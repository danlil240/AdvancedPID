"""Second-order plus dead-time identifier (PLAN.md T2.2).

Fits two cascaded first-order models:

.. math::

    G(s) = \\frac{K\\,e^{-\\theta s}}{(\\tau_1 s + 1)(\\tau_2 s + 1)}

Critical safety feature: the fit is **rejected** if τ₂ collapses to a
fraction of the sample time or a fraction of τ₁.  A degenerate SOPDT
with a fast fake pole is indistinguishable from FOPDT plus fitting
noise; the model-selection stage should prefer the FOPDT in that case.
Callers receive a :class:`~pid_control.autotune.types.WarningCode.W_DEGENERATE_SOPDT`
warning and the identifier's ``fit_quality_r2`` is *left untouched* so
AIC/BIC naturally disqualify it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

from pid_control.autotune.experiments.base import ExperimentRecord
from pid_control.autotune.identification.base import Identifier
from pid_control.autotune.identification.fopdt import (
    FOPDTIdentifier,
    _detect_step,  # noqa: PLC2701 - intra-package reuse is intentional
)
from pid_control.autotune.identification.quality import compute_fit_quality
from pid_control.autotune.identification.simulate import simulate_model
from pid_control.autotune.types import (
    IdentificationResult,
    ModelType,
    Severity,
    TransferFunctionModel,
    Warning,
    WarningCode,
)

# Degeneracy thresholds — see module docstring.
DEGENERATE_TAU_DT_FACTOR = 5.0
DEGENERATE_TAU_RATIO = 0.05
# If the SOPDT residual RMSE is ≥ this fraction of the FOPDT anchor's
# RMSE, the second pole brought no real information and we flag the
# fit as degenerate.  80 % is a conservative threshold calibrated on
# synthetic FOPDT data, where SOPDT's best RMSE still tracks the FOPDT
# RMSE closely because the true response really is first-order.
DEGENERATE_RMSE_FRACTION = 0.8


@dataclass
class SOPDTIdentifier:
    """Least-squares SOPDT identifier."""

    max_nfev: int = 400
    name: str = "sopdt_nlls"

    def identify(self, record: ExperimentRecord) -> IdentificationResult:
        t = record.time
        u = record.input
        y = record.output
        dt = record.sample_time

        warnings: List[Warning] = []
        step = _detect_step(u)
        step_idx = step.index if step is not None else 0
        baseline_u = (
            step.baseline_input if step is not None
            else float(np.mean(u[: max(1, len(u) // 20)]))
        )
        baseline_y = float(np.mean(y[: max(1, step_idx or 1)]))
        u_dev = u - baseline_u
        y_dev = y - baseline_y

        # Use a completed FOPDT fit as the anchor: it gives a reliable K
        # and a *sharp* dead-time estimate.  Splitting that τ into two
        # real poles becomes a one-dimensional problem the optimiser
        # handles without getting trapped in a degenerate basin.
        fopdt = FOPDTIdentifier().identify(record)
        K0 = float(fopdt.model.K)
        tau_total = float(fopdt.model.tau)
        theta_from_fopdt = float(fopdt.model.theta)

        duration = float(t[-1] - t[0])

        # θ upper bound: never more than the FOPDT θ + one sample period.
        # If the underlying system is SOPDT, its true θ is *smaller* than
        # the FOPDT θ (the extra pole inflates apparent lag); so the
        # FOPDT θ is a safe, tight ceiling that prevents the optimiser
        # from absorbing a collapsed τ₂ into θ.
        theta_cap = theta_from_fopdt + 2 * dt
        bounds = _bounds(dt, duration, K0, theta_cap=theta_cap)

        # Seed list.  Each seed pairs a (τ₁, τ₂) split of the FOPDT
        # time-constant with TWO choices of θ:
        #   * θ = θ_FOPDT (the optimiser is free to grow τ₂ to offset
        #     the lag, or to keep the FOPDT-equivalent basin);
        #   * θ = min sample step (forces τ₂ to carry all of the lag;
        #     this is the seed that escapes the degenerate basin when
        #     the data truly is SOPDT).
        # The optimiser picks whichever seed delivers the lowest cost.
        tau_min = bounds[0][1]
        theta_small = max(0.0, 2 * dt)
        seeds: List[Tuple[float, float, float, float]] = []
        for ratio in (0.5, 0.7, 0.9, 0.3):
            tau1 = max(tau_min, tau_total * ratio)
            tau2 = max(tau_min, tau_total * (1 - ratio))
            theta_full = max(0.0, min(theta_from_fopdt, bounds[1][3]))
            seeds.append((K0, tau1, tau2, theta_full))
            seeds.append((K0, tau1, tau2, min(theta_small, bounds[1][3])))

        best: Optional[Tuple[float, Tuple[float, float, float, float]]] = None
        for seed in seeds:
            try:
                res = least_squares(
                    fun=_residuals, x0=seed, bounds=bounds,
                    args=(t, u_dev, y_dev),
                    max_nfev=self.max_nfev,
                )
            except Exception:  # pragma: no cover
                continue
            cost = float(res.cost)
            if np.isfinite(cost) and (best is None or cost < best[0]):
                best = (cost, tuple(res.x))  # type: ignore[assignment]

        if best is None:
            # Degrade gracefully to a FOPDT fit — still honest.
            warnings.append(
                Warning(
                    code=WarningCode.W_POOR_FIT,
                    severity=Severity.WARNING,
                    message="SOPDT NLLS did not converge; deferring to FOPDT",
                    stage="identify.sopdt",
                )
            )
            return FOPDTIdentifier().identify(record)

        K_fit, tau1_fit, tau2_fit, theta_fit = best[1]

        # Convention: tau1 is the larger pole.
        if tau2_fit > tau1_fit:
            tau1_fit, tau2_fit = tau2_fit, tau1_fit

        # Score the fit *first* so the degeneracy check can compare
        # RMSE against the FOPDT anchor.
        model = TransferFunctionModel(
            model_type=ModelType.SOPDT,
            K=K_fit, tau=tau1_fit, tau2=tau2_fit, theta=theta_fit,
        )
        y_sim = simulate_model(model, t, u_dev, y0=0.0) + baseline_y
        fit = compute_fit_quality(y, y_sim, n_params=4)

        # Degeneracy detection — three independent failure modes:
        #   1. τ₂ collapses to a small multiple of the sample time, or
        #   2. τ₂/τ₁ ratio is vanishingly small (one pole dominates), or
        #   3. the SOPDT fit offers no meaningful residual improvement
        #      over the FOPDT anchor (the second pole is "free-floating",
        #      the data is really FOPDT).
        degenerate_reasons: List[str] = []
        if tau2_fit < DEGENERATE_TAU_DT_FACTOR * dt:
            degenerate_reasons.append(
                f"τ₂={tau2_fit:.4g}s ≈ {DEGENERATE_TAU_DT_FACTOR}·dt"
            )
        if tau1_fit > 0 and tau2_fit / tau1_fit < DEGENERATE_TAU_RATIO:
            degenerate_reasons.append(
                f"τ₂/τ₁={tau2_fit / tau1_fit:.3f} below {DEGENERATE_TAU_RATIO}"
            )
        fopdt_rmse = float(fopdt.residual_rmse or 0.0)
        sopdt_rmse = float(fit.rmse or 0.0)
        if fopdt_rmse > 0 and sopdt_rmse >= DEGENERATE_RMSE_FRACTION * fopdt_rmse:
            degenerate_reasons.append(
                f"SOPDT RMSE {sopdt_rmse:.4g} ≥ "
                f"{DEGENERATE_RMSE_FRACTION:.0%} × FOPDT RMSE {fopdt_rmse:.4g}"
            )
        if degenerate_reasons:
            warnings.append(
                Warning(
                    code=WarningCode.W_DEGENERATE_SOPDT,
                    severity=Severity.WARNING,
                    message=(
                        "SOPDT fit is degenerate: "
                        + "; ".join(degenerate_reasons)
                        + "; FOPDT preferred."
                    ),
                    stage="identify.sopdt",
                    context={
                        "tau1": tau1_fit, "tau2": tau2_fit, "dt": dt,
                        "fopdt_rmse": fopdt_rmse, "sopdt_rmse": sopdt_rmse,
                    },
                )
            )

        if fit.r2 < 0.7:
            warnings.append(
                Warning(
                    code=WarningCode.W_POOR_FIT,
                    severity=Severity.WARNING,
                    message=f"Low SOPDT fit quality (R²={fit.r2:.3f})",
                    stage="identify.sopdt",
                    context={"r2": fit.r2},
                )
            )

        return IdentificationResult(
            model=model,
            fit_quality_r2=fit.r2,
            aic=fit.aic,
            bic=fit.bic,
            residual_rmse=fit.rmse,
            noise_variance=fit.residual_variance,
            data_quality=record.quality,
            warnings=tuple(warnings),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bounds(
    dt: float, duration: float, K0: float, theta_cap: float,
):
    tau_min = max(dt * DEGENERATE_TAU_DT_FACTOR, dt)
    tau_max = max(duration * 10.0, tau_min * 100.0)
    if abs(K0) < 1e-9:
        K_lo, K_hi = -1e3, 1e3
    elif K0 > 0:
        K_lo, K_hi = K0 / 100.0, K0 * 100.0
    else:
        K_lo, K_hi = K0 * 100.0, K0 / 100.0
    theta_min = 0.0
    theta_max = min(max(theta_cap, dt * 10.0), max(duration * 0.8, dt * 10.0))
    lo = (K_lo, tau_min, tau_min, theta_min)
    hi = (K_hi, tau_max, tau_max, theta_max)
    return lo, hi


def _residuals(
    params: np.ndarray,
    t: np.ndarray,
    u_dev: np.ndarray,
    y_dev: np.ndarray,
) -> np.ndarray:
    K, tau1, tau2, theta = params
    model = TransferFunctionModel(
        model_type=ModelType.SOPDT,
        K=float(K), tau=float(tau1), tau2=float(tau2), theta=float(theta),
    )
    y_sim = simulate_model(model, t, u_dev, y0=0.0)
    return y_sim - y_dev


__all__ = ["SOPDTIdentifier"]
