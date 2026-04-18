"""First-order plus dead-time identifier (PLAN.md T2.1).

Approach
--------
1. **Step detection** — locate the largest rapid change in the input so
   we can reason about pre-/post-step baselines and the time at which
   excitation begins.  Handles ramp-like inputs too; see
   :func:`_detect_step`.
2. **Two-point seed** — compute an initial ``(K̂, τ̂, θ̂)`` from the
   time-to-28.3 % and time-to-63.2 % points of the response.  This is
   the classical Ziegler two-point method; it is scale-free and robust
   enough for any reasonable SNR.
3. **Non-linear least squares** — refine with a multi-start
   :func:`scipy.optimize.least_squares` on the (noise-free) FOPDT model
   response.  Bounds are wide but finite so the optimiser never leaves
   the physically meaningful cone.
4. **Quality scoring** — report R², residual RMSE, AIC and BIC so the
   model-selection stage (T2.4) can compare against SOPDT fairly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

from pid_control.autotune.experiments.base import ExperimentRecord
from pid_control.autotune.identification.base import Identifier
from pid_control.autotune.identification.quality import (
    compute_fit_quality,
)
from pid_control.autotune.identification.simulate import simulate_model
from pid_control.autotune.types import (
    IdentificationResult,
    ModelType,
    Severity,
    TransferFunctionModel,
    Warning,
    WarningCode,
)

# Heuristic floor on the FOPDT τ — prevents the optimiser collapsing to
# a pure gain + dead time (which is unidentifiable from step data).
_TAU_FLOOR_SAMPLES = 2


@dataclass
class FOPDTIdentifier:
    """NLLS identifier for ``G(s) = K * exp(-θs) / (τs + 1)``."""

    multistart: int = 3
    max_theta_over_tau: float = 10.0
    name: str = "fopdt_nlls"

    # ------------------------------------------------------------------
    # Protocol entrypoint
    # ------------------------------------------------------------------

    def identify(self, record: ExperimentRecord) -> IdentificationResult:
        t = record.time
        u = record.input
        y = record.output
        dt = record.sample_time

        warnings: List[Warning] = []
        step = _detect_step(u)
        if step is None:
            warnings.append(
                Warning(
                    code=WarningCode.W_POOR_FIT,
                    severity=Severity.WARNING,
                    message="No clear step detected in input; FOPDT fit may be unreliable",
                    stage="identify.fopdt",
                )
            )
            step_idx = 0
            baseline_u = float(np.mean(u[: max(1, len(u) // 20)]))
        else:
            step_idx = step.index
            baseline_u = step.baseline_input

        # Subtract operating points so FOPDT's zero-initial-state
        # assumption is respected.  We work entirely in "deviation
        # variables" from here on.
        baseline_y = float(np.mean(y[: max(1, step_idx or 1)]))
        y_dev = y - baseline_y
        u_dev = u - baseline_u

        # Two-point seed ---------------------------------------------------
        try:
            K0, tau0, theta0 = _two_point_seed(t, u_dev, y_dev, step_idx)
        except _IdentificationAborted as exc:
            warnings.append(
                Warning(
                    code=WarningCode.W_POOR_FIT,
                    severity=Severity.WARNING,
                    message=f"FOPDT seed failed: {exc}; using fallback guess",
                    stage="identify.fopdt",
                )
            )
            K0 = float((y_dev[-1] - y_dev[0]) / (u_dev[-1] - u_dev[0] or 1e-9))
            tau0 = max(dt * 10, (t[-1] - t[0]) * 0.1)
            theta0 = dt * 2

        bounds = _bounds_for(dt, t[-1] - t[0], K0)
        seeds = _seed_list(K0, tau0, theta0, bounds, n=self.multistart)

        # NLLS multi-start -------------------------------------------------
        best: Optional[Tuple[float, Tuple[float, float, float]]] = None
        for seed in seeds:
            try:
                res = least_squares(
                    fun=_residuals, x0=seed, bounds=bounds,
                    args=(t, u_dev, y_dev, dt),
                    max_nfev=300,
                )
            except Exception:  # pragma: no cover - defensive
                continue
            cost = float(res.cost)
            if np.isfinite(cost) and (best is None or cost < best[0]):
                best = (cost, tuple(res.x))  # type: ignore[assignment]

        if best is None:
            # Should be rare — fall back to the seed itself.
            K_fit, tau_fit, theta_fit = K0, tau0, theta0
            warnings.append(
                Warning(
                    code=WarningCode.W_POOR_FIT,
                    severity=Severity.WARNING,
                    message="FOPDT NLLS did not converge; reporting two-point seed",
                    stage="identify.fopdt",
                )
            )
        else:
            K_fit, tau_fit, theta_fit = best[1]

        # Final simulation for quality metrics -----------------------------
        model = TransferFunctionModel(
            model_type=ModelType.FOPDT, K=K_fit, tau=tau_fit, theta=theta_fit,
        )
        y_sim_dev = simulate_model(model, t, u_dev, y0=0.0)
        y_sim = y_sim_dev + baseline_y

        fit = compute_fit_quality(y, y_sim, n_params=3)

        if fit.r2 < 0.7:
            warnings.append(
                Warning(
                    code=WarningCode.W_POOR_FIT,
                    severity=Severity.WARNING,
                    message=f"Low fit quality (R²={fit.r2:.3f})",
                    stage="identify.fopdt",
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
# Supporting routines
# ---------------------------------------------------------------------------

class _IdentificationAborted(Exception):
    pass


@dataclass(frozen=True)
class _StepInfo:
    index: int
    baseline_input: float
    amplitude: float


def _detect_step(u: np.ndarray) -> Optional[_StepInfo]:
    """Locate the first sample where the input jumps by at least 25 % of
    its full range.  Returns ``None`` when no such jump exists (e.g.
    ramp or multi-level test signal).
    """
    du = np.diff(u)
    rng = float(np.ptp(u))
    if rng < 1e-12:
        return None
    threshold = 0.25 * rng
    idx = int(np.argmax(np.abs(du)))
    if abs(du[idx]) < threshold:
        return None
    baseline = float(np.mean(u[: max(1, idx)]))
    amplitude = float(u[idx + 1] - baseline)
    return _StepInfo(index=idx + 1, baseline_input=baseline, amplitude=amplitude)


def _two_point_seed(
    t: np.ndarray,
    u_dev: np.ndarray,
    y_dev: np.ndarray,
    step_idx: int,
) -> Tuple[float, float, float]:
    """Classical two-point (28.3 %, 63.2 %) estimate of (K, τ, θ).

    Robust to monotonic step responses; fails loudly on non-monotonic or
    flat data so the caller can fall back.
    """
    u_amp = float(u_dev[-1] - u_dev[0])
    if abs(u_amp) < 1e-9:
        raise _IdentificationAborted("Zero input amplitude")

    y_amp = float(y_dev[-1] - y_dev[0])
    if abs(y_amp) < 1e-9:
        raise _IdentificationAborted("Zero output response")

    K0 = y_amp / u_amp
    sign = np.sign(y_amp)

    # Normalise so we can chase 28.3 % and 63.2 % levels regardless of sign.
    y_norm = (y_dev - y_dev[step_idx]) / y_amp if abs(y_amp) > 1e-9 else y_dev

    def _time_at(level: float) -> Optional[float]:
        target = level
        above = np.where(sign * y_norm >= sign * target)[0]
        above = above[above >= step_idx]
        if above.size == 0:
            return None
        k = int(above[0])
        if k == step_idx:
            return float(t[k])
        y0, y1 = y_norm[k - 1], y_norm[k]
        if y1 == y0:
            return float(t[k])
        frac = (target - y0) / (y1 - y0)
        return float(t[k - 1] + frac * (t[k] - t[k - 1]))

    t28 = _time_at(0.283)
    t63 = _time_at(0.632)
    t0 = float(t[step_idx])
    if t28 is None or t63 is None or t63 <= t28:
        raise _IdentificationAborted("Response did not reach 63.2% level")

    tau0 = 1.5 * (t63 - t28)
    theta0 = max(0.0, t63 - tau0 - t0)
    return K0, tau0, theta0


def _bounds_for(
    dt: float, duration: float, K0: float,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Construct (lower, upper) bounds tuples for least_squares."""
    tau_min = max(dt * _TAU_FLOOR_SAMPLES, dt)
    tau_max = max(duration * 10.0, tau_min * 100.0)
    # Sign of K is determined by the two-point seed and held constant by
    # centring the bounds around it.  Magnitude bounded to ±100× for
    # numerical sanity.
    if abs(K0) < 1e-9:
        K_lo, K_hi = -1e3, 1e3
    elif K0 > 0:
        K_lo, K_hi = K0 / 100.0, K0 * 100.0
    else:
        K_lo, K_hi = K0 * 100.0, K0 / 100.0
    theta_min = 0.0
    theta_max = max(duration * 0.8, dt * 10.0)
    return (K_lo, tau_min, theta_min), (K_hi, tau_max, theta_max)


def _seed_list(
    K0: float, tau0: float, theta0: float,
    bounds: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
    n: int,
) -> List[Tuple[float, float, float]]:
    """Return ``n`` feasible seeds – the two-point estimate plus scaled
    variants to escape local minima."""
    lo, hi = bounds
    seeds: List[Tuple[float, float, float]] = [
        (K0, max(lo[1], tau0), max(lo[2], theta0))
    ]
    scales = (0.5, 2.0, 4.0)
    for s in scales[: max(0, n - 1)]:
        candidate = (
            float(np.clip(K0, lo[0], hi[0])),
            float(np.clip(tau0 * s, lo[1], hi[1])),
            float(np.clip(theta0 * s, lo[2], hi[2])),
        )
        seeds.append(candidate)
    return seeds[:n]


def _residuals(
    params: np.ndarray,
    t: np.ndarray,
    u_dev: np.ndarray,
    y_dev: np.ndarray,
    dt: float,
) -> np.ndarray:
    K, tau, theta = params
    model = TransferFunctionModel(
        model_type=ModelType.FOPDT, K=float(K), tau=float(tau), theta=float(theta),
    )
    y_sim = simulate_model(model, t, u_dev, y0=0.0)
    return y_sim - y_dev


__all__ = ["FOPDTIdentifier"]
