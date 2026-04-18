"""Integrator + dead-time (IPDT) identifier (PLAN.md T2.3).

Detects processes whose open-loop output keeps drifting under a constant
input — the hallmark of an integrating plant.  Fits the model::

    G(s) = K_i * exp(-θs) / s

where ``K_i`` is the integrator gain (output rate per unit input) and
``θ`` is the apparent dead time.

Detection heuristic:
 1. After the step, the output keeps rising/falling monotonically
    (or with small noise ripple) without settling.
 2. A linear fit ``y(t) = a + b·t`` on the post-step region yields a
    high R² (> 0.9) and the slope is consistent with the input magnitude.

If the plant passes the integrator test, fitting FOPDT/SOPDT is
inappropriate because their gain diverges; the data-quality stage will
also flag ``E_NO_STEADY_STATE``.  This identifier offers a proper model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.optimize import least_squares

from pid_control.autotune.experiments.base import ExperimentRecord
from pid_control.autotune.identification.base import Identifier
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


def _detect_step(u: np.ndarray):
    """Find the index of the largest magnitude jump in ``u``."""
    if len(u) < 2:
        return None
    diff = np.abs(np.diff(u))
    idx = int(np.argmax(diff))
    if diff[idx] < 1e-12:
        return None
    return idx


@dataclass
class IPDTIdentifier:
    """Identifier for integrator-plus-dead-time plants.

    Only emits a usable result when the data actually looks integrating.
    Otherwise returns a low-R² result so the model selector can discard it.
    """

    name: str = "ipdt"

    def identify(self, record: ExperimentRecord) -> IdentificationResult:
        t = record.time
        u = record.input
        y = record.output
        dt = record.sample_time

        warnings: List[Warning] = []

        step = _detect_step(u)
        if step is None:
            return self._unusable(warnings, t, u, y, dt)

        step_idx = step + 1
        baseline_u = float(np.mean(u[:max(1, step_idx)]))
        baseline_y = float(np.mean(y[:max(1, step_idx)]))
        delta_u = float(np.mean(u[step_idx:])) - baseline_u
        if abs(delta_u) < 1e-12:
            return self._unusable(warnings, t, u, y, dt)

        # Post-step region
        t_post = t[step_idx:] - t[step_idx]
        y_post = y[step_idx:] - baseline_y

        if len(t_post) < 10:
            return self._unusable(warnings, t, u, y, dt)

        # Linear fit on post-step output: y(t) ≈ slope * t + intercept
        coeffs = np.polyfit(t_post, y_post, 1)
        slope = coeffs[0]
        intercept = coeffs[1]
        y_lin = np.polyval(coeffs, t_post)
        ss_res = float(np.sum((y_post - y_lin) ** 2))
        ss_tot = float(np.sum((y_post - np.mean(y_post)) ** 2))
        r2_linear = 1.0 - ss_res / max(ss_tot, 1e-30)

        # Is this actually integrating?  Linear fit R² > 0.85 heuristic
        if r2_linear < 0.85:
            warnings.append(Warning(
                code=WarningCode.W_POOR_FIT,
                severity=Severity.WARNING,
                message=f"Linear fit R²={r2_linear:.3f} < 0.85; data may not be integrating.",
                stage="identify.ipdt",
            ))

        # K_i = slope / delta_u  (integrator gain)
        K_i = slope / delta_u

        # Dead-time estimation: find when output first departs from baseline
        noise_band = 3.0 * float(np.std(y[:max(1, step_idx)])) if step_idx > 1 else 0.0
        noise_band = max(noise_band, abs(y_post[-1]) * 0.01)
        theta = 0.0
        for i in range(len(y_post)):
            if abs(y_post[i] - intercept) > noise_band:
                theta = float(t_post[max(0, i - 1)])
                break

        # Refine with NLLS
        model_init = TransferFunctionModel(
            model_type=ModelType.IPDT, K=K_i, tau=0.0, theta=theta,
        )
        y_sim_init = simulate_model(model_init, t, u - baseline_u, y0=0.0) + baseline_y

        # NLLS refinement
        def residuals(params):
            K_fit, theta_fit = params
            m = TransferFunctionModel(
                model_type=ModelType.IPDT, K=K_fit, tau=0.0, theta=max(theta_fit, 0.0),
            )
            y_sim = simulate_model(m, t, u - baseline_u, y0=0.0) + baseline_y
            return y_sim - y

        try:
            sol = least_squares(
                residuals, [K_i, theta],
                bounds=([K_i * 0.01 if K_i > 0 else K_i * 100, 0.0],
                        [K_i * 100 if K_i > 0 else K_i * 0.01, float(t[-1]) * 0.5]),
                method="trf",
                max_nfev=200,
            )
            K_i, theta = float(sol.x[0]), float(sol.x[1])
        except Exception:
            pass  # Keep initial estimates

        model = TransferFunctionModel(
            model_type=ModelType.IPDT, K=K_i, tau=0.0, theta=theta,
        )
        y_sim = simulate_model(model, t, u - baseline_u, y0=0.0) + baseline_y

        quality = compute_fit_quality(y, y_sim, n_params=2)

        warnings.append(Warning(
            code=WarningCode.W_INTEGRATING,
            severity=Severity.INFO,
            message=f"Integrating plant detected: K_i={K_i:.4g}, theta={theta:.4g}",
            stage="identify.ipdt",
        ))

        return IdentificationResult(
            model=model,
            fit_quality_r2=quality.r2,
            aic=quality.aic,
            bic=quality.bic,
            residual_rmse=quality.rmse,
            noise_variance=quality.residual_variance,
            warnings=tuple(warnings),
        )

    def _unusable(self, warnings, t, u, y, dt) -> IdentificationResult:
        """Return a poor-quality result so the selector ignores it."""
        warnings.append(Warning(
            code=WarningCode.W_POOR_FIT,
            severity=Severity.WARNING,
            message="Cannot fit IPDT model to this data.",
            stage="identify.ipdt",
        ))
        return IdentificationResult(
            model=TransferFunctionModel(
                model_type=ModelType.IPDT, K=0.0, tau=0.0, theta=0.0,
            ),
            fit_quality_r2=-1.0,
            warnings=tuple(warnings),
        )
