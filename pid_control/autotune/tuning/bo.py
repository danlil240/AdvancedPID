"""Bayesian Optimization tuner using sklearn's GaussianProcessRegressor (PLAN.md T4.2, M6).

This replaces the legacy ``BayesianTuner`` fake GP with a proper acquisition-
function based loop.  Requires ``scikit-learn`` — if not installed the import
will raise :class:`ImportError` with a helpful message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "BOTuner requires scikit-learn.  Install it with:\n"
        "  pip install scikit-learn"
    ) from _exc

from pid_control.autotune.tuning.base import TunerOutcome
from pid_control.autotune.tuning.cost import CostEvaluator, CostSpec
from pid_control.autotune.types import (
    ActuatorLimits,
    IdentificationResult,
    Objective,
    PIDGains,
    Severity,
    Status,
    Warning,
    WarningCode,
)


def _expected_improvement(
    X: np.ndarray,
    gpr: GaussianProcessRegressor,
    f_best: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Compute the Expected Improvement acquisition function."""
    from scipy.stats import norm

    mu, sigma = gpr.predict(X, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    z = (f_best - mu - xi) / sigma
    return (f_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)


@dataclass
class BOTuner:
    """Bayesian-optimisation PID tuner (real GP, EI acquisition).

    Parameters
    ----------
    cost_spec : CostSpec | None
        Cost weights.  ``None`` → built from *objective*.
    n_initial : int
        Number of Latin-hypercube initial samples.
    max_iter : int
        Maximum BO iterations (each evaluates the real cost once).
    seed : int
        RNG seed.
    bounds_kp, bounds_ki, bounds_kd
        Search bounds.  Inferred from the model when ``None``.
    """

    cost_spec: Optional[CostSpec] = None
    n_initial: int = 10
    max_iter: int = 40
    seed: int = 42
    bounds_kp: Optional[Tuple[float, float]] = None
    bounds_ki: Optional[Tuple[float, float]] = None
    bounds_kd: Optional[Tuple[float, float]] = None
    name: str = "bo"

    def refine(
        self,
        identification: IdentificationResult,
        initial: PIDGains,
        objective: Objective,
        actuator: ActuatorLimits,
    ) -> TunerOutcome:
        spec = self.cost_spec or CostSpec.from_objective(objective)
        evaluator = CostEvaluator(
            identification=identification,
            objective=objective,
            actuator=actuator,
            cost_spec=spec,
        )

        bounds = self._resolve_bounds(identification, initial)
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])

        rng = np.random.RandomState(self.seed)
        initial_cost = evaluator.evaluate(initial)

        # Latin-hypercube initial samples + the initial guess
        n = 3  # dimensionality
        X_init = rng.uniform(lower, upper, size=(self.n_initial, n))
        X_init = np.vstack([
            np.array([[initial.kp, initial.ki, initial.kd]]),
            X_init,
        ])
        Y_init = np.array([evaluator.evaluate_array(x) for x in X_init])

        X_obs = X_init.copy()
        Y_obs = Y_init.copy()
        cost_history: List[float] = [float(Y_obs.min())]

        gpr = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=self.seed,
        )

        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            gpr.fit(X_obs, Y_obs)
            f_best = float(Y_obs.min())

            # Maximise EI via random search (cheap for n=3)
            X_cand = rng.uniform(lower, upper, size=(2000, n))
            ei = _expected_improvement(X_cand, gpr, f_best)
            x_next = X_cand[np.argmax(ei)]

            y_next = evaluator.evaluate_array(x_next)
            X_obs = np.vstack([X_obs, x_next.reshape(1, -1)])
            Y_obs = np.append(Y_obs, y_next)
            cost_history.append(float(Y_obs.min()))

            # Convergence: EI negligible
            if ei.max() < 1e-8:
                break

        best_idx = int(np.argmin(Y_obs))
        best_x = X_obs[best_idx]
        best_cost = float(Y_obs[best_idx])

        best = PIDGains(
            kp=float(best_x[0]),
            ki=float(best_x[1]),
            kd=float(best_x[2]),
            setpoint_weight_b=initial.setpoint_weight_b,
            setpoint_weight_c=initial.setpoint_weight_c,
            derivative_filter_n=initial.derivative_filter_n,
        )

        warnings: List[Warning] = []
        if n_iter >= self.max_iter:
            warnings.append(Warning(
                code=WarningCode.W_MAXITER,
                severity=Severity.WARNING,
                message=f"BO reached max iterations ({self.max_iter}).",
                stage="tuning.bo",
            ))

        if best_cost >= 1e10:
            status = Status.FAILED
        elif n_iter >= self.max_iter and best_cost >= initial_cost:
            status = Status.FAILED
        elif n_iter >= self.max_iter:
            status = Status.WARNING
        else:
            status = Status.OK

        return TunerOutcome(
            gains=best,
            status=status,
            cost=best_cost,
            initial_gains=initial,
            initial_cost=initial_cost,
            iterations=n_iter,
            cost_history=np.array(cost_history) if cost_history else None,
            warnings=tuple(warnings),
            meta={"method": "bayesian_optimization", "n_evaluations": len(Y_obs)},
        )

    def _resolve_bounds(
        self,
        identification: IdentificationResult,
        initial: PIDGains,
    ) -> List[Tuple[float, float]]:
        model = identification.model
        K = abs(model.K) or 1.0
        tau = max(float(model.tau), 0.1)
        kp_scale = tau / K
        ki_scale = kp_scale / tau
        kd_scale = kp_scale * tau

        def _bound(user: Optional[Tuple[float, float]], scale: float) -> Tuple[float, float]:
            if user is not None:
                return user
            return (0.0, max(10.0 * scale, abs(initial.kp) * 5.0))

        return [
            _bound(self.bounds_kp, kp_scale),
            _bound(self.bounds_ki, ki_scale),
            _bound(self.bounds_kd, kd_scale),
        ]
