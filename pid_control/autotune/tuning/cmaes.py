"""CMA-ES numerical tuner backend (PLAN.md T4.2).

Implements a (μ/μ_w, λ)-CMA-ES following Hansen 2016.  The implementation
is self-contained (no external ``cma`` package required) and wraps behind
the :class:`~pid_control.autotune.tuning.base.Tuner` protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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


@dataclass
class CMAESTuner:
    """CMA-ES (Covariance Matrix Adaptation – Evolution Strategy) tuner.

    Parameters
    ----------
    cost_spec : CostSpec | None
        Cost weights (defaults built from *objective* when ``None``).
    max_iter : int
        Maximum number of CMA-ES generations.
    pop_size : int | None
        Population size λ.  ``None`` → ``4 + floor(3·ln(n))``.
    sigma0 : float
        Initial step-size σ₀.
    seed : int
        RNG seed for reproducibility.
    bounds_kp, bounds_ki, bounds_kd
        Search bounds.  Inferred from the model when ``None``.
    """

    cost_spec: Optional[CostSpec] = None
    max_iter: int = 100
    pop_size: Optional[int] = None
    sigma0: float = 0.5
    seed: int = 42
    tol: float = 1e-8
    bounds_kp: Optional[Tuple[float, float]] = None
    bounds_ki: Optional[Tuple[float, float]] = None
    bounds_kd: Optional[Tuple[float, float]] = None
    name: str = "cmaes"

    # ------------------------------------------------------------------ #
    # Tuner protocol
    # ------------------------------------------------------------------ #
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

        x0 = np.array([initial.kp, initial.ki, initial.kd], dtype=float)
        initial_cost = evaluator.evaluate(initial)

        bounds = self._resolve_bounds(identification, initial)
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])

        best_x, best_cost, nit, cost_history = self._run_cmaes(
            evaluator.evaluate_array, x0, lower, upper,
        )

        best = PIDGains(
            kp=float(best_x[0]),
            ki=float(best_x[1]),
            kd=float(best_x[2]),
            setpoint_weight_b=initial.setpoint_weight_b,
            setpoint_weight_c=initial.setpoint_weight_c,
            derivative_filter_n=initial.derivative_filter_n,
        )

        warnings: List[Warning] = []
        if nit >= self.max_iter:
            warnings.append(Warning(
                code=WarningCode.W_MAXITER,
                severity=Severity.WARNING,
                message=f"CMA-ES reached max iterations ({self.max_iter}).",
                stage="tuning.cmaes",
            ))

        if best_cost >= 1e10:
            status = Status.FAILED
            warnings.append(Warning(
                code=WarningCode.W_COST_NOT_IMPROVED,
                severity=Severity.ERROR,
                message="CMA-ES: all candidates returned infinite cost.",
                stage="tuning.cmaes",
            ))
        elif nit >= self.max_iter and best_cost >= initial_cost:
            status = Status.FAILED
        elif nit >= self.max_iter:
            status = Status.WARNING
        else:
            status = Status.OK

        return TunerOutcome(
            gains=best,
            status=status,
            cost=best_cost,
            initial_gains=initial,
            initial_cost=initial_cost,
            iterations=nit,
            cost_history=np.array(cost_history) if cost_history else None,
            warnings=tuple(warnings),
            meta={"method": "cmaes"},
        )

    # ------------------------------------------------------------------ #
    # Core CMA-ES loop (Hansen 2016 canonical form, n=3)
    # ------------------------------------------------------------------ #
    def _run_cmaes(
        self,
        func,
        x0: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> Tuple[np.ndarray, float, int, List[float]]:
        rng = np.random.RandomState(self.seed)
        n = len(x0)
        lam = self.pop_size or (4 + int(3 * np.log(n)))
        mu = lam // 2

        # Recombination weights
        raw_w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = raw_w / raw_w.sum()
        mu_eff = 1.0 / (weights ** 2).sum()

        # Adaptation parameters
        c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0)
        d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + c_sigma
        c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
        c_1 = 2.0 / ((n + 1.3) ** 2 + mu_eff)
        c_mu_cov = min(1.0 - c_1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff))

        chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

        # State
        mean = x0.copy()
        sigma = self.sigma0
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        best_x = x0.copy()
        best_f = func(x0)
        cost_history: List[float] = [best_f]

        for gen in range(self.max_iter):
            # Sample population
            try:
                eig_vals, eig_vecs = np.linalg.eigh(C)
                eig_vals = np.maximum(eig_vals, 1e-20)
                sqrt_C = eig_vecs @ np.diag(np.sqrt(eig_vals)) @ eig_vecs.T
            except np.linalg.LinAlgError:
                C = np.eye(n)
                sqrt_C = np.eye(n)

            z_all = rng.randn(lam, n)
            xs = np.empty((lam, n))
            for k in range(lam):
                xs[k] = mean + sigma * (sqrt_C @ z_all[k])
                # Box constraint: clip to bounds
                xs[k] = np.clip(xs[k], lower, upper)

            # Evaluate
            fs = np.array([func(xs[k]) for k in range(lam)])

            # Sort by fitness
            idx = np.argsort(fs)
            xs = xs[idx]
            z_all = z_all[idx]
            fs = fs[idx]

            if fs[0] < best_f:
                best_f = fs[0]
                best_x = xs[0].copy()
            cost_history.append(best_f)

            # New mean
            new_mean = np.zeros(n)
            for i in range(mu):
                new_mean += weights[i] * xs[i]

            # Evolution paths
            inv_sqrt_C = eig_vecs @ np.diag(1.0 / np.sqrt(eig_vals)) @ eig_vecs.T
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_sqrt_C @ (new_mean - mean) / sigma)
            h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 2.0 / (n + 1)) * chi_n else 0.0
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (new_mean - mean) / sigma

            # Covariance update
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                y_i = (xs[i] - mean) / sigma
                rank_mu += weights[i] * np.outer(y_i, y_i)

            C = (1 - c_1 - c_mu_cov) * C + c_1 * rank_one + c_mu_cov * rank_mu

            # Step-size update
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))

            mean = new_mean

            # Convergence check
            if sigma < self.tol:
                break

        return best_x, best_f, gen + 1, cost_history

    # ------------------------------------------------------------------ #
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
