"""Differential-Evolution numerical tuner (PLAN.md T4.2).

Wraps ``scipy.optimize.differential_evolution`` behind the
:class:`~pid_control.autotune.tuning.base.Tuner` protocol so it can be
swapped with Nelder-Mead or other backends without touching the façade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import differential_evolution

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
class DETuner:
    """Differential-evolution PID tuner.

    Parameters
    ----------
    cost_spec
        Cost weights.  When ``None``, built from ``objective`` via
        :meth:`CostSpec.from_objective`.
    max_iter
        Maximum number of DE generations.
    pop_size
        Population multiplier (actual pop = pop_size × 3 dimensions).
    seed
        RNG seed for reproducibility.
    bounds_kp, bounds_ki, bounds_kd
        Search ranges.  When ``None`` they are inferred from the model.
    """

    cost_spec: Optional[CostSpec] = None
    max_iter: int = 80
    pop_size: int = 15
    seed: int = 42
    tol: float = 1e-6
    bounds_kp: Optional[Tuple[float, float]] = None
    bounds_ki: Optional[Tuple[float, float]] = None
    bounds_kd: Optional[Tuple[float, float]] = None
    name: str = "de"

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

        initial_cost = evaluator.evaluate(initial)
        bounds = self._resolve_bounds(identification, initial)
        cost_history: List[float] = []

        def callback(xk: np.ndarray, convergence: float = 0.0) -> bool:
            cost_history.append(evaluator.evaluate_array(xk))
            return False

        result = differential_evolution(
            evaluator.evaluate_array,
            bounds=bounds,
            maxiter=self.max_iter,
            popsize=self.pop_size,
            seed=self.seed,
            tol=self.tol,
            x0=np.array([initial.kp, initial.ki, initial.kd]),
            callback=callback,
            polish=True,
        )

        best = PIDGains(
            kp=float(result.x[0]),
            ki=float(result.x[1]),
            kd=float(result.x[2]),
            setpoint_weight_b=initial.setpoint_weight_b,
            setpoint_weight_c=initial.setpoint_weight_c,
            derivative_filter_n=initial.derivative_filter_n,
        )
        best_cost = float(result.fun)

        warnings: List[Warning] = []
        # PLAN T1.5 / C1: if a user-supplied bound is hit, surface
        # W_GAIN_CLIPPED so the caller knows their envelope was binding.
        user_bounds = {
            "kp": self.bounds_kp,
            "ki": self.bounds_ki,
            "kd": self.bounds_kd,
        }
        for idx, (name, bound) in enumerate(user_bounds.items()):
            if bound is None:
                continue
            lo, hi = bound
            val = float(result.x[idx])
            tol = 1e-6 * max(abs(hi - lo), 1.0)
            if abs(val - lo) <= tol or abs(val - hi) <= tol:
                warnings.append(Warning(
                    code=WarningCode.W_GAIN_CLIPPED,
                    severity=Severity.WARNING,
                    message=(
                        f"Optimal {name}={val:.4g} is at a user-provided "
                        f"bound [{lo:.4g}, {hi:.4g}]. Consider widening it."
                    ),
                    stage="tuning.de",
                    context={"gain": name, "value": val, "bounds": (lo, hi)},
                ))

        if not result.success:
            warnings.append(Warning(
                code=WarningCode.W_MAXITER,
                severity=Severity.WARNING,
                message=f"DE did not converge: {result.message}",
                stage="tuning.de",
            ))

        # Determine status
        if best_cost >= 1e10:
            status = Status.FAILED
            warnings.append(Warning(
                code=WarningCode.W_COST_NOT_IMPROVED,
                severity=Severity.ERROR,
                message="Cost function returned inf for all candidates — plant may be PID-inappropriate.",
                stage="tuning.de",
            ))
        elif not result.success and best_cost >= initial_cost:
            status = Status.FAILED
            warnings.append(Warning(
                code=WarningCode.W_COST_NOT_IMPROVED,
                severity=Severity.ERROR,
                message="Optimizer did not improve on the initial guess.",
                stage="tuning.de",
            ))
        elif not result.success:
            status = Status.WARNING
        else:
            status = Status.OK

        return TunerOutcome(
            gains=best,
            status=status,
            cost=best_cost,
            initial_gains=initial,
            initial_cost=initial_cost,
            iterations=result.nit,
            cost_history=np.array(cost_history) if cost_history else None,
            warnings=tuple(warnings),
            meta={
                "method": "differential_evolution",
                "nfev": result.nfev,
                "message": result.message,
            },
        )

    def _resolve_bounds(
        self,
        identification: IdentificationResult,
        initial: PIDGains,
    ) -> List[Tuple[float, float]]:
        """Infer sensible search bounds from the identified model."""
        model = identification.model
        K = abs(model.K) or 1.0
        tau = max(float(model.tau), 0.1)

        # Scale-aware bounds: roughly 0.1x to 10x the natural scale
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


@dataclass
class NelderMeadTuner:
    """Nelder-Mead simplex tuner (gradient-free, fast for local refinement)."""

    cost_spec: Optional[CostSpec] = None
    max_iter: int = 200
    name: str = "nelder_mead"

    def refine(
        self,
        identification: IdentificationResult,
        initial: PIDGains,
        objective: Objective,
        actuator: ActuatorLimits,
    ) -> TunerOutcome:
        from scipy.optimize import minimize

        spec = self.cost_spec or CostSpec.from_objective(objective)
        evaluator = CostEvaluator(
            identification=identification,
            objective=objective,
            actuator=actuator,
            cost_spec=spec,
        )
        initial_cost = evaluator.evaluate(initial)

        x0 = np.array([initial.kp, initial.ki, initial.kd])
        cost_history: List[float] = []

        def callback(xk: np.ndarray) -> None:
            cost_history.append(evaluator.evaluate_array(xk))

        result = minimize(
            evaluator.evaluate_array,
            x0,
            method="Nelder-Mead",
            options={"maxiter": self.max_iter, "xatol": 1e-6, "fatol": 1e-8},
            callback=callback,
        )

        best = PIDGains(
            kp=max(float(result.x[0]), 0.0),
            ki=max(float(result.x[1]), 0.0),
            kd=max(float(result.x[2]), 0.0),
            setpoint_weight_b=initial.setpoint_weight_b,
            setpoint_weight_c=initial.setpoint_weight_c,
            derivative_filter_n=initial.derivative_filter_n,
        )
        best_cost = float(result.fun)

        warnings: List[Warning] = []
        if not result.success:
            warnings.append(Warning(
                code=WarningCode.W_MAXITER,
                severity=Severity.WARNING,
                message=f"Nelder-Mead: {result.message}",
                stage="tuning.nm",
            ))

        status = Status.OK
        if best_cost >= 1e10:
            status = Status.FAILED
        elif not result.success:
            status = Status.WARNING if best_cost < initial_cost else Status.FAILED

        return TunerOutcome(
            gains=best,
            status=status,
            cost=best_cost,
            initial_gains=initial,
            initial_cost=initial_cost,
            iterations=result.nit,
            cost_history=np.array(cost_history) if cost_history else None,
            warnings=tuple(warnings),
            meta={"method": "nelder_mead", "nfev": result.nfev},
        )


__all__ = ["DETuner", "NelderMeadTuner"]
