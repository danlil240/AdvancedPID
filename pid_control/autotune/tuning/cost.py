"""Composable cost specification for numerical PID tuning (PLAN.md T4.3).

The cost function simulates the identified model in closed loop with
candidate PID gains and scores the resulting trajectory.  Every term
(IAE, ITAE, overshoot penalty, control effort, etc.) has a user-facing
weight so callers can express their priorities explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pid_control.autotune.identification.simulate import simulate_model
from pid_control.autotune.types import (
    ActuatorLimits,
    IdentificationResult,
    Objective,
    PIDGains,
)

# Sentinel for "no penalty"
_INF = 1e12


@dataclass(frozen=True)
class CostSpec:
    """Weights for each term in the tuning cost function.

    All weights default to zero except ``iae`` so that the minimal
    ``CostSpec()`` gives a pure IAE minimisation.  Pass
    :meth:`from_objective` to build sensible defaults from a user's
    :class:`Objective`.
    """

    iae: float = 1.0
    itae: float = 0.0
    ise: float = 0.0
    overshoot_above: float = 0.0       # penalty per % above target
    settling_pct_above: float = 0.0     # penalty per second above target
    du: float = 0.01                    # control total-variation weight
    Ms_penalty_above: float = 0.0       # per-unit penalty above threshold
    Mt_penalty_above: float = 0.0

    @staticmethod
    def from_objective(obj: Objective) -> "CostSpec":
        return CostSpec(
            iae=obj.iae_weight,
            itae=obj.itae_weight,
            ise=obj.ise_weight,
            overshoot_above=5.0 if obj.max_overshoot_pct < 100 else 0.0,
            du=obj.control_effort_weight,
        )


@dataclass
class CostEvaluator:
    """Evaluates candidate PID gains on the identified model.

    This replaces the hardcoded cost logic that was scattered across
    ``AutotuneFromData`` and ``RealtimeTuner``.  It uses the user's
    real setpoint and actuator limits rather than magic constants.
    """

    identification: IdentificationResult
    objective: Objective
    actuator: ActuatorLimits
    cost_spec: CostSpec
    setpoint: float = 1.0
    sim_duration: Optional[float] = None
    dt: Optional[float] = None

    def __post_init__(self) -> None:
        model = self.identification.model
        tau = float(model.tau)
        theta = float(model.theta)
        if self.sim_duration is None:
            self.sim_duration = max(10.0 * (tau + theta), 20.0 * theta, 5.0)
        if self.dt is None:
            self.dt = min(tau / 20.0, theta / 5.0, 0.05)
            self.dt = max(self.dt, 1e-4)

    def evaluate(self, gains: PIDGains) -> float:
        """Return scalar cost for ``gains``.  Lower is better.

        Returns ``_INF`` for unstable / divergent simulations.
        """
        dt = self.dt
        n = int(self.sim_duration / dt) + 1
        time = np.linspace(0.0, self.sim_duration, n)

        sp = np.full(n, self.setpoint)
        y = np.zeros(n)
        u = np.zeros(n)
        e_int = 0.0
        e_prev = 0.0
        e_filt = 0.0

        kp, ki, kd = gains.kp, gains.ki, gains.kd
        N = gains.derivative_filter_n
        lo = self.actuator.lower
        hi = self.actuator.upper

        model = self.identification.model

        for i in range(1, n):
            e = sp[i - 1] - y[i - 1]

            # Filtered derivative
            alpha = N * dt / (1.0 + N * dt) if N > 0 else 0.0
            e_filt = alpha * (e - e_prev) / dt + (1.0 - alpha) * e_filt if i > 1 else 0.0

            # Anti-windup: only integrate when output not saturated
            u_raw = kp * e + ki * e_int + kd * e_filt
            if lo <= u_raw <= hi:
                e_int += e * dt

            u_cmd = float(np.clip(kp * e + ki * e_int + kd * e_filt, lo, hi))

            # Rate limit
            if self.actuator.rate_limit is not None and i > 1:
                max_du = self.actuator.rate_limit * dt
                u_cmd = float(np.clip(u_cmd, u[i - 1] - max_du, u[i - 1] + max_du))

            u[i] = u_cmd

            # Simulate plant one step
            delay_samples = int(round(model.theta / dt))
            u_idx = max(i - delay_samples, 0)
            u_eff = u[u_idx]

            tau_p = max(float(model.tau), 1e-9)
            alpha_p = float(np.exp(-dt / tau_p))
            y[i] = alpha_p * y[i - 1] + model.K * (1.0 - alpha_p) * u_eff

            e_prev = e

            # Early abort: divergence
            if abs(y[i]) > 1e6 or not np.isfinite(y[i]):
                return _INF

        # --- Compute cost terms ---
        error = sp - y
        abs_error = np.abs(error)

        cost = 0.0

        # IAE
        if self.cost_spec.iae > 0:
            cost += self.cost_spec.iae * np.trapezoid(abs_error, time)

        # ITAE
        if self.cost_spec.itae > 0:
            cost += self.cost_spec.itae * np.trapezoid(time * abs_error, time)

        # ISE
        if self.cost_spec.ise > 0:
            cost += self.cost_spec.ise * np.trapezoid(error ** 2, time)

        # Overshoot penalty
        if self.cost_spec.overshoot_above > 0:
            peak = np.max(y)
            if self.setpoint != 0:
                os_pct = max(0.0, (peak - self.setpoint) / abs(self.setpoint) * 100.0)
            else:
                os_pct = max(0.0, peak * 100.0)
            excess = max(0.0, os_pct - self.objective.max_overshoot_pct)
            cost += self.cost_spec.overshoot_above * excess

        # Settling time penalty
        if self.cost_spec.settling_pct_above > 0 and self.objective.max_settling_time is not None:
            settle = _settling_time(time, y, self.setpoint, 0.02)
            excess = max(0.0, settle - self.objective.max_settling_time)
            cost += self.cost_spec.settling_pct_above * excess

        # Control effort (total variation)
        if self.cost_spec.du > 0:
            tv = np.sum(np.abs(np.diff(u)))
            cost += self.cost_spec.du * tv

        return float(cost)

    def evaluate_array(self, x: np.ndarray) -> float:
        """Evaluate from a flat array ``[kp, ki, kd]`` (optimizer interface)."""
        gains = PIDGains(kp=float(x[0]), ki=float(x[1]), kd=float(x[2]))
        return self.evaluate(gains)


def _settling_time(
    time: np.ndarray, y: np.ndarray, target: float, band: float,
) -> float:
    """Time at which ``y`` permanently enters ``target ± band*|target|``."""
    tol = band * abs(target) if target != 0 else band
    within = np.abs(y - target) <= tol
    # Walk backward to find last exit
    for i in range(len(within) - 1, -1, -1):
        if not within[i]:
            return float(time[min(i + 1, len(time) - 1)])
    return 0.0


__all__ = ["CostSpec", "CostEvaluator"]
