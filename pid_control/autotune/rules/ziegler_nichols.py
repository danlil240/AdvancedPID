"""Ziegler-Nichols open-loop (reaction-curve) tuning rule (PLAN.md T4.1)."""

from __future__ import annotations

from dataclasses import dataclass

from pid_control.autotune.rules._utils import to_fopdt_triplet
from pid_control.autotune.types import (
    ActuatorLimits,
    IdentificationResult,
    Objective,
    PIDGains,
)


@dataclass(frozen=True)
class ZieglerNicholsRule:
    """Classical Ziegler-Nichols reaction-curve rule for PID.

    For ``G(s) = K·e^{-θs}/(τs+1)``:

    * :math:`K_p = 1.2\\,\\tau/(K\\theta)`
    * :math:`T_i = 2\\theta \\Rightarrow K_i = K_p / T_i`
    * :math:`T_d = 0.5\\theta \\Rightarrow K_d = K_p · T_d`

    Aggressive; expect ~20 % overshoot.  Useful as an initial guess that
    the numerical tuner refines.
    """

    name: str = "ziegler_nichols"

    def apply(
        self,
        identification: IdentificationResult,
        objective: Objective,
        actuator: ActuatorLimits,
    ) -> PIDGains:
        K, tau, theta = to_fopdt_triplet(identification.model)
        # Guard against θ→0 (unrealistic for ZN; clamp to half a sample).
        theta = max(theta, 1e-6)
        K = K or 1e-9
        kp = 1.2 * tau / (abs(K) * theta)
        if K < 0:
            kp = -kp
        ti = 2.0 * theta
        td = 0.5 * theta
        ki = kp / ti if ti > 0 else 0.0
        kd = kp * td
        return PIDGains(kp=float(kp), ki=float(ki), kd=float(kd))


__all__ = ["ZieglerNicholsRule"]
