"""AMIGO (Åström-Hägglund) PID tuning rule (PLAN.md T4.1).

AMIGO is the reference balanced rule used in modern control textbooks:
it explicitly targets a maximum sensitivity ``M_s ≈ 1.4`` (robust)
while maintaining a reasonable response speed.
"""

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
class AMIGORule:
    """Åström-Hägglund AMIGO rule for FOPDT plants.

    Reference: Åström, K.J. & Hägglund, T. (2004) — "Revisiting the
    Ziegler-Nichols step response method for PID control."
    """

    name: str = "amigo"

    def apply(
        self,
        identification: IdentificationResult,
        objective: Objective,
        actuator: ActuatorLimits,
    ) -> PIDGains:
        K, tau, theta = to_fopdt_triplet(identification.model)
        K = K or 1e-9
        theta = max(theta, 1e-9)
        # AMIGO formulae (robust PID, Ms target ≈ 1.4)
        kp_mag = (1.0 / abs(K)) * (
            0.2 + 0.45 * (tau / theta)
        )
        ti = (theta * (0.4 * theta + 0.8 * tau)) / (theta + 0.1 * tau)
        td = (0.5 * theta * tau) / (0.3 * theta + tau)
        kp = kp_mag if K > 0 else -kp_mag
        ki = kp / ti if ti > 0 else 0.0
        kd = kp * td
        return PIDGains(kp=float(kp), ki=float(ki), kd=float(kd))


__all__ = ["AMIGORule"]
