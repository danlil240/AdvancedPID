"""Cohen-Coon tuning rule (PLAN.md T4.1).

Cohen-Coon handles plants with a larger ``θ/τ`` ratio (0.25 – 1) more
gracefully than classical Ziegler-Nichols by adjusting the integral
and derivative constants based on that ratio.
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
class CohenCoonRule:
    name: str = "cohen_coon"

    def apply(
        self,
        identification: IdentificationResult,
        objective: Objective,
        actuator: ActuatorLimits,
    ) -> PIDGains:
        K, tau, theta = to_fopdt_triplet(identification.model)
        theta = max(theta, 1e-6)
        K = K or 1e-9
        r = theta / tau

        kp_mag = (1.0 / abs(K)) * (tau / theta) * ((4.0 / 3.0) + r / 4.0)
        ti = theta * (32 + 6 * r) / (13 + 8 * r)
        td = theta * 4 / (11 + 2 * r)
        kp = kp_mag if K > 0 else -kp_mag
        ki = kp / ti if ti > 0 else 0.0
        kd = kp * td
        return PIDGains(kp=float(kp), ki=float(ki), kd=float(kd))


__all__ = ["CohenCoonRule"]
