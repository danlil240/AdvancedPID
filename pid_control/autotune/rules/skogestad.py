"""Skogestad SIMC tuning rule (PLAN.md T4.1).

SIMC ("Simple IMC") is Skogestad's practical recipe that produces a
well-behaved PI / PID tuning from a FOPDT model with a single tuning
knob ``τ_c``.  The rule is widely used in industry because of its
excellent robustness/performance trade-off.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pid_control.autotune.rules._utils import to_fopdt_triplet
from pid_control.autotune.types import (
    ActuatorLimits,
    IdentificationResult,
    Objective,
    PIDGains,
)


@dataclass(frozen=True)
class SIMCRule:
    """Skogestad's SIMC PID rule for FOPDT plants.

    Parameters
    ----------
    closed_loop_constant
        Desired closed-loop time constant ``τ_c``.  Default ``τ_c = θ``
        (Skogestad's "fast response" choice, reasonable robustness).
        Use ``τ_c = 2·θ`` or larger for a more robust controller.
    """

    closed_loop_constant: Optional[float] = None
    name: str = "simc"

    def apply(
        self,
        identification: IdentificationResult,
        objective: Objective,
        actuator: ActuatorLimits,
    ) -> PIDGains:
        K, tau, theta = to_fopdt_triplet(identification.model)
        K = K or 1e-9
        theta = max(theta, 1e-9)
        tau_c = (
            float(self.closed_loop_constant)
            if self.closed_loop_constant is not None else theta
        )
        kp_mag = tau / (abs(K) * (tau_c + theta))
        ti = min(tau, 4.0 * (tau_c + theta))
        # SIMC PID ≈ PI; add a small Kd only when τ is much larger than θ.
        td = 0.0
        kp = kp_mag if K > 0 else -kp_mag
        ki = kp / ti if ti > 0 else 0.0
        kd = kp * td
        return PIDGains(kp=float(kp), ki=float(ki), kd=float(kd))


# Back-compat alias
SkogestadRule = SIMCRule

__all__ = ["SIMCRule", "SkogestadRule"]
