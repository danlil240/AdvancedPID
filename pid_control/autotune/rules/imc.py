"""Internal-Model-Control tuning rule (PLAN.md T4.1).

IMC produces PID gains from a FOPDT model plus a single user-facing
tuning parameter ``λ`` (closed-loop time constant).  A sensible default
``λ = max(τ, 8·θ)`` yields a conservative controller; the user can
trade robustness for speed by shrinking ``λ``.
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
class IMCRule:
    """Internal-model-control PID rule for FOPDT plants.

    Parameters
    ----------
    closed_loop_constant
        Desired closed-loop time constant ``λ``.  When ``None`` (the
        default) the rule uses ``λ = max(τ, 8·θ)`` which is the standard
        "robust" recipe — it yields generous stability margins at the
        cost of a slower response.  Pass a smaller value to trade
        robustness for speed.
    """

    closed_loop_constant: Optional[float] = None
    name: str = "imc"

    def apply(
        self,
        identification: IdentificationResult,
        objective: Objective,
        actuator: ActuatorLimits,
    ) -> PIDGains:
        K, tau, theta = to_fopdt_triplet(identification.model)
        K = K or 1e-9
        lam = (
            float(self.closed_loop_constant)
            if self.closed_loop_constant is not None
            else max(tau, 8.0 * max(theta, 1e-9))
        )
        # IMC-PID for FOPDT (Rivera/Morari, 1986)
        kp_mag = (2 * tau + theta) / (abs(K) * (2 * lam + theta))
        ti = tau + theta / 2.0
        td = tau * theta / (2 * tau + theta) if (2 * tau + theta) > 0 else 0.0
        kp = kp_mag if K > 0 else -kp_mag
        ki = kp / ti if ti > 0 else 0.0
        kd = kp * td
        return PIDGains(kp=float(kp), ki=float(ki), kd=float(kd))


__all__ = ["IMCRule"]
