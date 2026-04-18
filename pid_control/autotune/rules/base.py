"""Analytical tuning-rule protocol (PLAN.md §4 stage 3a, task T1.2)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pid_control.autotune.types import (
    ActuatorLimits,
    IdentificationResult,
    Objective,
    PIDGains,
)


@runtime_checkable
class TuningRule(Protocol):
    """Convert an identified model into an initial :class:`PIDGains` guess.

    Rules are pure functions: no hidden state, no I/O, no optimisation.
    They provide the *starting point* the numerical tuner refines.
    """

    name: str

    def apply(
        self,
        identification: IdentificationResult,
        objective: Objective,
        actuator: ActuatorLimits,
    ) -> PIDGains: ...


__all__ = ["TuningRule"]
