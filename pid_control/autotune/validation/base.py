"""Validator protocol (PLAN.md §4 stage 4, task T1.2).

Validators run *after* numerical tuning and either attach warnings (for
soft issues like low phase margin) or downgrade the pipeline status to
``FAILED`` (for hard issues like instability on the identified model).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, Tuple, runtime_checkable

from pid_control.autotune.types import (
    ActuatorLimits,
    IdentificationResult,
    MarginReport,
    Objective,
    PerformanceReport,
    PIDGains,
    Status,
    Warning,
)


@dataclass(frozen=True)
class ValidationOutcome:
    """Aggregate of a single validator's findings.

    Validators compose additively: the façade concatenates all warnings
    and takes the strictest ``status`` value.
    """

    status: Status
    warnings: Tuple[Warning, ...] = ()
    margins: Optional[MarginReport] = None
    performance: Optional[PerformanceReport] = None


@runtime_checkable
class Validator(Protocol):
    """Post-tune safety / quality gate."""

    name: str

    def validate(
        self,
        identification: IdentificationResult,
        gains: PIDGains,
        objective: Objective,
        actuator: ActuatorLimits,
    ) -> ValidationOutcome: ...


__all__ = ["Validator", "ValidationOutcome"]
