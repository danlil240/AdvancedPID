"""Numerical tuner protocol (PLAN.md §4 stage 3b, task T1.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from pid_control.autotune.types import (
    ActuatorLimits,
    IdentificationResult,
    Objective,
    PIDGains,
    Status,
    Warning,
)


@dataclass(frozen=True)
class TunerOutcome:
    """Raw output of the numerical tuning stage.

    The full :class:`~pid_control.autotune.types.TuneResult` is assembled
    by the façade after validation runs against this outcome.
    """

    gains: PIDGains
    status: Status
    cost: float
    initial_gains: PIDGains
    initial_cost: float
    iterations: int
    cost_history: Optional[np.ndarray] = None
    warnings: Tuple[Warning, ...] = ()
    meta: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Tuner(Protocol):
    """Refine :class:`PIDGains` against the identified model.

    Tuners MUST be deterministic given their own ``seed`` field (so
    regression tests are stable).  They accept the initial guess from an
    upstream :class:`~pid_control.autotune.rules.base.TuningRule`.
    """

    name: str

    def refine(
        self,
        identification: IdentificationResult,
        initial: PIDGains,
        objective: Objective,
        actuator: ActuatorLimits,
    ) -> TunerOutcome: ...


__all__ = ["Tuner", "TunerOutcome"]
