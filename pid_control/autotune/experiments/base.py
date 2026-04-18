"""Experiment protocol (PLAN.md §4 stage 1, task T1.2).

An *experiment* turns either (a) a live plant or (b) a pre-recorded
dataset into an :class:`ExperimentRecord` — the minimal input an
identifier needs.  The protocol is deliberately narrow so new excitation
designs (relay, step, chirp, PRBS, …) can be plugged in without
touching the rest of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from pid_control.autotune.types import (
    ActuatorLimits,
    DataQuality,
    Trajectory,
    Warning,
)


@dataclass(frozen=True)
class ExperimentRecord:
    """Raw time series collected by an :class:`Experiment`.

    All arrays share length ``n_samples``; ``sample_time`` is the fixed
    step in seconds.  ``setpoint`` is optional because open-loop
    excitations (step, chirp) do not close the loop.  When an experiment
    is a simple wrapper over historical data, ``quality`` is populated by
    the experiment itself; otherwise the identifier / diagnostics stage
    computes it.
    """

    time: np.ndarray
    input: np.ndarray
    output: np.ndarray
    setpoint: Optional[np.ndarray] = None
    sample_time: float = 0.0
    operating_point_input: float = 0.0
    operating_point_output: float = 0.0
    actuator: Optional[ActuatorLimits] = None
    quality: Optional[DataQuality] = None
    warnings: Tuple[Warning, ...] = ()
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        lens = {len(self.time), len(self.input), len(self.output)}
        if self.setpoint is not None:
            lens.add(len(self.setpoint))
        if len(lens) != 1:
            raise ValueError(
                f"ExperimentRecord arrays must share length, got {lens}"
            )
        if self.sample_time <= 0:
            raise ValueError("sample_time must be positive")

    @property
    def n_samples(self) -> int:
        return len(self.time)

    def as_trajectory(self) -> Trajectory:
        """Return a :class:`Trajectory` view (fills a zero setpoint when
        absent — the trajectory type always expects one)."""
        sp = (
            self.setpoint
            if self.setpoint is not None
            else np.zeros_like(self.output)
        )
        return Trajectory(
            time=self.time, setpoint=sp,
            measurement=self.output, control=self.input,
        )


@runtime_checkable
class Experiment(Protocol):
    """Excitation protocol.

    Implementations either (a) drive a live plant to generate an
    :class:`ExperimentRecord` or (b) wrap a previously recorded dataset.

    The caller is responsible for any safety wrappers (rate limiters,
    abort thresholds): see ``pid_control.autotune.experiments.safety``.
    """

    name: str

    def run(self, plant: Any | None = None) -> ExperimentRecord: ...


__all__ = ["Experiment", "ExperimentRecord"]
