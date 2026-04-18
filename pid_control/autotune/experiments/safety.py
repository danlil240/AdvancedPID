"""Safety wrapper for live experiments (PLAN.md T3.5).

Decorates any :class:`Experiment` with amplitude clamping, rate limiting,
and abort-on-exceedance logic.  All live experiments should be composed
with this wrapper before running on real hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from pid_control.autotune.experiments.base import ExperimentRecord
from pid_control.autotune.types import (
    ActuatorLimits,
    Severity,
    Warning,
    WarningCode,
)


class SafePlantWrapper:
    """Wraps a plant object with safety clamping on the control input.

    Intercepts ``update(u, dt)`` calls, applying:
    - Amplitude clamping to ``[lo, hi]``.
    - Rate limiting: ``|u - u_prev| ≤ rate_limit * dt``.
    - Output abort: raises ``RuntimeError`` if ``|y|`` exceeds ``abort_limit``.
    """

    def __init__(
        self,
        plant: Any,
        actuator: Optional[ActuatorLimits] = None,
        abort_limit: Optional[float] = None,
    ) -> None:
        self._plant = plant
        self._actuator = actuator or ActuatorLimits()
        self._abort_limit = abort_limit
        self._u_prev = 0.0

    def update(self, u: float, dt: float = 0.0) -> float:
        # Amplitude clamp
        lo, hi = self._actuator.lower, self._actuator.upper
        u = max(lo, min(hi, u))

        # Rate limit
        if self._actuator.rate_limit is not None and dt > 0:
            max_du = self._actuator.rate_limit * dt
            u = max(self._u_prev - max_du, min(self._u_prev + max_du, u))

        self._u_prev = u
        # Adapt to plant's update signature
        import inspect
        sig = inspect.signature(self._plant.update)
        params = [p for p in sig.parameters.values() if p.name != 'self']
        if len(params) >= 2:
            y = float(self._plant.update(u, dt))
        else:
            y = float(self._plant.update(u))

        if self._abort_limit is not None and abs(y) > self._abort_limit:
            raise RuntimeError(
                f"Safety abort: plant output {y:.3g} exceeds limit ±{self._abort_limit:.3g}"
            )

        return y

    def __getattr__(self, name: str) -> Any:
        return getattr(self._plant, name)


@dataclass
class SafeExperiment:
    """Wraps another experiment with safety constraints.

    Usage::

        safe = SafeExperiment(
            inner=RelayExperiment(amplitude=0.5),
            actuator=ActuatorLimits(lower=-10, upper=10, rate_limit=5.0),
            abort_limit=50.0,
        )
        record = safe.run(plant)
    """

    inner: Any  # Experiment protocol
    actuator: Optional[ActuatorLimits] = None
    abort_limit: Optional[float] = None
    name: str = "safe"

    def run(self, plant: Any = None) -> ExperimentRecord:
        safe_plant = SafePlantWrapper(
            plant, actuator=self.actuator, abort_limit=self.abort_limit,
        )
        return self.inner.run(safe_plant)


__all__ = ["SafeExperiment", "SafePlantWrapper"]
