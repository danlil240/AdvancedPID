"""Back-compatibility shim for legacy ``AutotuneFromData`` users (T1.4).

Wraps the new :class:`~pid_control.autotune.api.PIDAutotuner` pipeline
behind the old ``AutotuneFromData`` interface so existing code keeps
working while emitting a :class:`DeprecationWarning`.

.. deprecated:: 0.2.0
   Use :class:`pid_control.autotune.PIDAutotuner` instead.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from pid_control.autotune.api import PIDAutotuner
from pid_control.autotune.types import Objective, Status


@dataclass
class AutotuneFromDataResult:
    """Mimics the legacy result shape."""

    identification: Any
    initial_gains: Dict[str, float]
    optimized_gains: Dict[str, float]
    improvement: float
    new_result: Any  # the real TuneResult

    def summary(self) -> str:
        if hasattr(self.new_result, "report"):
            return self.new_result.report("text")
        return str(self.optimized_gains)


class AutotuneFromDataCompat:
    """Drop-in replacement for the legacy ``AutotuneFromData``.

    .. deprecated:: 0.2.0
       Prefer ``PIDAutotuner.from_csv(path).tune()``.
    """

    def __init__(
        self,
        csv_path: str,
        time_col: str = "timestamp",
        input_col: str = "output",
        output_col: str = "measurement",
        setpoint_col: Optional[str] = "setpoint",
    ):
        warnings.warn(
            "AutotuneFromData is deprecated. "
            "Use pid_control.autotune.PIDAutotuner.from_csv() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._csv_path = csv_path
        self._columns = {
            "time": time_col,
            "input": input_col,
            "measurement": output_col,
        }

    def autotune(
        self,
        model_type: Any = None,
        tuning_rule: str = "ziegler_nichols",
        optimizer: str = "differential_evolution",
        bounds_scale: float = 2.0,
        max_iterations: int = 50,
        cost_function: Optional[Callable] = None,
        requirements: Any = None,
        prompt_for_requirements: bool = False,
    ) -> AutotuneFromDataResult:
        """Run the new pipeline and translate result to the legacy shape."""
        autotuner = PIDAutotuner.from_csv(self._csv_path, columns=self._columns)

        # Map legacy tuning_rule name
        if tuning_rule in ("cohen_coon", "cohen-coon"):
            from pid_control.autotune.rules.cohen_coon import CohenCoonRule
            autotuner.set_rule(CohenCoonRule())
        elif tuning_rule in ("imc", "IMC"):
            from pid_control.autotune.rules.imc import IMCRule
            autotuner.set_rule(IMCRule())
        elif tuning_rule in ("amigo", "AMIGO"):
            from pid_control.autotune.rules.amigo import AMIGORule
            autotuner.set_rule(AMIGORule())

        result = autotuner.tune()

        optimized = {
            "kp": float(result.gains.kp),
            "ki": float(result.gains.ki),
            "kd": float(result.gains.kd),
        }

        # Compute a rough "improvement" percentage from initial rule vs DE
        initial = {"kp": 0.0, "ki": 0.0, "kd": 0.0}
        if result.meta is not None and hasattr(result, "artifacts"):
            # Best effort: pull initial from the rule stage
            pass

        improvement = 0.0
        if (
            result.artifacts is not None
            and result.artifacts.cost_history is not None
            and len(result.artifacts.cost_history) >= 2
        ):
            c0 = float(result.artifacts.cost_history[0])
            cf = float(result.artifacts.cost_history[-1])
            if c0 > 0:
                improvement = (c0 - cf) / c0 * 100.0

        return AutotuneFromDataResult(
            identification=result.identification,
            initial_gains=initial,
            optimized_gains=optimized,
            improvement=improvement,
            new_result=result,
        )
