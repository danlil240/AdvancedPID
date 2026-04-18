"""Numerical tuners (PLAN.md §4 stage 3b)."""

from pid_control.autotune.tuning.base import Tuner, TunerOutcome
from pid_control.autotune.tuning.cost import CostEvaluator, CostSpec
from pid_control.autotune.tuning.de import DETuner, NelderMeadTuner
from pid_control.autotune.tuning.cmaes import CMAESTuner

__all__ = [
    "Tuner",
    "TunerOutcome",
    "CostEvaluator",
    "CostSpec",
    "DETuner",
    "NelderMeadTuner",
    "CMAESTuner",
]

# BOTuner requires scikit-learn (optional dependency).
try:
    from pid_control.autotune.tuning.bo import BOTuner
    __all__.append("BOTuner")
except ImportError:
    pass
