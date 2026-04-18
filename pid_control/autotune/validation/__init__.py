"""Post-tune validators (PLAN.md §4 stage 4)."""

from pid_control.autotune.validation.base import Validator, ValidationOutcome
from pid_control.autotune.validation.confidence import ConfidenceAggregator
from pid_control.autotune.validation.sim_benchmark import SimBenchmarkValidator
from pid_control.autotune.validation.stability import (
    RobustnessValidator,
    StabilityValidator,
    compute_margins,
)

__all__ = [
    "ValidationOutcome",
    "Validator",
    "ConfidenceAggregator",
    "RobustnessValidator",
    "SimBenchmarkValidator",
    "StabilityValidator",
    "compute_margins",
]
