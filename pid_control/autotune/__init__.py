"""Unified PID autotune pipeline (PLAN.md §4).

Public surface: :class:`PIDAutotuner` façade plus all typed result
objects and stage protocols.
"""

from pid_control.autotune.api import PIDAutotuner
from pid_control.autotune.types import (
    ActuatorLimits,
    Artifacts,
    Confidence,
    DataQuality,
    IdentificationResult,
    MarginReport,
    ModelType,
    Objective,
    PerformanceReport,
    PIDGains,
    Severity,
    Status,
    Trajectory,
    TransferFunctionModel,
    TuneMeta,
    TuneResult,
    Warning,
    WarningCode,
    merge_warnings,
)

__all__ = [
    "PIDAutotuner",
    "ActuatorLimits",
    "Artifacts",
    "Confidence",
    "DataQuality",
    "IdentificationResult",
    "MarginReport",
    "ModelType",
    "Objective",
    "PerformanceReport",
    "PIDGains",
    "Severity",
    "Status",
    "Trajectory",
    "TransferFunctionModel",
    "TuneMeta",
    "TuneResult",
    "Warning",
    "WarningCode",
    "merge_warnings",
]
