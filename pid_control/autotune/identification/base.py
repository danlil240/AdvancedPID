"""Identifier protocol (PLAN.md §4 stage 2, task T1.2)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pid_control.autotune.experiments.base import ExperimentRecord
from pid_control.autotune.types import IdentificationResult


@runtime_checkable
class Identifier(Protocol):
    """Turn an :class:`ExperimentRecord` into an :class:`IdentificationResult`.

    Identifiers MUST:
      * never raise for empty/flat data – emit :class:`~pid_control.autotune.types.WarningCode.E_DATA_FLAT`
        on :attr:`IdentificationResult.warnings` and return a
        best-effort model (or the caller can consult the paired
        :class:`~pid_control.autotune.types.DataQuality` to decide whether
        to short-circuit).
      * populate ``fit_quality_r2`` and, when possible, ``aic``/``bic``
        so model selection can compare alternatives fairly.
    """

    name: str

    def identify(self, record: ExperimentRecord) -> IdentificationResult: ...


__all__ = ["Identifier"]
