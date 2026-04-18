"""Wrap pre-recorded (CSV / array) data as an :class:`Experiment`.

This is the common entry point for the *offline* side of the autotune
pipeline.  It performs no excitation of its own – it simply validates
the input arrays, runs :func:`pid_control.autotune.diagnostics.data_quality.assess`
and emits a fully-populated :class:`ExperimentRecord`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from pid_control.autotune.diagnostics import data_quality
from pid_control.autotune.experiments.base import (
    Experiment,
    ExperimentRecord,
)
from pid_control.autotune.types import ActuatorLimits


# Column aliases accepted by :func:`load_csv`.  Each key maps to a list of
# case-insensitive names that are tried in order.  Users can override via
# ``columns=`` keyword if their schema is unusual.
DEFAULT_COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "time": ("time", "timestamp", "t", "secs", "seconds"),
    "input": ("input", "u", "control", "output",  # 'output' kept for
              "control_signal", "mv"),             # library-CSVLogger compat
    "measurement": ("measurement", "y", "pv", "process_variable"),
    "setpoint": ("setpoint", "sp", "reference", "ref"),
}


@dataclass
class FromDataExperiment:
    """Wrap pre-recorded data as an :class:`Experiment`.

    Parameters
    ----------
    time, input_signal, output : numpy arrays of equal length
    setpoint : optional reference series
    sample_time : overrides the inferred ``median(diff(time))``
    actuator : optional :class:`ActuatorLimits` describing the envelope
    operating_point_input, operating_point_output : float
        Baseline values (used by relay/closed-loop excitation; kept here
        for symmetry with live experiments).
    """

    time: np.ndarray
    input_signal: np.ndarray
    output: np.ndarray
    setpoint: Optional[np.ndarray] = None
    sample_time: Optional[float] = None
    actuator: Optional[ActuatorLimits] = None
    operating_point_input: float = 0.0
    operating_point_output: float = 0.0
    name: str = "from_data"

    def run(self, plant: Any | None = None) -> ExperimentRecord:  # noqa: ARG002
        t = np.asarray(self.time, dtype=float)
        u = np.asarray(self.input_signal, dtype=float)
        y = np.asarray(self.output, dtype=float)
        sp = (
            np.asarray(self.setpoint, dtype=float)
            if self.setpoint is not None else None
        )
        dt = (
            float(self.sample_time)
            if self.sample_time is not None
            else _infer_dt(t)
        )
        if dt <= 0:
            raise ValueError("Cannot infer positive sample_time from data")

        quality = data_quality.assess(t, u, y, sample_time=dt)

        return ExperimentRecord(
            time=t, input=u, output=y, setpoint=sp,
            sample_time=dt,
            operating_point_input=float(self.operating_point_input),
            operating_point_output=float(self.operating_point_output),
            actuator=self.actuator,
            quality=quality,
            warnings=quality.warnings,
            meta={"source": "from_data", "name": self.name},
        )


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def load_csv(
    path: str | Path,
    columns: Optional[Dict[str, str]] = None,
) -> FromDataExperiment:
    """Load a dataset from a CSV file and return a :class:`FromDataExperiment`.

    The loader is intentionally forgiving: it tries a handful of common
    column names (see :data:`DEFAULT_COLUMN_ALIASES`) and accepts either
    "native" experimental CSVs (``time, input, output``) or the library's
    own logger format (``timestamp, output, measurement, setpoint``).

    Pass ``columns={"time": "t", "input": "cv", ...}`` to force a
    specific mapping.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    rows: list[list[str]] = []
    with p.open("r", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        for row in reader:
            if row:
                rows.append(row)

    header_clean = [c.strip() for c in header]
    resolved = _resolve_columns(header_clean, columns)

    arr = np.array(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != len(header_clean):
        raise ValueError(
            f"CSV at {p} has inconsistent column count"
        )

    time_arr = arr[:, resolved["time"]]
    input_arr = arr[:, resolved["input"]]
    out_arr = arr[:, resolved["measurement"]]
    sp_arr = (
        arr[:, resolved["setpoint"]]
        if resolved.get("setpoint") is not None else None
    )

    return FromDataExperiment(
        time=time_arr,
        input_signal=input_arr,
        output=out_arr,
        setpoint=sp_arr,
        name=p.stem,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _infer_dt(t: np.ndarray) -> float:
    if len(t) < 2:
        return 0.0
    diffs = np.diff(t)
    return float(np.median(diffs))


def _resolve_columns(
    header: list[str],
    explicit: Optional[Dict[str, str]],
) -> Dict[str, Optional[int]]:
    lower = [h.lower() for h in header]

    def _find(names: Tuple[str, ...]) -> Optional[int]:
        for name in names:
            if name in lower:
                return lower.index(name)
        return None

    resolved: Dict[str, Optional[int]] = {}
    if explicit:
        for key, colname in explicit.items():
            if colname not in header:
                raise KeyError(
                    f"CSV is missing requested column {colname!r} "
                    f"(header: {header})"
                )
            resolved[key] = header.index(colname)
    for key, names in DEFAULT_COLUMN_ALIASES.items():
        if key in resolved:
            continue
        idx = _find(names)
        if idx is not None:
            resolved[key] = idx

    for required in ("time", "input", "measurement"):
        if resolved.get(required) is None:
            raise KeyError(
                f"CSV header {header} lacks a recognisable "
                f"{required!r} column (aliases: "
                f"{DEFAULT_COLUMN_ALIASES[required]}).  "
                f"Pass the ``columns=`` keyword to override."
            )

    # Avoid time and input pointing at the same column (common with the
    # library's own CSV, which names its logged control signal 'output'
    # while its measurement is 'measurement').
    if resolved["input"] == resolved["time"]:
        raise ValueError(
            "CSV column resolution produced time == input; "
            "please pass columns={'input': <name>, ...} explicitly."
        )

    return resolved


__all__ = ["FromDataExperiment", "load_csv", "DEFAULT_COLUMN_ALIASES"]
