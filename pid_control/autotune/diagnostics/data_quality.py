"""Data-quality diagnostics (PLAN.md §4, task T2.5).

The autotune pipeline MUST refuse to silently fit obviously bad data
(this is the user's non-negotiable rule).  Rather than letting an
identifier produce a nonsense R² on a double-integrator or flat dataset,
we detect the common pathologies up-front and return a
:class:`~pid_control.autotune.types.DataQuality` object whose
``status`` is consulted by the façade.

Design goals:
  * pure-NumPy, zero dependencies beyond the typed API
  * deterministic — identical arrays always produce identical findings
  * small and well-documented: each check has a single-paragraph
    docstring and a dedicated warning code so reports are explainable.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from pid_control.autotune.types import (
    DataQuality,
    Severity,
    Status,
    Warning,
    WarningCode,
)

# Detection thresholds (exposed so tests and advanced users can tune them).
MIN_SAMPLES = 20
MIN_DURATION_SECONDS = 1.0
FLAT_INPUT_RELATIVE_RANGE = 1e-6
STEADY_STATE_WINDOW_FRACTION = 0.15  # last 15 %
STEADY_STATE_SLOPE_TOLERANCE = 0.05
# fraction of initial->final swing still being traversed per unit time


def assess(
    time: np.ndarray,
    input_signal: np.ndarray,
    output: np.ndarray,
    sample_time: float | None = None,
) -> DataQuality:
    """Score a dataset and return a :class:`DataQuality`.

    The ``status`` field is the primary signal:
      * :attr:`Status.OK`      — fit away
      * :attr:`Status.WARNING` — fit, but flag concerns on the result
      * :attr:`Status.FAILED`  — refuse to fit; pipeline must short-circuit
    """
    t = np.asarray(time, dtype=float)
    u = np.asarray(input_signal, dtype=float)
    y = np.asarray(output, dtype=float)
    n = len(t)

    warnings: List[Warning] = []

    # --- Length / duration ---------------------------------------------
    if n < MIN_SAMPLES:
        warnings.append(
            Warning(
                code=WarningCode.E_TOO_SHORT,
                severity=Severity.ERROR,
                message=(
                    f"Dataset has only {n} samples; minimum is {MIN_SAMPLES}"
                ),
                stage="data_quality",
                context={"n_samples": n, "min_samples": MIN_SAMPLES},
            )
        )

    dt = float(sample_time) if sample_time else _infer_sample_time(t)
    duration = float(t[-1] - t[0]) if n > 1 else 0.0
    if duration < MIN_DURATION_SECONDS:
        warnings.append(
            Warning(
                code=WarningCode.E_TOO_SHORT,
                severity=Severity.ERROR,
                message=(
                    f"Dataset spans {duration:.3g}s; minimum is "
                    f"{MIN_DURATION_SECONDS}s"
                ),
                stage="data_quality",
                context={"duration": duration},
            )
        )

    # --- Excitation energy ---------------------------------------------
    u_range = float(np.ptp(u)) if u.size else 0.0
    u_scale = max(abs(np.mean(u)), 1.0)
    excitation = _excitation_energy(u)
    if u_range <= FLAT_INPUT_RELATIVE_RANGE * u_scale:
        warnings.append(
            Warning(
                code=WarningCode.E_DATA_FLAT,
                severity=Severity.ERROR,
                message=(
                    "Input signal is effectively constant "
                    f"(range={u_range:.3g}); cannot identify a plant"
                ),
                stage="data_quality",
                context={"input_range": u_range},
            )
        )

    # --- Steady-state check (excludes obvious integrators) --------------
    has_ss, slope_fraction = _has_steady_state(t, y)
    if not has_ss:
        warnings.append(
            Warning(
                code=WarningCode.E_NO_STEADY_STATE,
                severity=Severity.ERROR,
                message=(
                    "Output does not settle (possible integrator or unstable "
                    "plant); refusing to fit a first/second-order model"
                ),
                stage="data_quality",
                context={"tail_slope_fraction": slope_fraction},
            )
        )

    # --- Noise / SNR heuristic -----------------------------------------
    snr_db = _estimate_snr_db(y)
    if snr_db is not None and snr_db < 6.0:
        warnings.append(
            Warning(
                code=WarningCode.W_HIGH_NOISE,
                severity=Severity.WARNING,
                message=f"Low SNR estimate {snr_db:.1f} dB; fit may be unreliable",
                stage="data_quality",
                context={"snr_db": snr_db},
            )
        )

    status = _aggregate_status(warnings)
    return DataQuality(
        status=status,
        snr_db=snr_db,
        excitation_energy=excitation,
        has_steady_state=has_ss,
        n_samples=n,
        sample_time=dt,
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_sample_time(t: np.ndarray) -> float:
    if len(t) < 2:
        return 0.0
    return float(np.median(np.diff(t)))


def _excitation_energy(u: np.ndarray) -> float:
    """Variance of the input signal about its mean — a rough proxy for how
    much the experiment actually moved the plant away from its operating
    point.  Returned in *signal units squared* so it is commensurate with
    the dataset; the caller normalises it as needed.
    """
    if u.size == 0:
        return 0.0
    return float(np.var(u))


def _has_steady_state(t: np.ndarray, y: np.ndarray) -> Tuple[bool, float]:
    """Return (has_steady_state, tail_slope_fraction).

    We look at the last ``STEADY_STATE_WINDOW_FRACTION`` of the trace,
    fit a line, and compare its slope against the total output swing.
    If the tail is still moving by more than a fraction of the full
    swing per unit time, we declare "no steady state".
    """
    n = len(y)
    if n < MIN_SAMPLES:
        return False, float("inf")

    window = max(int(n * STEADY_STATE_WINDOW_FRACTION), 5)
    t_tail = t[-window:]
    y_tail = y[-window:]
    if t_tail[-1] == t_tail[0]:
        return False, float("inf")

    # Robust line fit (least-squares on the tail window)
    slope = np.polyfit(t_tail, y_tail, 1)[0]
    span = float(np.ptp(y)) or max(abs(np.mean(y)), 1.0)
    duration_tail = float(t_tail[-1] - t_tail[0])
    slope_fraction = abs(slope) * duration_tail / span
    return slope_fraction < STEADY_STATE_SLOPE_TOLERANCE, float(slope_fraction)


def _estimate_snr_db(y: np.ndarray) -> float | None:
    """Crude SNR estimate.

    Signal power = variance of the output.  Noise power = variance of
    the high-pass-filtered output (second difference).  This is
    conservative: slow drifts bleed into "signal".  We guard against
    divisions by zero and return ``None`` when the dataset is too short
    or flat for the estimate to be meaningful.
    """
    if len(y) < 5:
        return None
    signal_var = float(np.var(y))
    if signal_var <= 0:
        return None
    hp = np.diff(y, n=2)
    if hp.size == 0:
        return None
    noise_var = float(np.var(hp) / 6.0)  # 2nd-diff variance multiplier
    if noise_var <= 0:
        return None
    ratio = signal_var / noise_var
    if ratio <= 0:
        return None
    return float(10.0 * math.log10(ratio))


def _aggregate_status(warnings: List[Warning]) -> Status:
    if any(w.severity is Severity.ERROR for w in warnings):
        return Status.FAILED
    if any(w.severity is Severity.WARNING for w in warnings):
        return Status.WARNING
    return Status.OK


__all__ = ["assess"]
