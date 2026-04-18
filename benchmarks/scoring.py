"""Scoring helpers for the benchmark harness.

Given a closed-loop simulation trace this module computes the canonical
metrics the benchmark compares across runs:

  * Time-domain: rise time, settling time (2%, 5%), overshoot, SSE
  * Integral:    IAE, ISE, ITAE
  * Effort:      total variation, RMS, peak |u|, saturation fraction
  * Stability:   a coarse stable/oscillating flag

The functions are deliberately dependency-free beyond numpy so the baseline
snapshot can be regenerated on a minimal environment.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple

import numpy as np


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(fn(y, x))


@dataclass(frozen=True)
class TraceScore:
    iae: float
    ise: float
    itae: float
    rise_time: float
    settling_time_2pct: float
    overshoot_percent: float
    steady_state_error: float
    control_tv: float
    control_rms: float
    control_peak: float
    saturation_fraction: float
    stable: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _settling_time(t: np.ndarray, y: np.ndarray, sp: float, frac: float) -> float:
    band = max(frac * abs(sp), frac)
    outside = np.where(np.abs(y - sp) > band)[0]
    if outside.size == 0:
        return 0.0
    last = outside[-1]
    return float(t[min(last + 1, len(t) - 1)])


def score_trace(
    t: np.ndarray,
    sp: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    actuator: Optional[Tuple[float, float]] = None,
) -> TraceScore:
    t = np.asarray(t, dtype=float)
    sp = np.asarray(sp, dtype=float)
    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)
    err = sp - y
    abs_err = np.abs(err)
    sq_err = err * err

    sp_end = float(sp[-1])
    delta = sp_end - float(y[0])

    if abs(delta) < 1e-12:
        rise_time = 0.0
        overshoot = 0.0
    else:
        y_norm = (y - y[0]) / delta
        # rise time 10%-90%
        if np.any(y_norm >= 0.1) and np.any(y_norm >= 0.9):
            t10 = float(np.interp(0.1, y_norm, t))
            t90 = float(np.interp(0.9, y_norm, t))
            rise_time = max(0.0, t90 - t10)
        else:
            rise_time = float(t[-1] - t[0])

        if delta > 0:
            peak = float(np.max(y))
            overshoot = max(0.0, (peak - sp_end) / delta * 100.0)
        else:
            peak = float(np.min(y))
            overshoot = max(0.0, (sp_end - peak) / abs(delta) * 100.0)

    n_tail = max(5, len(y) // 20)
    ss = float(np.mean(y[-n_tail:]))
    sse = sp_end - ss

    # Stability heuristic: tail coefficient of variation.
    tail_std = float(np.std(y[-n_tail:]))
    tail_cv = tail_std / (abs(ss) if abs(ss) > 1e-9 else 1.0)
    stable = bool(np.isfinite(ss)) and tail_cv < 0.05 and abs_err[-1] < 1.5 * abs(delta or 1.0)

    # Actuator effort
    tv = float(np.sum(np.abs(np.diff(u))))
    rms = float(np.sqrt(np.mean(u * u)))
    peak_u = float(np.max(np.abs(u)))
    sat_frac = 0.0
    if actuator is not None:
        lo, hi = actuator
        at_limit = (u <= lo + 1e-9) | (u >= hi - 1e-9)
        sat_frac = float(np.mean(at_limit))

    return TraceScore(
        iae=_trapezoid(abs_err, t),
        ise=_trapezoid(sq_err, t),
        itae=_trapezoid(t * abs_err, t),
        rise_time=rise_time,
        settling_time_2pct=_settling_time(t, y, sp_end, 0.02),
        overshoot_percent=overshoot,
        steady_state_error=sse,
        control_tv=tv,
        control_rms=rms,
        control_peak=peak_u,
        saturation_fraction=sat_frac,
        stable=stable,
    )


__all__ = ["TraceScore", "score_trace"]
