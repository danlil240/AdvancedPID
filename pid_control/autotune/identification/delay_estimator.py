"""Cross-correlation delay estimator (PLAN.md T2.6).

Estimates the pure time delay (dead time / transport delay) between an
input signal and the corresponding output by computing the normalised
cross-correlation and locating its peak.

The estimated delay can seed the θ₀ initial guess in FOPDT / SOPDT
identifiers, improving convergence on noisy data.
"""

from __future__ import annotations

import numpy as np


def estimate_delay(
    time: np.ndarray,
    input_signal: np.ndarray,
    output_signal: np.ndarray,
    *,
    sample_time: float | None = None,
    max_delay_fraction: float = 0.5,
) -> float:
    """Return the estimated dead-time in seconds.

    Parameters
    ----------
    time : array
        Time vector (must be uniformly spaced).
    input_signal, output_signal : array
        Input (controller output / excitation) and measured output.
    sample_time : float, optional
        If *None*, inferred from ``time``.
    max_delay_fraction : float
        Maximum delay expressed as a fraction of the total record length.
        Cross-correlation peaks beyond this lag are ignored (guards
        against spurious secondary peaks in oscillatory data).

    Returns
    -------
    float
        Estimated delay in seconds (≥ 0).
    """
    u = np.asarray(input_signal, dtype=np.float64)
    y = np.asarray(output_signal, dtype=np.float64)
    n = len(u)
    if n < 4:
        return 0.0

    dt = float(sample_time) if sample_time else float(np.median(np.diff(time)))
    if dt <= 0:
        return 0.0

    # Work in deviation from mean so DC offset doesn't dominate
    u_dev = u - np.mean(u)
    y_dev = y - np.mean(y)

    # Normalise
    norm_u = np.linalg.norm(u_dev)
    norm_y = np.linalg.norm(y_dev)
    if norm_u < 1e-12 or norm_y < 1e-12:
        return 0.0  # flat signal — no meaningful delay

    u_norm = u_dev / norm_u
    y_norm = y_dev / norm_y

    # Full cross-correlation via FFT (output lags input → positive lag)
    corr = np.correlate(y_norm, u_norm, mode="full")
    # corr[n-1] is zero-lag; positive lags are corr[n-1:]
    positive_lags = corr[n - 1 :]

    # Restrict search to [0, max_delay_fraction * record_length]
    max_lag = max(1, int(max_delay_fraction * n))
    search = positive_lags[: max_lag + 1]

    peak_idx = int(np.argmax(search))
    delay = peak_idx * dt
    return max(0.0, delay)
