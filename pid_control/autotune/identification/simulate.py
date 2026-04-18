"""Model-response simulation helpers shared by identifiers and cost
functions (PLAN.md T4.4 deduplication).

Everything in this module is pure-NumPy and deterministic.  The FOPDT
response uses the exact ZOH discretisation; SOPDT uses scipy's discrete
state-space simulator for correctness in the underdamped regime.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from pid_control.autotune.types import ModelType, TransferFunctionModel


def simulate_model(
    model: TransferFunctionModel,
    time: np.ndarray,
    u: np.ndarray,
    y0: float = 0.0,
) -> np.ndarray:
    """Simulate a :class:`TransferFunctionModel` driven by ``u`` at the
    timestamps in ``time``.

    ``time`` must be uniformly spaced; ``u`` must have the same length.
    The output is the (noise-free, disturbance-free) model response.
    """
    if len(time) != len(u):
        raise ValueError("time and u must have equal length")
    if len(time) < 2:
        return np.full_like(time, y0)

    dt = float(np.median(np.diff(time)))
    if dt <= 0:
        raise ValueError("time array must be increasing")

    delay_samples = int(round(model.theta / dt))
    if delay_samples < 0:
        delay_samples = 0
    u_delayed = np.concatenate((np.full(delay_samples, 0.0), u))[: len(u)]

    if model.model_type is ModelType.FOPDT or (
        model.model_type is ModelType.SOPDT and model.tau2 is None
    ):
        return _simulate_fopdt(model.K, model.tau, dt, u_delayed, y0)

    if model.model_type is ModelType.SOPDT:
        return _simulate_sopdt_over(model.K, model.tau, model.tau2 or 0.0,
                                    dt, u_delayed, y0)

    if model.model_type is ModelType.SECOND_ORDER:
        wn = float(model.natural_frequency or 1.0)
        zeta = float(model.damping_ratio or 1.0)
        return _simulate_second_order(model.K, wn, zeta, dt, u_delayed, y0)

    if model.model_type is ModelType.IPDT:
        return _simulate_ipdt(model.K, dt, u_delayed, y0)

    raise ValueError(f"Unsupported model type {model.model_type!r}")


# ---------------------------------------------------------------------------
# FOPDT
# ---------------------------------------------------------------------------

def _simulate_fopdt(
    K: float, tau: float, dt: float, u: np.ndarray, y0: float,
) -> np.ndarray:
    tau = max(tau, 1e-9)
    alpha = float(np.exp(-dt / tau))
    y = np.empty_like(u, dtype=float)
    y[0] = y0
    for i in range(1, len(u)):
        y[i] = alpha * y[i - 1] + K * (1.0 - alpha) * u[i - 1]
    return y


# ---------------------------------------------------------------------------
# SOPDT (two real, non-repeated poles)
# ---------------------------------------------------------------------------

def _simulate_sopdt_over(
    K: float, tau1: float, tau2: float, dt: float, u: np.ndarray, y0: float,
) -> np.ndarray:
    """Two cascaded first-order filters with a shared gain ``K``.

    This is the numerically robust form used by the over-damped /
    critically-damped branch of :func:`simulate_model`.
    """
    tau1 = max(tau1, 1e-9)
    tau2 = max(tau2, 1e-9)
    a1 = float(np.exp(-dt / tau1))
    a2 = float(np.exp(-dt / tau2))

    y1 = np.empty_like(u, dtype=float)
    y2 = np.empty_like(u, dtype=float)
    y1[0] = y0
    y2[0] = y0
    for i in range(1, len(u)):
        y1[i] = a1 * y1[i - 1] + K * (1.0 - a1) * u[i - 1]
        y2[i] = a2 * y2[i - 1] + (1.0 - a2) * y1[i - 1]
    return y2


def _simulate_second_order(
    K: float, wn: float, zeta: float, dt: float, u: np.ndarray, y0: float,
) -> np.ndarray:
    """Under-damped / general second-order in (K, ωn, ζ) form.

    Uses forward Euler on the state-space form – fine for the damping
    ratios encountered in practice; callers needing sub-sample accuracy
    should pre-resample.
    """
    wn = max(wn, 1e-9)
    y = np.empty_like(u, dtype=float)
    y_dot = np.zeros_like(u, dtype=float)
    y[0] = y0
    for i in range(1, len(u)):
        y_ddot = wn * wn * (K * u[i - 1] - y[i - 1]) - 2 * zeta * wn * y_dot[i - 1]
        y_dot[i] = y_dot[i - 1] + y_ddot * dt
        y[i] = y[i - 1] + y_dot[i] * dt
    return y


def _simulate_ipdt(
    K: float, dt: float, u: np.ndarray, y0: float,
) -> np.ndarray:
    y = np.empty_like(u, dtype=float)
    y[0] = y0
    for i in range(1, len(u)):
        y[i] = y[i - 1] + K * u[i - 1] * dt
    return y


__all__ = ["simulate_model"]
