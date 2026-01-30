"""
Mathematical utility functions for PID control.
Uses numpy for efficient array operations.
"""

from typing import Optional, Union
import numpy as np
from numpy.typing import ArrayLike


def clamp(value: float, min_val: Optional[float], max_val: Optional[float]) -> float:
    """Clamp a value between minimum and maximum bounds."""
    if min_val is None and max_val is None:
        return value
    return float(np.clip(value, min_val, max_val))


def interpolate(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    """Linear interpolation between two points."""
    return float(np.interp(x, [x0, x1], [y0, y1]))


def moving_average(values: ArrayLike, window: int) -> np.ndarray:
    """Compute moving average using numpy convolution."""
    if window <= 0:
        raise ValueError("Window size must be positive")
    
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return np.array([])
    
    kernel = np.ones(window) / window
    # Use 'same' mode and handle edges
    result = np.convolve(arr, kernel, mode='same')
    # Fix edge effects
    for i in range(min(window - 1, len(arr))):
        result[i] = np.mean(arr[:i + 1])
    return result


def exponential_moving_average(values: ArrayLike, alpha: float) -> np.ndarray:
    """Compute exponential moving average using scipy."""
    if not 0 < alpha <= 1:
        raise ValueError("Alpha must be in (0, 1]")
    
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return np.array([])
    
    from scipy.ndimage import uniform_filter1d
    # EMA can be computed iteratively
    result = np.zeros_like(arr)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result


def derivative_estimate(values: ArrayLike, dt: float, method: str = "backward") -> np.ndarray:
    """Estimate derivative using numpy.gradient or diff."""
    if dt <= 0:
        raise ValueError("Time step must be positive")
    
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return np.zeros_like(arr)
    
    if method == "central":
        return np.gradient(arr, dt)
    elif method == "backward":
        result = np.zeros_like(arr)
        result[1:] = np.diff(arr) / dt
        return result
    elif method == "forward":
        result = np.zeros_like(arr)
        result[:-1] = np.diff(arr) / dt
        result[-1] = result[-2] if len(result) > 1 else 0.0
        return result
    else:
        raise ValueError(f"Unknown method: {method}")


def rms(values: ArrayLike) -> float:
    """Compute root mean square using numpy."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr ** 2)))


def integrate_trapezoid(values: ArrayLike, dt: float) -> float:
    """Integrate using numpy's trapezoid function."""
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return 0.0
    return float(np.trapz(arr, dx=dt))


def sign(x: float) -> int:
    """Return sign of x: -1, 0, or 1."""
    return int(np.sign(x))


def deadband(value: float, threshold: float) -> float:
    """Apply deadband to a value."""
    return 0.0 if abs(value) < threshold else value
