"""
Mathematical utility functions for PID control.
"""

from typing import List, Optional, Sequence
import math


def clamp(value: float, min_val: Optional[float], max_val: Optional[float]) -> float:
    """
    Clamp a value between minimum and maximum bounds.
    
    Args:
        value: Value to clamp
        min_val: Minimum bound (None for no lower bound)
        max_val: Maximum bound (None for no upper bound)
        
    Returns:
        Clamped value
    """
    if min_val is not None and value < min_val:
        return min_val
    if max_val is not None and value > max_val:
        return max_val
    return value


def interpolate(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    """
    Linear interpolation between two points.
    
    Args:
        x: Input value
        x0, x1: X coordinates of known points
        y0, y1: Y coordinates of known points
        
    Returns:
        Interpolated y value
    """
    if abs(x1 - x0) < 1e-12:
        return (y0 + y1) / 2.0
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def moving_average(values: Sequence[float], window: int) -> List[float]:
    """
    Compute moving average of a sequence.
    
    Args:
        values: Input sequence
        window: Window size for averaging
        
    Returns:
        List of moving averages
    """
    if window <= 0:
        raise ValueError("Window size must be positive")
    
    n = len(values)
    if n == 0:
        return []
    
    result = []
    window_sum = 0.0
    
    for i, val in enumerate(values):
        window_sum += val
        if i >= window:
            window_sum -= values[i - window]
            result.append(window_sum / window)
        else:
            result.append(window_sum / (i + 1))
    
    return result


def exponential_moving_average(
    values: Sequence[float], 
    alpha: float
) -> List[float]:
    """
    Compute exponential moving average.
    
    Args:
        values: Input sequence
        alpha: Smoothing factor (0 < alpha <= 1)
        
    Returns:
        List of EMA values
    """
    if not 0 < alpha <= 1:
        raise ValueError("Alpha must be in (0, 1]")
    
    if len(values) == 0:
        return []
    
    result = [float(values[0])]
    for val in values[1:]:
        ema = alpha * val + (1 - alpha) * result[-1]
        result.append(ema)
    
    return result


def derivative_estimate(
    values: Sequence[float],
    dt: float,
    method: str = "backward"
) -> List[float]:
    """
    Estimate derivative of a discrete signal.
    
    Args:
        values: Input sequence
        dt: Time step
        method: "backward", "forward", or "central"
        
    Returns:
        List of derivative estimates
    """
    if dt <= 0:
        raise ValueError("Time step must be positive")
    
    n = len(values)
    if n < 2:
        return [0.0] * n
    
    result = []
    
    if method == "backward":
        result.append(0.0)
        for i in range(1, n):
            result.append((values[i] - values[i-1]) / dt)
    
    elif method == "forward":
        for i in range(n - 1):
            result.append((values[i+1] - values[i]) / dt)
        result.append(result[-1] if result else 0.0)
    
    elif method == "central":
        result.append((values[1] - values[0]) / dt if n > 1 else 0.0)
        for i in range(1, n - 1):
            result.append((values[i+1] - values[i-1]) / (2 * dt))
        result.append((values[-1] - values[-2]) / dt if n > 1 else 0.0)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return result


def rms(values: Sequence[float]) -> float:
    """
    Compute root mean square of values.
    
    Args:
        values: Input sequence
        
    Returns:
        RMS value
    """
    if len(values) == 0:
        return 0.0
    return math.sqrt(sum(v * v for v in values) / len(values))


def integrate_trapezoid(values: Sequence[float], dt: float) -> float:
    """
    Integrate using trapezoidal rule.
    
    Args:
        values: Input sequence
        dt: Time step
        
    Returns:
        Integral value
    """
    if len(values) < 2:
        return 0.0
    
    total = 0.0
    for i in range(1, len(values)):
        total += (values[i] + values[i-1]) * dt / 2
    
    return total


def sign(x: float) -> int:
    """Return sign of x: -1, 0, or 1."""
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0


def deadband(value: float, threshold: float) -> float:
    """
    Apply deadband to a value.
    
    Args:
        value: Input value
        threshold: Deadband threshold (positive)
        
    Returns:
        0 if |value| < threshold, else value
    """
    if abs(value) < threshold:
        return 0.0
    return value
