"""
Signal filtering implementations for PID control.
Provides various filters for noise reduction and signal conditioning.
"""

from abc import ABC, abstractmethod
from typing import Optional, List
from collections import deque
import math


class BaseFilter(ABC):
    """Abstract base class for all filters."""
    
    @abstractmethod
    def update(self, value: float) -> float:
        """
        Update filter with new value and return filtered output.
        
        Args:
            value: New input value
            
        Returns:
            Filtered output value
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset filter state."""
        pass
    
    @property
    @abstractmethod
    def output(self) -> float:
        """Current filter output."""
        pass


class LowPassFilter(BaseFilter):
    """
    First-order IIR low-pass filter.
    
    Implements: y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
    
    The cutoff frequency fc and sample time dt determine alpha:
    alpha = dt / (tau + dt) where tau = 1 / (2 * pi * fc)
    """
    
    def __init__(
        self, 
        cutoff_freq: Optional[float] = None,
        sample_time: float = 0.01,
        alpha: Optional[float] = None
    ):
        """
        Initialize low-pass filter.
        
        Args:
            cutoff_freq: Cutoff frequency in Hz (used if alpha not provided)
            sample_time: Sample time in seconds
            alpha: Direct smoothing factor (0 < alpha <= 1)
                   If provided, overrides cutoff_freq calculation
        """
        if alpha is not None:
            if not 0 < alpha <= 1:
                raise ValueError("Alpha must be in (0, 1]")
            self._alpha = alpha
        elif cutoff_freq is not None:
            if cutoff_freq <= 0:
                raise ValueError("Cutoff frequency must be positive")
            if sample_time <= 0:
                raise ValueError("Sample time must be positive")
            tau = 1.0 / (2.0 * math.pi * cutoff_freq)
            self._alpha = sample_time / (tau + sample_time)
        else:
            self._alpha = 0.1  # Default smoothing
        
        self._output: float = 0.0
        self._initialized: bool = False
    
    def update(self, value: float) -> float:
        """Update filter with new value."""
        if not self._initialized:
            self._output = value
            self._initialized = True
        else:
            self._output = self._alpha * value + (1 - self._alpha) * self._output
        return self._output
    
    def reset(self) -> None:
        """Reset filter state."""
        self._output = 0.0
        self._initialized = False
    
    @property
    def output(self) -> float:
        """Current filter output."""
        return self._output
    
    @property
    def alpha(self) -> float:
        """Current alpha value."""
        return self._alpha
    
    @alpha.setter
    def alpha(self, value: float) -> None:
        """Set alpha value."""
        if not 0 < value <= 1:
            raise ValueError("Alpha must be in (0, 1]")
        self._alpha = value


class DerivativeFilter(BaseFilter):
    """
    Filtered derivative calculator.
    
    Combines derivative calculation with low-pass filtering
    to reduce noise amplification in derivative term.
    
    Implements: D[n] = (N * (x[n] - x[n-1]) + (1 - N*dt/tau) * D[n-1]) 
                     / (1 + N*dt/tau)
    
    Where N is the filter coefficient (typically 2-20).
    """
    
    def __init__(
        self,
        sample_time: float = 0.01,
        filter_coefficient: float = 10.0
    ):
        """
        Initialize derivative filter.
        
        Args:
            sample_time: Sample time in seconds
            filter_coefficient: N value (higher = less filtering, faster response)
        """
        if sample_time <= 0:
            raise ValueError("Sample time must be positive")
        if filter_coefficient <= 0:
            raise ValueError("Filter coefficient must be positive")
        
        self._dt = sample_time
        self._N = filter_coefficient
        self._prev_input: float = 0.0
        self._output: float = 0.0
        self._initialized: bool = False
    
    def update(self, value: float) -> float:
        """
        Update filter and compute filtered derivative.
        
        Args:
            value: New input value
            
        Returns:
            Filtered derivative
        """
        if not self._initialized:
            self._prev_input = value
            self._output = 0.0
            self._initialized = True
        else:
            # Filtered derivative formula
            alpha = self._N * self._dt
            self._output = (
                self._N * (value - self._prev_input) + 
                (1 - alpha) * self._output
            ) / (1 + alpha)
            self._prev_input = value
        
        return self._output
    
    def reset(self) -> None:
        """Reset filter state."""
        self._prev_input = 0.0
        self._output = 0.0
        self._initialized = False
    
    @property
    def output(self) -> float:
        """Current derivative output."""
        return self._output
    
    @property
    def sample_time(self) -> float:
        """Current sample time."""
        return self._dt
    
    @sample_time.setter
    def sample_time(self, value: float) -> None:
        """Set sample time."""
        if value <= 0:
            raise ValueError("Sample time must be positive")
        self._dt = value


class MedianFilter(BaseFilter):
    """
    Median filter for spike/outlier rejection.
    
    Particularly useful for removing impulse noise from sensors.
    """
    
    def __init__(self, window_size: int = 5):
        """
        Initialize median filter.
        
        Args:
            window_size: Number of samples in window (must be odd)
        """
        if window_size < 1:
            raise ValueError("Window size must be at least 1")
        # Make odd if even
        if window_size % 2 == 0:
            window_size += 1
        
        self._window_size = window_size
        self._buffer: deque = deque(maxlen=window_size)
        self._output: float = 0.0
    
    def update(self, value: float) -> float:
        """Update filter with new value."""
        self._buffer.append(value)
        sorted_values = sorted(self._buffer)
        self._output = sorted_values[len(sorted_values) // 2]
        return self._output
    
    def reset(self) -> None:
        """Reset filter state."""
        self._buffer.clear()
        self._output = 0.0
    
    @property
    def output(self) -> float:
        """Current filter output."""
        return self._output


class MovingAverageFilter(BaseFilter):
    """
    Simple moving average filter.
    """
    
    def __init__(self, window_size: int = 5):
        """
        Initialize moving average filter.
        
        Args:
            window_size: Number of samples to average
        """
        if window_size < 1:
            raise ValueError("Window size must be at least 1")
        
        self._window_size = window_size
        self._buffer: deque = deque(maxlen=window_size)
        self._sum: float = 0.0
        self._output: float = 0.0
    
    def update(self, value: float) -> float:
        """Update filter with new value."""
        if len(self._buffer) == self._window_size:
            self._sum -= self._buffer[0]
        self._buffer.append(value)
        self._sum += value
        self._output = self._sum / len(self._buffer)
        return self._output
    
    def reset(self) -> None:
        """Reset filter state."""
        self._buffer.clear()
        self._sum = 0.0
        self._output = 0.0
    
    @property
    def output(self) -> float:
        """Current filter output."""
        return self._output


class RateLimiter(BaseFilter):
    """
    Rate limiter to constrain signal rate of change.
    
    Useful for preventing sudden jumps in setpoint or control output.
    """
    
    def __init__(
        self,
        rising_rate: float,
        falling_rate: Optional[float] = None,
        sample_time: float = 0.01
    ):
        """
        Initialize rate limiter.
        
        Args:
            rising_rate: Maximum rate of increase (units/second)
            falling_rate: Maximum rate of decrease (units/second), 
                         defaults to rising_rate
            sample_time: Sample time in seconds
        """
        if rising_rate <= 0:
            raise ValueError("Rising rate must be positive")
        if sample_time <= 0:
            raise ValueError("Sample time must be positive")
        
        self._rising_rate = rising_rate
        self._falling_rate = falling_rate if falling_rate is not None else rising_rate
        self._dt = sample_time
        self._output: float = 0.0
        self._initialized: bool = False
    
    def update(self, value: float) -> float:
        """Update rate limiter with new value."""
        if not self._initialized:
            self._output = value
            self._initialized = True
        else:
            delta = value - self._output
            max_rise = self._rising_rate * self._dt
            max_fall = self._falling_rate * self._dt
            
            if delta > max_rise:
                self._output += max_rise
            elif delta < -max_fall:
                self._output -= max_fall
            else:
                self._output = value
        
        return self._output
    
    def reset(self) -> None:
        """Reset rate limiter state."""
        self._output = 0.0
        self._initialized = False
    
    @property
    def output(self) -> float:
        """Current output."""
        return self._output


class ButterworthFilter(BaseFilter):
    """
    Second-order Butterworth low-pass filter.
    
    Provides maximally flat frequency response in passband.
    """
    
    def __init__(self, cutoff_freq: float, sample_time: float = 0.01):
        """
        Initialize Butterworth filter.
        
        Args:
            cutoff_freq: Cutoff frequency in Hz
            sample_time: Sample time in seconds
        """
        if cutoff_freq <= 0:
            raise ValueError("Cutoff frequency must be positive")
        if sample_time <= 0:
            raise ValueError("Sample time must be positive")
        
        # Precompute coefficients using bilinear transform
        omega = 2.0 * math.pi * cutoff_freq
        omega_d = 2.0 / sample_time * math.tan(omega * sample_time / 2.0)
        
        k1 = math.sqrt(2.0) * omega_d
        k2 = omega_d * omega_d
        
        a0 = 4.0 + 2.0 * k1 * sample_time + k2 * sample_time * sample_time
        
        self._b0 = k2 * sample_time * sample_time / a0
        self._b1 = 2.0 * self._b0
        self._b2 = self._b0
        self._a1 = (2.0 * k2 * sample_time * sample_time - 8.0) / a0
        self._a2 = (4.0 - 2.0 * k1 * sample_time + k2 * sample_time * sample_time) / a0
        
        self._x1: float = 0.0
        self._x2: float = 0.0
        self._y1: float = 0.0
        self._y2: float = 0.0
        self._output: float = 0.0
    
    def update(self, value: float) -> float:
        """Update filter with new value."""
        self._output = (
            self._b0 * value + 
            self._b1 * self._x1 + 
            self._b2 * self._x2 -
            self._a1 * self._y1 - 
            self._a2 * self._y2
        )
        
        self._x2 = self._x1
        self._x1 = value
        self._y2 = self._y1
        self._y1 = self._output
        
        return self._output
    
    def reset(self) -> None:
        """Reset filter state."""
        self._x1 = 0.0
        self._x2 = 0.0
        self._y1 = 0.0
        self._y2 = 0.0
        self._output = 0.0
    
    @property
    def output(self) -> float:
        """Current filter output."""
        return self._output
