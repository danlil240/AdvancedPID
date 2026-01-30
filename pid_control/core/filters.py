"""
Signal filtering implementations for PID control.
Uses scipy.signal for robust filter implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional
from collections import deque
import numpy as np
from scipy import signal


class BaseFilter(ABC):
    """Abstract base class for all filters."""
    
    @abstractmethod
    def update(self, value: float) -> float:
        """Update filter with new value and return filtered output."""
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
    """First-order IIR low-pass filter."""
    
    def __init__(
        self, 
        cutoff_freq: Optional[float] = None,
        sample_time: float = 0.01,
        alpha: Optional[float] = None
    ):
        if alpha is not None:
            if not 0 < alpha <= 1:
                raise ValueError("Alpha must be in (0, 1]")
            self._alpha = alpha
        elif cutoff_freq is not None:
            if cutoff_freq <= 0:
                raise ValueError("Cutoff frequency must be positive")
            if sample_time <= 0:
                raise ValueError("Sample time must be positive")
            tau = 1.0 / (2.0 * np.pi * cutoff_freq)
            self._alpha = sample_time / (tau + sample_time)
        else:
            self._alpha = 0.1
        
        self._output: float = 0.0
        self._initialized: bool = False
    
    def update(self, value: float) -> float:
        if not self._initialized:
            self._output = value
            self._initialized = True
        else:
            self._output = self._alpha * value + (1 - self._alpha) * self._output
        return self._output
    
    def reset(self) -> None:
        self._output = 0.0
        self._initialized = False
    
    @property
    def output(self) -> float:
        return self._output
    
    @property
    def alpha(self) -> float:
        return self._alpha
    
    @alpha.setter
    def alpha(self, value: float) -> None:
        if not 0 < value <= 1:
            raise ValueError("Alpha must be in (0, 1]")
        self._alpha = value


class DerivativeFilter(BaseFilter):
    """Filtered derivative calculator with low-pass filtering."""
    
    def __init__(self, sample_time: float = 0.01, filter_coefficient: float = 10.0):
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
        if not self._initialized:
            self._prev_input = value
            self._output = 0.0
            self._initialized = True
        else:
            alpha = self._N * self._dt
            self._output = (
                self._N * (value - self._prev_input) + 
                (1 - alpha) * self._output
            ) / (1 + alpha)
            self._prev_input = value
        return self._output
    
    def reset(self) -> None:
        self._prev_input = 0.0
        self._output = 0.0
        self._initialized = False
    
    @property
    def output(self) -> float:
        return self._output
    
    @property
    def sample_time(self) -> float:
        return self._dt
    
    @sample_time.setter
    def sample_time(self, value: float) -> None:
        if value <= 0:
            raise ValueError("Sample time must be positive")
        self._dt = value


class MedianFilter(BaseFilter):
    """Median filter for spike/outlier rejection using numpy."""
    
    def __init__(self, window_size: int = 5):
        if window_size < 1:
            raise ValueError("Window size must be at least 1")
        if window_size % 2 == 0:
            window_size += 1
        
        self._window_size = window_size
        self._buffer: deque = deque(maxlen=window_size)
        self._output: float = 0.0
    
    def update(self, value: float) -> float:
        self._buffer.append(value)
        self._output = float(np.median(list(self._buffer)))
        return self._output
    
    def reset(self) -> None:
        self._buffer.clear()
        self._output = 0.0
    
    @property
    def output(self) -> float:
        return self._output


class MovingAverageFilter(BaseFilter):
    """Simple moving average filter using numpy."""
    
    def __init__(self, window_size: int = 5):
        if window_size < 1:
            raise ValueError("Window size must be at least 1")
        
        self._window_size = window_size
        self._buffer: deque = deque(maxlen=window_size)
        self._output: float = 0.0
    
    def update(self, value: float) -> float:
        self._buffer.append(value)
        self._output = float(np.mean(list(self._buffer)))
        return self._output
    
    def reset(self) -> None:
        self._buffer.clear()
        self._output = 0.0
    
    @property
    def output(self) -> float:
        return self._output


class RateLimiter(BaseFilter):
    """Rate limiter to constrain signal rate of change."""
    
    def __init__(
        self,
        rising_rate: float,
        falling_rate: Optional[float] = None,
        sample_time: float = 0.01
    ):
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
        if not self._initialized:
            self._output = value
            self._initialized = True
        else:
            delta = value - self._output
            max_rise = self._rising_rate * self._dt
            max_fall = self._falling_rate * self._dt
            self._output += np.clip(delta, -max_fall, max_rise)
        return self._output
    
    def reset(self) -> None:
        self._output = 0.0
        self._initialized = False
    
    @property
    def output(self) -> float:
        return self._output


class ButterworthFilter(BaseFilter):
    """Second-order Butterworth low-pass filter using scipy.signal."""
    
    def __init__(self, cutoff_freq: float, sample_time: float = 0.01, order: int = 2):
        if cutoff_freq <= 0:
            raise ValueError("Cutoff frequency must be positive")
        if sample_time <= 0:
            raise ValueError("Sample time must be positive")
        
        fs = 1.0 / sample_time
        nyquist = fs / 2.0
        
        if cutoff_freq >= nyquist:
            cutoff_freq = nyquist * 0.99
        
        normalized_cutoff = cutoff_freq / nyquist
        self._b, self._a = signal.butter(order, normalized_cutoff, btype='low')
        self._zi = signal.lfilter_zi(self._b, self._a)
        self._output: float = 0.0
        self._initialized: bool = False
    
    def update(self, value: float) -> float:
        if not self._initialized:
            self._zi = self._zi * value
            self._initialized = True
        
        result, self._zi = signal.lfilter(self._b, self._a, [value], zi=self._zi)
        self._output = float(result[0])
        return self._output
    
    def reset(self) -> None:
        self._zi = signal.lfilter_zi(self._b, self._a)
        self._output = 0.0
        self._initialized = False
    
    @property
    def output(self) -> float:
        return self._output
