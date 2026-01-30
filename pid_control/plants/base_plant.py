"""
Base plant model abstract class.
Defines the interface for all plant/process models.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np


class BasePlant(ABC):
    """
    Abstract base class for plant/process models.
    
    All plant implementations must inherit from this class
    and implement the required methods.
    """
    
    def __init__(self, sample_time: float = 0.01):
        """
        Initialize base plant.
        
        Args:
            sample_time: Sample time in seconds
        """
        if sample_time <= 0:
            raise ValueError("sample_time must be positive")
        
        self._dt = sample_time
        self._output: float = 0.0
        self._time: float = 0.0
        self._noise_std: float = 0.0
        self._disturbance: float = 0.0
    
    @abstractmethod
    def update(self, control_input: float) -> float:
        """
        Update plant state with control input.
        
        Args:
            control_input: Control signal from controller
            
        Returns:
            Plant output (measured value)
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset plant to initial state."""
        pass
    
    @property
    def output(self) -> float:
        """Current plant output."""
        return self._output
    
    @property
    def sample_time(self) -> float:
        """Sample time."""
        return self._dt
    
    @sample_time.setter
    def sample_time(self, value: float) -> None:
        """Set sample time."""
        if value <= 0:
            raise ValueError("sample_time must be positive")
        self._dt = value
    
    @property
    def time(self) -> float:
        """Current simulation time."""
        return self._time
    
    def set_noise(self, std: float) -> None:
        """
        Set measurement noise level.
        
        Args:
            std: Standard deviation of Gaussian noise
        """
        if std < 0:
            raise ValueError("Noise std must be non-negative")
        self._noise_std = std
    
    def set_disturbance(self, value: float) -> None:
        """
        Set constant disturbance.
        
        Args:
            value: Disturbance value added to output
        """
        self._disturbance = value
    
    def _add_noise(self, value: float) -> float:
        """Add measurement noise to output."""
        if self._noise_std > 0:
            return value + np.random.normal(0, self._noise_std)
        return value
    
    def _add_disturbance(self, value: float) -> float:
        """Add disturbance to output."""
        return value + self._disturbance
    
    def get_state(self) -> Dict[str, Any]:
        """Get current plant state as dictionary."""
        return {
            'output': self._output,
            'time': self._time,
            'noise_std': self._noise_std,
            'disturbance': self._disturbance,
        }
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get plant information/parameters."""
        pass
