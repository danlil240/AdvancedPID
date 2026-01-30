"""
First-order plant model.
Transfer function: G(s) = K / (tau*s + 1)
"""

from typing import Dict, Any, Optional
from pid_control.plants.base_plant import BasePlant


class FirstOrderPlant(BasePlant):
    """
    First-order (PT1) plant model.
    
    Transfer function: G(s) = K / (tau*s + 1)
    
    Discrete implementation using Euler method:
    y[n+1] = y[n] + dt/tau * (K*u[n] - y[n])
    
    Example:
        >>> plant = FirstOrderPlant(gain=2.0, time_constant=1.0, sample_time=0.01)
        >>> output = plant.update(control_input=1.0)
    """
    
    def __init__(
        self,
        gain: float = 1.0,
        time_constant: float = 1.0,
        sample_time: float = 0.01,
        initial_output: float = 0.0
    ):
        """
        Initialize first-order plant.
        
        Args:
            gain: Static gain K
            time_constant: Time constant tau in seconds
            sample_time: Sample time in seconds
            initial_output: Initial output value
        """
        super().__init__(sample_time)
        
        if time_constant <= 0:
            raise ValueError("time_constant must be positive")
        
        self._K = gain
        self._tau = time_constant
        self._initial_output = initial_output
        self._output = initial_output
    
    def update(self, control_input: float) -> float:
        """
        Update plant with control input.
        
        Args:
            control_input: Control signal u
            
        Returns:
            Plant output y with noise and disturbance
        """
        # First-order dynamics: dy/dt = (K*u - y) / tau
        # Euler discretization
        self._output += (self._dt / self._tau) * (
            self._K * control_input - self._output
        )
        
        self._time += self._dt
        
        # Apply disturbance and noise
        measured = self._add_disturbance(self._output)
        measured = self._add_noise(measured)
        
        return measured
    
    def reset(self) -> None:
        """Reset plant to initial state."""
        self._output = self._initial_output
        self._time = 0.0
    
    def get_info(self) -> Dict[str, Any]:
        """Get plant parameters."""
        return {
            'type': 'FirstOrderPlant',
            'gain': self._K,
            'time_constant': self._tau,
            'sample_time': self._dt,
        }
    
    @property
    def gain(self) -> float:
        """Static gain K."""
        return self._K
    
    @property
    def time_constant(self) -> float:
        """Time constant tau."""
        return self._tau
