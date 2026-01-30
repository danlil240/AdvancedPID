"""
Plant models with time delay (dead time).
"""

from typing import Dict, Any, Optional
from collections import deque
from pid_control.plants.base_plant import BasePlant
from pid_control.plants.first_order import FirstOrderPlant
from pid_control.plants.second_order import SecondOrderPlant


class DelayPlant(BasePlant):
    """
    Wrapper that adds pure time delay to any plant.
    
    Implements delay using a circular buffer (FIFO queue).
    
    Example:
        >>> base_plant = FirstOrderPlant(gain=2.0, time_constant=1.0)
        >>> delayed_plant = DelayPlant(base_plant, delay_time=0.5)
    """
    
    def __init__(
        self,
        base_plant: BasePlant,
        delay_time: float
    ):
        """
        Initialize delayed plant.
        
        Args:
            base_plant: Underlying plant model
            delay_time: Pure delay time in seconds
        """
        super().__init__(base_plant.sample_time)
        
        if delay_time < 0:
            raise ValueError("delay_time must be non-negative")
        
        self._base_plant = base_plant
        self._delay_time = delay_time
        
        # Calculate delay buffer size
        self._delay_samples = max(1, int(round(delay_time / self._dt)))
        
        # Initialize delay buffer with zeros
        self._delay_buffer: deque = deque(
            [0.0] * self._delay_samples, 
            maxlen=self._delay_samples
        )
        
        self._output = 0.0
    
    def update(self, control_input: float) -> float:
        """
        Update plant with delayed input.
        
        Args:
            control_input: Control signal (will be delayed)
            
        Returns:
            Delayed plant output
        """
        # Get delayed input from buffer
        delayed_input = self._delay_buffer[0]
        
        # Add current input to buffer (will come out after delay)
        self._delay_buffer.append(control_input)
        
        # Update base plant with delayed input
        self._output = self._base_plant.update(delayed_input)
        self._time = self._base_plant.time
        
        return self._output
    
    def reset(self) -> None:
        """Reset plant and delay buffer."""
        self._base_plant.reset()
        self._delay_buffer = deque(
            [0.0] * self._delay_samples,
            maxlen=self._delay_samples
        )
        self._output = 0.0
        self._time = 0.0
    
    def get_info(self) -> Dict[str, Any]:
        """Get plant parameters."""
        base_info = self._base_plant.get_info()
        return {
            'type': 'DelayPlant',
            'delay_time': self._delay_time,
            'delay_samples': self._delay_samples,
            'base_plant': base_info,
        }
    
    @property
    def delay_time(self) -> float:
        """Delay time in seconds."""
        return self._delay_time
    
    @property
    def base_plant(self) -> BasePlant:
        """Underlying base plant."""
        return self._base_plant


class FOPDTPlant(BasePlant):
    """
    First-Order Plus Dead Time (FOPDT) plant.
    
    Most common model for process control:
    G(s) = K * exp(-L*s) / (tau*s + 1)
    
    Where:
        - K: Process gain
        - tau: Time constant
        - L: Dead time (delay)
    """
    
    def __init__(
        self,
        gain: float = 1.0,
        time_constant: float = 1.0,
        dead_time: float = 0.5,
        sample_time: float = 0.01,
        initial_output: float = 0.0
    ):
        """
        Initialize FOPDT plant.
        
        Args:
            gain: Process gain K
            time_constant: Time constant tau
            dead_time: Dead time L
            sample_time: Sample time
            initial_output: Initial output
        """
        super().__init__(sample_time)
        
        if time_constant <= 0:
            raise ValueError("time_constant must be positive")
        if dead_time < 0:
            raise ValueError("dead_time must be non-negative")
        
        self._K = gain
        self._tau = time_constant
        self._L = dead_time
        self._initial_output = initial_output
        
        # Internal first-order system
        self._fo_state = initial_output
        
        # Delay buffer
        self._delay_samples = max(1, int(round(dead_time / sample_time)))
        self._delay_buffer: deque = deque(
            [0.0] * self._delay_samples,
            maxlen=self._delay_samples
        )
        
        self._output = initial_output
    
    def update(self, control_input: float) -> float:
        """Update FOPDT plant."""
        # Get delayed input
        delayed_input = self._delay_buffer[0]
        self._delay_buffer.append(control_input)
        
        # First-order dynamics
        self._fo_state += (self._dt / self._tau) * (
            self._K * delayed_input - self._fo_state
        )
        
        self._output = self._fo_state
        self._time += self._dt
        
        # Apply disturbance and noise
        measured = self._add_disturbance(self._output)
        measured = self._add_noise(measured)
        
        return measured
    
    def reset(self) -> None:
        """Reset plant."""
        self._fo_state = self._initial_output
        self._output = self._initial_output
        self._delay_buffer = deque(
            [0.0] * self._delay_samples,
            maxlen=self._delay_samples
        )
        self._time = 0.0
    
    def get_info(self) -> Dict[str, Any]:
        """Get plant parameters."""
        return {
            'type': 'FOPDTPlant',
            'gain': self._K,
            'time_constant': self._tau,
            'dead_time': self._L,
            'sample_time': self._dt,
        }
    
    @property
    def gain(self) -> float:
        return self._K
    
    @property
    def time_constant(self) -> float:
        return self._tau
    
    @property
    def dead_time(self) -> float:
        return self._L
    
    def get_tuning_suggestions(self) -> Dict[str, Dict[str, float]]:
        """
        Get PID tuning suggestions based on plant parameters.
        
        Returns various classical tuning formulas.
        """
        K = self._K
        tau = self._tau
        L = self._L
        
        # Avoid division by zero
        if L < 1e-6:
            L = 0.01 * tau
        
        suggestions = {}
        
        # Ziegler-Nichols (open loop)
        suggestions['ziegler_nichols'] = {
            'kp': 1.2 * tau / (K * L),
            'ki': 1.2 * tau / (K * L) / (2 * L),
            'kd': 1.2 * tau / (K * L) * 0.5 * L,
        }
        
        # Cohen-Coon
        r = L / tau
        suggestions['cohen_coon'] = {
            'kp': (1.35 / K) * (tau / L + 0.185),
            'ki': (1.35 / K) * (tau / L + 0.185) / (2.5 * L * (tau + 0.185 * L) / (tau + 0.611 * L)),
            'kd': (1.35 / K) * (tau / L + 0.185) * 0.37 * L * tau / (tau + 0.185 * L),
        }
        
        # IMC (lambda tuning) with lambda = tau
        lambda_c = tau  # Closed-loop time constant
        suggestions['imc_aggressive'] = {
            'kp': tau / (K * (lambda_c + L)),
            'ki': tau / (K * (lambda_c + L)) / tau,
            'kd': 0,
        }
        
        # IMC conservative (lambda = 3*tau)
        lambda_c = 3 * tau
        suggestions['imc_conservative'] = {
            'kp': tau / (K * (lambda_c + L)),
            'ki': tau / (K * (lambda_c + L)) / tau,
            'kd': 0,
        }
        
        return suggestions
