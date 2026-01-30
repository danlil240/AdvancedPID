"""
Nonlinear plant models for testing PID robustness.
"""

from typing import Dict, Any, Callable, Optional
import numpy as np
from pid_control.plants.base_plant import BasePlant


class NonlinearPlant(BasePlant):
    """
    Configurable nonlinear plant with saturation, dead-zone, and backlash.
    Base dynamics: First-order with nonlinear modifications.
    """
    
    def __init__(
        self,
        gain: float = 1.0,
        time_constant: float = 1.0,
        sample_time: float = 0.01,
        saturation_limits: Optional[tuple] = None,
        dead_zone: float = 0.0,
        backlash: float = 0.0,
        nonlinear_gain_func: Optional[Callable[[float], float]] = None,
        initial_output: float = 0.0
    ):
        super().__init__(sample_time)
        
        if time_constant <= 0:
            raise ValueError("time_constant must be positive")
        
        self._K = gain
        self._tau = time_constant
        self._saturation = saturation_limits
        self._dead_zone = dead_zone
        self._backlash = backlash
        self._nl_gain_func = nonlinear_gain_func
        self._initial_output = initial_output
        self._output = initial_output
        self._backlash_state = 0.0
        
        # Precompute discrete coefficient
        self._alpha = np.exp(-sample_time / time_constant)
    
    def update(self, control_input: float) -> float:
        """Update plant with nonlinear effects."""
        u = control_input
        
        # Apply input saturation using numpy
        if self._saturation is not None:
            u = np.clip(u, self._saturation[0], self._saturation[1])
        
        # Apply dead-zone
        if self._dead_zone > 0:
            u = np.sign(u) * np.maximum(0, np.abs(u) - self._dead_zone)
        
        # Apply backlash
        if self._backlash > 0:
            half = self._backlash / 2
            if u > self._backlash_state + half:
                self._backlash_state = u - half
            elif u < self._backlash_state - half:
                self._backlash_state = u + half
            u = self._backlash_state
        
        # Get effective gain
        effective_gain = self._K
        if self._nl_gain_func is not None:
            effective_gain *= self._nl_gain_func(self._output)
        
        # First-order dynamics with exact discretization
        self._output = self._alpha * self._output + effective_gain * (1 - self._alpha) * u
        self._time += self._dt
        
        measured = self._add_disturbance(self._output)
        measured = self._add_noise(measured)
        return measured
    
    def reset(self) -> None:
        self._output = self._initial_output
        self._time = 0.0
        self._backlash_state = 0.0
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'type': 'NonlinearPlant',
            'gain': self._K,
            'time_constant': self._tau,
            'saturation': self._saturation,
            'dead_zone': self._dead_zone,
            'backlash': self._backlash,
            'sample_time': self._dt,
        }


class FrictionPlant(BasePlant):
    """
    Plant model with Coulomb + viscous friction and stiction.
    Simulates realistic mechanical systems with friction effects.
    """
    
    def __init__(
        self,
        mass: float = 1.0,
        viscous_friction: float = 0.5,
        coulomb_friction: float = 0.1,
        stiction: float = 0.15,
        sample_time: float = 0.01,
        initial_position: float = 0.0,
        initial_velocity: float = 0.0
    ):
        super().__init__(sample_time)
        
        if mass <= 0:
            raise ValueError("mass must be positive")
        
        self._mass = mass
        self._b = viscous_friction
        self._fc = coulomb_friction
        self._fs = stiction
        
        self._position = initial_position
        self._velocity = initial_velocity
        self._initial_position = initial_position
        self._initial_velocity = initial_velocity
        self._output = initial_position
    
    def update(self, control_input: float) -> float:
        """Update plant with friction dynamics."""
        force = control_input
        
        # Calculate friction force
        if np.abs(self._velocity) < 1e-6:  # Static regime
            if np.abs(force) < self._fs:
                friction_force = force  # Friction cancels applied force
            else:
                friction_force = -np.sign(force) * self._fs
        else:
            # Moving: Coulomb + viscous friction
            friction_force = -np.sign(self._velocity) * self._fc - self._b * self._velocity
        
        # Semi-implicit Euler integration
        acceleration = (force + friction_force) / self._mass
        self._velocity += acceleration * self._dt
        self._position += self._velocity * self._dt
        
        self._output = self._position
        self._time += self._dt
        
        measured = self._add_disturbance(self._output)
        measured = self._add_noise(measured)
        return measured
    
    def reset(self) -> None:
        self._position = self._initial_position
        self._velocity = self._initial_velocity
        self._output = self._initial_position
        self._time = 0.0
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'type': 'FrictionPlant',
            'mass': self._mass,
            'viscous_friction': self._b,
            'coulomb_friction': self._fc,
            'stiction': self._fs,
            'sample_time': self._dt,
        }
    
    @property
    def velocity(self) -> float:
        return self._velocity
    
    @property
    def position(self) -> float:
        return self._position
