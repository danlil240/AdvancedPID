"""
Nonlinear plant models for testing PID robustness.
"""

from typing import Dict, Any, Callable, Optional
import math
from pid_control.plants.base_plant import BasePlant


class NonlinearPlant(BasePlant):
    """
    Configurable nonlinear plant with various nonlinearities.
    
    Supports:
    - Saturation
    - Dead-zone
    - Hysteresis
    - Backlash
    - Variable gain
    - Custom nonlinear functions
    
    Base dynamics: First-order with nonlinear modifications
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
        """
        Initialize nonlinear plant.
        
        Args:
            gain: Nominal static gain
            time_constant: Nominal time constant
            sample_time: Sample time
            saturation_limits: (min, max) tuple for input saturation
            dead_zone: Dead-zone threshold (symmetric)
            backlash: Backlash/hysteresis width
            nonlinear_gain_func: Custom gain function f(output) -> gain_multiplier
            initial_output: Initial output value
        """
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
        
        # Backlash state
        self._prev_input = 0.0
        self._backlash_state = 0.0
    
    def update(self, control_input: float) -> float:
        """
        Update plant with nonlinear effects.
        
        Args:
            control_input: Control signal u
            
        Returns:
            Plant output with nonlinearities
        """
        u = control_input
        
        # Apply input saturation
        if self._saturation is not None:
            u = max(self._saturation[0], min(self._saturation[1], u))
        
        # Apply dead-zone
        if self._dead_zone > 0:
            if abs(u) < self._dead_zone:
                u = 0.0
            elif u > 0:
                u = u - self._dead_zone
            else:
                u = u + self._dead_zone
        
        # Apply backlash
        if self._backlash > 0:
            u = self._apply_backlash(u)
        
        # Get effective gain
        effective_gain = self._K
        if self._nl_gain_func is not None:
            effective_gain *= self._nl_gain_func(self._output)
        
        # First-order dynamics
        self._output += (self._dt / self._tau) * (
            effective_gain * u - self._output
        )
        
        self._time += self._dt
        self._prev_input = control_input
        
        # Apply disturbance and noise
        measured = self._add_disturbance(self._output)
        measured = self._add_noise(measured)
        
        return measured
    
    def _apply_backlash(self, u: float) -> float:
        """Apply backlash nonlinearity."""
        half_backlash = self._backlash / 2
        
        if u > self._backlash_state + half_backlash:
            self._backlash_state = u - half_backlash
        elif u < self._backlash_state - half_backlash:
            self._backlash_state = u + half_backlash
        
        return self._backlash_state
    
    def reset(self) -> None:
        """Reset plant to initial state."""
        self._output = self._initial_output
        self._time = 0.0
        self._prev_input = 0.0
        self._backlash_state = 0.0
    
    def get_info(self) -> Dict[str, Any]:
        """Get plant parameters."""
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
    Plant model with friction (Coulomb + viscous).
    
    Simulates realistic mechanical systems with friction effects:
    - Coulomb friction (constant opposing force)
    - Viscous friction (velocity-dependent)
    - Stiction (static friction)
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
        """
        Initialize friction plant.
        
        Args:
            mass: Mass (kg)
            viscous_friction: Viscous friction coefficient (N*s/m)
            coulomb_friction: Coulomb friction force (N)
            stiction: Static friction force (N)
            sample_time: Sample time
            initial_position: Initial position
            initial_velocity: Initial velocity
        """
        super().__init__(sample_time)
        
        if mass <= 0:
            raise ValueError("mass must be positive")
        if viscous_friction < 0:
            raise ValueError("viscous_friction must be non-negative")
        if coulomb_friction < 0:
            raise ValueError("coulomb_friction must be non-negative")
        if stiction < 0:
            raise ValueError("stiction must be non-negative")
        
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
        """
        Update plant with friction dynamics.
        
        Args:
            control_input: Applied force (N)
            
        Returns:
            Position output
        """
        force = control_input
        
        # Calculate friction force
        if abs(self._velocity) < 1e-6:  # Essentially stationary
            # Static friction regime
            if abs(force) < self._fs:
                # Not enough force to overcome stiction
                friction_force = force  # Friction exactly cancels applied force
            else:
                # Breaking away from stiction
                friction_force = self._fs * (1 if force < 0 else -1)
        else:
            # Moving - Coulomb + viscous friction
            friction_force = (
                self._fc * (-1 if self._velocity > 0 else 1) +
                self._b * (-self._velocity)
            )
        
        # Acceleration
        acceleration = (force + friction_force) / self._mass
        
        # Integrate using semi-implicit Euler
        self._velocity += acceleration * self._dt
        self._position += self._velocity * self._dt
        
        self._output = self._position
        self._time += self._dt
        
        # Apply disturbance and noise
        measured = self._add_disturbance(self._output)
        measured = self._add_noise(measured)
        
        return measured
    
    def reset(self) -> None:
        """Reset plant to initial state."""
        self._position = self._initial_position
        self._velocity = self._initial_velocity
        self._output = self._initial_position
        self._time = 0.0
    
    def get_info(self) -> Dict[str, Any]:
        """Get plant parameters."""
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
        """Current velocity."""
        return self._velocity
    
    @property
    def position(self) -> float:
        """Current position."""
        return self._position
