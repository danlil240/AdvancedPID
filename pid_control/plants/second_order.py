"""
Second-order plant model.
Transfer function: G(s) = K * wn^2 / (s^2 + 2*zeta*wn*s + wn^2)
"""

from typing import Dict, Any
import math
from pid_control.plants.base_plant import BasePlant


class SecondOrderPlant(BasePlant):
    """
    Second-order plant model.
    
    Transfer function: G(s) = K * wn^2 / (s^2 + 2*zeta*wn*s + wn^2)
    
    Where:
        - K: Static gain
        - wn: Natural frequency (rad/s)
        - zeta: Damping ratio
    
    State-space representation:
        dx1/dt = x2
        dx2/dt = -wn^2*x1 - 2*zeta*wn*x2 + K*wn^2*u
        y = x1
    
    Example:
        >>> plant = SecondOrderPlant(
        ...     gain=1.0, 
        ...     natural_frequency=2.0,
        ...     damping_ratio=0.7
        ... )
        >>> output = plant.update(1.0)
    """
    
    def __init__(
        self,
        gain: float = 1.0,
        natural_frequency: float = 1.0,
        damping_ratio: float = 0.7,
        sample_time: float = 0.01,
        initial_output: float = 0.0,
        initial_velocity: float = 0.0
    ):
        """
        Initialize second-order plant.
        
        Args:
            gain: Static gain K
            natural_frequency: Natural frequency wn in rad/s
            damping_ratio: Damping ratio zeta (0=undamped, 1=critical)
            sample_time: Sample time in seconds
            initial_output: Initial output (position) value
            initial_velocity: Initial velocity value
        """
        super().__init__(sample_time)
        
        if natural_frequency <= 0:
            raise ValueError("natural_frequency must be positive")
        if damping_ratio < 0:
            raise ValueError("damping_ratio must be non-negative")
        
        self._K = gain
        self._wn = natural_frequency
        self._zeta = damping_ratio
        
        self._initial_output = initial_output
        self._initial_velocity = initial_velocity
        
        # State variables
        self._x1 = initial_output  # Position/output
        self._x2 = initial_velocity  # Velocity
        self._output = initial_output
    
    def update(self, control_input: float) -> float:
        """
        Update plant with control input using RK4 integration.
        
        Args:
            control_input: Control signal u
            
        Returns:
            Plant output y with noise and disturbance
        """
        # RK4 integration for better accuracy
        def dx1(x1, x2):
            return x2
        
        def dx2(x1, x2, u):
            return (
                -self._wn**2 * x1 
                - 2 * self._zeta * self._wn * x2 
                + self._K * self._wn**2 * u
            )
        
        dt = self._dt
        x1, x2 = self._x1, self._x2
        u = control_input
        
        # RK4 steps
        k1_x1 = dx1(x1, x2)
        k1_x2 = dx2(x1, x2, u)
        
        k2_x1 = dx1(x1 + 0.5*dt*k1_x1, x2 + 0.5*dt*k1_x2)
        k2_x2 = dx2(x1 + 0.5*dt*k1_x1, x2 + 0.5*dt*k1_x2, u)
        
        k3_x1 = dx1(x1 + 0.5*dt*k2_x1, x2 + 0.5*dt*k2_x2)
        k3_x2 = dx2(x1 + 0.5*dt*k2_x1, x2 + 0.5*dt*k2_x2, u)
        
        k4_x1 = dx1(x1 + dt*k3_x1, x2 + dt*k3_x2)
        k4_x2 = dx2(x1 + dt*k3_x1, x2 + dt*k3_x2, u)
        
        self._x1 += (dt/6) * (k1_x1 + 2*k2_x1 + 2*k3_x1 + k4_x1)
        self._x2 += (dt/6) * (k1_x2 + 2*k2_x2 + 2*k3_x2 + k4_x2)
        
        self._output = self._x1
        self._time += self._dt
        
        # Apply disturbance and noise
        measured = self._add_disturbance(self._output)
        measured = self._add_noise(measured)
        
        return measured
    
    def reset(self) -> None:
        """Reset plant to initial state."""
        self._x1 = self._initial_output
        self._x2 = self._initial_velocity
        self._output = self._initial_output
        self._time = 0.0
    
    def get_info(self) -> Dict[str, Any]:
        """Get plant parameters."""
        return {
            'type': 'SecondOrderPlant',
            'gain': self._K,
            'natural_frequency': self._wn,
            'damping_ratio': self._zeta,
            'sample_time': self._dt,
        }
    
    @property
    def gain(self) -> float:
        """Static gain K."""
        return self._K
    
    @property
    def natural_frequency(self) -> float:
        """Natural frequency wn."""
        return self._wn
    
    @property
    def damping_ratio(self) -> float:
        """Damping ratio zeta."""
        return self._zeta
    
    @property
    def velocity(self) -> float:
        """Current velocity (dx/dt)."""
        return self._x2
    
    def get_characteristic_times(self) -> Dict[str, float]:
        """
        Calculate characteristic response times.
        
        Returns:
            Dictionary with rise_time, settling_time, peak_time
        """
        zeta = self._zeta
        wn = self._wn
        
        result = {}
        
        if zeta < 1:  # Underdamped
            wd = wn * math.sqrt(1 - zeta**2)  # Damped frequency
            result['damped_frequency'] = wd
            result['peak_time'] = math.pi / wd
            result['rise_time'] = (math.pi - math.atan2(math.sqrt(1-zeta**2), -zeta)) / wd
            result['settling_time_2pct'] = 4 / (zeta * wn)
            result['settling_time_5pct'] = 3 / (zeta * wn)
            result['overshoot_percent'] = 100 * math.exp(-zeta * math.pi / math.sqrt(1 - zeta**2))
        elif zeta == 1:  # Critically damped
            result['rise_time'] = 1.8 / wn
            result['settling_time_2pct'] = 5.8 / wn
            result['settling_time_5pct'] = 4.7 / wn
            result['overshoot_percent'] = 0
        else:  # Overdamped
            p1 = -wn * (zeta - math.sqrt(zeta**2 - 1))
            p2 = -wn * (zeta + math.sqrt(zeta**2 - 1))
            result['pole_1'] = p1
            result['pole_2'] = p2
            result['settling_time_2pct'] = -4 / p1  # Dominant pole
            result['overshoot_percent'] = 0
        
        return result
