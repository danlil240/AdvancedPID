"""
Second-order plant model using python-control library.
Transfer function: G(s) = K * wn^2 / (s^2 + 2*zeta*wn*s + wn^2)
"""

from typing import Dict, Any
import numpy as np
import control as ct
from pid_control.plants.base_plant import BasePlant


class SecondOrderPlant(BasePlant):
    """
    Second-order plant model using python-control library.
    
    Transfer function: G(s) = K * wn^2 / (s^2 + 2*zeta*wn*s + wn^2)
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
        
        # Create transfer function: G(s) = K*wn^2 / (s^2 + 2*zeta*wn*s + wn^2)
        wn2 = natural_frequency ** 2
        self._tf = ct.TransferFunction(
            [gain * wn2],
            [1, 2 * damping_ratio * natural_frequency, wn2]
        )
        
        # Convert to discrete state-space for simulation
        self._sys_d = ct.sample_system(ct.tf2ss(self._tf), sample_time, method='zoh')
        self._state = np.array([[initial_output], [initial_velocity]])
        self._output = initial_output
    
    def update(self, control_input: float) -> float:
        """Update plant using discrete state-space model."""
        u = np.array([[control_input]])
        
        # State-space update: x[k+1] = A*x[k] + B*u[k], y[k] = C*x[k] + D*u[k]
        y = self._sys_d.C @ self._state + self._sys_d.D @ u
        self._state = self._sys_d.A @ self._state + self._sys_d.B @ u
        
        self._output = float(y[0, 0])
        self._time += self._dt
        
        measured = self._add_disturbance(self._output)
        measured = self._add_noise(measured)
        return measured
    
    def reset(self) -> None:
        self._state = np.array([[self._initial_output], [self._initial_velocity]])
        self._output = self._initial_output
        self._time = 0.0
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'type': 'SecondOrderPlant',
            'gain': self._K,
            'natural_frequency': self._wn,
            'damping_ratio': self._zeta,
            'sample_time': self._dt,
        }
    
    @property
    def transfer_function(self) -> ct.TransferFunction:
        """Get the continuous transfer function."""
        return self._tf
    
    @property
    def poles(self) -> np.ndarray:
        """Get system poles using python-control."""
        return ct.poles(self._tf)
    
    @property
    def gain(self) -> float:
        return self._K
    
    @property
    def natural_frequency(self) -> float:
        return self._wn
    
    @property
    def damping_ratio(self) -> float:
        return self._zeta
    
    @property
    def velocity(self) -> float:
        return float(self._state[1, 0])
    
    def get_characteristic_times(self) -> Dict[str, float]:
        """Calculate characteristic response times using python-control."""
        zeta = self._zeta
        wn = self._wn
        result = {}
        
        # Use python-control for step response info
        info = ct.step_info(self._tf)
        result['rise_time'] = info.get('RiseTime', 0)
        result['settling_time'] = info.get('SettlingTime', 0)
        result['overshoot_percent'] = info.get('Overshoot', 0)
        result['peak_time'] = info.get('PeakTime', 0)
        
        if zeta < 1:  # Underdamped
            result['damped_frequency'] = wn * np.sqrt(1 - zeta**2)
        
        return result
