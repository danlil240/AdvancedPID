"""
First-order plant model using python-control library.
Transfer function: G(s) = K / (tau*s + 1)
"""

from typing import Dict, Any
import numpy as np
import control as ct
from pid_control.plants.base_plant import BasePlant


class FirstOrderPlant(BasePlant):
    """
    First-order (PT1) plant model using python-control library.
    
    Transfer function: G(s) = K / (tau*s + 1)
    """
    
    def __init__(
        self,
        gain: float = 1.0,
        time_constant: float = 1.0,
        sample_time: float = 0.01,
        initial_output: float = 0.0
    ):
        super().__init__(sample_time)
        
        if time_constant <= 0:
            raise ValueError("time_constant must be positive")
        
        self._K = gain
        self._tau = time_constant
        self._initial_output = initial_output
        self._output = initial_output
        
        # Create transfer function using python-control
        self._tf = ct.TransferFunction([gain], [time_constant, 1])
        
        # Convert to discrete state-space for simulation
        self._sys_d = ct.sample_system(ct.tf2ss(self._tf), sample_time, method='zoh')
        self._state = np.zeros((self._sys_d.nstates, 1))
    
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
        self._state = np.zeros((self._sys_d.nstates, 1))
        self._output = self._initial_output
        self._time = 0.0
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'type': 'FirstOrderPlant',
            'gain': self._K,
            'time_constant': self._tau,
            'sample_time': self._dt,
        }
    
    @property
    def transfer_function(self) -> ct.TransferFunction:
        """Get the continuous transfer function."""
        return self._tf
    
    @property
    def gain(self) -> float:
        return self._K
    
    @property
    def time_constant(self) -> float:
        return self._tau
