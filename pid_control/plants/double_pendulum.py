"""
Double Inverted Pendulum on Cart

A highly nonlinear, underactuated system with two pendulums mounted on a cart.
Uses scipy.integrate for accurate ODE solving.
"""

import numpy as np
from typing import Dict, Any
from scipy.integrate import solve_ivp
from .base_plant import BasePlant


class DoublePendulumCart(BasePlant):
    """
    Double inverted pendulum on a cart.
    
    State vector: [x, x_dot, theta1, theta1_dot, theta2, theta2_dot]
    where:
        x = cart position
        theta1 = angle of first pendulum (from vertical, positive = clockwise)
        theta2 = angle of second pendulum (from vertical, positive = clockwise)
    
    Control input: Force applied to cart
    
    This is a highly nonlinear system that's challenging to stabilize.
    The PID controller will control cart position while trying to keep
    both pendulums upright.
    """
    
    def __init__(
        self,
        cart_mass: float = 1.0,
        pendulum1_mass: float = 0.1,
        pendulum2_mass: float = 0.1,
        pendulum1_length: float = 0.5,
        pendulum2_length: float = 0.5,
        friction: float = 0.1,
        gravity: float = 9.81,
        sample_time: float = 0.01,
        initial_angle1: float = 0.1,
        initial_angle2: float = 0.1,
        control_mode: str = 'position'
    ):
        super().__init__(sample_time)
        
        if cart_mass <= 0:
            raise ValueError("cart_mass must be positive")
        if pendulum1_mass <= 0:
            raise ValueError("pendulum1_mass must be positive")
        if pendulum2_mass <= 0:
            raise ValueError("pendulum2_mass must be positive")
        if pendulum1_length <= 0:
            raise ValueError("pendulum1_length must be positive")
        if pendulum2_length <= 0:
            raise ValueError("pendulum2_length must be positive")
        if friction < 0:
            raise ValueError("friction must be non-negative")
        
        self.M = cart_mass
        self.m1 = pendulum1_mass
        self.m2 = pendulum2_mass
        self.L1 = pendulum1_length
        self.L2 = pendulum2_length
        self.b = friction
        self.g = gravity
        self.control_mode = control_mode
        
        self.initial_angle1 = initial_angle1
        self.initial_angle2 = initial_angle2
        
        # State: [x, x_dot, theta1, theta1_dot, theta2, theta2_dot]
        self.state = np.array([0.0, 0.0, initial_angle1, 0.0, initial_angle2, 0.0])
        
        self._output = 0.0
        self._force = 0.0
    
    def _dynamics(self, state: np.ndarray, force: float) -> np.ndarray:
        """
        Compute state derivatives using equations of motion.
        
        This uses the Euler-Lagrange formulation for the double pendulum on cart.
        """
        x, x_dot, theta1, theta1_dot, theta2, theta2_dot = state
        
        # Precompute common terms
        s1 = np.sin(theta1)
        c1 = np.cos(theta1)
        s2 = np.sin(theta2)
        c2 = np.cos(theta2)
        
        # Mass matrix elements
        M11 = self.M + self.m1 + self.m2
        M12 = (self.m1 + self.m2) * self.L1 * c1
        M13 = self.m2 * self.L2 * c2
        
        M21 = M12
        M22 = (self.m1 + self.m2) * self.L1**2
        M23 = self.m2 * self.L1 * self.L2 * np.cos(theta1 - theta2)
        
        M31 = M13
        M32 = M23
        M33 = self.m2 * self.L2**2
        
        # Mass matrix
        M_matrix = np.array([
            [M11, M12, M13],
            [M21, M22, M23],
            [M31, M32, M33]
        ])
        
        # Right-hand side (forces/torques)
        F1 = (force - self.b * x_dot + 
              (self.m1 + self.m2) * self.L1 * theta1_dot**2 * s1 +
              self.m2 * self.L2 * theta2_dot**2 * s2)
        
        F2 = (-(self.m1 + self.m2) * self.g * self.L1 * s1 +
              self.m2 * self.L1 * self.L2 * theta2_dot**2 * np.sin(theta1 - theta2))
        
        F3 = (-self.m2 * self.g * self.L2 * s2 -
              self.m2 * self.L1 * self.L2 * theta1_dot**2 * np.sin(theta1 - theta2))
        
        F_vector = np.array([F1, F2, F3])
        
        # Solve for accelerations: M * acc = F
        try:
            accelerations = np.linalg.solve(M_matrix, F_vector)
        except np.linalg.LinAlgError:
            # Singular matrix - use pseudo-inverse
            accelerations = np.linalg.lstsq(M_matrix, F_vector, rcond=None)[0]
        
        x_ddot, theta1_ddot, theta2_ddot = accelerations
        
        # Return derivatives
        return np.array([
            x_dot,
            x_ddot,
            theta1_dot,
            theta1_ddot,
            theta2_dot,
            theta2_ddot
        ])
    
    def update(self, control_input: float) -> float:
        """Update system state using scipy.integrate.solve_ivp."""
        self._force = control_input + self._disturbance
        
        # Use scipy's solve_ivp for integration
        sol = solve_ivp(
            lambda t, y: self._dynamics(y, self._force),
            [0, self._dt],
            self.state,
            method='RK45',
            dense_output=False
        )
        
        self.state = sol.y[:, -1]
        
        # Normalize angles to [-pi, pi]
        self.state[2] = np.arctan2(np.sin(self.state[2]), np.cos(self.state[2]))
        self.state[4] = np.arctan2(np.sin(self.state[4]), np.cos(self.state[4]))
        
        # Output depends on control mode
        if self.control_mode == 'position':
            self._output = self.state[0]
        elif self.control_mode == 'angle':
            self._output = (self.state[2] + self.state[4]) / 2.0
        else:
            self._output = self.state[0]
        
        if self._noise_std > 0:
            self._output += np.random.normal(0, self._noise_std)
        
        self._time += self._dt
        return self._output
    
    def reset(self):
        """Reset the system to initial state."""
        self.state = np.array([
            0.0,
            0.0,
            self.initial_angle1,
            0.0,
            self.initial_angle2,
            0.0
        ])
        self._output = 0.0
        self._force = 0.0
        self._time = 0.0
        self._disturbance = 0.0
    
    @property
    def output(self) -> float:
        """Get current output."""
        return self._output
    
    @property
    def cart_position(self) -> float:
        """Get cart position."""
        return self.state[0]
    
    @property
    def cart_velocity(self) -> float:
        """Get cart velocity."""
        return self.state[1]
    
    @property
    def pendulum1_angle(self) -> float:
        """Get first pendulum angle (rad)."""
        return self.state[2]
    
    @property
    def pendulum1_velocity(self) -> float:
        """Get first pendulum angular velocity (rad/s)."""
        return self.state[3]
    
    @property
    def pendulum2_angle(self) -> float:
        """Get second pendulum angle (rad)."""
        return self.state[4]
    
    @property
    def pendulum2_velocity(self) -> float:
        """Get second pendulum angular velocity (rad/s)."""
        return self.state[5]
    
    @property
    def force(self) -> float:
        """Get applied force."""
        return self._force
    
    def get_full_state(self) -> Dict[str, float]:
        """Get complete state information."""
        return {
            'cart_position': self.cart_position,
            'cart_velocity': self.cart_velocity,
            'pendulum1_angle_rad': self.pendulum1_angle,
            'pendulum1_angle_deg': np.degrees(self.pendulum1_angle),
            'pendulum1_velocity': self.pendulum1_velocity,
            'pendulum2_angle_rad': self.pendulum2_angle,
            'pendulum2_angle_deg': np.degrees(self.pendulum2_angle),
            'pendulum2_velocity': self.pendulum2_velocity,
            'force': self.force,
            'time': self.time
        }
    
    def get_energy(self) -> Dict[str, float]:
        """Calculate system energy."""
        x, x_dot, theta1, theta1_dot, theta2, theta2_dot = self.state
        
        # Kinetic energy
        KE_cart = 0.5 * self.M * x_dot**2
        
        # Pendulum 1 kinetic energy
        v1x = x_dot + self.L1 * theta1_dot * np.cos(theta1)
        v1y = self.L1 * theta1_dot * np.sin(theta1)
        KE_pend1 = 0.5 * self.m1 * (v1x**2 + v1y**2)
        
        # Pendulum 2 kinetic energy (relative to pendulum 1 tip)
        v2x = x_dot + self.L1 * theta1_dot * np.cos(theta1) + self.L2 * theta2_dot * np.cos(theta2)
        v2y = self.L1 * theta1_dot * np.sin(theta1) + self.L2 * theta2_dot * np.sin(theta2)
        KE_pend2 = 0.5 * self.m2 * (v2x**2 + v2y**2)
        
        KE_total = KE_cart + KE_pend1 + KE_pend2
        
        # Potential energy (relative to cart level)
        PE_pend1 = self.m1 * self.g * self.L1 * (1 - np.cos(theta1))
        PE_pend2 = self.m2 * self.g * (
            self.L1 * (1 - np.cos(theta1)) + 
            self.L2 * (1 - np.cos(theta2))
        )
        
        PE_total = PE_pend1 + PE_pend2
        
        return {
            'kinetic': KE_total,
            'potential': PE_total,
            'total': KE_total + PE_total
        }
    
    def is_stable(self, angle_threshold: float = 0.2) -> bool:
        """
        Check if both pendulums are approximately upright.
        
        Args:
            angle_threshold: Maximum angle deviation from vertical (rad)
        
        Returns:
            True if both pendulums are within threshold of vertical
        """
        return (abs(self.pendulum1_angle) < angle_threshold and 
                abs(self.pendulum2_angle) < angle_threshold)
    
    def get_info(self) -> Dict[str, Any]:
        """Get plant information."""
        return {
            'type': 'DoublePendulumCart',
            'cart_mass': self.M,
            'pendulum1_mass': self.m1,
            'pendulum2_mass': self.m2,
            'pendulum1_length': self.L1,
            'pendulum2_length': self.L2,
            'friction': self.b,
            'gravity': self.g,
            'control_mode': self.control_mode,
            'sample_time': self.sample_time,
            'initial_angle1_deg': np.degrees(self.initial_angle1),
            'initial_angle2_deg': np.degrees(self.initial_angle2)
        }
