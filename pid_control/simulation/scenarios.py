"""
Simulation scenarios for PID testing.
Defines various test scenarios with different setpoint profiles and disturbances.
"""

from typing import Callable, Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np


class SetpointType(Enum):
    """Types of setpoint profiles."""
    STEP = "step"
    RAMP = "ramp"
    SINE = "sine"
    SQUARE = "square"
    STAIRCASE = "staircase"
    CUSTOM = "custom"


class DisturbanceType(Enum):
    """Types of disturbances."""
    NONE = "none"
    STEP = "step"
    PULSE = "pulse"
    SINE = "sine"
    RANDOM = "random"
    CUSTOM = "custom"


@dataclass
class SimulationScenario:
    """
    Defines a complete simulation scenario.
    
    Specifies setpoint profile, disturbances, noise, and timing.
    """
    
    name: str
    duration: float
    sample_time: float = 0.01
    
    # Setpoint configuration
    setpoint_type: SetpointType = SetpointType.STEP
    setpoint_initial: float = 0.0
    setpoint_final: float = 100.0
    setpoint_time: float = 0.0  # Time of setpoint change
    setpoint_params: Optional[Dict[str, Any]] = None
    setpoint_function: Optional[Callable[[float], float]] = None
    
    # Disturbance configuration
    disturbance_type: DisturbanceType = DisturbanceType.NONE
    disturbance_magnitude: float = 0.0
    disturbance_time: float = 0.0
    disturbance_params: Optional[Dict[str, Any]] = None
    disturbance_function: Optional[Callable[[float], float]] = None
    
    # Noise configuration
    measurement_noise_std: float = 0.0
    
    def get_setpoint(self, t: float) -> float:
        """
        Get setpoint value at time t.
        
        Args:
            t: Current time
            
        Returns:
            Setpoint value
        """
        if self.setpoint_function is not None:
            return self.setpoint_function(t)
        
        params = self.setpoint_params or {}
        
        if self.setpoint_type == SetpointType.STEP:
            if t < self.setpoint_time:
                return self.setpoint_initial
            return self.setpoint_final
        
        elif self.setpoint_type == SetpointType.RAMP:
            ramp_duration = params.get('ramp_duration', 1.0)
            if t < self.setpoint_time:
                return self.setpoint_initial
            elif t < self.setpoint_time + ramp_duration:
                progress = (t - self.setpoint_time) / ramp_duration
                return self.setpoint_initial + progress * (self.setpoint_final - self.setpoint_initial)
            return self.setpoint_final
        
        elif self.setpoint_type == SetpointType.SINE:
            amplitude = (self.setpoint_final - self.setpoint_initial) / 2
            offset = (self.setpoint_final + self.setpoint_initial) / 2
            frequency = params.get('frequency', 0.1)
            return offset + amplitude * np.sin(2 * np.pi * frequency * t)
        
        elif self.setpoint_type == SetpointType.SQUARE:
            period = params.get('period', 10.0)
            if int(t / (period / 2)) % 2 == 0:
                return self.setpoint_final
            return self.setpoint_initial
        
        elif self.setpoint_type == SetpointType.STAIRCASE:
            n_steps = params.get('n_steps', 5)
            step_duration = self.duration / n_steps
            current_step = min(int(t / step_duration), n_steps - 1)
            step_size = (self.setpoint_final - self.setpoint_initial) / (n_steps - 1) if n_steps > 1 else 0
            return self.setpoint_initial + current_step * step_size
        
        return self.setpoint_final
    
    def get_disturbance(self, t: float) -> float:
        """
        Get disturbance value at time t.
        
        Args:
            t: Current time
            
        Returns:
            Disturbance value
        """
        if self.disturbance_function is not None:
            return self.disturbance_function(t)
        
        params = self.disturbance_params or {}
        
        if self.disturbance_type == DisturbanceType.NONE:
            return 0.0
        
        elif self.disturbance_type == DisturbanceType.STEP:
            if t >= self.disturbance_time:
                return self.disturbance_magnitude
            return 0.0
        
        elif self.disturbance_type == DisturbanceType.PULSE:
            pulse_duration = params.get('pulse_duration', 1.0)
            if self.disturbance_time <= t < self.disturbance_time + pulse_duration:
                return self.disturbance_magnitude
            return 0.0
        
        elif self.disturbance_type == DisturbanceType.SINE:
            frequency = params.get('frequency', 0.5)
            if t >= self.disturbance_time:
                return self.disturbance_magnitude * np.sin(2 * np.pi * frequency * (t - self.disturbance_time))
            return 0.0
        
        elif self.disturbance_type == DisturbanceType.RANDOM:
            return np.random.normal(0, self.disturbance_magnitude)
        
        return 0.0


class ScenarioLibrary:
    """Pre-defined test scenarios."""
    
    @staticmethod
    def step_response(
        setpoint: float = 100.0,
        duration: float = 20.0,
        sample_time: float = 0.01
    ) -> SimulationScenario:
        """Basic step response test."""
        return SimulationScenario(
            name="Step Response",
            duration=duration,
            sample_time=sample_time,
            setpoint_type=SetpointType.STEP,
            setpoint_initial=0.0,
            setpoint_final=setpoint,
            setpoint_time=1.0
        )
    
    @staticmethod
    def step_with_disturbance(
        setpoint: float = 100.0,
        disturbance: float = 20.0,
        duration: float = 30.0,
        sample_time: float = 0.01
    ) -> SimulationScenario:
        """Step response with load disturbance."""
        return SimulationScenario(
            name="Step with Disturbance",
            duration=duration,
            sample_time=sample_time,
            setpoint_type=SetpointType.STEP,
            setpoint_initial=0.0,
            setpoint_final=setpoint,
            setpoint_time=1.0,
            disturbance_type=DisturbanceType.STEP,
            disturbance_magnitude=disturbance,
            disturbance_time=duration / 2
        )
    
    @staticmethod
    def tracking_sine(
        amplitude: float = 50.0,
        frequency: float = 0.1,
        offset: float = 50.0,
        duration: float = 30.0,
        sample_time: float = 0.01
    ) -> SimulationScenario:
        """Sinusoidal setpoint tracking."""
        return SimulationScenario(
            name="Sine Tracking",
            duration=duration,
            sample_time=sample_time,
            setpoint_type=SetpointType.SINE,
            setpoint_initial=offset - amplitude,
            setpoint_final=offset + amplitude,
            setpoint_params={'frequency': frequency}
        )
    
    @staticmethod
    def staircase_test(
        min_value: float = 0.0,
        max_value: float = 100.0,
        n_steps: int = 5,
        duration: float = 50.0,
        sample_time: float = 0.01
    ) -> SimulationScenario:
        """Staircase setpoint test."""
        return SimulationScenario(
            name="Staircase Test",
            duration=duration,
            sample_time=sample_time,
            setpoint_type=SetpointType.STAIRCASE,
            setpoint_initial=min_value,
            setpoint_final=max_value,
            setpoint_params={'n_steps': n_steps}
        )
    
    @staticmethod
    def noise_rejection(
        setpoint: float = 100.0,
        noise_std: float = 5.0,
        duration: float = 20.0,
        sample_time: float = 0.01
    ) -> SimulationScenario:
        """Test with measurement noise."""
        return SimulationScenario(
            name="Noise Rejection",
            duration=duration,
            sample_time=sample_time,
            setpoint_type=SetpointType.STEP,
            setpoint_initial=0.0,
            setpoint_final=setpoint,
            setpoint_time=1.0,
            measurement_noise_std=noise_std
        )
    
    @staticmethod
    def aggressive_setpoint_changes(
        low: float = 20.0,
        high: float = 80.0,
        period: float = 10.0,
        duration: float = 50.0,
        sample_time: float = 0.01
    ) -> SimulationScenario:
        """Square wave setpoint for aggressive testing."""
        return SimulationScenario(
            name="Aggressive Setpoint Changes",
            duration=duration,
            sample_time=sample_time,
            setpoint_type=SetpointType.SQUARE,
            setpoint_initial=low,
            setpoint_final=high,
            setpoint_params={'period': period}
        )
    
    @staticmethod
    def combined_challenges(
        setpoint: float = 100.0,
        disturbance: float = 15.0,
        noise_std: float = 2.0,
        duration: float = 40.0,
        sample_time: float = 0.01
    ) -> SimulationScenario:
        """Challenging scenario with multiple difficulties."""
        return SimulationScenario(
            name="Combined Challenges",
            duration=duration,
            sample_time=sample_time,
            setpoint_type=SetpointType.STAIRCASE,
            setpoint_initial=0.0,
            setpoint_final=setpoint,
            setpoint_params={'n_steps': 4},
            disturbance_type=DisturbanceType.PULSE,
            disturbance_magnitude=disturbance,
            disturbance_time=duration * 0.6,
            disturbance_params={'pulse_duration': 5.0},
            measurement_noise_std=noise_std
        )
    
    @staticmethod
    def custom(
        name: str,
        duration: float,
        setpoint_func: Callable[[float], float],
        disturbance_func: Optional[Callable[[float], float]] = None,
        noise_std: float = 0.0,
        sample_time: float = 0.01
    ) -> SimulationScenario:
        """Create custom scenario with function-defined profiles."""
        return SimulationScenario(
            name=name,
            duration=duration,
            sample_time=sample_time,
            setpoint_type=SetpointType.CUSTOM,
            setpoint_function=setpoint_func,
            disturbance_type=DisturbanceType.CUSTOM if disturbance_func else DisturbanceType.NONE,
            disturbance_function=disturbance_func,
            measurement_noise_std=noise_std
        )
