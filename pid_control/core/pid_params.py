"""
PID Controller Parameters Configuration.
Encapsulates all PID settings in a validated, immutable-friendly structure.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum
import json


class AntiWindupMethod(Enum):
    """Anti-windup methods for integral term."""
    NONE = "none"
    CLAMPING = "clamping"
    BACK_CALCULATION = "back_calculation"
    CONDITIONAL_INTEGRATION = "conditional_integration"


class DerivativeMode(Enum):
    """Derivative calculation mode."""
    ERROR = "error"  # Derivative of error (standard)
    MEASUREMENT = "measurement"  # Derivative of measurement (avoids derivative kick)


class ControllerType(Enum):
    """Controller type selection."""
    P = "P"
    PI = "PI"
    PD = "PD"
    PID = "PID"
    PIDA = "PIDA"  # PID with acceleration term


@dataclass
class PIDParams:
    """
    PID Controller Parameters.
    
    Encapsulates all tunable and configuration parameters for a PID controller.
    Provides validation and serialization capabilities.
    """
    
    # Core gains
    kp: float = 1.0  # Proportional gain
    ki: float = 0.0  # Integral gain
    kd: float = 0.0  # Derivative gain
    
    # Sample time
    sample_time: float = 0.01  # Sample time in seconds
    
    # Output limits (saturation)
    output_min: Optional[float] = None  # Minimum output limit
    output_max: Optional[float] = None  # Maximum output limit
    
    # Integral limits (separate from output limits)
    integral_min: Optional[float] = None
    integral_max: Optional[float] = None
    
    # Anti-windup configuration
    anti_windup: AntiWindupMethod = AntiWindupMethod.BACK_CALCULATION
    back_calculation_gain: float = 1.0  # Kb for back-calculation method
    
    # Derivative configuration
    derivative_mode: DerivativeMode = DerivativeMode.MEASUREMENT
    derivative_filter_coeff: float = 10.0  # N coefficient for derivative filter
    
    # Setpoint weighting (for two-degree-of-freedom PID)
    setpoint_weight_p: float = 1.0  # b - proportional setpoint weight
    setpoint_weight_d: float = 0.0  # c - derivative setpoint weight
    
    # Measurement filtering
    measurement_filter_alpha: Optional[float] = None  # Low-pass filter for measurement
    
    # Deadband
    error_deadband: float = 0.0  # Ignore errors smaller than this
    
    # Output rate limiting
    output_rate_limit: Optional[float] = None  # Maximum change per sample
    
    # Controller type
    controller_type: ControllerType = ControllerType.PID
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate all parameters."""
        # Validate gains based on controller type
        if self.kp < 0:
            raise ValueError("kp must be non-negative")
        if self.ki < 0:
            raise ValueError("ki must be non-negative")
        if self.kd < 0:
            raise ValueError("kd must be non-negative")
        
        if self.sample_time <= 0:
            raise ValueError("sample_time must be positive")
        
        # Validate output limits
        if self.output_min is not None and self.output_max is not None:
            if self.output_min >= self.output_max:
                raise ValueError("output_min must be less than output_max")
        
        # Validate integral limits
        if self.integral_min is not None and self.integral_max is not None:
            if self.integral_min >= self.integral_max:
                raise ValueError("integral_min must be less than integral_max")
        
        # Validate setpoint weights
        if not 0 <= self.setpoint_weight_p <= 1:
            raise ValueError("setpoint_weight_p must be in [0, 1]")
        if not 0 <= self.setpoint_weight_d <= 1:
            raise ValueError("setpoint_weight_d must be in [0, 1]")
        
        # Validate filter coefficient
        if self.derivative_filter_coeff <= 0:
            raise ValueError("derivative_filter_coeff must be positive")
        
        # Validate measurement filter
        if self.measurement_filter_alpha is not None:
            if not 0 < self.measurement_filter_alpha <= 1:
                raise ValueError("measurement_filter_alpha must be in (0, 1]")
        
        # Validate deadband
        if self.error_deadband < 0:
            raise ValueError("error_deadband must be non-negative")
        
        # Validate rate limit
        if self.output_rate_limit is not None and self.output_rate_limit <= 0:
            raise ValueError("output_rate_limit must be positive")
        
        # Validate back-calculation gain
        if self.back_calculation_gain < 0:
            raise ValueError("back_calculation_gain must be non-negative")
    
    def copy(self, **changes) -> 'PIDParams':
        """
        Create a copy with optional parameter changes.
        
        Args:
            **changes: Parameters to override
            
        Returns:
            New PIDParams instance
        """
        params = {
            'kp': self.kp,
            'ki': self.ki,
            'kd': self.kd,
            'sample_time': self.sample_time,
            'output_min': self.output_min,
            'output_max': self.output_max,
            'integral_min': self.integral_min,
            'integral_max': self.integral_max,
            'anti_windup': self.anti_windup,
            'back_calculation_gain': self.back_calculation_gain,
            'derivative_mode': self.derivative_mode,
            'derivative_filter_coeff': self.derivative_filter_coeff,
            'setpoint_weight_p': self.setpoint_weight_p,
            'setpoint_weight_d': self.setpoint_weight_d,
            'measurement_filter_alpha': self.measurement_filter_alpha,
            'error_deadband': self.error_deadband,
            'output_rate_limit': self.output_rate_limit,
            'controller_type': self.controller_type,
        }
        params.update(changes)
        return PIDParams(**params)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary of parameters
        """
        return {
            'kp': self.kp,
            'ki': self.ki,
            'kd': self.kd,
            'sample_time': self.sample_time,
            'output_min': self.output_min,
            'output_max': self.output_max,
            'integral_min': self.integral_min,
            'integral_max': self.integral_max,
            'anti_windup': self.anti_windup.value,
            'back_calculation_gain': self.back_calculation_gain,
            'derivative_mode': self.derivative_mode.value,
            'derivative_filter_coeff': self.derivative_filter_coeff,
            'setpoint_weight_p': self.setpoint_weight_p,
            'setpoint_weight_d': self.setpoint_weight_d,
            'measurement_filter_alpha': self.measurement_filter_alpha,
            'error_deadband': self.error_deadband,
            'output_rate_limit': self.output_rate_limit,
            'controller_type': self.controller_type.value,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PIDParams':
        """
        Create from dictionary.
        
        Args:
            data: Dictionary of parameters
            
        Returns:
            PIDParams instance
        """
        data = data.copy()
        
        # Convert enum strings to enums
        if 'anti_windup' in data and isinstance(data['anti_windup'], str):
            data['anti_windup'] = AntiWindupMethod(data['anti_windup'])
        if 'derivative_mode' in data and isinstance(data['derivative_mode'], str):
            data['derivative_mode'] = DerivativeMode(data['derivative_mode'])
        if 'controller_type' in data and isinstance(data['controller_type'], str):
            data['controller_type'] = ControllerType(data['controller_type'])
        
        return cls(**data)
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'PIDParams':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"PIDParams(Kp={self.kp:.4f}, Ki={self.ki:.4f}, Kd={self.kd:.4f}, "
            f"Ts={self.sample_time:.4f}s, "
            f"limits=[{self.output_min}, {self.output_max}], "
            f"anti_windup={self.anti_windup.value})"
        )


# Preset configurations
class PIDPresets:
    """Common PID parameter presets."""
    
    @staticmethod
    def aggressive() -> PIDParams:
        """Fast response, may have overshoot."""
        return PIDParams(
            kp=2.0, ki=1.0, kd=0.5,
            derivative_filter_coeff=20.0,
            anti_windup=AntiWindupMethod.BACK_CALCULATION
        )
    
    @staticmethod
    def moderate() -> PIDParams:
        """Balanced response."""
        return PIDParams(
            kp=1.0, ki=0.5, kd=0.25,
            derivative_filter_coeff=10.0,
            anti_windup=AntiWindupMethod.BACK_CALCULATION
        )
    
    @staticmethod
    def conservative() -> PIDParams:
        """Slow, stable response."""
        return PIDParams(
            kp=0.5, ki=0.2, kd=0.1,
            derivative_filter_coeff=5.0,
            anti_windup=AntiWindupMethod.BACK_CALCULATION
        )
    
    @staticmethod
    def pi_only() -> PIDParams:
        """PI controller (no derivative)."""
        return PIDParams(
            kp=1.0, ki=0.5, kd=0.0,
            controller_type=ControllerType.PI,
            anti_windup=AntiWindupMethod.BACK_CALCULATION
        )
    
    @staticmethod
    def pd_only() -> PIDParams:
        """PD controller (no integral)."""
        return PIDParams(
            kp=1.0, ki=0.0, kd=0.5,
            controller_type=ControllerType.PD,
            derivative_filter_coeff=10.0
        )
