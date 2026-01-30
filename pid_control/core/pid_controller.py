"""
Advanced PID Controller Implementation.

Features:
- Proportional, Integral, Derivative control
- Multiple anti-windup methods
- Derivative filtering (avoids noise amplification)
- Derivative on measurement (avoids derivative kick)
- Output saturation with proper integral handling
- Setpoint weighting (2-DOF PID)
- Bumpless transfer for parameter changes
- Measurement filtering
- Error deadband
- Output rate limiting
- Efficient CSV logging
"""

from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
import time

from pid_control.core.pid_params import (
    PIDParams, 
    AntiWindupMethod, 
    DerivativeMode,
    ControllerType
)
from pid_control.core.filters import LowPassFilter, DerivativeFilter
from pid_control.logging.csv_logger import CSVLogger
from pid_control.utils.math_utils import clamp, deadband


@dataclass
class PIDState:
    """Internal state of the PID controller."""
    timestamp: float = 0.0
    setpoint: float = 0.0
    measurement: float = 0.0
    error: float = 0.0
    error_filtered: float = 0.0
    
    # Component outputs
    p_term: float = 0.0
    i_term: float = 0.0
    d_term: float = 0.0
    
    # Pre/post saturation output
    output_unsat: float = 0.0
    output: float = 0.0
    
    # Additional diagnostics
    integral_accumulator: float = 0.0
    derivative_raw: float = 0.0
    anti_windup_active: bool = False
    saturated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            'timestamp': self.timestamp,
            'setpoint': self.setpoint,
            'measurement': self.measurement,
            'error': self.error,
            'error_filtered': self.error_filtered,
            'p_term': self.p_term,
            'i_term': self.i_term,
            'd_term': self.d_term,
            'output_unsat': self.output_unsat,
            'output': self.output,
            'integral_accumulator': self.integral_accumulator,
            'derivative_raw': self.derivative_raw,
            'anti_windup_active': self.anti_windup_active,
            'saturated': self.saturated,
        }


class PIDController:
    """
    Advanced PID Controller with professional-grade features.
    
    Supports all standard PID control enhancements for robust real-world
    performance including filtering, anti-windup, and saturation handling.
    
    Example:
        >>> params = PIDParams(kp=1.0, ki=0.5, kd=0.1, output_min=-10, output_max=10)
        >>> pid = PIDController(params, csv_path="pid_log.csv")
        >>> output = pid.update(setpoint=100.0, measurement=95.0)
    """
    
    def __init__(
        self,
        params: Optional[PIDParams] = None,
        csv_path: Optional[str] = None,
        auto_timestamp: bool = True
    ):
        """
        Initialize PID controller.
        
        Args:
            params: PID parameters (uses defaults if None)
            csv_path: Path for CSV logging (no logging if None)
            auto_timestamp: Use automatic timestamps if True
        """
        self._params = params if params is not None else PIDParams()
        self._auto_timestamp = auto_timestamp
        
        # Internal state
        self._state = PIDState()
        self._prev_measurement: float = 0.0
        self._prev_error: float = 0.0
        self._integral: float = 0.0
        self._prev_output: float = 0.0
        self._iteration: int = 0
        self._start_time: Optional[float] = None
        self._initialized: bool = False
        
        # Filters
        self._derivative_filter = DerivativeFilter(
            sample_time=self._params.sample_time,
            filter_coefficient=self._params.derivative_filter_coeff
        )
        
        self._measurement_filter: Optional[LowPassFilter] = None
        if self._params.measurement_filter_alpha is not None:
            self._measurement_filter = LowPassFilter(
                alpha=self._params.measurement_filter_alpha
            )
        
        # CSV Logger
        self._logger: Optional[CSVLogger] = None
        if csv_path is not None:
            self._logger = CSVLogger(
                csv_path,
                columns=[
                    'iteration', 'timestamp', 'setpoint', 'measurement',
                    'error', 'p_term', 'i_term', 'd_term',
                    'output_unsat', 'output', 'integral_accumulator',
                    'anti_windup_active', 'saturated'
                ]
            )
    
    @property
    def params(self) -> PIDParams:
        """Get current parameters."""
        return self._params
    
    @property
    def state(self) -> PIDState:
        """Get current state."""
        return self._state
    
    @property
    def output(self) -> float:
        """Get current output."""
        return self._state.output
    
    @property
    def integral(self) -> float:
        """Get current integral accumulator."""
        return self._integral
    
    def update(
        self,
        setpoint: float,
        measurement: float,
        timestamp: Optional[float] = None,
        feedforward: float = 0.0
    ) -> float:
        """
        Update PID controller with new setpoint and measurement.
        
        Args:
            setpoint: Desired value
            measurement: Actual measured value
            timestamp: Optional timestamp (auto-generated if not provided)
            feedforward: Optional feedforward term added to output
            
        Returns:
            Control output
        """
        # Handle timestamp
        if timestamp is None and self._auto_timestamp:
            if self._start_time is None:
                self._start_time = time.perf_counter()
            timestamp = time.perf_counter() - self._start_time
        elif timestamp is None:
            timestamp = self._iteration * self._params.sample_time
        
        # Apply measurement filter if configured
        if self._measurement_filter is not None:
            measurement = self._measurement_filter.update(measurement)
        
        # Calculate error
        error = setpoint - measurement
        
        # Apply deadband
        if abs(error) < self._params.error_deadband:
            error = 0.0
        
        # Initialize on first call
        if not self._initialized:
            self._prev_measurement = measurement
            self._prev_error = error
            self._initialized = True
        
        # Calculate P term with setpoint weighting
        error_p = self._params.setpoint_weight_p * setpoint - measurement
        p_term = self._params.kp * error_p
        
        # Calculate I term
        i_term = self._calculate_integral(error)
        
        # Calculate D term
        d_term = self._calculate_derivative(setpoint, measurement, error)
        
        # Sum components
        output_unsat = p_term + i_term + d_term + feedforward
        
        # Apply output saturation
        output = self._apply_saturation(output_unsat)
        
        # Apply anti-windup
        self._apply_anti_windup(output_unsat, output, error)
        
        # Apply rate limiting
        if self._params.output_rate_limit is not None:
            max_change = self._params.output_rate_limit * self._params.sample_time
            delta = output - self._prev_output
            if abs(delta) > max_change:
                output = self._prev_output + max_change * (1 if delta > 0 else -1)
        
        # Update state
        self._state = PIDState(
            timestamp=timestamp,
            setpoint=setpoint,
            measurement=measurement,
            error=error,
            error_filtered=error,
            p_term=p_term,
            i_term=i_term,
            d_term=d_term,
            output_unsat=output_unsat,
            output=output,
            integral_accumulator=self._integral,
            derivative_raw=self._derivative_filter.output,
            anti_windup_active=self._anti_windup_active(output_unsat, output),
            saturated=output != output_unsat
        )
        
        # Log if configured
        if self._logger is not None:
            self._logger.log({
                'iteration': self._iteration,
                'timestamp': timestamp,
                'setpoint': setpoint,
                'measurement': measurement,
                'error': error,
                'p_term': p_term,
                'i_term': i_term,
                'd_term': d_term,
                'output_unsat': output_unsat,
                'output': output,
                'integral_accumulator': self._integral,
                'anti_windup_active': int(self._state.anti_windup_active),
                'saturated': int(self._state.saturated)
            })
        
        # Update history
        self._prev_measurement = measurement
        self._prev_error = error
        self._prev_output = output
        self._iteration += 1
        
        return output
    
    def _calculate_integral(self, error: float) -> float:
        """Calculate integral term with limits."""
        # Trapezoidal integration
        self._integral += self._params.ki * error * self._params.sample_time
        
        # Apply integral limits
        if self._params.integral_min is not None:
            self._integral = max(self._integral, self._params.integral_min)
        if self._params.integral_max is not None:
            self._integral = min(self._integral, self._params.integral_max)
        
        return self._integral
    
    def _calculate_derivative(
        self, 
        setpoint: float, 
        measurement: float, 
        error: float
    ) -> float:
        """Calculate filtered derivative term."""
        if self._params.kd == 0:
            return 0.0
        
        if self._params.derivative_mode == DerivativeMode.MEASUREMENT:
            # Derivative on measurement (avoids derivative kick on setpoint change)
            derivative_input = -measurement
            # With setpoint weighting
            if self._params.setpoint_weight_d > 0:
                derivative_input = (
                    self._params.setpoint_weight_d * setpoint - measurement
                )
        else:
            # Derivative on error
            derivative_input = error
        
        # Use filtered derivative
        derivative = self._derivative_filter.update(derivative_input)
        
        return self._params.kd * derivative
    
    def _apply_saturation(self, output: float) -> float:
        """Apply output saturation limits."""
        return clamp(output, self._params.output_min, self._params.output_max)
    
    def _apply_anti_windup(
        self, 
        output_unsat: float, 
        output_sat: float,
        error: float
    ) -> None:
        """Apply anti-windup correction to integral term."""
        if self._params.anti_windup == AntiWindupMethod.NONE:
            return
        
        if self._params.anti_windup == AntiWindupMethod.CLAMPING:
            # Simple clamping - stop integration when saturated
            if output_sat != output_unsat:
                # Undo last integration step if saturated in same direction
                if (output_unsat > output_sat and error > 0) or \
                   (output_unsat < output_sat and error < 0):
                    self._integral -= self._params.ki * error * self._params.sample_time
        
        elif self._params.anti_windup == AntiWindupMethod.BACK_CALCULATION:
            # Back-calculation method
            saturation_error = output_sat - output_unsat
            self._integral += (
                self._params.back_calculation_gain * 
                saturation_error * 
                self._params.sample_time
            )
        
        elif self._params.anti_windup == AntiWindupMethod.CONDITIONAL_INTEGRATION:
            # Only integrate when output is not saturated OR error is reducing saturation
            if output_sat != output_unsat:
                # Check if error would increase saturation
                if (output_unsat > output_sat and error > 0) or \
                   (output_unsat < output_sat and error < 0):
                    self._integral -= self._params.ki * error * self._params.sample_time
    
    def _anti_windup_active(self, output_unsat: float, output_sat: float) -> bool:
        """Check if anti-windup is currently active."""
        return (
            self._params.anti_windup != AntiWindupMethod.NONE and
            output_sat != output_unsat
        )
    
    def set_params(self, params: PIDParams, bumpless: bool = True) -> None:
        """
        Update controller parameters.
        
        Args:
            params: New parameters
            bumpless: If True, adjust integral for bumpless transfer
        """
        if bumpless and self._initialized:
            # Adjust integral to maintain current output
            # output = kp_new * error + integral_new + kd_new * deriv
            # We want output to stay the same
            old_output = self._state.output
            new_p = params.kp * self._state.error
            new_d = params.kd * self._derivative_filter.output if params.kd > 0 else 0
            self._integral = old_output - new_p - new_d
        
        self._params = params
        
        # Update derivative filter
        self._derivative_filter = DerivativeFilter(
            sample_time=params.sample_time,
            filter_coefficient=params.derivative_filter_coeff
        )
        
        # Update measurement filter
        if params.measurement_filter_alpha is not None:
            self._measurement_filter = LowPassFilter(
                alpha=params.measurement_filter_alpha
            )
        else:
            self._measurement_filter = None
    
    def set_gains(
        self, 
        kp: Optional[float] = None,
        ki: Optional[float] = None,
        kd: Optional[float] = None,
        bumpless: bool = True
    ) -> None:
        """
        Update individual gains.
        
        Args:
            kp: New proportional gain (None to keep current)
            ki: New integral gain (None to keep current)
            kd: New derivative gain (None to keep current)
            bumpless: If True, adjust for bumpless transfer
        """
        new_params = self._params.copy(
            kp=kp if kp is not None else self._params.kp,
            ki=ki if ki is not None else self._params.ki,
            kd=kd if kd is not None else self._params.kd
        )
        self.set_params(new_params, bumpless=bumpless)
    
    def reset(self) -> None:
        """Reset controller state."""
        self._state = PIDState()
        self._prev_measurement = 0.0
        self._prev_error = 0.0
        self._integral = 0.0
        self._prev_output = 0.0
        self._iteration = 0
        self._start_time = None
        self._initialized = False
        
        self._derivative_filter.reset()
        if self._measurement_filter is not None:
            self._measurement_filter.reset()
    
    def set_integral(self, value: float) -> None:
        """
        Manually set integral term.
        
        Useful for bumpless transfer or initialization.
        
        Args:
            value: New integral value
        """
        self._integral = value
    
    def set_output_limits(
        self,
        output_min: Optional[float] = None,
        output_max: Optional[float] = None
    ) -> None:
        """Update output limits."""
        self._params = self._params.copy(
            output_min=output_min,
            output_max=output_max
        )
    
    def flush_log(self) -> None:
        """Flush any buffered log data to disk."""
        if self._logger is not None:
            self._logger.flush()
    
    def close(self) -> None:
        """Close controller and flush logs."""
        if self._logger is not None:
            self._logger.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
    
    def __repr__(self) -> str:
        return f"PIDController({self._params})"
