"""
Performance metrics calculation for PID control analysis.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class StepResponseMetrics:
    """Metrics from step response analysis."""
    rise_time: float          # Time to go from 10% to 90% of final value
    settling_time_2pct: float # Time to stay within 2% of final value
    settling_time_5pct: float # Time to stay within 5% of final value
    overshoot_percent: float  # Peak overshoot as percentage
    undershoot_percent: float # Maximum undershoot as percentage
    peak_time: float          # Time to reach peak value
    peak_value: float         # Peak value reached
    steady_state_value: float # Final steady-state value
    steady_state_error: float # Final error from setpoint
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'rise_time': self.rise_time,
            'settling_time_2pct': self.settling_time_2pct,
            'settling_time_5pct': self.settling_time_5pct,
            'overshoot_percent': self.overshoot_percent,
            'undershoot_percent': self.undershoot_percent,
            'peak_time': self.peak_time,
            'peak_value': self.peak_value,
            'steady_state_value': self.steady_state_value,
            'steady_state_error': self.steady_state_error,
        }


@dataclass
class ErrorMetrics:
    """Error-based performance metrics."""
    iae: float    # Integral Absolute Error
    ise: float    # Integral Square Error
    itae: float   # Integral Time-weighted Absolute Error
    itse: float   # Integral Time-weighted Square Error
    mae: float    # Mean Absolute Error
    mse: float    # Mean Square Error
    rmse: float   # Root Mean Square Error
    max_error: float  # Maximum absolute error
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'iae': self.iae,
            'ise': self.ise,
            'itae': self.itae,
            'itse': self.itse,
            'mae': self.mae,
            'mse': self.mse,
            'rmse': self.rmse,
            'max_error': self.max_error,
        }


@dataclass
class ControlEffortMetrics:
    """Metrics related to control effort."""
    total_variation: float    # Sum of |u[k] - u[k-1]|
    mean_absolute: float      # Mean |u|
    max_absolute: float       # Maximum |u|
    rms: float               # RMS of control signal
    saturation_time: float   # Fraction of time at limits
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'total_variation': self.total_variation,
            'mean_absolute': self.mean_absolute,
            'max_absolute': self.max_absolute,
            'rms': self.rms,
            'saturation_time': self.saturation_time,
        }


class PerformanceMetrics:
    """
    Comprehensive performance metrics calculator.
    
    Analyzes PID control data to extract various performance indicators.
    """
    
    def __init__(self):
        """Initialize metrics calculator."""
        pass
    
    def calculate_step_response_metrics(
        self,
        timestamps: np.ndarray,
        setpoints: np.ndarray,
        measurements: np.ndarray,
        initial_value: Optional[float] = None
    ) -> StepResponseMetrics:
        """
        Calculate step response metrics.
        
        Args:
            timestamps: Time values
            setpoints: Setpoint values (assumed constant step)
            measurements: Measured values
            initial_value: Initial value before step (defaults to first measurement)
            
        Returns:
            StepResponseMetrics object
        """
        if len(timestamps) < 2:
            raise ValueError("Need at least 2 data points")
        
        # Determine step parameters
        final_setpoint = setpoints[-1]
        y0 = initial_value if initial_value is not None else measurements[0]
        delta = final_setpoint - y0
        
        if abs(delta) < 1e-10:
            # No step - return default metrics
            return StepResponseMetrics(
                rise_time=0.0,
                settling_time_2pct=0.0,
                settling_time_5pct=0.0,
                overshoot_percent=0.0,
                undershoot_percent=0.0,
                peak_time=0.0,
                peak_value=measurements[-1],
                steady_state_value=measurements[-1],
                steady_state_error=final_setpoint - measurements[-1]
            )
        
        # Normalize response
        y_norm = (measurements - y0) / delta
        
        # Rise time (10% to 90%)
        rise_time = self._find_rise_time(timestamps, y_norm)
        
        # Settling times
        settling_2pct = self._find_settling_time(timestamps, measurements, final_setpoint, 0.02)
        settling_5pct = self._find_settling_time(timestamps, measurements, final_setpoint, 0.05)
        
        # Overshoot and undershoot
        if delta > 0:
            peak_idx = np.argmax(measurements)
            overshoot = max(0, (measurements[peak_idx] - final_setpoint) / delta * 100)
            trough_idx = np.argmin(measurements[peak_idx:]) + peak_idx if peak_idx < len(measurements) - 1 else peak_idx
            undershoot = max(0, (final_setpoint - measurements[trough_idx]) / delta * 100)
        else:
            peak_idx = np.argmin(measurements)
            overshoot = max(0, (final_setpoint - measurements[peak_idx]) / abs(delta) * 100)
            trough_idx = np.argmax(measurements[peak_idx:]) + peak_idx if peak_idx < len(measurements) - 1 else peak_idx
            undershoot = max(0, (measurements[trough_idx] - final_setpoint) / abs(delta) * 100)
        
        peak_time = timestamps[peak_idx]
        peak_value = measurements[peak_idx]
        
        # Steady-state (average of last 10%)
        n_ss = max(1, len(measurements) // 10)
        steady_state_value = np.mean(measurements[-n_ss:])
        steady_state_error = final_setpoint - steady_state_value
        
        return StepResponseMetrics(
            rise_time=rise_time,
            settling_time_2pct=settling_2pct,
            settling_time_5pct=settling_5pct,
            overshoot_percent=overshoot,
            undershoot_percent=undershoot,
            peak_time=peak_time,
            peak_value=peak_value,
            steady_state_value=steady_state_value,
            steady_state_error=steady_state_error
        )
    
    def _find_rise_time(
        self,
        timestamps: np.ndarray,
        y_norm: np.ndarray
    ) -> float:
        """Find rise time from 10% to 90% of normalized response."""
        # Find time at 10%
        t_10 = None
        for i, y in enumerate(y_norm):
            if y >= 0.1:
                if i > 0:
                    # Interpolate
                    t_10 = np.interp(0.1, [y_norm[i-1], y], [timestamps[i-1], timestamps[i]])
                else:
                    t_10 = timestamps[i]
                break
        
        # Find time at 90%
        t_90 = None
        for i, y in enumerate(y_norm):
            if y >= 0.9:
                if i > 0:
                    t_90 = np.interp(0.9, [y_norm[i-1], y], [timestamps[i-1], timestamps[i]])
                else:
                    t_90 = timestamps[i]
                break
        
        if t_10 is None:
            t_10 = timestamps[0]
        if t_90 is None:
            t_90 = timestamps[-1]
        
        return max(0, t_90 - t_10)
    
    def _find_settling_time(
        self,
        timestamps: np.ndarray,
        measurements: np.ndarray,
        final_value: float,
        tolerance: float
    ) -> float:
        """Find settling time within tolerance band."""
        band = tolerance * abs(final_value) if abs(final_value) > 1e-10 else tolerance
        
        within_band = np.abs(measurements - final_value) <= band
        
        # Find last time exiting the band
        for i in range(len(within_band) - 1, -1, -1):
            if not within_band[i]:
                if i < len(timestamps) - 1:
                    return timestamps[i + 1]
                return timestamps[-1]
        
        return 0.0  # Always within band
    
    def calculate_error_metrics(
        self,
        timestamps: np.ndarray,
        setpoints: np.ndarray,
        measurements: np.ndarray
    ) -> ErrorMetrics:
        """
        Calculate error-based metrics.
        
        Args:
            timestamps: Time values
            setpoints: Setpoint values
            measurements: Measured values
            
        Returns:
            ErrorMetrics object
        """
        if len(timestamps) < 2:
            raise ValueError("Need at least 2 data points")
        
        errors = setpoints - measurements
        abs_errors = np.abs(errors)
        sq_errors = errors ** 2
        
        dt = np.diff(timestamps)
        
        # Integral metrics (trapezoidal integration)
        iae = np.sum((abs_errors[:-1] + abs_errors[1:]) / 2 * dt)
        ise = np.sum((sq_errors[:-1] + sq_errors[1:]) / 2 * dt)
        
        # Time-weighted
        t_mid = (timestamps[:-1] + timestamps[1:]) / 2
        itae = np.sum(t_mid * (abs_errors[:-1] + abs_errors[1:]) / 2 * dt)
        itse = np.sum(t_mid * (sq_errors[:-1] + sq_errors[1:]) / 2 * dt)
        
        # Average metrics
        mae = np.mean(abs_errors)
        mse = np.mean(sq_errors)
        rmse = np.sqrt(mse)
        max_error = np.max(abs_errors)
        
        return ErrorMetrics(
            iae=iae,
            ise=ise,
            itae=itae,
            itse=itse,
            mae=mae,
            mse=mse,
            rmse=rmse,
            max_error=max_error
        )
    
    def calculate_control_effort_metrics(
        self,
        timestamps: np.ndarray,
        outputs: np.ndarray,
        output_limits: Optional[Tuple[float, float]] = None
    ) -> ControlEffortMetrics:
        """
        Calculate control effort metrics.
        
        Args:
            timestamps: Time values
            outputs: Control output values
            output_limits: (min, max) output limits for saturation calculation
            
        Returns:
            ControlEffortMetrics object
        """
        if len(outputs) < 2:
            raise ValueError("Need at least 2 data points")
        
        # Total variation
        total_variation = np.sum(np.abs(np.diff(outputs)))
        
        # Absolute metrics
        abs_outputs = np.abs(outputs)
        mean_absolute = np.mean(abs_outputs)
        max_absolute = np.max(abs_outputs)
        rms = np.sqrt(np.mean(outputs ** 2))
        
        # Saturation time
        if output_limits is not None:
            at_limits = (outputs <= output_limits[0] + 1e-10) | (outputs >= output_limits[1] - 1e-10)
            saturation_time = np.mean(at_limits)
        else:
            saturation_time = 0.0
        
        return ControlEffortMetrics(
            total_variation=total_variation,
            mean_absolute=mean_absolute,
            max_absolute=max_absolute,
            rms=rms,
            saturation_time=saturation_time
        )
    
    def calculate_all_metrics(
        self,
        timestamps: np.ndarray,
        setpoints: np.ndarray,
        measurements: np.ndarray,
        outputs: np.ndarray,
        output_limits: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """
        Calculate all available metrics.
        
        Returns:
            Dictionary containing all metrics
        """
        step_metrics = self.calculate_step_response_metrics(
            timestamps, setpoints, measurements
        )
        error_metrics = self.calculate_error_metrics(
            timestamps, setpoints, measurements
        )
        control_metrics = self.calculate_control_effort_metrics(
            timestamps, outputs, output_limits
        )
        
        return {
            'step_response': step_metrics.to_dict(),
            'error': error_metrics.to_dict(),
            'control_effort': control_metrics.to_dict(),
        }
    
    def compare_controllers(
        self,
        results: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple controller results.
        
        Args:
            results: Dictionary mapping controller names to data dicts
                     Each data dict should have 'timestamps', 'setpoints',
                     'measurements', 'outputs' keys
                     
        Returns:
            Dictionary with metrics for each controller
        """
        comparison = {}
        
        for name, data in results.items():
            metrics = self.calculate_all_metrics(
                data['timestamps'],
                data['setpoints'],
                data['measurements'],
                data['outputs'],
                data.get('output_limits')
            )
            comparison[name] = metrics
        
        return comparison
