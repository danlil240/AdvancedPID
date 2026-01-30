"""
Performance metrics calculation for PID control analysis.
Uses numpy for efficient vectorized calculations.
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class StepResponseMetrics:
    """Metrics from step response analysis."""
    rise_time: float
    settling_time_2pct: float
    settling_time_5pct: float
    overshoot_percent: float
    undershoot_percent: float
    peak_time: float
    peak_value: float
    steady_state_value: float
    steady_state_error: float
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class ErrorMetrics:
    """Error-based performance metrics."""
    iae: float
    ise: float
    itae: float
    itse: float
    mae: float
    mse: float
    rmse: float
    max_error: float
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class ControlEffortMetrics:
    """Metrics related to control effort."""
    total_variation: float
    mean_absolute: float
    max_absolute: float
    rms: float
    saturation_time: float
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class PerformanceMetrics:
    """Comprehensive performance metrics calculator using numpy."""
    
    def calculate_step_response_metrics(
        self,
        timestamps: np.ndarray,
        setpoints: np.ndarray,
        measurements: np.ndarray,
        initial_value: Optional[float] = None
    ) -> StepResponseMetrics:
        """Calculate step response metrics."""
        if len(timestamps) < 2:
            raise ValueError("Need at least 2 data points")
        
        final_setpoint = setpoints[-1]
        y0 = initial_value if initial_value is not None else measurements[0]
        delta = final_setpoint - y0
        
        if abs(delta) < 1e-10:
            return StepResponseMetrics(
                rise_time=0.0, settling_time_2pct=0.0, settling_time_5pct=0.0,
                overshoot_percent=0.0, undershoot_percent=0.0, peak_time=0.0,
                peak_value=measurements[-1], steady_state_value=measurements[-1],
                steady_state_error=final_setpoint - measurements[-1]
            )
        
        y_norm = (measurements - y0) / delta
        
        # Rise time using numpy searchsorted
        t_10 = np.interp(0.1, y_norm, timestamps) if np.any(y_norm >= 0.1) else timestamps[0]
        t_90 = np.interp(0.9, y_norm, timestamps) if np.any(y_norm >= 0.9) else timestamps[-1]
        rise_time = max(0, t_90 - t_10)
        
        # Settling times
        settling_2pct = self._find_settling_time(timestamps, measurements, final_setpoint, 0.02)
        settling_5pct = self._find_settling_time(timestamps, measurements, final_setpoint, 0.05)
        
        # Overshoot and undershoot
        if delta > 0:
            peak_idx = np.argmax(measurements)
            overshoot = max(0, (measurements[peak_idx] - final_setpoint) / delta * 100)
            undershoot = max(0, (final_setpoint - np.min(measurements[peak_idx:])) / delta * 100) if peak_idx < len(measurements) - 1 else 0
        else:
            peak_idx = np.argmin(measurements)
            overshoot = max(0, (final_setpoint - measurements[peak_idx]) / abs(delta) * 100)
            undershoot = max(0, (np.max(measurements[peak_idx:]) - final_setpoint) / abs(delta) * 100) if peak_idx < len(measurements) - 1 else 0
        
        n_ss = max(1, len(measurements) // 10)
        steady_state_value = np.mean(measurements[-n_ss:])
        
        return StepResponseMetrics(
            rise_time=rise_time, settling_time_2pct=settling_2pct, settling_time_5pct=settling_5pct,
            overshoot_percent=overshoot, undershoot_percent=undershoot,
            peak_time=timestamps[peak_idx], peak_value=measurements[peak_idx],
            steady_state_value=steady_state_value, steady_state_error=final_setpoint - steady_state_value
        )
    
    def _find_settling_time(self, timestamps: np.ndarray, measurements: np.ndarray, 
                            final_value: float, tolerance: float) -> float:
        """Find settling time within tolerance band."""
        band = tolerance * abs(final_value) if abs(final_value) > 1e-10 else tolerance
        within_band = np.abs(measurements - final_value) <= band
        
        # Find last index outside band
        outside_indices = np.where(~within_band)[0]
        if len(outside_indices) == 0:
            return 0.0
        
        last_outside = outside_indices[-1]
        return timestamps[min(last_outside + 1, len(timestamps) - 1)]
    
    def calculate_error_metrics(self, timestamps: np.ndarray, setpoints: np.ndarray,
                                measurements: np.ndarray) -> ErrorMetrics:
        """Calculate error-based metrics using numpy."""
        if len(timestamps) < 2:
            raise ValueError("Need at least 2 data points")
        
        errors = setpoints - measurements
        abs_errors = np.abs(errors)
        sq_errors = errors ** 2
        
        # Use numpy trapz for integration
        iae = np.trapz(abs_errors, timestamps)
        ise = np.trapz(sq_errors, timestamps)
        itae = np.trapz(timestamps * abs_errors, timestamps)
        itse = np.trapz(timestamps * sq_errors, timestamps)
        
        return ErrorMetrics(
            iae=iae, ise=ise, itae=itae, itse=itse,
            mae=np.mean(abs_errors), mse=np.mean(sq_errors),
            rmse=np.sqrt(np.mean(sq_errors)), max_error=np.max(abs_errors)
        )
    
    def calculate_control_effort_metrics(self, timestamps: np.ndarray, outputs: np.ndarray,
                                         output_limits: Optional[Tuple[float, float]] = None) -> ControlEffortMetrics:
        """Calculate control effort metrics."""
        if len(outputs) < 2:
            raise ValueError("Need at least 2 data points")
        
        saturation_time = 0.0
        if output_limits is not None:
            at_limits = (outputs <= output_limits[0] + 1e-10) | (outputs >= output_limits[1] - 1e-10)
            saturation_time = np.mean(at_limits)
        
        return ControlEffortMetrics(
            total_variation=np.sum(np.abs(np.diff(outputs))),
            mean_absolute=np.mean(np.abs(outputs)),
            max_absolute=np.max(np.abs(outputs)),
            rms=np.sqrt(np.mean(outputs ** 2)),
            saturation_time=saturation_time
        )
    
    def calculate_all_metrics(self, timestamps: np.ndarray, setpoints: np.ndarray,
                              measurements: np.ndarray, outputs: np.ndarray,
                              output_limits: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """Calculate all available metrics."""
        return {
            'step_response': self.calculate_step_response_metrics(timestamps, setpoints, measurements).to_dict(),
            'error': self.calculate_error_metrics(timestamps, setpoints, measurements).to_dict(),
            'control_effort': self.calculate_control_effort_metrics(timestamps, outputs, output_limits).to_dict(),
        }
    
    def compare_controllers(self, results: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, Any]]:
        """Compare multiple controller results."""
        return {
            name: self.calculate_all_metrics(
                data['timestamps'], data['setpoints'], data['measurements'],
                data['outputs'], data.get('output_limits')
            )
            for name, data in results.items()
        }
