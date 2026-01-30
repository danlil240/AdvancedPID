"""
Main PID Analyzer class for comprehensive log analysis.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import csv
import numpy as np

from pid_control.analyzer.metrics import (
    PerformanceMetrics,
    StepResponseMetrics,
    ErrorMetrics,
    ControlEffortMetrics
)
from pid_control.analyzer.plots import PIDPlotter


class PIDAnalyzer:
    """
    Comprehensive PID log analyzer.
    
    Loads PID log data from CSV files and provides deep analysis
    with metrics calculation and visualization.
    
    Example:
        >>> analyzer = PIDAnalyzer("pid_log.csv")
        >>> metrics = analyzer.analyze()
        >>> analyzer.plot_comprehensive()
        >>> PIDPlotter.show()
    """
    
    def __init__(
        self,
        csv_path: Optional[str] = None,
        data: Optional[Dict[str, np.ndarray]] = None
    ):
        """
        Initialize analyzer.
        
        Args:
            csv_path: Path to CSV log file
            data: Alternatively, provide data directly as dict
        """
        self._metrics_calc = PerformanceMetrics()
        self._plotter = PIDPlotter()
        
        self._data: Dict[str, np.ndarray] = {}
        self._metrics: Optional[Dict[str, Any]] = None
        
        if csv_path is not None:
            self.load_csv(csv_path)
        elif data is not None:
            self._data = {k: np.array(v) for k, v in data.items()}
    
    def load_csv(self, csv_path: str) -> None:
        """
        Load PID log data from CSV file.
        
        Args:
            csv_path: Path to CSV file
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        data_lists: Dict[str, List[float]] = {}
        
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                for key, value in row.items():
                    if key not in data_lists:
                        data_lists[key] = []
                    try:
                        data_lists[key].append(float(value))
                    except (ValueError, TypeError):
                        # Handle boolean columns
                        if value.lower() in ('true', '1'):
                            data_lists[key].append(1.0)
                        elif value.lower() in ('false', '0'):
                            data_lists[key].append(0.0)
                        else:
                            data_lists[key].append(np.nan)
        
        self._data = {k: np.array(v) for k, v in data_lists.items()}
        self._metrics = None  # Reset computed metrics
    
    def set_data(self, data: Dict[str, Union[np.ndarray, List[float]]]) -> None:
        """
        Set data directly.
        
        Args:
            data: Dictionary of data arrays
        """
        self._data = {k: np.array(v) for k, v in data.items()}
        self._metrics = None
    
    def analyze(self, output_limits: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive analysis.
        
        Args:
            output_limits: Optional output limits for saturation analysis
            
        Returns:
            Dictionary containing all metrics
        """
        if not self._data:
            raise RuntimeError("No data loaded")
        
        timestamps = self._get_column('timestamp')
        setpoints = self._get_column('setpoint')
        measurements = self._get_column('measurement')
        outputs = self._get_column('output')
        
        if timestamps is None or setpoints is None or measurements is None:
            raise ValueError("Required columns (timestamp, setpoint, measurement) not found")
        
        # Calculate all metrics
        self._metrics = self._metrics_calc.calculate_all_metrics(
            timestamps, setpoints, measurements, 
            outputs if outputs is not None else np.zeros_like(timestamps),
            output_limits
        )
        
        # Add additional analysis
        self._metrics['data_summary'] = self._calculate_data_summary()
        self._metrics['stability'] = self._analyze_stability()
        
        return self._metrics
    
    def _calculate_data_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics of the data."""
        timestamps = self._get_column('timestamp')
        
        summary = {
            'duration': timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0,
            'n_samples': len(timestamps),
            'sample_rate': len(timestamps) / (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0,
        }
        
        for col in ['setpoint', 'measurement', 'output', 'error']:
            data = self._get_column(col)
            if data is not None:
                summary[f'{col}_mean'] = np.mean(data)
                summary[f'{col}_std'] = np.std(data)
                summary[f'{col}_min'] = np.min(data)
                summary[f'{col}_max'] = np.max(data)
        
        return summary
    
    def _analyze_stability(self) -> Dict[str, Any]:
        """Analyze system stability characteristics."""
        measurements = self._get_column('measurement')
        timestamps = self._get_column('timestamp')
        
        if measurements is None or len(measurements) < 10:
            return {'stable': None, 'oscillating': None}
        
        # Check for oscillation using zero-crossings of derivative
        deriv = np.gradient(measurements, timestamps)
        zero_crossings = np.where(np.diff(np.signbit(deriv)))[0]
        
        # Check last 20% of data for stability
        n_tail = max(10, len(measurements) // 5)
        tail_std = np.std(measurements[-n_tail:])
        tail_mean = np.mean(measurements[-n_tail:])
        
        # Coefficient of variation in tail
        cv = tail_std / abs(tail_mean) if abs(tail_mean) > 1e-10 else tail_std
        
        # Oscillation frequency
        if len(zero_crossings) > 2:
            periods = np.diff(timestamps[zero_crossings])
            avg_period = np.mean(periods) * 2  # Full period is 2 half-periods
            oscillation_freq = 1.0 / avg_period if avg_period > 0 else 0
        else:
            oscillation_freq = 0
        
        return {
            'stable': cv < 0.05,  # Less than 5% variation
            'oscillating': oscillation_freq > 0.1,  # Significant oscillation
            'oscillation_frequency': oscillation_freq,
            'tail_coefficient_of_variation': cv,
            'n_zero_crossings': len(zero_crossings),
        }
    
    def get_step_response_metrics(self) -> StepResponseMetrics:
        """Get step response metrics."""
        timestamps = self._get_column('timestamp')
        setpoints = self._get_column('setpoint')
        measurements = self._get_column('measurement')
        
        return self._metrics_calc.calculate_step_response_metrics(
            timestamps, setpoints, measurements
        )
    
    def get_error_metrics(self) -> ErrorMetrics:
        """Get error-based metrics."""
        timestamps = self._get_column('timestamp')
        setpoints = self._get_column('setpoint')
        measurements = self._get_column('measurement')
        
        return self._metrics_calc.calculate_error_metrics(
            timestamps, setpoints, measurements
        )
    
    def get_control_effort_metrics(
        self,
        output_limits: Optional[Tuple[float, float]] = None
    ) -> ControlEffortMetrics:
        """Get control effort metrics."""
        timestamps = self._get_column('timestamp')
        outputs = self._get_column('output')
        
        return self._metrics_calc.calculate_control_effort_metrics(
            timestamps, outputs, output_limits
        )
    
    def _get_column(self, name: str) -> Optional[np.ndarray]:
        """Get column data by name."""
        return self._data.get(name)
    
    def plot_response(self, **kwargs) -> 'Figure':
        """Plot basic response."""
        timestamps = self._get_column('timestamp')
        setpoints = self._get_column('setpoint')
        measurements = self._get_column('measurement')
        
        return self._plotter.plot_response(
            timestamps, setpoints, measurements, **kwargs
        )
    
    def plot_comprehensive(self, **kwargs) -> 'Figure':
        """Plot comprehensive analysis."""
        timestamps = self._get_column('timestamp')
        setpoints = self._get_column('setpoint')
        measurements = self._get_column('measurement')
        outputs = self._get_column('output')
        p_terms = self._get_column('p_term')
        i_terms = self._get_column('i_term')
        d_terms = self._get_column('d_term')
        
        return self._plotter.plot_comprehensive(
            timestamps, setpoints, measurements, outputs,
            p_terms, i_terms, d_terms, **kwargs
        )
    
    def plot_pid_components(self, **kwargs) -> 'Figure':
        """Plot PID component breakdown."""
        timestamps = self._get_column('timestamp')
        p_terms = self._get_column('p_term')
        i_terms = self._get_column('i_term')
        d_terms = self._get_column('d_term')
        outputs = self._get_column('output')
        
        if any(x is None for x in [p_terms, i_terms, d_terms]):
            raise ValueError("PID component data not available")
        
        return self._plotter.plot_pid_components_stacked(
            timestamps, p_terms, i_terms, d_terms, outputs, **kwargs
        )
    
    def plot_frequency_analysis(self, **kwargs) -> 'Figure':
        """Plot frequency domain analysis."""
        timestamps = self._get_column('timestamp')
        measurements = self._get_column('measurement')
        outputs = self._get_column('output')
        
        return self._plotter.plot_frequency_analysis(
            timestamps, measurements, outputs, **kwargs
        )
    
    def plot_saturation(
        self,
        output_limits: Tuple[float, float],
        **kwargs
    ) -> 'Figure':
        """Plot saturation analysis."""
        timestamps = self._get_column('timestamp')
        outputs = self._get_column('output')
        outputs_unsat = self._get_column('output_unsat')
        anti_windup = self._get_column('anti_windup_active')
        
        if outputs_unsat is None:
            outputs_unsat = outputs
        
        return self._plotter.plot_saturation_analysis(
            timestamps, outputs, outputs_unsat, output_limits,
            anti_windup, **kwargs
        )
    
    def generate_report(self) -> str:
        """
        Generate text report of analysis.
        
        Returns:
            Formatted text report
        """
        if self._metrics is None:
            self.analyze()
        
        lines = [
            "=" * 60,
            "PID CONTROL ANALYSIS REPORT",
            "=" * 60,
            "",
            "DATA SUMMARY",
            "-" * 40,
        ]
        
        summary = self._metrics.get('data_summary', {})
        lines.extend([
            f"  Duration: {summary.get('duration', 0):.2f} s",
            f"  Samples: {summary.get('n_samples', 0)}",
            f"  Sample Rate: {summary.get('sample_rate', 0):.1f} Hz",
            "",
        ])
        
        lines.extend([
            "STEP RESPONSE METRICS",
            "-" * 40,
        ])
        
        step = self._metrics.get('step_response', {})
        lines.extend([
            f"  Rise Time: {step.get('rise_time', 0):.4f} s",
            f"  Settling Time (2%): {step.get('settling_time_2pct', 0):.4f} s",
            f"  Settling Time (5%): {step.get('settling_time_5pct', 0):.4f} s",
            f"  Overshoot: {step.get('overshoot_percent', 0):.2f} %",
            f"  Peak Time: {step.get('peak_time', 0):.4f} s",
            f"  Steady-State Error: {step.get('steady_state_error', 0):.6f}",
            "",
        ])
        
        lines.extend([
            "ERROR METRICS",
            "-" * 40,
        ])
        
        error = self._metrics.get('error', {})
        lines.extend([
            f"  IAE: {error.get('iae', 0):.4f}",
            f"  ISE: {error.get('ise', 0):.4f}",
            f"  ITAE: {error.get('itae', 0):.4f}",
            f"  RMSE: {error.get('rmse', 0):.6f}",
            f"  Max Error: {error.get('max_error', 0):.6f}",
            "",
        ])
        
        lines.extend([
            "CONTROL EFFORT METRICS",
            "-" * 40,
        ])
        
        control = self._metrics.get('control_effort', {})
        lines.extend([
            f"  Total Variation: {control.get('total_variation', 0):.4f}",
            f"  RMS Control: {control.get('rms', 0):.4f}",
            f"  Max Control: {control.get('max_absolute', 0):.4f}",
            f"  Saturation Time: {control.get('saturation_time', 0)*100:.1f} %",
            "",
        ])
        
        lines.extend([
            "STABILITY ANALYSIS",
            "-" * 40,
        ])
        
        stability = self._metrics.get('stability', {})
        lines.extend([
            f"  Stable: {stability.get('stable', 'Unknown')}",
            f"  Oscillating: {stability.get('oscillating', 'Unknown')}",
            f"  Oscillation Freq: {stability.get('oscillation_frequency', 0):.4f} Hz",
            "",
            "=" * 60,
        ])
        
        return "\n".join(lines)
    
    def save_report(self, path: str) -> None:
        """Save report to text file."""
        report = self.generate_report()
        with open(path, 'w') as f:
            f.write(report)
    
    @property
    def data(self) -> Dict[str, np.ndarray]:
        """Get raw data."""
        return self._data
    
    @property
    def columns(self) -> List[str]:
        """Get available column names."""
        return list(self._data.keys())
    
    @staticmethod
    def show_plots():
        """Display all plots."""
        PIDPlotter.show()
