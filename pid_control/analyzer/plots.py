"""
Comprehensive plotting utilities for PID analysis.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.axes import Axes


class PIDPlotter:
    """
    Professional plotting utilities for PID control analysis.
    
    Provides various visualization methods for understanding
    controller behavior and performance.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-whitegrid'):
        """
        Initialize plotter.
        
        Args:
            style: Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use('seaborn-whitegrid' if 'seaborn-whitegrid' in plt.style.available else 'default')
        
        self._colors = {
            'setpoint': '#2ecc71',
            'measurement': '#3498db',
            'error': '#e74c3c',
            'output': '#9b59b6',
            'p_term': '#f39c12',
            'i_term': '#1abc9c',
            'd_term': '#e67e22',
            'reference': '#7f8c8d',
        }
    
    def plot_response(
        self,
        timestamps: np.ndarray,
        setpoints: np.ndarray,
        measurements: np.ndarray,
        title: str = "PID Response",
        figsize: Tuple[int, int] = (12, 6),
        show_error: bool = True
    ) -> Figure:
        """
        Plot basic PID response with setpoint and measurement.
        
        Args:
            timestamps: Time values
            setpoints: Setpoint values
            measurements: Measured values
            title: Plot title
            figsize: Figure size
            show_error: Whether to show error on secondary axis
            
        Returns:
            Matplotlib Figure
        """
        fig, ax1 = plt.subplots(figsize=figsize)
        
        ax1.plot(timestamps, setpoints, '--', color=self._colors['setpoint'],
                 linewidth=2, label='Setpoint')
        ax1.plot(timestamps, measurements, '-', color=self._colors['measurement'],
                 linewidth=1.5, label='Measurement')
        
        ax1.set_xlabel('Time (s)', fontsize=12)
        ax1.set_ylabel('Value', fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        if show_error:
            ax2 = ax1.twinx()
            errors = setpoints - measurements
            ax2.fill_between(timestamps, 0, errors, alpha=0.2, color=self._colors['error'])
            ax2.plot(timestamps, errors, '-', color=self._colors['error'],
                    linewidth=1, alpha=0.7, label='Error')
            ax2.set_ylabel('Error', fontsize=12, color=self._colors['error'])
            ax2.tick_params(axis='y', labelcolor=self._colors['error'])
            ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        return fig
    
    def plot_comprehensive(
        self,
        timestamps: np.ndarray,
        setpoints: np.ndarray,
        measurements: np.ndarray,
        outputs: np.ndarray,
        p_terms: Optional[np.ndarray] = None,
        i_terms: Optional[np.ndarray] = None,
        d_terms: Optional[np.ndarray] = None,
        title: str = "PID Comprehensive Analysis",
        figsize: Tuple[int, int] = (14, 10)
    ) -> Figure:
        """
        Create comprehensive multi-panel analysis plot.
        
        Args:
            timestamps: Time values
            setpoints: Setpoint values
            measurements: Measured values
            outputs: Control output values
            p_terms: Proportional term values
            i_terms: Integral term values
            d_terms: Derivative term values
            title: Overall title
            figsize: Figure size
            
        Returns:
            Matplotlib Figure
        """
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
        
        # Main response plot (top, spans both columns)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(timestamps, setpoints, '--', color=self._colors['setpoint'],
                 linewidth=2.5, label='Setpoint')
        ax1.plot(timestamps, measurements, '-', color=self._colors['measurement'],
                 linewidth=1.5, label='Measurement')
        ax1.fill_between(timestamps, setpoints, measurements, alpha=0.1,
                        color=self._colors['error'])
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Process Value')
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Error plot (middle left)
        ax2 = fig.add_subplot(gs[1, 0])
        errors = setpoints - measurements
        ax2.plot(timestamps, errors, '-', color=self._colors['error'], linewidth=1.2)
        ax2.fill_between(timestamps, 0, errors, alpha=0.3, color=self._colors['error'])
        ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Error')
        ax2.set_title('Tracking Error', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # Control output plot (middle right)
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(timestamps, outputs, '-', color=self._colors['output'], linewidth=1.2)
        ax3.fill_between(timestamps, 0, outputs, alpha=0.3, color=self._colors['output'])
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Control Output')
        ax3.set_title('Control Signal', fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        # PID components plot (bottom left)
        ax4 = fig.add_subplot(gs[2, 0])
        if p_terms is not None:
            ax4.plot(timestamps, p_terms, '-', color=self._colors['p_term'],
                    linewidth=1.2, label='P')
        if i_terms is not None:
            ax4.plot(timestamps, i_terms, '-', color=self._colors['i_term'],
                    linewidth=1.2, label='I')
        if d_terms is not None:
            ax4.plot(timestamps, d_terms, '-', color=self._colors['d_term'],
                    linewidth=1.2, label='D')
        ax4.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Component Value')
        ax4.set_title('PID Components', fontsize=11)
        ax4.legend(loc='upper right', fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        # Phase portrait / control effort (bottom right)
        ax5 = fig.add_subplot(gs[2, 1])
        if len(errors) > 1:
            error_rate = np.gradient(errors, timestamps)
            ax5.scatter(errors[::5], error_rate[::5], c=timestamps[::5],
                       cmap='viridis', s=10, alpha=0.7)
            ax5.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
            ax5.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
            ax5.set_xlabel('Error')
            ax5.set_ylabel('Error Rate')
            ax5.set_title('Phase Portrait', fontsize=11)
            cbar = plt.colorbar(ax5.collections[0], ax=ax5)
            cbar.set_label('Time (s)', fontsize=9)
        ax5.grid(True, alpha=0.3)
        
        return fig
    
    def plot_pid_components_stacked(
        self,
        timestamps: np.ndarray,
        p_terms: np.ndarray,
        i_terms: np.ndarray,
        d_terms: np.ndarray,
        outputs: np.ndarray,
        title: str = "PID Component Contribution",
        figsize: Tuple[int, int] = (12, 8)
    ) -> Figure:
        """
        Plot stacked area chart showing each PID component's contribution.
        
        Args:
            timestamps: Time values
            p_terms: P component values
            i_terms: I component values
            d_terms: D component values
            outputs: Total output values
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib Figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Stacked area chart
        ax1.fill_between(timestamps, 0, p_terms, alpha=0.7,
                        color=self._colors['p_term'], label='Proportional')
        ax1.fill_between(timestamps, p_terms, p_terms + i_terms, alpha=0.7,
                        color=self._colors['i_term'], label='Integral')
        ax1.fill_between(timestamps, p_terms + i_terms, p_terms + i_terms + d_terms,
                        alpha=0.7, color=self._colors['d_term'], label='Derivative')
        ax1.plot(timestamps, outputs, 'k-', linewidth=1.5, label='Total Output')
        ax1.set_ylabel('Component Value')
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Individual components
        ax2.plot(timestamps, p_terms, '-', color=self._colors['p_term'],
                linewidth=1.5, label='P')
        ax2.plot(timestamps, i_terms, '-', color=self._colors['i_term'],
                linewidth=1.5, label='I')
        ax2.plot(timestamps, d_terms, '-', color=self._colors['d_term'],
                linewidth=1.5, label='D')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Component Value')
        ax2.set_title('Individual Components', fontsize=11)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_frequency_analysis(
        self,
        timestamps: np.ndarray,
        measurements: np.ndarray,
        outputs: np.ndarray,
        title: str = "Frequency Analysis",
        figsize: Tuple[int, int] = (12, 8)
    ) -> Figure:
        """
        Plot frequency domain analysis using FFT.
        
        Args:
            timestamps: Time values (evenly spaced)
            measurements: Measurement signal
            outputs: Control output signal
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Calculate FFT
        dt = timestamps[1] - timestamps[0]
        fs = 1.0 / dt
        n = len(timestamps)
        
        freqs = np.fft.fftfreq(n, dt)[:n//2]
        
        # Measurement FFT
        meas_fft = np.abs(np.fft.fft(measurements - np.mean(measurements)))[:n//2]
        meas_fft_db = 20 * np.log10(meas_fft + 1e-10)
        
        # Output FFT
        out_fft = np.abs(np.fft.fft(outputs - np.mean(outputs)))[:n//2]
        out_fft_db = 20 * np.log10(out_fft + 1e-10)
        
        # Time domain plots
        axes[0, 0].plot(timestamps, measurements, color=self._colors['measurement'])
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Measurement')
        axes[0, 0].set_title('Measurement Signal')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(timestamps, outputs, color=self._colors['output'])
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Control Output')
        axes[0, 1].set_title('Control Signal')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Frequency domain plots
        axes[1, 0].plot(freqs, meas_fft_db, color=self._colors['measurement'])
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('Magnitude (dB)')
        axes[1, 0].set_title('Measurement Spectrum')
        axes[1, 0].set_xlim([0, fs/4])
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(freqs, out_fft_db, color=self._colors['output'])
        axes[1, 1].set_xlabel('Frequency (Hz)')
        axes[1, 1].set_ylabel('Magnitude (dB)')
        axes[1, 1].set_title('Control Output Spectrum')
        axes[1, 1].set_xlim([0, fs/4])
        axes[1, 1].grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_comparison(
        self,
        results: Dict[str, Dict[str, np.ndarray]],
        title: str = "Controller Comparison",
        figsize: Tuple[int, int] = (14, 8)
    ) -> Figure:
        """
        Compare multiple controller results.
        
        Args:
            results: Dictionary mapping names to data dicts
                     Each dict has 'timestamps', 'setpoints', 'measurements', 'outputs'
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
        
        for (name, data), color in zip(results.items(), colors):
            t = data['timestamps']
            sp = data['setpoints']
            meas = data['measurements']
            out = data['outputs']
            err = sp - meas
            
            # Response comparison
            axes[0, 0].plot(t, meas, '-', color=color, linewidth=1.5, label=name)
            
            # Error comparison
            axes[0, 1].plot(t, err, '-', color=color, linewidth=1.2, label=name)
            
            # Control output comparison
            axes[1, 0].plot(t, out, '-', color=color, linewidth=1.2, label=name)
            
            # Error histogram
            axes[1, 1].hist(err, bins=30, alpha=0.5, color=color, label=name)
        
        # Add setpoint to response plot
        first_data = list(results.values())[0]
        axes[0, 0].plot(first_data['timestamps'], first_data['setpoints'],
                       'k--', linewidth=2, label='Setpoint')
        
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Process Value')
        axes[0, 0].set_title('Response Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Error')
        axes[0, 1].set_title('Error Comparison')
        axes[0, 1].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Control Output')
        axes[1, 0].set_title('Control Effort Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_xlabel('Error')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Error Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_metrics_radar(
        self,
        metrics: Dict[str, Dict[str, float]],
        title: str = "Performance Metrics Comparison",
        figsize: Tuple[int, int] = (10, 10)
    ) -> Figure:
        """
        Create radar chart comparing metrics across controllers.
        
        Args:
            metrics: Dict mapping controller names to metric dicts
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib Figure
        """
        # Extract common metrics
        first_metrics = list(metrics.values())[0]
        categories = list(first_metrics.keys())
        n_cats = len(categories)
        
        # Normalize metrics to [0, 1] for radar chart
        all_values = {cat: [] for cat in categories}
        for m in metrics.values():
            for cat in categories:
                all_values[cat].append(m.get(cat, 0))
        
        normalized = {}
        for name, m in metrics.items():
            normalized[name] = []
            for cat in categories:
                vals = all_values[cat]
                min_val, max_val = min(vals), max(vals)
                if max_val - min_val > 1e-10:
                    norm = (m.get(cat, 0) - min_val) / (max_val - min_val)
                else:
                    norm = 0.5
                normalized[name].append(norm)
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon
        
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(metrics)))
        
        for (name, values), color in zip(normalized.items(), colors):
            values = values + values[:1]  # Close the polygon
            ax.plot(angles, values, 'o-', linewidth=2, color=color, label=name)
            ax.fill(angles, values, alpha=0.25, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        return fig
    
    def plot_saturation_analysis(
        self,
        timestamps: np.ndarray,
        outputs: np.ndarray,
        outputs_unsat: np.ndarray,
        output_limits: Tuple[float, float],
        anti_windup_active: Optional[np.ndarray] = None,
        title: str = "Saturation Analysis",
        figsize: Tuple[int, int] = (12, 6)
    ) -> Figure:
        """
        Plot analysis of output saturation and anti-windup.
        
        Args:
            timestamps: Time values
            outputs: Saturated output values
            outputs_unsat: Pre-saturation output values
            output_limits: (min, max) output limits
            anti_windup_active: Boolean array of anti-windup activity
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib Figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Output with saturation
        ax1.plot(timestamps, outputs_unsat, '--', color='gray',
                linewidth=1, alpha=0.7, label='Unsaturated')
        ax1.plot(timestamps, outputs, '-', color=self._colors['output'],
                linewidth=1.5, label='Saturated')
        ax1.axhline(y=output_limits[0], color='red', linestyle=':',
                   alpha=0.5, label='Limits')
        ax1.axhline(y=output_limits[1], color='red', linestyle=':', alpha=0.5)
        ax1.fill_between(timestamps, output_limits[0], output_limits[1],
                        alpha=0.1, color='green')
        ax1.set_ylabel('Control Output')
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Saturation indicator
        is_saturated = (outputs <= output_limits[0] + 1e-10) | (outputs >= output_limits[1] - 1e-10)
        ax2.fill_between(timestamps, 0, is_saturated.astype(float),
                        alpha=0.5, color='red', label='Saturated')
        
        if anti_windup_active is not None:
            ax2.fill_between(timestamps, 0, anti_windup_active.astype(float) * 0.5,
                            alpha=0.5, color='orange', label='Anti-windup Active')
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Status')
        ax2.set_title('Saturation Status')
        ax2.set_ylim([-0.1, 1.1])
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def show():
        """Display all open plots."""
        plt.show()
    
    @staticmethod
    def save(fig: Figure, path: str, dpi: int = 150):
        """Save figure to file."""
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
