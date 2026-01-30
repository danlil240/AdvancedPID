"""
Control system analysis utilities using python-control library.
Provides frequency response, stability analysis, and system characterization.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import control as ct


class ControlSystemAnalyzer:
    """Analyze control systems using python-control library."""
    
    @staticmethod
    def step_response(sys: ct.TransferFunction, t_final: float = 10.0, 
                      num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Get step response of a system."""
        t = np.linspace(0, t_final, num_points)
        t_out, y_out = ct.step_response(sys, t)
        return t_out, y_out
    
    @staticmethod
    def impulse_response(sys: ct.TransferFunction, t_final: float = 10.0,
                         num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Get impulse response of a system."""
        t = np.linspace(0, t_final, num_points)
        t_out, y_out = ct.impulse_response(sys, t)
        return t_out, y_out
    
    @staticmethod
    def bode_data(sys: ct.TransferFunction, omega_range: Optional[Tuple[float, float]] = None,
                  num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get Bode plot data (magnitude in dB, phase in degrees, frequency)."""
        if omega_range:
            omega = np.logspace(np.log10(omega_range[0]), np.log10(omega_range[1]), num_points)
        else:
            omega = None
        
        mag, phase, omega_out = ct.frequency_response(sys, omega)
        mag_db = 20 * np.log10(np.abs(mag))
        phase_deg = np.angle(mag, deg=True)
        return omega_out, mag_db.flatten(), phase_deg.flatten()
    
    @staticmethod
    def stability_margins(sys: ct.TransferFunction) -> Dict[str, float]:
        """Calculate gain and phase margins."""
        gm, pm, wg, wp = ct.margin(sys)
        return {
            'gain_margin': gm,
            'gain_margin_db': 20 * np.log10(gm) if gm != np.inf else np.inf,
            'phase_margin_deg': pm,
            'gain_crossover_freq': wg,
            'phase_crossover_freq': wp,
        }
    
    @staticmethod
    def poles_zeros(sys: ct.TransferFunction) -> Dict[str, np.ndarray]:
        """Get poles and zeros of the system."""
        return {
            'poles': ct.poles(sys),
            'zeros': ct.zeros(sys),
        }
    
    @staticmethod
    def is_stable(sys: ct.TransferFunction) -> bool:
        """Check if system is stable (all poles in left half-plane)."""
        poles = ct.poles(sys)
        return np.all(np.real(poles) < 0)
    
    @staticmethod
    def dc_gain(sys: ct.TransferFunction) -> float:
        """Get DC gain of the system."""
        return float(ct.dcgain(sys))
    
    @staticmethod
    def step_info(sys: ct.TransferFunction) -> Dict[str, float]:
        """Get step response characteristics."""
        return ct.step_info(sys)
    
    @staticmethod
    def pid_transfer_function(kp: float, ki: float, kd: float, 
                              filter_coeff: float = 100.0) -> ct.TransferFunction:
        """Create PID controller transfer function with derivative filter.
        
        C(s) = Kp + Ki/s + Kd*s/(1 + s/N)
        """
        s = ct.TransferFunction([1, 0], [1])
        
        # P term
        C = kp
        
        # I term
        if ki != 0:
            C = C + ki / s
        
        # D term with filter
        if kd != 0:
            C = C + kd * s / (1 + s / filter_coeff)
        
        return C
    
    @staticmethod
    def closed_loop(plant: ct.TransferFunction, 
                    controller: ct.TransferFunction) -> ct.TransferFunction:
        """Create closed-loop transfer function."""
        return ct.feedback(controller * plant, 1)
    
    @staticmethod
    def sensitivity(plant: ct.TransferFunction,
                    controller: ct.TransferFunction) -> ct.TransferFunction:
        """Calculate sensitivity function S = 1/(1 + G*C)."""
        return 1 / (1 + controller * plant)
    
    @staticmethod
    def analyze_closed_loop(plant: ct.TransferFunction, kp: float, ki: float = 0, 
                            kd: float = 0) -> Dict[str, Any]:
        """Complete closed-loop analysis with PID controller."""
        analyzer = ControlSystemAnalyzer
        
        controller = analyzer.pid_transfer_function(kp, ki, kd)
        cl_sys = analyzer.closed_loop(plant, controller)
        
        return {
            'closed_loop_tf': cl_sys,
            'is_stable': analyzer.is_stable(cl_sys),
            'dc_gain': analyzer.dc_gain(cl_sys),
            'step_info': analyzer.step_info(cl_sys),
            'poles': ct.poles(cl_sys),
            'margins': analyzer.stability_margins(controller * plant),
        }
