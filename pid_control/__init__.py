"""
Advanced PID Control Library
============================

A professional, modular PID control library with:
- Robust PID controller with filtering, anti-windup, saturation
- Real-time tuning capabilities
- Comprehensive analysis and visualization
- Simulation framework for testing

Author: Advanced PID Control Project
"""

from pid_control.core.pid_controller import PIDController
from pid_control.core.pid_params import PIDParams
from pid_control.tuner.realtime_tuner import RealtimeTuner
from pid_control.analyzer.pid_analyzer import PIDAnalyzer
from pid_control.simulation.simulator import Simulator
from pid_control.identification.system_identifier import SystemIdentifier

__version__ = "1.0.0"
__all__ = [
    "PIDController",
    "PIDParams", 
    "RealtimeTuner",
    "PIDAnalyzer",
    "Simulator",
    "SystemIdentifier",
]
