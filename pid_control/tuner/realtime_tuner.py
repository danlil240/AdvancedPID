"""
Real-time PID tuner with online optimization.

Tunes PID parameters in real-time based on actual system response.
"""

from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass
from collections import deque
import numpy as np
import time

from pid_control.core.pid_controller import PIDController
from pid_control.core.pid_params import PIDParams
from pid_control.plants.base_plant import BasePlant
from pid_control.tuner.optimization_methods import (
    BaseTuner,
    GradientFreeTuner,
    BayesianTuner,
    GeneticTuner,
    DifferentialEvolutionTuner,
    TuningResult
)


@dataclass
class CostWeights:
    """Weights for multi-objective cost function."""
    iae: float = 1.0        # Integral Absolute Error
    ise: float = 0.0        # Integral Square Error
    itae: float = 0.0       # Integral Time-weighted Absolute Error
    overshoot: float = 1.0  # Overshoot penalty
    settling: float = 0.5   # Settling time penalty
    control_effort: float = 0.1  # Control effort penalty


class PerformanceEvaluator:
    """Evaluates control performance for tuning."""
    
    def __init__(self, weights: Optional[CostWeights] = None):
        """
        Initialize evaluator.
        
        Args:
            weights: Cost function weights
        """
        self._weights = weights or CostWeights()
    
    def calculate_cost(
        self,
        timestamps: np.ndarray,
        setpoints: np.ndarray,
        measurements: np.ndarray,
        outputs: np.ndarray
    ) -> float:
        """
        Calculate overall cost from performance data.
        
        Args:
            timestamps: Time values
            setpoints: Setpoint values
            measurements: Measured values
            outputs: Control outputs
            
        Returns:
            Total cost value
        """
        if len(timestamps) < 2:
            return float('inf')
        
        errors = setpoints - measurements
        abs_errors = np.abs(errors)
        dt = np.diff(timestamps)
        
        cost = 0.0
        
        # IAE - Integral Absolute Error
        if self._weights.iae > 0:
            iae = np.sum(abs_errors[:-1] * dt)
            cost += self._weights.iae * iae
        
        # ISE - Integral Square Error
        if self._weights.ise > 0:
            ise = np.sum(errors[:-1]**2 * dt)
            cost += self._weights.ise * ise
        
        # ITAE - Integral Time-weighted Absolute Error
        if self._weights.itae > 0:
            itae = np.sum(timestamps[:-1] * abs_errors[:-1] * dt)
            cost += self._weights.itae * itae
        
        # Overshoot penalty
        if self._weights.overshoot > 0:
            overshoot = self._calculate_overshoot(setpoints, measurements)
            cost += self._weights.overshoot * overshoot
        
        # Settling time penalty
        if self._weights.settling > 0:
            settling = self._calculate_settling_time(timestamps, setpoints, measurements)
            cost += self._weights.settling * settling
        
        # Control effort penalty (total variation)
        if self._weights.control_effort > 0:
            control_tv = np.sum(np.abs(np.diff(outputs)))
            cost += self._weights.control_effort * control_tv
        
        return cost
    
    def _calculate_overshoot(
        self,
        setpoints: np.ndarray,
        measurements: np.ndarray
    ) -> float:
        """Calculate maximum overshoot percentage."""
        # Find step changes in setpoint
        final_setpoint = setpoints[-1]
        initial_value = measurements[0]
        
        if abs(final_setpoint - initial_value) < 1e-10:
            return 0.0
        
        if final_setpoint > initial_value:
            peak = np.max(measurements)
            overshoot = max(0, (peak - final_setpoint) / (final_setpoint - initial_value) * 100)
        else:
            trough = np.min(measurements)
            overshoot = max(0, (final_setpoint - trough) / (initial_value - final_setpoint) * 100)
        
        return overshoot
    
    def _calculate_settling_time(
        self,
        timestamps: np.ndarray,
        setpoints: np.ndarray,
        measurements: np.ndarray,
        tolerance: float = 0.02
    ) -> float:
        """Calculate settling time (2% by default)."""
        final_setpoint = setpoints[-1]
        band = tolerance * abs(final_setpoint) if abs(final_setpoint) > 1e-10 else tolerance
        
        # Find last time outside settling band
        within_band = np.abs(measurements - final_setpoint) <= band
        
        if np.all(within_band):
            return 0.0
        
        # Find index where it permanently enters band
        for i in range(len(within_band) - 1, -1, -1):
            if not within_band[i]:
                if i < len(timestamps) - 1:
                    return timestamps[i + 1]
                return timestamps[-1]
        
        return 0.0


class RealtimeTuner:
    """
    Real-time PID tuner that optimizes parameters during operation.
    
    Supports multiple optimization strategies and can tune while
    the controller is running.
    
    Example:
        >>> plant = FirstOrderPlant(gain=2.0, time_constant=1.0)
        >>> params = PIDParams(kp=1.0, ki=0.5, kd=0.1)
        >>> pid = PIDController(params)
        >>> tuner = RealtimeTuner(pid, plant)
        >>> tuner.auto_tune(setpoint=100.0, duration=10.0)
    """
    
    def __init__(
        self,
        controller: PIDController,
        plant: Optional[BasePlant] = None,
        optimizer: str = 'differential_evolution',
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        cost_weights: Optional[CostWeights] = None
    ):
        """
        Initialize real-time tuner.
        
        Args:
            controller: PID controller to tune
            plant: Plant model for simulation (optional for online tuning)
            optimizer: Optimization method ('nelder_mead', 'bayesian', 'genetic', 'differential_evolution')
            bounds: Parameter bounds {'kp': (min, max), 'ki': ..., 'kd': ...}
            cost_weights: Weights for cost function
        """
        self._controller = controller
        self._plant = plant
        self._optimizer_type = optimizer
        
        # Default bounds
        self._bounds = bounds or {
            'kp': (0.0, 100.0),
            'ki': (0.0, 50.0),
            'kd': (0.0, 20.0)
        }
        
        self._evaluator = PerformanceEvaluator(cost_weights)
        
        # Data buffers for online tuning
        self._data_buffer: deque = deque(maxlen=10000)
        
        # Create optimizer
        self._optimizer = self._create_optimizer()
        
        # Tuning state
        self._is_tuning = False
        self._last_result: Optional[TuningResult] = None
    
    def _create_optimizer(self) -> BaseTuner:
        """Create optimizer instance based on type."""
        if self._optimizer_type == 'nelder_mead':
            return GradientFreeTuner(self._bounds)
        elif self._optimizer_type == 'bayesian':
            return BayesianTuner(self._bounds)
        elif self._optimizer_type == 'genetic':
            return GeneticTuner(self._bounds)
        elif self._optimizer_type == 'differential_evolution':
            return DifferentialEvolutionTuner(self._bounds)
        else:
            raise ValueError(f"Unknown optimizer: {self._optimizer_type}")
    
    def auto_tune(
        self,
        setpoint: float,
        duration: float,
        initial_measurement: float = 0.0,
        max_iterations: int = 50,
        apply_result: bool = True
    ) -> TuningResult:
        """
        Automatically tune PID parameters using simulation.
        
        Requires a plant model to be set.
        
        Args:
            setpoint: Target setpoint for tuning
            duration: Simulation duration in seconds
            initial_measurement: Initial process value
            max_iterations: Maximum optimization iterations
            apply_result: If True, apply tuned parameters to controller
            
        Returns:
            TuningResult with optimal parameters
        """
        if self._plant is None:
            raise RuntimeError("Plant model required for auto-tuning")
        
        self._is_tuning = True
        
        def cost_function(kp: float, ki: float, kd: float) -> float:
            return self._simulate_and_evaluate(kp, ki, kd, setpoint, duration, initial_measurement)
        
        self._optimizer.set_cost_function(cost_function)
        
        initial_params = {
            'kp': self._controller.params.kp,
            'ki': self._controller.params.ki,
            'kd': self._controller.params.kd
        }
        
        result = self._optimizer.optimize(initial_params, max_iterations)
        self._last_result = result
        self._is_tuning = False
        
        if apply_result and result.success:
            self._controller.set_gains(
                kp=result.kp,
                ki=result.ki,
                kd=result.kd,
                bumpless=True
            )
        
        return result
    
    def _simulate_and_evaluate(
        self,
        kp: float,
        ki: float,
        kd: float,
        setpoint: float,
        duration: float,
        initial_measurement: float
    ) -> float:
        """Run simulation and calculate cost."""
        # Create temporary controller with test parameters
        test_params = self._controller.params.copy(kp=kp, ki=ki, kd=kd)
        test_controller = PIDController(test_params)
        
        # Reset plant
        self._plant.reset()
        
        # Data collection
        timestamps = []
        setpoints = []
        measurements = []
        outputs = []
        
        dt = self._controller.params.sample_time
        n_steps = int(duration / dt)
        
        measurement = initial_measurement
        
        for i in range(n_steps):
            t = i * dt
            
            # Controller update
            output = test_controller.update(setpoint, measurement)
            
            # Plant update
            measurement = self._plant.update(output)
            
            # Store data
            timestamps.append(t)
            setpoints.append(setpoint)
            measurements.append(measurement)
            outputs.append(output)
        
        # Calculate cost
        cost = self._evaluator.calculate_cost(
            np.array(timestamps),
            np.array(setpoints),
            np.array(measurements),
            np.array(outputs)
        )
        
        return cost
    
    def record_data(
        self,
        timestamp: float,
        setpoint: float,
        measurement: float,
        output: float
    ) -> None:
        """
        Record data point for online tuning.
        
        Args:
            timestamp: Current time
            setpoint: Current setpoint
            measurement: Current measurement
            output: Current control output
        """
        self._data_buffer.append({
            'timestamp': timestamp,
            'setpoint': setpoint,
            'measurement': measurement,
            'output': output
        })
    
    def tune_from_data(
        self,
        max_iterations: int = 50,
        apply_result: bool = True
    ) -> TuningResult:
        """
        Tune using recorded data.
        
        Uses the data buffer to create a cost function that replays
        the recorded trajectory with different parameters.
        
        Args:
            max_iterations: Maximum optimization iterations
            apply_result: If True, apply tuned parameters
            
        Returns:
            TuningResult with optimal parameters
        """
        if len(self._data_buffer) < 10:
            raise RuntimeError("Not enough recorded data for tuning")
        
        if self._plant is None:
            raise RuntimeError("Plant model required for tuning from data")
        
        # Extract data
        data = list(self._data_buffer)
        timestamps = np.array([d['timestamp'] for d in data])
        setpoints = np.array([d['setpoint'] for d in data])
        
        self._is_tuning = True
        
        def cost_function(kp: float, ki: float, kd: float) -> float:
            # Simulate with recorded setpoints
            test_params = self._controller.params.copy(kp=kp, ki=ki, kd=kd)
            test_controller = PIDController(test_params)
            
            self._plant.reset()
            
            measurements = []
            outputs = []
            
            measurement = data[0]['measurement']
            
            for i, sp in enumerate(setpoints):
                output = test_controller.update(sp, measurement)
                measurement = self._plant.update(output)
                measurements.append(measurement)
                outputs.append(output)
            
            return self._evaluator.calculate_cost(
                timestamps,
                setpoints,
                np.array(measurements),
                np.array(outputs)
            )
        
        self._optimizer.set_cost_function(cost_function)
        
        initial_params = {
            'kp': self._controller.params.kp,
            'ki': self._controller.params.ki,
            'kd': self._controller.params.kd
        }
        
        result = self._optimizer.optimize(initial_params, max_iterations)
        self._last_result = result
        self._is_tuning = False
        
        if apply_result and result.success:
            self._controller.set_gains(
                kp=result.kp,
                ki=result.ki,
                kd=result.kd,
                bumpless=True
            )
        
        return result
    
    def ziegler_nichols_step(
        self,
        setpoint: float,
        step_size: float,
        duration: float
    ) -> Dict[str, float]:
        """
        Perform Ziegler-Nichols step response tuning.
        
        Args:
            setpoint: Initial setpoint
            step_size: Step change magnitude
            duration: Test duration
            
        Returns:
            Dictionary with tuning parameters
        """
        if self._plant is None:
            raise RuntimeError("Plant model required")
        
        self._plant.reset()
        
        dt = self._controller.params.sample_time
        n_steps = int(duration / dt)
        
        # Step response test (open loop)
        responses = []
        times = []
        
        for i in range(n_steps):
            t = i * dt
            
            # Apply step at t=0
            u = step_size
            y = self._plant.update(u)
            
            times.append(t)
            responses.append(y)
        
        times = np.array(times)
        responses = np.array(responses)
        
        # Find reaction curve parameters
        final_value = responses[-1]
        gain = final_value / step_size
        
        # Find maximum slope point
        derivatives = np.gradient(responses, times)
        max_slope_idx = np.argmax(derivatives)
        max_slope = derivatives[max_slope_idx]
        
        # Tangent line at inflection point
        # y = m*t + b, solve for dead time L where y=0
        # L = -b/m = t_inflect - y_inflect/m
        t_inflect = times[max_slope_idx]
        y_inflect = responses[max_slope_idx]
        
        L = t_inflect - y_inflect / max_slope  # Dead time
        tau = final_value / max_slope  # Time constant approximation
        
        L = max(L, 0.01)  # Ensure positive
        
        # Ziegler-Nichols PID formulas
        kp = 1.2 * tau / (gain * L)
        ki = kp / (2 * L)
        kd = kp * 0.5 * L
        
        return {
            'kp': kp,
            'ki': ki,
            'kd': kd,
            'K': gain,
            'L': L,
            'tau': tau
        }
    
    def relay_feedback_tune(
        self,
        setpoint: float,
        relay_amplitude: float,
        hysteresis: float = 0.0,
        n_cycles: int = 5,
        max_duration: float = 100.0
    ) -> Dict[str, float]:
        """
        Perform relay feedback (auto-tune) experiment.
        
        This method induces controlled oscillations to determine
        the ultimate gain and period.
        
        Args:
            setpoint: Operating setpoint
            relay_amplitude: Relay output amplitude
            hysteresis: Relay hysteresis (switching band)
            n_cycles: Number of oscillation cycles to measure
            max_duration: Maximum test duration
            
        Returns:
            Dictionary with tuning parameters
        """
        if self._plant is None:
            raise RuntimeError("Plant model required")
        
        self._plant.reset()
        
        dt = self._controller.params.sample_time
        max_steps = int(max_duration / dt)
        
        # Relay state
        relay_output = relay_amplitude
        crossings = []
        times = []
        
        measurement = self._plant.update(0)
        
        for i in range(max_steps):
            t = i * dt
            error = setpoint - measurement
            
            # Relay logic with hysteresis
            if relay_output > 0:  # Currently high
                if error < -hysteresis:
                    relay_output = -relay_amplitude
                    crossings.append(t)
            else:  # Currently low
                if error > hysteresis:
                    relay_output = relay_amplitude
                    crossings.append(t)
            
            measurement = self._plant.update(relay_output)
            times.append(t)
            
            # Check if we have enough cycles
            if len(crossings) >= 2 * n_cycles:
                break
        
        if len(crossings) < 4:
            raise RuntimeError("Could not establish oscillation")
        
        # Calculate period from crossings
        periods = []
        for i in range(2, len(crossings)):
            periods.append(crossings[i] - crossings[i-2])
        
        Tu = np.mean(periods)  # Ultimate period
        
        # Calculate amplitude from response
        # Ultimate gain Ku = 4*d / (pi*a) where d=relay amplitude, a=oscillation amplitude
        # This is an approximation
        Ku = 4 * relay_amplitude / (np.pi * relay_amplitude * 0.5)  # Simplified
        
        # Ziegler-Nichols from ultimate values
        kp = 0.6 * Ku
        ki = kp / (0.5 * Tu)
        kd = kp * 0.125 * Tu
        
        return {
            'kp': kp,
            'ki': ki,
            'kd': kd,
            'Ku': Ku,
            'Tu': Tu
        }
    
    @property
    def is_tuning(self) -> bool:
        """Check if tuning is in progress."""
        return self._is_tuning
    
    @property
    def last_result(self) -> Optional[TuningResult]:
        """Get last tuning result."""
        return self._last_result
    
    @property
    def bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds."""
        return self._bounds
    
    @bounds.setter
    def bounds(self, value: Dict[str, Tuple[float, float]]) -> None:
        """Set parameter bounds."""
        self._bounds = value
        self._optimizer = self._create_optimizer()
    
    def clear_data(self) -> None:
        """Clear recorded data buffer."""
        self._data_buffer.clear()
