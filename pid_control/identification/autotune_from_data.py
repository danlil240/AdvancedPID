"""
Integration between system identification and autotuning.

Provides a complete workflow from CSV data to optimized PID gains.
"""

from typing import Dict, Optional, Callable, List
import numpy as np
from dataclasses import dataclass

from pid_control.identification.csv_reader import CSVDataReader, ExperimentalData
from pid_control.identification.system_identifier import (
    SystemIdentifier,
    ModelType,
    IdentificationResult,
    TransferFunctionModel
)
from pid_control.tuner.optimization_methods import (
    BaseTuner,
    GradientFreeTuner,
    BayesianTuner,
    GeneticTuner,
    DifferentialEvolutionTuner,
    TuningResult
)


@dataclass
class AutotuneFromDataResult:
    """Complete result from data-driven autotuning."""
    identification: IdentificationResult
    initial_gains: Dict[str, float]
    optimized_gains: Dict[str, float]
    tuning_result: TuningResult
    improvement: float
    
    def summary(self) -> str:
        """Get comprehensive summary."""
        lines = [
            "=" * 70,
            "DATA-DRIVEN PID AUTOTUNING RESULTS",
            "=" * 70,
            "",
            "STEP 1: SYSTEM IDENTIFICATION",
            "-" * 70,
            f"Model: {self.identification.model}",
            f"Fit Quality (R²): {self.identification.fit_quality:.4f}",
            "",
            "STEP 2: INITIAL TUNING (Analytical)",
            "-" * 70,
            f"Tuning Rule: {self.identification.tuning_rule}",
            f"  Kp = {self.initial_gains['kp']:.4f}",
            f"  Ki = {self.initial_gains['ki']:.4f}",
            f"  Kd = {self.initial_gains['kd']:.4f}",
            "",
            "STEP 3: OPTIMIZATION (Numerical)",
            "-" * 70,
            f"Method: {self.tuning_result.message}",
            f"Iterations: {self.tuning_result.iterations}",
            f"Final Cost: {self.tuning_result.cost:.4f}",
            "",
            "OPTIMIZED PID GAINS:",
            f"  Kp = {self.optimized_gains['kp']:.4f}",
            f"  Ki = {self.optimized_gains['ki']:.4f}",
            f"  Kd = {self.optimized_gains['kd']:.4f}",
            "",
            f"Performance Improvement: {self.improvement:.2f}%",
            "=" * 70
        ]
        return "\n".join(lines)


@dataclass
class PerformanceRequirements:
    """Performance requirements to guide PID optimization."""
    max_overshoot_pct: Optional[float] = None
    max_settling_time: Optional[float] = None
    settling_band: float = 0.02
    overshoot_penalty_weight: float = 200.0
    settling_penalty_weight: float = 200.0


class AutotuneFromData:
    """
    Complete workflow for PID autotuning from experimental CSV data.
    
    Workflow:
    1. Load CSV data (input, output, time)
    2. Identify system transfer function
    3. Apply analytical tuning rule (Ziegler-Nichols, Cohen-Coon, etc.)
    4. Optimize gains using numerical optimization
    5. Return optimized gains with analysis
    """
    
    def __init__(
        self,
        csv_path: str,
        time_col: str = 'timestamp',
        input_col: str = 'output',
        output_col: str = 'measurement',
        setpoint_col: Optional[str] = 'setpoint'
    ):
        """
        Initialize autotuner from CSV data.
        
        Args:
            csv_path: Path to CSV file with experimental data
            time_col: Name of time column (default: 'timestamp' to match CSVLogger)
            input_col: Name of control input column (default: 'output' to match CSVLogger)
            output_col: Name of process output column (default: 'measurement' to match CSVLogger)
            setpoint_col: Name of setpoint column (optional)
        """
        self.csv_path = csv_path
        self.reader = CSVDataReader(csv_path)
        self.data = self.reader.read(
            time_col=time_col,
            input_col=input_col,
            output_col=output_col,
            setpoint_col=setpoint_col
        )
        self.identifier = SystemIdentifier(self.data)
    
    def autotune(
        self,
        model_type: ModelType = ModelType.AUTO,
        tuning_rule: str = 'ziegler_nichols',
        optimizer: str = 'differential_evolution',
        bounds_scale: float = 2.0,
        max_iterations: int = 50,
        cost_function: Optional[Callable] = None,
        requirements: Optional[PerformanceRequirements] = None,
        prompt_for_requirements: bool = False
    ) -> AutotuneFromDataResult:
        """
        Perform complete autotuning from data.
        
        Args:
            model_type: Transfer function model type (FOPDT, SOPDT, or AUTO for best fit)
            tuning_rule: Analytical tuning rule to use as starting point
            optimizer: Optimization method ('gradient_free', 'bayesian', 'genetic', 'differential_evolution')
            bounds_scale: Scale factor for parameter bounds around initial guess
            max_iterations: Maximum optimization iterations
            cost_function: Custom cost function (if None, uses default)
            requirements: Performance requirements (overshoot %, settling time)
            prompt_for_requirements: Prompt for requirements if not provided
        
        Returns:
            AutotuneFromDataResult with complete analysis
        """
        print("=" * 70)
        print("STARTING DATA-DRIVEN PID AUTOTUNING")
        print("=" * 70)
        
        print("\nStep 1: System Identification...")
        id_result = self.identifier.identify(
            model_type=model_type,
            tuning_rule=tuning_rule
        )
        print(f"  Model identified: {id_result.model}")
        print(f"  Fit quality (R²): {id_result.fit_quality:.4f}")
        
        initial_gains = id_result.recommended_gains
        print(f"\nStep 2: Initial Gains from {tuning_rule}:")
        print(f"  Kp = {initial_gains['kp']:.4f}")
        print(f"  Ki = {initial_gains['ki']:.4f}")
        print(f"  Kd = {initial_gains['kd']:.4f}")
        
        print(f"\nStep 3: Optimizing with {optimizer}...")
        
        bounds = self._create_bounds(initial_gains, bounds_scale)
        
        if requirements is None and prompt_for_requirements:
            requirements = self._prompt_requirements()
        
        if requirements is not None:
            print("\nPerformance requirements:")
            if requirements.max_overshoot_pct is not None:
                print(f"  Max overshoot: {requirements.max_overshoot_pct:.2f}%")
            if requirements.max_settling_time is not None:
                print(f"  Max settling time: {requirements.max_settling_time:.4f}s")
            print(f"  Settling band: ±{requirements.settling_band * 100:.1f}%")
        
        if cost_function is None:
            cost_function = self._create_default_cost_function(id_result.model, requirements)
        
        tuner = self._create_tuner(optimizer, bounds, cost_function)
        
        tuning_result = tuner.optimize(initial_gains, max_iterations=max_iterations)
        
        optimized_gains = {
            'kp': tuning_result.kp,
            'ki': tuning_result.ki,
            'kd': tuning_result.kd
        }
        
        print(f"\nOptimization complete!")
        print(f"  Iterations: {tuning_result.iterations}")
        print(f"  Final cost: {tuning_result.cost:.4f}")
        print(f"\nOptimized Gains:")
        print(f"  Kp = {optimized_gains['kp']:.4f}")
        print(f"  Ki = {optimized_gains['ki']:.4f}")
        print(f"  Kd = {optimized_gains['kd']:.4f}")
        
        initial_cost = cost_function(
            initial_gains['kp'],
            initial_gains['ki'],
            initial_gains['kd']
        )
        improvement = ((initial_cost - tuning_result.cost) / initial_cost) * 100 if initial_cost > 0 else 0
        
        print(f"\nPerformance improvement: {improvement:.2f}%")
        
        return AutotuneFromDataResult(
            identification=id_result,
            initial_gains=initial_gains,
            optimized_gains=optimized_gains,
            tuning_result=tuning_result,
            improvement=improvement
        )
    
    def compare_tuning_rules(self) -> Dict[str, Dict[str, float]]:
        """
        Compare different tuning rules for the identified system.
        
        Returns:
            Dictionary mapping rule names to PID gains
        """
        return self.identifier.compare_tuning_rules()
    
    def _create_bounds(
        self,
        initial_gains: Dict[str, float],
        scale: float
    ) -> Dict[str, tuple]:
        """Create parameter bounds around initial guess."""
        bounds = {}
        for param, value in initial_gains.items():
            if value < 1e-6:
                bounds[param] = (0.0, 1.0 * scale)
            else:
                bounds[param] = (
                    max(0.0, value / scale),
                    value * scale
                )
        return bounds
    
    def _create_tuner(
        self,
        optimizer: str,
        bounds: Dict[str, tuple],
        cost_function: Callable
    ) -> BaseTuner:
        """Create optimizer instance."""
        if optimizer == 'gradient_free':
            return GradientFreeTuner(bounds, cost_function)
        elif optimizer == 'bayesian':
            return BayesianTuner(bounds, cost_function, n_initial=5)
        elif optimizer == 'genetic':
            return GeneticTuner(bounds, cost_function, population_size=20)
        elif optimizer == 'differential_evolution':
            return DifferentialEvolutionTuner(bounds, cost_function, population_size=15)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
    
    def _create_default_cost_function(
        self,
        model: TransferFunctionModel,
        requirements: Optional[PerformanceRequirements] = None
    ) -> Callable:
        """
        Create default cost function based on identified model.
        
        Simulates closed-loop response and evaluates performance.
        """
        from scipy import signal
        
        t = self.data.time
        dt = self.data.sample_time or 0.01
        
        if model.model_type == "FOPDT":
            num = [model.K]
            den = [model.tau, 1]
        elif model.model_type == "SECOND_ORDER":
            # G(s) = K*wn^2 / (s^2 + 2*zeta*wn*s + wn^2)
            wn = 1.0 / model.tau if model.tau > 0 else 1.0
            zeta = model.zeta if model.zeta is not None else 0.7
            wn2 = wn ** 2
            num = [model.K * wn2]
            den = [1, 2 * zeta * wn, wn2]
        else:  # SOPDT
            num = [model.K]
            den = [model.tau * model.tau2, model.tau + model.tau2, 1]
        
        plant_sys = signal.TransferFunction(num, den)
        plant_discrete = signal.cont2discrete((plant_sys.num, plant_sys.den), dt, method='zoh')
        
        def cost_function(kp: float, ki: float, kd: float) -> float:
            """Evaluate PID performance on identified model using time-domain simulation."""
            if kp < 0 or ki < 0 or kd < 0:
                return 1e10
            
            if kp > 1000 or ki > 1000 or kd > 1000:
                return 1e10
            
            try:
                n_steps = len(t)
                setpoint = 1.0
                
                y = np.zeros(n_steps)
                u = np.zeros(n_steps)
                error_integral = 0.0
                error_prev = 0.0
                
                delay_samples = int(model.theta / dt) if model.theta > 0 else 0
                u_delayed = np.zeros(max(delay_samples + 1, 10))
                
                y1 = 0.0
                y2 = 0.0
                
                for i in range(n_steps):
                    error = setpoint - y[i]
                    error_integral += error * dt
                    error_derivative = (error - error_prev) / dt if i > 0 else 0.0
                    
                    u[i] = kp * error + ki * error_integral + kd * error_derivative
                    u[i] = np.clip(u[i], -10, 10)
                    
                    u_delayed = np.roll(u_delayed, 1)
                    u_delayed[0] = u[i]
                    
                    u_to_plant = u_delayed[min(delay_samples, len(u_delayed) - 1)]
                    
                    if i < n_steps - 1:
                        if model.model_type == "FOPDT":
                            dydt = (model.K * u_to_plant - y[i]) / model.tau
                        else:
                            dy1dt = (model.K * u_to_plant - y1) / model.tau
                            dy2dt = (y1 - y2) / model.tau2
                            y1 = y1 + dy1dt * dt
                            y2 = y2 + dy2dt * dt
                            dydt = dy2dt
                        
                        y[i + 1] = y[i] + dydt * dt
                    
                    error_prev = error
                
                error = setpoint - y
                iae = np.sum(np.abs(error)) * dt # Integral Absolute Error
                ise = np.sum(error ** 2) * dt # Integral Square Error
                
                if abs(setpoint) > 1e-9:
                    overshoot = max(0.0, np.max(y) - setpoint) / abs(setpoint) * 100.0
                else:
                    overshoot = max(0.0, np.max(y) - setpoint) * 100.0
                
                settling_band = requirements.settling_band if requirements is not None else 0.02
                settling_threshold = settling_band * abs(setpoint)
                within_band = np.abs(y - setpoint) <= settling_threshold
                outside_idx = np.where(~within_band)[0]
                if len(outside_idx) == 0:
                    settling_time = t[0]
                else:
                    last_outside = outside_idx[-1]
                    settling_time = t[min(last_outside + 1, len(t) - 1)]
                
                control_variation = np.sum(np.abs(np.diff(u))) * dt
                
                if np.max(np.abs(y)) > 100 or np.max(np.abs(u)) > 50:
                    return 1e10
                
                requirements_penalty = 0.0
                if requirements is not None:
                    if requirements.max_overshoot_pct is not None:
                        overshoot_violation = max(0.0, overshoot - requirements.max_overshoot_pct)
                        requirements_penalty += (overshoot_violation ** 2) * requirements.overshoot_penalty_weight
                    if requirements.max_settling_time is not None:
                        settling_violation = max(0.0, settling_time - requirements.max_settling_time)
                        requirements_penalty += (settling_violation ** 2) * requirements.settling_penalty_weight
                
                cost = (
                    ise * 100.0 +
                    iae * 10.0 +
                    overshoot * 50.0 +
                    settling_time * 2.0 +
                    control_variation * 0.1 +
                    requirements_penalty
                )
                
                if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                    return 1e10
                
                return cost
                
            except Exception as e:
                return 1e10
        
        return cost_function

    def _prompt_requirements(self) -> PerformanceRequirements:
        """Prompt user for performance requirements."""
        print("\nEnter performance requirements (press Enter to skip a field).")
        
        def _parse_optional_float(prompt: str) -> Optional[float]:
            value = input(prompt).strip()
            if not value:
                return None
            try:
                return float(value)
            except ValueError:
                print("  Invalid number, ignoring.")
                return None
        
        max_overshoot = _parse_optional_float("Max overshoot (%): ")
        max_settling = _parse_optional_float("Max settling time (s): ")
        settling_band = _parse_optional_float("Settling band (% of setpoint, default 2): ")
        
        return PerformanceRequirements(
            max_overshoot_pct=max_overshoot,
            max_settling_time=max_settling,
            settling_band=(settling_band / 100.0) if settling_band is not None else 0.02
        )
