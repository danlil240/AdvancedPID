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
        cost_function: Optional[Callable] = None
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
        
        if cost_function is None:
            cost_function = self._create_default_cost_function(id_result.model)
        
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
        model: TransferFunctionModel
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
        else:
            num = [model.K]
            den = [model.tau * model.tau2, model.tau + model.tau2, 1]
        
        plant_sys = signal.TransferFunction(num, den)
        plant_discrete = signal.cont2discrete((plant_sys.num, plant_sys.den), dt, method='zoh')
        
        def cost_function(kp: float, ki: float, kd: float) -> float:
            """Evaluate PID performance on identified model."""
            if kp < 0 or ki < 0 or kd < 0:
                return 1e10
            
            try:
                pid_num = [kd, kp, ki]
                pid_den = [0, 1, 0]
                pid_sys = signal.TransferFunction(pid_num, pid_den)
                pid_discrete = signal.cont2discrete((pid_sys.num, pid_sys.den), dt, method='tustin')
                
                closed_loop = signal.TransferFunction(
                    np.convolve(plant_discrete[0].flatten(), pid_discrete[0].flatten()),
                    np.convolve(plant_discrete[1].flatten(), pid_discrete[1].flatten()) +
                    np.convolve(plant_discrete[0].flatten(), pid_discrete[0].flatten())
                )
                
                setpoint = np.ones(len(t))
                _, y_response = signal.dlsim(
                    (closed_loop.num, closed_loop.den),
                    setpoint,
                    t=t
                )
                
                y_response = y_response.flatten()
                
                error = setpoint - y_response
                iae = np.sum(np.abs(error)) * dt
                
                overshoot = max(0, np.max(y_response) - 1.0)
                
                settling_threshold = 0.02
                settled_idx = len(y_response) - 1
                for i in range(len(y_response) - 1, -1, -1):
                    if abs(y_response[i] - 1.0) > settling_threshold:
                        settled_idx = i
                        break
                settling_time = t[settled_idx] if settled_idx < len(t) else t[-1]
                
                control_effort = kp + ki + kd
                
                cost = (
                    iae * 10.0 +
                    overshoot * 100.0 +
                    settling_time * 5.0 +
                    control_effort * 0.01
                )
                
                if np.any(np.isnan(y_response)) or np.any(np.isinf(y_response)):
                    return 1e10
                
                return cost
                
            except Exception:
                return 1e10
        
        return cost_function
