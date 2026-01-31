"""
System identification from experimental data.

Estimates transfer function models from input-output data using optimization:
- First Order Plus Dead Time (FOPDT)
- Second Order Plus Dead Time (SOPDT)
- Automatic selection of best fit
"""

from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np
from scipy import signal, optimize
from enum import Enum

from pid_control.identification.csv_reader import ExperimentalData


class ModelType(Enum):
    """Types of transfer function models."""
    FOPDT = "First Order Plus Dead Time"
    SOPDT = "Second Order Plus Dead Time"
    AUTO = "Automatic Selection (Best Fit)"


@dataclass
class TransferFunctionModel:
    """
    Transfer function model parameters.
    
    FOPDT: G(s) = K * exp(-theta*s) / (tau*s + 1)
    SOPDT: G(s) = K * exp(-theta*s) / (tau1*s + 1)(tau2*s + 1)
    """
    K: float
    tau: float
    theta: float
    tau2: Optional[float] = None
    zeta: Optional[float] = None
    model_type: str = "FOPDT"
    fit_quality: float = 0.0
    
    def __str__(self):
        if self.model_type == "FOPDT":
            return f"FOPDT: K={self.K:.3f}, tau={self.tau:.3f}, theta={self.theta:.3f}"
        elif self.model_type == "SOPDT":
            return f"SOPDT: K={self.K:.3f}, tau1={self.tau:.3f}, tau2={self.tau2:.3f}, theta={self.theta:.3f}"
        return f"Model: K={self.K:.3f}, tau={self.tau:.3f}, theta={self.theta:.3f}"


@dataclass
class IdentificationResult:
    """Result of system identification."""
    model: TransferFunctionModel
    recommended_gains: Dict[str, float]
    fit_quality: float
    simulated_output: np.ndarray
    time: np.ndarray
    method: str
    tuning_rule: str
    
    def summary(self) -> str:
        """Get summary string."""
        lines = [
            "=" * 70,
            "SYSTEM IDENTIFICATION RESULTS",
            "=" * 70,
            f"Model Type: {self.model.model_type}",
            f"Identification Method: {self.method}",
            f"Fit Quality (R²): {self.fit_quality:.4f}",
            "",
            "Transfer Function Parameters:",
            f"  Gain (K): {self.model.K:.4f}",
            f"  Time Constant (tau): {self.model.tau:.4f} s",
            f"  Dead Time (theta): {self.model.theta:.4f} s",
        ]
        
        if self.model.tau2 is not None:
            lines.append(f"  Second Time Constant (tau2): {self.model.tau2:.4f} s")
        
        lines.extend([
            "",
            f"Recommended PID Gains ({self.tuning_rule}):",
            f"  Kp = {self.recommended_gains['kp']:.4f}",
            f"  Ki = {self.recommended_gains['ki']:.4f}",
            f"  Kd = {self.recommended_gains['kd']:.4f}",
            "=" * 70
        ])
        
        return "\n".join(lines)


class SystemIdentifier:
    """
    Identify system transfer function from experimental data.
    
    Supports multiple identification methods and automatic tuning rule application.
    """
    
    def __init__(self, data: ExperimentalData):
        """
        Initialize system identifier.
        
        Args:
            data: Experimental data from CSV
        """
        self.data = data
        self._validate_data()
    
    def _validate_data(self):
        """Validate input data."""
        if len(self.data.time) < 20:
            raise ValueError("Need at least 20 data points for reliable identification")
        
        if np.std(self.data.input) < 1e-6:
            raise ValueError("Input signal has no variation - cannot identify system")
        
        if np.std(self.data.output) < 1e-6:
            raise ValueError("Output signal has no variation - cannot identify system")
    
    def identify(
        self,
        model_type: ModelType = ModelType.AUTO,
        tuning_rule: str = 'ziegler_nichols'
    ) -> IdentificationResult:
        """
        Identify system and recommend PID gains using optimization.
        
        Args:
            model_type: Type of transfer function model (FOPDT, SOPDT, or AUTO for best fit)
            tuning_rule: PID tuning rule to apply
        
        Returns:
            IdentificationResult with model and recommended gains
        """
        if model_type == ModelType.AUTO:
            fopdt_model = self._optimize_model(ModelType.FOPDT)
            sopdt_model = self._optimize_model(ModelType.SOPDT)
            
            fopdt_sim = self._simulate_model(fopdt_model)
            sopdt_sim = self._simulate_model(sopdt_model)
            
            fopdt_fit = self._calculate_fit_quality(self.data.output, fopdt_sim)
            sopdt_fit = self._calculate_fit_quality(self.data.output, sopdt_sim)
            
            if sopdt_fit > fopdt_fit:
                model = sopdt_model
                simulated_output = sopdt_sim
                fit_quality = sopdt_fit
            else:
                model = fopdt_model
                simulated_output = fopdt_sim
                fit_quality = fopdt_fit
        else:
            model = self._optimize_model(model_type)
            simulated_output = self._simulate_model(model)
            fit_quality = self._calculate_fit_quality(self.data.output, simulated_output)
        
        model.fit_quality = fit_quality
        recommended_gains = self._apply_tuning_rule(model, tuning_rule)
        
        return IdentificationResult(
            model=model,
            recommended_gains=recommended_gains,
            fit_quality=fit_quality,
            simulated_output=simulated_output,
            time=self.data.time,
            method='optimization',
            tuning_rule=tuning_rule
        )
    
    def _get_initial_guess(self, model_type: ModelType) -> TransferFunctionModel:
        """Get initial parameter guess using analytical step response analysis."""
        two_point_model = self._analytical_step_response_method()
        tangent_model = self._tangent_method()
        
        analytical_model = None
        if two_point_model is not None and tangent_model is not None:
            sim_two_point = self._simulate_model(two_point_model)
            sim_tangent = self._simulate_model(tangent_model)
            
            fit_two_point = self._calculate_fit_quality(self.data.output, sim_two_point)
            fit_tangent = self._calculate_fit_quality(self.data.output, sim_tangent)
            
            analytical_model = two_point_model if fit_two_point >= fit_tangent else tangent_model
        elif two_point_model is not None:
            analytical_model = two_point_model
        elif tangent_model is not None:
            analytical_model = tangent_model
        
        if analytical_model is not None:
            if model_type == ModelType.SOPDT:
                return TransferFunctionModel(
                    K=analytical_model.K,
                    tau=analytical_model.tau * 0.7,
                    theta=analytical_model.theta,
                    tau2=analytical_model.tau * 0.3,
                    model_type="SOPDT"
                )
            return analytical_model
        
        t = self.data.time
        u = self.data.input
        y = self.data.output
        
        delta_u = np.max(u) - np.min(u)
        delta_y = np.max(y) - np.min(y)
        
        if abs(delta_u) < 1e-6:
            K = 1.0
        else:
            K = delta_y / delta_u
        
        time_span = t[-1] - t[0]
        tau = time_span / 3.0
        theta = time_span * 0.1
        
        if model_type == ModelType.SOPDT:
            tau2 = tau * 0.3
            return TransferFunctionModel(
                K=K, tau=tau, theta=theta, tau2=tau2,
                model_type="SOPDT"
            )
        
        return TransferFunctionModel(K=K, tau=tau, theta=theta, model_type="FOPDT")
    
    def _analytical_step_response_method(self) -> Optional[TransferFunctionModel]:
        """
        Estimate FOPDT parameters using analytical step response analysis.
        
        Uses the two-point method (28.3% and 63.2% of steady-state response).
        This provides much better initial estimates than pure heuristics.
        """
        t = self.data.time
        u = self.data.input
        y = self.data.output
        
        step_start_idx = self._find_step_start()
        if step_start_idx is None:
            return None
        
        y0 = np.mean(y[max(0, step_start_idx-10):step_start_idx+1])
        u0 = np.mean(u[max(0, step_start_idx-10):step_start_idx+1])
        
        y_final = np.mean(y[-20:])
        u_final = np.mean(u[-20:])
        
        delta_u = u_final - u0
        delta_y = y_final - y0
        
        if abs(delta_u) < 1e-6:
            return None
        
        K = delta_y / delta_u
        
        y_target_283 = y0 + 0.283 * delta_y
        y_target_632 = y0 + 0.632 * delta_y
        
        t1_idx = None
        t2_idx = None
        
        for i in range(step_start_idx, len(y)):
            if t1_idx is None and y[i] >= y_target_283:
                t1_idx = i
            if t2_idx is None and y[i] >= y_target_632:
                t2_idx = i
                break
        
        if t1_idx is None or t2_idx is None:
            return None
        
        t1 = t[t1_idx] - t[step_start_idx]
        t2 = t[t2_idx] - t[step_start_idx]
        
        tau = 1.5 * (t2 - t1)
        theta = t2 - tau
        
        theta = max(0.0, theta)
        tau = max(0.01, tau)
        
        return TransferFunctionModel(K=K, tau=tau, theta=theta, model_type="FOPDT")
    
    def _tangent_method(self) -> Optional[TransferFunctionModel]:
        """
        Estimate FOPDT parameters using tangent method.
        
        Finds the point of maximum slope and draws a tangent line.
        """
        t = self.data.time
        y = self.data.output
        
        step_start_idx = self._find_step_start()
        if step_start_idx is None:
            return None
        
        y0 = np.mean(y[max(0, step_start_idx-10):step_start_idx+1])
        y_final = np.mean(y[-20:])
        delta_y = y_final - y0
        
        if abs(delta_y) < 1e-6:
            return None
        
        u0 = np.mean(self.data.input[max(0, step_start_idx-10):step_start_idx+1])
        u_final = np.mean(self.data.input[-20:])
        delta_u = u_final - u0
        
        if abs(delta_u) < 1e-6:
            return None
        
        K = delta_y / delta_u
        
        dy_dt = np.gradient(y, t)
        
        search_start = step_start_idx
        search_end = min(step_start_idx + int(len(y) * 0.5), len(y) - 1)
        
        max_slope_idx = search_start + np.argmax(dy_dt[search_start:search_end])
        max_slope = dy_dt[max_slope_idx]
        
        if max_slope < 1e-6:
            return None
        
        y_at_max_slope = y[max_slope_idx]
        t_at_max_slope = t[max_slope_idx]
        
        t_intercept_y0 = t_at_max_slope - (y_at_max_slope - y0) / max_slope
        t_intercept_yf = t_at_max_slope + (y_final - y_at_max_slope) / max_slope
        
        theta = max(0.0, t_intercept_y0 - t[step_start_idx])
        tau = max(0.01, t_intercept_yf - t_intercept_y0)
        
        return TransferFunctionModel(K=K, tau=tau, theta=theta, model_type="FOPDT")
    
    def _find_step_start(self) -> Optional[int]:
        """Find the index where a step change in input occurs."""
        u = self.data.input
        
        u_diff = np.abs(np.diff(u))
        threshold = np.std(u_diff) * 3
        
        step_candidates = np.where(u_diff > threshold)[0]
        
        if len(step_candidates) == 0:
            return 0
        
        return step_candidates[0]
    
    def _optimize_model(self, model_type: ModelType, use_multistart: bool = True) -> TransferFunctionModel:
        """Optimize model parameters to minimize prediction error with multi-start."""
        y = self.data.output
        initial_model = self._get_initial_guess(model_type)
        
        if model_type == ModelType.FOPDT:
            def objective(params):
                K, tau, theta = params
                if tau <= 0 or theta < 0:
                    return 1e10
                
                model = TransferFunctionModel(K=K, tau=tau, theta=theta, model_type="FOPDT")
                y_sim = self._simulate_model(model)
                
                error = self._weighted_error(y, y_sim)
                
                regularization = 0.001 * (theta / max(tau, 0.01)) if tau > 0 else 1e10
                
                return error + regularization
            
            bounds = [
                (initial_model.K * 0.1 if initial_model.K > 0 else -abs(initial_model.K) * 10, 
                 initial_model.K * 10 if initial_model.K > 0 else abs(initial_model.K) * 0.1),
                (initial_model.tau * 0.1, initial_model.tau * 10),
                (0, max(initial_model.theta * 3, 2.0))
            ]
            
            if use_multistart:
                best_result = None
                best_cost = float('inf')
                
                initial_guesses = [
                    [initial_model.K, initial_model.tau, initial_model.theta],
                    [initial_model.K * 0.9, initial_model.tau * 1.2, initial_model.theta * 0.8],
                    [initial_model.K * 1.1, initial_model.tau * 0.8, initial_model.theta * 1.2],
                    [initial_model.K, initial_model.tau * 1.5, initial_model.theta * 0.5],
                    [initial_model.K, initial_model.tau * 0.7, initial_model.theta * 1.5],
                ]
                
                for x0 in initial_guesses:
                    x0 = [max(bounds[i][0], min(bounds[i][1], x0[i])) for i in range(3)]
                    
                    result = optimize.minimize(
                        objective, x0, method='L-BFGS-B', bounds=bounds,
                        options={'maxiter': 2000, 'ftol': 1e-9, 'gtol': 1e-9}
                    )
                    
                    if result.fun < best_cost:
                        best_cost = result.fun
                        best_result = result
                
                result = best_result
            else:
                x0 = [initial_model.K, initial_model.tau, initial_model.theta]
                result = optimize.minimize(
                    objective, x0, method='L-BFGS-B', bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-9, 'gtol': 1e-9}
                )
            
            return TransferFunctionModel(
                K=result.x[0], tau=result.x[1], theta=result.x[2],
                model_type="FOPDT"
            )
        
        else:
            def objective(params):
                K, tau1, tau2, theta = params
                if tau1 <= 0 or tau2 <= 0 or theta < 0:
                    return 1e10
                
                model = TransferFunctionModel(
                    K=K, tau=tau1, theta=theta, tau2=tau2, model_type="SOPDT"
                )
                y_sim = self._simulate_model(model)
                
                error = self._weighted_error(y, y_sim)
                
                regularization = 0.001 * (theta / max(tau1, 0.01))
                
                return error + regularization
            
            bounds = [
                (initial_model.K * 0.1 if initial_model.K > 0 else -abs(initial_model.K) * 10,
                 initial_model.K * 10 if initial_model.K > 0 else abs(initial_model.K) * 0.1),
                (initial_model.tau * 0.1, initial_model.tau * 10),
                (0.01, initial_model.tau * 5),
                (0, max(initial_model.theta * 3, 2.0))
            ]
            
            if use_multistart:
                best_result = None
                best_cost = float('inf')
                
                initial_guesses = [
                    [initial_model.K, initial_model.tau, initial_model.tau2 or initial_model.tau * 0.3, initial_model.theta],
                    [initial_model.K * 0.9, initial_model.tau * 1.2, initial_model.tau * 0.25, initial_model.theta * 0.8],
                    [initial_model.K * 1.1, initial_model.tau * 0.8, initial_model.tau * 0.35, initial_model.theta * 1.2],
                ]
                
                for x0 in initial_guesses:
                    x0 = [max(bounds[i][0], min(bounds[i][1], x0[i])) for i in range(4)]
                    
                    result = optimize.minimize(
                        objective, x0, method='L-BFGS-B', bounds=bounds,
                        options={'maxiter': 2000, 'ftol': 1e-9, 'gtol': 1e-9}
                    )
                    
                    if result.fun < best_cost:
                        best_cost = result.fun
                        best_result = result
                
                result = best_result
            else:
                x0 = [
                    initial_model.K,
                    initial_model.tau,
                    initial_model.tau2 or initial_model.tau * 0.3,
                    initial_model.theta
                ]
                result = optimize.minimize(
                    objective, x0, method='L-BFGS-B', bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-9, 'gtol': 1e-9}
                )
            
            return TransferFunctionModel(
                K=result.x[0], tau=result.x[1], theta=result.x[3],
                tau2=result.x[2], model_type="SOPDT"
            )
    
    def _simulate_model(self, model: TransferFunctionModel) -> np.ndarray:
        """Simulate model response to input signal."""
        t = self.data.time
        u = self.data.input
        dt = self.data.sample_time or 0.01
        
        if model.model_type == "FOPDT":
            num = [model.K]
            den = [model.tau, 1]
        else:
            num = [model.K]
            den = [model.tau * model.tau2, model.tau + model.tau2, 1]
        
        sys = signal.TransferFunction(num, den)
        
        _, y_no_delay = signal.dlsim(
            signal.cont2discrete((sys.num, sys.den), dt, method='zoh'),
            u,
            t=t
        )
        
        y_no_delay = y_no_delay.flatten()
        
        n_target = len(t)
        if len(y_no_delay) < n_target:
            y_no_delay = np.pad(y_no_delay, (0, n_target - len(y_no_delay)), mode='edge')
        elif len(y_no_delay) > n_target:
            y_no_delay = y_no_delay[:n_target]
        
        if model.theta > 0:
            delay_samples = int(model.theta / dt)
            y_sim = np.zeros(n_target)
            if delay_samples < n_target:
                n_copy = min(len(y_no_delay), n_target - delay_samples)
                y_sim[delay_samples:delay_samples + n_copy] = y_no_delay[:n_copy]
        else:
            y_sim = y_no_delay
        
        y_initial = np.mean(self.data.output[:min(10, len(self.data.output))])
        y_sim = y_sim + y_initial
        
        return y_sim
    
    def _weighted_error(self, y_actual: np.ndarray, y_predicted: np.ndarray) -> float:
        """Calculate weighted prediction error emphasizing transient response."""
        error = y_actual - y_predicted
        
        n = len(error)
        weights = np.ones(n)
        
        transient_end = min(int(n * 0.4), n)
        weights[:transient_end] = 2.0
        
        weighted_sse = np.sum(weights * error ** 2)
        
        return weighted_sse
    
    def _calculate_fit_quality(self, y_actual: np.ndarray, y_predicted: np.ndarray) -> float:
        """Calculate R² coefficient of determination."""
        ss_res = np.sum((y_actual - y_predicted) ** 2)
        ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
        
        if ss_tot < 1e-10:
            return 0.0
        
        r_squared = 1 - (ss_res / ss_tot)
        return max(0.0, min(1.0, r_squared))
    
    def _apply_tuning_rule(
        self,
        model: TransferFunctionModel,
        rule: str
    ) -> Dict[str, float]:
        """
        Apply PID tuning rule to transfer function model.
        
        Supported rules:
        - ziegler_nichols: Classic Z-N tuning
        - cohen_coon: Cohen-Coon tuning
        - imc: Internal Model Control tuning
        - lambda_tuning: Lambda tuning (aggressive, moderate, conservative)
        """
        K = model.K
        tau = model.tau
        theta = model.theta
        
        if abs(K) < 1e-6:
            return {'kp': 1.0, 'ki': 0.1, 'kd': 0.1}
        
        if rule == 'ziegler_nichols':
            kp = 1.2 * tau / (K * theta) if theta > 0 else 0.6 * tau / K
            ki = kp / (2 * theta) if theta > 0 else kp / tau
            kd = kp * 0.5 * theta if theta > 0 else kp * tau * 0.125
        
        elif rule == 'cohen_coon':
            if theta > 0 and theta < tau:
                R = theta / tau
                kp = (1.0 / K) * (tau / theta) * (0.9 + R / 12)
                ki = kp * theta / (3.33 - 3 * R) * (1 + R / 4)
                kd = kp * theta * 0.5 * (1 - R / 6)
            else:
                kp = 1.2 * tau / (K * max(theta, 0.1))
                ki = kp / (2 * max(theta, 0.1))
                kd = kp * 0.5 * max(theta, 0.1)
        
        elif rule == 'imc':
            lambda_c = max(theta, 0.1 * tau)
            kp = tau / (K * (lambda_c + theta))
            ki = kp / tau
            kd = 0.0
        
        elif rule == 'lambda_tuning':
            lambda_c = max(theta, 0.8 * tau)
            kp = (2 * tau + theta) / (2 * K * lambda_c)
            ki = kp / tau
            kd = kp * tau * theta / (2 * tau + theta)
        
        elif rule == 'aggressive':
            lambda_c = max(theta * 0.5, 0.1 * tau)
            kp = tau / (K * (lambda_c + theta))
            ki = kp / (0.5 * tau)
            kd = kp * 0.5 * tau
        
        elif rule == 'conservative':
            lambda_c = max(theta * 2, tau)
            kp = tau / (K * (lambda_c + theta))
            ki = kp / (2 * tau)
            kd = kp * 0.1 * tau
        
        else:
            raise ValueError(f"Unknown tuning rule: {rule}")
        
        kp = max(0.0, min(abs(kp), 1000.0))
        ki = max(0.0, min(abs(ki), 100.0))
        kd = max(0.0, min(abs(kd), 100.0))
        
        return {'kp': kp, 'ki': ki, 'kd': kd}
    
    def compare_tuning_rules(self) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple tuning rules for the identified system.
        
        Returns:
            Dictionary mapping rule names to PID gains
        """
        model = self._optimize_model(ModelType.FOPDT)
        
        rules = [
            'ziegler_nichols',
            'cohen_coon',
            'imc',
            'lambda_tuning',
            'aggressive',
            'conservative'
        ]
        
        results = {}
        for rule in rules:
            try:
                gains = self._apply_tuning_rule(model, rule)
                results[rule] = gains
            except Exception as e:
                print(f"Warning: Failed to apply {rule}: {e}")
                continue
        
        return results
