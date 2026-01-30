"""
Optimization methods for PID tuning.
Provides various optimization algorithms for real-time parameter adjustment.
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import math
from scipy import optimize


@dataclass
class TuningResult:
    """Result of a tuning optimization."""
    kp: float
    ki: float
    kd: float
    cost: float
    iterations: int
    success: bool
    message: str
    history: List[Dict[str, float]]


class BaseTuner(ABC):
    """Abstract base class for tuning optimizers."""
    
    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        cost_function: Optional[Callable[[float, float, float], float]] = None
    ):
        """
        Initialize tuner.
        
        Args:
            bounds: Dictionary with 'kp', 'ki', 'kd' keys and (min, max) tuple values
            cost_function: Function that takes (kp, ki, kd) and returns cost
        """
        self._bounds = bounds
        self._cost_function = cost_function
        self._history: List[Dict[str, float]] = []
    
    @abstractmethod
    def optimize(
        self,
        initial_params: Dict[str, float],
        max_iterations: int = 100
    ) -> TuningResult:
        """
        Run optimization to find optimal PID parameters.
        
        Args:
            initial_params: Starting parameters {'kp', 'ki', 'kd'}
            max_iterations: Maximum number of iterations
            
        Returns:
            TuningResult with optimal parameters
        """
        pass
    
    def set_cost_function(self, func: Callable[[float, float, float], float]) -> None:
        """Set the cost function to optimize."""
        self._cost_function = func
    
    @property
    def history(self) -> List[Dict[str, float]]:
        """Get optimization history."""
        return self._history


class GradientFreeTuner(BaseTuner):
    """
    Gradient-free optimizer using Nelder-Mead simplex method.
    
    Good for noisy cost functions where gradients are unreliable.
    """
    
    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        cost_function: Optional[Callable[[float, float, float], float]] = None
    ):
        super().__init__(bounds, cost_function)
    
    def optimize(
        self,
        initial_params: Dict[str, float],
        max_iterations: int = 100
    ) -> TuningResult:
        """Run Nelder-Mead optimization."""
        if self._cost_function is None:
            raise ValueError("Cost function not set")
        
        self._history = []
        
        def objective(x):
            kp, ki, kd = x
            # Enforce bounds via penalty
            penalty = 0
            if kp < self._bounds['kp'][0] or kp > self._bounds['kp'][1]:
                penalty += 1e6
            if ki < self._bounds['ki'][0] or ki > self._bounds['ki'][1]:
                penalty += 1e6
            if kd < self._bounds['kd'][0] or kd > self._bounds['kd'][1]:
                penalty += 1e6
            
            cost = self._cost_function(kp, ki, kd) + penalty
            self._history.append({'kp': kp, 'ki': ki, 'kd': kd, 'cost': cost})
            return cost
        
        x0 = [initial_params['kp'], initial_params['ki'], initial_params['kd']]
        
        result = optimize.minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': max_iterations, 'xatol': 1e-4, 'fatol': 1e-4}
        )
        
        return TuningResult(
            kp=result.x[0],
            ki=result.x[1],
            kd=result.x[2],
            cost=result.fun,
            iterations=result.nit,
            success=result.success,
            message=result.message,
            history=self._history
        )


class BayesianTuner(BaseTuner):
    """
    Bayesian optimization tuner using Gaussian Process surrogate.
    
    Efficient for expensive-to-evaluate cost functions.
    Uses Expected Improvement acquisition function.
    """
    
    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        cost_function: Optional[Callable[[float, float, float], float]] = None,
        n_initial: int = 5,
        exploration_weight: float = 0.1
    ):
        """
        Initialize Bayesian tuner.
        
        Args:
            bounds: Parameter bounds
            cost_function: Cost function
            n_initial: Number of initial random samples
            exploration_weight: Trade-off between exploration and exploitation
        """
        super().__init__(bounds, cost_function)
        self._n_initial = n_initial
        self._xi = exploration_weight
    
    def optimize(
        self,
        initial_params: Dict[str, float],
        max_iterations: int = 50
    ) -> TuningResult:
        """Run Bayesian optimization."""
        if self._cost_function is None:
            raise ValueError("Cost function not set")
        
        self._history = []
        
        # Parameter bounds as arrays
        lb = np.array([self._bounds['kp'][0], self._bounds['ki'][0], self._bounds['kd'][0]])
        ub = np.array([self._bounds['kp'][1], self._bounds['ki'][1], self._bounds['kd'][1]])
        
        # Storage for observations
        X_observed = []
        y_observed = []
        
        # Initial random sampling
        for i in range(self._n_initial):
            if i == 0:
                # Use initial params for first sample
                x = np.array([initial_params['kp'], initial_params['ki'], initial_params['kd']])
            else:
                # Random sampling
                x = lb + np.random.random(3) * (ub - lb)
            
            y = self._cost_function(x[0], x[1], x[2])
            X_observed.append(x)
            y_observed.append(y)
            self._history.append({'kp': x[0], 'ki': x[1], 'kd': x[2], 'cost': y})
        
        X_observed = np.array(X_observed)
        y_observed = np.array(y_observed)
        
        # Bayesian optimization loop
        for iteration in range(max_iterations - self._n_initial):
            # Fit Gaussian Process (simplified RBF kernel)
            next_x = self._get_next_sample(X_observed, y_observed, lb, ub)
            
            # Evaluate cost function
            y = self._cost_function(next_x[0], next_x[1], next_x[2])
            
            X_observed = np.vstack([X_observed, next_x])
            y_observed = np.append(y_observed, y)
            self._history.append({'kp': next_x[0], 'ki': next_x[1], 'kd': next_x[2], 'cost': y})
        
        # Return best found parameters
        best_idx = np.argmin(y_observed)
        best_x = X_observed[best_idx]
        
        return TuningResult(
            kp=best_x[0],
            ki=best_x[1],
            kd=best_x[2],
            cost=y_observed[best_idx],
            iterations=max_iterations,
            success=True,
            message="Bayesian optimization completed",
            history=self._history
        )
    
    def _get_next_sample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray
    ) -> np.ndarray:
        """Get next sample point using Expected Improvement."""
        # Simplified acquisition function maximization
        best_y = np.min(y)
        
        # Random search for acquisition function maximum
        n_candidates = 1000
        candidates = lb + np.random.random((n_candidates, 3)) * (ub - lb)
        
        best_ei = -np.inf
        best_candidate = candidates[0]
        
        for candidate in candidates:
            ei = self._expected_improvement(candidate, X, y, best_y)
            if ei > best_ei:
                best_ei = ei
                best_candidate = candidate
        
        return best_candidate
    
    def _expected_improvement(
        self,
        x: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        best_y: float
    ) -> float:
        """Calculate Expected Improvement at point x."""
        # Simplified GP prediction using kernel-weighted average
        length_scale = 1.0
        
        # Calculate kernel distances
        dists = np.sum((X - x)**2, axis=1)
        weights = np.exp(-dists / (2 * length_scale**2))
        weights = weights / (np.sum(weights) + 1e-10)
        
        # Predicted mean and std
        mu = np.sum(weights * y)
        sigma = np.sqrt(np.sum(weights * (y - mu)**2)) + 1e-6
        
        # Expected Improvement
        z = (best_y - mu - self._xi) / sigma
        ei = (best_y - mu - self._xi) * self._norm_cdf(z) + sigma * self._norm_pdf(z)
        
        return ei
    
    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Standard normal CDF."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    @staticmethod
    def _norm_pdf(x: float) -> float:
        """Standard normal PDF."""
        return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)


class GeneticTuner(BaseTuner):
    """
    Genetic algorithm tuner for global optimization.
    
    Good for avoiding local minima in complex cost landscapes.
    """
    
    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        cost_function: Optional[Callable[[float, float, float], float]] = None,
        population_size: int = 20,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8
    ):
        """
        Initialize genetic tuner.
        
        Args:
            bounds: Parameter bounds
            cost_function: Cost function
            population_size: Number of individuals in population
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
        """
        super().__init__(bounds, cost_function)
        self._pop_size = population_size
        self._mutation_rate = mutation_rate
        self._crossover_rate = crossover_rate
    
    def optimize(
        self,
        initial_params: Dict[str, float],
        max_iterations: int = 50
    ) -> TuningResult:
        """Run genetic algorithm optimization."""
        if self._cost_function is None:
            raise ValueError("Cost function not set")
        
        self._history = []
        
        lb = np.array([self._bounds['kp'][0], self._bounds['ki'][0], self._bounds['kd'][0]])
        ub = np.array([self._bounds['kp'][1], self._bounds['ki'][1], self._bounds['kd'][1]])
        
        # Initialize population
        population = lb + np.random.random((self._pop_size, 3)) * (ub - lb)
        population[0] = [initial_params['kp'], initial_params['ki'], initial_params['kd']]
        
        # Evaluate initial population
        fitness = np.array([
            self._cost_function(ind[0], ind[1], ind[2]) for ind in population
        ])
        
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        for ind, fit in zip(population, fitness):
            self._history.append({'kp': ind[0], 'ki': ind[1], 'kd': ind[2], 'cost': fit})
        
        # Evolution loop
        for generation in range(max_iterations):
            # Selection (tournament)
            new_population = []
            for _ in range(self._pop_size):
                # Tournament selection
                idx1, idx2 = np.random.randint(0, self._pop_size, 2)
                winner = population[idx1] if fitness[idx1] < fitness[idx2] else population[idx2]
                new_population.append(winner.copy())
            
            # Crossover
            for i in range(0, self._pop_size - 1, 2):
                if np.random.random() < self._crossover_rate:
                    alpha = np.random.random()
                    child1 = alpha * new_population[i] + (1 - alpha) * new_population[i+1]
                    child2 = (1 - alpha) * new_population[i] + alpha * new_population[i+1]
                    new_population[i] = child1
                    new_population[i+1] = child2
            
            # Mutation
            for i in range(self._pop_size):
                if np.random.random() < self._mutation_rate:
                    mutation_idx = np.random.randint(0, 3)
                    new_population[i][mutation_idx] += np.random.normal(0, 0.1) * (ub[mutation_idx] - lb[mutation_idx])
                    new_population[i] = np.clip(new_population[i], lb, ub)
            
            population = np.array(new_population)
            
            # Evaluate new population
            fitness = np.array([
                self._cost_function(ind[0], ind[1], ind[2]) for ind in population
            ])
            
            # Track best
            gen_best_idx = np.argmin(fitness)
            if fitness[gen_best_idx] < best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_individual = population[gen_best_idx].copy()
            
            # Elitism - keep best individual
            worst_idx = np.argmax(fitness)
            population[worst_idx] = best_individual
            fitness[worst_idx] = best_fitness
            
            self._history.append({
                'kp': best_individual[0], 
                'ki': best_individual[1], 
                'kd': best_individual[2], 
                'cost': best_fitness
            })
        
        return TuningResult(
            kp=best_individual[0],
            ki=best_individual[1],
            kd=best_individual[2],
            cost=best_fitness,
            iterations=max_iterations,
            success=True,
            message="Genetic algorithm completed",
            history=self._history
        )


class DifferentialEvolutionTuner(BaseTuner):
    """
    Differential Evolution optimizer.
    
    Robust global optimizer that works well for PID tuning.
    """
    
    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        cost_function: Optional[Callable[[float, float, float], float]] = None,
        population_size: int = 15,
        mutation_factor: float = 0.8,
        crossover_rate: float = 0.7
    ):
        super().__init__(bounds, cost_function)
        self._pop_size = population_size
        self._F = mutation_factor
        self._CR = crossover_rate
    
    def optimize(
        self,
        initial_params: Dict[str, float],
        max_iterations: int = 100
    ) -> TuningResult:
        """Run differential evolution."""
        if self._cost_function is None:
            raise ValueError("Cost function not set")
        
        self._history = []
        
        bounds_list = [
            self._bounds['kp'],
            self._bounds['ki'],
            self._bounds['kd']
        ]
        
        def objective(x):
            cost = self._cost_function(x[0], x[1], x[2])
            self._history.append({'kp': x[0], 'ki': x[1], 'kd': x[2], 'cost': cost})
            return cost
        
        result = optimize.differential_evolution(
            objective,
            bounds_list,
            maxiter=max_iterations,
            popsize=self._pop_size,
            mutation=self._F,
            recombination=self._CR,
            seed=42,
            x0=[initial_params['kp'], initial_params['ki'], initial_params['kd']]
        )
        
        return TuningResult(
            kp=result.x[0],
            ki=result.x[1],
            kd=result.x[2],
            cost=result.fun,
            iterations=result.nit,
            success=result.success,
            message=result.message,
            history=self._history
        )
