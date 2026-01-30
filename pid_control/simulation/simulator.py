"""
PID Control Simulation Framework.
Provides tools for running simulations and visualizing results.
"""

from typing import Dict, Any, List, Optional, Tuple, Type
from dataclasses import dataclass, field
import numpy as np
import time

from pid_control.core.pid_controller import PIDController
from pid_control.core.pid_params import PIDParams
from pid_control.plants.base_plant import BasePlant
from pid_control.simulation.scenarios import SimulationScenario, ScenarioLibrary
from pid_control.analyzer.pid_analyzer import PIDAnalyzer
from pid_control.analyzer.plots import PIDPlotter


@dataclass
class SimulationResult:
    """Container for simulation results."""
    timestamps: np.ndarray
    setpoints: np.ndarray
    measurements: np.ndarray
    outputs: np.ndarray
    errors: np.ndarray
    p_terms: np.ndarray
    i_terms: np.ndarray
    d_terms: np.ndarray
    disturbances: np.ndarray
    
    # Metadata
    scenario_name: str = ""
    controller_params: Optional[Dict[str, Any]] = None
    plant_info: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary format."""
        return {
            'timestamp': self.timestamps,
            'setpoint': self.setpoints,
            'measurement': self.measurements,
            'output': self.outputs,
            'error': self.errors,
            'p_term': self.p_terms,
            'i_term': self.i_terms,
            'd_term': self.d_terms,
            'disturbance': self.disturbances,
        }


class Simulator:
    """
    PID Control Simulation Engine.
    
    Runs simulations with various scenarios and provides
    comprehensive analysis and visualization.
    
    Example:
        >>> from pid_control.plants import FirstOrderPlant
        >>> plant = FirstOrderPlant(gain=2.0, time_constant=1.0)
        >>> params = PIDParams(kp=1.0, ki=0.5, kd=0.1)
        >>> sim = Simulator(plant, params)
        >>> result = sim.run(ScenarioLibrary.step_response())
        >>> sim.plot_results(result)
    """
    
    def __init__(
        self,
        plant: BasePlant,
        pid_params: Optional[PIDParams] = None,
        csv_log_path: Optional[str] = None
    ):
        """
        Initialize simulator.
        
        Args:
            plant: Plant model to simulate
            pid_params: PID controller parameters
            csv_log_path: Optional path for CSV logging
        """
        self._plant = plant
        self._params = pid_params or PIDParams()
        self._csv_path = csv_log_path
        
        self._controller: Optional[PIDController] = None
        self._results: List[SimulationResult] = []
        self._plotter = PIDPlotter()
    
    def run(
        self,
        scenario: SimulationScenario,
        initial_output: float = 0.0
    ) -> SimulationResult:
        """
        Run a simulation scenario.
        
        Args:
            scenario: Simulation scenario to run
            initial_output: Initial controller output
            
        Returns:
            SimulationResult containing all data
        """
        start_time = time.perf_counter()
        
        # Reset plant and create controller
        self._plant.reset()
        self._plant.set_noise(scenario.measurement_noise_std)
        
        self._controller = PIDController(
            self._params.copy(sample_time=scenario.sample_time),
            csv_path=self._csv_path
        )
        
        # Allocate arrays
        n_steps = int(scenario.duration / scenario.sample_time)
        
        timestamps = np.zeros(n_steps)
        setpoints = np.zeros(n_steps)
        measurements = np.zeros(n_steps)
        outputs = np.zeros(n_steps)
        errors = np.zeros(n_steps)
        p_terms = np.zeros(n_steps)
        i_terms = np.zeros(n_steps)
        d_terms = np.zeros(n_steps)
        disturbances = np.zeros(n_steps)
        
        # Initialize
        measurement = self._plant.output
        output = initial_output
        
        # Run simulation loop
        for i in range(n_steps):
            t = i * scenario.sample_time
            
            # Get setpoint and disturbance
            setpoint = scenario.get_setpoint(t)
            disturbance = scenario.get_disturbance(t)
            
            # Apply disturbance to plant
            self._plant.set_disturbance(disturbance)
            
            # Controller update
            output = self._controller.update(setpoint, measurement, timestamp=t)
            state = self._controller.state
            
            # Plant update
            measurement = self._plant.update(output)
            
            # Store data
            timestamps[i] = t
            setpoints[i] = setpoint
            measurements[i] = measurement
            outputs[i] = output
            errors[i] = state.error
            p_terms[i] = state.p_term
            i_terms[i] = state.i_term
            d_terms[i] = state.d_term
            disturbances[i] = disturbance
        
        # Flush log
        self._controller.flush_log()
        
        execution_time = time.perf_counter() - start_time
        
        result = SimulationResult(
            timestamps=timestamps,
            setpoints=setpoints,
            measurements=measurements,
            outputs=outputs,
            errors=errors,
            p_terms=p_terms,
            i_terms=i_terms,
            d_terms=d_terms,
            disturbances=disturbances,
            scenario_name=scenario.name,
            controller_params=self._params.to_dict(),
            plant_info=self._plant.get_info(),
            execution_time=execution_time
        )
        
        self._results.append(result)
        return result
    
    def run_comparison(
        self,
        scenario: SimulationScenario,
        param_sets: Dict[str, PIDParams]
    ) -> Dict[str, SimulationResult]:
        """
        Run simulation with multiple parameter sets for comparison.
        
        Args:
            scenario: Simulation scenario
            param_sets: Dictionary mapping names to parameter sets
            
        Returns:
            Dictionary mapping names to results
        """
        results = {}
        
        for name, params in param_sets.items():
            # Update parameters
            self._params = params
            
            # Run simulation
            result = self.run(scenario)
            result.scenario_name = f"{scenario.name} - {name}"
            results[name] = result
        
        return results
    
    def run_batch(
        self,
        scenarios: List[SimulationScenario]
    ) -> List[SimulationResult]:
        """
        Run multiple scenarios.
        
        Args:
            scenarios: List of scenarios to run
            
        Returns:
            List of results
        """
        results = []
        for scenario in scenarios:
            result = self.run(scenario)
            results.append(result)
        return results
    
    def plot_results(
        self,
        result: SimulationResult,
        comprehensive: bool = True
    ) -> None:
        """
        Plot simulation results.
        
        Args:
            result: Simulation result to plot
            comprehensive: If True, show comprehensive analysis
        """
        if comprehensive:
            self._plotter.plot_comprehensive(
                result.timestamps,
                result.setpoints,
                result.measurements,
                result.outputs,
                result.p_terms,
                result.i_terms,
                result.d_terms,
                title=result.scenario_name
            )
        else:
            self._plotter.plot_response(
                result.timestamps,
                result.setpoints,
                result.measurements,
                title=result.scenario_name
            )
    
    def plot_comparison(
        self,
        results: Dict[str, SimulationResult],
        title: str = "Controller Comparison"
    ) -> None:
        """
        Plot comparison of multiple results.
        
        Args:
            results: Dictionary of results to compare
            title: Plot title
        """
        data = {
            name: {
                'timestamps': r.timestamps,
                'setpoints': r.setpoints,
                'measurements': r.measurements,
                'outputs': r.outputs,
            }
            for name, r in results.items()
        }
        
        self._plotter.plot_comparison(data, title=title)
    
    def analyze(self, result: SimulationResult) -> Dict[str, Any]:
        """
        Analyze simulation result.
        
        Args:
            result: Simulation result to analyze
            
        Returns:
            Analysis metrics dictionary
        """
        analyzer = PIDAnalyzer(data=result.to_dict())
        return analyzer.analyze(
            output_limits=(self._params.output_min, self._params.output_max)
            if self._params.output_min is not None else None
        )
    
    def set_params(self, params: PIDParams) -> None:
        """Update PID parameters."""
        self._params = params
    
    def set_plant(self, plant: BasePlant) -> None:
        """Update plant model."""
        self._plant = plant
    
    @property
    def results(self) -> List[SimulationResult]:
        """Get all simulation results."""
        return self._results
    
    @property
    def last_result(self) -> Optional[SimulationResult]:
        """Get most recent result."""
        return self._results[-1] if self._results else None
    
    def clear_results(self) -> None:
        """Clear stored results."""
        self._results.clear()
    
    @staticmethod
    def show():
        """Display all plots."""
        PIDPlotter.show()


class AnimatedSimulator:
    """
    Real-time animated simulation for visual demonstration.
    
    Shows live updating plots during simulation.
    """
    
    def __init__(
        self,
        plant: BasePlant,
        pid_params: PIDParams,
        update_interval: int = 10
    ):
        """
        Initialize animated simulator.
        
        Args:
            plant: Plant model
            pid_params: PID parameters
            update_interval: Update plot every N steps
        """
        self._plant = plant
        self._params = pid_params
        self._update_interval = update_interval
    
    def run_animated(
        self,
        scenario: SimulationScenario,
        figsize: Tuple[int, int] = (12, 8)
    ) -> SimulationResult:
        """
        Run simulation with live animation.
        
        Args:
            scenario: Scenario to run
            figsize: Figure size
            
        Returns:
            Final simulation result
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        
        # Setup
        self._plant.reset()
        self._plant.set_noise(scenario.measurement_noise_std)
        
        controller = PIDController(
            self._params.copy(sample_time=scenario.sample_time)
        )
        
        n_steps = int(scenario.duration / scenario.sample_time)
        
        # Data storage
        data = {
            't': [], 'sp': [], 'meas': [], 'out': [],
            'p': [], 'i': [], 'd': [], 'err': []
        }
        
        # Setup figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f"Live Simulation: {scenario.name}", fontsize=14)
        
        # Initialize lines
        lines = {}
        
        # Response plot
        lines['sp'], = axes[0, 0].plot([], [], 'g--', label='Setpoint')
        lines['meas'], = axes[0, 0].plot([], [], 'b-', label='Measurement')
        axes[0, 0].set_xlim(0, scenario.duration)
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].set_title('Response')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Error plot
        lines['err'], = axes[0, 1].plot([], [], 'r-')
        axes[0, 1].axhline(y=0, color='gray', linestyle=':')
        axes[0, 1].set_xlim(0, scenario.duration)
        axes[0, 1].set_ylabel('Error')
        axes[0, 1].set_title('Tracking Error')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Control output plot
        lines['out'], = axes[1, 0].plot([], [], 'm-')
        axes[1, 0].set_xlim(0, scenario.duration)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Control Output')
        axes[1, 0].set_title('Control Signal')
        axes[1, 0].grid(True, alpha=0.3)
        
        # PID components
        lines['p'], = axes[1, 1].plot([], [], 'orange', label='P')
        lines['i'], = axes[1, 1].plot([], [], 'cyan', label='I')
        lines['d'], = axes[1, 1].plot([], [], 'brown', label='D')
        axes[1, 1].axhline(y=0, color='gray', linestyle=':')
        axes[1, 1].set_xlim(0, scenario.duration)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Component')
        axes[1, 1].set_title('PID Components')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        measurement = self._plant.output
        
        def init():
            for line in lines.values():
                line.set_data([], [])
            return lines.values()
        
        def animate(frame):
            nonlocal measurement
            
            # Run multiple steps per frame
            for _ in range(self._update_interval):
                if len(data['t']) >= n_steps:
                    return lines.values()
                
                i = len(data['t'])
                t = i * scenario.sample_time
                
                setpoint = scenario.get_setpoint(t)
                disturbance = scenario.get_disturbance(t)
                self._plant.set_disturbance(disturbance)
                
                output = controller.update(setpoint, measurement, timestamp=t)
                state = controller.state
                measurement = self._plant.update(output)
                
                data['t'].append(t)
                data['sp'].append(setpoint)
                data['meas'].append(measurement)
                data['out'].append(output)
                data['err'].append(state.error)
                data['p'].append(state.p_term)
                data['i'].append(state.i_term)
                data['d'].append(state.d_term)
            
            # Update lines
            lines['sp'].set_data(data['t'], data['sp'])
            lines['meas'].set_data(data['t'], data['meas'])
            lines['err'].set_data(data['t'], data['err'])
            lines['out'].set_data(data['t'], data['out'])
            lines['p'].set_data(data['t'], data['p'])
            lines['i'].set_data(data['t'], data['i'])
            lines['d'].set_data(data['t'], data['d'])
            
            # Auto-scale y-axes
            for ax in axes.flat:
                ax.relim()
                ax.autoscale_view()
            
            return lines.values()
        
        n_frames = n_steps // self._update_interval + 1
        anim = FuncAnimation(
            fig, animate, init_func=init,
            frames=n_frames, interval=20, blit=False
        )
        
        plt.tight_layout()
        plt.show()
        
        # Return final result
        return SimulationResult(
            timestamps=np.array(data['t']),
            setpoints=np.array(data['sp']),
            measurements=np.array(data['meas']),
            outputs=np.array(data['out']),
            errors=np.array(data['err']),
            p_terms=np.array(data['p']),
            i_terms=np.array(data['i']),
            d_terms=np.array(data['d']),
            disturbances=np.zeros(len(data['t'])),
            scenario_name=scenario.name
        )
