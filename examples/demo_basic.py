#!/usr/bin/env python3
"""
Basic PID Controller Demo

Demonstrates:
- Basic PID controller setup
- Simple step response simulation
- CSV logging
- Basic plotting
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from pid_control.core.pid_controller import PIDController
from pid_control.core.pid_params import PIDParams, AntiWindupMethod
from pid_control.envs import FirstOrderEnv
from pid_control.simulation.simulator import Simulator
from pid_control.simulation.scenarios import ScenarioLibrary
from pid_control.analyzer.pid_analyzer import PIDAnalyzer


def main():
    print("=" * 60)
    print("Basic PID Controller Demo")
    print("=" * 60)
    
    # Create a first-order plant via Gymnasium environment
    # This represents a simple thermal system or tank level
    env = FirstOrderEnv(
        gain=2.0,          # Output changes by 2x input at steady state
        time_constant=5.0, # 5 second time constant
        sample_time=0.01
    )
    plant = env.plant
    
    print(f"\nPlant (via Gymnasium env): {plant.get_info()}")

    # --- Gymnasium live-render preview ---
    render_env = FirstOrderEnv(
        gain=2.0, time_constant=5.0, sample_time=0.01,
        setpoint=50.0, max_steps=3000, render_mode="human",
    )
    preview_ctrl = PIDController(PIDParams(
        kp=1.5, ki=0.3, kd=0.5, sample_time=0.01,
        output_min=-100, output_max=100,
    ))
    obs, _ = render_env.reset()
    terminated, truncated = False, False
    while not (terminated or truncated):
        ctrl = preview_ctrl.update(50.0, obs[0])
        obs, _, terminated, truncated, _ = render_env.step(np.array([ctrl]))
    render_env.close()

    # Configure PID controller
    params = PIDParams(
        kp=1.5,            # Proportional gain
        ki=0.3,            # Integral gain
        kd=0.5,            # Derivative gain
        sample_time=0.01,
        output_min=-100,   # Output saturation limits
        output_max=100,
        anti_windup=AntiWindupMethod.BACK_CALCULATION,
        derivative_filter_coeff=10.0
    )
    
    print(f"Controller: {params}")
    
    # Create simulator
    sim = Simulator(plant, params, csv_log_path="output/basic_demo.csv")
    
    # Run step response scenario
    scenario = ScenarioLibrary.step_response(
        setpoint=50.0,
        duration=30.0
    )
    
    print(f"\nRunning scenario: {scenario.name}")
    result = sim.run(scenario)
    
    print(f"Simulation completed in {result.execution_time:.3f}s")
    print(f"Final measurement: {result.measurements[-1]:.2f}")
    print(f"Final error: {result.errors[-1]:.4f}")
    
    # Analyze results
    print("\n" + "=" * 60)
    print("Analysis Results")
    print("=" * 60)
    
    metrics = sim.analyze(result)
    
    step_metrics = metrics['step_response']
    print(f"\nStep Response Metrics:")
    print(f"  Rise Time: {step_metrics['rise_time']:.3f}s")
    print(f"  Settling Time (2%): {step_metrics['settling_time_2pct']:.3f}s")
    print(f"  Overshoot: {step_metrics['overshoot_percent']:.1f}%")
    print(f"  Steady-State Error: {step_metrics['steady_state_error']:.4f}")
    
    error_metrics = metrics['error']
    print(f"\nError Metrics:")
    print(f"  IAE: {error_metrics['iae']:.2f}")
    print(f"  ISE: {error_metrics['ise']:.2f}")
    print(f"  RMSE: {error_metrics['rmse']:.4f}")
    
    # Plot results
    print("\nGenerating plots...")
    sim.plot_results(result, comprehensive=True)
    
    print("\nClose plot window to exit.")
    Simulator.show()


if __name__ == "__main__":
    # Create output directory
    Path("output").mkdir(exist_ok=True)
    main()
