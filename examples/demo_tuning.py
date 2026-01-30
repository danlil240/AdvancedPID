#!/usr/bin/env python3
"""
PID Tuning Demo

Demonstrates:
- Auto-tuning with different optimization methods
- Comparing tuned vs untuned controllers
- Ziegler-Nichols tuning
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pid_control.core.pid_controller import PIDController
from pid_control.core.pid_params import PIDParams
from pid_control.plants.first_order import FirstOrderPlant
from pid_control.plants.delay_plant import FOPDTPlant
from pid_control.tuner.realtime_tuner import RealtimeTuner, CostWeights
from pid_control.simulation.simulator import Simulator
from pid_control.simulation.scenarios import ScenarioLibrary


def main():
    print("=" * 60)
    print("PID Auto-Tuning Demo")
    print("=" * 60)
    
    # Create a challenging FOPDT plant (common in process control)
    plant = FOPDTPlant(
        gain=1.5,
        time_constant=3.0,
        dead_time=1.0,
        sample_time=0.01
    )
    
    print(f"\nPlant: {plant.get_info()}")
    
    # Get tuning suggestions based on plant model
    suggestions = plant.get_tuning_suggestions()
    print("\nPlant-based tuning suggestions:")
    for method, params in suggestions.items():
        print(f"  {method}: Kp={params['kp']:.2f}, Ki={params['ki']:.2f}, Kd={params['kd']:.2f}")
    
    # Start with a poor initial guess
    initial_params = PIDParams(
        kp=0.5,
        ki=0.1,
        kd=0.0,
        sample_time=0.01,
        output_min=-50,
        output_max=50
    )
    
    print(f"\nInitial parameters: Kp={initial_params.kp}, Ki={initial_params.ki}, Kd={initial_params.kd}")
    
    # Create controller and tuner
    controller = PIDController(initial_params)
    
    tuner = RealtimeTuner(
        controller,
        plant,
        optimizer='differential_evolution',
        bounds={
            'kp': (0.1, 10.0),
            'ki': (0.01, 5.0),
            'kd': (0.0, 3.0)
        },
        cost_weights=CostWeights(
            iae=1.0,
            overshoot=2.0,
            settling=1.0,
            control_effort=0.1
        )
    )
    
    # Run auto-tuning
    print("\n" + "-" * 40)
    print("Running auto-tuning (this may take a moment)...")
    print("-" * 40)
    
    result = tuner.auto_tune(
        setpoint=100.0,
        duration=30.0,
        max_iterations=30,
        apply_result=True
    )
    
    print(f"\nTuning completed!")
    print(f"  Success: {result.success}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Final cost: {result.cost:.4f}")
    print(f"\nOptimized parameters:")
    print(f"  Kp = {result.kp:.4f}")
    print(f"  Ki = {result.ki:.4f}")
    print(f"  Kd = {result.kd:.4f}")
    
    # Compare initial vs tuned parameters
    print("\n" + "=" * 60)
    print("Comparing Initial vs Tuned Controllers")
    print("=" * 60)
    
    tuned_params = initial_params.copy(
        kp=result.kp,
        ki=result.ki,
        kd=result.kd
    )
    
    param_sets = {
        'Initial (Poor)': initial_params,
        'Auto-Tuned': tuned_params,
        'Ziegler-Nichols': initial_params.copy(**suggestions['ziegler_nichols']),
        'IMC Conservative': initial_params.copy(**suggestions['imc_conservative']),
    }
    
    # Run comparison
    sim = Simulator(plant, initial_params)
    scenario = ScenarioLibrary.step_with_disturbance(
        setpoint=100.0,
        disturbance=20.0,
        duration=50.0
    )
    
    comparison_results = sim.run_comparison(scenario, param_sets)
    
    # Print comparison metrics
    print("\nPerformance Comparison:")
    print("-" * 70)
    print(f"{'Controller':<20} {'Rise Time':>10} {'Settling':>10} {'Overshoot':>10} {'IAE':>10}")
    print("-" * 70)
    
    for name, res in comparison_results.items():
        metrics = sim.analyze(res)
        sr = metrics['step_response']
        err = metrics['error']
        print(f"{name:<20} {sr['rise_time']:>10.2f}s {sr['settling_time_2pct']:>10.2f}s {sr['overshoot_percent']:>9.1f}% {err['iae']:>10.1f}")
    
    # Plot comparison
    print("\nGenerating comparison plots...")
    sim.plot_comparison(comparison_results, title="Controller Comparison: Initial vs Tuned")
    
    Simulator.show()


if __name__ == "__main__":
    main()
