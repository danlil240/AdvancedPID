#!/usr/bin/env python3
"""
Mass-Spring-Damper System PID Control Demo

Demonstrates PID control of a classic mechanical system:
- Second-order mass-spring-damper dynamics
- Position control with different damping ratios
- Comparison of tuning methods
- Response to various setpoint profiles
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from pid_control.core.pid_controller import PIDController
from pid_control.core.pid_params import PIDParams, AntiWindupMethod
from pid_control.plants.second_order import SecondOrderPlant
from pid_control.simulation.simulator import Simulator
from pid_control.simulation.scenarios import SimulationScenario, SetpointType
from pid_control.identification import SystemIdentifier, CSVDataReader, ModelType
from pid_control.identification.autotune_from_data import AutotuneFromData


def demo_open_loop():
    """
    Demonstrate open-loop response of mass-spring-damper system.
    Shows natural system behavior without control.
    """
    print("\n" + "=" * 70)
    print("DEMO 1: OPEN-LOOP RESPONSE")
    print("=" * 70)
    print("\nMass-spring-damper equation: m*x'' + c*x' + k*x = F")
    print("Where: m = mass, c = damping coefficient, k = spring constant")
    
    # Create mass-spring-damper with different damping ratios
    damping_configs = {
        'Underdamped (ζ=0.3)': 0.3,
        'Critically Damped (ζ=1.0)': 1.0,
        'Overdamped (ζ=2.0)': 2.0,
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Open-Loop Mass-Spring-Damper Response', fontsize=14, fontweight='bold')
    
    colors = {'Underdamped (ζ=0.3)': 'blue', 'Critically Damped (ζ=1.0)': 'green', 'Overdamped (ζ=2.0)': 'red'}
    
    for name, zeta in damping_configs.items():
        print(f"\nTesting {name}...")
        
        plant = SecondOrderPlant(
            gain=1.0,
            natural_frequency=2.0,
            damping_ratio=zeta,
            sample_time=0.01
        )
        
        # Simulate step input
        duration = 5.0
        dt = 0.01
        n_steps = int(duration / dt)
        
        times = np.zeros(n_steps)
        inputs = np.zeros(n_steps)
        outputs = np.zeros(n_steps)
        
        # Step input at t=0.5s
        for i in range(n_steps):
            t = i * dt
            times[i] = t
            inputs[i] = 10.0 if t >= 0.5 else 0.0
            outputs[i] = plant.update(inputs[i])
        
        char_times = plant.get_characteristic_times()
        print(f"  Rise time: {char_times.get('rise_time', 0):.2f}s")
        print(f"  Settling time: {char_times.get('settling_time', 0):.2f}s")
        print(f"  Overshoot: {char_times.get('overshoot_percent', 0):.1f}%")
        
        color = colors[name]
        
        # Response plot
        axes[0, 0].plot(times, outputs, label=name, linewidth=1.5, color=color)
        
        # Input plot
        axes[0, 1].plot(times, inputs, label=name, linewidth=1.5, color=color, alpha=0.7)
        
        # Phase portrait
        velocity = np.gradient(outputs, times)
        axes[1, 0].plot(outputs, velocity, label=name, linewidth=1.5, color=color, alpha=0.7)
        
        # Frequency response (approximate)
        axes[1, 1].plot(times, outputs, label=name, linewidth=1.5, color=color)
    
    axes[0, 0].set_ylabel('Position', fontsize=11)
    axes[0, 0].set_xlabel('Time (s)', fontsize=11)
    axes[0, 0].set_title('Step Response')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_ylabel('Input Force', fontsize=11)
    axes[0, 1].set_xlabel('Time (s)', fontsize=11)
    axes[0, 1].set_title('Input Signal')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    axes[1, 0].axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    axes[1, 0].set_xlabel('Position', fontsize=11)
    axes[1, 0].set_ylabel('Velocity', fontsize=11)
    axes[1, 0].set_title('Phase Portrait')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Time (s)', fontsize=11)
    axes[1, 1].set_ylabel('Position', fontsize=11)
    axes[1, 1].set_title('Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()


def demo_closed_loop_simple():
    """
    Demonstrate simple PID control of mass-spring-damper.
    """
    print("\n" + "=" * 70)
    print("DEMO 2: CLOSED-LOOP CONTROL - SIMPLE PID")
    print("=" * 70)
    
    plant = SecondOrderPlant(
        gain=1.0,
        natural_frequency=2.0,
        damping_ratio=0.3,
        sample_time=0.01
    )
    
    print("\nSystem: Underdamped mass-spring-damper")
    print("  Natural frequency: 2.0 rad/s")
    print("  Damping ratio: 0.3")
    
    # Simple PID tuning
    params = PIDParams(
        kp=10.0,
        ki=2.0,
        kd=3.0,
        sample_time=0.01
    )
    
    print(f"\nPID Parameters:")
    print(f"  Kp = {params.kp}")
    print(f"  Ki = {params.ki}")
    print(f"  Kd = {params.kd}")
    
    scenario = SimulationScenario(
        name="Simple PID Control",
        duration=10.0,
        sample_time=0.01,
        setpoint_type=SetpointType.STEP,
        setpoint_initial=0.0,
        setpoint_final=100.0,
        setpoint_time=1.0
    )
    
    controller = PIDController(params)
    sim = Simulator(plant, controller)
    result = sim.run(scenario)
    
    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle('Simple PID Control - Mass-Spring-Damper', fontsize=14, fontweight='bold')
    
    axes[0].plot(result.timestamps, result.setpoints, 'g--', linewidth=2, label='Setpoint', alpha=0.7)
    axes[0].plot(result.timestamps, result.measurements, 'b-', linewidth=1.5, label='Position')
    axes[0].set_ylabel('Position', fontsize=11)
    axes[0].set_title('System Response')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(result.timestamps, result.errors, 'r-', linewidth=1.5)
    axes[1].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    axes[1].set_ylabel('Error', fontsize=11)
    axes[1].set_title('Tracking Error')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(result.timestamps, result.outputs, 'm-', linewidth=1.5)
    axes[2].set_xlabel('Time (s)', fontsize=11)
    axes[2].set_ylabel('Control Force', fontsize=11)
    axes[2].set_title('Control Signal')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Performance metrics
    overshoot = np.max(result.measurements) - 100.0
    settling_idx = np.where(result.timestamps > 5.0)[0]
    if len(settling_idx) > 0:
        steady_state_error = np.mean(np.abs(result.errors[settling_idx[0]:]))
        print(f"\nPerformance:")
        print(f"  Overshoot: {overshoot:.1f}")
        print(f"  Steady-state error: {steady_state_error:.2f}")


def demo_closed_loop_advanced(csv_path):
    """
    Demonstrate advanced PID control with anti-windup and derivative filtering.
    Logs data to CSV for system identification.
    """
    print("\n" + "=" * 70)
    print("DEMO 3: CLOSED-LOOP CONTROL - ADVANCED PID")
    print("=" * 70)
    
    from pathlib import Path
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    
    plant = SecondOrderPlant(
        gain=1.0,
        natural_frequency=2.0,
        damping_ratio=0.3,
        sample_time=0.01
    )
    
    # Advanced PID with features
    params = PIDParams(
        kp=18.0,
        ki=6.0,
        kd=6.0,
        sample_time=0.01,
        anti_windup=AntiWindupMethod.BACK_CALCULATION,
        derivative_filter_coeff=10.0,
        setpoint_weight_p=0.6,
        output_min=-200.0,
        output_max=200.0
    )
    
    print(f"\nAdvanced PID Parameters:")
    print(f"  Kp = {params.kp}, Ki = {params.ki}, Kd = {params.kd}")
    print(f"  Anti-windup: {params.anti_windup.value}")
    print(f"  Derivative filter: N = {params.derivative_filter_coeff}")
    print(f"  Setpoint weight: b = {params.setpoint_weight_p}")
    print(f"  Output limits: [{params.output_min}, {params.output_max}]")
    
    scenario = SimulationScenario(
        name="Advanced PID Control",
        duration=15.0,
        sample_time=0.01,
        setpoint_type=SetpointType.STEP,
        setpoint_initial=0.0,
        setpoint_final=100.0,
        setpoint_time=1.0,
        disturbance_type='step',
        disturbance_magnitude=-15.0,
        disturbance_time=8.0
    )
    
    controller = PIDController(params, csv_path=csv_path)
    sim = Simulator(plant, controller)
    result = sim.run(scenario)
    
    print(f"\nData logged to: {csv_path}")
    
    # Plot results
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    fig.suptitle('Advanced PID Control - Mass-Spring-Damper', fontsize=14, fontweight='bold')
    
    axes[0].plot(result.timestamps, result.setpoints, 'g--', linewidth=2, label='Setpoint', alpha=0.7)
    axes[0].plot(result.timestamps, result.measurements, 'b-', linewidth=1.5, label='Position')
    axes[0].axvline(x=8.0, color='red', linestyle=':', linewidth=2, label='Disturbance', alpha=0.5)
    axes[0].set_ylabel('Position', fontsize=11)
    axes[0].set_title('System Response with Disturbance')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(result.timestamps, result.errors, 'r-', linewidth=1.5)
    axes[1].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    axes[1].set_ylabel('Error', fontsize=11)
    axes[1].set_title('Tracking Error')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(result.timestamps, result.p_terms, 'orange', linewidth=1.5, label='P', alpha=0.7)
    axes[2].plot(result.timestamps, result.i_terms, 'cyan', linewidth=1.5, label='I', alpha=0.7)
    axes[2].plot(result.timestamps, result.d_terms, 'brown', linewidth=1.5, label='D', alpha=0.7)
    axes[2].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    axes[2].set_ylabel('Component Value', fontsize=11)
    axes[2].set_title('PID Components')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    axes[3].plot(result.timestamps, result.outputs, 'm-', linewidth=1.5)
    axes[3].axhline(y=params.output_max, color='red', linestyle=':', alpha=0.3, label='Limits')
    axes[3].axhline(y=params.output_min, color='red', linestyle=':', alpha=0.3)
    axes[3].set_xlabel('Time (s)', fontsize=11)
    axes[3].set_ylabel('Control Force', fontsize=11)
    axes[3].set_title('Control Signal')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Performance metrics
    overshoot = np.max(result.measurements) - 100.0
    print(f"\nPerformance:")
    print(f"  Overshoot: {overshoot:.1f}")
    
    # Check disturbance recovery
    dist_idx = np.argmax(result.timestamps >= 8.0)
    if dist_idx > 0:
        max_dist_error = np.max(np.abs(result.errors[dist_idx:dist_idx+300]))
        print(f"  Max error after disturbance: {max_dist_error:.2f}")


def demo_system_identification(csv_path):
    """
    Identify system parameters from logged CSV data.
    """
    print("\n" + "=" * 70)
    print("DEMO 4: SYSTEM IDENTIFICATION FROM CSV DATA")
    print("=" * 70)
    
    print(f"\nReading data from: {csv_path}")
    
    reader = CSVDataReader(csv_path)
    data = reader.read()
    
    print(f"Loaded {len(data.time)} data points")
    print(f"Sample time: {data.sample_time:.4f} s")
    
    identifier = SystemIdentifier(data)
    
    print("\nIdentifying system model...")
    result = identifier.identify(
        model_type=ModelType.SECOND_ORDER,
        tuning_rule='ziegler_nichols'
    )
    
    print("\n" + result.summary())
    
    # Plot identification results
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('System Identification from Closed-Loop Data', fontsize=14, fontweight='bold')
    
    axes[0].plot(data.time, data.output, 'b-', linewidth=1.5, label='Actual Output', alpha=0.7)
    axes[0].plot(result.time, result.simulated_output, 'r--', linewidth=2, label='Model Output')
    axes[0].set_ylabel('Output', fontsize=11)
    axes[0].set_title(f'Model Fit (R² = {result.fit_quality:.4f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(data.time, data.input, 'g-', linewidth=1.5)
    axes[1].set_xlabel('Time (s)', fontsize=11)
    axes[1].set_ylabel('Input', fontsize=11)
    axes[1].set_title('Control Input')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return result


def demo_autotuning():
    """
    Demonstrate autotuning using system identification.
    """
    print("\n" + "=" * 70)
    print("DEMO 5: AUTOTUNING FROM SYSTEM IDENTIFICATION")
    print("=" * 70)
    
    # Generate fresh data for autotuning
    print("\nGenerating experimental data...")
    
    from pathlib import Path
    autotune_csv = "output/autotune_data.csv"
    Path(autotune_csv).parent.mkdir(parents=True, exist_ok=True)
    
    # Create test system
    plant = SecondOrderPlant(
        gain=1.0,
        natural_frequency=1.5,
        damping_ratio=0.4,
        sample_time=0.01
    )
    
    # Run with basic controller to generate data
    basic_params = PIDParams(kp=5.0, ki=1.0, kd=1.0, sample_time=0.01)
    controller = PIDController(basic_params, csv_path=autotune_csv)
    
    scenario = SimulationScenario(
        name="Autotune Data Collection",
        duration=12.0,
        sample_time=0.01,
        setpoint_type=SetpointType.STEP,
        setpoint_initial=0.0,
        setpoint_final=100.0,
        setpoint_time=1.0
    )
    
    sim = Simulator(plant, controller)
    sim.run(scenario)
    
    print(f"Data saved to: {autotune_csv}")
    
    # Perform autotuning
    print("\nRunning autotuning...")
    autotuner = AutotuneFromData(autotune_csv)
    
    result = autotuner.autotune(
        model_type=ModelType.AUTO,
        tuning_rule='imc',
        optimizer='differential_evolution',
        bounds_scale=2.0,
        max_iterations=30
    )
    
    print("\n" + result.summary())
    
    print("\n" + "=" * 70)
    print("RECOMMENDED PID GAINS (Optimized)")
    print("=" * 70)
    print(f"Kp = {result.optimized_gains['kp']:.4f}")
    print(f"Ki = {result.optimized_gains['ki']:.4f}")
    print(f"Kd = {result.optimized_gains['kd']:.4f}")
    print(f"\nPerformance Improvement: {result.improvement:.2f}%")
    print(f"Model Fit Quality (R²): {result.identification.fit_quality:.4f}")
    
    # Compare original vs optimized
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('Autotuning Results Comparison', fontsize=14, fontweight='bold')
    
    # Test both controllers
    plant.reset()
    original_controller = PIDController(basic_params)
    original_sim = Simulator(plant, original_controller)
    original_result = original_sim.run(scenario)
    
    plant.reset()
    optimized_params = PIDParams(
        kp=result.optimized_gains['kp'],
        ki=result.optimized_gains['ki'],
        kd=result.optimized_gains['kd'],
        sample_time=0.01
    )
    optimized_controller = PIDController(optimized_params)
    optimized_sim = Simulator(plant, optimized_controller)
    optimized_result = optimized_sim.run(scenario)
    
    axes[0].plot(original_result.timestamps, original_result.setpoints, 'g--', 
                linewidth=2, label='Setpoint', alpha=0.7)
    axes[0].plot(original_result.timestamps, original_result.measurements, 'b-', 
                linewidth=1.5, label='Original PID', alpha=0.7)
    axes[0].plot(optimized_result.timestamps, optimized_result.measurements, 'r-', 
                linewidth=1.5, label='Optimized PID')
    axes[0].set_ylabel('Position', fontsize=11)
    axes[0].set_title('Response Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(original_result.timestamps, original_result.errors, 'b-', 
                linewidth=1.5, label='Original Error', alpha=0.7)
    axes[1].plot(optimized_result.timestamps, optimized_result.errors, 'r-', 
                linewidth=1.5, label='Optimized Error')
    axes[1].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    axes[1].set_xlabel('Time (s)', fontsize=11)
    axes[1].set_ylabel('Error', fontsize=11)
    axes[1].set_title('Error Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()


def main():
    print("\n" + "=" * 70)
    print("MASS-SPRING-DAMPER SYSTEM PID CONTROL DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo shows PID control of a classic second-order mechanical system.")
    print("The mass-spring-damper is fundamental in mechanical engineering and")
    print("represents many real systems: suspension systems, robotic joints, etc.")
    
    demo_open_loop() # Open loop response
    demo_closed_loop_simple() # Closed loop response with simple PID
    csv_path = "output/demo_closed_loop_advanced.csv"
    demo_closed_loop_advanced(csv_path) # Closed loop response with advanced PID
    demo_system_identification(csv_path) # System identification with  output of the closed loop response
    demo_autotuning() # Autotuning of system identified
    
    print("\n" + "=" * 70)
    print("All demos complete. Close plot windows to exit.")
    print("=" * 70)
    
    plt.show()


if __name__ == "__main__":
    main()