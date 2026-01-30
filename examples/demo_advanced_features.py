#!/usr/bin/env python3
"""
Advanced PID Features Demo

Demonstrates:
- Anti-windup methods comparison
- Derivative filtering
- Setpoint weighting
- Bumpless transfer
- Saturation handling
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from pid_control.core.pid_controller import PIDController
from pid_control.core.pid_params import PIDParams, AntiWindupMethod, DerivativeMode
from pid_control.plants.second_order import SecondOrderPlant
from pid_control.simulation.simulator import Simulator
from pid_control.simulation.scenarios import SimulationScenario, SetpointType


def demo_anti_windup():
    """Compare different anti-windup methods."""
    print("\n" + "=" * 60)
    print("Anti-Windup Methods Comparison")
    print("=" * 60)
    
    plant = SecondOrderPlant(
        gain=1.0,
        natural_frequency=1.0,
        damping_ratio=0.5,
        sample_time=0.01
    )
    
    # Tight output limits to force saturation
    base_params = PIDParams(
        kp=5.0,
        ki=2.0,
        kd=1.0,
        sample_time=0.01,
        output_min=-10,
        output_max=10
    )
    
    methods = {
        'No Anti-Windup': AntiWindupMethod.NONE,
        'Clamping': AntiWindupMethod.CLAMPING,
        'Back-Calculation': AntiWindupMethod.BACK_CALCULATION,
        'Conditional Integration': AntiWindupMethod.CONDITIONAL_INTEGRATION,
    }
    
    scenario = SimulationScenario(
        name="Anti-Windup Test",
        duration=30.0,
        sample_time=0.01,
        setpoint_type=SetpointType.STEP,
        setpoint_initial=0.0,
        setpoint_final=50.0,  # Large step to cause saturation
        setpoint_time=1.0
    )
    
    param_sets = {
        name: base_params.copy(anti_windup=method)
        for name, method in methods.items()
    }
    
    sim = Simulator(plant, base_params)
    results = sim.run_comparison(scenario, param_sets)
    
    # Plot comparison
    sim.plot_comparison(results, title="Anti-Windup Methods Comparison")
    
    # Print metrics
    print("\nAnti-Windup Performance:")
    print("-" * 60)
    for name, res in results.items():
        metrics = sim.analyze(res)
        sr = metrics['step_response']
        print(f"{name:<25}: Settling={sr['settling_time_2pct']:.2f}s, Overshoot={sr['overshoot_percent']:.1f}%")


def demo_derivative_filtering():
    """Show effect of derivative filtering on noisy signals."""
    print("\n" + "=" * 60)
    print("Derivative Filtering Demo")
    print("=" * 60)
    
    plant = SecondOrderPlant(
        gain=1.0,
        natural_frequency=2.0,
        damping_ratio=0.7,
        sample_time=0.01
    )
    
    # Heavy noise to show filtering effect
    scenario = SimulationScenario(
        name="Noisy Signal Test",
        duration=20.0,
        sample_time=0.01,
        setpoint_type=SetpointType.STEP,
        setpoint_initial=0.0,
        setpoint_final=100.0,
        setpoint_time=1.0,
        measurement_noise_std=3.0  # Significant noise
    )
    
    # Compare different filter coefficients
    param_sets = {
        'No Filtering (N=100)': PIDParams(kp=2.0, ki=1.0, kd=0.5, derivative_filter_coeff=100.0),
        'Light Filtering (N=20)': PIDParams(kp=2.0, ki=1.0, kd=0.5, derivative_filter_coeff=20.0),
        'Medium Filtering (N=10)': PIDParams(kp=2.0, ki=1.0, kd=0.5, derivative_filter_coeff=10.0),
        'Heavy Filtering (N=5)': PIDParams(kp=2.0, ki=1.0, kd=0.5, derivative_filter_coeff=5.0),
    }
    
    sim = Simulator(plant, list(param_sets.values())[0])
    results = sim.run_comparison(scenario, param_sets)
    
    sim.plot_comparison(results, title="Effect of Derivative Filter Coefficient (N)")
    
    print("\nDerivative Filtering Results:")
    print("-" * 60)
    for name, res in results.items():
        metrics = sim.analyze(res)
        ctrl = metrics['control_effort']
        print(f"{name:<25}: Control TV={ctrl['total_variation']:.1f}")


def demo_setpoint_weighting():
    """Demonstrate setpoint weighting for reduced overshoot."""
    print("\n" + "=" * 60)
    print("Setpoint Weighting (2-DOF PID) Demo")
    print("=" * 60)
    
    plant = SecondOrderPlant(
        gain=1.0,
        natural_frequency=1.5,
        damping_ratio=0.3,  # Underdamped - prone to overshoot
        sample_time=0.01
    )
    
    scenario = SimulationScenario(
        name="Setpoint Weighting Test",
        duration=20.0,
        sample_time=0.01,
        setpoint_type=SetpointType.STEP,
        setpoint_initial=0.0,
        setpoint_final=100.0,
        setpoint_time=1.0
    )
    
    # Different setpoint weights
    param_sets = {
        'b=1.0 (Standard PID)': PIDParams(kp=3.0, ki=1.5, kd=1.0, setpoint_weight_p=1.0),
        'b=0.7': PIDParams(kp=3.0, ki=1.5, kd=1.0, setpoint_weight_p=0.7),
        'b=0.5': PIDParams(kp=3.0, ki=1.5, kd=1.0, setpoint_weight_p=0.5),
        'b=0.3': PIDParams(kp=3.0, ki=1.5, kd=1.0, setpoint_weight_p=0.3),
    }
    
    sim = Simulator(plant, list(param_sets.values())[0])
    results = sim.run_comparison(scenario, param_sets)
    
    sim.plot_comparison(results, title="Setpoint Weighting Effect on Overshoot")
    
    print("\nSetpoint Weighting Results:")
    print("-" * 60)
    for name, res in results.items():
        metrics = sim.analyze(res)
        sr = metrics['step_response']
        print(f"{name:<25}: Overshoot={sr['overshoot_percent']:.1f}%, Rise Time={sr['rise_time']:.2f}s")


def demo_bumpless_transfer():
    """Demonstrate bumpless parameter transfer."""
    print("\n" + "=" * 60)
    print("Bumpless Transfer Demo")
    print("=" * 60)
    
    plant = SecondOrderPlant(
        gain=1.0,
        natural_frequency=1.0,
        damping_ratio=0.7,
        sample_time=0.01
    )
    
    duration = 30.0
    dt = 0.01
    n_steps = int(duration / dt)
    
    # Run simulation with parameter change at t=15s
    timestamps = np.zeros(n_steps)
    setpoints = np.zeros(n_steps)
    measurements_bump = np.zeros(n_steps)
    measurements_nobump = np.zeros(n_steps)
    outputs_bump = np.zeros(n_steps)
    outputs_nobump = np.zeros(n_steps)
    
    initial_params = PIDParams(kp=1.0, ki=0.5, kd=0.2, sample_time=dt)
    new_params = PIDParams(kp=3.0, ki=1.5, kd=0.6, sample_time=dt)
    
    # Controller with bumpless transfer
    ctrl_nobump = PIDController(initial_params)
    plant_nobump = SecondOrderPlant(gain=1.0, natural_frequency=1.0, damping_ratio=0.7, sample_time=dt)
    
    # Controller without bumpless transfer
    ctrl_bump = PIDController(initial_params)
    plant_bump = SecondOrderPlant(gain=1.0, natural_frequency=1.0, damping_ratio=0.7, sample_time=dt)
    
    meas_nobump = 0.0
    meas_bump = 0.0
    
    for i in range(n_steps):
        t = i * dt
        sp = 50.0 if t >= 1.0 else 0.0
        
        # Change parameters at t=15s
        if i == int(15.0 / dt):
            ctrl_nobump.set_params(new_params, bumpless=True)
            ctrl_bump.set_params(new_params, bumpless=False)
        
        out_nobump = ctrl_nobump.update(sp, meas_nobump, timestamp=t)
        out_bump = ctrl_bump.update(sp, meas_bump, timestamp=t)
        
        meas_nobump = plant_nobump.update(out_nobump)
        meas_bump = plant_bump.update(out_bump)
        
        timestamps[i] = t
        setpoints[i] = sp
        measurements_nobump[i] = meas_nobump
        measurements_bump[i] = meas_bump
        outputs_nobump[i] = out_nobump
        outputs_bump[i] = out_bump
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    axes[0].plot(timestamps, setpoints, 'g--', label='Setpoint', linewidth=2)
    axes[0].plot(timestamps, measurements_bump, 'r-', label='Without Bumpless', linewidth=1.5)
    axes[0].plot(timestamps, measurements_nobump, 'b-', label='With Bumpless', linewidth=1.5)
    axes[0].axvline(x=15.0, color='gray', linestyle=':', label='Parameter Change')
    axes[0].set_ylabel('Process Value')
    axes[0].set_title('Bumpless Transfer Demo - Response', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(timestamps, outputs_bump, 'r-', label='Without Bumpless', linewidth=1.5)
    axes[1].plot(timestamps, outputs_nobump, 'b-', label='With Bumpless', linewidth=1.5)
    axes[1].axvline(x=15.0, color='gray', linestyle=':')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Control Output')
    axes[1].set_title('Control Signal')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    print("Parameter change at t=15s: Kp 1.0->3.0, Ki 0.5->1.5, Kd 0.2->0.6")
    print("Notice how bumpless transfer avoids the sudden jump in control output.")


def main():
    print("=" * 60)
    print("Advanced PID Features Demonstration")
    print("=" * 60)
    
    demo_anti_windup()
    demo_derivative_filtering()
    demo_setpoint_weighting()
    demo_bumpless_transfer()
    
    print("\n" + "=" * 60)
    print("All demos complete. Close plot windows to exit.")
    print("=" * 60)
    
    plt.show()


if __name__ == "__main__":
    main()
