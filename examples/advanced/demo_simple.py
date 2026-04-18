#!/usr/bin/env python3
"""
Simple PID Controller Demo
Demonstrates basic PID control with a second-order system using Gymnasium.
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt

from pid_control.envs import SecondOrderEnv
from pid_control.core.pid_controller import PIDController
from pid_control.core.pid_params import PIDParams


def main(show: bool = False) -> None:
    print("=" * 60)
    print("Basic PID Controller Demo (Gymnasium)")
    print("=" * 60)

    # Create a Gymnasium environment wrapping a second-order plant
    env = SecondOrderEnv(
        sample_time=0.1,
        gain=1.0,
        damping_ratio=0.707,
        initial_output=0.0,
        initial_velocity=0.0,
        natural_frequency=1.0,
        setpoint=1.0,
        max_steps=100,
        action_low=-10.0,
        action_high=10.0,
        render_mode="human" if show else None,
    )

    # Create PID controller
    pid_params = PIDParams(
        kp=2.0,
        ki=1.0,
        kd=0.5,
        sample_time=0.1,
        output_min=-10.0,
        output_max=10.0
    )
    controller = PIDController(pid_params)

    # Simulation parameters
    setpoint = 1.0
    dt = env.plant.sample_time

    # Data storage for plotting
    time = []
    output = []
    setpoints = []
    control_signal = []

    print(f"Running simulation for {env.max_steps} steps...")
    print(f"Setpoint: {setpoint}")
    print(f"PID parameters: Kp={pid_params.kp}, Ki={pid_params.ki}, Kd={pid_params.kd}")

    # Reset the environment
    obs, info = env.reset()
    measurement = obs[0]

    # Run simulation using Gymnasium step loop
    terminated, truncated = False, False
    step_idx = 0
    while not (terminated or truncated):
        current_time = step_idx * dt

        # Get control signal from PID controller
        control = controller.update(setpoint, measurement)

        # Step the Gymnasium environment
        obs, reward, terminated, truncated, info = env.step(np.array([control]))
        measurement = obs[0]

        # Store data
        time.append(current_time)
        output.append(measurement)
        setpoints.append(setpoint)
        control_signal.append(control)
        step_idx += 1

    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot system output
    ax1.plot(time, output, 'b-', linewidth=2, label='System Output')
    ax1.plot(time, setpoints, 'r--', linewidth=2, label='Setpoint')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Output')
    ax1.set_title('PID Control Response')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot control signal
    ax2.plot(time, control_signal, 'g-', linewidth=2, label='Control Signal')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Control Signal')
    ax2.set_title('PID Controller Output')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    # Calculate performance metrics
    steady_state_error = abs(output[-1] - setpoint)
    overshoot = max(output) - setpoint if max(output) > setpoint else 0
    settling_time = None
    
    # Find settling time (within 2% of setpoint)
    tolerance = 0.02 * setpoint
    for i in range(len(output) - 1, -1, -1):
        if abs(output[i] - setpoint) > tolerance:
            settling_time = time[i]
            break

    print(f"\nPerformance Metrics:")
    print(f"  Steady-state error: {steady_state_error:.4f}")
    print(f"  Overshoot: {overshoot:.4f}")
    print(f"  Settling time: {settling_time:.2f} s" if settling_time else "  Settling time: Not reached")

    env.close()

    if show:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple PID controller demo")
    parser.add_argument("--show", action="store_true", help="Show interactive plots")
    args = parser.parse_args()
    main(show=args.show)
