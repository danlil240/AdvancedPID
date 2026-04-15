#!/usr/bin/env python3
"""
Realtime tuner demo: car position control with acceleration commands.

Goal: move a car from rest to a target position and stop precisely.
We start with a very bad PID and let RealtimeTuner optimize it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from pid_control.core.pid_controller import PIDController
from pid_control.core.pid_params import PIDParams, AntiWindupMethod
from pid_control.envs import FrictionPlantEnv
from pid_control.simulation.simulator import Simulator
from pid_control.simulation.scenarios import SimulationScenario, SetpointType
from pid_control.tuner.realtime_tuner import RealtimeTuner, CostWeights


def build_car_env(sample_time: float) -> FrictionPlantEnv:
    """Car model: mass with friction, controlled by acceleration (force)."""
    return FrictionPlantEnv(
        mass=1.0,                # unit mass so input ~= acceleration
        viscous_friction=0.35,   # drag-like friction
        coulomb_friction=0.4,    # kinetic friction
        stiction=1.0,            # static friction (hard to start/stop precisely)
        sample_time=sample_time,
        initial_position=0.0,
        initial_velocity=0.0
    )


def print_metrics(label: str, sim: Simulator, result, target: float) -> None:
    metrics = sim.analyze(result)
    step = metrics["step_response"]
    error = metrics["error"]
    print(
        f"{label:<12} "
        f"rise={step['rise_time']:.2f}s  "
        f"settle2%={step['settling_time_2pct']:.2f}s  "
        f"overshoot={step['overshoot_percent']:.1f}%  "
        f"ss_error={abs(step['steady_state_error']):.2f}  "
        f"IAE={error['iae']:.1f}"
    )


def main() -> None:
    target_position = 50.0
    sample_time = 0.02
    duration = 22.0
    accel_limit = 2.0

    print("=" * 70)
    print("REALTIME PID TUNER DEMO - CAR STOPPING AT A TARGET POSITION")
    print("=" * 70)
    print("\nPlant: Friction + inertia (car-like), control output is acceleration")
    print(f"Target position: {target_position} m, accel limits: +/- {accel_limit} m/s^2")

    # Very bad PID: P-only, too weak for stiction -> large steady-state error
    bad_params = PIDParams(
        kp=0.05,
        ki=0.0,
        kd=0.0,
        sample_time=sample_time,
        output_min=-accel_limit,
        output_max=accel_limit,
        anti_windup=AntiWindupMethod.BACK_CALCULATION,
        derivative_filter_coeff=10.0
    )

    # Use a fresh plant for tuning via Gymnasium env
    env_for_tuning = build_car_env(sample_time)
    plant_for_tuning = env_for_tuning.plant
    tune_controller = PIDController(bad_params)

    tuner = RealtimeTuner(
        tune_controller,
        plant_for_tuning,
        optimizer="differential_evolution",
        bounds={
            "kp": (0.0, 8.0),
            "ki": (0.0, 2.0),
            "kd": (0.0, 4.0)
        },
        cost_weights=CostWeights(
            iae=0.1,
            itae=0.0,
            overshoot=0.0,
            settling=50.0,
            control_effort=0.0
        )
    )

    print("\nRunning auto-tuning (this may take a moment)...")
    result = tuner.auto_tune(
        setpoint=target_position,
        duration=duration,
        initial_measurement=0.0,
        max_iterations=40,
        apply_result=True
    )

    print("\nTuning complete")
    print(f"  Success: {result.success}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Final cost: {result.cost:.4f}")
    print(f"  Tuned gains: Kp={result.kp:.4f}, Ki={result.ki:.4f}, Kd={result.kd:.4f}")

    tuned_params = bad_params.copy(kp=result.kp, ki=result.ki, kd=result.kd)

    # Compare controllers on the same scenario
    scenario = SimulationScenario(
        name="Car Stop - Position Step",
        duration=duration,
        sample_time=sample_time,
        setpoint_type=SetpointType.STEP,
        setpoint_initial=0.0,
        setpoint_final=target_position,
        setpoint_time=1.0
    )

    sim = Simulator(build_car_env(sample_time).plant, PIDController(bad_params))
    comparison = sim.run_comparison(
        scenario,
        {"Very Bad": bad_params, "Auto-Tuned": tuned_params}
    )

    print("\nPerformance summary:")
    print_metrics("Very Bad", sim, comparison["Very Bad"], target_position)
    print_metrics("Auto-Tuned", sim, comparison["Auto-Tuned"], target_position)

    # --- Gymnasium live-render preview with tuned controller ---
    print("\nRunning Gymnasium live-render episode with tuned controller...")
    render_env = build_car_env(sample_time)
    render_env.render_mode = "human"
    render_env.set_setpoint(target_position)
    tuned_ctrl = PIDController(tuned_params)
    obs, _ = render_env.reset()
    terminated, truncated = False, False
    while not (terminated or truncated):
        pos = obs[0]
        ctrl = tuned_ctrl.update(target_position, pos)
        obs, _, terminated, truncated, _ = render_env.step(np.array([ctrl]))
    render_env.close()

    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Car Position Control: Very Bad vs Auto-Tuned", fontsize=14, fontweight="bold")

    # Position
    axes[0].plot(
        comparison["Very Bad"].timestamps,
        comparison["Very Bad"].setpoints,
        "k--",
        linewidth=1.5,
        label="Target"
    )
    axes[0].plot(
        comparison["Very Bad"].timestamps,
        comparison["Very Bad"].measurements,
        "r-",
        linewidth=1.5,
        label="Very Bad"
    )
    axes[0].plot(
        comparison["Auto-Tuned"].timestamps,
        comparison["Auto-Tuned"].measurements,
        "b-",
        linewidth=1.5,
        label="Auto-Tuned"
    )
    axes[0].set_ylabel("Position (m)")
    axes[0].set_title("Position Response")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Acceleration (control output)
    axes[1].plot(
        comparison["Very Bad"].timestamps,
        comparison["Very Bad"].outputs,
        "r-",
        linewidth=1.2,
        label="Very Bad accel"
    )
    axes[1].plot(
        comparison["Auto-Tuned"].timestamps,
        comparison["Auto-Tuned"].outputs,
        "b-",
        linewidth=1.2,
        label="Auto-Tuned accel"
    )
    axes[1].axhline(accel_limit, color="gray", linestyle=":", linewidth=1)
    axes[1].axhline(-accel_limit, color="gray", linestyle=":", linewidth=1)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Acceleration (m/s^2)")
    axes[1].set_title("Control Output (Acceleration)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
