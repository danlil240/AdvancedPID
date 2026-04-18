#!/usr/bin/env python3
"""
Double Inverted Pendulum with Autotuning

Demonstrates automatic PID tuning for the double inverted pendulum using:
- Genetic algorithm optimization
- Cost function based on settling time and control effort
- Automatic parameter search for stabilization
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

from pid_control.core.pid_controller import PIDController
from pid_control.core.pid_params import PIDParams, AntiWindupMethod
from pid_control.envs import DoublePendulumEnv
from pid_control.tuner.optimization_methods import GeneticTuner


EARLY_EXIT_ANGLE = 0.8  # radians (~46 degrees)
SUCCESS_ANGLE = 0.6     # radians (~34 degrees)
TUNE_DURATION = 4.0
VERIFY_DURATION = 8.0


def simulate_pendulum(kp, ki, kd, K_theta1, K_theta1_dot, K_theta2, K_theta2_dot,
                      duration=5.0, verbose=False, early_exit_angle=EARLY_EXIT_ANGLE,
                      return_metrics=False):
    """
    Simulate double pendulum with given PID and state feedback gains.
    
    Returns cost metric (lower is better).
    """
    # Create plant via Gymnasium env
    env = DoublePendulumEnv(
        cart_mass=1.0,
        pendulum1_mass=0.1,
        pendulum2_mass=0.1,
        pendulum1_length=0.5,
        pendulum2_length=0.5,
        friction=0.1,
        sample_time=0.005,
        initial_angle1=0.15,  # ~8.6 degrees
        initial_angle2=0.10,  # ~5.7 degrees
        control_mode='position',
        integrator='rk4'
    )
    plant = env.plant
    
    # Create controller
    pos_params = PIDParams(
        kp=kp,
        ki=ki,
        kd=kd,
        sample_time=0.005,
        output_min=-150.0,
        output_max=150.0,
        anti_windup=AntiWindupMethod.BACK_CALCULATION,
        derivative_filter_coeff=20.0,
        derivative_mode='measurement'
    )
    controller = PIDController(pos_params)
    
    # Simulation
    dt = plant.sample_time
    n_steps = int(duration / dt)
    
    cart_pos_target = 0.0
    total_error = 0.0
    total_control_effort = 0.0
    max_angle = 0.0
    # Track angle/position during the last 25% of the sim (settled behaviour).
    # The cost rewards driving these toward zero; using peak max_angle is
    # useless because it is floored by the initial disturbance.
    settle_start = int(0.75 * n_steps)
    settle_angle_sum = 0.0
    settle_pos_sum = 0.0
    settle_count = 0

    for i in range(n_steps):
        # State feedback control
        theta1 = plant.state[2]
        theta1_dot = plant.state[3]
        theta2 = plant.state[4]
        theta2_dot = plant.state[5]
        
        # Position control
        cart_pos = plant.state[0]
        pos_control = controller.update(cart_pos_target, cart_pos)
        
        # Angle stabilization feedback
        angle_feedback = (
            K_theta1 * theta1 + K_theta1_dot * theta1_dot +
            K_theta2 * theta2 + K_theta2_dot * theta2_dot
        )
        
        # Combined control
        force = pos_control + angle_feedback
        
        # Update plant
        plant.update(force)
        
        # Track metrics
        total_error += abs(cart_pos)
        total_control_effort += abs(force)
        max_angle = max(max_angle, abs(theta1), abs(theta2))
        if i >= settle_start:
            settle_angle_sum += max(abs(theta1), abs(theta2))
            settle_pos_sum += abs(cart_pos)
            settle_count += 1

        # Early exit for unstable candidates to speed up tuning
        if early_exit_angle is not None and max_angle > early_exit_angle:
            remaining = n_steps - i
            cost = 1e6 + remaining * 10.0 + max_angle * 1000.0
            if verbose:
                print(
                    f"  Kp={kp:.2f}, Ki={ki:.3f}, Kd={kd:.2f} -> Cost={cost:.2f}, "
                    f"MaxAngle={np.degrees(max_angle):.1f} deg"
                )
            if return_metrics:
                return cost, max_angle
            return cost

    settled_angle = settle_angle_sum / max(settle_count, 1)
    settled_pos = settle_pos_sum / max(settle_count, 1)

    # Cost rewards true settling: the GA is scored on the mean angle / position
    # over the final 25% of the sim, not on the initial-condition-floored max.
    cost = (
        settled_angle * 100000 +      # Drive steady-state angle to zero
        settled_pos * 5000 +          # Drive cart back to origin
        total_error * 2 +             # Transient cart error
        total_control_effort * 0.001 + # Small control-effort regularizer
        (max_angle > 0.4) * 50000      # Penalty for transient excursions > ~23 deg
    )
    
    if verbose:
        print(f"  Kp={kp:.2f}, Ki={ki:.3f}, Kd={kd:.2f} -> Cost={cost:.2f}, MaxAngle={np.degrees(max_angle):.1f} deg")
    
    if return_metrics:
        return cost, max_angle
    return cost


def autotune_position_pid():
    """
    Automatically tune the position PID controller for the double pendulum.
    State feedback gains are kept fixed.
    """
    print("=" * 70)
    print("DOUBLE PENDULUM AUTOTUNING")
    print("=" * 70)
    print("\nAutotuning position PID controller...")
    print("State feedback gains are fixed:")
    
    # Improved state feedback gains for better stabilization
    K_theta1 = 200.0
    K_theta1_dot = 40.0
    K_theta2 = 150.0
    K_theta2_dot = 35.0
    
    print(f"  K_theta1={K_theta1}, K_theta1_dot={K_theta1_dot}")
    print(f"  K_theta2={K_theta2}, K_theta2_dot={K_theta2_dot}")
    
    # Bounds cover both the tiny-gain regime (Kp~5-20 that works with strong
    # state feedback, as in demo_double_pendulum.py) and the larger-gain regime.
    bounds = {
        'kp': (2.0, 120.0),
        'ki': (0.0, 5.0),
        'kd': (0.0, 40.0)
    }

    # Define cost function for tuner
    def cost_function(kp, ki, kd):
        if not (bounds['kp'][0] <= kp <= bounds['kp'][1]):
            return 1e6
        if not (bounds['ki'][0] <= ki <= bounds['ki'][1]):
            return 1e6
        if not (bounds['kd'][0] <= kd <= bounds['kd'][1]):
            return 1e6
        return simulate_pendulum(kp, ki, kd, K_theta1, K_theta1_dot,
                                K_theta2, K_theta2_dot, duration=TUNE_DURATION)

    # Genetic tuner config with more iterations for better convergence
    max_attempts = 3
    max_iterations = 80
    population_size = 40
    
    mutation_rate = 0.15
    crossover_rate = 0.8
    
    # Initial guess near the hand-tuned working values from demo_double_pendulum.py.
    initial_params = {'kp': 10.0, 'ki': 0.5, 'kd': 5.0}
    
    print(f"\nInitial parameters: Kp={initial_params['kp']}, Ki={initial_params['ki']}, Kd={initial_params['kd']}")
    print("\nOptimizing with genetic algorithm... (this may take a bit)")
    
    best_result = None
    best_cost = float('inf')
    best_max_angle = None
    converged = False
    
    for attempt in range(1, max_attempts + 1):
        seed = 100 + attempt
        np.random.seed(seed)
        print(f"\nAttempt {attempt}/{max_attempts} (seed={seed})")
        
        tuner = GeneticTuner(
            bounds,
            cost_function,
            population_size=population_size,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate
        )
        
        result = tuner.optimize(initial_params, max_iterations=max_iterations)
        verify_cost, max_angle = simulate_pendulum(
            result.kp,
            result.ki,
            result.kd,
            K_theta1,
            K_theta1_dot,
            K_theta2,
            K_theta2_dot,
            duration=VERIFY_DURATION,
            verbose=False,
            early_exit_angle=None,
            return_metrics=True
        )
        
        if verify_cost < best_cost:
            best_cost = verify_cost
            best_result = result
            best_max_angle = max_angle
        
        if max_angle <= SUCCESS_ANGLE:
            converged = True
            break
    
    result = best_result
    
    print(f"\n{'='*70}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*70}")
    print(f"Converged: {converged}")
    print(f"Iterations: {result.iterations}")
    print(f"Final cost: {best_cost:.2f}")
    print(f"\nOptimal PID parameters:")
    print(f"  Kp = {result.kp:.3f}")
    print(f"  Ki = {result.ki:.3f}")
    print(f"  Kd = {result.kd:.3f}")
    print(f"  Max angle = {np.degrees(best_max_angle):.1f} deg")
    
    # Test the optimized controller
    print(f"\nTesting optimized controller...")
    final_cost = simulate_pendulum(
        result.kp,
        result.ki,
        result.kd,
        K_theta1,
        K_theta1_dot,
        K_theta2,
        K_theta2_dot,
        duration=VERIFY_DURATION,
        verbose=True,
        early_exit_angle=None
    )
    
    return result, K_theta1, K_theta1_dot, K_theta2, K_theta2_dot


def run_animated_demo(kp, ki, kd, K_theta1, K_theta1_dot, K_theta2, K_theta2_dot):
    """Run animated visualization with tuned parameters."""
    print(f"\n{'='*70}")
    print("RUNNING ANIMATED DEMO")
    print(f"{'='*70}")
    
    # Create plant via Gymnasium env
    env = DoublePendulumEnv(
        cart_mass=1.0,
        pendulum1_mass=0.1,
        pendulum2_mass=0.1,
        pendulum1_length=0.5,
        pendulum2_length=0.5,
        friction=0.1,
        sample_time=0.005,
        initial_angle1=0.15,
        initial_angle2=0.10,
        control_mode='position',
        integrator='rk4'
    )
    plant = env.plant
    
    # Create controller
    pos_params = PIDParams(
        kp=kp, ki=ki, kd=kd,
        sample_time=0.005,
        output_min=-150.0,
        output_max=150.0,
        anti_windup=AntiWindupMethod.BACK_CALCULATION,
        derivative_filter_coeff=20.0,
        derivative_mode='measurement'
    )
    controller = PIDController(pos_params)
    
    # Setup figure
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])
    
    ax_cart = fig.add_subplot(gs[0, :])
    ax_angle = fig.add_subplot(gs[1, 0])
    ax_force = fig.add_subplot(gs[1, 1])
    
    # Cart visualization
    ax_cart.set_xlim(-1.5, 1.5)
    ax_cart.set_ylim(-0.2, 1.2)
    ax_cart.set_aspect('equal')
    ax_cart.grid(True, alpha=0.3)
    ax_cart.set_title(f'Double Pendulum (Autotuned: Kp={kp:.2f}, Ki={ki:.3f}, Kd={kd:.2f})')
    
    from matplotlib.patches import Rectangle, Circle
    cart = Rectangle((-0.1, 0), 0.2, 0.1, fc='blue', ec='black')
    ax_cart.add_patch(cart)
    link1, = ax_cart.plot([], [], 'o-', lw=3, color='red', markersize=8)
    link2, = ax_cart.plot([], [], 'o-', lw=3, color='green', markersize=8)
    
    # Angle plot
    ax_angle.set_xlim(0, 10)
    ax_angle.set_ylim(-20, 20)
    ax_angle.grid(True, alpha=0.3)
    ax_angle.set_xlabel('Time (s)')
    ax_angle.set_ylabel('Angle (deg)')
    ax_angle.set_title('Pendulum Angles')
    line_angle1, = ax_angle.plot([], [], 'r-', label='theta1', linewidth=2)
    line_angle2, = ax_angle.plot([], [], 'g-', label='theta2', linewidth=2)
    ax_angle.legend()
    
    # Force plot
    ax_force.set_xlim(0, 10)
    ax_force.set_ylim(-50, 50)
    ax_force.grid(True, alpha=0.3)
    ax_force.set_xlabel('Time (s)')
    ax_force.set_ylabel('Force (N)')
    ax_force.set_title('Control Force')
    line_force, = ax_force.plot([], [], 'b-', linewidth=2)
    
    # Data storage
    data = {'t': [], 'angle1': [], 'angle2': [], 'force': []}
    
    cart_pos_target = 0.0
    max_points = 2000
    
    def init():
        return cart, link1, link2, line_angle1, line_angle2, line_force
    
    def animate(frame):
        # State feedback control
        theta1 = plant.state[2]
        theta1_dot = plant.state[3]
        theta2 = plant.state[4]
        theta2_dot = plant.state[5]
        
        cart_pos = plant.state[0]
        pos_control = controller.update(cart_pos_target, cart_pos)
        
        angle_feedback = (
            K_theta1 * theta1 + K_theta1_dot * theta1_dot +
            K_theta2 * theta2 + K_theta2_dot * theta2_dot
        )
        
        force = pos_control + angle_feedback
        # Use plant.update() directly to match the tuner's (unclipped) physics.
        # env.step() would clip to action_space bounds (+/-150) and break convergence
        # because the tuned gains rely on the full unclipped state-feedback authority.
        plant.update(force)
        
        t = frame * plant.sample_time
        
        # Update cart position
        cart.set_x(cart_pos - 0.1)
        
        # Update pendulums
        x0, y0 = cart_pos, 0.05
        x1 = x0 + plant.L1 * np.sin(theta1)
        y1 = y0 + plant.L1 * np.cos(theta1)
        link1.set_data([x0, x1], [y0, y1])
        
        x2 = x1 + plant.L2 * np.sin(theta2)
        y2 = y1 + plant.L2 * np.cos(theta2)
        link2.set_data([x1, x2], [y1, y2])
        
        # Store data
        data['t'].append(t)
        data['angle1'].append(np.degrees(theta1))
        data['angle2'].append(np.degrees(theta2))
        data['force'].append(force)
        
        # Update plots
        if len(data['t']) > 1:
            line_angle1.set_data(data['t'], data['angle1'])
            line_angle2.set_data(data['t'], data['angle2'])
            line_force.set_data(data['t'], data['force'])
            
            if t > 7:
                ax_angle.set_xlim(t - 7, t + 1)
                ax_force.set_xlim(t - 7, t + 1)
        
        return cart, link1, link2, line_angle1, line_angle2, line_force
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=max_points,
                        interval=20, blit=True, repeat=False)
    
    plt.tight_layout()
    print("\nAnimation running... Close window when done.")
    plt.show()
    
    return anim


def main(show: bool = False) -> None:
    # Run autotuning
    result, K_theta1, K_theta1_dot, K_theta2, K_theta2_dot = autotune_position_pid()
    
    print(f"\n{'='*70}")
    print("Autotuning complete. Run with --show to see animated demo.")
    
    if show:
        run_animated_demo(result.kp, result.ki, result.kd,
                         K_theta1, K_theta1_dot, K_theta2, K_theta2_dot)
    
    print("\nDone!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Double pendulum autotuning demo")
    parser.add_argument("--show", action="store_true", help="Show animated demo")
    args = parser.parse_args()
    main(show=args.show)
