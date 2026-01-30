#!/usr/bin/env python3
"""
Double Inverted Pendulum Stabilization Demo

Demonstrates PID control of a highly nonlinear, underactuated system:
- Double inverted pendulum on a cart
- Stabilization from initial disturbance
- Multiple PID strategies comparison
- Animated visualization with pendulum rendering
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

from pid_control.core.pid_controller import PIDController
from pid_control.core.pid_params import PIDParams, AntiWindupMethod
from pid_control.plants.double_pendulum import DoublePendulumCart


def run_stabilization_test():
    """
    Test PID stabilization of double pendulum from initial disturbance.
    """
    print("=" * 70)
    print("Double Inverted Pendulum Stabilization Test")
    print("=" * 70)
    
    # Create double pendulum with initial disturbance
    plant = DoublePendulumCart(
        cart_mass=1.0,
        pendulum1_mass=0.1,
        pendulum2_mass=0.1,
        pendulum1_length=0.5,
        pendulum2_length=0.5,
        friction=0.1,
        sample_time=0.005,  # Small timestep for stability
        initial_angle1=0.15,  # ~8.6 degrees
        initial_angle2=0.10,  # ~5.7 degrees
        control_mode='position'
    )
    
    print(f"\nPlant Configuration:")
    info = plant.get_info()
    print(f"  Cart mass: {info['cart_mass']} kg")
    print(f"  Pendulum masses: {info['pendulum1_mass']}, {info['pendulum2_mass']} kg")
    print(f"  Pendulum lengths: {info['pendulum1_length']}, {info['pendulum2_length']} m")
    print(f"  Initial angles: {info['initial_angle1_deg']:.1f}°, {info['initial_angle2_deg']:.1f}°")
    
    # Use state-feedback approach with gains for cart position and pendulum angles
    # This is more effective than simple PID on cart position alone
    
    # Position controller
    pos_params = PIDParams(
        kp=10.0,
        ki=0.5,
        kd=5.0,
        sample_time=0.005,
        output_min=-150.0,
        output_max=150.0,
        anti_windup=AntiWindupMethod.BACK_CALCULATION,
        derivative_filter_coeff=20.0,
        derivative_mode='measurement'
    )
    
    # Angle stabilization gains (proportional to angle and angular velocity)
    K_theta1 = 120.0  # Gain for pendulum 1 angle
    K_theta1_dot = 25.0  # Gain for pendulum 1 angular velocity
    K_theta2 = 80.0   # Gain for pendulum 2 angle
    K_theta2_dot = 20.0  # Gain for pendulum 2 angular velocity
    
    print(f"\nControl Strategy: State Feedback")
    print(f"  Position PID: Kp={pos_params.kp}, Ki={pos_params.ki}, Kd={pos_params.kd}")
    print(f"  Angle gains: K_θ1={K_theta1}, K_θ1_dot={K_theta1_dot}")
    print(f"               K_θ2={K_theta2}, K_θ2_dot={K_theta2_dot}")
    
    pos_controller = PIDController(pos_params)
    
    # Simulation parameters
    duration = 10.0
    dt = plant.sample_time
    n_steps = int(duration / dt)
    
    # Data storage
    times = np.zeros(n_steps)
    cart_positions = np.zeros(n_steps)
    cart_velocities = np.zeros(n_steps)
    angle1s = np.zeros(n_steps)
    angle2s = np.zeros(n_steps)
    forces = np.zeros(n_steps)
    setpoints = np.zeros(n_steps)
    
    # Run simulation
    print(f"\nRunning simulation for {duration}s...")
    
    setpoint = 0.0  # Keep cart at origin
    
    for i in range(n_steps):
        t = i * dt
        
        # Get current state
        cart_pos = plant.cart_position
        theta1 = plant.pendulum1_angle
        theta1_dot = plant.pendulum1_velocity
        theta2 = plant.pendulum2_angle
        theta2_dot = plant.pendulum2_velocity
        
        # Position control component
        force_pos = pos_controller.update(setpoint, cart_pos, timestamp=t)
        
        # State feedback for pendulum stabilization
        # Force to counteract pendulum angles and velocities
        force_stabilize = (K_theta1 * theta1 + K_theta1_dot * theta1_dot +
                          K_theta2 * theta2 + K_theta2_dot * theta2_dot)
        
        # Combined control force
        force = force_pos + force_stabilize
        
        # Apply output limits
        force = np.clip(force, pos_params.output_min, pos_params.output_max)
        
        # Update plant
        plant.update(force)
        
        # Store data
        times[i] = t
        cart_positions[i] = plant.cart_position
        cart_velocities[i] = plant.cart_velocity
        angle1s[i] = plant.pendulum1_angle
        angle2s[i] = plant.pendulum2_angle
        forces[i] = force
        setpoints[i] = setpoint
    
    # Analysis
    print(f"\nStabilization Results:")
    print(f"  Final cart position: {cart_positions[-1]:.4f} m")
    print(f"  Final pendulum 1 angle: {np.degrees(angle1s[-1]):.2f}°")
    print(f"  Final pendulum 2 angle: {np.degrees(angle2s[-1]):.2f}°")
    
    # Check stability
    final_stable = plant.is_stable(angle_threshold=0.05)  # ~2.9 degrees
    print(f"  System stable: {final_stable}")
    
    # Find when system first stabilizes
    stable_threshold = 0.05
    stable_mask = (np.abs(angle1s) < stable_threshold) & (np.abs(angle2s) < stable_threshold)
    if np.any(stable_mask):
        stabilization_time = times[np.argmax(stable_mask)]
        print(f"  Stabilization time: {stabilization_time:.2f}s")
    else:
        print(f"  System did not stabilize within {duration}s")
    
    # Calculate max angles
    max_angle1 = np.max(np.abs(angle1s))
    max_angle2 = np.max(np.abs(angle2s))
    print(f"  Max pendulum 1 angle: {np.degrees(max_angle1):.2f}°")
    print(f"  Max pendulum 2 angle: {np.degrees(max_angle2):.2f}°")
    
    # Plot results
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])
    
    # Cart position
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(times, setpoints, 'g--', linewidth=2, label='Setpoint', alpha=0.7)
    ax1.plot(times, cart_positions, 'b-', linewidth=1.5, label='Cart Position')
    ax1.set_ylabel('Position (m)', fontsize=11)
    ax1.set_title('Double Inverted Pendulum Stabilization', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Pendulum angles
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(times, np.degrees(angle1s), 'r-', linewidth=1.5, label='Pendulum 1')
    ax2.plot(times, np.degrees(angle2s), 'm-', linewidth=1.5, label='Pendulum 2')
    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(y=np.degrees(stable_threshold), color='green', linestyle=':', alpha=0.3, label='Stable zone')
    ax2.axhline(y=-np.degrees(stable_threshold), color='green', linestyle=':', alpha=0.3)
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Angle (degrees)', fontsize=11)
    ax2.set_title('Pendulum Angles')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Control force
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(times, forces, 'purple', linewidth=1.5)
    ax3.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax3.axhline(y=pos_params.output_max, color='red', linestyle=':', alpha=0.3, label='Limits')
    ax3.axhline(y=pos_params.output_min, color='red', linestyle=':', alpha=0.3)
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('Force (N)', fontsize=11)
    ax3.set_title('Control Force')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Phase portrait - Pendulum 1
    ax4 = fig.add_subplot(gs[2, 0])
    angle1_vel = np.degrees(np.diff(angle1s))/dt
    scatter = ax4.scatter(np.degrees(angle1s[:-1]), angle1_vel,
                         c=times[:-1], cmap='viridis', s=1, alpha=0.5)
    ax4.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax4.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax4.set_xlabel('Angle (degrees)', fontsize=11)
    ax4.set_ylabel('Angular Velocity (deg/s)', fontsize=11)
    ax4.set_title('Phase Portrait - Pendulum 1')
    plt.colorbar(scatter, ax=ax4, label='Time (s)')
    ax4.grid(True, alpha=0.3)
    
    # Phase portrait - Pendulum 2
    ax5 = fig.add_subplot(gs[2, 1])
    angle2_vel = np.degrees(np.diff(angle2s))/dt
    scatter = ax5.scatter(np.degrees(angle2s[:-1]), angle2_vel,
                         c=times[:-1], cmap='plasma', s=1, alpha=0.5)
    ax5.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax5.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax5.set_xlabel('Angle (degrees)', fontsize=11)
    ax5.set_ylabel('Angular Velocity (deg/s)', fontsize=11)
    ax5.set_title('Phase Portrait - Pendulum 2')
    plt.colorbar(scatter, ax=ax5, label='Time (s)')
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return times, cart_positions, angle1s, angle2s, forces, plant


def create_animated_visualization():
    """
    Create animated visualization of the double pendulum.
    """
    print("\n" + "=" * 70)
    print("Animated Double Pendulum Visualization")
    print("=" * 70)
    
    # Create system
    plant = DoublePendulumCart(
        cart_mass=1.0,
        pendulum1_mass=0.1,
        pendulum2_mass=0.1,
        pendulum1_length=0.5,
        pendulum2_length=0.5,
        friction=0.1,
        sample_time=0.01,
        initial_angle1=0.2,
        initial_angle2=0.15,
        control_mode='position'
    )
    
    params = PIDParams(
        kp=80.0, ki=2.0, kd=35.0,
        sample_time=0.01,
        output_min=-150.0, output_max=150.0,
        anti_windup=AntiWindupMethod.BACK_CALCULATION,
        derivative_filter_coeff=15.0,
        derivative_mode='measurement'
    )
    
    controller = PIDController(params)
    
    # Setup figure
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])
    
    # Animation axis
    ax_anim = fig.add_subplot(gs[0, :])
    ax_anim.set_xlim(-2, 2)
    ax_anim.set_ylim(-0.5, 1.5)
    ax_anim.set_aspect('equal')
    ax_anim.grid(True, alpha=0.3)
    ax_anim.set_xlabel('Position (m)')
    ax_anim.set_ylabel('Height (m)')
    ax_anim.set_title('Double Inverted Pendulum on Cart', fontsize=14, fontweight='bold')
    
    # Cart
    cart_width = 0.3
    cart_height = 0.1
    cart = Rectangle((0, 0), cart_width, cart_height, fc='blue', ec='black', linewidth=2)
    ax_anim.add_patch(cart)
    
    # Pendulum links
    link1, = ax_anim.plot([], [], 'ro-', linewidth=3, markersize=8, label='Pendulum 1')
    link2, = ax_anim.plot([], [], 'mo-', linewidth=3, markersize=8, label='Pendulum 2')
    
    # Ground
    ax_anim.axhline(y=0, color='brown', linewidth=3, alpha=0.5)
    ax_anim.legend(loc='upper right')
    
    # Angle plots
    ax_angle = fig.add_subplot(gs[1, 0])
    line_angle1, = ax_angle.plot([], [], 'r-', linewidth=1.5, label='Pendulum 1')
    line_angle2, = ax_angle.plot([], [], 'm-', linewidth=1.5, label='Pendulum 2')
    ax_angle.set_xlim(0, 8)
    ax_angle.set_ylim(-20, 20)
    ax_angle.set_xlabel('Time (s)')
    ax_angle.set_ylabel('Angle (degrees)')
    ax_angle.set_title('Pendulum Angles')
    ax_angle.legend()
    ax_angle.grid(True, alpha=0.3)
    
    # Force plot
    ax_force = fig.add_subplot(gs[1, 1])
    line_force, = ax_force.plot([], [], 'purple', linewidth=1.5)
    ax_force.set_xlim(0, 8)
    ax_force.set_ylim(-100, 100)
    ax_force.set_xlabel('Time (s)')
    ax_force.set_ylabel('Force (N)')
    ax_force.set_title('Control Force')
    ax_force.grid(True, alpha=0.3)
    
    # Data storage
    max_points = 800
    data = {
        't': [],
        'angle1': [],
        'angle2': [],
        'force': []
    }
    
    def init():
        cart.set_xy((-cart_width/2, 0))
        link1.set_data([], [])
        link2.set_data([], [])
        line_angle1.set_data([], [])
        line_angle2.set_data([], [])
        line_force.set_data([], [])
        return cart, link1, link2, line_angle1, line_angle2, line_force
    
    def animate(frame):
        t = frame * plant.sample_time
        
        # Control
        cart_pos = plant.cart_position
        force = controller.update(0.0, cart_pos, timestamp=t)
        plant.update(force)
        
        # Update cart position
        cart.set_xy((cart_pos - cart_width/2, 0))
        
        # Calculate pendulum positions
        x0 = cart_pos
        y0 = cart_height
        
        # Pendulum 1
        x1 = x0 + plant.L1 * np.sin(plant.pendulum1_angle)
        y1 = y0 + plant.L1 * np.cos(plant.pendulum1_angle)
        link1.set_data([x0, x1], [y0, y1])
        
        # Pendulum 2 (from tip of pendulum 1)
        x2 = x1 + plant.L2 * np.sin(plant.pendulum2_angle)
        y2 = y1 + plant.L2 * np.cos(plant.pendulum2_angle)
        link2.set_data([x1, x2], [y1, y2])
        
        # Store data
        data['t'].append(t)
        data['angle1'].append(np.degrees(plant.pendulum1_angle))
        data['angle2'].append(np.degrees(plant.pendulum2_angle))
        data['force'].append(force)
        
        # Update plots
        if len(data['t']) > 1:
            line_angle1.set_data(data['t'], data['angle1'])
            line_angle2.set_data(data['t'], data['angle2'])
            line_force.set_data(data['t'], data['force'])
            
            # Auto-scroll
            if t > 7:
                ax_angle.set_xlim(t - 7, t + 1)
                ax_force.set_xlim(t - 7, t + 1)
        
        return cart, link1, link2, line_angle1, line_angle2, line_force
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=max_points,
                        interval=20, blit=True, repeat=False)
    
    plt.tight_layout()
    print("\nAnimation running... Close window when done.")
    
    return anim


def main():
    print("\n" + "=" * 70)
    print("DOUBLE INVERTED PENDULUM STABILIZATION DEMO")
    print("=" * 70)
    
    # Run stabilization test
    times, positions, angle1s, angle2s, forces, plant = run_stabilization_test()
    
    # Check if stabilization succeeded
    final_stable = plant.is_stable(angle_threshold=0.05)
    
    if final_stable:
        print("\n" + "=" * 70)
        print("✓ SUCCESS: PID controller successfully stabilized the double pendulum!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("⚠ PARTIAL: System improved but not fully stabilized.")
        print("  Try adjusting PID gains for better performance.")
        print("=" * 70)
    
    # Create animated visualization
    print("\nStarting animated visualization...")
    anim = create_animated_visualization()
    
    plt.show()


if __name__ == "__main__":
    main()
