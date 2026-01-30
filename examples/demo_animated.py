#!/usr/bin/env python3
"""
Animated Real-Time PID Simulation Demo

Shows live updating plots during simulation for
an engaging, educational experience.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

from pid_control.core.pid_controller import PIDController
from pid_control.core.pid_params import PIDParams
from pid_control.plants.second_order import SecondOrderPlant


def run_interactive_simulation():
    """
    Interactive simulation with real-time parameter adjustment.
    """
    print("=" * 60)
    print("Interactive PID Simulation")
    print("=" * 60)
    print("\nUse sliders to adjust PID gains in real-time!")
    print("Click 'Reset' to restart the simulation.\n")
    
    # Setup
    dt = 0.02
    plant = SecondOrderPlant(
        gain=1.0,
        natural_frequency=1.5,
        damping_ratio=0.5,
        sample_time=dt
    )
    
    params = PIDParams(
        kp=2.0, ki=1.0, kd=0.5,
        sample_time=dt,
        output_min=-100, output_max=100
    )
    
    controller = PIDController(params)
    
    # Data storage
    max_points = 1000
    data = {
        't': np.zeros(max_points),
        'sp': np.zeros(max_points),
        'meas': np.zeros(max_points),
        'out': np.zeros(max_points),
        'err': np.zeros(max_points),
    }
    current_idx = [0]
    measurement = [0.0]
    running = [True]
    
    # Setup figure
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle('Interactive PID Controller', fontsize=16, fontweight='bold')
    
    # Create axes
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((3, 2), (1, 0))
    ax3 = plt.subplot2grid((3, 2), (1, 1))
    
    # Leave space for sliders at bottom
    plt.subplots_adjust(bottom=0.25)
    
    # Initialize lines
    line_sp, = ax1.plot([], [], 'g--', linewidth=2, label='Setpoint')
    line_meas, = ax1.plot([], [], 'b-', linewidth=1.5, label='Measurement')
    ax1.set_xlim(0, 20)
    ax1.set_ylim(-20, 120)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Value')
    ax1.set_title('System Response')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    line_err, = ax2.plot([], [], 'r-', linewidth=1.2)
    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlim(0, 20)
    ax2.set_ylim(-50, 100)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Error')
    ax2.set_title('Tracking Error')
    ax2.grid(True, alpha=0.3)
    
    line_out, = ax3.plot([], [], 'm-', linewidth=1.2)
    ax3.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax3.axhline(y=100, color='red', linestyle=':', alpha=0.3)
    ax3.axhline(y=-100, color='red', linestyle=':', alpha=0.3)
    ax3.set_xlim(0, 20)
    ax3.set_ylim(-120, 120)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Control Output')
    ax3.set_title('Control Signal')
    ax3.grid(True, alpha=0.3)
    
    # Create sliders
    ax_kp = plt.axes([0.15, 0.15, 0.3, 0.03])
    ax_ki = plt.axes([0.15, 0.10, 0.3, 0.03])
    ax_kd = plt.axes([0.15, 0.05, 0.3, 0.03])
    
    slider_kp = Slider(ax_kp, 'Kp', 0.0, 10.0, valinit=params.kp, color='orange')
    slider_ki = Slider(ax_ki, 'Ki', 0.0, 5.0, valinit=params.ki, color='cyan')
    slider_kd = Slider(ax_kd, 'Kd', 0.0, 3.0, valinit=params.kd, color='brown')
    
    # Create buttons
    ax_reset = plt.axes([0.6, 0.10, 0.1, 0.04])
    ax_pause = plt.axes([0.75, 0.10, 0.1, 0.04])
    
    btn_reset = Button(ax_reset, 'Reset')
    btn_pause = Button(ax_pause, 'Pause')
    
    # Metrics display
    ax_metrics = plt.axes([0.55, 0.02, 0.4, 0.06])
    ax_metrics.axis('off')
    metrics_text = ax_metrics.text(0, 0.5, '', fontsize=10, family='monospace',
                                   verticalalignment='center')
    
    def update_params(val):
        controller.set_gains(
            kp=slider_kp.val,
            ki=slider_ki.val,
            kd=slider_kd.val,
            bumpless=True
        )
    
    slider_kp.on_changed(update_params)
    slider_ki.on_changed(update_params)
    slider_kd.on_changed(update_params)
    
    def reset(event):
        nonlocal measurement
        current_idx[0] = 0
        measurement[0] = 0.0
        controller.reset()
        plant.reset()
        data['t'][:] = 0
        data['sp'][:] = 0
        data['meas'][:] = 0
        data['out'][:] = 0
        data['err'][:] = 0
    
    def toggle_pause(event):
        running[0] = not running[0]
        btn_pause.label.set_text('Resume' if not running[0] else 'Pause')
    
    btn_reset.on_clicked(reset)
    btn_pause.on_clicked(toggle_pause)
    
    def animate(frame):
        if not running[0]:
            return line_sp, line_meas, line_err, line_out
        
        idx = current_idx[0]
        
        if idx >= max_points:
            # Shift data left
            data['t'][:-1] = data['t'][1:]
            data['sp'][:-1] = data['sp'][1:]
            data['meas'][:-1] = data['meas'][1:]
            data['out'][:-1] = data['out'][1:]
            data['err'][:-1] = data['err'][1:]
            idx = max_points - 1
        
        t = idx * dt
        
        # Generate setpoint (multiple steps)
        if t < 2:
            sp = 0.0
        elif t < 8:
            sp = 50.0
        elif t < 14:
            sp = 80.0
        else:
            sp = 30.0
        
        # Run controller
        output = controller.update(sp, measurement[0], timestamp=t)
        state = controller.state
        measurement[0] = plant.update(output)
        
        # Store data
        data['t'][idx] = t
        data['sp'][idx] = sp
        data['meas'][idx] = measurement[0]
        data['out'][idx] = output
        data['err'][idx] = state.error
        
        # Update lines
        valid_idx = idx + 1
        line_sp.set_data(data['t'][:valid_idx], data['sp'][:valid_idx])
        line_meas.set_data(data['t'][:valid_idx], data['meas'][:valid_idx])
        line_err.set_data(data['t'][:valid_idx], data['err'][:valid_idx])
        line_out.set_data(data['t'][:valid_idx], data['out'][:valid_idx])
        
        # Update x-axis limits
        if t > 18:
            for ax in [ax1, ax2, ax3]:
                ax.set_xlim(t - 18, t + 2)
        
        # Update metrics
        if valid_idx > 10:
            recent_err = data['err'][max(0, valid_idx-100):valid_idx]
            mae = np.mean(np.abs(recent_err))
            metrics_text.set_text(
                f'Current Error: {state.error:+7.2f}  |  '
                f'MAE (recent): {mae:6.2f}  |  '
                f'Output: {output:+7.2f}'
            )
        
        current_idx[0] = idx + 1
        
        return line_sp, line_meas, line_err, line_out
    
    anim = FuncAnimation(fig, animate, frames=None, interval=20, blit=False)
    
    plt.show()


def run_animated_comparison():
    """
    Side-by-side animated comparison of different tunings.
    """
    print("\n" + "=" * 60)
    print("Animated Tuning Comparison")
    print("=" * 60)
    
    dt = 0.02
    
    configs = {
        'Conservative': PIDParams(kp=1.0, ki=0.3, kd=0.2, sample_time=dt),
        'Moderate': PIDParams(kp=2.0, ki=1.0, kd=0.5, sample_time=dt),
        'Aggressive': PIDParams(kp=5.0, ki=2.5, kd=1.0, sample_time=dt),
    }
    
    plants = {name: SecondOrderPlant(
        gain=1.0, natural_frequency=1.5, damping_ratio=0.5, sample_time=dt
    ) for name in configs}
    
    controllers = {name: PIDController(params) for name, params in configs.items()}
    measurements = {name: 0.0 for name in configs}
    
    max_points = 800
    data = {name: {
        't': np.zeros(max_points),
        'meas': np.zeros(max_points),
    } for name in configs}
    
    current_idx = [0]
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('Real-Time Tuning Comparison', fontsize=14, fontweight='bold')
    
    colors = {'Conservative': 'blue', 'Moderate': 'green', 'Aggressive': 'red'}
    
    line_sp, = ax.plot([], [], 'k--', linewidth=2.5, label='Setpoint')
    lines = {name: ax.plot([], [], '-', color=colors[name], linewidth=1.5, label=name)[0]
             for name in configs}
    
    ax.set_xlim(0, 16)
    ax.set_ylim(-20, 120)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Process Value', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    def animate(frame):
        idx = current_idx[0]
        
        if idx >= max_points:
            return list(lines.values()) + [line_sp]
        
        t = idx * dt
        
        # Setpoint profile
        if t < 1:
            sp = 0.0
        elif t < 6:
            sp = 50.0
        elif t < 11:
            sp = 80.0
        else:
            sp = 30.0
        
        for name in configs:
            output = controllers[name].update(sp, measurements[name], timestamp=t)
            measurements[name] = plants[name].update(output)
            data[name]['t'][idx] = t
            data[name]['meas'][idx] = measurements[name]
        
        # Update lines
        valid_idx = idx + 1
        
        # Update setpoint line
        sp_data = np.where(data['Conservative']['t'][:valid_idx] < 1, 0,
                          np.where(data['Conservative']['t'][:valid_idx] < 6, 50,
                                  np.where(data['Conservative']['t'][:valid_idx] < 11, 80, 30)))
        line_sp.set_data(data['Conservative']['t'][:valid_idx], sp_data)
        
        for name in configs:
            lines[name].set_data(data[name]['t'][:valid_idx], data[name]['meas'][:valid_idx])
        
        current_idx[0] = idx + 1
        
        return list(lines.values()) + [line_sp]
    
    anim = FuncAnimation(fig, animate, frames=max_points, interval=20, blit=True, repeat=False)
    
    plt.tight_layout()
    plt.show()


def main():
    print("=" * 60)
    print("ANIMATED PID DEMONSTRATIONS")
    print("=" * 60)
    
    print("\n1. Running Interactive Simulation...")
    print("   (Close the window to continue to next demo)")
    run_interactive_simulation()
    
    print("\n2. Running Animated Comparison...")
    run_animated_comparison()
    
    print("\nAll animated demos complete!")


if __name__ == "__main__":
    main()
