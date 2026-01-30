#!/usr/bin/env python3
"""
Spectacular PID Simulation Demonstrations

Mind-blowing visualizations showcasing:
- 3D phase space trajectories
- Real-time animated control
- Multi-plant comparison
- Robustness analysis
- Interactive parameter exploration
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

from pid_control.core.pid_controller import PIDController
from pid_control.core.pid_params import PIDParams, AntiWindupMethod
from pid_control.plants.first_order import FirstOrderPlant
from pid_control.plants.second_order import SecondOrderPlant
from pid_control.plants.nonlinear import NonlinearPlant, FrictionPlant
from pid_control.plants.delay_plant import FOPDTPlant
from pid_control.simulation.simulator import Simulator
from pid_control.simulation.scenarios import SimulationScenario, SetpointType, ScenarioLibrary


def demo_3d_phase_space():
    """
    Spectacular 3D phase space visualization.
    Shows error, error derivative, and integral evolving over time.
    """
    print("\n" + "=" * 60)
    print("3D Phase Space Trajectory")
    print("=" * 60)
    
    plant = SecondOrderPlant(
        gain=1.0,
        natural_frequency=1.5,
        damping_ratio=0.3,  # Underdamped for interesting trajectory
        sample_time=0.01
    )
    
    params = PIDParams(
        kp=4.0, ki=2.0, kd=1.5,
        sample_time=0.01,
        output_min=-50, output_max=50
    )
    
    controller = PIDController(params)
    
    # Run simulation
    duration = 25.0
    dt = 0.01
    n_steps = int(duration / dt)
    
    errors = []
    error_rates = []
    integrals = []
    times = []
    
    measurement = 0.0
    prev_error = 0.0
    
    for i in range(n_steps):
        t = i * dt
        setpoint = 100.0 if t > 1.0 else 0.0
        
        output = controller.update(setpoint, measurement, timestamp=t)
        state = controller.state
        measurement = plant.update(output)
        
        error = state.error
        error_rate = (error - prev_error) / dt if i > 0 else 0
        
        errors.append(error)
        error_rates.append(error_rate)
        integrals.append(state.integral_accumulator)
        times.append(t)
        
        prev_error = error
    
    errors = np.array(errors)
    error_rates = np.array(error_rates)
    integrals = np.array(integrals)
    times = np.array(times)
    
    # Create stunning 3D plot
    fig = plt.figure(figsize=(16, 12))
    
    # Main 3D trajectory
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Color by time
    colors = cm.viridis(times / times.max())
    
    for i in range(len(errors) - 1):
        ax1.plot(errors[i:i+2], error_rates[i:i+2], integrals[i:i+2],
                color=colors[i], linewidth=1.5)
    
    # Mark start and end
    ax1.scatter([errors[0]], [error_rates[0]], [integrals[0]], 
               color='green', s=100, marker='o', label='Start')
    ax1.scatter([errors[-1]], [error_rates[-1]], [integrals[-1]], 
               color='red', s=100, marker='*', label='End (Equilibrium)')
    
    ax1.set_xlabel('Error', fontsize=11)
    ax1.set_ylabel('Error Rate', fontsize=11)
    ax1.set_zlabel('Integral', fontsize=11)
    ax1.set_title('3D Phase Space Trajectory', fontsize=14, fontweight='bold')
    ax1.legend()
    
    # 2D projections
    ax2 = fig.add_subplot(222)
    scatter = ax2.scatter(errors, error_rates, c=times, cmap='viridis', s=1, alpha=0.7)
    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Error')
    ax2.set_ylabel('Error Rate')
    ax2.set_title('Phase Portrait (Error vs Error Rate)')
    plt.colorbar(scatter, ax=ax2, label='Time (s)')
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(223)
    scatter = ax3.scatter(errors, integrals, c=times, cmap='viridis', s=1, alpha=0.7)
    ax3.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax3.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Error')
    ax3.set_ylabel('Integral')
    ax3.set_title('Error vs Integral Accumulator')
    plt.colorbar(scatter, ax=ax3, label='Time (s)')
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(224)
    scatter = ax4.scatter(error_rates, integrals, c=times, cmap='viridis', s=1, alpha=0.7)
    ax4.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax4.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax4.set_xlabel('Error Rate')
    ax4.set_ylabel('Integral')
    ax4.set_title('Error Rate vs Integral')
    plt.colorbar(scatter, ax=ax4, label='Time (s)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    print("Phase space shows the controller's journey to equilibrium!")


def demo_gain_surface():
    """
    3D surface plot showing performance vs PID gains.
    """
    print("\n" + "=" * 60)
    print("PID Gain Performance Surface")
    print("=" * 60)
    
    plant = FirstOrderPlant(gain=2.0, time_constant=2.0, sample_time=0.05)
    
    # Grid of Kp and Ki values
    kp_range = np.linspace(0.5, 5.0, 20)
    ki_range = np.linspace(0.1, 3.0, 20)
    
    Kp, Ki = np.meshgrid(kp_range, ki_range)
    IAE = np.zeros_like(Kp)
    Overshoot = np.zeros_like(Kp)
    
    print("Computing performance surface (this may take a moment)...")
    
    for i in range(len(kp_range)):
        for j in range(len(ki_range)):
            params = PIDParams(
                kp=kp_range[i], ki=ki_range[j], kd=0.3,
                sample_time=0.05, output_min=-100, output_max=100
            )
            
            controller = PIDController(params)
            plant.reset()
            
            errors = []
            measurements = []
            measurement = 0.0
            
            for k in range(400):  # 20 second simulation
                sp = 50.0 if k > 20 else 0.0
                output = controller.update(sp, measurement)
                measurement = plant.update(output)
                errors.append(abs(sp - measurement))
                measurements.append(measurement)
            
            IAE[j, i] = sum(errors) * 0.05
            
            # Calculate overshoot
            final_sp = 50.0
            max_val = max(measurements)
            if max_val > final_sp:
                Overshoot[j, i] = (max_val - final_sp) / final_sp * 100
            else:
                Overshoot[j, i] = 0
    
    # Create surface plots
    fig = plt.figure(figsize=(16, 6))
    
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(Kp, Ki, IAE, cmap='coolwarm', alpha=0.8)
    ax1.set_xlabel('Kp')
    ax1.set_ylabel('Ki')
    ax1.set_zlabel('IAE')
    ax1.set_title('Integral Absolute Error Surface', fontsize=14, fontweight='bold')
    fig.colorbar(surf1, ax=ax1, shrink=0.5)
    
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(Kp, Ki, Overshoot, cmap='RdYlGn_r', alpha=0.8)
    ax2.set_xlabel('Kp')
    ax2.set_ylabel('Ki')
    ax2.set_zlabel('Overshoot %')
    ax2.set_title('Overshoot Surface', fontsize=14, fontweight='bold')
    fig.colorbar(surf2, ax=ax2, shrink=0.5)
    
    plt.tight_layout()
    print("Lower IAE = better tracking. Lower overshoot = more stable.")


def demo_multi_plant_battle():
    """
    Dramatic comparison of PID controlling different plant types.
    """
    print("\n" + "=" * 60)
    print("Multi-Plant Control Battle")
    print("=" * 60)
    
    plants = {
        'First Order': FirstOrderPlant(gain=2.0, time_constant=2.0, sample_time=0.01),
        'Second Order Underdamped': SecondOrderPlant(gain=1.0, natural_frequency=1.5, damping_ratio=0.3, sample_time=0.01),
        'Second Order Overdamped': SecondOrderPlant(gain=1.0, natural_frequency=1.0, damping_ratio=1.5, sample_time=0.01),
        'With Dead Time': FOPDTPlant(gain=1.5, time_constant=2.0, dead_time=0.5, sample_time=0.01),
        'Nonlinear': NonlinearPlant(gain=2.0, time_constant=1.5, saturation_limits=(-30, 30), dead_zone=0.5, sample_time=0.01),
    }
    
    # Aggressive PID that will behave differently on each plant
    params = PIDParams(
        kp=3.0, ki=1.5, kd=0.8,
        sample_time=0.01,
        output_min=-50, output_max=50,
        anti_windup=AntiWindupMethod.BACK_CALCULATION
    )
    
    duration = 25.0
    dt = 0.01
    n_steps = int(duration / dt)
    
    results = {}
    
    for name, plant in plants.items():
        controller = PIDController(params)
        plant.reset()
        
        timestamps = []
        setpoints = []
        measurements = []
        outputs = []
        
        measurement = 0.0
        
        for i in range(n_steps):
            t = i * dt
            # Challenging setpoint profile
            if t < 2:
                sp = 0.0
            elif t < 10:
                sp = 50.0
            elif t < 15:
                sp = 80.0
            else:
                sp = 30.0
            
            output = controller.update(sp, measurement, timestamp=t)
            measurement = plant.update(output)
            
            timestamps.append(t)
            setpoints.append(sp)
            measurements.append(measurement)
            outputs.append(output)
        
        results[name] = {
            't': np.array(timestamps),
            'sp': np.array(setpoints),
            'meas': np.array(measurements),
            'out': np.array(outputs)
        }
    
    # Create dramatic multi-panel plot
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(plants)))
    
    # Main comparison
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.plot(results['First Order']['t'], results['First Order']['sp'],
                'k--', linewidth=2.5, label='Setpoint', zorder=10)
    
    for (name, data), color in zip(results.items(), colors):
        ax_main.plot(data['t'], data['meas'], '-', color=color, 
                    linewidth=1.5, label=name, alpha=0.8)
    
    ax_main.set_ylabel('Process Value', fontsize=12)
    ax_main.set_title('PID Controller vs Multiple Plant Types', fontsize=16, fontweight='bold')
    ax_main.legend(loc='upper right', ncol=2)
    ax_main.grid(True, alpha=0.3)
    ax_main.set_xlim([0, duration])
    
    # Error comparison
    ax_err = fig.add_subplot(gs[1, 0])
    for (name, data), color in zip(results.items(), colors):
        error = data['sp'] - data['meas']
        ax_err.plot(data['t'], error, '-', color=color, linewidth=1, alpha=0.8)
    ax_err.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax_err.set_xlabel('Time (s)')
    ax_err.set_ylabel('Error')
    ax_err.set_title('Tracking Error by Plant Type')
    ax_err.grid(True, alpha=0.3)
    
    # Control effort comparison
    ax_ctrl = fig.add_subplot(gs[1, 1])
    for (name, data), color in zip(results.items(), colors):
        ax_ctrl.plot(data['t'], data['out'], '-', color=color, linewidth=1, alpha=0.8)
    ax_ctrl.set_xlabel('Time (s)')
    ax_ctrl.set_ylabel('Control Output')
    ax_ctrl.set_title('Control Effort by Plant Type')
    ax_ctrl.grid(True, alpha=0.3)
    
    # Performance metrics bar chart
    ax_bar = fig.add_subplot(gs[2, :])
    
    metrics_iae = []
    metrics_os = []
    names = []
    
    for name, data in results.items():
        error = np.abs(data['sp'] - data['meas'])
        iae = np.sum(error) * dt
        metrics_iae.append(iae)
        
        # Overshoot at first step
        mask = (data['t'] > 2) & (data['t'] < 10)
        if np.any(mask):
            max_val = np.max(data['meas'][mask])
            os = max(0, (max_val - 50) / 50 * 100)
        else:
            os = 0
        metrics_os.append(os)
        names.append(name)
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax_bar.bar(x - width/2, metrics_iae, width, label='IAE', color='steelblue')
    ax_bar2 = ax_bar.twinx()
    bars2 = ax_bar2.bar(x + width/2, metrics_os, width, label='Overshoot %', color='coral')
    
    ax_bar.set_xlabel('Plant Type')
    ax_bar.set_ylabel('IAE', color='steelblue')
    ax_bar2.set_ylabel('Overshoot %', color='coral')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(names, rotation=15, ha='right')
    ax_bar.set_title('Performance Metrics Comparison')
    
    # Add legend
    ax_bar.legend(loc='upper left')
    ax_bar2.legend(loc='upper right')
    
    plt.tight_layout()
    print("Same PID, different plants - watch how behavior varies!")


def demo_disturbance_rejection():
    """
    Dramatic visualization of disturbance rejection.
    """
    print("\n" + "=" * 60)
    print("Disturbance Rejection Showcase")
    print("=" * 60)
    
    plant = SecondOrderPlant(
        gain=1.0, natural_frequency=1.5, damping_ratio=0.7, sample_time=0.01
    )
    
    params = PIDParams(
        kp=5.0, ki=3.0, kd=1.5,
        sample_time=0.01,
        output_min=-100, output_max=100
    )
    
    controller = PIDController(params)
    
    duration = 40.0
    dt = 0.01
    n_steps = int(duration / dt)
    
    timestamps = []
    setpoints = []
    measurements = []
    outputs = []
    disturbances = []
    
    measurement = 0.0
    
    for i in range(n_steps):
        t = i * dt
        sp = 50.0 if t > 1.0 else 0.0
        
        # Complex disturbance pattern
        disturbance = 0.0
        if 10 < t < 12:  # Step disturbance
            disturbance = 20.0
        elif 20 < t < 25:  # Ramp disturbance
            disturbance = 5.0 * (t - 20)
        elif 30 < t < 35:  # Sinusoidal disturbance
            disturbance = 15.0 * np.sin(2 * np.pi * 0.5 * (t - 30))
        
        plant.set_disturbance(disturbance)
        
        output = controller.update(sp, measurement, timestamp=t)
        measurement = plant.update(output)
        
        timestamps.append(t)
        setpoints.append(sp)
        measurements.append(measurement)
        outputs.append(output)
        disturbances.append(disturbance)
    
    timestamps = np.array(timestamps)
    setpoints = np.array(setpoints)
    measurements = np.array(measurements)
    outputs = np.array(outputs)
    disturbances = np.array(disturbances)
    
    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Response with disturbance regions highlighted
    ax1 = axes[0]
    ax1.fill_between(timestamps, 0, 1, where=(timestamps > 10) & (timestamps < 12),
                    transform=ax1.get_xaxis_transform(), alpha=0.2, color='red', label='Step Disturbance')
    ax1.fill_between(timestamps, 0, 1, where=(timestamps > 20) & (timestamps < 25),
                    transform=ax1.get_xaxis_transform(), alpha=0.2, color='orange', label='Ramp Disturbance')
    ax1.fill_between(timestamps, 0, 1, where=(timestamps > 30) & (timestamps < 35),
                    transform=ax1.get_xaxis_transform(), alpha=0.2, color='purple', label='Sine Disturbance')
    
    ax1.plot(timestamps, setpoints, 'g--', linewidth=2.5, label='Setpoint')
    ax1.plot(timestamps, measurements, 'b-', linewidth=1.5, label='Measurement')
    ax1.set_ylabel('Process Value', fontsize=12)
    ax1.set_title('Disturbance Rejection Demonstration', fontsize=16, fontweight='bold')
    ax1.legend(loc='upper right', ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Disturbance signal
    ax2 = axes[1]
    ax2.fill_between(timestamps, 0, disturbances, alpha=0.5, color='red')
    ax2.plot(timestamps, disturbances, 'r-', linewidth=1.5)
    ax2.set_ylabel('Disturbance', fontsize=12)
    ax2.set_title('Applied Disturbance')
    ax2.grid(True, alpha=0.3)
    
    # Control response
    ax3 = axes[2]
    ax3.plot(timestamps, outputs, 'm-', linewidth=1.5)
    ax3.fill_between(timestamps, 0, outputs, alpha=0.3, color='purple')
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Control Output', fontsize=12)
    ax3.set_title('Controller Response to Disturbances')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    print("Watch how the controller fights back against various disturbances!")


def demo_robustness_analysis():
    """
    Spectacular robustness visualization across parameter variations.
    """
    print("\n" + "=" * 60)
    print("Robustness Analysis Visualization")
    print("=" * 60)
    
    # Nominal plant
    nominal_gain = 2.0
    nominal_tau = 2.0
    
    # Test PID robustness against plant variations
    params = PIDParams(
        kp=2.0, ki=1.0, kd=0.5,
        sample_time=0.01,
        output_min=-100, output_max=100
    )
    
    # Vary plant gain and time constant
    gain_variations = [0.5, 0.75, 1.0, 1.25, 1.5]  # Multipliers
    tau_variations = [0.5, 0.75, 1.0, 1.25, 1.5]
    
    duration = 15.0
    dt = 0.01
    n_steps = int(duration / dt)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Gain variations
    ax1 = axes[0, 0]
    colors_gain = plt.cm.RdYlBu(np.linspace(0, 1, len(gain_variations)))
    
    for mult, color in zip(gain_variations, colors_gain):
        plant = FirstOrderPlant(
            gain=nominal_gain * mult,
            time_constant=nominal_tau,
            sample_time=dt
        )
        controller = PIDController(params)
        
        measurements = []
        measurement = 0.0
        
        for i in range(n_steps):
            t = i * dt
            sp = 50.0 if t > 1.0 else 0.0
            output = controller.update(sp, measurement)
            measurement = plant.update(output)
            measurements.append(measurement)
        
        times = np.arange(n_steps) * dt
        label = f'K×{mult:.2f}' if mult != 1.0 else 'Nominal'
        lw = 2.5 if mult == 1.0 else 1.2
        ax1.plot(times, measurements, color=color, linewidth=lw, label=label)
    
    ax1.plot(times, [50.0 if t > 1.0 else 0.0 for t in times], 'k--', linewidth=2, label='Setpoint')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Measurement')
    ax1.set_title('Robustness to Plant Gain Variation', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Time constant variations
    ax2 = axes[0, 1]
    colors_tau = plt.cm.PuOr(np.linspace(0, 1, len(tau_variations)))
    
    for mult, color in zip(tau_variations, colors_tau):
        plant = FirstOrderPlant(
            gain=nominal_gain,
            time_constant=nominal_tau * mult,
            sample_time=dt
        )
        controller = PIDController(params)
        
        measurements = []
        measurement = 0.0
        
        for i in range(n_steps):
            t = i * dt
            sp = 50.0 if t > 1.0 else 0.0
            output = controller.update(sp, measurement)
            measurement = plant.update(output)
            measurements.append(measurement)
        
        label = f'τ×{mult:.2f}' if mult != 1.0 else 'Nominal'
        lw = 2.5 if mult == 1.0 else 1.2
        ax2.plot(times, measurements, color=color, linewidth=lw, label=label)
    
    ax2.plot(times, [50.0 if t > 1.0 else 0.0 for t in times], 'k--', linewidth=2, label='Setpoint')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Measurement')
    ax2.set_title('Robustness to Time Constant Variation', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Heatmap of IAE for gain and tau variations
    ax3 = axes[1, 0]
    
    iae_matrix = np.zeros((len(tau_variations), len(gain_variations)))
    
    for i, tau_mult in enumerate(tau_variations):
        for j, gain_mult in enumerate(gain_variations):
            plant = FirstOrderPlant(
                gain=nominal_gain * gain_mult,
                time_constant=nominal_tau * tau_mult,
                sample_time=dt
            )
            controller = PIDController(params)
            
            total_error = 0.0
            measurement = 0.0
            
            for k in range(n_steps):
                t = k * dt
                sp = 50.0 if t > 1.0 else 0.0
                output = controller.update(sp, measurement)
                measurement = plant.update(output)
                total_error += abs(sp - measurement) * dt
            
            iae_matrix[i, j] = total_error
    
    im = ax3.imshow(iae_matrix, cmap='YlOrRd', aspect='auto',
                   extent=[gain_variations[0], gain_variations[-1],
                          tau_variations[-1], tau_variations[0]])
    ax3.set_xlabel('Gain Multiplier')
    ax3.set_ylabel('Time Constant Multiplier')
    ax3.set_title('IAE Sensitivity Map', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax3, label='IAE')
    
    # Mark nominal point
    ax3.plot(1.0, 1.0, 'w*', markersize=15, markeredgecolor='black')
    
    # Plot 4: Overshoot sensitivity
    ax4 = axes[1, 1]
    
    os_matrix = np.zeros((len(tau_variations), len(gain_variations)))
    
    for i, tau_mult in enumerate(tau_variations):
        for j, gain_mult in enumerate(gain_variations):
            plant = FirstOrderPlant(
                gain=nominal_gain * gain_mult,
                time_constant=nominal_tau * tau_mult,
                sample_time=dt
            )
            controller = PIDController(params)
            
            measurements = []
            measurement = 0.0
            
            for k in range(n_steps):
                t = k * dt
                sp = 50.0 if t > 1.0 else 0.0
                output = controller.update(sp, measurement)
                measurement = plant.update(output)
                measurements.append(measurement)
            
            max_val = max(measurements)
            os_matrix[i, j] = max(0, (max_val - 50) / 50 * 100)
    
    im2 = ax4.imshow(os_matrix, cmap='RdYlGn_r', aspect='auto',
                    extent=[gain_variations[0], gain_variations[-1],
                           tau_variations[-1], tau_variations[0]])
    ax4.set_xlabel('Gain Multiplier')
    ax4.set_ylabel('Time Constant Multiplier')
    ax4.set_title('Overshoot Sensitivity Map', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=ax4, label='Overshoot %')
    ax4.plot(1.0, 1.0, 'w*', markersize=15, markeredgecolor='black')
    
    plt.suptitle('PID Robustness Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    print("White star marks nominal plant. Colors show performance degradation.")


def main():
    print("=" * 60)
    print("SPECTACULAR PID SIMULATIONS")
    print("Mind-Blowing Visualizations of Control Theory")
    print("=" * 60)
    
    demo_3d_phase_space()
    demo_gain_surface()
    demo_multi_plant_battle()
    demo_disturbance_rejection()
    demo_robustness_analysis()
    
    print("\n" + "=" * 60)
    print("All spectacular demos complete!")
    print("Close all plot windows to exit.")
    print("=" * 60)
    
    plt.show()


if __name__ == "__main__":
    main()
