#!/usr/bin/env python3
"""
System Identification and Autotuning from CSV Data

This demo shows the complete workflow:
1. Generate synthetic experimental data (or load your own CSV)
2. Identify system transfer function from data
3. Apply analytical tuning rules
4. Optimize PID gains numerically
5. Visualize results and comparisons

This is useful when you have real experimental data from a system
and want to find optimal PID gains without knowing the system model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from pid_control.identification import (
    CSVDataReader,
    SystemIdentifier,
    ModelType
)
from pid_control.identification.autotune_from_data import AutotuneFromData
from pid_control.identification.visualizer import IdentificationVisualizer


def generate_sample_data(output_path: str = "output/sample_system_data.csv"):
    """
    Generate synthetic experimental data from a known system.
    
    This simulates collecting data from a real system with:
    - Step input
    - First-order dynamics with dead time
    - Measurement noise
    """
    print("=" * 70)
    print("GENERATING SAMPLE EXPERIMENTAL DATA")
    print("=" * 70)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    K = 2.5
    tau = 1.5
    theta = 0.3
    
    print(f"\nTrue system parameters:")
    print(f"  Gain (K): {K}")
    print(f"  Time constant (tau): {tau} s")
    print(f"  Dead time (theta): {theta} s")
    
    dt = 0.02
    duration = 10.0
    t = np.arange(0, duration, dt)
    
    u = np.zeros_like(t)
    u[t >= 1.0] = 1.0
    
    y = np.zeros_like(t)
    delay_samples = int(theta / dt)
    
    for i in range(1, len(t)):
        if i > delay_samples:
            u_delayed = u[i - delay_samples]
        else:
            u_delayed = 0.0
        
        dydt = (K * u_delayed - y[i-1]) / tau
        y[i] = y[i-1] + dydt * dt
    
    noise_level = 0.02
    y_noisy = y + np.random.normal(0, noise_level * np.std(y), len(y))
    
    setpoint = np.ones_like(t) * K
    
    with open(output_path, 'w') as f:
        f.write("timestamp,output,measurement,setpoint\n")
        for i in range(len(t)):
            f.write(f"{t[i]:.4f},{u[i]:.6f},{y_noisy[i]:.6f},{setpoint[i]:.6f}\n")
    
    print(f"\nGenerated {len(t)} data points")
    print(f"Sample time: {dt} s")
    print(f"Duration: {duration} s")
    print(f"Noise level: {noise_level * 100:.1f}% of signal std")
    print(f"\nData saved to: {output_path}")
    
    return output_path


def demo_basic_identification(csv_path: str):
    """
    Demonstrate basic system identification with automatic model selection.
    """
    print("\n" + "=" * 70)
    print("DEMO 1: BASIC SYSTEM IDENTIFICATION")
    print("=" * 70)
    
    reader = CSVDataReader(csv_path)
    data = reader.read()
    
    print(f"\nLoaded {len(data.time)} data points from CSV")
    print(f"Sample time: {data.sample_time:.4f} s")
    
    identifier = SystemIdentifier(data)
    
    print("\nIdentifying system model using optimization...")
    result = identifier.identify(
        model_type=ModelType.AUTO,
        tuning_rule='ziegler_nichols'
    )
    
    print("\n" + result.summary())
    
    print("\nGenerating visualization...")
    IdentificationVisualizer.plot_identification_result(
        result,
        data.output,
        save_path="output/identification_basic.png"
    )
    
    return result


def demo_compare_tuning_rules(csv_path: str):
    """
    Compare different analytical tuning rules.
    """
    print("\n" + "=" * 70)
    print("DEMO 2: COMPARING TUNING RULES")
    print("=" * 70)
    
    reader = CSVDataReader(csv_path)
    data = reader.read()
    identifier = SystemIdentifier(data)
    
    print("\nComparing tuning rules:")
    rules_comparison = identifier.compare_tuning_rules()
    
    print("\n" + "-" * 70)
    for rule_name, gains in rules_comparison.items():
        print(f"\n{rule_name}:")
        print(f"  Kp = {gains['kp']:.4f}")
        print(f"  Ki = {gains['ki']:.4f}")
        print(f"  Kd = {gains['kd']:.4f}")
    print("-" * 70)
    
    print("\nGenerating comparison visualization...")
    IdentificationVisualizer.plot_tuning_rules_comparison(
        rules_comparison,
        save_path="output/tuning_rules_comparison.png"
    )
    
    return rules_comparison


def demo_full_autotune(csv_path: str):
    """
    Demonstrate complete autotuning workflow with optimization.
    """
    print("\n" + "=" * 70)
    print("DEMO 3: COMPLETE AUTOTUNING WITH OPTIMIZATION")
    print("=" * 70)
    
    autotuner = AutotuneFromData(csv_path)
    
    print("\nRunning complete autotuning workflow...")
    print("This will:")
    print("  1. Identify system transfer function")
    print("  2. Apply Ziegler-Nichols tuning rule")
    print("  3. Optimize gains using differential evolution")
    
    result = autotuner.autotune(
        model_type=ModelType.AUTO,
        tuning_rule='ziegler_nichols',
        optimizer='differential_evolution',
        bounds_scale=3.0,
        max_iterations=50
    )
    
    print("\n" + result.summary())
    
    print("\nGenerating comprehensive visualization...")
    IdentificationVisualizer.plot_autotune_comparison(
        result,
        autotuner.data,
        save_path="output/autotune_complete.png"
    )
    
    return result


def demo_advanced_optimization(csv_path: str):
    """
    Demonstrate advanced optimization with different methods.
    """
    print("\n" + "=" * 70)
    print("DEMO 4: COMPARING OPTIMIZATION METHODS")
    print("=" * 70)
    
    optimizers = ['gradient_free', 'genetic', 'differential_evolution']
    results = {}
    
    for optimizer in optimizers:
        print(f"\n{'-' * 70}")
        print(f"Testing optimizer: {optimizer}")
        print(f"{'-' * 70}")
        
        autotuner = AutotuneFromData(csv_path)
        
        result = autotuner.autotune(
            model_type=ModelType.AUTO,
            tuning_rule='imc',
            optimizer=optimizer,
            bounds_scale=2.5,
            max_iterations=30
        )
        
        results[optimizer] = result
        
        print(f"\nOptimized gains ({optimizer}):")
        print(f"  Kp = {result.optimized_gains['kp']:.4f}")
        print(f"  Ki = {result.optimized_gains['ki']:.4f}")
        print(f"  Kd = {result.optimized_gains['kd']:.4f}")
        print(f"  Final cost: {result.tuning_result.cost:.4f}")
        print(f"  Improvement: {result.improvement:.2f}%")
    
    print("\n" + "=" * 70)
    print("OPTIMIZATION METHODS COMPARISON")
    print("=" * 70)
    
    for optimizer, result in results.items():
        print(f"\n{optimizer}:")
        print(f"  Cost: {result.tuning_result.cost:.4f}")
        print(f"  Improvement: {result.improvement:.2f}%")
        print(f"  Iterations: {result.tuning_result.iterations}")
    
    return results


def demo_with_real_data():
    """
    Template for using your own CSV data.
    """
    print("\n" + "=" * 70)
    print("USING YOUR OWN CSV DATA")
    print("=" * 70)
    
    print("\nTo use your own experimental data:")
    print("\n1. Prepare CSV file with columns:")
    print("   - time: timestamp (seconds)")
    print("   - input: control signal (e.g., voltage, force)")
    print("   - output: measured process variable (e.g., position, temperature)")
    print("   - setpoint: (optional) desired output value")
    
    print("\n2. Example CSV format:")
    print("   time,input,output,setpoint")
    print("   0.0,0.0,25.2,50.0")
    print("   0.01,5.2,25.3,50.0")
    print("   0.02,8.1,25.5,50.0")
    print("   ...")
    
    print("\n3. Load and autotune:")
    print("   ```python")
    print("   from pid_control.identification.autotune_from_data import AutotuneFromData")
    print("   ")
    print("   autotuner = AutotuneFromData('your_data.csv')")
    print("   result = autotuner.autotune()")
    print("   print(result.summary())")
    print("   ```")
    
    print("\n4. Required data characteristics:")
    print("   - At least 20-50 data points (more is better)")
    print("   - Input signal with variation (any type: step, ramp, random)")
    print("   - Consistent sample time")
    print("   - Output should respond to input changes")
    
    print("\n5. Optional: Specify existing PID gains used during data collection")
    print("   This helps with analysis but is not required for identification")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("SYSTEM IDENTIFICATION & AUTOTUNING DEMO")
    print("=" * 70)
    print("\nThis demo shows how to:")
    print("  • Identify system dynamics from CSV data")
    print("  • Compare analytical tuning rules")
    print("  • Optimize PID gains numerically")
    print("  • Visualize results with comprehensive plots")
    
    csv_path = generate_sample_data()
    
    input("\nPress Enter to start Demo 1: Basic Identification...")
    demo_basic_identification(csv_path)
    
    input("\nPress Enter to start Demo 2: Compare Tuning Rules...")
    demo_compare_tuning_rules(csv_path)
    
    input("\nPress Enter to start Demo 3: Complete Autotuning...")
    result = demo_full_autotune(csv_path)
    
    print("\n" + "=" * 70)
    print("FINAL RECOMMENDED PID GAINS")
    print("=" * 70)
    print(f"\nKp = {result.optimized_gains['kp']:.4f}")
    print(f"Ki = {result.optimized_gains['ki']:.4f}")
    print(f"Kd = {result.optimized_gains['kd']:.4f}")
    print(f"\nFit Quality (R²): {result.identification.fit_quality:.4f}")
    print(f"Performance Improvement: {result.improvement:.2f}%")
    
    response = input("\nRun Demo 4: Compare Optimization Methods? (y/n): ")
    if response.lower() == 'y':
        demo_advanced_optimization(csv_path)
    
    demo_with_real_data()
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    print("\nGenerated files in output/ directory:")
    print("  - sample_system_data.csv")
    print("  - identification_basic.png")
    print("  - tuning_rules_comparison.png")
    print("  - autotune_complete.png")
    print("\nYou can now use these techniques with your own CSV data!")


if __name__ == '__main__':
    main()
