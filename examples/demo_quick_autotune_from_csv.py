#!/usr/bin/env python3
"""
Quick Start: Autotune PID from CSV Data

Minimal example showing how to get optimal PID gains from experimental data.
Perfect for when you have CSV data from your system and need PID parameters.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pid_control.identification.autotune_from_data import AutotuneFromData
from pid_control.identification.visualizer import IdentificationVisualizer


def quick_autotune(csv_path: str):
    """
    Get optimal PID gains from CSV data in 3 lines of code.
    
    CSV Requirements:
    - Columns: time, input, output (required)
    - Optional: setpoint
    - At least 20-50 data points
    - Clear step response or dynamic behavior
    """
    autotuner = AutotuneFromData(csv_path)
    
    result = autotuner.autotune()
    
    print(result.summary())
    
    IdentificationVisualizer.plot_autotune_comparison(
        result,
        autotuner.data,
        save_path="output/quick_autotune_result.png"
    )
    
    return result


def main():
    """
    Example usage with your own CSV file.
    """
    print("=" * 70)
    print("QUICK AUTOTUNE FROM CSV DATA")
    print("=" * 70)
    
    print("\nThis script will:")
    print("  1. Load your CSV data")
    print("  2. Identify the system transfer function")
    print("  3. Calculate optimal PID gains")
    print("  4. Generate visualization plots")
    
    csv_path = input("\nEnter path to your CSV file (or press Enter for demo): ").strip()
    
    if not csv_path:
        print("\nGenerating demo data...")
        import numpy as np
        
        Path("output").mkdir(exist_ok=True)
        csv_path = "output/demo_data.csv"
        
        t = np.arange(0, 10, 0.02)
        u = np.zeros_like(t)
        u[t >= 1.0] = 1.0
        
        y = np.zeros_like(t)
        K, tau, theta = 2.0, 1.2, 0.2
        delay_samples = int(theta / 0.02)
        
        for i in range(1, len(t)):
            u_delayed = u[i - delay_samples] if i > delay_samples else 0.0
            dydt = (K * u_delayed - y[i-1]) / tau
            y[i] = y[i-1] + dydt * 0.02
        
        y += np.random.normal(0, 0.02, len(y))
        
        with open(csv_path, 'w') as f:
            f.write("timestamp,output,measurement,setpoint\n")
            for i in range(len(t)):
                f.write(f"{t[i]:.4f},{u[i]:.6f},{y[i]:.6f},{K:.6f}\n")
        
        print(f"Demo data saved to: {csv_path}")
    
    print(f"\nProcessing: {csv_path}")
    print("\nRunning autotuning (this may take 30-60 seconds)...\n")
    
    result = quick_autotune(csv_path)
    
    print("\n" + "=" * 70)
    print("OPTIMAL PID GAINS")
    print("=" * 70)
    print(f"\nKp = {result.optimized_gains['kp']:.4f}")
    print(f"Ki = {result.optimized_gains['ki']:.4f}")
    print(f"Kd = {result.optimized_gains['kd']:.4f}")
    print("\nUse these gains in your PID controller!")
    print("\nVisualization saved to: output/quick_autotune_result.png")


if __name__ == '__main__':
    main()
