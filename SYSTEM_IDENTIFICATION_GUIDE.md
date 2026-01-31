# System Identification and Autotuning from CSV Data

## Overview

This guide shows how to identify system dynamics from experimental data and automatically tune PID controllers. This is useful when you have real data from your system but don't know the mathematical model.

## Complete Workflow

```
CSV Data → System Identification → Analytical Tuning → Numerical Optimization → Optimal PID Gains
```

## Quick Start

### Minimal Example (3 lines)

```python
from pid_control.identification.autotune_from_data import AutotuneFromData

autotuner = AutotuneFromData('your_data.csv')
result = autotuner.autotune()
print(f"Optimal gains: Kp={result.optimized_gains['kp']:.4f}, "
      f"Ki={result.optimized_gains['ki']:.4f}, Kd={result.optimized_gains['kd']:.4f}")
```

## CSV Data Requirements

### Required Columns
- **timestamp**: Timestamp in seconds (must be monotonically increasing)
- **output**: Control signal from PID controller (e.g., voltage, force, PWM duty cycle)
- **measurement**: Measured process variable (e.g., position, temperature, speed)

### Optional Columns
- **setpoint**: Desired output value
- **error**: Tracking error (output - setpoint)

### Example CSV Format

```csv
timestamp,output,measurement,setpoint
0.00,0.0,25.2,50.0
0.01,5.2,25.3,50.0
0.02,8.1,25.5,50.0
0.03,10.5,25.8,50.0
...
```

**Note:** These column names match the output from `CSVLogger` in the PID library.

### Data Collection Guidelines

1. **Sample Rate**: 10-100 Hz typical (faster for fast systems)
2. **Duration**: Long enough to see system response (3-5 time constants)
3. **Input Signal**: Step input is ideal, but any dynamic input works
4. **Data Points**: Minimum 20-50, recommended 100-500
5. **Steady State**: System should reach or approach steady state
6. **Noise**: Some noise is okay, the algorithms are robust

## Step-by-Step Usage

### Step 1: Load CSV Data

```python
from pid_control.identification import CSVDataReader

reader = CSVDataReader('experimental_data.csv')
data = reader.read(
    time_col='timestamp',
    input_col='output',
    output_col='measurement',
    setpoint_col='setpoint'  # optional
)

print(f"Loaded {len(data.time)} samples")
print(f"Sample time: {data.sample_time:.4f} s")
```

### Step 2: Identify System Transfer Function

```python
from pid_control.identification import SystemIdentifier, ModelType

identifier = SystemIdentifier(data)

# Identify First Order Plus Dead Time (FOPDT) model
result = identifier.identify(
    model_type=ModelType.FOPDT,
    method='step_response',  # or 'optimization'
    tuning_rule='ziegler_nichols'
)

print(result.summary())
print(f"Transfer function: K={result.model.K:.3f}, "
      f"tau={result.model.tau:.3f}, theta={result.model.theta:.3f}")
```

### Step 3: Compare Tuning Rules

```python
# Compare different analytical tuning methods
rules_comparison = identifier.compare_tuning_rules()

for rule_name, gains in rules_comparison.items():
    print(f"\n{rule_name}:")
    print(f"  Kp={gains['kp']:.4f}, Ki={gains['ki']:.4f}, Kd={gains['kd']:.4f}")
```

### Step 4: Optimize with Numerical Methods

```python
from pid_control.identification.autotune_from_data import AutotuneFromData

autotuner = AutotuneFromData('experimental_data.csv')

result = autotuner.autotune(
    model_type=ModelType.FOPDT,
    identification_method='step_response',
    tuning_rule='ziegler_nichols',
    optimizer='differential_evolution',
    bounds_scale=2.0,
    max_iterations=50
)

print(result.summary())
```

### Step 5: Visualize Results

```python
from pid_control.identification.visualizer import IdentificationVisualizer

# Plot identification results
IdentificationVisualizer.plot_identification_result(
    result.identification,
    autotuner.data.output,
    save_path='identification_result.png'
)

# Plot complete autotuning comparison
IdentificationVisualizer.plot_autotune_comparison(
    result,
    autotuner.data,
    save_path='autotune_comparison.png'
)
```

## Model Types

### FOPDT (First Order Plus Dead Time)
```
G(s) = K * exp(-θ*s) / (τ*s + 1)
```
- **K**: Steady-state gain
- **τ**: Time constant (how fast system responds)
- **θ**: Dead time (delay before response starts)

**Best for**: Most industrial processes, temperature control, level control

### SOPDT (Second Order Plus Dead Time)
```
G(s) = K * exp(-θ*s) / ((τ1*s + 1)(τ2*s + 1))
```
- **K**: Steady-state gain
- **τ1, τ2**: Time constants
- **θ**: Dead time

**Best for**: Systems with oscillatory behavior, mechanical systems

## Tuning Rules

### Ziegler-Nichols
Classic tuning rule, good starting point. Can be aggressive.

### Cohen-Coon
Better for systems with significant dead time.

### IMC (Internal Model Control)
Conservative tuning, good stability margins.

### Lambda Tuning
Adjustable aggressiveness, good for various applications.

### Aggressive
Fast response, may have overshoot.

### Conservative
Slow but very stable, no overshoot.

## Optimization Methods

### Differential Evolution (Recommended)
- Robust global optimizer
- Good for PID tuning
- Handles noisy cost functions well

```python
optimizer='differential_evolution'
```

### Genetic Algorithm
- Global search
- Good for avoiding local minima
- Slower but thorough

```python
optimizer='genetic'
```

### Gradient-Free (Nelder-Mead)
- Fast local optimization
- Good when close to optimum
- Can get stuck in local minima

```python
optimizer='gradient_free'
```

### Bayesian Optimization
- Sample-efficient
- Good for expensive evaluations
- Balances exploration/exploitation

```python
optimizer='bayesian'
```

## Advanced Usage

### Custom Cost Function

```python
def my_cost_function(kp, ki, kd):
    """Custom cost function for specific requirements."""
    # Simulate closed-loop response
    # Calculate performance metrics
    # Return cost (lower is better)
    return cost

result = autotuner.autotune(
    cost_function=my_cost_function,
    max_iterations=100
)
```

### Adjust Search Bounds

```python
result = autotuner.autotune(
    bounds_scale=3.0,  # Search ±3x around initial guess
    max_iterations=100
)
```

### With Existing PID Parameters

If you collected data with a PID controller already running:

```python
data = reader.read_with_pid_params(
    kp_value=1.5,
    ki_value=0.2,
    kd_value=0.1
)
```

## Complete Example

```python
from pid_control.identification.autotune_from_data import AutotuneFromData
from pid_control.identification.visualizer import IdentificationVisualizer

# 1. Load data
autotuner = AutotuneFromData('my_system_data.csv')

# 2. Run complete autotuning
result = autotuner.autotune(
    tuning_rule='ziegler_nichols',
    optimizer='differential_evolution',
    max_iterations=50
)

# 3. Get optimal gains
print(f"\nOptimal PID Gains:")
print(f"Kp = {result.optimized_gains['kp']:.4f}")
print(f"Ki = {result.optimized_gains['ki']:.4f}")
print(f"Kd = {result.optimized_gains['kd']:.4f}")

# 4. Visualize
IdentificationVisualizer.plot_autotune_comparison(
    result, autotuner.data, save_path='results.png'
)

# 5. Use gains in your controller
from pid_control import PIDController, PIDParams

params = PIDParams(
    kp=result.optimized_gains['kp'],
    ki=result.optimized_gains['ki'],
    kd=result.optimized_gains['kd'],
    sample_time=autotuner.data.sample_time
)
controller = PIDController(params)
```

## Troubleshooting

### Poor Fit Quality (R² < 0.7)

**Causes:**
- Insufficient data
- High noise levels
- Non-linear system behavior
- Wrong model type

**Solutions:**
- Collect more data
- Use longer time window
- Try SOPDT model
- Filter data before identification

### Unstable Gains

**Causes:**
- Poor system identification
- Aggressive tuning rule
- Wrong model parameters

**Solutions:**
- Use conservative tuning rule
- Increase bounds_scale
- Try different identification method
- Verify data quality

### Optimization Not Converging

**Causes:**
- Bad initial guess
- Narrow search bounds
- Complex cost landscape

**Solutions:**
- Use different optimizer
- Increase max_iterations
- Adjust bounds_scale
- Try different tuning rule for initial guess

## Demo Scripts

Run the included examples:

```bash
# Complete demonstration
python examples/demo_system_identification.py

# Quick start example
python examples/demo_quick_autotune_from_csv.py
```

## API Reference

### AutotuneFromData

Main class for complete workflow.

**Methods:**
- `autotune()`: Run complete identification and optimization
- `compare_tuning_rules()`: Compare analytical tuning rules

### SystemIdentifier

System identification from experimental data.

**Methods:**
- `identify()`: Identify transfer function model
- `compare_tuning_rules()`: Compare tuning methods

### CSVDataReader

Read experimental data from CSV files.

**Methods:**
- `read()`: Load CSV data
- `read_with_pid_params()`: Load with PID parameters
- `detect_step_response()`: Find step response region

### IdentificationVisualizer

Visualization tools.

**Methods:**
- `plot_identification_result()`: Plot model fit
- `plot_autotune_comparison()`: Plot optimization results
- `plot_tuning_rules_comparison()`: Compare tuning rules

## Performance Tips

1. **Data Quality**: Clean, consistent data is crucial
2. **Sample Rate**: Match your system dynamics (10-100x faster than time constant)
3. **Duration**: Capture full response (3-5 time constants minimum)
4. **Initial Guess**: Better initial guess = faster optimization
5. **Bounds**: Reasonable bounds prevent unrealistic solutions

## References

- Ziegler-Nichols tuning (1942)
- Cohen-Coon tuning (1953)
- Internal Model Control (IMC)
- System Identification Theory and Practice

## Support

For issues or questions, see the main README.md or examples directory.
