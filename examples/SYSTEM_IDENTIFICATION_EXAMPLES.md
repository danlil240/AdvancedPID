# System Identification Examples

This directory contains examples demonstrating the complete workflow for identifying system dynamics from CSV data and obtaining optimal PID gains.

## Available Examples

### 1. Quick Start: `demo_quick_autotune_from_csv.py`

**Purpose:** Minimal example to get optimal PID gains from CSV data in 3 lines of code.

**Usage:**
```bash
python examples/demo_quick_autotune_from_csv.py
```

**What it does:**
- Loads CSV data (or generates demo data)
- Identifies system transfer function
- Applies analytical tuning + numerical optimization
- Outputs optimal PID gains
- Generates visualization

**Best for:** Quick results when you just need PID gains

---

### 2. Complete Demo: `demo_system_identification.py`

**Purpose:** Comprehensive demonstration of all system identification features.

**Usage:**
```bash
python examples/demo_system_identification.py
```

**What it includes:**

#### Demo 1: Basic Identification
- Load CSV data
- Identify FOPDT model
- Apply Ziegler-Nichols tuning
- Visualize model fit

#### Demo 2: Compare Tuning Rules
- Compare 6 different tuning methods
- Ziegler-Nichols, Cohen-Coon, IMC, Lambda, Aggressive, Conservative
- Visualize gain differences

#### Demo 3: Complete Autotuning
- Full workflow: CSV → System ID → Analytical → Optimization
- Uses differential evolution optimizer
- Shows performance improvement
- Comprehensive visualization

#### Demo 4: Compare Optimizers
- Test multiple optimization methods
- Gradient-free, Genetic, Differential Evolution
- Compare convergence and results

**Best for:** Learning all features and understanding the workflow

---

## Workflow Overview

```
┌─────────────────┐
│   CSV Data      │
│ (time, input,   │
│  output)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ System          │
│ Identification  │
│ (Transfer Func) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Analytical      │
│ Tuning Rules    │
│ (Initial Gains) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Numerical       │
│ Optimization    │
│ (Refine Gains)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Optimal PID     │
│ Gains + Plots   │
└─────────────────┘
```

## Using Your Own Data

### Step 1: Prepare CSV File

Create a CSV file with your experimental data:

```csv
timestamp,output,measurement,setpoint
0.00,0.0,25.2,50.0
0.01,5.2,25.3,50.0
0.02,8.1,25.5,50.0
...
```

**Requirements:**
- `timestamp`: Monotonically increasing timestamps (seconds)
- `output`: Control signal from PID controller (actuator command)
- `measurement`: Measured process variable
- `setpoint`: (optional) Desired output value

**Note:** These column names match the output from `CSVLogger` in the PID library.

### Step 2: Run Quick Autotune

```python
from pid_control.identification.autotune_from_data import AutotuneFromData

autotuner = AutotuneFromData('your_data.csv')
result = autotuner.autotune()

print(f"Kp = {result.optimized_gains['kp']:.4f}")
print(f"Ki = {result.optimized_gains['ki']:.4f}")
print(f"Kd = {result.optimized_gains['kd']:.4f}")
```

### Step 3: Use Gains in Your Controller

```python
from pid_control import PIDController, PIDParams

params = PIDParams(
    kp=result.optimized_gains['kp'],
    ki=result.optimized_gains['ki'],
    kd=result.optimized_gains['kd'],
    sample_time=autotuner.data.sample_time
)

controller = PIDController(params)

# Use in your control loop
output = controller.update(setpoint, measurement)
```

## Customization Options

### Choose Model Type

```python
from pid_control.identification import ModelType

result = autotuner.autotune(
    model_type=ModelType.FOPDT,  # or ModelType.SOPDT
)
```

### Choose Tuning Rule

```python
result = autotuner.autotune(
    tuning_rule='ziegler_nichols',  # or 'cohen_coon', 'imc', 'lambda_tuning'
)
```

### Choose Optimizer

```python
result = autotuner.autotune(
    optimizer='differential_evolution',  # or 'genetic', 'bayesian', 'gradient_free'
)
```

### Adjust Search Bounds

```python
result = autotuner.autotune(
    bounds_scale=3.0,  # Search ±3x around initial guess
    max_iterations=100
)
```

## Output Files

The examples generate files in the `output/` directory:

- `sample_system_data.csv` - Generated demo data
- `identification_basic.png` - Model fit visualization
- `tuning_rules_comparison.png` - Comparison of tuning methods
- `autotune_complete.png` - Complete workflow results
- `quick_autotune_result.png` - Quick start results

## Understanding the Results

### Identification Quality

**R² (Coefficient of Determination):**
- R² > 0.95: Excellent fit
- R² > 0.85: Good fit
- R² > 0.70: Acceptable fit
- R² < 0.70: Poor fit (collect better data)

### Transfer Function Parameters

**FOPDT Model:** G(s) = K·e^(-θs) / (τs + 1)

- **K (Gain):** Output change per unit input change
- **τ (Time Constant):** Time to reach 63.2% of final value
- **θ (Dead Time):** Delay before response begins

### PID Gains

- **Kp:** Proportional gain (immediate response)
- **Ki:** Integral gain (eliminate steady-state error)
- **Kd:** Derivative gain (damping, reduce overshoot)

### Performance Improvement

Shows percentage improvement from analytical tuning to optimized gains:
- Positive: Optimization improved performance
- Negative: Initial analytical gains were already good

## Troubleshooting

### Issue: Low R² (< 0.7)

**Solutions:**
1. Collect more data (longer duration)
2. Ensure clear step response in data
3. Try optimization identification method
4. Check for data quality issues

### Issue: Unrealistic Gains

**Solutions:**
1. Verify data quality
2. Try different tuning rule
3. Adjust bounds_scale
4. Use conservative tuning rule first

### Issue: Optimization Not Converging

**Solutions:**
1. Increase max_iterations
2. Try different optimizer
3. Use wider bounds_scale
4. Check initial analytical gains quality

## Advanced Usage

### Compare Multiple Tuning Rules

```python
rules_comparison = autotuner.compare_tuning_rules()

for rule, gains in rules_comparison.items():
    print(f"{rule}: Kp={gains['kp']:.3f}, Ki={gains['ki']:.3f}, Kd={gains['kd']:.3f}")
```

### Custom Cost Function

```python
def my_cost_function(kp, ki, kd):
    # Simulate system with these gains
    # Calculate performance metric
    return cost

result = autotuner.autotune(cost_function=my_cost_function)
```

### Visualize Results

```python
from pid_control.identification.visualizer import IdentificationVisualizer

# Plot identification
IdentificationVisualizer.plot_identification_result(
    result.identification,
    autotuner.data.output,
    save_path='my_identification.png'
)

# Plot complete comparison
IdentificationVisualizer.plot_autotune_comparison(
    result,
    autotuner.data,
    save_path='my_autotune.png'
)
```

## Next Steps

1. **Run the demos** to understand the workflow
2. **Prepare your CSV data** following the requirements
3. **Start with quick autotune** for fast results
4. **Experiment with options** to optimize for your system
5. **Implement gains** in your real controller
6. **Iterate if needed** with different tuning rules or optimizers

## Additional Resources

- [SYSTEM_IDENTIFICATION_GUIDE.md](../SYSTEM_IDENTIFICATION_GUIDE.md) - Complete documentation
- [DATA_REQUIREMENTS.md](../DATA_REQUIREMENTS.md) - Data collection guidelines
- [README.md](../README.md) - Main library documentation

## Support

For questions or issues:
1. Check the documentation files
2. Review the example code
3. Verify your data meets requirements
4. Try different tuning rules/optimizers
