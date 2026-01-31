# Data Requirements for System Identification

## Overview

This document specifies the data requirements for using the system identification and autotuning features to obtain optimal PID gains from experimental data.

## Required Data

### CSV File Format

Your CSV file must contain the following columns:

| Column | Description | Units | Required |
|--------|-------------|-------|----------|
| `timestamp` | Timestamp | seconds | ✅ Yes |
| `output` | Control signal from PID (actuator command) | any | ✅ Yes |
| `measurement` | Measured process variable | any | ✅ Yes |
| `setpoint` | Desired output value | same as measurement | ⚪ Optional |
| `error` | Tracking error | same as measurement | ⚪ Optional |

**Note:** These column names match the output from `CSVLogger` in the PID library.

### Example CSV

```csv
timestamp,output,measurement,setpoint
0.00,0.0,25.2,50.0
0.01,5.2,25.3,50.0
0.02,8.1,25.5,50.0
0.03,10.5,25.8,50.0
0.04,12.3,26.2,50.0
...
```

## Data Collection Guidelines

### 1. Sample Rate

**Requirement:** Sample time should be 10-100x faster than system dynamics

| System Type | Typical Sample Rate |
|-------------|-------------------|
| Fast (motors, servos) | 100-1000 Hz |
| Medium (temperature, pressure) | 1-10 Hz |
| Slow (chemical processes) | 0.1-1 Hz |

**Rule of thumb:** If your system time constant is τ, sample at least 10/τ Hz

### 2. Duration

**Requirement:** Capture complete system response

- **Minimum:** 3-5 time constants
- **Recommended:** 5-10 time constants
- **Include:** Transient response + settling to steady state

**Example:** If τ = 2 seconds, collect at least 10-20 seconds of data

### 3. Number of Data Points

| Quality | Data Points | Use Case |
|---------|-------------|----------|
| Minimum | 20-50 | Quick identification |
| Good | 100-500 | Standard applications |
| Excellent | 500-2000 | High precision needed |

### 4. Input Signal Requirements

**Best:** Step input (sudden change in control signal)
- Clear transient response
- Easy to analyze
- Most reliable identification

**Acceptable:** Dynamic input with variation
- Ramp signals
- Multi-step sequences
- Pseudo-random binary sequence (PRBS)

**Not suitable:**
- Constant input (no variation)
- Random noise only
- Very slow changes

### 5. Output Response

**Required characteristics:**
- Clear response to input changes
- System reaches or approaches steady state
- Measurable signal variation

**Signal-to-noise ratio:**
- Minimum: 10:1 (acceptable)
- Good: 20:1 or better
- Excellent: 50:1 or better

### 6. System State

**Initial conditions:**
- System should start at or near steady state
- Or clearly defined initial condition
- Avoid transients before data collection starts

**During collection:**
- No external disturbances if possible
- Consistent operating conditions
- No parameter changes mid-test

## Data Quality Checklist

Before using your data for system identification, verify:

- [ ] Timestamp column is monotonically increasing
- [ ] No missing or NaN values
- [ ] Output (control signal) shows clear variation
- [ ] Measurement responds to output changes
- [ ] Sample rate is consistent
- [ ] Sufficient duration captured
- [ ] System reaches steady state (or shows clear trend)
- [ ] No obvious outliers or data corruption

## Example Data Collection Scenarios

### Scenario 1: Temperature Control System

```
System: Heating element controlling temperature
Time constant: ~30 seconds

Data requirements:
- Sample rate: 1-5 Hz (0.2-1.0 second intervals)
- Duration: 3-5 minutes
- Input: Heater power (0-100%)
- Output: Temperature (°C)
- Test: Step heater from 0% to 50%
```

### Scenario 2: Motor Position Control

```
System: DC motor with encoder
Time constant: ~0.1 seconds

Data requirements:
- Sample rate: 50-200 Hz (0.005-0.02 second intervals)
- Duration: 2-5 seconds
- Input: Motor voltage or PWM (%)
- Output: Position (degrees or counts)
- Test: Step command from 0 to target position
```

### Scenario 3: Liquid Level Control

```
System: Tank with pump
Time constant: ~60 seconds

Data requirements:
- Sample rate: 0.5-2 Hz (0.5-2 second intervals)
- Duration: 5-10 minutes
- Input: Pump speed (%)
- Output: Level (cm or %)
- Test: Step pump speed change
```

## Current PID Gains (Optional)

If you collected data with a PID controller already running, you can provide the gains used:

```python
from pid_control.identification import CSVDataReader

reader = CSVDataReader('data.csv')
data = reader.read_with_pid_params(
    kp_value=1.5,
    ki_value=0.2,
    kd_value=0.1
)
```

This helps with:
- Understanding current performance
- Comparing optimized vs current gains
- Validating identification results

## What the System Identifies

From your CSV data, the system will identify:

### Transfer Function Parameters

**FOPDT Model:** G(s) = K·e^(-θs) / (τs + 1)

- **K (Gain):** Steady-state output change / input change
- **τ (Time Constant):** How fast system responds (63.2% rise time)
- **θ (Dead Time):** Delay before response starts

### Example Identification Output

```
Transfer Function Parameters:
  Gain (K): 2.450
  Time Constant (τ): 1.234 s
  Dead Time (θ): 0.156 s
  
Fit Quality (R²): 0.9823

Recommended PID Gains (Ziegler-Nichols):
  Kp = 3.456
  Ki = 1.234
  Kd = 0.567
```

## Troubleshooting Poor Identification

### Problem: Low R² (< 0.7)

**Possible causes:**
- Insufficient data duration
- High noise levels
- Non-linear system behavior
- Wrong model type (try SOPDT)
- Data doesn't capture full response

**Solutions:**
- Collect longer duration data
- Increase signal-to-noise ratio
- Filter data before identification
- Use optimization method instead of step response

### Problem: Unrealistic Parameters

**Possible causes:**
- Bad data quality
- No clear step response
- System not reaching steady state

**Solutions:**
- Verify data collection procedure
- Check for data corruption
- Ensure input has clear step change
- Collect data over longer duration

### Problem: Poor Tuning Results

**Possible causes:**
- Inaccurate system identification
- Wrong tuning rule for application
- System non-linearity

**Solutions:**
- Improve identification quality first
- Try different tuning rules
- Use numerical optimization
- Adjust bounds_scale parameter

## Quick Validation Test

After collecting data, quickly check if it's suitable:

```python
from pid_control.identification import CSVDataReader
import numpy as np

reader = CSVDataReader('your_data.csv')
data = reader.read()

print(f"Data points: {len(data.time)}")
print(f"Duration: {data.time[-1] - data.time[0]:.2f} s")
print(f"Sample time: {data.sample_time:.4f} s")
print(f"Input range: {np.min(data.input):.2f} to {np.max(data.input):.2f}")
print(f"Output range: {np.min(data.output):.2f} to {np.max(data.output):.2f}")
print(f"Input variation: {np.std(data.input):.4f}")
print(f"Output variation: {np.std(data.output):.4f}")

# Quick plot
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
ax1.plot(data.time, data.input)
ax1.set_ylabel('Input')
ax1.grid(True)
ax2.plot(data.time, data.output)
ax2.set_ylabel('Output')
ax2.set_xlabel('Time (s)')
ax2.grid(True)
plt.tight_layout()
plt.show()
```

## Summary

**Minimum requirements for successful identification:**
1. ✅ CSV with time, input, output columns
2. ✅ At least 20-50 data points
3. ✅ Clear input variation (step preferred)
4. ✅ Observable output response
5. ✅ Consistent sample rate
6. ✅ Sufficient duration (3-5 time constants)

**For best results:**
- 100-500+ data points
- Step input with clean response
- Low noise (SNR > 20:1)
- System reaches steady state
- Sample rate 10-100x faster than dynamics

Once you have suitable data, the system identification workflow will automatically:
1. Estimate transfer function parameters
2. Apply analytical tuning rules
3. Optimize gains numerically
4. Provide comprehensive visualizations
5. Output optimal PID gains ready to use
