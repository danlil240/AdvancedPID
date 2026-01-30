# Advanced PID Control Library

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A professional-grade, modular Python library for PID control with advanced features, real-time tuning, comprehensive analysis, and stunning visualizations.

## ✨ Features

### Core PID Controller
- **Robust Implementation** with all professional features:
  - Proportional, Integral, Derivative control
  - **Multiple anti-windup methods**: Clamping, Back-calculation, Conditional integration
  - **Derivative filtering** to reduce noise amplification
  - **Derivative on measurement** to avoid derivative kick on setpoint changes
  - **Setpoint weighting** (2-DOF PID) for reduced overshoot
  - **Bumpless transfer** for smooth parameter changes
  - Output saturation with proper integral handling
  - Error deadband
  - Output rate limiting
  - Efficient CSV logging with buffering

### Real-Time Tuner
- **Multiple optimization algorithms**:
  - Nelder-Mead (gradient-free)
  - Bayesian Optimization
  - Genetic Algorithm
  - Differential Evolution
- **Classical tuning methods**:
  - Ziegler-Nichols (step response)
  - Relay feedback auto-tune
  - Cohen-Coon
  - IMC tuning
- Configurable cost functions with multiple objectives

### Analyzer
- **Comprehensive metrics**:
  - Step response: rise time, settling time, overshoot, peak time
  - Error metrics: IAE, ISE, ITAE, RMSE
  - Control effort: total variation, RMS, saturation time
- **Professional visualizations**:
  - Response plots with error overlay
  - PID component breakdown
  - Phase portraits
  - Frequency analysis
  - Saturation analysis
  - Controller comparison charts
  - Radar charts for metrics comparison

### Plant Models
- First-order (PT1)
- Second-order with configurable damping
- FOPDT (First-Order Plus Dead Time)
- Nonlinear plants (saturation, dead-zone, backlash)
- Friction models (Coulomb, viscous, stiction)
- Delay wrapper for any plant

### Simulation Framework
- Pre-defined test scenarios
- Custom scenario creation
- Batch simulation
- Animated real-time visualization
- Interactive parameter adjustment

## 📁 Project Structure

```
AdvancedPID/
├── pid_control/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── pid_controller.py    # Main PID controller
│   │   ├── pid_params.py        # Parameter configuration
│   │   └── filters.py           # Signal filters
│   ├── tuner/
│   │   ├── __init__.py
│   │   ├── realtime_tuner.py    # Real-time tuning
│   │   └── optimization_methods.py
│   ├── analyzer/
│   │   ├── __init__.py
│   │   ├── pid_analyzer.py      # Main analyzer
│   │   ├── metrics.py           # Performance metrics
│   │   └── plots.py             # Visualization
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── simulator.py         # Simulation engine
│   │   └── scenarios.py         # Test scenarios
│   ├── plants/
│   │   ├── __init__.py
│   │   ├── base_plant.py
│   │   ├── first_order.py
│   │   ├── second_order.py
│   │   ├── nonlinear.py
│   │   └── delay_plant.py
│   ├── logging/
│   │   ├── __init__.py
│   │   ├── csv_logger.py
│   │   └── data_buffer.py
│   └── utils/
│       ├── __init__.py
│       ├── validators.py
│       └── math_utils.py
├── examples/
│   ├── demo_basic.py
│   ├── demo_tuning.py
│   ├── demo_advanced_features.py
│   ├── demo_spectacular_simulations.py
│   └── demo_animated.py
├── tests/
│   ├── test_pid_controller.py
│   └── test_plants.py
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
# Clone or download the project
cd AdvancedPID

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from pid_control import PIDController, PIDParams
from pid_control.plants import FirstOrderPlant

# Create a plant
plant = FirstOrderPlant(gain=2.0, time_constant=1.0, sample_time=0.01)

# Configure PID controller
params = PIDParams(
    kp=1.5,
    ki=0.5,
    kd=0.2,
    sample_time=0.01,
    output_min=-100,
    output_max=100
)

# Create controller with CSV logging
pid = PIDController(params, csv_path="pid_log.csv")

# Control loop
measurement = 0.0
setpoint = 100.0

for i in range(1000):
    output = pid.update(setpoint, measurement)
    measurement = plant.update(output)

pid.close()
```

### Simulation and Visualization

```python
from pid_control import PIDParams, Simulator
from pid_control.plants import SecondOrderPlant
from pid_control.simulation import ScenarioLibrary

# Create plant and controller params
plant = SecondOrderPlant(gain=1.0, natural_frequency=2.0, damping_ratio=0.5)
params = PIDParams(kp=3.0, ki=1.5, kd=0.5)

# Run simulation
sim = Simulator(plant, params)
result = sim.run(ScenarioLibrary.step_response(setpoint=100.0))

# Analyze and plot
metrics = sim.analyze(result)
print(f"Rise Time: {metrics['step_response']['rise_time']:.3f}s")
print(f"Overshoot: {metrics['step_response']['overshoot_percent']:.1f}%")

sim.plot_results(result, comprehensive=True)
Simulator.show()
```

### Auto-Tuning

```python
from pid_control import PIDController, PIDParams, RealtimeTuner
from pid_control.plants import FOPDTPlant

# Create plant and initial controller
plant = FOPDTPlant(gain=2.0, time_constant=3.0, dead_time=0.5)
initial_params = PIDParams(kp=1.0, ki=0.5, kd=0.1)
controller = PIDController(initial_params)

# Create tuner and auto-tune
tuner = RealtimeTuner(
    controller, 
    plant,
    optimizer='differential_evolution'
)

result = tuner.auto_tune(setpoint=100.0, duration=30.0)

print(f"Optimized: Kp={result.kp:.3f}, Ki={result.ki:.3f}, Kd={result.kd:.3f}")
```

### Analyzing Logged Data

```python
from pid_control import PIDAnalyzer

# Load and analyze CSV log
analyzer = PIDAnalyzer("pid_log.csv")
metrics = analyzer.analyze()

# Generate report
print(analyzer.generate_report())

# Plot comprehensive analysis
analyzer.plot_comprehensive()
PIDAnalyzer.show_plots()
```

## 🎮 Running Examples

```bash
# Basic demo
python examples/demo_basic.py

# Auto-tuning demo
python examples/demo_tuning.py

# Advanced features (anti-windup, filtering, etc.)
python examples/demo_advanced_features.py

# Spectacular visualizations (3D plots, surfaces, etc.)
python examples/demo_spectacular_simulations.py

# Interactive animated simulation
python examples/demo_animated.py
```

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=pid_control
```

## 📊 Anti-Windup Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `NONE` | No anti-windup | Testing only |
| `CLAMPING` | Stop integration when saturated | Simple systems |
| `BACK_CALCULATION` | Feed back saturation error | Most applications |
| `CONDITIONAL_INTEGRATION` | Selective integration | Aggressive tuning |

## 🔧 PID Parameters Reference

```python
PIDParams(
    # Core gains
    kp=1.0,                  # Proportional gain
    ki=0.0,                  # Integral gain
    kd=0.0,                  # Derivative gain
    
    # Timing
    sample_time=0.01,        # Sample time (seconds)
    
    # Output limits
    output_min=None,         # Minimum output
    output_max=None,         # Maximum output
    
    # Anti-windup
    anti_windup=AntiWindupMethod.BACK_CALCULATION,
    back_calculation_gain=1.0,
    
    # Derivative handling
    derivative_mode=DerivativeMode.MEASUREMENT,  # Avoid derivative kick
    derivative_filter_coeff=10.0,                # Filter coefficient N
    
    # Setpoint weighting (2-DOF)
    setpoint_weight_p=1.0,   # b: 0=no kick, 1=full response
    setpoint_weight_d=0.0,   # c: derivative setpoint weight
    
    # Additional features
    error_deadband=0.0,      # Ignore small errors
    output_rate_limit=None,  # Max change per sample
)
```

## 📈 Performance Tips

1. **Start with PI control** - add derivative only if needed
2. **Use derivative filtering** - coefficient N between 5-20
3. **Enable derivative on measurement** - prevents setpoint kicks
4. **Use back-calculation anti-windup** - most robust method
5. **Set reasonable output limits** - prevents actuator damage
6. **Log data for analysis** - use the CSV logger

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Classical PID control theory
- Modern optimization techniques
- The control systems community
