# Advanced PID Control Library

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#known-limitations)

A Python library for SISO PID control with a typed autotune pipeline
(`PIDAutotuner`), stability/robustness validation, system identification,
simulation, and analysis.  Designed to fail loudly on bad data rather
than silently produce wrong gains.

> **Status:** v0.2 refactor is complete (see [PLAN.md](PLAN.md) and
> [CHANGES.md](CHANGES.md)).  Legacy `AutotuneFromData` / `RealtimeTuner`
> still work through a deprecation shim — prefer `PIDAutotuner`.

## ✨ Features

### `PIDAutotuner` — Unified autotune façade *(primary API)*
- **One pipeline**: Experiment → Identify → Tune → Validate.
- **Typed results** (`TuneResult`): `status`, typed `warnings` with stable
  codes (`W_GAIN_CLIPPED`, `W_LOW_MARGIN`, `E_DATA_FLAT`, …), and a
  confidence score in [0, 1].
- **Honest failure**: flat or integrator-like data is rejected with
  `Status.FAILED` and `E_*` error codes instead of silent zero-gain
  controllers.
- **Stability & robustness validation**: gain/phase margin, Ms, Mt, delay
  margin, and ±20 %/±50 % parameter perturbation sweeps.
- **Classical rules as first-class citizens**: Ziegler-Nichols, Cohen-Coon,
  IMC, SIMC (Skogestad), AMIGO — each pluggable via `.set_rule(...)`.
- **Numerical refinement backends**: Differential Evolution (default),
  Nelder-Mead, CMA-ES, and Bayesian Optimization (real `sklearn` GP).
- **Reports & plots**: `result.report("text"|"md"|"json")`, headless-safe
  `result.plot(save_path=...)`, JSON persistence via `result.save()`.
- **CLI**: `pidtune csv …` / `pidtune plant fopdt …` / `pidtune bench …`.

### Core PID Controller *(preserved from v0.1)*
- Proportional, Integral, Derivative control
- **Anti-windup**: Clamping, Back-calculation, Conditional integration
- **Derivative filtering** (first-order, configurable `N`)
- **Derivative on measurement** to avoid setpoint-change kick
- **Setpoint weighting** (2-DOF PID: `b`, `c`)
- **Bumpless transfer** for online gain changes
- Output saturation with proper integral handling
- **Error deadband** now suppresses P, I, **and** D (bug fix in v0.2)
- Output rate limiting
- Efficient CSV logging with buffering

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
│   ├── core/                    # PIDController, PIDParams, filters
│   ├── plants/                  # FOPDT, SOPDT, FrictionPlant, …
│   ├── envs/                    # Gymnasium wrappers
│   ├── autotune/                # ★ Unified pipeline (v0.2)
│   │   ├── types.py             #   Frozen dataclasses (TuneResult, …)
│   │   ├── api.py               #   PIDAutotuner façade
│   │   ├── compat.py            #   Deprecation shim for v0.1 APIs
│   │   ├── experiments/         #   step / relay / chirp / from_data
│   │   ├── identification/      #   fopdt, sopdt, ipdt, AIC selector
│   │   ├── rules/               #   ZN, CC, IMC, SIMC, AMIGO
│   │   ├── tuning/              #   DE, NM, CMA-ES, BO + cost
│   │   ├── validation/          #   margins, robustness, confidence
│   │   └── diagnostics/         #   reports, plots, data-quality
│   ├── cli/                     # `pidtune` entry point
│   ├── analyzer/                # Offline metrics & plots
│   ├── simulation/              # Scenario simulator
│   ├── logging/                 # CSV logger + buffer
│   ├── identification/          # (legacy) FOPDT / SOPDT ID + autotune
│   ├── tuner/                   # (legacy) RealtimeTuner
│   └── utils/                   # Shared helpers
├── examples/
│   ├── 01_quickstart_plant.py   # ★ start here
│   ├── 02_quickstart_csv.py
│   ├── 03_from_plant_objective.py
│   ├── 04_relay_autotune.py
│   ├── 05_compare_rules.py
│   ├── 06_diagnosing_bad_data.py
│   ├── data/                    # Sample CSV datasets
│   └── advanced/                # Legacy interactive demos
├── benchmarks/                  # Plant zoo, smoke & full runs, baseline
├── tests/                       # 97 tests across 7 files
├── tools/run_examples.py        # Headless CI runner
├── docs/                        # MIGRATION.md, ARCHITECTURE.md
└── .github/workflows/ci.yml     # pytest + example runner
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

### Autotune from CSV data *(recommended)*

```python
from pid_control.autotune import PIDAutotuner

result = PIDAutotuner.from_csv("examples/data/fopdt_step.csv").tune()

print(result.report())                # human-readable summary
if result.is_usable:
    ctrl = result.build_controller()  # ready-to-run PIDController
else:
    for w in result.warnings:
        print(w.severity.value, w.code.value, w.message)

result.save("output/result.json")     # reload later with TuneResult.load
```

**CSV column mapping** — default expects `timestamp,output,measurement`
(matching this library's `CSVLogger`).  Override for generic data:

```python
PIDAutotuner.from_csv(
    "your.csv",
    columns={"time": "t", "input": "u", "output": "y"},
).tune()
```

See [USAGE_TUTORIAL.md](USAGE_TUTORIAL.md) and
[docs/MIGRATION.md](docs/MIGRATION.md) for the full workflow.

### Autotune from a plant model

```python
from pid_control.autotune import PIDAutotuner, Objective
from pid_control.plants import FOPDTPlant

plant  = FOPDTPlant(gain=2.0, time_constant=3.0, dead_time=0.5)
result = (
    PIDAutotuner.from_plant(plant)
    .with_objective(Objective(max_overshoot_pct=5.0, min_phase_margin_deg=45))
    .with_actuator_limits(lower=-10.0, upper=10.0)
    .tune()
)
print(result.gains.kp, result.gains.ki, result.gains.kd)
print(result.confidence.score)        # 0..1 aggregate trust score
```

### CLI

```bash
pidtune csv examples/data/fopdt_step.csv --rule simc -o result.json
pidtune plant fopdt --K 1.5 --tau 3.0 --theta 0.5
pidtune bench --suite smoke
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

All six golden examples run headlessly and write artifacts under
`./output/`.  They are also exercised by CI on every push.

```bash
python examples/01_quickstart_plant.py --output ./output
python examples/02_quickstart_csv.py
python examples/03_from_plant_objective.py
python examples/04_relay_autotune.py
python examples/05_compare_rules.py
python examples/06_diagnosing_bad_data.py    # shows a FAILED result on bad data

# Or run them all through the headless CI runner:
python -m tools.run_examples
```

Legacy interactive demos (`demo_basic.py`, `demo_animated.py`,
`demo_double_pendulum*.py`, …) live under `examples/advanced/` and may
open blocking plot or Gym render windows.

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

## Known Limitations

- **SISO only** — the library tunes a single PID loop at a time. MIMO plants, cascade loops, and feedforward structures are out of scope.
- **Linear-model identification** — the autotune pipeline fits FOPDT / SOPDT transfer functions. Highly nonlinear, time-varying, or unstable-open-loop plants will produce poor fits and unreliable gains.
- **No real-time guarantees** — pure Python with NumPy/SciPy. Not suitable for hard-real-time embedded control without a compiled wrapper.
- **Real-hardware relay autotune is experimental** — the biased-ATV
  relay experiment (`RelayExperiment`) has unit-test coverage on
  simulated plants but has not been validated on physical hardware;
  always compose it with `SafeExperiment` / abort limits first.
- **Identification is offline** — the pipeline expects a recorded step
  or relay response; it does not currently perform closed-loop
  identification while a plant is running in production.

## When *Not* to Use PID

PID control is a workhorse, but it is the wrong tool when:

| Situation | Better alternative |
|---|---|
| The plant is significantly nonlinear across its operating range | Gain-scheduled PID, MPC, or adaptive control |
| Multiple interacting loops (MIMO) | Decoupling + multi-loop PID, or full MPC |
| The plant is open-loop unstable *and* has large dead time | State-feedback or model-predictive control |
| You need optimal trajectories, not setpoint tracking | Trajectory optimization / optimal control |
| Safety-critical hard-real-time (< 1 ms) | C/C++ control libraries with RTOS integration |

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
