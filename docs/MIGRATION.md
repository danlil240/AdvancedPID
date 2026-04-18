# Migration Guide: Old API → New PIDAutotuner (v0.2)

This guide maps the legacy `AutotuneFromData` / `RealtimeTuner` surface to the
new `PIDAutotuner` pipeline.  The old classes still work via a compatibility
shim but emit `DeprecationWarning`; they will be removed in v0.3.

---

## Quick comparison

| Old | New |
|---|---|
| `AutotuneFromData(csv).autotune()` | `PIDAutotuner.from_csv(csv).tune()` |
| `RealtimeTuner(ctrl, plant).auto_tune(sp, dur)` | `PIDAutotuner.from_plant(plant).tune()` |
| `SystemIdentifier(data).identify()` | Handled internally by the pipeline |
| Result: `dict` of gains | Result: `TuneResult` (typed, inspectable) |

---

## 1. CSV-based autotuning

### Before (v0.1)

```python
from pid_control.identification.autotune_from_data import AutotuneFromData

at = AutotuneFromData(
    "data.csv",
    time_col="timestamp",
    input_col="output",       # controller output column
    output_col="measurement", # plant measurement column
)
result = at.autotune(
    tuning_rule="cohen_coon",
    optimizer="differential_evolution",
    max_iterations=50,
)
print(result.optimized_gains)  # {'kp': ..., 'ki': ..., 'kd': ...}
```

### After (v0.2)

```python
from pid_control.autotune import PIDAutotuner
from pid_control.autotune.rules import CohenCoonRule

result = (
    PIDAutotuner.from_csv(
        "data.csv",
        columns={"time": "timestamp", "input": "output", "output": "measurement"},
    )
    .set_rule(CohenCoonRule())
    .tune()
)

print(result.gains)            # PIDGains(kp=..., ki=..., kd=...)
print(result.status)           # Status.OK | WARNING | FAILED
print(result.report())         # human-readable summary
result.save("result.json")     # persist & reload later
```

**Key differences:**

| Aspect | Old | New |
|---|---|---|
| Column mapping | Positional constructor args | `columns` dict |
| Tuning rule | String `"cohen_coon"` | Rule object `CohenCoonRule()` |
| Result type | `AutotuneFromDataResult` with `.optimized_gains` dict | `TuneResult` with typed `.gains`, `.status`, `.confidence` |
| Identification | Exposed via `result.identification` (old type) | `result.identification` (new `IdentificationResult` with R², AIC, BIC) |
| Validation | None | Automatic stability + robustness checks |
| Reporting | `result.summary()` | `result.report("text"\|"md"\|"json")` |
| Persistence | Manual | `result.save(path)` / `TuneResult.load(path)` |

---

## 2. Plant-based autotuning

### Before (v0.1)

```python
from pid_control import PIDController, PIDParams, RealtimeTuner
from pid_control.plants import FOPDTPlant

plant = FOPDTPlant(gain=2.0, time_constant=3.0, dead_time=0.5)
ctrl = PIDController(PIDParams(kp=1.0, ki=0.5, kd=0.1))

tuner = RealtimeTuner(ctrl, plant, optimizer="differential_evolution")
result = tuner.auto_tune(setpoint=100.0, duration=30.0)
print(result.kp, result.ki, result.kd)
```

### After (v0.2)

```python
from pid_control.autotune import PIDAutotuner
from pid_control.plants import FOPDTPlant

plant = FOPDTPlant(gain=2.0, time_constant=3.0, dead_time=0.5)
result = PIDAutotuner.from_plant(plant).tune()

print(result.gains.kp, result.gains.ki, result.gains.kd)
ctrl = result.build_controller()   # ready-to-use PIDController
```

---

## 3. Tuning rules

| Old string | New class |
|---|---|
| `"ziegler_nichols"` | `ZieglerNicholsRule()` |
| `"cohen_coon"` | `CohenCoonRule()` |
| `"imc"` | `IMCRule()` (default) |
| `"amigo"` | `AMIGORule()` |
| *(new)* | `SIMCRule()` |

```python
from pid_control.autotune.rules import IMCRule, SIMCRule, CohenCoonRule
result = PIDAutotuner.from_csv("data.csv").set_rule(SIMCRule()).tune()
```

---

## 4. Working with TuneResult

```python
result = PIDAutotuner.from_csv("data.csv").tune()

# Status & confidence
if result.is_usable:
    ctrl = result.build_controller()

# Warnings
for w in result.warnings:
    print(w.code, w.severity, w.message)

if result.has_warning(WarningCode.W_LOW_MARGIN):
    print("Consider reducing aggressiveness")

# Reports & plots
print(result.report("text"))
result.plot(kind="all", save_path="plots/")

# Persistence
result.save("tune.json")
loaded = TuneResult.load("tune.json")
```

---

## 5. CLI (new in v0.2)

The `pidtune` command provides the same functionality without writing Python:

```bash
# From CSV data
pidtune csv data.csv --rule simc -o result.json

# From a built-in plant
pidtune plant fopdt --K 2.0 --tau 3.0 --theta 0.5

# Run benchmarks
pidtune bench --suite smoke
```

---

## 6. Compatibility shim

If you cannot migrate immediately, the old `AutotuneFromData` import still
works but routes through `PIDAutotuner` internally:

```python
# This still works but emits DeprecationWarning
from pid_control.identification.autotune_from_data import AutotuneFromData
result = AutotuneFromData("data.csv").autotune()

# Access the new TuneResult underneath
new_result = result.new_result  # TuneResult object
```

To silence the warning during migration:

```python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning,
                        module="pid_control.autotune.compat")
```
