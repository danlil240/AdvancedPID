# AdvancedPID — Usage Tutorial

Step-by-step walkthrough of the v0.2 `PIDAutotuner` pipeline.  If you
already used the v0.1 `AutotuneFromData` / `RealtimeTuner` APIs, see
[docs/MIGRATION.md](docs/MIGRATION.md) for a diff-style translation
guide; this document is the forward-looking "start here" tour.

---

## 0. Install

```bash
# from a clean venv, at the repo root
python -m pip install -e ".[dev]"
```

Python 3.10+ is required.  Verify:

```bash
python -c "from pid_control.autotune import PIDAutotuner; print('ok')"
pidtune --help                 # CLI entry point
```

---

## 1. The four-stage pipeline

`PIDAutotuner.tune()` runs the same four stages regardless of input:

```
  (1) Experiment      (2) Identification     (3) Tuning          (4) Validation
  --------------      ------------------     ------------        -----------------
  CSV / arrays    →   FOPDT / SOPDT     →    Rule + DE     →     margins, robust,
  live plant          AIC/BIC selector       cost on real        confidence score
  (step/relay/chirp)  noise-aware fit        setpoint + limits   + typed warnings
```

Every stage is swappable via a builder method; the defaults are:

| Stage | Default implementation |
|---|---|
| Experiment | `FromDataExperiment` (CSV/arrays) or `StepExperiment` (plant) |
| Identification | `AutoIdentifier` — FOPDT vs SOPDT by lowest AIC |
| Rule | `IMCRule` with λ = max(τ, 8·θ) |
| Tuner | `DETuner` — differential evolution, 80 generations, seed=42 |
| Validators | `StabilityValidator` + `RobustnessValidator` |

---

## 2. Recipe A — I have a CSV

This is the most common case.

```python
from pid_control.autotune import PIDAutotuner

result = PIDAutotuner.from_csv("examples/data/fopdt_step.csv").tune()
print(result.report())
```

### 2.1 Column mapping

Default columns are `timestamp`, `output`, `measurement` (matching this
library's own `CSVLogger`).  For generic data:

```python
PIDAutotuner.from_csv(
    "my_run.csv",
    columns={"time": "t", "input": "u", "output": "y", "setpoint": "sp"},
).tune()
```

Only `time`, `input`, `output` are required; `setpoint` is optional and
used only for reporting.

### 2.2 What the pipeline checks before tuning

`DataQuality` (stage 0.5) will downgrade the run to `Status.FAILED`
with an `E_*` error code if any of the following are true:

| Check | Error code | Typical cause |
|---|---|---|
| Flat input (no excitation) | `E_DATA_FLAT` | Step never applied |
| No steady-state reached | `E_NO_STEADY_STATE` | Integrator-like plant |
| Too few samples | `E_TOO_SHORT` | < 50 samples |

See `examples/06_diagnosing_bad_data.py` for a worked example — it
intentionally feeds an integrator dataset in and shows the typed rejection.

### 2.3 Inspecting the result

```python
if result.is_usable:
    ctrl = result.build_controller()   # PIDController ready to go
    print("Kp", ctrl.params.kp, "Ki", ctrl.params.ki, "Kd", ctrl.params.kd)

print(result.status)                   # Status.OK | WARNING | FAILED
print(result.confidence.score)         # 0..1 aggregate trust
print(result.identification.model)     # e.g. "fopdt: K=2, tau=3, theta=0.5"
print(result.performance.margins)      # gain / phase margin, Ms, Mt

for w in result.warnings:
    print(w.severity.value, w.code.value, "—", w.message)
```

### 2.4 Persisting and reloading

```python
result.save("tune.json")                      # JSON; arrays are stripped
from pid_control.autotune.types import TuneResult
reloaded = TuneResult.load("tune.json")
```

---

## 3. Recipe B — I have a plant model

Simulated plants (or anything implementing `.update(u, dt) -> y` and
`.reset()`) can be tuned directly:

```python
from pid_control.autotune import PIDAutotuner, Objective
from pid_control.plants import FOPDTPlant

plant = FOPDTPlant(gain=1.5, time_constant=3.0, dead_time=0.5)

result = (
    PIDAutotuner.from_plant(plant)
    .with_objective(Objective(
        max_overshoot_pct=5.0,
        min_phase_margin_deg=45,
        max_Ms=1.6,
        control_effort_weight=0.05,
    ))
    .with_actuator_limits(lower=-10.0, upper=10.0, rate_limit=50.0)
    .tune()
)
```

The default experiment is `StepExperiment`, which performs a bounded
open-loop step using the actuator limits (so it is hardware-safe when
the plant is wrapped with `SafeExperiment`).

---

## 4. Recipe C — I have raw arrays

If your data is already in NumPy arrays (common in Jupyter):

```python
import numpy as np
from pid_control.autotune import PIDAutotuner

t = np.arange(0, 30, 0.02)
u = np.where(t >= 1.0, 1.0, 0.0)
y = ...  # whatever your plant produced

result = PIDAutotuner.from_arrays(t, u, y).tune()
```

---

## 5. Customising stages

Every stage can be replaced.  Example: use Skogestad's SIMC rule and
refine with Nelder–Mead instead of DE, with custom cost weights and
seeded reproducibility:

```python
from pid_control.autotune import PIDAutotuner
from pid_control.autotune.rules import SIMCRule
from pid_control.autotune.tuning.de import NelderMeadTuner
from pid_control.autotune.tuning.cost import CostSpec

result = (
    PIDAutotuner.from_csv("my.csv")
    .set_rule(SIMCRule(closed_loop_constant=2.0))
    .set_tuner(NelderMeadTuner(max_iter=300))
    .set_cost(CostSpec(iae_weight=1.0, itae_weight=0.2, du_weight=0.05,
                        ms_penalty_above=1.6))
    .tune()
)
```

Available building blocks:

- **Rules**: `ZieglerNicholsRule`, `CohenCoonRule`, `IMCRule`,
  `SIMCRule`, `AMIGORule`.
- **Tuners**: `DETuner`, `NelderMeadTuner`, `CMAESTuner`, `BOTuner`
  (scikit-learn GP).  Pass `None` to `set_tuner` to skip numerical
  refinement and keep the analytical rule's gains verbatim.
- **Experiments**: `FromDataExperiment`, `StepExperiment`,
  `RelayExperiment`, `ChirpExperiment`, `SafeExperiment` (safety wrapper).
- **Validators**: `StabilityValidator`, `RobustnessValidator`,
  `SimBenchmarkValidator`.

---

## 6. Warning codes — what to branch on

Stable codes you can pattern-match safely:

| Code | Severity | Meaning |
|---|---|---|
| `E_DATA_FLAT` | ERROR | Input signal is constant — no excitation |
| `E_NO_STEADY_STATE` | ERROR | Output never settles — integrator-like |
| `E_TOO_SHORT` | ERROR | Fewer than ~50 samples |
| `E_UNSTABLE` | ERROR | Closed-loop poles in RHP |
| `W_POOR_FIT` | WARNING | R² / AIC below threshold |
| `W_DEGENERATE_SOPDT` | WARNING | `τ₂ ≪ sample_time` — fell back to FOPDT |
| `W_GAIN_CLIPPED` | WARNING | Gain hit a user-supplied bound |
| `W_MAXITER` | WARNING | Optimizer hit its iteration cap |
| `W_COST_NOT_IMPROVED` | ERROR | Cost never improved over the initial guess |
| `W_LOW_MARGIN` | WARNING | Phase margin or Ms outside objective |
| `W_FRAGILE` | WARNING | Worst-case under ±20 %/±50 % perturbation is unstable |
| `W_SATURATION` | INFO | Actuator saturated during simulation |
| `W_HIGH_NOISE` | WARNING | Noise variance is a large fraction of signal |

```python
from pid_control.autotune import WarningCode

if result.has_warning(WarningCode.W_LOW_MARGIN):
    # reduce aggressiveness or raise min_phase_margin_deg
    ...
```

---

## 7. Headless plotting

`TuneResult.plot()` never calls `plt.show()` unless you ask:

```python
# Save PNGs to disk (CI-safe)
result.plot(kind="all", save_path="output/plots/")

# Open interactively
result.plot(kind="response", show=True)
```

Kinds: `fit` (identified model vs data), `response` (closed-loop step),
`margins` (Bode + sensitivity), `cost` (optimizer history), `all`.

---

## 8. CLI

Every Python recipe has a CLI analogue:

```bash
# From a CSV
pidtune csv examples/data/fopdt_step.csv --rule simc -o result.json

# From a built-in plant
pidtune plant fopdt --K 2.0 --tau 3.0 --theta 0.5 --format md

# Run the smoke benchmark
pidtune bench --suite smoke --out benchmarks/results/smoke.json
```

`--format {text,md,json}` mirrors `result.report(fmt=...)`.

---

## 9. Benchmarks

```bash
python -m benchmarks.smoke                              # ~30 s
python -m benchmarks.run --suite full --out bench.json  # longer
python -m benchmarks.smoke --compare benchmarks/results/baseline_pre_refactor.json
```

The baseline snapshot at
`benchmarks/results/baseline_pre_refactor.json` was captured before the
v0.2 refactor so regressions on safety metrics are detectable.

---

## 10. Troubleshooting

| Symptom | Likely cause & fix |
|---|---|
| `Status.FAILED` with `E_DATA_FLAT` | Your CSV's input column never changes — make sure you pass the actuator command, not a constant reference. |
| `Status.FAILED` with `E_NO_STEADY_STATE` | The plant behaves like an integrator; PID alone is not stabilising it.  Consider an integrator-aware model (IPDT) or a different controller topology. |
| `Status.WARNING` with `W_LOW_MARGIN` | The cost optimum is aggressive.  Tighten `Objective.min_phase_margin_deg` or `max_Ms`, or relax `max_overshoot_pct`. |
| `W_GAIN_CLIPPED` | You passed `bounds_kp=…` to the tuner and the optimum landed on the edge.  Widen the bound or accept the clipped value. |
| `DeprecationWarning: AutotuneFromData is deprecated` | You are on the v0.1 API; switch to `PIDAutotuner.from_csv(...)` (see [docs/MIGRATION.md](docs/MIGRATION.md)). |
| Plots pop up in CI | Use `result.plot(save_path=..., show=False)` — `show` defaults to `False` but some legacy demos still open Matplotlib windows; prefer the six golden examples. |

---

## 11. Where to go next

- [README.md](README.md) — short feature summary and installation.
- [docs/MIGRATION.md](docs/MIGRATION.md) — v0.1 → v0.2 diff.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — how to add new rules,
  identifiers, validators.
- [CHANGES.md](CHANGES.md) — what changed in the v0.2 refactor.
- [PLAN.md](PLAN.md) — the audit and roadmap that produced v0.2.
