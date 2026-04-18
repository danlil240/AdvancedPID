# Architecture Guide

> Intended audience: developers extending or debugging the `pid_control` library.

---

## Package layout

```
pid_control/
├── core/               # PIDController, PIDParams, filters
├── plants/             # Simulation plant models (BasePlant, FOPDT, …)
├── envs/               # Gymnasium-compatible wrappers
├── autotune/           # ★ The new pipeline (v0.2)
│   ├── types.py        # All frozen dataclasses & enums
│   ├── api.py          # PIDAutotuner façade
│   ├── compat.py       # Old-API shim
│   ├── experiments/    # Stage 1 — data acquisition
│   ├── identification/ # Stage 2 — model fitting
│   ├── rules/          # Stage 3a — analytical gain rules
│   ├── tuning/         # Stage 3b — numerical refinement
│   ├── validation/     # Stage 4 — stability & robustness checks
│   └── diagnostics/    # Reporting & plotting
├── identification/     # Legacy system identification (deprecated)
├── tuner/              # Legacy RealtimeTuner (deprecated)
├── analyzer/           # Offline analysis & metrics
├── simulation/         # Scenario-based simulation engine
├── logging/            # CSV logger & data buffer
├── cli/                # `pidtune` CLI entry point
└── utils/              # Validators, math helpers
```

---

## Autotune pipeline

`PIDAutotuner.tune()` executes five stages in sequence.  Each stage is a
pluggable protocol object injected via the builder chain.

```
┌─────────────┐     ┌────────────────┐     ┌──────────────┐
│ 1. Experiment│ ──▶ │ 2. Identifier  │ ──▶ │ 3a. Rule     │
│   (data)     │     │ (FOPDT/SOPDT)  │     │ (analytical) │
└─────────────┘     └────────────────┘     └──────┬───────┘
                                                   │
                                                   ▼
                                           ┌──────────────┐
                                           │ 3b. Tuner    │
                                           │ (numerical)  │
                                           └──────┬───────┘
                                                   │
                                                   ▼
                                           ┌──────────────┐     ┌────────────┐
                                           │ 4. Validators│ ──▶ │ TuneResult │
                                           │ (margins)    │     └────────────┘
                                           └──────────────┘
```

Each stage can short-circuit the pipeline by emitting an `ERROR`-severity
warning, which sets `Status.FAILED` on the final `TuneResult`.

### Default components

| Stage | Default | Alternatives |
|---|---|---|
| Experiment | `StepExperiment` (for plants) / passthrough (CSV) | Custom `Experiment` |
| Identifier | `AutoIdentifier` (picks FOPDT vs SOPDT by AIC) | `FOPDTIdentifier`, `SOPDTIdentifier`, `IPDTIdentifier` |
| Rule | `IMCRule` | `SIMCRule`, `AMIGORule`, `CohenCoonRule`, `ZieglerNicholsRule` |
| Tuner | `DETuner` (Differential Evolution) | `NelderMeadTuner`, `None` (skip) |
| Validators | `[StabilityValidator, RobustnessValidator]` | Any `Validator` protocol impl |

---

## Stage protocols

All protocols are `@runtime_checkable` and live alongside their
implementations.  No base-class inheritance is required — duck typing via
`Protocol` suffices.

### Experiment

```python
class Experiment(Protocol):
    name: str
    def run(self, plant: Any | None = None) -> ExperimentRecord: ...
```

`ExperimentRecord` bundles `time`, `input`, `output`, `setpoint`,
`sample_time`, `quality: DataQuality`, and `warnings`.

### Identifier

```python
class Identifier(Protocol):
    name: str
    def identify(self, record: ExperimentRecord) -> IdentificationResult: ...
```

Returns a `TransferFunctionModel` (K, tau, theta, optional tau2) plus
fit-quality metrics (R², AIC, BIC).

### TuningRule

```python
class TuningRule(Protocol):
    name: str
    def apply(self, identification: IdentificationResult,
              objective: Objective, actuator: ActuatorLimits) -> PIDGains: ...
```

Pure function — no state, no side effects.

### Tuner

```python
class Tuner(Protocol):
    name: str
    def refine(self, identification: IdentificationResult,
               initial: PIDGains, objective: Objective,
               actuator: ActuatorLimits) -> TunerOutcome: ...
```

`TunerOutcome` carries refined `gains`, scalar `cost`, iteration count, and
optional `cost_history` array.

### Validator

```python
class Validator(Protocol):
    name: str
    def validate(self, identification: IdentificationResult,
                 gains: PIDGains, objective: Objective,
                 actuator: ActuatorLimits) -> ValidationOutcome: ...
```

Multiple validators run in sequence.  The façade takes the strictest
`status` and concatenates all warnings.

### CostSpec

```python
@dataclass(frozen=True)
class CostSpec:
    iae: float = 1.0
    itae: float = 0.0
    ise: float = 0.0
    overshoot_above: float = 0.0
    settling_pct_above: float = 0.0
    du: float = 0.01          # control-effort penalty
    Ms_penalty_above: float = 0.0
    Mt_penalty_above: float = 0.0
```

Built from `Objective` via `CostSpec.from_objective(obj)`.  The `Tuner`
feeds `CostSpec` to a `CostEvaluator` that simulates the closed loop and
returns a scalar.

---

## Adding a new tuning rule

1. Create `pid_control/autotune/rules/my_rule.py`:

```python
from dataclasses import dataclass
from pid_control.autotune.types import (
    ActuatorLimits, IdentificationResult, Objective, PIDGains,
)

@dataclass(frozen=True)
class MyRule:
    name: str = "my_rule"

    def apply(
        self,
        identification: IdentificationResult,
        objective: Objective,
        actuator: ActuatorLimits,
    ) -> PIDGains:
        m = identification.model
        # your gain formulas here
        return PIDGains(kp=..., ki=..., kd=...)
```

2. Re-export from `pid_control/autotune/rules/__init__.py`.

3. Use it:

```python
PIDAutotuner.from_csv("data.csv").set_rule(MyRule()).tune()
```

No registration, no metaclass — just satisfy the protocol.

---

## Adding a new identifier

Same pattern as rules:

```python
@dataclass(frozen=True)
class MyIdentifier:
    name: str = "my_identifier"

    def identify(self, record) -> IdentificationResult:
        # fit your model to record.time, record.output, record.input
        return IdentificationResult(model=..., fit_quality_r2=...)
```

Plug in via `.set_identifier(MyIdentifier())`.

---

## Adding a new validator

```python
@dataclass(frozen=True)
class MyValidator:
    name: str = "my_validator"

    def validate(self, identification, gains, objective, actuator):
        from pid_control.autotune.validation.base import ValidationOutcome
        warnings = []
        # ... check whatever you need ...
        return ValidationOutcome(
            status=Status.OK,
            warnings=tuple(warnings),
        )
```

Add via `.add_validator(MyValidator())` (appends) or
`.set_validators([...])` (replaces defaults).

---

## Types & conventions

- **All result types are frozen dataclasses** — immutable by design.
  Use `dataclasses.replace()` (or the `.with_warnings()` helper) to derive
  modified copies.
- **Warnings use stable `WarningCode` enums** so downstream code can branch
  deterministically.  Never match on `.message` strings.
- **`TuneResult` is the single top-level artifact.** Everything the caller
  needs (gains, diagnostics, margins, plots) is accessible from it.
- **Artifacts (heavy arrays) live on `TuneResult.artifacts`** and are *not*
  serialized by `.save()`.  Callers who need raw trajectories should process
  them before discarding the result object.

---

## CLI

The `pidtune` CLI in `pid_control/cli/main.py` exposes three subcommands
(`csv`, `plant`, `bench`) that wrap `PIDAutotuner` and the benchmark
harness.  See `pidtune --help` or the [Migration Guide](MIGRATION.md).
