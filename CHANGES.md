# CHANGES — PLAN.md Verification Pass (18 Apr 2026)

This file documents the audit of [PLAN.md](PLAN.md) and the gap-closing
edits applied during the verification pass.  Treat it as the
"what actually changed" companion to the forward-looking PLAN.

---

## TL;DR

- [x] **All 64 PLAN tasks are now implemented** (previously 60/64 — 4 gaps
  closed in this pass, listed below).
- [x] **97 tests pass** (70 fast + 27 pipeline) after the changes.
- [x] **Six golden examples** are the only scripts at
  `examples/` root; legacy interactive demos moved to `examples/advanced/`.
- [x] Documentation: new **`README.md`** (honest capability list),
  new **`USAGE_TUTORIAL.md`** (step-by-step walkthrough), existing
  `docs/MIGRATION.md` and `docs/ARCHITECTURE.md` verified.

---

## 1. Verification matrix

Status checked against every task in PLAN.md §9.

| WS | Task | Status | Notes |
|---|---|---|---|
| W0 | T0.1 Fix `error_deadband` P-term bypass (C4) | ✅ | `core/pid_controller.py:193-223` suppresses P, I, D; test `test_error_deadband_suppresses_all_terms` passes. |
| W0 | T0.2 Remove `examples/demo.py`, purge `steering-*.log` | ✅ | `find_by_name demo.py` and `steering-*.log` both return 0 results. |
| W0 | T0.3 `setup.py` deps + `python_requires>=3.10` | ✅ | `control`, `gymnasium` present; `python_requires=">=3.10"`. |
| W0 | T0.4 `np.trapz` → `np.trapezoid` | ✅ | Guarded via `getattr(np, "trapezoid", np.trapz)` in `analyzer/metrics.py` and `utils/math_utils.py`. |
| W0 | T0.5 CI (`.github/workflows/ci.yml`) | ✅ | Matrix over 3.10/3.11/3.12 × ubuntu/windows, plus a golden-examples job. |
| W1 | T1.1 `types.py` (frozen dataclasses) | ✅ | `PIDGains`, `TuneResult`, `Warning`, `Objective`, `ActuatorLimits`, … |
| W1 | T1.2 Stage protocols | ✅ | `base.py` in `experiments/`, `identification/`, `rules/`, `tuning/`, `validation/`. |
| W1 | T1.3 `PIDAutotuner` façade | ✅ | `autotune/api.py`; all 5 builder methods present. |
| W1 | T1.4 Back-compat shim | ✅ | `autotune/compat.py` wraps `AutotuneFromData` with `DeprecationWarning`. |
| W1 | T1.5 Remove silent gain clamps | ✅ **(fixed this pass)** | Legacy `system_identifier.py` now emits `UserWarning("W_GAIN_CLIPPED: …")`; new `DETuner` emits typed `WarningCode.W_GAIN_CLIPPED`. |
| W2 | T2.1–T2.7 Identification correctness | ✅ | `fopdt.py`, `sopdt.py`, `ipdt.py`, `selector.py` (AIC/BIC), `data_quality.py`, `delay_estimator.py`, `noise_variance` populated on `IdentificationResult`. |
| W3 | T3.1–T3.5 Experiments | ✅ | `relay.py` (biased ATV + output hysteresis + `Ku = 4d/(π·a_y)`), `step.py`, `chirp.py`, `from_data.py`, `safety.py`. |
| W4 | T4.1–T4.5 Tuning | ✅ | 5 rules, 4 tuner backends (DE/NM/CMA-ES/BO with real `sklearn` GP), composable `CostSpec`, honest `Status` reporting. |
| W5 | T5.1–T5.5 Validation | ✅ | `StabilityValidator` (margins via `control`), `RobustnessValidator` (±20 %/±50 %), `SimBenchmarkValidator`, `ConfidenceAggregator`, `W_LOW_MARGIN` gating. |
| W6 | T6.1–T6.4 Reporting | ✅ | `report("text"/"md"/"json")`, headless-safe `plot(save_path=…, show=False)`, `pidtune` CLI with `csv`/`plant`/`bench`, `TuneResult.save/load`. |
| W7 | T7.1–T7.5 Examples | ✅ **(fixed this pass)** | Six numbered golden examples at `examples/`; legacy demos moved to `examples/advanced/`; sample CSVs in `examples/data/`. |
| W8 | T8.1–T8.3 Tests | ✅ | 97 total (regression + property + pipeline integration). |
| W9 | T9.1–T9.3 Benchmarks | ✅ | `benchmarks/run.py`, `smoke.py`, `plant_zoo.py`, `results/baseline_pre_refactor.json`. |
| W10 | T10.1 Honest README | ✅ **(fixed this pass)** | Rewritten to lead with `PIDAutotuner`, correct Python badge, drop "Perfect for real systems" oversell. |
| W10 | T10.2 `docs/MIGRATION.md` | ✅ | Side-by-side v0.1 → v0.2 diff with CLI section. |
| W10 | T10.3 `docs/ARCHITECTURE.md` | ✅ | Pipeline diagram + extension points. |
| W10 | T10.4 Redundant top-level `.md` removed | ✅ | `IMPLEMENTATION_SUMMARY.md` and `OPTIMIZATION_IMPROVEMENTS.md` are no longer present. |

---

## 2. Gaps found and closed in this pass

### 2.1 Duplicate `__all__` dropped `PIDAutotuner` from wildcard imports *(bug)*

**File**: `pid_control/autotune/__init__.py`

Two `__all__` assignments in a row meant `from pid_control.autotune import *`
silently omitted `PIDAutotuner` (the headline class).  The second
assignment was removed.

```py
# Before (lines 30-73): two __all__ assignments; second overrides first
__all__ = ["PIDAutotuner", "ActuatorLimits", …]
__all__ = ["ActuatorLimits", …]             # ← drops PIDAutotuner

# After: single canonical __all__
__all__ = ["PIDAutotuner", "ActuatorLimits", …]
```

Verified with:

```bash
python -c "import pid_control.autotune as a; assert 'PIDAutotuner' in a.__all__"
```

### 2.2 T1.5 — silent gain clamps in legacy identifier

**File**: `pid_control/identification/system_identifier.py` (lines 693-716)

The legacy `_apply_tuning_rule` still had the unconditional
`max(0, min(abs(k), cap))` clamp that PLAN C1 called out.  Replaced with
a `_bounded(name, value, cap)` helper that emits a `UserWarning`
tagged with `W_GAIN_CLIPPED` whenever the cap actually fires, so
headless scripts can catch it via `warnings.catch_warnings`.

The new `PIDAutotuner` pipeline was already clean (no clamps in
`autotune/rules/*.py`), so this change only affects the deprecated
v0.1 path.

### 2.3 Emit `WarningCode.W_GAIN_CLIPPED` from the DE tuner

**File**: `pid_control/autotune/tuning/de.py` (lines 104-128)

`WarningCode.W_GAIN_CLIPPED` was defined in `types.py` but no stage
was emitting it.  Now, when the DE optimum lands on a user-supplied
`bounds_kp` / `bounds_ki` / `bounds_kd` (within a 1e-6 relative
tolerance), the tuner appends a typed `Warning` with the offending gain
name, value, and bounds in `context`.

### 2.4 T7.3 — legacy demos moved to `examples/advanced/`

PLAN §7 mandated that interactive / heavy demos be moved under
`examples/advanced/`.  They were still at the root.  Moved 12 files:

```
demo_advanced_features.py        demo_animated.py
demo_basic.py                    demo_double_pendulum.py
demo_double_pendulum_autotune.py demo_mass_spring_damper.py
demo_quick_autotune_from_csv.py  demo_realtime_tuner_car_stop.py
demo_simple.py                   demo_spectacular_simulations.py
demo_system_identification.py    demo_tuning.py
```

Every moved file had `sys.path.insert(0, Path(__file__).parent.parent)`
assuming it sat directly under `examples/`.  All 12 were rewritten to
use `Path(__file__).resolve().parents[2]` so imports still resolve to
the repo root from one level deeper.

`run_all_demos.py` at the repo root was updated so its module paths
match the new location (`examples.advanced.demo_*`).  `tools/run_examples.py`
already uses a non-recursive glob, so it automatically runs only the
six golden examples now.

### 2.5 T10.1 — README rewrite

**File**: `README.md`

- Python badge corrected: 3.8+ → 3.10+ (matches `setup.py`).
- Added "Status: Alpha" badge.
- Tagline rewritten from "professional-grade" / "Perfect for real
  systems" → a factual description that highlights the pipeline's
  honest-failure design.
- Lead feature is now `PIDAutotuner`; classical controller description
  demoted to "preserved from v0.1".
- Project structure block updated to show `autotune/`, `cli/`,
  `benchmarks/`, `docs/`, `examples/advanced/` — all of which were
  missing from the old tree.
- Quick-start examples now use `PIDAutotuner.from_csv(...)` and
  `.from_plant(...)` instead of the deprecated `AutotuneFromData` /
  `RealtimeTuner`.
- Incorrect "planned but not yet available" line about CMA-ES / BO
  backends removed — both have been implemented (`tuning/cmaes.py`,
  `tuning/bo.py`).
- Examples list now points at the six numbered scripts and the headless
  CI runner.

---

## 3. New documentation artifacts

| File | Purpose |
|---|---|
| `USAGE_TUTORIAL.md` | New 10-section tutorial: pipeline overview, three input recipes (CSV / plant / arrays), stage customisation, warning taxonomy, headless plotting, CLI, benchmarks, troubleshooting. |
| `CHANGES.md` | *(this file)* Verification log. |

Existing docs (unchanged, re-verified):

- `docs/MIGRATION.md` — v0.1 → v0.2 side-by-side.
- `docs/ARCHITECTURE.md` — pipeline internals and extension points.
- `SYSTEM_IDENTIFICATION_GUIDE.md` — legacy identification details.
- `CSV_COLUMN_MAPPING.md`, `DATA_REQUIREMENTS.md` — CSV specifics.

---

## 4. Verification commands (all green locally)

```bash
# Imports
python -c "from pid_control.autotune import PIDAutotuner, Objective, Status; print('ok')"
python -c "import pid_control.autotune as a; assert 'PIDAutotuner' in a.__all__"

# Fast tests (<10 s)
python -m pytest tests/test_pid_controller.py tests/test_plants.py \
                 tests/test_autotune_types.py tests/test_autotune_data_quality.py \
                 tests/test_autotune_identification.py tests/test_property.py

# Full pipeline tests (~3-4 min on a laptop)
python -m pytest tests/test_autotune_pipeline.py

# CLI
python -m pid_control.cli.main plant fopdt --K 1.5 --tau 3 --theta 0.5 --rule imc
pidtune csv examples/data/fopdt_step.csv -o out.json     # after `pip install -e .`

# Headless examples
$env:MPLBACKEND = "Agg"
python -m tools.run_examples

# Benchmark smoke suite (< 30 s)
python -m benchmarks.smoke
```

Expected output summary:

- 97 tests pass, 0 fail.
- CLI prints a `PID Autotune Result` block with `Status : OK` for a
  well-posed FOPDT plant.
- All six golden examples exit 0 and write artefacts under `./output/`.

---

## 5. Known residual caveats

- **`control` library FutureWarning** — `control>=0.10,<0.12` emits
  `FutureWarning: fresp attribute is deprecated`.  Harmless; will be
  resolved when we bump to `control>=0.11`.
- **Pipeline tests are slow** (~3–4 min) because each runs the full
  DE + `control.margin` stack.  `tests/test_autotune_pipeline.py` is
  opt-in; the fast suite (9 s) is the default in CI's PR job.
- **Legacy `UserWarning("W_GAIN_CLIPPED: …")`** from
  `system_identifier.py` uses Python warnings, not the typed
  `WarningCode.W_GAIN_CLIPPED` — that is intentional because the v0.1
  `IdentificationResult` doesn't carry a warnings tuple.  v0.3 will
  remove the legacy path entirely.
