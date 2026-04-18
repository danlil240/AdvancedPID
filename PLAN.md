AdvancedPID — Principal Engineer's Audit & Roadmap to World-Class

**Status: ALL TASKS COMPLETE (64/64)** — Implementation finished.

Scope: everything under @c:/Users/danli/Projects/AdvancedPID as of 17 Apr 2026. Findings are grounded in code reads and live test/example runs (Python 3.11, numpy 2.2.6, scipy 1.16.0, matplotlib 3.10.3, control 0.10.2, gymnasium 1.2.3, pytest 9.0.2). Bugs reported here were reproduced.

1. Executive Summary
Where the project stands today

Mid-sized library (~7.5k LOC of Python) covering PID core, plants, simulation, analyzer, system identification, Gymnasium wrappers, and a real-time tuner.
The PID core controller (@c:/Users/danli/Projects/AdvancedPID/pid_control/core/pid_controller.py:1-476) is the strongest part: anti-windup, setpoint weighting, filtered derivative, CSV logging, bumpless transfer.
The autotune stack — which the README advertises as the headline feature — is the weakest. It ships three overlapping entry points (AutotuneFromData, RealtimeTuner, SystemIdentifier), silently tunes obviously bad data (verified on a double-integrator plant), has an algorithmically broken relay feedback routine, bounds gains with arbitrary clamps, and offers zero robustness / confidence diagnostics.
Examples are a mix of useful demos and noise (@c:/Users/danli/Projects/AdvancedPID/examples/demo.py:1-286 is unrelated vehicle-steering code that executes at import time; several demos block on input() / plt.show()).
Tests are thin (35 cases, 1 failing — verified).
Docs are overstated ("professional-grade", "perfect for real systems") relative to actual robustness.
The gap to world-class

The library has good primitives but lacks the engineering spine a real autotune framework needs:

No unified API. Every user assembles workflows manually.
No safety layer. Nothing detects PID-inappropriate plants, non-minimum-phase behavior, ill-posed experiments, or untrustworthy fits.
No confidence reporting. Every tuning result looks equally certain.
No benchmark harness. Changes to the tuner cannot be evaluated against a baseline.
Weak separation of concerns. AutotuneFromData mixes CSV I/O, identification, cost design, optimization, clipping, and reporting in one file.
Recommended direction

Refactor into an explicit four-stage pipeline — Experiment → Identify → Tune → Validate — behind a single top-level façade PIDAutotuner. Introduce typed Result objects with warnings + confidence. Add a safety/diagnostics module, a pluggable tuning-rule registry, and a benchmark harness. Do this as a series of bounded, parallelizable child tasks (Section 9). Do not break the existing PID controller class — it is the asset worth preserving.

2. Current Architecture Summary
2.1 Module map
Module	Role	Key files
Core PID	The controller	@c:/Users/danli/Projects/AdvancedPID/pid_control/core/pid_controller.py, @c:/Users/danli/Projects/AdvancedPID/pid_control/core/pid_params.py, @c:/Users/danli/Projects/AdvancedPID/pid_control/core/filters.py
Plants	Simulated processes	@c:/Users/danli/Projects/AdvancedPID/pid_control/plants/*
Envs	Gymnasium wrappers	@c:/Users/danli/Projects/AdvancedPID/pid_control/envs/pid_envs.py
Simulation	Offline sim engine, scenarios	@c:/Users/danli/Projects/AdvancedPID/pid_control/simulation/simulator.py, @c:/Users/danli/Projects/AdvancedPID/pid_control/simulation/scenarios.py
Analyzer	Metrics + plots from CSV/data	@c:/Users/danli/Projects/AdvancedPID/pid_control/analyzer/*
Identification	FOPDT/SOPDT fit from CSV	@c:/Users/danli/Projects/AdvancedPID/pid_control/identification/system_identifier.py, @c:/Users/danli/Projects/AdvancedPID/pid_control/identification/autotune_from_data.py
Tuner	Numerical PID optimization	@c:/Users/danli/Projects/AdvancedPID/pid_control/tuner/realtime_tuner.py, @c:/Users/danli/Projects/AdvancedPID/pid_control/tuner/optimization_methods.py
Logging	CSV output for live loops	@c:/Users/danli/Projects/AdvancedPID/pid_control/logging/*
2.2 How a user gets from "I have data / a plant" to tuned gains today
There are three independent user-facing paths, each with subtly different assumptions:

CSV → gains — AutotuneFromData(csv_path).autotune() in @c:/Users/danli/Projects/AdvancedPID/pid_control/identification/autotune_from_data.py:83-219. Runs: read CSV → SystemIdentifier.identify(AUTO) → ZN/Cohen-Coon/IMC initial gains → scipy DE on a hand-rolled closed-loop sim of the identified model. Reports "improvement %".
Simulation-in-the-loop — RealtimeTuner(ctrl, plant).auto_tune(setpoint, duration) in @c:/Users/danli/Projects/AdvancedPID/pid_control/tuner/realtime_tuner.py:227-278. Given a plant object, runs a full closed-loop sim per candidate and minimizes a weighted-sum cost.
Classical rules only — RealtimeTuner.ziegler_nichols_step(...), RealtimeTuner.relay_feedback_tune(...), or FOPDTPlant.get_tuning_suggestions() for users who already have a model.
Nothing stitches these together or tells the user which one to use.

2.3 Documentation / onboarding surface
Top-level .md files: README.md, CSV_COLUMN_MAPPING.md, DATA_REQUIREMENTS.md, IMPLEMENTATION_SUMMARY.md, OPTIMIZATION_IMPROVEMENTS.md, SYSTEM_IDENTIFICATION_GUIDE.md, plus c:\Users\danli\Projects\AdvancedPID/examples/AUTOTUNING_GUIDE.md, c:\Users\danli\Projects\AdvancedPID/examples/SYSTEM_IDENTIFICATION_EXAMPLES.md. Content is generally decent but scattered; the README oversells capabilities (claims "Perfect for real systems") without warning about known limitations.

3. Key Weaknesses, Ranked
Issues below are reproduced, with file:line citations. Ranking follows Critical → High → Medium → Low.

3.1 Critical
C1. Silent failure when plant is PID-inappropriate. Verified. Reproduced on a pure double-integrator dataset: SystemIdentifier.identify(AUTO) reported SECOND_ORDER fit with R²=0.992, autotuner reported "success" with 0.00% improvement — but internally the cost function returned 1e10 every single evaluation and DE quit after 1 iteration. The "optimized" Ki was silently clipped to the hardcoded ceiling 100.0.

Cost gate at @c:/Users/danli/Projects/AdvancedPID/pid_control/identification/autotune_from_data.py:368-369 sets cost to 1e10 with no diagnostic; optimizer quits and claims success.
Hardcoded ceiling: @c:/Users/danli/Projects/AdvancedPID/pid_control/identification/system_identifier.py:693-695 clips kp, ki, kd to [0,1000] / [0,100] / [0,100] without any warning.
Violates the user's own non-negotiable rule: "Do not silently tune obviously bad data."
C2. relay_feedback_tune is algorithmically broken. Verified on FOPDT: RuntimeError "Could not establish oscillation".

@c:/Users/danli/Projects/AdvancedPID/pid_control/tuner/realtime_tuner.py:510-596. The relay is applied as an absolute control value and the loop expects error to cross zero around setpoint. If setpoint > plant_gain*relay_amplitude, the plant can never reach the setpoint and the relay never switches. Real relay auto-tune biases the control around the current operating point, not around an arbitrary setpoint.
Ku = 4 * relay_amplitude / (π * relay_amplitude * 0.5) is mathematically wrong (the classic formula is Ku = 4d / (π·a) where a is the oscillation amplitude of the output, not a fraction of the input). The code never measures output amplitude.
C3. ziegler_nichols_step uses raw open-loop slope incorrectly. @c:/Users/danli/Projects/AdvancedPID/pid_control/tuner/realtime_tuner.py:434-508.

Applies a constant step and assumes responses[0] == 0 (no bias correction).
Computes tau = final_value / max_slope — missing the -L term of the tangent method. For FOPDT: T_ss_tangent = L + τ, so τ = y∞/slope_max − L. Current code can give τ off by up to a time-constant, propagating into Kp, Ki, Kd.
No sanity check on plant actually reaching steady state.
C4. error_deadband is bypassed by the P-term. Verified — test currently FAILS.

tests/test_pid_controller.py::TestPIDController::test_error_deadband asserts output == 0.0 but gets 3.0.
Root cause: @c:/Users/danli/Projects/AdvancedPID/pid_control/core/pid_controller.py:193-208 applies deadband to error (used for I-term) but P-term uses an independently computed error_p = setpoint_weight_p * setpoint - measurement that ignores the deadband entirely. D-term has a similar independent path at lines 289-314.
C5. demo.py pollutes examples and runs at import time.

@c:/Users/danli/Projects/AdvancedPID/examples/demo.py:1-286 is unrelated python-control optimal-control vehicle-steering code with no if __name__ == "__main__": guard — everything executes on import. Running importlib.import_module("examples.demo") executes a multi-second optimization just by being imported. Also writes 2 MB of .log files at the repo root (observed steering-*.log files in the workspace).
3.2 High
H1. Three overlapping autotune APIs, no recommended path. README shows two totally different workflows; SystemIdentifier vs AutotuneFromData vs RealtimeTuner have no shared result type. AutotuneFromData and RealtimeTuner duplicate: closed-loop simulation, cost function, DE optimizer wiring. Users cannot reliably pick the right tool.

H2. Cost function baked into AutotuneFromData is brittle and hardcoded.

@c:/Users/danli/Projects/AdvancedPID/pid_control/identification/autotune_from_data.py:297-397:
Hardcoded setpoint = 1.0, no relation to the user's data.
Hardcoded np.clip(u, -10, 10) — disconnected from the actual actuator limits.
Hardcoded failure threshold max|y|>100 or max|u|>50 → returns 1e10. Arbitrary.
Euler integration (unconditionally), even though identification uses signal.dlsim.
Magic weights ise*100 + iae*10 + overshoot*50 + settling*2 + TV*0.1.
H3. No robustness scoring. Zero analysis of phase/gain margin, sensitivity peak Ms, or complementary sensitivity Mt. A tuned result could have margins near zero and be reported identically to a safe tuning.

H4. No confidence / quality flags on the identification.

SOPDT auto-selection often degenerates into FOPDT + a fast fake pole (verified: on a pure FOPDT dataset, _optimize_model(SOPDT) returned tau2=0.041 with BadCoefficients warning — @c:/Users/danli/Projects/AdvancedPID/pid_control/identification/system_identifier.py:420-483). R² alone is reported; no check for tau2 ≪ sample_time, imaginary poles, negative gain vs physics, non-minimum-phase residuals, or noise floor.
H5. System ID picks the best-R² model without penalizing complexity.

@c:/Users/danli/Projects/AdvancedPID/pid_control/identification/system_identifier.py:139-157 picks max(fit_quality) across FOPDT/SOPDT/SECOND_ORDER. Higher-order models will always win on noisy data → overfitting. Needs AIC/BIC or F-test-style comparison.
H6. Relay/Ziegler-Nichols live experiments do not close the loop for excitation and can damage plants. No safety limits on open-loop step amplitude, no ramping, no abort conditions. Not deployable on real hardware.

H7. RealtimeTuner.tune_from_data assumes the plant is a resettable simulator. @c:/Users/danli/Projects/AdvancedPID/pid_control/tuner/realtime_tuner.py:356-432 resets self._plant and re-simulates — which is meaningless if _plant is a real device. The method pretends to be "from data" but cannot work without a model.

H8. success flag is unreliable.

RealtimeTuner.auto_tune returned success=False on my FOPDT test (DE hit max-iter). With apply_result=True, it silently does nothing — @c:/Users/danli/Projects/AdvancedPID/pid_control/tuner/realtime_tuner.py:270-277. The user has no signal that their controller wasn't updated.
3.3 Medium
M1. Plant update() returns measurement (noise+disturbance applied), but plant.output returns the clean state. This is inconsistent across plant classes and leads to subtle bugs when users read plant.output directly.

**M2. Simulator.run calls plant.set_noise(scenario.measurement_noise_std) mutating plant state — @c:/Users/danli/Projects/AdvancedPID/pid_control/simulation/simulator.py:115. Reusing the same plant in another simulation silently inherits noise from the last scenario.

M3. Reproducibility is inconsistent. DifferentialEvolutionTuner hardcodes seed=42 (@c:/Users/danli/Projects/AdvancedPID/pid_control/tuner/optimization_methods.py:461) while BayesianTuner uses np.random.random without seeding, and GeneticTuner depends on callers setting np.random.seed.

M4. IdentificationResult.recommended_gains clamps to arbitrary [0,1000]/[0,100] without telling anyone. Documented above (C1 tie-in).

M5. AutotuneFromData and SystemIdentifier duplicate FOPDT/SOPDT/SECOND_ORDER simulation code. @c:/Users/danli/Projects/AdvancedPID/pid_control/identification/system_identifier.py:554-602 vs @c:/Users/danli/Projects/AdvancedPID/pid_control/identification/autotune_from_data.py:275-346. Drift risk.

M6. BayesianTuner is a fake GP. Kernel-weighted average is not a proper Gaussian process; sigma is noise, not posterior std. Calling it "Bayesian optimization" is misleading. Either fix it (use scikit-learn's GaussianProcessRegressor) or rename it.

M7. CSV column default mismatch vs generic CSVs. CSVDataReader.read default columns are time_col='timestamp', input_col='output', output_col='measurement' — these match the library's own CSVLogger but not typical experimental data, where users expect time, input, output. Confusing for outside data.

M8. Examples mix interactive input() calls with programmatic demos. demo_system_identification.py uses input() — unusable for CI, batch runs, or automated verification.

M9. Blocking plt.show() in visualizer functions. @c:/Users/danli/Projects/AdvancedPID/pid_control/identification/visualizer.py:119,245,295 always call plt.show() even when save_path was provided — forces users into interactive runs.

M10. Test coverage gaps. Only 35 tests total; none for the tuner, optimizer, identification, simulation loop, scenarios, or analyzer. One test is failing.

M11. PIDController constructs DerivativeFilter with the initial sample_time and never respects a later change unless set_params is called. Users can accidentally call the controller with a different timestamp spacing and the filter math (which bakes dt) quietly desynchronizes.

M12. PIDParams validation rejects negative kp/ki/kd. For processes with negative gain (e.g., some HVAC), a negative Kp is the correct choice. This is a usability trap — user must wrap-and-invert.

3.4 Low
L1. ControllerType enum is defined but unused (@c:/Users/danli/Projects/AdvancedPID/pid_control/core/pid_params.py:26-32). L2. PIDState.error_filtered == error — the field exists in the dataclass but is just a copy (@c:/Users/danli/Projects/AdvancedPID/pid_control/core/pid_controller.py:237-238). L3. README 📁 Project Structure is stale (missing envs/, identification/). L4. Top-level repo has 2 MB of steering-*.log files from a rogue demo run. L5. c:\Users\danli\Projects\AdvancedPID/advanced_pid_control.egg-info/ is committed (should be gitignored). L6. numpy.trapz is deprecated in recent numpy (@c:/Users/danli/Projects/AdvancedPID/pid_control/analyzer/metrics.py:139-142) — use numpy.trapezoid (ships in numpy 2.x). L7. No type checking. from typing import ... is used but no mypy/pyright config; # type: ignore etc. absent. L8. setup.py's install_requires does not include control or gymnasium even though they're required — requirements.txt does. Pip-installing the package will fail at import time.

4. Target Architecture Proposal
The design principle: a four-stage pipeline with explicit boundaries. Users interact with one façade; power users plug in or replace any stage.

pid_control/
├── core/                       # PRESERVE: controller + params (minimal fixes only)
├── plants/                     # PRESERVE: simulated plants
├── envs/                       # PRESERVE: Gymnasium wrappers
│
├── autotune/                   # NEW: unified, staged pipeline
│   ├── __init__.py             #   exports PIDAutotuner + dataclasses
│   ├── api.py                  #   PIDAutotuner façade (one-line usage)
│   ├── types.py                #   TuneRequest, TuneResult, Diagnostics, Confidence
│   ├── experiments/            #   Stage 1: excitation design
│   │   ├── base.py             #     Experiment protocol
│   │   ├── step.py             #     open-loop + biased step
│   │   ├── relay.py            #     CORRECT biased relay (ATV / autotune variation)
│   │   ├── chirp.py            #     linear + log sine sweep
│   │   └── from_data.py        #     wraps pre-recorded CSV
│   ├── identification/         #   Stage 2: plant characterization
│   │   ├── base.py             #     Identifier protocol + TransferFunctionModel
│   │   ├── fopdt.py            #     Standalone FOPDT (two-point + tangent + NLLS)
│   │   ├── sopdt.py            #     SOPDT with regularization for tau2≥k*dt
│   │   ├── integrator.py       #     Integrator+dead-time detection (IPDT)
│   │   ├── delay_estimator.py  #     Cross-correlation delay
│   │   └── quality.py          #     AIC/BIC, residual tests, noise estimation
│   ├── rules/                  #   Stage 3a: analytical tuning-rule registry
│   │   ├── registry.py
│   │   ├── ziegler_nichols.py
│   │   ├── cohen_coon.py
│   │   ├── imc.py, lambda_.py, amigo.py, skogestad.py
│   ├── tuning/                 #   Stage 3b: numerical refinement
│   │   ├── base.py             #     Tuner protocol
│   │   ├── de.py, nm.py, bo.py
│   │   └── cost.py             #     composable cost terms (IAE, ITAE, Ms, Mt, du)
│   ├── validation/             #   Stage 4: post-tune safety/quality gates
│   │   ├── stability.py        #     closed-loop poles / Ms, Mt, gain/phase margins
│   │   ├── robustness.py       #     gain/delay perturbation tests
│   │   ├── sim_benchmark.py    #     standard reference scenarios
│   │   └── confidence.py       #     aggregate → Confidence score + warnings
│   └── diagnostics/            #   Cross-cutting
│       ├── data_quality.py     #     SNR, excitation energy, steady-state check
│       └── reporters.py        #     Markdown / JSON / HTML reports
│
├── simulation/                 # PRESERVE + cleanup (noise-mutation fix)
├── analyzer/                   # PRESERVE + add margin/sensitivity metrics
├── logging/                    # PRESERVE
├── io/                         # NEW: CSVReader moved here, format-agnostic
└── cli/                        # NEW: `pidtune` entry point
 
tests/                          # Greatly expanded; one-file-per-module
├── test_core/                  # controller, params, filters
├── test_plants/
├── test_autotune/
│   ├── test_api.py
│   ├── test_experiments_*.py
│   ├── test_identification_*.py
│   ├── test_tuning_*.py
│   └── test_validation_*.py
├── test_simulation/
└── benchmarks/                 # Regression benchmarks (Phase 6)
 
examples/                       # Pruned + restructured (Section 7)
├── 01_quickstart.py
├── 02_from_csv.py
├── 03_from_plant.py
├── 04_relay_autotune_live.py
├── 05_custom_cost.py
└── advanced/*
Design principles baked into the target
Everything returns typed results. No dicts-of-dicts.
Every stage can fail loudly. TuneResult.status ∈ {OK, WARNING, FAILED}; warnings are enumerated.
Identification ≠ Tuning. A user can do ID only, tuning only, or the full pipeline.
No hidden clamps. If gains are clipped, a warning is attached to the result.
Confidence is a first-class output. A scalar in [0,1] with contributing factors.
Experiments are safe by default. Bounded amplitude/rate; abort-on-runaway; dry-run mode.
Deterministic unless the user asks otherwise. All stochastic paths take rng: np.random.Generator.
5. Proposed API Design
5.1 Beginner — one-liner with strong defaults
python
from pid_control.autotune import PIDAutotuner
 
# From CSV
result = PIDAutotuner.from_csv("heater_step.csv").tune()
print(result.report())             # human-readable summary
controller = result.build_controller()  # ready-to-run PIDController
 
# From simulated/known plant
from pid_control.plants import FOPDTPlant
plant = FOPDTPlant(gain=1.5, time_constant=3.0, dead_time=1.0)
result = PIDAutotuner.from_plant(plant).tune()
5.2 Intermediate — tell it what you care about
python
from pid_control.autotune import PIDAutotuner, Objective
 
result = (
    PIDAutotuner.from_csv("heater_step.csv")
    .with_objective(
        Objective(
            max_overshoot_pct=5.0,
            max_settling_time=4.0,
            min_phase_margin_deg=45,
            min_Ms=1.4,           # sensitivity peak
            control_effort_weight=0.1,
        )
    )
    .with_actuator_limits(lower=-10.0, upper=10.0, rate_limit=50.0)
    .tune()
)
 
if result.status is Status.OK:
    controller = result.build_controller()
else:
    for w in result.warnings:
        print(w.level, w.code, w.message)
5.3 Advanced — plug in your own stages
python
from pid_control.autotune import PIDAutotuner
from pid_control.autotune.experiments import RelayExperiment
from pid_control.autotune.identification import FOPDTIdentifier
from pid_control.autotune.tuning import DifferentialEvolutionTuner, CostSpec
 
tuner = (
    PIDAutotuner(plant)
    .set_experiment(RelayExperiment(
        bias="operating_point", amplitude=0.05, hysteresis=0.005,
        n_cycles=6, abort_if_abs_output_gt=50.0,
    ))
    .set_identifier(FOPDTIdentifier(method="two_point", multistart=5))
    .set_tuner(DifferentialEvolutionTuner(
        cost=CostSpec(iae=1.0, itae=0.0, du=0.01, Ms_penalty_above=1.6),
        max_iter=80, seed=0,
    ))
    .add_validator("margins", "robustness", "sim_benchmark")
)
result = tuner.tune()
5.4 Core result objects
python
@dataclass(frozen=True)
class TuneResult:
    gains: PIDGains                      # kp, ki, kd + filter N, setpoint weights
    status: Status                       # OK | WARNING | FAILED
    confidence: float                    # [0, 1]
    identification: IdentificationResult # model + quality
    performance: PerformanceReport       # rise, settling, overshoot, IAE, Ms, Mt, margins
    warnings: tuple[Warning, ...]        # typed & coded
    artifacts: Artifacts                 # trajectories, residuals, for plotting
    meta: TuneMeta                       # versions, seeds, timing, cost history
 
    def report(self, fmt: Literal["md","json","text"] = "text") -> str: ...
    def plot(self, kind: Literal["fit","margins","response","all"] = "all"): ...
    def build_controller(self, **overrides) -> PIDController: ...
    def save(self, path: str | Path) -> None: ...
5.5 Warning taxonomy (stable codes users can branch on)
Code	Level	Meaning
E_DATA_FLAT	ERROR	Input has no excitation
E_NO_STEADY_STATE	ERROR	Response never settled
E_UNSTABLE	ERROR	Closed-loop model unstable
W_POOR_FIT	WARNING	R² or AIC below threshold
W_DEGENERATE_SOPDT	WARNING	Second pole ≪ sample time
W_GAIN_CLIPPED	WARNING	Result clamped at bound
W_LOW_MARGIN	WARNING	Phase margin < 30° or Ms > 2
W_NONMIN_PHASE	WARNING	Initial inverse response detected
W_SATURATION	INFO	Actuator saturated in simulation
5.6 CLI
$ pidtune csv heater.csv --objective overshoot=5,settle=4 --out heater_tune.json
$ pidtune plant fopdt --K 1.5 --tau 3.0 --theta 1.0 --format markdown
$ pidtune bench --suite standard --save bench.json
6. Algorithm Upgrade Roadmap
Scoped, measurable, non-fluffy. [P] = priority; [T] = estimated size (S=≤1d, M=1-3d, L=3-7d).

6.1 Excitation / Experiment design
[P1, M] Fix and re-implement relay auto-tune (C2). Use biased ATV (auto-tune variation): hold operating output u0, toggle u = u0 ± d, measure output oscillation amplitude a_y, compute Ku = 4d / (π·a_y), Tu from zero-crossings. Add hysteresis on output, not error. Add abort-on-runaway.
[P1, S] Fix and harden ZN step test (C3). Correct tangent math: τ = y∞/s_max − L. Require explicit pre-step dwell; measure baseline; warn if plant not settled.
[P2, M] Chirp / log-sweep experiment. Frequency-domain fit for lightly damped and higher-order plants.
[P2, S] Closed-loop step experiment. Optional "around existing gains" mode — useful when the plant cannot be opened.
[P3, S] Safety wrapper. Amplitude/rate clamp + abort thresholds for live experiments on hardware.
6.2 Plant characterization
[P1, S] Degenerate-SOPDT detector (H4). Flag tau2 < k·sample_time or tau2/tau1 < 0.05; drop to FOPDT.
[P1, M] AIC/BIC-based model selection (H5). Replace max(R²) with information-criterion comparison.
[P1, S] Integrator detection. If residual output drifts with constant input → flag IPDT candidate; optionally fit integrator+delay model.
[P1, S] Data quality score. SNR, excitation energy, steady-state-reached check — returned as DataQuality dataclass; blocks tuning if ERROR-level.
[P2, M] Non-minimum-phase detection. Inverse-response flag from early residual sign.
[P2, S] Noise variance estimation. Use pre-step baseline; propagate to fit-quality confidence.
[P3, M] Subspace / ARX / N4SID identification using control or sippy for users with rich data.
6.3 Tuning strategies
[P1, S] Tuning-rule registry. Move all ZN/CC/IMC/Lambda into autotune/rules/ with one pluggable interface; add AMIGO and Skogestad SIMC. Remove the silent [0,1000]/[0,100] clamps; instead return W_GAIN_CLIPPED only when a user-supplied bound is hit.
[P1, M] Unified numerical tuner. One Tuner abstract base class; fold RealtimeTuner and AutotuneFromData optimization into a single engine. DE, NelderMead, CMA-ES, BO as backends.
[P1, M] Composable cost. CostSpec(iae=..., itae=..., overshoot_above=..., du=..., Ms_penalty_above=..., settling_time_weight=...). Each term documented, unit-tested.
[P1, S] Real setpoint and actuator limits in cost. Pull from Objective + ActuatorLimits; do not hardcode.
[P2, M] 2-DOF PID tuning. Optimize (Kp, Ki, Kd, b, c, N) with sensible priors on b, c, N.
[P2, M] Anti-windup aware cost. Simulate with the same anti-windup the user will use in production.
[P2, S] Derivative-filter aware tuning. Tune N jointly or set from noise estimate.
6.4 Validation & safety
[P1, M] Margin computation. Use python-control to compute gain margin, phase margin, Ms, Mt, crossover frequencies on the identified loop. Add thresholds to Objective.
[P1, M] Robustness stress tests. Perturb identified K, τ, θ by ±20% / ±50%; re-simulate; report worst-case IAE + stability retained. Flag W_FRAGILE if worst-case explodes.
[P1, S] Reject clearly-unsafe tunings. If margin < configurable floor (default PM>30°, Ms<2.0) → downgrade to WARNING at minimum; gate build_controller() behind an opt-in if FAILED.
[P2, M] Confidence score aggregator. Weighted combination of fit quality, robustness, margin, noise-aware information-gain; deterministic and documented.
6.5 Output quality
[P1, S] Standard reports. Text + Markdown + JSON TuneResult.report().
[P1, M] Standard plots. Fit overlay, margin / Bode, step & disturbance response, cost history, robustness sweep. One result.plot() generates them all.
[P2, S] Save/load. TuneResult.save() / load() — freeze for audit.
7. Example Verification Plan
7.1 Current examples matrix (based on direct inspection; import-tested for all, runtime-tested for the core autotune paths)
#	File	Intent	Observed status	Root cause / note
1	examples/demo.py	??	BROKEN / OFF-TOPIC	Unrelated vehicle-steering code; runs on import; writes steering-*.log at repo root. Delete.
2	examples/demo_basic.py	Minimal PID step	Likely OK (functional), but launches a blocking Gym render window that runs 3000 steps; awkward for headless use.	
3	examples/demo_simple.py	Very minimal Gym sim	OK functionally, render_mode="human" blocks without display.	
4	examples/demo_tuning.py	DE autotune on FOPDT	Runs, but the final "live render" at line @c:/Users/danli/Projects/AdvancedPID/examples/demo_tuning.py:144-156 runs 5000 steps and never exits cleanly in headless contexts. Also uses RealtimeTuner which has the success=False trap (H8).	
5	examples/demo_advanced_features.py	Anti-windup / DOF / bumpless	Likely OK, 4 sub-demos all call plt.show() once at the end — fine interactive, slow on batch.	
6	examples/demo_system_identification.py	Full ID + autotune	UNUSABLE in batch: blocking input() calls at multiple points @c:/Users/danli/Projects/AdvancedPID/examples/demo_system_identification.py:299-317.	
7	examples/demo_quick_autotune_from_csv.py	CSV quick-start	UNUSABLE in batch: input() at @c:/Users/danli/Projects/AdvancedPID/examples/demo_quick_autotune_from_csv.py:56.	
8	examples/demo_realtime_tuner_car_stop.py	RealtimeTuner on friction plant	Runs, but opens blocking Gym render window. Demonstrates H8 silently.	
9	c:\Users\danli\Projects\AdvancedPID/examples/demo_double_pendulum.py	Hand-tuned double pendulum	Runs interactively.	
10	examples/demo_double_pendulum_autotune.py	GA autotune of pendulum	Runs but contains interactive input() at @c:/Users/danli/Projects/AdvancedPID/examples/demo_double_pendulum_autotune.py:432-437.	
11	c:\Users\danli\Projects\AdvancedPID/examples/demo_mass_spring_damper.py	MSD visualization	Not runtime-verified.	
12	c:\Users\danli\Projects\AdvancedPID/examples/demo_animated.py	Live animation	Interactive only.	
13	c:\Users\danli\Projects\AdvancedPID/examples/demo_spectacular_simulations.py	3D viz showcase	Interactive only; heavy.	
All 13 modules import cleanly (verified). The failures are in UX (interactive traps, blocking renders) and in the underlying autotune (silent failures, see §3).

7.2 Proposed "golden" set (minimum viable examples)
New c:\Users\danli\Projects\AdvancedPID/examples/ layout, all runnable headlessly, all producing saved artifacts under ./output/:

01_quickstart_plant.py — from-plant autotune, 15 LOC, writes output/quickstart.md + quickstart.png.
02_quickstart_csv.py — from-CSV autotune (ships sample CSV in examples/data/fopdt_step.csv).
03_from_plant_objective.py — tuning with overshoot/settling/margin constraints.
04_relay_autotune.py — biased-ATV relay autotune (live experiment, simulated).
05_compare_rules.py — ZN vs Cohen-Coon vs IMC vs optimized, overlay plot.
06_diagnosing_bad_data.py — demonstrates an integrator dataset being rejected with a warning (fixing C1).
Advanced:

advanced/custom_cost.py, advanced/custom_experiment.py, advanced/double_pendulum_autotune.py, advanced/live_animation.py.
7.3 Standardized launcher
Every example:

ends with if __name__ == "__main__": main(),
accepts --show/--no-show (default --no-show when MPLBACKEND=Agg),
takes no input() calls,
writes artifacts to a configurable --output dir.
A tools/run_examples.py runner discovers examples/*.py, runs each with --no-show, captures stdout/stderr, asserts zero errors, and emits a report. Used in CI.

7.4 Reproduction commands
# Install dev deps
python -m pip install -e ".[dev]"
 
# Run all unit tests
python -m pytest tests/ -v
 
# Run all examples headlessly (CI mode)
$env:MPLBACKEND="Agg"; python tools/run_examples.py
 
# Run one example
python examples/01_quickstart_plant.py --output ./output
8. Benchmarking Strategy
8.1 Plant zoo (reference battery)
Category	Plants (parametric)
First-order	FOPDT(K, τ, θ) grid over τ/θ ∈ {0.2, 1, 3, 10}, K ∈ {0.2, 1, 5}
Second-order	SOPDT(K, τ₁, τ₂, θ); underdamped (ζ=0.3), critical, overdamped
Integrating	IPDT(K, θ)
Unstable	(K)/(τs−1) with θ (smoke test only; PID may not suffice)
Nonminimum-phase	(1−βs)·G(s) with FOPDT core
Nonlinear	saturation, dead-zone, backlash, stiction (use existing NonlinearPlant, FrictionPlant)
Noise / disturbance	Gaussian, 1/f, step load, ramp load
Actuator saturation	limits ±1, ±10, ±∞
8.2 Metrics
Per (plant, controller, scenario):

Time-domain: rise time, settling time (2%, 5%), overshoot, steady-state error.
Integral: IAE, ISE, ITAE.
Effort: control TV, RMS, peak, % saturation time.
Robustness: Ms, Mt, gain margin, phase margin, delay margin.
Worst-case across plant perturbations (±20%, ±50% of K, τ, θ).
8.3 Reference controllers
baseline_zn: Ziegler-Nichols from true FOPDT.
baseline_imc: IMC λ = τ from true FOPDT.
current_autotune: repo's AutotuneFromData.autotune() with ZN + DE.
new_autotune: proposed pipeline, default settings.
new_autotune_strict: proposed pipeline with margin-first objective.
8.4 Harness design
benchmarks/run.py runs the full matrix; caches seeds; writes a JSON file benchmarks/results/<date>.json and a Markdown summary.
Compare run N against run M; emit a diff with regressions highlighted.
Acceptance thresholds for promotion of the new autotune:
Safety: zero "claimed success" on plants where no PID stabilization exists (integrator+delay with bad ratio, unstable plants). Compared to baseline which silently produces numbers.
Robustness: Ms ≤ 1.6 on at least 80% of FOPDT plants with default objective.
Performance: median IAE improvement ≥ 0% vs baseline_imc, ≥ 20% vs baseline_zn.
Reproducibility: identical seeds → identical gains to 1e-6.
Speed: one full FOPDT autotune in < 5 s for default settings.
8.5 Continuous regression
A benchmarks/smoke.py with a small subset (~20 plants, < 30 s) runs on every PR; full matrix nightly.

9. Detailed Child-Task Execution Plan
Designed to be parallelizable across sub-agents / contributors without stepping on each other. Tasks are ordered by dependency layer (Layer 0 = no deps). Every task has: purpose, files, inputs, outputs, deps, DoD, tests.

Mission summary
Convert AdvancedPID into a small, safe, professional PID autotuning framework with one unified façade, typed results, explicit safety/confidence, a benchmark harness, and a cleaned example set. Preserve the existing PID controller core.

Current state → Target state
From: 3 overlapping autotune APIs, silent failures, algorithmic bugs in relay/ZN, sparse tests, mixed examples.
To: one PIDAutotuner façade, typed TuneResult with warnings/confidence, safe experiments, validated tunings, benchmark harness, cleaned examples.
Workstreams
W1 API Redesign & Façade
W2 Identification correctness
W3 Experiments (excitation) correctness + safety
W4 Tuning engine + cost + rules registry
W5 Validation (margins, robustness, confidence)
W6 Reporting, plotting, CLI
W7 Examples cleanup + runner
W8 Test suite expansion
W9 Benchmarks + promotion gate
W10 Docs & migration
Dependency graph (top-level)
W1 ─┬─► W2, W3, W4, W5  (API types must land first)
    └─► W6 (depends on W1 + W5)
W2 ┐
W3 ┼─► W4 (tuner consumes ID + experiments)
W4 ┘
W4 + W5 ─► W9 (benchmarks need the full pipeline)
Everything ─► W7 (examples use final API) ─► W10 (docs reflect reality)
W8 runs in parallel with W2–W5.
Milestones
M0 (week 0.5): Housekeeping & safety net. C4 fix, C5 fix, egg-info gitignored, tests green.
M1 (week 1–2): API skeleton. W1 done; empty stubs compile.
M2 (week 2–4): Correct algorithms. W2, W3, W4 done; old API emulated on top.
M3 (week 4–5): Safety & reporting. W5, W6 done.
M4 (week 5–6): Examples + benchmarks. W7, W8, W9 done.
M5 (week 6–7): Docs & release. W10 done; v0.2.0 cut.
Child tasks
W0 — Housekeeping & safety net (all [P0])
T0.1 Fix error_deadband P-term bypass (C4). Files: pid_control/core/pid_controller.py. DoD: existing failing test passes + new test for D-term under deadband + identical output when deadband=0. [S]
T0.2 Remove examples/demo.py (C5) and purge steering-*.log from repo root. Update .gitignore. DoD: grep steering *.log returns nothing; no import-time side effects anywhere in c:\Users\danli\Projects\AdvancedPID/examples/. [S]
T0.3 Gitignore c:\Users\danli\Projects\AdvancedPID/advanced_pid_control.egg-info/ + fix setup.py install_requires to include control, gymnasium. Add python_requires>=3.10. DoD: pip install -e . and python -c "import pid_control" succeed on a fresh venv. [S]
T0.4 Migrate np.trapz → np.trapezoid (L6). [S]
T0.5 Add CI config (.github/workflows/ci.yml) running pytest + example runner on push. [S]
W1 — API redesign
T1.1 Define pid_control/autotune/types.py: PIDGains, TransferFunctionModel, DataQuality, IdentificationResult, PerformanceReport, Warning, Status, Confidence, TuneResult, Objective, ActuatorLimits. All frozen dataclasses. [M]
T1.2 Define stage protocols (Experiment, Identifier, TuningRule, Tuner, Validator) under pid_control/autotune/{experiments,identification,rules,tuning,validation}/base.py. [S]
T1.3 Implement PIDAutotuner façade (autotune/api.py): from_csv, from_plant, from_data_arrays, .with_objective, .with_actuator_limits, .set_experiment/identifier/tuner, .tune(). Composes stages. [M] — depends on T1.1, T1.2.
T1.4 Back-compat shim: keep AutotuneFromData and RealtimeTuner.auto_tune importable; wrap the new engine underneath with a DeprecationWarning. [S]
T1.5 Remove silent gain clamps (C1 partial, M4): replace unconditional [0,1000]/[0,100] clamp in system_identifier._apply_tuning_rule with an _apply_bounds(gains, bounds, warnings) that appends a W_GAIN_CLIPPED warning. [S]
W2 — Identification correctness
T2.1 Extract FOPDT identifier into autotune/identification/fopdt.py. Keep two-point + tangent + NLLS multi-start. [M]
T2.2 Extract SOPDT identifier into sopdt.py with regularizer: reject tau2 < 5·dt or tau2/tau1 < 0.05 → raise DegenerateModel with W_DEGENERATE_SOPDT. Suppress BadCoefficients warning in the API layer and surface it as a typed warning instead (H4). [M]
T2.3 Integrator + dead-time identifier (ipdt.py). Detect non-returning step response. [M]
T2.4 Model selection via AIC/BIC (identification/quality.py): replace max(R²) in identify(AUTO) (H5). [M]
T2.5 Data-quality scoring (diagnostics/data_quality.py): SNR, excitation energy, steady-state check. Blocks tune() with E_DATA_FLAT or E_NO_STEADY_STATE on ERROR. [M]
T2.6 Delay estimation via cross-correlation (delay_estimator.py). Use as initial guess for FOPDT/SOPDT. [S]
T2.7 Noise variance from pre-step baseline; feed into confidence aggregator. [S]
W3 — Experiments correctness & safety
T3.1 Rewrite relay feedback (C2): experiments/relay.py with biased ATV, hysteresis on output, output-amplitude measurement, correct Ku = 4d/(π·a_y), proper Tu from zero-crossing intervals. Unit-test on FOPDT and SOPDT fixtures. [M]
T3.2 Rewrite open-loop step (C3): experiments/step.py with tangent τ = y∞/s_max − L correction, explicit pre-step baseline, ramp entry option. [S]
T3.3 Chirp experiment (chirp.py) with log/linear sweep. [M]
T3.4 FromDataExperiment wrapping CSV/array data as a virtual experiment for downstream stages. [S]
T3.5 Safety wrapper (experiments/safety.py): amplitude clamp, rate limit, abort-on-exceedance. All live experiments composed with it. [S]
W4 — Tuning engine, cost, rules
T4.1 Tuning-rule registry (autotune/rules/{registry,ziegler_nichols,cohen_coon,imc,lambda_,amigo,skogestad}.py). Each rule is a pure function (model, config) -> PIDGains. Remove duplication across FOPDTPlant.get_tuning_suggestions, SystemIdentifier._apply_tuning_rule. [M]
T4.2 Unified numerical tuner base (tuning/base.py) with DE, NelderMead, CMAES, BO backends (BO: real sklearn.gaussian_process). Deprecate BayesianTuner's fake GP (M6). [L]
T4.3 Composable cost (tuning/cost.py): CostSpec(iae, itae, du, overshoot_above, settling_pct_above, Ms_penalty_above, Mt_penalty_above); each term individually unit-tested. Use the user's real setpoint + ActuatorLimits; delete hardcoded setpoint=1.0, np.clip(u,-10,10) (H2). [M]
T4.4 Delete duplicated closed-loop sim code in autotune_from_data.py (M5); route through one helper autotune/sim/fixed_step.py. [S]
T4.5 Honest status: if DE/NM hits maxiter but improved from initial → Status.OK with W_MAXITER; if cost never improved → Status.FAILED. Fixes H8. [S]
W5 — Validation
T5.1 Margin computation (validation/stability.py): build closed-loop transfer function from identified model + tuned gains with control, compute Ms, Mt, GM, PM, delay margin. [M]
T5.2 Robustness sweep (validation/robustness.py): perturb K, τ, θ by user-configurable fractions; re-simulate; report worst-case IAE + stability. [M]
T5.3 Simulation benchmark (validation/sim_benchmark.py): step, disturbance, noise scenarios against identified model. [S]
T5.4 Confidence aggregator (validation/confidence.py): deterministic weighted sum of (data quality, fit quality, robustness, margin, noise). Document formula in docstring. [M]
T5.5 Margin-aware gating: default Objective enforces PM>30°, Ms<2.0; violation → Status.WARNING and W_LOW_MARGIN. [S]
W6 — Reporting, plotting, CLI
T6.1 TuneResult.report(fmt) emits text / markdown / JSON. Deterministic, no timestamps except opt-in. [S]
T6.2 TuneResult.plot(kind) with save_path parameter that never calls plt.show(); show() is an explicit method (M9). [M]
T6.3 pidtune CLI (pid_control/cli/main.py, setup.py entry point). Subcommands: csv, plant, bench. [M]
T6.4 TuneResult.save()/load() as JSON (+ optional NPZ for artifacts). [S]
W7 — Examples cleanup
T7.1 Delete examples/demo.py (subsumed by T0.2). [S]
T7.2 Build 6 golden examples per §7.2. Each: <50 LOC, --show/--no-show,**
Let me finish with a more compact output.

T7.2 Build 6 golden examples per §7.2. Each ≤50 LOC, headless, saves artifacts. [M]
T7.3 Move interactive demos to examples/advanced/ and strip input() calls. [S]
T7.4 tools/run_examples.py headless runner feeding CI. [S]
T7.5 Ship examples/data/*.csv sample datasets (FOPDT step, SOPDT, integrator, noisy). [S]
W8 — Tests
T8.1 Regression tests for every fixed bug (C1–C5, H1–H8, M1–M12 where applicable). [M]
T8.2 Property tests (hypothesis) for: PIDController.update linearity in error for P-only; integral-accumulator monotonicity; filter convergence. [M]
T8.3 Integration tests for PIDAutotuner: one FOPDT, one SOPDT, one integrator (must reject), one noisy FOPDT. [M]
T8.4 Coverage target ≥ 80% on pid_control/autotune/ + pid_control/core/. [tracking]
W9 — Benchmarks
T9.1 Harness (benchmarks/run.py) per §8.4. JSON+MD output. [M]
T9.2 Smoke subset (benchmarks/smoke.py) ≤ 30 s, runs on every PR. [S]
T9.3 Baseline snapshot (benchmarks/results/baseline_pre_refactor.json) captured before W2–W5 land. Non-regression gate. [S]
W10 — Docs & migration
T10.1 New README with honest capability list, known limitations, “when not to use PID” section. [S]
T10.2 Migration guide docs/MIGRATION.md: old AutotuneFromData/RealtimeTuner → new PIDAutotuner. Code diffs. [S]
T10.3 Developer guide docs/ARCHITECTURE.md: stage protocol, how to add a rule / identifier / validator. [S]
T10.4 Delete / consolidate redundant top-level .md files (IMPLEMENTATION_SUMMARY.md, OPTIMIZATION_IMPROVEMENTS.md). [S]
Parallelization guide
Agent A owns W0 + W1 (blocking for everyone else).
Agent B owns W2 + W8 (tests alongside identification).
Agent C owns W3 + W4 (experiments + tuning).
Agent D owns W5 + W6 (validation + reporting).
Agent E owns W7 + W9 + W10 (examples, benchmarks, docs).
Every agent writes against the typed interfaces from W1. No file in pid_control/autotune/* is shared between agents B–E except types.py (frozen) and base.py protocols (modified only via RFC).

Acceptance criteria (global)
All existing unit tests pass.
New tests reach the coverage target.
Golden examples run headlessly with zero stderr.
Benchmark harness reports no regressions vs baseline_pre_refactor.json on safety metrics.
PIDAutotuner.from_csv(bad_data).tune() on a double-integrator dataset returns Status.FAILED with E_NO_STEADY_STATE (not silent "0% improvement success").
Risks
R1 Over-scoping. Mitigation: merge in layers (M0→M5), not one big PR.
R2 Breaking existing users. Mitigation: W1.T1.4 back-compat shim + deprecation warnings for ≥1 minor release.
R3 Robustness metrics disagreeing across plant types. Mitigation: thresholds configurable; validators are advisory, not blocking, by default.
R4 python-control version drift. Mitigation: pin control>=0.10,<0.12; adapter in autotune/validation/stability.py.
10. Recommended First Implementation Steps
In order, minimum friction, maximum signal:

T0.1 — Fix error_deadband P-term bypass. 5 lines, unblocks the red test. @c:/Users/danli/Projects/AdvancedPID/pid_control/core/pid_controller.py:206-209 apply deadband to the error used by both P and D paths (or recompute P after zeroing error).
T0.2 — Delete examples/demo.py; delete repo-root steering-*.log; add *.log, c:\Users\danli\Projects\AdvancedPID/advanced_pid_control.egg-info/ to .gitignore.
T0.3 — Fix setup.py install_requires (add control, gymnasium).
T9.3 baseline snapshot — before any algorithmic refactor, freeze current behaviour on the plant zoo so regressions are detectable.
T1.1 + T1.2 — Land the typed types.py + stage protocols. Even empty, they unlock parallel work.
T4.5 + T1.5 — Remove silent gain clamps and fix the status lie. One afternoon of work, huge credibility win.
T2.5 + C1 fix — Reject obviously bad data (double-integrator, flat input, no steady state) with typed E_* errors.
T3.1 — Rewrite relay autotune correctly. Pairs with T3.5 safety wrapper.
T5.1 — Margin computation using python-control; immediately elevates every tuning result.
T1.3 PIDAutotuner façade on top of everything above. Promote in README.
Assumptions & caveats
I assumed the repository state shown under @c:/Users/danli/Projects/AdvancedPID/ is current as of the conversation start.
I ran tests / examples with Python 3.11 because the default python on this machine (3.13) has no packages installed. The library imports fine on 3.11; I did not verify 3.8 support claimed in setup.py.
I tested only the autotune path end-to-end programmatically; I did not open GUI windows. "Likely OK" for interactive demos means import/construction succeeded; actual user workflows with render windows were not exercised.
Benchmark thresholds in §8 are starting proposals — tune them against the baseline_pre_refactor snapshot.
The proposed Confidence scalar is useful as a summary but should never be a sole gate; users should branch on the typed warnings list.
Summary
The project has a strong PID core and useful plant/simulation scaffolding, but its autotune stack — the headline feature — silently fails on PID-inappropriate plants, contains algorithmic bugs in relay and Z-N routines, and has no safety or confidence reporting. One unit test is red, one examples/demo.py is unrelated and executes on import. The path forward is not a rewrite: preserve the controller, delete the noise, and build a single PIDAutotuner façade around a typed four-stage pipeline (Experiment → Identify → Tune → Validate) with explicit warnings, margin/robustness analysis, a benchmark harness, and a pruned golden example set. The work is scoped into ~35 child tasks across 10 workstreams that 5 agents can execute in parallel on top of frozen interfaces from W1. Start with T0.1–T0.3 (safety net), then T9.3 baseline snapshot, then T1.1 types, then parallelize.

Want me to start executing? The highest-value first move is T0.1 + T0.2 + T0.3 + T9.3 (the safety net + baseline), which I can land in one pass without touching architecture.