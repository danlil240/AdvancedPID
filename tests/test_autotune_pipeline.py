"""Integration tests for the full PIDAutotuner pipeline (PLAN.md T8.3).

Each test exercises the complete Experiment → Identify → Tune → Validate
pipeline via the public ``PIDAutotuner`` façade, verifying:
  - Status codes are correct.
  - Gains are physically reasonable.
  - Warnings fire on the right pathologies.
  - The controller can be built from successful results.
"""

from __future__ import annotations

import numpy as np
import pytest

from pid_control.autotune.api import PIDAutotuner
from pid_control.autotune.types import (
    Objective,
    Status,
    WarningCode,
)


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def _fopdt_step_data(K=2.0, tau=3.0, theta=0.5, dt=0.02, T=30.0, step_t=1.0,
                     noise=0.0, seed=0):
    """Generate a clean FOPDT step-response dataset."""
    t = np.arange(0.0, T, dt)
    u = np.zeros_like(t)
    u[t >= step_t] = 1.0
    y = np.zeros_like(t)
    ds = int(theta / dt)
    for i in range(1, len(t)):
        ud = u[i - ds] if i > ds else 0.0
        y[i] = y[i - 1] + dt / tau * (K * ud - y[i - 1])
    if noise > 0:
        rng = np.random.default_rng(seed)
        y += rng.normal(0, noise * max(np.std(y), 1e-6), len(y))
    return t, u, y


def _sopdt_step_data(K=1.5, tau1=2.0, tau2=0.8, theta=0.3, dt=0.02, T=40.0,
                     step_t=1.0, noise=0.0, seed=0):
    """Generate a clean SOPDT step-response dataset."""
    t = np.arange(0.0, T, dt)
    u = np.zeros_like(t)
    u[t >= step_t] = 1.0
    y1 = np.zeros_like(t)
    y2 = np.zeros_like(t)
    ds = int(theta / dt)
    for i in range(1, len(t)):
        ud = u[i - ds] if i > ds else 0.0
        y1[i] = y1[i - 1] + dt / tau1 * (K * ud - y1[i - 1])
        y2[i] = y2[i - 1] + dt / tau2 * (y1[i - 1] - y2[i - 1])
    if noise > 0:
        rng = np.random.default_rng(seed)
        y2 += rng.normal(0, noise * max(np.std(y2), 1e-6), len(y2))
    return t, u, y2


def _integrator_data(K_i=0.5, theta=0.2, dt=0.02, T=20.0, step_t=1.0):
    """Generate an integrating (IPDT) dataset — output ramps forever."""
    t = np.arange(0.0, T, dt)
    u = np.zeros_like(t)
    u[t >= step_t] = 1.0
    y = np.zeros_like(t)
    ds = int(theta / dt)
    for i in range(1, len(t)):
        ud = u[i - ds] if i > ds else 0.0
        y[i] = y[i - 1] + K_i * ud * dt
    return t, u, y


def _flat_input_data(dt=0.02, T=10.0):
    """Generate data with no excitation (constant input)."""
    t = np.arange(0.0, T, dt)
    u = np.ones_like(t) * 5.0
    y = np.ones_like(t) * 2.5 + np.random.default_rng(0).normal(0, 0.01, len(t))
    return t, u, y


# ---------------------------------------------------------------------------
# FOPDT pipeline
# ---------------------------------------------------------------------------

class TestFOPDTPipeline:
    """Full pipeline on a clean FOPDT plant."""

    def test_successful_tune(self):
        t, u, y = _fopdt_step_data(K=2.0, tau=3.0, theta=0.5)
        result = PIDAutotuner.from_arrays(t, u, y).tune()

        assert result.status in (Status.OK, Status.WARNING)
        assert result.is_usable
        assert result.gains.kp > 0
        assert result.gains.ki > 0
        assert result.confidence.score > 0.0
        assert result.identification is not None
        assert result.identification.fit_quality_r2 > 0.90

    def test_builds_controller(self):
        t, u, y = _fopdt_step_data()
        result = PIDAutotuner.from_arrays(t, u, y).tune()
        if result.is_usable:
            ctrl = result.build_controller()
            assert ctrl is not None
            assert ctrl.params.kp == pytest.approx(result.gains.kp, rel=1e-6)

    def test_with_tight_objective(self):
        t, u, y = _fopdt_step_data(K=2.0, tau=3.0, theta=0.5)
        result = (
            PIDAutotuner.from_arrays(t, u, y)
            .with_objective(Objective(
                max_overshoot_pct=5.0,
                min_phase_margin_deg=45.0,
            ))
            .tune()
        )
        assert result.status in (Status.OK, Status.WARNING)

    def test_report_text(self):
        t, u, y = _fopdt_step_data()
        result = PIDAutotuner.from_arrays(t, u, y).tune()
        text = result.report("text")
        assert "Kp" in text
        assert "Status" in text

    def test_report_markdown(self):
        t, u, y = _fopdt_step_data()
        result = PIDAutotuner.from_arrays(t, u, y).tune()
        md = result.report("md")
        assert "# PID Autotune Result" in md

    def test_report_json(self):
        import json
        t, u, y = _fopdt_step_data()
        result = PIDAutotuner.from_arrays(t, u, y).tune()
        payload = json.loads(result.report("json"))
        assert "gains" in payload
        assert "status" in payload

    def test_save_load_json(self, tmp_path):
        t, u, y = _fopdt_step_data()
        result = PIDAutotuner.from_arrays(t, u, y).tune()
        out = tmp_path / "result.json"
        result.save(str(out))
        assert out.exists()
        import json
        data = json.loads(out.read_text())
        assert data["status"] in ("ok", "warning", "failed")

    def test_noisy_fopdt(self):
        t, u, y = _fopdt_step_data(noise=0.05, seed=42)
        result = PIDAutotuner.from_arrays(t, u, y).tune()
        # Should still produce a usable result on moderate noise
        assert result.status in (Status.OK, Status.WARNING)
        assert result.identification is not None


# ---------------------------------------------------------------------------
# SOPDT pipeline
# ---------------------------------------------------------------------------

class TestSOPDTPipeline:

    def test_successful_tune(self):
        t, u, y = _sopdt_step_data(K=1.5, tau1=2.0, tau2=0.8, theta=0.3)
        result = PIDAutotuner.from_arrays(t, u, y).tune()
        assert result.status in (Status.OK, Status.WARNING)
        assert result.is_usable
        assert result.identification is not None
        assert result.identification.fit_quality_r2 > 0.90


# ---------------------------------------------------------------------------
# Integrator rejection (C1 fix verification)
# ---------------------------------------------------------------------------

class TestIntegratorRejection:
    """An integrating (ramp-response) dataset MUST be rejected — never
    silently tuned (this was the C1 bug)."""

    def test_integrator_rejected(self):
        t, u, y = _integrator_data()
        result = PIDAutotuner.from_arrays(t, u, y).tune()
        assert result.status is Status.FAILED
        assert result.has_warning(WarningCode.E_NO_STEADY_STATE)

    def test_cannot_build_controller_from_failed(self):
        t, u, y = _integrator_data()
        result = PIDAutotuner.from_arrays(t, u, y).tune()
        with pytest.raises(RuntimeError, match="FAILED"):
            result.build_controller()


# ---------------------------------------------------------------------------
# Flat-input rejection
# ---------------------------------------------------------------------------

class TestFlatInputRejection:

    def test_flat_input_rejected(self):
        t, u, y = _flat_input_data()
        result = PIDAutotuner.from_arrays(t, u, y).tune()
        assert result.status is Status.FAILED
        assert result.has_warning(WarningCode.E_DATA_FLAT)


# ---------------------------------------------------------------------------
# Actuator limits
# ---------------------------------------------------------------------------

class TestActuatorLimits:

    def test_with_actuator_limits(self):
        t, u, y = _fopdt_step_data()
        result = (
            PIDAutotuner.from_arrays(t, u, y)
            .with_actuator_limits(lower=-5.0, upper=5.0)
            .tune()
        )
        assert result.status in (Status.OK, Status.WARNING)


# ---------------------------------------------------------------------------
# Rule-only mode (no numerical tuner)
# ---------------------------------------------------------------------------

class TestRuleOnlyMode:

    def test_rule_only(self):
        t, u, y = _fopdt_step_data()
        result = (
            PIDAutotuner.from_arrays(t, u, y)
            .set_tuner(None)
            .tune()
        )
        assert result.status in (Status.OK, Status.WARNING)
        assert result.gains.kp > 0

    def test_different_rules(self):
        t, u, y = _fopdt_step_data()
        from pid_control.autotune.rules.cohen_coon import CohenCoonRule
        from pid_control.autotune.rules.skogestad import SkogestadRule

        for rule_cls in (CohenCoonRule, SkogestadRule):
            result = (
                PIDAutotuner.from_arrays(t, u, y)
                .set_rule(rule_cls())
                .set_tuner(None)
                .tune()
            )
            assert result.status in (Status.OK, Status.WARNING)
            assert result.gains.kp > 0


# ---------------------------------------------------------------------------
# Back-compat shim
# ---------------------------------------------------------------------------

class TestBackCompat:

    def test_compat_shim_warns(self, tmp_path):
        """AutotuneFromDataCompat emits a DeprecationWarning."""
        # Create a minimal CSV
        csv_path = tmp_path / "test.csv"
        t, u, y = _fopdt_step_data(T=10.0)
        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "output", "measurement"])
            for i in range(len(t)):
                writer.writerow([t[i], u[i], y[i]])

        from pid_control.autotune.compat import AutotuneFromDataCompat
        with pytest.warns(DeprecationWarning, match="deprecated"):
            _ = AutotuneFromDataCompat(str(csv_path))


# ---------------------------------------------------------------------------
# NelderMead tuner
# ---------------------------------------------------------------------------

class TestNelderMeadTuner:

    def test_nelder_mead_runs(self):
        from pid_control.autotune.tuning.de import NelderMeadTuner
        t, u, y = _fopdt_step_data(K=2.0, tau=3.0, theta=0.5)
        result = (
            PIDAutotuner.from_arrays(t, u, y)
            .set_tuner(NelderMeadTuner(max_iter=100))
            .tune()
        )
        assert result.status in (Status.OK, Status.WARNING)
        assert result.gains.kp > 0


# ---------------------------------------------------------------------------
# From-plant path
# ---------------------------------------------------------------------------

class TestFromPlant:

    def test_from_plant_fopdt(self):
        from pid_control.plants import DelayPlant, FirstOrderPlant
        base = FirstOrderPlant(gain=2.0, time_constant=3.0, sample_time=0.02)
        plant = DelayPlant(base, delay_time=0.5)
        result = PIDAutotuner.from_plant(plant).tune()
        assert result.status in (Status.OK, Status.WARNING)
        assert result.is_usable
        assert result.gains.kp > 0

    def test_from_plant_builds_controller(self):
        from pid_control.plants import DelayPlant, FirstOrderPlant
        base = FirstOrderPlant(gain=1.5, time_constant=2.0, sample_time=0.02)
        plant = DelayPlant(base, delay_time=0.3)
        result = PIDAutotuner.from_plant(plant).tune()
        if result.is_usable:
            ctrl = result.build_controller()
            assert ctrl.params.kp == pytest.approx(result.gains.kp, rel=1e-6)


# ---------------------------------------------------------------------------
# From-CSV path
# ---------------------------------------------------------------------------

class TestFromCSV:

    def test_from_csv_fopdt(self, tmp_path):
        import csv
        csv_path = tmp_path / "step.csv"
        t, u, y = _fopdt_step_data(K=2.0, tau=3.0, theta=0.5, T=20.0)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "input", "output"])
            for i in range(len(t)):
                writer.writerow([t[i], u[i], y[i]])
        result = PIDAutotuner.from_csv(
            str(csv_path),
            columns={"time": "time", "input": "input", "measurement": "output"},
        ).tune()
        assert result.status in (Status.OK, Status.WARNING)
        assert result.identification is not None
        assert result.identification.fit_quality_r2 > 0.90


# ---------------------------------------------------------------------------
# Validation outcomes
# ---------------------------------------------------------------------------

class TestValidationOutcomes:

    def test_margins_populated(self):
        t, u, y = _fopdt_step_data(K=2.0, tau=3.0, theta=0.5)
        result = PIDAutotuner.from_arrays(t, u, y).tune()
        if result.performance is not None and result.performance.margins is not None:
            m = result.performance.margins
            assert m.phase_margin_deg is None or m.phase_margin_deg > 0
            assert m.sensitivity_peak is None or m.sensitivity_peak > 0

    def test_confidence_in_range(self):
        t, u, y = _fopdt_step_data()
        result = PIDAutotuner.from_arrays(t, u, y).tune()
        assert 0.0 <= result.confidence.score <= 1.0

    def test_confidence_has_contributions(self):
        t, u, y = _fopdt_step_data()
        result = PIDAutotuner.from_arrays(t, u, y).tune()
        assert len(result.confidence.contributions) > 0

    def test_sim_benchmark_validator(self):
        from pid_control.autotune.validation.sim_benchmark import SimBenchmarkValidator
        from pid_control.autotune.types import (
            ActuatorLimits, IdentificationResult, Objective,
            PIDGains, TransferFunctionModel, ModelType,
        )
        model = TransferFunctionModel(
            model_type=ModelType.FOPDT, K=2.0, tau=3.0, theta=0.5,
        )
        ident = IdentificationResult(model=model, fit_quality_r2=0.99)
        gains = PIDGains(kp=1.5, ki=0.5, kd=0.3)
        v = SimBenchmarkValidator()
        outcome = v.validate(ident, gains, Objective(), ActuatorLimits())
        assert outcome.status in (Status.OK, Status.WARNING)


# ---------------------------------------------------------------------------
# TuneResult helpers
# ---------------------------------------------------------------------------

class TestTuneResultHelpers:

    def test_has_warning(self):
        t, u, y = _integrator_data()
        result = PIDAutotuner.from_arrays(t, u, y).tune()
        assert result.has_warning(WarningCode.E_NO_STEADY_STATE)
        assert not result.has_warning(WarningCode.W_SATURATION)

    def test_warnings_for_severity(self):
        from pid_control.autotune.types import Severity
        t, u, y = _integrator_data()
        result = PIDAutotuner.from_arrays(t, u, y).tune()
        errors = result.warnings_for(Severity.ERROR)
        assert len(errors) >= 1

    def test_to_dict_roundtrip(self):
        import json
        t, u, y = _fopdt_step_data(T=10.0)
        result = PIDAutotuner.from_arrays(t, u, y).set_tuner(None).tune()
        d = result.to_dict()
        text = json.dumps(d, default=float)
        parsed = json.loads(text)
        assert parsed["status"] in ("ok", "warning", "failed")
        assert "gains" in parsed
