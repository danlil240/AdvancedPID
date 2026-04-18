"""Tests for the typed autotune API surface (PLAN.md T1.1).

These tests pin the shape of :class:`TuneResult` and friends so the
downstream workstreams can refactor on stable foundations.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pid_control.autotune import (
    ActuatorLimits,
    Artifacts,
    Confidence,
    DataQuality,
    IdentificationResult,
    MarginReport,
    ModelType,
    Objective,
    PerformanceReport,
    PIDGains,
    Severity,
    Status,
    Trajectory,
    TransferFunctionModel,
    TuneMeta,
    TuneResult,
    Warning,
    WarningCode,
    merge_warnings,
)


# ---------------------------------------------------------------------------
# Leaf dataclasses
# ---------------------------------------------------------------------------

def test_actuator_limits_validates_upper_above_lower():
    with pytest.raises(ValueError):
        ActuatorLimits(lower=5.0, upper=1.0)


def test_actuator_limits_rate_limit_positive():
    with pytest.raises(ValueError):
        ActuatorLimits(lower=-1.0, upper=1.0, rate_limit=-1.0)


def test_actuator_limits_is_bounded_flag():
    assert ActuatorLimits(lower=-1.0, upper=1.0).is_bounded
    assert not ActuatorLimits().is_bounded


def test_pid_gains_as_dict_roundtrip():
    g = PIDGains(kp=1.5, ki=0.2, kd=0.0, setpoint_weight_b=0.6)
    d = g.as_dict()
    assert d == {"kp": 1.5, "ki": 0.2, "kd": 0.0, "b": 0.6, "c": 0.0, "N": 10.0}


def test_trajectory_rejects_mismatched_arrays():
    with pytest.raises(ValueError):
        Trajectory(
            time=np.zeros(3),
            setpoint=np.zeros(2),
            measurement=np.zeros(3),
            control=np.zeros(3),
        )


def test_warning_to_dict_includes_code_value():
    w = Warning(
        code=WarningCode.W_LOW_MARGIN,
        severity=Severity.WARNING,
        message="phase margin 22° < 30°",
        stage="validation.margins",
        context={"phase_margin_deg": 22.0},
    )
    d = w.to_dict()
    assert d["code"] == "W_LOW_MARGIN"
    assert d["severity"] == "warning"
    assert d["context"]["phase_margin_deg"] == 22.0


def test_merge_warnings_preserves_order():
    a = Warning(WarningCode.W_POOR_FIT, Severity.WARNING, "a")
    b = Warning(WarningCode.W_GAIN_CLIPPED, Severity.WARNING, "b")
    c = Warning(WarningCode.W_LOW_MARGIN, Severity.WARNING, "c")
    merged = merge_warnings([a], [b, c])
    assert [w.code for w in merged] == [a.code, b.code, c.code]


# ---------------------------------------------------------------------------
# TuneResult helpers
# ---------------------------------------------------------------------------

def _make_result(status: Status = Status.OK, warnings=()) -> TuneResult:
    model = TransferFunctionModel(
        model_type=ModelType.FOPDT, K=1.0, tau=1.0, theta=0.1
    )
    ident = IdentificationResult(model=model, fit_quality_r2=0.99)
    perf = PerformanceReport(
        iae=1.0, ise=0.5, itae=2.0,
        rise_time=0.2, settling_time_2pct=1.0,
        overshoot_percent=3.0, steady_state_error=0.0,
        control_total_variation=1.0, control_rms=1.0, control_peak=2.0,
        saturation_fraction=0.0,
        margins=MarginReport(gain_margin_db=12.0, phase_margin_deg=60.0),
    )
    return TuneResult(
        gains=PIDGains(kp=1.0, ki=0.5, kd=0.05),
        status=status,
        confidence=Confidence(score=0.9, contributions={"fit": 0.4, "margin": 0.5}),
        identification=ident,
        performance=perf,
        warnings=tuple(warnings),
        artifacts=Artifacts(),
        meta=TuneMeta(library_version="0.2.0-dev", elapsed_seconds=0.5, seed=0),
    )


def test_tune_result_is_usable_tracks_status():
    assert _make_result(Status.OK).is_usable
    assert _make_result(Status.WARNING).is_usable
    assert not _make_result(Status.FAILED).is_usable


def test_tune_result_warnings_filter_by_severity():
    warns = [
        Warning(WarningCode.W_LOW_MARGIN, Severity.WARNING, "pm low"),
        Warning(WarningCode.W_SATURATION, Severity.INFO, "saturated"),
    ]
    r = _make_result(Status.WARNING, warnings=warns)
    assert r.warnings_for(Severity.INFO)[0].code is WarningCode.W_SATURATION
    assert r.has_warning(WarningCode.W_LOW_MARGIN)
    assert not r.has_warning(WarningCode.W_POOR_FIT)


def test_tune_result_to_dict_roundtrips_through_json(tmp_path: Path):
    r = _make_result(Status.OK, warnings=[
        Warning(WarningCode.W_GAIN_CLIPPED, Severity.WARNING, "kp hit upper bound"),
    ])
    target = tmp_path / "result.json"
    r.save(target)
    payload = json.loads(target.read_text())
    assert payload["status"] == "ok"
    assert payload["gains"]["kp"] == pytest.approx(1.0)
    assert payload["warnings"][0]["code"] == "W_GAIN_CLIPPED"
    assert payload["identification"]["model"]["type"] == "fopdt"
    assert payload["performance"]["margins"]["phase_margin_deg"] == 60.0


def test_build_controller_refuses_failed_result():
    r = _make_result(Status.FAILED)
    with pytest.raises(RuntimeError):
        r.build_controller()


def test_build_controller_returns_working_pid():
    r = _make_result(Status.OK)
    pid = r.build_controller(sample_time=0.05)
    assert pid is not None
    # Smoke-run a single update to confirm wiring.
    out = pid.update(setpoint=1.0, measurement=0.0)
    assert np.isfinite(out)


def test_data_quality_dataclass_defaults():
    dq = DataQuality(
        status=Status.OK, snr_db=20.0, excitation_energy=1.0,
        has_steady_state=True, n_samples=500, sample_time=0.01,
    )
    assert dq.status is Status.OK
    assert dq.warnings == ()


def test_objective_has_sensible_defaults():
    obj = Objective()
    assert obj.max_overshoot_pct > 0
    assert obj.max_Ms > 1.0
    assert obj.iae_weight == 1.0
