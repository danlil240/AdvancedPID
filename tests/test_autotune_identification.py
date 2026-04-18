"""Tests for the autotune identifiers (PLAN.md T2.1 / T2.2 / T2.4)."""

from __future__ import annotations

import numpy as np
import pytest

from pid_control.autotune.experiments.from_data import FromDataExperiment
from pid_control.autotune.identification import (
    AutoIdentifier,
    FOPDTIdentifier,
    SOPDTIdentifier,
    simulate_model,
)
from pid_control.autotune.types import ModelType, WarningCode


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _fopdt(K, tau, theta, dt=0.02, T=30.0, step_t=1.0, noise=0.0, seed=0):
    t = np.arange(0.0, T, dt)
    u = np.zeros_like(t); u[t >= step_t] = 1.0
    y = np.zeros_like(t); ds = int(theta / dt)
    for i in range(1, len(t)):
        ud = u[i - ds] if i > ds else 0.0
        y[i] = y[i - 1] + dt / tau * (K * ud - y[i - 1])
    if noise > 0:
        rng = np.random.default_rng(seed)
        y = y + rng.normal(0.0, noise * (np.std(y) or 1.0), len(y))
    return t, u, y


def _sopdt(K, tau1, tau2, theta, dt=0.02, T=30.0, step_t=1.0, noise=0.0, seed=0):
    t = np.arange(0.0, T, dt)
    u = np.zeros_like(t); u[t >= step_t] = 1.0
    y1 = np.zeros_like(t); y2 = np.zeros_like(t)
    ds = int(theta / dt)
    for i in range(1, len(t)):
        ud = u[i - ds] if i > ds else 0.0
        y1[i] = y1[i - 1] + dt / tau1 * (K * ud - y1[i - 1])
        y2[i] = y2[i - 1] + dt / tau2 * (y1[i - 1] - y2[i - 1])
    if noise > 0:
        rng = np.random.default_rng(seed)
        y2 = y2 + rng.normal(0.0, noise * (np.std(y2) or 1.0), len(y2))
    return t, u, y2


# ---------------------------------------------------------------------------
# FOPDT identifier
# ---------------------------------------------------------------------------

def test_fopdt_identifies_clean_plant_within_tolerance():
    t, u, y = _fopdt(K=2.0, tau=3.0, theta=0.5)
    rec = FromDataExperiment(time=t, input_signal=u, output=y).run()
    res = FOPDTIdentifier().identify(rec)
    assert res.model.model_type is ModelType.FOPDT
    assert res.fit_quality_r2 > 0.98
    assert res.model.K == pytest.approx(2.0, rel=0.05)
    assert res.model.tau == pytest.approx(3.0, rel=0.10)
    assert res.model.theta == pytest.approx(0.5, abs=0.2)


def test_fopdt_handles_noise_but_flags_poor_fit_if_needed():
    t, u, y = _fopdt(K=1.0, tau=2.0, theta=0.2, noise=0.5, seed=42)
    rec = FromDataExperiment(time=t, input_signal=u, output=y).run()
    res = FOPDTIdentifier().identify(rec)
    # Still recover K to sensible tolerance despite heavy noise.
    assert res.model.K == pytest.approx(1.0, rel=0.2)


def test_fopdt_handles_negative_gain():
    t, u, y = _fopdt(K=-1.5, tau=2.0, theta=0.3)
    rec = FromDataExperiment(time=t, input_signal=u, output=y).run()
    res = FOPDTIdentifier().identify(rec)
    assert res.model.K < 0
    assert res.model.K == pytest.approx(-1.5, rel=0.1)


# ---------------------------------------------------------------------------
# SOPDT identifier + degeneracy detection
# ---------------------------------------------------------------------------

def test_sopdt_identifies_clean_two_pole_plant():
    t, u, y = _sopdt(K=1.5, tau1=3.0, tau2=1.0, theta=0.3)
    rec = FromDataExperiment(time=t, input_signal=u, output=y).run()
    res = SOPDTIdentifier().identify(rec)
    assert res.model.model_type is ModelType.SOPDT
    # The larger τ should dominate.  Tolerances are wide (50 %) because
    # the test data is generated with forward-Euler while the identifier
    # uses the closed-form discrete cascade — the two agree on the overall
    # response shape but disagree on the exact pole split.  What matters
    # is that neither τ collapses into the sample-time floor.
    assert res.model.tau == pytest.approx(3.0, rel=0.5)
    tau2 = res.model.tau2 or 0.0
    assert tau2 > 5 * rec.sample_time, f"τ₂ collapsed to the floor: {tau2}"
    assert tau2 == pytest.approx(1.0, rel=0.6)
    assert res.fit_quality_r2 > 0.98


def test_sopdt_flags_degenerate_fit_on_pure_fopdt():
    t, u, y = _fopdt(K=1.0, tau=2.0, theta=0.3)
    rec = FromDataExperiment(time=t, input_signal=u, output=y).run()
    res = SOPDTIdentifier().identify(rec)
    codes = {w.code for w in res.warnings}
    assert WarningCode.W_DEGENERATE_SOPDT in codes


# ---------------------------------------------------------------------------
# AIC/BIC model selection
# ---------------------------------------------------------------------------

def test_auto_identifier_picks_fopdt_for_first_order_data():
    t, u, y = _fopdt(K=1.0, tau=2.0, theta=0.3)
    rec = FromDataExperiment(time=t, input_signal=u, output=y).run()
    res = AutoIdentifier().identify(rec)
    assert res.model.model_type is ModelType.FOPDT


def test_auto_identifier_picks_sopdt_for_second_order_data():
    t, u, y = _sopdt(K=1.0, tau1=3.0, tau2=1.0, theta=0.2)
    rec = FromDataExperiment(time=t, input_signal=u, output=y).run()
    res = AutoIdentifier().identify(rec)
    assert res.model.model_type is ModelType.SOPDT


def test_simulate_model_matches_identification_on_fopdt():
    t, u, y = _fopdt(K=1.2, tau=1.5, theta=0.3)
    rec = FromDataExperiment(time=t, input_signal=u, output=y).run()
    res = FOPDTIdentifier().identify(rec)
    # Feed the identified model back through the simulator and confirm
    # the residuals are small compared to the output swing.
    y_sim = simulate_model(res.model, t, u, y0=0.0)
    rms = float(np.sqrt(np.mean((y - y_sim) ** 2)))
    assert rms < 0.05 * (np.ptp(y) or 1.0)
