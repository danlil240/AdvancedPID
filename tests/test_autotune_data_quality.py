"""Tests for the data-quality diagnostic (PLAN.md T2.5).

These tests encode the library's non-negotiable rule: the autotune
pipeline MUST refuse to silently fit obviously bad data.  The concrete
checks here are:

  * flat input -> ``E_DATA_FLAT`` -> ``Status.FAILED``
  * integrator-like (no steady-state) output -> ``E_NO_STEADY_STATE`` -> FAILED
  * dataset shorter than ``MIN_SAMPLES`` -> ``E_TOO_SHORT`` -> FAILED
  * a clean FOPDT-style step response -> ``Status.OK``
  * a noisy step -> ``Status.OK`` or a ``W_HIGH_NOISE`` warning (never
    an error)
"""

from __future__ import annotations

import numpy as np
import pytest

from pid_control.autotune.diagnostics import data_quality
from pid_control.autotune.types import Status, WarningCode


def _fopdt_step(K=1.5, tau=1.0, theta=0.2, dt=0.01, duration=10.0, noise=0.0,
                 seed=0):
    t = np.arange(0.0, duration, dt)
    u = np.zeros_like(t)
    u[t >= 1.0] = 1.0
    y = np.zeros_like(t)
    ds = int(theta / dt)
    for i in range(1, len(t)):
        ud = u[i - ds] if i > ds else 0.0
        y[i] = y[i - 1] + dt / tau * (K * ud - y[i - 1])
    if noise > 0:
        rng = np.random.default_rng(seed)
        y = y + rng.normal(0.0, noise * (np.std(y) or 1.0), len(y))
    return t, u, y


def test_accepts_clean_fopdt_step():
    t, u, y = _fopdt_step()
    dq = data_quality.assess(t, u, y)
    assert dq.status is Status.OK
    assert dq.has_steady_state
    assert dq.excitation_energy > 0


def test_rejects_flat_input():
    t = np.linspace(0, 10, 1000)
    u = np.full_like(t, 2.0)            # absolutely constant
    y = 3.0 + 0.01 * np.random.default_rng(0).normal(size=len(t))
    dq = data_quality.assess(t, u, y)
    assert dq.status is Status.FAILED
    codes = {w.code for w in dq.warnings}
    assert WarningCode.E_DATA_FLAT in codes


def test_rejects_integrator_like_output():
    # Double integrator under a constant input — never reaches steady state.
    dt, T = 0.01, 20.0
    t = np.arange(0, T, dt)
    u = np.zeros_like(t)
    u[t >= 1.0] = 1.0
    v = np.zeros_like(t); y = np.zeros_like(t)
    for i in range(1, len(t)):
        v[i] = v[i - 1] + u[i - 1] * dt
        y[i] = y[i - 1] + v[i - 1] * dt
    dq = data_quality.assess(t, u, y)
    assert dq.status is Status.FAILED
    codes = {w.code for w in dq.warnings}
    assert WarningCode.E_NO_STEADY_STATE in codes


def test_rejects_too_short_dataset():
    t = np.linspace(0, 0.1, 5)
    u = np.zeros_like(t)
    u[2:] = 1.0
    y = np.linspace(0, 1, 5)
    dq = data_quality.assess(t, u, y)
    assert dq.status is Status.FAILED
    codes = {w.code for w in dq.warnings}
    assert WarningCode.E_TOO_SHORT in codes


def test_noisy_step_passes_but_may_flag_noise():
    t, u, y = _fopdt_step(noise=0.3, seed=1)
    dq = data_quality.assess(t, u, y)
    # Must still be fittable; just may carry a W_HIGH_NOISE hint.
    assert dq.status in (Status.OK, Status.WARNING)
    if dq.status is Status.WARNING:
        assert any(w.code is WarningCode.W_HIGH_NOISE for w in dq.warnings)
