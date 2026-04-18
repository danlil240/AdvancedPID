"""Property-based tests using hypothesis (PLAN.md T8.2).

Tests:
1. P-only controller: output is linear in error (homogeneity + additivity).
2. Integral accumulator: monotonically non-decreasing for constant positive error.
3. Low-pass filter: converges to steady input.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from pid_control.core.pid_controller import PIDController
from pid_control.core.pid_params import PIDParams
from pid_control.core.filters import LowPassFilter


# ---------------------------------------------------------------------------
# 1. P-only linearity: output = kp * (setpoint - measurement)
# ---------------------------------------------------------------------------

_reasonable_float = st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False)


@given(
    kp=st.floats(min_value=0.01, max_value=100.0, allow_nan=False),
    setpoint=_reasonable_float,
    measurement=_reasonable_float,
    scale=st.floats(min_value=0.01, max_value=10.0, allow_nan=False),
)
@settings(max_examples=200)
def test_p_only_homogeneity(kp: float, setpoint: float, measurement: float, scale: float):
    """P-only controller output scales linearly with Kp."""
    pid1 = PIDController(PIDParams(kp=kp, ki=0.0, kd=0.0))
    pid2 = PIDController(PIDParams(kp=kp * scale, ki=0.0, kd=0.0))

    out1 = pid1.update(setpoint=setpoint, measurement=measurement, timestamp=0.0)
    out2 = pid2.update(setpoint=setpoint, measurement=measurement, timestamp=0.0)

    assert abs(out2 - scale * out1) < 1e-6 * (1 + abs(out2))


@given(
    kp=st.floats(min_value=0.01, max_value=100.0, allow_nan=False),
    sp1=_reasonable_float,
    m1=_reasonable_float,
    sp2=_reasonable_float,
    m2=_reasonable_float,
)
@settings(max_examples=200)
def test_p_only_additivity(
    kp: float, sp1: float, m1: float, sp2: float, m2: float,
):
    """P-only output for combined errors == sum of individual outputs."""
    pid_a = PIDController(PIDParams(kp=kp, ki=0.0, kd=0.0))
    pid_b = PIDController(PIDParams(kp=kp, ki=0.0, kd=0.0))
    pid_ab = PIDController(PIDParams(kp=kp, ki=0.0, kd=0.0))

    out_a = pid_a.update(setpoint=sp1, measurement=m1, timestamp=0.0)
    out_b = pid_b.update(setpoint=sp2, measurement=m2, timestamp=0.0)
    # Combined error = (sp1 + sp2) - (m1 + m2)  should produce same output.
    out_ab = pid_ab.update(setpoint=sp1 + sp2, measurement=m1 + m2, timestamp=0.0)

    assert abs(out_ab - (out_a + out_b)) < 1e-6 * (1 + abs(out_ab))


# ---------------------------------------------------------------------------
# 2. Integral accumulator monotonicity for constant positive error
# ---------------------------------------------------------------------------

@given(
    ki=st.floats(min_value=0.01, max_value=50.0, allow_nan=False),
    error=st.floats(min_value=0.01, max_value=1e3, allow_nan=False),
    n_steps=st.integers(min_value=2, max_value=50),
)
@settings(max_examples=200)
def test_integral_monotonically_nondecreasing(ki: float, error: float, n_steps: int):
    """With constant positive error, the integral accumulator never decreases."""
    pid = PIDController(PIDParams(kp=0.0, ki=ki, kd=0.0))

    prev_integral = -float("inf")
    for i in range(n_steps):
        pid.update(setpoint=error, measurement=0.0, timestamp=i * 0.01)
        cur = pid.integral
        assert cur >= prev_integral - 1e-12, (
            f"Step {i}: integral went from {prev_integral} to {cur}"
        )
        prev_integral = cur


# ---------------------------------------------------------------------------
# 3. Low-pass filter convergence to a constant input
# ---------------------------------------------------------------------------

@given(
    alpha=st.floats(min_value=0.01, max_value=1.0, allow_nan=False),
    target=st.floats(min_value=-1e4, max_value=1e4, allow_nan=False),
)
@settings(max_examples=200)
def test_lowpass_converges(alpha: float, target: float):
    """A first-order IIR low-pass filter converges to its constant input."""
    filt = LowPassFilter(alpha=alpha)
    # Feed a constant signal enough times to converge.
    for _ in range(500):
        out = filt.update(target)
    assert abs(out - target) < 1e-3 * (1 + abs(target))
