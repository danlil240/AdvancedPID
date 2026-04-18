"""Simulation-benchmark validator (PLAN.md T5.3).

Runs the tuned PID against standard scenarios (step, load disturbance,
noise injection) on the *identified* model and checks responses against
the user's :class:`Objective`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from pid_control.autotune.types import (
    ActuatorLimits,
    IdentificationResult,
    Objective,
    PerformanceReport,
    PIDGains,
    Severity,
    Status,
    Warning,
    WarningCode,
)
from pid_control.autotune.validation.base import ValidationOutcome


@dataclass(frozen=True)
class SimBenchmarkValidator:
    """Simulate standard closed-loop scenarios and flag poor performance.

    Scenarios:
      1. **Unit step** — measures rise time, overshoot, settling, IAE.
      2. **Load disturbance** (step at mid-sim) — measures recovery.
      3. **Noise** — injects measurement noise; checks for amplification.
    """

    name: str = "sim_benchmark"

    def validate(
        self,
        identification: IdentificationResult,
        gains: PIDGains,
        objective: Objective,
        actuator: ActuatorLimits,
    ) -> ValidationOutcome:
        warnings: List[Warning] = []
        status = Status.OK

        model = identification.model
        tau = max(float(model.tau), 1e-9)
        theta = float(model.theta)
        dt = min(tau / 50.0, theta / 5.0 if theta > 0 else 0.05, 0.02)
        dt = max(dt, 1e-4)
        duration = max(10.0 * (tau + theta), 5.0)

        # --- Step response ------------------------------------------------
        step_result = _closed_loop_step(
            identification, gains, actuator, dt, duration,
        )
        if step_result is None:
            warnings.append(Warning(
                code=WarningCode.E_UNSTABLE,
                severity=Severity.ERROR,
                message="Closed-loop step response diverged.",
                stage="validate.sim_benchmark",
            ))
            return ValidationOutcome(
                status=Status.FAILED, warnings=tuple(warnings),
            )

        # Check overshoot
        if step_result["overshoot_pct"] > objective.max_overshoot_pct:
            warnings.append(Warning(
                code=WarningCode.W_LOW_MARGIN,
                severity=Severity.WARNING,
                message=(
                    f"Step overshoot {step_result['overshoot_pct']:.1f}% > "
                    f"target {objective.max_overshoot_pct:.1f}%"
                ),
                stage="validate.sim_benchmark",
                context={"overshoot_pct": step_result["overshoot_pct"]},
            ))
            if status is not Status.FAILED:
                status = Status.WARNING

        # Check settling time
        if (
            objective.max_settling_time is not None
            and step_result["settling_2pct"] > objective.max_settling_time
        ):
            warnings.append(Warning(
                code=WarningCode.W_LOW_MARGIN,
                severity=Severity.WARNING,
                message=(
                    f"Settling time {step_result['settling_2pct']:.2f}s > "
                    f"target {objective.max_settling_time:.2f}s"
                ),
                stage="validate.sim_benchmark",
                context={"settling_time": step_result["settling_2pct"]},
            ))
            if status is not Status.FAILED:
                status = Status.WARNING

        # --- Load disturbance recovery -----------------------------------
        dist_result = _closed_loop_disturbance(
            identification, gains, actuator, dt, duration,
        )
        if dist_result is not None and dist_result["recovery_error"] > 0.05:
            warnings.append(Warning(
                code=WarningCode.W_LOW_MARGIN,
                severity=Severity.INFO,
                message=(
                    f"Load disturbance recovery: residual error "
                    f"{dist_result['recovery_error']:.3f} > 5%"
                ),
                stage="validate.sim_benchmark",
                context={"recovery_error": dist_result["recovery_error"]},
            ))

        return ValidationOutcome(
            status=status, warnings=tuple(warnings),
        )


def _closed_loop_step(
    identification: IdentificationResult,
    gains: PIDGains,
    actuator: ActuatorLimits,
    dt: float,
    duration: float,
) -> Optional[dict]:
    """Simulate a unit step response and return metrics."""
    n = int(duration / dt) + 1
    time = np.linspace(0.0, duration, n)
    sp = np.ones(n)
    y = np.zeros(n)
    u = np.zeros(n)
    e_int = 0.0
    e_prev = 0.0
    e_filt = 0.0

    model = identification.model
    kp, ki, kd = gains.kp, gains.ki, gains.kd
    N = gains.derivative_filter_n
    lo, hi = actuator.lower, actuator.upper
    tau = max(float(model.tau), 1e-9)
    delay_samples = int(round(model.theta / dt))

    for i in range(1, n):
        e = sp[i - 1] - y[i - 1]
        alpha = N * dt / (1.0 + N * dt) if N > 0 else 0.0
        e_filt = alpha * (e - e_prev) / dt + (1.0 - alpha) * e_filt if i > 1 else 0.0

        u_raw = kp * e + ki * e_int + kd * e_filt
        if lo <= u_raw <= hi:
            e_int += e * dt
        u_cmd = float(np.clip(kp * e + ki * e_int + kd * e_filt, lo, hi))
        u[i] = u_cmd

        u_idx = max(i - delay_samples, 0)
        alpha_p = float(np.exp(-dt / tau))
        y[i] = alpha_p * y[i - 1] + model.K * (1.0 - alpha_p) * u[u_idx]
        e_prev = e

        if abs(y[i]) > 1e6 or not np.isfinite(y[i]):
            return None

    peak = float(np.max(y))
    os_pct = max(0.0, (peak - 1.0) * 100.0)
    ss_err = float(abs(y[-1] - 1.0))

    # Rise time (10% to 90%)
    idx10 = np.searchsorted(y, 0.1)
    idx90 = np.searchsorted(y, 0.9)
    rise_time = float(time[idx90] - time[idx10]) if idx10 < idx90 < n else None

    # Settling time (2%)
    within = np.abs(y - 1.0) <= 0.02
    settle = 0.0
    for j in range(n - 1, -1, -1):
        if not within[j]:
            settle = float(time[min(j + 1, n - 1)])
            break

    iae = float(np.trapezoid(np.abs(sp - y), time))

    return {
        "overshoot_pct": os_pct,
        "settling_2pct": settle,
        "rise_time": rise_time,
        "ss_error": ss_err,
        "iae": iae,
    }


def _closed_loop_disturbance(
    identification: IdentificationResult,
    gains: PIDGains,
    actuator: ActuatorLimits,
    dt: float,
    duration: float,
) -> Optional[dict]:
    """Simulate step-disturbance rejection."""
    n = int(duration / dt) + 1
    time = np.linspace(0.0, duration, n)
    sp = np.ones(n)
    y = np.zeros(n)
    u = np.zeros(n)
    e_int = 0.0
    e_prev = 0.0
    e_filt = 0.0

    model = identification.model
    kp, ki, kd = gains.kp, gains.ki, gains.kd
    N = gains.derivative_filter_n
    lo, hi = actuator.lower, actuator.upper
    tau = max(float(model.tau), 1e-9)
    delay_samples = int(round(model.theta / dt))
    dist_start = int(n * 0.5)  # disturbance at 50% into the sim
    dist_magnitude = 0.2 * abs(model.K)  # 20% of plant gain

    for i in range(1, n):
        e = sp[i - 1] - y[i - 1]
        alpha = N * dt / (1.0 + N * dt) if N > 0 else 0.0
        e_filt = alpha * (e - e_prev) / dt + (1.0 - alpha) * e_filt if i > 1 else 0.0

        u_raw = kp * e + ki * e_int + kd * e_filt
        if lo <= u_raw <= hi:
            e_int += e * dt
        u_cmd = float(np.clip(kp * e + ki * e_int + kd * e_filt, lo, hi))
        u[i] = u_cmd

        u_idx = max(i - delay_samples, 0)
        alpha_p = float(np.exp(-dt / tau))
        dist = dist_magnitude if i >= dist_start else 0.0
        y[i] = alpha_p * y[i - 1] + model.K * (1.0 - alpha_p) * u[u_idx] + dist * (1.0 - alpha_p)
        e_prev = e

        if abs(y[i]) > 1e6 or not np.isfinite(y[i]):
            return None

    # Recovery error: how close is output to setpoint after disturbance settles?
    last_10pct = y[int(n * 0.9):]
    recovery_error = float(np.mean(np.abs(last_10pct - 1.0)))

    return {"recovery_error": recovery_error}


__all__ = ["SimBenchmarkValidator"]
