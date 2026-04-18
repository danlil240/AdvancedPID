"""Stability-margin validation (PLAN.md T5.1).

Builds the closed-loop transfer function from the identified model +
tuned gains using the ``control`` library and reports gain margin,
phase margin, sensitivity peak, and complementary-sensitivity peak.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from pid_control.autotune.types import (
    ActuatorLimits,
    IdentificationResult,
    MarginReport,
    ModelType,
    Objective,
    PerformanceReport,
    PIDGains,
    Severity,
    Status,
    Warning,
    WarningCode,
)
from pid_control.autotune.validation.base import ValidationOutcome


def _build_loop_tf(identification: IdentificationResult, gains: PIDGains):
    """Build the open-loop transfer function ``L(s) = C(s) · G(s)``."""
    import control as ct

    model = identification.model
    K = model.K
    tau = max(float(model.tau), 1e-9)
    theta = float(model.theta)

    # Plant G(s)
    if model.model_type is ModelType.SOPDT and model.tau2 is not None:
        tau2 = max(float(model.tau2), 1e-9)
        G = ct.tf([K], [tau * tau2, tau + tau2, 1.0])
    elif model.model_type is ModelType.SECOND_ORDER:
        wn = float(model.natural_frequency or 1.0)
        zeta = float(model.damping_ratio or 1.0)
        G = ct.tf([K * wn ** 2], [1.0, 2 * zeta * wn, wn ** 2])
    else:
        # FOPDT or fallback
        G = ct.tf([K], [tau, 1.0])

    # Padé delay approximation (2nd order)
    if theta > 0:
        delay_num, delay_den = _pade(theta, order=3)
        delay_tf = ct.tf(delay_num, delay_den)
        G = G * delay_tf

    # PID controller: C(s) = Kp + Ki/s + Kd·N·s/(s+N)
    kp, ki, kd = gains.kp, gains.ki, gains.kd
    N = gains.derivative_filter_n

    # P term
    C = ct.tf([kp], [1.0])
    # I term
    if abs(ki) > 1e-12:
        C = C + ct.tf([ki], [1.0, 0.0])
    # Filtered D term
    if abs(kd) > 1e-12 and N > 0:
        C = C + ct.tf([kd * N, 0.0], [1.0, N])

    return C * G


def _pade(theta: float, order: int = 3):
    """Padé approximation of e^{-θs}."""
    import control as ct
    num, den = ct.pade(theta, order)
    return num, den


def compute_margins(
    identification: IdentificationResult, gains: PIDGains,
) -> MarginReport:
    """Compute stability margins from identified model + gains."""
    import control as ct

    try:
        L = _build_loop_tf(identification, gains)
    except Exception:
        return MarginReport()

    try:
        gm, pm, _, _ = ct.margin(L)
        gm_db = float(20 * np.log10(gm)) if gm is not None and np.isfinite(gm) and gm > 0 else None
        pm_deg = float(pm) if pm is not None and np.isfinite(pm) else None
    except Exception:
        gm_db, pm_deg = None, None

    # Sensitivity peaks
    Ms, Mt = None, None
    try:
        S = ct.feedback(1.0, L)
        T = ct.feedback(L, 1.0)
        omega = np.logspace(-3, 3, 2000)
        S_resp = ct.frequency_response(S, omega)
        T_resp = ct.frequency_response(T, omega)
        S_mag = np.abs(getattr(S_resp, 'frdata', getattr(S_resp, 'fresp', None))).flatten()
        T_mag = np.abs(getattr(T_resp, 'frdata', getattr(T_resp, 'fresp', None))).flatten()
        Ms = float(np.max(S_mag))
        Mt = float(np.max(T_mag))
    except Exception:
        pass

    # Delay margin
    dm = None
    if pm_deg is not None:
        try:
            _, _, wpc, _ = ct.margin(L)
            if wpc is not None and np.isfinite(wpc) and wpc > 0:
                dm = float(np.radians(pm_deg) / wpc)
        except Exception:
            pass

    return MarginReport(
        gain_margin_db=gm_db,
        phase_margin_deg=pm_deg,
        delay_margin_s=dm,
        sensitivity_peak=Ms,
        complementary_sensitivity_peak=Mt,
    )


@dataclass(frozen=True)
class StabilityValidator:
    """Check loop margins against the user's :class:`Objective` thresholds."""

    name: str = "stability"

    def validate(
        self,
        identification: IdentificationResult,
        gains: PIDGains,
        objective: Objective,
        actuator: ActuatorLimits,
    ) -> ValidationOutcome:
        margins = compute_margins(identification, gains)
        warnings: List[Warning] = []
        status = Status.OK

        # Phase margin
        if margins.phase_margin_deg is not None:
            if margins.phase_margin_deg < objective.min_phase_margin_deg:
                warnings.append(Warning(
                    code=WarningCode.W_LOW_MARGIN,
                    severity=Severity.WARNING,
                    message=(
                        f"Phase margin {margins.phase_margin_deg:.1f}° < "
                        f"target {objective.min_phase_margin_deg:.1f}°"
                    ),
                    stage="validate.stability",
                    context={"phase_margin_deg": margins.phase_margin_deg},
                ))
                status = Status.WARNING
            if margins.phase_margin_deg <= 0:
                warnings.append(Warning(
                    code=WarningCode.E_UNSTABLE,
                    severity=Severity.ERROR,
                    message="Closed-loop is unstable (non-positive phase margin).",
                    stage="validate.stability",
                ))
                status = Status.FAILED

        # Sensitivity peak
        if margins.sensitivity_peak is not None:
            if margins.sensitivity_peak > objective.max_Ms:
                warnings.append(Warning(
                    code=WarningCode.W_LOW_MARGIN,
                    severity=Severity.WARNING,
                    message=(
                        f"Sensitivity peak Ms={margins.sensitivity_peak:.2f} > "
                        f"target {objective.max_Ms:.2f}"
                    ),
                    stage="validate.stability",
                    context={"Ms": margins.sensitivity_peak},
                ))
                if status is not Status.FAILED:
                    status = Status.WARNING

        # Complementary sensitivity peak
        if margins.complementary_sensitivity_peak is not None:
            if margins.complementary_sensitivity_peak > objective.max_Mt:
                warnings.append(Warning(
                    code=WarningCode.W_LOW_MARGIN,
                    severity=Severity.WARNING,
                    message=(
                        f"Complementary sensitivity Mt={margins.complementary_sensitivity_peak:.2f} > "
                        f"target {objective.max_Mt:.2f}"
                    ),
                    stage="validate.stability",
                    context={"Mt": margins.complementary_sensitivity_peak},
                ))
                if status is not Status.FAILED:
                    status = Status.WARNING

        return ValidationOutcome(
            status=status,
            warnings=tuple(warnings),
            margins=margins,
        )


@dataclass(frozen=True)
class RobustnessValidator:
    """Perturb model parameters and check the tuning survives (PLAN.md T5.2)."""

    perturbation_fractions: tuple = (0.2, 0.5)
    name: str = "robustness"

    def validate(
        self,
        identification: IdentificationResult,
        gains: PIDGains,
        objective: Objective,
        actuator: ActuatorLimits,
    ) -> ValidationOutcome:
        from pid_control.autotune.identification.simulate import simulate_model

        warnings: List[Warning] = []
        model = identification.model
        dt = max(float(model.tau) / 50.0, 0.01)
        duration = max(10.0 * float(model.tau + model.theta), 5.0)
        n = int(duration / dt) + 1
        time = np.linspace(0.0, duration, n)
        sp = np.ones(n)

        worst_iae = 0.0
        any_unstable = False

        for frac in self.perturbation_fractions:
            for sign in (-1, +1):
                for param in ("K", "tau", "theta"):
                    perturbed = _perturb_model(model, param, sign * frac)
                    from pid_control.autotune.types import IdentificationResult as IR
                    pid_result = IR(
                        model=perturbed,
                        fit_quality_r2=identification.fit_quality_r2,
                    )
                    try:
                        iae = _closed_loop_iae(pid_result, gains, time, sp, dt, actuator)
                    except Exception:
                        any_unstable = True
                        continue
                    if not np.isfinite(iae) or iae > 1e8:
                        any_unstable = True
                    else:
                        worst_iae = max(worst_iae, iae)

        status = Status.OK
        if any_unstable:
            warnings.append(Warning(
                code=WarningCode.W_FRAGILE,
                severity=Severity.WARNING,
                message="Tuning is unstable under model perturbation.",
                stage="validate.robustness",
            ))
            status = Status.WARNING

        return ValidationOutcome(
            status=status,
            warnings=tuple(warnings),
        )


def _perturb_model(model, param: str, frac: float):
    """Return a new TransferFunctionModel with one parameter perturbed."""
    from pid_control.autotune.types import TransferFunctionModel
    d = {
        "model_type": model.model_type,
        "K": model.K,
        "tau": model.tau,
        "theta": model.theta,
        "tau2": model.tau2,
        "natural_frequency": model.natural_frequency,
        "damping_ratio": model.damping_ratio,
    }
    if param in d and d[param] is not None:
        d[param] = d[param] * (1.0 + frac)
    d["theta"] = max(d["theta"], 0.0)
    d["tau"] = max(d["tau"], 1e-9)
    return TransferFunctionModel(**d)


def _closed_loop_iae(identification, gains, time, sp, dt, actuator):
    """Quick closed-loop sim returning IAE."""
    n = len(time)
    y = np.zeros(n)
    u = np.zeros(n)
    e_int = 0.0
    model = identification.model
    kp, ki, kd = gains.kp, gains.ki, gains.kd
    lo, hi = actuator.lower, actuator.upper
    tau = max(float(model.tau), 1e-9)
    delay_samples = int(round(model.theta / dt))

    for i in range(1, n):
        e = sp[i - 1] - y[i - 1]
        e_int += e * dt
        u_cmd = float(np.clip(kp * e + ki * e_int + kd * (e - (sp[max(i - 2, 0)] - y[max(i - 2, 0)])) / dt, lo, hi))
        u[i] = u_cmd
        u_idx = max(i - delay_samples, 0)
        alpha_p = float(np.exp(-dt / tau))
        y[i] = alpha_p * y[i - 1] + model.K * (1.0 - alpha_p) * u[u_idx]
        if abs(y[i]) > 1e6:
            return float("inf")

    return float(np.trapezoid(np.abs(sp - y), time))


__all__ = ["StabilityValidator", "RobustnessValidator", "compute_margins"]
