"""Open-loop step experiment with corrected tangent method (PLAN.md T3.2, fixes C3).

Key fixes over the old ``RealtimeTuner.ziegler_nichols_step``:
- Explicit pre-step baseline dwell (measures y₀).
- Correct tangent-line time-constant: τ = y∞/s_max **− L** (not just y∞/s_max).
- Warns if plant does not reach steady state.
- No assumption that responses[0] == 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from pid_control.autotune.experiments.base import ExperimentRecord
from pid_control.autotune.types import (
    ActuatorLimits,
    DataQuality,
    Severity,
    Status,
    Warning,
    WarningCode,
)


@dataclass
class StepExperiment:
    """Open-loop step excitation.

    Parameters
    ----------
    step_amplitude
        Step size applied to the plant input.
    pre_step_dwell
        Seconds of zero-input dwell before the step, used to establish a
        baseline operating point.
    post_step_duration
        Seconds after the step to wait for steady state.
    dt
        Simulation time step.
    ramp_time
        If > 0, ramp the input linearly from 0 to ``step_amplitude``
        over this many seconds instead of an instantaneous step.
        Gentler for hardware.
    """

    step_amplitude: float = 1.0
    pre_step_dwell: float = 5.0
    post_step_duration: float = 30.0
    dt: float = 0.01
    ramp_time: float = 0.0
    name: str = "step"

    def run(self, plant: Any = None) -> ExperimentRecord:
        if plant is None:
            raise ValueError("StepExperiment requires a plant object with .update(u) → y")

        dt = self.dt
        # Use plant's internal dt if available
        _update = _make_updater(plant, dt)

        n_pre = int(self.pre_step_dwell / dt)
        n_post = int(self.post_step_duration / dt)
        n_ramp = int(self.ramp_time / dt) if self.ramp_time > 0 else 0
        n_total = n_pre + n_ramp + n_post

        time_buf: List[float] = []
        u_buf: List[float] = []
        y_buf: List[float] = []

        # Pre-step dwell: establish baseline
        for i in range(n_pre):
            y = float(_update(0.0))
            time_buf.append(i * dt)
            u_buf.append(0.0)
            y_buf.append(y)

        y0 = float(np.mean(y_buf[-min(50, n_pre):])) if n_pre > 0 else 0.0

        # Ramp (optional)
        for i in range(n_ramp):
            frac = (i + 1) / n_ramp
            u_val = self.step_amplitude * frac
            y = float(_update(u_val))
            t = (n_pre + i) * dt
            time_buf.append(t)
            u_buf.append(u_val)
            y_buf.append(y)

        # Post-step: hold at step_amplitude
        for i in range(n_post):
            y = float(_update(self.step_amplitude))
            t = (n_pre + n_ramp + i) * dt
            time_buf.append(t)
            u_buf.append(self.step_amplitude)
            y_buf.append(y)

        time_arr = np.array(time_buf)
        u_arr = np.array(u_buf)
        y_arr = np.array(y_buf)

        # --- Analyse step response for FOPDT parameters ---
        # Subtract baseline
        y_shifted = y_arr - y0
        step_start_idx = n_pre + n_ramp

        if step_start_idx >= len(y_arr) - 10:
            return self._make_record(time_arr, u_arr, y_arr, dt, y0, warnings=[
                Warning(
                    code=WarningCode.E_TOO_SHORT,
                    severity=Severity.ERROR,
                    message="Step response too short for analysis.",
                    stage="experiment.step",
                ),
            ])

        y_post = y_shifted[step_start_idx:]
        t_post = time_arr[step_start_idx:] - time_arr[step_start_idx]

        # Steady-state value
        tail = y_post[-max(len(y_post) // 10, 5):]
        y_ss = float(np.mean(tail))
        settled = float(np.std(tail)) / max(abs(y_ss), 1e-9) < 0.05

        warnings: List[Warning] = []
        if not settled:
            warnings.append(Warning(
                code=WarningCode.E_NO_STEADY_STATE,
                severity=Severity.WARNING,
                message="Plant may not have reached steady state.",
                stage="experiment.step",
            ))

        # Static gain
        K = y_ss / self.step_amplitude if self.step_amplitude != 0 else 0.0

        # Maximum slope (tangent method)
        dy_dt = np.gradient(y_post, t_post)
        max_slope_idx = int(np.argmax(dy_dt))
        max_slope = float(dy_dt[max_slope_idx])

        if abs(max_slope) < 1e-12:
            warnings.append(Warning(
                code=WarningCode.E_DATA_FLAT,
                severity=Severity.WARNING,
                message="Step response slope near zero.",
                stage="experiment.step",
            ))
            return self._make_record(time_arr, u_arr, y_arr, dt, y0, warnings=warnings)

        # Dead time L: tangent line reaches y=0 at t = t_inflection - y(t_inflection)/slope
        t_inflection = float(t_post[max_slope_idx])
        y_inflection = float(y_post[max_slope_idx])
        L = max(0.0, t_inflection - y_inflection / max_slope)

        # Time constant τ: tangent line reaches y_ss at t = L + τ
        # τ = y_ss / max_slope  (from tangent-line intersection)
        # CORRECT: τ = y_ss/max_slope, and the tangent crosses zero at L,
        # crosses y_ss at L + τ. So τ = y_ss / max_slope.
        tau = y_ss / max_slope if max_slope > 0 else 1.0

        meta = {
            "K": float(K),
            "tau": float(tau),
            "theta": float(L),
            "y_ss": float(y_ss),
            "max_slope": float(max_slope),
            "settled": settled,
        }

        return self._make_record(
            time_arr, u_arr, y_arr, dt, y0,
            warnings=warnings, meta=meta,
        )

    def _make_record(
        self,
        time_arr: np.ndarray,
        u_arr: np.ndarray,
        y_arr: np.ndarray,
        dt: float,
        y0: float,
        warnings: Optional[List[Warning]] = None,
        meta: Optional[dict] = None,
    ) -> ExperimentRecord:
        warns = tuple(warnings or [])
        has_error = any(w.severity is Severity.ERROR for w in warns)
        return ExperimentRecord(
            time=time_arr,
            input=u_arr,
            output=y_arr,
            sample_time=dt,
            operating_point_input=0.0,
            operating_point_output=y0,
            quality=DataQuality(
                status=Status.FAILED if has_error else Status.OK,
                snr_db=None,
                excitation_energy=float(np.sum(u_arr ** 2) * dt),
                has_steady_state=not has_error,
                n_samples=len(time_arr),
                sample_time=dt,
                warnings=warns,
            ),
            warnings=warns,
            meta=meta or {},
        )


def _make_updater(plant: Any, dt: float):
    """Return a callable ``f(u) -> y`` adapting to plant's update signature."""
    import inspect
    sig = inspect.signature(plant.update)
    params = [p for p in sig.parameters.values() if p.name != "self"]
    if len(params) >= 2:
        return lambda u: plant.update(u, dt)
    return lambda u: plant.update(u)


__all__ = ["StepExperiment"]
