"""Biased relay auto-tune experiment (PLAN.md T3.1, fixes C2).

Implements the correct Auto-Tune Variation (ATV) algorithm:

1. Hold control output at operating point ``u0``.
2. Toggle ``u = u0 ± d`` based on *output-side* hysteresis.
3. After ``n_cycles`` of sustained oscillation, measure:
   - Output oscillation amplitude ``a_y`` (peak-to-peak / 2).
   - Period ``T_u`` from zero-crossing intervals.
4. Compute ultimate gain ``K_u = 4d / (π · a_y)``.
5. Return ``K_u``, ``T_u`` on the :class:`ExperimentRecord`.

Key fixes over the old ``RealtimeTuner.relay_feedback_tune``:
- Relay bias at operating point (not absolute setpoint).
- Hysteresis on *output*, not error.
- Correct ``K_u`` formula using measured output amplitude.
- Abort-on-runaway (output exceeds safe bound).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
class RelayExperiment:
    """Biased relay (ATV) excitation.

    Parameters
    ----------
    amplitude
        Relay half-amplitude ``d``.  Control toggles between ``u0-d``
        and ``u0+d``.
    hysteresis
        Output hysteresis band.  The relay switches only when the output
        crosses ``y0 ± hysteresis``.  Prevents chattering on noisy plants.
    n_cycles
        Number of full oscillation cycles to collect before stopping.
    max_duration
        Hard time-out (seconds) to prevent infinite loops.
    abort_limit
        If ``|y - y0|`` exceeds this, the experiment aborts with an error.
    dt
        Simulation time step.
    """

    amplitude: float = 1.0
    hysteresis: float = 0.01
    n_cycles: int = 6
    max_duration: float = 300.0
    abort_limit: Optional[float] = None
    dt: float = 0.01
    name: str = "relay_atv"

    def run(self, plant: Any = None) -> ExperimentRecord:
        if plant is None:
            raise ValueError("RelayExperiment requires a plant object with .update(u) → y")

        dt = self.dt
        d = self.amplitude
        hyst = self.hysteresis
        max_steps = int(self.max_duration / dt)
        _update = _make_updater(plant, dt)

        # --- Measure operating point ---
        # Assume the plant is currently at its operating point.
        # Run a brief settle period with the current output.
        y0 = float(_update(0.0))
        u0 = 0.0

        time_buf: List[float] = [0.0]
        u_buf: List[float] = [u0]
        y_buf: List[float] = [y0]

        relay_high = True  # Start with positive half
        u_cmd = u0 + d
        crossings: List[float] = []
        last_cross_direction = 0  # +1 for upward, -1 for downward

        for step in range(1, max_steps):
            t = step * dt

            y = float(_update(u_cmd))
            time_buf.append(t)
            u_buf.append(u_cmd)
            y_buf.append(y)

            # Abort on runaway
            if self.abort_limit is not None and abs(y - y0) > self.abort_limit:
                return self._make_error_record(
                    time_buf, u_buf, y_buf, dt,
                    "Output exceeded abort limit — plant may be unstable.",
                )

            # Relay switching with hysteresis on output
            deviation = y - y0
            if relay_high and deviation > hyst:
                # Switch to low
                relay_high = False
                u_cmd = u0 - d
                if last_cross_direction != -1:
                    crossings.append(t)
                    last_cross_direction = -1
            elif not relay_high and deviation < -hyst:
                # Switch to high
                relay_high = True
                u_cmd = u0 + d
                if last_cross_direction != 1:
                    crossings.append(t)
                    last_cross_direction = 1

            # Check if we have enough full cycles (2 crossings per cycle)
            full_cycles = (len(crossings) - 1) // 2
            if full_cycles >= self.n_cycles:
                break

        # --- Analyze oscillation ---
        time_arr = np.array(time_buf)
        u_arr = np.array(u_buf)
        y_arr = np.array(y_buf)

        if len(crossings) < 3:
            return self._make_error_record(
                time_buf, u_buf, y_buf, dt,
                "Could not establish oscillation — plant may not respond to relay amplitude.",
            )

        # Period from consecutive same-direction crossings
        periods: List[float] = []
        for i in range(2, len(crossings)):
            periods.append(crossings[i] - crossings[i - 2])  # full cycle = 2 half-cycles

        Tu = float(np.median(periods)) if periods else float(crossings[-1] - crossings[0])

        # Output amplitude: use the steady-state oscillation region
        # (last n_cycles worth of data)
        osc_start = crossings[max(0, len(crossings) - 2 * self.n_cycles)]
        osc_mask = time_arr >= osc_start
        y_osc = y_arr[osc_mask] - y0
        a_y = (float(np.max(y_osc)) - float(np.min(y_osc))) / 2.0

        if a_y < 1e-12:
            return self._make_error_record(
                time_buf, u_buf, y_buf, dt,
                "Output oscillation amplitude near zero — plant gain may be too small.",
            )

        Ku = 4.0 * d / (np.pi * a_y)

        return ExperimentRecord(
            time=time_arr,
            input=u_arr,
            output=y_arr,
            sample_time=dt,
            operating_point_input=u0,
            operating_point_output=y0,
            quality=DataQuality(
                status=Status.OK,
                snr_db=None,
                excitation_energy=float(np.sum(u_arr ** 2) * dt),
                has_steady_state=True,
                n_samples=len(time_arr),
                sample_time=dt,
            ),
            meta={
                "Ku": float(Ku),
                "Tu": float(Tu),
                "a_y": float(a_y),
                "n_crossings": len(crossings),
            },
        )

    def _make_error_record(
        self,
        time_buf: List[float],
        u_buf: List[float],
        y_buf: List[float],
        dt: float,
        message: str,
    ) -> ExperimentRecord:
        time_arr = np.array(time_buf)
        return ExperimentRecord(
            time=time_arr,
            input=np.array(u_buf),
            output=np.array(y_buf),
            sample_time=dt,
            quality=DataQuality(
                status=Status.FAILED,
                snr_db=None,
                excitation_energy=0.0,
                has_steady_state=False,
                n_samples=len(time_arr),
                sample_time=dt,
                warnings=(
                    Warning(
                        code=WarningCode.E_DATA_FLAT,
                        severity=Severity.ERROR,
                        message=message,
                        stage="experiment.relay",
                    ),
                ),
            ),
            warnings=(
                Warning(
                    code=WarningCode.E_DATA_FLAT,
                    severity=Severity.ERROR,
                    message=message,
                    stage="experiment.relay",
                ),
            ),
        )


def _make_updater(plant: Any, dt: float):
    """Return a callable ``f(u) -> y`` adapting to plant's update signature."""
    import inspect
    sig = inspect.signature(plant.update)
    params = [p for p in sig.parameters.values() if p.name != "self"]
    if len(params) >= 2:
        return lambda u: plant.update(u, dt)
    return lambda u: plant.update(u)


__all__ = ["RelayExperiment"]
