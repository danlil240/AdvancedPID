"""Chirp (frequency-sweep) experiment (PLAN.md T3.3).

Generates a sinusoidal sweep (linear or logarithmic) that excites a
plant across a range of frequencies.  Useful for lightly-damped or
higher-order plants where a simple step response provides limited
information about dynamic behaviour.

The experiment drives a plant (or records the designed input for
offline use) and returns an :class:`ExperimentRecord`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
from scipy.signal import chirp as scipy_chirp

from pid_control.autotune.experiments.base import ExperimentRecord
from pid_control.autotune.types import DataQuality, Status, Warning


@dataclass(frozen=True)
class ChirpExperiment:
    """Linear or logarithmic sine-sweep experiment.

    Parameters
    ----------
    f_start : float
        Start frequency in Hz.
    f_end : float
        End frequency in Hz.
    duration : float
        Total sweep duration in seconds.
    amplitude : float
        Peak amplitude of the chirp signal (added to *bias*).
    bias : float
        DC offset (operating-point control output).
    method : ``"linear"`` or ``"logarithmic"``
        Sweep type.  Logarithmic gives equal energy per decade and is
        preferred when the bandwidth spans more than one decade.
    dt : float
        Sample time.
    pre_dwell : float
        Seconds of constant *bias* before the sweep begins (baseline).
    post_dwell : float
        Seconds of constant *bias* after the sweep ends (settling).
    """

    f_start: float = 0.01
    f_end: float = 1.0
    duration: float = 100.0
    amplitude: float = 1.0
    bias: float = 0.0
    method: str = "logarithmic"
    dt: float = 0.01
    pre_dwell: float = 5.0
    post_dwell: float = 5.0
    name: str = "chirp"

    def __post_init__(self) -> None:
        if self.f_start <= 0 or self.f_end <= 0:
            raise ValueError("f_start and f_end must be positive")
        if self.f_start >= self.f_end:
            raise ValueError("f_start must be less than f_end")
        if self.duration <= 0:
            raise ValueError("duration must be positive")
        if self.amplitude <= 0:
            raise ValueError("amplitude must be positive")
        if self.method not in ("linear", "logarithmic"):
            raise ValueError("method must be 'linear' or 'logarithmic'")

    def _generate_input(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(time, input)`` arrays for the full experiment."""
        n_pre = max(1, int(self.pre_dwell / self.dt))
        n_sweep = max(1, int(self.duration / self.dt))
        n_post = max(1, int(self.post_dwell / self.dt))
        n_total = n_pre + n_sweep + n_post

        t = np.arange(n_total) * self.dt
        u = np.full(n_total, self.bias, dtype=np.float64)

        # Sweep portion
        t_sweep = np.arange(n_sweep) * self.dt
        sweep_method = "logarithmic" if self.method == "logarithmic" else "linear"
        sweep = self.amplitude * scipy_chirp(
            t_sweep, f0=self.f_start, f1=self.f_end,
            t1=self.duration, method=sweep_method,
        )
        u[n_pre: n_pre + n_sweep] = self.bias + sweep

        return t, u

    def run(self, plant: Any | None = None) -> ExperimentRecord:
        """Drive *plant* with the chirp signal.

        Parameters
        ----------
        plant : BasePlant or None
            If *None*, returns the designed input with a zero output
            (useful for offline experiment design / recording the
            command signal to apply manually).
        """
        t, u = self._generate_input()
        n = len(t)
        y = np.zeros(n, dtype=np.float64)

        if plant is not None:
            if hasattr(plant, "reset"):
                plant.reset()
            for i in range(n):
                y[i] = plant.update(u[i])

        warnings: tuple[Warning, ...] = ()
        quality: Optional[DataQuality] = None
        if plant is None:
            # No plant → output is empty; quality is meaningless
            quality = DataQuality(
                status=Status.OK,
                snr_db=None,
                excitation_energy=float(np.sum((u - self.bias) ** 2) * self.dt),
                has_steady_state=True,
                n_samples=n,
                sample_time=self.dt,
            )

        return ExperimentRecord(
            time=t,
            input=u,
            output=y,
            sample_time=self.dt,
            operating_point_input=self.bias,
            operating_point_output=float(y[0]) if plant is not None else 0.0,
            quality=quality,
            warnings=warnings,
            meta={
                "experiment": self.name,
                "f_start": self.f_start,
                "f_end": self.f_end,
                "duration": self.duration,
                "method": self.method,
                "amplitude": self.amplitude,
            },
        )
