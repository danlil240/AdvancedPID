"""Reference plant battery used by the benchmark harness.

Each entry is a :class:`PlantCase` describing:
  * a stable, parametric simulated plant (from ``pid_control.plants``),
  * the canonical FOPDT/SOPDT ground-truth parameters (for sanity checks),
  * a recommended setpoint and actuator-limit pair,
  * an ``expected_success`` flag — ``True`` if a reasonable PID must be
    achievable, ``False`` if the plant is *intentionally* PID-inappropriate
    (e.g. a pure double integrator) and should be REJECTED by a correct
    autotuner.

The zoo is intentionally small so the smoke benchmark finishes in < 30 s.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Any

from pid_control.plants.first_order import FirstOrderPlant
from pid_control.plants.second_order import SecondOrderPlant
from pid_control.plants.delay_plant import FOPDTPlant
from pid_control.plants.base_plant import BasePlant


@dataclass(frozen=True)
class PlantCase:
    """A single benchmark case."""

    case_id: str
    description: str
    factory: Callable[[], BasePlant]
    setpoint: float
    actuator_low: float
    actuator_high: float
    # Ground-truth FOPDT parameters for post-hoc scoring (nullable for plants
    # that do not admit a clean FOPDT equivalent).
    K_true: float | None = None
    tau_true: float | None = None
    theta_true: float | None = None
    # True => a reasonable PID tuning must exist.  False => autotuner must
    # refuse and raise an error / return FAILED status (exercised by the
    # data-driven path once the safety layer lands).
    expected_success: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


def _fopdt(K: float, tau: float, theta: float, dt: float = 0.01) -> Callable[[], BasePlant]:
    def _factory() -> BasePlant:
        return FOPDTPlant(gain=K, time_constant=tau, dead_time=theta, sample_time=dt)
    return _factory


def _first_order(K: float, tau: float, dt: float = 0.01) -> Callable[[], BasePlant]:
    def _factory() -> BasePlant:
        return FirstOrderPlant(gain=K, time_constant=tau, sample_time=dt)
    return _factory


def _second_order(K: float, wn: float, zeta: float, dt: float = 0.01) -> Callable[[], BasePlant]:
    def _factory() -> BasePlant:
        return SecondOrderPlant(
            gain=K, natural_frequency=wn, damping_ratio=zeta, sample_time=dt
        )
    return _factory


def smoke_zoo() -> list[PlantCase]:
    """Small plant battery (~20 cases) used in the smoke benchmark.

    Covers the regimes that PID is routinely expected to handle: low/high
    dynamics ratio, slow/fast plants, noticeable dead time, and a few
    underdamped oscillators.  Unstable and integrating plants are listed in
    :func:`stress_zoo` instead.
    """
    cases: list[PlantCase] = []

    # --- FOPDT grid over tau/theta ratio --------------------------------
    for K in (0.5, 1.0, 2.5):
        for tau, theta in [
            (5.0, 0.5),   # tau/theta = 10 (easy)
            (2.0, 0.5),   # tau/theta = 4
            (1.0, 0.5),   # tau/theta = 2
            (1.0, 1.0),   # tau/theta = 1 (difficult)
        ]:
            cid = f"fopdt_K{K}_tau{tau}_th{theta}"
            cases.append(
                PlantCase(
                    case_id=cid,
                    description=f"FOPDT K={K} τ={tau}s θ={theta}s",
                    factory=_fopdt(K, tau, theta),
                    setpoint=1.0,
                    actuator_low=-10.0,
                    actuator_high=10.0,
                    K_true=K,
                    tau_true=tau,
                    theta_true=theta,
                )
            )

    # --- Pure first-order (no dead time) -------------------------------
    for K, tau in [(1.0, 1.0), (2.0, 3.0), (0.5, 0.5)]:
        cid = f"first_K{K}_tau{tau}"
        cases.append(
            PlantCase(
                case_id=cid,
                description=f"FirstOrder K={K} τ={tau}s",
                factory=_first_order(K, tau),
                setpoint=1.0,
                actuator_low=-10.0,
                actuator_high=10.0,
                K_true=K,
                tau_true=tau,
                theta_true=0.0,
            )
        )

    # --- Second-order (damping sweep) ----------------------------------
    for zeta in (0.3, 0.7, 1.5):
        cid = f"second_wn1_zeta{zeta}"
        cases.append(
            PlantCase(
                case_id=cid,
                description=f"SecondOrder ωn=1 ζ={zeta}",
                factory=_second_order(1.0, 1.0, zeta),
                setpoint=1.0,
                actuator_low=-10.0,
                actuator_high=10.0,
                K_true=1.0,
                tau_true=1.0 / zeta if zeta > 0 else None,  # rough equivalent
                theta_true=0.0,
            )
        )

    return cases


def stress_zoo() -> list[PlantCase]:
    """Additional cases where the current autotuner is expected to struggle
    or outright fail. Used by the full benchmark (not the smoke path) to
    document the baseline's blind spots.

    The current implementation does not actually *refuse* bad data — it will
    happily produce nonsense.  The baseline captures that behaviour so the
    new safety layer can demonstrate correct rejection in the same harness.
    """
    return []  # populated once safety gating lands; kept minimal for now


__all__ = ["PlantCase", "smoke_zoo", "stress_zoo"]
