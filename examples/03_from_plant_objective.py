"""03 — Autotune with objectives and actuator limits.

Shows how to constrain the tuner with performance targets.

Usage:
    python examples/03_from_plant_objective.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pid_control.autotune import PIDAutotuner, Objective, Status


def main(show: bool = False, output: str = "output") -> None:
    # Synthetic FOPDT data
    K, tau, theta, dt = 1.5, 4.0, 0.8, 0.02
    t = np.arange(0.0, 40.0, dt)
    u = np.zeros_like(t); u[t >= 1.0] = 1.0
    y = np.zeros_like(t); ds = int(theta / dt)
    for i in range(1, len(t)):
        ud = u[i - ds] if i > ds else 0.0
        y[i] = y[i - 1] + dt / tau * (K * ud - y[i - 1])

    result = (
        PIDAutotuner.from_arrays(t, u, y)
        .with_objective(Objective(
            max_overshoot_pct=5.0,
            max_settling_time=8.0,
            min_phase_margin_deg=45.0,
            max_Ms=1.6,
        ))
        .with_actuator_limits(lower=-10.0, upper=10.0, rate_limit=50.0)
        .tune()
    )

    print(result.report())

    if result.warnings:
        print("\nWarnings raised:")
        for w in result.warnings:
            print(f"  {w.code.value}: {w.message}")

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    result.save(out / "objective_tune.json")

    if show:
        result.plot(show=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--output", default="output")
    args = parser.parse_args()
    main(show=args.show, output=args.output)
