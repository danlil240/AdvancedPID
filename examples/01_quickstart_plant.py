"""01 — Quickstart: autotune from a known plant model.

Demonstrates the simplest possible PIDAutotuner usage:
give it a plant, get back tuned gains.

Usage:
    python examples/01_quickstart_plant.py
    python examples/01_quickstart_plant.py --show     # interactive plot
    python examples/01_quickstart_plant.py --output ./output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure the repo root is on sys.path for local development
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pid_control.autotune import PIDAutotuner, Objective, Status


def main(show: bool = False, output: str = "output") -> None:
    # 1. Generate a synthetic FOPDT step dataset (real users have a CSV)
    K, tau, theta, dt = 2.0, 3.0, 0.5, 0.02
    t = np.arange(0.0, 30.0, dt)
    u = np.zeros_like(t); u[t >= 1.0] = 1.0
    y = np.zeros_like(t); ds = int(theta / dt)
    for i in range(1, len(t)):
        ud = u[i - ds] if i > ds else 0.0
        y[i] = y[i - 1] + dt / tau * (K * ud - y[i - 1])

    # 2. One-liner autotune
    result = PIDAutotuner.from_arrays(t, u, y).tune()

    # 3. Inspect
    print(result.report())

    if result.is_usable:
        ctrl = result.build_controller()
        print(f"\nController ready: Kp={ctrl.params.kp:.4f}, "
              f"Ki={ctrl.params.ki:.4f}, Kd={ctrl.params.kd:.4f}")

    # 4. Save artifacts
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    result.save(out / "quickstart_plant.json")

    if show:
        result.plot(show=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quickstart: tune from plant data")
    parser.add_argument("--show", action="store_true", help="Show interactive plot")
    parser.add_argument("--output", default="output", help="Output directory")
    args = parser.parse_args()
    main(show=args.show, output=args.output)
