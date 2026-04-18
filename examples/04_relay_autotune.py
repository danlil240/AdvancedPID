"""04 — Autotune from a simulated plant (step experiment).

Demonstrates using PIDAutotuner.from_plant() with a live plant
object rather than pre-recorded data.  The pipeline automatically
runs a step experiment, identifies the plant, and tunes the PID.

Usage:
    python examples/04_relay_autotune.py
    python examples/04_relay_autotune.py --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pid_control.autotune import PIDAutotuner
from pid_control.plants import DelayPlant, FirstOrderPlant


def main(show: bool = False, output: str = "output") -> None:
    # Build a FOPDT plant: K=2, tau=3, theta=0.5
    base = FirstOrderPlant(gain=2.0, time_constant=3.0, sample_time=0.01)
    plant = DelayPlant(base, delay_time=0.5)

    result = (
        PIDAutotuner.from_plant(plant)
        .tune()
    )

    print(result.report())

    if result.is_usable:
        ctrl = result.build_controller()
        print(f"\nController: Kp={ctrl.params.kp:.4f}, "
              f"Ki={ctrl.params.ki:.4f}, Kd={ctrl.params.kd:.4f}")

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    result.save(out / "from_plant_step.json")

    if show:
        result.plot(show=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autotune from simulated plant")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--output", default="output")
    args = parser.parse_args()
    main(show=args.show, output=args.output)
