"""05 — Compare tuning rules on the same plant.

Runs ZN, Cohen-Coon, IMC, AMIGO, and Skogestad on the same FOPDT
dataset and prints a side-by-side comparison.

Usage:
    python examples/05_compare_rules.py
    python examples/05_compare_rules.py --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pid_control.autotune import PIDAutotuner, Status
from pid_control.autotune.rules.ziegler_nichols import ZieglerNicholsRule
from pid_control.autotune.rules.cohen_coon import CohenCoonRule
from pid_control.autotune.rules.imc import IMCRule
from pid_control.autotune.rules.amigo import AMIGORule
from pid_control.autotune.rules.skogestad import SkogestadRule


def main(show: bool = False, output: str = "output") -> None:
    # Synthetic FOPDT data
    K, tau, theta, dt = 2.0, 3.0, 0.5, 0.02
    t = np.arange(0.0, 30.0, dt)
    u = np.zeros_like(t); u[t >= 1.0] = 1.0
    y = np.zeros_like(t); ds = int(theta / dt)
    for i in range(1, len(t)):
        ud = u[i - ds] if i > ds else 0.0
        y[i] = y[i - 1] + dt / tau * (K * ud - y[i - 1])

    rules = {
        "Ziegler-Nichols": ZieglerNicholsRule(),
        "Cohen-Coon": CohenCoonRule(),
        "IMC": IMCRule(),
        "AMIGO": AMIGORule(),
        "Skogestad SIMC": SkogestadRule(),
    }

    print(f"{'Rule':20s} | {'Kp':>8s} | {'Ki':>8s} | {'Kd':>8s} | Status")
    print("-" * 70)

    for name, rule in rules.items():
        result = (
            PIDAutotuner.from_arrays(t, u, y)
            .set_rule(rule)
            .set_tuner(None)  # analytical rule only, no DE
            .tune()
        )
        g = result.gains
        print(f"{name:20s} | {g.kp:8.4f} | {g.ki:8.4f} | {g.kd:8.4f} | {result.status.value}")

    print()

    # Also run the full pipeline (IMC + DE refinement) for comparison
    result_full = PIDAutotuner.from_arrays(t, u, y).tune()
    g = result_full.gains
    print(f"{'IMC + DE optimized':20s} | {g.kp:8.4f} | {g.ki:8.4f} | {g.kd:8.4f} | {result_full.status.value}")
    print(f"\nConfidence: {result_full.confidence.score:.2f}")

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    result_full.save(out / "compare_rules.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--output", default="output")
    args = parser.parse_args()
    main(show=args.show, output=args.output)
