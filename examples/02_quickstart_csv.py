"""02 — Quickstart: autotune from a CSV file.

Load step-response data from a CSV and get tuned PID gains.

Usage:
    python examples/02_quickstart_csv.py
    python examples/02_quickstart_csv.py --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pid_control.autotune import PIDAutotuner


def main(show: bool = False, output: str = "output") -> None:
    csv_path = Path(__file__).parent / "data" / "fopdt_step.csv"
    if not csv_path.exists():
        print(f"Sample data not found at {csv_path}.")
        print("Run  python examples/data/generate_samples.py  first.")
        return

    result = PIDAutotuner.from_csv(str(csv_path)).tune()

    print(result.report())

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    result.save(out / "quickstart_csv.json")

    if show:
        result.plot(show=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quickstart: tune from CSV")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--output", default="output")
    args = parser.parse_args()
    main(show=args.show, output=args.output)
