"""06 — Diagnosing bad data: integrator rejection.

Demonstrates that the pipeline correctly rejects data from an integrating
plant — it returns Status.FAILED with E_NO_STEADY_STATE instead of
silently producing nonsense gains (this was the C1 bug).

Usage:
    python examples/06_diagnosing_bad_data.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pid_control.autotune import PIDAutotuner, Status, WarningCode


def main(show: bool = False, output: str = "output") -> None:
    csv_path = Path(__file__).parent / "data" / "integrator.csv"
    if not csv_path.exists():
        print(f"Sample data not found at {csv_path}.")
        print("Run  python examples/data/generate_samples.py  first.")
        return

    print("Attempting to autotune from an integrating (ramping) dataset...")
    print("The old API silently produced 'success' with 0% improvement.")
    print("The new pipeline correctly rejects this data.\n")

    result = PIDAutotuner.from_csv(str(csv_path)).tune()

    print(result.report())

    assert result.status is Status.FAILED, "Expected FAILED for integrator data"
    assert result.has_warning(WarningCode.E_NO_STEADY_STATE), (
        "Expected E_NO_STEADY_STATE warning"
    )

    print("\n[OK] Correctly rejected with Status.FAILED and E_NO_STEADY_STATE.")

    try:
        result.build_controller()
        print("ERROR: Should not have been able to build controller!")
    except RuntimeError as e:
        print(f"[OK] build_controller() correctly refused: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--output", default="output")
    args = parser.parse_args()
    main(show=args.show, output=args.output)
