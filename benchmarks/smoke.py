#!/usr/bin/env python
"""Fast smoke benchmark (≤ 30 s) suitable for PR checks.

Usage::

    python -m benchmarks.smoke                     # run quick subset
    python -m benchmarks.smoke --compare results/baseline_pre_refactor.json

This runs the first 6 smoke-zoo cases with short collect/replay durations
and a low iteration cap so the whole suite fits in ~30 s on a laptop.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

# Make project root importable.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

from benchmarks.plant_zoo import smoke_zoo  # noqa: E402
from benchmarks.run import (  # noqa: E402
    CaseResult,
    _run_one,
    _aggregate,
    _compare,
)

# Tuned so the entire suite finishes < 30 s on commodity hardware.
_MAX_CASES = 6
_COLLECT_DURATION = 10.0
_REPLAY_DURATION = 10.0
_MAX_ITER = 30
_SEED = 42


def _run_smoke(tmp_dir: Path) -> List[CaseResult]:
    cases = smoke_zoo()[:_MAX_CASES]
    results: List[CaseResult] = []
    for c in cases:
        r = _run_one(
            c,
            tmp_dir=tmp_dir,
            step_amplitude=1.0,
            collect_duration=_COLLECT_DURATION,
            replay_duration=_REPLAY_DURATION,
            max_iter=_MAX_ITER,
            seed=_SEED,
        )
        results.append(r)
    return results


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Quick smoke benchmark (≤30 s).")
    parser.add_argument(
        "--compare",
        default=None,
        help="Path to a baseline JSON to compare against.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Write results JSON to this path.",
    )
    args = parser.parse_args(argv)

    tmp_dir = _ROOT / "output" / "_bench_smoke"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    results = _run_smoke(tmp_dir)
    elapsed = time.monotonic() - t0

    summary = _aggregate(results)
    print(f"Smoke benchmark: {summary['ok']}/{summary['total_cases']} ok, "
          f"{summary['errors']} errors, {elapsed:.1f} s elapsed")
    if summary["median_iae"] is not None:
        print(f"  median IAE={summary['median_iae']:.3f}  "
              f"overshoot={summary['median_overshoot_pct']:.1f}%  "
              f"settling={summary['median_settling_time']:.2f} s")

    payload = {
        "suite": "smoke-quick",
        "elapsed_seconds": elapsed,
        "summary": summary,
        "results": [r.to_dict() for r in results],
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Results saved to {out_path}")

    if args.compare:
        return _compare(Path(args.compare), payload)

    return 1 if summary["errors"] > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
