#!/usr/bin/env python
"""Headless runner for examples/ — intended for CI.

Usage::

    python -m tools.run_examples          # run all examples
    python -m tools.run_examples --glob "01_*"  # run matching subset

Sets ``MPLBACKEND=Agg`` so no display is required.
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import subprocess
import sys
import time
from pathlib import Path

# Force headless matplotlib before any example can import it.
os.environ["MPLBACKEND"] = "Agg"

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

# Files that are not runnable scripts (modules, guides, data generators).
SKIP = {
    "__init__.py",
    "generate_samples.py",
}


def discover(glob: str = "*.py") -> list[Path]:
    """Return sorted list of example scripts matching *glob*."""
    scripts = []
    for p in sorted(EXAMPLES_DIR.glob(glob)):
        if p.name in SKIP or not p.suffix == ".py":
            continue
        scripts.append(p)
    return scripts


def run_one(script: Path, timeout: int) -> tuple[bool, str]:
    """Run a single example and return (ok, message)."""
    env = {**os.environ, "MPLBACKEND": "Agg"}
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(script.parent.parent),
        )
        if proc.returncode != 0:
            msg = proc.stderr[-500:] if proc.stderr else "(no stderr)"
            return False, f"exit {proc.returncode}: {msg}"
        return True, "ok"
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"
    except Exception as exc:
        return False, str(exc)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run examples headlessly for CI.")
    ap.add_argument("--glob", default="*.py", help="Filename glob (default: *.py)")
    ap.add_argument("--timeout", type=int, default=120, help="Per-script timeout in seconds")
    args = ap.parse_args(argv)

    scripts = discover(args.glob)
    if not scripts:
        print(f"No examples match '{args.glob}'")
        return 1

    print(f"Running {len(scripts)} example(s) …\n")
    failures: list[tuple[str, str]] = []
    t0 = time.monotonic()

    for script in scripts:
        name = script.name
        ok, msg = run_one(script, args.timeout)
        symbol = "PASS" if ok else "FAIL"
        print(f"  [{symbol}] {name}" + ("" if ok else f"  — {msg}"))
        if not ok:
            failures.append((name, msg))

    elapsed = time.monotonic() - t0
    print(f"\n{len(scripts) - len(failures)}/{len(scripts)} passed in {elapsed:.1f}s")

    if failures:
        print("\nFailed:")
        for name, msg in failures:
            print(f"  {name}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
