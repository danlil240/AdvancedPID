"""Benchmark runner for the PID autotune stack.

Usage::

    python -m benchmarks.run --suite smoke --out benchmarks/results/baseline_pre_refactor.json
    python -m benchmarks.run --suite smoke --compare benchmarks/results/baseline_pre_refactor.json

The smoke suite is designed to complete in well under a minute on a
laptop.  Results are emitted as JSON + a human-readable Markdown
summary next to it.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import traceback
import warnings
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Make sure the project root is importable when running ``python -m benchmarks.run``.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.plant_zoo import PlantCase, smoke_zoo  # noqa: E402
from benchmarks.scoring import TraceScore, score_trace  # noqa: E402

from pid_control.core.pid_controller import PIDController  # noqa: E402
from pid_control.core.pid_params import PIDParams, AntiWindupMethod  # noqa: E402
from pid_control.identification.autotune_from_data import (  # noqa: E402
    AutotuneFromData,
)

# Deterministic, matplotlib-safe for CI.
os.environ.setdefault("MPLBACKEND", "Agg")


# ---------------------------------------------------------------------------
# Data collection utilities
# ---------------------------------------------------------------------------

def _collect_open_loop_step(
    case: PlantCase,
    step_amplitude: float = 1.0,
    duration: float = 20.0,
) -> Dict[str, np.ndarray]:
    """Run a simple open-loop step on the simulated plant and return time,
    input, output arrays suitable for ``AutotuneFromData``.
    """
    plant = case.factory()
    dt = plant.sample_time
    n = int(duration / dt)
    t = np.arange(n) * dt
    u = np.full(n, step_amplitude, dtype=float)
    u[: int(1.0 / dt)] = 0.0  # 1s of pre-step baseline
    y = np.zeros(n, dtype=float)
    for i in range(n):
        y[i] = plant.update(float(u[i]))
    return {"time": t, "input": u, "output": y}


def _write_csv(path: Path, arr: Dict[str, np.ndarray], setpoint: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = arr["time"]
    u = arr["input"]
    y = arr["output"]
    sp = np.full_like(t, setpoint)
    with path.open("w") as f:
        f.write("timestamp,output,measurement,setpoint\n")
        for i in range(len(t)):
            f.write(f"{t[i]:.6f},{u[i]:.6f},{y[i]:.6f},{sp[i]:.6f}\n")


# ---------------------------------------------------------------------------
# Closed-loop replay using the tuned gains
# ---------------------------------------------------------------------------

def _closed_loop_replay(
    case: PlantCase,
    gains: Dict[str, float],
    duration: float = 20.0,
) -> Dict[str, np.ndarray]:
    plant = case.factory()
    dt = plant.sample_time
    n = int(duration / dt)

    params = PIDParams(
        kp=float(gains.get("kp", 0.0)),
        ki=float(gains.get("ki", 0.0)),
        kd=float(gains.get("kd", 0.0)),
        sample_time=dt,
        output_min=case.actuator_low,
        output_max=case.actuator_high,
        anti_windup=AntiWindupMethod.BACK_CALCULATION,
    )
    ctrl = PIDController(params)
    sp = case.setpoint

    t = np.arange(n) * dt
    y = np.zeros(n, dtype=float)
    u = np.zeros(n, dtype=float)
    sp_vec = np.full(n, sp, dtype=float)

    meas = 0.0
    for i in range(n):
        u[i] = ctrl.update(sp, meas, timestamp=float(t[i]))
        meas = plant.update(float(u[i]))
        y[i] = meas

    return {"time": t, "sp": sp_vec, "u": u, "y": y}


# ---------------------------------------------------------------------------
# Per-case runner
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    case_id: str
    description: str
    status: str                      # "ok" | "skip" | "error"
    gains: Optional[Dict[str, float]]
    identified_model: Optional[Dict[str, Any]]
    fit_quality: Optional[float]
    improvement: Optional[float]
    score: Optional[TraceScore]
    error: Optional[str]
    elapsed_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.score is not None:
            d["score"] = self.score.to_dict()
        return d


def _run_one(
    case: PlantCase,
    tmp_dir: Path,
    step_amplitude: float,
    collect_duration: float,
    replay_duration: float,
    max_iter: int,
    seed: int,
) -> CaseResult:
    start = time.perf_counter()
    csv_path = tmp_dir / f"{case.case_id}.csv"

    # 1. Generate open-loop data & write CSV
    try:
        arr = _collect_open_loop_step(
            case, step_amplitude=step_amplitude, duration=collect_duration
        )
        _write_csv(csv_path, arr, setpoint=case.setpoint)
    except Exception as exc:  # pragma: no cover - defensive
        return CaseResult(
            case_id=case.case_id,
            description=case.description,
            status="error",
            gains=None,
            identified_model=None,
            fit_quality=None,
            improvement=None,
            score=None,
            error=f"data-collection: {exc!r}",
            elapsed_seconds=time.perf_counter() - start,
        )

    # 2. Run the existing autotune path (silence its banner prints).
    buf = io.StringIO()
    try:
        with warnings.catch_warnings(), redirect_stdout(buf):
            warnings.simplefilter("ignore")
            np.random.seed(seed)
            auto = AutotuneFromData(str(csv_path))
            result = auto.autotune(max_iterations=max_iter)
        gains = {
            "kp": float(result.optimized_gains["kp"]),
            "ki": float(result.optimized_gains["ki"]),
            "kd": float(result.optimized_gains["kd"]),
        }
        model = {
            "model_type": str(result.identification.model.model_type),
            "K": float(result.identification.model.K),
            "tau": float(result.identification.model.tau),
            "theta": float(result.identification.model.theta),
        }
        fit_quality = float(result.identification.fit_quality)
        improvement = float(result.improvement)
    except Exception as exc:
        return CaseResult(
            case_id=case.case_id,
            description=case.description,
            status="error",
            gains=None,
            identified_model=None,
            fit_quality=None,
            improvement=None,
            score=None,
            error=f"autotune: {exc!r}\n{traceback.format_exc()}",
            elapsed_seconds=time.perf_counter() - start,
        )

    # 3. Replay the tuned controller in closed loop and score it.
    try:
        replay = _closed_loop_replay(case, gains, duration=replay_duration)
        score = score_trace(
            replay["time"], replay["sp"], replay["y"], replay["u"],
            actuator=(case.actuator_low, case.actuator_high),
        )
    except Exception as exc:
        return CaseResult(
            case_id=case.case_id,
            description=case.description,
            status="error",
            gains=gains,
            identified_model=model,
            fit_quality=fit_quality,
            improvement=improvement,
            score=None,
            error=f"replay: {exc!r}",
            elapsed_seconds=time.perf_counter() - start,
        )

    return CaseResult(
        case_id=case.case_id,
        description=case.description,
        status="ok",
        gains=gains,
        identified_model=model,
        fit_quality=fit_quality,
        improvement=improvement,
        score=score,
        error=None,
        elapsed_seconds=time.perf_counter() - start,
    )


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _aggregate(results: List[CaseResult]) -> Dict[str, Any]:
    ok = [r for r in results if r.status == "ok" and r.score is not None]
    stable = [r for r in ok if r.score.stable]
    iae = np.array([r.score.iae for r in ok], dtype=float) if ok else np.empty(0)
    overshoot = np.array([r.score.overshoot_percent for r in ok], dtype=float) if ok else np.empty(0)
    settling = np.array([r.score.settling_time_2pct for r in ok], dtype=float) if ok else np.empty(0)

    return {
        "total_cases": len(results),
        "ok": len(ok),
        "errors": sum(1 for r in results if r.status == "error"),
        "stable": len(stable),
        "stable_fraction": (len(stable) / len(ok)) if ok else 0.0,
        "median_iae": float(np.median(iae)) if iae.size else None,
        "median_overshoot_pct": float(np.median(overshoot)) if overshoot.size else None,
        "median_settling_time": float(np.median(settling)) if settling.size else None,
    }


def _write_markdown(path: Path, summary: Dict[str, Any], results: List[CaseResult]) -> None:
    lines: List[str] = []
    lines.append(f"# Benchmark summary — {path.stem}")
    lines.append("")
    lines.append(f"- total cases: **{summary['total_cases']}**")
    lines.append(f"- ok: **{summary['ok']}**")
    lines.append(f"- errors: **{summary['errors']}**")
    lines.append(f"- stable fraction: **{summary['stable_fraction']:.2%}**")
    if summary["median_iae"] is not None:
        lines.append(f"- median IAE: **{summary['median_iae']:.3f}**")
        lines.append(f"- median overshoot: **{summary['median_overshoot_pct']:.2f} %**")
        lines.append(f"- median settling time: **{summary['median_settling_time']:.3f} s**")
    lines.append("")
    lines.append("| case | status | Kp | Ki | Kd | IAE | overshoot% | settling_s | stable |")
    lines.append("|------|--------|----|----|----|-----|-----------|------------|--------|")
    for r in results:
        if r.score is None or r.gains is None:
            lines.append(
                f"| `{r.case_id}` | {r.status} | - | - | - | - | - | - | - |"
            )
            continue
        lines.append(
            f"| `{r.case_id}` | {r.status} | "
            f"{r.gains['kp']:.3f} | {r.gains['ki']:.3f} | {r.gains['kd']:.3f} | "
            f"{r.score.iae:.2f} | {r.score.overshoot_percent:.1f} | "
            f"{r.score.settling_time_2pct:.2f} | {r.score.stable} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _compare(baseline_path: Path, current: Dict[str, Any]) -> int:
    """Compare ``current`` against a previously saved baseline JSON.

    Returns an exit code: 0 = no regression, 1 = regression detected.
    """
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    b_sum = baseline["summary"]
    c_sum = current["summary"]

    regressions: List[str] = []

    if c_sum["errors"] > b_sum["errors"]:
        regressions.append(
            f"errors increased: {b_sum['errors']} → {c_sum['errors']}"
        )
    if c_sum["stable_fraction"] + 1e-9 < b_sum["stable_fraction"]:
        regressions.append(
            f"stable_fraction regressed: {b_sum['stable_fraction']:.3f} → {c_sum['stable_fraction']:.3f}"
        )
    if (
        b_sum["median_iae"] is not None
        and c_sum["median_iae"] is not None
        and c_sum["median_iae"] > 1.25 * b_sum["median_iae"]
    ):
        regressions.append(
            f"median IAE worsened: {b_sum['median_iae']:.3f} → {c_sum['median_iae']:.3f}"
        )

    if regressions:
        print("REGRESSIONS vs baseline:")
        for r in regressions:
            print(f"  - {r}")
        return 1
    print("No regressions detected.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the PID autotune benchmark.")
    parser.add_argument("--suite", choices=["smoke"], default="smoke")
    parser.add_argument(
        "--out",
        default=str(_ROOT / "benchmarks" / "results" / "run.json"),
        help="Where to write the JSON results file.",
    )
    parser.add_argument(
        "--compare",
        default=None,
        help="Path to a baseline JSON to compare against. Exit 1 on regression.",
    )
    parser.add_argument("--max-iter", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    cases = smoke_zoo() if args.suite == "smoke" else []
    if not cases:
        print(f"No cases in suite {args.suite!r}", file=sys.stderr)
        return 2

    tmp_dir = _ROOT / "output" / "_bench_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    results: List[CaseResult] = []
    t0 = time.perf_counter()
    for case in cases:
        res = _run_one(
            case,
            tmp_dir=tmp_dir,
            step_amplitude=1.0,
            collect_duration=20.0,
            replay_duration=20.0,
            max_iter=args.max_iter,
            seed=args.seed,
        )
        tag = res.status.upper()
        extra = ""
        if res.score is not None:
            extra = (
                f"IAE={res.score.iae:.2f} "
                f"oscillating={not res.score.stable} "
                f"overshoot={res.score.overshoot_percent:.1f}%"
            )
        elif res.error:
            extra = f"error={res.error[:120]}"
        print(f"[{tag:5s}] {case.case_id:32s} {extra}")
        results.append(res)
    total = time.perf_counter() - t0

    summary = _aggregate(results)
    payload = {
        "suite": args.suite,
        "seed": args.seed,
        "max_iter": args.max_iter,
        "elapsed_seconds": total,
        "summary": summary,
        "results": [r.to_dict() for r in results],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    _write_markdown(out_path.with_suffix(".md"), summary, results)
    print(
        f"\nFinished {args.suite} suite in {total:.1f}s → {out_path}"
        f" ({summary['ok']}/{summary['total_cases']} ok, "
        f"{summary['stable_fraction']:.0%} stable)"
    )

    if args.compare is not None:
        return _compare(Path(args.compare), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
