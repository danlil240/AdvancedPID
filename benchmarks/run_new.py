"""New-API benchmark runner using PIDAutotuner (PLAN.md §9, T9).

Usage::

    python -m benchmarks.run_new --suite smoke
    python -m benchmarks.run_new --suite smoke --compare benchmarks/results/baseline_pre_refactor.json

Mirrors the shape of ``benchmarks.run`` but drives the new
:class:`~pid_control.autotune.api.PIDAutotuner` pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.plant_zoo import PlantCase, smoke_zoo  # noqa: E402
from benchmarks.scoring import TraceScore, score_trace  # noqa: E402

from pid_control.autotune.api import PIDAutotuner  # noqa: E402
from pid_control.autotune.types import ActuatorLimits, Objective, Status  # noqa: E402
from pid_control.core.pid_controller import PIDController  # noqa: E402
from pid_control.core.pid_params import PIDParams, AntiWindupMethod  # noqa: E402

os.environ.setdefault("MPLBACKEND", "Agg")


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _collect_open_loop_step(
    case: PlantCase,
    step_amplitude: float = 1.0,
    duration: float = 20.0,
) -> Dict[str, np.ndarray]:
    plant = case.factory()
    dt = plant.sample_time
    n = int(duration / dt)
    t = np.arange(n) * dt
    u = np.full(n, step_amplitude, dtype=float)
    u[: int(1.0 / dt)] = 0.0  # 1s pre-step baseline
    y = np.zeros(n, dtype=float)
    for i in range(n):
        y[i] = plant.update(float(u[i]))
    return {"time": t, "input": u, "output": y}


# ---------------------------------------------------------------------------
# Closed-loop replay
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
    status: str          # "ok" | "skip" | "error"
    tune_status: str     # pipeline status: "ok" | "warning" | "failed"
    gains: Optional[Dict[str, float]]
    identified_model: Optional[Dict[str, Any]]
    fit_quality: Optional[float]
    confidence: Optional[float]
    score: Optional[TraceScore]
    error: Optional[str]
    elapsed_seconds: float
    n_warnings: int

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.score is not None:
            d["score"] = self.score.to_dict()
        return d


def _run_one(
    case: PlantCase,
    max_iter: int,
    seed: int,
    collect_duration: float,
    replay_duration: float,
) -> CaseResult:
    start = time.perf_counter()

    # 1. Collect open-loop step data
    try:
        arr = _collect_open_loop_step(case, duration=collect_duration)
    except Exception as exc:
        return CaseResult(
            case_id=case.case_id, description=case.description,
            status="error", tune_status="failed",
            gains=None, identified_model=None, fit_quality=None,
            confidence=None, score=None, n_warnings=0,
            error=f"data-collection: {exc!r}",
            elapsed_seconds=time.perf_counter() - start,
        )

    # 2. Run the new PIDAutotuner pipeline
    try:
        result = (
            PIDAutotuner.from_arrays(
                arr["time"], arr["input"], arr["output"],
            )
            .with_actuator_limits(
                lower=case.actuator_low, upper=case.actuator_high,
            )
            .tune()
        )

        tune_status = result.status.value
        gains = {
            "kp": float(result.gains.kp),
            "ki": float(result.gains.ki),
            "kd": float(result.gains.kd),
        }
        fit_quality = (
            float(result.identification.fit_quality_r2)
            if result.identification else None
        )
        confidence = float(result.confidence.score)
        ident_model = None
        if result.identification is not None:
            m = result.identification.model
            ident_model = {
                "model_type": m.model_type.value,
                "K": float(m.K),
                "tau": float(m.tau),
                "theta": float(m.theta),
            }
        n_warnings = len(result.warnings)
    except Exception as exc:
        return CaseResult(
            case_id=case.case_id, description=case.description,
            status="error", tune_status="failed",
            gains=None, identified_model=None, fit_quality=None,
            confidence=None, score=None, n_warnings=0,
            error=f"autotune: {exc!r}\n{traceback.format_exc()}",
            elapsed_seconds=time.perf_counter() - start,
        )

    # If pipeline FAILED, skip replay
    if tune_status == "failed":
        return CaseResult(
            case_id=case.case_id, description=case.description,
            status="ok" if not case.expected_success else "error",
            tune_status=tune_status,
            gains=gains, identified_model=ident_model,
            fit_quality=fit_quality, confidence=confidence,
            score=None, n_warnings=n_warnings,
            error="pipeline returned FAILED" if case.expected_success else None,
            elapsed_seconds=time.perf_counter() - start,
        )

    # 3. Replay the tuned controller in closed loop and score it
    try:
        replay = _closed_loop_replay(case, gains, duration=replay_duration)
        score = score_trace(
            replay["time"], replay["sp"], replay["y"], replay["u"],
            actuator=(case.actuator_low, case.actuator_high),
        )
    except Exception as exc:
        return CaseResult(
            case_id=case.case_id, description=case.description,
            status="error", tune_status=tune_status,
            gains=gains, identified_model=ident_model,
            fit_quality=fit_quality, confidence=confidence,
            score=None, n_warnings=n_warnings,
            error=f"replay: {exc!r}",
            elapsed_seconds=time.perf_counter() - start,
        )

    return CaseResult(
        case_id=case.case_id, description=case.description,
        status="ok", tune_status=tune_status,
        gains=gains, identified_model=ident_model,
        fit_quality=fit_quality, confidence=confidence,
        score=score, n_warnings=n_warnings, error=None,
        elapsed_seconds=time.perf_counter() - start,
    )


# ---------------------------------------------------------------------------
# Summary / reporting
# ---------------------------------------------------------------------------

def _aggregate(results: List[CaseResult]) -> Dict[str, Any]:
    ok = [r for r in results if r.status == "ok" and r.score is not None]
    stable = [r for r in ok if r.score.stable]
    iae = np.array([r.score.iae for r in ok], dtype=float) if ok else np.empty(0)
    overshoot = np.array([r.score.overshoot_percent for r in ok], dtype=float) if ok else np.empty(0)
    settling = np.array([r.score.settling_time_2pct for r in ok], dtype=float) if ok else np.empty(0)
    confidence = np.array([r.confidence for r in results if r.confidence is not None], dtype=float)

    return {
        "total_cases": len(results),
        "ok": len(ok),
        "errors": sum(1 for r in results if r.status == "error"),
        "stable": len(stable),
        "stable_fraction": len(stable) / len(ok) if ok else 0.0,
        "median_iae": float(np.median(iae)) if iae.size else None,
        "median_overshoot_pct": float(np.median(overshoot)) if overshoot.size else None,
        "median_settling_time": float(np.median(settling)) if settling.size else None,
        "mean_confidence": float(np.mean(confidence)) if confidence.size else None,
    }


def _write_markdown(path: Path, summary: Dict[str, Any], results: List[CaseResult]) -> None:
    lines: List[str] = []
    lines.append(f"# Benchmark (new API) — {path.stem}")
    lines.append("")
    lines.append(f"- total cases: **{summary['total_cases']}**")
    lines.append(f"- ok: **{summary['ok']}**")
    lines.append(f"- errors: **{summary['errors']}**")
    lines.append(f"- stable fraction: **{summary['stable_fraction']:.2%}**")
    if summary["median_iae"] is not None:
        lines.append(f"- median IAE: **{summary['median_iae']:.3f}**")
        lines.append(f"- median overshoot: **{summary['median_overshoot_pct']:.2f}%**")
        lines.append(f"- median settling: **{summary['median_settling_time']:.3f}s**")
    if summary["mean_confidence"] is not None:
        lines.append(f"- mean confidence: **{summary['mean_confidence']:.3f}**")
    lines.append("")
    lines.append("| case | status | tune | Kp | Ki | Kd | conf | IAE | OS% | settle | stable |")
    lines.append("|------|--------|------|----|----|----|------|-----|-----|--------|--------|")
    for r in results:
        g = r.gains or {}
        if r.score is not None:
            lines.append(
                f"| `{r.case_id}` | {r.status} | {r.tune_status} | "
                f"{g.get('kp', 0):.3f} | {g.get('ki', 0):.3f} | {g.get('kd', 0):.3f} | "
                f"{r.confidence or 0:.2f} | {r.score.iae:.2f} | "
                f"{r.score.overshoot_percent:.1f} | {r.score.settling_time_2pct:.2f} | "
                f"{r.score.stable} |"
            )
        else:
            lines.append(
                f"| `{r.case_id}` | {r.status} | {r.tune_status} | "
                f"- | - | - | {r.confidence or 0:.2f} | - | - | - | - |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _compare(baseline_path: Path, current: Dict[str, Any]) -> int:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    b_sum = baseline["summary"]
    c_sum = current["summary"]
    regressions: List[str] = []

    if c_sum["errors"] > b_sum["errors"]:
        regressions.append(
            f"errors increased: {b_sum['errors']} -> {c_sum['errors']}"
        )
    if c_sum["stable_fraction"] + 1e-9 < b_sum["stable_fraction"]:
        regressions.append(
            f"stable_fraction regressed: {b_sum['stable_fraction']:.3f} -> "
            f"{c_sum['stable_fraction']:.3f}"
        )
    if (
        b_sum.get("median_iae") is not None
        and c_sum.get("median_iae") is not None
        and c_sum["median_iae"] > 1.25 * b_sum["median_iae"]
    ):
        regressions.append(
            f"median IAE worsened: {b_sum['median_iae']:.3f} -> "
            f"{c_sum['median_iae']:.3f}"
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
    parser = argparse.ArgumentParser(
        description="Benchmark the new PIDAutotuner pipeline."
    )
    parser.add_argument("--suite", choices=["smoke"], default="smoke")
    parser.add_argument(
        "--out",
        default=str(_ROOT / "benchmarks" / "results" / "run_new.json"),
    )
    parser.add_argument("--compare", default=None)
    parser.add_argument("--max-iter", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    cases = smoke_zoo() if args.suite == "smoke" else []
    if not cases:
        print(f"No cases in suite {args.suite!r}", file=sys.stderr)
        return 2

    results: List[CaseResult] = []
    t0 = time.perf_counter()
    for case in cases:
        r = _run_one(
            case,
            max_iter=args.max_iter,
            seed=args.seed,
            collect_duration=20.0,
            replay_duration=20.0,
        )
        tag = r.status.upper()
        extra = ""
        if r.score is not None:
            extra = (
                f"IAE={r.score.iae:.2f} "
                f"stable={r.score.stable} "
                f"OS={r.score.overshoot_percent:.1f}% "
                f"conf={r.confidence or 0:.2f}"
            )
        elif r.error:
            extra = f"err={r.error[:100]}"
        print(f"[{tag:5s}] {case.case_id:32s} {extra}")
        results.append(r)
    total = time.perf_counter() - t0

    summary = _aggregate(results)
    payload = {
        "suite": args.suite,
        "api": "PIDAutotuner",
        "seed": args.seed,
        "max_iter": args.max_iter,
        "elapsed_seconds": total,
        "summary": summary,
        "results": [r.to_dict() for r in results],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8",
    )
    _write_markdown(out_path.with_suffix(".md"), summary, results)
    print(
        f"\nFinished {args.suite} in {total:.1f}s -> {out_path}"
        f" ({summary['ok']}/{summary['total_cases']} ok, "
        f"{summary['stable_fraction']:.0%} stable, "
        f"conf={summary['mean_confidence'] or 0:.2f})"
    )

    if args.compare is not None:
        return _compare(Path(args.compare), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
