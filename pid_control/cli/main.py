"""``pidtune`` command-line interface (PLAN.md T6.3).

Subcommands
-----------
csv     Autotune from a CSV data file.
plant   Autotune from a built-in plant model.
bench   Run the benchmark harness.

Examples::

    pidtune csv examples/data/fopdt_step.csv -o result.json
    pidtune plant fopdt --K 1.0 --tau 2.0 --theta 0.5 --rule simc
    pidtune bench --suite smoke --out benchmarks/results/run.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


# ── Rule lookup ──────────────────────────────────────────────────────────
_RULES = {
    "imc": "pid_control.autotune.rules.imc.IMCRule",
    "ziegler_nichols": "pid_control.autotune.rules.ziegler_nichols.ZieglerNicholsRule",
    "cohen_coon": "pid_control.autotune.rules.cohen_coon.CohenCoonRule",
    "amigo": "pid_control.autotune.rules.amigo.AMIGORule",
    "simc": "pid_control.autotune.rules.skogestad.SIMCRule",
}


def _resolve_rule(name: str):
    """Return an instantiated TuningRule from a string key."""
    if name not in _RULES:
        raise ValueError(
            f"Unknown rule '{name}'. Choose from: {', '.join(sorted(_RULES))}"
        )
    module_path, cls_name = _RULES[name].rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)()


# ── Subcommands ──────────────────────────────────────────────────────────

def _cmd_csv(args: argparse.Namespace) -> int:
    """Autotune from a CSV file."""
    from pid_control.autotune import PIDAutotuner

    columns = None
    if args.columns:
        # Parse "time=t,input=u,output=y" style
        columns = {}
        for pair in args.columns.split(","):
            k, _, v = pair.partition("=")
            columns[k.strip()] = v.strip()

    tuner = PIDAutotuner.from_csv(args.csv_path, columns=columns)

    if args.rule:
        tuner = tuner.set_rule(_resolve_rule(args.rule))

    result = tuner.tune()

    # Output
    fmt = args.format or "text"
    print(result.report(fmt=fmt))

    if args.output:
        result.save(args.output)
        print(f"\nResult saved to {args.output}", file=sys.stderr)

    if args.plot:
        result.plot(show=True)

    return 0 if result.is_usable else 1


def _cmd_plant(args: argparse.Namespace) -> int:
    """Autotune from a built-in plant model."""
    from pid_control.autotune import PIDAutotuner
    from pid_control.plants import (
        FirstOrderPlant,
        FOPDTPlant,
        SecondOrderPlant,
    )

    plant_map = {
        "first_order": lambda a: FirstOrderPlant(
            gain=a.K, time_constant=a.tau, sample_time=a.dt,
        ),
        "fopdt": lambda a: FOPDTPlant(
            gain=a.K, time_constant=a.tau, dead_time=a.theta,
            sample_time=a.dt,
        ),
        "second_order": lambda a: SecondOrderPlant(
            gain=a.K, natural_frequency=a.wn, damping_ratio=a.zeta,
            sample_time=a.dt,
        ),
    }

    if args.plant_type not in plant_map:
        print(f"Unknown plant type '{args.plant_type}'. "
              f"Choose from: {', '.join(sorted(plant_map))}",
              file=sys.stderr)
        return 2

    plant = plant_map[args.plant_type](args)

    tuner = PIDAutotuner.from_plant(plant)

    if args.rule:
        tuner = tuner.set_rule(_resolve_rule(args.rule))

    result = tuner.tune()

    fmt = args.format or "text"
    print(result.report(fmt=fmt))

    if args.output:
        result.save(args.output)
        print(f"\nResult saved to {args.output}", file=sys.stderr)

    if args.plot:
        result.plot(show=True)

    return 0 if result.is_usable else 1


def _cmd_bench(args: argparse.Namespace) -> int:
    """Run the benchmark harness (delegates to benchmarks.run)."""
    from benchmarks.run import main as bench_main

    bench_argv = ["--suite", args.suite, "--out", args.out]
    if args.compare:
        bench_argv += ["--compare", args.compare]
    if args.max_iter:
        bench_argv += ["--max-iter", str(args.max_iter)]
    if args.seed is not None:
        bench_argv += ["--seed", str(args.seed)]

    return bench_main(bench_argv)


# ── Argument parser ──────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pidtune",
        description="Advanced PID autotuning CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── csv ───────────────────────────────────────────────────────────
    p_csv = sub.add_parser(
        "csv", help="Autotune from a CSV data file",
    )
    p_csv.add_argument("csv_path", type=str, help="Path to the CSV file")
    p_csv.add_argument(
        "-c", "--columns", type=str, default=None,
        help="Column mapping, e.g. 'time=t,input=u,output=y'",
    )
    p_csv.add_argument(
        "-r", "--rule", type=str, default=None,
        choices=sorted(_RULES), help="Tuning rule (default: imc)",
    )
    p_csv.add_argument(
        "-f", "--format", type=str, default="text",
        choices=["text", "md", "json"],
        help="Report output format (default: text)",
    )
    p_csv.add_argument("-o", "--output", type=str, default=None,
                       help="Save TuneResult JSON to this path")
    p_csv.add_argument("--plot", action="store_true",
                       help="Show diagnostic plots")
    p_csv.set_defaults(func=_cmd_csv)

    # ── plant ─────────────────────────────────────────────────────────
    p_plant = sub.add_parser(
        "plant", help="Autotune from a built-in plant model",
    )
    p_plant.add_argument(
        "plant_type", type=str,
        choices=["first_order", "fopdt", "second_order"],
        help="Plant model",
    )
    p_plant.add_argument("--K", type=float, default=1.0, help="Plant gain")
    p_plant.add_argument("--tau", type=float, default=1.0,
                         help="Time constant (first_order/fopdt)")
    p_plant.add_argument("--theta", type=float, default=0.5,
                         help="Dead time (fopdt only)")
    p_plant.add_argument("--wn", type=float, default=1.0,
                         help="Natural frequency (second_order)")
    p_plant.add_argument("--zeta", type=float, default=0.7,
                         help="Damping ratio (second_order)")
    p_plant.add_argument("--dt", type=float, default=0.01,
                         help="Sample time (default: 0.01)")
    p_plant.add_argument(
        "-r", "--rule", type=str, default=None,
        choices=sorted(_RULES), help="Tuning rule (default: imc)",
    )
    p_plant.add_argument(
        "-f", "--format", type=str, default="text",
        choices=["text", "md", "json"],
        help="Report output format (default: text)",
    )
    p_plant.add_argument("-o", "--output", type=str, default=None,
                         help="Save TuneResult JSON to this path")
    p_plant.add_argument("--plot", action="store_true",
                         help="Show diagnostic plots")
    p_plant.set_defaults(func=_cmd_plant)

    # ── bench ─────────────────────────────────────────────────────────
    p_bench = sub.add_parser(
        "bench", help="Run the benchmark harness",
    )
    p_bench.add_argument(
        "--suite", type=str, default="smoke", help="Benchmark suite name",
    )
    p_bench.add_argument(
        "--out", type=str, default="benchmarks/results/run.json",
        help="Output JSON path",
    )
    p_bench.add_argument(
        "--compare", type=str, default=None,
        help="Baseline JSON for regression check",
    )
    p_bench.add_argument("--max-iter", type=int, default=None)
    p_bench.add_argument("--seed", type=int, default=None)
    p_bench.set_defaults(func=_cmd_bench)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
