"""Report formatting for TuneResult (PLAN.md T6.1).

Emits text, Markdown, or JSON summaries of a :class:`TuneResult`.
Deterministic — no timestamps unless the caller's ``TuneMeta`` supplies them.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pid_control.autotune.types import TuneResult


def report_text(result: "TuneResult") -> str:
    """Plain-text report suitable for terminal output."""
    lines: list[str] = []
    _header(lines, "PID Autotune Result")
    lines.append(f"Status      : {result.status.value.upper()}")
    lines.append(f"Confidence  : {result.confidence.score:.2f}")
    lines.append("")

    # Gains
    g = result.gains
    lines.append("Tuned Gains:")
    lines.append(f"  Kp = {g.kp:.6g}")
    lines.append(f"  Ki = {g.ki:.6g}")
    lines.append(f"  Kd = {g.kd:.6g}")
    if g.derivative_filter_n != 10.0:
        lines.append(f"  N  = {g.derivative_filter_n:.4g}")
    if g.setpoint_weight_b != 1.0:
        lines.append(f"  b  = {g.setpoint_weight_b:.4g}")
    if g.setpoint_weight_c != 0.0:
        lines.append(f"  c  = {g.setpoint_weight_c:.4g}")
    lines.append("")

    # Identification
    if result.identification is not None:
        ident = result.identification
        lines.append(f"Identified model: {ident.model}")
        lines.append(f"  Fit R²  = {ident.fit_quality_r2:.4f}")
        if ident.aic is not None:
            lines.append(f"  AIC     = {ident.aic:.2f}")
        if ident.bic is not None:
            lines.append(f"  BIC     = {ident.bic:.2f}")
        lines.append("")

    # Margins
    if result.performance is not None and result.performance.margins is not None:
        m = result.performance.margins
        lines.append("Stability margins:")
        if m.gain_margin_db is not None:
            lines.append(f"  Gain margin  = {m.gain_margin_db:.1f} dB")
        if m.phase_margin_deg is not None:
            lines.append(f"  Phase margin = {m.phase_margin_deg:.1f}°")
        if m.sensitivity_peak is not None:
            lines.append(f"  Ms           = {m.sensitivity_peak:.3f}")
        if m.complementary_sensitivity_peak is not None:
            lines.append(f"  Mt           = {m.complementary_sensitivity_peak:.3f}")
        if m.delay_margin_s is not None:
            lines.append(f"  Delay margin = {m.delay_margin_s:.4g} s")
        lines.append("")

    # Performance
    if result.performance is not None:
        p = result.performance
        lines.append("Performance (step to 1.0):")
        lines.append(f"  IAE        = {p.iae:.4g}")
        lines.append(f"  Overshoot  = {p.overshoot_percent:.2f}%")
        if p.rise_time is not None:
            lines.append(f"  Rise time  = {p.rise_time:.4g} s")
        if p.settling_time_2pct is not None:
            lines.append(f"  Settling   = {p.settling_time_2pct:.4g} s (2%)")
        lines.append(f"  SS error   = {p.steady_state_error:.4g}")
        lines.append(f"  Ctrl TV    = {p.control_total_variation:.4g}")
        lines.append("")

    # Confidence breakdown
    if result.confidence.contributions:
        lines.append("Confidence breakdown:")
        for name, val in result.confidence.contributions.items():
            lines.append(f"  {name:20s} = {val:.3f}")
        lines.append("")

    # Warnings
    if result.warnings:
        lines.append(f"Warnings ({len(result.warnings)}):")
        for w in result.warnings:
            lines.append(f"  [{w.severity.value:7s}] {w.code.value}: {w.message}")
        lines.append("")

    # Meta
    if result.meta is not None:
        lines.append(f"Elapsed: {result.meta.elapsed_seconds:.2f}s  "
                      f"Seed: {result.meta.seed}  "
                      f"Cost evals: {result.meta.cost_evaluations}")

    return "\n".join(lines)


def report_markdown(result: "TuneResult") -> str:
    """Markdown report suitable for documentation or CI artifacts."""
    lines: list[str] = []
    lines.append("# PID Autotune Result")
    lines.append("")
    lines.append(f"**Status**: `{result.status.value.upper()}`  ")
    lines.append(f"**Confidence**: {result.confidence.score:.2f}")
    lines.append("")

    # Gains table
    g = result.gains
    lines.append("## Tuned Gains")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| Kp | {g.kp:.6g} |")
    lines.append(f"| Ki | {g.ki:.6g} |")
    lines.append(f"| Kd | {g.kd:.6g} |")
    if g.derivative_filter_n != 10.0:
        lines.append(f"| N | {g.derivative_filter_n:.4g} |")
    lines.append("")

    # Identification
    if result.identification is not None:
        ident = result.identification
        lines.append("## Identified Model")
        lines.append("")
        lines.append(f"`{ident.model}`")
        lines.append("")
        lines.append(f"- R² = {ident.fit_quality_r2:.4f}")
        if ident.aic is not None:
            lines.append(f"- AIC = {ident.aic:.2f}")
        if ident.bic is not None:
            lines.append(f"- BIC = {ident.bic:.2f}")
        lines.append("")

    # Margins
    if result.performance is not None and result.performance.margins is not None:
        m = result.performance.margins
        lines.append("## Stability Margins")
        lines.append("")
        lines.append("| Margin | Value |")
        lines.append("|--------|-------|")
        if m.gain_margin_db is not None:
            lines.append(f"| Gain margin | {m.gain_margin_db:.1f} dB |")
        if m.phase_margin_deg is not None:
            lines.append(f"| Phase margin | {m.phase_margin_deg:.1f}° |")
        if m.sensitivity_peak is not None:
            lines.append(f"| Ms | {m.sensitivity_peak:.3f} |")
        if m.complementary_sensitivity_peak is not None:
            lines.append(f"| Mt | {m.complementary_sensitivity_peak:.3f} |")
        lines.append("")

    # Performance
    if result.performance is not None:
        p = result.performance
        lines.append("## Performance")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| IAE | {p.iae:.4g} |")
        lines.append(f"| Overshoot | {p.overshoot_percent:.2f}% |")
        if p.rise_time is not None:
            lines.append(f"| Rise time | {p.rise_time:.4g} s |")
        if p.settling_time_2pct is not None:
            lines.append(f"| Settling (2%) | {p.settling_time_2pct:.4g} s |")
        lines.append(f"| SS error | {p.steady_state_error:.4g} |")
        lines.append("")

    # Warnings
    if result.warnings:
        lines.append("## Warnings")
        lines.append("")
        for w in result.warnings:
            lines.append(f"- **{w.code.value}** ({w.severity.value}): {w.message}")
        lines.append("")

    return "\n".join(lines)


def report_json(result: "TuneResult") -> str:
    """JSON report from ``to_dict()``."""
    return json.dumps(result.to_dict(), indent=2, default=float)


def _header(lines: list[str], text: str) -> None:
    lines.append("=" * 60)
    lines.append(f"  {text}")
    lines.append("=" * 60)
    lines.append("")
