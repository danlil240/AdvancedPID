"""Cross-cutting diagnostics (data quality, reporters, plotting)."""

from pid_control.autotune.diagnostics.data_quality import assess as assess_data_quality
from pid_control.autotune.diagnostics.reporters import (
    report_json,
    report_markdown,
    report_text,
)
from pid_control.autotune.diagnostics.plotting import plot_result

__all__ = [
    "assess_data_quality",
    "plot_result",
    "report_json",
    "report_markdown",
    "report_text",
]
