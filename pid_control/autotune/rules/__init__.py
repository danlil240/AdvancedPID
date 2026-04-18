"""Analytical tuning rules (PLAN.md §4 stage 3a)."""

from pid_control.autotune.rules.base import TuningRule
from pid_control.autotune.rules.amigo import AMIGORule
from pid_control.autotune.rules.cohen_coon import CohenCoonRule
from pid_control.autotune.rules.imc import IMCRule
from pid_control.autotune.rules.skogestad import SIMCRule
from pid_control.autotune.rules.ziegler_nichols import ZieglerNicholsRule

__all__ = [
    "TuningRule",
    "AMIGORule",
    "CohenCoonRule",
    "IMCRule",
    "SIMCRule",
    "ZieglerNicholsRule",
]
