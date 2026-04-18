"""Confidence aggregator (PLAN.md T5.4).

Produces a scalar ``Confidence.score`` in [0, 1] from weighted
sub-scores: data quality, fit quality, margin headroom, and
robustness.  The formula is documented and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from pid_control.autotune.types import (
    Confidence,
    IdentificationResult,
    MarginReport,
    Objective,
    Status,
)


@dataclass(frozen=True)
class ConfidenceAggregator:
    """Compute a composite confidence score.

    Sub-scores (each in [0, 1]):

    * **fit**: maps R² linearly from 0.8→0 to 1.0→1.
    * **margin**: phase-margin headroom relative to objective.
    * **Ms**: sensitivity peak penalty (Ms ≤ 1.4 → 1.0, Ms ≥ 2.5 → 0).
    * **data**: 1.0 when data quality passed, 0.5 otherwise.

    Weights default to equal; callers can tune.
    """

    w_fit: float = 0.30
    w_margin: float = 0.30
    w_Ms: float = 0.25
    w_data: float = 0.15

    def compute(
        self,
        identification: Optional[IdentificationResult],
        margins: Optional[MarginReport],
        objective: Objective,
        data_ok: bool = True,
    ) -> Confidence:
        contributions: Dict[str, float] = {}
        total_weight = 0.0
        weighted_sum = 0.0

        # --- Fit quality ---
        if identification is not None:
            r2 = identification.fit_quality_r2
            fit_score = max(0.0, min(1.0, (r2 - 0.8) / 0.2))
            contributions["fit"] = fit_score * self.w_fit
            weighted_sum += fit_score * self.w_fit
            total_weight += self.w_fit

        # --- Phase margin ---
        if margins is not None and margins.phase_margin_deg is not None:
            target = objective.min_phase_margin_deg
            pm = margins.phase_margin_deg
            if pm <= 0:
                margin_score = 0.0
            elif pm >= target * 1.5:
                margin_score = 1.0
            else:
                margin_score = max(0.0, pm / (target * 1.5))
            contributions["margin"] = margin_score * self.w_margin
            weighted_sum += margin_score * self.w_margin
            total_weight += self.w_margin

        # --- Sensitivity peak ---
        if margins is not None and margins.sensitivity_peak is not None:
            Ms = margins.sensitivity_peak
            if Ms <= 1.4:
                ms_score = 1.0
            elif Ms >= 2.5:
                ms_score = 0.0
            else:
                ms_score = max(0.0, 1.0 - (Ms - 1.4) / (2.5 - 1.4))
            contributions["Ms"] = ms_score * self.w_Ms
            weighted_sum += ms_score * self.w_Ms
            total_weight += self.w_Ms

        # --- Data quality ---
        data_score = 1.0 if data_ok else 0.5
        contributions["data"] = data_score * self.w_data
        weighted_sum += data_score * self.w_data
        total_weight += self.w_data

        score = weighted_sum / total_weight if total_weight > 0 else 0.0
        return Confidence(score=float(max(0.0, min(1.0, score))),
                          contributions=contributions)


__all__ = ["ConfidenceAggregator"]
