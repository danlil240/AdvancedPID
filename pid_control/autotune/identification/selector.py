"""Information-criterion model selection (PLAN.md T2.4).

Given several candidate identifications of the same dataset, prefer the
one with the *lowest* AIC (falling back to BIC, then R²).  Models that
carry a :class:`~pid_control.autotune.types.WarningCode.W_DEGENERATE_SOPDT`
warning are demoted so an over-parameterised SOPDT cannot beat a clean
FOPDT by a hair's breadth.

Usage::

    from pid_control.autotune.identification import AutoIdentifier
    result = AutoIdentifier().identify(record)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from pid_control.autotune.experiments.base import ExperimentRecord
from pid_control.autotune.identification.base import Identifier
from pid_control.autotune.identification.fopdt import FOPDTIdentifier
from pid_control.autotune.identification.sopdt import SOPDTIdentifier
from pid_control.autotune.types import (
    IdentificationResult,
    Severity,
    Warning,
    WarningCode,
    merge_warnings,
)

# Penalty added to AIC for degenerate / over-parameterised models.  Large
# enough to dwarf the natural AIC improvement an SOPDT gets over a FOPDT
# on real step data when the second pole is meaningless.
_DEGENERATE_PENALTY = 1e3


@dataclass
class AutoIdentifier:
    """Runs every identifier and keeps the one with the best AIC.

    The default candidate list is ``[FOPDTIdentifier(), SOPDTIdentifier()]``
    which covers the overwhelming majority of industrial plants.  Users
    who want to add integrator / higher-order models can pass their own
    list through ``candidates``.
    """

    candidates: Sequence[Identifier] = field(
        default_factory=lambda: (FOPDTIdentifier(), SOPDTIdentifier())
    )
    name: str = "auto_ic"

    def identify(self, record: ExperimentRecord) -> IdentificationResult:
        results: List[IdentificationResult] = []
        for ident in self.candidates:
            try:
                res = ident.identify(record)
            except Exception as exc:  # pragma: no cover - defensive
                results.append(
                    _error_result(ident, exc)
                )
                continue
            results.append(res)

        best = _pick_best(results)
        # Preserve diagnostics from every candidate so explain-ability is
        # preserved in reports.  Rejected candidates surface as INFO-level
        # warnings.
        all_warnings = list(best.warnings)
        for r in results:
            if r is best:
                continue
            label = r.model.model_type.value
            all_warnings.append(
                Warning(
                    code=WarningCode.W_POOR_FIT,
                    severity=Severity.INFO,
                    message=(
                        f"Rejected {label} fit: R²={r.fit_quality_r2:.3f}"
                        + (f", AIC={r.aic:.1f}" if r.aic is not None else "")
                    ),
                    stage="identify.auto",
                    context={
                        "model": label,
                        "r2": r.fit_quality_r2,
                        "aic": r.aic,
                        "bic": r.bic,
                    },
                )
            )
        return IdentificationResult(
            model=best.model,
            fit_quality_r2=best.fit_quality_r2,
            aic=best.aic,
            bic=best.bic,
            residual_rmse=best.residual_rmse,
            noise_variance=best.noise_variance,
            data_quality=best.data_quality,
            warnings=tuple(all_warnings),
        )


def _pick_best(results: Sequence[IdentificationResult]) -> IdentificationResult:
    if not results:
        raise RuntimeError("No identification candidates ran")

    # Score: adjusted AIC.  Penalise degenerate models so they do not
    # silently win by a fractional margin.
    def _score(r: IdentificationResult) -> float:
        base = r.aic if r.aic is not None else -r.fit_quality_r2
        penalty = 0.0
        if any(w.code is WarningCode.W_DEGENERATE_SOPDT for w in r.warnings):
            penalty += _DEGENERATE_PENALTY
        if any(w.code is WarningCode.W_POOR_FIT and w.severity is Severity.ERROR
               for w in r.warnings):
            penalty += _DEGENERATE_PENALTY
        return base + penalty

    return min(results, key=_score)


def _error_result(ident: Identifier, exc: Exception) -> IdentificationResult:
    from pid_control.autotune.types import (
        ModelType,
        TransferFunctionModel,
    )
    model = TransferFunctionModel(
        model_type=ModelType.UNKNOWN, K=0.0, tau=1.0, theta=0.0,
    )
    return IdentificationResult(
        model=model,
        fit_quality_r2=0.0,
        warnings=(
            Warning(
                code=WarningCode.W_POOR_FIT,
                severity=Severity.WARNING,
                message=f"{ident.name} raised {exc!r}",
                stage="identify.auto",
            ),
        ),
    )


__all__ = ["AutoIdentifier"]
