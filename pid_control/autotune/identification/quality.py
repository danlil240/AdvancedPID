"""Fit-quality metrics (R², RMSE, AIC, BIC) used by the identifiers and
model-selection stage (PLAN.md T2.4).

These helpers are dependency-free beyond numpy.  All functions operate
on 1-D arrays of equal length; negative / zero-variance inputs are
handled by returning ``nan``-marked fields where appropriate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class FitQuality:
    """Aggregated fit metrics for an identified model."""

    r2: float
    rmse: float
    residual_variance: float
    aic: Optional[float] = None
    bic: Optional[float] = None
    n_params: int = 0
    n_samples: int = 0


def compute_fit_quality(
    y_actual: np.ndarray,
    y_model: np.ndarray,
    n_params: int,
) -> FitQuality:
    """Score how well ``y_model`` reproduces ``y_actual``.

    ``n_params`` is the *effective* number of free parameters (e.g. 3
    for FOPDT, 4 for SOPDT with dead time); it is used for the
    information-criterion penalties.  When :math:`n \\le p`, AIC / BIC
    are not computed.
    """
    y_actual = np.asarray(y_actual, dtype=float)
    y_model = np.asarray(y_model, dtype=float)
    if y_actual.shape != y_model.shape:
        raise ValueError("y_actual and y_model must have identical shape")

    residuals = y_actual - y_model
    n = int(residuals.size)
    rss = float(np.sum(residuals * residuals))
    var_y = float(np.var(y_actual))

    if n == 0 or var_y == 0.0:
        r2 = 0.0
        rmse = float("nan")
        res_var = float("nan")
    else:
        ss_tot = float(np.sum((y_actual - np.mean(y_actual)) ** 2))
        r2 = 1.0 - (rss / ss_tot) if ss_tot > 0 else 0.0
        rmse = math.sqrt(rss / n)
        res_var = rss / max(n - n_params, 1)

    aic: Optional[float] = None
    bic: Optional[float] = None
    if n > n_params + 1 and rss > 0:
        # Gaussian log-likelihood with estimated variance.
        sigma2 = rss / n
        log_like = -0.5 * n * (math.log(2 * math.pi * sigma2) + 1.0)
        k = n_params + 1  # + 1 for the noise variance
        aic = 2 * k - 2 * log_like
        bic = k * math.log(n) - 2 * log_like

    return FitQuality(
        r2=float(r2),
        rmse=float(rmse) if np.isfinite(rmse) else float("nan"),
        residual_variance=float(res_var) if np.isfinite(res_var) else float("nan"),
        aic=aic,
        bic=bic,
        n_params=n_params,
        n_samples=n,
    )


__all__ = ["FitQuality", "compute_fit_quality"]
