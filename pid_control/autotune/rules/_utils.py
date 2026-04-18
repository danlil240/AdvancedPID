"""Shared helpers for tuning rules.

Every classical rule is formulated on a FOPDT model ``(K, τ, θ)``.
When the identifier returned a higher-order model we project it onto
FOPDT using Skogestad's half-rule:

    * A SOPDT ``(K, τ₁, τ₂, θ)`` becomes
      ``K_eq = K, τ_eq = τ₁ + τ₂/2, θ_eq = θ + τ₂/2``.
    * Anything else falls back to ``(K, τ, θ)`` as-is.

This keeps the classical rule library scoped narrowly while still
accepting any :class:`IdentificationResult` the pipeline produces.
"""

from __future__ import annotations

from typing import Tuple

from pid_control.autotune.types import ModelType, TransferFunctionModel


def to_fopdt_triplet(model: TransferFunctionModel) -> Tuple[float, float, float]:
    """Project ``model`` onto an equivalent FOPDT ``(K, τ, θ)``.

    The caller is responsible for refusing degenerate cases (K≈0) — the
    rules clamp the tiny divisors but that's numerical hygiene, not
    physical meaning.
    """
    K = float(model.K)
    tau = max(float(model.tau), 1e-9)
    theta = max(float(model.theta), 0.0)

    if model.model_type is ModelType.SOPDT and model.tau2 is not None:
        tau2 = float(model.tau2)
        tau, tau_small = (tau, tau2) if tau >= tau2 else (tau2, tau)
        # Skogestad half-rule
        tau_eq = tau + tau_small / 2.0
        theta_eq = theta + tau_small / 2.0
        return K, max(tau_eq, 1e-9), max(theta_eq, 0.0)

    return K, tau, theta


__all__ = ["to_fopdt_triplet"]
