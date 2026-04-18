"""Typed result objects for the autotune pipeline (PLAN.md §5, task T1.1).

Everything the autotune surface returns is defined here as a frozen
dataclass.  The pipeline (experiment → identify → tune → validate)
composes these types without mutating them — callers can branch safely
on the ``status`` and ``warnings`` fields and persist results as JSON.

No algorithmic logic lives in this module: the point is to freeze a
stable shape that parallel workstreams can code against.  Keep it
lightweight (no dependencies beyond the standard library + numpy).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Status(str, Enum):
    """Final status of a :class:`TuneResult`.

    * ``OK`` — the pipeline converged and all validators passed.
    * ``WARNING`` — a result was produced but at least one validator raised
      a non-fatal concern (e.g. poor margin, saturation).  ``gains`` are
      still usable but the caller should inspect ``warnings``.
    * ``FAILED`` — no trustworthy controller could be produced.  Callers
      MUST NOT ``build_controller()`` from a ``FAILED`` result; doing so
      will raise.
    """

    OK = "ok"
    WARNING = "warning"
    FAILED = "failed"


class Severity(str, Enum):
    """Severity level of a :class:`Warning`.

    ``INFO``     — informational, no action needed.
    ``WARNING``  — result is still usable but caller should look.
    ``ERROR``    — pipeline must short-circuit to :attr:`Status.FAILED`.
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ModelType(str, Enum):
    """Identified plant model family."""

    FOPDT = "fopdt"
    SOPDT = "sopdt"
    SECOND_ORDER = "second_order"
    IPDT = "ipdt"            # integrator + dead time
    UNKNOWN = "unknown"


class WarningCode(str, Enum):
    """Stable warning codes users can branch on (see PLAN.md §5.5)."""

    # Experiment / data quality
    E_DATA_FLAT = "E_DATA_FLAT"
    E_NO_STEADY_STATE = "E_NO_STEADY_STATE"
    E_TOO_SHORT = "E_TOO_SHORT"
    E_UNSTABLE = "E_UNSTABLE"

    # Identification
    W_POOR_FIT = "W_POOR_FIT"
    W_DEGENERATE_SOPDT = "W_DEGENERATE_SOPDT"
    W_NONMIN_PHASE = "W_NONMIN_PHASE"
    W_INTEGRATING = "W_INTEGRATING"
    W_HIGH_NOISE = "W_HIGH_NOISE"

    # Tuning
    W_GAIN_CLIPPED = "W_GAIN_CLIPPED"
    W_MAXITER = "W_MAXITER"
    W_COST_NOT_IMPROVED = "W_COST_NOT_IMPROVED"

    # Validation
    W_LOW_MARGIN = "W_LOW_MARGIN"
    W_FRAGILE = "W_FRAGILE"
    W_SATURATION = "W_SATURATION"


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Warning:
    """A single typed warning produced by the pipeline.

    Warnings are accumulated by every stage and surfaced on the final
    :class:`TuneResult`.  ``code`` is stable across releases so callers can
    write deterministic handlers.
    """

    code: WarningCode
    severity: Severity
    message: str
    stage: str = ""
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code.value,
            "severity": self.severity.value,
            "message": self.message,
            "stage": self.stage,
            "context": dict(self.context),
        }


# ---------------------------------------------------------------------------
# Controller-facing types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PIDGains:
    """Minimal controller-facing gain set.

    The full :class:`~pid_control.core.pid_params.PIDParams` carries many
    more fields (anti-windup, filter ``N``, derivative mode, …).  Keep this
    type small and let :meth:`TuneResult.build_controller` translate.
    """

    kp: float
    ki: float = 0.0
    kd: float = 0.0
    setpoint_weight_b: float = 1.0  # 2-DOF PID b-weight on P-term (0..1)
    setpoint_weight_c: float = 0.0  # 2-DOF c-weight on D-term (0..1)
    derivative_filter_n: float = 10.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "kp": float(self.kp),
            "ki": float(self.ki),
            "kd": float(self.kd),
            "b": float(self.setpoint_weight_b),
            "c": float(self.setpoint_weight_c),
            "N": float(self.derivative_filter_n),
        }


@dataclass(frozen=True)
class ActuatorLimits:
    """Actuator envelope used by cost functions and saturation logic."""

    lower: float = -float("inf")
    upper: float = float("inf")
    rate_limit: Optional[float] = None  # max |du/dt|; None disables

    def __post_init__(self) -> None:
        if self.upper <= self.lower:
            raise ValueError(
                f"ActuatorLimits: upper ({self.upper}) must exceed "
                f"lower ({self.lower})"
            )
        if self.rate_limit is not None and self.rate_limit <= 0:
            raise ValueError("ActuatorLimits.rate_limit must be positive")

    @property
    def is_bounded(self) -> bool:
        return np.isfinite(self.lower) and np.isfinite(self.upper)


@dataclass(frozen=True)
class Objective:
    """What the user cares about.

    Fields without an explicit weight behave as soft targets; a violation
    incurs a penalty in the cost and is *not* a fatal error.  The
    validation layer (T5.x) is responsible for converting egregious
    violations into :class:`WarningCode` values.
    """

    # Time-domain targets (soft)
    max_overshoot_pct: float = 15.0
    max_settling_time: Optional[float] = None  # seconds; None = scaled by plant
    max_rise_time: Optional[float] = None

    # Robustness targets (typically harder constraints)
    min_phase_margin_deg: float = 30.0
    max_Ms: float = 2.0
    max_Mt: float = 1.8

    # Effort
    control_effort_weight: float = 0.01

    # Cost composition (classical integral terms)
    iae_weight: float = 1.0
    itae_weight: float = 0.0
    ise_weight: float = 0.0


# ---------------------------------------------------------------------------
# Experiment / identification / performance
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataQuality:
    """Diagnostics computed by the data-quality stage (T2.5).

    All fields are informational.  ``status == Status.FAILED`` blocks the
    pipeline before identification.
    """

    status: Status
    snr_db: Optional[float]
    excitation_energy: float
    has_steady_state: bool
    n_samples: int
    sample_time: float
    warnings: Tuple[Warning, ...] = ()


@dataclass(frozen=True)
class TransferFunctionModel:
    """Continuous-time transfer-function parameters (FOPDT / SOPDT / IPDT)."""

    model_type: ModelType
    K: float
    tau: float
    theta: float = 0.0
    tau2: Optional[float] = None
    # Derived fields useful downstream (filled by identifiers when relevant).
    natural_frequency: Optional[float] = None
    damping_ratio: Optional[float] = None

    def __str__(self) -> str:  # pragma: no cover - human diagnostics only
        if self.model_type is ModelType.SOPDT and self.tau2 is not None:
            return (
                f"{self.model_type.value}: K={self.K:.4g}, "
                f"tau1={self.tau:.4g}, tau2={self.tau2:.4g}, theta={self.theta:.4g}"
            )
        return (
            f"{self.model_type.value}: K={self.K:.4g}, "
            f"tau={self.tau:.4g}, theta={self.theta:.4g}"
        )


@dataclass(frozen=True)
class IdentificationResult:
    """Output of the identification stage."""

    model: TransferFunctionModel
    fit_quality_r2: float
    aic: Optional[float] = None
    bic: Optional[float] = None
    residual_rmse: Optional[float] = None
    noise_variance: Optional[float] = None
    data_quality: Optional[DataQuality] = None
    warnings: Tuple[Warning, ...] = ()


@dataclass(frozen=True)
class MarginReport:
    """Loop-stability margins computed on the identified model + gains."""

    gain_margin_db: Optional[float] = None
    phase_margin_deg: Optional[float] = None
    delay_margin_s: Optional[float] = None
    sensitivity_peak: Optional[float] = None            # Ms
    complementary_sensitivity_peak: Optional[float] = None  # Mt


@dataclass(frozen=True)
class PerformanceReport:
    """Time + frequency performance estimates from the simulated loop."""

    iae: float
    ise: float
    itae: float
    rise_time: Optional[float]
    settling_time_2pct: Optional[float]
    overshoot_percent: float
    steady_state_error: float
    control_total_variation: float
    control_rms: float
    control_peak: float
    saturation_fraction: float
    margins: MarginReport = field(default_factory=MarginReport)


@dataclass(frozen=True)
class Trajectory:
    """A single simulated or recorded time-series used by reports/plots."""

    time: np.ndarray
    setpoint: np.ndarray
    measurement: np.ndarray
    control: np.ndarray

    def __post_init__(self) -> None:
        lens = {len(self.time), len(self.setpoint),
                len(self.measurement), len(self.control)}
        if len(lens) != 1:
            raise ValueError(
                f"Trajectory arrays must share length, got {lens}"
            )


@dataclass(frozen=True)
class Artifacts:
    """Bundles numeric artefacts for plotting/persistence.

    Heavy arrays live here rather than on :class:`TuneResult` so callers
    can discard them cheaply after reporting.
    """

    closed_loop_trajectory: Optional[Trajectory] = None
    identification_trajectory: Optional[Trajectory] = None
    residuals: Optional[np.ndarray] = None
    cost_history: Optional[np.ndarray] = None


@dataclass(frozen=True)
class TuneMeta:
    """Bookkeeping: versions, seeds, timing.  Useful for reproducibility."""

    library_version: str
    elapsed_seconds: float
    seed: Optional[int] = None
    experiment: str = ""
    identifier: str = ""
    tuner: str = ""
    cost_evaluations: int = 0


@dataclass(frozen=True)
class Confidence:
    """Aggregate trust score in the final tuning.

    ``score`` is in [0, 1]; see :mod:`pid_control.autotune.validation.confidence`
    for the formula.  ``contributions`` maps each sub-score to its weighted
    contribution so reports can explain *why* the score is what it is.
    """

    score: float
    contributions: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# TuneResult – the top-level artifact
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TuneResult:
    """The single object a user receives from :meth:`PIDAutotuner.tune`."""

    gains: PIDGains
    status: Status
    confidence: Confidence
    identification: Optional[IdentificationResult]
    performance: Optional[PerformanceReport]
    warnings: Tuple[Warning, ...] = ()
    artifacts: Artifacts = field(default_factory=Artifacts)
    meta: Optional[TuneMeta] = None

    # --- Convenience helpers -------------------------------------------

    @property
    def is_usable(self) -> bool:
        """Callers may ``build_controller()`` iff this is True."""
        return self.status is not Status.FAILED

    def warnings_for(self, severity: Severity) -> Tuple[Warning, ...]:
        return tuple(w for w in self.warnings if w.severity is severity)

    def has_warning(self, code: WarningCode) -> bool:
        return any(w.code is code for w in self.warnings)

    # --- Serialization -------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "status": self.status.value,
            "gains": self.gains.as_dict(),
            "confidence": {
                "score": float(self.confidence.score),
                "contributions": dict(self.confidence.contributions),
            },
            "warnings": [w.to_dict() for w in self.warnings],
        }
        if self.identification is not None:
            payload["identification"] = {
                "model": {
                    "type": self.identification.model.model_type.value,
                    "K": float(self.identification.model.K),
                    "tau": float(self.identification.model.tau),
                    "theta": float(self.identification.model.theta),
                    "tau2": (
                        float(self.identification.model.tau2)
                        if self.identification.model.tau2 is not None else None
                    ),
                },
                "fit_quality_r2": float(self.identification.fit_quality_r2),
                "aic": self.identification.aic,
                "bic": self.identification.bic,
                "residual_rmse": self.identification.residual_rmse,
                "noise_variance": self.identification.noise_variance,
                "warnings": [w.to_dict() for w in self.identification.warnings],
            }
        if self.performance is not None:
            payload["performance"] = {
                k: getattr(self.performance, k)
                for k in (
                    "iae", "ise", "itae", "rise_time", "settling_time_2pct",
                    "overshoot_percent", "steady_state_error",
                    "control_total_variation", "control_rms",
                    "control_peak", "saturation_fraction",
                )
            }
            payload["performance"]["margins"] = asdict(self.performance.margins)
        if self.meta is not None:
            payload["meta"] = asdict(self.meta)
        return payload

    def report(self, fmt: str = "text") -> str:
        """Human-readable summary in *text*, *md* (Markdown), or *json*.

        >>> result.report()         # plain text for the terminal
        >>> result.report("md")     # Markdown for docs / CI artifacts
        >>> result.report("json")   # JSON for machine consumption
        """
        from pid_control.autotune.diagnostics.reporters import (
            report_json,
            report_markdown,
            report_text,
        )
        if fmt in ("md", "markdown"):
            return report_markdown(self)
        if fmt == "json":
            return report_json(self)
        return report_text(self)

    def plot(
        self,
        kind: str = "all",
        save_path: Optional[str] = None,
        show: bool = False,
    ):
        """Generate diagnostic plots.

        Parameters
        ----------
        kind : ``"fit"``, ``"response"``, ``"margins"``, or ``"all"``
        save_path : if given, figures are saved (PNG) instead of displayed.
        show : call ``plt.show()`` — **default False** so headless runs
               are safe.
        """
        from pid_control.autotune.diagnostics.plotting import plot_result
        return plot_result(self, kind=kind, save_path=save_path, show=show)

    def save(self, path: str | Path) -> None:
        """Persist as JSON (arrays in ``artifacts`` are not serialized)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, default=float),
                     encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TuneResult":
        """Reconstruct a :class:`TuneResult` from a JSON file saved by
        :meth:`save`.

        Arrays in ``artifacts`` are NOT restored (they are not serialized).
        """
        p = Path(path)
        payload = json.loads(p.read_text(encoding="utf-8"))
        return cls._from_dict(payload)

    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> "TuneResult":
        """Reconstruct from the dict produced by :meth:`to_dict`."""
        gains_d = d["gains"]
        gains = PIDGains(
            kp=gains_d["kp"],
            ki=gains_d["ki"],
            kd=gains_d["kd"],
            setpoint_weight_b=gains_d.get("b", 1.0),
            setpoint_weight_c=gains_d.get("c", 0.0),
            derivative_filter_n=gains_d.get("N", 10.0),
        )
        status = Status(d["status"])
        conf_d = d.get("confidence", {})
        confidence = Confidence(
            score=conf_d.get("score", 0.0),
            contributions=conf_d.get("contributions", {}),
        )
        warnings_list = tuple(
            Warning(
                code=WarningCode(w["code"]),
                severity=Severity(w["severity"]),
                message=w["message"],
                stage=w.get("stage", ""),
                context=w.get("context", {}),
            )
            for w in d.get("warnings", [])
        )

        identification: Optional[IdentificationResult] = None
        if "identification" in d and d["identification"] is not None:
            id_d = d["identification"]
            model_d = id_d["model"]
            identification = IdentificationResult(
                model=TransferFunctionModel(
                    model_type=ModelType(model_d["type"]),
                    K=model_d["K"],
                    tau=model_d["tau"],
                    theta=model_d.get("theta", 0.0),
                    tau2=model_d.get("tau2"),
                ),
                fit_quality_r2=id_d["fit_quality_r2"],
                aic=id_d.get("aic"),
                bic=id_d.get("bic"),
                residual_rmse=id_d.get("residual_rmse"),
                noise_variance=id_d.get("noise_variance"),
                warnings=tuple(
                    Warning(
                        code=WarningCode(w["code"]),
                        severity=Severity(w["severity"]),
                        message=w["message"],
                        stage=w.get("stage", ""),
                        context=w.get("context", {}),
                    )
                    for w in id_d.get("warnings", [])
                ),
            )

        performance: Optional[PerformanceReport] = None
        if "performance" in d and d["performance"] is not None:
            p_d = d["performance"]
            margins_d = p_d.get("margins", {})
            performance = PerformanceReport(
                iae=p_d["iae"],
                ise=p_d["ise"],
                itae=p_d["itae"],
                rise_time=p_d.get("rise_time"),
                settling_time_2pct=p_d.get("settling_time_2pct"),
                overshoot_percent=p_d["overshoot_percent"],
                steady_state_error=p_d["steady_state_error"],
                control_total_variation=p_d["control_total_variation"],
                control_rms=p_d["control_rms"],
                control_peak=p_d["control_peak"],
                saturation_fraction=p_d["saturation_fraction"],
                margins=MarginReport(
                    gain_margin_db=margins_d.get("gain_margin_db"),
                    phase_margin_deg=margins_d.get("phase_margin_deg"),
                    delay_margin_s=margins_d.get("delay_margin_s"),
                    sensitivity_peak=margins_d.get("sensitivity_peak"),
                    complementary_sensitivity_peak=margins_d.get(
                        "complementary_sensitivity_peak"
                    ),
                ),
            )

        meta: Optional[TuneMeta] = None
        if "meta" in d and d["meta"] is not None:
            m_d = d["meta"]
            meta = TuneMeta(
                library_version=m_d.get("library_version", "unknown"),
                elapsed_seconds=m_d.get("elapsed_seconds", 0.0),
                seed=m_d.get("seed"),
                experiment=m_d.get("experiment", ""),
                identifier=m_d.get("identifier", ""),
                tuner=m_d.get("tuner", ""),
                cost_evaluations=m_d.get("cost_evaluations", 0),
            )

        return cls(
            gains=gains,
            status=status,
            confidence=confidence,
            identification=identification,
            performance=performance,
            warnings=warnings_list,
            meta=meta,
        )

    # --- Mutation helpers (return new frozen copies) -------------------

    def with_warnings(self, extra: Iterable[Warning]) -> "TuneResult":
        merged = tuple(self.warnings) + tuple(extra)
        return replace(self, warnings=merged)

    def build_controller(self, **overrides: Any):
        """Construct a :class:`~pid_control.core.pid_controller.PIDController`
        from the tuned gains.

        Imported lazily so this module stays dependency-light and circular
        imports are avoided.  Raises ``RuntimeError`` on ``FAILED`` results
        unless ``force=True`` is explicitly passed.
        """
        force = bool(overrides.pop("force", False))
        if self.status is Status.FAILED and not force:
            codes = ", ".join(sorted({w.code.value for w in self.warnings}))
            raise RuntimeError(
                f"Refusing to build controller from FAILED TuneResult "
                f"(warnings: {codes or 'none'}).  Pass force=True to override."
            )

        from pid_control.core.pid_controller import PIDController  # noqa: WPS433
        from pid_control.core.pid_params import PIDParams  # noqa: WPS433

        kwargs: Dict[str, Any] = {
            "kp": float(self.gains.kp),
            "ki": float(self.gains.ki),
            "kd": float(self.gains.kd),
            "setpoint_weight_p": float(self.gains.setpoint_weight_b),
            "setpoint_weight_d": float(self.gains.setpoint_weight_c),
            "derivative_filter_coeff": float(self.gains.derivative_filter_n),
        }
        kwargs.update(overrides)
        return PIDController(PIDParams(**kwargs))


# ---------------------------------------------------------------------------
# Helpers shared by the stage implementations
# ---------------------------------------------------------------------------

def merge_warnings(*sources: Sequence[Warning]) -> Tuple[Warning, ...]:
    """Concatenate several warning sequences while preserving order."""
    out: List[Warning] = []
    for s in sources:
        out.extend(s)
    return tuple(out)


__all__ = [
    "ActuatorLimits",
    "Artifacts",
    "Confidence",
    "DataQuality",
    "IdentificationResult",
    "MarginReport",
    "ModelType",
    "Objective",
    "PerformanceReport",
    "PIDGains",
    "Severity",
    "Status",
    "Trajectory",
    "TransferFunctionModel",
    "TuneMeta",
    "TuneResult",
    "Warning",
    "WarningCode",
    "merge_warnings",
]
