"""PIDAutotuner façade — one-line entry point (PLAN.md T1.3, §5).

Example::

    from pid_control.autotune.api import PIDAutotuner

    result = PIDAutotuner.from_csv("heater_step.csv").tune()
    print(result.status, result.gains)
    controller = result.build_controller()
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from pid_control.autotune.experiments.base import ExperimentRecord
from pid_control.autotune.experiments.from_data import (
    FromDataExperiment,
    load_csv,
)
from pid_control.autotune.identification.base import Identifier
from pid_control.autotune.identification.selector import AutoIdentifier
from pid_control.autotune.rules.base import TuningRule
from pid_control.autotune.rules.imc import IMCRule
from pid_control.autotune.tuning.base import Tuner, TunerOutcome
from pid_control.autotune.tuning.cost import CostSpec
from pid_control.autotune.tuning.de import DETuner
from pid_control.autotune.validation.base import ValidationOutcome, Validator
from pid_control.autotune.validation.confidence import ConfidenceAggregator
from pid_control.autotune.validation.stability import (
    RobustnessValidator,
    StabilityValidator,
    compute_margins,
)
from pid_control.autotune.types import (
    ActuatorLimits,
    Artifacts,
    Confidence,
    IdentificationResult,
    MarginReport,
    Objective,
    PerformanceReport,
    PIDGains,
    Severity,
    Status,
    Trajectory,
    TuneMeta,
    TuneResult,
    Warning,
    WarningCode,
    merge_warnings,
)


class PIDAutotuner:
    """Unified autotune façade composing all four pipeline stages.

    Construction
    ------------
    Use one of the class-method factories (``from_csv``, ``from_plant``,
    ``from_arrays``).  Optionally chain ``.with_objective()``,
    ``.with_actuator_limits()``, ``.set_identifier()``, etc.  Then call
    ``.tune()`` to run the pipeline and receive a :class:`TuneResult`.

    Default pipeline (every stage is replaceable):

    1. **Experiment** — ``FromDataExperiment`` (CSV / arrays) or user-
       supplied live experiment.
    2. **Identification** — ``AutoIdentifier`` picks best of FOPDT / SOPDT
       by AIC.
    3. **Rule** — ``IMCRule`` provides the initial PID guess.
    4. **Tuner** — ``DETuner`` (differential evolution, 80 gens).
    5. **Validation** — ``StabilityValidator`` + ``RobustnessValidator``.
    """

    # --- Factories --------------------------------------------------------

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        columns: Optional[Dict[str, str]] = None,
    ) -> "PIDAutotuner":
        """Create an autotuner from a CSV file."""
        experiment = load_csv(path, columns=columns)
        inst = cls()
        inst._experiment = experiment
        return inst

    @classmethod
    def from_arrays(
        cls,
        time: np.ndarray,
        input_signal: np.ndarray,
        output: np.ndarray,
        setpoint: Optional[np.ndarray] = None,
        sample_time: Optional[float] = None,
    ) -> "PIDAutotuner":
        """Create an autotuner from pre-recorded arrays."""
        experiment = FromDataExperiment(
            time=np.asarray(time, dtype=float),
            input_signal=np.asarray(input_signal, dtype=float),
            output=np.asarray(output, dtype=float),
            setpoint=np.asarray(setpoint, dtype=float) if setpoint is not None else None,
            sample_time=sample_time,
        )
        inst = cls()
        inst._experiment = experiment
        return inst

    @classmethod
    def from_plant(
        cls,
        plant: Any,
        experiment: Optional[Any] = None,
    ) -> "PIDAutotuner":
        """Create an autotuner that will excite a plant object.

        If ``experiment`` is not supplied, a default
        :class:`StepExperiment` is used.
        """
        inst = cls()
        inst._plant = plant
        if experiment is not None:
            inst._experiment = experiment
        else:
            from pid_control.autotune.experiments.step import StepExperiment
            inst._experiment = StepExperiment()
        return inst

    # --- Constructor (internal) -------------------------------------------

    def __init__(self) -> None:
        self._experiment: Any = None
        self._plant: Any = None
        self._identifier: Identifier = AutoIdentifier()
        self._rule: TuningRule = IMCRule()
        self._tuner: Optional[Tuner] = DETuner()
        self._validators: List[Validator] = [
            StabilityValidator(),
            RobustnessValidator(),
        ]
        self._objective: Objective = Objective()
        self._actuator: ActuatorLimits = ActuatorLimits()
        self._cost_spec: Optional[CostSpec] = None
        self._seed: int = 42

    # --- Builder methods --------------------------------------------------

    def with_objective(self, objective: Objective) -> "PIDAutotuner":
        self._objective = objective
        return self

    def with_actuator_limits(
        self,
        lower: float = -float("inf"),
        upper: float = float("inf"),
        rate_limit: Optional[float] = None,
    ) -> "PIDAutotuner":
        self._actuator = ActuatorLimits(lower=lower, upper=upper, rate_limit=rate_limit)
        return self

    def set_experiment(self, experiment: Any) -> "PIDAutotuner":
        self._experiment = experiment
        return self

    def set_identifier(self, identifier: Identifier) -> "PIDAutotuner":
        self._identifier = identifier
        return self

    def set_rule(self, rule: TuningRule) -> "PIDAutotuner":
        self._rule = rule
        return self

    def set_tuner(self, tuner: Optional[Tuner]) -> "PIDAutotuner":
        """Set the numerical tuner.  Pass ``None`` to skip numerical
        refinement and use only the analytical rule."""
        self._tuner = tuner
        return self

    def set_cost(self, cost_spec: CostSpec) -> "PIDAutotuner":
        self._cost_spec = cost_spec
        return self

    def set_validators(self, validators: Sequence[Validator]) -> "PIDAutotuner":
        self._validators = list(validators)
        return self

    def add_validator(self, *validators: Validator) -> "PIDAutotuner":
        self._validators.extend(validators)
        return self

    # --- Pipeline execution -----------------------------------------------

    def tune(self) -> TuneResult:
        """Run the full Experiment → Identify → Tune → Validate pipeline.

        Returns a :class:`TuneResult` with ``status``, ``gains``,
        ``confidence``, ``warnings``, and supporting diagnostics.
        Never raises on expected pipeline failures — those are signalled
        through ``Status.FAILED`` + typed warnings.
        """
        t0 = _time.monotonic()
        all_warnings: List[Warning] = []

        # ------------------------------------------------------------------
        # Stage 1: Experiment
        # ------------------------------------------------------------------
        try:
            record = self._run_experiment()
        except Exception as exc:
            return self._fail(str(exc), all_warnings, t0)

        if record.quality is not None:
            all_warnings.extend(record.quality.warnings)
            if record.quality.status is Status.FAILED:
                return self._fail_from_data(record, all_warnings, t0)

        all_warnings.extend(record.warnings)

        # ------------------------------------------------------------------
        # Stage 2: Identification
        # ------------------------------------------------------------------
        try:
            identification = self._identifier.identify(record)
        except Exception as exc:
            all_warnings.append(Warning(
                code=WarningCode.W_POOR_FIT,
                severity=Severity.ERROR,
                message=f"Identification failed: {exc}",
                stage="identify",
            ))
            return self._fail("Identification failed", all_warnings, t0)

        all_warnings.extend(identification.warnings)

        # Check for ERROR-level identification warnings
        if any(w.severity is Severity.ERROR for w in identification.warnings):
            return TuneResult(
                gains=PIDGains(kp=0, ki=0, kd=0),
                status=Status.FAILED,
                confidence=Confidence(score=0.0),
                identification=identification,
                performance=None,
                warnings=tuple(all_warnings),
                meta=self._meta(t0),
            )

        # ------------------------------------------------------------------
        # Stage 3a: Analytical rule → initial gains
        # ------------------------------------------------------------------
        try:
            initial_gains = self._rule.apply(
                identification, self._objective, self._actuator,
            )
        except Exception as exc:
            all_warnings.append(Warning(
                code=WarningCode.W_POOR_FIT,
                severity=Severity.WARNING,
                message=f"Tuning rule failed: {exc}; using fallback gains.",
                stage="rule",
            ))
            initial_gains = PIDGains(kp=1.0, ki=0.1, kd=0.0)

        # ------------------------------------------------------------------
        # Stage 3b: Numerical refinement (optional)
        # ------------------------------------------------------------------
        final_gains = initial_gains
        tuner_outcome: Optional[TunerOutcome] = None

        if self._tuner is not None:
            try:
                tuner_outcome = self._tuner.refine(
                    identification, initial_gains, self._objective, self._actuator,
                )
                final_gains = tuner_outcome.gains
                all_warnings.extend(tuner_outcome.warnings)

                if tuner_outcome.status is Status.FAILED:
                    return TuneResult(
                        gains=initial_gains,
                        status=Status.FAILED,
                        confidence=Confidence(score=0.0),
                        identification=identification,
                        performance=None,
                        warnings=tuple(all_warnings),
                        artifacts=Artifacts(
                            cost_history=tuner_outcome.cost_history,
                        ),
                        meta=self._meta(t0, tuner_outcome),
                    )
            except Exception as exc:
                all_warnings.append(Warning(
                    code=WarningCode.W_MAXITER,
                    severity=Severity.WARNING,
                    message=f"Numerical tuner error: {exc}; using rule-based gains.",
                    stage="tuning",
                ))

        # ------------------------------------------------------------------
        # Stage 4: Validation
        # ------------------------------------------------------------------
        margins: Optional[MarginReport] = None
        pipeline_status = Status.OK

        for validator in self._validators:
            try:
                outcome = validator.validate(
                    identification, final_gains, self._objective, self._actuator,
                )
                all_warnings.extend(outcome.warnings)
                if outcome.margins is not None:
                    margins = outcome.margins
                # Strictest status wins
                if outcome.status is Status.FAILED:
                    pipeline_status = Status.FAILED
                elif outcome.status is Status.WARNING and pipeline_status is not Status.FAILED:
                    pipeline_status = Status.WARNING
            except Exception as exc:
                all_warnings.append(Warning(
                    code=WarningCode.W_LOW_MARGIN,
                    severity=Severity.INFO,
                    message=f"Validator '{getattr(validator, 'name', '?')}' error: {exc}",
                    stage="validate",
                ))

        # ------------------------------------------------------------------
        # Confidence
        # ------------------------------------------------------------------
        data_ok = record.quality is None or record.quality.status is not Status.FAILED
        confidence = ConfidenceAggregator().compute(
            identification, margins, self._objective, data_ok=data_ok,
        )

        # ------------------------------------------------------------------
        # Performance report (from margins if available)
        # ------------------------------------------------------------------
        performance = None
        if margins is not None:
            performance = self._compute_performance(
                identification, final_gains, margins,
            )

        # ------------------------------------------------------------------
        # Assemble result
        # ------------------------------------------------------------------
        return TuneResult(
            gains=final_gains,
            status=pipeline_status,
            confidence=confidence,
            identification=identification,
            performance=performance,
            warnings=tuple(all_warnings),
            artifacts=Artifacts(
                identification_trajectory=record.as_trajectory() if record else None,
                cost_history=tuner_outcome.cost_history if tuner_outcome else None,
            ),
            meta=self._meta(t0, tuner_outcome),
        )

    # --- Internal helpers -------------------------------------------------

    def _run_experiment(self) -> ExperimentRecord:
        if self._experiment is None:
            raise ValueError("No experiment configured — use from_csv(), from_plant(), or set_experiment().")
        return self._experiment.run(plant=self._plant)

    def _fail(
        self, message: str, warnings: List[Warning], t0: float,
    ) -> TuneResult:
        return TuneResult(
            gains=PIDGains(kp=0, ki=0, kd=0),
            status=Status.FAILED,
            confidence=Confidence(score=0.0),
            identification=None,
            performance=None,
            warnings=tuple(warnings),
            meta=self._meta(t0),
        )

    def _fail_from_data(
        self, record: ExperimentRecord, warnings: List[Warning], t0: float,
    ) -> TuneResult:
        return TuneResult(
            gains=PIDGains(kp=0, ki=0, kd=0),
            status=Status.FAILED,
            confidence=Confidence(score=0.0),
            identification=None,
            performance=None,
            warnings=tuple(warnings),
            artifacts=Artifacts(
                identification_trajectory=record.as_trajectory(),
            ),
            meta=self._meta(t0),
        )

    def _meta(
        self, t0: float, tuner_outcome: Optional[TunerOutcome] = None,
    ) -> TuneMeta:
        return TuneMeta(
            library_version="0.2.0-dev",
            elapsed_seconds=_time.monotonic() - t0,
            seed=self._seed,
            experiment=getattr(self._experiment, "name", ""),
            identifier=getattr(self._identifier, "name", ""),
            tuner=getattr(self._tuner, "name", "") if self._tuner else "",
            cost_evaluations=(
                tuner_outcome.meta.get("nfev", 0) if tuner_outcome else 0
            ),
        )

    def _compute_performance(
        self,
        identification: IdentificationResult,
        gains: PIDGains,
        margins: MarginReport,
    ) -> Optional[PerformanceReport]:
        """Quick closed-loop sim to populate time-domain metrics."""
        from pid_control.autotune.tuning.cost import CostEvaluator, CostSpec

        try:
            evaluator = CostEvaluator(
                identification=identification,
                objective=self._objective,
                actuator=self._actuator,
                cost_spec=CostSpec(),
            )
            dt = evaluator.dt
            n = int(evaluator.sim_duration / dt) + 1
            time = np.linspace(0.0, evaluator.sim_duration, n)
            sp = np.ones(n)
            y = np.zeros(n)
            u = np.zeros(n)
            e_int = 0.0
            e_prev = 0.0
            e_filt = 0.0
            kp, ki, kd = gains.kp, gains.ki, gains.kd
            N = gains.derivative_filter_n
            lo, hi = self._actuator.lower, self._actuator.upper
            model = identification.model
            tau = max(float(model.tau), 1e-9)
            delay_samples = int(round(model.theta / dt))

            for i in range(1, n):
                e = sp[i - 1] - y[i - 1]
                alpha = N * dt / (1.0 + N * dt) if N > 0 else 0.0
                e_filt = alpha * (e - e_prev) / dt + (1.0 - alpha) * e_filt if i > 1 else 0.0
                u_raw = kp * e + ki * e_int + kd * e_filt
                if lo <= u_raw <= hi:
                    e_int += e * dt
                u_cmd = float(np.clip(kp * e + ki * e_int + kd * e_filt, lo, hi))
                u[i] = u_cmd
                u_idx = max(i - delay_samples, 0)
                alpha_p = float(np.exp(-dt / tau))
                y[i] = alpha_p * y[i - 1] + model.K * (1.0 - alpha_p) * u[u_idx]
                e_prev = e
                if abs(y[i]) > 1e6:
                    return None

            error = sp - y
            abs_error = np.abs(error)

            # Metrics
            iae = float(np.trapezoid(abs_error, time))
            ise = float(np.trapezoid(error ** 2, time))
            itae = float(np.trapezoid(time * abs_error, time))

            peak = float(np.max(y))
            os_pct = max(0.0, (peak - 1.0) * 100.0)
            ss_err = float(abs(y[-1] - 1.0))

            # Rise time (10% to 90%)
            rise_time = None
            idx10 = np.searchsorted(y, 0.1)
            idx90 = np.searchsorted(y, 0.9)
            if idx10 < idx90 < n:
                rise_time = float(time[idx90] - time[idx10])

            # Settling time (2%)
            within = np.abs(y - 1.0) <= 0.02
            settle = None
            for j in range(n - 1, -1, -1):
                if not within[j]:
                    settle = float(time[min(j + 1, n - 1)])
                    break
            if settle is None:
                settle = 0.0

            tv = float(np.sum(np.abs(np.diff(u))))
            u_rms = float(np.sqrt(np.mean(u ** 2)))
            u_peak = float(np.max(np.abs(u)))
            sat_frac = float(np.mean((u <= lo) | (u >= hi))) if np.isfinite(lo) and np.isfinite(hi) else 0.0

            return PerformanceReport(
                iae=iae, ise=ise, itae=itae,
                rise_time=rise_time,
                settling_time_2pct=settle,
                overshoot_percent=os_pct,
                steady_state_error=ss_err,
                control_total_variation=tv,
                control_rms=u_rms,
                control_peak=u_peak,
                saturation_fraction=sat_frac,
                margins=margins,
            )
        except Exception:
            return None


__all__ = ["PIDAutotuner"]
