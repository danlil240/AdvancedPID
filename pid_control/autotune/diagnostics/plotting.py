"""Diagnostic plotting for TuneResult (PLAN.md T6.2).

All functions accept a ``save_path`` and default ``show=False`` so they
work headlessly in CI.  ``plt.show()`` is never called unless the caller
explicitly passes ``show=True``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from pid_control.autotune.types import TuneResult


def plot_result(
    result: "TuneResult",
    kind: str = "all",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Main entry — dispatches to individual plot routines.

    ``kind`` is one of ``"fit"``, ``"response"``, ``"margins"``, or
    ``"all"`` (the default).
    """
    import matplotlib.pyplot as plt

    figs = []
    if kind in ("all", "fit"):
        fig = _plot_identification(result)
        if fig is not None:
            figs.append(("fit", fig))
    if kind in ("all", "response"):
        fig = _plot_closed_loop(result)
        if fig is not None:
            figs.append(("response", fig))
    if kind in ("all", "margins"):
        fig = _plot_bode(result)
        if fig is not None:
            figs.append(("margins", fig))
    if kind in ("all", "cost"):
        fig = _plot_cost_history(result)
        if fig is not None:
            figs.append(("cost", fig))

    if save_path is not None:
        sp = Path(save_path)
        sp.mkdir(parents=True, exist_ok=True)
        for name, fig in figs:
            fig.savefig(sp / f"{name}.png", dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return {name: fig for name, fig in figs}


def _plot_identification(result: "TuneResult"):
    """Overlay identified model vs recorded data."""
    import matplotlib.pyplot as plt
    from pid_control.autotune.identification.simulate import simulate_model

    art = result.artifacts
    if art is None or art.identification_trajectory is None:
        return None

    traj = art.identification_trajectory
    t = traj.time
    u = traj.control
    y_data = traj.measurement

    if result.identification is None:
        return None

    model = result.identification.model
    try:
        y_model = simulate_model(model, t, u, y0=float(y_data[0]))
    except Exception:
        return None

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, y_data, "b-", label="Measured", alpha=0.8)
    axes[0].plot(t, y_model, "r--", label=f"Model ({model.model_type.value})", linewidth=2)
    axes[0].set_ylabel("Output")
    axes[0].legend()
    axes[0].set_title(
        f"System Identification — R²={result.identification.fit_quality_r2:.4f}"
    )
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, u, "g-", label="Input", alpha=0.8)
    axes[1].set_ylabel("Input")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def _plot_closed_loop(result: "TuneResult"):
    """Step response of tuned controller on identified plant."""
    import matplotlib.pyplot as plt

    perf = result.performance
    if perf is None or result.identification is None:
        return None

    # Simulate a unit step
    from pid_control.autotune.types import IdentificationResult
    model = result.identification.model
    g = result.gains
    tau = max(float(model.tau), 1e-9)
    dt = max(tau / 100.0, 0.001)
    duration = max(10.0 * (tau + model.theta), 5.0)
    n = int(duration / dt) + 1
    t = np.linspace(0.0, duration, n)
    sp = np.ones(n)
    y = np.zeros(n)
    u = np.zeros(n)
    e_int = 0.0
    e_prev = 0.0
    e_filt = 0.0
    kp, ki, kd = g.kp, g.ki, g.kd
    N = g.derivative_filter_n
    delay_samples = int(round(model.theta / dt))

    for i in range(1, n):
        e = sp[i - 1] - y[i - 1]
        alpha = N * dt / (1.0 + N * dt) if N > 0 else 0.0
        e_filt = alpha * (e - e_prev) / dt + (1.0 - alpha) * e_filt if i > 1 else 0.0
        u_raw = kp * e + ki * e_int + kd * e_filt
        e_int += e * dt
        u[i] = u_raw
        u_idx = max(i - delay_samples, 0)
        alpha_p = float(np.exp(-dt / tau))
        y[i] = alpha_p * y[i - 1] + model.K * (1.0 - alpha_p) * u[u_idx]
        e_prev = e
        if abs(y[i]) > 1e6:
            break

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, sp, "k--", label="Setpoint", alpha=0.5)
    axes[0].plot(t, y, "b-", label="Output", linewidth=1.5)
    axes[0].set_ylabel("Output")
    axes[0].legend()
    info = (f"Kp={kp:.3g}, Ki={ki:.3g}, Kd={kd:.3g}"
            f" | OS={perf.overshoot_percent:.1f}%"
            f" | IAE={perf.iae:.3g}")
    axes[0].set_title(f"Closed-Loop Step Response — {info}")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, u, "g-", label="Control", linewidth=1.5)
    axes[1].set_ylabel("Control")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def _plot_bode(result: "TuneResult"):
    """Bode plot of the open-loop transfer function with margin annotations."""
    if result.identification is None:
        return None

    try:
        import control as ct
        import matplotlib.pyplot as plt
        from pid_control.autotune.validation.stability import _build_loop_tf
        L = _build_loop_tf(result.identification, result.gains)
    except Exception:
        return None

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    omega = np.logspace(-3, 3, 2000)
    try:
        resp = ct.frequency_response(L, omega)
        frdata = getattr(resp, 'frdata', getattr(resp, 'fresp', None))
        mag = np.abs(frdata).flatten()
        phase = np.angle(frdata, deg=True).flatten()
    except Exception:
        return None

    axes[0].semilogx(omega, 20 * np.log10(mag), "b-", linewidth=1.5)
    axes[0].axhline(0, color="k", linestyle="--", linewidth=0.5)
    axes[0].set_ylabel("Magnitude [dB]")
    axes[0].set_title("Open-Loop Bode Plot")
    axes[0].grid(True, alpha=0.3, which="both")

    axes[1].semilogx(omega, phase, "b-", linewidth=1.5)
    axes[1].axhline(-180, color="k", linestyle="--", linewidth=0.5)
    axes[1].set_ylabel("Phase [°]")
    axes[1].set_xlabel("Frequency [rad/s]")
    axes[1].grid(True, alpha=0.3, which="both")

    # Annotate margins if available
    if result.performance is not None:
        m = result.performance.margins
        if m.phase_margin_deg is not None:
            axes[1].axhline(-180 + m.phase_margin_deg,
                            color="r", linestyle=":", alpha=0.7,
                            label=f"PM={m.phase_margin_deg:.1f}°")
            axes[1].legend()
        if m.gain_margin_db is not None:
            axes[0].axhline(-m.gain_margin_db,
                            color="r", linestyle=":", alpha=0.7,
                            label=f"GM={m.gain_margin_db:.1f} dB")
            axes[0].legend()

    fig.tight_layout()
    return fig


def _plot_cost_history(result: "TuneResult"):
    """Cost convergence during numerical tuning."""
    import matplotlib.pyplot as plt

    art = result.artifacts
    if art is None or art.cost_history is None:
        return None
    hist = np.asarray(art.cost_history)
    if hist.size < 2:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist, "b-", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    ax.set_title("Optimizer Cost Convergence")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
