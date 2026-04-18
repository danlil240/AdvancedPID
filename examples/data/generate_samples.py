"""Generate sample CSV datasets for the golden examples (T7.5).

Run once; the outputs are committed into examples/data/.
"""
import csv
import os
import numpy as np


def _write(filename, t, u, y, sp=None):
    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["time", "input", "measurement"]
        if sp is not None:
            header.append("setpoint")
        w.writerow(header)
        for i in range(len(t)):
            row = [f"{t[i]:.4f}", f"{u[i]:.6f}", f"{y[i]:.6f}"]
            if sp is not None:
                row.append(f"{sp[i]:.6f}")
            w.writerow(row)
    print(f"  wrote {path} ({len(t)} rows)")


def fopdt_step():
    K, tau, theta, dt = 2.0, 3.0, 0.5, 0.02
    T = 30.0
    t = np.arange(0.0, T, dt)
    u = np.zeros_like(t)
    u[t >= 1.0] = 1.0
    y = np.zeros_like(t)
    ds = int(theta / dt)
    for i in range(1, len(t)):
        ud = u[i - ds] if i > ds else 0.0
        y[i] = y[i - 1] + dt / tau * (K * ud - y[i - 1])
    rng = np.random.default_rng(42)
    y += rng.normal(0, 0.01, len(y))
    _write("fopdt_step.csv", t, u, y)


def sopdt_step():
    K, tau1, tau2, theta, dt = 1.5, 2.0, 0.8, 0.3, 0.02
    T = 40.0
    t = np.arange(0.0, T, dt)
    u = np.zeros_like(t)
    u[t >= 1.0] = 1.0
    y1 = np.zeros_like(t)
    y2 = np.zeros_like(t)
    ds = int(theta / dt)
    for i in range(1, len(t)):
        ud = u[i - ds] if i > ds else 0.0
        y1[i] = y1[i - 1] + dt / tau1 * (K * ud - y1[i - 1])
        y2[i] = y2[i - 1] + dt / tau2 * (y1[i - 1] - y2[i - 1])
    rng = np.random.default_rng(42)
    y2 += rng.normal(0, 0.005, len(y2))
    _write("sopdt_step.csv", t, u, y2)


def integrator():
    K_i, theta, dt = 0.5, 0.2, 0.02
    T = 20.0
    t = np.arange(0.0, T, dt)
    u = np.zeros_like(t)
    u[t >= 1.0] = 1.0
    y = np.zeros_like(t)
    ds = int(theta / dt)
    for i in range(1, len(t)):
        ud = u[i - ds] if i > ds else 0.0
        y[i] = y[i - 1] + K_i * ud * dt
    _write("integrator.csv", t, u, y)


def noisy_fopdt():
    K, tau, theta, dt = 1.0, 5.0, 1.0, 0.05
    T = 60.0
    t = np.arange(0.0, T, dt)
    u = np.zeros_like(t)
    u[t >= 2.0] = 1.0
    y = np.zeros_like(t)
    ds = int(theta / dt)
    for i in range(1, len(t)):
        ud = u[i - ds] if i > ds else 0.0
        y[i] = y[i - 1] + dt / tau * (K * ud - y[i - 1])
    rng = np.random.default_rng(123)
    y += rng.normal(0, 0.05, len(y))
    _write("noisy_fopdt.csv", t, u, y)


if __name__ == "__main__":
    print("Generating sample datasets...")
    fopdt_step()
    sopdt_step()
    integrator()
    noisy_fopdt()
    print("Done.")
