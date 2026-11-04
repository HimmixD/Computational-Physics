"""Numerical experiments for exercise 3, tasks 1a--1f.

The linear reaction chain is

    A --k1--> B --k2--> C,

with x(0) = (1, 0, 0).  The module contains the four requested time
integrators and creates all figures and tables used in the accompanying
``exercise_03_task_1_output.md`` file.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


INITIAL_STATE = np.array([1.0, 0.0, 0.0])


def system_matrix(k1: float = 1.0, k2: float = 1.0) -> np.ndarray:
    """Return the matrix J in ``x_dot = J x`` for the reaction chain."""
    if k1 <= 0 or k2 <= 0:
        raise ValueError("k1 and k2 must be positive")
    return np.array(
        [[-k1, 0.0, 0.0], [k1, -k2, 0.0], [0.0, k2, 0.0]],
        dtype=float,
    )


def analytical_solution(t: np.ndarray, k1: float = 1.0, k2: float = 1.0) -> np.ndarray:
    """Evaluate the exact solution for one or more times.

    The equal-rate expression is used separately to avoid cancellation in the
    formula containing ``k2-k1``.
    """
    times = np.asarray(t, dtype=float)
    if np.any(times < 0):
        raise ValueError("times must be non-negative")
    a = np.exp(-k1 * times)
    if np.isclose(k1, k2):
        c = 1.0 - a * (1.0 + k1 * times)
        b = k1 * times * a
    else:
        b = k1 * (np.exp(-k1 * times) - np.exp(-k2 * times)) / (k2 - k1)
        c = 1.0 - (k2 * np.exp(-k1 * times) - k1 * np.exp(-k2 * times)) / (k2 - k1)
    return np.column_stack((a, b, c))


def _time_grid(final_time: float, dt: float) -> np.ndarray:
    """Return an exact-step grid, requiring that ``final_time / dt`` is integral."""
    if final_time <= 0 or dt <= 0:
        raise ValueError("final_time and dt must be positive")
    steps = int(round(final_time / dt))
    if not np.isclose(steps * dt, final_time, rtol=1e-10, atol=1e-12):
        raise ValueError("dt must divide final_time")
    return np.linspace(0.0, final_time, steps + 1)


def explicit_euler(jacobian: np.ndarray, state: np.ndarray, dt: float) -> np.ndarray:
    """Advance one step with explicit Euler."""
    with np.errstate(over="ignore", invalid="ignore"):
        return state + dt * (jacobian @ state)


def heun(jacobian: np.ndarray, state: np.ndarray, dt: float) -> np.ndarray:
    """Advance one step with explicit trapezoidal rule (Heun)."""
    slope_1 = jacobian @ state
    predictor = state + dt * slope_1
    return state + 0.5 * dt * (slope_1 + jacobian @ predictor)


def rk4(jacobian: np.ndarray, state: np.ndarray, dt: float) -> np.ndarray:
    """Advance one step with classical fourth-order Runge--Kutta."""
    k_1 = jacobian @ state
    k_2 = jacobian @ (state + 0.5 * dt * k_1)
    k_3 = jacobian @ (state + 0.5 * dt * k_2)
    k_4 = jacobian @ (state + dt * k_3)
    return state + dt * (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6.0


def implicit_euler(jacobian: np.ndarray, state: np.ndarray, dt: float) -> np.ndarray:
    """Advance one step with implicit Euler by solving ``(I-dt J)x_next=x``."""
    return np.linalg.solve(np.eye(jacobian.shape[0]) - dt * jacobian, state)


METHODS = {
    "Explicit Euler": explicit_euler,
    "Heun": heun,
    "RK4": rk4,
    "Implicit Euler": implicit_euler,
}


def integrate(
    method: str,
    dt: float,
    final_time: float = 10.0,
    k1: float = 1.0,
    k2: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the reaction chain and return ``(times, states)``."""
    if method not in METHODS:
        raise KeyError(f"unknown method: {method}")
    times = _time_grid(final_time, dt)
    jacobian = system_matrix(k1, k2)
    states = np.empty((times.size, 3), dtype=float)
    states[0] = INITIAL_STATE
    stepper = METHODS[method]
    for index in range(times.size - 1):
        states[index + 1] = stepper(jacobian, states[index], dt)
    return times, states


def max_error(method: str, dt: float, final_time: float, k1: float, k2: float) -> float:
    """Return the maximum componentwise error on the integration grid."""
    times, numerical = integrate(method, dt, final_time, k1, k2)
    return float(np.max(np.abs(numerical - analytical_solution(times, k1, k2))))


def stability_function(method: str, z: np.ndarray | complex) -> np.ndarray | complex:
    """Return the scalar stability function R(z) of a requested method."""
    if method == "Explicit Euler":
        return 1.0 + z
    if method == "Heun":
        return 1.0 + z + z**2 / 2.0
    if method == "RK4":
        return 1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0
    if method == "Implicit Euler":
        return 1.0 / (1.0 - z)
    raise KeyError(f"unknown method: {method}")


def _log_grid(start: float, stop: float, points: int = 18) -> np.ndarray:
    """Return logarithmically spaced steps that divide 10 exactly."""
    minimum_steps = int(round(10.0 / stop))
    maximum_steps = int(round(10.0 / start))
    step_counts = np.unique(
        np.round(np.logspace(np.log10(minimum_steps), np.log10(maximum_steps), points)).astype(int)
    )
    return 10.0 / step_counts


def make_plots(output_dir: Path) -> dict[str, Path]:
    """Generate the figures for tasks 1c--1f and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # 1c: convergence slopes for k2=10.
    dts = _log_grid(10.0 / 2048.0, 10.0 / 8.0, 24)
    errors = {name: np.array([max_error(name, dt, 10.0, 1.0, 10.0) for dt in dts]) for name in METHODS}
    figure, axis = plt.subplots(figsize=(7.5, 5))
    for name, values in errors.items():
        axis.loglog(dts, values, "o-", label=name)
    axis.loglog(dts, errors["Explicit Euler"][-1] * (dts / dts[-1]), "k--", alpha=0.5, label=r"$O(\Delta t)$")
    axis.loglog(dts, errors["RK4"][-1] * (dts / dts[-1]) ** 4, "k:", alpha=0.7, label=r"$O(\Delta t^4)$")
    axis.set(xlabel=r"$\Delta t$", ylabel="maximum absolute error", title="1c: accuracy for $k_2=10$")
    axis.grid(True, which="both", ls=":", alpha=0.5)
    axis.legend()
    path = output_dir / "task_1c_accuracy_loglog.png"
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths["accuracy_plot"] = path

    # 1d: stability threshold and a trajectory just beyond it.
    critical = {"Explicit Euler": 2.0 / 10.0, "Heun": 2.0 / 10.0, "RK4": 2.7852935634 / 10.0, "Implicit Euler": np.inf}
    stability_counts = np.unique(np.round(np.linspace(32, 1000, 180)).astype(int))
    dts_stability = 10.0 / stability_counts
    final_norms = {}
    for name in METHODS:
        final_norms[name] = np.array(
            [np.max(np.abs(integrate(name, float(dt), 10.0, 1.0, 10.0)[1][-1])) for dt in dts_stability]
        )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, values in final_norms.items():
        axes[0].semilogy(dts_stability, values, label=name)
    axes[0].axhline(1.0, color="k", ls=":", lw=1)
    axes[0].set(xlabel=r"$\Delta t$", ylabel=r"$\max |x(10)|$", title="1d: stability scan")
    axes[0].grid(True, which="both", ls=":", alpha=0.5)
    axes[0].legend()
    dt_above = 10.0 / 47.0
    for name in ("Explicit Euler", "Heun", "RK4", "Implicit Euler"):
        times, states = integrate(name, dt_above, 10.0, 1.0, 10.0)
        axes[1].plot(times, states[:, 2], label=name)
    axes[1].set(xlabel="t", ylabel="C(t)", title=r"Just above Euler threshold ($\Delta t=0.21$)")
    axes[1].grid(True, ls=":", alpha=0.5)
    axes[1].legend()
    path = output_dir / "task_1d_stability.png"
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths["stability_plot"] = path

    # 1e: stability regions and eigenvalue locations z=lambda*dt.
    grid = np.linspace(-4.0, 2.0, 500)
    real, imag = np.meshgrid(grid, grid)
    z = real + 1j * imag
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    for axis, name in zip(axes.flat, METHODS):
        region = np.abs(stability_function(name, z)) <= 1.0
        axis.contourf(real, imag, region, levels=[0.5, 1.5], alpha=0.35, cmap="Blues")
        for dt, marker in ((0.05, "o"), (0.15, "s"), (0.3, "^")):
            eigenvalues = np.array([-1.0, -10.0, 0.0]) * dt
            axis.scatter(eigenvalues, np.zeros(3), s=22, marker=marker, label=fr"$\Delta t={dt}$")
        axis.axhline(0, color="k", lw=0.6)
        axis.axvline(0, color="k", lw=0.6)
        axis.set_title(name)
        axis.grid(True, ls=":", alpha=0.4)
    axes[0, 0].set_ylabel("Im(z)")
    axes[1, 0].set_ylabel("Im(z)")
    axes[1, 0].set_xlabel("Re(z)")
    axes[1, 1].set_xlabel("Re(z)")
    axes[0, 1].legend(fontsize=8, loc="upper right")
    figure.suptitle(r"1e: stability regions $|R(z)|\leq1$ and $z=\lambda\Delta t$")
    path = output_dir / "task_1e_stability_regions.png"
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths["stability_regions"] = path

    # Eigenmodes are the columns of the eigenvector matrix.
    jacobian = system_matrix(1.0, 10.0)
    eigenvalues, eigenvectors = np.linalg.eig(jacobian)
    order = np.argsort(eigenvalues)
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for axis, index in zip(axes, order):
        vector = np.real(eigenvectors[:, index])
        vector /= np.max(np.abs(vector))
        axis.bar(("A", "B", "C"), vector, color=("tab:blue", "tab:orange", "tab:green"))
        axis.axhline(0, color="k", lw=0.7)
        axis.set_title(fr"$\lambda={eigenvalues[index]:.3g}$")
        axis.set_ylim(-1.1, 1.1)
        axis.grid(True, axis="y", ls=":", alpha=0.5)
    figure.suptitle("1e: normalized eigenmodes for $k_1=1$, $k_2=10$")
    path = output_dir / "task_1e_eigenmodes.png"
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths["eigenmodes"] = path

    # 1f: largest step meeting the requested error target.
    candidate_counts = np.unique(np.round(np.logspace(1, 5, 80)).astype(int))
    candidate_steps = 10.0 / candidate_counts
    stiffness_rows = []
    for name in ("Explicit Euler", "Implicit Euler"):
        valid = [(float(dt), max_error(name, float(dt), 10.0, 1.0, 1000.0)) for dt in candidate_steps]
        acceptable = [(dt, error) for dt, error in valid if error <= 1e-2]
        dt_best, error_best = max(acceptable, key=lambda pair: pair[0])
        stiffness_rows.append((name, dt_best, int(round(10.0 / dt_best)), error_best))
    figure, axis = plt.subplots(figsize=(7.5, 5))
    for name in ("Explicit Euler", "Implicit Euler"):
        values = np.array([max_error(name, float(dt), 10.0, 1.0, 1000.0) for dt in candidate_steps])
        values[~np.isfinite(values)] = np.nan
        axis.loglog(candidate_steps, values, label=name)
    axis.axhline(1e-2, color="k", ls="--", label=r"target $10^{-2}$")
    axis.set_ylim(1e-14, 1e8)
    axis.set(xlabel=r"$\Delta t$", ylabel="maximum absolute error", title="1f: stiffness for $k_2=1000$")
    axis.grid(True, which="both", ls=":", alpha=0.5)
    axis.legend()
    path = output_dir / "task_1f_stiffness.png"
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths["stiffness_plot"] = path
    paths["_stiffness_rows"] = stiffness_rows  # type: ignore[assignment]
    paths["_critical"] = critical  # type: ignore[assignment]
    paths["_accuracy"] = errors  # type: ignore[assignment]
    paths["_eigenvalues"] = eigenvalues  # type: ignore[assignment]
    return paths


def write_report(output_dir: Path, results: dict[str, Path]) -> Path:
    """Write the explanatory report and machine-readable numerical tables."""
    accuracy = results["_accuracy"]  # type: ignore[assignment]
    dts = _log_grid(10.0 / 2048.0, 10.0 / 8.0, 24)
    slopes = {
        name: np.polyfit(np.log(dts[-8:]), np.log(values[-8:]), 1)[0]
        for name, values in accuracy.items()
    }
    critical = results["_critical"]  # type: ignore[assignment]
    eigenvalues = results["_eigenvalues"]  # type: ignore[assignment]
    stiffness_rows = results["_stiffness_rows"]  # type: ignore[assignment]
    with (output_dir / "accuracy_data.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["dt", *METHODS])
        for index, dt in enumerate(dts):
            writer.writerow([dt, *(accuracy[name][index] for name in METHODS)])
    lines = [
        "# Exercise 3 - Task 1: linear reaction chain",
        "",
        "## 1a - Standard form and conservation law",
        "",
        "With $x=(A,B,C)^T$, the reaction equations are",
        "",
        "$$\\dot x=Jx,\\qquad J=\\begin{pmatrix}-k_1&0&0\\\\k_1&-k_2&0\\\\0&k_2&0\\end{pmatrix}.$$",
        "",
        "The first row describes loss of A, the second the gain from A and loss to C, and the third the gain of C.",
        "Adding the three equations gives $\\dot A+\\dot B+\\dot C=0$. Therefore $A+B+C$ is constant and equals 1 for the given initial condition.",
        "",
        "## 1b - Methods and correctness checks",
        "",
        "The implementation is in [`exercise_03_ode.py`](../exercise_03_ode.py). It contains explicit Euler, Heun, classical RK4, and implicit Euler.",
        "For implicit Euler the linear system $(I-\\Delta t J)x_{n+1}=x_n$ is solved directly.",
        "Correctness is checked against the analytical solution, by convergence slopes, and by the invariant $A+B+C=1$.",
        "The exact solution used for comparison is",
        "",
        "$$C(t)=1-\\frac{k_2e^{-k_1t}-k_1e^{-k_2t}}{k_2-k_1},$$",
        "$$C(t)=1-e^{-k_1t}(1+k_1t)\\quad(k_1=k_2).$$",
        "",
        "## 1c - Accuracy for $k_2=10$",
        "",
        "![Accuracy](task_1c_accuracy_loglog.png)",
        "",
        "The measured slopes on the finest eight step sizes are:",
        "",
        "| Method | expected order | measured slope |",
        "|---|---:|---:|",
    ]
    expected = {"Explicit Euler": 1, "Heun": 2, "RK4": 4, "Implicit Euler": 1}
    lines.extend(f"| {name} | {expected[name]} | {slopes[name]:.3f} |" for name in METHODS)
    lines.extend(
        [
            "",
            "Thus the log-log plot confirms first order for both Euler methods, second order for Heun, and fourth order for RK4 until round-off or stability effects dominate.",
            "",
            "## 1d - Stability",
            "",
            "![Stability](task_1d_stability.png)",
            "",
            "For the negative real eigenvalue $-k_2$, the critical steps are approximately",
            "",
            "| Method | critical $\\Delta t$ for $k_2=10$ |",
            "|---|---:|",
        ]
    )
    lines.extend(
        f"| {name} | {'unconditional' if not np.isfinite(value) else f'{value:.6g}'} |"
        for name, value in critical.items()
    )
    lines.extend(
        [
            "",
            "For explicit Euler and Heun the real-axis boundary is $\\Delta t=2/k_{\\max}$; for RK4 it is $\\Delta t\\approx2.7853/k_{\\max}$. Here $k_{\\max}=\\max(k_1,k_2)$.",
            "Just above the critical step, the fast mode changes from decaying to alternating and growing. The computed concentration therefore develops oscillations and eventually unphysical negative values.",
            "",
            "## 1e - Eigenvalues, eigenvectors, and stability regions",
            "",
            f"The eigenvalues for $k_1=1$, $k_2=10$ are {', '.join(f'${value:.6g}$' for value in sorted(eigenvalues))}; in general they are $0$, $-k_1$, and $-k_2$.",
            "The eigenmode belonging to the most negative eigenvalue $-\\max(k_1,k_2)$ becomes unstable first, because its $z=\\lambda\\Delta t$ reaches the left stability boundary first.",
            "Implicit Euler has $R(z)=1/(1-z)$. For $\\operatorname{Re}(z)\\le0$, $|1-z|\\ge1$, so $|R(z)|\\le1$; it is therefore A-stable and always stable for this reaction problem.",
            "",
            "![Stability regions](task_1e_stability_regions.png)",
            "",
            "![Eigenmodes](task_1e_eigenmodes.png)",
            "",
            "## 1f - Stiffness, $k_2=1000$",
            "",
            "The table uses the largest tested step with maximum error at most $10^{-2}$ over $t\\in[0,10]$.",
            "",
            "| Method | selected $\\Delta t$ | steps | maximum error |",
            "|---|---:|---:|---:|",
        ]
    )
    lines.extend(f"| {name} | {dt:.6g} | {steps} | {error:.3e} |" for name, dt, steps, error in stiffness_rows)
    lines.extend(
        [
            "",
            "![Stiffness](task_1f_stiffness.png)",
            "",
            "Explicit Euler is limited by stability: $\\Delta t\\le2/k_2=0.002$, so the fast transient forces thousands of steps even though the slow dynamics last until $t=10$.",
            "Implicit Euler has no such stability restriction. Its step size is instead limited by accuracy, especially by resolving the fast initial transient and by the first-order truncation error.",
        ]
    )
    report = output_dir / "exercise_03_task_1_output.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    """Generate all task 1 output files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("exercise_03_task_1_output"))
    args = parser.parse_args()
    results = make_plots(args.output)
    report = write_report(args.output, results)
    print(f"Generated task 1 output in {args.output}")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
