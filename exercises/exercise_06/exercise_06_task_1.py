"""Exercise 6, task 1: stationary one-dimensional Schrödinger equation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq


def dirichlet_grid(n: int, left: float, right: float) -> tuple[np.ndarray, float]:
    """Interior grid points and spacing for a Dirichlet box [left, right]."""
    if n < 1:
        raise ValueError("n must be positive")
    dx = (right - left) / (n + 1)
    return left + dx * np.arange(1, n + 1), dx


def solve_stationary(
    n: int,
    left: float,
    right: float,
    potential,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve H=-1/2 d²/dx²+V on an interior Dirichlet grid."""
    points, dx = dirichlet_grid(n, left, right)
    diagonal = 1.0 / dx**2 + np.asarray(potential(points), dtype=float)
    off_diagonal = np.full(n - 1, -0.5 / dx**2, dtype=float)
    energies, states = eigh_tridiagonal(diagonal, off_diagonal)
    return points, energies, dx


def solve_grid_for_states(n: int, left: float, right: float, potential, count: int):
    points, energies, dx = solve_stationary(n, left, right, potential)
    return points, energies[:count], dx


def save_energy_error_csv(path: Path, state_numbers: np.ndarray, numerical: np.ndarray,
                          exact: np.ndarray) -> None:
    np.savetxt(
        path,
        np.column_stack((state_numbers, numerical, exact, numerical - exact,
                         np.abs(numerical - exact))),
        delimiter=",",
        header="state_n,script_energy,analytic_energy,absolute_error_signed,absolute_error",
        comments="",
    )


def save_dx_error_csv(path: Path, dx_values: np.ndarray, level_errors: dict[int, np.ndarray]) -> None:
    header = ["dx"] + [f"error_n{n}" for n in sorted(level_errors)]
    columns = [dx_values] + [level_errors[n] for n in sorted(level_errors)]
    np.savetxt(path, np.column_stack(columns), delimiter=",", header=",".join(header), comments="")


def plot_energies(output_dir: Path, name: str, title: str, state_numbers: np.ndarray,
                  numerical: np.ndarray, exact: np.ndarray) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(state_numbers, numerical, "o-", label="Finite difference")
    axis.plot(state_numbers, exact, "x--", label="Analytic")
    axis.set(xlabel=r"state $n$", ylabel=r"energy $E_n$", title=title)
    axis.grid(True, ls=":", alpha=0.5)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / f"task_{name}_energies.png", dpi=180)
    plt.close(figure)


def plot_error_vs_n(output_dir: Path, name: str, title: str, state_numbers: np.ndarray,
                    errors: np.ndarray) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.semilogy(state_numbers, np.maximum(errors, 1e-14), "o-")
    axis.set(xlabel=r"state $n$", ylabel=r"$|E_{\mathrm{script}}-E_{\mathrm{analytic}}|$",
             title=title)
    axis.grid(True, which="both", ls=":", alpha=0.5)
    figure.tight_layout()
    figure.savefig(output_dir / f"task_{name}_error_vs_n.png", dpi=180)
    plt.close(figure)


def plot_error_vs_dx(output_dir: Path, name: str, title: str,
                     dx_values: np.ndarray, level_errors: dict[int, np.ndarray]) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    for state_n in sorted(level_errors):
        axis.loglog(dx_values, np.maximum(level_errors[state_n], 1e-14), "o-",
                    label=f"n={state_n}")
    axis.set(xlabel=r"grid spacing $\Delta x$", ylabel=r"$|E_{\mathrm{script}}-E_{\mathrm{analytic}}|$",
             title=title)
    axis.grid(True, which="both", ls=":", alpha=0.5)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / f"task_{name}_error_vs_dx.png", dpi=180)
    plt.close(figure)


def convergence_by_dx(n_values: np.ndarray, left: float, right: float, potential,
                      exact_energies, levels: list[int]):
    """Compute errors for selected quantum levels over a grid refinement sequence."""
    dx_values = np.empty(n_values.size)
    level_errors = {n: np.empty(n_values.size) for n in levels}
    for i, n in enumerate(n_values):
        _, energies, dx = solve_stationary(int(n), left, right, potential)
        dx_values[i] = dx
        for n in levels:
            level_errors[n][i] = abs(energies[n] - exact_energies[n])
    return dx_values, level_errors


def task_1a(output_dir: Path) -> dict:
    left, right = -8.0, 8.0
    n = 800
    points, energies, dx = solve_stationary(n, left, right, lambda x: 0.5 * x**2)

    level_count = 12
    state_numbers = np.arange(level_count)
    numerical = energies[:level_count]
    exact = state_numbers + 0.5

    plot_energies(output_dir, "1a", "1a: harmonic oscillator energy levels",
                  state_numbers, numerical, exact)
    errors = np.abs(numerical - exact)
    plot_error_vs_n(output_dir, "1a", "1a: fixed-grid error versus quantum number",
                    state_numbers, errors)
    save_energy_error_csv(output_dir / "task_1a_fixed_grid.csv", state_numbers, numerical, exact)

    n_values = np.array([50, 75, 100, 150, 200, 300, 400, 600, 800, 1200])
    dx_values, level_errors = convergence_by_dx(
        n_values, left, right, lambda x: 0.5 * x**2,
        exact, [0, 1, 2, 5, 10]
    )
    plot_error_vs_dx(output_dir, "1a", "1a: harmonic oscillator grid-convergence",
                     dx_values, level_errors)
    save_dx_error_csv(output_dir / "task_1a_error_vs_dx.csv", dx_values, level_errors)

    return {"n_fixed": n, "dx_fixed": dx, "energies": numerical, "exact": exact,
            "errors": errors, "dx_values": dx_values, "level_errors": level_errors}


def task_1b(output_dir: Path) -> dict:
    left, right = 0.0, 1.0
    n = 1000
    points, energies, dx = solve_stationary(n, left, right, lambda x: 0.0 * x)

    level_count = 12
    state_numbers = np.arange(1, level_count + 1)
    numerical = energies[:level_count]
    exact = state_numbers**2 * np.pi**2 / 2.0

    plot_energies(output_dir, "1b", "1b: infinite square-well energy levels",
                  state_numbers, numerical, exact)
    errors = np.abs(numerical - exact)
    plot_error_vs_n(output_dir, "1b", "1b: fixed-grid error versus quantum number",
                    state_numbers, errors)
    save_energy_error_csv(output_dir / "task_1b_fixed_grid.csv", state_numbers, numerical, exact)

    n_values = np.array([50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 1600])
    exact_lookup = lambda n: n**2 * np.pi**2 / 2.0
    dx_values = np.empty(n_values.size)
    level_errors = {n: np.empty(n_values.size) for n in (1, 2, 5, 10, 20)}
    for i, grid_n in enumerate(n_values):
        _, grid_energies, grid_dx = solve_stationary(int(grid_n), left, right, lambda x: 0.0 * x)
        dx_values[i] = grid_dx
        for state_n in level_errors:
            level_errors[state_n][i] = abs(grid_energies[state_n - 1] - exact_lookup(state_n))
    plot_error_vs_dx(output_dir, "1b", "1b: infinite-well grid-convergence",
                     dx_values, level_errors)
    save_dx_error_csv(output_dir / "task_1b_error_vs_dx.csv", dx_values, level_errors)

    return {"n_fixed": n, "dx_fixed": dx, "energies": numerical, "exact": exact,
            "errors": errors, "dx_values": dx_values, "level_errors": level_errors}


def finite_well_exact(v0: float = 10.0, a: float = 1.0) -> np.ndarray:
    """Find all negative bound-state energies by solving the even/odd equations in k."""
    kappa_limit = np.sqrt(2.0 * v0)
    roots: list[float] = []

    def add_root(func, left: float, right: float) -> None:
        left = max(left, 1e-10)
        right = min(right, kappa_limit)
        if left >= right:
            return
        try:
            root = brentq(func, left, right, xtol=1e-13, rtol=1e-13)
            if not roots or abs(root - roots[-1]) > 1e-8:
                roots.append(root)
        except ValueError:
            pass

    def kappa(k: float) -> float:
        value = 2.0 * v0 - k**2
        return np.sqrt(max(value, 0.0))

    def even(k: float) -> float:
        return k * np.tan(k * a) - kappa(k)

    def odd(k: float) -> float:
        if abs(np.sin(k * a)) < 1e-14:
            return np.nan
        return -k / np.tan(k * a) - kappa(k)

    # Sampling handles the finite number of parity sectors robustly: sign changes
    # at tangent poles are separated because the k-grid is split at each pole.
    critical = [0.0]
    for pole in np.arange(np.pi / (2 * a), kappa_limit, np.pi / (2 * a)):
        if pole < kappa_limit:
            critical.append(pole)
    critical.append(kappa_limit)
    critical = sorted(set(critical))

    for sector_left, sector_right in zip(critical[:-1], critical[1:]):
        left = sector_left + 1e-8
        right = sector_right - 1e-8
        if left >= right:
            continue
        # Select parity from k*a at the midpoint.  Even: tan positive/negative;
        # odd: the cotangent form is used around odd multiples of pi/2.
        midpoint = 0.5 * (left + right)
        angle = midpoint * a
        reduced = angle % np.pi
        parity = "odd" if reduced > np.pi / 2.0 else "even"
        func = even if parity == "even" else odd
        ks = np.linspace(left, right, 300)
        values = np.array([func(k) for k in ks])
        finite = np.isfinite(values)
        for i in range(len(ks) - 1):
            if finite[i] and finite[i + 1] and values[i] * values[i + 1] < 0:
                add_root(func, float(ks[i]), float(ks[i + 1]))

    roots.sort()
    energies = np.array([k**2 / 2.0 - v0 for k in roots])
    return energies[energies < 0.0]


def task_1c(output_dir: Path) -> dict:
    v0, a = 10.0, 1.0
    left, right = -6.0, 6.0
    n = 1200

    def potential(x):
        return np.where(np.abs(x) < a, -v0, 0.0)

    points, energies, dx = solve_stationary(n, left, right, potential)
    bound_numerical = energies[energies < 0.0]
    exact = finite_well_exact(v0, a)
    count = min(bound_numerical.size, exact.size)

    state_numbers = np.arange(1, count + 1)
    numerical = bound_numerical[:count]
    exact = exact[:count]

    # 1c energy visualization: bound levels together with the finite potential.
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.15, 1]})
    x_plot = np.linspace(left, right, 2000)
    v_plot = potential(x_plot)
    axes[0].plot(x_plot, v_plot, "k-", lw=2)
    axes[0].axhline(0.0, color="0.6", lw=0.8)
    for i, (e_num, e_exact) in enumerate(zip(numerical, exact), start=1):
        axes[0].axhline(e_num, color="C0", lw=1.4, ls="--", alpha=0.9)
        axes[0].axhline(e_exact, color="C1", lw=1.4, ls=":", alpha=0.9)
        axes[0].text(right - 0.2, e_exact, f"n={i}", ha="right", va="bottom", fontsize=8)
    axes[0].set(xlabel="x", ylabel="V(x), E", title="1c: finite well and bound levels")
    axes[0].grid(True, ls=":", alpha=0.5)

    axes[1].plot(state_numbers, numerical, "o-", label="Finite difference")
    axes[1].plot(state_numbers, exact, "x--", label="Analytic transcendental equations")
    axes[1].set(xlabel=r"bound-state $n$", ylabel=r"energy $E_n$", title="1c: numerical vs analytic")
    axes[1].grid(True, ls=":", alpha=0.5)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "task_1c_energies.png", dpi=180)
    plt.close(figure)

    errors = np.abs(numerical - exact)
    plot_error_vs_n(output_dir, "1c", "1c: fixed-grid error versus bound-state number",
                    state_numbers, errors)
    save_energy_error_csv(output_dir / "task_1c_fixed_grid.csv", state_numbers, numerical, exact)

    n_values = np.array([100, 150, 200, 300, 400, 600, 800, 1000, 1200, 1600, 2000])
    dx_values = np.empty(n_values.size)
    level_errors = {n: np.empty(n_values.size) for n in state_numbers}
    for i, grid_n in enumerate(n_values):
        _, grid_energies, grid_dx = solve_stationary(int(grid_n), left, right, potential)
        dx_values[i] = grid_dx
        grid_bound = grid_energies[grid_energies < 0.0]
        for state_n in state_numbers:
            level_errors[state_n][i] = abs(grid_bound[state_n - 1] - exact[state_n - 1])
    plot_error_vs_dx(output_dir, "1c", "1c: finite-well grid-convergence",
                     dx_values, level_errors)
    save_dx_error_csv(output_dir / "task_1c_error_vs_dx.csv", dx_values, level_errors)

    return {"n_fixed": n, "dx_fixed": dx, "numerical": numerical, "exact": exact,
            "errors": errors, "bound_count_numeric": int(bound_numerical.size),
            "bound_count_exact": int(finite_well_exact(v0, a).size),
            "dx_values": dx_values, "level_errors": level_errors}


def generate_outputs(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {"1a": task_1a(output_dir), "1b": task_1b(output_dir), "1c": task_1c(output_dir)}

    with (output_dir / "summary.txt").open("w", encoding="utf-8") as handle:
        for task, result in results.items():
            handle.write(f"Task {task}: fixed grid N={result['n_fixed']}, dx={result['dx_fixed']:.8g}\n")
            if task == "1c":
                handle.write(f"  exact bound states: {result['bound_count_exact']}\n")
                handle.write(f"  numerical bound states: {result['bound_count_numeric']}\n")
                values = zip(result["numerical"], result["exact"], result["errors"])
            else:
                values = zip(result["energies"], result["exact"], result["errors"])
            for n, (script, analytic, error) in enumerate(values, start=(1 if task == "1b" or task == "1c" else 0)):
                handle.write(f"  n={n}: script={script:.12g}, analytic={analytic:.12g}, abs_error={error:.12g}\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "exercise_06_task_1_output",
                        help="Output directory")
    args = parser.parse_args()
    results = generate_outputs(args.output)
    print(f"Generated exercise 1 outputs in {args.output.resolve()}")
    for task, result in results.items():
        print(task, "fixed-grid max error:", float(np.max(result["errors"])))


if __name__ == "__main__":
    main()
