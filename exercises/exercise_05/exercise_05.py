"""Numerical experiments for Exercise 5, task 1 (finite differences).

Run this file from the repository root with ``python exercises/exercise_05/
exercise_05.py``.  All figures and CSV files are written to
``exercise_05_output``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import eye, kron, lil_matrix
from scipy.sparse.linalg import spsolve


def grid(n: int) -> tuple[np.ndarray, float]:
    """Return the N interior points and their spacing on [0, 1]."""
    if n < 1:
        raise ValueError("n must be positive")
    dx = 1.0 / (n + 1)
    return np.arange(1, n + 1, dtype=float) * dx, dx


def finite_difference_weights(nodes: np.ndarray, x0: float, derivative: int) -> np.ndarray:
    """Return polynomial-interpolation weights for a derivative at ``x0``."""
    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim != 1 or nodes.size <= derivative:
        raise ValueError("not enough distinct nodes for requested derivative")
    if np.unique(nodes).size != nodes.size:
        raise ValueError("finite-difference nodes must be distinct")
    powers = np.arange(nodes.size)
    matrix = (nodes[:, None] - x0) ** powers[None, :] / np.array(
        [math.factorial(int(power)) for power in powers]
    )
    return np.linalg.solve(matrix.T, np.eye(nodes.size)[derivative])


def _stencil_indices(index: int, total_points: int, width: int) -> np.ndarray:
    """Choose a contiguous stencil, shifted at the two boundaries."""
    start = min(max(index - width // 2, 0), total_points - width)
    return np.arange(start, start + width)


def second_derivative_matrix(
    n: int, dx: float | None = None, order: int = 2
) -> "lil_matrix[np.float64]":
    """Build the sparse ``u -> u''`` matrix on a Dirichlet grid.

    The boundary values are not unknowns and are assumed to be zero.  For
    order 2, 4, and 6, centered stencils are used in the interior and
    wider one-sided stencils close to the boundary to retain the order.
    """
    if order not in (2, 4, 6):
        raise ValueError("order must be 2, 4, or 6")
    points, spacing = grid(n)
    if dx is not None and not np.isclose(dx, spacing):
        raise ValueError("dx is inconsistent with n on [0, 1]")
    all_points = np.concatenate(([0.0], points, [1.0]))
    matrix = lil_matrix((n, n), dtype=float)
    for row, x0 in enumerate(points, start=1):
        centered_width = order + 1
        boundary_width = order + 2
        width = boundary_width if row <= order // 2 or row > n - order // 2 else centered_width
        indices = _stencil_indices(row, all_points.size, width)
        normalized_nodes = (all_points[indices] - x0) / spacing
        weights = finite_difference_weights(normalized_nodes, 0.0, 2) / spacing**2
        for full_index, weight in zip(indices, weights):
            if 0 < full_index < n + 1:
                matrix[row - 1, full_index - 1] += weight
    return matrix


def dirichlet_operator(n: int, order: int = 2):
    """Return the sparse positive operator ``A = -d²/dx²``."""
    return -second_derivative_matrix(n, order=order).tocsr()


def solve_dirichlet(n: int, order: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Solve ``-u'' = pi² sin(pi x)`` with homogeneous Dirichlet data."""
    points, _ = grid(n)
    solution = spsolve(dirichlet_operator(n, order), np.pi**2 * np.sin(np.pi * points))
    return points, solution


def neumann_operator(n: int):
    """Return ``A`` for u(0)=0 and the second-order condition u'(1)=0.

    A second-order backward derivative at the right boundary eliminates the
    boundary value from the equation at the last interior point.
    """
    points, dx = grid(n)
    operator = dirichlet_operator(n).tolil()
    operator[-1, :] = 0.0
    operator[-1, -1] = 2.0 / (3.0 * dx**2)
    if n > 1:
        operator[-1, -2] = -2.0 / (3.0 * dx**2)
    return points, operator.tocsr()


def solve_neumann(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Solve the mixed-boundary problem from task 1d."""
    points, operator = neumann_operator(n)
    rhs = (np.pi**2 / 4.0) * np.sin(np.pi * points / 2.0)
    return points, spsolve(operator, rhs)


def solve_2d(n: int) -> tuple[np.ndarray, float, float]:
    """Solve task 1e and return the grid, infinity error, and runtime."""
    points, _ = grid(n)
    one_d = dirichlet_operator(n)
    operator = kron(eye(n, format="csr"), one_d) + kron(one_d, eye(n, format="csr"))
    rhs_grid = 2.0 * np.pi**2 * np.sin(np.pi * points[:, None]) * np.sin(np.pi * points[None, :])
    start = time.perf_counter()
    numerical = spsolve(operator, rhs_grid.ravel(order="C"))
    runtime = time.perf_counter() - start
    exact = rhs_grid / (2.0 * np.pi**2)
    return points, float(np.max(np.abs(numerical - exact.ravel(order="C")))), runtime


def _slope(spacings: np.ndarray, errors: np.ndarray, tail: int = 4) -> float:
    count = min(tail, errors.size)
    return float(np.polyfit(np.log(spacings[-count:]), np.log(errors[-count:]), 1)[0])


def _slope_before_roundoff(spacings: np.ndarray, errors: np.ndarray) -> float:
    """Fit the last few decreasing-error points before the round-off floor."""
    minimum = int(np.argmin(errors))
    start = max(0, minimum - 3)
    return _slope(spacings[start : minimum + 1], errors[start : minimum + 1], tail=4)


def generate_outputs(output_dir: Path) -> None:
    """Generate all task-1 numerical data and figures."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1a/1b: the tridiagonal matrix, convergence, and round-off turn-around.
    n_check = 7
    operator = dirichlet_operator(n_check)
    print(
        f"1a: N={n_check}, shape={operator.shape}, nnz={operator.nnz}, "
        f"symmetric={np.allclose(operator.toarray(), operator.toarray().T)}, "
        f"min(eigenvalue)={np.linalg.eigvalsh(operator.toarray()).min():.6g}"
    )
    n_values = np.array([2**k - 1 for k in range(2, 21)] + [1_000_000])
    spacings = 1.0 / (n_values + 1)
    errors = []
    for n in n_values:
        points, numerical = solve_dirichlet(int(n))
        errors.append(np.max(np.abs(numerical - np.sin(np.pi * points))))
    errors = np.array(errors)
    slope = _slope(spacings[2:12], errors[2:12], tail=8)
    np.savetxt(
        output_dir / "task_1b_convergence.csv",
        np.column_stack((n_values, spacings, errors)),
        delimiter=",",
        header="N,dx,max_error",
        comments="",
    )
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.loglog(spacings, errors, "o-", label=f"central 3-point (slope {slope:.2f})")
    axis.loglog(spacings, np.pi**3 * spacings**2 / 24.0, "--", label=r"$O(\Delta x^2)$")
    axis.set(xlabel=r"$\Delta x$", ylabel=r"$\max_j|u_j-u(x_j)|$", title="1b: Dirichlet convergence")
    axis.grid(True, which="both", ls=":", alpha=0.5)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "task_1b_convergence.png", dpi=180)
    plt.close(figure)

    # 1c: higher-order stencils, including their boundary treatment.
    comparison_n = np.array([7, 15, 31, 63, 127, 255, 511, 1023])
    comparison_dx = 1.0 / (comparison_n + 1)
    comparison_errors = {}
    for order in (2, 4, 6):
        values = []
        for n in comparison_n:
            points, numerical = solve_dirichlet(int(n), order)
            values.append(np.max(np.abs(numerical - np.sin(np.pi * points))))
        comparison_errors[order] = np.array(values)
    figure, axis = plt.subplots(figsize=(7, 5))
    rows = []
    for order, values in comparison_errors.items():
        measured = _slope_before_roundoff(comparison_dx, values)
        rows.append((order, measured))
        axis.loglog(comparison_dx, values, "o-", label=f"order {order} (slope {measured:.2f})")
    axis.set(xlabel=r"$\Delta x$", ylabel="maximum error", title="1c: higher-order stencils")
    axis.grid(True, which="both", ls=":", alpha=0.5)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "task_1c_stencils.png", dpi=180)
    plt.close(figure)
    with (output_dir / "task_1c_orders.csv").open("w", encoding="utf-8") as file:
        file.write("stencil_order,measured_slope\n")
        file.writelines(f"{order},{measured:.8f}\n" for order, measured in rows)

    # 1d: mixed boundary condition.
    neumann_n = np.array([7, 15, 31, 63, 127, 255, 511, 1023])
    neumann_dx = 1.0 / (neumann_n + 1)
    neumann_errors = []
    for n in neumann_n:
        points, numerical = solve_neumann(int(n))
        neumann_errors.append(np.max(np.abs(numerical - np.sin(np.pi * points / 2.0))))
    neumann_errors = np.array(neumann_errors)
    neumann_slope = _slope(neumann_dx, neumann_errors)
    np.savetxt(
        output_dir / "task_1d_neumann.csv",
        np.column_stack((neumann_n, neumann_dx, neumann_errors)),
        delimiter=",",
        header="N,dx,max_error",
        comments="",
    )
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.loglog(neumann_dx, neumann_errors, "o-", label=f"ghost-point Neumann (slope {neumann_slope:.2f})")
    axis.set(xlabel=r"$\Delta x$", ylabel="maximum error", title=r"1d: $u(0)=0,\ u'(1)=0$")
    axis.grid(True, which="both", ls=":", alpha=0.5)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "task_1d_neumann.png", dpi=180)
    plt.close(figure)

    # 1e: tensor-product Laplacian and scaling.
    two_d_n = np.array([7, 15, 31, 63, 95])
    two_d = np.array([solve_2d(int(n))[1:] for n in two_d_n])
    np.savetxt(
        output_dir / "task_1e_2d_scaling.csv",
        np.column_stack((two_d_n, two_d)),
        delimiter=",",
        header="N,max_error,solve_time_seconds",
        comments="",
    )
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].loglog(two_d_n + 1, two_d[:, 0], "o-", label="error")
    axes[0].loglog(two_d_n + 1, two_d[0, 0] * ((two_d_n[0] + 1) / (two_d_n + 1)) ** 2, "--", label=r"$O(\Delta x^2)$")
    axes[0].set(xlabel=r"$1/\Delta x=N+1$", ylabel="maximum error", title="1e: 2D convergence")
    axes[1].loglog(two_d_n**2, two_d[:, 1], "o-")
    axes[1].set(xlabel="number of unknowns", ylabel="solve time [s]", title="1e: sparse solve scaling")
    for axis in axes:
        axis.grid(True, which="both", ls=":", alpha=0.5)
        axis.legend() if axis is axes[0] else None
    figure.tight_layout()
    figure.savefig(output_dir / "task_1e_2d_scaling.png", dpi=180)
    plt.close(figure)

    print(f"1b slope before round-off: {slope:.3f}")
    print("1c slopes: " + ", ".join(f"order {order}: {value:.3f}" for order, value in rows))
    print(f"1d slope: {neumann_slope:.3f}")
    print("Outputs written to", output_dir)


def stability_function(method: str, z: np.ndarray | float) -> np.ndarray | float:
    """Return the scalar stability function for a one-step method."""
    if method == "Explicit Euler":
        return 1.0 + z
    if method == "Heun":
        return 1.0 + z + z**2 / 2.0
    if method == "RK4":
        return 1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0
    if method == "Implicit Euler":
        return 1.0 / (1.0 - z)
    if method == "Crank-Nicolson":
        return (1.0 + z / 2.0) / (1.0 - z / 2.0)
    raise KeyError(f"unknown method: {method}")


TIME_METHODS = ("Explicit Euler", "Implicit Euler", "Crank-Nicolson")
STABILITY_METHODS = ("Explicit Euler", "Heun", "RK4", "Implicit Euler", "Crank-Nicolson")


def heat_step(method: str, state: np.ndarray, operator, dt: float) -> np.ndarray:
    """Advance ``u_t=-Au`` by one step."""
    if method == "Explicit Euler":
        return state - dt * (operator @ state)
    if method == "Implicit Euler":
        return spsolve(eye(operator.shape[0], format="csr") + dt * operator, state)
    if method == "Crank-Nicolson":
        identity = eye(operator.shape[0], format="csr")
        return spsolve(identity + 0.5 * dt * operator, (identity - 0.5 * dt * operator) @ state)
    raise KeyError(f"unknown heat method: {method}")


def integrate_heat(method: str, n: int, initial: np.ndarray, final_time: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the semi-discrete heat equation with an exact final step."""
    if final_time <= 0.0 or dt <= 0.0:
        raise ValueError("final_time and dt must be positive")
    points, _ = grid(n)
    operator = dirichlet_operator(n)
    times = np.arange(0.0, final_time, dt)
    times = np.append(times, final_time) if times.size == 0 or times[-1] < final_time else times
    states = np.empty((times.size, n), dtype=float)
    states[0] = np.asarray(initial, dtype=float)
    if states[0].shape != (n,):
        raise ValueError("initial state must have shape (n,)")
    identity = eye(n, format="csc")
    implicit_matrix = None
    if method == "Implicit Euler":
        implicit_matrix = (identity + dt * operator).tocsc()
    elif method == "Crank-Nicolson":
        implicit_matrix = (identity + 0.5 * dt * operator).tocsc()
    from scipy.sparse.linalg import factorized

    solve_implicit = factorized(implicit_matrix) if implicit_matrix is not None else None
    for index in range(times.size - 1):
        step = times[index + 1] - times[index]
        if method == "Explicit Euler":
            states[index + 1] = states[index] - step * (operator @ states[index])
        elif method == "Implicit Euler":
            if step == dt:
                states[index + 1] = solve_implicit(states[index])
            else:
                states[index + 1] = spsolve(identity + step * operator, states[index])
        elif method == "Crank-Nicolson":
            if step == dt:
                rhs = (identity - 0.5 * dt * operator) @ states[index]
                states[index + 1] = solve_implicit(rhs)
            else:
                left = identity + 0.5 * step * operator
                right = (identity - 0.5 * step * operator) @ states[index]
                states[index + 1] = spsolve(left, right)
        else:
            raise KeyError(f"unknown heat method: {method}")
    return times, states


def task_2_accuracy(output_dir: Path) -> dict[str, float]:
    """Generate method-of-lines accuracy plots and return temporal slopes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    final_time = 0.1
    n = 50
    points, _ = grid(n)
    initial = np.sin(np.pi * points)
    dt_values = np.array([1e-4, 5e-5, 2.5e-5, 1.25e-5, 6.25e-6])
    temporal = {}
    for method in TIME_METHODS:
        pde_errors = []
        ode_errors = []
        for dt in dt_values:
            _, states = integrate_heat(method, n, initial, final_time, dt)
            numerical = states[-1]
            lambda_1 = 4.0 / (1.0 / (n + 1))**2 * np.sin(np.pi / (2 * (n + 1)))**2
            pde_errors.append(np.max(np.abs(numerical - np.exp(-np.pi**2 * final_time) * initial)))
            ode_errors.append(np.max(np.abs(numerical - np.exp(-lambda_1 * final_time) * initial)))
        temporal[method] = (np.array(pde_errors), np.array(ode_errors))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    temporal_slopes = {}
    for method, (pde_errors, ode_errors) in temporal.items():
        axes[0].loglog(dt_values, pde_errors, "o-", label=method)
        axes[1].loglog(dt_values, ode_errors, "o-", label=method)
        temporal_slopes[method] = float(np.polyfit(np.log(dt_values), np.log(ode_errors), 1)[0])
    axes[0].set(title="2b: temporal error vs exact PDE", xlabel=r"$\Delta t$", ylabel="maximum error")
    axes[1].set(title="2b: temporal error vs exact ODE", xlabel=r"$\Delta t$", ylabel="maximum error")
    for axis in axes:
        axis.grid(True, which="both", ls=":", alpha=0.5)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "task_2b_temporal_accuracy.png", dpi=180)
    plt.close(figure)

    spatial_n = np.array([10, 20, 40, 80, 160])
    spatial_dt = 1e-4
    spatial_pde = []
    spatial_ode = []
    for n_value in spatial_n:
        points, _ = grid(int(n_value))
        initial = np.sin(np.pi * points)
        _, states = integrate_heat("Crank-Nicolson", int(n_value), initial, final_time, spatial_dt)
        lambda_1 = 4.0 * (n_value + 1)**2 * np.sin(np.pi / (2 * (n_value + 1)))**2
        spatial_pde.append(np.max(np.abs(states[-1] - np.exp(-np.pi**2 * final_time) * initial)))
        spatial_ode.append(np.max(np.abs(states[-1] - np.exp(-lambda_1 * final_time) * initial)))
    spatial_pde = np.array(spatial_pde)
    spatial_ode = np.array(spatial_ode)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.loglog(1.0 / (spatial_n + 1), spatial_pde, "o-", label="exact PDE")
    axis.loglog(1.0 / (spatial_n + 1), spatial_ode, "o-", label="exact ODE")
    axis.set(title="2b: spatial accuracy (Crank-Nicolson)", xlabel=r"$\Delta x$", ylabel="maximum error")
    axis.grid(True, which="both", ls=":", alpha=0.5)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "task_2b_spatial_accuracy.png", dpi=180)
    plt.close(figure)
    np.savetxt(
        output_dir / "task_2b_accuracy.csv",
        np.column_stack((dt_values, *[temporal[name][0] for name in TIME_METHODS], *[temporal[name][1] for name in TIME_METHODS])),
        delimiter=",",
        header="dt," + ",".join(f"{name}_pde" for name in TIME_METHODS) + "," + ",".join(f"{name}_ode" for name in TIME_METHODS),
        comments="",
    )
    np.savetxt(
        output_dir / "task_2b_spatial.csv",
        np.column_stack((spatial_n, spatial_pde, spatial_ode)),
        delimiter=",",
        header="N,pde_error,ode_error",
        comments="",
    )
    return temporal_slopes


def task_2_stability(output_dir: Path) -> None:
    """Generate eigenmode, stability-region, and critical-step results."""
    n_values = (10, 20, 40, 80)
    critical_rows = []
    for n in n_values:
        eigenvalues = 4.0 * (n + 1)**2 * np.sin(np.arange(1, n + 1) * np.pi / (2 * (n + 1)))**2
        critical_rows.append((n, eigenvalues[-1], 2.0 / eigenvalues[-1], 2.0 / eigenvalues[-1], 2.7852935634 / eigenvalues[-1]))
    np.savetxt(
        output_dir / "task_2c_critical_dt.csv",
        np.array(critical_rows),
        delimiter=",",
        header="N,lambda_max,dt_explicit_euler,dt_heun,dt_rk4",
        comments="",
    )

    n = 40
    points, _ = grid(n)
    operator = dirichlet_operator(n).toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(operator)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for mode in range(3):
        axes[0].plot(points, eigenvectors[:, mode] * np.sign(eigenvectors[0, mode]), label=f"k={mode + 1}")
    axes[0].set(title="2c: first eigenvectors", xlabel="x", ylabel=r"$v_k(x)$")
    axes[0].grid(True, ls=":", alpha=0.5)
    axes[0].legend()
    dt_mark = 0.8 * 2.0 / eigenvalues[-1]
    axes[1].plot(-eigenvalues * dt_mark, np.zeros_like(eigenvalues), "o", ms=3)
    axes[1].axhline(0.0, color="k", lw=0.7)
    axes[1].set(title=fr"Eigenvalue locations ($N={n}$, $\Delta t={dt_mark:.2e}$)", xlabel=r"$z_k=-\lambda_k\Delta t$", yticks=[])
    axes[1].grid(True, axis="x", ls=":", alpha=0.5)
    figure.tight_layout()
    figure.savefig(output_dir / "task_2c_eigenvectors_and_modes.png", dpi=180)
    plt.close(figure)

    real = np.linspace(-5.0, 2.0, 500)
    imag = np.linspace(-4.0, 4.0, 500)
    real_grid, imag_grid = np.meshgrid(real, imag)
    z = real_grid + 1j * imag_grid
    figure, axes = plt.subplots(2, 3, figsize=(13, 8))
    for axis, method in zip(axes.flat, STABILITY_METHODS):
        magnitude = np.abs(stability_function(method, z))
        axis.contourf(real_grid, imag_grid, magnitude <= 1.0, levels=[0.5, 1.5], alpha=0.35, cmap="Blues")
        axis.contour(real_grid, imag_grid, magnitude, levels=[1.0], colors="k")
        axis.plot(-eigenvalues * dt_mark, np.zeros_like(eigenvalues), "r.", ms=2)
        axis.set_title(method)
        axis.grid(True, ls=":", alpha=0.4)
    axes[1, 2].axis("off")
    for axis in axes[0, :]:
        axis.set_xlabel("Re(z)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Im(z)")
    figure.suptitle("2c: stability regions and semi-discrete heat eigenvalues")
    figure.tight_layout()
    figure.savefig(output_dir / "task_2c_stability_regions.png", dpi=180)
    plt.close(figure)

    # Just above the explicit-Euler limit: the highest mode grows and alternates.
    initial = np.sin(20 * np.pi * points)
    dt_above = 1.05 * 2.0 / eigenvalues[-1]
    _, unstable_states = integrate_heat("Explicit Euler", n, initial, 20.0 * dt_above, dt_above)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(points, initial, label="initial highest mode")
    axis.plot(points, unstable_states[-1], label="after 20 steps")
    axis.set(title="2c: just above the explicit-Euler stability limit", xlabel="x", ylabel="u")
    axis.grid(True, ls=":", alpha=0.5)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "task_2c_above_critical.png", dpi=180)
    plt.close(figure)


def task_2_stiffness(output_dir: Path) -> None:
    """Generate the non-smooth-data and stiff-mode experiments."""
    n = 99
    points, _ = grid(n)
    initial = np.where((points > 0.25) & (points < 0.75), 1.0, 0.0)
    final_time = 0.5
    dt = 0.01
    times_ie, states_ie = integrate_heat("Implicit Euler", n, initial, final_time, dt)
    times_cn, states_cn = integrate_heat("Crank-Nicolson", n, initial, final_time, dt)
    operator = dirichlet_operator(n).toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(operator)
    coefficients_ie = states_ie @ eigenvectors
    coefficients_cn = states_cn @ eigenvectors
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    z = np.linspace(-1000.0, 0.0, 2000)
    axes[0].plot(z, np.exp(z), label=r"$e^z$")
    axes[0].plot(z, np.abs(stability_function("Implicit Euler", z)), label="implicit Euler")
    axes[0].plot(z, np.abs(stability_function("Crank-Nicolson", z)), label="Crank-Nicolson")
    axes[0].set(title=r"2d: amplification factors, $z\in[-1000,0]$", xlabel="z", ylabel=r"$|R(z)|$")
    axes[0].grid(True, ls=":", alpha=0.5)
    axes[0].legend()
    axes[1].semilogy(times_ie, np.maximum(np.abs(coefficients_ie[:, -1]), 1e-16), label="implicit Euler")
    axes[1].semilogy(times_cn, np.maximum(np.abs(coefficients_cn[:, -1]), 1e-16), label="Crank-Nicolson")
    axes[1].set(title="2d: highest-mode amplitude", xlabel="t", ylabel=r"$|a_N(t)|$")
    axes[1].grid(True, which="both", ls=":", alpha=0.5)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "task_2d_stiffness_and_modes.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for state, time_value in zip(states_ie[::10], times_ie[::10]):
        axes[0].plot(points, state, alpha=0.35, label=f"t={time_value:.2f}" if time_value in (0.0, 0.1, 0.5) else None)
    for state, time_value in zip(states_cn[::10], times_cn[::10]):
        axes[1].plot(points, state, alpha=0.35, label=f"t={time_value:.2f}" if time_value in (0.0, 0.1, 0.5) else None)
    axes[0].set_title("Implicit Euler")
    axes[1].set_title("Crank-Nicolson")
    for axis in axes:
        axis.set(xlabel="x", ylabel="u(x,t)")
        axis.grid(True, ls=":", alpha=0.5)
        axis.legend()
    figure.suptitle(r"2d: non-smooth initial data ($N=99$, $\Delta t=0.01$)")
    figure.tight_layout()
    figure.savefig(output_dir / "task_2d_step_solution.png", dpi=180)
    plt.close(figure)

    explicit_dt = 0.5 / (n + 1)**2
    np.savetxt(
        output_dir / "task_2d_explicit_steps.csv",
        np.array([[n, explicit_dt, final_time / explicit_dt, 0.1 / explicit_dt]]),
        delimiter=",",
        header="N,stable_dt_max,steps_to_t_0.5,steps_to_t_0.1",
        comments="",
    )


def generate_task_2_outputs(output_dir: Path) -> None:
    """Generate all numerical results for task 2."""
    output_dir.mkdir(parents=True, exist_ok=True)
    slopes = task_2_accuracy(output_dir)
    task_2_stability(output_dir)
    task_2_stiffness(output_dir)
    print("2b temporal slopes vs exact ODE: " + ", ".join(f"{name}: {value:.3f}" for name, value in slopes.items()))
    print("Task 2 outputs written to", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("exercise_05_task_1_output"),
    )
    parser.add_argument(
        "--task-2-output-dir",
        type=Path,
        default=Path(__file__).with_name("exercise_05_task_2_output"),
    )
    args = parser.parse_args()
    generate_outputs(args.output_dir)
    generate_task_2_outputs(args.task_2_output_dir)


if __name__ == "__main__":
    main()
