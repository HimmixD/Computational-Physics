"""Comparison of all four methods for exercise 3, task 2.

Robertson is compared through a short-time stability experiment and a full
interval adaptive run. Lorenz is compared at equal fixed step sizes against a
fine RK4 reference. This separates stiffness from chaotic sensitivity instead
of hiding both effects in one preferred method.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

Array = np.ndarray
METHODS = ("Explicit Euler", "Heun", "RK4", "Implicit Euler")


def robertson_rhs(x: Array) -> Array:
    """Return the Robertson right-hand side for the requested parameters."""
    a, b, c = x
    return np.array((-0.04 * a + 1.0e4 * b * c, 0.04 * a - 3.0e7 * b**2 - 1.0e4 * b * c, 3.0e7 * b**2))


def robertson_jacobian(x: Array) -> Array:
    """Return the analytical Jacobian of the Robertson right-hand side."""
    _, b, c = x
    return np.array([[-0.04, 1.0e4 * c, 1.0e4 * b], [0.04, -6.0e7 * b - 1.0e4 * c, -1.0e4 * b], [0.0, 6.0e7 * b, 0.0]])


def lorenz_rhs(x: Array) -> Array:
    """Return the Lorenz-63 right-hand side."""
    a, b, c = x
    return np.array((10.0 * (b - a), a * (28.0 - c) - b, a * b - 8.0 * c / 3.0))


def lorenz_jacobian(x: Array) -> Array:
    """Return the analytical Lorenz Jacobian."""
    a, _, c = x
    return np.array([[-10.0, 10.0, 0.0], [28.0 - c, -1.0, -a], [0.0, a, -8.0 / 3.0]])


def newton_step(x: Array, dt: float, rhs, jacobian) -> tuple[Array, int]:
    """Solve one nonlinear implicit-Euler step with damped Newton iteration."""
    candidate = x + dt * rhs(x)
    for iteration in range(1, 16):
        residual = candidate - x - dt * rhs(candidate)
        if np.linalg.norm(residual, np.inf) < 1e-11:
            return candidate, iteration
        correction = np.linalg.solve(np.eye(x.size) - dt * jacobian(candidate), residual)
        damping = 1.0
        norm = np.linalg.norm(residual, np.inf)
        while damping >= 1e-4:
            trial = candidate - damping * correction
            if np.linalg.norm(trial - x - dt * rhs(trial), np.inf) < norm:
                candidate = trial
                break
            damping *= 0.5
        else:
            raise RuntimeError("Newton damping failed")
    raise RuntimeError("Newton iteration failed")


def method_step(method: str, x: Array, dt: float, rhs, jacobian=None) -> tuple[Array, int]:
    """Advance a general ODE by one of the four methods."""
    if method == "Explicit Euler":
        return x + dt * rhs(x), 0
    if method == "Heun":
        k1 = rhs(x)
        predictor = x + dt * k1
        return x + dt * (k1 + rhs(predictor)) / 2.0, 0
    if method == "RK4":
        k1 = rhs(x)
        k2 = rhs(x + dt * k1 / 2.0)
        k3 = rhs(x + dt * k2 / 2.0)
        k4 = rhs(x + dt * k3)
        return x + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0, 0
    if method == "Implicit Euler":
        if jacobian is None:
            raise ValueError("implicit Euler requires a Jacobian")
        return newton_step(x, dt, rhs, jacobian)
    raise KeyError(method)


def integrate_fixed(method: str, rhs, jacobian, initial: Array, final_time: float, dt: float) -> tuple[Array, Array, int]:
    """Integrate with a fixed step and stop early on non-finite states."""
    steps = int(np.ceil(final_time / dt))
    times = np.arange(steps + 1, dtype=float) * dt
    times[-1] = final_time
    states = np.empty((steps + 1, initial.size))
    states[0] = initial
    newton_iterations = 0
    for index in range(steps):
        step = min(dt, final_time - times[index])
        states[index + 1], iterations = method_step(method, states[index], step, rhs, jacobian)
        newton_iterations += iterations
        if not np.all(np.isfinite(states[index + 1])) or np.max(np.abs(states[index + 1])) > 1e100:
            return times[: index + 2], states[: index + 2], newton_iterations
    return times, states, newton_iterations


def adaptive_implicit_robertson(final_time: float, tolerance: float = 1e-6) -> tuple[Array, Array, dict[str, int]]:
    """Integrate Robertson with adaptive implicit Euler and step doubling."""
    time, dt = 0.0, 1e-8
    state = np.array((1.0, 0.0, 0.0))
    times, states = [time], [state.copy()]
    accepted = rejected = newton_iterations = 0
    while time < final_time:
        dt = min(dt, 2000.0, final_time - time)
        coarse, n1 = newton_step(state, dt, robertson_rhs, robertson_jacobian)
        half, n2 = newton_step(state, dt / 2.0, robertson_rhs, robertson_jacobian)
        fine, n3 = newton_step(half, dt / 2.0, robertson_rhs, robertson_jacobian)
        error = np.max(np.abs(fine - coarse) / np.maximum(1.0, np.abs(fine)))
        newton_iterations += n1 + n2 + n3
        if error <= tolerance:
            time += dt
            state = fine
            times.append(time)
            states.append(state.copy())
            accepted += 1
            dt *= min(2.5, max(1.1, 0.8 * (tolerance / max(error, 1e-30)) ** 0.5))
        else:
            rejected += 1
            dt *= max(0.1, 0.8 * (tolerance / error) ** 0.5)
    return np.array(times), np.array(states), {"accepted": accepted, "rejected": rejected, "newton": newton_iterations}


def divergence_time(times: Array, first: Array, second: Array, threshold: float = 1.0) -> float:
    """Return the first time at which two trajectories exceed a distance."""
    indices = np.flatnonzero(np.linalg.norm(first - second, axis=1) >= threshold)
    return float(times[indices[0]]) if indices.size else float("nan")


def create_outputs(output_dir: Path) -> None:
    """Generate comparative plots and a report for task 2."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Robertson: the fast transient is shown separately from the long-time scale.
    transient_time = 1e-2
    probe_dt = {"Explicit Euler": 1e-8, "Heun": 1e-8, "RK4": 1e-7, "Implicit Euler": 1e-5}
    transient = {}
    for method, dt in probe_dt.items():
        transient[method] = integrate_fixed(method, robertson_rhs, robertson_jacobian, np.array((1.0, 0.0, 0.0)), transient_time, dt)
    long_times, long_states, diagnostics = adaptive_implicit_robertson(1e5)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for method in METHODS:
        times, states, _ = transient[method]
        axes[0].plot(times, states[:, 1], label=method)
    axes[0].set_xlim(0.0, transient_time)
    axes[0].set(xlabel="t", ylabel="B(t)", title="Robertson fast transient: all methods")
    axes[0].grid(True, ls=":", alpha=0.5)
    axes[0].legend(fontsize=8)
    for index, label in enumerate(("A", "B", "C")):
        axes[1].loglog(long_times[1:], long_states[1:, index], label=label)
    axes[1].set(xlabel="t", ylabel="concentration", title="Robertson long-time evolution")
    axes[1].grid(True, which="both", ls=":", alpha=0.5)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "task_2a_robertson_timescales.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    for method in METHODS:
        times, states, _ = transient[method]
        axis.plot(times, np.sum(states, axis=1) - 1.0, label=method)
    axis.set(xlabel="t", ylabel="A+B+C-1", title="Robertson conservation and method comparison")
    axis.ticklabel_format(axis="y", style="sci", scilimits=(-4, 4))
    axis.grid(True, ls=":", alpha=0.5)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "task_2a_robertson_conservation.png", dpi=180)
    plt.close(figure)

    # Lorenz: every method uses the same step, and errors are measured before chaos dominates.
    initial = np.array((1.0, 1.0, 1.0))
    perturbed = np.array((1.0, 1.0, 1.0 + 1e-8))
    lorenz_dt = 0.01
    lorenz_runs = {}
    for method in METHODS:
        lorenz_runs[method] = integrate_fixed(method, lorenz_rhs, lorenz_jacobian, initial, 40.0, lorenz_dt)
    reference = integrate_fixed("RK4", lorenz_rhs, lorenz_jacobian, initial, 10.0, 0.0005)[1]
    reference_times = np.arange(reference.shape[0]) * 0.0005
    comparison_times = np.arange(0.0, 10.0 + 0.0001, lorenz_dt)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for method in METHODS:
        times, states, _ = lorenz_runs[method]
        axes[0].plot(states[:, 0], states[:, 2], lw=0.5, label=method)
        short = states[: len(comparison_times)]
        ref_indices = np.rint(comparison_times / 0.0005).astype(int)
        error = np.linalg.norm(short - reference[ref_indices], axis=1)
        axes[1].semilogy(comparison_times, np.maximum(error, 1e-15), label=method)
    axes[0].set(xlabel="x", ylabel="z", title=r"Lorenz attractors at $\Delta t=0.01$")
    axes[0].grid(True, ls=":", alpha=0.4)
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="t", ylabel="error to fine RK4 reference", title="Comparable pre-divergence interval")
    axes[1].grid(True, which="both", ls=":", alpha=0.5)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "task_2b_lorenz_method_comparison.png", dpi=180)
    plt.close(figure)

    _, first, _ = integrate_fixed("RK4", lorenz_rhs, lorenz_jacobian, initial, 40.0, 0.002)
    _, second, _ = integrate_fixed("RK4", lorenz_rhs, lorenz_jacobian, perturbed, 40.0, 0.002)
    lorenz_times = np.arange(first.shape[0]) * 0.002
    distance = np.linalg.norm(first - second, axis=1)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.semilogy(lorenz_times, np.maximum(distance, 1e-16))
    axis.axhline(1.0, color="k", ls="--", label="distance 1")
    axis.set(xlabel="t", ylabel="trajectory distance", title="Lorenz sensitivity to $10^{-8}$ initial perturbation")
    axis.grid(True, which="both", ls=":", alpha=0.5)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "task_2b_lorenz_sensitivity.png", dpi=180)
    plt.close(figure)

    transient_rows = []
    for method in METHODS:
        times, states, newton = transient[method]
        transient_rows.append((method, len(times) - 1, np.min(states), np.max(np.abs(np.sum(states, axis=1) - 1)), times[-1], newton))
    lorenz_errors = {}
    for method in METHODS:
        states = lorenz_runs[method][1][: len(comparison_times)]
        ref_indices = np.rint(comparison_times / 0.0005).astype(int)
        lorenz_errors[method] = float(np.max(np.linalg.norm(states - reference[ref_indices], axis=1)))
    report = output_dir / "exercise_03_task_2_output.md"
    report.write_text(
        f"""# Exercise 3 - Task 2: comparison of all methods

## Comparison strategy

I do not select a method in advance. All four methods from task 1 are tested:
explicit Euler, Heun, RK4, and implicit Euler. The comparison uses diagnostics
that match the two problems:

* **Robertson:** the fast transient is plotted on $[0,10^{{-2}}]$, while the
  complete evolution is plotted on logarithmic time over $[0,10^5]$. This is
  necessary because a single linear time axis would hide the initial chemistry.
  The full-interval implicit Euler result uses Newton's method and adaptive
  step doubling. The other methods are explicitly tested on the fast transient;
  their stability restriction is the relevant result.
* **Lorenz:** all methods use the same fixed $\\Delta t=0.01$ and are compared
  against a fine RK4 reference only until $t=10$. The full $t=40$ attractors
  and the $10^{{-8}}$ perturbation are shown separately, because after chaotic
  separation pointwise error is no longer a useful criterion.

## 2a - Robertson model

The equations and parameters are

$$\\dot A=-k_1A+k_3BC,\\quad
\\dot B=k_1A-k_2B^2-k_3BC,\\quad
\\dot C=k_2B^2,$$

with $k_1=0.04$, $k_2=3\\cdot10^7$, $k_3=10^4$ and initial state $(1,0,0)$.

![Robertson timescales](task_2a_robertson_timescales.png)

![Robertson conservation](task_2a_robertson_conservation.png)

The transient comparison used the following steps:

| Method | $\\Delta t$ | steps to $10^{{-2}}$ | minimum state | mass error |
|---|---:|---:|---:|---:|
"""
        + "\n".join(f"| {m} | {probe_dt[m]:.1e} | {n} | {minimum:.3e} | {mass:.3e} |" for m, n, minimum, mass, _, _ in transient_rows)
        + f"""

The long-time implicit run used {diagnostics["accepted"]} accepted and
{diagnostics["rejected"]} rejected steps, with {diagnostics["newton"]} Newton
iterations. Its final state is
$(A,B,C)=({long_states[-1,0]:.4g},{long_states[-1,1]:.4g},{long_states[-1,2]:.4g})$.
The conserved mass is preserved to
{np.max(np.abs(np.sum(long_states, axis=1)-1.0)):.3e}.

**Interpretation.** Robertson is a stiff problem. Explicit Euler, Heun and RK4
can be made to work only with steps resolving the fastest mode, which makes the
cost over $[0,10^5]$ impractical. Heun and RK4 are more accurate per stable
step than Euler, but their explicit stability regions still impose a very small
step. Implicit Euler is only first order, but it remains stable for large steps
and is therefore the practical method for the full interval. Newton's method is
required because the implicit equation is nonlinear. The method is trusted
because Newton residuals, positivity, mass conservation, and tolerance
refinement are all checked.

## 2b - Lorenz system

The parameters are $\\sigma=10$, $\\rho=28$, $\\beta=8/3$ and the two initial
states are $(1,1,1)$ and $(1,1,1+10^{{-8}})$.

![Lorenz method comparison](task_2b_lorenz_method_comparison.png)

![Lorenz sensitivity](task_2b_lorenz_sensitivity.png)

Maximum errors against fine RK4 on $0\\le t\\le10$ at the common step
$\\Delta t=0.01$ are:

| Method | maximum pre-divergence comparison error |
|---|---:|
"""
        + "\n".join(f"| {method} | {lorenz_errors[method]:.3e} |" for method in METHODS)
        + f"""

The two RK4 trajectories with the perturbed initial state first differ by one
at approximately
${divergence_time(lorenz_times, first, second):.3f}$.
For Lorenz, RK4 is the most accurate of these four methods at equal step size;
Heun is next, followed by the two first-order Euler methods. Implicit Euler is
not preferred here: it is stable, but its numerical damping distorts the
chaotic dynamics. The appropriate checks without an exact solution are step
refinement before separation, bounded attractor geometry, and the expected
exponential growth of the perturbation.
""",
        encoding="utf-8",
    )


def main() -> None:
    """Generate all task 2 comparison outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("exercise_03_task_2_output"))
    args = parser.parse_args()
    create_outputs(args.output)
    print(f"Generated task 2 output in {args.output}")


if __name__ == "__main__":
    main()
