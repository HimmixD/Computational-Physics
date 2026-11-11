"""Exercise 4, tasks 1a--1d: numerical methods for the lunar orbit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
try:
    from exercises.ode_methods import (
        explicit_euler,
        heun,
        implicit_euler,
        integrate_first_order,
        integrate_second_order,
        symplectic_euler,
        velocity_verlet,
    )
except ModuleNotFoundError:
    from ode_methods import (
        explicit_euler,
        heun,
        implicit_euler,
        integrate_first_order,
        integrate_second_order,
        symplectic_euler,
        velocity_verlet,
    )

G = 6.674e-11
EARTH_MASS = 5.972e24
MU = G * EARTH_MASS
MOON_RADIUS = 384_400e3
DAY = 86400.0
DT = 6.0 * 3600.0
METHODS = ("Explicit Euler", "Implicit Euler", "Heun", "Symplectic Euler", "Velocity Verlet")


def circular_initial_conditions(radius: float = MOON_RADIUS) -> tuple[np.ndarray, np.ndarray]:
    """Return position and velocity for a counter-clockwise circular orbit."""
    return np.array([radius, 0.0]), np.array([0.0, np.sqrt(MU / radius)])


def orbital_period(radius: float = MOON_RADIUS) -> float:
    """Return the Kepler period in seconds."""
    return 2.0 * np.pi * np.sqrt(radius**3 / MU)


def acceleration(_time: float, position: np.ndarray) -> np.ndarray:
    """Earth's point-mass gravitational acceleration."""
    radius = np.linalg.norm(position)
    if radius == 0.0:
        raise ValueError("gravitational acceleration is undefined at r=0")
    return -MU * position / radius**3


def orbit_rhs(_time: float, state: np.ndarray) -> np.ndarray:
    position, velocity = state[:2], state[2:]
    return np.concatenate((velocity, acceleration(0.0, position)))


def orbit_jacobian(_time: float, state: np.ndarray) -> np.ndarray:
    position = state[:2]
    radius = np.linalg.norm(position)
    if radius == 0.0:
        raise ValueError("gravitational Jacobian is undefined at r=0")
    gravity_gradient = MU * (3.0 * np.outer(position, position) / radius**5 - np.eye(2) / radius**3)
    return np.block([[np.zeros((2, 2)), np.eye(2)], [gravity_gradient, np.zeros((2, 2))]])


def integrate_orbit(method: str, final_time: float, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    position, velocity = circular_initial_conditions()
    if method == "Implicit Euler":
        initial = np.concatenate((position, velocity))
        times = np.arange(int(np.floor(final_time / dt)) + 1, dtype=float) * dt
        if times[-1] < final_time:
            times = np.append(times, final_time)
        states = np.empty((times.size, 4))
        states[0] = initial
        iterations = 0
        for index in range(times.size - 1):
            step = times[index + 1] - times[index]
            try:
                states[index + 1], count = implicit_euler(
                    orbit_rhs,
                    orbit_jacobian,
                    times[index],
                    states[index],
                    step,
                )
            except RuntimeError as error:
                warnings.warn(
                    f"Implicit Euler stopped at t={times[index] / DAY:.2f} d: {error}",
                    RuntimeWarning,
                )
                states[index + 1 :] = np.nan
                break
            iterations += count
        return times, states[:, :2], states[:, 2:], iterations
    if method in {"Explicit Euler", "Heun"}:
        stepper = explicit_euler if method == "Explicit Euler" else heun
        times, states = integrate_first_order(
            stepper, orbit_rhs, np.concatenate((position, velocity)), final_time, dt
        )
        return times, states[:, :2], states[:, 2:], 0
    stepper = symplectic_euler if method == "Symplectic Euler" else velocity_verlet
    times, positions, velocities = integrate_second_order(
        stepper, acceleration, position, velocity, final_time, dt
    )
    return times, positions, velocities, 0


def energy(position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    return 0.5 * np.sum(velocity**2, axis=-1) - MU / np.linalg.norm(position, axis=-1)


def angular_momentum(position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    return position[:, 0] * velocity[:, 1] - position[:, 1] * velocity[:, 0]


def generate_outputs(output_dir: Path) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    period = orbital_period()
    position0, velocity0 = circular_initial_conditions()
    np.savetxt(
        output_dir / "initial_conditions.txt",
        np.array([position0, velocity0]),
        header="rows: r(0) [m], v(0) [m/s]",
    )

    # 1b: ten-orbit trajectories.
    figure, axis = plt.subplots(figsize=(7, 7))
    implicit_iterations = 0
    for method in METHODS:
        _, positions, _, iterations = integrate_orbit(method, 10.0 * period, DT)
        if method == "Implicit Euler":
            implicit_iterations = iterations
        axis.plot(positions[:, 0] / MOON_RADIUS, positions[:, 1] / MOON_RADIUS, label=method)
    axis.plot(0.0, 0.0, "ko", label="Earth")
    axis.set_aspect("equal")
    axis.set(xlabel=r"$x/r_M$", ylabel=r"$y/r_M$", title=r"1b: ten lunar orbits, $\Delta t=6$ h")
    axis.grid(True, ls=":", alpha=0.5)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "task_1b_orbits.png", dpi=180)
    plt.close(figure)

    # 1c: invariants over one hundred orbits.
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for method in METHODS:
        times, positions, velocities, _ = integrate_orbit(method, 100.0 * period, DT)
        relative_energy = np.abs((energy(positions, velocities) - energy(positions[:1], velocities[:1])[0]) / energy(positions[:1], velocities[:1])[0])
        relative_angular_momentum = np.abs((angular_momentum(positions, velocities) - angular_momentum(positions[:1], velocities[:1])[0]) / angular_momentum(positions[:1], velocities[:1])[0])
        axes[0].semilogy(times / period, np.maximum(relative_energy, 1e-18), label=method)
        axes[1].semilogy(times / period, np.maximum(relative_angular_momentum, 1e-18), label=method)
    axes[0].set(xlabel="orbits", ylabel=r"$|E-E_0|/|E_0|$", title="1c: relative energy error")
    axes[1].set(xlabel="orbits", ylabel=r"$|L-L_0|/|L_0|$", title="1c: relative angular-momentum error")
    for axis in axes:
        axis.grid(True, which="both", ls=":", alpha=0.5)
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output_dir / "task_1c_invariants.png", dpi=180)
    plt.close(figure)

    # 1d: error at half a period and observed slopes.
    step_counts = np.array([200, 400, 800, 1600, 3200, 6400])
    errors = {method: [] for method in METHODS}
    for count in step_counts:
        dt = period / count
        for method in METHODS:
            _, positions, _, _ = integrate_orbit(method, period / 2.0, dt)
            errors[method].append(np.linalg.norm(positions[-1] - np.array([-MOON_RADIUS, 0.0])))
    errors = {method: np.array(values) for method, values in errors.items()}
    figure, axis = plt.subplots(figsize=(7.5, 5))
    for method, values in errors.items():
        axis.loglog(1.0 / step_counts, values, "o-", label=method)
    axis.set(xlabel=r"$\Delta t/T$", ylabel=r"$\|\mathbf{r}(T/2)-(-r_M,0)\|$ [m]", title="1d: half-period error")
    axis.grid(True, which="both", ls=":", alpha=0.5)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "task_1d_convergence.png", dpi=180)
    plt.close(figure)

    slopes = {method: float(np.polyfit(np.log(period / step_counts), np.log(values), 1)[0]) for method, values in errors.items()}
    with (output_dir / "task_1d_errors.csv").open("w", encoding="utf-8") as file:
        file.write("method," + ",".join(f"N={count}" for count in step_counts) + ",slope\n")
        for method in METHODS:
            file.write(method + "," + ",".join(f"{value:.8e}" for value in errors[method]) + f",{slopes[method]:.6f}\n")
    return {
        "period_days": period / DAY,
        "implicit_newton_iterations": implicit_iterations,
        **{f"slope_{method}": value for method, value in slopes.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("exercise_04_task_1_output"))
    args = parser.parse_args()
    results = generate_outputs(args.output_dir)
    print(f"Kepler period: {results['period_days']:.6f} d")
    print(f"Implicit Euler Newton iterations (before breakdown): {results['implicit_newton_iterations']}")
    for key, value in results.items():
        if key.startswith("slope_"):
            print(f"{key.removeprefix('slope_')}: {value:.4f}")


if __name__ == "__main__":
    main()
