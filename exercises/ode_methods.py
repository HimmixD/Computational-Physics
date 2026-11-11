"""Reusable numerical methods for ordinary differential equations.

The first-order steppers operate on ``y' = rhs(t, y)``.  The symplectic
Euler and velocity-Verlet steppers operate on second-order systems
``q'' = acceleration(t, q)`` and keep position and velocity separate.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Array = np.ndarray
RHS = Callable[[float, Array], Array]
Jacobian = Callable[[float, Array], Array]
Acceleration = Callable[[float, Array], Array]


def explicit_euler(rhs: RHS, t: float, y: Array, dt: float) -> Array:
    """Advance ``y' = rhs(t, y)`` by explicit Euler."""
    return y + dt * rhs(t, y)


def heun(rhs: RHS, t: float, y: Array, dt: float) -> Array:
    """Advance ``y' = rhs(t, y)`` by the explicit trapezoidal rule."""
    k1 = rhs(t, y)
    predictor = y + dt * k1
    return y + 0.5 * dt * (k1 + rhs(t + dt, predictor))


def rk4(rhs: RHS, t: float, y: Array, dt: float) -> Array:
    """Advance ``y' = rhs(t, y)`` with classical fourth-order Runge--Kutta."""
    k1 = rhs(t, y)
    k2 = rhs(t + dt / 2.0, y + dt * k1 / 2.0)
    k3 = rhs(t + dt / 2.0, y + dt * k2 / 2.0)
    k4 = rhs(t + dt, y + dt * k3)
    return y + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def implicit_euler(
    rhs: RHS,
    jacobian: Jacobian,
    t: float,
    y: Array,
    dt: float,
    *,
    initial_guess: Array | None = None,
    tolerance: float = 1e-12,
    max_iterations: int = 40,
) -> tuple[Array, int]:
    """Advance with implicit Euler using Newton's method.

    The Newton initial guess defaults to one explicit-Euler step.  The
    returned iteration count includes the converged Newton iteration.
    """
    candidate = (
        np.array(initial_guess, dtype=float, copy=True)
        if initial_guess is not None
        else explicit_euler(rhs, t, y, dt)
    )
    identity = np.eye(y.size)
    for iteration in range(1, max_iterations + 1):
        residual = candidate - y - dt * rhs(t + dt, candidate)
        scale = 1.0 + np.linalg.norm(candidate, ord=np.inf)
        if np.linalg.norm(residual, ord=np.inf) <= tolerance * scale:
            return candidate, iteration
        scale_vector = np.maximum(1.0, np.abs(candidate))
        scaled_system = (identity - dt * jacobian(t + dt, candidate)) * (
            scale_vector[np.newaxis, :] / scale_vector[:, np.newaxis]
        )
        correction = scale_vector * np.linalg.solve(scaled_system, residual / scale_vector)
        residual_norm = np.linalg.norm(residual, ord=np.inf)
        damping = 1.0
        while damping >= 1e-4:
            trial = candidate - damping * correction
            trial_residual = trial - y - dt * rhs(t + dt, trial)
            if np.linalg.norm(trial_residual, ord=np.inf) < residual_norm:
                candidate = trial
                break
            damping *= 0.5
        else:
            candidate -= correction
    raise RuntimeError(f"Newton iteration did not converge after {max_iterations} steps")


def symplectic_euler(
    acceleration: Acceleration,
    t: float,
    position: Array,
    velocity: Array,
    dt: float,
) -> tuple[Array, Array]:
    """Advance ``q'' = acceleration(t, q)`` with velocity-first symplectic Euler."""
    next_velocity = velocity + dt * acceleration(t, position)
    next_position = position + dt * next_velocity
    return next_position, next_velocity


def velocity_verlet(
    acceleration: Acceleration,
    t: float,
    position: Array,
    velocity: Array,
    dt: float,
) -> tuple[Array, Array]:
    """Advance ``q'' = acceleration(t, q)`` with velocity Verlet."""
    current_acceleration = acceleration(t, position)
    next_position = position + dt * velocity + 0.5 * dt**2 * current_acceleration
    next_acceleration = acceleration(t + dt, next_position)
    next_velocity = velocity + 0.5 * dt * (current_acceleration + next_acceleration)
    return next_position, next_velocity


def integrate_first_order(
    stepper: Callable[..., Array],
    rhs: RHS,
    initial: Array,
    final_time: float,
    dt: float,
) -> tuple[Array, Array]:
    """Integrate a first-order ODE with a fixed step and exact final time."""
    times = _time_grid(final_time, dt)
    states = np.empty((times.size, initial.size), dtype=float)
    states[0] = initial
    for index, (time, next_time) in enumerate(zip(times[:-1], times[1:])):
        states[index + 1] = stepper(rhs, float(time), states[index], float(next_time - time))
    return times, states


def integrate_second_order(
    stepper: Callable[..., tuple[Array, Array]],
    acceleration: Acceleration,
    initial_position: Array,
    initial_velocity: Array,
    final_time: float,
    dt: float,
) -> tuple[Array, Array, Array]:
    """Integrate a second-order ODE with a fixed step."""
    times = _time_grid(final_time, dt)
    positions = np.empty((times.size, initial_position.size), dtype=float)
    velocities = np.empty_like(positions)
    positions[0] = initial_position
    velocities[0] = initial_velocity
    for index, (time, next_time) in enumerate(zip(times[:-1], times[1:])):
        positions[index + 1], velocities[index + 1] = stepper(
            acceleration,
            float(time),
            positions[index],
            velocities[index],
            float(next_time - time),
        )
    return times, positions, velocities


def _time_grid(final_time: float, dt: float) -> Array:
    if final_time <= 0.0 or dt <= 0.0:
        raise ValueError("final_time and dt must be positive")
    steps = int(np.floor(final_time / dt))
    times = np.arange(steps + 1, dtype=float) * dt
    if times[-1] < final_time:
        times = np.append(times, final_time)
    return times
