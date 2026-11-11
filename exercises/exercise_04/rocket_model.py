"""Reusable planar rocket and orbital mechanics helpers for exercise 4."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Array = np.ndarray
Direction = Callable[[float, Array, Array], Array]

G = 6.674e-11
EARTH_MASS = 5.972e24
MU = G * EARTH_MASS
EARTH_RADIUS = 6371e3
EARTH_ROTATION = 7.292e-5
EXHAUST_VELOCITY = 4.4e3


def gravity(position: Array) -> Array:
    """Return Earth's point-mass gravitational acceleration."""
    radius = np.linalg.norm(position)
    if radius <= 0.0:
        raise ValueError("gravity is undefined at the origin")
    return -MU * position / radius**3


def radial_and_azimuthal(position: Array) -> tuple[Array, Array]:
    """Return the local radial and counter-clockwise azimuthal unit vectors."""
    radius = np.linalg.norm(position)
    radial = position / radius
    azimuthal = np.array((-radial[1], radial[0]))
    return radial, azimuthal


def pitch_direction(theta: float, eastward: bool = True) -> Direction:
    """Create the PDF's direction ``sin(theta) er +/- cos(theta) ephi``."""
    sign = 1.0 if eastward else -1.0

    def direction(_time: float, position: Array, _velocity: Array) -> Array:
        radial, azimuthal = radial_and_azimuthal(position)
        return np.sin(theta) * radial + sign * np.cos(theta) * azimuthal

    return direction


def variable_pitch_direction(tp: float, eastward: bool = True) -> Direction:
    """Return the PDF pitch program ``theta=90 deg max(0, 1-t/tp)``."""
    if tp <= 0.0:
        raise ValueError("pitch time must be positive")
    sign = 1.0 if eastward else -1.0

    def direction(time: float, position: Array, _velocity: Array) -> Array:
        radial, azimuthal = radial_and_azimuthal(position)
        theta = 0.5 * np.pi * max(0.0, 1.0 - time / tp)
        return np.sin(theta) * radial + sign * np.cos(theta) * azimuthal

    return direction


def rocket_rhs(
    time: float,
    state: Array,
    *,
    thrust_acceleration: float,
    exhaust_velocity: float = EXHAUST_VELOCITY,
    fuel_fraction: float = 18.0 / 19.0,
    direction: Direction,
    thrust_end: float | None = None,
    stage_initial_mass_fraction: float = 1.0,
    inert_mass_fraction: float = 0.0,
) -> Array:
    """Return the 2-D rocket RHS with linearly decreasing mass.

    ``thrust_acceleration`` is ``F/m0`` and ``fuel_fraction`` is the fraction
    of initial mass burned.  The thrust acceleration is therefore scaled by
    ``1/(1-fuel_fraction*time/thrust_end)`` while the engine is on.
    """
    position, velocity = state[:2], state[2:]
    acceleration = gravity(position)
    if thrust_end is not None and 0.0 <= time < thrust_end:
        mass_fraction = (
            inert_mass_fraction
            + stage_initial_mass_fraction
            * (1.0 - fuel_fraction * time / thrust_end)
        )
        acceleration = acceleration + thrust_acceleration * direction(time, position, velocity) / mass_fraction
    return np.concatenate((velocity, acceleration))


def rk4_step(rhs: Callable[[float, Array], Array], time: float, state: Array, dt: float) -> Array:
    """Advance a general first-order system by one RK4 step."""
    k1 = rhs(time, state)
    k2 = rhs(time + dt / 2.0, state + dt * k1 / 2.0)
    k3 = rhs(time + dt / 2.0, state + dt * k2 / 2.0)
    k4 = rhs(time + dt, state + dt * k3)
    return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def integrate(
    rhs: Callable[[float, Array], Array],
    initial_state: Array,
    final_time: float,
    dt: float,
    *,
    stop: Callable[[float, Array], bool] | None = None,
) -> tuple[Array, Array]:
    """Integrate with fixed RK4 steps and optionally stop at an event."""
    if final_time <= 0.0 or dt <= 0.0:
        raise ValueError("final_time and dt must be positive")
    count = int(np.ceil(final_time / dt))
    times = [0.0]
    states = [np.array(initial_state, dtype=float)]
    for index in range(count):
        time = times[-1]
        step = min(dt, final_time - time)
        state = rk4_step(rhs, time, states[-1], step)
        times.append(time + step)
        states.append(state)
        if stop is not None and stop(times[-1], state):
            break
    return np.array(times), np.array(states)


def initial_launch_state() -> Array:
    """Return the co-rotating equatorial launch state from the PDF."""
    return np.array(
        [EARTH_RADIUS, 0.0, 0.0, EARTH_ROTATION * EARTH_RADIUS],
        dtype=float,
    )


def specific_energy(position: Array, velocity: Array) -> Array:
    """Return specific mechanical energy."""
    return 0.5 * np.sum(velocity**2, axis=-1) - MU / np.linalg.norm(position, axis=-1)


def specific_angular_momentum(position: Array, velocity: Array) -> Array:
    """Return the scalar z-component of specific angular momentum."""
    return position[..., 0] * velocity[..., 1] - position[..., 1] * velocity[..., 0]


def mass_ratio(time: Array | float, thrust_end: float, fuel_fraction: float) -> Array | float:
    """Return m(t)/m0 while burning, clipped to the dry mass afterwards."""
    return np.maximum(1.0 - fuel_fraction * np.asarray(time) / thrust_end, 1.0 - fuel_fraction)
