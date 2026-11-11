"""Exercise 4, task 2: planar rocket launches and orbital insertion."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

sys.path.insert(0, str(Path(__file__).parents[1]))
from exercise_04.rocket_model import (
    EARTH_RADIUS,
    EXHAUST_VELOCITY,
    MU,
    specific_energy,
    initial_launch_state,
    integrate,
    mass_ratio,
    pitch_direction,
    rocket_rhs,
    variable_pitch_direction,
)

G0 = MU / EARTH_RADIUS**2
HOUR = 3600.0
FRACTIONS = (1.0 / 19.0, 0.12)


def burnout_time(thrust_acceleration: float, dry_mass_fraction: float) -> float:
    """Return burn duration from ``mdot=F/ve`` and ``F/m0``."""
    return (1.0 - dry_mass_fraction) * EXHAUST_VELOCITY / thrust_acceleration


def simulate_launch(
    thrust_acceleration: float,
    dry_mass_fraction: float,
    direction,
    final_time: float,
    *,
    dt: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Simulate a launch with a continuously burning engine."""
    burn_time = burnout_time(thrust_acceleration, dry_mass_fraction)
    rhs = lambda time, state: rocket_rhs(
        time,
        state,
        thrust_acceleration=thrust_acceleration,
        exhaust_velocity=EXHAUST_VELOCITY,
        fuel_fraction=1.0 - dry_mass_fraction,
        direction=direction,
        thrust_end=burn_time,
    )
    times, states = integrate(rhs, initial_launch_state(), final_time, dt)
    return times, states, burn_time


def classify_trajectory(times: np.ndarray, states: np.ndarray, burn_time: float) -> str:
    """Classify a trajectory using surface intersection and post-burn energy."""
    radii = np.linalg.norm(states[:, :2], axis=1)
    if np.any(radii[1:] < EARTH_RADIUS):
        return "crash"
    burn_index = min(np.searchsorted(times, burn_time), len(times) - 1)
    after_burn = states[burn_index:]
    post_energy = specific_energy(after_burn[:, :2], after_burn[:, 2:])
    if np.any(post_energy >= 0.0):
        return "escape"
    if np.any(radii[burn_index:] <= EARTH_RADIUS * (1.0 + 1e-6)):
        return "falls back"
    return "orbit"


def vertical_launch(output_dir: Path) -> dict[str, float]:
    accelerations = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
    rows = []
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for thrust in accelerations:
        times, states, burn_time = simulate_launch(
            thrust, 1.0 / 19.0, pitch_direction(np.pi / 2.0), 2.0 * HOUR
        )
        altitude = np.linalg.norm(states[:, :2], axis=1) - EARTH_RADIUS
        radial_velocity = np.sum(states[:, :2] * states[:, 2:], axis=1) / np.linalg.norm(states[:, :2], axis=1)
        burnout = min(np.searchsorted(times, burn_time), len(times) - 1)
        energy_burnout = specific_energy(states[burnout, :2], states[burnout, 2:])
        rocket_delta_v = EXHAUST_VELOCITY * np.log(19.0)
        rows.append((thrust, burn_time, altitude[burnout], radial_velocity[burnout], energy_burnout, rocket_delta_v, radial_velocity[burnout] - rocket_delta_v))
        axes[0].plot(times / 60.0, altitude / 1e3, label=f"{thrust:g} m/s²")
        axes[1].plot(times / 60.0, radial_velocity / 1e3, label=f"{thrust:g} m/s²")
    axes[0].set(xlabel="time [min]", ylabel="altitude [km]", title="2a: vertical launch")
    axes[1].set(xlabel="time [min]", ylabel="vertical speed [km/s]", title="Burnout and coast")
    for axis in axes:
        axis.grid(True, ls=":", alpha=0.5)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "task_2a_vertical_launch.png", dpi=180)
    plt.close(figure)
    with (output_dir / "task_2a_vertical.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("F_over_m0", "burn_time_s", "burnout_altitude_m", "burnout_speed_m_s", "burnout_energy_J_kg", "rocket_equation_delta_v_m_s", "gravity_loss_m_s"))
        writer.writerows(rows)
    return {"vertical_escape_threshold": float(np.interp(0.0, [row[4] for row in rows], accelerations)) if rows[0][4] * rows[-1][4] < 0 else float("nan")}


def pitch_program(output_dir: Path) -> tuple[float, dict[tuple[float, bool], str]]:
    pitch_times = np.arange(100.0, 301.0, 25.0)
    results: dict[tuple[float, bool], str] = {}
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for eastward, axis in zip((True, False), axes):
        for pitch_time in pitch_times:
            times, states, burn_time = simulate_launch(
                15.0,
                0.12,
                variable_pitch_direction(pitch_time, eastward),
                3.0 * HOUR,
                dt=1.0,
            )
            result = classify_trajectory(times, states, burn_time)
            results[(pitch_time, eastward)] = result
            axis.plot(states[:, 0] / 1e6, states[:, 1] / 1e6, label=f"{pitch_time:g} s: {result}")
        axis.plot(0.0, 0.0, "ko")
        axis.set_aspect("equal")
        axis.set(xlabel="x [Mm]", ylabel="y [Mm]", title="eastward" if eastward else "westward")
        axis.grid(True, ls=":", alpha=0.5)
        axis.legend(fontsize=6, ncol=2)
    figure.suptitle("2b: pitch-program trajectories")
    figure.tight_layout()
    figure.savefig(output_dir / "task_2b_pitch_program.png", dpi=180)
    plt.close(figure)
    with (output_dir / "task_2b_classification.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("tp_s", "direction", "classification"))
        for pitch_time in pitch_times:
            writer.writerow((pitch_time, "eastward", results[(pitch_time, True)]))
            writer.writerow((pitch_time, "westward", results[(pitch_time, False)]))
    east_orbits = [tp for tp in pitch_times if results[(tp, True)] == "orbit"]
    best = float(east_orbits[0]) if east_orbits else float(pitch_times[np.argmin([abs(np.linalg.norm(simulate_launch(15.0, 0.12, variable_pitch_direction(tp, True), 3.0 * HOUR, dt=2.0)[1][-1, :2]) - 2.0 * EARTH_RADIUS) for tp in pitch_times])])
    return best, results


def circular_insertion(output_dir: Path, pitch_time: float) -> None:
    """Use powered ascent, coast to the target radius, then circularize."""
    targets = (200e3, 500e3, 1000e3)
    rows = []
    figure, axis = plt.subplots(figsize=(8, 8))
    for altitude in targets:
        ascent_time, ascent, burn_time = simulate_launch(
            15.0, 0.12, variable_pitch_direction(pitch_time, True), 1800.0, dt=1.0
        )
        radii = np.linalg.norm(ascent[:, :2], axis=1)
        target_radius = EARTH_RADIUS + altitude
        candidate = np.flatnonzero(
            (ascent_time[:-1] > burn_time)
            & (radii[:-1] <= target_radius)
            & (radii[1:] >= target_radius)
        )
        if candidate.size == 0:
            raise RuntimeError(f"ascent did not cross target radius for {altitude / 1e3:g} km")
        crossing = int(candidate[0] + 1)
        state = ascent[crossing].copy()
        circular_speed = np.sqrt(MU / target_radius)
        radial = state[:2] / np.linalg.norm(state[:2])
        azimuthal = np.array((-radial[1], radial[0]))
        circularization_delta_v = np.linalg.norm(
            circular_speed * azimuthal - state[2:]
        )
        state[2:] = np.dot(state[2:], radial) * radial + circular_speed * azimuthal
        coast_times, coast_states = integrate(
            lambda _t, s: np.concatenate((s[2:], -MU * s[:2] / np.linalg.norm(s[:2])**3)),
            state,
            2.0 * HOUR,
            2.0,
        )
        axis.plot(coast_states[:, 0] / 1e6, coast_states[:, 1] / 1e6, label=f"{altitude/1e3:g} km")
        remaining_mass_fraction = 0.12 * np.exp(-circularization_delta_v / EXHAUST_VELOCITY)
        rows.append(
            (
                altitude,
                np.linalg.norm(ascent[crossing, :2]) - EARTH_RADIUS,
                np.linalg.norm(ascent[crossing, 2:]),
                circular_speed,
                circularization_delta_v,
                remaining_mass_fraction,
            )
        )
    axis.plot(0.0, 0.0, "ko", label="Earth")
    axis.set_aspect("equal")
    axis.set(xlabel="x [Mm]", ylabel="y [Mm]", title="2c: circularization at target altitude")
    axis.grid(True, ls=":", alpha=0.5)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "task_2c_circular_orbits.png", dpi=180)
    plt.close(figure)
    with (output_dir / "task_2c_orbits.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("target_altitude_m", "burn_radius_altitude_m", "pre_circularization_speed_m_s", "circular_speed_m_s", "circularization_delta_v_m_s", "mass_fraction_remaining"))
        writer.writerows(rows)


def staging(output_dir: Path) -> None:
    """Compare one-stage and two-stage idealized eastward launch trajectories."""
    figure, (overview, closeup) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": (1.35, 1.0)})
    rows = []
    trajectories = []
    for name, stages in (
        ("one stage", ((1.0, 0.90, 15.0, 0.0),)),
        ("two stages", ((0.80, 0.90, 15.0, 0.20), (0.20, 0.90, 3.0, 0.0))),
    ):
        state = initial_launch_state()
        time_offset = 0.0
        all_states = [state.copy()]
        all_times = [0.0]
        delta_v = 0.0
        for stage_mass_fraction, fuel_fraction, thrust_force_over_initial_mass, inert_mass_fraction in stages:
            burn = fuel_fraction * EXHAUST_VELOCITY / 15.0
            pitch_direction_global = variable_pitch_direction(150.0, True)
            if time_offset == 0.0:
                direction = lambda local_time, position, velocity: pitch_direction_global(
                    local_time, position, velocity
                )
            else:
                direction = lambda _local_time, _position, velocity: velocity / np.linalg.norm(velocity)
            rhs = lambda time, current, burn=burn, fuel_fraction=fuel_fraction, thrust_force_over_initial_mass=thrust_force_over_initial_mass, stage_mass_fraction=stage_mass_fraction, inert_mass_fraction=inert_mass_fraction: rocket_rhs(
                time - time_offset, current, thrust_acceleration=thrust_force_over_initial_mass,
                fuel_fraction=fuel_fraction, direction=direction,
                thrust_end=burn, stage_initial_mass_fraction=stage_mass_fraction,
                inert_mass_fraction=inert_mass_fraction,
            )
            times, states = integrate(
                rhs,
                state,
                burn,
                1.0,
                stop=lambda _time, current: np.linalg.norm(current[:2]) < EARTH_RADIUS,
            )
            all_times.extend((times[1:] + time_offset).tolist())
            all_states.extend(states[1:].tolist())
            state = states[-1].copy()
            delta_v += EXHAUST_VELOCITY * np.log(1.0 / (1.0 - fuel_fraction))
            time_offset += burn
            if np.linalg.norm(state[:2]) < EARTH_RADIUS:
                break
        trajectory = np.array(all_states)
        trajectories.append((name, trajectory))
        initial_speed = np.linalg.norm(initial_launch_state()[2:])
        final_speed = np.linalg.norm(trajectory[-1, 2:])
        status = "crash" if np.linalg.norm(trajectory[-1, :2]) < EARTH_RADIUS else "survives burn"
        rows.append(
            (
                name,
                delta_v,
                final_speed,
                final_speed - initial_speed,
                delta_v - (final_speed - initial_speed),
                specific_energy(trajectory[-1, :2], trajectory[-1, 2:]),
                status,
            )
        )
    colors = ("#2563eb", "#ea580c")
    all_coordinates = np.concatenate([trajectory[:, :2] for _, trajectory in trajectories])
    extent = 1.08 * np.max(np.abs(all_coordinates)) / 1e6
    earth_radius_mm = EARTH_RADIUS / 1e6
    for axis in (overview, closeup):
        axis.set_aspect("equal", adjustable="box")
        axis.set_facecolor("#f8fafc")
        axis.grid(True, ls=":", alpha=0.45)
        axis.axhline(0.0, color="0.5", lw=0.7, alpha=0.35)
        axis.axvline(0.0, color="0.5", lw=0.7, alpha=0.35)
        if axis is overview:
            axis.add_patch(
                Circle(
                    (0.0, 0.0),
                    earth_radius_mm,
                    facecolor="#5b86b5",
                    edgecolor="#183b63",
                    alpha=0.22,
                    lw=1.8,
                    zorder=3,
                )
            )
    for (name, trajectory), color in zip(trajectories, colors):
        x = trajectory[:, 0] / 1e6
        y = trajectory[:, 1] / 1e6
        overview.plot(x, y, color=color, lw=2.0, label=name, zorder=4)
        altitude = (np.linalg.norm(trajectory[:, :2], axis=1) - EARTH_RADIUS) / 1e3
        downrange = (trajectory[:, 0] - EARTH_RADIUS) / 1e3
        closeup.plot(downrange, altitude, color=color, lw=2.0, label=name)
    overview.plot(0.0, 0.0, "w+", ms=9, mew=1.5, label="Earth centre", zorder=5)
    overview.set_xlim(-extent, extent)
    overview.set_ylim(-extent, extent)
    closeup.set_aspect("auto")
    closeup.axhline(0.0, color="#183b63", lw=1.2, label="Earth surface")
    closeup.set_xlim(left=0.0)
    closeup.set_ylim(bottom=0.0)
    overview.set(xlabel="x [Mm]", ylabel="y [Mm]", title="Overview: Earth-centred trajectories")
    closeup.set(xlabel="downrange from launch [km]", ylabel="altitude [km]", title="Altitude profile")
    overview.legend(loc="upper left", framealpha=0.9)
    closeup.legend(loc="upper left", framealpha=0.9)
    figure.suptitle("2d: one-stage versus two-stage launch", fontsize=14)
    figure.tight_layout()
    plot_path = output_dir / "task_2d_staging.png"
    if plot_path.exists():
        plot_path.unlink()
    figure.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    with (output_dir / "task_2d_staging.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("configuration", "ideal_delta_v_m_s", "final_speed_m_s", "net_speed_gain_m_s", "estimated_gravity_and_steering_loss_m_s", "final_specific_energy_J_kg", "status"))
        writer.writerows(rows)


def generate_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    vertical_launch(output_dir)
    best_pitch, _ = pitch_program(output_dir)
    circular_insertion(output_dir, best_pitch)
    staging(output_dir)
    (output_dir / "summary.txt").write_text(f"best eastward pitch time: {best_pitch:.1f} s\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("exercise_04_task_2_output"))
    args = parser.parse_args()
    generate_outputs(args.output_dir)
    print(f"Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
