"""Exercise 6, task 2: time-dependent one-dimensional Schrödinger equation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu


def kinetic_hamiltonian(x: np.ndarray, potential: np.ndarray) -> tuple[np.diags, np.ndarray]:
    dx = x[1] - x[0]
    n = x.size
    main = 1.0 / dx**2 + potential
    off = np.full(n - 1, -0.5 / dx**2)
    matrix = diags((off, main, off), (-1, 0, 1), format="csc")
    return matrix, dx


def gaussian_packet(x: np.ndarray, x0: float, sigma: float, k0: float) -> np.ndarray:
    psi = np.exp(-(x - x0) ** 2 / (4.0 * sigma**2)) * np.exp(1j * k0 * (x - x0))
    return psi / np.sqrt(np.sum(np.abs(psi) ** 2) * (x[1] - x[0]))


def exact_free_state(x: np.ndarray, t: float, x0: float, sigma: float, k0: float) -> np.ndarray:
    alpha = 1.0 + 1j * t / (2.0 * sigma**2)
    center = x0 + k0 * t
    prefactor = (2.0 * np.pi * sigma**2) ** -0.25
    prefactor /= np.sqrt(alpha)
    envelope = np.exp(-(x - center) ** 2 / (4.0 * sigma**2 * alpha))
    phase = np.exp(1j * k0 * (x - x0) - 1j * k0**2 * t / 2.0)
    return prefactor * envelope * phase


def propagate_cn(x: np.ndarray, psi: np.ndarray, potential: np.ndarray,
                 dt: float, steps: int, sample_every: int = 1):
    h, dx = kinetic_hamiltonian(x, potential)
    identity = diags(np.ones(x.size), 0, format="csc")
    lhs = identity + 1j * dt * 0.5 * h
    rhs = identity - 1j * dt * 0.5 * h
    solver = splu(lhs)
    times = [0.0]
    norms = [float(np.sum(np.abs(psi)**2) * dx)]
    means = [float(np.sum(x * np.abs(psi)**2) * dx)]
    variances = [float(np.sum((x-means[-1])**2*np.abs(psi)**2) * dx)]

    for step in range(1, steps + 1):
        psi = solver.solve(rhs @ psi)
        if step % sample_every == 0:
            norm = float(np.sum(np.abs(psi)**2) * dx)
            mean = float(np.sum(x * np.abs(psi)**2) * dx / norm)
            variance = float(np.sum((x-mean)**2*np.abs(psi)**2) * dx / norm)
            times.append(step * dt)
            norms.append(norm)
            means.append(mean)
            variances.append(variance)
    return np.asarray(times), np.asarray(norms), np.asarray(means), np.sqrt(np.maximum(variances, 0.0)), psi


def propagate_free_explicit(x: np.ndarray, psi: np.ndarray, dt: float,
                           steps: int, sample_every: int = 1):
    h, dx = kinetic_hamiltonian(x, np.zeros(x.size))
    times, norms, means, widths = [0.0], [], [], []
    norm = float(np.sum(np.abs(psi)**2)*dx)
    mean = float(np.sum(x*np.abs(psi)**2)*dx/norm)
    var = float(np.sum((x-mean)**2*np.abs(psi)**2)*dx/norm)
    norms.append(norm); means.append(mean); widths.append(np.sqrt(var))
    for step in range(1, steps+1):
        psi = psi - 1j*dt*(h @ psi)
        if step % sample_every == 0:
            norm = float(np.sum(np.abs(psi)**2)*dx)
            mean = float(np.sum(x*np.abs(psi)**2)*dx/norm)
            var = float(np.sum((x-mean)**2*np.abs(psi)**2)*dx/norm)
            times.append(step*dt); norms.append(norm); means.append(mean)
            widths.append(np.sqrt(max(var,0.0)))
    return np.asarray(times), np.asarray(norms), np.asarray(means), np.asarray(widths), psi


def propagate_free_implicit(x: np.ndarray, psi: np.ndarray, dt: float,
                           steps: int, sample_every: int = 1):
    h, dx = kinetic_hamiltonian(x, np.zeros(x.size))
    identity = diags(np.ones(x.size), 0, format="csc")
    solver = splu(identity + 1j*dt*h)
    times, norms, means, widths = [0.0], [], [], []
    norm = float(np.sum(np.abs(psi)**2)*dx)
    mean = float(np.sum(x*np.abs(psi)**2)*dx/norm)
    var = float(np.sum((x-mean)**2*np.abs(psi)**2)*dx/norm)
    norms.append(norm); means.append(mean); widths.append(np.sqrt(var))
    for step in range(1, steps+1):
        psi = solver.solve(psi)
        if step % sample_every == 0:
            norm = float(np.sum(np.abs(psi)**2)*dx)
            mean = float(np.sum(x*np.abs(psi)**2)*dx/norm)
            var = float(np.sum((x-mean)**2*np.abs(psi)**2)*dx/norm)
            times.append(step*dt); norms.append(norm); means.append(mean)
            widths.append(np.sqrt(max(var,0.0)))
    return np.asarray(times), np.asarray(norms), np.asarray(means), np.asarray(widths), psi


def task_2a(output_dir: Path) -> dict:
    x = np.arange(-30.0, 30.0 + 0.10, 0.10)
    sigma, k0, x0 = 1.0, 2.0, -5.0
    psi0 = gaussian_packet(x, x0, sigma, k0)
    dt = 0.0001
    final = 8.0
    steps = int(round(final/dt))

    h, dx = kinetic_hamiltonian(x, np.zeros(x.size))
    identity = diags(np.ones(x.size), 0, format="csc")

    # Explicit Euler is deliberately run here; for this fine grid it remains stable.
    explicit = propagate_free_explicit(x, psi0.copy(), dt, steps, 80)
    implicit = propagate_free_implicit(x, psi0.copy(), dt, steps, 80)
    crank = propagate_cn(x, psi0.copy(), np.zeros(x.size), dt, steps, 80)

    times = crank[0]
    exact_norm = np.ones_like(times)
    exact_mean = x0 + k0*times
    exact_width = np.sqrt(sigma**2 + (times/(2.0*sigma))**2)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    labels = (("Explicit Euler", explicit), ("Implicit Euler", implicit), ("Crank-Nicolson", crank))
    quantities = (1, 2, 3)
    titles = (r"$\|\psi(t)\|^2$ (normalization)", r"$\langle x\rangle(t)$", r"width $\sigma(t)$")
    exact = (exact_norm, exact_mean, exact_width)
    for axis, quantity, title, exact_values in zip(axes, quantities, titles, exact):
        for label, result in labels:
            axis.plot(result[0], result[quantity], label=label, lw=1.4)
        axis.plot(times, exact_values, "k--", lw=2, label="Analytic")
        axis.set(xlabel="t", title=title)
        axis.grid(True, ls=":", alpha=0.5)
    axes[0].set_ylabel(r"norm")
    axes[1].set_ylabel("position")
    axes[2].set_ylabel("width")
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "task_2a_free_method_comparison.png", dpi=180)
    plt.close(figure)

    # Grid-resolution study: density error for several k0 and spatial spacings.
    k_values = np.array([1.0, 2.0, 4.0, 8.0])
    dx_values = np.array([0.5, 0.25, 0.125, 0.0625])
    errors = np.empty((k_values.size, dx_values.size))
    for i, k in enumerate(k_values):
        for j, grid_dx in enumerate(dx_values):
            xg = np.arange(-12.0, 12.0 + grid_dx, grid_dx)
            psi = gaussian_packet(xg, -4.0, 1.0, k)
            dtg = min(0.002, 0.2*grid_dx**2)
            tg = 2.0
            stepsg = int(round(tg/dtg))
            _, _, _, _, psit = propagate_cn(xg, psi, np.zeros(xg.size), dtg, stepsg, stepsg)
            exact = exact_free_state(xg, tg, -4.0, 1.0, k)
            errors[i,j] = np.sqrt(np.sum(np.abs(psit-exact)**2)*grid_dx)

    figure, axis = plt.subplots(figsize=(8,5.5))
    image = axis.imshow(errors, origin="lower", aspect="auto", cmap="magma_r",
                        extent=[-0.5, dx_values.size-0.5, k_values[0]-0.5, k_values[-1]+0.5])
    axis.set_xticks(np.arange(dx_values.size), [str(v) for v in dx_values])
    axis.set_yticks(k_values)
    axis.set(xlabel=r"grid spacing $\Delta x$", ylabel=r"initial momentum $k_0$",
             title=r"2a: $L^2$ density-state error at $t=2$ (CN)")
    for i in range(k_values.size):
        for j in range(dx_values.size):
            axis.text(j, k_values[i], f"{errors[i,j]:.3g}", ha="center", va="center",
                      color="white" if errors[i,j] > np.percentile(errors,75) else "black", fontsize=8)
    figure.colorbar(image, ax=axis, label=r"$\|\psi_{\mathrm{s}}-\psi_{\mathrm{e}}\|_2$")
    figure.tight_layout()
    figure.savefig(output_dir / "task_2a_grid_size_vs_k0.png", dpi=180)
    plt.close(figure)

    np.savetxt(output_dir / "task_2a_grid_study.csv", np.column_stack((k_values[:,None]*np.ones((1,dx_values.size)), dx_values*np.ones((k_values.size,1)), errors)), delimiter=",")
    return {"times": times, "exact_norm": exact_norm,
            "exact_mean": exact_mean, "exact_width": exact_width}


def task_2b(output_dir: Path) -> dict:
    x = np.arange(-12.0, 12.0 + 0.05, 0.05)
    omega = 1.0
    sigma = 1.0/np.sqrt(2.0)
    x0 = 3.0
    psi0 = gaussian_packet(x, x0, sigma, 0.0)
    potential = 0.5*omega**2*x**2
    dt = 0.002
    final = 12.0
    steps = int(round(final/dt))

    times, _, means, widths, psi = propagate_cn(x, psi0, potential, dt, steps, 100)
    classical = x0*np.cos(times)

    figure, axes = plt.subplots(1, 2, figsize=(12,4.8))
    axes[0].plot(times, means, "o-", markevery=20, ms=3, label="Quantum packet")
    axes[0].plot(times, classical, "k--", lw=2, label="Classical particle")
    axes[0].set(xlabel="t", ylabel=r"$\langle x\rangle$", title="2b: center of packet")
    axes[0].grid(True, ls=":", alpha=0.5); axes[0].legend()

    axes[1].plot(times, widths, label="Numerical packet width")
    axes[1].axhline(sigma, color="k", ls="--", label="Ground-state width")
    axes[1].set(xlabel="t", ylabel=r"$\sigma(t)$", title="2b: spreading of shifted ground state")
    axes[1].grid(True, ls=":", alpha=0.5); axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "task_2b_harmonic_comparison.png", dpi=180)
    plt.close(figure)
    return {"period_mean": np.asarray(means), "period_width": np.asarray(widths), "times": times}


def task_2c(output_dir: Path) -> dict:
    x = np.arange(0.0, 1.0 + 0.002, 0.002)
    psi0 = gaussian_packet(x, 0.3, 0.05, 0.0)
    potential = np.zeros(x.size)
    h, dx = kinetic_hamiltonian(x, potential)
    dt = 0.00005
    trev = 4.0/np.pi
    final = trev
    steps = int(round(final/dt))

    identity = diags(np.ones(x.size), 0, format="csc")
    lhs = identity + 1j*dt*0.5*h
    rhs = identity - 1j*dt*0.5*h
    solver = splu(lhs)
    psi = psi0.copy()

    special_times = [0.0, trev/4.0, trev/2.0, trev]
    densities = {}
    density_times = np.linspace(0.0, final, 401)
    heat = np.empty((density_times.size, x.size))
    target_steps = {int(round(t/dt)): t for t in special_times}

    for i, target_time in enumerate(density_times):
        target_step = int(round(target_time/dt))
        if i == 0:
            heat[i] = np.abs(psi)**2
        else:
            previous = int(round(density_times[i-1]/dt))
            for _ in range(previous, target_step):
                psi = solver.solve(rhs @ psi)
            heat[i] = np.abs(psi)**2

    for target_step, target_time in target_steps.items():
        psi_check = psi0.copy()
        solver_check = splu(lhs)
        for _ in range(target_step):
            psi_check = solver_check.solve(rhs @ psi_check)
        densities[target_time] = np.abs(psi_check)**2

    figure, axes = plt.subplots(1, 2, figsize=(13,5), gridspec_kw={"width_ratios":[1.6,1]})
    image = axes[0].imshow(heat, origin="lower", aspect="auto", extent=[x[0],x[-1],0,trev], cmap="turbo")
    for t in special_times:
        axes[0].axhline(t, color="white", lw=1, ls="--")
    axes[0].set(xlabel="x", ylabel="t", title=r"2c: $|\psi(x,t)|^2$ in infinite well")
    figure.colorbar(image, ax=axes[0], label=r"density")

    colors = ("C0","C1","C2","C3")
    for (t,density), color in zip(densities.items(), colors):
        axes[1].plot(x, density, color=color, lw=1.8, label=f"t={t:.4f}")
    axes[1].set(xlabel="x", ylabel=r"$|\psi|^2$", title=r"special times")
    axes[1].grid(True, ls=":", alpha=0.5); axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "task_2c_infinite_well_revivals.png", dpi=180)
    plt.close(figure)
    return {"trev": trev}


def complex_absorber(x: np.ndarray, strength: float = 1.0) -> np.ndarray:
    edge = 0.18*(x[-1]-x[0])
    distance_left = np.maximum(0.0, x[0]+edge-x)
    distance_right = np.maximum(0.0, x-x[-1]+edge)
    w = strength*(distance_left/edge)**2 + strength*(distance_right/edge)**2
    return w


def analytic_barrier_transmission(energies: np.ndarray, v0: float, b: float) -> np.ndarray:
    transmitted = np.empty(energies.size)
    for i, energy in enumerate(energies):
        if energy < v0:
            kappa = np.sqrt(2.0*(v0-energy))
            argument = kappa*b
            denominator = 1.0 + v0**2*np.sinh(argument)**2/(4.0*energy*(v0-energy))
        elif energy-v0 > 1e-12:
            k = np.sqrt(2.0*(energy-v0))
            argument = k*b
            denominator = 1.0 + v0**2*np.sin(argument)**2/(4.0*energy*(energy-v0))
        else:
            denominator = 1.0 + v0**2*b**2/4.0
        transmitted[i] = 1.0/denominator
    return transmitted


def transmit_probability(x: np.ndarray, k0: float, sigma: float, x0: float,
                        v0: float, b: float, with_absorber: bool,
                        final_time: float | None = None) -> tuple[float,np.ndarray,np.ndarray]:
    dx = x[1]-x[0]
    potential = np.where((x>=0.0)&(x<=b), v0, 0.0)
    if with_absorber:
        w = complex_absorber(x, strength=2.0)
        potential = potential - 1j*w
    psi = gaussian_packet(x, x0, sigma, k0)
    initial_norm = float(np.sum(np.abs(psi)**2)*dx)
    h, _ = kinetic_hamiltonian(x, potential)
    dt = 0.01
    final = 70.0/k0 if final_time is None else final_time
    steps = int(round(final/dt))
    identity = diags(np.ones(x.size),0,format="csc")
    lhs = identity + 1j*dt*0.5*h
    rhs = identity - 1j*dt*0.5*h
    solver = splu(lhs)
    norms=[]
    times=[]
    right_prob=[]
    barrier_prob=[]
    for step in range(steps):
        psi = solver.solve(rhs@psi)
        if step % 20 == 0:
            norms.append(float(np.sum(np.abs(psi)**2)*dx))
            times.append(step*dt)
            screen = (x>b) & (x < x[-1] - 0.18*(x[-1]-x[0]))
            right_prob.append(float(np.sum(np.abs(psi[screen])**2)*dx/initial_norm))
            barrier_prob.append(float(np.sum(np.abs(psi[(x>=0)&(x<=b)])**2)*dx/norms[-1]))
    return right_prob[-1], np.asarray(times), np.asarray(right_prob)


def task_2d(output_dir: Path) -> dict:
    x = np.arange(-80.0,80.0+0.10,0.10)
    sigma=5.0
    x0=-18.0
    v0=1.0
    b=1.0
    energies=np.linspace(0.2,2.5,24)
    k_values=np.sqrt(2.0*energies)
    numeric=[]
    numeric_abs=[]
    right_time_examples=[]
    for k in k_values:
        tprob, _, tright = transmit_probability(x,k,sigma,x0,v0,b,False)
        aprob, _, aright = transmit_probability(x,k,sigma,x0,v0,b,True)
        numeric.append(tprob); numeric_abs.append(aprob)
    numeric=np.asarray(numeric); numeric_abs=np.asarray(numeric_abs)
    analytic=analytic_barrier_transmission(energies,v0,b)

    figure, axes=plt.subplots(1,2,figsize=(13,5))
    axes[0].plot(energies,analytic,"k--",lw=2,label="Analytic transmission")
    axes[0].plot(energies,numeric,"o-",label="Domain without absorber")
    axes[0].plot(energies,numeric_abs,"s-",label="Complex absorbing boundary")
    axes[0].axvspan(0.2,v0,color="C1",alpha=0.08,label="tunneling E<V0")
    axes[0].axvline(v0,color="C3",ls=":",label="barrier top")
    axes[0].set(xlabel="packet energy $E=k_0^2/2$",ylabel="transmitted probability",
               title="2d: barrier transmission and boundary effects")
    axes[0].grid(True,ls=":",alpha=0.5); axes[0].legend(fontsize=8)

    k_example=np.sqrt(0.4)
    _,times_plain,plain=transmit_probability(x,k_example,sigma,x0,v0,b,False,150.0)
    _,times_abs,absorbed=transmit_probability(x,k_example,sigma,x0,v0,b,True,150.0)
    axes[1].plot(times_plain,plain,"o-",markevery=12,label="without absorber")
    axes[1].plot(times_abs,absorbed,"s-",markevery=12,label="with -iW(x)")
    axes[1].set(xlabel="time",ylabel=r"probability to right of barrier",
               title=r"2d: $E=k_0^2/2\approx0.2$, time of arrival")
    axes[1].grid(True,ls=":",alpha=0.5); axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir/"task_2d_tunneling_and_absorbing_boundary.png",dpi=180)
    plt.close(figure)

    np.savetxt(output_dir/"task_2d_transmission.csv",
               np.column_stack((energies,k_values,analytic,numeric,numeric_abs)),delimiter=",",
               header="energy,k0,analytic,without_absorber,with_complex_absorber",comments="")
    return {"energies":energies,"analytic":analytic,"numeric":numeric,"numeric_abs":numeric_abs}


def generate_outputs(output_dir:Path)->dict:
    output_dir.mkdir(parents=True,exist_ok=True)
    return {"2a":task_2a(output_dir),"2b":task_2b(output_dir),
            "2c":task_2c(output_dir),"2d":task_2d(output_dir)}


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,
                        default=Path(__file__).resolve().parent/"exercise_06_task_2_output")
    args=parser.parse_args()
    generate_outputs(args.output)
    print(f"Generated task-2 outputs in {args.output.resolve()}")

if __name__=="__main__":
    main()
