import math
import random
import time
import matplotlib.pyplot as plt
import numpy as np


def estimate_pi_loops(n: int, seed: int = 42) -> float:
    """Estimate pi using standard Python loops (point-by-point)."""
    rnd = random.Random(seed)
    points_inside = 0
    for _ in range(n):
        x = rnd.random()
        y = rnd.random()
        if x * x + y * y <= 1.0:
            points_inside += 1
    return 4.0 * points_inside / n


def estimate_pi_numpy(n: int, rng: np.random.Generator) -> float:
    """Estimate pi using vectorized NumPy operations (no per-point loop)."""
    pts = rng.random((n, 2))
    points_inside = np.count_nonzero(pts[:, 0] * pts[:, 0] + pts[:, 1] * pts[:, 1] <= 1.0)
    return 4.0 * points_inside / n


def benchmark_function(func, *args, repeat: int = 1) -> float:
    """Benchmark a function call and return minimum execution time in seconds."""
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - t0)
    return min(times)


def plot_errors(n_values: list[int], errors: list[float], output_file: str = "errors.png") -> None:
    """Plot Monte Carlo absolute errors vs sample size N on a log-log scale."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(n_values, errors, marker="o", linestyle="-", color="#1f77b4", label=r"Monte Carlo Error $|\hat{\pi} - \pi|$")

    ref_scaling = errors[0] * np.sqrt(n_values[0])
    theory_line = [ref_scaling / np.sqrt(n) for n in n_values]
    ax.loglog(n_values, theory_line, linestyle="--", color="#d62728", alpha=0.8, label=r"Theoretical trend $\mathcal{O}(1/\sqrt{N})$")

    ax.set_title(r"Monte Carlo Approximation of $\pi$: Error vs $N$", fontsize=14, pad=12)
    ax.set_xlabel("Number of Samples ($N$)", fontsize=12)
    ax.set_ylabel(r"Absolute Error $|\hat{\pi} - \pi|$", fontsize=12)
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend(fontsize=11)
    fig.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close(fig)
    print(f"Error plot saved to {output_file}")


def plot_speed_comparison(
    n_values: list[int],
    loop_times: list[float],
    numpy_times: list[float],
    output_file: str = "speed_numpy_vs_loops.png",
) -> None:
    """Plot execution times and speedup factor comparing loops vs NumPy."""
    speedups = [t_loop / t_np for t_loop, t_np in zip(loop_times, numpy_times)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left plot: Execution time vs N (Log-Log)
    ax1.loglog(n_values, loop_times, marker="s", linestyle="-", color="#d62728", label="Python Loops")
    ax1.loglog(n_values, numpy_times, marker="o", linestyle="-", color="#1f77b4", label="NumPy (Vectorized)")
    ax1.set_title("Execution Time vs Sample Size ($N$)", fontsize=13, pad=10)
    ax1.set_xlabel("Number of Samples ($N$)", fontsize=11)
    ax1.set_ylabel("Execution Time (seconds)", fontsize=11)
    ax1.grid(True, which="both", ls=":", alpha=0.6)
    ax1.legend(fontsize=11)

    # Right plot: Speedup factor vs N
    ax2.semilogx(n_values, speedups, marker="^", linestyle="-", color="#2ca02c", label=r"Speedup ($T_{\mathrm{loops}} / T_{\mathrm{numpy}}$)")
    ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.7, label="Parity (1x)")
    ax2.set_title("NumPy Speedup Factor over Loops", fontsize=13, pad=10)
    ax2.set_xlabel("Number of Samples ($N$)", fontsize=11)
    ax2.set_ylabel("Speedup Multiplier (x)", fontsize=11)
    ax2.grid(True, which="both", ls=":", alpha=0.6)
    ax2.legend(fontsize=11)

    fig.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close(fig)
    print(f"Speed comparison plot saved to {output_file}")


def main():
    seed = 42
    rng = np.random.default_rng(seed)

    n_values = [10**k for k in range(2, 8)]  # 10^2 to 10^7
    pi_estimates = []
    errors = []
    loop_times = []
    numpy_times = []

    print(f"{'N':>10} | {'Loop Time [s]':>14} | {'NumPy Time [s]':>15} | {'Speedup':>10} | {'Est Pi (NumPy)':>15} | {'Abs Error':>12}")
    print("-" * 88)

    for n in n_values:
        # Benchmark Python loops (multiple runs for smaller N for accurate measurement)
        repeat_loops = 5 if n <= 10**4 else 1
        t_loop = benchmark_function(estimate_pi_loops, n, seed, repeat=repeat_loops)
        loop_times.append(t_loop)

        # Benchmark NumPy (vectorized)
        repeat_numpy = 5 if n <= 10**4 else 1
        t_numpy = benchmark_function(estimate_pi_numpy, n, rng, repeat=repeat_numpy)
        numpy_times.append(t_numpy)

        # Estimate pi and calculate error using NumPy
        est_pi = estimate_pi_numpy(n, rng)
        err = abs(est_pi - math.pi)
        pi_estimates.append(est_pi)
        errors.append(err)

        speedup = t_loop / t_numpy
        print(f"{n:10.0e} | {t_loop:14.6f} | {t_numpy:15.6f} | {speedup:9.2f}x | {est_pi:15.7f} | {err:12.6e}")

    # Generate both figures
    plot_errors(n_values, errors, output_file="errors.png")
    plot_speed_comparison(n_values, loop_times, numpy_times, output_file="speed_numpy_vs_loops.png")


if __name__ == "__main__":
    main()
