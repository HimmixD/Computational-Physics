import math
import matplotlib.pyplot as plt
import numpy as np


def estimate_pi_monte_carlo(n: int, rng: np.random.Generator) -> float:
    """Estimate pi using n random points in the unit square [0, 1] x [0, 1].

    Points are counted inside the quarter circle if x^2 + y^2 <= 1.
    Since area(quarter_circle) / area(square) = pi / 4,
    pi is approximated by 4 * (inside / n).
    """
    # For large n (e.g. 10^7), process in chunks to keep memory usage low and fast
    chunk_size = 10_000_000
    points_inside = 0
    remaining = n

    while remaining > 0:
        current_chunk = min(remaining, chunk_size)
        x = rng.random(current_chunk)
        y = rng.random(current_chunk)
        points_inside += np.count_nonzero(x * x + y * y <= 1.0)
        remaining -= current_chunk

    return 4.0 * points_inside / n


def main():
    seed = 42
    rng = np.random.default_rng(seed)

    n_values = [10**k for k in range(2, 8)]  # 10^2 to 10^7
    pi_estimates = []
    errors = []

    print(f"{'N':>10} | {'Estimated Pi':>14} | {'Absolute Error':>15}")
    print("-" * 45)

    for n in n_values:
        est = estimate_pi_monte_carlo(n, rng)
        err = abs(est - math.pi)
        pi_estimates.append(est)
        errors.append(err)
        print(f"{n:10.0e} | {est:14.7f} | {err:15.7e}")

    # Plot error vs N on log-log scale
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(n_values, errors, marker="o", linestyle="-", color="#1f77b4", label="Monte Carlo Error $|\\hat{\\pi} - \\pi|$")

    # Add theoretical 1 / sqrt(N) reference line normalized to the first error point
    ref_scaling = errors[0] * np.sqrt(n_values[0])
    theory_line = [ref_scaling / np.sqrt(n) for n in n_values]
    ax.loglog(n_values, theory_line, linestyle="--", color="#d62728", alpha=0.8, label=r"Theoretical trend $\mathcal{O}(1/\sqrt{N})$")

    ax.set_title("Monte Carlo Approximation of $\\pi$: Error vs $N$", fontsize=14, pad=12)
    ax.set_xlabel("Number of Samples ($N$)", fontsize=12)
    ax.set_ylabel("Absolute Error $|\\hat{\\pi} - \\pi|$", fontsize=12)
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend(fontsize=11)
    fig.tight_layout()

    output_file = "errors.png"
    plt.savefig(output_file, dpi=300)
    plt.close(fig)
    print(f"\nPlot saved successfully to {output_file}")


if __name__ == "__main__":
    main()

