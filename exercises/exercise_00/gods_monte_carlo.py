"""gods_monte_carlo.py

Faster-than-1/sqrt(N) Monte Carlo approximation of Pi.

Theoretical Background:
-----------------------
1. Standard Pseudo-Random Monte Carlo:
   By the Central Limit Theorem (CLT), independent uniform sampling in [0, 1]^2
   yields an error standard deviation scaling strictly as:
       Error ~ O(N^(-1/2)) = O(1 / sqrt(N))

2. Stratified Sampling (Jittered Grid):
   We partition the unit square [0, 1]^2 into a K x K grid (with K = floor(sqrt(N)),
   total samples N_actual = K^2). Exactly one point is sampled with random jitter
   inside each cell [i/K, (i+1)/K] x [j/K, (j+1)/K].
   - Cells completely inside the circle (r_max <= 1) contribute 1 with ZERO variance.
   - Cells completely outside the circle (r_min > 1) contribute 0 with ZERO variance.
   - Only the cells intersecting the boundary circle curve contribute variance.
   The perimeter length is finite (pi / 2), so only O(K) = O(sqrt(N)) cells intersect
   the boundary.
   Sum of variances: Var(sum) ~ O(sqrt(N))
   Variance of estimator: Var(pi_hat) = (4/N)^2 * Var(sum) ~ O(N^(-3/2))
   Root-Mean-Squared Error (RMSE):
       RMSE ~ sqrt(Var) ~ O(N^(-3/4)) = O(N^(-0.75))
   This provably beats the standard O(N^(-0.50)) convergence rate!

3. Quasi-Monte Carlo (Sobol Sequence):
   Uses deterministic low-discrepancy points. While smooth integrands achieve
   O(log(N)^2 / N) via the Koksma-Hlawka inequality, the indicator function of
   the quarter circle has a curved boundary discontinuity, which limits QMC
   convergence for discs in 2D to approximately O(N^(-3/4)).
"""

import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc


def estimate_pi_standard(n: int, rng: np.random.Generator) -> float:
    """Standard Monte Carlo with uniform pseudo-random sampling."""
    pts = rng.random((n, 2))
    inside = np.count_nonzero(pts[:, 0] * pts[:, 0] + pts[:, 1] * pts[:, 1] <= 1.0)
    return 4.0 * inside / n


def estimate_pi_stratified(n: int, rng: np.random.Generator) -> tuple[float, int]:
    """Stratified (jittered grid) Monte Carlo sampling.

    Divides [0, 1]^2 into a K x K grid where K = floor(sqrt(N)).
    Samples exactly 1 random jittered point in each cell.
    Returns (estimated_pi, actual_n).
    """
    k = math.isqrt(n)
    actual_n = k * k
    dx = 1.0 / k

    # Vectorized grid index generation
    i = np.arange(k)
    gx, gy = np.meshgrid(i, i, indexing="ij")

    # Add random uniform jitter within each sub-cell
    x = (gx + rng.random((k, k))) * dx
    y = (gy + rng.random((k, k))) * dx

    inside = np.count_nonzero(x * x + y * y <= 1.0)
    return 4.0 * inside / actual_n, actual_n


def estimate_pi_sobol(n: int, scramble: bool = True, seed: int | None = None) -> tuple[float, int]:
    """Quasi-Monte Carlo using Sobol low-discrepancy sequence.

    Uses base-2 point set 2^m where m = round(log2(N)) for optimal discrepancy.
    Returns (estimated_pi, actual_n).
    """
    m = max(1, int(round(np.log2(n))))
    actual_n = 2**m
    sampler = qmc.Sobol(d=2, scramble=scramble, seed=seed)
    pts = sampler.random_base2(m=m)
    inside = np.count_nonzero(pts[:, 0] * pts[:, 0] + pts[:, 1] * pts[:, 1] <= 1.0)
    return 4.0 * inside / actual_n, actual_n


def run_numerical_verification():
    """Rigorously verify empirical convergence rates across multiple trials."""
    target_ns = [10**k for k in range(2, 7)]  # 10^2 up to 10^6
    n_trials = 25

    std_rmses = []
    strat_rmses = []
    sobol_rmses = []
    strat_ns = []
    sobol_ns = []

    print("=" * 80)
    print("NUMERICAL VERIFICATION: RMSE across 25 independent trials")
    print("=" * 80)
    print(f"{'Target N':>10} | {'Standard MC RMSE':>18} | {'Stratified RMSE':>18} | {'Sobol QMC RMSE':>18}")
    print("-" * 80)

    for n in target_ns:
        # 1. Standard MC
        errs_std = []
        for t in range(n_trials):
            rng = np.random.default_rng(10_000 + t)
            est = estimate_pi_standard(n, rng)
            errs_std.append((est - math.pi) ** 2)
        std_rmse = math.sqrt(float(np.mean(errs_std)))
        std_rmses.append(std_rmse)

        # 2. Stratified MC
        errs_strat = []
        actual_strat_n = n
        for t in range(n_trials):
            rng = np.random.default_rng(20_000 + t)
            est, actual_strat_n = estimate_pi_stratified(n, rng)
            errs_strat.append((est - math.pi) ** 2)
        strat_rmse = math.sqrt(float(np.mean(errs_strat)))
        strat_rmses.append(strat_rmse)
        strat_ns.append(actual_strat_n)

        # 3. Sobol QMC
        errs_sobol = []
        actual_sobol_n = n
        for t in range(n_trials):
            est, actual_sobol_n = estimate_pi_sobol(n, scramble=True, seed=30_000 + t)
            errs_sobol.append((est - math.pi) ** 2)
        sobol_rmse = math.sqrt(float(np.mean(errs_sobol)))
        sobol_rmses.append(sobol_rmse)
        sobol_ns.append(actual_sobol_n)

        print(f"{n:10.0e} | {std_rmse:18.6e} | {strat_rmse:18.6e} | {sobol_rmse:18.6e}")

    # Power law regressions: log(RMSE) = -alpha * log(N) + const
    slope_std, _ = np.polyfit(np.log(target_ns), np.log(std_rmses), 1)
    slope_strat, _ = np.polyfit(np.log(strat_ns), np.log(strat_rmses), 1)
    slope_sobol, _ = np.polyfit(np.log(sobol_ns), np.log(sobol_rmses), 1)

    print("-" * 80)
    print("FITTED EMPIRICAL CONVERGENCE EXPONENTS (RMSE ~ N^(-alpha)):")
    print(f"  Standard Pseudo-Random MC: alpha = {-slope_std:.3f}   (Theory: 0.500 = 1/sqrt(N))")
    print(f"  Stratified Sampling (Grid): alpha = {-slope_strat:.3f}   (Theory: 0.750 = N^(-3/4))")
    print(f"  Sobol Quasi-Monte Carlo:   alpha = {-slope_sobol:.3f}   (Theory: ~0.750)")
    print("=" * 80)

    # Plot convergence comparison
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    ax.loglog(target_ns, std_rmses, "s-", color="#d62728", lw=1.8, label=f"Standard MC (fit: $\\sim N^{{{slope_std:.2f}}}$)")
    ax.loglog(strat_ns, strat_rmses, "o-", color="#1f77b4", lw=1.8, label=f"Stratified Grid (fit: $\\sim N^{{{slope_strat:.2f}}}$)")
    ax.loglog(sobol_ns, sobol_rmses, "^-", color="#2ca02c", lw=1.8, label=f"Sobol QMC (fit: $\\sim N^{{{slope_sobol:.2f}}}$)")

    # Reference asymptotes
    ref_n = np.array([target_ns[0], target_ns[-1]])
    ref_half = std_rmses[0] * np.sqrt(ref_n[0]) / np.sqrt(ref_n)
    ref_three_fourth = strat_rmses[0] * (ref_n[0] ** 0.75) / (ref_n**0.75)

    ax.loglog(ref_n, ref_half, "k--", alpha=0.6, label=r"Theory: $\mathcal{O}(N^{-1/2})$")
    ax.loglog(ref_n, ref_three_fourth, "k:", alpha=0.6, label=r"Theory: $\mathcal{O}(N^{-3/4})$")

    ax.set_title(r"Monte Carlo Convergence Rates: Standard vs Faster Methods", fontsize=13, pad=12)
    ax.set_xlabel(r"Sample Count $N$", fontsize=11)
    ax.set_ylabel(r"Root-Mean-Squared Error (RMSE)", fontsize=11)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=10)
    fig.tight_layout()

    out_plot = "gods_monte_carlo_convergence.png"
    plt.savefig(out_plot, dpi=300)
    plt.close(fig)
    print(f"Convergence comparison graphic saved to {out_plot}")


def main():
    run_numerical_verification()


if __name__ == "__main__":
    main()

