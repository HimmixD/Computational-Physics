"""Compare a naive DFT with NumPy's FFT.

The default signal is the triangular wave from exercise_01.pdf.  Running this
file creates four figures in ``exercise_01_output``:

* the DFT amplitudes and all Fourier components,
* coefficient and reconstruction errors relative to ``numpy.fft.fft``,
* execution-time scaling,
* sampling and observation-window effects compared with the analytic
  Fourier series of the triangle wave.

Examples
--------
    python dft_vs_numpyfft.py
    python dft_vs_numpyfft.py --signal sine --output-dir results
    python dft_vs_numpyfft.py --signal gaussian --show
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


def dft(signal: np.ndarray) -> np.ndarray:
    """Return the forward DFT using the definition from the exercise."""
    values = np.asarray(signal, dtype=complex)
    if values.ndim != 1:
        raise ValueError("signal must be a one-dimensional array")
    n = values.size
    if n == 0:
        raise ValueError("signal must contain at least one sample")
    indices = np.arange(n)
    kernel = np.exp(-2j * np.pi * np.outer(indices, indices) / n)
    return kernel @ values


def inverse_dft(coefficients: np.ndarray) -> np.ndarray:
    """Return the inverse DFT using the definition from the exercise."""
    values = np.asarray(coefficients, dtype=complex)
    if values.ndim != 1:
        raise ValueError("coefficients must be a one-dimensional array")
    n = values.size
    if n == 0:
        raise ValueError("coefficients must contain at least one value")
    indices = np.arange(n)
    kernel = np.exp(2j * np.pi * np.outer(indices, indices) / n)
    return (kernel @ values) / n


def triangle_signal(t: np.ndarray, period: float = 2e-3) -> np.ndarray:
    """Triangle wave matching the first graphic in exercise_01.pdf."""
    phase = np.mod(t, period) / period
    return np.where(phase <= 0.5, 1.0 - 4.0 * phase, -3.0 + 4.0 * phase)


def sample_signal(name: str, t: np.ndarray) -> np.ndarray:
    """Return one of the signals used in the optional signal comparison."""
    if name == "triangle":
        return triangle_signal(t)
    if name == "sine":
        return np.sin(2 * np.pi * 1_000 * t)
    if name == "rectangle":
        return np.where(np.mod(t, 2e-3) < 1e-3, 1.0, -1.0)
    if name == "sawtooth":
        return 2.0 * np.mod(t, 2e-3) / 2e-3 - 1.0
    if name == "gaussian":
        return np.exp(-((t - 4e-3) / 0.7e-3) ** 2)
    raise ValueError(f"unknown signal {name!r}")


def frequency_axis(n: int, dt: float) -> np.ndarray:
    """Return frequencies in cycles per second, ordered for plotting."""
    return np.fft.fftshift(np.fft.fftfreq(n, d=dt))


def benchmark(function: Callable[[np.ndarray], np.ndarray], values: np.ndarray, repeats: int) -> float:
    """Return the fastest of several timed calls."""
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        function(values)
        timings.append(time.perf_counter() - start)
    return min(timings)


def plot_dft_results(
    t: np.ndarray,
    signal: np.ndarray,
    coefficients: np.ndarray,
    output_file: Path,
    show: bool,
) -> None:
    n = signal.size
    dt = t[1] - t[0]
    frequencies = frequency_axis(n, dt)
    shifted = np.fft.fftshift(coefficients) / n
    reconstructed = inverse_dft(coefficients).real
    amplitude = np.abs(shifted)
    positive = frequencies > 0
    fit_mask = positive & (amplitude > max(amplitude) * 1e-4)
    fit_exponent, fit_intercept = np.polyfit(
        np.log(frequencies[fit_mask]), np.log(amplitude[fit_mask]), 1
    )
    fit_amplitude = np.exp(fit_intercept) * frequencies[positive] ** fit_exponent

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].loglog(frequencies[positive], amplitude[positive], "o", ms=4, label="DFT amplitudes")
    axes[0].loglog(
        frequencies[positive],
        fit_amplitude,
        "--",
        label=fr"fit: $|a(f)| \propto f^{{{fit_exponent:.2f}}}$",
    )
    axes[0].set(
        xlabel="frequency [Hz]",
        ylabel=r"$|a_k|$",
        title="DFT amplitudes and decay fit",
    )
    axes[0].legend()

    sample_indices = np.arange(n)
    for k in range(n):
        component = (
            coefficients[k] * np.exp(2j * np.pi * k * sample_indices / n) / n
        ).real
        axes[1].plot(t * 1e3, component, color="tab:blue", alpha=0.7, lw=0.8)
    axes[1].plot(t * 1e3, signal, "ko", ms=3, label="original samples")
    axes[1].plot(t * 1e3, reconstructed, color="tab:red", lw=1.5, label="sum of all components")
    axes[1].set(
        xlabel="time [ms]",
        ylabel="signal",
        title="All Fourier components and original signal",
    )
    axes[1].legend()
    for axis in axes:
        axis.grid(True, ls=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def plot_fft_comparison(
    coefficients: np.ndarray,
    numpy_coefficients: np.ndarray,
    signal: np.ndarray,
    output_file: Path,
    show: bool,
) -> None:
    coefficient_error = np.abs(coefficients - numpy_coefficients)
    reconstruction_error = np.abs(inverse_dft(coefficients) - np.fft.ifft(numpy_coefficients))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].semilogy(np.arange(signal.size), coefficient_error + 1e-18, "o", ms=3)
    axes[0].set(xlabel="coefficient index $k$", ylabel=r"$|A_k^{DFT}-A_k^{FFT}|$",
                title="DFT versus NumPy FFT")
    axes[1].semilogy(np.arange(signal.size), reconstruction_error + 1e-18, "o", ms=3)
    axes[1].set(xlabel="sample index $j$", ylabel="absolute error", title="Inverse-transform error")
    for axis in axes:
        axis.grid(True, ls=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def plot_timing(output_file: Path, show: bool) -> None:
    sizes = np.array([16, 32, 64, 128, 256, 512])
    dft_times = []
    fft_times = []
    for n in sizes:
        values = np.sin(2 * np.pi * np.arange(n) / n)
        repeats = 3 if n <= 128 else 1
        dft_times.append(benchmark(dft, values, repeats))
        fft_times.append(benchmark(np.fft.fft, values, repeats))
    dft_times = np.asarray(dft_times)
    fft_times = np.asarray(fft_times)

    print("\nTiming comparison")
    print(f"{'N':>6} {'naive DFT [s]':>16} {'numpy FFT [s]':>16} {'speedup':>12}")
    for n, dft_time, fft_time in zip(sizes, dft_times, fft_times):
        print(f"{n:6d} {dft_time:16.6e} {fft_time:16.6e} {dft_time / fft_time:12.2f}x")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].loglog(sizes, dft_times, "o-", label="naive DFT")
    axes[0].loglog(sizes, fft_times, "s-", label="NumPy FFT")
    axes[0].set(xlabel="system size $N$", ylabel="time [s]", title="Execution time")
    axes[0].legend()
    axes[1].semilogx(sizes, dft_times / fft_times, "o-")
    axes[1].set(xlabel="system size $N$", ylabel="speedup", title="FFT speedup")
    for axis in axes:
        axis.grid(True, which="both", ls=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def triangle_fourier_coefficient(harmonic: np.ndarray) -> np.ndarray:
    """Analytic complex Fourier coefficients of the unit triangle wave."""
    harmonic = np.asarray(harmonic, dtype=int)
    result = np.zeros(harmonic.shape, dtype=float)
    nonzero = (harmonic != 0) & (harmonic % 2 != 0)
    result[nonzero] = 8.0 / (np.pi**2 * harmonic[nonzero] ** 2)
    return result


def sampled_analytic_coefficients(n: int, periods: int, aliases: int = 1000) -> np.ndarray:
    """Analytic DFT coefficients including sampling aliases.

    For an observation window containing ``periods`` exact triangle periods,
    DFT bin k represents harmonic k/periods.  Sampling folds all harmonics
    separated by N/periods onto that bin.
    """
    bins = np.arange(n)
    harmonic = bins / periods
    result = np.zeros(n, dtype=float)
    valid = np.isclose(harmonic, np.rint(harmonic))
    integer_harmonic = np.rint(harmonic[valid]).astype(int)
    alias_step = n // periods
    for alias in range(-aliases, aliases + 1):
        result[valid] += triangle_fourier_coefficient(integer_harmonic + alias * alias_step)
    return result


def plot_analytic_comparison(output_file: Path, show: bool) -> None:
    period = 2e-3
    samples_per_period = np.array([8, 16, 32, 64, 128])
    periods = np.array([1e-3, 2e-3, 4e-3, 8e-3])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    for samples in samples_per_period:
        dt = period / samples
        n = samples
        t = np.arange(n) * dt
        numerical = np.fft.fft(triangle_signal(t, period=period)) / n
        positive = np.arange(1, n // 2)
        analytic = triangle_fourier_coefficient(positive)
        axes[0].plot(
            positive / (n / 2),
            np.abs(numerical[positive] - analytic),
            "o",
            ms=3,
            label=fr"$\Delta t={dt * 1e6:.1f}\,\mu$s",
        )
    axes[0].set(xlabel=r"frequency / Nyquist frequency", ylabel="absolute coefficient error",
                title=r"Changing $\Delta t$ at fixed period time $T=2$ ms")

    dt = period / 64
    for period_time in periods:
        n = int(round(period_time / dt))
        t = np.arange(n) * dt
        numerical = np.fft.fft(triangle_signal(t, period=period_time)) / n
        positive = np.arange(1, n // 2)
        analytic = triangle_fourier_coefficient(positive)
        axes[1].plot(
            positive / (n / 2),
            np.abs(numerical[positive] - analytic),
            "o",
            ms=3,
            label=fr"$T={period_time * 1e3:.1f}$ ms",
        )
    axes[1].set(xlabel=r"frequency / Nyquist frequency", ylabel="absolute coefficient error",
                title=fr"Changing period time $T$ at fixed $\Delta t={dt * 1e6:.1f}\,\mu$s")
    for axis in axes:
        axis.grid(True, ls=":", alpha=0.6)
        axis.legend()
    fig.suptitle(
        "Analytic triangle-wave coefficients versus sampled FFT",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--signal",
        choices=("triangle", "sine", "rectangle", "sawtooth", "gaussian"),
        default="triangle",
        help="signal to transform (default: triangle from the exercise PDF)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("exercise_01_output"))
    parser.add_argument("--show", action="store_true", help="display figures interactively")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    n = 256
    total_time = 8e-3
    t = np.arange(n) * total_time / n
    signal = sample_signal(args.signal, t)
    coefficients = dft(signal)
    numpy_coefficients = np.fft.fft(signal)

    print(f"Signal: {args.signal}, N={n}, dt={t[1] - t[0]:.3e} s")
    print(f"Maximum DFT/FFT coefficient error: {np.max(np.abs(coefficients - numpy_coefficients)):.3e}")
    print(f"Maximum inverse-DFT reconstruction error: {np.max(np.abs(inverse_dft(coefficients).real - signal)):.3e}")
    print(
        "Near-Nyquist errors grow because a sampled spectrum is periodic in frequency: "
        "unresolved higher harmonics fold back (alias) into bins near the Nyquist limit. "
        "Reducing dt raises the Nyquist frequency; increasing T improves frequency resolution."
    )

    plot_dft_results(t, signal, coefficients, args.output_dir / "dft_results.png", args.show)
    plot_fft_comparison(
        coefficients,
        numpy_coefficients,
        signal,
        args.output_dir / "dft_vs_numpyfft_error.png",
        args.show,
    )
    plot_timing(args.output_dir / "dft_vs_numpyfft_timing.png", args.show)
    plot_analytic_comparison(args.output_dir / "analytic_sampling_effects.png", args.show)
    print(f"Figures saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
