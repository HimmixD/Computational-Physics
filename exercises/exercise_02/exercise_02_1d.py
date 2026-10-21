"""Exercise 2, task 1: one-dimensional demagnetizing field via FFT convolution.

The source and evaluation cells lie on the x-axis and have unit spacing.  The
point-dipole expression gives K_xx(x) = 1 / (2*pi*|x|**3) away from x=0.
The singular self-interaction is excluded here (K_xx(0) = 0); a finite-cell
demagnetizing factor would be a separate physical model.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def dipole_kernel_1d(size: int, *, dtype: np.dtype = np.float64) -> np.ndarray:
    """Return the centred 1D kernel K_xx for integer cell displacements."""
    if size < 1:
        raise ValueError("size must be positive")
    distances = np.arange(size, dtype=dtype)
    kernel = np.zeros(size, dtype=dtype)
    nonzero = distances != 0
    kernel[nonzero] = 1.0 / (2.0 * np.pi * distances[nonzero] ** 3)
    return kernel


def direct_convolution(magnetization: np.ndarray) -> np.ndarray:
    """Compute h_i = sum_j K_xx(i-j) m_j in O(N^2) time."""
    values = np.asarray(magnetization, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("magnetization must be a non-empty one-dimensional array")
    indices = np.arange(values.size)
    distances = np.abs(indices[:, None] - indices[None, :])
    kernel = np.zeros_like(distances, dtype=float)
    nonzero = distances != 0
    kernel[nonzero] = 1.0 / (2.0 * np.pi * distances[nonzero] ** 3)
    return kernel @ values


def next_fast_length(target: int) -> int:
    """Return a small 2/3/5-smooth length suitable for NumPy's FFT."""
    if target < 1:
        raise ValueError("target must be positive")
    length = target
    while True:
        candidate = length
        for factor in (2, 3, 5):
            while candidate % factor == 0:
                candidate //= factor
        if candidate == 1:
            return length
        length += 1


def _fft_kernel_spectrum(size: int, fft_size: int) -> np.ndarray:
    """Arrange the symmetric kernel for a linear convolution and transform it."""
    if fft_size < 2 * size - 1:
        raise ValueError("fft_size must be at least 2*size - 1")
    kernel = np.zeros(fft_size, dtype=float)
    values = dipole_kernel_1d(size)
    kernel[:size] = values
    if size > 1:
        kernel[-(size - 1) :] = values[:0:-1]
    return np.fft.rfft(kernel)


def fft_convolution(
    magnetization: np.ndarray,
    *,
    fft_size: int | None = None,
    kernel_spectrum: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the same linear convolution using zero-padded real FFTs."""
    values = np.asarray(magnetization, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("magnetization must be a non-empty one-dimensional array")
    size = values.size
    if fft_size is None:
        fft_size = next_fast_length(2 * size - 1)
    if fft_size < 2 * size - 1:
        raise ValueError("fft_size must be at least 2*size - 1")
    spectrum = _fft_kernel_spectrum(size, fft_size) if kernel_spectrum is None else kernel_spectrum
    padded = np.zeros(fft_size, dtype=float)
    padded[:size] = values
    return np.fft.irfft(np.fft.rfft(padded) * spectrum, n=fft_size)[:size]


def optimized_fft_convolution(
    magnetization: np.ndarray,
    kernel_spectrum: np.ndarray,
    fft_size: int,
) -> np.ndarray:
    """FFT convolution for repeated calls with a precomputed kernel spectrum."""
    values = np.asarray(magnetization, dtype=float)
    padded = np.zeros(fft_size, dtype=float)
    padded[: values.size] = values
    return np.fft.irfft(np.fft.rfft(padded) * kernel_spectrum, n=fft_size)[: values.size]


def dipole_field_xz(
    x: np.ndarray, z: np.ndarray, moment: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return (h_x, h_z) of an x-directed point dipole in the x-z plane."""
    coordinates_x, coordinates_z = np.meshgrid(
        np.asarray(x, dtype=float), np.asarray(z, dtype=float), indexing="xy"
    )
    radius_squared = coordinates_x**2 + coordinates_z**2
    field_x = np.full_like(radius_squared, np.nan)
    field_z = np.full_like(radius_squared, np.nan)
    valid = radius_squared > 0.0
    radius = np.sqrt(radius_squared[valid])
    field_x[valid] = moment * (3.0 * coordinates_x[valid] ** 2 / radius_squared[valid] - 1.0) / (
        4.0 * np.pi * radius**3
    )
    field_z[valid] = moment * 3.0 * coordinates_x[valid] * coordinates_z[valid] / (
        4.0 * np.pi * radius**5
    )
    return field_x, field_z


def benchmark(function, values: np.ndarray, repeats: int = 5) -> float:
    """Return the best runtime in seconds after one warm-up call."""
    function(values)
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        function(values)
        timings.append(time.perf_counter() - start)
    return min(timings)


def create_figures(output_dir: Path) -> list[Path]:
    """Create plots for tasks 1a--1e and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    x = np.arange(-16, 17)
    field = np.full(x.shape, np.nan, dtype=float)
    nonzero = x != 0
    field[nonzero] = 1.0 / (2.0 * np.pi * np.abs(x[nonzero]) ** 3)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.stem(x, field, basefmt=" ", label=r"$K_{xx}(x)=1/(2\pi |x|^3)$")
    axis.set(xlabel="x [cell units]", ylabel=r"$h_x$ for $m_x=1$", title="1a: field of one x-directed dipole")
    axis.grid(True, ls=":", alpha=0.6)
    axis.legend()
    path = output_dir / "task_1a_single_dipole_kernel.png"
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)

    coordinates = np.linspace(-4.0, 4.0, 25)
    field_x, field_z = dipole_field_xz(coordinates, coordinates)
    magnitude = np.hypot(field_x, field_z)
    direction_x = field_x / magnitude
    direction_z = field_z / magnitude
    figure, axis = plt.subplots(figsize=(7, 6))
    color_values = np.log10(magnitude)
    image = axis.contourf(
        coordinates,
        coordinates,
        color_values,
        levels=20,
        cmap="viridis",
        alpha=0.82,
    )
    axis.quiver(
        *np.meshgrid(coordinates, coordinates),
        direction_x,
        direction_z,
        color="white",
        pivot="mid",
        scale=18,
        width=0.004,
    )
    axis.scatter(0.0, 0.0, marker="o", color="crimson", s=45, label="dipole at origin")
    figure.colorbar(image, ax=axis, label=r"$\log_{10}|\vec h|$")
    axis.set(
        xlabel="x [cell units]",
        ylabel="z [cell units]",
        title="1a: x-z slice of the point-dipole field",
        aspect="equal",
    )
    axis.legend(loc="upper right")
    axis.grid(True, ls=":", alpha=0.35, color="white")
    path = output_dir / "task_1a_single_dipole_vector_field_xz.png"
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
    paths.append(path)

    size = 128
    magnetization = np.zeros(size)
    magnetization[size // 4 : 3 * size // 4] = 1.0
    direct = direct_convolution(magnetization)
    fft = fft_convolution(magnetization)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(magnetization, label=r"$m_x$")
    axis.plot(direct, label="direct sum")
    axis.plot(fft, "--", label="FFT convolution")
    axis.set(xlabel="cell index", ylabel=r"$h_x$", title="1b/1c: field of a finite magnetized segment")
    axis.grid(True, ls=":", alpha=0.6)
    axis.legend()
    path = output_dir / "task_1b_1c_segment_field.png"
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)

    sizes = np.array([32, 64, 128, 256, 512, 1024, 2048])
    direct_times = []
    fft_times = []
    optimized_times = []
    for n in sizes:
        values = np.ones(n)
        direct_times.append(benchmark(direct_convolution, values, repeats=3))
        fft_size = next_fast_length(2 * n - 1)
        spectrum = _fft_kernel_spectrum(n, fft_size)
        fft_times.append(benchmark(fft_convolution, values, repeats=3))
        optimized_times.append(
            benchmark(lambda v: optimized_fft_convolution(v, spectrum, fft_size), values, repeats=3)
        )
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.loglog(sizes, direct_times, "o-", label="direct O(N²)")
    axis.loglog(sizes, fft_times, "o-", label="FFT, setup included")
    axis.loglog(sizes, optimized_times, "o-", label="optimized real FFT")
    axis.set(xlabel="N", ylabel="best runtime [s]", title="1d/1e: convolution timing")
    axis.grid(True, which="both", ls=":", alpha=0.6)
    axis.legend()
    path = output_dir / "task_1d_1e_timing.png"
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)

    return paths


def write_report(output_dir: Path) -> Path:
    """Run the numerical comparisons and write the reviewable task output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sizes = np.array([32, 64, 128, 256, 512, 1024, 2048])
    rows = []
    for n in sizes:
        values = np.ones(n)
        direct = direct_convolution(values)
        fft = fft_convolution(values)
        fft_size = next_fast_length(2 * n - 1)
        spectrum = _fft_kernel_spectrum(n, fft_size)
        optimized = optimized_fft_convolution(values, spectrum, fft_size)
        rows.append(
            (
                n,
                fft_size,
                benchmark(direct_convolution, values, repeats=3),
                benchmark(fft_convolution, values, repeats=3),
                benchmark(lambda v: optimized_fft_convolution(v, spectrum, fft_size), values, repeats=3),
                np.max(np.abs(direct - fft)),
                np.max(np.abs(direct - optimized)),
            )
        )

    lines = [
        "# Exercise 2 - Task 1: Demagnetizing field in 1D via FFT convolution",
        "",
        "All cells have unit spacing. The point-dipole field is",
        r"$\vec h(\vec r)=\frac{1}{4\pi r^3}\left(3(\vec m\cdot\hat r)\hat r-\vec m\right)$.",
        "For `m = m_x e_x` and `r = x e_x`, this gives",
        r"$h_x(x)=K_{xx}(x)m_x=\frac{m_x}{2\pi |x|^3}$ for $x\ne0$.",
        "",
        "## 1a - Single dipole and self-interaction",
        "",
        "On the x-axis the field is positive and even: it points in the +x direction on both sides of an x-directed dipole, and its magnitude decays as |x|^-3. The point-dipole model is singular at x=0. In this implementation the self term is excluded, `K_xx(0)=0`, so a cell does not act on itself. A nonzero finite-cell self-demagnetizing factor would require replacing the point kernel by a volume-averaged cell kernel.",
        "",
        "![Single-dipole kernel](task_1a_single_dipole_kernel.png)",
        "",
        "The vector field itself is shown below in the x-z plane. Arrow directions are normalized for visibility; the color encodes the logarithm of the field magnitude. The point at the origin is omitted from the formula because the point-dipole field is singular there.",
        "",
        "![Single-dipole vector field](task_1a_single_dipole_vector_field_xz.png)",
        "",
        "## 1b - Direct summation",
        "",
        r"$h_i=\sum_{j\ne i}\frac{m_j}{2\pi |i-j|^3}$ is evaluated directly.",
        "At the segment ends the field is smaller than in the middle because one side has no magnetized source cells. Near the middle, contributions from both sides add and the field is approximately symmetric and nearly constant away from the ends. The exact values still show edge effects because the kernel is long-ranged.",
        "",
        "## 1c - Linear convolution with the FFT",
        "",
        r"The convolution theorem is $\hat h_k=\hat K_k\hat m_k$.",
        "A DFT computes a circular convolution. To obtain the desired linear convolution, both arrays are embedded in arrays of length L >= 2N-1 and padded with zeros. The kernel is arranged by displacement: `K[0]=0`, `K[d]=1/(2*pi*d^3)` for d=1,...,N-1, and `K[L-d]=K[d]`. This places positive and negative displacements at their wrapped FFT indices and prevents wrap-around terms from entering the first N output cells.",
        "",
        "## 1d - Direct versus FFT scaling",
        "",
        "The direct algorithm uses O(N²) pair interactions. The FFT method uses O(L log L) operations with L approximately 2N. Therefore FFT convolution becomes increasingly faster as N grows. N does not need to be a power of two: NumPy supports arbitrary lengths, although lengths with small prime factors are usually faster. The implementation selects a 2/3/5-smooth length at least 2N-1.",
        "",
        "| N | FFT length | direct [s] | FFT [s] | optimized FFT [s] | max direct-FFT error | max direct-optimized error |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {n} | {length} | {direct_time:.3e} | {fft_time:.3e} | {optimized_time:.3e} | {error:.3e} | {optimized_error:.3e} |"
        for n, length, direct_time, fft_time, optimized_time, error, optimized_error in rows
    )
    lines.extend(
        [
            "",
            "![Timing benchmark](task_1d_1e_timing.png)",
            "",
            "## 1e - FFT optimization",
            "",
            "The optimized version uses real-input transforms (`rfft`/`irfft`), computes the symmetric kernel spectrum only once, reuses the spectrum for repeated magnetization arrays, and chooses a smooth FFT length. These changes reduce memory and avoid redundant work while preserving the linear-convolution result. For repeated evaluations with the same N, kernel-spectrum reuse is the most important algorithmic optimization; FFT planning would be an additional optimization in libraries that expose reusable plans.",
            "",
            "![Segment comparison](task_1b_1c_segment_field.png)",
        ]
    )
    path = output_dir / "exercise_02_task_1_output.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("exercise_02_task_1_output"))
    args = parser.parse_args()
    paths = create_figures(args.output_dir)
    report = write_report(args.output_dir)
    print(f"Wrote report: {report}")
    for path in paths:
        print(f"Wrote figure: {path}")


if __name__ == "__main__":
    main()
