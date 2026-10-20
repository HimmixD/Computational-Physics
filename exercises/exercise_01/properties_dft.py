"""Numerical verification of the DFT properties from Exercise 2.

Running this file verifies the symmetry, shift theorem, Parseval identity,
Cooley--Tukey recursion, and a recursive radix-2 FFT.  It also creates plots
and a typeset PDF report in ``properties_dft_output``.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def dft(values: np.ndarray) -> np.ndarray:
    """Compute the direct DFT from its definition."""
    values = np.asarray(values, dtype=complex)
    n = values.size
    indices = np.arange(n)
    return np.exp(-2j * np.pi * np.outer(indices, indices) / n) @ values


def inverse_dft(coefficients: np.ndarray) -> np.ndarray:
    """Compute the inverse DFT matching the normalization used by ``dft``."""
    coefficients = np.asarray(coefficients, dtype=complex)
    n = coefficients.size
    if n == 0:
        raise ValueError("inverse_dft requires a non-empty array")
    indices = np.arange(n)
    return np.exp(2j * np.pi * np.outer(indices, indices) / n) @ coefficients / n


def recursive_fft(values: np.ndarray) -> np.ndarray:
    """Compute a radix-2 FFT recursively using even/odd decomposition."""
    values = np.asarray(values, dtype=complex)
    n = values.size
    if n == 0 or n & (n - 1):
        raise ValueError("recursive_fft requires a non-empty power-of-two length")
    if n == 1:
        return values.copy()
    even = recursive_fft(values[::2])
    odd = recursive_fft(values[1::2])
    twiddle = np.exp(-2j * np.pi * np.arange(n // 2) / n)
    upper = even + twiddle * odd
    lower = even - twiddle * odd
    return np.concatenate((upper, lower))


def cooley_tukey_terms(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return even DFT, odd DFT, and their recombined radix-2 result."""
    values = np.asarray(values, dtype=complex)
    even = dft(values[::2])
    odd = dft(values[1::2])
    twiddle = np.exp(-2j * np.pi * np.arange(values.size // 2) / values.size)
    combined = np.concatenate((even + twiddle * odd, even - twiddle * odd))
    return even, odd, combined


@dataclass
class Check:
    name: str
    error: float
    tolerance: float
    explanation: str

    @property
    def passed(self) -> bool:
        return self.error <= self.tolerance


def run_checks() -> tuple[list[Check], dict[str, np.ndarray | float]]:
    rng = np.random.default_rng(20261014)
    n = 32
    signal = rng.normal(size=n)
    coefficients = dft(signal)

    complex_signal = rng.normal(size=n) + 1j * rng.normal(size=n)
    complex_coefficients = dft(complex_signal)
    conjugated_signal_error = np.max(
        np.abs(dft(np.conjugate(complex_signal)) - np.conjugate(complex_coefficients[(-np.arange(n)) % n]))
    )
    symmetry_rhs = np.conjugate(coefficients[(-np.arange(n)) % n])
    symmetry_error = np.max(np.abs(coefficients - symmetry_rhs))

    shift = 7
    shifted_signal = np.roll(signal, shift)
    shifted_coefficients = dft(shifted_signal)
    shift_prediction = coefficients * np.exp(-2j * np.pi * np.arange(n) * shift / n)
    shift_error = np.max(np.abs(shifted_coefficients - shift_prediction))

    signal_energy = np.sum(np.abs(signal) ** 2)
    coefficient_energy = np.sum(np.abs(coefficients) ** 2) / n
    parseval_error = abs(signal_energy - coefficient_energy)
    inverse_error = np.max(np.abs(inverse_dft(coefficients) - signal))

    even, odd, combined = cooley_tukey_terms(signal)
    recursion_error = np.max(np.abs(combined - coefficients))

    recursive_result = recursive_fft(signal)
    fft_error = np.max(np.abs(recursive_result - np.fft.fft(signal)))

    checks = [
        Check(
            "Conjugation identity",
            float(conjugated_signal_error),
            1e-10,
            "For any signal, DFT(conjugate(a)) equals the conjugate of A at "
            "the reversed index (−k) modulo N.",
        ),
        Check(
            "Real-signal N-k symmetry",
            float(symmetry_error),
            1e-10,
            "For real samples, A_k = conjugate(A_(N−k) mod N); therefore positive "
            "and negative frequency bins contain conjugate information.",
        ),
        Check(
            "DFT / inverse DFT",
            float(inverse_error),
            1e-10,
            "Applying the inverse DFT to the DFT coefficients reconstructs the "
            "original signal sample by sample.",
        ),
        Check(
            "Parseval theorem",
            float(parseval_error),
            1e-10,
            "The sum of |aⱼ|² equals 1/N times the sum of |Aₖ|² for this DFT convention.",
        ),
        Check(
            "Cooley--Tukey recombination",
            float(recursion_error),
            1e-10,
            "The even and odd transforms combine as Aₖ = Eₖ + ωₙᵏOₖ and "
            "Aₖ₊ₙ⁄₂ = Eₖ − ωₙᵏOₖ.",
        ),
        Check(
            "Recursive radix-2 FFT",
            float(fft_error),
            1e-10,
            "The recursive implementation agrees with NumPy's FFT.",
        ),
    ]
    data = {
        "signal": signal,
        "coefficients": coefficients,
        "complex_signal": complex_signal,
        "complex_coefficients": complex_coefficients,
        "shifted_coefficients": shifted_coefficients,
        "shift_prediction": shift_prediction,
        "shift_error": float(shift_error),
        "even": even,
        "odd": odd,
        "combined": combined,
        "recursive_result": recursive_result,
        "signal_energy": signal_energy,
        "coefficient_energy": coefficient_energy,
        "inverse_signal": inverse_dft(coefficients),
    }
    return checks, data


def benchmark_fft(output_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sizes = 2 ** np.arange(3, 11)
    recursive_times = []
    numpy_times = []
    for n in sizes:
        values = np.sin(2 * np.pi * np.arange(n) / n)
        repeats = 5 if n <= 64 else 2
        recursive_times.append(min(_timed_call(recursive_fft, values) for _ in range(repeats)))
        numpy_times.append(min(_timed_call(np.fft.fft, values) for _ in range(repeats)))

    recursive_times = np.asarray(recursive_times)
    numpy_times = np.asarray(numpy_times)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].loglog(sizes, recursive_times, "o-", label="recursive radix-2 FFT")
    axes[0].loglog(sizes, numpy_times, "s-", label="NumPy FFT")
    axes[0].set(xlabel="N", ylabel="time [s]", title="FFT timing")
    axes[0].legend()
    axes[1].semilogx(sizes, recursive_times / numpy_times, "o-", color="#d95f02")
    axes[1].set(xlabel="N", ylabel="time ratio", title="Recursive / NumPy time")
    for axis in axes:
        axis.grid(True, which="both", ls=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_dir / "recursive_fft_timing.png", dpi=200)
    plt.close(fig)
    return sizes, recursive_times, numpy_times


def _timed_call(function, values: np.ndarray) -> float:
    start = time.perf_counter()
    function(values)
    return time.perf_counter() - start


def create_plots(output_dir: Path, data: dict[str, np.ndarray | float]) -> list[Path]:
    coefficients = np.asarray(data["coefficients"])
    signal = np.asarray(data["signal"])
    n = signal.size
    indices = np.arange(n)
    paths: list[Path] = []

    figure, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    axes[0, 0].semilogy(indices, np.abs(coefficients - np.conjugate(coefficients[(-indices) % n])) + 1e-18, "o")
    axes[0, 0].set(title="Conjugate-symmetry residual", xlabel="k", ylabel="absolute residual")
    axes[0, 1].plot(indices, np.abs(data["shifted_coefficients"]), "o", label="shifted signal")
    axes[0, 1].plot(indices, np.abs(data["shift_prediction"]), "--", label="phase-factor prediction")
    axes[0, 1].set(title="Shift theorem: magnitudes", xlabel="k", ylabel=r"$|A_k|$")
    axes[0, 1].legend()
    axes[1, 0].plot(indices, np.abs(data["combined"] - coefficients), "o")
    axes[1, 0].set(title="Cooley--Tukey residual", xlabel="k", ylabel="absolute residual")
    axes[1, 1].plot(indices, np.abs(data["recursive_result"] - np.fft.fft(signal)), "o")
    axes[1, 1].set(title="Recursive FFT residual", xlabel="k", ylabel="absolute residual")
    for axis in axes.flat:
        axis.grid(True, ls=":", alpha=0.6)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    figure.suptitle("DFT properties: numerical residuals", fontsize=14)
    figure.tight_layout()
    residual_path = output_dir / "dft_property_residuals.png"
    figure.savefig(residual_path, dpi=200)
    plt.close(figure)
    paths.append(residual_path)

    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(indices, np.abs(signal) ** 2, "o-", label=r"$|a_j|^2$")
    axis.plot(indices, np.abs(coefficients) ** 2 / n, "s--", label=r"$|A_j|^2/N$")
    axis.set(xlabel="index", ylabel="energy contribution", title="Parseval theorem: total energy is preserved")
    axis.legend()
    axis.grid(True, ls=":", alpha=0.6)
    parseval_path = output_dir / "parseval_energy.png"
    figure.tight_layout()
    figure.savefig(parseval_path, dpi=200)
    plt.close(figure)
    paths.append(parseval_path)
    return paths


def create_radix2_plot(output_dir: Path) -> Path:
    """Draw the three butterfly stages of an 8-point radix-2 FFT."""
    figure, axis = plt.subplots(figsize=(11, 5.2))
    stage_x = [0.0, 1.0, 2.0, 3.0]
    labels = [f"a{index}" for index in range(8)]
    y = np.arange(7, -1, -1)
    axis.scatter([stage_x[0]] * 8, y, color="#17365D", zorder=3)
    for position, label in zip(y, labels):
        axis.text(stage_x[0] - 0.12, position, label, ha="right", va="center")

    pairings = [
        [(0, 4), (1, 5), (2, 6), (3, 7)],
        [(0, 2), (1, 3), (4, 6), (5, 7)],
        [(0, 1), (2, 3), (4, 5), (6, 7)],
    ]
    for stage, pairs in enumerate(pairings):
        left_x, right_x = stage_x[stage], stage_x[stage + 1]
        axis.scatter([right_x] * 8, y, color="#4F6B82", zorder=3)
        for upper, lower in pairs:
            axis.plot([left_x, right_x], [y[upper], y[upper]], color="#AAB7C4")
            axis.plot([left_x, right_x], [y[lower], y[lower]], color="#AAB7C4")
            axis.plot([left_x, right_x], [y[upper], y[lower]], color="#D95F02", alpha=0.8)
            axis.plot([left_x, right_x], [y[lower], y[upper]], color="#D95F02", alpha=0.8)
            axis.text((left_x + right_x) / 2, (y[upper] + y[lower]) / 2 + 0.12,
                      "× ω", color="#D95F02", ha="center", fontsize=8)
        axis.text(right_x, 7.65, f"stage {stage + 1}", ha="center", fontweight="bold")
    for index, position in enumerate(y):
        axis.text(stage_x[-1] + 0.12, position, f"X{index}", ha="left", va="center")
    axis.set_xlim(-0.65, 3.65)
    axis.set_ylim(-0.7, 7.9)
    axis.axis("off")
    axis.set_title("Radix-2 FFT: even/odd butterfly recombination for N = 8", pad=12)
    figure.tight_layout()
    path = output_dir / "radix2_butterfly.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path


def build_pdf(
    output_dir: Path,
    checks: list[Check],
    data: dict[str, np.ndarray | float],
    plot_paths: list[Path],
    timing: tuple[np.ndarray, np.ndarray, np.ndarray],
    radix2_plot_path: Path,
) -> Path:
    pdf_path = output_dir / "properties_dft_report.pdf"
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, fontSize=22, leading=28, textColor=colors.HexColor("#17365D")))
    styles.add(ParagraphStyle(name="Subtitle", parent=styles["Normal"], alignment=TA_CENTER, fontSize=11, textColor=colors.HexColor("#4F6B82"), spaceAfter=18))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], textColor=colors.HexColor("#17365D"), spaceBefore=12, spaceAfter=7))
    styles.add(ParagraphStyle(name="BodySmall", parent=styles["BodyText"], fontSize=9.5, leading=13))
    formula_dir = output_dir / "formula_images"
    formula_dir.mkdir(exist_ok=True)
    formula_images = {
        "dft": _formula_image(
            r"A_k = \sum_{j=0}^{N-1} a_j\,e^{-2\pi i k j/N}",
            formula_dir / "dft_definition.png",
        ),
        "symmetry": _formula_image(
            r"A_k = A^*_{(-k)\,\mathrm{mod}\,N}",
            formula_dir / "symmetry.png",
        ),
        "parseval": _formula_image(
            r"\sum_j |a_j|^2 = \frac{1}{N}\sum_k |A_k|^2",
            formula_dir / "parseval.png",
        ),
        "recursion": _formula_image(
            r"A_k=E_k+\omega_N^k O_k,\qquad A_{k+N/2}=E_k-\omega_N^k O_k",
            formula_dir / "recursion.png",
        ),
        "inverse": _formula_image(
            r"a_j = \frac{1}{N}\sum_{k=0}^{N-1} A_k\,e^{2\pi i k j/N}",
            formula_dir / "inverse.png",
        ),
    }
    document = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title="Properties of the Discrete Fourier Transform",
        author="Computational Physics",
    )
    story = [
        Paragraph("Properties of the Discrete Fourier Transform", styles["TitleCenter"]),
        Paragraph("Exercise 2 numerical verification and recursive FFT report", styles["Subtitle"]),
        Paragraph(
            "This report checks the algebraic properties of the DFT on a reproducible "
            "real-valued test signal of length N=32. All calculations use the convention:",
            styles["BodyText"],
        ),
        Image(str(formula_images["dft"]), width=10 * cm, height=0.9 * cm),
        Spacer(1, 0.35 * cm),
        Paragraph("Numerical validation table", styles["Section"]),
    ]
    result_rows = [["Property", "Maximum error", "Tolerance", "Status"]]
    for check in checks:
        result_rows.append([check.name, f"{check.error:.3e}", f"{check.tolerance:.0e}", "PASS" if check.passed else "FAIL"])
    table = Table(result_rows, colWidths=[6.4 * cm, 3.2 * cm, 2.7 * cm, 2.0 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F2F6FA")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#AAB7C4")),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6FA")]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story += [table, Spacer(1, 0.3 * cm)]
    for check in checks:
        story.append(Paragraph(f"<b>{check.name}.</b> {check.explanation}", styles["BodySmall"]))
        story.append(Spacer(1, 0.12 * cm))

    story += [
        PageBreak(),
        Paragraph("Results and visual checks", styles["Section"]),
        Image(str(plot_paths[0]), width=17.6 * cm, height=13.2 * cm),
        Spacer(1, 0.2 * cm),
        Paragraph(
            "The residuals are at floating-point round-off level. The shift theorem changes "
            "phase but not spectral magnitude, which is why the two magnitude curves overlap.",
            styles["BodySmall"],
        ),
        Spacer(1, 0.25 * cm),
        Image(str(plot_paths[1]), width=17.6 * cm, height=8.8 * cm),
        Paragraph(
            f"Parseval check: signal energy = {float(data['signal_energy']):.8f}, "
            f"coefficient energy / N = {float(data['coefficient_energy']):.8f}.",
            styles["BodySmall"],
        ),
        Spacer(1, 0.2 * cm),
        Paragraph(
            "<b>Why are the triangle-wave coefficients purely real?</b> "
            "The sampled triangle is real and even when it is centred at the origin. "
            "In the DFT, the terms at +j and −j then pair to a cosine, so the sine "
            "parts cancel and A_k is real. If the same triangle is shifted by m samples, "
            "the shift theorem multiplies A_k by exp(−2πikm/N), adding a phase. "
            "Thus the purely real coefficients are a consequence of centring/evenness, "
            "not of the triangle shape alone. The numerical shift-theorem residual "
            f"for the test signal is {float(data['shift_error']):.3e}.",
            styles["BodySmall"],
        ),
        Spacer(1, 0.2 * cm),
        Paragraph("The DFT and its inverse", styles["Section"]),
        Image(str(formula_images["inverse"]), width=13 * cm, height=1.0 * cm),
        Paragraph(
            "The inverse contains the compensating factor 1/N. Numerically, the "
            "DFT/inverse-DFT round trip is included in the validation table above.",
            styles["BodySmall"],
        ),
        PageBreak(),
        Paragraph("Cooley--Tukey recursion and radix-2 FFT", styles["Section"]),
        Paragraph(
            "Splitting the samples into even and odd subsequences gives "
            "Eₖ = DFT(a₀, a₂, …) and Oₖ = DFT(a₁, a₃, …). With "
            "ωₙ = exp(−2πi/N), applying the identities below recursively "
            "produces the radix-2 FFT, reducing the operation count from "
            "O(N²) to O(N log₂ N).",
            styles["BodyText"],
        ),
        Image(str(formula_images["recursion"]), width=13 * cm, height=1.0 * cm),
        Spacer(1, 0.25 * cm),
        Image(str(radix2_plot_path), width=17.6 * cm, height=8.3 * cm),
        Paragraph(
            "Each butterfly combines an even and an odd subtransform. The upper "
            "output uses E + ωO and the lower output uses E − ωO; applying this "
            "operation recursively is exactly what the code below implements.",
            styles["BodySmall"],
        ),
        Preformatted(
            "def recursive_fft(values):\n"
            "    if len(values) == 1:\n"
            "        return values.copy()\n"
            "    even = recursive_fft(values[::2])\n"
            "    odd = recursive_fft(values[1::2])\n"
            "    twiddle = exp(-2j*pi*arange(N//2)/N)\n"
            "    upper = even + twiddle * odd\n"
            "    lower = even - twiddle * odd\n"
            "    return concatenate((upper, lower))",
            styles["Code"],
        ),
        Spacer(1, 0.25 * cm),
    ]
    sizes, recursive_times, numpy_times = timing
    timing_rows = [["N", "Recursive FFT [s]", "NumPy FFT [s]", "Ratio"]]
    for n, recursive_time, numpy_time in zip(sizes, recursive_times, numpy_times):
        timing_rows.append([str(n), f"{recursive_time:.3e}", f"{numpy_time:.3e}", f"{recursive_time / numpy_time:.2f}"])
    timing_table = Table(timing_rows, colWidths=[2.3 * cm, 4.5 * cm, 4.2 * cm, 2.4 * cm])
    timing_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#AAB7C4")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6FA")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story += [timing_table, Spacer(1, 0.25 * cm), Image(str(output_dir / "recursive_fft_timing.png"), width=17.6 * cm, height=7.4 * cm)]
    document.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)
    return pdf_path


def _draw_footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#AAB7C4"))
    canvas.line(document.leftMargin, 1.15 * cm, A4[0] - document.rightMargin, 1.15 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#4F6B82"))
    canvas.drawString(document.leftMargin, 0.75 * cm, "Computational Physics — Exercise 1, Task 2")
    canvas.drawRightString(A4[0] - document.rightMargin, 0.75 * cm, f"Page {document.page}")
    canvas.restoreState()


def _formula_image(formula: str, path: Path) -> Path:
    figure = plt.figure(figsize=(8, 0.8))
    figure.text(0.5, 0.5, f"${formula}$", ha="center", va="center", fontsize=18)
    figure.savefig(path, dpi=220, transparent=True, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("properties_dft_output"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checks, data = run_checks()
    plot_paths = create_plots(args.output_dir, data)
    radix2_plot_path = create_radix2_plot(args.output_dir)
    timing = benchmark_fft(args.output_dir)
    pdf_path = build_pdf(args.output_dir, checks, data, plot_paths, timing, radix2_plot_path)

    print("DFT property checks")
    for check in checks:
        print(f"{check.name:28s}: {check.error:.3e} ({'PASS' if check.passed else 'FAIL'})")
    print(f"PDF report: {pdf_path.resolve()}")
    print(f"Plots: {', '.join(str(path.resolve()) for path in plot_paths)}")


if __name__ == "__main__":
    main()
