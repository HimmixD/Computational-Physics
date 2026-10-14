"""Compare the spectra of standard signals and discuss spectral leakage.

The script creates plots and an explanatory PDF in ``comparison_signals_output``.
Signals are sampled on a centred time grid so parity is visible in the DFT:
real-even signals have real spectra, real-odd signals have imaginary spectra,
and signals with neither symmetry generally have complex spectra.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def make_signals(n: int, duration: float, seed: int = 7) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return a centred time grid and five signals with comparable scales."""
    t = (np.arange(n) - n // 2) * duration / n
    frequency = 5.0 / duration
    rng = np.random.default_rng(seed)
    signals = {
        "rectangle": np.where(np.abs(t) < 0.18 * duration, 1.0, 0.0),
        "sawtooth": 2.0 * np.mod(t + duration / 2, duration) / duration - 1.0,
        "sine": np.sin(2 * np.pi * frequency * t),
        "Gaussian pulse": np.exp(-0.5 * (t / (0.08 * duration)) ** 2),
        "noise": rng.normal(0.0, 0.35, n),
    }
    return t, signals


def spectrum(signal: np.ndarray, duration: float, window: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return centred frequencies and DFT coefficients normalized by N."""
    values = signal if window is None else signal * window
    frequencies = np.fft.fftshift(np.fft.fftfreq(signal.size, duration / signal.size))
    coefficients = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(values))) / signal.size
    return frequencies, coefficients


def spectral_character(coefficients: np.ndarray) -> tuple[str, float, float]:
    """Classify a spectrum using the relative real and imaginary energy."""
    scale = max(float(np.max(np.abs(coefficients))), 1e-15)
    real_fraction = float(np.max(np.abs(coefficients.real)) / scale)
    imaginary_fraction = float(np.max(np.abs(coefficients.imag)) / scale)
    if imaginary_fraction < 1e-8:
        return "real", real_fraction, imaginary_fraction
    if real_fraction < 1e-8:
        return "imaginary", real_fraction, imaginary_fraction
    return "complex", real_fraction, imaginary_fraction


def decay_fit(frequencies: np.ndarray, coefficients: np.ndarray) -> tuple[float, float]:
    """Fit log(|a_k|) against log(f) away from DC and the noise floor."""
    positive = frequencies > 0
    amplitudes = np.abs(coefficients)
    threshold = max(np.max(amplitudes) * 1e-3, np.median(amplitudes[positive]) * 3)
    mask = positive & (amplitudes > threshold)
    if np.count_nonzero(mask) < 3:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(np.log(frequencies[mask]), np.log(amplitudes[mask]), 1)
    return float(slope), float(intercept)


def create_signal_plot(output_dir: Path, t: np.ndarray, signals: dict[str, np.ndarray]) -> Path:
    figure, axes = plt.subplots(len(signals), 2, figsize=(12, 14), sharex="col")
    duration = t[-1] - t[0] + (t[1] - t[0])
    for row, (name, signal) in enumerate(signals.items()):
        frequencies, coefficients = spectrum(signal, duration)
        axes[row, 0].plot(t * 1e3, signal, color="#246A9B", lw=1)
        axes[row, 0].set_ylabel(name)
        axes[row, 1].plot(frequencies, np.abs(coefficients), color="#C44E52", lw=1)
        axes[row, 1].set_yscale("log")
        axes[row, 1].set_ylabel("|aₖ|")
        axes[row, 0].grid(True, ls=":", alpha=0.5)
        axes[row, 1].grid(True, ls=":", alpha=0.5)
    axes[-1, 0].set_xlabel("time [ms]")
    axes[-1, 1].set_xlabel("frequency [Hz]")
    figure.suptitle("Signals and their spectra", fontsize=16)
    figure.tight_layout()
    path = output_dir / "signals_and_spectra.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def create_complexity_plot(output_dir: Path, t: np.ndarray, signals: dict[str, np.ndarray]) -> Path:
    duration = t[-1] - t[0] + (t[1] - t[0])
    figure, axes = plt.subplots(len(signals), 2, figsize=(12, 14), sharex="col")
    for row, (name, signal) in enumerate(signals.items()):
        frequencies, coefficients = spectrum(signal, duration)
        axes[row, 0].plot(frequencies, coefficients.real, label="real")
        axes[row, 0].plot(frequencies, coefficients.imag, "--", label="imaginary")
        axes[row, 0].set_ylabel(name)
        axes[row, 1].loglog(frequencies[frequencies > 0], np.abs(coefficients[frequencies > 0]), ".")
        slope, _ = decay_fit(frequencies, coefficients)
        label = f"fit slope {slope:.2f}" if np.isfinite(slope) else "no reliable power-law fit"
        axes[row, 1].text(0.04, 0.08, label, transform=axes[row, 1].transAxes, fontsize=9)
        for axis in axes[row]:
            axis.grid(True, ls=":", alpha=0.5)
    axes[0, 0].legend()
    axes[-1, 0].set_xlabel("frequency [Hz]")
    axes[-1, 1].set_xlabel("frequency [Hz]")
    figure.suptitle("Real/imaginary parts and high-frequency decay", fontsize=16)
    figure.tight_layout()
    path = output_dir / "spectral_character_and_decay.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def create_window_plot(output_dir: Path, n: int = 512, duration: float = 1.0) -> Path:
    t = (np.arange(n) - n // 2) * duration / n
    frequency = 5.5 / duration
    signal = np.sin(2 * np.pi * frequency * t)
    frequencies, unwindowed = spectrum(signal, duration)
    hann = np.hanning(n)
    _, windowed = spectrum(signal, duration, hann)
    figure, axes = plt.subplots(2, 1, figsize=(10, 7))
    axes[0].plot(t, signal, label="non-integer-period sine")
    axes[0].set(xlabel="time [s]", ylabel="signal", title="A signal that does not fit the sample window")
    axes[0].legend()
    axes[1].semilogy(frequencies, np.abs(unwindowed) + 1e-12, label="rectangular window")
    axes[1].semilogy(frequencies, np.abs(windowed) + 1e-12, label="Hann window")
    axes[1].set(xlabel="frequency [Hz]", ylabel="|aₖ|", title="Spectral leakage and windowing")
    axes[1].legend()
    for axis in axes:
        axis.grid(True, ls=":", alpha=0.6)
    figure.tight_layout()
    path = output_dir / "window_mismatch_and_leakage.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def build_report(output_dir: Path, table_rows: list[list[str]], plot_paths: list[Path]) -> Path:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, fontSize=22, textColor=colors.HexColor("#17365D")))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], textColor=colors.HexColor("#17365D"), spaceBefore=10, spaceAfter=7))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=9.5, leading=13))
    path = output_dir / "comparison_signals_report.pdf"
    document = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=1.6 * cm, leftMargin=1.6 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    story = [
        Paragraph("Comparison of Signal Spectra", styles["TitleCenter"]),
        Paragraph("Parity, smoothness, and finite-window effects", styles["Subtitle"] if "Subtitle" in styles else styles["Normal"]),
        Paragraph(
            "The DFT is computed for a centred, real-valued sample grid. A real signal that is "
            "even about the origin has a real spectrum; a real odd signal has an imaginary "
            "spectrum; without either parity the spectrum is generally complex. Smooth signals "
            "have rapidly decaying coefficients, while jumps create slowly decaying high-frequency tails.",
            styles["BodyText"],
        ),
        Spacer(1, 0.25 * cm),
        Paragraph("Numerical classification", styles["Section"]),
    ]
    table = Table(table_rows, colWidths=[3.2 * cm, 3.0 * cm, 4.0 * cm, 3.0 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#AAB7C4")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6FA")]),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story += [table, Spacer(1, 0.3 * cm), Image(str(plot_paths[0]), width=17.7 * cm, height=20 * cm), PageBreak()]
    story += [
        Paragraph("What controls spectral decay?", styles["Section"]),
        Paragraph(
            "The rectangle and sawtooth contain discontinuities, so their coefficients have algebraic "
            "tails. The sine concentrates energy into a small number of frequencies. The Gaussian is "
            "infinitely differentiable and has very rapid, approximately Gaussian decay. White noise "
            "has no deterministic smoothness, so its spectrum fluctuates around a broad noise floor "
            "and a power-law fit is not physically meaningful.",
            styles["BodyText"],
        ),
        Spacer(1, 0.25 * cm),
        Image(str(plot_paths[1]), width=17.7 * cm, height=20 * cm),
        PageBreak(),
        Paragraph("When the signal does not fit in the sample window", styles["Section"]),
        Paragraph(
            "The DFT assumes that the finite sample window repeats periodically. If the first and "
            "last samples do not join smoothly, that artificial boundary is a jump. Its Fourier "
            "coefficients spread energy into neighbouring bins: this is spectral leakage. A sine "
            "with 5.5 cycles in the window is a direct example.",
            styles["BodyText"],
        ),
        Image(str(plot_paths[2]), width=17.7 * cm, height=12.4 * cm),
        Paragraph(
            "<b>Ways to improve the result:</b> choose a window containing an integer number of "
            "periods when possible; increase the observation time to improve frequency resolution; "
            "remove a mean or trend; apply a Hann, Hamming, or Blackman window to reduce boundary "
            "leakage; and use zero-padding only to interpolate the plotted spectrum. Zero-padding "
            "does not add information or remove leakage, while windowing trades reduced sidelobes "
            "for a wider main lobe.",
            styles["BodyText"],
        ),
    ]
    document.build(story)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("comparison_signals_output"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    t, signals = make_signals(512, 1.0)
    duration = t[-1] - t[0] + (t[1] - t[0])
    rows = [["Signal", "Spectrum type", "Real / |A| max", "Imaginary / |A| max"]]
    for name, signal in signals.items():
        frequencies, coefficients = spectrum(signal, duration)
        kind, real_fraction, imaginary_fraction = spectral_character(coefficients)
        slope, _ = decay_fit(frequencies, coefficients)
        slope_text = f"{slope:.2f}" if np.isfinite(slope) else "not fitted"
        rows.append([name, kind, f"{real_fraction:.3f}", f"{imaginary_fraction:.3f} (slope {slope_text})"])
    plot_paths = [
        create_signal_plot(args.output_dir, t, signals),
        create_complexity_plot(args.output_dir, t, signals),
        create_window_plot(args.output_dir),
    ]
    report = build_report(args.output_dir, rows, plot_paths)
    print(f"PDF report: {report.resolve()}")
    print(f"Plots: {', '.join(str(path.resolve()) for path in plot_paths)}")


if __name__ == "__main__":
    main()
