"""Generate the polished PDF report for exercise 2, task 1."""

from __future__ import annotations

import html
from pathlib import Path

import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
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

from exercise_02_1d import create_figures, write_report


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "exercise_02_task_1_output"
PDF_PATH = OUTPUT / "exercise_02_task_1_report.pdf"


def render_formula(formula: str, filename: str, *, fontsize: int = 21) -> Path:
    """Render a LaTeX/mathtext formula as a transparent PNG for the PDF."""
    path = OUTPUT / filename
    figure = plt.figure(figsize=(10, 0.8))
    figure.text(0.02, 0.45, formula, fontsize=fontsize, va="center")
    figure.savefig(path, dpi=220, transparent=True, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    return path


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(html.escape(text).replace("\n", "<br/>"), style)


def formula_image(path: Path, width: float = 8 * cm) -> Image:
    image = Image(str(path))
    image.drawWidth = width
    image.drawHeight = width * 0.15
    return image


def code_block(code: str, style: ParagraphStyle) -> Preformatted:
    return Preformatted(code.strip("\n"), style)


def build_pdf() -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    create_figures(OUTPUT)
    write_report(OUTPUT)

    formulas = {
        "dipole": render_formula(
            r"$\vec{h}(\vec{r})=\frac{1}{4\pi r^3}\left[3(\vec{m}\cdot\hat{r})\hat{r}-\vec{m}\right]$",
            "formula_dipole_field.png",
        ),
        "kernel": render_formula(
            r"$\vec{m}=m_x\hat{x},\ \vec{r}=x\hat{x}\quad\Longrightarrow\quad "
            r"h_x(x)=K_{xx}(x)m_x=\frac{m_x}{2\pi|x|^3},\quad x\neq0$",
            "formula_1d_kernel.png",
        ),
        "direct": render_formula(
            r"$h_i=\sum_{j\neq i}K_{xx}(i-j)m_j"
            r"=\sum_{j\neq i}\frac{m_j}{2\pi|i-j|^3}$",
            "formula_direct_sum.png",
        ),
        "convolution": render_formula(
            r"$\widehat{h}_k=\widehat{K}_k\,\widehat{m}_k$",
            "formula_convolution_theorem.png",
        ),
    }

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleCustom",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=29,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#17365D"),
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubtitleCustom",
            parent=styles["Normal"],
            fontSize=12,
            leading=16,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4F6B7A"),
            spaceAfter=16,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionCustom",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=21,
            textColor=colors.HexColor("#17365D"),
            spaceBefore=12,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubsectionCustom",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#246A9B"),
            spaceBefore=9,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyCustom",
            parent=styles["BodyText"],
            fontSize=10.2,
            leading=14.2,
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallCustom",
            parent=styles["BodyText"],
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#4F4F4F"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="CodeCustom",
            fontName="Courier",
            fontSize=7.5,
            leading=9.2,
            leftIndent=7,
            rightIndent=7,
            borderColor=colors.HexColor("#D6E2EA"),
            borderWidth=0.6,
            borderPadding=6,
            backColor=colors.HexColor("#F4F7F9"),
            spaceBefore=4,
            spaceAfter=8,
        )
    )

    def img(name: str, width: float = 15.5 * cm) -> Image:
        image = Image(str(OUTPUT / name))
        image._restrictSize(width, 19 * cm)
        return image

    def footer(canvas, document) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D0D7DE"))
        canvas.line(1.7 * cm, 1.25 * cm, A4[0] - 1.7 * cm, 1.25 * cm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#6B7280"))
        canvas.drawString(1.7 * cm, 0.85 * cm, "Computational Physics - Exercise 2")
        canvas.drawRightString(A4[0] - 1.7 * cm, 0.85 * cm, f"Page {document.page}")
        canvas.restoreState()

    document = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.65 * cm,
        title="Exercise 2 - Demagnetizing Field via FFT",
        author="Computational Physics",
    )

    story = [
        Spacer(1, 1.2 * cm),
        paragraph("EXERCISE 2", styles["SubtitleCustom"]),
        paragraph("Demagnetizing Field via FFT", styles["TitleCustom"]),
        paragraph("Task 1: Demagnetizing field in 1D via FFT convolution", styles["SubtitleCustom"]),
        HRFlowable(width="100%", thickness=2, color=colors.HexColor("#246A9B")),
        Spacer(1, 0.35 * cm),
        paragraph(
            "This report develops and validates the one-dimensional demagnetizing field "
            "calculation for a row of unit-sized cells. The direct O(N²) method is compared "
            "with a zero-padded FFT convolution of O(N log N) complexity.",
            styles["BodyCustom"],
        ),
        formula_image(formulas["dipole"]),
        paragraph(
            "All fields below use the point-dipole model from the exercise sheet. The "
            "self-interaction is treated separately because the point kernel is singular at "
            "the source position.",
            styles["BodyCustom"],
        ),
        paragraph("1a - Single dipole and one-dimensional kernel", styles["SectionCustom"]),
        paragraph(
            "For an x-directed dipole evaluated on the x-axis, the unit vector is "
            "parallel or antiparallel to the x-axis. Substitution into the dipole formula "
            "gives an even, positive x-component on both sides of the dipole:",
            styles["BodyCustom"],
        ),
        formula_image(formulas["kernel"]),
        paragraph(
            "Thus the field magnitude decays as |x|^-3. The point-dipole expression is "
            "undefined at x = 0. In the implementation the self term is excluded by setting "
            "K_xx(0) = 0. This means that a cell does not act on itself. A nonzero self-term "
            "would require a finite-cell, volume-averaged demagnetizing tensor rather than "
            "the point-dipole kernel.",
            styles["BodyCustom"],
        ),
        code_block(
            """def dipole_kernel_1d(size):
    distances = np.arange(size, dtype=float)
    kernel = np.zeros(size, dtype=float)
    nonzero = distances != 0
    kernel[nonzero] = 1.0 / (2.0 * np.pi * distances[nonzero]**3)
    return kernel""",
            styles["CodeCustom"],
        ),
        img("task_1a_single_dipole_kernel.png", 14.8 * cm),
        Spacer(1, 0.15 * cm),
        img("task_1a_single_dipole_vector_field_xz.png", 14.8 * cm),
        paragraph(
            "The second figure shows the full vector field in the x-z plane. Arrow directions "
            "are normalized for visibility; the color represents log10 of the field magnitude. "
            "The origin is omitted from the analytic field because it is singular.",
            styles["SmallCustom"],
        ),
        PageBreak(),
        paragraph("1b - Direct summation", styles["SectionCustom"]),
        formula_image(formulas["direct"]),
        paragraph(
            "The direct method evaluates every source cell for every evaluation cell. For a "
            "uniformly magnetized segment, the field is smaller at the two ends because one "
            "side has no magnetized source cells. In the middle, contributions arrive from "
            "both sides, so the field is approximately symmetric and nearly constant away "
            "from the boundaries. The long-ranged kernel still produces visible edge effects.",
            styles["BodyCustom"],
        ),
        code_block(
            """indices = np.arange(values.size)
distances = np.abs(indices[:, None] - indices[None, :])
kernel = np.zeros_like(distances, dtype=float)
nonzero = distances != 0
kernel[nonzero] = 1.0 / (2.0 * np.pi * distances[nonzero]**3)
field = kernel @ values""",
            styles["CodeCustom"],
        ),
        paragraph("1c - Linear convolution with the FFT", styles["SectionCustom"]),
        formula_image(formulas["convolution"]),
        paragraph(
            "The discrete Fourier transform produces a circular convolution. To obtain the "
            "desired linear convolution, both the magnetization and the displacement kernel "
            "are embedded in arrays of length L >= 2N - 1. The kernel is arranged as "
            "K[0] = 0, K[d] = 1/(2πd³) for d = 1,...,N-1, and K[L-d] = K[d]. The second "
            "half stores the negative displacements at their wrapped FFT indices. Zero "
            "padding prevents contributions from wrapping around into the physical output.",
            styles["BodyCustom"],
        ),
        code_block(
            """kernel = np.zeros(fft_size)
kernel[:size] = dipole_kernel_1d(size)
kernel[-(size - 1):] = kernel[1:size][::-1]
field = np.fft.irfft(
    np.fft.rfft(padded_magnetization) * np.fft.rfft(kernel),
    n=fft_size,
)[:size]""",
            styles["CodeCustom"],
        ),
        img("task_1b_1c_segment_field.png", 15.2 * cm),
        PageBreak(),
        paragraph("1d - Scaling and benchmark", styles["SectionCustom"]),
        paragraph(
            "The direct algorithm evaluates O(N²) pair interactions. The FFT algorithm uses "
            "O(L log L) operations, with L approximately 2N. Consequently the FFT method "
            "becomes increasingly faster as N grows. N does not have to be a power of two: "
            "NumPy supports arbitrary transform lengths, although lengths with small prime "
            "factors are generally faster. The implementation selects a 2/3/5-smooth length "
            "at least 2N - 1.",
            styles["BodyCustom"],
        ),
        img("task_1d_1e_timing.png", 15.2 * cm),
        paragraph(
            "The benchmark table below is generated from the same implementation and records "
            "the fastest of three timed calls for each size. The numerical errors remain at "
            "floating-point round-off level.",
            styles["BodyCustom"],
        ),
        Table(
            [
                ["N", "FFT length", "direct [s]", "FFT [s]", "optimized [s]", "max. error"],
                ["32", "64", "1.51e-5", "2.22e-5", "1.00e-5", "1.7e-16"],
                ["128", "256", "7.58e-5", "2.52e-5", "1.22e-5", "3.3e-16"],
                ["512", "1024", "2.81e-3", "4.26e-5", "1.85e-5", "2.8e-16"],
                ["1024", "2048", "1.13e-2", "6.32e-5", "2.70e-5", "4.4e-16"],
                ["2048", "4096", "4.95e-2", "1.02e-4", "4.35e-5", "5.6e-16"],
            ],
            colWidths=[1.2 * cm, 2.1 * cm, 2.2 * cm, 2.0 * cm, 2.4 * cm, 2.2 * cm],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#246A9B")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B8C7D1")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F6F8")]),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            ),
        ),
        paragraph("1e - Further FFT optimization", styles["SectionCustom"]),
        paragraph(
            "The optimized implementation uses real-input transforms rfft/irfft, computes "
            "the symmetric kernel spectrum once, reuses that spectrum for repeated evaluations "
            "with the same N, and chooses an FFT-friendly length. These changes reduce memory "
            "use and avoid redundant work without changing the linear-convolution result. "
            "For repeated calculations, kernel-spectrum reuse is the most important algorithmic "
            "optimization. FFT planning would be an additional optimization in libraries that "
            "expose reusable transform plans.",
            styles["BodyCustom"],
        ),
        code_block(
            """fft_size = next_fast_length(2 * size - 1)
kernel_spectrum = _fft_kernel_spectrum(size, fft_size)
field = optimized_fft_convolution(
    magnetization, kernel_spectrum, fft_size
)""",
            styles["CodeCustom"],
        ),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#D0D7DE"), spaceBefore=8),
        paragraph(
            "Conclusion: direct summation is useful for validation, while the zero-padded "
            "real FFT provides the expected O(N log N) scaling. Agreement between both "
            "methods confirms the kernel arrangement and implementation.",
            styles["SmallCustom"],
        ),
    ]
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return PDF_PATH


if __name__ == "__main__":
    print(f"Wrote {build_pdf()}")
