"""generate_summary_pdf.py

Generates a publication-quality scientific summary PDF for Exercise 0:
Approximating Pi via Monte Carlo Methods (Computational Physics).
"""

import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


class NumberedCanvas(canvas.Canvas):
    """Canvas wrapper that dynamically adds page numbers and running header/footer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_header_footer(self, total_pages):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))

        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "Computational Physics — Exercise 0: Monte Carlo Approximation of π")
            self.setStrokeColor(colors.HexColor("#dddddd"))
            self.setLineWidth(0.5)
            self.line(54, 745, 558, 745)

        # Footer (all pages)
        self.setStrokeColor(colors.HexColor("#dddddd"))
        self.setLineWidth(0.5)
        self.line(54, 45, 558, 45)
        footer_text = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(558, 32, footer_text)
        self.drawString(54, 32, "Universität Wien · Faculty of Physics · WS 2026/27")
        self.restoreState()


def build_pdf():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(base_dir, "exercise_00_summary.pdf")

    # Document margins: 54pt = 0.75 in
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Refined Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1a365d"),
        spaceAfter=3,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#4a5568"),
        spaceAfter=6,
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor("#2b6cb0"),
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#2d3748"),
        spaceBefore=5,
        spaceAfter=2,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#2d3748"),
        spaceAfter=4,
    )

    callout_style = ParagraphStyle(
        "Callout",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#1a365d"),
    )

    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        alignment=1,  # Center
    )

    table_header = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=1,
    )

    story = []

    # =========================================================================
    # PAGE 1: Foundations, Baseline Algorithm, and Vectorization Performance
    # =========================================================================
    story.append(Paragraph("Monte Carlo Estimation of &pi;: Performance & Convergence", title_style))
    story.append(
        Paragraph(
            "<b>Course:</b> Computational Physics (WS 2026/27) · Universität Wien &nbsp;|&nbsp; <b>Exercise 0:</b> First Steps with Agentic AI",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#2b6cb0"), spaceAfter=6))

    # SECTION 1
    story.append(Paragraph("1. Mathematical Principle & Baseline Algorithm", h1_style))
    intro_text = (
        "The Monte Carlo approximation of &pi; samples points uniformly in the unit square [0, 1] &times; [0, 1]. "
        "A point (<i>x, y</i>) falls inside the quarter circle if <i>x</i><sup>2</sup> + <i>y</i><sup>2</sup> &le; 1. "
        "Because Area(quarter circle) / Area(square) = &pi; / 4, the fraction of points inside approximates &pi; / 4:"
    )
    story.append(Paragraph(intro_text, body_style))

    box1 = (
        "<b>Estimator:</b> &pi;<sub>est</sub> = 4 &times; (<i>N</i><sub>inside</sub> / <i>N</i>) "
        "&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;"
        "<b>Standard Error:</b> &sigma;<sub>&pi;</sub> = 4 &radic;[<i>p</i>(1 &minus; <i>p</i>) / <i>N</i>] "
        "&sim; <i>O</i>(<i>N</i><sup>&minus;1/2</sup>) = <i>O</i>(1 / &radic;<i>N</i>)"
    )
    box1_table = Table([[Paragraph(box1, callout_style)]], colWidths=[504])
    box1_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#edf2f7")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
                ("PADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(box1_table)
    story.append(Spacer(1, 4))

    # Figure 1 & Figure 2 side by side
    img_principle_path = os.path.join(base_dir, "mc_principle.png")
    img_errors_path = os.path.join(base_dir, "errors.png")

    fig12_data = [
        [
            Image(img_principle_path, width=200, height=170),
            Image(img_errors_path, width=285, height=170),
        ],
        [
            Paragraph("<b>Figure 1:</b> Monte Carlo sampling for <i>N</i>=1500.", table_cell),
            Paragraph("<b>Figure 2:</b> Error vs <i>N</i> scaling as <i>O</i>(1/&radic;<i>N</i>).", table_cell),
        ],
    ]
    fig12_table = Table(fig12_data, colWidths=[210, 294])
    fig12_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("PADDING", (0, 0), (-1, -1), 1),
            ]
        )
    )
    story.append(fig12_table)
    story.append(Spacer(1, 4))

    # SECTION 2
    story.append(Paragraph("2. Computational Optimization: Python Loops vs. Vectorized NumPy", h1_style))
    perf_text = (
        "Pure Python loops (iterating over scalar coordinates with <code>random.random()</code>) incur bytecode "
        "interpretation overhead on every coordinate. By eliminating loops in favor of contiguous 2D NumPy arrays "
        "(<code>pts = rng.random((N, 2))</code>), random generation and condition checks execute in compiled C routines. "
        "Execution times were benchmarked across <i>N</i> = 10<sup>2</sup> &hellip; 10<sup>7</sup> samples:"
    )
    story.append(Paragraph(perf_text, body_style))

    timing_data = [
        [
            Paragraph("<b>N</b>", table_header),
            Paragraph("<b>Python Loops [s]</b>", table_header),
            Paragraph("<b>NumPy (Vectorized) [s]</b>", table_header),
            Paragraph("<b>Speedup Multiplier</b>", table_header),
            Paragraph("<b>Estimated &pi;</b>", table_header),
            Paragraph("<b>Absolute Error</b>", table_header),
        ],
        [
            Paragraph("10<sup>2</sup>", table_cell),
            Paragraph("0.000029", table_cell),
            Paragraph("0.000008", table_cell),
            Paragraph("3.6&times;", table_cell),
            Paragraph("3.0400000", table_cell),
            Paragraph("1.016 &times; 10<sup>&minus;1</sup>", table_cell),
        ],
        [
            Paragraph("10<sup>3</sup>", table_cell),
            Paragraph("0.000268", table_cell),
            Paragraph("0.000021", table_cell),
            Paragraph("12.7&times;", table_cell),
            Paragraph("3.2760000", table_cell),
            Paragraph("1.344 &times; 10<sup>&minus;1</sup>", table_cell),
        ],
        [
            Paragraph("10<sup>4</sup>", table_cell),
            Paragraph("0.001319", table_cell),
            Paragraph("0.000053", table_cell),
            Paragraph("<b>25.0&times;</b>", table_cell),
            Paragraph("3.1776000", table_cell),
            Paragraph("3.601 &times; 10<sup>&minus;2</sup>", table_cell),
        ],
        [
            Paragraph("10<sup>5</sup>", table_cell),
            Paragraph("0.013779", table_cell),
            Paragraph("0.001375", table_cell),
            Paragraph("10.0&times;", table_cell),
            Paragraph("3.1527600", table_cell),
            Paragraph("1.117 &times; 10<sup>&minus;2</sup>", table_cell),
        ],
        [
            Paragraph("10<sup>6</sup>", table_cell),
            Paragraph("0.135143", table_cell),
            Paragraph("0.015708", table_cell),
            Paragraph("8.6&times;", table_cell),
            Paragraph("3.1419160", table_cell),
            Paragraph("3.233 &times; 10<sup>&minus;4</sup>", table_cell),
        ],
        [
            Paragraph("10<sup>7</sup>", table_cell),
            Paragraph("1.276377", table_cell),
            Paragraph("0.165382", table_cell),
            Paragraph("7.7&times;", table_cell),
            Paragraph("3.1409632", table_cell),
            Paragraph("6.295 &times; 10<sup>&minus;4</sup>", table_cell),
        ],
    ]
    timing_table = Table(timing_data, colWidths=[60, 95, 110, 85, 75, 79])
    timing_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b6cb0")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
                ("PADDING", (0, 0), (-1, -1), 2.5),
            ]
        )
    )
    story.append(timing_table)

    # =========================================================================
    # PAGE 2: Speedup Plot & Beating 1/sqrt(N)
    # =========================================================================
    story.append(PageBreak())

    img_speed_path = os.path.join(base_dir, "speed_numpy_vs_loops.png")
    speed_img_table = Table(
        [
            [Image(img_speed_path, width=490, height=170)],
            [Paragraph("<b>Figure 3:</b> Execution time and speedup multiplier comparing Python loops vs. vectorized NumPy.", table_cell)],
        ],
        colWidths=[504],
    )
    speed_img_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("PADDING", (0, 0), (-1, -1), 1),
            ]
        )
    )
    story.append(speed_img_table)
    story.append(Spacer(1, 4))

    # SECTION 3
    sec3_elements = []
    sec3_elements.append(Paragraph("3. Beating the Central Limit Theorem: Faster Convergence Than 1/&radic;N", h1_style))
    q_text = (
        "<b>Assignment Question (2d):</b> <i>Ask the agent to make the method converge faster than 1/&radic;N. "
        "What does it propose? Is its claim true? Verify it numerically.</i>"
    )
    q_table = Table([[Paragraph(q_text, callout_style)]], colWidths=[504])
    q_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#feebc8")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#fbd38d")),
                ("PADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    sec3_elements.append(q_table)
    sec3_elements.append(Spacer(1, 3))

    theory_text = (
        "<b>Method: Stratified Sampling on a Jittered Grid (Implemented in <code>gods_monte_carlo.py</code>)</b><br/>"
        "We divide [0, 1] &times; [0, 1] into a <i>K &times; K</i> grid where <i>K</i> = floor(&radic;<i>N</i>) "
        "(total samples <i>N</i><sub>actual</sub> = <i>K</i><sup>2</sup>) and place exactly one uniformly jittered point inside each cell. "
        "<br/>&bull; <b>Interior cells</b> (<i>r</i><sub>max</sub> &le; 1) contribute 1 with <b>zero variance</b>. "
        "<br/>&bull; <b>Exterior cells</b> (<i>r</i><sub>min</sub> &gt; 1) contribute 0 with <b>zero variance</b>. "
        "<br/>&bull; Only <b>boundary cells</b> intersecting the circle arc contribute variance. "
        "Because the arc has finite length (&pi;/2), only <i>O</i>(<i>K</i>) = <i>O</i>(&radic;<i>N</i>) cells intersect the boundary. "
        "<br/>Hence, Var(&pi;<sub>est</sub>) &sim; (4/<i>N</i>)<sup>2</sup> &times; <i>O</i>(&radic;<i>N</i>) = <i>O</i>(<i>N</i><sup>&minus;3/2</sup>), "
        "yielding an error standard deviation of <b>RMSE &sim; <i>O</i>(<i>N</i><sup>&minus;3/4</sup>) = <i>O</i>(<i>N</i><sup>&minus;0.75</sup>)</b>, "
        "strictly faster than standard Monte Carlo's <i>O</i>(<i>N</i><sup>&minus;0.50</sup>)."
    )
    sec3_elements.append(Paragraph(theory_text, body_style))
    story.extend(sec3_elements)

    # Numerical verification table
    qmc_data = [
        [
            Paragraph("<b>Target N</b>", table_header),
            Paragraph("<b>Standard MC RMSE</b>", table_header),
            Paragraph("<b>Stratified Grid RMSE</b>", table_header),
            Paragraph("<b>Sobol QMC RMSE</b>", table_header),
            Paragraph("<b>Accuracy Gain</b>", table_header),
        ],
        [
            Paragraph("10<sup>2</sup>", table_cell),
            Paragraph("1.455 &times; 10<sup>&minus;1</sup>", table_cell),
            Paragraph("8.231 &times; 10<sup>&minus;2</sup>", table_cell),
            Paragraph("4.772 &times; 10<sup>&minus;2</sup>", table_cell),
            Paragraph("1.8&times;", table_cell),
        ],
        [
            Paragraph("10<sup>3</sup>", table_cell),
            Paragraph("5.256 &times; 10<sup>&minus;2</sup>", table_cell),
            Paragraph("1.036 &times; 10<sup>&minus;2</sup>", table_cell),
            Paragraph("1.088 &times; 10<sup>&minus;2</sup>", table_cell),
            Paragraph("5.1&times;", table_cell),
        ],
        [
            Paragraph("10<sup>4</sup>", table_cell),
            Paragraph("1.735 &times; 10<sup>&minus;2</sup>", table_cell),
            Paragraph("2.125 &times; 10<sup>&minus;3</sup>", table_cell),
            Paragraph("2.411 &times; 10<sup>&minus;3</sup>", table_cell),
            Paragraph("8.2&times;", table_cell),
        ],
        [
            Paragraph("10<sup>5</sup>", table_cell),
            Paragraph("4.855 &times; 10<sup>&minus;3</sup>", table_cell),
            Paragraph("2.726 &times; 10<sup>&minus;4</sup>", table_cell),
            Paragraph("1.749 &times; 10<sup>&minus;4</sup>", table_cell),
            Paragraph("17.8&times;", table_cell),
        ],
        [
            Paragraph("10<sup>6</sup>", table_cell),
            Paragraph("1.404 &times; 10<sup>&minus;3</sup>", table_cell),
            Paragraph("<b>6.642 &times; 10<sup>&minus;5</sup></b>", table_cell),
            Paragraph("<b>5.712 &times; 10<sup>&minus;5</sup></b>", table_cell),
            Paragraph("<b>21.1&times;</b>", table_cell),
        ],
    ]
    qmc_table = Table(qmc_data, colWidths=[70, 115, 115, 110, 94])
    qmc_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b6cb0")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
                ("PADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    story.append(qmc_table)
    story.append(Spacer(1, 3))

    img_conv_path = os.path.join(base_dir, "gods_monte_carlo_convergence.png")
    conv_flowables = [
        Image(img_conv_path, width=420, height=160),
        Paragraph(
            "<b>Figure 4:</b> Empirical convergence verification: Standard MC scales as &sim;<i>N</i><sup>&minus;0.51</sup>, "
            "while Stratified Grid and Sobol QMC achieve &sim;<i>N</i><sup>&minus;0.78</sup> and &sim;<i>N</i><sup>&minus;0.77</sup>.",
            table_cell,
        ),
    ]
    conv_table = Table([[item] for item in conv_flowables], colWidths=[504])
    conv_table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("PADDING", (0, 0), (-1, -1), 0)]))
    story.append(conv_table)

    # =========================================================================
    # PAGE 3: Reflections & Agentic AI Insights
    # =========================================================================
    story.append(PageBreak())

    story.append(Paragraph("4. Exercise Reflection: Working with Agentic AI", h1_style))

    refl_a = (
        "<b>a) Where did the agent save time, and where did you have to correct or verify it?</b><br/>"
        "&bull; <b>Time Savings:</b> Fast generation of clean boilerplate, automated NumPy vectorization without dimension mismatches, "
        "reproducible multi-seed benchmark loops, and publication-ready matplotlib graphics.<br/>"
        "&bull; <b>Critical Verification Required:</b> Theoretical convergence claims must be rigorously audited. "
        "A common pitfall with AI agents is asserting that Quasi-Monte Carlo achieves <i>O</i>(1/<i>N</i>) for this problem. "
        "In reality, the Koksma-Hlawka bound requires bounded variation in the sense of Hardy and Krause (BVHK); "
        "because the quarter circle boundary is curved and oblique to coordinate axes, the variation is unbounded. "
        "Consequently, QMC error on circular sets in 2D degenerates to <i>O</i>(<i>N</i><sup>&minus;3/4</sup>). "
        "Multi-trial empirical RMSE regression confirmed this exact ~0.77 slope."
    )
    story.append(Paragraph(refl_a, body_style))
    story.append(Spacer(1, 5))

    refl_b = (
        "<b>b) What should you write into prompts (or CLAUDE.md / project instruction files)?</b><br/>"
        "&bull; <b>State strict numerical and mathematical expectations:</b> Specify required convergence rates, "
        "statistical validation criteria (e.g., minimum of 20 runs to compute RMSE rather than single-run luck), and error definitions.<br/>"
        "&bull; <b>Enforce vectorized coding standards:</b> Explicitly require pure array-based NumPy operations and prohibit per-sample loops.<br/>"
        "&bull; <b>Preserve files and verify diffs:</b> Direct the agent to create distinct filenames for major variants (e.g. <code>gods_monte_carlo.py</code>) "
        "and maintain backward compatibility with previous scripts."
    )
    story.append(Paragraph(refl_b, body_style))
    story.append(Spacer(1, 5))

    refl_c = (
        "<b>c) How do you ensure you understand every line of code written by the agent?</b><br/>"
        "&bull; <b>Inspect diffs prior to committing:</b> Always review <code>git diff</code> line-by-line to verify changes against intentions.<br/>"
        "&bull; <b>Interrogate the agent for proofs:</b> Prompt the agent to derive intermediate formulas (e.g., how perimeter scaling leads to "
        "<i>O</i>(<i>N</i><sup>&minus;3/2</sup>) variance in stratified sampling).<br/>"
        "&bull; <b>Sanity-check boundary cases:</b> Run unit checks on small <i>N</i> (e.g. <i>N</i>=1, <i>N</i>=100) to confirm expected array shapes, "
        "normalizations, and unbiasedness."
    )
    story.append(Paragraph(refl_c, body_style))
    story.append(Spacer(1, 10))

    # Executive Summary Callout Box
    summary_box_data = [
        [
            Paragraph(
                "<b>Key Takeaways:</b><br/>"
                "<b>1. Baseline Rate:</b> Standard pseudo-random Monte Carlo is bound to <i>O</i>(1/&radic;<i>N</i>) by the Central Limit Theorem.<br/>"
                "<b>2. Computational Speed:</b> Vectorized NumPy eliminates Python loop overhead, delivering a <b>25&times; speedup</b> at <i>N</i>=10<sup>4</sup>.<br/>"
                "<b>3. Accelerated Convergence:</b> Stratified sampling eliminates interior/exterior cell variance, achieving proven "
                "<b><i>O</i>(<i>N</i><sup>&minus;3/4</sup>)</b> convergence and reducing estimation error by <b>21.1&times;</b> at <i>N</i>=10<sup>6</sup>.",
                callout_style,
            )
        ]
    ]
    summary_box = Table(summary_box_data, colWidths=[504])
    summary_box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ebf8ff")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#3182ce")),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(summary_box)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated: {pdf_path}")


if __name__ == "__main__":
    build_pdf()
