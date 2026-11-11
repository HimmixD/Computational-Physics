"""Generate a readable PDF report for exercise 4, tasks 1 and 2."""

from __future__ import annotations

import csv
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).parent
OUTPUT = ROOT / "exercise_04_report.pdf"
TASK_1_OUTPUT = ROOT / "exercise_04_task_1_output"
TASK_2_OUTPUT = ROOT / "exercise_04_task_2_output"


def equation(expression: str, width: float = 13.5 * cm) -> Image:
    """Render a LaTeX/mathtext equation into a crisp report image."""
    figure = plt.figure(figsize=(width / 72.0, 0.62), dpi=220)
    figure.patch.set_alpha(0.0)
    figure.text(0.5, 0.5, f"${expression}$", ha="center", va="center", fontsize=15)
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=220, transparent=True, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    buffer.seek(0)
    return Image(buffer, width=width, height=0.62 * cm)


def image(path: Path, width: float = 16.8 * cm) -> Image:
    """Add a plot while preserving its aspect ratio."""
    from PIL import Image as PILImage

    with PILImage.open(path) as source:
        ratio = source.height / source.width
    return Image(str(path), width=width, height=width * ratio)


def table(data: list[list[str]], widths: list[float] | None = None) -> Table:
    result = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEADING", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9aa9b8")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return result


def read_csv(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.reader(file))


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def build_report() -> None:
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleCentered",
            parent=styles["Title"],
            alignment=TA_CENTER,
            textColor=colors.HexColor("#17365d"),
            fontSize=22,
            leading=27,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Subtitle",
            parent=styles["Normal"],
            alignment=TA_CENTER,
            fontSize=11,
            leading=15,
            textColor=colors.HexColor("#4f5b66"),
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading1"],
            textColor=colors.HexColor("#1f4e79"),
            spaceBefore=8,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Subsection",
            parent=styles["Heading2"],
            textColor=colors.HexColor("#2f5597"),
            spaceBefore=8,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyReport",
            parent=styles["BodyText"],
            fontSize=9.6,
            leading=14,
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Caption",
            parent=styles["BodyText"],
            alignment=TA_CENTER,
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#53616d"),
            spaceBefore=3,
            spaceAfter=10,
        )
    )

    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title="Exercise 4 - Orbits and Rocket Launch",
        author="Computational Physics",
    )
    story: list = []
    body = styles["BodyReport"]

    story.extend(
        [
            Spacer(1, 2.0 * cm),
            paragraph("Exercise 4", styles["TitleCentered"]),
            paragraph("Symplectic Integrators for Orbits and Rocket Launch", styles["Subtitle"]),
            paragraph(
                "Zusammenfassender Bericht zu den Aufgaben 1 und 2. "
                "Der Bericht verbindet die Modellgleichungen, numerischen Verfahren, "
                "KontrollgrÃ¶ÃŸen und die erzeugten Simulationsergebnisse.",
                body,
            ),
            Spacer(1, 0.4 * cm),
            paragraph(
                "<b>Verwendete Konstanten:</b> "
                "G = 6.674Â·10<super>âˆ’11</super> m<super>3</super> kg<super>âˆ’1</super> s<super>âˆ’2</super>, "
                "M<sub>E</sub> = 5.972Â·10<super>24</super> kg. "
                "Die Implementierung verwendet SI-Einheiten.",
                body,
            ),
            paragraph(
                "Die Abbildungen sind keine isolierten Illustrationen: Jede Grafik beantwortet "
                "eine konkrete Frage des Ãœbungsblatts und wird im Text physikalisch interpretiert.",
                body,
            ),
            PageBreak(),
        ]
    )

    story.append(paragraph("1. Mondbahn um die Erde", styles["Section"]))
    story.append(paragraph("1a â€“ Anfangsbedingungen und Umlaufzeit", styles["Subsection"]))
    story.append(
        paragraph(
            "Die Erde wird als ruhender Punkt im Ursprung modelliert. FÃ¼r den Mond gilt die "
            "Zentralkraftgleichung. Bei einer Kreisbahn muss die Gravitationsbeschleunigung "
            "gleich der Zentripetalbeschleunigung sein:",
            body,
        )
    )
    story.append(equation(r"\ddot{r}=-\frac{GM}{|r|^3}r"))
    story.append(equation(r"\frac{v_M^2}{r_M}=\frac{GM}{r_M^2},\qquad v_M=\sqrt{\frac{GM}{r_M}}"))
    story.append(
        paragraph(
            "Die Wahl r(0) = (r<sub>M</sub>, 0) und v(0) = (0, v<sub>M</sub>) erzeugt eine "
            "gegen den Uhrzeigersinn laufende Bahn. Numerisch ergibt sich:",
            body,
        )
    )
    story.append(
        table(
            [
                ["GrÃ¶ÃŸe", "Wert"],
                ["Mondradius rM", "3.84400Â·10<sup>8</sup> m"],
                ["Anfangsgeschwindigkeit vM", "1.018266Â·10<sup>3</sup> m/s"],
                ["Kepler-Periode T", "27.452898 d"],
                ["Siderischer Monat", "27.32 d"],
            ],
            [7 * cm, 7 * cm],
        )
    )
    story.append(
        paragraph(
            "Die kleine Abweichung zwischen idealer Kepler-Periode und siderischem Monat "
            "ist physikalisch plausibel: Das reale Erde-Mond-System besitzt einen Baryzentrums-"
            "anteil, die Mondbahn ist nicht exakt kreisfÃ¶rmig und Sonne sowie andere KÃ¶rper "
            "stÃ¶ren die Bewegung. ZusÃ¤tzlich sind die im Blatt angegebenen Konstanten gerundet.",
            body,
        )
    )

    story.append(paragraph("1b â€“ Vergleich der Integratoren", styles["Subsection"]))
    story.append(
        paragraph(
            "FÃ¼r die Bahn wurden expliziter Euler, impliziter Euler, Heun, symplektischer Euler "
            "und Velocity-Verlet implementiert. Die Schrittweite betrÃ¤gt 6 h; dargestellt sind "
            "zehn UmlÃ¤ufe. Der implizite Euler-Schritt wird als nichtlineares Gleichungssystem "
            "mit Newton gelÃ¶st. Als Startwert dient ein expliziter Euler-Schritt.",
            body,
        )
    )
    story.append(
        paragraph(
            "Die Grafik zeigt den charakteristischen Langzeitfehler: Expliziter Euler gewinnt "
            "Energie und spiralt nach auÃŸen, impliziter Euler verliert Energie und spiralt nach "
            "innen. Heun ist zwar formal zweiter Ordnung, aber nicht symplektisch. Symplektischer "
            "Euler und Velocity-Verlet halten die Energiefehler bounded beziehungsweise oszillierend "
            "und bewahren dadurch die qualitative Bahnstruktur.",
            body,
        )
    )
    story.append(image(TASK_1_OUTPUT / "task_1b_orbits.png", 14.8 * cm))
    story.append(paragraph("Abb. 1: Zehn numerische Mondbahnen bei Î”t = 6 h.", styles["Caption"]))

    story.append(paragraph("1c â€“ Energie und Drehimpuls", styles["Subsection"]))
    story.append(equation(r"E=\frac{1}{2}|v|^2-\frac{GM}{|r|}"))
    story.append(equation(r"L=r\times v"))
    story.append(
        paragraph(
            "FÃ¼r die exakte Bewegung verschwinden beide Zeitableitungen. FÃ¼r die Energie "
            "heben sich Arbeitsbeitrag der Gravitation und kinetische Ã„nderung auf; beim "
            "Drehimpuls verschwinden beide Kreuzprodukte, weil Kraft und Ortsvektor parallel "
            "sind. Die 100-Umlauf-Auswertung prÃ¼ft diese ErhaltungsgrÃ¶ÃŸen numerisch.",
            body,
        )
    )
    story.append(image(TASK_1_OUTPUT / "task_1c_invariants.png", 16.2 * cm))
    story.append(paragraph("Abb. 2: Relative Fehler von Energie und Drehimpuls Ã¼ber 100 UmlÃ¤ufe.", styles["Caption"]))
    story.append(
        paragraph(
            "Der symplektische Euler-Schritt erhÃ¤lt in dieser zentralen Kraftdiskretisierung "
            "den skalaren Drehimpuls exakt. SymplektizitÃ¤t bedeutet jedoch nicht, dass jede "
            "ErhaltungsgrÃ¶ÃŸe exakt bleibt: Besonders die Energie zeigt bei symplektischen "
            "Verfahren typischerweise bounded Fehler statt eines sÃ¤kularen Drifts. Deshalb kann "
            "symplektischer Euler langfristig besser aussehen als Heun, obwohl Heun hÃ¶here Ordnung besitzt.",
            body,
        )
    )

    story.append(paragraph("1d â€“ Konvergenz am halben Umlauf", styles["Subsection"]))
    story.append(
        paragraph(
            "Bei t = T/2 lautet die exakte Position r = (âˆ’rM, 0). Die log-log-Darstellung "
            "bestÃ¤tigt die erwarteten Ordnungen: Euler-Verfahren erster Ordnung, Heun zweiter "
            "Ordnung, symplektischer Euler erster Ordnung und Velocity-Verlet zweiter Ordnung.",
            body,
        )
    )
    story.append(image(TASK_1_OUTPUT / "task_1d_convergence.png", 15.0 * cm))
    story.append(paragraph("Abb. 3: Fehler bei T/2 fÃ¼r Î”t = T/200 bis T/6400.", styles["Caption"]))
    convergence = read_csv(TASK_1_OUTPUT / "task_1d_errors.csv")
    rows = [["Methode", "gemessene Steigung"]]
    for row in convergence[1:]:
        rows.append([row[0], row[-1]])
    story.append(table(rows, [9 * cm, 5 * cm]))
    story.append(
        paragraph(
            "FÃ¼r eine 1000-jÃ¤hrige Simulation ist Velocity-Verlet die sinnvollste Wahl unter "
            "den getesteten Methoden: Es ist zweiter Ordnung, zeitumkehrbar fÃ¼r das autonome "
            "Problem und besitzt gute Langzeit-Eigenschaften. Die Schrittweite muss zusÃ¤tzlich "
            "Ã¼ber eine Phasenfehler-Studie festgelegt werden.",
            body,
        )
    )

    story.append(PageBreak())
    story.append(paragraph("2. Raketenstart in der Ã„quatorebene", styles["Section"]))
    story.append(paragraph("Modell und Anfangszustand", styles["Subsection"]))
    story.append(
        paragraph(
            "Die Rakete startet am Ã„quator mit der Rotationsgeschwindigkeit der Erde. WÃ¤hrend "
            "des Brennens nimmt die Masse linear ab; nach Brennschluss wirkt nur noch die Gravitation.",
            body,
        )
    )
    story.append(equation(r"\ddot{r}=-\frac{GM}{|r|^3}r+\frac{F}{m(t)}e(t)"))
    story.append(equation(r"m(t)=m_0-\dot m\,t,\qquad F=\dot m\,v_e"))
    story.append(equation(r"r_0=(R_E,0),\qquad v_0=(0,\omega_E R_E)"))
    story.append(
        paragraph(
            "Die allgemeine rechte Seite ist im wiederverwendbaren Modul "
            "<i>rocket_model.py</i> gekapselt. Die Auswertung nutzt klassische RK4-Schritte "
            "(0.5 s fÃ¼r 2a, 1 s fÃ¼r die lÃ¤ngeren Trajektorien).",
            body,
        )
    )

    story.append(paragraph("2a â€“ Vertikaler Start", styles["Subsection"]))
    story.append(equation(r"e=e_r,\qquad t_b=\left(1-\frac{m_1}{m_0}\right)\frac{v_e}{F/m_0}"))
    story.append(
        paragraph(
            "FÃ¼r m1 = m0/19 liefert die ideale Raketengleichung Î”v = ve ln(19) = 12.956 km/s. "
            "Die Simulation erreicht geringere Geschwindigkeiten, weil wÃ¤hrend des Brennens "
            "Gravitation wirkt. Der erste getestete Wert mit nichtnegativer spezifischer Energie "
            "ist F/m0 = 18 m/sÂ²; der genaue Schwellenwert liegt zwischen 16 und 18 m/sÂ².",
            body,
        )
    )
    story.append(image(TASK_2_OUTPUT / "task_2a_vertical_launch.png", 16.2 * cm))
    story.append(paragraph("Abb. 4: HÃ¶he und vertikale Geschwindigkeit fÃ¼r fÃ¼nf vertikale Schubwerte.", styles["Caption"]))
    vertical = read_csv(TASK_2_OUTPUT / "task_2a_vertical.csv")
    rows = [["F/m0 [m/sÂ²]", "tb [s]", "vB [km/s]", "EB [MJ/kg]"]]
    for row in vertical[1:]:
        rows.append([row[0], f"{float(row[1]):.1f}", f"{float(row[3])/1000:.3f}", f"{float(row[4])/1e6:.3f}"])
    story.append(table(rows, [3.4 * cm] * 4))
    story.append(
        paragraph(
            "Ein rein vertikaler Start erzeugt keine Kreisbahn: FÃ¼r eine Umlaufbahn benÃ¶tigt "
            "die Rakete am jeweiligen Radius eine passende tangentiale Geschwindigkeit. Sie "
            "kehrt bei E < 0 zurÃ¼ck oder entkommt bei E â‰¥ 0 radial.",
            body,
        )
    )

    story.append(paragraph("2b â€“ Pitch-Programm", styles["Subsection"]))
    story.append(equation(r"e=\sin\theta\,e_r\pm\cos\theta\,e_\phi"))
    story.append(equation(r"\theta(t)=90^\circ\max\left(0,1-\frac{t}{t_p}\right)"))
    story.append(
        paragraph(
            "Der Start nach Osten nutzt die vorhandene Erdrotation und erhÃ¶ht die anfÃ¤ngliche "
            "tangentiale Geschwindigkeit. Die Klassifikation unterscheidet Absturz, RÃ¼ckfall, "
            "gebundene Bahn und Flucht. Im getesteten Raster sind ostwÃ¤rts tp = 150, 175 und "
            "200 s als Orbit klassifiziert; westwÃ¤rts stÃ¼rzen alle getesteten FÃ¤lle ab.",
            body,
        )
    )
    story.append(image(TASK_2_OUTPUT / "task_2b_pitch_program.png", 16.2 * cm))
    story.append(paragraph("Abb. 5: Ost- und westwÃ¤rtige Trajektorien des Pitch-Programms.", styles["Caption"]))
    classification = read_csv(TASK_2_OUTPUT / "task_2b_classification.csv")
    rows = [["tp [s]", "OstwÃ¤rts", "WestwÃ¤rts"]]
    for index in range(1, len(classification), 2):
        rows.append([classification[index][0], classification[index][2], classification[index + 1][2]])
    story.append(table(rows, [4 * cm, 4.5 * cm, 4.5 * cm]))

    story.append(paragraph("2c â€“ Kreisbahnen durch Circularization", styles["Subsection"]))
    story.append(equation(r"v_circ=\sqrt{\frac{GM}{R_E+h}}"))
    story.append(
        paragraph(
            "Ein einzelner Brennvorgang endet im Allgemeinen auf einer Ellipse. Die Simulation "
            "fÃ¼hrt deshalb einen zweiten, gezielten Burn am Zielradius aus. Dabei wird die "
            "Geschwindigkeit auf die lokale Kreisbahngeschwindigkeit korrigiert.",
            body,
        )
    )
    story.append(image(TASK_2_OUTPUT / "task_2c_circular_orbits.png", 15.5 * cm))
    story.append(paragraph("Abb. 6: ZielhÃ¶hen 200 km, 500 km und 1000 km nach Circularization.", styles["Caption"]))
    circular = read_csv(TASK_2_OUTPUT / "task_2c_orbits.csv")
    rows = [["h [km]", "v vor Burn [km/s]", "vcirc [km/s]", "Î”vcirc [km/s]", "m/m0 danach"]]
    for row in circular[1:]:
        rows.append([f"{float(row[0])/1000:.0f}", f"{float(row[2])/1000:.3f}", f"{float(row[3])/1000:.3f}", f"{float(row[4])/1000:.3f}", f"{float(row[5]):.4f}"])
    story.append(table(rows, [2.4 * cm] * 5))

    story.append(paragraph("2d â€“ Stufung", styles["Subsection"]))
    story.append(equation(r"\Delta v_k=v_e\ln\left(\frac{m_{0,k}}{m_{1,k}}\right)"))
    story.append(
        paragraph(
            "Beim zweistufigen Aufbau wird die leere erste Stufe abgeworfen, sodass sie nicht "
            "mehr mitbeschleunigt werden muss. Ideal ergeben sich 10.13 km/s fÃ¼r eine Stufe "
            "und 20.26 km/s fÃ¼r zwei Stufen. Die reale Simulation zeigt zusÃ¤tzlich die Verluste "
            "durch Gravitation und die nicht optimale FlugfÃ¼hrung. Mit dem unverÃ¤nderten "
            "tp = 150-s-Programm schneidet die zweistufige Testbahn nach der Trennung die Erde; "
            "dies ist ein gÃ¼ltiges Simulationsergebnis und kein erfolgreicher Orbit.",
            body,
        )
    )
    story.append(image(TASK_2_OUTPUT / "task_2d_staging.png", 16.2 * cm))
    story.append(paragraph("Abb. 7: Earth-centred Ãœbersicht und HÃ¶henprofil der Stufenvergleiche.", styles["Caption"]))
    staging = read_csv(TASK_2_OUTPUT / "task_2d_staging.csv")
    rows = [["Konfiguration", "ideales Î”v [km/s]", "Status", "Energie [MJ/kg]"]]
    for row in staging[1:]:
        rows.append([row[0], f"{float(row[1])/1000:.3f}", row[-1], f"{float(row[5])/1e6:.3f}"])
    story.append(table(rows, [4.5 * cm, 4 * cm, 3.2 * cm, 4 * cm]))
    story.append(
        paragraph(
            "<b>Zusammenfassung:</b> Die Mondaufgabe zeigt, warum symplektische Verfahren fÃ¼r "
            "Langzeitbahnen wichtig sind. Die Raketenaufgabe ergÃ¤nzt dies um variable Masse, "
            "Schubsteuerung, Erdrotation, Energie-Klassifikation und gezielte Orbitinsertion. "
            "In beiden Teilen ist die Wahl der numerischen Methode beziehungsweise der "
            "FlugfÃ¼hrung entscheidend dafÃ¼r, ob ein physikalisch plausibles Ergebnis entsteht.",
            body,
        )
    )

    document.build(story)
    print(f"Report written to {OUTPUT}")


if __name__ == "__main__":
    build_report()



