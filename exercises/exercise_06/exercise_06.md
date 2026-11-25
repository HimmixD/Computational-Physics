# Exercise 6 - Aufgabe 1: stationäre Schrödingergleichung

Die Implementierungen stehen in [`exercise_06_task_1.py`](exercise_06_task_1.py)
und [`exercise_06_task_2.py`](exercise_06_task_2.py). Sie werden mit

```text
python exercises/exercise_06/exercise_06_task_1.py
python exercises/exercise_06/exercise_06_task_2.py
```

ausgeführt. Die numerischen Daten und neun geforderten Plots werden nach
[`exercise_06_task_1_output/`](exercise_06_task_1_output/) geschrieben.

Für einen äquidistanten Dirichlet-Gridsatz
$x_j=j\Delta x$ wird die zweite Ableitung durch

$$
\psi''(x_j)\approx\frac{\psi_{j-1}-2\psi_j+\psi_{j+1}}{\Delta x^2}
$$

diskretisiert. Damit entsteht die tridiagonale Matrix

$$
H_{jj}=\frac{1}{\Delta x^2}+V(x_j),\qquad
H_{j,j\pm1}=-\frac{1}{2\Delta x^2},
$$

also exakt die angegebene Hamiltonmatrix mit $\hbar=m=1$. Der Fehler des
zentralen Dreipunkt-Sterns ist lokal $O(\Delta x^2)$.

## 1a - Harmonischer Oszillator

Mit $V(x)=x^2/2$ lautet die analytische Energie

$$
E_n=n+\frac12,
$$

wobei $n=0,1,2,\ldots$. Das verwendete feste Gitter hat
$N=800$ innere Punkte auf $[-8,8]$, also $\Delta x=0.019975031$.
Die ersten Ergebnisse sind:

| $n$ | script | analytisch | absoluter Fehler |
|---:|---:|---:|---:|
| 0 | 0.49998753088 | 0.50000000000 | $1.25\cdot10^{-5}$ |
| 1 | 1.49993765316 | 1.50000000000 | $6.23\cdot10^{-5}$ |
| 5 | 5.49923929661 | 5.50000000000 | $7.61\cdot10^{-4}$ |
| 10 | 10.497243668 | 10.50000000000 | $2.76\cdot10^{-3}$ |
| 11 | 11.496694814 | 11.50000000000 | $3.31\cdot10^{-3}$ |

Die tief liegenden Zustände sind daher sehr genau. Höhere Zustände werden
zuerst ungenauer, weil ihre Wellenfunktionen weiter außen eine größere
Amplitude haben und der endlich gewählte rechte/linke Dirichlet-Rand bei
$x=\pm8$ die Wellenfunktion stärker beeinflusst. Zusätzlich wird das
Diskretisierungsgitter für die zunehmend schnell variierenden Zustände
relativ grob. Die Genauigkeit ist somit durch die endliche Box, durch
$\Delta x$ und für sehr feine Gitter schließlich durch Rundungs-/Eigenlöserfehler
begrenzt. Der globale Fehler der unteren Zustände fällt bei Verfeinerung
annähernd quadratisch mit $\Delta x$.

Erzeugte Plots:

- [`task_1a_energies.png`](exercise_06_task_1_output/task_1a_energies.png)
- [`task_1a_error_vs_n.png`](exercise_06_task_1_output/task_1a_error_vs_n.png)
- [`task_1a_error_vs_dx.png`](exercise_06_task_1_output/task_1a_error_vs_dx.png)

## 1b - Unendlicher Kasten

Auf $[0,1]$ gilt

$$
E_n=\frac{n^2\pi^2}{2},\qquad n=1,2,\ldots
$$

Das feste Gitter verwendet $N=1000$ innere Punkte, $\Delta x\approx0.001$.
Die Fehler steigen deutlich mit $n$:

| $n$ | script | analytisch | absoluter Fehler |
|---:|---:|---:|---:|
| 1 | 4.93479815007 | 4.93480220054 | $4.05\cdot10^{-6}$ |
| 5 | 123.367523405 | 123.370055014 | $2.53\cdot10^{-3}$ |
| 10 | 493.439715316 | 493.480220054 | $4.05\cdot10^{-2}$ |
| 12 | 710.527527466 | 710.611516878 | $8.40\cdot10^{-2}$ |

Die exakte Wellenfunktion besitzt für größere $n$ eine kürzere räumliche
Wellenlänge $\lambda_n=2/n$. Daher braucht ein hoher Zustand deutlich feinere
Gitter: Der relative Fehler ist bei festem $\Delta x$ für hohe $n$ größer.
Für jeden einzelnen Zustand sollte die Energiefehlerkurve bei Verfeinerung
mit steigender Gitterauflösung quadratisch abfallen, solange der Solver-bzw.
Rundungsfehler nicht dominiert. Die Randpunkte werden durch die
Dirichletbedingung $\psi=0$ unmittelbar erfüllt.

Erzeugte Plots:

- [`task_1b_energies.png`](exercise_06_task_1_output/task_1b_energies.png)
- [`task_1b_error_vs_n.png`](exercise_06_task_1_output/task_1b_error_vs_n.png)
- [`task_1b_error_vs_dx.png`](exercise_06_task_1_output/task_1b_error_vs_dx.png)

## 1c - Endlicher Kasten

Mit $V_0=10$, $a=1$ und $V=-10$ innerhalb $|x|<1$ bzw. $V=0$ außerhalb
führen die Ansätze in den Bereichen zu den gegebenen Gleichungen:

$$
 k\tan(ka)=\kappa\quad\text{(gerade Zustände)},
$$

$$
-k\cot(ka)=\kappa\quad\text{(ungerade Zustände)},
$$

mit

$$
k=\sqrt{2(E+V_0)},\qquad \kappa=\sqrt{-2E}.
$$

Numerisch ergeben sich auf $[-6,6]$ mit $N=1200$ genau drei gebundene
Zustände, ebenso analytisch aus den Transzendentengleichungen:

| $n$ | Parität | script | analytisch | absoluter Fehler |
|---:|---|---:|---:|---:|
| 1 | gerade | -9.17923230751 | -9.18025992623 | $1.03\cdot10^{-3}$ |
| 2 | ungerade | -6.77520702144 | -6.77906002141 | $3.85\cdot10^{-3}$ |
| 3 | gerade | -3.04687054696 | -3.05423350877 | $7.36\cdot10^{-3}$ |

Die Reihenfolge ist gerade, ungerade, gerade, passend zum Paritätsschema.
Die Anzahl gebundener Zustände folgt aus $z_0=a\sqrt{2V_0}=\sqrt{20}$ und den
Tangent-/Kotangentenschnitten. Hier ist $z_0\approx4.472$; es gibt genau
drei negative Energiezustände.

Die endliche Box muss deutlich größer als $a=1$ sein, weil die gebundenen
Wellenfunktionen außerhalb exponentiell abfallen. Die gewählte Box $[-6,6]$
ist auf dieser Skala ausreichend; ein zu kleiner äußerer Bereich würde durch
die künstliche Nullrandbedingung die tiefsten Energien erhöhen. Wieder steigt
der Fehler mit dem Zustand an, da der dritte Zustand weniger stark gebunden
ist und eine größere Außenwellenlänge besitzt. Die Verfeinerungskurven gegen
$\Delta x$ zeigen für alle drei Zustände die erwartete quadratische
Diskretisierungskonvergenz bis zu den numerischen Rundungsgrenzen.

Erzeugte Plots:

- [`task_1c_energies.png`](exercise_06_task_1_output/task_1c_energies.png)
- [`task_1c_error_vs_n.png`](exercise_06_task_1_output/task_1c_error_vs_n.png)
- [`task_1c_error_vs_dx.png`](exercise_06_task_1_output/task_1c_error_vs_dx.png)

# Aufgabe 2 - zeitabhängige eindimensionale Schrödingergleichung

Die zeitabhängige Rechnung löst

$$
i\partial_t\psi=H\psi
$$

mit dem Matrix-Hamiltonoperator der zentralen Dreipunkt-Diskretisierung.
Für $N_t$ Zeitschritte gilt bei der expliziten Euler-Methode

$$
\psi^{n+1}=(I-i\Delta tH)\psi^n,
$$
bei impliziter Euler-Methode

$$
(I+i\Delta tH)\psi^{n+1}=\psi^n,
$$
und beim Crank-Nicolson-Verfahren

$$
\left(I+i\frac{\Delta t}{2}H\right)\psi^{n+1}
=
\left(I-i\frac{\Delta t}{2}H\right)\psi^n.
$$

Das normierte Anfangspaket ist

$$
\psi(x,0)=\frac{1}{(2\pi\sigma^2)^{1/4}}
\exp\left(-\frac{(x-x_0)^2}{4\sigma^2}\right)e^{ik_0x}.
$$

## 2a - Freies Paket

Für $\sigma=1$, $k_0=2$ und $x_0=-5$ wurden die drei Zeitintegratoren auf
$[-30,30]$ mit $\Delta x=0.10$ und $\Delta t=0.0001$ verglichen. Die
analytischen Ergebnisse sind

$$
\|\psi(t)\|=1,\qquad
\langle x\rangle(t)=x_0+k_0t,
$$

sowie

$$
\sigma(t)=\sqrt{\sigma^2+\frac{t^2}{4\sigma^2}}.
$$

Der erzeugte Vergleichsplot zeigt alle drei Integratoren gemeinsam:

- [`task_2a_free_method_comparison.png`](exercise_06_task_2_output/task_2a_free_method_comparison.png)

Nur Crank-Nicolson bewahrt die diskrete Norm für beliebige Zeitschritte:
bei $A=I+i\Delta tH/2$ erfüllt die Cayley-Iteration
$\psi^{n+1}=A^{-1}A^*\psi^n$ mit hermiteschem $H$ die Gleichung
$\|\psi^{n+1}\|=\|\psi^n\|$. Expliziter Euler erzeugt künstlich zusätzliche
Norm ($I-i\Delta tH$ ist nicht unitär) und kann instabil werden. Impliziter
Euler ist stabil, aber gedämpft: $\|(I+i\Delta tH)^{-1}\|<1$.

Die zusätzliche Rasterstudie verwendet $k_0=1,2,4,8$ und
$\Delta x=0.5,0.25,0.125,0.0625$ bei $t=2$. Die Heatmap enthält den
$L^2$-Fehler der numerischen Dichtewellenfunktion gegenüber dem freien
analytischen Paket:

- [`task_2a_grid_size_vs_k0.png`](exercise_06_task_2_output/task_2a_grid_size_vs_k0.png)
- [`task_2a_grid_study.csv`](exercise_06_task_2_output/task_2a_grid_study.csv)

Praktisch muss das Gitter mehrere Punkte pro räumlicher Wellenlänge
$\lambda=2\pi/k_0$ besitzen. Bei fester Wellenzahl ist die Auflösung
proportional zu $\lambda$, also grober für kleines $k_0$ und feiner für großes
$k_0$. Eine brauchbare Richtlinie ist mindestens 10–20 Punkte pro
Wellenlänge. In der Heatmap wird dies sichtbar: gleiche relative Auflösung
$\Delta x/\lambda$ führt näherungsweise zu ähnlichem Fehler.

## 2b - Harmonischer Oszillator

Als Anfangszustand wird das zum Zentrum $x_0=3$ verschobene Grundzustandspaket
$x_0=3$, $\sigma=1/\sqrt2$ verwendet. Für dieses kohärente Zustandspaket ist

$$
\langle x\rangle(t)=x_0\cos t,
$$

während die Breite konstant bleibt. Der Vergleich mit dem klassischen
Teilchen und die Breitenkurve stehen gemeinsam in einem Plot:

- [`task_2b_harmonic_comparison.png`](exercise_06_task_2_output/task_2b_harmonic_comparison.png)

Das quantenmechanische Paket folgt im harmonischen Potential der klassischen
Zentrumstrajektorie. Anders als ein freies Paket ändert die Breite des
Grundzustands im harmonischen Oszillator nicht: die Gesamtkurve ist eine
kohärente Oszillation des verschobenen Grundzustands. Das Paket verbreitert
sich daher nicht, sondern das Wahrscheinlichkeitsmaximum pendelt mit $x_0$.

## 2c - Unendlicher Kasten

Für $x\in[0,1]$ wird der Dirichlet-Kasten mit dem engen Paket
$x_0=0.3$, $\sigma=0.05$, $k_0=0$ gerechnet. Die Eigenenergieabstände
sind

$$
E_n-E_m=\frac{\pi^2}{2}(n^2-m^2).
$$

Der größte gemeinsame Abstand der Eigenfrequenzen führt zur Quantendynamik
mit

$$
T_{rev}=\frac{4}{\pi}
$$

und einer halbperiodischen Spiegelstruktur bei $T_{rev}/2$. Der Plot enthält
sowohl die Zeitdichte $|\psi(x,t)|^2$ als Heatmap als auch Schnitte bei
$0$, $T_{rev}/4$, $T_{rev}/2$ und $T_{rev}$:

- [`task_2c_infinite_well_revivals.png`](exercise_06_task_2_output/task_2c_infinite_well_revivals.png)

Das Paket wird zunächst am linken/rechten Rand reflektiert und zerfasert
zwischenzeitlich. Bei $T_{rev}/2$ erscheint eine räumlich gespiegelte
Dichteverteilung bei $1-x_0$, weil sich alle Phasen um ganzzahlige Vielfache
von $\pi$ aufgebaut haben. Bei $T_{rev}$ ist die ursprüngliche Verteilung
beinahe wiederhergestellt, da sämtliche Eigenzustandsphasen wieder gemeinsam
mit ganzzahligen Vielfachen von $2\pi$ auftreten.

## 2d - Tunneln und komplexes Absorptionspotenzial

Die Verwendung von $\sigma=5$ und $k_0=\sqrt{2E}$ erzeugt Paketenergien
$E=k_0^2/2$ im Bereich $0.2$ bis $2.5$. Die numerisch gemessene rechte
Transmissionswahrscheinlichkeit wird nach dem Durchgang mit

$$
P_T=\int_{x>b}|\psi|^2dx
$$

bestimmt. Für $E<V_0$ wird sie mit

$$
T=
\left[1+
\frac{V_0^2\sinh^2(\kappa b)}
{4E(V_0-E)}
\right]^{-1}
$$

verglichen, wobei $\kappa=\sqrt(2(V_0-E))$.

Der geforderte Vergleich zeigt im linken Panel die analytische Kurve,
das Modell ohne Absorber und das Modell mit dem komplexen Potenzial
$V-iW(x)$ an den Domainenden. Das rechte Panel zeigt die kumulative rechte
Wahrscheinlichkeit als Funktion der Zeit für $E\approx0.2$:

- [`task_2d_tunneling_and_absorbing_boundary.png`](exercise_06_task_2_output/task_2d_tunneling_and_absorbing_boundary.png)
- [`task_2d_transmission.csv`](exercise_06_task_2_output/task_2d_transmission.csv)

Bei Erreichen des rechten Domainrandes werden die Pakete ohne Absorber
reflektiert. Diese künstlich zurücklaufende Dichte kann später wieder die
Messregion erreichen oder sich mit dem direkt transmittierten Paket
überlagern, wodurch die gemessene Transmission verfälscht wird. Das komplexe
Potenzial $V-iW(x)$ wirkt nahe der Ränder wie eine kontrollierte
elektronenarme Verlustzone: die Amplitude nimmt dort gemäß
$-iW$ ab, während der physikalische Bereich in der Mitte nahezu unverändert
bleibt. Daher werden Reflexionen unterdrückt. $W(x)$ sollte glatt von Null
zum Rand anwachsen, damit zusätzliche numerische Reflexion vermieden wird.

# Aufgabe 3 - zweidimensionale Schrödingergleichung

Die Implementierung steht in [`exercise_06_task_3.py`](exercise_06_task_3.py)
und erzeugt alle Daten und Abbildungen in
[`exercise_06_task_3_output/`](exercise_06_task_3_output/). Ausführung:

```text
python exercises/exercise_06/exercise_06_task_3.py
```

Aufgabe 3a und 3b verwenden 2D-Split-Operatoren mit separaten kinetischen
FFT-Phasen und Potentialphasen. Aufgabe 3c verwendet ein 2D-Finite-Difference-
Gitter und eine normerhaltende Crank-Nicolson-Methode. Das magnetische Gitter
Hamiltonian enthält Peierls-Phasen auf allen Gitterbindungen.

## 3a - Doppelspalt

Ein Paket mit $k_0=5$ propagiert in positive $y$-Richtung auf eine Wand bei
$y=12$. Die Wand hat Dicke $0.5$, Höhe $V=200$ und die zwei Öffnungen der
Breite $1$ sind bei $x=-2$ und $x=2$. Damit ist der Mittenabstand $d=4$.
Die Intensität wird über $27\leq y\leq29$ zeitlich akkumuliert.

Mit $\lambda=2\pi/k_0=1.256637$ werden die Haupt-Richtungen aus
$\sin\theta_m=m\lambda/d$ verglichen. Die Tabellen und Profildaten stehen in
[`task_3a_maxima.txt`](exercise_06_task_3_output/task_3a_maxima.txt) und
[`task_3a_screen_profiles.csv`](exercise_06_task_3_output/task_3a_screen_profiles.csv).
Der Vergleichsplot ist
[`task_3a_double_slit_comparison.png`](exercise_06_task_3_output/task_3a_double_slit_comparison.png).
Beide Kurven sind flächenweise auf die gleiche Normierung gebracht.

Die Summe der Einzelspaltprofile ist die inkohärente Intensitätssumme.
Sie enthält keinen Interferenzterm
$2\operatorname{Re}(\psi_1^*\psi_2)$. Daher unterscheidet sich das echte
Doppelspaltprofil dort von der Einzelsumme durch konstruktive und destruktive
Interferenz. Der theoretische Winkelvergleich bleibt unabhängig von einer
gemeinsamen Profilnormierung gültig.

## 3b - Mach-Zehnder-Interferometer

Das Paket startet bei $x=-13$, $y=-8$ und bewegt sich entlang $+x$ mit
$k_0=8$ auf den ersten 45°-Strahlteiler $y=-x-16$ bei $(-8,-8)$. BS1 teilt das Paket in
(1) einen unteren horizontalen Arm entlang $y=-8$ und (2) einen oberen vertikalen Arm entlang $x=-8$ auf.
Zwei 45°-harte Spiegel drehen die Wege: der Spiegel $y=x-12$ dreht den unteren Arm bei $(4,-8)$ nach oben, der Spiegel $y=x+12$ dreht den oberen Arm bei $(-8,4)$ nach rechts. Die beiden Arme treffen am
zweiten 45°-Strahlteiler $y=-x+8$ bei $(4,4)$ zusammen. Rechte und obere Regionen nach
BS2 dienen als Messports.

Für die $45°$-Geometrie wird die normal zum dünnen Splitter stehende
Komponente auf $E_n=16$ gesetzt. Die Barrierenformel liefert
$V_{BS}\approx29.40184278$; dabei sind $T=R=1/2$ bis auf Rundungsfehler.
Wert und Prüfdaten stehen in
[`task_3b_beam_splitter_tuning.csv`](exercise_06_task_3_output/task_3b_beam_splitter_tuning.csv).
Die echte 2D-Geometrie und die resultierende Dichte sind in
[`task_3b_mzi_geometry.png`](exercise_06_task_3_output/task_3b_mzi_geometry.png)
gezeigt.

Die ausgegebenen Wahrscheinlichkeiten werden auf die Summe der beiden
Messports normiert, sodass die CSV-Werte die Portverteilung darstellen:

- [`task_3b_phase_disk.csv`](exercise_06_task_3_output/task_3b_phase_disk.csv)
- [`task_3b_middle_disk.csv`](exercise_06_task_3_output/task_3b_middle_disk.csv)
- [`task_3b_mzi_phase_sweep.png`](exercise_06_task_3_output/task_3b_mzi_phase_sweep.png)

Ein skalares $V_{phase}$ in nur einem Arm erzeugt eine relative Phase
$\exp(-iV_{phase}t)$ zwischen den Wegen und verschiebt die Interferenz
am zweiten Strahlteiler. Ein gleich großes, mittiges Potential wird dagegen
von beiden möglichen Wegen in symmetrischer Weise durchlaufen. Es ändert die
gemeinsame oder fast gemeinsame Phase, nicht die relative Armphase. Kleine
Reständerungen in einem diskreten Paket sind daher numerische Geometrie- und
Messartefakte; im idealen MZI bleibt der mittlere Scheibenterm wirkungslos.

## 3c - Aharonov-Bohm-Effekt

Die MZI-Geometrie bleibt erhalten. Der skalare Scheibenterm in der Mitte wird
jedoch durch einen Solenoid mit Radius $r_0=1$ ersetzt. Seine Mitte ist bei
$(0,0)$, konsistent mit der Position des mittleren Scheibenortes. Das
Coulomb-/symmetrische Vektorpotential in Bezug auf diese Mitte lautet

$$
\mathbf A(\mathbf r)=
\begin{cases}
\left(-\frac{B_0y}{2},\frac{B_0x}{2}\right),&r\le r_0,\\
\left(-\frac{B_0r_0^2y}{2r^2},\frac{B_0r_0^2x}{2r^2}\right),&r>r_0.
\end{cases}
$$

Der Crank-Nicolson-Hamiltonian verwendet auf jeder Kante eine Peierls-Phase,
$H_{j,k}=-e^{iqA\cdot\Delta r}/(2\Delta r^2)$ für die kinetische 5-Punkt-
Diskretisierung und $+A^2/(2)$ im Diamantterm. Die resultierenden Portwerte
stehen in
[`task_3c_AB_output_probabilities.csv`](exercise_06_task_3_output/task_3c_AB_output_probabilities.csv)
und der Plot in
[`task_3c_aharonov_bohm_ports.png`](exercise_06_task_3_output/task_3c_aharonov_bohm_ports.png).

Entlang der klassischen MZI-Wege ist $\mathbf B=\nabla\times\mathbf A=0$;
es wirkt daher keine klassische Lorentzkraft. Nichtlokal beeinflusst
$\mathbf A$ jedoch die beiden geschlossenen Wegamplituden über

$$
\Delta\phi=q\oint\mathbf A\cdot d\mathbf r
=q\int_S\mathbf B\cdot d\mathbf S
=q\pi r_0^2B_0.
$$

Damit verschiebt sich die relative AB-Phase und die Portwahrscheinlichkeit
ändert sich trotz verschwindendem $\mathbf B$ auf den Teilchenwegen.

Eine Eichtransformation $\mathbf A\to\mathbf A+\nabla\chi$ verändert
observable Phasen nur um Eichgrößen. $\mathbf A$ kann außerhalb eines
kompakten Gebiets nicht für einen global geschlossenen Pfad bei nicht
verschwindendem eingeschlossenen Fluss gleichzeitig null gemacht werden:
sonst wäre das Linienintegral um den Flussschlauch gleich null. Lokal kann
man $\mathbf A=0$ wählen, aber nicht mit einer überall eindeutigen globalen
Eichung. Die gegebene symmetrische Lösung ist außerhalb wegen $r^{-1}$ zwar
nicht kompakt abgestützt, erzeugt aber genau den gewünschten AB-Fluss.
