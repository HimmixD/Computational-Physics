# Exercise 4 - Task 2: Rocket launch

The numerical implementation is in
[`exercise_04_rocket.py`](exercise_04_rocket.py), using the reusable model in
[`rocket_model.py`](rocket_model.py).  All positions are in metres, velocities
in metres per second, and times in seconds.

The equation integrated is

$$
\ddot{\mathbf r}
= -\frac{GM}{|\mathbf r|^3}\mathbf r
 + \frac{F}{m(t)}\mathbf e(t),
\qquad
m(t)=m_0-\dot m\,t,
\qquad
F=\dot m v_e.
$$

The rocket starts at

$$
\mathbf r_0=(R_E,0),\qquad
\mathbf v_0=(0,\omega_E R_E).
$$

The code uses fixed-step classical RK4.  The 0.5 s step is used for the
vertical-launch study and the 1 s step for the longer trajectory scans.

## 2a - Vertical launch

For $\theta=90^\circ$,

$$
\mathbf e=\mathbf e_r.
$$

With $m_1=m_0/19$ the burn time follows directly from
$\dot m=F/v_e$:

$$
t_b=\frac{m_0-m_1}{\dot m}
=\left(1-\frac1{19}\right)\frac{v_e}{F/m_0}.
$$

The results are in
[`task_2a_vertical.csv`](exercise_04_task_2_output/task_2a_vertical.csv) and
the altitude/speed plot is
[`task_2a_vertical_launch.png`](exercise_04_task_2_output/task_2a_vertical_launch.png).
The ideal rocket-equation velocity increment is

$$
\Delta v_\mathrm{rocket}=v_e\ln(m_0/m_1)
=4.4\,\mathrm{km/s}\ln(19)
=12.956\,\mathrm{km/s}.
$$

The numerical burnout speeds are smaller because gravity acts during the
burn.  In addition, the initial rotational speed contributes to the actual
inertial velocity, while the rocket equation describes only the velocity
increment produced by the engine in free space.

The burnout specific energy is

$$
E_b=\frac12|\mathbf v_b|^2-\frac{GM}{|\mathbf r_b|}.
$$

The tested launch accelerations show that $18\,\mathrm{m/s^2}$ is the first
listed value with $E_b\geq0$, so it escapes after burnout in this idealized
model.  Interpolating the coarse list is not used as a physical threshold;
the exact threshold lies between 16 and 18 m/s$^2$ and should be found with a
finer parameter scan.

A purely vertical launch cannot create a circular orbit.  At burnout it has
almost no tangential velocity, whereas an orbit requires the local circular
speed in the tangential direction.  It either falls back if bound or escapes
radially if its energy is non-negative.

## 2b - Pitch program

The implemented direction is exactly

$$
\mathbf e=\sin\theta\,\mathbf e_r
\mathord{\pm}\cos\theta\,\mathbf e_\phi,
\qquad
\theta(t)=90^\circ\max(0,1-t/t_p).
$$

The classifications for $t_p=100,125,\ldots,300$ s are in
[`task_2b_classification.csv`](exercise_04_task_2_output/task_2b_classification.csv),
and the trajectories are in
[`task_2b_pitch_program.png`](exercise_04_task_2_output/task_2b_pitch_program.png).
For the implemented scan the best eastward cases are around $t_p=150$--$200$
s; the selected value is recorded in `summary.txt`.  Eastward launch is
preferable because Earth already supplies the rotational velocity in the
eastward direction.  A westward launch starts with a tangential velocity
penalty and therefore needs more propellant velocity increment.

The labels mean: `crash` intersects the Earth, `falls back` remains bound and
returns to the surface, `orbit` remains above the surface with negative
specific energy for the simulated interval, and `escape` reaches
$E\geq0$ after burnout.

## 2c - Circular orbits

A single pitch-program burn generally leaves an ellipse because the velocity
at engine cutoff is not exactly the circular velocity at the current radius.
The implemented strategy is:

1. perform the eastward pitch-program ascent;
2. coast until the trajectory crosses the desired altitude;
3. relight the engine at that point and apply the velocity correction
   $\Delta\mathbf v=\mathbf v_\mathrm{circ}-\mathbf v$;
4. coast after circularization.

The circular speed is

$$
v_\mathrm{circ}=\sqrt{\frac{GM}{R_E+h}}.
$$

The result is shown in
[`task_2c_circular_orbits.png`](exercise_04_task_2_output/task_2c_circular_orbits.png)
and tabulated in
[`task_2c_orbits.csv`](exercise_04_task_2_output/task_2c_orbits.csv).  The
fuel remaining after the insertion burn is estimated with the rocket equation
from the 12% mass remaining after ascent:

$$
m_\mathrm{after}=0.12m_0\exp(-\Delta v_\mathrm{circ}/v_e).
$$

The correction has to be different at each altitude because both the required
circular speed and the ascent state vary with radius.  One fixed burn cannot
simultaneously satisfy the circular-orbit condition at several radii.

## 2d - Two stages

For the one-stage case, the complete mass is $m_0$ and 90% is fuel.  For the
two-stage case, the first stage has mass $0.8m_0$ and the second stage
$0.2m_0$; each stage contains 90% fuel.  The empty first stage is dropped
before the second stage ignites.

During first-stage burning, the remaining second-stage mass is included in
$m(t)$; after separation only the second-stage mass remains.
Following the exercise's specification, the first-stage input uses
$F/m_0=15\,\mathrm{m/s^2}$ and the second-stage engine uses
$F/(0.2m_0)=15\,\mathrm{m/s^2}$.

For each stage,

$$
\Delta v_k=v_e\ln\left(\frac{m_{0,k}}{m_{1,k}}\right).
$$

Thus the ideal values are approximately $10.13$ km/s for one stage and
$20.26$ km/s for two identical mass ratios.  The trajectories and numerical
values are in
[`task_2d_staging.png`](exercise_04_task_2_output/task_2d_staging.png) and
[`task_2d_staging.csv`](exercise_04_task_2_output/task_2d_staging.csv).

Staging helps because the empty first-stage structure is no longer accelerated
by the second stage.  The actual velocity gain is lower than the ideal
rocket-equation value because gravity acts during the burns and because the
rocket spends time climbing rather than converting all thrust into inertial
velocity.  For the one-stage trajectory in the generated table, the
ideal-to-net-speed difference is about $0.988\,\mathrm{km/s}$.  The prescribed
$t_p=150$ s pitch profile causes the two-stage trajectory to intersect the
Earth after the first-stage separation; its reported speed is therefore only
the last valid pre-crash value and is not a successful-orbit performance
measure.  A real staged launch would retune the pitch guidance after
separation (for example, by holding a prograde flight-path angle) so that the
additional ideal $\Delta v$ can be used rather than lost to gravity.
