# Exercise 5 - Task 2: Time-dependent heat equation

The implementation is in [`exercise_05.py`](exercise_05.py). Running

```text
python exercises/exercise_05/exercise_05.py
```

generates the Task-1 results and the Task-2 results in
[`exercise_05_task_2_output/`](exercise_05_task_2_output/).

## 2a - Method of lines

With the same spatial operator as in Task 1,

$$
A=-D_{xx},\qquad \dot{\mathbf u}=-A\mathbf u.
$$

The script implements explicit Euler, implicit Euler, and
Crank--Nicolson:

$$
\begin{aligned}
\text{explicit Euler:}\quad&
\mathbf u_{n+1}=(I-\Delta t A)\mathbf u_n,\\
\text{implicit Euler:}\quad&
(I+\Delta t A)\mathbf u_{n+1}=\mathbf u_n,\\
\text{Crank--Nicolson:}\quad&
(I+\tfrac12\Delta t A)\mathbf u_{n+1}
=(I-\tfrac12\Delta t A)\mathbf u_n.
\end{aligned}
$$

The time-grid construction, one-step formulas, and stability functions from
Exercise 3 can be reused conceptually unchanged. The only change is that
the ODE right-hand side is now the sparse linear map $-\!A\mathbf u$.
For implicit methods the sparse factorization of the left-hand matrix is
reused for all equal-sized time steps.

## 2b - Accuracy

For the initial condition $\sin(\pi x)$, the exact PDE solution is

$$
u(x,t)=e^{-\pi^2t}\sin(\pi x).
$$

The spatially discrete problem has the exact solution

$$
\mathbf u(t)=e^{-\lambda_1t}\mathbf v_1,
\qquad
\lambda_1=\frac{4}{\Delta x^2}
\sin^2\left(\frac{\pi}{2(N+1)}\right),
$$

where $\mathbf v_1$ is the first discrete sine mode. The script compares
both references. The corresponding figures are
[`task_2b_temporal_accuracy.png`](exercise_05_task_2_output/task_2b_temporal_accuracy.png)
and
[`task_2b_spatial_accuracy.png`](exercise_05_task_2_output/task_2b_spatial_accuracy.png).

Expected orders:

| comparison | expected order |
|---|---:|
| explicit Euler in time | 1 |
| implicit Euler in time | 1 |
| Crank--Nicolson in time | 2 |
| central finite differences in space vs PDE | 2 |

When comparing with the exact ODE solution, the spatial discretization error
is removed. When comparing with the exact PDE solution, the $O(\Delta x^2)$
error remains. Refining $\Delta t$ therefore eventually reaches a plateau
set by the spatial error, floating-point round-off, and the sparse linear
solve.

## 2c - Stability and eigenmodes

For an eigenmode with eigenvalue $\lambda_k$ the semi-discrete equation is
$\dot a_k=-\lambda_k a_k$. Thus the stability argument is
$z_k=-\lambda_k\Delta t$.

The stability functions are

$$
\begin{aligned}
R_{\rm EE}(z)&=1+z,\\
R_{\rm Heun}(z)&=1+z+\frac{z^2}{2},\\
R_{\rm RK4}(z)&=1+z+\frac{z^2}{2}+\frac{z^3}{6}+\frac{z^4}{24},\\
R_{\rm IE}(z)&=\frac{1}{1-z},\\
R_{\rm CN}(z)&=\frac{1+z/2}{1-z/2}.
\end{aligned}
$$

On the negative real axis, explicit Euler and Heun are stable for
$\lambda_{\max}\Delta t\le2$, while RK4 is stable up to
$\lambda_{\max}\Delta t\le2.7852935634$. Therefore

$$
\Delta t_{\rm EE,Heun}=\frac{2}{\lambda_{\max}},\qquad
\Delta t_{\rm RK4}=\frac{2.7852935634}{\lambda_{\max}},
$$

with

$$
\lambda_{\max}=\frac{4}{\Delta x^2}
\cos^2\left(\frac{\pi}{2(N+1)}\right).
$$

The values for $N=10,20,40,80$ are in
[`task_2c_critical_dt.csv`](exercise_05_task_2_output/task_2c_critical_dt.csv).
The mode plots and stability regions are in
[`task_2c_eigenvectors_and_modes.png`](exercise_05_task_2_output/task_2c_eigenvectors_and_modes.png)
and
[`task_2c_stability_regions.png`](exercise_05_task_2_output/task_2c_stability_regions.png).

The highest eigenvalue determines stability because it produces the most
negative value of $z_k$. Just above the critical step size, the highest
mode has $|R|>1$ and alternates sign at every step, producing rapidly
growing grid-scale oscillations; this is shown in
[`task_2c_above_critical.png`](exercise_05_task_2_output/task_2c_above_critical.png).

Implicit Euler and Crank--Nicolson are stable for every positive $\Delta t$
on the negative real axis. Implicit Euler is additionally L-stable:
$R_{\rm IE}(z)\to0$ for $z\to-\infty$. Crank--Nicolson is A-stable but not
L-stable:

$$
|R_{\rm CN}(z)|\to1\quad(z\to-\infty),
$$

so very stiff modes can persist as sign-alternating components.

## 2d - Stiffness and non-smooth data

The initial step function excites many eigenmodes, including high-frequency
ones. For $N=99$, $\Delta x=0.01$ and $\Delta t=0.01$, so

$$
r=\frac{\Delta t}{\Delta x^2}=100.
$$

For large negative $z$, implicit Euler strongly damps the high modes, while
Crank--Nicolson maps them close to $-1$. Consequently Crank--Nicolson can
show small alternating wiggles and is visibly worse for the non-smooth
initial data, even though it was second-order and often best in the smooth
accuracy test. The high-mode amplitude plot is in
[`task_2d_stiffness_and_modes.png`](exercise_05_task_2_output/task_2d_stiffness_and_modes.png).

The fast modes do not disappear mathematically in finite time. Implicit Euler
reduces them geometrically; Crank--Nicolson reduces them much more slowly in
the stiff limit and may only alternate in sign.

Explicit Euler requires

$$
\Delta t\le\frac{1}{2(N+1)^2}
\approx5.0\cdot10^{-5}
\quad(N=99).
$$

The exact step counts to representative final times are recorded in
[`task_2d_explicit_steps.csv`](exercise_05_task_2_output/task_2d_explicit_steps.csv).
Since $\Delta t_{\max}=O(\Delta x^2)$, refining the grid increases the
number of explicit steps per fixed physical time like $O(N^2)$.
