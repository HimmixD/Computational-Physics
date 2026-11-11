# Exercise 5 - Task 1: Finite differences

The implementation is in [`exercise_05.py`](exercise_05.py). Run it from the
repository root:

```text
python exercises/exercise_05/exercise_05.py
```

The script writes all numerical tables and figures to
[`exercise_05_task_1_output/`](exercise_05_task_1_output/).

## 1a - Stationary Dirichlet problem

For the interior grid points

$$x_j=j\Delta x,\qquad \Delta x=\frac{1}{N+1},\qquad j=1,\ldots,N,$$

the central three-point approximation is

$$
u''(x_j)\approx\frac{u_{j-1}-2u_j+u_{j+1}}{\Delta x^2}.
$$

With homogeneous boundary values, the matrix for $A=-d^2/dx^2$ is

$$
A=\frac{1}{\Delta x^2}
\begin{pmatrix}
2&-1&&\\
-1&2&-1&\\
&\ddots&\ddots&\ddots\\
&&-1&2
\end{pmatrix}.
$$

Before solving, the matrix can be checked independently of the right-hand
side: it must have shape `(N, N)`, exactly `3N-2` nonzero entries, be
symmetric positive definite, and have the expected diagonal/off-diagonal
pattern. In the script these checks are printed for `N=7`. The eigenvalues

$$
\lambda_k=\frac{4}{\Delta x^2}\sin^2\left(\frac{k\pi}{2(N+1)}\right)
$$

are positive, which is an additional check of the sign convention and
boundary treatment.

## 1b - Convergence and round-off

The exact solution is $u(x)=\sin(\pi x)$, and the truncation error of the
central stencil is $O(\Delta x^2)$. The measured slope in the log-log plot
[`task_1b_convergence.png`](exercise_05_task_1_output/task_1b_convergence.png)
is therefore approximately 2.

For very fine grids the error no longer decreases. The discretization error
eventually becomes smaller than floating-point and sparse-solver errors.
Moreover, the condition number of the discrete Laplacian grows like
$O(\Delta x^{-2})$, so round-off is amplified. The script includes the
requested sequence through $N=10^6$; the turn-around is visible in the CSV
file and plot.

## 1c - Higher-order stencils

The script constructs finite-difference weights by differentiating the
interpolating polynomial. This makes the stencil generator reusable rather
than hard-coding individual coefficients. Five-point and seven-point
stencils give formal orders four and six, respectively, and the measured
slopes are listed in `task_1c_orders.csv`.

Near a boundary a centered stencil would require points outside the domain.
The implementation shifts the stencil inward and uses a slightly wider
one-sided stencil there. Boundary values are known and are omitted from the
unknown vector. This preserves the intended formal order without inventing
exterior values.

## 1d - Mixed Dirichlet/Neumann boundary conditions

Here

$$
u(x)=\sin\left(\frac{\pi x}{2}\right),\qquad
f(x)=\frac{\pi^2}{4}\sin\left(\frac{\pi x}{2}\right).
$$

At $x=1$ the derivative is imposed with a second-order backward relation,

$$
\frac{3u_{N+1}-4u_N+u_{N-1}}{2\Delta x}=0
\quad\Longrightarrow\quad u_{N+1}=\frac{4u_N-u_{N-1}}{3}.
$$

Substitution into the centered equation at $x_N$ gives the last row

$$
A_{N,N-1}=-\frac{2}{3\Delta x^2},\qquad
A_{N,N}=\frac{2}{3\Delta x^2}.
$$

The boundary condition is second-order accurate, so the global maximum error
also converges with slope approximately 2. A lower-order one-sided derivative
would contaminate the global solution: elliptic problems couple every grid
point, so a first-order boundary truncation error generally reduces the
overall convergence order to first order.

## 1e - Two-dimensional extension

For the tensor-product grid, the discrete Laplacian is assembled as

$$
A_{2D}=I\otimes A_{1D}+A_{1D}\otimes I.
$$

This is the Kronecker-sum construction suggested in the assignment. With
$u(x,y)=\sin(\pi x)\sin(\pi y)$, the right-hand side is
$2\pi^2\sin(\pi x)\sin(\pi y)$. The maximum error is $O(\Delta x^2)$.

The number of unknowns is $N^2$. The sparse matrix has only $O(N^2)$
nonzeros, but a direct sparse factorization also introduces fill-in, so the
measured runtime grows substantially faster than linearly in the number of
unknowns. The exact timing depends on the sparse ordering and the machine;
the generated CSV records the measurements for the current run.
