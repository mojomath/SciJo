# ===----------------------------------------------------------------------=== #
# SciJo: Linalg module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Linalg Module (`scijo.linalg`)
=================================
SciPy-style linear algebra: matrix decompositions, linear system solvers,
and norms. Built on top of NuMojo's linear algebra primitives
(`numojo.linalg`), with a `scipy.linalg`-compatible surface (e.g.
1-D right-hand sides, `(P, L, U) = lu(A)`, string-valued `ord` for `norm`).

Available Functions
--------------------
- `lu`                — Pivoted LU decomposition, `A == P @ L @ U`.
- `qr`                — Reduced QR decomposition.
- `cholesky`           — Cholesky decomposition of a positive-definite matrix.
- `solve`              — Solve `Ax = b` for a square, non-singular `A`.
- `solve_triangular`   — Solve `Ax = b` for triangular `A`.
- `inv`                — Matrix inverse.
- `pinv`               — Moore-Penrose pseudo-inverse (full-rank only).
- `lstsq`              — Least-squares solution to `Ax = b`.
- `norm`               — Vector/matrix norms.
- `det`                — Determinant.
- `trace`              — Sum of the diagonal elements.

Also re-exported from `numojo.linalg` for convenience: `matmul`, `dot`,
`outer`, `kron`, `tensordot`, `cross`, `diagonal`.

Examples
--------
    ```mojo
    from scijo.linalg import solve, lu, norm
    import numojo as nm

    var A = nm.fromstring[nm.f64]("[[3, 1], [1, 2]]")
    var b = nm.fromstring[nm.f64]("[9, 8]")
    var x = solve(A, b)
    print(norm(x))
    ```

Not Yet Implemented
--------------------
Eigenvalues/eigenvectors, SVD, and anything derived from them (matrix
2-norm/condition number, rank-deficient `pinv`/`lstsq`) are not yet
available in NuMojo's linalg primitives and are tracked as future work.
"""

# ===----------------------------------------------------------------------=== #
# External
# ===----------------------------------------------------------------------=== #
from numojo.routines.linalg.misc import diagonal
from numojo.routines.linalg.norms import (
    det,
    trace,
)
from numojo.routines.linalg.products import (
    cross,
    dot,
    kron,
    matmul,
    outer,
    tensordot,
)

# ===----------------------------------------------------------------------=== #
# SciJo
# ===----------------------------------------------------------------------=== #
from .decompositions import (
    cholesky,
    lu,
    qr,
)
from .norms import norm
from .solving import (
    inv,
    lstsq,
    pinv,
    solve,
    solve_triangular,
)
