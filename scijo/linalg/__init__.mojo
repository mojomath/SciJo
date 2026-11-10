# ===----------------------------------------------------------------------=== #
# SciJo: Linalg module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Linalg Module (`scijo.linalg`)
=================================
Linear algebra: matrix decompositions, eigenvalue/SVD problems, linear
system solvers, and norms. Built on top of NuMojo's linear algebra
primitives (`numojo.linalg`).

Available Functions
--------------------
- `lu`                — Pivoted LU decomposition, `A == P @ L @ U`.
- `qr`                — Reduced QR decomposition.
- `cholesky`           — Cholesky decomposition of a positive-definite matrix.
- `eigh`               — Eigenvalues/eigenvectors of a symmetric matrix.
- `eigvalsh`           — Eigenvalues of a symmetric matrix.
- `eig`                — Eigenvalues/eigenvectors of a general (non-symmetric) matrix.
- `eigvals`            — Eigenvalues of a general (non-symmetric) matrix.
- `svd`                — Singular value decomposition, `A == U @ diag(s) @ Vt`.
- `svdvals`            — Singular values only.
- `matrix_rank`        — Estimated rank, via SVD.
- `cond`               — 2-norm condition number, via SVD.
- `solve`              — Solve `Ax = b` for a square, non-singular `A`.
- `solve_triangular`   — Solve `Ax = b` for triangular `A`.
- `inv`                — Matrix inverse.
- `pinv`               — Moore-Penrose pseudo-inverse, via SVD (rank-deficient OK).
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
from .eigen import (
    eig,
    eigh,
    eigvals,
    eigvalsh,
)
from .norms import norm
from .solving import (
    inv,
    lstsq,
    pinv,
    solve,
    solve_triangular,
)
from .svd import (
    cond,
    matrix_rank,
    svd,
    svdvals,
)
