"""
A tour of `scijo.linalg`: matrix decompositions (`lu`, `qr`, `cholesky`),
eigenvalues/SVD (`eigh`, `eigvalsh`, `eig`, `eigvals`, `svd`, `svdvals`,
`matrix_rank`, `cond`), linear system solvers (`solve`, `solve_triangular`,
`inv`, `pinv`, `lstsq`), and norms (`norm`, `det`, `trace`).

Run with (from the repository root, after `pixi run package`):

```bash
mojo run -I . -I tests/ examples/linalg_examples.mojo
```
"""

# ===----------------------------------------------------------------------=== #
# External
# ===----------------------------------------------------------------------=== #
import numojo as nm

# ===----------------------------------------------------------------------=== #
# SciJo
# ===----------------------------------------------------------------------=== #
from scijo.linalg import (
    cholesky,
    cond,
    det,
    eig,
    eigh,
    eigvals,
    eigvalsh,
    inv,
    lstsq,
    lu,
    matmul,
    matrix_rank,
    norm,
    pinv,
    qr,
    solve,
    solve_triangular,
    svd,
    svdvals,
    trace,
)
from scijo.prelude import *


def main() raises:
    decompositions()
    eigenvalues_and_svd()
    solving()
    norms_and_scalars()


# ===----------------------------------------------------------------------=== #
# Decompositions
# ===----------------------------------------------------------------------=== #


def decompositions() raises:
    var A = nm.fromstring[f64]("[[4, 3], [6, 3]]")

    var PLU = lu(A)
    var P = PLU[0].copy()
    var L = PLU[1].copy()
    var U = PLU[2].copy()
    print("A == P @ L @ U:")
    print(matmul(P, matmul(L, U)))

    var B = nm.fromstring[f64]("[[1, 2], [3, 4], [5, 6]]")
    var QR = qr(B)
    var Q = QR[0].copy()
    var R = QR[1].copy()
    print("B == Q @ R:")
    print(matmul(Q, R))

    var C = nm.fromstring[f64]("[[4, 2], [2, 3]]")
    var L2 = cholesky(C)
    print("C == L @ L.T:")
    print(matmul(L2, nm.transpose(L2)))


# ===----------------------------------------------------------------------=== #
# Eigenvalues and SVD
# ===----------------------------------------------------------------------=== #


def eigenvalues_and_svd() raises:
    var A = nm.fromstring[f64]("[[2, 1], [1, 2]]")
    var w_v = eigh(A)
    print("eigh(A) eigenvalues:")
    print(w_v[0])
    print("eigh(A) eigenvectors (columns):")
    print(w_v[1])
    print("eigvalsh(A):")
    print(eigvalsh(A))

    # General (non-symmetric) matrices may have complex eigenvalues -
    # here, a 90-degree rotation has eigenvalues +-1j.
    var Rot = nm.fromstring[f64]("[[0, -1], [1, 0]]")
    print("eigvals(Rot):")
    print(eigvals(Rot))
    var w_v_general = eig(Rot)
    print("eig(Rot) eigenvectors (columns):")
    print(w_v_general[1])

    var B = nm.fromstring[f64]("[[1, 2], [3, 4], [5, 6]]")
    var usvt = svd(B)
    print("svd(B) singular values:")
    print(usvt[1])
    print("svdvals(B):")
    print(svdvals(B))
    print("cond(B):", cond(B))
    print("matrix_rank(B):", matrix_rank(B))


# ===----------------------------------------------------------------------=== #
# Solving
# ===----------------------------------------------------------------------=== #


def solving() raises:
    var A = nm.fromstring[f64]("[[3, 1], [1, 2]]")
    var b = nm.fromstring[f64]("[9, 8]")
    print("solve(A, b):")
    print(solve(A, b))

    var Lower = nm.fromstring[f64]("[[2, 0], [1, 3]]")
    print("solve_triangular(Lower, b):")
    print(solve_triangular(Lower, b, lower=True))

    print("inv(A):")
    print(inv(A))

    var Tall = nm.fromstring[f64]("[[1, 1], [1, 2], [1, 3]]")
    print("pinv(Tall):")
    print(pinv(Tall))

    var y = nm.fromstring[f64]("[6, 0, 0]")
    print("lstsq(Tall, y):")
    print(lstsq(Tall, y))


# ===----------------------------------------------------------------------=== #
# Norms, determinant, trace
# ===----------------------------------------------------------------------=== #


def norms_and_scalars() raises:
    var x = nm.fromstring[f64]("[3, 4]")
    print("norm(x):", norm(x))
    print("norm(x, ord='1'):", norm(x, ord="1"))
    print("norm(x, ord='inf'):", norm(x, ord="inf"))

    var A = nm.fromstring[f64]("[[1, 2], [3, 4]]")
    print("det(A):", det(A))
    print("trace(A):", trace(A))
