"""
A tour of `scijo.linalg`: matrix decompositions (`lu`, `qr`, `cholesky`),
eigenvalues/SVD (`eigh`, `eigvalsh`, `svd`, `svdvals`, `matrix_rank`,
`cond`), linear system solvers (`solve`, `solve_triangular`, `inv`,
`pinv`, `lstsq`), norms (`norm`, `det`, `trace`), and matrix
functions/structured matrices (`expm`, `block_diag`, `toeplitz`,
`null_space`, `orth`).

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
    block_diag,
    cholesky,
    cond,
    det,
    eigh,
    eigvalsh,
    expm,
    inv,
    lstsq,
    lu,
    matmul,
    matrix_rank,
    norm,
    null_space,
    orth,
    pinv,
    qr,
    solve,
    solve_triangular,
    svd,
    svdvals,
    toeplitz,
    trace,
)
from scijo.prelude import *


def main() raises:
    decompositions()
    eigenvalues_and_svd()
    solving()
    norms_and_scalars()
    matrix_functions()


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


# ===----------------------------------------------------------------------=== #
# Matrix functions and structured matrices
# ===----------------------------------------------------------------------=== #


def matrix_functions() raises:
    print()
    print("=" * 80)
    print("MATRIX FUNCTIONS: expm, block_diag, toeplitz, null_space, orth")
    print("=" * 80)

    var A = nm.fromstring[f64]("[[0, 1], [-1, 0]]")
    print("expm([[0, 1], [-1, 0]]) (a rotation by 1 radian):")
    print(expm(A))

    var B1 = nm.fromstring[f64]("[[1, 2], [3, 4]]")
    var B2 = nm.fromstring[f64]("[[5]]")
    var blocks: List[nm.NDArray[f64]] = [B1^, B2^]
    print("block_diag([[1,2],[3,4]], [[5]]):")
    print(block_diag(blocks))

    var c = nm.fromstring[f64]("[1, 2, 3]")
    print("toeplitz([1, 2, 3]) (symmetric):")
    print(toeplitz(c))

    var singular = nm.fromstring[f64]("[[1, 2], [2, 4]]")
    print("null_space([[1, 2], [2, 4]]) (rank-deficient):")
    print(null_space(singular))
    print("orth([[1, 2], [2, 4]]):")
    print(orth(singular))
