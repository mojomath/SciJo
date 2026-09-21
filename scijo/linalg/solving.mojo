# ===----------------------------------------------------------------------=== #
# SciJo: Linalg module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Linear System Solvers (`scijo.linalg.solving`)
=================================================
SciPy-style wrappers for solving linear systems, matrix inversion, and
least-squares problems, built on top of NuMojo's linear algebra primitives.

Examples
--------
    ```mojo
    from scijo.linalg import solve
    import numojo as nm

    var A = nm.fromstring[nm.f64]("[[3, 1], [1, 2]]")
    var b = nm.fromstring[nm.f64]("[9, 8]")
    var x = solve(A, b)
    ```
"""

# ===----------------------------------------------------------------------=== #
# External
# ===----------------------------------------------------------------------=== #
from numojo.core.error import NumojoError
from numojo.core.indexing.item import Item
from numojo.core.ndarray import NDArray
from numojo.core.type_aliases import Shape
from numojo.routines.creation import (
    eye,
    full,
)
from numojo.routines.linalg.solving import (
    inv as _inv,
    lstsq as _lstsq,
    solve as _solve,
)


def solve[
    dtype: DType
](A: NDArray[dtype], b: NDArray[dtype]) raises -> NDArray[dtype]:
    """Solve the linear system `Ax = b`.

    Thin wrapper around `numojo.linalg.solve` that additionally accepts a
    1-D right-hand side, matching `scipy.linalg.solve`.

    Parameters:
        dtype: Data type of the solution.

    Args:
        A: Non-singular, square matrix of shape `(m, m)`.
        b: Right-hand side, of shape `(m,)` or `(m, n)`.

    Returns:
        The solution `x`, matching the shape of `b`.
    """
    if b.ndim == 1:
        var m = b.shape[0]
        var B = b.reshape(Shape(m, 1))
        var X = _solve(A, B)
        return X.reshape(Shape(m))
    return _solve(A, b)


def inv[dtype: DType](A: NDArray[dtype]) raises -> NDArray[dtype]:
    """Compute the inverse of a non-singular square matrix.

    Thin wrapper around `numojo.linalg.inv`.

    Parameters:
        dtype: Data type of the inverse matrix.

    Args:
        A: Non-singular, square matrix.

    Returns:
        The inverse of `A`.
    """
    return _inv(A)


def pinv[dtype: DType](A: NDArray[dtype]) raises -> NDArray[dtype]:
    """Compute the Moore-Penrose pseudo-inverse of a full-rank matrix.

    Computed as the least-squares solution to `A @ X = I`, matching
    `scipy.linalg.pinv` for full column- or row-rank input. Rank-deficient
    matrices are not currently supported (see `numojo.linalg.lstsq`).

    Parameters:
        dtype: Data type of the pseudo-inverse.

    Args:
        A: Matrix of shape `(m, n)`, of full column or row rank.

    Returns:
        The pseudo-inverse of `A`, of shape `(n, m)`.
    """
    var m = A.shape[0]
    return _lstsq(A, eye[dtype](m, m))


def lstsq[
    dtype: DType
](A: NDArray[dtype], b: NDArray[dtype]) raises -> NDArray[dtype]:
    """Compute the least-squares solution to `Ax = b`.

    Thin wrapper around `numojo.linalg.lstsq`. See that function for
    details and current limitations (no rank-deficient support).

    Parameters:
        dtype: Data type of the solution.

    Args:
        A: Coefficient matrix of shape `(m, n)`.
        b: Right-hand side, of shape `(m,)` or `(m, p)`.

    Returns:
        The least-squares solution `x`, of shape `(n,)` or `(n, p)`.
    """
    return _lstsq(A, b)


def solve_triangular[
    dtype: DType
](A: NDArray[dtype], b: NDArray[dtype], lower: Bool = True) raises -> NDArray[
    dtype
]:
    """Solve the triangular system `Ax = b`.

    Uses forward substitution when `lower=True` and back substitution
    when `lower=False`, matching `scipy.linalg.solve_triangular`. Unlike
    `solve`, this does not go through LU decomposition, so it is both
    faster and applicable when `A` is only known to be triangular.

    Parameters:
        dtype: Data type of the solution.

    Args:
        A: Triangular matrix of shape `(m, m)`. Only the triangle selected
            by `lower` is read; the other triangle is ignored.
        b: Right-hand side, of shape `(m,)` or `(m, n)`.
        lower: If `True` (default), `A` is treated as lower triangular.
            If `False`, `A` is treated as upper triangular.

    Returns:
        The solution `x`, matching the shape of `b`.

    Raises:
        NumojoError: If `A` is not 2-dimensional and square, or if `b`'s
            leading dimension does not match `A`.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise Error(
            NumojoError(
                category="shape",
                message="A must be a square 2-dimensional matrix!",
                location="solve_triangular",
            )
        )

    var m = A.shape[0]
    var b_was_1d = b.ndim == 1
    var B = b.reshape(Shape(m, 1)) if b_was_1d else b.copy()
    if B.shape[0] != m:
        raise Error(
            NumojoError(
                category="shape",
                message="A and b have incompatible shapes!",
                location="solve_triangular",
            )
        )
    var n = B.shape[1]

    var X = full[dtype](Shape(m, n), fill_value=SIMD[dtype, 1](0))

    for col in range(n):
        if lower:
            for i in range(m):
                var value = B.item(i, col)
                for j in range(i):
                    value = value - A.item(i, j) * X.item(j, col)
                X[Item(i, col)] = value / A.item(i, i)
        else:
            for i in range(m - 1, -1, -1):
                var value = B.item(i, col)
                for j in range(i + 1, m):
                    value = value - A.item(i, j) * X.item(j, col)
                X[Item(i, col)] = value / A.item(i, i)

    if b_was_1d:
        return X.reshape(Shape(m))
    return X^
