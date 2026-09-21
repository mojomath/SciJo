# ===----------------------------------------------------------------------=== #
# SciJo: Linalg module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Matrix Decompositions (`scijo.linalg.decompositions`)
========================================================
SciPy-style matrix decomposition wrappers built on top of NuMojo's linear
algebra primitives.

Examples
--------
    ```mojo
    from scijo.linalg import lu, qr, cholesky
    import numojo as nm

    var A = nm.fromstring[nm.f64]("[[4, 3], [6, 3]]")
    var P: nm.NDArray
    var L: nm.NDArray
    var U: nm.NDArray
    P, L, U = lu(A)  # A == P @ L @ U
    ```
"""

# ===----------------------------------------------------------------------=== #
# External
# ===----------------------------------------------------------------------=== #
from numojo.core.error import NumojoError
from numojo.core.ndarray import NDArray
from numojo.routines.linalg.decompositions import (
    cholesky as _cholesky,
    lu_decomposition,
    partial_pivoting,
    qr as _qr,
)
from numojo.routines.manipulation import transpose


def lu[
    dtype: DType
](A: NDArray[dtype]) raises -> Tuple[
    NDArray[dtype], NDArray[dtype], NDArray[dtype]
]:
    """Compute the pivoted LU decomposition of a square matrix.

    Matches `scipy.linalg.lu`: returns `(P, L, U)` such that
    `A == P @ L @ U`, where `P` is a permutation matrix, `L` is unit
    lower triangular, and `U` is upper triangular.

    Parameters:
        dtype: Data type of the input and output matrices.

    Args:
        A: Square input matrix of shape `(n, n)`.

    Returns:
        A tuple `(P, L, U)`.

    Raises:
        NumojoError: If the array is not 2-dimensional or not square.
    """

    if A.ndim != 2:
        raise Error(
            NumojoError(
                category="shape",
                message="The array is not 2-dimensional!",
                location="lu",
            )
        )
    if A.shape[0] != A.shape[1]:
        raise Error(
            NumojoError(
                category="shape",
                message="The matrix is not square!",
                location="lu",
            )
        )

    var pivoted = partial_pivoting(A.copy())
    var A_piv = pivoted[0].copy()
    var P = pivoted[1].copy()

    var L_U = lu_decomposition[dtype](A_piv)
    var L = L_U[0].copy()
    var U = L_U[1].copy()

    # `partial_pivoting` returns P such that `P @ A == A_piv`. Permutation
    # matrices are orthogonal, so `A == P.T @ A_piv == P.T @ L @ U`.
    return transpose(P), L^, U^


def qr[
    dtype: DType
](A: NDArray[dtype]) raises -> Tuple[NDArray[dtype], NDArray[dtype]]:
    """Compute the reduced QR decomposition of a matrix.

    Thin wrapper around `numojo.linalg.qr`. See that function for details.

    Parameters:
        dtype: Data type of the input and output matrices.

    Args:
        A: Input matrix of shape `(m, n)`.

    Returns:
        A tuple `(Q, R)` such that `Q @ R` reconstructs `A`.
    """
    return _qr[dtype](A)


def cholesky[
    dtype: DType
](A: NDArray[dtype], lower: Bool = True) raises -> NDArray[dtype]:
    """Compute the Cholesky decomposition of a symmetric positive-definite matrix.

    Parameters:
        dtype: Data type of the input and output matrices.

    Args:
        A: Symmetric positive-definite matrix of shape `(n, n)`.
        lower: If `True` (default), return the lower-triangular factor `L`
            such that `A == L @ L.T`. If `False`, return the
            upper-triangular factor `U == L.T` such that `A == U.T @ U`,
            matching `scipy.linalg.cholesky`'s default `lower=False`.

    Returns:
        The Cholesky factor, lower- or upper-triangular per `lower`.

    Raises:
        NumojoError: If the array is not 2-dimensional, not square, or not
            positive-definite.
    """
    var L = _cholesky[dtype](A)
    if lower:
        return L^
    return transpose(L)
