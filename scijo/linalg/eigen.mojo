# ===----------------------------------------------------------------------=== #
# SciJo: Linalg module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Symmetric Eigenvalue Problems (`scijo.linalg.eigen`)
=======================================================
Eigenvalues and eigenvectors of real symmetric matrices, via the classic
cyclic Jacobi eigenvalue algorithm (Golub & Van Loan, *Matrix
Computations*, ch. 8.4). Converges quadratically once the off-diagonal
entries are small, and - unlike a fixed-sweep-count Jacobi - iterates
until the off-diagonal norm actually falls below `tol`, so it keeps
working (up to `maxsweeps`) on matrices that would need more than a
handful of sweeps.

Only the symmetric (Hermitian, for real input) eigenvalue problem is
covered here. The general (non-symmetric) eigenvalue problem, where
eigenvalues may be complex, is tracked as future work.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.math import sqrt

# ===----------------------------------------------------------------------=== #
# External
# ===----------------------------------------------------------------------=== #
from numojo.core import Shape
from numojo.core.error import NumojoError
from numojo.core.ndarray import NDArray
from numojo.routines.creation import zeros


def eigh[
    dtype: DType
](
    A: NDArray[dtype],
    lower: Bool = True,
    tol: Scalar[dtype] = 1e-12,
    maxsweeps: Int = 100,
) raises -> Tuple[NDArray[dtype], NDArray[dtype]]:
    """Compute the eigenvalues and eigenvectors of a real symmetric matrix.

    Returns `(w, v)` with eigenvalues `w` ascending and the eigenvectors
    as the columns of `v`, such that `A @ v[:, i] == w[i] * v[:, i]` and
    `A == v @ diag(w) @ v.T`.

    Only one triangle of `A` is read; the matrix is treated as symmetric
    regardless of what the other triangle contains.

    Parameters:
        dtype: Data type of the input and output matrices.

    Args:
        A: Symmetric matrix of shape `(n, n)`.
        lower: If `True` (default), read `A`'s lower triangle. If `False`,
            read its upper triangle.
        tol: Convergence tolerance on the off-diagonal norm.
        maxsweeps: Maximum number of full Jacobi sweeps.

    Returns:
        A tuple `(w, v)`: eigenvalues ascending, eigenvectors as columns.

    Raises:
        NumojoError: If the array is not 2-dimensional or not square.
    """
    if A.ndim != 2:
        raise Error(
            NumojoError(
                category="shape",
                message="The array is not 2-dimensional!",
                location="eigh",
            )
        )
    if A.shape[0] != A.shape[1]:
        raise Error(
            NumojoError(
                category="shape",
                message="The matrix is not square!",
                location="eigh",
            )
        )

    var n = A.shape[0]
    var work = zeros[dtype](Shape(n, n))
    if lower:
        for i in range(n):
            for j in range(i + 1):
                var value = A.item(i, j)
                work.store(i * n + j, val=value)
                work.store(j * n + i, val=value)
    else:
        for i in range(n):
            for j in range(i, n):
                var value = A.item(i, j)
                work.store(i * n + j, val=value)
                work.store(j * n + i, val=value)

    var vectors = zeros[dtype](Shape(n, n))
    for i in range(n):
        vectors.store(i * n + i, val=Scalar[dtype](1.0))

    for _ in range(maxsweeps):
        var off_norm_sq: Scalar[dtype] = 0.0
        for p in range(n - 1):
            for q in range(p + 1, n):
                var apq = work.load(p * n + q)
                off_norm_sq += apq * apq
        if off_norm_sq <= tol * tol:
            break

        for p in range(n - 1):
            for q in range(p + 1, n):
                var apq = work.load(p * n + q)
                if abs(apq) <= Scalar[dtype](1e-300):
                    continue
                var app = work.load(p * n + p)
                var aqq = work.load(q * n + q)
                var theta = (aqq - app) / (Scalar[dtype](2.0) * apq)
                var t: Scalar[dtype]
                if theta >= 0:
                    t = Scalar[dtype](1.0) / (
                        theta + sqrt(Scalar[dtype](1.0) + theta * theta)
                    )
                else:
                    t = Scalar[dtype](-1.0) / (
                        -theta + sqrt(Scalar[dtype](1.0) + theta * theta)
                    )
                var c = Scalar[dtype](1.0) / sqrt(Scalar[dtype](1.0) + t * t)
                var s = t * c

                for k in range(n):
                    var akp = work.load(k * n + p)
                    var akq = work.load(k * n + q)
                    work.store(k * n + p, val=c * akp - s * akq)
                    work.store(k * n + q, val=s * akp + c * akq)
                for k in range(n):
                    var apk = work.load(p * n + k)
                    var aqk = work.load(q * n + k)
                    work.store(p * n + k, val=c * apk - s * aqk)
                    work.store(q * n + k, val=s * apk + c * aqk)
                for k in range(n):
                    var vkp = vectors.load(k * n + p)
                    var vkq = vectors.load(k * n + q)
                    vectors.store(k * n + p, val=c * vkp - s * vkq)
                    vectors.store(k * n + q, val=s * vkp + c * vkq)

    # Ascending sort, carrying the matching
    # eigenvector column along with each eigenvalue.
    return _eigh_sort[dtype](work, vectors, n)


def _eigh_sort[
    dtype: DType
](
    work: NDArray[dtype], vectors_unsorted: NDArray[dtype], n: Int
) raises -> Tuple[NDArray[dtype], NDArray[dtype]]:
    """Ascending-sort the eigenvalues on `work`'s diagonal, permuting the
    matching columns of `vectors_unsorted` alongside them.
    """
    var values = zeros[dtype](Shape(n))
    for i in range(n):
        values.store(i, val=work.load(i * n + i))

    var order = List[Int]()
    for i in range(n):
        order.append(i)
    for i in range(1, n):
        var key = order[i]
        var key_val = values.load(key)
        var j = i - 1
        while j >= 0 and values.load(order[j]) > key_val:
            order[j + 1] = order[j]
            j -= 1
        order[j + 1] = key

    var sorted_values = zeros[dtype](Shape(n))
    var sorted_vectors = zeros[dtype](Shape(n, n))
    for i in range(n):
        var src = order[i]
        sorted_values.store(i, val=values.load(src))
        for k in range(n):
            sorted_vectors.store(
                k * n + i, val=vectors_unsorted.load(k * n + src)
            )
    return sorted_values^, sorted_vectors^


def eigvalsh[
    dtype: DType
](
    A: NDArray[dtype],
    lower: Bool = True,
    tol: Scalar[dtype] = 1e-12,
    maxsweeps: Int = 100,
) raises -> NDArray[dtype]:
    """Compute the eigenvalues of a real symmetric matrix.

    `eigh(A)[0]`: the same Jacobi rotations are the work regardless of
    whether the accumulated eigenvector matrix is kept, so this exists for
    the caller who only wants the eigenvalues, not because it is cheaper.

    Parameters:
        dtype: Data type of the input and output.

    Args:
        A: Symmetric matrix of shape `(n, n)`.
        lower: If `True` (default), read `A`'s lower triangle.
        tol: Convergence tolerance on the off-diagonal norm.
        maxsweeps: Maximum number of full Jacobi sweeps.

    Returns:
        Eigenvalues, ascending.
    """
    var result = eigh[dtype](A, lower, tol, maxsweeps)
    return result[0].copy()
