# ===----------------------------------------------------------------------=== #
# SciJo: Linalg module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Singular Value Decomposition (`scijo.linalg.svd`)
====================================================
Singular value decomposition via one-sided Jacobi orthogonalization
(Hestenes, 1958): rotates pairs of columns of `A` until they are
mutually orthogonal, accumulating the rotations into `V`. The column
norms are then the singular values and the normalized columns are `U`.

Chosen over forming `A.T @ A` and calling an eigensolver on it, which
squares the condition number and loses up to half the significant digits
of the small singular values.
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


def _svd_tall[
    dtype: DType
](X: NDArray[dtype], maxsweeps: Int, tol: Scalar[dtype]) raises -> Tuple[
    NDArray[dtype], NDArray[dtype], NDArray[dtype]
]:
    """One-sided Jacobi SVD of a matrix with at least as many rows as
    columns. Returns `(U, s, V)`, descending by singular value, such that
    `X == U @ diag(s) @ V.T`.
    """
    var p = X.shape[0]
    var q = X.shape[1]
    var work = X.copy()
    var V = zeros[dtype](Shape(q, q))
    for i in range(q):
        V.store(i * q + i, val=Scalar[dtype](1.0))

    for _ in range(maxsweeps):
        var off_norm_sq: Scalar[dtype] = 0.0
        for a in range(q - 1):
            for b in range(a + 1, q):
                var alpha: Scalar[dtype] = 0.0
                var beta: Scalar[dtype] = 0.0
                var gamma: Scalar[dtype] = 0.0
                for i in range(p):
                    var ia = work.load(i * q + a)
                    var ib = work.load(i * q + b)
                    alpha += ia * ia
                    beta += ib * ib
                    gamma += ia * ib
                off_norm_sq += gamma * gamma

                if abs(gamma) <= tol * sqrt(
                    alpha * beta + Scalar[dtype](1e-300)
                ):
                    continue

                var theta = (beta - alpha) / (Scalar[dtype](2.0) * gamma)
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

                for i in range(p):
                    var ia = work.load(i * q + a)
                    var ib = work.load(i * q + b)
                    work.store(i * q + a, val=c * ia - s * ib)
                    work.store(i * q + b, val=s * ia + c * ib)
                for i in range(q):
                    var va = V.load(i * q + a)
                    var vb = V.load(i * q + b)
                    V.store(i * q + a, val=c * va - s * vb)
                    V.store(i * q + b, val=s * va + c * vb)

        if off_norm_sq <= tol * tol:
            break

    var values = zeros[dtype](Shape(q))
    var U = zeros[dtype](Shape(p, q))
    for j in range(q):
        var norm_sq: Scalar[dtype] = 0.0
        for i in range(p):
            var v = work.load(i * q + j)
            norm_sq += v * v
        var sigma = sqrt(norm_sq)
        values.store(j, val=sigma)
        var safe = sigma if sigma > Scalar[dtype](1e-300) else Scalar[dtype](
            1e-300
        )
        for i in range(p):
            U.store(i * q + j, val=work.load(i * q + j) / safe)

    # Descending sort, permuting the matching
    # columns of `U` and `V` alongside each singular value.
    var order = List[Int]()
    for i in range(q):
        order.append(i)
    for i in range(1, q):
        var key = order[i]
        var key_val = values.load(key)
        var j = i - 1
        while j >= 0 and values.load(order[j]) < key_val:
            order[j + 1] = order[j]
            j -= 1
        order[j + 1] = key

    var sorted_values = zeros[dtype](Shape(q))
    var sorted_U = zeros[dtype](Shape(p, q))
    var sorted_V = zeros[dtype](Shape(q, q))
    for i in range(q):
        var src = order[i]
        sorted_values.store(i, val=values.load(src))
        for k in range(p):
            sorted_U.store(k * q + i, val=U.load(k * q + src))
        for k in range(q):
            sorted_V.store(k * q + i, val=V.load(k * q + src))

    return sorted_U^, sorted_values^, sorted_V^


def svd[
    dtype: DType
](
    A: NDArray[dtype], maxsweeps: Int = 60, tol: Scalar[dtype] = 1e-12
) raises -> Tuple[NDArray[dtype], NDArray[dtype], NDArray[dtype]]:
    """Compute the (thin) singular value decomposition of a matrix.

    Returns the economy-sized (thin) factors `(U, s, Vt)` with
    `k = min(m, n)`, such that `A == U @ diag(s) @ Vt`. Singular values
    are descending.

    Parameters:
        dtype: Data type of the input and output matrices.

    Args:
        A: Matrix of shape `(m, n)`.
        maxsweeps: Maximum number of full Jacobi sweeps.
        tol: Convergence tolerance on the off-diagonal (cross-column)
            norm.

    Returns:
        A tuple `(U, s, Vt)` of shapes `(m, k)`, `(k,)`, `(k, n)`.

    Raises:
        NumojoError: If the array is not 2-dimensional.
    """
    if A.ndim != 2:
        raise Error(
            NumojoError(
                category="shape",
                message="The array is not 2-dimensional!",
                location="svd",
            )
        )

    var m = A.shape[0]
    var n = A.shape[1]

    if m >= n:
        var result = _svd_tall[dtype](A, maxsweeps, tol)
        var U = result[0].copy()
        var s = result[1].copy()
        var V = result[2].copy()
        var Vt = zeros[dtype](Shape(n, n))
        for i in range(n):
            for j in range(n):
                Vt.store(i * n + j, val=V.load(j * n + i))
        return U^, s^, Vt^
    else:
        var At = zeros[dtype](Shape(n, m))
        for i in range(m):
            for j in range(n):
                At.store(j * m + i, val=A.item(i, j))
        var result = _svd_tall[dtype](At, maxsweeps, tol)
        var Ut = result[0].copy()
        var s = result[1].copy()
        var V = result[2].copy()
        # A == At.T == (U' diag(s) V'.T).T == V' diag(s) U'.T
        var Vt = zeros[dtype](Shape(m, n))
        for i in range(m):
            for j in range(n):
                Vt.store(i * n + j, val=Ut.load(j * m + i))
        return V^, s^, Vt^


def svdvals[
    dtype: DType
](
    A: NDArray[dtype], maxsweeps: Int = 60, tol: Scalar[dtype] = 1e-12
) raises -> NDArray[dtype]:
    """Compute the singular values of a matrix, without either set of
    vectors.

    `svd(A)[1]`: the Jacobi sweeps are the work, and dropping `U` and
    `Vt` does not shorten them. Exists for the caller who only wants the
    singular values.

    Parameters:
        dtype: Data type of the input and output.

    Args:
        A: Matrix of shape `(m, n)`.
        maxsweeps: Maximum number of full Jacobi sweeps.
        tol: Convergence tolerance on the off-diagonal (cross-column)
            norm.

    Returns:
        Singular values, descending, of shape `(min(m, n),)`.
    """
    var result = svd[dtype](A, maxsweeps, tol)
    return result[1].copy()


def matrix_rank[
    dtype: DType
](A: NDArray[dtype], rcond: Scalar[dtype] = 1e-12) raises -> Int:
    """Estimate the rank of a matrix.

    Counts the singular values that exceed `rcond` times the largest
    singular value, matching `numpy.linalg.matrix_rank`'s default
    threshold convention.

    Parameters:
        dtype: Data type of the input.

    Args:
        A: Matrix of shape `(m, n)`.
        rcond: Relative threshold, as a fraction of the largest singular
            value.

    Returns:
        The estimated rank.
    """
    var s = svdvals[dtype](A)
    var k = len(s)
    var largest: Scalar[dtype] = 0.0
    for i in range(k):
        var v = s.load(i)
        if v > largest:
            largest = v
    var threshold = largest * rcond
    var rank = 0
    for i in range(k):
        if s.load(i) > threshold:
            rank += 1
    return rank


def cond[dtype: DType](A: NDArray[dtype]) raises -> Scalar[dtype]:
    """Compute the 2-norm condition number of a matrix.

    The ratio of the largest to the smallest singular value (the 2-norm
    condition number).

    Parameters:
        dtype: Data type of the input and output.

    Args:
        A: Matrix of shape `(m, n)`.

    Returns:
        The condition number `smax / smin`.
    """
    var s = svdvals[dtype](A)
    var k = len(s)
    var largest: Scalar[dtype] = s.load(0)
    var smallest: Scalar[dtype] = s.load(0)
    for i in range(1, k):
        var v = s.load(i)
        if v > largest:
            largest = v
        if v < smallest:
            smallest = v
    if smallest <= Scalar[dtype](1e-300):
        return Scalar[dtype](1e300)
    return largest / smallest
