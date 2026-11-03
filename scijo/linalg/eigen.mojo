# ===----------------------------------------------------------------------=== #
# SciJo: Linalg module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Eigenvalue Problems (`scijo.linalg.eigen`)
=============================================
Eigenvalues and eigenvectors of real matrices.

`eigh`/`eigvalsh` cover the symmetric case, via the classic cyclic Jacobi
eigenvalue algorithm (Golub & Van Loan, *Matrix Computations*, ch. 8.4).
Converges quadratically once the off-diagonal entries are small, and -
unlike a fixed-sweep-count Jacobi - iterates until the off-diagonal norm
actually falls below `tol`, so it keeps working (up to `maxsweeps`) on
matrices that would need more than a handful of sweeps.

`eig`/`eigvals` cover the general (non-symmetric) case, where eigenvalues
may be complex: the shifted QR algorithm (Wilkinson shift from the
trailing 2x2 submatrix), carried out entirely in complex arithmetic so
every eigenvalue - real or a complex-conjugate pair - converges to its
own 1x1 diagonal entry, with no separate real/complex case split needed.
`eig`'s eigenvectors come from shifted inverse iteration against the
converged eigenvalues, on the original (real) matrix cast to complex.

Deliberately not implemented here: reduction to upper Hessenberg form
before the QR iterations (NuMojo's own real-only `eig` also skips it) -
it would cut the per-iteration cost from `O(n^3)` to `O(n^2)`, at the
price of a real Householder reduction plus a QR step restricted to the
shrinking deflated submatrix, neither of which is needed for correctness.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.math import sqrt

# ===----------------------------------------------------------------------=== #
# External
# ===----------------------------------------------------------------------=== #
from numojo.core import Shape
from numojo.core.complex import (
    ComplexNDArray,
    ComplexSIMD,
)
from numojo.core.dtype import ComplexDType
from numojo.core.error import NumojoError
from numojo.core.layout import NDArrayShape
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


# ===----------------------------------------------------------------------=== #
# General (non-symmetric) eigenvalue problem
# ===----------------------------------------------------------------------=== #


def _check_square[dtype: DType](A: NDArray[dtype], location: String) raises:
    """Raises unless `A` is a square 2-D matrix."""
    if A.ndim != 2:
        raise Error(
            NumojoError(
                category="shape",
                message="The array is not 2-dimensional!",
                location=location,
            )
        )
    if A.shape[0] != A.shape[1]:
        raise Error(
            NumojoError(
                category="shape",
                message="The matrix is not square!",
                location=location,
            )
        )


def _csqrt[cdtype: ComplexDType](z: ComplexSIMD[cdtype]) -> ComplexSIMD[cdtype]:
    """Principal complex square root, via the standard real-arithmetic
    formula (avoids trigonometric functions)."""
    var r = sqrt(z.re * z.re + z.im * z.im)
    var re = sqrt((r + z.re) * 0.5)
    var im_mag = sqrt((r - z.re) * 0.5)
    var im = im_mag if z.im >= 0 else -im_mag
    return ComplexSIMD[cdtype](re, im)


def _wilkinson_shift[
    cdtype: ComplexDType
](M: ComplexNDArray[cdtype], n: Int) raises -> ComplexSIMD[cdtype]:
    """The eigenvalue of the trailing 2x2 submatrix closer to `M[n-1, n-1]`.

    The standard choice of shift for the (complex) shifted QR algorithm:
    whichever root of the trailing 2x2 block's characteristic polynomial
    is closer to the entry about to be deflated converges fastest.
    """
    var a = M.item(n - 2, n - 2)
    var b = M.item(n - 2, n - 1)
    var c = M.item(n - 1, n - 2)
    var d = M.item(n - 1, n - 1)
    var trace = a + d
    var det = a * d - b * c
    var disc = _csqrt[cdtype](
        trace * trace - ComplexSIMD[cdtype](4.0, 0.0) * det
    )
    var mu1 = (trace + disc) * ComplexSIMD[cdtype](0.5, 0.0)
    var mu2 = (trace - disc) * ComplexSIMD[cdtype](0.5, 0.0)
    if (mu1 - d).__abs__() <= (mu2 - d).__abs__():
        return mu1
    return mu2


def _householder_vector[
    cdtype: ComplexDType
](x: List[ComplexSIMD[cdtype]]) -> List[ComplexSIMD[cdtype]]:
    """The unit Householder vector `v` reflecting `x` onto `alpha * e1`.

    Returns an empty list if `x` is already (numerically) a multiple of
    `e1`, signaling "no reflection needed" to the caller.
    """
    var length = len(x)
    var xnorm_sq: Scalar[cdtype.dtype] = 0.0
    for i in range(length):
        xnorm_sq += x[i].norm()
    var xnorm = sqrt(xnorm_sq)
    if xnorm <= Scalar[cdtype.dtype](1e-300):
        return List[ComplexSIMD[cdtype]]()

    var x0 = x[0]
    var x0_abs = x0.__abs__()
    var alpha: ComplexSIMD[cdtype]
    if x0_abs <= Scalar[cdtype.dtype](1e-300):
        alpha = ComplexSIMD[cdtype](xnorm, 0.0)
    else:
        var phase = x0 / ComplexSIMD[cdtype](x0_abs, 0.0)
        alpha = -phase * ComplexSIMD[cdtype](xnorm, 0.0)

    var v = List[ComplexSIMD[cdtype]](capacity=length)
    v.append(x0 - alpha)
    for i in range(1, length):
        v.append(x[i])

    var vnorm_sq: Scalar[cdtype.dtype] = 0.0
    for i in range(length):
        vnorm_sq += v[i].norm()
    var vnorm = sqrt(vnorm_sq)
    if vnorm <= Scalar[cdtype.dtype](1e-300):
        return List[ComplexSIMD[cdtype]]()

    for i in range(length):
        v[i] = v[i] / ComplexSIMD[cdtype](vnorm, 0.0)
    return v^


def _complex_qr_step[
    cdtype: ComplexDType
](mut M: ComplexNDArray[cdtype], n: Int) raises:
    """One full shifted-QR step on `M` (already shifted by the caller),
    in place: factors `M = QR` via complex Householder reflections, then
    recombines `M := R @ Q` by re-applying the same reflectors from the
    right, without ever materializing `Q` itself.
    """
    var vlist = List[List[ComplexSIMD[cdtype]]](capacity=n - 1)

    for k in range(n - 1):
        var length = n - k
        var x = List[ComplexSIMD[cdtype]](capacity=length)
        for i in range(length):
            x.append(M.item(k + i, k))
        var v = _householder_vector[cdtype](x)
        vlist.append(v.copy())
        if len(v) == 0:
            continue

        for c in range(k, n):
            var s = ComplexSIMD[cdtype](0.0, 0.0)
            for i in range(length):
                s += v[i].conj() * M.item(k + i, c)
            var two_s = ComplexSIMD[cdtype](2.0, 0.0) * s
            for i in range(length):
                M.itemset([k + i, c], M.item(k + i, c) - v[i] * two_s)

    for k in range(n - 1):
        var v = vlist[k].copy()
        if len(v) == 0:
            continue
        var length = n - k

        for r in range(n):
            var t = ComplexSIMD[cdtype](0.0, 0.0)
            for i in range(length):
                t += M.item(r, k + i) * v[i]
            var two_t = ComplexSIMD[cdtype](2.0, 0.0) * t
            for i in range(length):
                M.itemset(
                    [r, k + i],
                    M.item(r, k + i) - two_t * v[i].conj(),
                )


def _general_eig_values[
    dtype: DType, cdtype: ComplexDType
](A: NDArray[dtype], tol: Scalar[dtype], maxiter: Int) raises -> ComplexNDArray[
    cdtype
] where dtype.is_floating_point():
    """The shifted-QR core shared by `eig` and `eigvals`: returns the
    converged (complex) diagonal of the shifted-QR iteration, one entry
    per eigenvalue.
    """
    var n = A.shape[0]
    var A_re = rebind_var[NDArray[cdtype.dtype]](A.copy())
    var M = ComplexNDArray[cdtype](A_re^, zeros[cdtype.dtype](Shape(n, n)))

    var frob_sq: Scalar[dtype] = 0.0
    for i in range(n):
        for j in range(n):
            var v = A.item(i, j)
            frob_sq += v * v
    var frob = sqrt(frob_sq)

    comptime unshifted_iters = 30
    for iter in range(maxiter):
        var off: Scalar[cdtype.dtype] = 0.0
        for i in range(1, n):
            off += M.item(i, i - 1).__abs__()
        if off <= Scalar[cdtype.dtype](tol) * (
            Scalar[cdtype.dtype](frob) + 1.0
        ):
            break

        var shift: ComplexSIMD[cdtype]
        if n < 2 or iter < unshifted_iters:
            shift = ComplexSIMD[cdtype](0.0, 0.0)
        else:
            shift = _wilkinson_shift[cdtype](M, n)

        for i in range(n):
            M.itemset([i, i], M.item(i, i) - shift)
        _complex_qr_step[cdtype](M, n)
        for i in range(n):
            M.itemset([i, i], M.item(i, i) + shift)

    var w = ComplexNDArray[cdtype](NDArrayShape(n))
    for i in range(n):
        w.itemset([i], M.item(i, i))
    return w^


def eigvals[
    dtype: DType,
    *,
    cdtype: ComplexDType = ComplexDType(mlir_value=dtype._mlir_value),
](
    A: NDArray[dtype],
    tol: Scalar[dtype] = 1e-10,
    maxiter: Int = 500,
) raises -> ComplexNDArray[cdtype] where dtype.is_floating_point():
    """Computes the eigenvalues of a general real square matrix.

    Unlike `eigvalsh`, `A` need not be symmetric, and the returned
    eigenvalues may be complex (returned as a `ComplexNDArray`, always -
    even when every eigenvalue happens to be real). See the module
    docstring for the algorithm (complex-arithmetic shifted QR).

    Unlike `numpy.linalg.eigvals`, the eigenvalues are not returned in
    any particular order.

    Parameters:
        dtype: Data type of the input matrix.
        cdtype: The complex data type of the returned eigenvalues.
            Defaults to the complex type matching `dtype`.

    Args:
        A: Square matrix of shape `(n, n)`.
        tol: Convergence tolerance on the subdiagonal norm.
        maxiter: Maximum number of shifted-QR iterations.

    Returns:
        The `n` eigenvalues, as a `ComplexNDArray` of shape `(n,)`.

    Raises:
        NumojoError: If the array is not 2-dimensional or not square.

    Examples:
        ```mojo
        import numojo as nm
        from scijo.linalg import eigvals

        var A = nm.fromstring[nm.f64]("[[0, -1], [1, 0]]")
        var w = eigvals(A)  # +-1j
        ```
    """
    _check_square[dtype](A, "eigvals")
    return _general_eig_values[dtype, cdtype](A, tol, maxiter)


def _complex_solve_vec[
    cdtype: ComplexDType
](
    A: ComplexNDArray[cdtype], b: List[ComplexSIMD[cdtype]], n: Int
) raises -> List[ComplexSIMD[cdtype]]:
    """Solves the small dense complex linear system `A @ x = b` via
    Gaussian elimination with partial pivoting.
    """
    var M = ComplexNDArray[cdtype](A._re.copy(), A._im.copy())
    var rhs = List[ComplexSIMD[cdtype]](capacity=n)
    for i in range(n):
        rhs.append(b[i])

    for col in range(n):
        var piv = col
        var piv_val = M.item(col, col).__abs__()
        for row in range(col + 1, n):
            var val = M.item(row, col).__abs__()
            if val > piv_val:
                piv_val = val
                piv = row
        if piv != col:
            for c in range(n):
                var tmp = M.item(col, c)
                M.itemset([col, c], M.item(piv, c))
                M.itemset([piv, c], tmp)
            var tmp_b = rhs[col]
            rhs[col] = rhs[piv]
            rhs[piv] = tmp_b

        var diag = M.item(col, col)
        if diag.__abs__() <= Scalar[cdtype.dtype](1e-300):
            diag = ComplexSIMD[cdtype](1e-300, 0.0)
        for row in range(col + 1, n):
            var factor = M.item(row, col) / diag
            if factor.__abs__() <= Scalar[cdtype.dtype](1e-300):
                continue
            for c in range(col, n):
                M.itemset(
                    [row, c],
                    M.item(row, c) - factor * M.item(col, c),
                )
            rhs[row] = rhs[row] - factor * rhs[col]

    var x = List[ComplexSIMD[cdtype]](capacity=n)
    for _ in range(n):
        x.append(ComplexSIMD[cdtype](0.0, 0.0))
    for row in range(n - 1, -1, -1):
        var s = rhs[row]
        for c in range(row + 1, n):
            s = s - M.item(row, c) * x[c]
        var diag = M.item(row, row)
        if diag.__abs__() <= Scalar[cdtype.dtype](1e-300):
            diag = ComplexSIMD[cdtype](1e-300, 0.0)
        x[row] = s / diag
    return x^


def _inverse_iteration_eigenvector[
    cdtype: ComplexDType
](A_c: ComplexNDArray[cdtype], lam: ComplexSIMD[cdtype], n: Int) raises -> List[
    ComplexSIMD[cdtype]
]:
    """Recovers an eigenvector for `lam` via shifted inverse iteration.

    `lam` is nudged by a tiny amount before shifting so `A - lam*I` is
    never exactly singular to working precision (it is always nearly
    singular by construction, since `lam` is an already-converged
    eigenvalue).
    """
    var eps = Scalar[cdtype.dtype](1e-10) * (
        lam.__abs__() + Scalar[cdtype.dtype](1.0)
    )
    var shifted = ComplexNDArray[cdtype](A_c._re.copy(), A_c._im.copy())
    var lam_shifted = lam + ComplexSIMD[cdtype](eps, 0.0)
    for i in range(n):
        shifted.itemset([i, i], shifted.item(i, i) - lam_shifted)

    var v = List[ComplexSIMD[cdtype]](capacity=n)
    for _ in range(n):
        v.append(ComplexSIMD[cdtype](1.0, 0.0))

    for _ in range(3):
        v = _complex_solve_vec[cdtype](shifted, v, n)
        var norm_sq: Scalar[cdtype.dtype] = 0.0
        for i in range(n):
            norm_sq += v[i].norm()
        var norm = sqrt(norm_sq)
        if norm > 0:
            for i in range(n):
                v[i] = v[i] / ComplexSIMD[cdtype](norm, 0.0)
    return v^


def eig[
    dtype: DType,
    *,
    cdtype: ComplexDType = ComplexDType(mlir_value=dtype._mlir_value),
](
    A: NDArray[dtype],
    tol: Scalar[dtype] = 1e-10,
    maxiter: Int = 500,
) raises -> Tuple[
    ComplexNDArray[cdtype], ComplexNDArray[cdtype]
] where dtype.is_floating_point():
    """Computes the eigenvalues and eigenvectors of a general real square matrix.

    Unlike `eig` in `numojo.linalg` (which raises if it detects complex
    eigenvalues), both the eigenvalues and the eigenvectors here may be
    complex - the shifted QR algorithm runs entirely in complex
    arithmetic (see the module docstring), so a complex-conjugate pair
    converges just as readily as a real eigenvalue. Eigenvectors come
    from shifted inverse iteration on the original matrix, using each
    converged eigenvalue as the shift.

    Parameters:
        dtype: Data type of the input matrix.
        cdtype: The complex data type of the returned eigenvalues and
            eigenvectors. Defaults to the complex type matching `dtype`.

    Args:
        A: Square matrix of shape `(n, n)`.
        tol: Convergence tolerance on the subdiagonal norm (eigenvalues).
        maxiter: Maximum number of shifted-QR iterations.

    Returns:
        A tuple `(w, v)`: eigenvalues (shape `(n,)`), and eigenvectors as
        the columns of `v` (shape `(n, n)`), each normalized to unit
        length, such that `A @ v[:, i] == w[i] * v[:, i]`.

    Raises:
        NumojoError: If the array is not 2-dimensional or not square.

    Examples:
        ```mojo
        import numojo as nm
        from scijo.linalg import eig

        var A = nm.fromstring[nm.f64]("[[0, -1], [1, 0]]")
        var w: nm.ComplexNDArray[nm.cf64]
        var v: nm.ComplexNDArray[nm.cf64]
        w, v = eig(A)
        ```
    """
    _check_square[dtype](A, "eig")
    var n = A.shape[0]
    var w = _general_eig_values[dtype, cdtype](A, tol, maxiter)
    var A_re2 = rebind_var[NDArray[cdtype.dtype]](A.copy())
    var A_c = ComplexNDArray[cdtype](A_re2^, zeros[cdtype.dtype](Shape(n, n)))

    var V = ComplexNDArray[cdtype](NDArrayShape(n, n))
    for col in range(n):
        var lam = w.item(col)
        var v = _inverse_iteration_eigenvector[cdtype](A_c, lam, n)
        for row in range(n):
            V.itemset([row, col], v[row])

    return w^, V^
