# ===----------------------------------------------------------------------=== #
# SciJo: Linalg module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Matrix Functions and Structured Matrices (`scijo.linalg.matrix_fns`)
=======================================================================
`expm` (the matrix exponential), `block_diag` and `toeplitz` (structured
matrix construction), and `null_space`/`orth` (orthonormal bases for a
matrix's null space and column space).

Scoped simplifications, noted explicitly rather than left as oversights:
- `expm` uses scaling-and-squaring with a diagonal Padé[6/6] approximant
  (Moler & Van Loan), not SciPy's higher-order Padé[13/13] - accurate to
  roughly 1e-9 or better in practice, but with a smaller stable norm range
  per squaring step than the higher-order approximant.
- `null_space`/`orth` are computed from the eigendecomposition of the
  Gram matrix (`A^T @ A` or `A @ A^T`, via the existing symmetric
  eigensolver `eigh`) rather than from the SVD. This sidesteps needing a
  *full* (not just thin) SVD to recover every null-space direction of a
  wide matrix, at the cost of squaring the condition number relative to
  an SVD-based implementation - adequate for well-conditioned inputs, less
  accurate than SciPy's SVD-based `null_space`/`orth` on ill-conditioned
  ones.
- `block_diag` takes a `List[NDArray[dtype]]` rather than SciPy's variadic
  `*arrs`.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.math import ceil, log2, sqrt

# ===----------------------------------------------------------------------=== #
# External
# ===----------------------------------------------------------------------=== #
from numojo import identity, zeros
from numojo.core import Shape
from numojo.core.ndarray import NDArray
from numojo.routines.linalg.products import matmul
from numojo.routines.manipulation import transpose

# ===----------------------------------------------------------------------=== #
# SciJo
# ===----------------------------------------------------------------------=== #
from scijo.linalg.eigen import eigh
from scijo.linalg.norms import norm
from scijo.linalg.solving import solve

# ===----------------------------------------------------------------------=== #
# expm
# ===----------------------------------------------------------------------=== #


def expm[
    dtype: DType
](A: NDArray[dtype]) raises -> NDArray[dtype] where dtype.is_floating_point():
    """Computes the matrix exponential `exp(A)`.

    Example: eA = expm(A)

    Scaling-and-squaring with a diagonal Padé[6/6] approximant (Moler &
    Van Loan, 2003): `A` is scaled down by a power of 2 until its 1-norm
    is small, the Padé approximant of the scaled matrix is evaluated by
    solving one linear system, and the result is squared back up.

    Parameters:
        dtype: The floating-point data type.

    Args:
        A: Square matrix of shape `(n, n)`.

    Returns:
        The matrix exponential of `A`, of shape `(n, n)`.

    Raises:
        Error: If `A` is not a square 2-D matrix.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise Error("expm requires a square 2-D matrix")

    var n = A.shape[0]
    var normA = Float64(norm[dtype](A, ord="1"))

    var s = 0
    if normA > 1.0:
        s = Int(ceil(log2(normA)))
        if s < 0:
            s = 0

    var scale = Scalar[dtype](2.0) ** Scalar[dtype](s)
    var B = A / scale

    comptime c: List[Float64] = [
        1.0,
        0.5,
        5.0 / 44.0,
        1.0 / 66.0,
        1.0 / 792.0,
        1.0 / 15840.0,
        1.0 / 665280.0,
    ]
    ref pade_c = materialize[c]()

    var powers = List[NDArray[dtype]]()
    powers.append(identity[dtype](n))
    for i in range(1, 7):
        powers.append(matmul(powers[i - 1], B))

    var N = zeros[dtype](Shape(n, n))
    var D = zeros[dtype](Shape(n, n))
    for j in range(7):
        var cj = Scalar[dtype](pade_c[j])
        N = N + cj * powers[j]
        var sign: Scalar[dtype] = 1.0 if j % 2 == 0 else -1.0
        D = D + (sign * cj) * powers[j]

    var X = solve(D, N)
    for _ in range(s):
        X = matmul(X, X)

    return X^


# ===----------------------------------------------------------------------=== #
# block_diag
# ===----------------------------------------------------------------------=== #


def block_diag[
    dtype: DType
](arrays: List[NDArray[dtype]]) raises -> NDArray[dtype]:
    """Constructs a block-diagonal matrix from a list of 2-D matrices.

    Example: D = block_diag(arrays)

    Parameters:
        dtype: The data type.

    Args:
        arrays: A list of 2-D matrices, each of arbitrary shape
            `(m_i, n_i)`.

    Returns:
        A block-diagonal matrix of shape `(sum(m_i), sum(n_i))`, with
        every array placed on the diagonal and zeros elsewhere.

    Raises:
        Error: If arrays is empty, or any array is not 2-D.
    """
    var k = len(arrays)
    if k == 0:
        raise Error("block_diag requires at least one array")

    var total_rows = 0
    var total_cols = 0
    for i in range(k):
        if arrays[i].ndim != 2:
            raise Error("block_diag requires every array to be 2-D")
        total_rows += arrays[i].shape[0]
        total_cols += arrays[i].shape[1]

    var result = zeros[dtype](Shape(total_rows, total_cols))

    var row_off = 0
    var col_off = 0
    for i in range(k):
        var m = arrays[i].shape[0]
        var n = arrays[i].shape[1]
        for r in range(m):
            for c in range(n):
                result.store(row_off + r, col_off + c, val=arrays[i].item(r, c))
        row_off += m
        col_off += n

    return result^


# ===----------------------------------------------------------------------=== #
# toeplitz
# ===----------------------------------------------------------------------=== #


def toeplitz[
    dtype: DType
](c: NDArray[dtype], r: Optional[NDArray[dtype]] = None) raises -> NDArray[
    dtype
]:
    """Constructs a Toeplitz matrix from its first column and first row.

    Example: T = toeplitz(c, r)

    `T[i, j] = c[i - j]` for `i >= j`, and `T[i, j] = r[j - i]` for `i < j`.
    `r[0]` is ignored; `c[0]` is used for the main diagonal instead
    (matching SciPy's convention).

    Parameters:
        dtype: The data type.

    Args:
        c: First column, of shape `(m,)`.
        r: First row, of shape `(n,)`. If `None`, `r = c` (a symmetric
            Toeplitz matrix).

    Returns:
        A Toeplitz matrix of shape `(m, n)` where `n = r.size` if given,
        else `n = m`.

    Raises:
        Error: If c or r is empty.
    """
    if c.size == 0:
        raise Error("toeplitz requires a non-empty first column")

    var m = c.size
    var n: Int
    var row: NDArray[dtype]
    if r:
        if r.value().size == 0:
            raise Error("toeplitz requires a non-empty first row")
        n = r.value().size
        row = r.value().copy()
    else:
        n = m
        row = c.copy()

    var result = zeros[dtype](Shape(m, n))
    for i in range(m):
        for j in range(n):
            if i >= j:
                result.store(i, j, val=c.item(i - j))
            else:
                result.store(i, j, val=row.item(j - i))

    return result^


# ===----------------------------------------------------------------------=== #
# null_space / orth
# ===----------------------------------------------------------------------=== #


def _default_rcond[dtype: DType](m: Int, n: Int) -> Scalar[dtype]:
    """SciPy's default `rcond`: `max(m, n) * eps`, applied to the
    singular values (not the Gram-matrix eigenvalues).
    """
    var eps: Scalar[dtype] = 2.220446049250313e-16
    var max_mn = m if m > n else n
    return Scalar[dtype](max_mn) * eps


def null_space[
    dtype: DType
](A: NDArray[dtype], rcond: Optional[Scalar[dtype]] = None) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """Computes an orthonormal basis for the null space of a matrix.

    Example: ns = null_space(A)

    Via the eigendecomposition of the Gram matrix `A^T @ A` (see the
    module docstring for the SVD-vs-Gram-matrix tradeoff).

    Parameters:
        dtype: The floating-point data type.

    Args:
        A: Matrix of shape `(m, n)`.
        rcond: Relative condition number: singular values `<= rcond *
            largest singular value` are treated as zero. Defaults to
            `max(m, n) * eps`.

    Returns:
        An orthonormal basis for the null space, of shape `(n, k)` where
        `k` is the null space dimension.

    Raises:
        Error: If A is not 2-D.
    """
    if A.ndim != 2:
        raise Error("null_space requires a 2-D matrix")

    var m = A.shape[0]
    var n = A.shape[1]
    var gram = matmul(transpose(A), A)
    var eig = eigh(gram)
    ref w = eig[0]
    ref v = eig[1]

    var max_eig = w.item(n - 1)
    var largest_sv = sqrt(max_eig) if max_eig > 0 else Scalar[dtype](0.0)
    var rc = rcond.value() if rcond else _default_rcond[dtype](m, n)
    var threshold = (rc * largest_sv) * (rc * largest_sv)

    var count = 0
    for i in range(n):
        if w.item(i) <= threshold:
            count += 1

    var result = zeros[dtype](Shape(n, count))
    var col = 0
    for i in range(n):
        if w.item(i) <= threshold:
            for r in range(n):
                result.store(r, col, val=v.item(r, i))
            col += 1

    return result^


def orth[
    dtype: DType
](A: NDArray[dtype], rcond: Optional[Scalar[dtype]] = None) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """Computes an orthonormal basis for the range (column space) of a matrix.

    Example: ob = orth(A)

    Via the eigendecomposition of the Gram matrix `A @ A^T` (see the
    module docstring for the SVD-vs-Gram-matrix tradeoff).

    Parameters:
        dtype: The floating-point data type.

    Args:
        A: Matrix of shape `(m, n)`.
        rcond: Relative condition number: singular values `<= rcond *
            largest singular value` are treated as zero. Defaults to
            `max(m, n) * eps`.

    Returns:
        An orthonormal basis for the range, of shape `(m, k)` where `k`
        is the rank of `A`, ordered by decreasing singular value.

    Raises:
        Error: If A is not 2-D.
    """
    if A.ndim != 2:
        raise Error("orth requires a 2-D matrix")

    var m = A.shape[0]
    var n = A.shape[1]
    var gram = matmul(A, transpose(A))
    var eig = eigh(gram)
    ref w = eig[0]
    ref v = eig[1]

    var max_eig = w.item(m - 1)
    var largest_sv = sqrt(max_eig) if max_eig > 0 else Scalar[dtype](0.0)
    var rc = rcond.value() if rcond else _default_rcond[dtype](m, n)
    var threshold = (rc * largest_sv) * (rc * largest_sv)

    var count = 0
    for i in range(m):
        if w.item(i) > threshold:
            count += 1

    var result = zeros[dtype](Shape(m, count))
    var col = 0
    for i in range(m - 1, -1, -1):
        if w.item(i) > threshold:
            for r in range(m):
                result.store(r, col, val=v.item(r, i))
            col += 1

    return result^
