"""
Tests for the linalg module (decompositions, solvers, norms).

To run: `mojo test tests/test_linalg.mojo -I .` from the project root directory.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.math import sqrt
from std.python import Python, PythonObject
from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_true,
    TestSuite,
)

# ===----------------------------------------------------------------------=== #
# External
# ===----------------------------------------------------------------------=== #
import numojo as nm

# ===----------------------------------------------------------------------=== #
# SciJo
# ===----------------------------------------------------------------------=== #
import scijo as sj
from scijo.linalg import (
    cholesky,
    cond,
    det,
    eigh,
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


def _assert_matrix_almost_equal[
    dtype: DType
](
    A: nm.NDArray[dtype], B: nm.NDArray[dtype], atol: Scalar[dtype] = 1e-8
) raises:
    assert_equal(A.shape[0], B.shape[0])
    assert_equal(A.shape[1], B.shape[1])
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            assert_almost_equal(A.item(i, j), B.item(i, j), atol=Float64(atol))


def test_solve_2x2() raises:
    var A = nm.fromstring[sj.f64]("[[3, 1], [1, 2]]")
    var b = nm.fromstring[sj.f64]("[9, 8]")
    var x = solve(A, b)
    assert_almost_equal(x.item(0), 2.0, atol=1e-8)
    assert_almost_equal(x.item(1), 3.0, atol=1e-8)


def test_solve_matrix_rhs() raises:
    var A = nm.fromstring[sj.f64]("[[3, 1], [1, 2]]")
    var I = nm.fromstring[sj.f64]("[[1, 0], [0, 1]]")
    var X = solve(A, I)
    var A_inv = inv(A)
    _assert_matrix_almost_equal(X, A_inv)


def test_inv_roundtrip() raises:
    var A = nm.fromstring[sj.f64]("[[4, 7], [2, 6]]")
    var A_inv = inv(A)
    var product = matmul(A, A_inv)
    var I = nm.fromstring[sj.f64]("[[1, 0], [0, 1]]")
    _assert_matrix_almost_equal(product, I)


def test_det() raises:
    var A = nm.fromstring[sj.f64]("[[1, 2], [3, 4]]")
    assert_almost_equal(det(A), -2.0, atol=1e-8)


def test_trace() raises:
    var A = nm.fromstring[sj.f64]("[[1, 2], [3, 4]]")
    var t = trace(A)
    assert_almost_equal(t.item(), 5.0, atol=1e-8)


def test_lu_reconstructs_a() raises:
    var A = nm.fromstring[sj.f64]("[[4, 3], [6, 3]]")
    var PLU = lu(A)
    var P = PLU[0].copy()
    var L = PLU[1].copy()
    var U = PLU[2].copy()
    var reconstructed = matmul(P, matmul(L, U))
    _assert_matrix_almost_equal(A, reconstructed)


def test_qr_reconstructs_a() raises:
    var A = nm.fromstring[sj.f64]("[[1, 2], [3, 4], [5, 6]]")
    var QR = qr(A)
    var Q = QR[0].copy()
    var R = QR[1].copy()
    var reconstructed = matmul(Q, R)
    _assert_matrix_almost_equal(A, reconstructed)


def test_cholesky_lower_and_upper() raises:
    var A = nm.fromstring[sj.f64]("[[4, 2], [2, 3]]")
    var L = cholesky(A)
    var reconstructed_lower = matmul(L, nm.transpose(L))
    _assert_matrix_almost_equal(A, reconstructed_lower)

    var U = cholesky(A, lower=False)
    var reconstructed_upper = matmul(nm.transpose(U), U)
    _assert_matrix_almost_equal(A, reconstructed_upper)


def test_lstsq_overdetermined() raises:
    var A = nm.fromstring[sj.f64]("[[1, 1], [1, 2], [1, 3]]")
    var y = nm.fromstring[sj.f64]("[6, 0, 0]")
    var x = lstsq(A, y)
    # Least-squares fit of y = a + b*t at t=1,2,3 for y=6,0,0.
    assert_almost_equal(x.item(0), 8.0, atol=1e-6)
    assert_almost_equal(x.item(1), -3.0, atol=1e-6)


def test_pinv_full_rank() raises:
    var A = nm.fromstring[sj.f64]("[[1, 0], [0, 1], [1, 1]]")
    var A_pinv = pinv(A)
    var should_be_identity = matmul(A_pinv, A)
    var I = nm.fromstring[sj.f64]("[[1, 0], [0, 1]]")
    _assert_matrix_almost_equal(should_be_identity, I, atol=1e-6)


def test_solve_triangular_lower() raises:
    var L = nm.fromstring[sj.f64]("[[2, 0], [1, 3]]")
    var b = nm.fromstring[sj.f64]("[4, 7]")
    var x = solve_triangular(L, b, lower=True)
    assert_almost_equal(x.item(0), 2.0, atol=1e-8)
    assert_almost_equal(x.item(1), 5.0 / 3.0, atol=1e-8)


def test_solve_triangular_upper() raises:
    var U = nm.fromstring[sj.f64]("[[2, 1], [0, 3]]")
    var b = nm.fromstring[sj.f64]("[4, 9]")
    var x = solve_triangular(U, b, lower=False)
    assert_almost_equal(x.item(1), 3.0, atol=1e-8)
    assert_almost_equal(x.item(0), 0.5, atol=1e-8)


def test_norm_vector() raises:
    var x = nm.fromstring[sj.f64]("[3, 4]")
    assert_almost_equal(norm(x), 5.0, atol=1e-8)
    assert_almost_equal(norm(x, ord="1"), 7.0, atol=1e-8)
    assert_almost_equal(norm(x, ord="inf"), 4.0, atol=1e-8)


def test_norm_matrix() raises:
    var A = nm.fromstring[sj.f64]("[[1, -2], [-3, 4]]")
    assert_almost_equal(norm(A, ord="fro"), sqrt(30.0), atol=1e-8)
    assert_almost_equal(norm(A, ord="1"), 6.0, atol=1e-8)
    assert_almost_equal(norm(A, ord="inf"), 7.0, atol=1e-8)


def test_eigh_symmetric_2x2() raises:
    """Eigenvalues/vectors of [[2, 1], [1, 2]] are 1 -> [1,-1]/sqrt2 and
    3 -> [1,1]/sqrt2, by hand.
    """
    var A = nm.fromstring[sj.f64]("[[2, 1], [1, 2]]")
    var result = eigh(A)
    var w = result[0].copy()
    var v = result[1].copy()
    assert_almost_equal(w.item(0), 1.0, atol=1e-8)
    assert_almost_equal(w.item(1), 3.0, atol=1e-8)
    assert_almost_equal(abs(v.item(0, 0)), 1.0 / sqrt(2.0), atol=1e-8)
    assert_almost_equal(abs(v.item(1, 0)), 1.0 / sqrt(2.0), atol=1e-8)
    assert_almost_equal(abs(v.item(0, 1)), 1.0 / sqrt(2.0), atol=1e-8)
    assert_almost_equal(abs(v.item(1, 1)), 1.0 / sqrt(2.0), atol=1e-8)


def test_eigh_reconstructs_a() raises:
    """A == v @ diag(w) @ v.T for a 3x3 symmetric matrix."""
    var A = nm.fromstring[sj.f64]("[[4, 1, 0], [1, 3, 1], [0, 1, 2]]")
    var result = eigh(A)
    var w = result[0].copy()
    var v = result[1].copy()

    var n = A.shape[0]
    var D = nm.zeros[sj.f64](nm.Shape(n, n))
    for i in range(n):
        D.store(i * n + i, val=w.item(i))
    var reconstructed = matmul(matmul(v, D), nm.transpose(v))
    _assert_matrix_almost_equal(A, reconstructed, atol=1e-6)

    # A @ v[:, i] == w[i] * v[:, i] for every eigenpair.
    for i in range(n):
        var col = nm.zeros[sj.f64](nm.Shape(n))
        for k in range(n):
            col.store(k, val=v.item(k, i))
        var Av = matmul(A, col)
        for k in range(n):
            assert_almost_equal(Av.item(k), w.item(i) * col.item(k), atol=1e-6)


def test_eigvalsh_matches_eigh() raises:
    var A = nm.fromstring[sj.f64]("[[4, 1, 0], [1, 3, 1], [0, 1, 2]]")
    var w_only = eigvalsh(A)
    var result = eigh(A)
    var w = result[0].copy()
    for i in range(len(w)):
        assert_almost_equal(w_only.item(i), w.item(i), atol=1e-10)


def test_eigh_matches_numpy() raises:
    var np: PythonObject
    try:
        np = Python.import_module("numpy")
    except:
        print("NumPy not available, skipping eigh comparison test")
        return

    var A = nm.fromstring[sj.f64]("[[4, 1, 0], [1, 3, 1], [0, 1, 2]]")
    var result = eigh(A)
    var w = result[0].copy()

    var py_A = np.array(
        Python.list(
            Python.list(4.0, 1.0, 0.0),
            Python.list(1.0, 3.0, 1.0),
            Python.list(0.0, 1.0, 2.0),
        )
    )
    var py_result = np.linalg.eigh(py_A)
    var py_w = py_result[0]

    for i in range(len(w)):
        assert_almost_equal(
            w.item(i), Float64(py=py_w[i]), atol=1e-8, msg="eigh vs numpy"
        )


def test_svd_reconstructs_a_tall() raises:
    """A == U @ diag(s) @ Vt for a 3x2 (tall) matrix."""
    var A = nm.fromstring[sj.f64]("[[1, 2], [3, 4], [5, 6]]")
    var result = svd(A)
    var U = result[0].copy()
    var s = result[1].copy()
    var Vt = result[2].copy()

    var k = len(s)
    var D = nm.zeros[sj.f64](nm.Shape(k, k))
    for i in range(k):
        D.store(i * k + i, val=s.item(i))
    var reconstructed = matmul(matmul(U, D), Vt)
    _assert_matrix_almost_equal(A, reconstructed, atol=1e-6)


def test_svd_reconstructs_a_wide() raises:
    """A == U @ diag(s) @ Vt for a 2x3 (wide) matrix."""
    var A = nm.fromstring[sj.f64]("[[1, 2, 3], [4, 5, 6]]")
    var result = svd(A)
    var U = result[0].copy()
    var s = result[1].copy()
    var Vt = result[2].copy()

    var k = len(s)
    var D = nm.zeros[sj.f64](nm.Shape(k, k))
    for i in range(k):
        D.store(i * k + i, val=s.item(i))
    var reconstructed = matmul(matmul(U, D), Vt)
    _assert_matrix_almost_equal(A, reconstructed, atol=1e-6)


def test_svdvals_matches_svd() raises:
    var A = nm.fromstring[sj.f64]("[[1, 2], [3, 4], [5, 6]]")
    var s_only = svdvals(A)
    var result = svd(A)
    var s = result[1].copy()
    for i in range(len(s)):
        assert_almost_equal(s_only.item(i), s.item(i), atol=1e-10)


def test_svd_matches_numpy() raises:
    var np: PythonObject
    try:
        np = Python.import_module("numpy")
    except:
        print("NumPy not available, skipping svd comparison test")
        return

    var A = nm.fromstring[sj.f64]("[[1, 2], [3, 4], [5, 6]]")
    var s = svdvals(A)

    var py_A = np.array(
        Python.list(
            Python.list(1.0, 2.0),
            Python.list(3.0, 4.0),
            Python.list(5.0, 6.0),
        )
    )
    var py_s = np.linalg.svd(py_A, compute_uv=False)

    for i in range(len(s)):
        assert_almost_equal(
            s.item(i), Float64(py=py_s[i]), atol=1e-8, msg="svd vs numpy"
        )


def test_matrix_rank() raises:
    var full_rank = nm.fromstring[sj.f64]("[[1, 2], [3, 4], [5, 6]]")
    assert_equal(matrix_rank(full_rank), 2)

    var deficient = nm.fromstring[sj.f64]("[[1, 2], [2, 4]]")
    assert_equal(matrix_rank(deficient), 1)


def test_cond() raises:
    var A = nm.fromstring[sj.f64]("[[1, 0], [0, 4]]")
    # Diagonal matrix: singular values are the diagonal entries themselves.
    assert_almost_equal(cond(A), 4.0, atol=1e-8)


def test_pinv_rank_deficient() raises:
    """Rank-deficient pinv (not supported by the old lstsq-based pinv):
    `A @ A_pinv @ A == A` is the defining Moore-Penrose property that
    still holds even when `A` is singular.
    """
    var A = nm.fromstring[sj.f64]("[[1, 2], [2, 4]]")
    var A_pinv = pinv(A)
    var reconstructed = matmul(matmul(A, A_pinv), A)
    _assert_matrix_almost_equal(A, reconstructed, atol=1e-6)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
