"""
Tests for the linalg module (decompositions, solvers, norms).

To run: `mojo test tests/test_linalg.mojo -I .` from the project root directory.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.math import sqrt
from std.testing import (
    assert_almost_equal,
    assert_equal,
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
    det,
    inv,
    lstsq,
    lu,
    matmul,
    norm,
    pinv,
    qr,
    solve,
    solve_triangular,
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
    assert_almost_equal(t.item(0), 5.0, atol=1e-8)


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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
