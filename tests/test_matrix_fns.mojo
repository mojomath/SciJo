"""
Tests for expm, block_diag, toeplitz, null_space, and orth.
"""

from std.python import Python, PythonObject
from std.testing import (
    assert_almost_equal,
    assert_true,
    TestSuite,
)

import numojo as nm

import scijo as sj
from scijo.linalg import (
    block_diag,
    expm,
    matmul,
    norm,
    null_space,
    orth,
    toeplitz,
)


def _assert_matrix_almost_equal(
    a: nm.NDArray[sj.f64], b: nm.NDArray[sj.f64], atol: Float64 = 1e-6
) raises:
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            assert_almost_equal(a.item(i, j), b.item(i, j), atol=atol)


def test_expm_identity() raises:
    """Exp(0) == I."""
    var A = nm.zeros[sj.f64](nm.Shape(3, 3))
    var result = expm(A)
    for i in range(3):
        for j in range(3):
            var expected: Float64 = 1.0 if i == j else 0.0
            assert_almost_equal(result.item(i, j), expected, atol=1e-10)


def test_expm_diagonal_matches_scalar_exp() raises:
    """Exp(diag(a, b)) == diag(e^a, e^b)."""
    from std.math import exp

    var A = nm.fromstring[sj.f64]("[[1.0, 0.0], [0.0, 2.0]]")
    var result = expm(A)
    assert_almost_equal(result.item(0, 0), exp(1.0), atol=1e-8)
    assert_almost_equal(result.item(1, 1), exp(2.0), atol=1e-8)
    assert_almost_equal(result.item(0, 1), 0.0, atol=1e-8)
    assert_almost_equal(result.item(1, 0), 0.0, atol=1e-8)


def test_expm_matches_scipy() raises:
    var np: PythonObject
    var scipy_linalg: PythonObject
    try:
        np = Python.import_module("numpy")
        scipy_linalg = Python.import_module("scipy.linalg")
    except:
        print("SciPy not available, skipping expm SciPy test")
        return

    var A = nm.fromstring[sj.f64](
        "[[0.5, 1.0, 0.2], [0.0, 0.3, -0.4], [0.1, 0.0, 0.2]]"
    )
    var result = expm(A)

    var py_A = np.array(
        Python.list(
            Python.list(0.5, 1.0, 0.2),
            Python.list(0.0, 0.3, -0.4),
            Python.list(0.1, 0.0, 0.2),
        )
    )
    var py_result = scipy_linalg.expm(py_A)

    for i in range(3):
        for j in range(3):
            assert_almost_equal(
                result.item(i, j),
                Float64(py=py_result[i][j]),
                atol=1e-6,
                msg="expm should match SciPy",
            )


def test_expm_large_norm_matches_scipy() raises:
    """Exercises the scaling-and-squaring path (norm > 1)."""
    var np: PythonObject
    var scipy_linalg: PythonObject
    try:
        np = Python.import_module("numpy")
        scipy_linalg = Python.import_module("scipy.linalg")
    except:
        print("SciPy not available, skipping expm large-norm SciPy test")
        return

    var A = nm.fromstring[sj.f64](
        "[[1.0, 2.0, -1.0], [0.5, -3.0, 2.0], [-1.0, 1.0, 4.0]]"
    )
    var result = expm(A)

    var py_A = np.array(
        Python.list(
            Python.list(1.0, 2.0, -1.0),
            Python.list(0.5, -3.0, 2.0),
            Python.list(-1.0, 1.0, 4.0),
        )
    )
    var py_result = scipy_linalg.expm(py_A)

    for i in range(3):
        for j in range(3):
            assert_almost_equal(
                result.item(i, j),
                Float64(py=py_result[i][j]),
                atol=1e-5,
                msg="expm (large norm) should match SciPy",
            )


def test_expm_non_square_raises() raises:
    var A = nm.zeros[sj.f64](nm.Shape(2, 3))
    var raised = False
    try:
        _ = expm(A)
    except:
        raised = True
    assert_true(raised, msg="non-square matrix should raise")


def test_block_diag_basic() raises:
    var A = nm.fromstring[sj.f64]("[[1.0, 2.0], [3.0, 4.0]]")
    var B = nm.fromstring[sj.f64]("[[5.0]]")
    var arrays: List[nm.NDArray[sj.f64]] = [A^, B^]
    var result = block_diag(arrays)

    assert_true(result.shape[0] == 3)
    assert_true(result.shape[1] == 3)
    assert_almost_equal(result.item(0, 0), 1.0, atol=1e-10)
    assert_almost_equal(result.item(0, 1), 2.0, atol=1e-10)
    assert_almost_equal(result.item(1, 0), 3.0, atol=1e-10)
    assert_almost_equal(result.item(1, 1), 4.0, atol=1e-10)
    assert_almost_equal(result.item(2, 2), 5.0, atol=1e-10)
    assert_almost_equal(result.item(0, 2), 0.0, atol=1e-10)
    assert_almost_equal(result.item(2, 0), 0.0, atol=1e-10)


def test_block_diag_matches_scipy() raises:
    var np: PythonObject
    var scipy_linalg: PythonObject
    try:
        np = Python.import_module("numpy")
        scipy_linalg = Python.import_module("scipy.linalg")
    except:
        print("SciPy not available, skipping block_diag SciPy test")
        return

    var A = nm.fromstring[sj.f64]("[[1.0, 2.0], [3.0, 4.0]]")
    var B = nm.fromstring[sj.f64]("[[5.0, 6.0, 7.0]]")
    var C = nm.fromstring[sj.f64]("[[8.0], [9.0]]")
    var arrays: List[nm.NDArray[sj.f64]] = [A^, B^, C^]
    var result = block_diag(arrays)

    var py_A = np.array(
        Python.list(Python.list(1.0, 2.0), Python.list(3.0, 4.0))
    )
    var py_B = np.array(Python.list(Python.list(5.0, 6.0, 7.0)))
    var py_C = np.array(Python.list(Python.list(8.0), Python.list(9.0)))
    var py_result = scipy_linalg.block_diag(py_A, py_B, py_C)

    assert_true(result.shape[0] == Int(py=py_result.shape[0]))
    assert_true(result.shape[1] == Int(py=py_result.shape[1]))
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            assert_almost_equal(
                result.item(i, j),
                Float64(py=py_result[i][j]),
                atol=1e-10,
                msg="block_diag should match SciPy",
            )


def test_block_diag_empty_raises() raises:
    var arrays = List[nm.NDArray[sj.f64]]()
    var raised = False
    try:
        _ = block_diag(arrays)
    except:
        raised = True
    assert_true(raised, msg="empty list should raise")


def test_toeplitz_symmetric() raises:
    var c = nm.fromstring[sj.f64]("[1.0, 2.0, 3.0]")
    var result = toeplitz(c)

    var expected = nm.fromstring[sj.f64](
        "[[1.0, 2.0, 3.0], [2.0, 1.0, 2.0], [3.0, 2.0, 1.0]]"
    )
    _assert_matrix_almost_equal(result, expected, atol=1e-10)


def test_toeplitz_rectangular_matches_scipy() raises:
    var np: PythonObject
    var scipy_linalg: PythonObject
    try:
        np = Python.import_module("numpy")
        scipy_linalg = Python.import_module("scipy.linalg")
    except:
        print("SciPy not available, skipping toeplitz SciPy test")
        return

    var c = nm.fromstring[sj.f64]("[1.0, 2.0, 3.0, 4.0]")
    var r = nm.fromstring[sj.f64]("[1.0, 5.0, 6.0]")
    var result = toeplitz(c, r^)

    var py_c = np.array(Python.list(1.0, 2.0, 3.0, 4.0))
    var py_r = np.array(Python.list(1.0, 5.0, 6.0))
    var py_result = scipy_linalg.toeplitz(py_c, py_r)

    assert_true(result.shape[0] == 4)
    assert_true(result.shape[1] == 3)
    for i in range(4):
        for j in range(3):
            assert_almost_equal(
                result.item(i, j),
                Float64(py=py_result[i][j]),
                atol=1e-10,
                msg="toeplitz should match SciPy",
            )


def test_toeplitz_empty_raises() raises:
    var c = nm.zeros[sj.f64](nm.Shape(0))
    var raised = False
    try:
        _ = toeplitz(c)
    except:
        raised = True
    assert_true(raised, msg="empty first column should raise")


def test_null_space_rank_deficient() raises:
    """A rank-1 2x2 matrix has a 1-D null space."""
    var A = nm.fromstring[sj.f64]("[[1.0, 2.0], [2.0, 4.0]]")
    var ns = null_space(A)

    assert_true(ns.shape[0] == 2)
    assert_true(ns.shape[1] == 1)

    # A @ ns should be (near) zero.
    var Av = matmul(A, ns)
    assert_almost_equal(Av.item(0, 0), 0.0, atol=1e-8)
    assert_almost_equal(Av.item(1, 0), 0.0, atol=1e-8)

    # The basis vector should be unit norm.
    var col = nm.zeros[sj.f64](nm.Shape(2))
    col.store(0, val=ns.item(0, 0))
    col.store(1, val=ns.item(1, 0))
    assert_almost_equal(norm(col), 1.0, atol=1e-8)


def test_null_space_full_rank_is_empty() raises:
    var A = nm.fromstring[sj.f64]("[[1.0, 0.0], [0.0, 1.0]]")
    var ns = null_space(A)
    assert_true(ns.shape[1] == 0)


def test_orth_full_rank_matches_ncols() raises:
    var A = nm.fromstring[sj.f64]("[[1.0, 0.0], [0.0, 1.0]]")
    var ob = orth(A)
    assert_true(ob.shape[0] == 2)
    assert_true(ob.shape[1] == 2)


def test_orth_rank_deficient() raises:
    """A rank-1 2x2 matrix has a 1-D column space."""
    var A = nm.fromstring[sj.f64]("[[1.0, 2.0], [2.0, 4.0]]")
    var ob = orth(A)
    assert_true(ob.shape[1] == 1)

    # The basis vector should be unit norm.
    var col = nm.zeros[sj.f64](nm.Shape(2))
    col.store(0, val=ob.item(0, 0))
    col.store(1, val=ob.item(1, 0))
    assert_almost_equal(norm(col), 1.0, atol=1e-8)


def test_null_space_and_orth_rank_complementary() raises:
    """Dim(null_space(A)) + dim(orth(A)) == n (rank-nullity theorem)."""
    var A = nm.fromstring[sj.f64](
        "[[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 1.0, 1.0]]"
    )
    var ns = null_space(A)
    var ob = orth(A)
    assert_true(ns.shape[1] + ob.shape[1] == 3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
