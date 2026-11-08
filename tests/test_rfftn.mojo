"""
Tests for rfftn, irfftn, rfft2, and irfft2.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.python import (
    Python,
    PythonObject,
)
from std.testing import (
    assert_almost_equal,
    assert_true,
    TestSuite,
)

# ===----------------------------------------------------------------------=== #
# External
# ===----------------------------------------------------------------------=== #
import numojo as nm
from numojo.core.layout import NDArrayShape

# ===----------------------------------------------------------------------=== #
# SciJo
# ===----------------------------------------------------------------------=== #
from scijo.fft import (
    irfft2,
    irfftn,
    rfft2,
    rfftn,
)


def _make_2d_real_arr() raises -> nm.NDArray[nm.f64]:
    var arr = nm.NDArray[nm.f64](NDArrayShape(4, 4))
    for i in range(4):
        for j in range(4):
            arr.store(i, j, val=Float64(i * 4 + j))
    return arr^


def test_rfft2_matches_numpy() raises:
    var np: PythonObject
    try:
        np = Python.import_module("numpy")
    except:
        print("NumPy not available, skipping test_rfft2_matches_numpy")
        return

    var arr = _make_2d_real_arr()
    var result = rfft2[nm.f64](arr)

    var np_arr = np.arange(16, dtype=np.float64).reshape(4, 4)
    var np_result = np.fft.rfft2(np_arr)

    assert_true(result.shape[0] == 4)
    assert_true(result.shape[1] == 3)
    for i in range(4):
        for j in range(3):
            var val = result.item(i, j)
            var np_val = np_result[i][j]
            assert_almost_equal(
                Float64(val.re),
                Float64(py=np_val.real),
                atol=1e-8,
                msg="rfft2 real mismatch",
            )
            assert_almost_equal(
                Float64(val.im),
                Float64(py=np_val.imag),
                atol=1e-8,
                msg="rfft2 imag mismatch",
            )


def test_rfftn_matches_numpy_all_axes() raises:
    var np: PythonObject
    try:
        np = Python.import_module("numpy")
    except:
        print("NumPy not available, skipping test_rfftn_matches_numpy_all_axes")
        return

    var arr = _make_2d_real_arr()
    var result = rfftn[nm.f64](arr)

    var np_arr = np.arange(16, dtype=np.float64).reshape(4, 4)
    var np_result = np.fft.rfftn(np_arr)

    assert_true(result.shape[0] == 4)
    assert_true(result.shape[1] == 3)
    for i in range(4):
        for j in range(3):
            var val = result.item(i, j)
            var np_val = np_result[i][j]
            assert_almost_equal(
                Float64(val.re),
                Float64(py=np_val.real),
                atol=1e-8,
                msg="rfftn real mismatch",
            )
            assert_almost_equal(
                Float64(val.im),
                Float64(py=np_val.imag),
                atol=1e-8,
                msg="rfftn imag mismatch",
            )


def test_rfft2_irfft2_roundtrip() raises:
    var arr = _make_2d_real_arr()
    var freq = rfft2[nm.f64](arr)
    var back = irfft2[nm.f64](freq, s=4)

    for i in range(4):
        for j in range(4):
            assert_almost_equal(
                back.item(i, j),
                arr.item(i, j),
                atol=1e-8,
                msg="rfft2/irfft2 round trip mismatch",
            )


def test_rfftn_irfftn_roundtrip() raises:
    var arr = _make_2d_real_arr()
    var freq = rfftn[nm.f64](arr)
    var back = irfftn[nm.f64](freq, s=4)

    for i in range(4):
        for j in range(4):
            assert_almost_equal(
                back.item(i, j),
                arr.item(i, j),
                atol=1e-8,
                msg="rfftn/irfftn round trip mismatch",
            )


def test_rfftn_single_axis_matches_numpy() raises:
    var np: PythonObject
    try:
        np = Python.import_module("numpy")
    except:
        print(
            "NumPy not available, skipping test_rfftn_single_axis_matches_numpy"
        )
        return

    var arr = _make_2d_real_arr()
    var axes: List[Int] = [1]
    var result = rfftn[nm.f64](arr, axes^)

    var np_arr = np.arange(16, dtype=np.float64).reshape(4, 4)
    var np_result = np.fft.rfftn(np_arr, axes=Python.list(1))

    assert_true(result.shape[0] == 4)
    assert_true(result.shape[1] == 3)
    for i in range(4):
        for j in range(3):
            var val = result.item(i, j)
            var np_val = np_result[i][j]
            assert_almost_equal(
                Float64(val.re),
                Float64(py=np_val.real),
                atol=1e-8,
                msg="rfftn (single axis) real mismatch",
            )
            assert_almost_equal(
                Float64(val.im),
                Float64(py=np_val.imag),
                atol=1e-8,
                msg="rfftn (single axis) imag mismatch",
            )


def test_rfftn_3d_matches_numpy() raises:
    var np: PythonObject
    try:
        np = Python.import_module("numpy")
    except:
        print("NumPy not available, skipping test_rfftn_3d_matches_numpy")
        return

    var arr = nm.NDArray[nm.f64](NDArrayShape(2, 2, 4))
    for i in range(2):
        for j in range(2):
            for k in range(4):
                arr.store(i, j, k, val=Float64(i * 8 + j * 4 + k))

    var result = rfftn[nm.f64](arr)

    var np_arr = np.arange(16, dtype=np.float64).reshape(2, 2, 4)
    var np_result = np.fft.rfftn(np_arr)

    assert_true(result.shape[0] == 2)
    assert_true(result.shape[1] == 2)
    assert_true(result.shape[2] == 3)
    for i in range(2):
        for j in range(2):
            for k in range(3):
                var val = result.item(i, j, k)
                var np_val = np_result[i][j][k]
                assert_almost_equal(
                    Float64(val.re),
                    Float64(py=np_val.real),
                    atol=1e-8,
                    msg="rfftn (3-D) real mismatch",
                )
                assert_almost_equal(
                    Float64(val.im),
                    Float64(py=np_val.imag),
                    atol=1e-8,
                    msg="rfftn (3-D) imag mismatch",
                )


def test_rfftn_empty_axes_raises() raises:
    var arr = _make_2d_real_arr()
    var axes = List[Int]()
    var raised = False
    try:
        _ = rfftn[nm.f64](arr, axes^)
    except:
        raised = True
    assert_true(raised, msg="empty axes should raise")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
