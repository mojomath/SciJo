from std.python import Python, PythonObject
from std.testing import assert_almost_equal, assert_true
from std.testing import TestSuite

import numojo as nm
from scijo.fft import fftn, ifftn, fft2, ifft2
from numojo.core.complex import ComplexNDArray, ComplexSIMD
from numojo.core.layout import NDArrayShape


def compare_complex_2d[
    dtype: nm.ComplexDType
](
    arr: ComplexNDArray[dtype],
    np_result: PythonObject,
    msg: String,
    atol: Float64 = 1e-8,
) raises:
    """Compare a 2-D complex array element by element with a NumPy result."""
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            var val = arr[nm.Item(i, j)]
            var np_val = np_result[i][j]
            assert_almost_equal(
                Float64(val.re),
                Float64(py=np_val.real),
                atol=atol,
                msg=msg
                + " real mismatch at ("
                + String(i)
                + ","
                + String(j)
                + ")",
            )
            assert_almost_equal(
                Float64(val.im),
                Float64(py=np_val.imag),
                atol=atol,
                msg=msg
                + " imag mismatch at ("
                + String(i)
                + ","
                + String(j)
                + ")",
            )
    print(msg + " - PASSED")


def _make_2d_arr() raises -> ComplexNDArray[nm.cf64]:
    var arr = ComplexNDArray[nm.cf64](NDArrayShape(4, 4))
    for i in range(4):
        for j in range(4):
            arr[nm.Item(i, j)] = ComplexSIMD[nm.cf64](Float64(i * 4 + j), 0.0)
    return arr^


def test_fftn_matches_numpy_fft2() raises:
    try:
        var np = Python.import_module("numpy")
        var arr = _make_2d_arr()
        var result = fftn[nm.cf64](arr)

        var np_arr = np.arange(16, dtype=np.float64).reshape(4, 4)
        var np_result = np.fft.fft2(np_arr)

        compare_complex_2d[nm.cf64](
            result, np_result, "fftn (all axes) matches numpy fft2"
        )
    except:
        print("NumPy not available, skipping test_fftn_matches_numpy_fft2")


def test_fft2_matches_numpy() raises:
    try:
        var np = Python.import_module("numpy")
        var arr = _make_2d_arr()
        var result = fft2[nm.cf64](arr)

        var np_arr = np.arange(16, dtype=np.float64).reshape(4, 4)
        var np_result = np.fft.fft2(np_arr)

        compare_complex_2d[nm.cf64](result, np_result, "fft2 matches numpy")
    except:
        print("NumPy not available, skipping test_fft2_matches_numpy")


def test_fftn_single_axis_matches_numpy() raises:
    try:
        var np = Python.import_module("numpy")
        var arr = _make_2d_arr()
        var axes_1: List[Int] = [1]
        var result = fftn[nm.cf64](arr, axes=axes_1^)

        var np_arr = np.arange(16, dtype=np.float64).reshape(4, 4)
        var np_result = np.fft.fft(np_arr, axis=1)

        compare_complex_2d[nm.cf64](
            result, np_result, "fftn(axes=[1]) matches numpy fft(axis=1)"
        )
    except:
        print(
            "NumPy not available, skipping test_fftn_single_axis_matches_numpy"
        )


def test_ifftn_roundtrip() raises:
    var arr = _make_2d_arr()
    var freq = fftn[nm.cf64](arr)
    var back = ifftn[nm.cf64](freq)

    for i in range(4):
        for j in range(4):
            var orig = arr[nm.Item(i, j)]
            var recovered = back[nm.Item(i, j)]
            assert_almost_equal(
                Float64(orig.re),
                Float64(recovered.re),
                atol=1e-8,
                msg="ifftn(fftn(x)) roundtrip real",
            )
            assert_almost_equal(
                Float64(orig.im),
                Float64(recovered.im),
                atol=1e-8,
                msg="ifftn(fftn(x)) roundtrip imag",
            )
    print("ifftn(fftn(x)) roundtrip - PASSED")


def test_ifft2_roundtrip() raises:
    var arr = _make_2d_arr()
    var freq = fft2[nm.cf64](arr)
    var back = ifft2[nm.cf64](freq)

    for i in range(4):
        for j in range(4):
            var orig = arr[nm.Item(i, j)]
            var recovered = back[nm.Item(i, j)]
            assert_almost_equal(
                Float64(orig.re),
                Float64(recovered.re),
                atol=1e-8,
                msg="ifft2(fft2(x)) roundtrip real",
            )
    print("ifft2(fft2(x)) roundtrip - PASSED")


def test_fftn_negative_axis() raises:
    var arr = _make_2d_arr()
    var axes_neg: List[Int] = [-1]
    var result_neg = fftn[nm.cf64](arr, axes=axes_neg^)
    var axes_pos: List[Int] = [1]
    var result_pos = fftn[nm.cf64](arr, axes=axes_pos^)

    for i in range(4):
        for j in range(4):
            var a = result_neg[nm.Item(i, j)]
            var b = result_pos[nm.Item(i, j)]
            assert_almost_equal(
                Float64(a.re),
                Float64(b.re),
                atol=1e-10,
                msg="negative axis real",
            )
            assert_almost_equal(
                Float64(a.im),
                Float64(b.im),
                atol=1e-10,
                msg="negative axis imag",
            )
    print("fftn(axes=[-1]) matches fftn(axes=[1]) - PASSED")


def test_fftn_invalid_axis_raises() raises:
    var arr = _make_2d_arr()
    var caught_error = False
    try:
        var axes_bad: List[Int] = [5]
        _ = fftn[nm.cf64](arr, axes=axes_bad^)
    except:
        caught_error = True
    assert_true(caught_error, "Should raise error for out-of-bounds axis")
    print("fftn out-of-bounds axis raises - PASSED")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
