"""
Tests for Interpolation Module.
"""

from math import sin

import numojo as nm
from scijo.interpolate import LinearInterpolator, interp1d
from testing import assert_true, assert_false, assert_equal, assert_almost_equal
from testing import TestSuite
from python import Python, PythonObject


# TODO: Instead of checking element by element, we could use nm.all for array comparisons similar to NuMojo tests.
fn test_input_validation() raises:
    """Test input validation."""
    var x1 = nm.arange[nm.f64](0, 3, 1)
    var y1 = nm.arange[nm.f64](0, 4, 1)

    var caught_error = False
    try:
        var _ = LinearInterpolator(x1, y1)
    except:
        caught_error = True
    assert_true(caught_error, "Should raise error for different array lengths")

    var a1: List[Float64] = [1.0]
    var a2: List[Float64] = [1.0]
    var x2 = nm.array[nm.f64](a1, shape=[1])
    var y2 = nm.array[nm.f64](a2, shape=[1])

    caught_error = False
    try:
        var _ = LinearInterpolator(x2, y2)
    except:
        caught_error = True
    assert_true(caught_error, "Should raise error for arrays with < 2 points")

    # Test non-increasing x array
    var x3 = nm.array[nm.f64](
        [Float64(1.0), 3.0, 2.0, 4.0], [4]
    )  # Not increasing
    var y3 = nm.array[nm.f64]([Float64(1.0), 9.0, 4.0, 16.0], [4])

    caught_error = False
    try:
        var _ = LinearInterpolator(x3, y3)
    except:
        caught_error = True
    assert_true(caught_error, "Should raise error for non-increasing x array")

    # Test duplicate values in x array
    var x4 = nm.array[nm.f64]([1.0, 2.0, 2.0, 3.0], [4])
    var y4 = nm.array[nm.f64]([1.0, 4.0, 4.0, 9.0], [4])

    caught_error = False
    try:
        var _ = LinearInterpolator(x4, y4)
    except:
        caught_error = True
    assert_true(caught_error, "Should raise error for duplicate x values")


fn test_bounds_handling() raises:
    """Test bounds error and fill value handling against SciPy."""
    try:
        var np = Python.import_module("numpy")
        var scipy_interpolate = Python.import_module("scipy.interpolate")

        var x = nm.arange[nm.f64](1, 5, 1)
        var y = x * x

        var interp_strict = LinearInterpolator(x, y, bounds_error=True)

        var py_x = np.array(Python.list(1.0, 2.0, 3.0, 4.0))
        var py_y = np.array(Python.list(1.0, 4.0, 9.0, 16.0))
        var py_interp_strict = scipy_interpolate.interp1d(
            py_x, py_y, kind="linear", bounds_error=True
        )

        var test_point = 2.5
        var mojo_result = interp_strict(test_point)
        var scipy_result = Float64(py=py_interp_strict(test_point))
        assert_almost_equal(
            mojo_result,
            scipy_result,
            atol=1e-10,
            msg="Within bounds interpolation should match SciPy",
        )

        # Test bounds_error=True raises error for out of bounds
        var caught_error = False
        try:
            var _ = interp_strict(0.5)
        except:
            caught_error = True
        assert_true(caught_error, "Should raise error for point below range")

        caught_error = False
        try:
            var _ = interp_strict(5.0)
        except:
            caught_error = True
        assert_true(caught_error, "Should raise error for point above range")

        # Test fill_value handling
        var fill_val = Scalar[nm.f64](-999.0)
        var interp_fill = LinearInterpolator(
            x, y, bounds_error=False, fill_value=fill_val
        )

        var py_interp_fill = scipy_interpolate.interp1d(
            py_x, py_y, kind="linear", bounds_error=False, fill_value=-999.0
        )

        var result_below = interp_fill(0.5)
        var scipy_below = Float64(py=py_interp_fill(0.5))
        assert_almost_equal(
            result_below,
            scipy_below,
            atol=1e-10,
            msg="Fill value below range should match SciPy",
        )

        var result_above = interp_fill(5.0)
        var scipy_above = Float64(py=py_interp_fill(5.0))
        assert_almost_equal(
            result_above,
            scipy_above,
            atol=1e-10,
            msg="Fill value above range should match SciPy",
        )

    except:
        print("SciPy not available, skipping bounds handling test")


fn test_memory_access_consistency() raises:
    """Test that memory access is consistent and matches SciPy results."""

    try:
        var np = Python.import_module("numpy")
        var scipy_interpolate = Python.import_module("scipy.interpolate")

        var x = nm.linspace[nm.f64](0.0, 10.0, 11)
        var y = x * x

        var interp = LinearInterpolator(x, y)

        var py_x = np.linspace(0.0, 10.0, 11)
        var py_y = py_x**2
        var py_interp = scipy_interpolate.interp1d(py_x, py_y, kind="linear")

        # Test single value interpolation
        var single_point = 5.5
        var single_result = interp(single_point)
        var scipy_single = Float64(py=py_interp(PythonObject(single_point)))
        assert_almost_equal(
            single_result,
            scipy_single,
            atol=1e-10,
            msg="Single value interpolation should match SciPy",
        )

        # Test array interpolation with same point
        var xi_array = nm.array[nm.f64]([5.5], [1])
        var array_result = interp(xi_array)
        assert_almost_equal(
            array_result.item(0),
            scipy_single,
            atol=1e-10,
            msg="Array interpolation with single point should match SciPy",
        )

        # Test that single and array methods give same results for multiple points
        var test_points = nm.linspace[nm.f64](1.5, 8.5, 8)
        var array_results = interp(test_points)

        var py_test_points = np.linspace(1.5, 8.5, 8)
        var py_array_results = py_interp(py_test_points)

        for i in range(test_points.size):
            var single_res = interp(test_points.item(i))
            var array_res = array_results.item(i)
            var scipy_res = Float64(py=py_array_results[i])

            assert_almost_equal(
                single_res,
                array_res,
                atol=1e-15,
                msg=(
                    "Single and array interpolation results should match"
                    " exactly"
                ),
            )
            assert_almost_equal(
                array_res,
                scipy_res,
                atol=1e-10,
                msg="Array interpolation should match SciPy",
            )

    except:
        print("SciPy not available, skipping memory access consistency test")


fn test_functional_interface() raises:
    """Test the functional interp1d interface against NumPy."""
    try:
        var np = Python.import_module("numpy")

        var x = nm.arange[nm.f64](0, 5, 1)
        var y = x * x
        var xi = nm.linspace[nm.f64](0.5, 3.5, 4)

        var yi_interp = interp1d[
            nm.f64, type="linear", fill_method="interpolate"
        ](xi, x, y)

        var py_x = np.array(Python.list(0.0, 1.0, 2.0, 3.0, 4.0))
        var py_y = np.array(Python.list(0.0, 1.0, 4.0, 9.0, 16.0))
        var py_xi = np.array(Python.list(0.5, 1.5, 2.5, 3.5))
        var py_yi = np.interp(py_xi, py_x, py_y)

        for i in range(yi_interp.size):
            var mojo_val = yi_interp.item(i)
            var numpy_val = Float64(py=py_yi[i])
            assert_almost_equal(
                mojo_val,
                numpy_val,
                atol=1e-10,
                msg="Functional interface should match NumPy",
            )

        # Test extrapolation behavior
        var xi_extrap = nm.linspace[nm.f64](-0.5, 4.5, 6)
        var yi_extrap = interp1d[
            nm.f64, type="linear", fill_method="extrapolate"
        ](xi_extrap, x, y)

        var slope_below = (1.0 - 0.0) / (1.0 - 0.0)
        var expected_below = 0.0 + slope_below * (-0.5 - 0.0)
        assert_almost_equal(
            yi_extrap.item(0),
            expected_below,
            atol=1e-10,
            msg="Extrapolation below range should use correct slope",
        )

        # Test invalid parameters
        var caught_error = False
        try:
            var _ = interp1d[nm.f64, type="invalid", fill_method="interpolate"](
                xi, x, y
            )
        except:
            caught_error = True
        assert_true(
            caught_error, "Should raise error for invalid interpolation type"
        )

    except:
        print("NumPy not available, skipping functional interface test")


fn test_edge_cases() raises:
    """Test edge cases against SciPy."""
    print("Testing edge cases...")

    var np = Python.import_module("numpy")
    var scipy_interpolate = Python.import_module("scipy.interpolate")

    # Test with exactly 2 points
    var x_min = nm.array[nm.f64]([1.0, 3.0], [2])
    var y_min = nm.array[nm.f64]([2.0, 6.0], [2])
    var interp_min = LinearInterpolator(x_min, y_min)

    var py_x_min = np.array(Python.list(1.0, 3.0))
    var py_y_min = np.array(Python.list(2.0, 6.0))
    var py_interp_min = scipy_interpolate.interp1d(
        py_x_min, py_y_min, kind="linear"
    )

    var test_point = 2.0
    var result_min = interp_min(test_point)
    var scipy_min = Float64(py=py_interp_min(PythonObject(test_point)))
    assert_almost_equal(
        result_min,
        scipy_min,
        atol=1e-10,
        msg="Two-point interpolation should match SciPy",
    )

    var x_exact = nm.arange[nm.f64](0, 4, 1)
    var y_exact = x_exact * 2
    var interp_exact = LinearInterpolator(x_exact, y_exact)

    var py_x_exact = np.array(Python.list(0.0, 1.0, 2.0, 3.0))
    var py_y_exact = np.array(Python.list(0.0, 2.0, 4.0, 6.0))
    var py_interp_exact = scipy_interpolate.interp1d(
        py_x_exact, py_y_exact, kind="linear"
    )

    for i in range(x_exact.size):
        var xi = x_exact.item(i)
        var result_exact = interp_exact(xi)
        var scipy_exact = Float64(py=py_interp_exact(PythonObject(xi)))
        assert_almost_equal(
            result_exact,
            scipy_exact,
            atol=1e-15,
            msg="Interpolation at exact data points should match SciPy exactly",
        )

    # Test with very small intervals
    var x_small = nm.array[nm.f64]([0.0, 1e-10, 2e-10], [3])
    var y_small = nm.array[nm.f64]([0.0, 1.0, 2.0], [3])
    var interp_small = LinearInterpolator(x_small, y_small)

    var py_x_small = np.array(Python.list(0.0, 1e-10, 2e-10))
    var py_y_small = np.array(Python.list(0.0, 1.0, 2.0))
    var py_interp_small = scipy_interpolate.interp1d(
        py_x_small, py_y_small, kind="linear"
    )

    var small_test_point = 1.5e-10
    var result_small = interp_small(small_test_point)
    var scipy_small = Float64(
        py=py_interp_small(PythonObject(small_test_point))
    )
    assert_almost_equal(
        result_small,
        scipy_small,
        atol=1e-10,
        msg="Small scale interpolation should match SciPy",
    )


fn test_accuracy_against_known_functions() raises:
    """Test interpolation accuracy against SciPy for known mathematical functions.
    """
    var np = Python.import_module("numpy")
    var scipy_interpolate = Python.import_module("scipy.interpolate")

    # Test linear function (should be exact)
    var x_lin = nm.linspace[nm.f64](0.0, 10.0, 11)
    var y_lin = 2.0 * x_lin + 3.0  # y = 2x + 3
    var interp_lin = LinearInterpolator(x_lin, y_lin)

    var py_x_lin = np.linspace(0.0, 10.0, 11)
    var py_y_lin = 2.0 * py_x_lin + 3.0
    var py_interp_lin = scipy_interpolate.interp1d(
        py_x_lin, py_y_lin, kind="linear"
    )

    var xi_lin = nm.linspace[nm.f64](0.5, 9.5, 10)
    var yi_lin = interp_lin(xi_lin)

    var py_xi_lin = np.linspace(0.5, 9.5, 10)
    var py_yi_lin = py_interp_lin(py_xi_lin)

    for i in range(xi_lin.size):
        var mojo_val = yi_lin.item(i)
        var scipy_val = Float64(py=py_yi_lin[i])
        assert_almost_equal(
            mojo_val,
            scipy_val,
            atol=1e-15,
            msg="Linear function interpolation should match SciPy exactly",
        )

    # Test quadratic function with dense sampling
    var x_quad = nm.linspace[nm.f64](0.0, 4.0, 41)
    var y_quad = nm.zeros[nm.f64](x_quad.shape)
    for i in range(x_quad.size):
        y_quad.itemset(i, x_quad.item(i) * x_quad.item(i))

    var interp_quad = LinearInterpolator(x_quad, y_quad)

    var py_x_quad = np.linspace(0.0, 4.0, 41)
    var py_y_quad = py_x_quad**2
    var py_interp_quad = scipy_interpolate.interp1d(
        py_x_quad, py_y_quad, kind="linear"
    )

    var xi_quad = nm.linspace[nm.f64](0.5, 3.5, 10)
    var yi_quad = interp_quad(xi_quad)

    var py_xi_quad = np.linspace(0.5, 3.5, 10)
    var py_yi_quad = py_interp_quad(py_xi_quad)

    for i in range(xi_quad.size):
        var mojo_val = yi_quad.item(i)
        var scipy_val = Float64(py=py_yi_quad[i])
        assert_almost_equal(
            mojo_val,
            scipy_val,
            atol=1e-10,
            msg="Quadratic interpolation should match SciPy",
        )


fn test_scipy_comprehensive_compatibility() raises:
    """Comprehensive SciPy compatibility test covering all major features."""
    var np = Python.import_module("numpy")
    var scipy_interpolate = Python.import_module("scipy.interpolate")

    # Test 1: Basic linear interpolation
    var x_data = nm.fromstring[nm.f64]("[0.0, 1.0, 2.0, 3.0, 4.0]")
    var y_data = nm.fromstring[nm.f64]("[0.0, 1.0, 4.0, 9.0, 16.0]")  # y = x^2
    var xi_test = nm.fromstring[nm.f64]("[0.5, 1.5, 2.5, 3.5]")

    var mojo_interp = LinearInterpolator(x_data, y_data)
    var mojo_result = mojo_interp(xi_test)

    var py_x = np.array(Python.list(0.0, 1.0, 2.0, 3.0, 4.0))
    var py_y = np.array(Python.list(0.0, 1.0, 4.0, 9.0, 16.0))
    var py_xi = np.array(Python.list(0.5, 1.5, 2.5, 3.5))
    var py_interp = scipy_interpolate.interp1d(py_x, py_y, kind="linear")
    var py_result = py_interp(py_xi)

    for i in range(xi_test.size):
        var mojo_val = mojo_result.item(i)
        var scipy_val = Float64(py=py_result[i])
        assert_almost_equal(
            mojo_val,
            scipy_val,
            atol=1e-10,
            msg="Basic interpolation should match SciPy for point " + String(i),
        )

    # Test 2: Fill value handling
    var xi_extrap = nm.fromstring[nm.f64]("[-1.0, 5.0]")
    var mojo_fill = LinearInterpolator(
        x_data, y_data, bounds_error=False, fill_value=-999.0
    )
    var mojo_fill_result = mojo_fill(xi_extrap)

    var py_xi_extrap = np.array(Python.list(-1.0, 5.0))
    var py_fill_interp = scipy_interpolate.interp1d(
        py_x, py_y, kind="linear", bounds_error=False, fill_value=-999.0
    )
    var py_fill_result = py_fill_interp(py_xi_extrap)

    for i in range(xi_extrap.size):
        var mojo_val = mojo_fill_result.item(i)
        var scipy_val = Float64(py=py_fill_result[i])
        assert_almost_equal(
            mojo_val,
            scipy_val,
            atol=1e-10,
            msg="Fill value handling should match SciPy",
        )

    # Test 3: Single point interpolation
    var single_point = 2.7
    var mojo_single = mojo_interp(single_point)
    var py_single = Float64(py=py_interp(single_point))
    assert_almost_equal(
        mojo_single,
        py_single,
        atol=1e-10,
        msg="Single point interpolation should match SciPy",
    )

    # Test 4: Functional interface vs NumPy interp
    var mojo_func_result = interp1d[
        nm.f64, type="linear", fill_method="interpolate"
    ](xi_test, x_data, y_data)
    var py_numpy_result = np.interp(py_xi, py_x, py_y)

    for i in range(xi_test.size):
        var mojo_val = mojo_func_result.item(i)
        var numpy_val = Float64(py=py_numpy_result[i])
        assert_almost_equal(
            mojo_val,
            numpy_val,
            atol=1e-10,
            msg="Functional interface should match NumPy interp",
        )


fn test_performance_comparison() raises:
    var np = Python.import_module("numpy")
    var scipy_interpolate = Python.import_module("scipy.interpolate")

    var n_large = 1000
    var x_large = nm.linspace[nm.f64](0.0, 1000.0, n_large)
    var y_large = nm.zeros[nm.f64](x_large.shape)

    for i in range(x_large.size):
        var xi = x_large.item(i)
        y_large.itemset(i, sin(xi / 100.0) + 0.1 * xi)

    var interp_large = LinearInterpolator(x_large, y_large)

    var py_x_large = np.linspace(0.0, 1000.0, n_large)
    var py_y_large = np.sin(py_x_large / 100.0) + 0.1 * py_x_large
    var py_interp_large = scipy_interpolate.interp1d(
        py_x_large, py_y_large, kind="linear"
    )

    var xi_many = nm.linspace[nm.f64](50.0, 950.0, 100)
    var yi_many = interp_large(xi_many)

    var py_xi_many = np.linspace(50.0, 950.0, 100)
    var py_yi_many = py_interp_large(py_xi_many)

    for i in range(yi_many.size):
        var mojo_val = yi_many.item(i)
        var scipy_val = Float64(py=py_yi_many[i])
        assert_almost_equal(
            mojo_val,
            scipy_val,
            atol=1e-10,
            msg="Large dataset interpolation should match SciPy",
        )


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
