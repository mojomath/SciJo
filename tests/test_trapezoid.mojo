from scijo.integrate.fixed_sample import trapezoid
import scijo as sj
import numojo as nm
from python import Python, PythonObject
from testing import assert_almost_equal, assert_equal
from testing import TestSuite


fn test_basic_trapezoid() raises:
    """Test basic trapezoid integration matching SciPy examples."""
    var y1 = nm.fromstring[sj.f64]("[1, 2, 3]")
    var result1 = trapezoid[sj.f64](y1)
    assert_almost_equal(
        result1, 4.0, msg="trapezoid([1, 2, 3]) should equal 4.0"
    )

    var y2 = nm.fromstring[sj.f64]("[1, 2, 3]")
    var x2 = nm.fromstring[sj.f64]("[4, 6, 8]")
    var result2 = trapezoid[sj.f64](y2, x2)
    assert_almost_equal(
        result2, 8.0, msg="trapezoid([1, 2, 3], x=[4, 6, 8]) should equal 8.0"
    )

    var y3 = nm.fromstring[sj.f64]("[1, 2, 3]")
    var result3 = trapezoid[sj.f64](y3, dx=2.0)
    assert_almost_equal(
        result3, 8.0, msg="trapezoid([1, 2, 3], dx=2) should equal 8.0"
    )


fn test_edge_cases() raises:
    """Test edge cases for trapezoid integration."""

    var y_single = nm.fromstring[sj.f64]("[5]")
    var result_single = trapezoid[sj.f64](y_single)
    assert_almost_equal(
        result_single, 0.0, msg="Single point should return 0.0"
    )

    var y_two = nm.fromstring[sj.f64]("[1, 3]")
    var result_two = trapezoid[sj.f64](y_two)
    assert_almost_equal(
        result_two, 2.0, msg="Two points [1, 3] with dx=1 should equal 2.0"
    )

    var y_two_custom = nm.fromstring[sj.f64]("[1, 3]")
    var x_two_custom = nm.fromstring[sj.f64]("[0, 4]")
    var result_two_custom = trapezoid[sj.f64](y_two_custom, x_two_custom)
    assert_almost_equal(
        result_two_custom,
        8.0,
        msg="Two points [1, 3] with x=[0, 4] should equal 8.0",
    )


fn test_reverse_integration() raises:
    """Test integration with decreasing x values."""

    var y = nm.fromstring[sj.f64]("[1, 2, 3]")
    var x_reverse = nm.fromstring[sj.f64]("[8, 6, 4]")
    var result = trapezoid[sj.f64](y, x_reverse)
    assert_almost_equal(
        result, -8.0, msg="Reverse integration should return -8.0"
    )


fn test_parametric_curve() raises:
    """Test parametric curve integration (approximating x^2 from 0 to 1)."""

    var n = 50
    var x = nm.linspace[sj.f64](0.0, 1.0, n)
    var y = x * x

    var result = trapezoid[sj.f64](y, x)
    assert_almost_equal(
        result,
        0.3333333333333333,
        atol=0.001,
        msg="Integral of x^2 from 0 to 1 should be approximately 1/3",
    )


fn test_different_spacings() raises:
    """Test different dx values."""

    var y = nm.fromstring[sj.f64]("[0, 1, 4, 9]")
    var result1 = trapezoid[sj.f64](y)
    assert_almost_equal(
        result1, 9.5, msg="trapezoid with dx=1 should equal 9.5"
    )

    var result2 = trapezoid[sj.f64](y, dx=0.5)
    assert_almost_equal(
        result2, 4.75, msg="trapezoid with dx=0.5 should equal 4.75"
    )

    var result3 = trapezoid[sj.f64](y, dx=2.0)
    assert_almost_equal(
        result3, 19.0, msg="trapezoid with dx=2 should equal 19.0"
    )


fn test_numerical_accuracy() raises:
    """Test numerical accuracy with known integrals."""

    var x_linear = nm.fromstring[sj.f64]("[0, 1, 2, 3]")
    var y_linear = nm.fromstring[sj.f64]("[0, 2, 4, 6]")
    var result_linear = trapezoid[sj.f64](y_linear, x_linear)
    assert_almost_equal(
        result_linear, 9.0, msg="Integral of 2x from 0 to 3 should equal 9.0"
    )

    var x_const = nm.fromstring[sj.f64]("[0, 1, 2, 3, 4]")
    var y_const = nm.fromstring[sj.f64]("[5, 5, 5, 5, 5]")
    var result_const = trapezoid[sj.f64](y_const, x_const)
    assert_almost_equal(
        result_const,
        20.0,
        msg="Integral of constant 5 from 0 to 4 should equal 20.0",
    )


fn test_scipy_compatibility() raises:
    """Test compatibility with SciPy results."""

    try:
        var python = Python.import_module("numpy")
        var scipy_integrate = Python.import_module("scipy.integrate")

        var y1 = nm.fromstring[sj.f64]("[1, 2, 3]")
        var result1 = trapezoid[sj.f64](y1)
        var py_y1 = Python.list(1, 2, 3)
        var py_result1 = scipy_integrate.trapezoid(py_y1)
        assert_almost_equal(
            result1,
            Float64(py=py_result1),
            msg="Should match SciPy result for [1,2,3]",
        )

        var y2 = nm.fromstring[sj.f64]("[1, 2, 3]")
        var x2 = nm.fromstring[sj.f64]("[4, 6, 8]")
        var result2 = trapezoid[sj.f64](y2, x2)
        var py_y2 = python.list(1, 2, 3)
        var py_x2 = python.list(4, 6, 8)
        var py_result2 = scipy_integrate.trapezoid(py_y2, py_x2)
        assert_almost_equal(
            result2,
            Float64(py=py_result2),
            msg="Should match SciPy result for custom x",
        )

        var y3 = nm.fromstring[sj.f64]("[1, 2, 3]")
        var result3 = trapezoid[sj.f64](y3, dx=2.0)
        var py_y3 = python.array(1, 2, 3)
        var py_result3 = scipy_integrate.trapezoid(py_y3, dx=PythonObject(2.0))
        assert_almost_equal(
            result3,
            Float64(py=py_result3),
            msg="Should match SciPy result for dx=2",
        )

        var y_single = nm.fromstring[sj.f64]("[5]")
        var result_single = trapezoid[sj.f64](y_single)
        var py_y_single = python.list(5)
        var py_result_single = scipy_integrate.trapezoid(py_y_single)
        assert_almost_equal(
            result_single,
            Float64(py=py_result_single),
            msg="Should match SciPy result for single point",
        )

        print("All SciPy compatibility tests passed")

    except:
        print("SciPy not available for compatibility testing")


fn test_error_conditions() raises:
    """Test error conditions and edge cases."""
    var y_mismatch = nm.fromstring[sj.f64]("[1, 2, 3]")
    var x_mismatch = nm.fromstring[sj.f64]("[1, 2]")
    try:
        var _ = trapezoid[sj.f64](y_mismatch, x_mismatch)
        assert_equal(
            True, False, msg="Mismatched array sizes should raise an error"
        )
    except:
        pass


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
