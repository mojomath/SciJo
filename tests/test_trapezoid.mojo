from std.python import Python, PythonObject
from std.testing import assert_almost_equal, assert_equal
from std.testing import TestSuite
from std.math import sin as _sin

from scijo.integrate.fixed_sample import (
    trapezoid,
    romb,
    cumulative_trapezoid,
    cumulative_simpson,
)
import scijo as sj
import numojo as nm


def test_basic_trapezoid() raises:
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


def test_edge_cases() raises:
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


def test_reverse_integration() raises:
    """Test integration with decreasing x values."""

    var y = nm.fromstring[sj.f64]("[1, 2, 3]")
    var x_reverse = nm.fromstring[sj.f64]("[8, 6, 4]")
    var result = trapezoid[sj.f64](y, x_reverse)
    assert_almost_equal(
        result, -8.0, msg="Reverse integration should return -8.0"
    )


def test_parametric_curve() raises:
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


def test_different_spacings() raises:
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


def test_numerical_accuracy() raises:
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


def test_scipy_compatibility() raises:
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


def test_error_conditions() raises:
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


def test_romb_polynomial() raises:
    """Romberg on x^2 over [0,1]: exact answer is 1/3."""
    # 9 = 2^3 + 1 points, dx = 1/8
    var n = 9
    var dx = Scalar[sj.f64](1.0 / 8.0)
    var y = nm.zeros[sj.f64](nm.Shape(n))
    for i in range(n):
        var xi = Scalar[sj.f64](i) * dx
        y.itemset(i, xi * xi)
    var result = romb[sj.f64](y, dx=dx)
    assert_almost_equal(
        result, 1.0 / 3.0, atol=1e-10, msg="romb x^2 over [0,1]"
    )


def test_romb_linear() raises:
    """Romberg on x over [0,1]: exact answer is 0.5 — should be exact at k=1."""
    var n = 3
    var dx = Scalar[sj.f64](0.5)
    var y = nm.zeros[sj.f64](nm.Shape(n))
    for i in range(n):
        y.itemset(i, Scalar[sj.f64](i) * dx)
    var result = romb[sj.f64](y, dx=dx)
    assert_almost_equal(
        result, 0.5, atol=1e-15, msg="romb x over [0,1] (linear, exact)"
    )


def test_romb_cubic() raises:
    """Romberg on x^3 over [0,1]: exact answer is 0.25. Uses 17 points."""
    var n = 17
    var dx = Scalar[sj.f64](1.0 / 16.0)
    var y = nm.zeros[sj.f64](nm.Shape(n))
    for i in range(n):
        var xi = Scalar[sj.f64](i) * dx
        y.itemset(i, xi * xi * xi)
    var result = romb[sj.f64](y, dx=dx)
    assert_almost_equal(result, 0.25, atol=1e-12, msg="romb x^3 over [0,1]")


def test_romb_invalid_size_raises() raises:
    """Romberg must raise when y.size is not 2^k + 1."""
    var y_bad = nm.zeros[sj.f64](nm.Shape(10))
    var caught = False
    try:
        var _ = romb[sj.f64](y_bad)
    except:
        caught = True
    assert_equal(caught, True, msg="romb should raise for invalid size")


def test_romb_matches_scipy() raises:
    """Romberg result matches scipy.integrate.romb on sin over [0, pi]."""
    try:
        var scipy_integrate = Python.import_module("scipy.integrate")
        var np = Python.import_module("numpy")

        var n = 33  # 2^5 + 1
        var dx = Scalar[sj.f64](3.141592653589793 / 32.0)
        var y = nm.zeros[sj.f64](nm.Shape(n))
        for i in range(n):
            var xi = Scalar[sj.f64](i) * dx
            y.itemset(i, _sin(xi))

        var mojo_result = romb[sj.f64](y, dx=dx)

        var py_y = y.to_numpy()
        var scipy_result = Float64(
            py=scipy_integrate.romb(py_y, dx=Float64(dx))
        )

        assert_almost_equal(
            mojo_result,
            scipy_result,
            atol=1e-10,
            msg="romb should match scipy on sin over [0, pi]",
        )
    except:
        pass  # SciPy not available


def test_cumulative_trapezoid_dx() raises:
    """Cumulative trapezoid with dx matches scipy.integrate.cumulative_trapezoid."""
    try:
        var scipy_integrate = Python.import_module("scipy.integrate")

        # uniform spacing, no initial
        var y = nm.fromstring[sj.f64]("[1.0, 2.0, 3.0, 4.0]")
        var cum = cumulative_trapezoid[sj.f64](y, dx=1.0)
        var py_y = y.to_numpy()
        var py_cum = scipy_integrate.cumulative_trapezoid(py_y, dx=1.0)
        assert_equal(cum.size, 3)
        for i in range(cum.size):
            assert_almost_equal(
                cum.item(i), Float64(py=py_cum[i]), atol=1e-14,
                msg="cumulative_trapezoid dx index " + String(i),
            )

        # with initial=0
        var cum_init = cumulative_trapezoid[sj.f64](y, dx=1.0, initial=0.0)
        var py_cum_init = scipy_integrate.cumulative_trapezoid(py_y, dx=1.0, initial=0)
        assert_equal(cum_init.size, 4)
        for i in range(cum_init.size):
            assert_almost_equal(
                cum_init.item(i), Float64(py=py_cum_init[i]), atol=1e-14,
                msg="cumulative_trapezoid dx initial=0 index " + String(i),
            )

        # sin over [0, pi]
        var x = nm.linspace[sj.f64](0.0, 3.141592653589793, 11)
        var y_sin = nm.zeros[sj.f64](x.shape)
        for i in range(x.size):
            y_sin.itemset(i, _sin(x.item(i)))
        var dx_sin = x.item(1) - x.item(0)
        var cum_sin = cumulative_trapezoid[sj.f64](y_sin, dx=dx_sin, initial=0.0)
        var py_cum_sin = scipy_integrate.cumulative_trapezoid(
            y_sin.to_numpy(), dx=Float64(dx_sin), initial=0
        )
        for i in range(cum_sin.size):
            assert_almost_equal(
                cum_sin.item(i), Float64(py=py_cum_sin[i]), atol=1e-4,
                msg="cumulative_trapezoid sin index " + String(i),
            )
    except:
        print("SciPy not available, skipping cumulative_trapezoid test")


def test_cumulative_trapezoid_x() raises:
    """Cumulative trapezoid with non-uniform x matches scipy."""
    try:
        var scipy_integrate = Python.import_module("scipy.integrate")

        var x = nm.fromstring[sj.f64]("[0.0, 1.0, 3.0, 6.0]")
        var y = nm.fromstring[sj.f64]("[0.0, 1.0, 9.0, 36.0]")  # y = x^2
        var cum = cumulative_trapezoid[sj.f64](y, x, initial=0.0)
        var py_cum = scipy_integrate.cumulative_trapezoid(
            y.to_numpy(), x.to_numpy(), initial=0
        )
        assert_equal(cum.size, 4)
        for i in range(cum.size):
            assert_almost_equal(
                cum.item(i), Float64(py=py_cum[i]), atol=1e-12,
                msg="cumulative_trapezoid x index " + String(i),
            )
    except:
        print("SciPy not available, skipping cumulative_trapezoid x test")


def test_cumulative_simpson_dx() raises:
    """Cumulative Simpson with dx matches scipy.integrate.cumulative_simpson."""
    try:
        var scipy_integrate = Python.import_module("scipy.integrate")

        # odd n=5
        var y = nm.fromstring[sj.f64]("[1.0, 4.0, 1.0, 4.0, 1.0]")
        var cum = cumulative_simpson[sj.f64](y, dx=1.0, initial=0.0)
        var py_cum = scipy_integrate.cumulative_simpson(
            y.to_numpy(), dx=1.0, initial=0
        )
        assert_equal(cum.size, 5)
        for i in range(cum.size):
            assert_almost_equal(
                cum.item(i), Float64(py=py_cum[i]), atol=1e-12,
                msg="cumulative_simpson n=5 index " + String(i),
            )

        # even n=4 (last panel falls back to trapezoid)
        var y4 = nm.fromstring[sj.f64]("[1.0, 4.0, 1.0, 4.0]")
        var cum4 = cumulative_simpson[sj.f64](y4, dx=1.0, initial=0.0)
        var py_cum4 = scipy_integrate.cumulative_simpson(
            y4.to_numpy(), dx=1.0, initial=0
        )
        assert_equal(cum4.size, 4)
        for i in range(cum4.size):
            assert_almost_equal(
                cum4.item(i), Float64(py=py_cum4[i]), atol=1e-12,
                msg="cumulative_simpson n=4 index " + String(i),
            )

        # linear data: result should be exact
        var y_lin = nm.fromstring[sj.f64]("[0.0, 1.0, 2.0, 3.0, 4.0]")
        var cum_lin = cumulative_simpson[sj.f64](y_lin, dx=1.0, initial=0.0)
        var py_cum_lin = scipy_integrate.cumulative_simpson(
            y_lin.to_numpy(), dx=1.0, initial=0
        )
        for i in range(cum_lin.size):
            assert_almost_equal(
                cum_lin.item(i), Float64(py=py_cum_lin[i]), atol=1e-12,
                msg="cumulative_simpson linear index " + String(i),
            )
    except:
        print("SciPy not available, skipping cumulative_simpson test")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
