"""
Tests for nonlinear least squares (scijo.optimize.least_squares, curve_fit).

To run: `mojo test tests/test_least_squares.mojo -I .` from the project root directory.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
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
from scijo.optimize import curve_fit, least_squares


def test_least_squares_linear_overdetermined() raises:
    """Fit y = a*x + b to noiseless data on a line: a=2, b=1."""

    def residual[
        dtype: DType
    ](
        params: nm.NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) capturing raises -> nm.NDArray[dtype]:
        var xdata = nm.fromstring[dtype]("[0, 1, 2, 3, 4]")
        var ydata = nm.fromstring[dtype]("[1, 3, 5, 7, 9]")
        var a = params.item(0)
        var b = params.item(1)
        return a * xdata + b - ydata

    var p0 = nm.fromstring[sj.f64]("[1, 1]")
    var result = least_squares[sj.f64, residual](p0, atol=1e-12, rtol=1e-12)
    assert_equal(result.success, True)
    assert_almost_equal(result.x.item(0), 2.0, atol=1e-6)
    assert_almost_equal(result.x.item(1), 1.0, atol=1e-6)
    assert_almost_equal(Float64(result.cost), 0.0, atol=1e-10)
    assert_equal(result.method, "lm")


def test_least_squares_nonlinear_exponential() raises:
    """Fit y = a*exp(b*x) to noiseless exponential data: a=2, b=0.5."""

    def residual[
        dtype: DType
    ](
        params: nm.NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) capturing raises -> nm.NDArray[dtype] where dtype.is_floating_point():
        var xdata = nm.fromstring[dtype]("[0, 1, 2, 3, 4, 5]")
        var a = params.item(0)
        var b = params.item(1)
        var pred = a * nm.exp(b * xdata)
        var ydata = 2.0 * nm.exp(0.5 * xdata)
        return pred - ydata

    var p0 = nm.fromstring[sj.f64]("[1, 1]")
    var result = least_squares[sj.f64, residual](
        p0, atol=1e-10, rtol=1e-12, maxiter=200
    )
    assert_equal(result.success, True)
    assert_almost_equal(result.x.item(0), 2.0, atol=1e-4)
    assert_almost_equal(result.x.item(1), 0.5, atol=1e-4)


def test_least_squares_args_free_overload() raises:
    """The args-free overload should behave identically to the args-taking one.
    """

    def residual[
        dtype: DType
    ](params: nm.NDArray[dtype]) capturing raises -> nm.NDArray[dtype]:
        var xdata = nm.fromstring[dtype]("[0, 1, 2, 3]")
        var ydata = nm.fromstring[dtype]("[1, 3, 5, 7]")
        return params.item(0) * xdata + params.item(1) - ydata

    var p0 = nm.fromstring[sj.f64]("[1, 1]")
    var result = least_squares[sj.f64, residual](p0, atol=1e-12, rtol=1e-12)
    assert_equal(result.success, True)
    assert_almost_equal(result.x.item(0), 2.0, atol=1e-6)
    assert_almost_equal(result.x.item(1), 1.0, atol=1e-6)


def test_least_squares_unsupported_method_raises() raises:
    """An unsupported method should raise, not silently misbehave."""

    def residual[
        dtype: DType
    ](
        params: nm.NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) capturing raises -> nm.NDArray[dtype]:
        return params.copy()

    var p0 = nm.fromstring[sj.f64]("[1, 1]")
    var raised = False
    try:
        _ = least_squares[sj.f64, residual, method="trf"](p0)
    except:
        raised = True
    assert_equal(raised, True)


def test_curve_fit_line() raises:
    """Curve_fit should recover a=2, b=1 for a linear model."""

    def line[
        dtype: DType
    ](
        xdata: nm.NDArray[dtype], params: nm.NDArray[dtype]
    ) capturing raises -> nm.NDArray[dtype]:
        return params.item(0) * xdata + params.item(1)

    var xdata = nm.fromstring[sj.f64]("[0, 1, 2, 3, 4]")
    var ydata = nm.fromstring[sj.f64]("[1, 3, 5, 7, 9]")
    var p0 = nm.fromstring[sj.f64]("[1, 1]")

    var result = curve_fit[sj.f64, line](
        xdata, ydata, p0, atol=1e-12, rtol=1e-12
    )
    assert_equal(result.success, True)
    assert_almost_equal(result.x.item(0), 2.0, atol=1e-6)
    assert_almost_equal(result.x.item(1), 1.0, atol=1e-6)


def test_curve_fit_quadratic() raises:
    """Curve_fit should recover a=1, b=-2, c=1 for y = a*x^2 + b*x + c."""

    def quad[
        dtype: DType
    ](
        xdata: nm.NDArray[dtype], params: nm.NDArray[dtype]
    ) capturing raises -> nm.NDArray[dtype]:
        var a = params.item(0)
        var b = params.item(1)
        var c = params.item(2)
        return a * xdata * xdata + b * xdata + c

    var xdata = nm.fromstring[sj.f64]("[-2, -1, 0, 1, 2, 3]")
    var ydata = nm.fromstring[sj.f64]("[9, 4, 1, 0, 1, 4]")
    var p0 = nm.fromstring[sj.f64]("[1, 1, 1]")

    var result = curve_fit[sj.f64, quad](
        xdata, ydata, p0, atol=1e-12, rtol=1e-12, maxiter=200
    )
    assert_equal(result.success, True)
    assert_almost_equal(result.x.item(0), 1.0, atol=1e-5)
    assert_almost_equal(result.x.item(1), -2.0, atol=1e-5)
    assert_almost_equal(result.x.item(2), 1.0, atol=1e-5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
