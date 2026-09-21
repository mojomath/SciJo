"""
Tests for multi-dimensional minimization (scijo.optimize.minimize).

To run: `mojo test tests/test_minimize.mojo -I .` from the project root directory.
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
from scijo.optimize import minimize


def test_minimize_quadratic_bowl() raises:
    """F(x, y) = (x - 3)^2 + (y + 1)^2, minimum at (3, -1)."""

    @parameter
    def f[
        dtype: DType
    ](
        x: nm.NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) capturing raises -> Scalar[dtype]:
        var a = x.item(0) - 3.0
        var b = x.item(1) + 1.0
        return a * a + b * b

    var x0 = nm.fromstring[sj.f64]("[0, 0]")
    var result = minimize[sj.f64, f](x0, xatol=1e-8, fatol=1e-10)
    assert_equal(result.success, True)
    assert_almost_equal(result.x.item(0), 3.0, atol=1e-4)
    assert_almost_equal(result.x.item(1), -1.0, atol=1e-4)
    assert_almost_equal(result.fun, 0.0, atol=1e-6)
    assert_equal(result.method, "Nelder-Mead")


def test_minimize_rosenbrock() raises:
    """Classic Rosenbrock banana function, minimum at (1, 1)."""

    @parameter
    def rosenbrock[
        dtype: DType
    ](
        x: nm.NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) capturing raises -> Scalar[dtype]:
        var a = 1.0 - x.item(0)
        var b = x.item(1) - x.item(0) * x.item(0)
        return a * a + 100.0 * b * b

    var x0 = nm.fromstring[sj.f64]("[-1.2, 1.0]")
    var result = minimize[sj.f64, rosenbrock](x0, maxiter=2000)
    assert_equal(result.success, True)
    assert_almost_equal(result.x.item(0), 1.0, atol=1e-3)
    assert_almost_equal(result.x.item(1), 1.0, atol=1e-3)


def test_minimize_with_args() raises:
    """F(x) = a * ||x||^2, with `a` threaded through `args`."""

    @parameter
    def scaled_sphere[
        dtype: DType
    ](
        x: nm.NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) capturing raises -> Scalar[dtype]:
        var a = args.value()[0]
        return a * (x.item(0) * x.item(0) + x.item(1) * x.item(1))

    var x0 = nm.fromstring[sj.f64]("[1, 1]")
    var arglist: List[Scalar[sj.f64]] = [2.0]
    var result = minimize[sj.f64, scaled_sphere](x0, args=arglist^)
    assert_equal(result.success, True)
    assert_almost_equal(result.x.item(0), 0.0, atol=1e-3)
    assert_almost_equal(result.x.item(1), 0.0, atol=1e-3)


def test_minimize_args_free_overload() raises:
    """The args-free overload captures its coefficient from the closure."""
    var scale: Scalar[sj.f64] = 5.0

    @parameter
    def scaled_sphere[
        dtype: DType
    ](x: nm.NDArray[dtype]) capturing raises -> Scalar[dtype]:
        return scale.cast[dtype]() * (
            x.item(0) * x.item(0) + x.item(1) * x.item(1)
        )

    var x0 = nm.fromstring[sj.f64]("[2, -2]")
    var result = minimize[sj.f64, scaled_sphere](x0)
    assert_equal(result.success, True)
    assert_almost_equal(result.x.item(0), 0.0, atol=1e-3)
    assert_almost_equal(result.x.item(1), 0.0, atol=1e-3)


def test_minimize_empty_x0_raises() raises:
    """An empty initial guess should raise, not misbehave."""

    @parameter
    def f[dtype: DType](x: nm.NDArray[dtype]) capturing raises -> Scalar[dtype]:
        return Scalar[dtype](0.0)

    var x0 = nm.zeros[sj.f64](nm.Shape(0))
    var raised = False
    try:
        _ = minimize[sj.f64, f](x0)
    except:
        raised = True
    assert_true(raised, "Should raise error for an empty x0")


def test_minimize_unsupported_method_raises() raises:
    """An unsupported method should raise, not silently do the wrong thing."""

    @parameter
    def f[dtype: DType](x: nm.NDArray[dtype]) capturing raises -> Scalar[dtype]:
        return x.item(0) * x.item(0)

    var x0 = nm.fromstring[sj.f64]("[1]")
    var raised = False
    try:
        _ = minimize[sj.f64, f, method="BFGS"](x0)
    except:
        raised = True
    assert_true(raised, "Should raise error for an unsupported method")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
