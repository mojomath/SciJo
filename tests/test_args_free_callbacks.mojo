"""
Tests for the args-free callback overloads: `func(x)` instead of
`func(x, args)`, added so callbacks that don't need extra parameters
don't have to carry an unused `args` argument.

To run: `mojo test tests/test_args_free_callbacks.mojo -I .` from the
project root directory.
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
from scijo.differentiate import (
    derivative,
    hessian,
    jacobian,
)
from scijo.integrate import quad
from scijo.linalg import matmul
from scijo.optimize import (
    bisect,
    brent,
    minimize_scalar,
    newton,
    root,
    root_scalar,
    secant,
)


def test_derivative_args_free() raises:
    @parameter
    def f[dtype: DType](x: Scalar[dtype]) capturing -> Scalar[dtype]:
        return x * x

    var result = derivative[sj.f64, f](1.0)
    assert_almost_equal(result.df, 2.0, atol=1e-6)


def test_jacobian_args_free() raises:
    @parameter
    def f[
        dtype: DType
    ](x: nm.NDArray[dtype]) capturing raises -> nm.NDArray[dtype]:
        return x * x

    var x = nm.fromstring[sj.f64]("[1.0, 2.0]")
    var J = jacobian[sj.f64, f](x)
    assert_almost_equal(J.item(0, 0), 2.0, atol=1e-4)
    assert_almost_equal(J.item(0, 1), 0.0, atol=1e-4)
    assert_almost_equal(J.item(1, 0), 0.0, atol=1e-4)
    assert_almost_equal(J.item(1, 1), 4.0, atol=1e-4)


def test_hessian_args_free() raises:
    @parameter
    def f[dtype: DType](x: nm.NDArray[dtype]) capturing raises -> Scalar[dtype]:
        return x.item(0) * x.item(0) + x.item(1) * x.item(1)

    var x = nm.fromstring[sj.f64]("[1.0, 2.0]")
    var H = hessian[sj.f64, f](x)
    assert_almost_equal(H.item(0, 0), 2.0, atol=1e-3)
    assert_almost_equal(H.item(1, 1), 2.0, atol=1e-3)
    assert_almost_equal(H.item(0, 1), 0.0, atol=1e-3)


def test_quad_args_free() raises:
    @parameter
    def f[dtype: DType](x: Scalar[dtype]) capturing -> Scalar[dtype]:
        return x

    var result = quad[sj.f64, f](a=0.0, b=1.0)
    assert_almost_equal(result.integral, 0.5, atol=1e-6)


def test_root_scalar_family_args_free() raises:
    @parameter
    def f[dtype: DType](x: Scalar[dtype]) capturing -> Scalar[dtype]:
        return x * x - 2.0

    @parameter
    def fprime[dtype: DType](x: Scalar[dtype]) capturing -> Scalar[dtype]:
        return 2.0 * x

    var r_bisect = root_scalar[sj.f64, f, method="bisect"](bracket=(0.0, 2.0))
    assert_almost_equal(r_bisect.root, sqrt(2.0), atol=1e-6)

    var r_brent = root_scalar[sj.f64, f, method="brent"](bracket=(0.0, 2.0))
    assert_almost_equal(r_brent.root, sqrt(2.0), atol=1e-8)

    var r_secant = root_scalar[sj.f64, f, method="secant"](x0=1.0, x1=2.0)
    assert_almost_equal(r_secant.root, sqrt(2.0), atol=1e-6)

    var r_newton = root_scalar[sj.f64, f, fprime, method="newton"](x0=1.0)
    assert_almost_equal(r_newton.root, sqrt(2.0), atol=1e-8)

    assert_almost_equal(
        bisect[sj.f64, f]((0.0, 2.0)).root, sqrt(2.0), atol=1e-6
    )
    assert_almost_equal(brent[sj.f64, f]((0.0, 2.0)).root, sqrt(2.0), atol=1e-8)
    assert_almost_equal(secant[sj.f64, f](1.0, 2.0).root, sqrt(2.0), atol=1e-6)
    assert_almost_equal(
        newton[sj.f64, f, fprime](x0=1.0).root, sqrt(2.0), atol=1e-8
    )


def test_minimize_scalar_args_free() raises:
    @parameter
    def objective[dtype: DType](x: Scalar[dtype]) capturing -> Scalar[dtype]:
        return (x - 2.0) * (x - 2.0) + 1.0

    var result = minimize_scalar[sj.f64, objective, method="Brent"](
        bracket=(0.0, 4.0)
    )
    assert_almost_equal(result.x, 2.0, atol=1e-6)
    assert_almost_equal(result.fun, 1.0, atol=1e-6)


def test_root_args_free() raises:
    @parameter
    def f[
        dtype: DType
    ](x: nm.NDArray[dtype]) capturing raises -> nm.NDArray[dtype]:
        var A = nm.fromstring[dtype]("[[3, 1], [1, 2]]")
        var b = nm.fromstring[dtype]("[9, 8]")
        return matmul(A, x) - b

    @parameter
    def jac[
        dtype: DType
    ](x: nm.NDArray[dtype]) capturing raises -> nm.NDArray[dtype]:
        return nm.fromstring[dtype]("[[3, 1], [1, 2]]")

    var x0 = nm.fromstring[sj.f64]("[0, 0]")

    var r1 = root[sj.f64, f](x0)
    assert_equal(r1.success, True)
    assert_almost_equal(r1.x.item(0), 2.0, atol=1e-8)
    assert_almost_equal(r1.x.item(1), 3.0, atol=1e-8)

    var r2 = root[sj.f64, f, jac](x0)
    assert_equal(r2.success, True)
    assert_almost_equal(r2.x.item(0), 2.0, atol=1e-8)
    assert_almost_equal(r2.x.item(1), 3.0, atol=1e-8)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
