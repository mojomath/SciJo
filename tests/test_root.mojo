"""
Tests for multi-dimensional root-finding (scijo.optimize.root).

To run: `mojo test tests/test_root.mojo -I .` from the project root directory.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
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
from scijo.linalg import matmul
from scijo.optimize import root


def test_root_linear_system() raises:
    """Solving a linear system: f(x) = Ax - b, root at x = A^-1 b."""

    @parameter
    def f[
        dtype: DType
    ](
        x: nm.NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) capturing raises -> nm.NDArray[dtype]:
        var A = nm.fromstring[dtype]("[[3, 1], [1, 2]]")
        var b = nm.fromstring[dtype]("[9, 8]")
        return matmul(A, x) - b

    var x0 = nm.fromstring[sj.f64]("[0, 0]")
    var result = root[sj.f64, f](x0, atol=1e-10, rtol=1e-12)
    assert_equal(result.success, True)
    assert_almost_equal(result.x.item(0), 2.0, atol=1e-8)
    assert_almost_equal(result.x.item(1), 3.0, atol=1e-8)
    assert_equal(result.method, "newton")


def test_root_nonlinear_system() raises:
    """Classic 2-eqn nonlinear system: x^2 + y^2 = 4, x*y = 1."""

    @parameter
    def f[
        dtype: DType
    ](
        x: nm.NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) capturing raises -> nm.NDArray[dtype]:
        var out = nm.zeros[dtype](nm.Shape(2))
        out.store(0, x.item(0) * x.item(0) + x.item(1) * x.item(1) - 4.0)
        out.store(1, x.item(0) * x.item(1) - 1.0)
        return out^

    var x0 = nm.fromstring[sj.f64]("[1.5, 1.0]")
    var result = root[sj.f64, f](x0, atol=1e-10, rtol=1e-12)
    assert_equal(result.success, True)
    # Check the residual, rather than a specific root (there are 4 by symmetry).
    assert_almost_equal(result.fun.item(0), 0.0, atol=1e-6)
    assert_almost_equal(result.fun.item(1), 0.0, atol=1e-6)


def test_root_with_analytic_jacobian() raises:
    """Same linear system, but with an analytic Jacobian supplied."""

    @parameter
    def f[
        dtype: DType
    ](
        x: nm.NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) capturing raises -> nm.NDArray[dtype]:
        var A = nm.fromstring[dtype]("[[3, 1], [1, 2]]")
        var b = nm.fromstring[dtype]("[9, 8]")
        return matmul(A, x) - b

    @parameter
    def jac[
        dtype: DType
    ](
        x: nm.NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) capturing raises -> nm.NDArray[dtype]:
        return nm.fromstring[dtype]("[[3, 1], [1, 2]]")

    var x0 = nm.fromstring[sj.f64]("[0, 0]")
    var result = root[sj.f64, f, jac](x0, atol=1e-10, rtol=1e-12)
    assert_equal(result.success, True)
    assert_almost_equal(result.x.item(0), 2.0, atol=1e-8)
    assert_almost_equal(result.x.item(1), 3.0, atol=1e-8)
    # An analytic Jacobian converges in a single Newton step on a linear system.
    assert_equal(result.nit, 1)


def test_root_shape_mismatch_raises() raises:
    """F(x) with a different length than x should raise, not misbehave."""

    @parameter
    def f[
        dtype: DType
    ](
        x: nm.NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) capturing raises -> nm.NDArray[dtype]:
        return nm.zeros[dtype](nm.Shape(3))

    var x0 = nm.fromstring[sj.f64]("[0, 0]")
    var raised = False
    try:
        _ = root[sj.f64, f](x0)
    except:
        raised = True
    assert_equal(raised, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
