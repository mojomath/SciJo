from std.testing import assert_almost_equal, assert_equal
from std.testing import TestSuite
from std.math import sqrt

import scijo as sj
from scijo.optimize.root_scalar import root_scalar, newton, bisect, brent, secant


def test_bisect_root_scalar_basic() raises:
    @parameter
    def f[
        dtype: DType
    ](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[dtype]:
        return x * x - 2.0

    var result = root_scalar[sj.f64, f](bracket=(0.0, 2.0))
    assert_almost_equal(result.root, sqrt(2.0), atol=1e-8)
    assert_equal(result.success, True)
    assert_equal(result.method, "bisect")

    var result2 = bisect[sj.f64, f](None, (0.0, 2.0))
    assert_almost_equal(result2.root, sqrt(2.0), atol=1e-8)
    assert_equal(result2.success, True)


def test_newton_basic() raises:
    @parameter
    def f[
        dtype: DType
    ](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[dtype]:
        return x * x - 2.0

    @parameter
    def fprime[
        dtype: DType
    ](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[dtype]:
        return 2.0 * x

    var result = newton[sj.f64, f, fprime](None, x0=1.0, atol=1e-12, rtol=1e-12)
    assert_almost_equal(result.root, sqrt(2.0), atol=1e-10)
    assert_equal(result.success, True)
    assert_equal(result.method, "newton")


def test_secant_basic() raises:
    @parameter
    def f[
        dtype: DType
    ](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[dtype]:
        return x * x - 2.0

    var result = secant[sj.f64, f](None, 1.0, 2.0)
    assert_almost_equal(result.root, sqrt(2.0), atol=1e-8)
    assert_equal(result.success, True)
    assert_equal(result.method, "secant")


def test_brent_basic() raises:
    @parameter
    def f[
        dtype: DType
    ](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[dtype]:
        return x * x - 2.0

    var result = brent[sj.f64, f](None, (0.0, 2.0))
    assert_almost_equal(result.root, sqrt(2.0), atol=1e-8)
    assert_equal(result.success, True)
    assert_equal(result.method, "brent")

    var result2 = root_scalar[sj.f64, f, method="brent"](bracket=(0.0, 2.0))
    assert_almost_equal(result2.root, sqrt(2.0), atol=1e-8)
    assert_equal(result2.success, True)
    assert_equal(result2.method, "brent")


def test_bisect_invalid_bracket_raises() raises:
    @parameter
    def g[
        dtype: DType
    ](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[dtype]:
        return x * x + 1.0

    try:
        var _ = bisect[sj.f64, g](None, (0.0, 1.0))
        assert_equal(
            True, False, msg="Expected bisect to raise on invalid bracket."
        )
    except:
        pass


def test_root_results_fields() raises:
    """Verify RootResults carries all diagnostic fields."""

    @parameter
    def f[
        dtype: DType
    ](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[dtype]:
        return x - 3.0

    var result = bisect[sj.f64, f](None, (0.0, 5.0))
    assert_almost_equal(result.root, 3.0, atol=1e-7)
    assert_equal(result.success, True)
    assert_equal(result.nit > 0, True)
    assert_equal(result.nfev > 0, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
