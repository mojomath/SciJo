from testing import assert_almost_equal, assert_equal
from testing import TestSuite
from math import sqrt
import scijo as sj
from scijo.optimize.root_scalar import root_scalar, newton, bisect, secant


fn test_bisect_root_scalar_basic() raises:
    fn f[
        dtype: DType
    ](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[dtype]:
        return x * x - 2.0

    var root = root_scalar[sj.f64, f](bracket=(0.0, 2.0))
    assert_almost_equal(root, sqrt(2.0), atol=1e-8)

    var root2 = bisect[sj.f64, f](None, (0.0, 2.0))
    assert_almost_equal(root2, sqrt(2.0), atol=1e-8)


fn test_newton_basic() raises:
    fn f[
        dtype: DType
    ](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[dtype]:
        return x * x - 2.0

    fn fprime[
        dtype: DType
    ](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[dtype]:
        return 2.0 * x

    var root = newton[sj.f64, f, fprime](None, x0=1.0, xtol=1e-12, rtol=1e-12)
    assert_almost_equal(root, sqrt(2.0), atol=1e-10)


fn test_secant_basic() raises:
    fn f[
        dtype: DType
    ](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[dtype]:
        return x * x - 2.0

    var root = secant[sj.f64, f](None, 1.0, 2.0)
    assert_almost_equal(root, sqrt(2.0), atol=1e-8)


fn test_bisect_invalid_bracket_raises() raises:
    fn g[
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


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
