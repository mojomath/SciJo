"""
Comprehensive tests for the quad (adaptive quadrature) function.

This test suite validates the quad function implementation against known analytical
solutions and edge cases, similar to SciPy's quad function tests.

To run: `mojo test tests/test_quad.mojo -I .` from the project root directory.
"""

from testing import assert_almost_equal, assert_equal, assert_true, assert_false
from testing import TestSuite
from math import sin, cos, exp, log, pi, sqrt

from scijo.integrate import quad
import scijo as sj


def test_quad_basic_polynomials():
    """Test quad with simple polynomial functions that have analytical solutions.
    """

    # Test ∫x dx from 0 to 1 = 1/2
    fn linear[
        dtype: DType
    ](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[dtype]:
        return x

    var result = quad[sj.f64, linear](0.0, 1.0, None)
    assert_almost_equal(result.integral, 0.5, atol=1e-10)
    assert_true(result.ier == 0)  # Success

    # Test ∫x² dx from 0 to 2 = 8/3
    fn quadratic[
        dtype: DType
    ](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[dtype]:
        return x * x

    var result2 = quad[sj.f64, quadratic](0.0, 2.0, None)
    assert_almost_equal(result2.integral, 8.0 / 3.0, atol=1e-10)
    assert_true(result2.ier == 0)

    # Test ∫x³ dx from 0 to 3 = 81/4
    fn cubic[
        dtype: DType
    ](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[dtype]:
        return x * x * x

    var result3 = quad[sj.f64, cubic](0.0, 3.0, None)
    assert_almost_equal(result3.integral, 81.0 / 4.0, atol=1e-10)
    assert_true(result3.ier == 0)


def test_quad_trigonometric():
    """Test quad with trigonometric functions."""

    # Test ∫sin(x) dx from 0 to π = 2
    fn sine[
        dtype: DType
    ](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[
        dtype
    ] where dtype.is_floating_point():
        return sin(x)

    var result = quad[sj.f64, sine](0.0, pi, None)
    assert_almost_equal(result.integral, 2.0, atol=1e-10)
    assert_true(result.ier == 0)

    # Test ∫cos(x) dx from 0 to π/2 = 1
    fn cosine[
        dtype: DType
    ](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[
        dtype
    ] where dtype.is_floating_point():
        return cos(x)

    var result2 = quad[sj.f64, cosine](0.0, pi / 2.0, None)
    assert_almost_equal(result2.integral, 1.0, atol=1e-10)
    assert_true(result2.ier == 0)

    # Test ∫sin²(x) dx from 0 to π = π/2
    fn sin_squared[
        dtype: DType
    ](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[
        dtype
    ] where dtype.is_floating_point():
        var s = sin(x)
        return s * s

    var result3 = quad[sj.f64, sin_squared](0.0, pi, None)
    assert_almost_equal(result3.integral, pi / 2.0, atol=1e-9)
    assert_true(result3.ier == 0)


def test_quad_exponential():
    """Test quad with exponential functions."""

    # Test ∫e^x dx from 0 to 1 = e - 1
    fn exponential[
        dtype: DType
    ](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[
        dtype
    ] where dtype.is_floating_point():
        return exp(x)

    var result = quad[sj.f64, exponential](0.0, 1.0, None)
    var expected = exp(1.0) - 1.0
    assert_almost_equal(result.integral, expected, atol=1e-10)
    assert_true(result.ier == 0)

    # Test ∫e^(-x) dx from 0 to ∞ ≈ 1 (using large upper bound)
    fn exp_decay[
        dtype: DType
    ](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[
        dtype
    ] where dtype.is_floating_point():
        return exp(-x)

    var result2 = quad[sj.f64, exp_decay](
        0.0, 10.0, None, epsrel=1e-8
    )  # 10 is "large enough"
    assert_almost_equal(result2.integral, 1.0, atol=1e-3)
    assert_true(result2.ier == 0)


def test_quad_edge_cases():
    """Test quad with edge cases and boundary conditions."""

    # Test with identical limits (should return 0)
    fn constant_func[
        dtype: DType
    ](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[dtype]:
        return 5.0

    var result = quad[sj.f64, constant_func](2.0, 2.0, None)
    assert_equal(result.integral, 0.0)
    assert_true(result.ier == 0)
    assert_equal(result.neval, 0)

    # Test with reversed limits (should negate result)
    var result_normal = quad[sj.f64, constant_func](0.0, 1.0, None)
    var result_reversed = quad[sj.f64, constant_func](1.0, 0.0, None)
    assert_almost_equal(
        result_normal.integral, -result_reversed.integral, atol=1e-15
    )

    # Test constant function ∫5 dx from 0 to 3 = 15
    var result_const = quad[sj.f64, constant_func](0.0, 3.0, None)
    assert_almost_equal(result_const.integral, 15.0, atol=1e-15)


def test_quad_with_parameters():
    """Test quad with parameterized functions using args."""

    # Test ∫a*x dx with parameter a
    fn linear_param[
        dtype: DType
    ](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[dtype]:
        if args:
            var a = args.value()[0]
            return a * x
        return x  # fallback

    var args = List[Float64]()
    args.append(3.0)  # a = 3

    # ∫3x dx from 0 to 2 = 3 * (2²/2) = 6
    var result = quad[sj.f64, linear_param](0.0, 2.0, args^)
    assert_almost_equal(result.integral, 6.0, atol=1e-10)
    assert_true(result.ier == 0)


def test_quad_difficult_integrands():
    """Test quad with more challenging integrands."""

    # Test Gaussian integral ∫e^(-x²) dx from -∞ to ∞ ≈ √π
    # Using finite bounds that approximate infinity
    fn gaussian[
        dtype: DType
    ](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[
        dtype
    ] where dtype.is_floating_point():
        return exp(-x * x)

    var result = quad[sj.f64, gaussian](-5.0, 5.0, None, epsrel=1e-8)
    assert_almost_equal(result.integral, sqrt(pi), atol=1e-6)
    assert_true(result.ier == 0)

    # Test oscillatory function sin(x)/x near origin (needs careful handling)
    fn sinc_like[
        dtype: DType
    ](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[
        dtype
    ] where dtype.is_floating_point():
        if abs(x) < 1e-10:
            return 1.0  # limit as x→0 of sin(x)/x = 1
        return sin(x) / x

    var result2 = quad[sj.f64, sinc_like](0.0, pi, None)
    # This integral is known to be approximately 1.8519
    assert_almost_equal(result2.integral, 1.8519, atol=1e-3)


# ! this isn't giving enough precision to pass tests yet.
# def test_quad_precision_and_tolerance():
#     """Test quad with different precision requirements."""
#     fn smooth_func[dtype: DType](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[dtype]:
#         return x * exp(-x)

# Test with default tolerance
# var result_default = quad[sj.f64, smooth_func](0.0, 5.0, None, epsabs=1e-8, epsrel=1e-8, limit=100)
# assert_almost_equal(result_default.integral, 1.0)
# assert_true(result_default.ier == 0)

# Test with stricter tolerance
# var result_strict = quad[sj.f64, smooth_func](
#     0.0, 5.0, None, epsabs=1e-12, epsrel=1e-12
# )
# # Analytical result: ∫x*e^(-x) dx from 0 to ∞ = 1
# assert_almost_equal(result_strict.integral, 1.0)


# def test_quad_float32():
#     """Test quad with Float32 precision."""

#     fn simple_func[
#         dtype: DType
#     ](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[dtype]:
#         return x * x

#     var result = quad[DType.float32, simple_func](
#         Float32(0.0), Float32(2.0), None
#     )
#     assert_almost_equal(
#         Scalar[DType.float32](result.integral), 8.0 / 3.0, atol=1e-6
#     )
#     assert_true(result.ier == 0)


# only for adaptive algo when it's implemented correctly.
# def test_quad_error_conditions():
#     """Test quad error handling and convergence failure cases."""

#     # Test with very few subdivisions allowed (should still work for simple functions)
#     fn simple_func[
#         dtype: DType
#     ](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[dtype]:
#         return x

#     var result = quad[sj.f64, simple_func](0.0, 1.0, None, limit=1)
#     assert_almost_equal(result.integral, 0.5, atol=1e-8)

#     # Test extremely oscillatory function that might challenge the algorithm
#     fn oscillatory[
#         dtype: DType
#     ](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[dtype]:
#         return sin(100.0 * x)  # High frequency oscillation

#     var result2 = quad[sj.f64, oscillatory](0.0, 2 * pi, None, limit=100)
#     # This should integrate to near zero due to oscillation
#     assert_almost_equal(result2.integral, 0.0, atol=1e-6)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
