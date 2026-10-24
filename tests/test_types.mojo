# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.math import cos, exp, sin
from std.testing import assert_almost_equal, assert_true, TestSuite

# ===----------------------------------------------------------------------=== #
# SciJo
# ===----------------------------------------------------------------------=== #
from scijo import f64
from scijo.types import Complex, Dual, Numeric, Primal, variable, variable2


# ===----------------------------------------------------------------------=== #
# Functions written once against `Numeric`, reused for both value and
# derivative checks below.
# ===----------------------------------------------------------------------=== #


def constant_function[T: Numeric](x: T) -> T:
    """F(x) = 5, f'(x) = 0."""
    return T.constant(5.0)


def linear_function[T: Numeric](x: T) -> T:
    """F(x) = 3x + 2, f'(x) = 3."""
    return 3.0 * x + 2.0


def quadratic_function[T: Numeric](x: T) -> T:
    """F(x) = 2x^2 + 3x + 1, f'(x) = 4x + 3."""
    return 2.0 * x * x + 3.0 * x + 1.0


def cubic_function[T: Numeric](x: T) -> T:
    """F(x) = x^3 - 2x^2 + x - 5, f'(x) = 3x^2 - 4x + 1."""
    return x * x * x - 2.0 * x * x + x - 5.0


def sin_function[T: Numeric](x: T) -> T:
    """F(x) = sin(x), f'(x) = cos(x)."""
    return x.sin()


def cos_function[T: Numeric](x: T) -> T:
    """F(x) = cos(x), f'(x) = -sin(x)."""
    return x.cos()


def exp_function[T: Numeric](x: T) -> T:
    """F(x) = e^x, f'(x) = e^x."""
    return x.exp()


# ===----------------------------------------------------------------------=== #
# Primal: calling a `Numeric` function at `Primal` should match calling the
# same formula directly at `Scalar[f64]`.
# ===----------------------------------------------------------------------=== #


def test_primal_matches_scalar_arithmetic() raises:
    var x = Primal[f64](2.0)
    var result = quadratic_function(x)
    assert_almost_equal(
        result.v, 2.0 * 2.0 * 2.0 + 3.0 * 2.0 + 1.0, msg="Primal arithmetic"
    )


def test_primal_matches_scalar_transcendentals() raises:
    var x = Primal[f64](0.5)
    assert_almost_equal(sin_function(x).v, sin(0.5), msg="Primal sin")
    assert_almost_equal(cos_function(x).v, cos(0.5), msg="Primal cos")
    assert_almost_equal(exp_function(x).v, exp(0.5), msg="Primal exp")


# ===----------------------------------------------------------------------=== #
# Complex: arithmetic and transcendentals should match hand-derived complex
# identities.
# ===----------------------------------------------------------------------=== #


def test_complex_arithmetic() raises:
    var a = Complex[f64](3.0, 4.0)
    var b = Complex[f64](1.0, 2.0)

    var sum = a + b
    assert_almost_equal(sum.re, 4.0, msg="(3+4i)+(1+2i) re")
    assert_almost_equal(sum.im, 6.0, msg="(3+4i)+(1+2i) im")

    var product = a * b
    assert_almost_equal(product.re, -5.0, msg="(3+4i)(1+2i) re = 3-8")
    assert_almost_equal(product.im, 10.0, msg="(3+4i)(1+2i) im = 6+4")

    var quotient = a / b
    assert_almost_equal(quotient.re, 2.2, msg="(3+4i)/(1+2i) re")
    assert_almost_equal(quotient.im, -0.4, msg="(3+4i)/(1+2i) im")


def test_complex_matches_quadratic() raises:
    var x = Complex[f64](2.0, -1.0)
    var result = quadratic_function(x)
    # F(x) = 2x^2 + 3x + 1 at x = 2-i: x^2 = 3-4i, 2x^2 = 6-8i,
    # 3x = 6-3i, so F(x) = (6-8i) + (6-3i) + 1 = 13 - 11i.
    assert_almost_equal(result.re, 13.0, msg="Complex quadratic re")
    assert_almost_equal(result.im, -11.0, msg="Complex quadratic im")


def test_complex_exp_euler_identity() raises:
    # e^(i*pi) = -1, the imaginary axis of exp's known closed form.
    var pi = Complex[f64](0.0, 3.141592653589793)
    var result = pi.exp()
    assert_almost_equal(result.re, -1.0, msg="e^(i*pi) re")
    assert_almost_equal(result.im, 0.0, msg="e^(i*pi) im")


def test_complex_sqrt_matches_square() raises:
    var z = Complex[f64](3.0, 4.0)
    var root = z.sqrt()
    var squared = root * root
    assert_almost_equal(squared.re, z.re, msg="sqrt(z)^2 re == z")
    assert_almost_equal(squared.im, z.im, msg="sqrt(z)^2 im == z")


def test_complex_log_matches_exp() raises:
    var z = Complex[f64](3.0, 4.0)
    var roundtrip = z.log().exp()
    assert_almost_equal(roundtrip.re, z.re, msg="exp(log(z)) re == z")
    assert_almost_equal(roundtrip.im, z.im, msg="exp(log(z)) im == z")


# ===----------------------------------------------------------------------=== #
# Dual: the derivative component should match the closed-form derivative
# exactly (no finite-difference tolerance needed).
# ===----------------------------------------------------------------------=== #


def test_dual_constant_derivative() raises:
    var x = variable(2.0)
    var result = constant_function(x)
    assert_almost_equal(result.value.v, 5.0, msg="F(2) = 5")
    assert_almost_equal(result.deriv.v, 0.0, msg="f'(x) = 0")


def test_dual_linear_derivative() raises:
    var x = variable(1.5)
    var result = linear_function(x)
    assert_almost_equal(result.value.v, 3.0 * 1.5 + 2.0, msg="F(1.5)")
    assert_almost_equal(result.deriv.v, 3.0, msg="f'(x) = 3")


def test_dual_quadratic_derivative() raises:
    var x0 = 2.0
    var x = variable(x0)
    var result = quadratic_function(x)
    assert_almost_equal(
        result.value.v, 2.0 * x0 * x0 + 3.0 * x0 + 1.0, msg="F(2)"
    )
    assert_almost_equal(result.deriv.v, 4.0 * x0 + 3.0, msg="f'(x) = 4x + 3")


def test_dual_cubic_derivative() raises:
    for x0 in [0.0, 1.0, -1.0, 2.5]:
        var x = variable(x0)
        var result = cubic_function(x)
        var expected_df = 3.0 * x0 * x0 - 4.0 * x0 + 1.0
        assert_almost_equal(
            result.deriv.v, expected_df, msg="cubic derivative at " + String(x0)
        )


def test_dual_trigonometric_derivatives() raises:
    var x_sin = variable(0.5)
    var result_sin = sin_function(x_sin)
    assert_almost_equal(result_sin.value.v, sin(0.5), msg="sin(0.5)")
    assert_almost_equal(
        result_sin.deriv.v, cos(0.5), msg="d/dx sin(x) = cos(x)"
    )

    var x_cos = variable(1.0)
    var result_cos = cos_function(x_cos)
    assert_almost_equal(result_cos.value.v, cos(1.0), msg="cos(1.0)")
    assert_almost_equal(
        result_cos.deriv.v, -sin(1.0), msg="d/dx cos(x) = -sin(x)"
    )


def test_dual_exponential_derivative() raises:
    var x = variable(1.0)
    var result = exp_function(x)
    assert_almost_equal(result.value.v, exp(1.0), msg="exp(1)")
    assert_almost_equal(result.deriv.v, exp(1.0), msg="d/dx exp(x) = exp(x)")


# ===----------------------------------------------------------------------=== #
# Nesting: `variable2` should produce both the first and second derivative
# from one call, with no new code written for the second order.
# ===----------------------------------------------------------------------=== #


def test_nested_dual_second_derivative() raises:
    """F(x) = x^3: f'(x) = 3x^2, f''(x) = 6x."""

    def cube[T: Numeric](x: T) -> T:
        return x * x * x

    var x0 = 2.0
    var x = variable2(x0)
    var result = cube(x)

    assert_almost_equal(result.value.value.v, x0 * x0 * x0, msg="F(2) = 8")
    assert_almost_equal(
        result.value.deriv.v, 3.0 * x0 * x0, msg="f'(2) = 12, via inner Dual"
    )
    assert_almost_equal(
        result.deriv.value.v, 3.0 * x0 * x0, msg="f'(2) = 12, via outer Dual"
    )
    assert_almost_equal(result.deriv.deriv.v, 6.0 * x0, msg="f''(2) = 12")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
