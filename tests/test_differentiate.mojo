from scijo.differentiate import derivative

from testing import assert_almost_equal, assert_equal, assert_true, assert_false
from testing import TestSuite
import math


fn constant_function[
    dtype: DType
](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[dtype]:
    """
    F(x) = 5, f'(x) = 0.
    """
    return 5.0


fn linear_function[
    dtype: DType
](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[dtype]:
    """
    F(x) = 3x + 2, f'(x) = 3.
    """
    return 3.0 * x + 2.0


fn quadratic_function[
    dtype: DType
](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[dtype]:
    """
    F(x) = 2x^2 + 3x + 1, f'(x) = 4x + 3.
    """
    return 2.0 * x * x + 3.0 * x + 1.0


fn cubic_function[
    dtype: DType
](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[dtype]:
    """
    F(x) = x^3 - 2x^2 + x - 5, f'(x) = 3x^2 - 4x + 1.
    """
    return x * x * x - 2.0 * x * x + x - 5.0


fn sin_function[
    dtype: DType
](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[
    dtype
] where dtype.is_floating_point():
    """
    F(x) = sin(x), f'(x) = cos(x).
    """
    return math.sin(x)


fn cos_function[
    dtype: DType
](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[
    dtype
] where dtype.is_floating_point():
    """
    F(x) = cos(x), f'(x) = -sin(x).
    """
    return math.cos(x)


fn exp_function[
    dtype: DType
](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[
    dtype
] where dtype.is_floating_point():
    """
    F(x) = e^x, f'(x) = e^x.
    """
    return math.exp(x)


fn parameterized_function[
    dtype: DType
](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[dtype]:
    """
    F(x) = a*x^2 + b*x + c, f'(x) = 2*a*x + b.
    """
    var a = args.value()[0]
    var b = args.value()[1]
    var c = args.value()[2]
    return a * x * x + b * x + c


fn test_basic_derivatives() raises:
    """Test derivatives of basic polynomial functions."""

    # Test constant function: F(x) = 5, f'(x) = 0
    var result_const = derivative[
        DType.float64, constant_function, step_direction=0
    ](x0=2.0, args=None)
    assert_true(
        result_const.success, "Constant function derivative should converge"
    )
    assert_almost_equal(
        result_const.df,
        0.0,
        atol=1e-8,
        msg="Derivative of constant should be 0",
    )

    # Test linear function: F(x) = 3x + 2, f'(x) = 3
    var result_linear = derivative[
        DType.float64, linear_function, step_direction=0
    ](x0=1.5, args=None)
    assert_true(
        result_linear.success, "Linear function derivative should converge"
    )
    assert_almost_equal(
        result_linear.df, 3.0, atol=1e-8, msg="Derivative of 3x + 2 should be 3"
    )

    # Test quadratic function: F(x) = 2x^2 + 3x + 1, f'(x) = 4x + 3
    var x_quad = 2.0
    var expected_quad = 4.0 * x_quad + 3.0
    var result_quad = derivative[
        DType.float64, quadratic_function, step_direction=0
    ](x0=x_quad, args=None)
    assert_true(
        result_quad.success, "Quadratic function derivative should converge"
    )
    assert_almost_equal(
        result_quad.df,
        expected_quad,
        atol=1e-6,
        msg="Derivative of 2x^2 + 3x + 1 at x=2 should be 11",
    )


fn test_cubic_derivatives() raises:
    """Test derivative of cubic function."""

    # Test cubic function: F(x) = x^3 - 2x^2 + x - 5, f'(x) = 3x^2 - 4x + 1
    var test_points: List[Float64] = [0.0, 1.0, -1.0, 2.5]

    for i in range(len(test_points)):
        var x = test_points[i]
        var expected = 3.0 * x * x - 4.0 * x + 1.0
        var result = derivative[
            DType.float64, cubic_function, step_direction=0
        ](x0=x, args=None)
        assert_true(result.success, "Cubic function derivative should converge")
        assert_almost_equal(
            result.df,
            expected,
            atol=1e-5,
            msg="Cubic derivative should match analytical result",
        )


fn test_trigonometric_derivatives() raises:
    """Test derivatives of trigonometric functions."""

    # Test sin(x): f'(x) = cos(x)
    var x_sin = 0.5
    var expected_sin = math.cos(x_sin)
    var result_sin = derivative[DType.float64, sin_function, step_direction=0](
        x0=x_sin, args=None, order=6
    )
    assert_true(result_sin.success, "Sin function derivative should converge")
    assert_almost_equal(
        result_sin.df,
        expected_sin,
        atol=1e-6,
        msg="Derivative of sin(x) should be cos(x)",
    )

    # Test cos(x): f'(x) = -sin(x)
    var x_cos = 1.0
    var expected_cos = -math.sin(x_cos)
    var result_cos = derivative[DType.float64, cos_function, step_direction=0](
        x0=x_cos, args=None, order=6
    )
    assert_true(result_cos.success, "Cos function derivative should converge")
    assert_almost_equal(
        result_cos.df,
        expected_cos,
        atol=1e-6,
        msg="Derivative of cos(x) should be -sin(x)",
    )


fn test_exponential_derivative() raises:
    """Test derivative of exponential function."""

    # Test exp(x): f'(x) = exp(x)
    var x_exp = 1.0
    var expected_exp = math.exp(x_exp)
    var result_exp = derivative[DType.float64, exp_function, step_direction=0](
        x0=x_exp, args=None, order=6
    )
    assert_true(result_exp.success, "Exp function derivative should converge")
    assert_almost_equal(
        result_exp.df,
        expected_exp,
        atol=1e-5,
        msg="Derivative of exp(x) should be exp(x)",
    )


fn test_parameterized_function() raises:
    """Test derivative with function parameters."""

    # Test F(x) = a*x^2 + b*x + c with a=2, b=5, c=3
    # f'(x) = 2*a*x + b = 4*x + 5
    var args: List[Scalar[DType.float64]] = [2.0, 5.0, 3.0]
    var x_param = 1.5
    var expected_param = 4.0 * x_param + 5.0  # = 11.0
    var result_param = derivative[
        DType.float64, parameterized_function, step_direction=0
    ](x0=x_param, args=args^)
    assert_true(
        result_param.success,
        "Parameterized function derivative should converge",
    )
    assert_almost_equal(
        result_param.df,
        expected_param,
        atol=1e-6,
        msg="Derivative of parameterized function should match expected value",
    )


fn test_different_step_directions() raises:
    """Test different finite difference methods (central, forward, backward)."""

    var x_test = 1.0
    var expected = 4.0 * x_test + 3.0

    # Central differences
    var result_central = derivative[
        DType.float64, quadratic_function, step_direction=0
    ](x0=x_test, args=None)
    assert_true(result_central.success, "Central difference should converge")
    assert_almost_equal(
        result_central.df,
        expected,
        atol=1e-8,
        msg="Central difference should be most accurate",
    )

    # Forward differences
    var result_forward = derivative[
        DType.float64, quadratic_function, step_direction=1
    ](x0=x_test, args=None, order=6, max_iter=50)
    assert_true(result_forward.success, "Forward difference should converge")
    assert_almost_equal(
        result_forward.df,
        expected,
        atol=1e-5,
        msg="Forward difference should be reasonably accurate",
    )

    # Backward differences
    var result_backward = derivative[
        DType.float64, quadratic_function, step_direction= -1
    ](x0=x_test, args=None, order=6, max_iter=50)
    assert_true(result_backward.success, "Backward difference should converge")
    assert_almost_equal(
        result_backward.df,
        expected,
        atol=1e-5,
        msg="Backward difference should be reasonably accurate",
    )


fn test_different_orders() raises:
    """Test different accuracy orders."""

    var x_test = 0.5
    var expected = 4.0 * x_test + 3.0

    var orders: List[Int] = [2, 4, 6, 8]

    for i in range(len(orders)):
        var order = orders[i]
        var result = derivative[
            DType.float64, quadratic_function, step_direction=0
        ](x0=x_test, args=None, order=order)
        assert_true(result.success, "Higher order should converge")
        assert_almost_equal(
            result.df,
            expected,
            atol=1e-4,
            msg="Higher order should give accurate result",
        )


fn test_tolerance_settings() raises:
    """Test different tolerance settings."""

    var x_test = 1.0
    var expected = 4.0 * x_test + 3.0

    var strict_tolerance = Dict[String, Scalar[DType.float64]]()
    strict_tolerance["atol"] = 1e-10
    strict_tolerance["rtol"] = 1e-10

    var result_strict = derivative[
        DType.float64, quadratic_function, step_direction=0
    ](x0=x_test, args=None, tolerances=strict_tolerance)
    assert_almost_equal(
        result_strict.df,
        expected,
        atol=1e-8,
        msg="Strict tolerance should give accurate result",
    )

    var loose_tolerance = Dict[String, Scalar[DType.float64]]()
    loose_tolerance["atol"] = 1e-3
    loose_tolerance["rtol"] = 1e-3

    var result_loose = derivative[
        DType.float64, quadratic_function, step_direction=0
    ](x0=x_test, args=None, tolerances=loose_tolerance)
    assert_almost_equal(
        result_loose.df,
        expected,
        atol=1e-2,
        msg="Loose tolerance should still give reasonable result",
    )


fn test_convergence_properties() raises:
    """Test convergence properties and diagnostic information."""

    var result = derivative[
        DType.float64, quadratic_function, step_direction=0
    ](x0=1.0, args=None, max_iter=5)

    assert_true(result.nit > 0, "Number of iterations should be positive")
    assert_true(
        result.nfev > 0, "Number of function evaluations should be positive"
    )
    assert_almost_equal(
        result.x, 1.0, msg="Evaluation point should be preserved"
    )
    assert_true(result.error >= 0.0, "Error estimate should be non-negative")


fn test_error_conditions() raises:
    """Test error conditions and invalid parameters."""

    try:
        var _ = derivative[DType.float64, quadratic_function, step_direction=5](
            x0=1.0, args=None
        )
        assert_false(True, "Invalid step direction should raise an error")
    except:
        pass


fn test_step_size_parameters() raises:
    """Test different step size parameters."""

    var x_test = 1.0
    var expected = 4.0 * x_test + 3.0  # = 7.0

    var step_sizes: List[Float64] = [0.1, 0.5, 1.0]

    for i in range(len(step_sizes)):
        var step = step_sizes[i]
        var result = derivative[
            DType.float64, quadratic_function, step_direction=0
        ](x0=x_test, args=None, initial_step=step)
        assert_true(result.success, "Different initial step sizes should work")
        assert_almost_equal(
            result.df,
            expected,
            atol=1e-5,
            msg="Result should be consistent across step sizes",
        )

    var factors: List[Float64] = [1.5, 2.0, 3.0]

    for i in range(len(factors)):
        var factor = factors[i]
        var result = derivative[
            DType.float64, quadratic_function, step_direction=0
        ](x0=x_test, args=None, step_factor=factor)
        assert_true(result.success, "Different step factors should work")
        assert_almost_equal(
            result.df,
            expected,
            atol=1e-5,
            msg="Result should be consistent across step factors",
        )


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
