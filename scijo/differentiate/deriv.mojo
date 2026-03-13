# ===----------------------------------------------------------------------=== #
# Scijo: Differentiate - Derivative
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Differentiate Module - Numerical Differentiation (scijo.differentiate.derivative)

Numerical differentiation using finite difference methods. Provides functions to
compute first-order derivatives of scalar functions using central, forward, and
backward finite difference schemes with adaptive step sizing.

References:
    - SciPy derivative documentation.
    - Wikipedia: Finite difference coefficient
      https://en.wikipedia.org/wiki/Finite_difference_coefficient
"""


from .utility import (
    DiffResult,
    generate_central_finite_difference_table,
    generate_forward_finite_difference_table,
    generate_backward_finite_difference_table,
)


# ===----------------------------------------------------------------------=== #
# Derivative
# ===----------------------------------------------------------------------=== #


fn derivative[
    dtype: DType,
    func: fn[dtype: DType](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) -> Scalar[dtype],
    *,
    step_direction: Int = 0,
](
    x0: Scalar[dtype],
    args: Optional[List[Scalar[dtype]]] = None,
    tolerances: Dict[String, Scalar[dtype]] = {"atol": 1e-6, "rtol": 1e-6},
    max_iter: Int = 10,
    order: Int = 8,
    initial_step: Scalar[dtype] = 0.5,
    step_factor: Scalar[dtype] = 2.0,
) raises -> DiffResult[dtype] where dtype.is_floating_point():
    """Computes the first derivative of a scalar function using finite differences.

    Provides a unified interface for computing first-order derivatives using
    central, forward, or backward finite difference methods with adaptive step
    size reduction for improved accuracy.

    Parameters:
        dtype: The floating-point data type.
        func: Function to differentiate with signature fn(x, args) -> Scalar[dtype].
        step_direction: Direction of finite difference
            (central=0, forward=1, backward=-1). Keyword-only.

    Args:
        x0: Point at which to evaluate the derivative.
        args: Optional arguments to pass to the function.
        tolerances: Convergence tolerances with "atol" (absolute) and "rtol" (relative) keys.
        max_iter: Maximum number of iterations.
        order: Accuracy order for finite differences.
            Central: {2, 4, 6, 8}. Forward/Backward: {1, 2, 3, 4, 5, 6}.
        initial_step: Initial step size for finite differences.
        step_factor: Factor by which to reduce step size in each iteration (must be > 1).

    Returns:
        DiffResult[dtype] containing the derivative, convergence status,
        estimated error, iteration count, and function evaluation count.

    Raises:
        Error: If step_direction is not in {-1, 0, 1}.
        Error: If the specified order is not supported for the chosen method.

    Examples:
        ```mojo
        from scijo.differentiate import derivative
        from scijo.prelude import *

        fn f[dtype: DType](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[dtype]:
            return x * x

        var res = derivative[f64, f](1.0)
        ```
    """

    @parameter
    if step_direction == 0:
        comptime first_order_coefficients = generate_central_finite_difference_table[
            dtype
        ]()
        return _derivative_central_difference[dtype, func](
            x0,
            args,
            tolerances=tolerances,
            order=order,
            initial_step=initial_step,
            step_factor=step_factor,
            max_iter=max_iter,
        )
    elif step_direction == 1:
        comptime first_order_coefficients = generate_forward_finite_difference_table[
            dtype
        ]()
        return _derivative_forward_difference[dtype, func](
            x0,
            args,
            tolerances,
            order,
            initial_step,
            step_factor,
            max_iter,
        )
    elif step_direction == -1:
        comptime first_order_coefficients = generate_backward_finite_difference_table[
            dtype
        ]()
        return _derivative_backward_difference[dtype, func](
            x0,
            args,
            tolerances,
            order,
            initial_step,
            step_factor,
            max_iter,
        )
    else:
        raise Error(
            "SciJo Derivative: Invalid step direction parameter.\n"
            "  Expected: step_direction ∈ {-1, 0, 1}\n"
            "  Got: step_direction = "
            + String(step_direction)
            + "\n  Valid options:\n    • step_direction = 0:  Central"
            " differences (highest accuracy, requires f(x±h))\n    •"
            " step_direction = 1:  Forward differences (for left boundaries,"
            " uses f(x+h))\n    • step_direction = -1: Backward differences"
            " (for right boundaries, uses f(x-h))"
        )


# ===----------------------------------------------------------------------=== #
# Internal methods
# ===----------------------------------------------------------------------=== #


fn _derivative_central_difference[
    dtype: DType,
    func: fn[dtype: DType](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) -> Scalar[dtype],
](
    x0: Scalar[dtype],
    args: Optional[List[Scalar[dtype]]],
    tolerances: Dict[String, Scalar[dtype]] = {"atol": 1e-6, "rtol": 1e-6},
    order: Int = 8,
    initial_step: Scalar[dtype] = 0.5,
    step_factor: Scalar[dtype] = 2.0,
    max_iter: Int = 10,
) raises -> DiffResult[dtype]:
    """Computes first derivative using central finite difference method.

    Uses symmetric stencils around the evaluation point with adaptive step size
    reduction until convergence within specified tolerances.

    Parameters:
        dtype: The floating-point data type.
        func: Function to differentiate with signature fn(x, args) -> Scalar[dtype].

    Args:
        x0: Point at which to evaluate the derivative.
        args: Optional arguments to pass to the function.
        tolerances: Convergence tolerances with "atol" and "rtol" keys.
        order: Accuracy order (2, 4, 6, or 8).
        initial_step: Initial step size for finite differences.
        step_factor: Factor by which to reduce step size in each iteration.
        max_iter: Maximum number of iterations.

    Returns:
        DiffResult[dtype] containing the derivative and convergence information.

    Raises:
        Error: If the specified order is not in {2, 4, 6, 8}.
        Error: If tolerances, step size, step factor, or max_iter are invalid.
    """
    comptime first_order_coefficients_compiletime: Dict[
        Int, List[Scalar[dtype]]
    ] = generate_central_finite_difference_table[dtype]()
    var first_order_coefficients = materialize[
        first_order_coefficients_compiletime
    ]()

    var diff_estimate: Scalar[dtype] = 0.0
    var prev_diff: Scalar[dtype] = 0.0
    var atol: Scalar[dtype] = tolerances["atol"]
    var rtol: Scalar[dtype] = tolerances["rtol"]

    if atol < 0:
        raise Error(
            "SciJo Derivative (Central): Invalid absolute tolerances.\n"
            "  Expected: atol ≥ 0\n"
            "  Got: atol = "
            + String(atol)
            + "\n  Note: Absolute tolerances must be non-negative for"
            " convergence testing."
        )
    if rtol < 0:
        raise Error(
            "SciJo Derivative (Central): Invalid relative tolerances.\n"
            "  Expected: rtol ≥ 0\n"
            "  Got: rtol = "
            + String(rtol)
            + "\n  Note: Relative tolerances must be non-negative for"
            " convergence testing."
        )

    if initial_step <= 0:
        raise Error(
            "SciJo Derivative (Central): Invalid initial step size.\n"
            "  Expected: initial_step > 0\n"
            "  Got: initial_step = "
            + String(initial_step)
            + "\n  Note: Step size must be positive for finite difference"
            " computation."
        )
    if step_factor <= 1:
        raise Error(
            "SciJo Derivative (Central): Invalid step reduction factor.\n"
            "  Expected: step_factor > 1\n"
            "  Got: step_factor = "
            + String(step_factor)
            + "\n  Note: Step factor must be > 1 for Richardson extrapolation"
            " convergence."
        )

    if max_iter <= 0:
        raise Error(
            "SciJo Derivative (Central): Invalid maximum iterations.\n"
            "  Expected: max_iter > 0\n"
            "  Got: max_iter = "
            + String(max_iter)
            + "\n  Note: At least one iteration is required for derivative"
            " computation."
        )

    var coefficients: List[Scalar[dtype]]
    if order in (2, 4, 6, 8):
        coefficients = first_order_coefficients[order].copy()
    else:
        raise Error(
            "SciJo Derivative (Central): Invalid accuracy order specified.\n"
            "  Expected: order ∈ {2, 4, 6, 8}\n"
            "  Got: order = "
            + String(order)
            + "\n  Note: Higher orders provide better accuracy but require more"
            " function evaluations.\n  Available orders map to truncation"
            " errors: 2→O(h²), 4→O(h⁴), 6→O(h⁶), 8→O(h⁸)"
        )
    var step: Scalar[dtype] = initial_step

    for i in range(max_iter):
        diff_estimate = 0.0
        var j: Int = 0
        for ref coeff in coefficients:
            diff_estimate += coeff * func(
                x0 + step * Scalar[dtype](j - len(coefficients) // 2), args
            )
            j += 1
        diff_estimate /= step
        if i > 0:
            var diff_change = abs(diff_estimate - prev_diff)
            var tolerance_threshold = atol + rtol * abs(diff_estimate)
            if diff_change < tolerance_threshold:
                return DiffResult[dtype](
                    success=True,
                    df=diff_estimate,
                    error=diff_change,
                    nit=i + 1,
                    nfev=(i + 1) * len(coefficients),
                    x=x0,
                )

        prev_diff = diff_estimate
        step /= step_factor

    return DiffResult[dtype](
        success=False,
        df=diff_estimate,
        error=0.0,
        nit=max_iter,
        nfev=max_iter * len(coefficients),
        x=x0,
    )


fn _derivative_forward_difference[
    dtype: DType,
    func: fn[dtype: DType](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) -> Scalar[dtype],
](
    x0: Scalar[dtype],
    args: Optional[List[Scalar[dtype]]],
    tolerances: Dict[String, Scalar[dtype]] = {"atol": 1e-6, "rtol": 1e-6},
    order: Int = 8,
    initial_step: Scalar[dtype] = 0.5,
    step_factor: Scalar[dtype] = 2.0,
    max_iter: Int = 10,
) raises -> DiffResult[dtype]:
    """Computes first derivative using forward finite difference method.

    Uses one-sided stencils in the forward direction with adaptive step size
    reduction. Suitable for left boundary conditions or when backward evaluations
    are not available.

    Parameters:
        dtype: The floating-point data type.
        func: Function to differentiate with signature fn(x, args) -> Scalar[dtype].

    Args:
        x0: Point at which to evaluate the derivative.
        args: Optional arguments to pass to the function.
        tolerances: Convergence tolerances with "atol" and "rtol" keys.
        order: Accuracy order (1, 2, 3, 4, 5, or 6).
        initial_step: Initial step size for finite differences.
        step_factor: Factor by which to reduce step size in each iteration.
        max_iter: Maximum number of iterations.

    Returns:
        DiffResult[dtype] containing the derivative and convergence information.

    Raises:
        Error: If the specified order is not in {1, 2, 3, 4, 5, 6}.
        Error: If tolerances, step size, step factor, or max_iter are invalid.
    """
    comptime first_order_coefficients_compiletime: Dict[
        Int, List[Scalar[dtype]]
    ] = generate_forward_finite_difference_table[dtype]()
    var first_order_coefficients = materialize[
        first_order_coefficients_compiletime
    ]()

    var diff_estimate: Scalar[dtype] = 0.0
    var prev_diff: Scalar[dtype] = 0.0
    var atol: Scalar[dtype] = tolerances["atol"]
    var rtol: Scalar[dtype] = tolerances["rtol"]

    if atol < 0:
        raise Error(
            "SciJo Derivative (Forward): Invalid absolute tolerances.\n"
            "  Expected: atol ≥ 0\n"
            "  Got: atol = "
            + String(atol)
            + "\n  Note: Absolute tolerances must be non-negative for"
            " convergence testing."
        )
    if rtol < 0:
        raise Error(
            "SciJo Derivative (Forward): Invalid relative tolerances.\n"
            "  Expected: rtol ≥ 0\n"
            "  Got: rtol = "
            + String(rtol)
            + "\n  Note: Relative tolerances must be non-negative for"
            " convergence testing."
        )

    if initial_step <= 0:
        raise Error(
            "SciJo Derivative (Forward): Invalid initial step size.\n"
            "  Expected: initial_step > 0\n"
            "  Got: initial_step = "
            + String(initial_step)
            + "\n  Note: Step size must be positive for finite difference"
            " computation."
        )
    if step_factor <= 1:
        raise Error(
            "SciJo Derivative (Forward): Invalid step reduction factor.\n"
            "  Expected: step_factor > 1\n"
            "  Got: step_factor = "
            + String(step_factor)
            + "\n  Note: Step factor must be > 1 for Richardson extrapolation"
            " convergence."
        )

    if max_iter <= 0:
        raise Error(
            "SciJo Derivative (Forward): Invalid maximum iterations.\n"
            "  Expected: max_iter > 0\n"
            "  Got: max_iter = "
            + String(max_iter)
            + "\n  Note: At least one iteration is required for derivative"
            " computation."
        )

    if order in (1, 2, 3, 4, 5, 6):
        coefficients = first_order_coefficients[order].copy()
    else:
        raise Error(
            "SciJo Derivative (Forward): Invalid accuracy order specified.\n"
            "  Expected: order ∈ {1, 2, 3, 4, 5, 6}\n"
            "  Got: order = "
            + String(order)
            + "\n  Note: Forward differences are limited to order 6 due to"
            " numerical stability.\n  Available orders map to truncation"
            " errors: 1→O(h), 2→O(h²), ..., 6→O(h⁶)"
        )
    var step: Scalar[dtype] = initial_step

    for i in range(max_iter):
        diff_estimate = 0.0
        var j: Int = 0
        for ref coeff in coefficients:
            diff_estimate += coeff * func(x0 + step * Scalar[dtype](j), args)
            j += 1
        diff_estimate /= step
        if i > 0:
            var diff_change = abs(diff_estimate - prev_diff)
            var tolerance_threshold = atol + rtol * abs(diff_estimate)
            if diff_change < tolerance_threshold:
                return DiffResult[dtype](
                    success=True,
                    df=diff_estimate,
                    error=diff_change,
                    nit=i + 1,
                    nfev=(i + 1) * len(coefficients),
                    x=x0,
                )

        prev_diff = diff_estimate
        step /= step_factor

    return DiffResult[dtype](
        success=False,
        df=diff_estimate,
        error=0.0,
        nit=max_iter,
        nfev=max_iter * len(coefficients),
        x=x0,
    )


fn _derivative_backward_difference[
    dtype: DType,
    func: fn[dtype: DType](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) -> Scalar[dtype],
](
    x0: Scalar[dtype],
    args: Optional[List[Scalar[dtype]]],
    tolerances: Dict[String, Scalar[dtype]] = {"atol": 1e-6, "rtol": 1e-6},
    order: Int = 8,
    initial_step: Scalar[dtype] = 0.5,
    step_factor: Scalar[dtype] = 2.0,
    max_iter: Int = 10,
) raises -> DiffResult[dtype]:
    """Computes first derivative using backward finite difference method.

    Uses one-sided stencils in the backward direction with adaptive step size
    reduction. Suitable for right boundary conditions or when forward evaluations
    are not available.

    Parameters:
        dtype: The floating-point data type.
        func: Function to differentiate with signature fn(x, args) -> Scalar[dtype].

    Args:
        x0: Point at which to evaluate the derivative.
        args: Optional arguments to pass to the function.
        tolerances: Convergence tolerances with "atol" and "rtol" keys.
        order: Accuracy order (1, 2, 3, 4, 5, or 6).
        initial_step: Initial step size for finite differences.
        step_factor: Factor by which to reduce step size in each iteration.
        max_iter: Maximum number of iterations.

    Returns:
        DiffResult[dtype] containing the derivative and convergence information.

    Raises:
        Error: If the specified order is not in {1, 2, 3, 4, 5, 6}.
        Error: If tolerances, step size, step factor, or max_iter are invalid.
    """
    comptime first_order_coefficients_compiletime: Dict[
        Int, List[Scalar[dtype]]
    ] = generate_backward_finite_difference_table[dtype]()
    var first_order_coefficients = materialize[
        first_order_coefficients_compiletime
    ]()

    var diff_estimate: Scalar[dtype] = 0.0
    var prev_diff: Scalar[dtype] = 0.0
    var atol: Scalar[dtype] = tolerances["atol"]
    var rtol: Scalar[dtype] = tolerances["rtol"]

    if atol < 0:
        raise Error(
            "SciJo Derivative (Backward): Invalid absolute tolerances.\n"
            "  Expected: atol ≥ 0\n"
            "  Got: atol = "
            + String(atol)
            + "\n  Note: Absolute tolerances must be non-negative for"
            " convergence testing."
        )
    if rtol < 0:
        raise Error(
            "SciJo Derivative (Backward): Invalid relative tolerances.\n"
            "  Expected: rtol ≥ 0\n"
            "  Got: rtol = "
            + String(rtol)
            + "\n  Note: Relative tolerances must be non-negative for"
            " convergence testing."
        )

    if initial_step <= 0:
        raise Error(
            "SciJo Derivative (Backward): Invalid initial step size.\n"
            "  Expected: initial_step > 0\n"
            "  Got: initial_step = "
            + String(initial_step)
            + "\n  Note: Step size must be positive for finite difference"
            " computation."
        )
    if step_factor <= 1:
        raise Error(
            "SciJo Derivative (Backward): Invalid step reduction factor.\n"
            "  Expected: step_factor > 1\n"
            "  Got: step_factor = "
            + String(step_factor)
            + "\n  Note: Step factor must be > 1 for Richardson extrapolation"
            " convergence."
        )

    if max_iter <= 0:
        raise Error(
            "SciJo Derivative (Backward): Invalid maximum iterations.\n"
            "  Expected: max_iter > 0\n"
            "  Got: max_iter = "
            + String(max_iter)
            + "\n  Note: At least one iteration is required for derivative"
            " computation."
        )

    if order in (1, 2, 3, 4, 5, 6):
        coefficients = first_order_coefficients[order].copy()
    else:
        raise Error(
            "SciJo Derivative (Backward): Invalid accuracy order specified.\n"
            "  Expected: order ∈ {1, 2, 3, 4, 5, 6}\n"
            "  Got: order = "
            + String(order)
            + "\n  Note: Backward differences are limited to order 6 due to"
            " numerical stability.\n  Available orders map to truncation"
            " errors: 1→O(h), 2→O(h²), ..., 6→O(h⁶)"
        )
    var step: Scalar[dtype] = initial_step

    for i in range(max_iter):
        diff_estimate = 0.0
        var j: Scalar[dtype] = 0
        for ref coeff in coefficients:
            diff_estimate += coeff * func(x0 + step * j, args)
            j += 1
        diff_estimate /= step
        if i > 0:
            var diff_change = abs(diff_estimate - prev_diff)
            var tolerance_threshold = atol + rtol * abs(diff_estimate)
            if diff_change < tolerance_threshold:
                return DiffResult[dtype](
                    success=True,
                    df=diff_estimate,
                    error=diff_change,
                    nit=i + 1,
                    nfev=(i + 1) * len(coefficients),
                    x=x0,
                )

        prev_diff = diff_estimate
        step /= step_factor

    return DiffResult[dtype](
        success=False,
        df=diff_estimate,
        error=0.0,
        nit=max_iter,
        nfev=max_iter * len(coefficients),
        x=x0,
    )
