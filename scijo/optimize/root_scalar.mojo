# ===----------------------------------------------------------------------=== #
# Scijo: Optimize - Root Scalar
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Optimize Module - Scalar Root-Finding (scijo.optimize.root_scalar)

Scalar root-finding methods for nonlinear equations. Includes bracketing methods
(bisection), derivative-based methods (Newton-Raphson), and derivative-free
methods (secant).
"""

# TODO: check if we are using the right tolerance conditions in all methods.

# ===----------------------------------------------------------------------=== #
# Root scalar
# ===----------------------------------------------------------------------=== #


fn root_scalar[
    dtype: DType,
    f: fn[dtype: DType](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) -> Scalar[dtype],
    fprime: Optional[
        fn[
            dtype: DType
        ](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[
            dtype
        ]
    ] = None,
    *,
    method: String = "bisect",
](
    args: Optional[List[Scalar[dtype]]] = None,
    x0: Optional[Scalar[dtype]] = None,
    x1: Optional[Scalar[dtype]] = None,
    bracket: Optional[Tuple[Scalar[dtype], Scalar[dtype]]] = None,
    xtol: Scalar[dtype] = 1e-8,
    rtol: Scalar[dtype] = 1e-8,
    maxiter: Int = 100,
) raises -> Scalar[dtype]:
    """Finds a root of a scalar function using the specified method.

    Parameters:
        dtype: The floating-point data type.
        f: Function f(x, args) -> Scalar[dtype] for which to find a root.
        fprime: Derivative function f'(x, args). Required for Newton's method.
        method: Root-finding algorithm: "bisect", "newton", or "secant".

    Args:
        args: Optional arguments forwarded to f and fprime.
        x0: Initial guess. Required for Newton and secant methods.
        x1: Second initial guess. Required for secant method.
        bracket: (a, b) tuple where f(a) and f(b) have opposite signs. Required for bisection.
        xtol: Absolute tolerance for convergence.
        rtol: Relative tolerance for convergence.
        maxiter: Maximum number of iterations.

    Raises:
        Error: If required inputs for the chosen method are missing or invalid.

    Returns:
        The approximate root of the function.
    """

    @parameter
    if method == "newton":
        if not fprime:
            raise Error(
                "Scijo [root_scalar]: Derivative fprime must be provided for"
                " Newton's method."
            )
        return newton[dtype, f, fprime.value()](args, x0, xtol, rtol, maxiter)
    elif method == "bisect":
        if not bracket:
            raise Error(
                "Scijo [root_scalar]: Bracket must be provided for bisection"
                " method."
            )
        return bisect[dtype, f](args, bracket.value(), xtol, rtol, maxiter)
    elif method == "secant":
        if not (x0 and x1):
            raise Error(
                "Scijo [root_scalar]: Initial guesses x0 and x1 must be"
                " provided for secant method."
            )
        return secant[dtype, f](
            args, x0.value(), x1.value(), xtol, rtol, maxiter
        )
    else:
        raise Error(
            "Scijo [root_scalar]: Unsupported method: " + String(method)
        )


# ===----------------------------------------------------------------------=== #
# Root scalar methods
# ===----------------------------------------------------------------------=== #


fn newton[
    dtype: DType,
    f: fn[dtype: DType](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) -> Scalar[dtype],
    fprime: fn[dtype: DType](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) -> Scalar[dtype],
    method: String = "newton",
](
    args: Optional[List[Scalar[dtype]]],
    x0: Optional[Scalar[dtype]] = None,
    xtol: Scalar[dtype] = 1e-8,
    rtol: Scalar[dtype] = 1e-8,
    maxiter: Int = 100,
) raises -> Scalar[dtype]:
    """Finds a root using the Newton-Raphson method.

    Terminates when the step size or function value falls below
    max(xtol, rtol * |x|), or when maxiter is reached.

    Parameters:
        dtype: The floating-point data type.
        f: Function for which to find a root.
        fprime: Derivative of f.
        method: Method name (unused, for signature compatibility).

    Args:
        args: Optional arguments forwarded to f and fprime.
        x0: Initial guess. Required.
        xtol: Absolute tolerance for convergence.
        rtol: Relative tolerance for convergence.
        maxiter: Maximum number of iterations.

    Raises:
        Error: If x0 is not provided or the derivative is zero at any step.

    Returns:
        The approximate root as a Scalar[dtype].
    """
    var xn: Scalar[dtype]
    if x0:
        xn = x0.value()
    else:
        raise Error(
            "Scijo [newton]: Initial guess x0 must be provided for Newton's"
            " method."
        )

    for _ in range(maxiter):
        var fx = f(xn, args)
        var fpx = fprime(xn, args)

        if fpx == 0:
            raise Error(
                "Scijo [newton]: Derivative is zero. Newton-Raphson step would"
                " divide by zero."
            )

        var delta = fx / fpx
        var xn_next = xn - delta

        var tol_x = max(xtol, rtol * abs(xn_next))
        var tol_f = max(xtol, rtol * abs(fx))

        if abs(fx) <= tol_f:
            return xn_next
        if abs(delta) <= tol_x:
            return xn_next

        xn = xn_next

    return xn


fn bisect[
    dtype: DType,
    f: fn[dtype: DType](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) -> Scalar[dtype],
](
    args: Optional[List[Scalar[dtype]]],
    bracket: Tuple[Scalar[dtype], Scalar[dtype]],
    xtol: Scalar[dtype] = 1e-8,
    rtol: Scalar[dtype] = 1e-8,
    maxiter: Int = 100,
) raises -> Scalar[dtype]:
    """Finds a root using the bisection method over a bracket [a, b].

    Requires f(a) and f(b) to have opposite signs. Terminates when the
    interval half-width falls below max(xtol, rtol * |c|), f(c) == 0,
    or maxiter is reached.

    Parameters:
        dtype: The floating-point data type.
        f: Function for which to find a root.

    Args:
        args: Optional arguments forwarded to f.
        bracket: (a, b) tuple where f(a) and f(b) have opposite signs.
        xtol: Absolute tolerance for convergence.
        rtol: Relative tolerance for convergence.
        maxiter: Maximum number of iterations.

    Raises:
        Error: If f(a) and f(b) do not have opposite signs.

    Returns:
        The approximate root as a Scalar[dtype].
    """
    var a: Scalar[dtype] = bracket[0]
    var b: Scalar[dtype] = bracket[1]

    var fa = f(a, args)
    var fb = f(b, args)

    if fa == 0:
        return a
    if fb == 0:
        return b

    if fa * fb > 0:
        raise Error(
            "Scijo [newton]: f(a) and f(b) must have opposite signs (bracket"
            " does not enclose a root)."
        )

    for _ in range(maxiter):
        var c = (a + b) / 2
        var fc = f(c, args)

        if fc == 0:
            return c

        var tol_x = max(xtol, rtol * abs(c))
        var half_width = abs(b - a) / 2
        if half_width <= tol_x:
            return c

        if fa * fc < 0:
            b = c
        else:
            a = c
            fa = fc

    return (a + b) / 2


fn secant[
    dtype: DType,
    f: fn[dtype: DType](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) -> Scalar[dtype],
](
    args: Optional[List[Scalar[dtype]]],
    x0: Scalar[dtype],
    x1: Scalar[dtype],
    xtol: Scalar[dtype] = 1e-8,
    rtol: Scalar[dtype] = 1e-8,
    maxiter: Int = 100,
) raises -> Scalar[dtype]:
    """Finds a root using the secant method.

    A derivative-free method that approximates the derivative using finite
    differences between the two most recent iterates.

    Parameters:
        dtype: The floating-point data type.
        f: Function for which to find a root.

    Args:
        args: Optional arguments forwarded to f.
        x0: First initial guess.
        x1: Second initial guess.
        xtol: Absolute tolerance for convergence.
        rtol: Relative tolerance for convergence.
        maxiter: Maximum number of iterations.

    Raises:
        Error: If zero slope is encountered.

    Returns:
        The approximate root as a Scalar[dtype].
    """
    var a: Scalar[dtype] = x0
    var b: Scalar[dtype] = x1

    for _ in range(maxiter):
        var f0 = f(a, args)
        var f1 = f(b, args)

        var denom = f1 - f0
        if denom == 0:
            raise Error(
                "Scijo [newton]: Secant method encountered zero slope (f1 - f0"
                " == 0)."
            )

        var xn = b - (f1 * (b - a)) / denom

        var fxn = f(xn, args)
        var tol_x = max(xtol, rtol * abs(xn))
        var tol_f = max(xtol, rtol * abs(fxn))

        if abs(fxn) <= tol_f:
            return xn

        if abs(xn - b) <= tol_x:
            return xn

        a = b
        b = xn

    return b
