# ===----------------------------------------------------------------------=== #
# SciJo: Optimize module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Scalar Root-Finding (`scijo.optimize.root_scalar`)
=====================================================
Scalar root-finding routines with SciPy-like frontend signatures.
"""

# ===----------------------------------------------------------------------=== #
# External
# ===----------------------------------------------------------------------=== #
from msl import (
    root_bisect as msl_root_bisect,
    root_brent as msl_root_brent,
    root_newton as msl_root_newton,
    root_secant as msl_root_secant,
)

# ===----------------------------------------------------------------------=== #
# SciJo
# ===----------------------------------------------------------------------=== #
from scijo.optimize.utility import RootResult


def _root_message(success: Bool, errno: Int) -> String:
    """Maps MSL status codes into user-facing SciJo messages.

    Args:
        success: Whether the backend solver reported convergence.
        errno: MSL error/status code.

    Returns:
        Human-readable status string for RootResult.
    """
    if success:
        return "converged"
    if errno == 11:
        return "maximum iterations exceeded"
    if errno == 1:
        return "domain error (invalid bracket or singular derivative)"
    return "failed"


def root_scalar[
    dtype: DType,
    f: def[dtype: DType](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) capturing -> Scalar[dtype],
    *,
    method: String = "bisect",
](
    args: Optional[List[Scalar[dtype]]] = None,
    x0: Optional[Scalar[dtype]] = None,
    x1: Optional[Scalar[dtype]] = None,
    bracket: Optional[Tuple[Scalar[dtype], Scalar[dtype]]] = None,
    atol: Scalar[dtype] = 1e-8,
    rtol: Scalar[dtype] = 1e-8,
    maxiter: Int = 100,
) raises -> RootResult[dtype]:
    """Finds a root of a scalar function (bisect or secant overload).

    Parameters:
        dtype: The floating-point data type.
        f: Scalar objective function.
        method: Root-finding method. Supported: `"bisect"`,
            `"brent"`, `"secant"`.

    Args:
        args: Optional arguments forwarded to `f`.
        x0: First initial guess (used by secant).
        x1: Second initial guess (used by secant).
        bracket: Bracketing interval `(a, b)` (used by bisect).
        atol: Absolute convergence tolerance.
        rtol: Relative convergence tolerance.
        maxiter: Maximum solver iterations.

    Returns:
        RootResult[dtype] with root estimate and diagnostics.

    Raises:
        Error: If required inputs for a method are missing.
        Error: If `"newton"` is requested without a derivative overload.
        Error: If an unsupported method is requested.
    """
    comptime if method == "bisect":
        if not bracket:
            raise Error(
                "Scijo [root_scalar]: Bracket must be provided for bisection"
                " method."
            )
        return bisect[dtype, f](args, bracket.value(), atol, rtol, maxiter)

    elif method == "brent":
        if not bracket:
            raise Error(
                "Scijo [root_scalar]: Bracket must be provided for brent"
                " method."
            )
        return brent[dtype, f](args, bracket.value(), atol, rtol, maxiter)

    elif method == "secant":
        if not (x0 and x1):
            raise Error(
                "Scijo [root_scalar]: Initial guesses x0 and x1 must be"
                " provided for secant method."
            )
        return secant[dtype, f](
            args, x0.value(), x1.value(), atol, rtol, maxiter
        )

    elif method == "newton":
        raise Error(
            "Scijo [root_scalar]: Newton method requires the overload"
            " root_scalar[dtype, f, fprime](...)."
        )

    else:
        raise Error(
            "Scijo [root_scalar]: Unsupported method: " + String(method)
        )


def root_scalar[
    dtype: DType,
    f: def[dtype: DType](x: Scalar[dtype]) capturing -> Scalar[dtype],
    *,
    method: String = "bisect",
](
    x0: Optional[Scalar[dtype]] = None,
    x1: Optional[Scalar[dtype]] = None,
    bracket: Optional[Tuple[Scalar[dtype], Scalar[dtype]]] = None,
    atol: Scalar[dtype] = 1e-8,
    rtol: Scalar[dtype] = 1e-8,
    maxiter: Int = 100,
) raises -> RootResult[dtype]:
    """Finds a root of a scalar function (args-free bisect/secant overload).

    For functions that do not need `args`: `f` takes only `x`, so extra
    parameters (if any) are captured directly from the enclosing scope
    instead of threaded through `args`. See the `args`-taking overload
    `root_scalar[dtype, f](args=..., ...)` for a function reused across
    call sites with different `args` values.

    Parameters:
        dtype: The floating-point data type.
        f: Scalar objective function with signature `def(x) -> Scalar[dtype]`.
        method: Root-finding method. Supported: `"bisect"`,
            `"brent"`, `"secant"`.

    Args:
        x0: First initial guess (used by secant).
        x1: Second initial guess (used by secant).
        bracket: Bracketing interval `(a, b)` (used by bisect).
        atol: Absolute convergence tolerance.
        rtol: Relative convergence tolerance.
        maxiter: Maximum solver iterations.

    Returns:
        RootResult[dtype] with root estimate and diagnostics.

    Raises:
        Error: If required inputs for a method are missing.
        Error: If `"newton"` is requested without a derivative overload.
        Error: If an unsupported method is requested.
    """

    @parameter
    def _with_args[
        dtype2: DType
    ](
        x: Scalar[dtype2], args: Optional[List[Scalar[dtype2]]]
    ) capturing -> Scalar[dtype2]:
        return f(x)

    return root_scalar[dtype, _with_args, method=method](
        x0=x0, x1=x1, bracket=bracket, atol=atol, rtol=rtol, maxiter=maxiter
    )


def root_scalar[
    dtype: DType,
    f: def[dtype: DType](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) capturing -> Scalar[dtype],
    fprime: def[dtype: DType](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) capturing -> Scalar[dtype],
    *,
    method: String = "newton",
](
    args: Optional[List[Scalar[dtype]]] = None,
    x0: Optional[Scalar[dtype]] = None,
    x1: Optional[Scalar[dtype]] = None,
    bracket: Optional[Tuple[Scalar[dtype], Scalar[dtype]]] = None,
    atol: Scalar[dtype] = 1e-8,
    rtol: Scalar[dtype] = 1e-8,
    maxiter: Int = 100,
) raises -> RootResult[dtype]:
    """Finds a root of a scalar function (newton-enabled overload).

    Parameters:
        dtype: The floating-point data type.
        f: Scalar objective function.
        fprime: Derivative of `f`.
        method: Root-finding method. Supported: `"newton"`,
            `"bisect"`, `"brent"`, `"secant"`.

    Args:
        args: Optional arguments forwarded to `f` and `fprime`.
        x0: Initial guess for Newton; first guess for secant.
        x1: Second initial guess for secant.
        bracket: Bracketing interval `(a, b)` for bisect.
        atol: Absolute convergence tolerance.
        rtol: Relative convergence tolerance.
        maxiter: Maximum solver iterations.

    Returns:
        RootResult[dtype] with root estimate and diagnostics.

    Raises:
        Error: If required method-specific inputs are missing.
        Error: If an unsupported method is requested.
    """
    comptime if method == "newton":
        return newton[dtype, f, fprime](args, x0, atol, rtol, maxiter)

    elif method == "bisect":
        if not bracket:
            raise Error(
                "Scijo [root_scalar]: Bracket must be provided for bisection"
                " method."
            )
        return bisect[dtype, f](args, bracket.value(), atol, rtol, maxiter)

    elif method == "brent":
        if not bracket:
            raise Error(
                "Scijo [root_scalar]: Bracket must be provided for brent"
                " method."
            )
        return brent[dtype, f](args, bracket.value(), atol, rtol, maxiter)

    elif method == "secant":
        if not (x0 and x1):
            raise Error(
                "Scijo [root_scalar]: Initial guesses x0 and x1 must be"
                " provided for secant method."
            )
        return secant[dtype, f](
            args, x0.value(), x1.value(), atol, rtol, maxiter
        )

    else:
        raise Error(
            "Scijo [root_scalar]: Unsupported method: " + String(method)
        )


def root_scalar[
    dtype: DType,
    f: def[dtype: DType](x: Scalar[dtype]) capturing -> Scalar[dtype],
    fprime: def[dtype: DType](x: Scalar[dtype]) capturing -> Scalar[dtype],
    *,
    method: String = "newton",
](
    x0: Optional[Scalar[dtype]] = None,
    x1: Optional[Scalar[dtype]] = None,
    bracket: Optional[Tuple[Scalar[dtype], Scalar[dtype]]] = None,
    atol: Scalar[dtype] = 1e-8,
    rtol: Scalar[dtype] = 1e-8,
    maxiter: Int = 100,
) raises -> RootResult[dtype]:
    """Finds a root of a scalar function (args-free newton-enabled overload).

    For functions that do not need `args`: `f` and `fprime` take only `x`,
    so extra parameters (if any) are captured directly from the enclosing
    scope instead of threaded through `args`. See the `args`-taking
    overload `root_scalar[dtype, f, fprime](args=..., ...)` for functions
    reused across call sites with different `args` values.

    Parameters:
        dtype: The floating-point data type.
        f: Scalar objective function with signature `def(x) -> Scalar[dtype]`.
        fprime: Derivative of `f`, with the same signature.
        method: Root-finding method. Supported: `"newton"`,
            `"bisect"`, `"brent"`, `"secant"`.

    Args:
        x0: Initial guess for Newton; first guess for secant.
        x1: Second initial guess for secant.
        bracket: Bracketing interval `(a, b)` for bisect.
        atol: Absolute convergence tolerance.
        rtol: Relative convergence tolerance.
        maxiter: Maximum solver iterations.

    Returns:
        RootResult[dtype] with root estimate and diagnostics.

    Raises:
        Error: If required method-specific inputs are missing.
        Error: If an unsupported method is requested.
    """

    @parameter
    def _f_with_args[
        dtype2: DType
    ](
        x: Scalar[dtype2], args: Optional[List[Scalar[dtype2]]]
    ) capturing -> Scalar[dtype2]:
        return f(x)

    @parameter
    def _fprime_with_args[
        dtype2: DType
    ](
        x: Scalar[dtype2], args: Optional[List[Scalar[dtype2]]]
    ) capturing -> Scalar[dtype2]:
        return fprime(x)

    return root_scalar[dtype, _f_with_args, _fprime_with_args, method=method](
        x0=x0, x1=x1, bracket=bracket, atol=atol, rtol=rtol, maxiter=maxiter
    )


def newton[
    dtype: DType,
    f: def[dtype: DType](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) capturing -> Scalar[dtype],
    fprime: def[dtype: DType](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) capturing -> Scalar[dtype],
](
    args: Optional[List[Scalar[dtype]]],
    x0: Optional[Scalar[dtype]] = None,
    atol: Scalar[dtype] = 1e-8,
    rtol: Scalar[dtype] = 1e-8,
    maxiter: Int = 100,
) raises -> RootResult[dtype]:
    """Finds a root using Newton-Raphson.

    Parameters:
        dtype: The floating-point data type.
        f: Scalar objective function.
        fprime: Derivative of `f`.

    Args:
        args: Optional arguments forwarded to `f` and `fprime`.
        x0: Initial guess for Newton's method.
        atol: Absolute convergence tolerance.
        rtol: Relative convergence tolerance.
        maxiter: Maximum solver iterations.

    Returns:
        RootResult[dtype] containing convergence and diagnostic fields.

    Raises:
        Error: If `x0` is not provided.
    """
    if not x0:
        raise Error(
            "Scijo [newton]: Initial guess x0 must be provided for Newton's"
            " method."
        )

    @parameter
    def wrapped_fn(x: Float64) -> Float64:
        return Float64(f(Scalar[dtype](x), args))

    @parameter
    def wrapped_dfn(x: Float64) -> Float64:
        return Float64(fprime(Scalar[dtype](x), args))

    var result = msl_root_newton[wrapped_fn, wrapped_dfn](
        Float64(x0.value()),
        epsabs=Float64(atol),
        epsrel=Float64(rtol),
        max_iter=maxiter,
    )

    return RootResult[dtype](
        root=Scalar[dtype](result.root),
        nit=result.nit,
        nfev=result.nfev,
        success=result.success,
        message=_root_message(result.success, result.errno),
        method="newton",
    )


def bisect[
    dtype: DType,
    f: def[dtype: DType](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) capturing -> Scalar[dtype],
](
    args: Optional[List[Scalar[dtype]]],
    bracket: Tuple[Scalar[dtype], Scalar[dtype]],
    atol: Scalar[dtype] = 1e-8,
    rtol: Scalar[dtype] = 1e-8,
    maxiter: Int = 100,
) raises -> RootResult[dtype]:
    """Finds a root using bisection.

    Parameters:
        dtype: The floating-point data type.
        f: Scalar objective function.

    Args:
        args: Optional arguments forwarded to `f`.
        bracket: Bracketing interval `(a, b)`.
        atol: Absolute convergence tolerance.
        rtol: Relative convergence tolerance.
        maxiter: Maximum solver iterations.

    Returns:
        RootResult[dtype] containing convergence and diagnostic fields.
    """

    @parameter
    def wrapped_fn(x: Float64) -> Float64:
        return Float64(f(Scalar[dtype](x), args))

    var result = msl_root_bisect[wrapped_fn](
        Float64(bracket[0]),
        Float64(bracket[1]),
        epsabs=Float64(atol),
        epsrel=Float64(rtol),
        max_iter=maxiter,
    )

    return RootResult[dtype](
        root=Scalar[dtype](result.root),
        nit=result.nit,
        nfev=result.nfev,
        success=result.success,
        message=_root_message(result.success, result.errno),
        method="bisect",
    )


def bisect[
    dtype: DType,
    f: def[dtype: DType](x: Scalar[dtype]) capturing -> Scalar[dtype],
](
    bracket: Tuple[Scalar[dtype], Scalar[dtype]],
    atol: Scalar[dtype] = 1e-8,
    rtol: Scalar[dtype] = 1e-8,
    maxiter: Int = 100,
) raises -> RootResult[dtype]:
    """Finds a root using bisection (args-free overload).

    For functions that do not need `args`: `f` takes only `x`, so extra
    parameters (if any) are captured directly from the enclosing scope
    instead of threaded through `args`. See the `args`-taking overload
    `bisect[dtype, f](args, bracket, ...)` for a function reused across
    call sites with different `args` values.

    Parameters:
        dtype: The floating-point data type.
        f: Scalar objective function with signature `def(x) -> Scalar[dtype]`.

    Args:
        bracket: Bracketing interval `(a, b)`.
        atol: Absolute convergence tolerance.
        rtol: Relative convergence tolerance.
        maxiter: Maximum solver iterations.

    Returns:
        RootResult[dtype] containing convergence and diagnostic fields.
    """

    @parameter
    def _with_args[
        dtype2: DType
    ](
        x: Scalar[dtype2], args: Optional[List[Scalar[dtype2]]]
    ) capturing -> Scalar[dtype2]:
        return f(x)

    return bisect[dtype, _with_args](None, bracket, atol, rtol, maxiter)


def brent[
    dtype: DType,
    f: def[dtype: DType](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) capturing -> Scalar[dtype],
](
    args: Optional[List[Scalar[dtype]]],
    bracket: Tuple[Scalar[dtype], Scalar[dtype]],
    atol: Scalar[dtype] = 1e-8,
    rtol: Scalar[dtype] = 1e-8,
    maxiter: Int = 100,
) raises -> RootResult[dtype]:
    """Finds a root using Brent's method.

    Parameters:
        dtype: The floating-point data type.
        f: Scalar objective function.

    Args:
        args: Optional arguments forwarded to `f`.
        bracket: Bracketing interval `(a, b)`.
        atol: Absolute convergence tolerance.
        rtol: Relative convergence tolerance.
        maxiter: Maximum solver iterations.

    Returns:
        RootResult[dtype] containing convergence and diagnostic fields.
    """

    @parameter
    def wrapped_fn(x: Float64) -> Float64:
        return Float64(f(Scalar[dtype](x), args))

    var result = msl_root_brent[wrapped_fn](
        Float64(bracket[0]),
        Float64(bracket[1]),
        epsabs=Float64(atol),
        epsrel=Float64(rtol),
        max_iter=maxiter,
    )

    return RootResult[dtype](
        root=Scalar[dtype](result.root),
        nit=result.nit,
        nfev=result.nfev,
        success=result.success,
        message=_root_message(result.success, result.errno),
        method="brent",
    )


def brent[
    dtype: DType,
    f: def[dtype: DType](x: Scalar[dtype]) capturing -> Scalar[dtype],
](
    bracket: Tuple[Scalar[dtype], Scalar[dtype]],
    atol: Scalar[dtype] = 1e-8,
    rtol: Scalar[dtype] = 1e-8,
    maxiter: Int = 100,
) raises -> RootResult[dtype]:
    """Finds a root using Brent's method (args-free overload).

    For functions that do not need `args`: `f` takes only `x`, so extra
    parameters (if any) are captured directly from the enclosing scope
    instead of threaded through `args`. See the `args`-taking overload
    `brent[dtype, f](args, bracket, ...)` for a function reused across
    call sites with different `args` values.

    Parameters:
        dtype: The floating-point data type.
        f: Scalar objective function with signature `def(x) -> Scalar[dtype]`.

    Args:
        bracket: Bracketing interval `(a, b)`.
        atol: Absolute convergence tolerance.
        rtol: Relative convergence tolerance.
        maxiter: Maximum solver iterations.

    Returns:
        RootResult[dtype] containing convergence and diagnostic fields.
    """

    @parameter
    def _with_args[
        dtype2: DType
    ](
        x: Scalar[dtype2], args: Optional[List[Scalar[dtype2]]]
    ) capturing -> Scalar[dtype2]:
        return f(x)

    return brent[dtype, _with_args](None, bracket, atol, rtol, maxiter)


def secant[
    dtype: DType,
    f: def[dtype: DType](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) capturing -> Scalar[dtype],
](
    args: Optional[List[Scalar[dtype]]],
    x0: Scalar[dtype],
    x1: Scalar[dtype],
    atol: Scalar[dtype] = 1e-8,
    rtol: Scalar[dtype] = 1e-8,
    maxiter: Int = 100,
) raises -> RootResult[dtype]:
    """Finds a root using the secant method.

    Parameters:
        dtype: The floating-point data type.
        f: Scalar objective function.

    Args:
        args: Optional arguments forwarded to `f`.
        x0: First initial guess.
        x1: Second initial guess.
        atol: Absolute convergence tolerance.
        rtol: Relative convergence tolerance.
        maxiter: Maximum solver iterations.

    Returns:
        RootResult[dtype] containing convergence and diagnostic fields.
    """

    @parameter
    def wrapped_fn(x: Float64) -> Float64:
        return Float64(f(Scalar[dtype](x), args))

    var result = msl_root_secant[wrapped_fn](
        Float64(x0),
        Float64(x1),
        epsabs=Float64(atol),
        epsrel=Float64(rtol),
        max_iter=maxiter,
    )

    return RootResult[dtype](
        root=Scalar[dtype](result.root),
        nit=result.nit,
        nfev=result.nfev,
        success=result.success,
        message=_root_message(result.success, result.errno),
        method="secant",
    )


def secant[
    dtype: DType,
    f: def[dtype: DType](x: Scalar[dtype]) capturing -> Scalar[dtype],
](
    x0: Scalar[dtype],
    x1: Scalar[dtype],
    atol: Scalar[dtype] = 1e-8,
    rtol: Scalar[dtype] = 1e-8,
    maxiter: Int = 100,
) raises -> RootResult[dtype]:
    """Finds a root using the secant method (args-free overload).

    For functions that do not need `args`: `f` takes only `x`, so extra
    parameters (if any) are captured directly from the enclosing scope
    instead of threaded through `args`. See the `args`-taking overload
    `secant[dtype, f](args, x0, x1, ...)` for a function reused across
    call sites with different `args` values.

    Parameters:
        dtype: The floating-point data type.
        f: Scalar objective function with signature `def(x) -> Scalar[dtype]`.

    Args:
        x0: First initial guess.
        x1: Second initial guess.
        atol: Absolute convergence tolerance.
        rtol: Relative convergence tolerance.
        maxiter: Maximum solver iterations.

    Returns:
        RootResult[dtype] containing convergence and diagnostic fields.
    """

    @parameter
    def _with_args[
        dtype2: DType
    ](
        x: Scalar[dtype2], args: Optional[List[Scalar[dtype2]]]
    ) capturing -> Scalar[dtype2]:
        return f(x)

    return secant[dtype, _with_args](None, x0, x1, atol, rtol, maxiter)


def newton[
    dtype: DType,
    f: def[dtype: DType](x: Scalar[dtype]) capturing -> Scalar[dtype],
    fprime: def[dtype: DType](x: Scalar[dtype]) capturing -> Scalar[dtype],
](
    x0: Optional[Scalar[dtype]] = None,
    atol: Scalar[dtype] = 1e-8,
    rtol: Scalar[dtype] = 1e-8,
    maxiter: Int = 100,
) raises -> RootResult[dtype]:
    """Finds a root using Newton-Raphson (args-free overload).

    For functions that do not need `args`: `f` and `fprime` take only `x`,
    so extra parameters (if any) are captured directly from the enclosing
    scope instead of threaded through `args`. See the `args`-taking
    overload `newton[dtype, f, fprime](args, x0=..., ...)` for functions
    reused across call sites with different `args` values.

    Parameters:
        dtype: The floating-point data type.
        f: Scalar objective function with signature `def(x) -> Scalar[dtype]`.
        fprime: Derivative of `f`, with the same signature.

    Args:
        x0: Initial guess for Newton's method.
        atol: Absolute convergence tolerance.
        rtol: Relative convergence tolerance.
        maxiter: Maximum solver iterations.

    Returns:
        RootResult[dtype] containing convergence and diagnostic fields.

    Raises:
        Error: If `x0` is not provided.
    """

    @parameter
    def _f_with_args[
        dtype2: DType
    ](
        x: Scalar[dtype2], args: Optional[List[Scalar[dtype2]]]
    ) capturing -> Scalar[dtype2]:
        return f(x)

    @parameter
    def _fprime_with_args[
        dtype2: DType
    ](
        x: Scalar[dtype2], args: Optional[List[Scalar[dtype2]]]
    ) capturing -> Scalar[dtype2]:
        return fprime(x)

    return newton[dtype, _f_with_args, _fprime_with_args](
        None, x0, atol, rtol, maxiter
    )
