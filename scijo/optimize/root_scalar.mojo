# ===----------------------------------------------------------------------=== #
# SciJo: Optimize module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Scalar Root-Finding (`scijo.optimize.root_scalar`)
=====================================================
Scalar root-finding routines: bisection, Brent's method, secant, and
Newton's method.
"""

# ===----------------------------------------------------------------------=== #
# SciJo
# ===----------------------------------------------------------------------=== #
from scijo.optimize.utility import RootResult


def root_scalar[
    dtype: DType,
    f: def[dtype: DType](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) thin capturing[_] -> Scalar[dtype] where dtype.is_floating_point(),
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
) raises -> RootResult[dtype] where dtype.is_floating_point():
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
    f: def[dtype: DType](x: Scalar[dtype]) thin capturing[_] -> Scalar[
        dtype
    ] where dtype.is_floating_point(),
    *,
    method: String = "bisect",
](
    x0: Optional[Scalar[dtype]] = None,
    x1: Optional[Scalar[dtype]] = None,
    bracket: Optional[Tuple[Scalar[dtype], Scalar[dtype]]] = None,
    atol: Scalar[dtype] = 1e-8,
    rtol: Scalar[dtype] = 1e-8,
    maxiter: Int = 100,
) raises -> RootResult[dtype] where dtype.is_floating_point():
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

    def _with_args[
        dtype2: DType
    ](
        x: Scalar[dtype2], args: Optional[List[Scalar[dtype2]]]
    ) capturing -> Scalar[dtype2] where dtype2.is_floating_point():
        return f(x)

    return root_scalar[dtype, _with_args, method=method](
        x0=x0, x1=x1, bracket=bracket, atol=atol, rtol=rtol, maxiter=maxiter
    )


def root_scalar[
    dtype: DType,
    f: def[dtype: DType](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) thin capturing[_] -> Scalar[dtype] where dtype.is_floating_point(),
    fprime: def[dtype: DType](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) thin capturing[_] -> Scalar[dtype] where dtype.is_floating_point(),
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
) raises -> RootResult[dtype] where dtype.is_floating_point():
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
    f: def[dtype: DType](x: Scalar[dtype]) thin capturing[_] -> Scalar[
        dtype
    ] where dtype.is_floating_point(),
    fprime: def[dtype: DType](x: Scalar[dtype]) thin capturing[_] -> Scalar[
        dtype
    ] where dtype.is_floating_point(),
    *,
    method: String = "newton",
](
    x0: Optional[Scalar[dtype]] = None,
    x1: Optional[Scalar[dtype]] = None,
    bracket: Optional[Tuple[Scalar[dtype], Scalar[dtype]]] = None,
    atol: Scalar[dtype] = 1e-8,
    rtol: Scalar[dtype] = 1e-8,
    maxiter: Int = 100,
) raises -> RootResult[dtype] where dtype.is_floating_point():
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

    def _f_with_args[
        dtype2: DType
    ](
        x: Scalar[dtype2], args: Optional[List[Scalar[dtype2]]]
    ) capturing -> Scalar[dtype2] where dtype2.is_floating_point():
        return f(x)

    def _fprime_with_args[
        dtype2: DType
    ](
        x: Scalar[dtype2], args: Optional[List[Scalar[dtype2]]]
    ) capturing -> Scalar[dtype2] where dtype2.is_floating_point():
        return fprime(x)

    return root_scalar[dtype, _f_with_args, _fprime_with_args, method=method](
        x0=x0, x1=x1, bracket=bracket, atol=atol, rtol=rtol, maxiter=maxiter
    )


def newton[
    dtype: DType,
    f: def[dtype: DType](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) thin capturing[_] -> Scalar[dtype] where dtype.is_floating_point(),
    fprime: def[dtype: DType](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) thin capturing[_] -> Scalar[dtype] where dtype.is_floating_point(),
](
    args: Optional[List[Scalar[dtype]]],
    x0: Optional[Scalar[dtype]] = None,
    atol: Scalar[dtype] = 1e-8,
    rtol: Scalar[dtype] = 1e-8,
    maxiter: Int = 100,
) raises -> RootResult[dtype] where dtype.is_floating_point():
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

    var x = x0.value()
    var fx = f(x, args)
    var nfev = 1

    for i in range(maxiter):
        var dfx = fprime(x, args)
        nfev += 1
        if dfx == 0:
            return RootResult[dtype](
                root=x,
                nit=i,
                nfev=nfev,
                success=False,
                message="Derivative is zero; Newton's method stalled.",
                method="newton",
            )

        var step = fx / dfx
        var x_new = x - step
        var fx_new = f(x_new, args)
        nfev += 1

        if abs(step) <= atol + rtol * abs(x_new) or fx_new == 0:
            return RootResult[dtype](
                root=x_new,
                nit=i + 1,
                nfev=nfev,
                success=True,
                message="converged",
                method="newton",
            )

        x = x_new
        fx = fx_new

    return RootResult[dtype](
        root=x,
        nit=maxiter,
        nfev=nfev,
        success=False,
        message="maximum iterations exceeded",
        method="newton",
    )


def bisect[
    dtype: DType,
    f: def[dtype: DType](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) thin capturing[_] -> Scalar[dtype] where dtype.is_floating_point(),
](
    args: Optional[List[Scalar[dtype]]],
    bracket: Tuple[Scalar[dtype], Scalar[dtype]],
    atol: Scalar[dtype] = 1e-8,
    rtol: Scalar[dtype] = 1e-8,
    maxiter: Int = 100,
) raises -> RootResult[dtype] where dtype.is_floating_point():
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

    Raises:
        Error: If `f` does not change sign over `bracket`.
    """
    var lo: Scalar[dtype] = bracket[0]
    var hi: Scalar[dtype] = bracket[1]
    var flo = f(lo, args)
    var fhi = f(hi, args)
    var nfev = 2

    if flo == 0:
        return RootResult[dtype](
            root=lo,
            nit=0,
            nfev=nfev,
            success=True,
            message="converged",
            method="bisect",
        )
    if fhi == 0:
        return RootResult[dtype](
            root=hi,
            nit=0,
            nfev=nfev,
            success=True,
            message="converged",
            method="bisect",
        )
    if flo * fhi > 0:
        raise Error("Scijo [bisect]: f(a) and f(b) must have opposite signs.")

    for i in range(maxiter):
        var mid = (lo + hi) / 2
        var fmid = f(mid, args)
        nfev += 1

        if fmid == 0 or (hi - lo) / 2 <= atol + rtol * abs(mid):
            return RootResult[dtype](
                root=mid,
                nit=i + 1,
                nfev=nfev,
                success=True,
                message="converged",
                method="bisect",
            )

        if flo * fmid < 0:
            hi = mid
        else:
            lo = mid
            flo = fmid

    return RootResult[dtype](
        root=(lo + hi) / 2,
        nit=maxiter,
        nfev=nfev,
        success=False,
        message="maximum iterations exceeded",
        method="bisect",
    )


def bisect[
    dtype: DType,
    f: def[dtype: DType](x: Scalar[dtype]) thin capturing[_] -> Scalar[
        dtype
    ] where dtype.is_floating_point(),
](
    bracket: Tuple[Scalar[dtype], Scalar[dtype]],
    atol: Scalar[dtype] = 1e-8,
    rtol: Scalar[dtype] = 1e-8,
    maxiter: Int = 100,
) raises -> RootResult[dtype] where dtype.is_floating_point():
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

    def _with_args[
        dtype2: DType
    ](
        x: Scalar[dtype2], args: Optional[List[Scalar[dtype2]]]
    ) capturing -> Scalar[dtype2] where dtype2.is_floating_point():
        return f(x)

    return bisect[dtype, _with_args](None, bracket, atol, rtol, maxiter)


def brent[
    dtype: DType,
    f: def[dtype: DType](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) thin capturing[_] -> Scalar[dtype] where dtype.is_floating_point(),
](
    args: Optional[List[Scalar[dtype]]],
    bracket: Tuple[Scalar[dtype], Scalar[dtype]],
    atol: Scalar[dtype] = 1e-8,
    rtol: Scalar[dtype] = 1e-8,
    maxiter: Int = 100,
) raises -> RootResult[dtype] where dtype.is_floating_point():
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

    Raises:
        Error: If `f` does not change sign over `bracket`.
    """
    var a: Scalar[dtype] = bracket[0]
    var b: Scalar[dtype] = bracket[1]
    var fa = f(a, args)
    var fb = f(b, args)
    var nfev = 2

    if fa == 0:
        return RootResult[dtype](
            root=a,
            nit=0,
            nfev=nfev,
            success=True,
            message="converged",
            method="brent",
        )
    if fb == 0:
        return RootResult[dtype](
            root=b,
            nit=0,
            nfev=nfev,
            success=True,
            message="converged",
            method="brent",
        )
    if fa * fb > 0:
        raise Error("Scijo [brent]: f(a) and f(b) must have opposite signs.")

    if abs(fa) < abs(fb):
        var tmp_x = a
        var tmp_f = fa
        a = b
        fa = fb
        b = tmp_x
        fb = tmp_f

    var c = a
    var fc = fa
    var d = b - a
    var mflag = True

    for i in range(maxiter):
        var tol1 = atol + rtol * abs(b)

        if fb == 0 or abs(b - c) <= tol1:
            return RootResult[dtype](
                root=b,
                nit=i,
                nfev=nfev,
                success=True,
                message="converged",
                method="brent",
            )

        var s: Scalar[dtype]
        if fa != fc and fb != fc:
            # Inverse quadratic interpolation.
            s = (
                a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb))
            )
        else:
            # Secant step.
            s = b - fb * (b - a) / (fb - fa)

        var lo: Scalar[dtype]
        var hi: Scalar[dtype]
        if a < b:
            lo = (3 * a + b) / 4
            hi = b
        else:
            lo = b
            hi = (3 * a + b) / 4

        var bisect_condition = (
            (s < lo or s > hi)
            or (mflag and abs(s - b) >= abs(b - c) / 2)
            or (not mflag and abs(s - b) >= abs(c - d) / 2)
            or (mflag and abs(b - c) <= tol1)
            or (not mflag and abs(c - d) <= tol1)
        )
        if bisect_condition:
            s = (a + b) / 2
            mflag = True
        else:
            mflag = False

        var fs = f(s, args)
        nfev += 1

        d = c
        c = b
        fc = fb

        if fa * fs < 0:
            b = s
            fb = fs
        else:
            a = s
            fa = fs

        if abs(fa) < abs(fb):
            var tmp2_x = a
            var tmp2_f = fa
            a = b
            fa = fb
            b = tmp2_x
            fb = tmp2_f

    return RootResult[dtype](
        root=b,
        nit=maxiter,
        nfev=nfev,
        success=False,
        message="maximum iterations exceeded",
        method="brent",
    )


def brent[
    dtype: DType,
    f: def[dtype: DType](x: Scalar[dtype]) thin capturing[_] -> Scalar[
        dtype
    ] where dtype.is_floating_point(),
](
    bracket: Tuple[Scalar[dtype], Scalar[dtype]],
    atol: Scalar[dtype] = 1e-8,
    rtol: Scalar[dtype] = 1e-8,
    maxiter: Int = 100,
) raises -> RootResult[dtype] where dtype.is_floating_point():
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

    def _with_args[
        dtype2: DType
    ](
        x: Scalar[dtype2], args: Optional[List[Scalar[dtype2]]]
    ) capturing -> Scalar[dtype2] where dtype2.is_floating_point():
        return f(x)

    return brent[dtype, _with_args](None, bracket, atol, rtol, maxiter)


def secant[
    dtype: DType,
    f: def[dtype: DType](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) thin capturing[_] -> Scalar[dtype] where dtype.is_floating_point(),
](
    args: Optional[List[Scalar[dtype]]],
    x0: Scalar[dtype],
    x1: Scalar[dtype],
    atol: Scalar[dtype] = 1e-8,
    rtol: Scalar[dtype] = 1e-8,
    maxiter: Int = 100,
) raises -> RootResult[dtype] where dtype.is_floating_point():
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
    var x_prev = x0
    var x_curr = x1
    var f_prev = f(x_prev, args)
    var f_curr = f(x_curr, args)
    var nfev = 2

    if f_prev == 0:
        return RootResult[dtype](
            root=x_prev,
            nit=0,
            nfev=nfev,
            success=True,
            message="converged",
            method="secant",
        )
    if f_curr == 0:
        return RootResult[dtype](
            root=x_curr,
            nit=0,
            nfev=nfev,
            success=True,
            message="converged",
            method="secant",
        )

    for i in range(maxiter):
        if f_curr == f_prev:
            return RootResult[dtype](
                root=x_curr,
                nit=i,
                nfev=nfev,
                success=False,
                message="secant method stalled (zero denominator)",
                method="secant",
            )

        var x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        var f_new = f(x_new, args)
        nfev += 1

        if abs(x_new - x_curr) <= atol + rtol * abs(x_new) or f_new == 0:
            return RootResult[dtype](
                root=x_new,
                nit=i + 1,
                nfev=nfev,
                success=True,
                message="converged",
                method="secant",
            )

        x_prev = x_curr
        f_prev = f_curr
        x_curr = x_new
        f_curr = f_new

    return RootResult[dtype](
        root=x_curr,
        nit=maxiter,
        nfev=nfev,
        success=False,
        message="maximum iterations exceeded",
        method="secant",
    )


def secant[
    dtype: DType,
    f: def[dtype: DType](x: Scalar[dtype]) thin capturing[_] -> Scalar[
        dtype
    ] where dtype.is_floating_point(),
](
    x0: Scalar[dtype],
    x1: Scalar[dtype],
    atol: Scalar[dtype] = 1e-8,
    rtol: Scalar[dtype] = 1e-8,
    maxiter: Int = 100,
) raises -> RootResult[dtype] where dtype.is_floating_point():
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

    def _with_args[
        dtype2: DType
    ](
        x: Scalar[dtype2], args: Optional[List[Scalar[dtype2]]]
    ) capturing -> Scalar[dtype2] where dtype2.is_floating_point():
        return f(x)

    return secant[dtype, _with_args](None, x0, x1, atol, rtol, maxiter)


def newton[
    dtype: DType,
    f: def[dtype: DType](x: Scalar[dtype]) thin capturing[_] -> Scalar[
        dtype
    ] where dtype.is_floating_point(),
    fprime: def[dtype: DType](x: Scalar[dtype]) thin capturing[_] -> Scalar[
        dtype
    ] where dtype.is_floating_point(),
](
    x0: Optional[Scalar[dtype]] = None,
    atol: Scalar[dtype] = 1e-8,
    rtol: Scalar[dtype] = 1e-8,
    maxiter: Int = 100,
) raises -> RootResult[dtype] where dtype.is_floating_point():
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

    def _f_with_args[
        dtype2: DType
    ](
        x: Scalar[dtype2], args: Optional[List[Scalar[dtype2]]]
    ) capturing -> Scalar[dtype2] where dtype2.is_floating_point():
        return f(x)

    def _fprime_with_args[
        dtype2: DType
    ](
        x: Scalar[dtype2], args: Optional[List[Scalar[dtype2]]]
    ) capturing -> Scalar[dtype2] where dtype2.is_floating_point():
        return fprime(x)

    return newton[dtype, _f_with_args, _fprime_with_args](
        None, x0, atol, rtol, maxiter
    )
