# ===----------------------------------------------------------------------=== #
# SciJo: Optimize module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Scalar Minimization (`scijo.optimize.min_scalar`)
====================================================
Provides scalar minimization algorithms including Brent's method, golden
section search, and bounded minimization.

Examples
--------
    ```mojo
    from scijo.prelude import *
    from scijo.optimize import minimize_scalar

    def objective[dtype: DType](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) capturing -> Scalar[dtype]:
        return (x - 2) * (x - 2) + 1

    var result = minimize_scalar[f64, objective, method="Brent"](
        bracket=(0.0, 4.0),
        atol=1e-8,
        maxiter=100
    )
    ```
"""

comptime _phi: Float64 = 1.618033988749895
"""Inverse of the golden ratio, used in optimization algorithms."""
comptime _invphi: Float64 = 0.3819660112501051
"""Inverse of the golden ratio conjugate, used in optimization algorithms."""

# ===----------------------------------------------------------------------=== #
# Scalar minimization
# ===----------------------------------------------------------------------=== #


struct OptimizeResult[dtype: DType](ImplicitlyCopyable, Writable):
    """Result structure for scalar minimization operations."""

    var x: Scalar[Self.dtype]
    """The minimizer."""
    var fun: Scalar[Self.dtype]
    """The function value at the minimizer."""
    var success: Bool
    """Whether the optimization converged."""
    var message: String
    """Human-readable status message."""
    var nit: Int
    """Number of iterations performed."""
    var nfev: Int
    """Number of function evaluations used."""

    def __init__(
        out self,
        x: Scalar[Self.dtype],
        fun: Scalar[Self.dtype],
        success: Bool,
        message: String,
        nit: Int,
        nfev: Int,
    ):
        self.x = x
        self.fun = fun
        self.success = success
        self.message = message
        self.nit = nit
        self.nfev = nfev

    def __str__(self) raises -> String:
        return String(
            "OptimizeResult(success={}, x={}, fun={}, nit={}, nfev={},"
            " message='{}')"
        ).format(
            self.success, self.x, self.fun, self.nit, self.nfev, self.message
        )

    def write_to[W: Writer](self, mut writer: W):
        try:
            writer.write(
                String(
                    "Minimize Result\n"
                    "===============\n"
                    "Success : {}\n"
                    "x       : {}\n"
                    "f(x)    : {}\n"
                    "Iters   : {}\n"
                    "Evals   : {}\n"
                    "Message : {}\n"
                ).format(
                    self.success,
                    self.x,
                    self.fun,
                    self.nit,
                    self.nfev,
                    self.message,
                )
            )
        except e:
            writer.write("Error displaying OptimizeResult: " + String(e) + "\n")


# ===----------------------------------------------------------------------=== #
# Implementation of scalar minimization algorithms: .
# Brent's method, Golden section search, and bounded minimization
# ===----------------------------------------------------------------------=== #


def _brent_minimize[
    dtype: DType,
    f: def[dtype: DType](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) capturing -> Scalar[dtype],
](
    args: Optional[List[Scalar[dtype]]],
    bracket: Tuple[Scalar[dtype], Scalar[dtype]],
    tol: Scalar[dtype],
    maxiter: Int,
) raises -> OptimizeResult[dtype]:
    var a: Scalar[dtype] = bracket[0]
    var b: Scalar[dtype] = bracket[1]

    if a == b:
        raise Error(
            "Scijo [_brent_minimize]: Bracket endpoints must be distinct."
        )

    if a > b:
        var tmp = a
        a = b
        b = tmp

    var fa = f(a, args)
    var fb = f(b, args)
    var nfev = 2

    if fb > fa:
        var tmpx = a
        var tmpf = fa
        a = b
        b = tmpx
        fb = tmpf

    var c = b + (b - a) * Scalar[dtype](_phi)
    var fc = f(c, args)
    nfev += 1
    var bracket_iter = 0
    while fb > fc and bracket_iter < 50:
        a = b
        b = c
        fb = fc
        c = b + (b - a) * Scalar[dtype](_phi)
        fc = f(c, args)
        nfev += 1
        bracket_iter += 1

    if a > c:
        var tmp2 = a
        a = c
        c = tmp2

    var x = b
    var w = b
    var v = b
    var fx = fb
    var fw = fb
    var fv = fb
    var d: Scalar[dtype]
    var e: Scalar[dtype] = 0.0

    for i in range(maxiter):
        var m = (a + c) / 2
        var tol1 = tol * abs(x) + Scalar[dtype](1e-12)
        var tol2 = tol1 * 2

        if abs(c - a) <= tol2:
            return OptimizeResult[dtype](
                x=x,
                fun=fx,
                success=True,
                message="Optimization terminated successfully.",
                nit=i,
                nfev=nfev,
            )

        if abs(e) > tol1:
            var r = (x - w) * (fx - fv)
            var q = (x - v) * (fx - fw)
            var p = (x - v) * q - (x - w) * r
            q = (q - r) * 2
            if q > 0:
                p = -p
            q = abs(q)
            if (
                abs(p) < abs(Scalar[dtype](0.5) * q * e)
                and p > q * (a - x)
                and p < q * (c - x)
            ):
                d = p / q
                var u1 = x + d
                if (u1 - a) < tol2 or (c - u1) < tol2:
                    d = tol1 if x < m else -tol1
            else:
                e = c - x if x < m else a - x
                d = Scalar[dtype](_invphi) * e
        else:
            e = c - x if x < m else a - x
            d = Scalar[dtype](_invphi) * e

        var u = x + d if abs(d) >= tol1 else x + (tol1 if d > 0 else -tol1)
        var fu = f(u, args)
        nfev += 1

        if fu <= fx:
            if u < x:
                c = x
            else:
                a = x
            v = w
            fv = fw
            w = x
            fw = fx
            x = u
            fx = fu
        else:
            if u < x:
                a = u
            else:
                c = u
            if fu <= fw or w == x:
                v = w
                fv = fw
                w = u
                fw = fu
            elif fu <= fv or v == x or v == w:
                v = u
                fv = fu

    return OptimizeResult[dtype](
        x=x,
        fun=fx,
        success=False,
        message="Maximum number of iterations exceeded.",
        nit=maxiter,
        nfev=nfev,
    )


def _golden_minimize[
    dtype: DType,
    f: def[dtype: DType](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) capturing -> Scalar[dtype],
](
    args: Optional[List[Scalar[dtype]]],
    bounds: Tuple[Scalar[dtype], Scalar[dtype]],
    tol: Scalar[dtype],
    maxiter: Int,
) raises -> OptimizeResult[dtype]:
    var a: Scalar[dtype] = bounds[0]
    var b: Scalar[dtype] = bounds[1]

    if a == b:
        raise Error("Scijo [_golden_minimize]: Bounds must be distinct.")

    if a > b:
        var tmp = a
        a = b
        b = tmp

    var c = b - Scalar[dtype](_invphi) * (b - a)
    var d = a + Scalar[dtype](_invphi) * (b - a)
    var fc = f(c, args)
    var fd = f(d, args)
    var nfev = 2

    for i in range(maxiter):
        if abs(b - a) <= tol * (abs(c) + abs(d)):
            var x = c if fc < fd else d
            var fx = fc if fc < fd else fd
            return OptimizeResult[dtype](
                x=x,
                fun=fx,
                success=True,
                message="Optimization terminated successfully.",
                nit=i,
                nfev=nfev,
            )

        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - Scalar[dtype](_invphi) * (b - a)
            fc = f(c, args)
            nfev += 1
        else:
            a = c
            c = d
            fc = fd
            d = a + Scalar[dtype](_invphi) * (b - a)
            fd = f(d, args)
            nfev += 1

    var x2 = c if fc < fd else d
    var fx2 = fc if fc < fd else fd
    return OptimizeResult[dtype](
        x=x2,
        fun=fx2,
        success=False,
        message="Maximum number of iterations exceeded.",
        nit=maxiter,
        nfev=nfev,
    )


def _bounded_minimize[
    dtype: DType,
    f: def[dtype: DType](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) capturing -> Scalar[dtype],
](
    args: Optional[List[Scalar[dtype]]],
    bounds: Tuple[Scalar[dtype], Scalar[dtype]],
    tol: Scalar[dtype],
    maxiter: Int,
) raises -> OptimizeResult[dtype]:
    var a: Scalar[dtype] = bounds[0]
    var b: Scalar[dtype] = bounds[1]

    if a == b:
        raise Error("Scijo [_bounded_minimize]: Bounds must be distinct.")

    if a > b:
        var tmp = a
        a = b
        b = tmp

    var x = a + Scalar[dtype](_invphi) * (b - a)
    var w = x
    var v = x
    var fx = f(x, args)
    var fw = fx
    var fv = fx
    var nfev = 1
    var d: Scalar[dtype]
    var e: Scalar[dtype] = 0

    for i in range(maxiter):
        var m = (a + b) / 2
        var tol1 = tol * abs(x) + Scalar[dtype](1e-12)
        var tol2 = tol1 * 2

        if abs(b - a) <= tol2:
            return OptimizeResult[dtype](
                x=x,
                fun=fx,
                success=True,
                message="Optimization terminated successfully.",
                nit=i,
                nfev=nfev,
            )

        if abs(e) > tol1:
            var r = (x - w) * (fx - fv)
            var q = (x - v) * (fx - fw)
            var p = (x - v) * q - (x - w) * r
            q = (q - r) * 2
            if q > 0:
                p = -p
            q = abs(q)
            if (
                abs(p) < abs(Scalar[dtype](0.5) * q * e)
                and p > q * (a - x)
                and p < q * (b - x)
            ):
                d = p / q
                var u1 = x + d
                if (u1 - a) < tol2 or (b - u1) < tol2:
                    d = tol1 if x < m else -tol1
            else:
                e = b - x if x < m else a - x
                d = Scalar[dtype](_invphi) * e
        else:
            e = b - x if x < m else a - x
            d = Scalar[dtype](_invphi) * e

        var u = x + d if abs(d) >= tol1 else x + (tol1 if d > 0 else -tol1)
        if u < a + tol2:
            u = a + tol2
        if u > b - tol2:
            u = b - tol2

        var fu = f(u, args)
        nfev += 1

        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v = w
            fv = fw
            w = x
            fw = fx
            x = u
            fx = fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v = w
                fv = fw
                w = u
                fw = fu
            elif fu <= fv or v == x or v == w:
                v = u
                fv = fu

    return OptimizeResult[dtype](
        x=x,
        fun=fx,
        success=False,
        message="Maximum number of iterations exceeded.",
        nit=maxiter,
        nfev=nfev,
    )


def minimize_scalar[
    dtype: DType,
    f: def[dtype: DType](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) capturing -> Scalar[dtype],
    *,
    method: String = "Brent",
](
    bracket: Optional[Tuple[Scalar[dtype], Scalar[dtype]]] = None,
    bounds: Optional[Tuple[Scalar[dtype], Scalar[dtype]]] = None,
    args: Optional[List[Scalar[dtype]]] = None,
    atol: Scalar[dtype] = 1e-8,
    maxiter: Int = 500,
) raises -> OptimizeResult[dtype]:
    """Minimize a scalar function using the specified method.

    Parameters:
        dtype: The floating-point data type.
        f: Function f(x, args) -> Scalar[dtype] to minimize.
        method: Optimization algorithm: "Brent", "Golden", or "Bounded".

    Args:
        bracket: (a, b) tuple specifying an initial interval for Brent and Golden methods.
        bounds: (lower, upper) tuple specifying the search interval for Bounded method.
        args: Optional arguments forwarded to f.
        atol: Absolute tolerance for convergence.
        maxiter: Maximum number of iterations.

    Returns:
        OptimizeResult[dtype] containing the optimization result.

    Examples:
        ```mojo
        from scijo.prelude import *
        from scijo.optimize import minimize_scalar

        def objective[dtype: DType](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) capturing -> Scalar[dtype]:
            return (x - 2) * (x - 2) + 1

        var result = minimize_scalar[f64, objective, method="Brent"](
            bracket=(0.0, 4.0),
            atol=1e-8,
            maxiter=100
        )
        print(result)
        ```
    """

    if method == "Brent" or method == "brent":
        if bracket:
            return _brent_minimize[dtype, f](
                args, bracket.value(), atol, maxiter
            )
        if bounds:
            return _brent_minimize[dtype, f](
                args, bounds.value(), atol, maxiter
            )
        raise Error("bracket or bounds must be provided for Brent method.")

    if method == "Golden" or method == "golden":
        if bracket:
            return _golden_minimize[dtype, f](
                args, bracket.value(), atol, maxiter
            )
        if bounds:
            return _golden_minimize[dtype, f](
                args, bounds.value(), atol, maxiter
            )
        raise Error("bracket or bounds must be provided for Golden method.")

    if method == "Bounded" or method == "bounded":
        if not bounds:
            raise Error("bounds must be provided for bounded method.")
        return _bounded_minimize[dtype, f](args, bounds.value(), atol, maxiter)

    raise Error(
        "Unsupported method: "
        + String(method)
        + ". Supported methods: 'Brent', 'Golden', 'Bounded'."
    )
