# ===----------------------------------------------------------------------=== #
# Scijo: Optimize - Minimize Scalar
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Optimize Module - Minimize scalar (scijo.optimize.min_scalar)
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
    var fun: Scalar[Self.dtype]
    var success: Bool
    var status: Int
    var message: String
    var nit: Int
    var nfev: Int

    fn __init__(
        out self,
        x: Scalar[Self.dtype],
        fun: Scalar[Self.dtype],
        success: Bool,
        status: Int,
        message: String,
        nit: Int,
        nfev: Int,
    ):
        self.x = x
        self.fun = fun
        self.success = success
        self.status = status
        self.message = message
        self.nit = nit
        self.nfev = nfev

    fn __str__(self) raises -> String:
        return (
            t"Result(success={self.success}, x={self.x}, fun={self.fun},"
            t" status={self.status}, nit={self.nit}, nfev={self.nfev})"
        )

    fn write_to[W: Writer](self, mut writer: W):
        """Writes the array to a writer.

        Args:
            writer: The writer to write the array to.
        """
        writer.write(
            t"Result(success={self.success}, x={self.x}, fun={self.fun},"
            t" status={self.status}, nit={self.nit}, nfev={self.nfev})"
        )


# ===----------------------------------------------------------------------=== #
# Implementation of scalar minimization algorithms: .
# Brent's method, Golden section search, and bounded minimization
# ===----------------------------------------------------------------------=== #


fn _brent_minimize[
    dtype: DType,
    f: fn[dtype: DType](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) -> Scalar[dtype],
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
    var d: Scalar[dtype] = 0
    var e: Scalar[dtype] = 0

    for i in range(maxiter):
        var m = (a + c) / 2
        var tol1 = tol * abs(x) + Scalar[dtype](1e-12)
        var tol2 = tol1 * 2

        if abs(c - a) <= tol2:
            return OptimizeResult[dtype](
                x=x,
                fun=fx,
                success=True,
                status=0,
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
                var d = p / q
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
        status=1,
        message="Maximum number of iterations exceeded.",
        nit=maxiter,
        nfev=nfev,
    )


fn _golden_minimize[
    dtype: DType,
    f: fn[dtype: DType](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) -> Scalar[dtype],
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
                status=0,
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
        status=1,
        message="Maximum number of iterations exceeded.",
        nit=maxiter,
        nfev=nfev,
    )


fn _bounded_minimize[
    dtype: DType,
    f: fn[dtype: DType](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) -> Scalar[dtype],
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
    var d: Scalar[dtype] = 0
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
                status=0,
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
        status=1,
        message="Maximum number of iterations exceeded.",
        nit=maxiter,
        nfev=nfev,
    )


fn minimize_scalar[
    dtype: DType,
    f: fn[dtype: DType](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) -> Scalar[dtype],
    *,
    method: String = "Brent",
](
    Bracket: Optional[Tuple[Scalar[dtype], Scalar[dtype]]] = None,
    bounds: Optional[Tuple[Scalar[dtype], Scalar[dtype]]] = None,
    args: Optional[List[Scalar[dtype]]] = None,
    tol: Scalar[dtype] = 1e-8,
    maxiter: Int = 500,
) raises -> OptimizeResult[dtype]:
    """Minimize a scalar function using the specified method.

    Parameters:
        dtype: The floating-point data type.
        f: Function f(x, args) -> Scalar[dtype] to minimize.
        method: Optimization algorithm: "Brent", "Golden", or "Bounded".

    Args:
        Bracket: (a, b) tuple specifying an initial interval for bracketing methods.
        bounds: (lower, upper) tuple specifying the search interval for bounded methods.
        args: Optional arguments forwarded to f.
        tol: Tolerance for convergence.
        maxiter: Maximum number of iterations.

    Returns:
        OptimizeResult[dtype] containing the optimization result.

    Examples:
        ```mojo
        from scijo.prelude import *
        from scijo.optimize import minimize_scalar

        fn objective[dtype: DType](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[dtype]:
            return (x - 2) * (x - 2) + 1

        var result = minimize_scalar[Float64, objective, method="Brent"](
            Bracket=(0.0, 4.0),
            tol=1e-8,
            maxiter=100
        )
        print(result)
        # Output: Result(success=True, x=2.0, fun=1.0, status=0, nit=5, nfev=8)
        ```
    """

    if method == "Brent" or method == "brent":
        if Bracket:
            return _brent_minimize[dtype, f](
                args, Bracket.value(), tol, maxiter
            )
        if bounds:
            return _brent_minimize[dtype, f](args, bounds.value(), tol, maxiter)
        raise Error("Bracket or bounds must be provided for Brent method.")

    if method == "Golden" or method == "golden":
        if Bracket:
            return _golden_minimize[dtype, f](
                args, Bracket.value(), tol, maxiter
            )
        if bounds:
            return _golden_minimize[dtype, f](
                args, bounds.value(), tol, maxiter
            )
        raise Error("Bracket or bounds must be provided for Golden method.")

    if method == "Bounded" or method == "bounded":
        if not bounds:
            raise Error("Bounds must be provided for bounded method.")
        return _bounded_minimize[dtype, f](args, bounds.value(), tol, maxiter)

    raise Error(
        "Unsupported method: "
        + String(method)
        + ". Supported methods: 'Brent', 'Golden', 'Bounded'."
    )
