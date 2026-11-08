# ===----------------------------------------------------------------------=== #
# SciJo: Integrate module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Fixed-Order and Multidimensional Quadrature (`scijo.integrate.fixed_quad`)
=============================================================================
`fixed_quad`, fixed-order Gauss-Legendre quadrature (no adaptivity, no error
estimate), plus `dblquad`/`tplquad`/`nquad`, multidimensional integration via
fixed-order tensor-product Gauss-Legendre quadrature (`n` points per axis).

Scoped simplification, noted explicitly rather than left as an oversight:
`dblquad`/`tplquad`/`nquad` here are fixed-order (tensor-product
Gauss-Legendre), not SciPy's adaptive QUADPACK-based nested quadrature.
`scijo.integrate.quad`'s own integrand-function type is assumed non-raising
(matching the rest of this codebase's convention), so an adaptive `quad`
call cannot be nested inside a closure passed back into `quad` itself for
an outer axis; fixed-order tensor-product quadrature sidesteps that
entirely, at the cost of adaptivity. Increase `n` for better accuracy on
integrands that are not well-approximated by a modest-degree polynomial on
each axis.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.math import cos

# ===----------------------------------------------------------------------=== #
# SciJo
# ===----------------------------------------------------------------------=== #
from scijo.integrate.utility import IntegralResult

# ===----------------------------------------------------------------------=== #
# Gauss-Legendre nodes and weights
# ===----------------------------------------------------------------------=== #


def _gauss_legendre_nodes_weights[
    dtype: DType
](n: Int) raises -> Tuple[
    List[Scalar[dtype]], List[Scalar[dtype]]
] where dtype.is_floating_point():
    """Nodes and weights for `n`-point Gauss-Legendre quadrature on [-1, 1].

    Example: nodes, weights = _gauss_legendre_nodes_weights[f64](5)

    Newton's method applied to the three-term Legendre recurrence (the
    classic `gauleg` algorithm, Press et al., *Numerical Recipes*), using
    the symmetry of the roots about 0 to compute only the first half.

    Parameters:
        dtype: The floating-point data type.

    Args:
        n: The number of quadrature points, n >= 1.

    Returns:
        A tuple `(nodes, weights)`, each of length `n`.

    Raises:
        Error: If n < 1.
    """
    if n < 1:
        raise Error("n must be >= 1, got " + String(n))

    var nodes = List[Scalar[dtype]](capacity=n)
    var weights = List[Scalar[dtype]](capacity=n)
    for _ in range(n):
        nodes.append(0.0)
        weights.append(0.0)

    var pi_val = Scalar[dtype](3.14159265358979323846)
    var eps = Scalar[dtype](3e-14)
    var m = (n + 1) // 2

    for i in range(m):
        var z = cos(
            pi_val * (Scalar[dtype](i) + 0.75) / (Scalar[dtype](n) + 0.5)
        )
        var z1: Scalar[dtype] = z + 1.0
        var pp: Scalar[dtype] = 0.0

        while abs(z - z1) > eps:
            var p1: Scalar[dtype] = 1.0
            var p2: Scalar[dtype] = 0.0
            for j in range(1, n + 1):
                var p3 = p2
                p2 = p1
                p1 = (
                    (2.0 * Scalar[dtype](j) - 1.0) * z * p2
                    - (Scalar[dtype](j) - 1.0) * p3
                ) / Scalar[dtype](j)
            pp = Scalar[dtype](n) * (z * p1 - p2) / (z * z - 1.0)
            z1 = z
            z = z1 - p1 / pp

        nodes[i] = -z
        nodes[n - 1 - i] = z
        weights[i] = 2.0 / ((1.0 - z * z) * pp * pp)
        weights[n - 1 - i] = weights[i]

    return (nodes^, weights^)


# ===----------------------------------------------------------------------=== #
# fixed_quad
# ===----------------------------------------------------------------------=== #


def fixed_quad[
    dtype: DType,
    integrand_func: def[dtype: DType](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) thin capturing[_] -> Scalar[dtype] where dtype.is_floating_point(),
](
    a: Scalar[dtype],
    b: Scalar[dtype],
    args: Optional[List[Scalar[dtype]]] = None,
    n: Int = 5,
) raises -> Scalar[dtype] where dtype.is_floating_point():
    """Computes a definite integral using fixed-order Gauss-Legendre quadrature.

    Example: result = fixed_quad[f64, integrand](0.0, 1.0, n=5)

    Unlike `quad`, this applies a single `n`-point Gauss-Legendre rule over
    the whole interval with no adaptive subdivision and no error estimate -
    accurate only when the integrand is well-approximated by a degree
    `2n - 1` polynomial on `[a, b]`. SciPy's `fixed_quad` returns a
    `(value, None)` pair; since the second element is always `None`, this
    returns just the value.

    Parameters:
        dtype: The floating-point data type.
        integrand_func: Integrand function with signature
            `def(x, args) -> Scalar[dtype]`.

    Args:
        a: Lower integration limit.
        b: Upper integration limit.
        args: Optional arguments to pass to the integrand function.
        n: The order of the Gauss-Legendre rule (number of points), n >= 1.

    Returns:
        The estimated integral value.

    Raises:
        Error: If n < 1.
    """
    var nw = _gauss_legendre_nodes_weights[dtype](n)
    ref nodes = nw[0]
    ref weights = nw[1]

    var half = (b - a) * Scalar[dtype](0.5)
    var center = (b + a) * Scalar[dtype](0.5)

    var total: Scalar[dtype] = 0.0
    for i in range(n):
        var x = center + half * nodes[i]
        total += weights[i] * integrand_func(x, args)

    return total * half


def fixed_quad[
    dtype: DType,
    integrand_func: def[dtype: DType](x: Scalar[dtype]) capturing[_] -> Scalar[
        dtype
    ],
](
    a: Scalar[dtype],
    b: Scalar[dtype],
    n: Int = 5,
) raises -> Scalar[
    dtype
] where dtype.is_floating_point():
    """Computes a definite integral using fixed-order Gauss-Legendre
    quadrature (args-free overload).

    Example: result = fixed_quad[f64, integrand](0.0, 1.0, n=5)

    See the `args`-taking overload for the algorithm. For integrands that
    do not need `args`: `integrand_func` takes only `x`, so extra
    parameters (if any) are captured directly from the enclosing scope.

    Parameters:
        dtype: The floating-point data type.
        integrand_func: Integrand function with signature
            `def(x) -> Scalar[dtype]`.

    Args:
        a: Lower integration limit.
        b: Upper integration limit.
        n: The order of the Gauss-Legendre rule (number of points), n >= 1.

    Returns:
        The estimated integral value.

    Raises:
        Error: If n < 1.
    """

    def _with_args[
        dtype2: DType
    ](
        x: Scalar[dtype2], args: Optional[List[Scalar[dtype2]]]
    ) capturing -> Scalar[dtype2]:
        return integrand_func(x)

    return fixed_quad[dtype, _with_args](a=a, b=b, args=None, n=n)


# ===----------------------------------------------------------------------=== #
# dblquad
# ===----------------------------------------------------------------------=== #


def dblquad[
    dtype: DType,
    func: def[dtype: DType](
        y: Scalar[dtype], x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) thin capturing[_] -> Scalar[dtype] where dtype.is_floating_point(),
    gfun: def[dtype: DType](x: Scalar[dtype]) thin capturing[_] -> Scalar[
        dtype
    ] where dtype.is_floating_point(),
    hfun: def[dtype: DType](x: Scalar[dtype]) thin capturing[_] -> Scalar[
        dtype
    ] where dtype.is_floating_point(),
](
    a: Scalar[dtype],
    b: Scalar[dtype],
    args: Optional[List[Scalar[dtype]]] = None,
    n: Int = 30,
) raises -> IntegralResult[dtype] where dtype.is_floating_point():
    """Computes a double integral over a region bounded by functions of x.

    Example: result = dblquad[f64, func, gfun, hfun](0.0, 1.0)

    Integrates `func(y, x)` over `y` in `[gfun(x), hfun(x)]` and `x` in
    `[a, b]`, matching `scipy.integrate.dblquad`'s argument order (`y`
    before `x`, the innermost variable first), via fixed-order
    tensor-product Gauss-Legendre quadrature (`n` points per axis - see
    the module docstring for why this is fixed-order rather than adaptive).

    Parameters:
        dtype: The floating-point data type.
        func: Integrand with signature `def(y, x, args) -> Scalar[dtype]`.
        gfun: Lower bound of y as a function of x.
        hfun: Upper bound of y as a function of x.

    Args:
        a: Lower integration limit for x.
        b: Upper integration limit for x.
        args: Optional arguments to pass to `func`.
        n: The order of the Gauss-Legendre rule (number of points per
            axis), n >= 1.

    Returns:
        IntegralResult[dtype] containing the integral value and the total
        function evaluation count. `abserr` is not estimated (left at 0);
        `ier` is 0 on success.

    Raises:
        Error: If n < 1.
    """
    var nw_x = _gauss_legendre_nodes_weights[dtype](n)
    ref nodes_x = nw_x[0]
    ref weights_x = nw_x[1]
    var half_x = (b - a) * Scalar[dtype](0.5)
    var center_x = (b + a) * Scalar[dtype](0.5)

    var nw_y = _gauss_legendre_nodes_weights[dtype](n)
    ref nodes_y = nw_y[0]
    ref weights_y = nw_y[1]

    var total: Scalar[dtype] = 0.0
    for i in range(n):
        var x = center_x + half_x * nodes_x[i]
        var lo = gfun(x)
        var hi = hfun(x)
        var half_y = (hi - lo) * Scalar[dtype](0.5)
        var center_y = (hi + lo) * Scalar[dtype](0.5)

        var inner_total: Scalar[dtype] = 0.0
        for j in range(n):
            var y = center_y + half_y * nodes_y[j]
            inner_total += weights_y[j] * func(y, x, args)

        total += weights_x[i] * (inner_total * half_y)

    return IntegralResult[dtype](
        integral=total * half_x, abserr=0, nfev=n * n, ier=0
    )


def dblquad[
    dtype: DType,
    func: def[dtype: DType](y: Scalar[dtype], x: Scalar[dtype]) capturing[
        _
    ] -> Scalar[dtype],
    gfun: def[dtype: DType](x: Scalar[dtype]) thin capturing[_] -> Scalar[
        dtype
    ] where dtype.is_floating_point(),
    hfun: def[dtype: DType](x: Scalar[dtype]) thin capturing[_] -> Scalar[
        dtype
    ] where dtype.is_floating_point(),
](
    a: Scalar[dtype],
    b: Scalar[dtype],
    n: Int = 30,
) raises -> IntegralResult[
    dtype
] where dtype.is_floating_point():
    """Computes a double integral over a region bounded by functions of x
    (args-free overload).

    Example: result = dblquad[f64, func, gfun, hfun](0.0, 1.0)

    See the `args`-taking overload for the algorithm.

    Parameters:
        dtype: The floating-point data type.
        func: Integrand with signature `def(y, x) -> Scalar[dtype]`.
        gfun: Lower bound of y as a function of x.
        hfun: Upper bound of y as a function of x.

    Args:
        a: Lower integration limit for x.
        b: Upper integration limit for x.
        n: The order of the Gauss-Legendre rule (number of points per
            axis), n >= 1.

    Returns:
        IntegralResult[dtype] containing the integral value and the total
        function evaluation count. `abserr` is not estimated (left at 0);
        `ier` is 0 on success.
    """

    def _with_args[
        dtype2: DType
    ](
        y: Scalar[dtype2],
        x: Scalar[dtype2],
        args: Optional[List[Scalar[dtype2]]],
    ) capturing -> Scalar[dtype2]:
        return func(y, x)

    return dblquad[dtype, _with_args, gfun, hfun](a=a, b=b, args=None, n=n)


# ===----------------------------------------------------------------------=== #
# tplquad
# ===----------------------------------------------------------------------=== #


def tplquad[
    dtype: DType,
    func: def[dtype: DType](
        z: Scalar[dtype],
        y: Scalar[dtype],
        x: Scalar[dtype],
        args: Optional[List[Scalar[dtype]]],
    ) thin capturing[_] -> Scalar[dtype] where dtype.is_floating_point(),
    qfun: def[dtype: DType](x: Scalar[dtype], y: Scalar[dtype]) thin capturing[
        _
    ] -> Scalar[dtype] where dtype.is_floating_point(),
    rfun: def[dtype: DType](x: Scalar[dtype], y: Scalar[dtype]) thin capturing[
        _
    ] -> Scalar[dtype] where dtype.is_floating_point(),
    gfun: def[dtype: DType](x: Scalar[dtype]) thin capturing[_] -> Scalar[
        dtype
    ] where dtype.is_floating_point(),
    hfun: def[dtype: DType](x: Scalar[dtype]) thin capturing[_] -> Scalar[
        dtype
    ] where dtype.is_floating_point(),
](
    a: Scalar[dtype],
    b: Scalar[dtype],
    args: Optional[List[Scalar[dtype]]] = None,
    n: Int = 20,
) raises -> IntegralResult[dtype] where dtype.is_floating_point():
    """Computes a triple integral over a region bounded by functions of x and y.

    Example: result = tplquad[f64, func, qfun, rfun, gfun, hfun](0.0, 1.0)

    Integrates `func(z, y, x)` over `z` in `[qfun(x, y), rfun(x, y)]`, `y`
    in `[gfun(x), hfun(x)]`, and `x` in `[a, b]` - matching
    `scipy.integrate.tplquad`'s argument order (innermost variable first),
    via fixed-order tensor-product Gauss-Legendre quadrature (`n` points
    per axis - see the module docstring for why this is fixed-order rather
    than adaptive).

    Parameters:
        dtype: The floating-point data type.
        func: Integrand with signature `def(z, y, x, args) -> Scalar[dtype]`.
        qfun: Lower bound of z as a function of (x, y).
        rfun: Upper bound of z as a function of (x, y).
        gfun: Lower bound of y as a function of x.
        hfun: Upper bound of y as a function of x.

    Args:
        a: Lower integration limit for x.
        b: Upper integration limit for x.
        args: Optional arguments to pass to `func`.
        n: The order of the Gauss-Legendre rule (number of points per
            axis), n >= 1.

    Returns:
        IntegralResult[dtype] containing the integral value and the total
        function evaluation count. `abserr` is not estimated (left at 0);
        `ier` is 0 on success.

    Raises:
        Error: If n < 1.
    """
    var nw_x = _gauss_legendre_nodes_weights[dtype](n)
    ref nodes_x = nw_x[0]
    ref weights_x = nw_x[1]
    var half_x = (b - a) * Scalar[dtype](0.5)
    var center_x = (b + a) * Scalar[dtype](0.5)

    var nw_y = _gauss_legendre_nodes_weights[dtype](n)
    ref nodes_y = nw_y[0]
    ref weights_y = nw_y[1]

    var nw_z = _gauss_legendre_nodes_weights[dtype](n)
    ref nodes_z = nw_z[0]
    ref weights_z = nw_z[1]

    var total: Scalar[dtype] = 0.0
    for i in range(n):
        var x = center_x + half_x * nodes_x[i]
        var y_lo = gfun(x)
        var y_hi = hfun(x)
        var half_y = (y_hi - y_lo) * Scalar[dtype](0.5)
        var center_y = (y_hi + y_lo) * Scalar[dtype](0.5)

        var mid_total: Scalar[dtype] = 0.0
        for j in range(n):
            var y = center_y + half_y * nodes_y[j]
            var z_lo = qfun(x, y)
            var z_hi = rfun(x, y)
            var half_z = (z_hi - z_lo) * Scalar[dtype](0.5)
            var center_z = (z_hi + z_lo) * Scalar[dtype](0.5)

            var inner_total: Scalar[dtype] = 0.0
            for k in range(n):
                var z = center_z + half_z * nodes_z[k]
                inner_total += weights_z[k] * func(z, y, x, args)

            mid_total += weights_y[j] * (inner_total * half_z)

        total += weights_x[i] * (mid_total * half_y)

    return IntegralResult[dtype](
        integral=total * half_x, abserr=0, nfev=n * n * n, ier=0
    )


def tplquad[
    dtype: DType,
    func: def[dtype: DType](
        z: Scalar[dtype], y: Scalar[dtype], x: Scalar[dtype]
    ) capturing[_] -> Scalar[dtype],
    qfun: def[dtype: DType](x: Scalar[dtype], y: Scalar[dtype]) thin capturing[
        _
    ] -> Scalar[dtype] where dtype.is_floating_point(),
    rfun: def[dtype: DType](x: Scalar[dtype], y: Scalar[dtype]) thin capturing[
        _
    ] -> Scalar[dtype] where dtype.is_floating_point(),
    gfun: def[dtype: DType](x: Scalar[dtype]) thin capturing[_] -> Scalar[
        dtype
    ] where dtype.is_floating_point(),
    hfun: def[dtype: DType](x: Scalar[dtype]) thin capturing[_] -> Scalar[
        dtype
    ] where dtype.is_floating_point(),
](
    a: Scalar[dtype],
    b: Scalar[dtype],
    n: Int = 20,
) raises -> IntegralResult[
    dtype
] where dtype.is_floating_point():
    """Computes a triple integral over a region bounded by functions of x and y
    (args-free overload).

    Example: result = tplquad[f64, func, qfun, rfun, gfun, hfun](0.0, 1.0)

    See the `args`-taking overload for the algorithm.

    Parameters:
        dtype: The floating-point data type.
        func: Integrand with signature `def(z, y, x) -> Scalar[dtype]`.
        qfun: Lower bound of z as a function of (x, y).
        rfun: Upper bound of z as a function of (x, y).
        gfun: Lower bound of y as a function of x.
        hfun: Upper bound of y as a function of x.

    Args:
        a: Lower integration limit for x.
        b: Upper integration limit for x.
        n: The order of the Gauss-Legendre rule (number of points per
            axis), n >= 1.

    Returns:
        IntegralResult[dtype] containing the integral value and the total
        function evaluation count. `abserr` is not estimated (left at 0);
        `ier` is 0 on success.
    """

    def _with_args[
        dtype2: DType
    ](
        z: Scalar[dtype2],
        y: Scalar[dtype2],
        x: Scalar[dtype2],
        args: Optional[List[Scalar[dtype2]]],
    ) capturing -> Scalar[dtype2]:
        return func(z, y, x)

    return tplquad[dtype, _with_args, qfun, rfun, gfun, hfun](
        a=a, b=b, args=None, n=n
    )


# ===----------------------------------------------------------------------=== #
# nquad
# ===----------------------------------------------------------------------=== #


def _nquad_rec[
    dtype: DType,
    func: def[dtype: DType](
        x: List[Scalar[dtype]], args: Optional[List[Scalar[dtype]]]
    ) thin capturing[_] -> Scalar[dtype] where dtype.is_floating_point(),
](
    ranges: List[Tuple[Scalar[dtype], Scalar[dtype]]],
    coords: List[Scalar[dtype]],
    dim: Int,
    ndim: Int,
    args: Optional[List[Scalar[dtype]]],
    n: Int,
) raises -> Scalar[dtype] where dtype.is_floating_point():
    """Recursively integrates `func` over `ranges[dim:]`, holding
    `coords[:dim]` fixed, via one fixed-order (`n`-point Gauss-Legendre)
    tensor-product axis per remaining dimension.
    """
    if dim == ndim:
        return func(coords, args)

    var bounds = ranges[dim]
    var nw = _gauss_legendre_nodes_weights[dtype](n)
    ref nodes = nw[0]
    ref weights = nw[1]
    var half = (bounds[1] - bounds[0]) * Scalar[dtype](0.5)
    var center = (bounds[1] + bounds[0]) * Scalar[dtype](0.5)

    var total: Scalar[dtype] = 0.0
    for i in range(n):
        var x = center + half * nodes[i]
        var next_coords = coords.copy()
        next_coords.append(x)
        total += weights[i] * _nquad_rec[dtype, func](
            ranges, next_coords, dim + 1, ndim, args, n
        )

    return total * half


def nquad[
    dtype: DType,
    func: def[dtype: DType](
        x: List[Scalar[dtype]], args: Optional[List[Scalar[dtype]]]
    ) thin capturing[_] -> Scalar[dtype] where dtype.is_floating_point(),
](
    ranges: List[Tuple[Scalar[dtype], Scalar[dtype]]],
    args: Optional[List[Scalar[dtype]]] = None,
    n: Int = 20,
) raises -> IntegralResult[dtype] where dtype.is_floating_point():
    """Integrates a scalar function of N variables over a hyperrectangle.

    Example: result = nquad[f64, func](ranges)

    Integrates `func(x, args)`, where `x` has length `len(ranges)`, over
    the hyperrectangle `ranges[0] x ranges[1] x ... x ranges[-1]`, via
    fixed-order tensor-product Gauss-Legendre quadrature (`n` points per
    axis, recursing one axis at a time, outermost first - see the module
    docstring for why this is fixed-order rather than adaptive).

    Scoped simplification: only constant per-axis bounds are supported
    here (`ranges` is a plain list of `(lo, hi)` pairs), not SciPy's
    `nquad` generality of bounds given as callables of the outer
    variables. `dblquad`/`tplquad` already cover that fuller form for 2
    and 3 dimensions.

    Parameters:
        dtype: The floating-point data type.
        func: Integrand with signature `def(x, args) -> Scalar[dtype]`,
            where `x` is a list of length `len(ranges)`.

    Args:
        ranges: A list of `(lo, hi)` bounds, one per integration variable.
        args: Optional arguments to pass to `func`.
        n: The order of the Gauss-Legendre rule (number of points per
            axis), n >= 1.

    Returns:
        IntegralResult[dtype] containing the integral value. The
        `abserr`/`nfev` fields are not meaningfully aggregated across the
        nested recursion and are left at 0; `ier` is 0 on success.

    Raises:
        Error: If ranges is empty.
    """
    var ndim = len(ranges)
    if ndim == 0:
        raise Error("nquad requires at least one integration range")

    var coords = List[Scalar[dtype]]()
    var integral = _nquad_rec[dtype, func](ranges, coords, 0, ndim, args, n)
    return IntegralResult[dtype](integral=integral, abserr=0, nfev=0, ier=0)
