# ===----------------------------------------------------------------------=== #
# SciJo: Integrate module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Quadrature Integration (`scijo.integrate.quadrature`)
========================================================
General-purpose numerical integration. This module exposes `quad`,
dispatching to QNG, QAG, or QAGS quadrature, all built on the
Gauss-Kronrod rules and adaptive-subdivision priority queue in
`scijo.integrate.utility`.

References:
    - Netlib QUADPACK: https://www.netlib.org/quadpack/
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.math import pow

# ===----------------------------------------------------------------------=== #
# SciJo
# ===----------------------------------------------------------------------=== #
from scijo.integrate.utility import (
    gk15_gauss_weights,
    gk15_kronrod_weights,
    gk15_nodes,
    gk21_gauss_weights,
    gk21_kronrod_weights,
    gk21_nodes,
    gk43_gauss21_weights,
    gk43_kronrod_weights,
    gk43_nodes,
    gk87_gauss43_weights,
    gk87_kronrod_weights,
    gk87_nodes,
    IntegralResult,
    machine_epsilon,
    QAGSInterval,
    QAGSPriorityQueue,
)

# ===----------------------------------------------------------------------=== #
# QAG rule aliases
# ===----------------------------------------------------------------------=== #

comptime QAG_GK15: Int = 1
"""7-point Gauss / 15-point Kronrod rule for `quad[..., method="qag"]`."""
comptime QAG_GK21: Int = 2
"""10-point Gauss / 21-point Kronrod rule (default) for `quad[..., method="qag"]`."""
comptime QAG_GK31: Int = 3
"""15-point Gauss / 31-point Kronrod rule for `quad[..., method="qag"]`."""
comptime QAG_GK41: Int = 4
"""20-point Gauss / 41-point Kronrod rule for `quad[..., method="qag"]`."""
comptime QAG_GK51: Int = 5
"""25-point Gauss / 51-point Kronrod rule for `quad[..., method="qag"]`."""
comptime QAG_GK61: Int = 6
"""30-point Gauss / 61-point Kronrod rule for `quad[..., method="qag"]`."""

# ===----------------------------------------------------------------------=== #
# Single-interval Gauss-Kronrod evaluation
# ===----------------------------------------------------------------------=== #


struct _GKResult[dtype: DType](Copyable, Movable):
    """Result of evaluating one Gauss-Kronrod rule over a single interval."""

    var result: Scalar[Self.dtype]
    """Kronrod (higher-order) estimate of the integral over the interval."""
    var abserr: Scalar[Self.dtype]
    """Estimated absolute error (rescaled |Kronrod - embedded Gauss|)."""
    var resabs: Scalar[Self.dtype]
    """Kronrod-weighted integral of |f|, used for error rescaling."""
    var resasc: Scalar[Self.dtype]
    """Kronrod-weighted integral of |f - mean(f)|, used for error rescaling."""
    var nfev: Int
    """Number of function evaluations used by this rule."""

    def __init__(
        out self,
        result: Scalar[Self.dtype] = 0,
        abserr: Scalar[Self.dtype] = 0,
        resabs: Scalar[Self.dtype] = 0,
        resasc: Scalar[Self.dtype] = 0,
        nfev: Int = 0,
    ):
        """Constructs a _GKResult from a single Gauss-Kronrod rule evaluation.

        Args:
            result: Kronrod (higher-order) estimate of the integral.
            abserr: Estimated absolute error.
            resabs: Kronrod-weighted integral of |f|.
            resasc: Kronrod-weighted integral of |f - mean(f)|.
            nfev: Number of function evaluations used.
        """
        self.result = result
        self.abserr = abserr
        self.resabs = resabs
        self.resasc = resasc
        self.nfev = nfev


def _rescale_error[
    dtype: DType
](
    err: Scalar[dtype], result_abs: Scalar[dtype], result_asc: Scalar[dtype]
) -> Scalar[dtype] where dtype.is_floating_point():
    """QUADPACK's standard error-rescaling heuristic (Piessens et al. 1983):
    shrinks the raw |Kronrod - Gauss| difference when it is small relative to
    the oscillation of `f` on the interval, and floors it at roughly 50 ULPs
    of `resabs` to avoid reporting an error estimate below machine precision.
    """
    var e = abs(err)
    var eps = Scalar[dtype](machine_epsilon[dtype]())
    if result_asc != 0 and e != 0:
        var scale = pow(200.0 * e / result_asc, Scalar[dtype](1.5))
        if scale < 1.0:
            e = result_asc * scale
        else:
            e = result_asc
    var tiny = Scalar[dtype](1e-292)
    if result_abs > tiny / (50.0 * eps):
        var min_err = 50.0 * eps * result_abs
        if min_err > e:
            e = min_err
    return e


def _gk_eval[
    dtype: DType,
    fn_: def(x: Scalar[dtype]) capturing[_] -> Scalar[dtype],
](
    nodes: List[Float64],
    kw: List[Float64],
    gw: List[Float64],
    a: Scalar[dtype],
    b: Scalar[dtype],
) raises -> _GKResult[dtype] where dtype.is_floating_point():
    """Evaluates a Gauss-Kronrod rule over `[a, b]`.

    `nodes[0]` is the center (0.0); `nodes[1:]` are the ascending positive
    half-nodes, each contributing `f(center - x)` and `f(center + x)`.
    `kw` holds the Kronrod weight for each node. `gw` holds the embedded
    lower-order Gauss weight for each node, or 0.0 where that node isn't
    part of the embedded rule (e.g. the 10-point Gauss rule embedded in the
    21-point Kronrod rule has no center node).
    """
    var n = len(nodes)
    var center = (a + b) * Scalar[dtype](0.5)
    var half_length = (b - a) * Scalar[dtype](0.5)
    var abs_half = abs(half_length)

    var f_center = fn_(center)
    var result_kronrod = f_center * Scalar[dtype](kw[0])
    var result_gauss = f_center * Scalar[dtype](gw[0])
    var result_abs = abs(result_kronrod)

    var fv1 = List[Scalar[dtype]](capacity=n - 1)
    var fv2 = List[Scalar[dtype]](capacity=n - 1)
    for i in range(1, n):
        var x = half_length * Scalar[dtype](nodes[i])
        var f1 = fn_(center - x)
        var f2 = fn_(center + x)
        fv1.append(f1)
        fv2.append(f2)
        var fsum = f1 + f2
        result_kronrod += Scalar[dtype](kw[i]) * fsum
        result_gauss += Scalar[dtype](gw[i]) * fsum
        result_abs += Scalar[dtype](kw[i]) * (abs(f1) + abs(f2))

    var mean = result_kronrod * Scalar[dtype](0.5)
    var result_asc = Scalar[dtype](kw[0]) * abs(f_center - mean)
    for i in range(1, n):
        result_asc += Scalar[dtype](kw[i]) * (
            abs(fv1[i - 1] - mean) + abs(fv2[i - 1] - mean)
        )

    var err = (result_kronrod - result_gauss) * half_length

    var res = _GKResult[dtype]()
    res.result = result_kronrod * half_length
    res.resabs = result_abs * abs_half
    res.resasc = result_asc * abs_half
    res.abserr = _rescale_error[dtype](err, res.resabs, res.resasc)
    res.nfev = 2 * (n - 1) + 1
    return res^


def _qk15[
    dtype: DType,
    fn_: def(x: Scalar[dtype]) capturing[_] -> Scalar[dtype],
](a: Scalar[dtype], b: Scalar[dtype]) raises -> _GKResult[
    dtype
] where dtype.is_floating_point():
    """7-point Gauss / 15-point Kronrod rule."""
    ref nds = materialize[gk15_nodes]()
    ref kws = materialize[gk15_kronrod_weights]()
    ref gws = materialize[gk15_gauss_weights]()
    var nodes = List[Float64]()
    var kw = List[Float64]()
    var gw = List[Float64]()
    for i in range(8):
        nodes.append(nds[i])
        kw.append(kws[i])
        gw.append(0.0)
    # Gauss-7 has a center node: indices 0, 2, 4, 6.
    gw[0] = gws[0]
    gw[2] = gws[1]
    gw[4] = gws[2]
    gw[6] = gws[3]
    return _gk_eval[dtype, fn_](nodes, kw, gw, a, b)


def _qk21[
    dtype: DType,
    fn_: def(x: Scalar[dtype]) capturing[_] -> Scalar[dtype],
](a: Scalar[dtype], b: Scalar[dtype]) raises -> _GKResult[
    dtype
] where dtype.is_floating_point():
    """10-point Gauss / 21-point Kronrod rule."""
    ref nds = materialize[gk21_nodes]()
    ref kws = materialize[gk21_kronrod_weights]()
    ref gws = materialize[gk21_gauss_weights]()
    var nodes = List[Float64]()
    var kw = List[Float64]()
    var gw = List[Float64]()
    for i in range(11):
        nodes.append(nds[i])
        kw.append(kws[i])
        gw.append(0.0)
    # Gauss-10 has no center node: indices 1, 3, 5, 7, 9.
    gw[1] = gws[0]
    gw[3] = gws[1]
    gw[5] = gws[2]
    gw[7] = gws[3]
    gw[9] = gws[4]
    return _gk_eval[dtype, fn_](nodes, kw, gw, a, b)


def _qk43[
    dtype: DType,
    fn_: def(x: Scalar[dtype]) capturing[_] -> Scalar[dtype],
](a: Scalar[dtype], b: Scalar[dtype]) raises -> _GKResult[
    dtype
] where dtype.is_floating_point():
    """21-point Gauss / 43-point Kronrod rule."""
    ref nds = materialize[gk43_nodes]()
    ref kws = materialize[gk43_kronrod_weights]()
    ref gws = materialize[gk43_gauss21_weights]()
    var nodes = List[Float64]()
    var kw = List[Float64]()
    var gw = List[Float64]()
    for i in range(22):
        nodes.append(nds[i])
        kw.append(kws[i])
        gw.append(0.0)
    # Gauss-21 has a center node: indices 0, 2, 4, ..., 20.
    for j in range(11):
        gw[2 * j] = gws[j]
    return _gk_eval[dtype, fn_](nodes, kw, gw, a, b)


def _qk87[
    dtype: DType,
    fn_: def(x: Scalar[dtype]) capturing[_] -> Scalar[dtype],
](a: Scalar[dtype], b: Scalar[dtype]) raises -> _GKResult[
    dtype
] where dtype.is_floating_point():
    """43-point Gauss / 87-point Kronrod rule."""
    ref nds = materialize[gk87_nodes]()
    ref kws = materialize[gk87_kronrod_weights]()
    ref gws = materialize[gk87_gauss43_weights]()
    var nodes = List[Float64]()
    var kw = List[Float64]()
    var gw = List[Float64]()
    for i in range(44):
        nodes.append(nds[i])
        kw.append(kws[i])
        gw.append(0.0)
    # Gauss-43 has a center node: indices 0, 2, 4, ..., 42.
    for j in range(22):
        gw[2 * j] = gws[j]
    return _gk_eval[dtype, fn_](nodes, kw, gw, a, b)


def _qk_dispatch[
    dtype: DType,
    fn_: def(x: Scalar[dtype]) capturing[_] -> Scalar[dtype],
](rule: Int, a: Scalar[dtype], b: Scalar[dtype]) raises -> _GKResult[
    dtype
] where dtype.is_floating_point():
    """Maps a `qag_rule` value to the closest available Gauss-Kronrod rule.

    Only 15-, 21-, 43-, and 87-point rules are tabulated (the same tables
    `qng` uses for its accuracy cascade), so the 31- and 41-point requests
    are served by the 43-point rule and the 51- and 61-point requests by
    the 87-point rule - both strictly more accurate than what was asked
    for, so results still meet the requested tolerance.
    """
    if rule <= QAG_GK15:
        return _qk15[dtype, fn_](a, b)
    elif rule == QAG_GK21:
        return _qk21[dtype, fn_](a, b)
    elif rule == QAG_GK31 or rule == QAG_GK41:
        return _qk43[dtype, fn_](a, b)
    else:
        return _qk87[dtype, fn_](a, b)


# ===----------------------------------------------------------------------=== #
# QNG: non-adaptive Gauss-Kronrod-Patterson cascade
# ===----------------------------------------------------------------------=== #


def _qng_impl[
    dtype: DType,
    fn_: def(x: Scalar[dtype]) capturing[_] -> Scalar[dtype],
](
    a: Scalar[dtype],
    b: Scalar[dtype],
    epsabs: Scalar[dtype],
    epsrel: Scalar[dtype],
) raises -> IntegralResult[dtype] where dtype.is_floating_point():
    """Tries increasingly accurate fixed Gauss-Kronrod rules (21, 43, then
    87 points) over the whole interval, accepting the first whose estimated
    error meets the requested tolerance. Each rule's embedded lower-order
    Gauss estimate equals the *previous* rule's Kronrod estimate, so this
    reproduces QUADPACK's `qng` cascade exactly.
    """

    var r1 = _qk21[dtype, fn_](a, b)
    if r1.abserr <= _qag_tol(epsabs, epsrel, r1.result):
        return IntegralResult[dtype](
            integral=r1.result, abserr=r1.abserr, nfev=r1.nfev, ier=0
        )

    var r2 = _qk43[dtype, fn_](a, b)
    var nfev2 = r1.nfev + r2.nfev
    if r2.abserr <= _qag_tol(epsabs, epsrel, r2.result):
        return IntegralResult[dtype](
            integral=r2.result, abserr=r2.abserr, nfev=nfev2, ier=0
        )

    var r3 = _qk87[dtype, fn_](a, b)
    var nfev3 = nfev2 + r3.nfev
    var ier = 0 if r3.abserr <= _qag_tol(epsabs, epsrel, r3.result) else 1
    return IntegralResult[dtype](
        integral=r3.result, abserr=r3.abserr, nfev=nfev3, ier=ier
    )


# ===----------------------------------------------------------------------=== #
# QAG / QAGS: globally-adaptive subdivision
# ===----------------------------------------------------------------------=== #


def _qag_impl[
    dtype: DType,
    fn_: def(x: Scalar[dtype]) capturing[_] -> Scalar[dtype],
](
    a: Scalar[dtype],
    b: Scalar[dtype],
    epsabs: Scalar[dtype],
    epsrel: Scalar[dtype],
    limit: Int,
    qag_rule: Int,
    extrapolate: Bool,
) raises -> IntegralResult[dtype] where dtype.is_floating_point():
    """Globally-adaptive quadrature: repeatedly bisects the subinterval with
    the largest error estimate, tracked via a max-error priority queue, until
    the total estimated error meets the requested tolerance or `limit`
    subintervals have been used. When `extrapolate` is set (`method="qags"`),
    Wynn's epsilon algorithm is additionally applied to the sequence of
    running integral estimates to accelerate convergence on integrands that
    are difficult for plain subdivision (e.g. those with a mild endpoint
    singularity).
    """
    var r0 = _qk_dispatch[dtype, fn_](qag_rule, a, b)
    var total_integral = r0.result
    var total_error = r0.abserr
    var nfev = r0.nfev

    var pq = QAGSPriorityQueue[dtype]()
    pq.push(
        QAGSInterval[dtype](
            Float64(a), Float64(b), Float64(r0.result), Float64(r0.abserr)
        )
    )

    var history = List[Scalar[dtype]]()
    history.append(total_integral)

    var n_intervals = 1
    while (
        total_error > _qag_tol(epsabs, epsrel, total_integral)
        and n_intervals < limit
    ):
        var worst = pq.pop()
        total_integral -= Scalar[dtype](worst.integral)
        total_error -= Scalar[dtype](worst.error)

        var mid = Scalar[dtype]((worst.a + worst.b) * 0.5)
        var r1 = _qk_dispatch[dtype, fn_](qag_rule, Scalar[dtype](worst.a), mid)
        var r2 = _qk_dispatch[dtype, fn_](qag_rule, mid, Scalar[dtype](worst.b))
        nfev += r1.nfev + r2.nfev

        total_integral += r1.result + r2.result
        total_error += r1.abserr + r2.abserr

        pq.push(
            QAGSInterval[dtype](
                worst.a,
                Float64(mid),
                Float64(r1.result),
                Float64(r1.abserr),
                worst.level + 1,
            )
        )
        pq.push(
            QAGSInterval[dtype](
                Float64(mid),
                worst.b,
                Float64(r2.result),
                Float64(r2.abserr),
                worst.level + 1,
            )
        )
        n_intervals += 1
        history.append(total_integral)

        if extrapolate and len(history) >= 4:
            var extrap = _wynn_epsilon[dtype](history)
            if extrap[1] < total_error:
                total_error = extrap[1]

    var final_integral = total_integral
    var final_error = total_error
    if extrapolate and len(history) >= 4:
        var extrap = _wynn_epsilon[dtype](history)
        if extrap[1] <= total_error:
            final_integral = extrap[0]
            final_error = extrap[1]

    var ier = (
        0 if final_error <= _qag_tol(epsabs, epsrel, final_integral) else 1
    )
    return IntegralResult[dtype](
        integral=final_integral, abserr=final_error, nfev=nfev, ier=ier
    )


def _qag_tol[
    dtype: DType
](epsabs: Scalar[dtype], epsrel: Scalar[dtype], value: Scalar[dtype]) -> Scalar[
    dtype
] where dtype.is_floating_point():
    var rel = epsrel * abs(value)
    return epsabs if epsabs > rel else rel


def _wynn_epsilon[
    dtype: DType
](seq: List[Scalar[dtype]]) -> Tuple[
    Scalar[dtype], Scalar[dtype]
] where dtype.is_floating_point():
    """Applies Wynn's epsilon algorithm (Wynn, 1956) to accelerate
    convergence of the partial-sum sequence `seq`, returning an
    (extrapolated_value, error_estimate) pair. A standard technique for
    accelerating slowly-converging sequences, independent of any particular
    quadrature package.
    """
    var n = len(seq)
    var tiny = Scalar[dtype](1e-300)
    var huge = Scalar[dtype](1e30)
    if n < 3:
        return (seq[n - 1], huge)

    var eps = List[Scalar[dtype]]()
    for v in seq:
        eps.append(v)

    var result = eps[len(eps) - 1]
    var abserr = huge
    while len(eps) >= 3:
        var m = len(eps)
        var next_eps = List[Scalar[dtype]](capacity=m - 2)
        var singular = False
        for i in range(m - 2):
            var diff = eps[i + 2] - eps[i + 1]
            if abs(diff) < tiny:
                singular = True
                break
            next_eps.append(eps[i] + 1.0 / diff)
        if singular:
            break
        eps = next_eps^
        var new_m = len(eps)
        if new_m >= 1:
            result = eps[new_m - 1]
        if new_m >= 2:
            abserr = abs(eps[new_m - 1] - eps[new_m - 2])

    return (result, abserr)


# ===----------------------------------------------------------------------=== #
# Public frontend
# ===----------------------------------------------------------------------=== #


def quad[
    dtype: DType,
    integrand_func: def[dtype: DType](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) thin capturing[_] -> Scalar[dtype] where dtype.is_floating_point(),
    *,
    method: String = "qng",
    qag_rule: Int = 2,
](
    a: Scalar[dtype],
    b: Scalar[dtype],
    args: Optional[List[Scalar[dtype]]] = None,
    atol: Scalar[dtype] = 1.49e-8,
    rtol: Scalar[dtype] = 1.49e-8,
    limit: Int = 50,
) raises -> IntegralResult[dtype] where dtype.is_floating_point():
    """Computes the definite integral of a scalar function over [a, b].

    Parameters:
        dtype: The floating-point data type.
        integrand_func: Integrand function with signature
            `def(x, args) -> Scalar[dtype]`.
        method: Quadrature method name. Supports `"qng"`, `"qag"`, and `"qags"`.
        qag_rule: Gauss-Kronrod rule key used when `method="qag"`.

    Args:
        a: Lower integration limit.
        b: Upper integration limit.
        args: Optional arguments to pass to the integrand function.
        atol: Absolute error tolerance.
        rtol: Relative error tolerance.
        limit: Maximum number of subintervals for adaptive methods (`qag`, `qags`).

    Returns:
        IntegralResult[dtype] containing integral value, absolute error
        estimate, and status fields.

    Raises:
        Error: If an unsupported integration method is requested or if `limit <= 0`.

    NOTES:
        Method `"qng"` performs non-adaptive Gauss-Kronrod-Patterson quadrature.
        Method `"qag"` performs adaptive Gauss-Kronrod integration.
        Method `"qags"` performs adaptive integration with Wynn epsilon extrapolation.
    """

    if limit <= 0:
        raise Error(
            "Scijo [quad]: Invalid limit. Expected limit > 0, got "
            + String(limit)
        )

    if a == b:
        return IntegralResult[dtype](integral=0, abserr=0, nfev=0, ier=0)

    var sign: Scalar[dtype] = 1
    var lo = a
    var hi = b
    if lo > hi:
        var tmp = lo
        lo = hi
        hi = tmp
        sign = -1

    def wrapped_fn(x: Scalar[dtype]) capturing -> Scalar[dtype]:
        return integrand_func(x, args)

    comptime if method == "qng":
        var result = _qng_impl[dtype, wrapped_fn](lo, hi, atol, rtol)
        return IntegralResult[dtype](
            integral=sign * result.integral,
            abserr=result.abserr,
            nfev=result.nfev,
            ier=result.ier,
        )
    elif method == "qag":
        var result = _qag_impl[dtype, wrapped_fn](
            lo, hi, atol, rtol, limit, qag_rule, extrapolate=False
        )
        return IntegralResult[dtype](
            integral=sign * result.integral,
            abserr=result.abserr,
            nfev=result.nfev,
            ier=result.ier,
        )
    elif method == "qags":
        var result = _qag_impl[dtype, wrapped_fn](
            lo, hi, atol, rtol, limit, qag_rule, extrapolate=True
        )
        return IntegralResult[dtype](
            integral=sign * result.integral,
            abserr=result.abserr,
            nfev=result.nfev,
            ier=result.ier,
        )
    else:
        raise Error(
            "Unsupported quad method: "
            + String(method)
            + ". Supported methods: 'qng', 'qag', 'qags'."
        )


def quad[
    dtype: DType,
    integrand_func: def[dtype: DType](x: Scalar[dtype]) capturing[_] -> Scalar[
        dtype
    ],
    *,
    method: String = "qng",
    qag_rule: Int = 2,
](
    a: Scalar[dtype],
    b: Scalar[dtype],
    atol: Scalar[dtype] = 1.49e-8,
    rtol: Scalar[dtype] = 1.49e-8,
    limit: Int = 50,
) raises -> IntegralResult[dtype] where dtype.is_floating_point():
    """Computes the definite integral of a scalar function over [a, b] (args-free overload).

    For integrands that do not need `args`: `integrand_func` takes only
    `x`, so extra parameters (if any) are captured directly from the
    enclosing scope instead of threaded through `args`. See the
    two-parameter overload `quad[dtype, integrand_func](..., args=...)`
    for a function reused across call sites with different `args` values.

    Parameters:
        dtype: The floating-point data type.
        integrand_func: Integrand function with signature `def(x) -> Scalar[dtype]`.
        method: Quadrature method name. Supports `"qng"`, `"qag"`, and `"qags"`.
        qag_rule: Gauss-Kronrod rule key used when `method="qag"`.

    Args:
        a: Lower integration limit.
        b: Upper integration limit.
        atol: Absolute error tolerance.
        rtol: Relative error tolerance.
        limit: Maximum number of subintervals for adaptive methods (`qag`, `qags`).

    Returns:
        IntegralResult[dtype] containing integral value, absolute error
        estimate, and status fields.

    Raises:
        Error: If an unsupported integration method is requested or if `limit <= 0`.
    """

    def _with_args[
        dtype2: DType
    ](
        x: Scalar[dtype2], args: Optional[List[Scalar[dtype2]]]
    ) capturing -> Scalar[dtype2]:
        return integrand_func(x)

    return quad[dtype, _with_args, method=method, qag_rule=qag_rule](
        a=a, b=b, atol=atol, rtol=rtol, limit=limit
    )
