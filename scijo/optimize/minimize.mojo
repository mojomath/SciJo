# ===----------------------------------------------------------------------=== #
# SciJo: Optimize module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Multi-Dimensional Minimization (`scijo.optimize.minimize`)
=============================================================
Minimization for scalar-valued functions of several variables.

`method="Powell"` and `method="L-BFGS-B"` additionally accept a `bounds`
argument (a `(lo, hi)` pair of shape-`(n,)` arrays) for simple box
constraints. `"L-BFGS-B"` here is a projected-gradient variant of
limited-memory BFGS (two-loop recursion, projected Armijo line search,
convergence on the projected-gradient norm) - not the full active-set
algorithm of Byrd, Lu, Nocedal & Zhu (1995); it is correct but does not
replicate that paper's active-set bookkeeping.
"""

# ===----------------------------------------------------------------------=== #
# External
# ===----------------------------------------------------------------------=== #
from numojo.core.ndarray import NDArray
from numojo.core.type_aliases import Shape
from numojo.routines.creation import zeros

# ===----------------------------------------------------------------------=== #
# SciJo
# ===----------------------------------------------------------------------=== #
from scijo.linalg.norms import norm
from scijo.optimize.utility import OptimizeResultVector

comptime _nm_alpha: Float64 = 1.0
"""Reflection coefficient."""
comptime _nm_gamma: Float64 = 2.0
"""Expansion coefficient."""
comptime _nm_rho: Float64 = 0.5
"""Contraction coefficient."""
comptime _nm_sigma: Float64 = 0.5
"""Shrink coefficient."""

comptime _ls_c1: Float64 = 1e-4
"""Armijo sufficient-decrease constant for the backtracking line search."""
comptime _ls_shrink: Float64 = 0.5
"""Step-size shrink factor for the backtracking line search."""
comptime _ls_max_iter: Int = 60
"""Maximum number of step halvings tried by the backtracking line search."""

comptime _invphi: Float64 = 0.3819660112501051
"""Inverse of the golden ratio conjugate, used by the Powell line search."""

comptime _lbfgs_m: Int = 10
"""Number of (s, y) correction pairs kept by L-BFGS-B."""


def minimize[
    dtype: DType,
    f: def[dtype: DType](
        x: NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) raises thin capturing[_] -> Scalar[dtype] where dtype.is_floating_point(),
    *,
    method: String = "Nelder-Mead",
](
    x0: NDArray[dtype],
    args: Optional[List[Scalar[dtype]]] = None,
    bounds: Optional[Tuple[NDArray[dtype], NDArray[dtype]]] = None,
    xatol: Scalar[dtype] = 1e-4,
    fatol: Scalar[dtype] = 1e-4,
    gtol: Scalar[dtype] = 1e-5,
    maxiter: Int = 1000,
    initial_step: Scalar[dtype] = 0.05,
    fd_step: Scalar[dtype] = 1.4901161193847656e-8,
) raises -> OptimizeResultVector[dtype] where dtype.is_floating_point():
    """Minimizes a scalar-valued function of several variables.

    `f` must map `R^n -> R`.

    With `method="Nelder-Mead"` (the default), uses the derivative-free
    simplex algorithm: it only ever evaluates `f`, never a gradient.

    With `method="BFGS"` or `method="CG"`, the gradient of `f` is
    estimated with central finite differences at every iteration (there
    is currently no overload accepting an analytic gradient). `"BFGS"`
    maintains a running approximation of the inverse Hessian and usually
    converges in fewer iterations; `"CG"` (nonlinear conjugate gradient,
    Polak-Ribiere with automatic restart) uses O(n) memory per iteration
    instead of BFGS's O(n^2) and so scales better to large `n`. Both use
    a backtracking line search satisfying the Armijo condition.

    With `method="Powell"`, uses a derivative-free sequential line
    minimization along a rotating set of conjugate directions (Press et
    al., *Numerical Recipes*): each iteration line-minimizes along every
    current direction, then - if it passes Powell's acceleration test -
    replaces the direction responsible for the largest single-line
    decrease with the overall displacement direction.

    With `method="L-BFGS-B"`, uses limited-memory BFGS (two-loop
    recursion over the last few `(s, y)` correction pairs) with a
    projected Armijo line search; see the module docstring for how this
    differs from the reference algorithm. The gradient is estimated with
    central finite differences, as for `"BFGS"`/`"CG"`.

    `bounds`, if given, is a `(lo, hi)` pair of shape-`(n,)` arrays and is
    only honored by `method="Powell"` and `method="L-BFGS-B"`; it is
    ignored by every other method.

    Parameters:
        dtype: The floating-point data type.
        f: Scalar-valued function with signature `def(x, args) -> Scalar[dtype]`.
        method: Minimization method: `"Nelder-Mead"`, `"BFGS"`, `"CG"`,
            `"Powell"`, or `"L-BFGS-B"`.

    Args:
        x0: Initial guess, of shape `(n,)`.
        args: Optional arguments forwarded to `f`.
        bounds: Optional `(lo, hi)` box constraints, each of shape `(n,)`
            (only used by `method="Powell"` or `method="L-BFGS-B"`).
        xatol: Absolute tolerance on the simplex's point-to-point spread
            (`method="Nelder-Mead"`) or on the per-iteration displacement
            (`method="Powell"`).
        fatol: Absolute tolerance on the simplex's function-value spread
            (only used by `method="Nelder-Mead"` or `method="Powell"`).
        gtol: Absolute tolerance on the (projected) gradient norm
            (only used by `method="BFGS"`, `method="CG"`, or
            `method="L-BFGS-B"`).
        maxiter: Maximum number of iterations.
        initial_step: Fractional step used to build the initial simplex:
            each non-pivot vertex offsets one coordinate of `x0` by
            `initial_step * x0[i]` (or a small fixed step if `x0[i] == 0`)
            (only used by `method="Nelder-Mead"`).
        fd_step: Finite-difference step size used to estimate the
            gradient (only used by `method="BFGS"`, `method="CG"`, or
            `method="L-BFGS-B"`).

    Returns:
        OptimizeResultVector[dtype] with the minimizer and diagnostics.

    Raises:
        Error: If `x0` is empty.
        Error: If an unsupported method is requested.

    Examples:
        ```mojo
        import numojo as nm
        from scijo.optimize import minimize
        from scijo.prelude import *

        def rosenbrock[
            dtype: DType
        ](
            x: NDArray[dtype], args: Optional[List[Scalar[dtype]]]
        ) capturing raises -> Scalar[dtype]:
            var a = 1.0 - x.item(0)
            var b = x.item(1) - x.item(0) * x.item(0)
            return a * a + 100.0 * b * b

        var x0 = nm.fromstring[f64]("[-1.2, 1.0]")
        var result = minimize[f64, rosenbrock, method="BFGS"](x0)
        # result.x ~= [1, 1]
        ```
    """
    comptime if method == "Nelder-Mead":
        return _nelder_mead[dtype, f](
            x0, args, xatol, fatol, maxiter, initial_step
        )
    elif method == "BFGS":
        return _bfgs[dtype, f](x0, args, gtol, maxiter, fd_step)
    elif method == "CG":
        return _cg[dtype, f](x0, args, gtol, maxiter, fd_step)
    elif method == "Powell":
        return _powell[dtype, f](x0, args, bounds, xatol, fatol, maxiter)
    elif method == "L-BFGS-B":
        return _lbfgsb[dtype, f](x0, args, bounds, gtol, maxiter, fd_step)
    else:
        raise Error("Scijo [minimize]: Unsupported method: " + String(method))


def minimize[
    dtype: DType,
    f: def[dtype: DType](x: NDArray[dtype]) raises thin capturing[_] -> Scalar[
        dtype
    ] where dtype.is_floating_point(),
    *,
    method: String = "Nelder-Mead",
](
    x0: NDArray[dtype],
    bounds: Optional[Tuple[NDArray[dtype], NDArray[dtype]]] = None,
    xatol: Scalar[dtype] = 1e-4,
    fatol: Scalar[dtype] = 1e-4,
    gtol: Scalar[dtype] = 1e-5,
    maxiter: Int = 1000,
    initial_step: Scalar[dtype] = 0.05,
    fd_step: Scalar[dtype] = 1.4901161193847656e-8,
) raises -> OptimizeResultVector[dtype] where dtype.is_floating_point():
    """Minimizes a scalar-valued function of several variables (args-free overload).

    For functions that do not need `args`: `f` takes only `x`, so extra
    parameters (if any) are captured directly from the enclosing scope
    instead of threaded through `args`. See the `args`-taking overload
    `minimize[dtype, f](..., args=...)` for a function reused across call
    sites with different `args` values.

    Parameters:
        dtype: The floating-point data type.
        f: Scalar-valued function with signature `def(x) -> Scalar[dtype]`.
        method: Minimization method: `"Nelder-Mead"`, `"BFGS"`, `"CG"`,
            `"Powell"`, or `"L-BFGS-B"`.

    Args:
        x0: Initial guess, of shape `(n,)`.
        bounds: Optional `(lo, hi)` box constraints, each of shape `(n,)`
            (only used by `method="Powell"` or `method="L-BFGS-B"`).
        xatol: Absolute tolerance on the simplex's point-to-point spread
            (`method="Nelder-Mead"`) or on the per-iteration displacement
            (`method="Powell"`).
        fatol: Absolute tolerance on the simplex's function-value spread
            (only used by `method="Nelder-Mead"` or `method="Powell"`).
        gtol: Absolute tolerance on the (projected) gradient norm
            (only used by `method="BFGS"`, `method="CG"`, or
            `method="L-BFGS-B"`).
        maxiter: Maximum number of iterations.
        initial_step: Fractional step used to build the initial simplex
            (only used by `method="Nelder-Mead"`).
        fd_step: Finite-difference step size used to estimate the
            gradient (only used by `method="BFGS"`, `method="CG"`, or
            `method="L-BFGS-B"`).

    Returns:
        OptimizeResultVector[dtype] with the minimizer and diagnostics.

    Raises:
        Error: If `x0` is empty.
        Error: If an unsupported method is requested.
    """

    def _with_args[
        dtype2: DType
    ](
        x: NDArray[dtype2], args: Optional[List[Scalar[dtype2]]]
    ) capturing raises -> Scalar[dtype2] where dtype2.is_floating_point():
        return f(x)

    return minimize[dtype, _with_args, method=method](
        x0=x0,
        bounds=bounds,
        xatol=xatol,
        fatol=fatol,
        gtol=gtol,
        maxiter=maxiter,
        initial_step=initial_step,
        fd_step=fd_step,
    )


def _max_abs_diff[
    dtype: DType
](a: NDArray[dtype], b: NDArray[dtype]) raises -> Scalar[dtype]:
    """Returns the maximum absolute elementwise difference between `a` and `b`.
    """
    var m: Scalar[dtype] = 0.0
    for i in range(a.size):
        var d = abs(a.load(i) - b.load(i))
        if d > m:
            m = d
    return m


def _nelder_mead[
    dtype: DType,
    f: def[dtype: DType](
        x: NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) raises thin capturing[_] -> Scalar[dtype] where dtype.is_floating_point(),
](
    x0: NDArray[dtype],
    args: Optional[List[Scalar[dtype]]],
    xatol: Scalar[dtype],
    fatol: Scalar[dtype],
    maxiter: Int,
    initial_step: Scalar[dtype],
) raises -> OptimizeResultVector[dtype] where dtype.is_floating_point():
    """Nelder-Mead simplex minimization."""
    var n = x0.size
    if n == 0:
        raise Error("Scijo [minimize]: x0 must not be empty.")

    var alpha = Scalar[dtype](_nm_alpha)
    var gamma = Scalar[dtype](_nm_gamma)
    var rho = Scalar[dtype](_nm_rho)
    var sigma = Scalar[dtype](_nm_sigma)

    # Build the initial simplex: x0 plus one perturbation per dimension.
    var simplex = List[NDArray[dtype]]()
    var fvals = List[Scalar[dtype]]()
    simplex.append(x0.copy())
    fvals.append(f(x0, args))
    var nfev = 1

    for i in range(n):
        var xi = x0.copy()
        var v = x0.load(i)
        var step = initial_step * v if v != 0 else Scalar[dtype](0.00025)
        xi.store(i, val=v + step)
        fvals.append(f(xi, args))
        nfev += 1
        simplex.append(xi^)

    var it = 0
    var converged = False

    while it < maxiter:
        # Sort the simplex (and its function values) ascending by value.
        for i in range(1, n + 1):
            var j = i
            while j > 0 and fvals[j] < fvals[j - 1]:
                fvals.swap_elements(j, j - 1)
                simplex.swap_elements(j, j - 1)
                j -= 1

        var max_x_spread: Scalar[dtype] = 0.0
        for i in range(1, n + 1):
            var d = _max_abs_diff(simplex[i], simplex[0])
            if d > max_x_spread:
                max_x_spread = d
        var max_f_spread: Scalar[dtype] = 0.0
        for i in range(1, n + 1):
            var d = abs(fvals[i] - fvals[0])
            if d > max_f_spread:
                max_f_spread = d

        if max_x_spread <= xatol and max_f_spread <= fatol:
            converged = True
            break

        # Centroid of every vertex but the worst (index n).
        var centroid = zeros[dtype](Shape(n))
        for i in range(n):
            centroid = centroid + simplex[i]
        centroid = centroid / Scalar[dtype](n)

        var worst = simplex[n].copy()
        var f_worst = fvals[n]

        var xr = centroid + alpha * (centroid - worst)
        var fr = f(xr, args)
        nfev += 1

        if fr < fvals[0]:
            var xe = centroid + gamma * (xr - centroid)
            var fe = f(xe, args)
            nfev += 1
            if fe < fr:
                simplex[n] = xe^
                fvals[n] = fe
            else:
                simplex[n] = xr^
                fvals[n] = fr
        elif fr < fvals[n - 1]:
            simplex[n] = xr^
            fvals[n] = fr
        else:
            var did_contract = False
            if fr < f_worst:
                var xc = centroid + rho * (xr - centroid)
                var fc = f(xc, args)
                nfev += 1
                if fc <= fr:
                    simplex[n] = xc^
                    fvals[n] = fc
                    did_contract = True
            else:
                var xc = centroid + rho * (worst - centroid)
                var fc = f(xc, args)
                nfev += 1
                if fc < f_worst:
                    simplex[n] = xc^
                    fvals[n] = fc
                    did_contract = True

            if not did_contract:
                # Shrink every vertex but the best (index 0) toward it.
                for i in range(1, n + 1):
                    var xi = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = f(xi, args)
                    nfev += 1
                    simplex[i] = xi^

        it += 1

    # Final ordering, in case the loop exited via `maxiter` without one.
    for i in range(1, n + 1):
        var j = i
        while j > 0 and fvals[j] < fvals[j - 1]:
            fvals.swap_elements(j, j - 1)
            simplex.swap_elements(j, j - 1)
            j -= 1

    return OptimizeResultVector[dtype](
        simplex[0].copy(),
        fvals[0],
        it,
        nfev,
        converged,
        "converged" if converged else "maximum iterations exceeded",
        "Nelder-Mead",
    )


# ===----------------------------------------------------------------------=== #
# Shared linear-algebra helpers for the gradient-based methods
# ===----------------------------------------------------------------------=== #


def _dot[
    dtype: DType
](a: NDArray[dtype], b: NDArray[dtype]) raises -> Scalar[dtype]:
    """Returns the dot product of two vectors of equal length."""
    var s: Scalar[dtype] = 0.0
    for i in range(a.size):
        s += a.load(i) * b.load(i)
    return s


def _matvec[
    dtype: DType
](A: NDArray[dtype], v: NDArray[dtype], n: Int) raises -> NDArray[dtype]:
    """Returns `A @ v` for a flattened `(n, n)` matrix `A` and vector `v`."""
    var result = zeros[dtype](Shape(n))
    for i in range(n):
        var s: Scalar[dtype] = 0.0
        for j in range(n):
            s += A.load(i * n + j) * v.load(j)
        result.store(i, val=s)
    return result^


def _fd_gradient[
    dtype: DType,
    f: def[dtype: DType](
        x: NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) raises thin capturing[_] -> Scalar[dtype] where dtype.is_floating_point(),
](
    x: NDArray[dtype],
    args: Optional[List[Scalar[dtype]]],
    step: Scalar[dtype],
) raises -> NDArray[dtype] where dtype.is_floating_point():
    """Estimates the gradient of `f` at `x` with central finite differences."""
    var n = x.size
    var g = zeros[dtype](Shape(n))
    for i in range(n):
        var xp = x.copy()
        var xm = x.copy()
        xp.store(i, val=x.load(i) + step)
        xm.store(i, val=x.load(i) - step)
        g.store(i, val=(f(xp, args) - f(xm, args)) / (2.0 * step))
    return g^


def _armijo_line_search[
    dtype: DType,
    f: def[dtype: DType](
        x: NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) raises thin capturing[_] -> Scalar[dtype] where dtype.is_floating_point(),
](
    x: NDArray[dtype],
    fx: Scalar[dtype],
    g: NDArray[dtype],
    p: NDArray[dtype],
    args: Optional[List[Scalar[dtype]]],
) raises -> Tuple[
    Scalar[dtype], Scalar[dtype], Int
] where dtype.is_floating_point():
    """Backtracking line search satisfying the Armijo sufficient-decrease condition.

    Starting from a unit step, halves the step size until
    `f(x + alpha*p) <= f(x) + c1*alpha*(grad f(x) . p)`, or until the
    halving budget is exhausted. Returns `(alpha, f(x + alpha*p), nfev)`.
    """
    var slope = _dot(g, p)
    if slope >= 0:
        # `p` is not a descent direction (can happen right after a
        # skipped curvature update or a restart); fall back to a small
        # safe step.
        var alpha = Scalar[dtype](1e-4)
        return (alpha, f(x + alpha * p, args), 1)

    var c1 = Scalar[dtype](_ls_c1)
    var shrink = Scalar[dtype](_ls_shrink)
    var alpha: Scalar[dtype] = 1.0
    var f_new = fx
    var nfev = 0
    var it = 0
    while it < _ls_max_iter:
        f_new = f(x + alpha * p, args)
        nfev += 1
        if f_new <= fx + c1 * alpha * slope:
            break
        alpha *= shrink
        it += 1
    return (alpha, f_new, nfev)


# ===----------------------------------------------------------------------=== #
# BFGS
# ===----------------------------------------------------------------------=== #


def _bfgs_update[
    dtype: DType
](mut H: NDArray[dtype], s: NDArray[dtype], y: NDArray[dtype], n: Int) raises:
    """Updates the inverse-Hessian approximation `H` with the BFGS secant formula.

    Uses the standard rank-2 inverse update (Nocedal & Wright, eq. 6.17),
    expanded into O(n^2) work instead of the O(n^3) two-sided matrix
    product form. If the curvature condition `s . y > 0` is not met, the
    update is skipped and `H` is left unchanged, since applying it would
    make `H` indefinite.
    """
    var sy = _dot(s, y)
    if sy <= 1e-10:
        return

    var rho = Scalar[dtype](1.0) / sy
    var Hy = _matvec(H, y, n)
    var yHy = _dot(y, Hy)
    var coef = rho * (Scalar[dtype](1.0) + rho * yHy)

    for i in range(n):
        for j in range(n):
            var val = (
                H.load(i * n + j)
                + coef * s.load(i) * s.load(j)
                - rho * (s.load(i) * Hy.load(j) + Hy.load(i) * s.load(j))
            )
            H.store(i * n + j, val=val)


def _bfgs[
    dtype: DType,
    f: def[dtype: DType](
        x: NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) raises thin capturing[_] -> Scalar[dtype] where dtype.is_floating_point(),
](
    x0: NDArray[dtype],
    args: Optional[List[Scalar[dtype]]],
    gtol: Scalar[dtype],
    maxiter: Int,
    fd_step: Scalar[dtype],
) raises -> OptimizeResultVector[dtype] where dtype.is_floating_point():
    """BFGS quasi-Newton minimization with a finite-difference gradient."""
    var n = x0.size
    if n == 0:
        raise Error("Scijo [minimize]: x0 must not be empty.")

    var x = x0.copy()
    var fx = f(x, args)
    var nfev = 1
    var g = _fd_gradient[dtype, f](x, args, fd_step)
    nfev += 2 * n

    # Inverse-Hessian approximation, initialized to the identity.
    var H = zeros[dtype](Shape(n, n))
    for i in range(n):
        H.store(i * n + i, val=Scalar[dtype](1.0))

    var it = 0
    var converged = norm(g) <= gtol
    while it < maxiter and not converged:
        var p = -_matvec(H, g, n)
        var ls = _armijo_line_search[dtype, f](x, fx, g, p, args)
        var alpha = ls[0]
        nfev += ls[2]

        var x_new = x + alpha * p
        var g_new = _fd_gradient[dtype, f](x_new, args, fd_step)
        nfev += 2 * n

        var s = x_new - x
        var y = g_new - g
        _bfgs_update(H, s, y, n)

        x = x_new^
        fx = ls[1]
        g = g_new^
        it += 1
        converged = norm(g) <= gtol

    return OptimizeResultVector[dtype](
        x^,
        fx,
        it,
        nfev,
        converged,
        "converged" if converged else "maximum iterations exceeded",
        "BFGS",
    )


# ===----------------------------------------------------------------------=== #
# Nonlinear conjugate gradient
# ===----------------------------------------------------------------------=== #


def _cg[
    dtype: DType,
    f: def[dtype: DType](
        x: NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) raises thin capturing[_] -> Scalar[dtype] where dtype.is_floating_point(),
](
    x0: NDArray[dtype],
    args: Optional[List[Scalar[dtype]]],
    gtol: Scalar[dtype],
    maxiter: Int,
    fd_step: Scalar[dtype],
) raises -> OptimizeResultVector[dtype] where dtype.is_floating_point():
    """Nonlinear conjugate gradient minimization (Polak-Ribiere, with
    automatic restarts) with a finite-difference gradient.
    """
    var n = x0.size
    if n == 0:
        raise Error("Scijo [minimize]: x0 must not be empty.")

    var x = x0.copy()
    var fx = f(x, args)
    var nfev = 1
    var g = _fd_gradient[dtype, f](x, args, fd_step)
    nfev += 2 * n
    var p = -g

    var it = 0
    var converged = norm(g) <= gtol
    while it < maxiter and not converged:
        var ls = _armijo_line_search[dtype, f](x, fx, g, p, args)
        var alpha = ls[0]
        nfev += ls[2]

        var x_new = x + alpha * p
        var g_new = _fd_gradient[dtype, f](x_new, args, fd_step)
        nfev += 2 * n

        # Polak-Ribiere+ coefficient, clamped to zero (which restarts
        # along the steepest-descent direction) when negative.
        var gg = _dot(g, g)
        var beta: Scalar[dtype] = 0.0
        if gg > 1e-300:
            beta = _dot(g_new, g_new - g) / gg
        if beta < 0:
            beta = 0.0

        x = x_new^
        fx = ls[1]
        it += 1

        var p_new = -g_new + beta * p
        # Periodic restart along steepest descent, which bounds how far
        # conjugacy (and hence convergence) can degrade between resets.
        if it % n == 0:
            p = -g_new
        else:
            p = p_new^
        g = g_new^
        converged = norm(g) <= gtol

    return OptimizeResultVector[dtype](
        x^,
        fx,
        it,
        nfev,
        converged,
        "converged" if converged else "maximum iterations exceeded",
        "CG",
    )


# ===----------------------------------------------------------------------=== #
# Box-constraint helpers, shared by Powell and L-BFGS-B
# ===----------------------------------------------------------------------=== #


def _clip_box[
    dtype: DType
](
    x: NDArray[dtype], lo: NDArray[dtype], hi: NDArray[dtype], n: Int
) raises -> NDArray[dtype]:
    """Projects `x` onto the box `[lo, hi]`, elementwise."""
    var result = x.copy()
    for i in range(n):
        var v = result.load(i)
        var l = lo.load(i)
        var h = hi.load(i)
        if v < l:
            v = l
        if v > h:
            v = h
        result.store(i, val=v)
    return result^


def _alpha_box_bounds[
    dtype: DType
](
    x: NDArray[dtype],
    direction: NDArray[dtype],
    lo: NDArray[dtype],
    hi: NDArray[dtype],
    n: Int,
) raises -> Tuple[Scalar[dtype], Scalar[dtype]]:
    """Returns the feasible `alpha` interval `[alo, ahi]` such that
    `x + alpha*direction` stays within `[lo, hi]`, elementwise.
    """
    var alo = Scalar[dtype](-1e30)
    var ahi = Scalar[dtype](1e30)
    for i in range(n):
        var d = direction.load(i)
        if abs(d) <= Scalar[dtype](1e-300):
            continue
        var t_lo = (lo.load(i) - x.load(i)) / d
        var t_hi = (hi.load(i) - x.load(i)) / d
        if d > 0:
            if t_lo > alo:
                alo = t_lo
            if t_hi < ahi:
                ahi = t_hi
        else:
            if t_hi > alo:
                alo = t_hi
            if t_lo < ahi:
                ahi = t_lo
    return (alo, ahi)


# ===----------------------------------------------------------------------=== #
# Powell
# ===----------------------------------------------------------------------=== #


def _brent_line_1d[
    dtype: DType,
    f: def[dtype: DType](
        x: NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) raises thin capturing[_] -> Scalar[dtype] where dtype.is_floating_point(),
](
    x: NDArray[dtype],
    direction: NDArray[dtype],
    args: Optional[List[Scalar[dtype]]],
    a0: Scalar[dtype],
    b0: Scalar[dtype],
    tol: Scalar[dtype],
    maxiter: Int,
) raises -> Tuple[
    Scalar[dtype], NDArray[dtype], Scalar[dtype], Int
] where dtype.is_floating_point():
    """Brent's method, minimizing `f(x + alpha*direction)` for `alpha` in
    the bounded interval `[a0, b0]`. Returns `(alpha, x_new, f_new, nfev)`.
    """
    var a = a0
    var b = b0
    if a > b:
        var tmp = a
        a = b
        b = tmp

    var invphi = Scalar[dtype](_invphi)
    var alpha = a + invphi * (b - a)
    var w = alpha
    var v = alpha
    var fx = f(x + alpha * direction, args)
    var fw = fx
    var fv = fx
    var nfev = 1
    var d: Scalar[dtype] = 0.0
    var e: Scalar[dtype] = 0.0

    for _ in range(maxiter):
        var m = (a + b) / Scalar[dtype](2.0)
        var tol1 = tol * abs(alpha) + Scalar[dtype](1e-12)
        var tol2 = tol1 * Scalar[dtype](2.0)
        if abs(b - a) <= tol2:
            break

        var use_golden = True
        if abs(e) > tol1:
            var r = (alpha - w) * (fx - fv)
            var q = (alpha - v) * (fx - fw)
            var p = (alpha - v) * q - (alpha - w) * r
            q = (q - r) * Scalar[dtype](2.0)
            if q > 0:
                p = -p
            q = abs(q)
            if (
                abs(p) < abs(Scalar[dtype](0.5) * q * e)
                and p > q * (a - alpha)
                and p < q * (b - alpha)
            ):
                d = p / q
                var u1 = alpha + d
                if (u1 - a) < tol2 or (b - u1) < tol2:
                    d = tol1 if alpha < m else -tol1
                use_golden = False

        if use_golden:
            e = b - alpha if alpha < m else a - alpha
            d = invphi * e

        var u = alpha + d if abs(d) >= tol1 else alpha + (
            tol1 if d > 0 else -tol1
        )
        if u < a + tol2:
            u = a + tol2
        if u > b - tol2:
            u = b - tol2

        var fu = f(x + u * direction, args)
        nfev += 1

        if fu <= fx:
            if u < alpha:
                b = alpha
            else:
                a = alpha
            v = w
            fv = fw
            w = alpha
            fw = fx
            alpha = u
            fx = fu
        else:
            if u < alpha:
                a = u
            else:
                b = u
            if fu <= fw or w == alpha:
                v = w
                fv = fw
                w = u
                fw = fu
            elif fu <= fv or v == alpha or v == w:
                v = u
                fv = fu

    var x_new = x + alpha * direction
    return (alpha, x_new^, fx, nfev)


def _powell_line_search[
    dtype: DType,
    f: def[dtype: DType](
        x: NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) raises thin capturing[_] -> Scalar[dtype] where dtype.is_floating_point(),
](
    x: NDArray[dtype],
    direction: NDArray[dtype],
    args: Optional[List[Scalar[dtype]]],
    bounds: Optional[Tuple[NDArray[dtype], NDArray[dtype]]],
    n: Int,
    tol: Scalar[dtype],
) raises -> Tuple[
    NDArray[dtype], Scalar[dtype], Int
] where dtype.is_floating_point():
    """One Powell line minimization along `direction`, honoring `bounds`
    if given. Returns `(x_new, f_new, nfev)`.
    """
    var dir_norm = norm(direction)
    if dir_norm <= Scalar[dtype](1e-300):
        return (x.copy(), f(x, args), 1)

    var alo: Scalar[dtype]
    var ahi: Scalar[dtype]
    if bounds:
        ref bb = bounds.value()
        var ab = _alpha_box_bounds[dtype](x, direction, bb[0], bb[1], n)
        alo = ab[0]
        ahi = ab[1]
        if alo >= ahi:
            return (x.copy(), f(x, args), 1)
    else:
        var scale = Scalar[dtype](1.0)
        for i in range(n):
            var xi_abs = abs(x.load(i))
            if xi_abs > scale:
                scale = xi_abs
        var span = Scalar[dtype](10.0) * scale / dir_norm
        if span < Scalar[dtype](1.0):
            span = Scalar[dtype](1.0)
        alo = -span
        ahi = span

    var res = _brent_line_1d[dtype, f](x, direction, args, alo, ahi, tol, 100)
    return (res[1].copy(), res[2], res[3])


def _powell[
    dtype: DType,
    f: def[dtype: DType](
        x: NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) raises thin capturing[_] -> Scalar[dtype] where dtype.is_floating_point(),
](
    x0: NDArray[dtype],
    args: Optional[List[Scalar[dtype]]],
    bounds: Optional[Tuple[NDArray[dtype], NDArray[dtype]]],
    xatol: Scalar[dtype],
    fatol: Scalar[dtype],
    maxiter: Int,
) raises -> OptimizeResultVector[dtype] where dtype.is_floating_point():
    """Powell's method: sequential line minimization along a rotating set
    of conjugate directions, with the classic acceleration test deciding
    whether to replace a direction (Press et al., *Numerical Recipes*).
    """
    var n = x0.size
    if n == 0:
        raise Error("Scijo [minimize]: x0 must not be empty.")

    var x = x0.copy()
    if bounds:
        ref bb = bounds.value()
        x = _clip_box[dtype](x, bb[0], bb[1], n)
    var fx = f(x, args)
    var nfev = 1

    var directions = List[NDArray[dtype]]()
    for i in range(n):
        var e_i = zeros[dtype](Shape(n))
        e_i.store(i, val=Scalar[dtype](1.0))
        directions.append(e_i^)

    var it = 0
    var converged = False
    while it < maxiter:
        var x_start = x.copy()
        var f_start = fx
        var biggest_decrease: Scalar[dtype] = 0.0
        var biggest_idx = 0

        for i in range(n):
            var f_before = fx
            var res = _powell_line_search[dtype, f](
                x, directions[i], args, bounds, n, xatol
            )
            x = res[0].copy()
            fx = res[1]
            nfev += res[2]
            var decrease = f_before - fx
            if decrease > biggest_decrease:
                biggest_decrease = decrease
                biggest_idx = i

        if (
            Scalar[dtype](2.0) * (f_start - fx)
            <= fatol * (abs(f_start) + abs(fx)) + Scalar[dtype](1e-300)
            and _max_abs_diff(x, x_start) <= xatol
        ):
            converged = True
            break

        var extrap_dir = x - x_start
        var x_extrap = x + extrap_dir
        if bounds:
            ref bb = bounds.value()
            x_extrap = _clip_box[dtype](x_extrap, bb[0], bb[1], n)
        var f_extrap = f(x_extrap, args)
        nfev += 1

        if f_extrap < f_start:
            var t1 = f_start - fx - biggest_decrease
            var t2 = f_start - f_extrap
            var t = (
                Scalar[dtype](2.0)
                * (f_start - Scalar[dtype](2.0) * fx + f_extrap)
                * t1
                * t1
                - biggest_decrease * t2 * t2
            )
            if t < 0:
                var new_direction = extrap_dir.copy()
                var res2 = _powell_line_search[dtype, f](
                    x, new_direction, args, bounds, n, xatol
                )
                x = res2[0].copy()
                fx = res2[1]
                nfev += res2[2]
                directions[biggest_idx] = new_direction^

        it += 1

    return OptimizeResultVector[dtype](
        x^,
        fx,
        it,
        nfev,
        converged,
        "converged" if converged else "maximum iterations exceeded",
        "Powell",
    )


# ===----------------------------------------------------------------------=== #
# L-BFGS-B (projected-gradient variant)
# ===----------------------------------------------------------------------=== #


def _build_proj_grad[
    dtype: DType
](
    x: NDArray[dtype],
    g: NDArray[dtype],
    bounds: Optional[Tuple[NDArray[dtype], NDArray[dtype]]],
    n: Int,
) raises -> NDArray[dtype]:
    """Zeroes out gradient components that are "blocked": `x` sits at a
    box boundary and the unconstrained step would push further outside.
    """
    if not bounds:
        return g.copy()
    ref bb = bounds.value()
    ref lo = bb[0]
    ref hi = bb[1]
    var pg = g.copy()
    var eps = Scalar[dtype](1e-12)
    for i in range(n):
        var xi = x.load(i)
        var gi = g.load(i)
        if xi <= lo.load(i) + eps and gi > 0:
            pg.store(i, val=Scalar[dtype](0.0))
        elif xi >= hi.load(i) - eps and gi < 0:
            pg.store(i, val=Scalar[dtype](0.0))
    return pg^


def _lbfgs_Hg[
    dtype: DType
](
    g: NDArray[dtype],
    s_hist: List[NDArray[dtype]],
    y_hist: List[NDArray[dtype]],
    rho_hist: List[Scalar[dtype]],
) raises -> NDArray[dtype]:
    """Two-loop recursion (Nocedal & Wright, Algorithm 7.4): returns
    `H_k @ g`, the limited-memory BFGS inverse-Hessian-vector product.
    """
    var m = len(s_hist)
    var q = g.copy()
    var alpha_hist = List[Scalar[dtype]](capacity=m)
    for _ in range(m):
        alpha_hist.append(Scalar[dtype](0.0))

    for i in range(m - 1, -1, -1):
        var alpha_i = rho_hist[i] * _dot(s_hist[i], q)
        alpha_hist[i] = alpha_i
        q = q - alpha_i * y_hist[i]

    var gamma = Scalar[dtype](1.0)
    if m > 0:
        var last = m - 1
        var yy = _dot(y_hist[last], y_hist[last])
        if yy > Scalar[dtype](1e-300):
            gamma = _dot(s_hist[last], y_hist[last]) / yy

    var r = gamma * q
    for i in range(m):
        var beta = rho_hist[i] * _dot(y_hist[i], r)
        r = r + (alpha_hist[i] - beta) * s_hist[i]
    return r^


def _lbfgsb[
    dtype: DType,
    f: def[dtype: DType](
        x: NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) raises thin capturing[_] -> Scalar[dtype] where dtype.is_floating_point(),
](
    x0: NDArray[dtype],
    args: Optional[List[Scalar[dtype]]],
    bounds: Optional[Tuple[NDArray[dtype], NDArray[dtype]]],
    gtol: Scalar[dtype],
    maxiter: Int,
    fd_step: Scalar[dtype],
) raises -> OptimizeResultVector[dtype] where dtype.is_floating_point():
    """Limited-memory BFGS with simple box constraints: a two-loop
    recursion over the last `_lbfgs_m` `(s, y)` pairs, a projected Armijo
    backtracking line search, and convergence on the projected-gradient
    norm. See the module docstring for how this differs from the
    reference active-set L-BFGS-B algorithm.
    """
    var n = x0.size
    if n == 0:
        raise Error("Scijo [minimize]: x0 must not be empty.")

    var x = x0.copy()
    if bounds:
        ref bb = bounds.value()
        x = _clip_box[dtype](x, bb[0], bb[1], n)

    var fx = f(x, args)
    var nfev = 1
    var g = _fd_gradient[dtype, f](x, args, fd_step)
    nfev += 2 * n

    var s_hist = List[NDArray[dtype]]()
    var y_hist = List[NDArray[dtype]]()
    var rho_hist = List[Scalar[dtype]]()

    var it = 0
    var pg = _build_proj_grad[dtype](x, g, bounds, n)
    var converged = norm(pg) <= gtol

    while it < maxiter and not converged:
        var Hg = _lbfgs_Hg[dtype](g, s_hist, y_hist, rho_hist)
        var p = -Hg

        var alpha = Scalar[dtype](1.0)
        var x_new = x + alpha * p
        if bounds:
            ref bb = bounds.value()
            x_new = _clip_box[dtype](x_new, bb[0], bb[1], n)
        var f_new = f(x_new, args)
        nfev += 1

        var c1 = Scalar[dtype](_ls_c1)
        var ls_it = 0
        while f_new > fx + c1 * _dot(g, x_new - x) and ls_it < _ls_max_iter:
            alpha *= Scalar[dtype](_ls_shrink)
            x_new = x + alpha * p
            if bounds:
                ref bb2 = bounds.value()
                x_new = _clip_box[dtype](x_new, bb2[0], bb2[1], n)
            f_new = f(x_new, args)
            nfev += 1
            ls_it += 1

        var g_new = _fd_gradient[dtype, f](x_new, args, fd_step)
        nfev += 2 * n

        var s = x_new - x
        var y = g_new - g
        var sy = _dot(s, y)
        if sy > Scalar[dtype](1e-10):
            if len(s_hist) >= _lbfgs_m:
                _ = s_hist.pop(0)
                _ = y_hist.pop(0)
                _ = rho_hist.pop(0)
            s_hist.append(s.copy())
            y_hist.append(y.copy())
            rho_hist.append(Scalar[dtype](1.0) / sy)

        x = x_new^
        fx = f_new
        g = g_new^
        it += 1
        pg = _build_proj_grad[dtype](x, g, bounds, n)
        converged = norm(pg) <= gtol

    return OptimizeResultVector[dtype](
        x^,
        fx,
        it,
        nfev,
        converged,
        "converged" if converged else "maximum iterations exceeded",
        "L-BFGS-B",
    )
