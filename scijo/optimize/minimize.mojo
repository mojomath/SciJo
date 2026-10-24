# ===----------------------------------------------------------------------=== #
# SciJo: Optimize module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Multi-Dimensional Minimization (`scijo.optimize.minimize`)
=============================================================
Minimization for scalar-valued functions of several variables.
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

    Parameters:
        dtype: The floating-point data type.
        f: Scalar-valued function with signature `def(x, args) -> Scalar[dtype]`.
        method: Minimization method: `"Nelder-Mead"`, `"BFGS"`, or `"CG"`.

    Args:
        x0: Initial guess, of shape `(n,)`.
        args: Optional arguments forwarded to `f`.
        xatol: Absolute tolerance on the simplex's point-to-point spread
            (only used by `method="Nelder-Mead"`).
        fatol: Absolute tolerance on the simplex's function-value spread
            (only used by `method="Nelder-Mead"`).
        gtol: Absolute tolerance on the gradient norm `||grad f(x)||`
            (only used by `method="BFGS"` or `method="CG"`).
        maxiter: Maximum number of iterations.
        initial_step: Fractional step used to build the initial simplex:
            each non-pivot vertex offsets one coordinate of `x0` by
            `initial_step * x0[i]` (or a small fixed step if `x0[i] == 0`)
            (only used by `method="Nelder-Mead"`).
        fd_step: Finite-difference step size used to estimate the
            gradient (only used by `method="BFGS"` or `method="CG"`).

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
        method: Minimization method: `"Nelder-Mead"`, `"BFGS"`, or `"CG"`.

    Args:
        x0: Initial guess, of shape `(n,)`.
        xatol: Absolute tolerance on the simplex's point-to-point spread
            (only used by `method="Nelder-Mead"`).
        fatol: Absolute tolerance on the simplex's function-value spread
            (only used by `method="Nelder-Mead"`).
        gtol: Absolute tolerance on the gradient norm `||grad f(x)||`
            (only used by `method="BFGS"` or `method="CG"`).
        maxiter: Maximum number of iterations.
        initial_step: Fractional step used to build the initial simplex
            (only used by `method="Nelder-Mead"`).
        fd_step: Finite-difference step size used to estimate the
            gradient (only used by `method="BFGS"` or `method="CG"`).

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
