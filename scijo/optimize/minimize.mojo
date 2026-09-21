# ===----------------------------------------------------------------------=== #
# SciJo: Optimize module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Multi-Dimensional Minimization (`scijo.optimize.minimize`)
=============================================================
Minimization for scalar-valued functions of several variables, matching
`scipy.optimize.minimize`'s frontend signature.
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
from scijo.optimize.utility import OptimizeResultVector

comptime _nm_alpha: Float64 = 1.0
"""Reflection coefficient."""
comptime _nm_gamma: Float64 = 2.0
"""Expansion coefficient."""
comptime _nm_rho: Float64 = 0.5
"""Contraction coefficient."""
comptime _nm_sigma: Float64 = 0.5
"""Shrink coefficient."""


def minimize[
    dtype: DType,
    f: def[dtype: DType](
        x: NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) capturing raises -> Scalar[dtype],
    *,
    method: String = "Nelder-Mead",
](
    x0: NDArray[dtype],
    args: Optional[List[Scalar[dtype]]] = None,
    xatol: Scalar[dtype] = 1e-4,
    fatol: Scalar[dtype] = 1e-4,
    maxiter: Int = 1000,
    initial_step: Scalar[dtype] = 0.05,
) raises -> OptimizeResultVector[dtype]:
    """Minimizes a scalar-valued function of several variables.

    `f` must map `R^n -> R`. Uses the Nelder-Mead simplex algorithm, which
    is derivative-free: it only ever evaluates `f`, never a gradient or
    Hessian, matching `scipy.optimize.minimize(..., method="Nelder-Mead")`.

    Parameters:
        dtype: The floating-point data type.
        f: Scalar-valued function with signature `def(x, args) -> Scalar[dtype]`.
        method: Minimization method. Only `"Nelder-Mead"` is currently
            supported.

    Args:
        x0: Initial guess, of shape `(n,)`.
        args: Optional arguments forwarded to `f`.
        xatol: Absolute tolerance on the simplex's point-to-point spread.
        fatol: Absolute tolerance on the simplex's function-value spread.
        maxiter: Maximum number of iterations.
        initial_step: Fractional step used to build the initial simplex:
            each non-pivot vertex offsets one coordinate of `x0` by
            `initial_step * x0[i]` (or a small fixed step if `x0[i] == 0`).

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
        var result = minimize[f64, rosenbrock](x0)
        # result.x ~= [1, 1]
        ```
    """
    comptime if method == "Nelder-Mead":
        return _nelder_mead[dtype, f](
            x0, args, xatol, fatol, maxiter, initial_step
        )
    else:
        raise Error("Scijo [minimize]: Unsupported method: " + String(method))


def minimize[
    dtype: DType,
    f: def[dtype: DType](x: NDArray[dtype]) capturing raises -> Scalar[dtype],
    *,
    method: String = "Nelder-Mead",
](
    x0: NDArray[dtype],
    xatol: Scalar[dtype] = 1e-4,
    fatol: Scalar[dtype] = 1e-4,
    maxiter: Int = 1000,
    initial_step: Scalar[dtype] = 0.05,
) raises -> OptimizeResultVector[dtype]:
    """Minimizes a scalar-valued function of several variables (args-free overload).

    For functions that do not need `args`: `f` takes only `x`, so extra
    parameters (if any) are captured directly from the enclosing scope
    instead of threaded through `args`. See the `args`-taking overload
    `minimize[dtype, f](..., args=...)` for a function reused across call
    sites with different `args` values.

    Parameters:
        dtype: The floating-point data type.
        f: Scalar-valued function with signature `def(x) -> Scalar[dtype]`.
        method: Minimization method. Only `"Nelder-Mead"` is currently
            supported.

    Args:
        x0: Initial guess, of shape `(n,)`.
        xatol: Absolute tolerance on the simplex's point-to-point spread.
        fatol: Absolute tolerance on the simplex's function-value spread.
        maxiter: Maximum number of iterations.
        initial_step: Fractional step used to build the initial simplex.

    Returns:
        OptimizeResultVector[dtype] with the minimizer and diagnostics.

    Raises:
        Error: If `x0` is empty.
        Error: If an unsupported method is requested.
    """

    @parameter
    def _with_args[
        dtype2: DType
    ](
        x: NDArray[dtype2], args: Optional[List[Scalar[dtype2]]]
    ) capturing raises -> Scalar[dtype2]:
        return f(x)

    return minimize[dtype, _with_args, method=method](
        x0=x0,
        xatol=xatol,
        fatol=fatol,
        maxiter=maxiter,
        initial_step=initial_step,
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
    ) capturing raises -> Scalar[dtype],
](
    x0: NDArray[dtype],
    args: Optional[List[Scalar[dtype]]],
    xatol: Scalar[dtype],
    fatol: Scalar[dtype],
    maxiter: Int,
    initial_step: Scalar[dtype],
) raises -> OptimizeResultVector[dtype]:
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
