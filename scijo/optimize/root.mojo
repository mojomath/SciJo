# ===----------------------------------------------------------------------=== #
# SciJo: Optimize module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Multi-Dimensional Root-Finding (`scijo.optimize.root`)
=========================================================
Root-finding for vector-valued functions: Newton's method (with a
finite-difference or analytic Jacobian) and Broyden's method.
"""

# ===----------------------------------------------------------------------=== #
# External
# ===----------------------------------------------------------------------=== #
from numojo.core import Shape
from numojo.core.ndarray import NDArray
from numojo.routines.creation import zeros

# ===----------------------------------------------------------------------=== #
# SciJo
# ===----------------------------------------------------------------------=== #
from scijo.differentiate.jacob import jacobian
from scijo.linalg.norms import norm
from scijo.linalg.solving import solve
from scijo.optimize.utility import RootResultVector


def root[
    dtype: DType,
    f: def[dtype: DType](
        x: NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) raises thin capturing[_] -> NDArray[
        dtype
    ] where dtype.is_floating_point(),
    *,
    method: String = "newton",
](
    x0: NDArray[dtype],
    args: Optional[List[Scalar[dtype]]] = None,
    atol: Scalar[dtype] = 1e-8,
    rtol: Scalar[dtype] = 1e-8,
    maxiter: Int = 100,
    fd_step: Scalar[dtype] = 1e-6,
) raises -> RootResultVector[dtype] where dtype.is_floating_point():
    """Finds a root of a vector-valued function (finite-difference overload).

    `f` must map `R^n -> R^n` (as many outputs as inputs), so that the
    Jacobian is square and invertible at a root.

    With `method="newton"` (the default), the Jacobian is estimated with
    central finite differences at every iteration (via
    `scijo.differentiate.jacobian`); use the `root[dtype, f, jac](...)`
    overload to supply an analytic Jacobian instead and skip that cost.

    With `method="broyden1"`, no Jacobian (finite-difference or analytic)
    is ever formed: Broyden's "good" method maintains a low-rank running
    approximation of the inverse Jacobian instead, updated from the last
    step and the resulting change in `f`, at the cost of usually needing
    more iterations than Newton's method to reach the same tolerance.

    Parameters:
        dtype: The floating-point data type.
        f: Vector-valued function with signature `def(x, args) -> NDArray[dtype]`.
        method: Root-finding method: `"newton"` or `"broyden1"`.

    Args:
        x0: Initial guess, of shape `(n,)`.
        args: Optional arguments forwarded to `f`.
        atol: Absolute convergence tolerance on `||f(x)||`.
        rtol: Relative convergence tolerance on the step size.
        maxiter: Maximum number of iterations.
        fd_step: Finite-difference step size used to estimate the Jacobian
            (only used by `method="newton"`).

    Returns:
        RootResultVector[dtype] with the root estimate and diagnostics.

    Raises:
        Error: If `f(x0)` is not the same length as `x0`.
        Error: If an unsupported method is requested.
    """
    comptime if method == "newton":
        return _newton_fd[dtype, f](x0, args, atol, rtol, maxiter, fd_step)
    elif method == "broyden1":
        return _broyden1[dtype, f](x0, args, atol, rtol, maxiter)
    else:
        raise Error("Scijo [root]: Unsupported method: " + String(method))


def root[
    dtype: DType,
    f: def[dtype: DType](
        x: NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) raises thin capturing[_] -> NDArray[
        dtype
    ] where dtype.is_floating_point(),
    jac: def[dtype: DType](
        x: NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) raises thin capturing[_] -> NDArray[
        dtype
    ] where dtype.is_floating_point(),
    *,
    method: String = "newton",
](
    x0: NDArray[dtype],
    args: Optional[List[Scalar[dtype]]] = None,
    atol: Scalar[dtype] = 1e-8,
    rtol: Scalar[dtype] = 1e-8,
    maxiter: Int = 100,
) raises -> RootResultVector[dtype] where dtype.is_floating_point():
    """Finds a root of a vector-valued function (analytic-Jacobian overload).

    Parameters:
        dtype: The floating-point data type.
        f: Vector-valued function with signature `def(x, args) -> NDArray[dtype]`.
        jac: Jacobian of `f`, returning an `(n, n)` matrix.
        method: Root-finding method. Only `"newton"` is currently supported.

    Args:
        x0: Initial guess, of shape `(n,)`.
        args: Optional arguments forwarded to `f` and `jac`.
        atol: Absolute convergence tolerance on `||f(x)||`.
        rtol: Relative convergence tolerance on the Newton step size.
        maxiter: Maximum number of Newton iterations.

    Returns:
        RootResultVector[dtype] with the root estimate and diagnostics.

    Raises:
        Error: If `f(x0)` is not the same length as `x0`.
        Error: If an unsupported method is requested.
    """
    comptime if method == "newton":
        return _newton_jac[dtype, f, jac](x0, args, atol, rtol, maxiter)
    else:
        raise Error("Scijo [root]: Unsupported method: " + String(method))


def root[
    dtype: DType,
    f: def[dtype: DType](x: NDArray[dtype]) raises thin capturing[_] -> NDArray[
        dtype
    ] where dtype.is_floating_point(),
    *,
    method: String = "newton",
](
    x0: NDArray[dtype],
    atol: Scalar[dtype] = 1e-8,
    rtol: Scalar[dtype] = 1e-8,
    maxiter: Int = 100,
    fd_step: Scalar[dtype] = 1e-6,
) raises -> RootResultVector[dtype] where dtype.is_floating_point():
    """Finds a root of a vector-valued function (args-free finite-difference overload).

    For functions that do not need `args`: `f` takes only `x`, so extra
    parameters (if any) are captured directly from the enclosing scope
    instead of threaded through `args`. See the `args`-taking overload
    `root[dtype, f](..., args=...)` for a function reused across call
    sites with different `args` values.

    Parameters:
        dtype: The floating-point data type.
        f: Vector-valued function with signature `def(x) -> NDArray[dtype]`.
        method: Root-finding method: `"newton"` or `"broyden1"`.

    Args:
        x0: Initial guess, of shape `(n,)`.
        atol: Absolute convergence tolerance on `||f(x)||`.
        rtol: Relative convergence tolerance on the step size.
        maxiter: Maximum number of iterations.
        fd_step: Finite-difference step size used to estimate the Jacobian
            (only used by `method="newton"`).

    Returns:
        RootResultVector[dtype] with the root estimate and diagnostics.

    Raises:
        Error: If `f(x0)` is not the same length as `x0`.
        Error: If an unsupported method is requested.
    """

    def _with_args[
        dtype2: DType
    ](
        x: NDArray[dtype2], args: Optional[List[Scalar[dtype2]]]
    ) capturing raises -> NDArray[dtype2] where dtype2.is_floating_point():
        return f(x)

    return root[dtype, _with_args, method=method](
        x0=x0, atol=atol, rtol=rtol, maxiter=maxiter, fd_step=fd_step
    )


def root[
    dtype: DType,
    f: def[dtype: DType](x: NDArray[dtype]) raises thin capturing[_] -> NDArray[
        dtype
    ] where dtype.is_floating_point(),
    jac: def[dtype: DType](x: NDArray[dtype]) raises capturing[_] -> NDArray[
        dtype
    ],
    *,
    method: String = "newton",
](
    x0: NDArray[dtype],
    atol: Scalar[dtype] = 1e-8,
    rtol: Scalar[dtype] = 1e-8,
    maxiter: Int = 100,
) raises -> RootResultVector[dtype] where dtype.is_floating_point():
    """Finds a root of a vector-valued function (args-free analytic-Jacobian overload).

    For functions that do not need `args`: `f` and `jac` take only `x`, so
    extra parameters (if any) are captured directly from the enclosing
    scope instead of threaded through `args`. See the `args`-taking
    overload `root[dtype, f, jac](..., args=...)` for functions reused
    across call sites with different `args` values.

    Parameters:
        dtype: The floating-point data type.
        f: Vector-valued function with signature `def(x) -> NDArray[dtype]`.
        jac: Jacobian of `f`, returning an `(n, n)` matrix.
        method: Root-finding method. Only `"newton"` is currently supported.

    Args:
        x0: Initial guess, of shape `(n,)`.
        atol: Absolute convergence tolerance on `||f(x)||`.
        rtol: Relative convergence tolerance on the Newton step size.
        maxiter: Maximum number of Newton iterations.

    Returns:
        RootResultVector[dtype] with the root estimate and diagnostics.

    Raises:
        Error: If `f(x0)` is not the same length as `x0`.
        Error: If an unsupported method is requested.
    """

    def _f_with_args[
        dtype2: DType
    ](
        x: NDArray[dtype2], args: Optional[List[Scalar[dtype2]]]
    ) capturing raises -> NDArray[dtype2] where dtype2.is_floating_point():
        return f(x)

    def _jac_with_args[
        dtype2: DType
    ](
        x: NDArray[dtype2], args: Optional[List[Scalar[dtype2]]]
    ) capturing raises -> NDArray[dtype2] where dtype2.is_floating_point():
        return jac(x)

    return root[dtype, _f_with_args, _jac_with_args, method=method](
        x0=x0, atol=atol, rtol=rtol, maxiter=maxiter
    )


def _newton_fd[
    dtype: DType,
    f: def[dtype: DType](
        x: NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) raises thin capturing[_] -> NDArray[
        dtype
    ] where dtype.is_floating_point(),
](
    x0: NDArray[dtype],
    args: Optional[List[Scalar[dtype]]],
    atol: Scalar[dtype],
    rtol: Scalar[dtype],
    maxiter: Int,
    fd_step: Scalar[dtype],
) raises -> RootResultVector[dtype] where dtype.is_floating_point():
    """Multivariate Newton's method with a finite-difference Jacobian."""
    var x = x0.copy()
    var n = len(x)
    var fx = f(x, args)
    var nfev = 1

    if len(fx) != n:
        raise Error(
            "Scijo [root]: f(x) must return a vector of the same length as"
            " x (got "
            + String(len(fx))
            + " for n="
            + String(n)
            + ")."
        )

    var it = 0
    while it < maxiter:
        if norm(fx) <= atol:
            return RootResultVector[dtype](
                x^, fx^, it, nfev, True, "converged", "newton"
            )

        var J = jacobian[dtype, f](x, args, step=fd_step)
        nfev += 2 * n
        var dx = solve(J, fx)
        x = x - dx
        fx = f(x, args)
        nfev += 1
        it += 1

        if norm(dx) <= rtol * (norm(x) + atol):
            break

    var converged = norm(fx) <= atol
    return RootResultVector[dtype](
        x^,
        fx^,
        it,
        nfev,
        converged,
        "converged" if converged else "maximum iterations exceeded",
        "newton",
    )


def _newton_jac[
    dtype: DType,
    f: def[dtype: DType](
        x: NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) raises thin capturing[_] -> NDArray[
        dtype
    ] where dtype.is_floating_point(),
    jac: def[dtype: DType](
        x: NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) raises thin capturing[_] -> NDArray[
        dtype
    ] where dtype.is_floating_point(),
](
    x0: NDArray[dtype],
    args: Optional[List[Scalar[dtype]]],
    atol: Scalar[dtype],
    rtol: Scalar[dtype],
    maxiter: Int,
) raises -> RootResultVector[dtype] where dtype.is_floating_point():
    """Multivariate Newton's method with an analytic Jacobian."""
    var x = x0.copy()
    var n = len(x)
    var fx = f(x, args)
    var nfev = 1

    if len(fx) != n:
        raise Error(
            "Scijo [root]: f(x) must return a vector of the same length as"
            " x (got "
            + String(len(fx))
            + " for n="
            + String(n)
            + ")."
        )

    var it = 0
    while it < maxiter:
        if norm(fx) <= atol:
            return RootResultVector[dtype](
                x^, fx^, it, nfev, True, "converged", "newton"
            )

        var J = jac(x, args)
        var dx = solve(J, fx)
        x = x - dx
        fx = f(x, args)
        nfev += 1
        it += 1

        if norm(dx) <= rtol * (norm(x) + atol):
            break

    var converged = norm(fx) <= atol
    return RootResultVector[dtype](
        x^,
        fx^,
        it,
        nfev,
        converged,
        "converged" if converged else "maximum iterations exceeded",
        "newton",
    )


def _broyden1[
    dtype: DType,
    f: def[dtype: DType](
        x: NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) raises thin capturing[_] -> NDArray[
        dtype
    ] where dtype.is_floating_point(),
](
    x0: NDArray[dtype],
    args: Optional[List[Scalar[dtype]]],
    atol: Scalar[dtype],
    rtol: Scalar[dtype],
    maxiter: Int,
) raises -> RootResultVector[dtype] where dtype.is_floating_point():
    """Broyden's "good" method: quasi-Newton root-finding without a Jacobian.

    Maintains a running approximation `H` of the inverse Jacobian, starting
    at the identity and updated after every step with a rank-1 correction
    (Sherman-Morrison applied to Broyden's Jacobian update), so `f` is the
    only function ever evaluated - no finite-difference columns, no analytic
    derivative.

    References:
        Broyden, C. G. (1965). "A class of methods for solving nonlinear
        simultaneous equations." Mathematics of Computation, 19(92), 577-593.
    """
    var x = x0.copy()
    var n = len(x)
    var fx = f(x, args)
    var nfev = 1

    if len(fx) != n:
        raise Error(
            "Scijo [root]: f(x) must return a vector of the same length as"
            " x (got "
            + String(len(fx))
            + " for n="
            + String(n)
            + ")."
        )

    var H = zeros[dtype](Shape(n, n))
    for i in range(n):
        H.store(i * n + i, val=Scalar[dtype](1.0))

    var it = 0
    while it < maxiter:
        if norm(fx) <= atol:
            return RootResultVector[dtype](
                x^, fx^, it, nfev, True, "converged", "broyden1"
            )

        # Step: s = -H @ f(x).
        var s = zeros[dtype](Shape(n))
        for i in range(n):
            var acc: Scalar[dtype] = 0.0
            for j in range(n):
                acc += H.load(i * n + j) * fx.load(j)
            s.store(i, val=-acc)

        var x_new = x + s
        var fx_new = f(x_new, args)
        nfev += 1
        var y = fx_new - fx

        # Rank-1 update: H += (s - H@y) (s^T H) / (s^T H y).
        var Hy = zeros[dtype](Shape(n))
        for i in range(n):
            var acc: Scalar[dtype] = 0.0
            for j in range(n):
                acc += H.load(i * n + j) * y.load(j)
            Hy.store(i, val=acc)

        var sH = zeros[dtype](Shape(n))
        for j in range(n):
            var acc: Scalar[dtype] = 0.0
            for i in range(n):
                acc += s.load(i) * H.load(i * n + j)
            sH.store(j, val=acc)

        var denom: Scalar[dtype] = 0.0
        for i in range(n):
            denom += s.load(i) * Hy.load(i)

        if abs(denom) > Scalar[dtype](1e-14):
            for i in range(n):
                var num_i = (s.load(i) - Hy.load(i)) / denom
                for j in range(n):
                    H.store(
                        i * n + j,
                        val=H.load(i * n + j) + num_i * sH.load(j),
                    )
        # Else: the update direction is singular this step; keep the
        # existing H rather than divide by ~0.

        var s_norm = norm(s)
        x = x_new^
        fx = fx_new^
        it += 1

        if s_norm <= rtol * (norm(x) + atol):
            break

    var converged = norm(fx) <= atol
    return RootResultVector[dtype](
        x^,
        fx^,
        it,
        nfev,
        converged,
        "converged" if converged else "maximum iterations exceeded",
        "broyden1",
    )
