# ===----------------------------------------------------------------------=== #
# SciJo: Optimize module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Multi-Dimensional Root-Finding (`scijo.optimize.root`)
=========================================================
Root-finding for vector-valued functions, matching `scipy.optimize.root`'s
frontend signature.
"""

# ===----------------------------------------------------------------------=== #
# External
# ===----------------------------------------------------------------------=== #
from numojo.core.ndarray import NDArray

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
    ) capturing raises -> NDArray[dtype],
    *,
    method: String = "newton",
](
    x0: NDArray[dtype],
    args: Optional[List[Scalar[dtype]]] = None,
    atol: Scalar[dtype] = 1e-8,
    rtol: Scalar[dtype] = 1e-8,
    maxiter: Int = 100,
    fd_step: Scalar[dtype] = 1e-6,
) raises -> RootResultVector[dtype]:
    """Finds a root of a vector-valued function (finite-difference overload).

    `f` must map `R^n -> R^n` (as many outputs as inputs), so that the
    Jacobian is square and invertible at a root. The Jacobian is estimated
    with central finite differences at every iteration (via
    `scijo.differentiate.jacobian`); use the
    `root[dtype, f, jac](...)` overload to supply an analytic Jacobian
    instead and skip that cost.

    Parameters:
        dtype: The floating-point data type.
        f: Vector-valued function with signature `def(x, args) -> NDArray[dtype]`.
        method: Root-finding method. Only `"newton"` is currently supported.

    Args:
        x0: Initial guess, of shape `(n,)`.
        args: Optional arguments forwarded to `f`.
        atol: Absolute convergence tolerance on `||f(x)||`.
        rtol: Relative convergence tolerance on the Newton step size.
        maxiter: Maximum number of Newton iterations.
        fd_step: Finite-difference step size used to estimate the Jacobian.

    Returns:
        RootResultVector[dtype] with the root estimate and diagnostics.

    Raises:
        Error: If `f(x0)` is not the same length as `x0`.
        Error: If an unsupported method is requested.
    """
    comptime if method == "newton":
        return _newton_fd[dtype, f](x0, args, atol, rtol, maxiter, fd_step)
    else:
        raise Error("Scijo [root]: Unsupported method: " + String(method))


def root[
    dtype: DType,
    f: def[dtype: DType](
        x: NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) capturing raises -> NDArray[dtype],
    jac: def[dtype: DType](
        x: NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) capturing raises -> NDArray[dtype],
    *,
    method: String = "newton",
](
    x0: NDArray[dtype],
    args: Optional[List[Scalar[dtype]]] = None,
    atol: Scalar[dtype] = 1e-8,
    rtol: Scalar[dtype] = 1e-8,
    maxiter: Int = 100,
) raises -> RootResultVector[dtype]:
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


def _newton_fd[
    dtype: DType,
    f: def[dtype: DType](
        x: NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) capturing raises -> NDArray[dtype],
](
    x0: NDArray[dtype],
    args: Optional[List[Scalar[dtype]]],
    atol: Scalar[dtype],
    rtol: Scalar[dtype],
    maxiter: Int,
    fd_step: Scalar[dtype],
) raises -> RootResultVector[dtype]:
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
    ) capturing raises -> NDArray[dtype],
    jac: def[dtype: DType](
        x: NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) capturing raises -> NDArray[dtype],
](
    x0: NDArray[dtype],
    args: Optional[List[Scalar[dtype]]],
    atol: Scalar[dtype],
    rtol: Scalar[dtype],
    maxiter: Int,
) raises -> RootResultVector[dtype]:
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
