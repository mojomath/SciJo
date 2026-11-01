# ===----------------------------------------------------------------------=== #
# SciJo: Optimize module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Nonlinear Least Squares (`scijo.optimize.least_squares`)
===========================================================
Minimizes `0.5 * sum(f(x)**2)` for a vector-valued residual function `f`,
via the Levenberg-Marquardt algorithm.
"""

# ===----------------------------------------------------------------------=== #
# External
# ===----------------------------------------------------------------------=== #
from numojo.core import Shape
from numojo.core.ndarray import NDArray
from numojo.routines.linalg.products import matmul
from numojo.routines.manipulation import transpose

# ===----------------------------------------------------------------------=== #
# SciJo
# ===----------------------------------------------------------------------=== #
from scijo.differentiate.jacob import jacobian
from scijo.linalg.norms import norm
from scijo.linalg.solving import solve
from scijo.optimize.utility import LeastSquaresResult


def least_squares[
    dtype: DType,
    f: def[dtype: DType](
        x: NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) raises thin capturing[_] -> NDArray[
        dtype
    ] where dtype.is_floating_point(),
    *,
    method: String = "lm",
](
    x0: NDArray[dtype],
    args: Optional[List[Scalar[dtype]]] = None,
    atol: Scalar[dtype] = 1e-8,
    rtol: Scalar[dtype] = 1e-8,
    maxiter: Int = 100,
    fd_step: Scalar[dtype] = 1e-6,
) raises -> LeastSquaresResult[dtype] where dtype.is_floating_point():
    """Minimizes `0.5 * sum(f(x)**2)` over a vector-valued residual `f`.

    Unlike `scijo.optimize.root`, `f` may return more (or fewer) components
    than `x` has — the classic curve-fitting setting, where `f` is one
    residual per data point and `x` is the (usually much shorter) parameter
    vector. With `method="lm"` (the only method currently supported), the
    Levenberg-Marquardt algorithm forms a finite-difference Jacobian at
    every iteration (via `scijo.differentiate.jacobian`) and damps the
    Gauss-Newton step `(J^T J) dx = -J^T f` by adding `lambda * diag(J^T J)`
    to the normal-equations matrix, increasing `lambda` whenever a trial
    step fails to reduce the cost and decreasing it otherwise.

    Parameters:
        dtype: The floating-point data type.
        f: Vector-valued residual function with signature
            `def(x, args) -> NDArray[dtype]`.
        method: Least-squares method. Only `"lm"` is currently supported.

    Args:
        x0: Initial guess for the parameters, of shape `(n,)`.
        args: Optional arguments forwarded to `f`.
        atol: Absolute convergence tolerance on `||J^T f||` (the gradient
            of the cost).
        rtol: Relative convergence tolerance on the step size.
        maxiter: Maximum number of iterations.
        fd_step: Finite-difference step size used to estimate the Jacobian.

    Returns:
        LeastSquaresResult[dtype] with the solution and diagnostics.

    Raises:
        Error: If an unsupported method is requested.
    """
    comptime if method == "lm":
        return _levenberg_marquardt[dtype, f](
            x0, args, atol, rtol, maxiter, fd_step
        )
    else:
        raise Error(
            "Scijo [least_squares]: Unsupported method: " + String(method)
        )


def least_squares[
    dtype: DType,
    f: def[dtype: DType](x: NDArray[dtype]) raises thin capturing[_] -> NDArray[
        dtype
    ] where dtype.is_floating_point(),
    *,
    method: String = "lm",
](
    x0: NDArray[dtype],
    atol: Scalar[dtype] = 1e-8,
    rtol: Scalar[dtype] = 1e-8,
    maxiter: Int = 100,
    fd_step: Scalar[dtype] = 1e-6,
) raises -> LeastSquaresResult[dtype] where dtype.is_floating_point():
    """Minimizes `0.5 * sum(f(x)**2)` (args-free overload).

    For functions that do not need `args`: `f` takes only `x`, so extra
    parameters (if any) are captured directly from the enclosing scope
    instead of threaded through `args`. See the `args`-taking overload
    `least_squares[dtype, f](..., args=...)` for a function reused across
    call sites with different `args` values.

    Parameters:
        dtype: The floating-point data type.
        f: Vector-valued residual function with signature
            `def(x) -> NDArray[dtype]`.
        method: Least-squares method. Only `"lm"` is currently supported.

    Args:
        x0: Initial guess for the parameters, of shape `(n,)`.
        atol: Absolute convergence tolerance on `||J^T f||` (the gradient
            of the cost).
        rtol: Relative convergence tolerance on the step size.
        maxiter: Maximum number of iterations.
        fd_step: Finite-difference step size used to estimate the Jacobian.

    Returns:
        LeastSquaresResult[dtype] with the solution and diagnostics.

    Raises:
        Error: If an unsupported method is requested.
    """

    def _with_args[
        dtype2: DType
    ](
        x: NDArray[dtype2], args: Optional[List[Scalar[dtype2]]]
    ) capturing raises -> NDArray[dtype2] where dtype2.is_floating_point():
        return f(x)

    return least_squares[dtype, _with_args, method=method](
        x0=x0, atol=atol, rtol=rtol, maxiter=maxiter, fd_step=fd_step
    )


def _levenberg_marquardt[
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
) raises -> LeastSquaresResult[dtype] where dtype.is_floating_point():
    """Levenberg-Marquardt nonlinear least squares with a finite-difference
    Jacobian.

    References:
        Marquardt, D. W. (1963). "An Algorithm for Least-Squares Estimation
        of Nonlinear Parameters." SIAM Journal on Applied Mathematics,
        11(2), 431-441.
    """
    var x = x0.copy()
    var n = len(x)
    var r = f(x, args)
    var nfev = 1
    var cost = 0.5 * norm(r) ** 2
    var lam: Scalar[dtype] = 1e-3

    var it = 0
    while it < maxiter:
        var J = jacobian[dtype, f](x, args, step=fd_step)
        nfev += 2 * n
        var Jt = transpose(J)
        var JTJ = matmul(Jt, J)
        var JTr = matmul(Jt, r)

        if norm(JTr) <= atol:
            return LeastSquaresResult[dtype](
                x^, r^, cost, it, nfev, True, "converged (gtol)", "lm"
            )

        var accepted = False
        var x_new = x.copy()
        var r_new = r.copy()
        var cost_new = cost
        var dx = x.copy()
        var inner = 0
        while inner < 30:
            var A = JTJ.copy()
            for i in range(n):
                A.store(i * n + i, val=A.load(i * n + i) * (1.0 + lam))
            dx = solve(A, -JTr)
            x_new = x + dx
            r_new = f(x_new, args)
            nfev += 1
            cost_new = 0.5 * norm(r_new) ** 2
            if cost_new < cost:
                accepted = True
                var shrunk = lam / 10.0
                lam = shrunk if shrunk > 1e-12 else Scalar[dtype](1e-12)
                break
            else:
                lam *= 10.0
                inner += 1

        if not accepted:
            return LeastSquaresResult[dtype](
                x^,
                r^,
                cost,
                it,
                nfev,
                False,
                "unable to reduce cost further (stalled)",
                "lm",
            )

        var dx_norm = norm(dx)
        x = x_new^
        r = r_new^
        cost = cost_new
        it += 1

        if dx_norm <= rtol * (norm(x) + atol):
            return LeastSquaresResult[dtype](
                x^, r^, cost, it, nfev, True, "converged (xtol)", "lm"
            )

    return LeastSquaresResult[dtype](
        x^, r^, cost, it, nfev, False, "maximum iterations exceeded", "lm"
    )
