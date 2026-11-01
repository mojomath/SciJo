# ===----------------------------------------------------------------------=== #
# SciJo: Optimize module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Curve Fitting (`scijo.optimize.curve_fit`)
=============================================
Fits a parametric model to `(xdata, ydata)` data by nonlinear least
squares, via `scijo.optimize.least_squares`.
"""

# ===----------------------------------------------------------------------=== #
# External
# ===----------------------------------------------------------------------=== #
from numojo.core.ndarray import NDArray

# ===----------------------------------------------------------------------=== #
# SciJo
# ===----------------------------------------------------------------------=== #
from scijo.optimize.least_squares import least_squares
from scijo.optimize.utility import LeastSquaresResult


def curve_fit[
    dtype: DType,
    model: def[dtype: DType](
        xdata: NDArray[dtype], params: NDArray[dtype]
    ) raises thin capturing[_] -> NDArray[
        dtype
    ] where dtype.is_floating_point(),
](
    xdata: NDArray[dtype],
    ydata: NDArray[dtype],
    p0: NDArray[dtype],
    atol: Scalar[dtype] = 1e-8,
    rtol: Scalar[dtype] = 1e-8,
    maxiter: Int = 100,
    fd_step: Scalar[dtype] = 1e-6,
) raises -> LeastSquaresResult[dtype] where dtype.is_floating_point():
    """Fits `model(xdata, params)` to `ydata` by nonlinear least squares.

    Minimizes `0.5 * sum((model(xdata, params) - ydata)**2)` over `params`,
    via `scijo.optimize.least_squares` (Levenberg-Marquardt).

    Parameters:
        dtype: The floating-point data type.
        model: Model function with signature
            `def(xdata, params) -> NDArray[dtype]`, evaluated at every
            point of `xdata` for a given `params` vector.

    Args:
        xdata: Independent-variable data, of shape `(m,)`.
        ydata: Dependent-variable data to fit, of shape `(m,)`.
        p0: Initial guess for `params`, of shape `(n,)`.
        atol: Absolute convergence tolerance on `||J^T r||` (the gradient
            of the cost).
        rtol: Relative convergence tolerance on the step size.
        maxiter: Maximum number of iterations.
        fd_step: Finite-difference step size used to estimate the Jacobian.

    Returns:
        LeastSquaresResult[dtype]: `x` holds the fitted `params`; `fun`
        holds the residuals `model(xdata, params) - ydata` at the fit.

    Raises:
        Error: If `model(xdata, p0)` is not the same length as `xdata`.

    Examples:
        ```mojo
        import numojo as nm
        from scijo.optimize import curve_fit
        from scijo.prelude import *

        def line[dtype: DType](xdata: NDArray[dtype], params: NDArray[dtype]) capturing raises -> NDArray[dtype]:
            return params.item(0) * xdata + params.item(1)

        var xdata = nm.fromstring[f64]("[0, 1, 2, 3]")
        var ydata = nm.fromstring[f64]("[1, 3, 5, 7]")
        var p0 = nm.fromstring[f64]("[1, 1]")
        var result = curve_fit[f64, line](xdata, ydata, p0)
        # result.x ~= [2, 1]
        ```
    """

    def _residual[
        dtype2: DType
    ](
        params: NDArray[dtype2], args: Optional[List[Scalar[dtype2]]]
    ) capturing raises -> NDArray[dtype2] where dtype2.is_floating_point():
        return model[dtype2](rebind[NDArray[dtype2]](xdata), params) - rebind[
            NDArray[dtype2]
        ](ydata)

    return least_squares[dtype, _residual](
        x0=p0, atol=atol, rtol=rtol, maxiter=maxiter, fd_step=fd_step
    )
