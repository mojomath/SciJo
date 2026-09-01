# ===----------------------------------------------------------------------=== #
# SciJo: Differentiate module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Jacobian Matrix Computation (`scijo.differentiate.jacob`)
============================================================
Computes the Jacobian matrix of a vector-valued function using central finite
differences with parallelized column evaluation.

Examples
--------
    ```mojo
    from scijo.prelude import *
    from scijo.differentiate import jacobian

    def f[dtype: DType](x: NDArray[dtype], args: Optional[List[Scalar[dtype]]]) capturing raises -> NDArray[dtype]:
        return x * x

    var x = nm.array[f64]([1.0, 2.0])
    var J = jacobian[f64, f](x)
    ```
"""

# ===----------------------------------------------------------------------=== #
# Max
# ===----------------------------------------------------------------------=== #
from max.algorithm.backend.cpu import parallelize

# ===----------------------------------------------------------------------=== #
# External
# ===----------------------------------------------------------------------=== #
from numojo.core import (
    NDArray,
    Shape,
)
from numojo.routines.creation import zeros


def jacobian[
    dtype: DType,
    jacob_func: def[dtype: DType](
        x: NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) capturing raises -> NDArray[dtype],
](
    x: NDArray[dtype],
    args: Optional[List[Scalar[dtype]]] = None,
    step: Scalar[dtype] = 0.5,
) raises -> NDArray[dtype]:
    """Computes the Jacobian matrix of a vector-valued function using central finite differences.

    Evaluates J[i, j] = ∂f_i/∂x_j using the central difference formula
    (f(x + h*e_j) - f(x - h*e_j)) / (2h), where e_j is the j-th unit vector.
    Each column of the Jacobian is computed in parallel.

    Parameters:
        dtype: The floating-point data type.
        jacob_func: Vector-valued function with signature def(x, args) -> NDArray[dtype].

    Args:
        x: Input vector of shape (n,) at which to evaluate the Jacobian.
        args: Optional arguments to pass to the function.
        step: Finite difference step size. Defaults to 0.5.

    Returns:
        NDArray[dtype] of shape (m, n) representing the Jacobian matrix,
        where m is the output dimension and n is the input dimension.

    Raises:
        Error: If function evaluation fails for any column perturbation.

    Examples:
        ```mojo
        import numojo as nm
        from scijo.differentiate import jacobian
        from scijo.prelude import *

        def f[dtype: DType](x: NDArray[dtype], args: Optional[List[Scalar[dtype]]]) capturing raises -> NDArray[dtype]:
            return x * x

        var x = nm.array[f64]([1.0, 2.0])
        var J = jacobian[f64, f](x)
        ```
    """
    var n: Int = len(x)
    var f0: NDArray[dtype] = jacob_func(x, args)
    var m: Int = len(f0)

    var jacob: NDArray[dtype] = zeros[dtype](Shape(m, n))
    var errors = List[String]()

    @parameter
    def closure(j: Int):
        try:
            var x_plus = x.copy()
            var x_minus = x.copy()
            x_plus.store(j, val=x.load(j) + step)
            x_minus.store(j, val=x.load(j) - step)
            var f_plus = jacob_func(x_plus, args)
            var f_minus = jacob_func(x_minus, args)

            var col: NDArray[dtype] = (f_plus - f_minus) / (2.0 * step)

            for i in range(m):
                jacob.store(i * n + j, val=col.load(i))
        except e:
            errors.append("column " + String(j) + ": " + String(e))

    parallelize[closure](n, num_workers=n)

    if len(errors) > 0:
        raise Error("SciJo [jacobian]: " + errors[0])

    return jacob^


# ===----------------------------------------------------------------------=== #
# Hessian
# ===----------------------------------------------------------------------=== #


def hessian[
    dtype: DType,
    hess_func: def[dtype: DType](
        x: NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) capturing raises -> Scalar[dtype],
](
    x: NDArray[dtype],
    args: Optional[List[Scalar[dtype]]] = None,
    step: Scalar[dtype] = 1e-5,
) raises -> NDArray[dtype]:
    """Computes the Hessian matrix of a scalar-valued function.

    Uses the central finite difference formula for second-order mixed partial
    derivatives, matching `scipy.differentiate.hessian` (numerical approximation):

    H[i, j] = (f(x + h*ei + h*ej) - f(x + h*ei - h*ej)
                - f(x - h*ei + h*ej) + f(x - h*ei - h*ej)) / (4 * h²)

    The diagonal entries use the standard second-derivative formula:

    H[i, i] = (f(x + h*ei) - 2*f(x) + f(x - h*ei)) / h²

    Parameters:
        dtype: The floating-point data type.
        hess_func: Scalar-valued function with signature
            ``def(x: NDArray[dtype], args) -> Scalar[dtype]``.

    Args:
        x: Input vector of shape (n,) at which to evaluate the Hessian.
        args: Optional arguments to pass to the function.
        step: Finite difference step size. Defaults to 1e-5.

    Returns:
        Symmetric NDArray of shape (n, n) containing the Hessian matrix.

    Raises:
        Error: If function evaluation fails.

    Examples:
        ```mojo
        import numojo as nm
        from scijo.differentiate import hessian
        from scijo.prelude import *

        def f[dtype: DType](x: NDArray[dtype], args: Optional[List[Scalar[dtype]]]) capturing raises -> Scalar[dtype]:
            return x.item(0) * x.item(0) + x.item(1) * x.item(1)

        var x = nm.array[f64]([1.0, 2.0])
        var H = hessian[f64, f](x)
        # H ≈ [[2, 0], [0, 2]]
        ```
    """
    var n: Int = x.size
    var H = zeros[dtype](Shape(n, n))
    var f0 = hess_func(x, args)
    var h2 = step * step

    for i in range(n):
        var xi_pp = x.copy()
        var xi_mm = x.copy()
        xi_pp.store(i, val=x.load(i) + step)
        xi_mm.store(i, val=x.load(i) - step)
        var diag = (
            hess_func(xi_pp, args) - 2.0 * f0 + hess_func(xi_mm, args)
        ) / h2
        H.store(i * n + i, val=diag)

        for j in range(i + 1, n):
            var x_pp = x.copy()
            var x_pm = x.copy()
            var x_mp = x.copy()
            var x_mm = x.copy()
            x_pp.store(i, val=x.load(i) + step)
            x_pp.store(j, val=x.load(j) + step)
            x_pm.store(i, val=x.load(i) + step)
            x_pm.store(j, val=x.load(j) - step)
            x_mp.store(i, val=x.load(i) - step)
            x_mp.store(j, val=x.load(j) + step)
            x_mm.store(i, val=x.load(i) - step)
            x_mm.store(j, val=x.load(j) - step)
            var val = (
                hess_func(x_pp, args)
                - hess_func(x_pm, args)
                - hess_func(x_mp, args)
                + hess_func(x_mm, args)
            ) / (4.0 * h2)
            H.store(i * n + j, val=val)
            H.store(j * n + i, val=val)

    return H^
