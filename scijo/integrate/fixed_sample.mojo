# ===----------------------------------------------------------------------=== #
# Scijo: Integrate - Fixed Sample
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Integrate Module - Fixed Sample Methods (scijo.integrate.fixed_sample)

Integration methods for discrete, evenly or unevenly spaced sample data.
Includes the composite trapezoidal rule, Simpson's rule, and Romberg integration.
"""

from numojo.core.ndarray import NDArray, NDArrayShape
import numojo as nm


fn trapezoid[
    dtype: DType
](
    y: NDArray[dtype],
    dx: Scalar[dtype] = 1.0,
    axis: Int = -1,
) raises -> Scalar[
    dtype
] where dtype.is_floating_point():
    """Integrates along the given axis using the composite trapezoidal rule.

    Computes ∫ y(x) dx using evenly spaced points with spacing `dx`.

    Parameters:
        dtype: The floating-point data type.

    Args:
        y: Input array to integrate. Must be 1-D.
        dx: The spacing between sample points. Defaults to 1.0.
        axis: The axis along which to integrate. Currently only 1-D is supported.

    Returns:
        Definite integral approximated by the trapezoidal rule.
        Returns 0.0 for arrays with fewer than 2 elements.

    Raises:
        Error: If y is not 1-D.
        Error: If y is empty.
    """

    if y.ndim != 1:
        raise Error(
            t"Scijo [trapezoid]: Expected y to be 1-D array, received"
            t" ndim={{y.ndim}}."
        )

    if y.size == 0:
        raise Error(
            t"Scijo [trapezoid]: y.size = 0, Cannot interage over an empty"
            t" array."
        )

    if y.size == 1:
        return Scalar[dtype](0.0)

    var integral: Scalar[dtype] = 0.0
    for i in range(y.size - 1):
        var y_i = y.item(i)
        var y_i1 = y.item(i + 1)
        integral += (y_i + y_i1) * dx * 0.5

    return integral


fn trapezoid[
    dtype: DType
](
    y: NDArray[dtype],
    x: NDArray[dtype],
    axis: Int = -1,
) raises -> Scalar[
    dtype
] where dtype.is_floating_point():
    """Integrates along the given axis using the composite trapezoidal rule.

    Computes ∫ y(x) dx along the parametric curve defined by `x` and `y`.

    Parameters:
        dtype: The floating-point data type.

    Args:
        y: Input array to integrate. Must be 1-D.
        x: Array of sample points corresponding to the y values.
        axis: The axis along which to integrate. Currently only 1-D is supported.

    Returns:
        Definite integral approximated by the trapezoidal rule.
        Returns 0.0 for arrays with fewer than 2 elements.

    Raises:
        Error: If y or x are not 1-D, or if their sizes differ.
        Error: If y is empty.
    """
    if y.ndim != 1:
        raise Error(
            NumojoError(
                category="shape",
                message=String(
                    "Expected y to be 1-D, received ndim={}. Pass a 1-D NDArray"
                    " for y (e.g. shape (N,))."
                ).format(y.ndim),
                location="trapezoid(y, x)",
            )
        )

    if y.size == 0:
        raise Error(
            NumojoError(
                category="value",
                message=(
                    "Cannot integrate over an empty array. Provide a non-empty"
                    " array for y."
                ),
                location="trapezoid(y, x)",
            )
        )

    if y.size == 1:
        return Scalar[dtype](0.0)

    if x.ndim != 1:
        raise Error(
            NumojoError(
                category="shape",
                message=String(
                    "Expected x to be 1-D, received ndim={}. Provide a 1-D"
                    " NDArray for x."
                ).format(x.ndim),
                location="trapezoid(y, x)",
            )
        )

    if y.size != x.size:
        raise Error(
            NumojoError(
                category="shape",
                message=(
                    String(
                        "Size mismatch: y.size={} != x.size={}. Ensure x and y"
                        " have identical lengths."
                    ).format(y.size, x.size)
                ),
                location="trapezoid(y, x)",
            )
        )

    var integral: Scalar[dtype] = 0.0
    for i in range(y.size - 1):
        var y_i = y.item(i)
        var y_i1 = y.item(i + 1)
        var x_i = x.item(i)
        var x_i1 = x.item(i + 1)
        var dx_segment = x_i1 - x_i
        integral += (y_i + y_i1) * dx_segment * 0.5

    return integral


fn simpson[
    dtype: DType
](
    y: NDArray[dtype],
    dx: Scalar[dtype] = 1.0,
    axis: Int = -1,
) raises -> Scalar[
    dtype
]:
    """Integrates along the given axis using Simpson's rule.

    Computes ∫ y(x) dx using evenly spaced points with spacing `dx`.

    Parameters:
        dtype: The floating-point data type.

    Args:
        y: Input array to integrate. Must be 1-D.
        dx: The spacing between sample points. Defaults to 1.0.
        axis: The axis along which to integrate. Currently only 1-D is supported.

    Returns:
        Definite integral approximated by Simpson's rule.

    Raises:
        Error: If y is not 1-D.
    """
    if y.ndim != 1:
        raise Error(
            NumojoError(
                category="shape",
                message=String(
                    "Expected y to be 1-D, received ndim={}. Pass a 1-D NDArray"
                    " for y (e.g. shape (N,)). Only 1-D arrays are supported"
                    " currently."
                ).format(y.ndim),
                location="simpson(y, dx=1.0)",
            )
        )
    var integral: Scalar[dtype] = 0.0
    comptime multiplier: Scalar[dtype] = 1.0 / 6.0
    for i in range(0, y.size - 1, 2):
        integral += (
            multiplier
            * dx
            * 2
            * (y.item(i) + 4.0 * y.item(i + 1) + y.item(i + 2))
        )

    return integral


fn simpson[
    dtype: DType
](
    y: NDArray[dtype],
    x: NDArray[dtype],
    axis: Int = -1,
) raises -> Scalar[
    dtype
]:
    """Integrates along the given axis using Simpson's rule.

    Computes ∫ y(x) dx along the parametric curve defined by `x` and `y`.
    For arrays with an even number of points, the last panel falls back to
    the trapezoidal rule.

    Parameters:
        dtype: The floating-point data type.

    Args:
        y: Input array to integrate. Must be 1-D.
        x: Array of sample points corresponding to the y values.
        axis: The axis along which to integrate. Currently only 1-D is supported.

    Returns:
        Definite integral approximated by Simpson's rule.

    Raises:
        Error: If y or x are not 1-D, or if their sizes differ.
    """
    if y.ndim != 1:
        raise Error(
            NumojoError(
                category="shape",
                message=String(
                    "Expected y to be 1-D, received ndim={}. Pass a 1-D NDArray"
                    " for y (e.g. shape (N,)). Only 1-D arrays are supported"
                    " currently."
                ).format(y.ndim),
                location="simpson(y, x)",
            )
        )

    if x.ndim != 1:
        raise Error(
            NumojoError(
                category="shape",
                message=String(
                    "Expected x to be 1-D, received ndim={}. Provide a 1-D"
                    " NDArray for x."
                ).format(x.ndim),
                location="simpson(y, x)",
            )
        )

    if y.size != x.size:
        raise Error(
            NumojoError(
                category="shape",
                message=(
                    String(
                        "Size mismatch: y.size={} != x.size={}. Ensure x and y"
                        " have identical lengths."
                    ).format(y.size, x.size)
                ),
                location="simpson(y, x)",
            )
        )

    var integral: Scalar[dtype] = 0.0
    comptime multiplier: Scalar[dtype] = 1.0 / 6.0
    for i in range(1, y.size - 1, 2):
        var dx_segment = x.item(i + 1) - x.item(i - 1)
        integral += (
            multiplier
            * dx_segment
            * (y.item(i - 1) + 4.0 * y.item(i) + y.item(i + 1))
        )

    if y.size % 2 == 0:
        var y_n1 = y.item(y.size - 2)
        var y_n = y.item(y.size - 1)
        var x_n1 = x.item(x.size - 2)
        var x_n = x.item(x.size - 1)
        var dx_last = x_n - x_n1
        integral += (y_n1 + y_n) * dx_last * 0.5

    return integral


# TODO: fix the loop implementation.
fn romb[
    dtype: DType
](y: NDArray[dtype], dx: Scalar[dtype] = 1.0, axis: Int = -1) raises -> Scalar[
    dtype
]:
    """Integrates along the given axis using Romberg integration.

    Computes ∫ y(x) dx using evenly spaced points with spacing `dx` and
    Richardson extrapolation for accelerated convergence.

    Parameters:
        dtype: The floating-point data type.

    Args:
        y: Input array to integrate. Must be 1-D.
        dx: The spacing between sample points. Defaults to 1.0.
        axis: The axis along which to integrate. Currently only 1-D is supported.

    Returns:
        Definite integral approximated by Romberg integration.

    Raises:
        Error: If y is not 1-D.
    """
    var maxiter: Int = 10
    if y.ndim != 1:
        raise Error(
            NumojoError(
                category="shape",
                message=String(
                    "Expected y to be 1-D, received ndim={}. Pass a 1-D NDArray"
                    " for y (e.g. shape (N,)). Only 1-D arrays are supported"
                    " currently."
                ).format(y.ndim),
                location="romb(y, dx=1.0)",
            )
        )

    var step: Scalar[dtype] = dx
    var Rone: NDArray[dtype] = nm.zeros[dtype](NDArrayShape(maxiter))
    var Rtwo: NDArray[dtype] = nm.zeros[dtype](NDArrayShape(maxiter))

    var R1 = Rone.unsafe_ptr()
    var R2 = Rtwo.unsafe_ptr()

    R1[0] = 0.5 * dx * (y.item(0) + y.item(y.size - 1))

    for i in range(1, maxiter):
        step /= 2.0
        var c: Scalar[dtype] = 0
        var ep: Int = 2 * (i - 1)
        for j in range(1, ep + 1):
            c += y.item(Int(2 * j - 1))
        R2[0] = step * c + 0.5 * R1[0]

        for j in range(1, i + 1):
            var const: Scalar[dtype] = Scalar[dtype](4.0) ** j
            R2[j] = (const * R2[j - 1] - R1[j - 1]) / (const - 1.0)

        if i > 1 and abs(R1[i - 1] - R2[i]) < Scalar[dtype](1e-6):
            return R2[i]

        var temp = R1
        R1 = R2
        R2 = temp

    return Rone.item(maxiter - 1)
