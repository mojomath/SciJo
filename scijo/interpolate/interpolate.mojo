# ===----------------------------------------------------------------------=== #
# Scijo: Interpolate - Interpolation Functions
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Interpolate Module - Interpolation Functions (scijo.interpolate.interpolate)

Linear interpolation utilities for 1-D data. Provides a reusable
`LinearInterpolator` and a functional `interp1d` interface.

Example: interp = interp1d(x, y); yq = interp(Scalar[DType.float64](0.5))
"""

from numojo import zeros

from .utility import _binary_search, _validate_interpolation_input


# ===----------------------------------------------------------------------=== #
# Linear interpolator
# ===----------------------------------------------------------------------=== #


# TODO: Add extrapolation and fill_value handling to LinearInterpolator
struct LinearInterpolator[dtype: DType = DType.float64](Copyable, Movable):
    """A callable linear interpolation object similar to scipy.interpolate.interp1d.

    Stores the interpolation data (x, y) and provides a callable interface that
    can interpolate single values or arrays of values using binary search.
    Out-of-bounds behavior is controlled by `bounds_error` and `fill_value`.

    Parameters:
        dtype: The floating-point data type. Defaults to DType.float64.

    Examples:
        ```mojo
        import numojo as nm
        from scijo.interpolate import interp1d
        from scijo.prelude import *

        var x = nm.arange[f64](0.0, 1.0, 0.5)  # [0.0, 0.5, 1.0]
        var y = nm.array[f64]([0.0, 0.25, 1.0])  # y = x^2
        var interp = interp1d(x, y, bounds_error=False, fill_value=0.0)
        var yq1 = interp(Scalar[f64](0.25))
        var yq2 = interp(nm.array[f64]([0.1, 0.5, 0.9]))
        var yq3 = interp(Scalar[f64](1.5))
        ```
    """

    var x: NDArray[Self.dtype]
    """The x-coordinates of the data points."""
    var y: NDArray[Self.dtype]
    """The y-coordinates of the data points."""
    var bounds_error: Bool
    """If True, raise error when interpolating outside bounds."""
    var fill_value: Optional[Scalar[Self.dtype]]
    """Value to use for out-of-bounds points when bounds_error is False."""

    fn __init__(
        out self,
        x: NDArray[Self.dtype],
        y: NDArray[Self.dtype],
        bounds_error: Bool = True,
        fill_value: Optional[Scalar[Self.dtype]] = None,
    ) raises:
        """Initializes the linear interpolator.

        Example: LinearInterpolator(x, y, bounds_error=False, fill_value=0.0)

        Args:
            x: The x-coordinates of the data points, must be strictly increasing.
            y: The y-coordinates of the data points, same length as x.
            bounds_error: If True, raise error when interpolating outside bounds.
                          If False, use fill_value or extrapolate linearly.
            fill_value: Value to use for points outside the data range when
                       bounds_error is False. If None, extrapolate linearly.

        Raises:
            Error: If x and y have different lengths, have fewer than 2 points,
                   or x is not strictly increasing.
        """
        _validate_interpolation_input(x, y)

        self.x = x.deep_copy()
        self.y = y.deep_copy()
        self.bounds_error = bounds_error
        self.fill_value = fill_value

    fn __call__(self, xi: Scalar[Self.dtype]) raises -> Scalar[Self.dtype]:
        """Interpolates a single value.

        Example: yq = interp(Scalar[dtype](0.25))

        Args:
            xi: The point at which to interpolate.

        Returns:
            The interpolated value at xi.

        Raises:
            Error: If bounds_error is True and xi is outside data range.
        """

        var x_min = self.x._buf.ptr[0]
        var x_max = self.x._buf.ptr[self.x.size - 1]

        if xi < x_min or xi > x_max:
            if self.bounds_error:
                raise Error(
                    "Interpolation point "
                    + String(xi)
                    + " is outside data range ["
                    + String(x_min)
                    + ", "
                    + String(x_max)
                    + "]"
                )
            elif self.fill_value:
                return self.fill_value.value()

        if xi <= x_min:
            if self.fill_value:
                return self.fill_value.value()
            return self.y._buf.ptr[0]
        elif xi >= x_max:
            if self.fill_value:
                return self.fill_value.value()
            return self.y._buf.ptr[self.y.size - 1]

        var j: Int = _binary_search(self.x, xi)

        var x0: Scalar[Self.dtype] = self.x._buf.ptr[j - 1]
        var x1: Scalar[Self.dtype] = self.x._buf.ptr[j]
        var y0: Scalar[Self.dtype] = self.y._buf.ptr[j - 1]
        var y1: Scalar[Self.dtype] = self.y._buf.ptr[j]

        var slope: Scalar[Self.dtype] = (y1 - y0) / (x1 - x0)
        return y0 + slope * (xi - x0)

    fn __call__(self, xi: NDArray[Self.dtype]) raises -> NDArray[Self.dtype]:
        """Interpolates an array of values.

        Example: yq = interp(xq_array)

        Args:
            xi: Array of points at which to interpolate.

        Returns:
            Array of interpolated values with the same shape as xi.

        Raises:
            Error: If bounds_error is True and any point in xi is outside data range.
        """
        var result: NDArray[Self.dtype] = zeros[Self.dtype](xi.shape)
        var x_min: Scalar[Self.dtype] = self.x._buf.ptr[0]
        var x_max: Scalar[Self.dtype] = self.x._buf.ptr[self.x.size - 1]

        for i in range(xi.size):
            var x_val: Scalar[Self.dtype] = xi._buf.ptr[i]

            if x_val < x_min or x_val > x_max:
                if self.bounds_error:
                    raise Error(
                        "Interpolation point "
                        + String(x_val)
                        + " is outside data range ["
                        + String(x_min)
                        + ", "
                        + String(x_max)
                        + "]"
                    )
                elif self.fill_value:
                    result._buf.ptr[i] = self.fill_value.value()
                    continue

            if x_val <= x_min:
                if self.fill_value and x_val < x_min:
                    result._buf.ptr[i] = self.fill_value.value()
                else:
                    result._buf.ptr[i] = self.y._buf.ptr[0]
                continue
            elif x_val >= x_max:
                if self.fill_value and x_val > x_max:
                    result._buf.ptr[i] = self.fill_value.value()
                else:
                    result._buf.ptr[i] = self.y._buf.ptr[self.y.size - 1]
                continue

            var j: Int = _binary_search(self.x, x_val)

            var x0: Scalar[Self.dtype] = self.x._buf.ptr[j - 1]
            var x1: Scalar[Self.dtype] = self.x._buf.ptr[j]
            var y0: Scalar[Self.dtype] = self.y._buf.ptr[j - 1]
            var y1: Scalar[Self.dtype] = self.y._buf.ptr[j]

            var slope: Scalar[Self.dtype] = (y1 - y0) / (x1 - x0)
            result._buf.ptr[i] = y0 + slope * (x_val - x0)

        return result^


# ===----------------------------------------------------------------------=== #
# interp1d (constructor)
# ===----------------------------------------------------------------------=== #


# TODO: Add more interpolation methods like 'quadratic', 'cubic'.
# TODO: Add both interpolate and extrapolate fill methods.
fn interp1d[
    dtype: DType = DType.float64
](
    x: NDArray[dtype],
    y: NDArray[dtype],
    bounds_error: Bool = True,
    fill_value: Optional[Scalar[dtype]] = None,
) raises -> LinearInterpolator[dtype]:
    """Creates a callable LinearInterpolator from data points.

    Example: interp = interp1d(x, y, bounds_error=False, fill_value=0.0)

    Parameters:
        dtype: The floating-point data type. Defaults to DType.float64.

    Args:
        x: The x-coordinates of the data points, must be strictly increasing.
        y: The y-coordinates of the data points, same length as x.
        bounds_error: If True, raise error when interpolating outside bounds.
            If False, use fill_value or extrapolate linearly.
        fill_value: Value to use for out-of-bounds points when bounds_error
            is False. If None, extrapolate linearly.

    Raises:
        Error: If x and y have different lengths, have fewer than 2 points,
            or x is not strictly increasing.

    Returns:
        A callable LinearInterpolator object.

    Examples:
        ```mojo
        import numojo as nm
        from scijo.interpolate import interp1d
        from scijo.prelude import *

        var x = nm.arange[f64](0.0, 1.0, 0.5)  # [0.0, 0.5, 1.0]
        var y = nm.array[f64]([0.0, 0.25, 1.0])  # y = x^2
        var interp = interp1d(x, y, bounds_error=False, fill_value=0.0)
        var yq1 = interp(Scalar[f64](0.25))
        var yq2 = interp(nm.array[f64]([0.1, 0.5, 0.9]))
        var yq3 = interp(Scalar[f64](1.5))
        ```
    """
    return LinearInterpolator[dtype](x, y, bounds_error, fill_value)


# ===----------------------------------------------------------------------=== #
# interp1d (functional)
# ===----------------------------------------------------------------------=== #


fn interp1d[
    dtype: DType = DType.float64,
    type: String = "linear",
    fill_method: String = "interpolate",
](
    xi: NDArray[dtype],
    x: NDArray[dtype],
    y: NDArray[dtype],
) raises -> NDArray[
    dtype
]:
    """Interpolates the values of y at the points xi using the specified method.

    Example: yq = interp1d(xq, x, y)

    Functional interface similar to numpy.interp that directly returns
    interpolated values without creating a reusable interpolator object.

    Parameters:
        dtype: The floating-point data type. Defaults to DType.float64.
        type: The interpolation method. Currently supported: "linear".
        fill_method: Out-of-bounds handling: "interpolate" (clamp to boundary
            values) or "extrapolate" (linear extrapolation).

    Args:
        xi: Array of points at which to interpolate.
        x: Array of x-coordinates of data points, must be strictly increasing.
        y: Array of y-coordinates of data points, same length as x.

    Raises:
        Error: If inputs are invalid or method/fill_method is unsupported.

    Returns:
        NDArray of interpolated values at the points xi.

    Examples:
        ```mojo
        import numojo as nm
        from scijo.interpolate import interp1d
        from scijo.prelude import *

        var x = nm.arange[f64](0.0, 1.0, 0.5)  # [0.0, 0.5, 1.0]
        var y = x * x  # y = x^2
        var xq = nm.array[f64]([0.1, 0.5, 0.9])
        var yq = interp1d[f64, type="linear", fill_method="interpolate"](xq, x, y)
        ```
    """
    _validate_interpolation_input(x, y)

    @parameter
    if type == "linear" and fill_method == "extrapolate":
        return _interp1d_linear_extrapolate(xi, x, y)
    elif type == "linear" and fill_method == "interpolate":
        return _interp1d_linear_interpolate(xi, x, y)
    else:
        raise Error(
            String(
                "Invalid interpolation method: {} with fill_method: {}."
                " Supported: type='linear' with fill_method='interpolate' or"
                " 'extrapolate'"
            ).format(type, fill_method)
        )


# ===----------------------------------------------------------------------=== #
# Internal linear helpers
# ===----------------------------------------------------------------------=== #


fn _interp1d_linear_interpolate[
    dtype: DType
](xi: NDArray[dtype], x: NDArray[dtype], y: NDArray[dtype]) raises -> NDArray[
    dtype
]:
    """Linear interpolation with boundary clamping.

    Example: yq = _interp1d_linear_interpolate(xq, x, y)

    For points outside the data range, returns the nearest boundary value.

    Parameters:
        dtype: The floating-point data type.

    Args:
        xi: Array of interpolation points.
        x: Array of x-coordinates (must be sorted).
        y: Array of y-coordinates.

    Returns:
        Array of interpolated values.
    """
    var result: NDArray[dtype] = NDArray[dtype](xi.shape)
    var x_min: Scalar[dtype] = x._buf.ptr[0]
    var x_max: Scalar[dtype] = x._buf.ptr[x.size - 1]

    for i in range(xi.size):
        var xi_val: Scalar[dtype] = xi._buf.ptr[i]

        if xi_val <= x_min:
            result.itemset(i, y._buf.ptr[0])
        elif xi_val >= x_max:
            result.itemset(i, y._buf.ptr[y.size - 1])
        else:
            var j: Int = _binary_search(x, xi_val)

            var x0: Scalar[dtype] = x._buf.ptr[j - 1]
            var x1: Scalar[dtype] = x._buf.ptr[j]
            var y0: Scalar[dtype] = y._buf.ptr[j - 1]
            var y1: Scalar[dtype] = y._buf.ptr[j]
            var t: Scalar[dtype] = (xi_val - x0) / (x1 - x0)
            result._buf.ptr[i] = y0 + t * (y1 - y0)

    return result^


fn _interp1d_linear_extrapolate[
    dtype: DType
](xi: NDArray[dtype], x: NDArray[dtype], y: NDArray[dtype]) raises -> NDArray[
    dtype
]:
    """Linear interpolation with linear extrapolation beyond boundaries.

    Example: yq = _interp1d_linear_extrapolate(xq, x, y)

    For points outside the data range, extrapolates using the slope of the
    nearest boundary segment.

    Parameters:
        dtype: The floating-point data type.

    Args:
        xi: Array of interpolation points.
        x: Array of x-coordinates (must be sorted).
        y: Array of y-coordinates.

    Returns:
        Array of interpolated/extrapolated values.
    """
    var result: NDArray[dtype] = NDArray[dtype](xi.shape)
    var x_min: Scalar[dtype] = x._buf.ptr[0]
    var x_max: Scalar[dtype] = x._buf.ptr[x.size - 1]

    for i in range(xi.size):
        var xi_val: Scalar[dtype] = xi._buf.ptr[i]

        if xi_val < x_min:
            var slope = (y._buf.ptr[1] - y._buf.ptr[0]) / (
                x._buf.ptr[1] - x._buf.ptr[0]
            )
            result.itemset(i, y._buf.ptr[0] + slope * (xi_val - x._buf.ptr[0]))
        elif xi_val > x_max:
            var slope = (y._buf.ptr[y.size - 1] - y._buf.ptr[y.size - 2]) / (
                x._buf.ptr[x.size - 1] - x._buf.ptr[x.size - 2]
            )
            result.itemset(
                i,
                y._buf.ptr[y.size - 1]
                + slope * (xi_val - x._buf.ptr[x.size - 1]),
            )
        else:
            if xi_val == x_min:
                result.itemset(i, y._buf.ptr[0])
            elif xi_val == x_max:
                result.itemset(i, y._buf.ptr[y.size - 1])
            else:
                var j: Int = _binary_search(x, xi_val)

                var x0: Scalar[dtype] = x._buf.ptr[j - 1]
                var x1: Scalar[dtype] = x._buf.ptr[j]
                var y0: Scalar[dtype] = y._buf.ptr[j - 1]
                var y1: Scalar[dtype] = y._buf.ptr[j]
                var t: Scalar[dtype] = (xi_val - x0) / (x1 - x0)
                result._buf.ptr[i] = y0 + t * (y1 - y0)

    return result^


# ===----------------------------------------------------------------------=== #
# Higher-order interpolation methods
# ===----------------------------------------------------------------------=== #


# fn _interp1d_quadratic_interpolate[dtype: DType](
#     xi: NDArray[dtype], x: NDArray[dtype], y: NDArray[dtype]
# ) raises -> NDArray[dtype]:
#     """Quadratic interpolation with boundary clamping."""
#     pass

# fn _interp1d_cubic_interpolate[dtype: DType](
#     xi: NDArray[dtype], x: NDArray[dtype], y: NDArray[dtype]
# ) raises -> NDArray[dtype]:
#     """Cubic interpolation with boundary clamping."""
#     pass
