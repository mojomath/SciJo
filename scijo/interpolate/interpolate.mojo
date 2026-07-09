# ===----------------------------------------------------------------------=== #
# SciJo: Interpolate module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Linear Interpolation (`scijo.interpolate.interpolate`)
=========================================================
Linear interpolation utilities for 1-D data. Provides a reusable
`LinearInterpolator` and a functional `interp1d` interface.

Examples
--------
    ```mojo
    from scijo.interpolate import interp1d

    var x = nm.arange[f64](0.0, 1.0, 0.5)
    var y = nm.array[f64]([0.0, 0.25, 1.0])
    var interp = interp1d(x, y, bounds_error=False, fill_value=0.0)
    var yq1 = interp(Scalar[f64](0.25))
    var yq2 = interp(nm.array[f64]([0.1, 0.5, 0.9]))
    ```
"""

from numojo import zeros
from numojo.core import Shape
from scijo.interpolate.utility import (
    _binary_search,
    _validate_interpolation_input,
)


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

    def __init__(
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

        self.x = x.copy()
        self.y = y.copy()
        self.bounds_error = bounds_error
        self.fill_value = fill_value

    def __call__(self, xi: Scalar[Self.dtype]) raises -> Scalar[Self.dtype]:
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
            if self.fill_value:
                return self.fill_value.value()
            if xi < x_min:
                return self.y._buf.ptr[0]
            return self.y._buf.ptr[self.y.size - 1]

        var j: Int = _binary_search(self.x, xi)

        var x0: Scalar[Self.dtype] = self.x._buf.ptr[j - 1]
        var x1: Scalar[Self.dtype] = self.x._buf.ptr[j]
        var y0: Scalar[Self.dtype] = self.y._buf.ptr[j - 1]
        var y1: Scalar[Self.dtype] = self.y._buf.ptr[j]

        var slope: Scalar[Self.dtype] = (y1 - y0) / (x1 - x0)
        return y0 + slope * (xi - x0)

    def __call__(self, xi: NDArray[Self.dtype]) raises -> NDArray[Self.dtype]:
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
                if self.fill_value:
                    result._buf.ptr[i] = self.fill_value.value()
                elif x_val < x_min:
                    result._buf.ptr[i] = self.y._buf.ptr[0]
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
# CubicSpline interpolator
# ===----------------------------------------------------------------------=== #


struct CubicSpline[dtype: DType = DType.float64, bc_type: String = "natural"](
    Copyable, Movable
):
    """A callable natural cubic spline interpolator, matching scipy.interpolate.CubicSpline.

    Constructs a piecewise cubic polynomial that passes through all data points
    with continuous first and second derivatives. Only `bc_type="natural"` is
    currently supported (second derivatives at endpoints are zero).

    Parameters:
        dtype: The floating-point data type. Defaults to DType.float64.
        bc_type: Boundary condition type. Only "natural" is supported.

    Examples:
        ```mojo
        import numojo as nm
        from scijo.interpolate import CubicSpline

        var x = nm.linspace[nm.f64](0.0, 10.0, 11)
        var y = x * x
        var cs = CubicSpline(x, y)
        var yi = cs(nm.linspace[nm.f64](0.5, 9.5, 10))
        ```
    """

    var x: NDArray[Self.dtype]
    var y: NDArray[Self.dtype]
    var _b: NDArray[Self.dtype]
    var _c: NDArray[Self.dtype]
    var _d: NDArray[Self.dtype]

    def __init__(
        out self, x: NDArray[Self.dtype], y: NDArray[Self.dtype]
    ) raises:
        """Constructs the cubic spline from data points.

        Args:
            x: Strictly increasing x-coordinates of the data points.
            y: Y-coordinates of the data points, same length as x.

        Raises:
            Error: If inputs are invalid or bc_type is unsupported.
        """
        comptime if Self.bc_type != "natural":
            raise Error(
                "CubicSpline only supports bc_type='natural' currently."
            )

        _validate_interpolation_input(x, y)
        var n = x.size
        self.x = x.copy()
        self.y = y.copy()

        var h = NDArray[Self.dtype](Shape(n - 1))
        for i in range(n - 1):
            h._buf.ptr[i] = x._buf.ptr[i + 1] - x._buf.ptr[i]

        var alpha = NDArray[Self.dtype](Shape(n))
        alpha._buf.ptr[0] = 0.0
        alpha._buf.ptr[n - 1] = 0.0
        for i in range(1, n - 1):
            alpha._buf.ptr[i] = (
                3.0 * (y._buf.ptr[i + 1] - y._buf.ptr[i]) / h._buf.ptr[i]
                - 3.0 * (y._buf.ptr[i] - y._buf.ptr[i - 1]) / h._buf.ptr[i - 1]
            )

        var l = NDArray[Self.dtype](Shape(n))
        var mu = NDArray[Self.dtype](Shape(n))
        var z = NDArray[Self.dtype](Shape(n))
        l._buf.ptr[0] = 1.0
        mu._buf.ptr[0] = 0.0
        z._buf.ptr[0] = 0.0

        for i in range(1, n - 1):
            l._buf.ptr[i] = (
                2.0 * (x._buf.ptr[i + 1] - x._buf.ptr[i - 1])
                - h._buf.ptr[i - 1] * mu._buf.ptr[i - 1]
            )
            mu._buf.ptr[i] = h._buf.ptr[i] / l._buf.ptr[i]
            z._buf.ptr[i] = (
                alpha._buf.ptr[i] - h._buf.ptr[i - 1] * z._buf.ptr[i - 1]
            ) / l._buf.ptr[i]

        l._buf.ptr[n - 1] = 1.0
        z._buf.ptr[n - 1] = 0.0

        var c = NDArray[Self.dtype](Shape(n))
        var b = NDArray[Self.dtype](Shape(n - 1))
        var d = NDArray[Self.dtype](Shape(n - 1))
        c._buf.ptr[n - 1] = 0.0

        for j in range(n - 2, -1, -1):
            c._buf.ptr[j] = z._buf.ptr[j] - mu._buf.ptr[j] * c._buf.ptr[j + 1]
            b._buf.ptr[j] = (y._buf.ptr[j + 1] - y._buf.ptr[j]) / h._buf.ptr[
                j
            ] - h._buf.ptr[j] * (c._buf.ptr[j + 1] + 2.0 * c._buf.ptr[j]) / 3.0
            d._buf.ptr[j] = (c._buf.ptr[j + 1] - c._buf.ptr[j]) / (
                3.0 * h._buf.ptr[j]
            )

        self._b = b^
        self._c = c^
        self._d = d^

    def __call__(self, xi: Scalar[Self.dtype]) raises -> Scalar[Self.dtype]:
        """Evaluates the spline at a single point.

        Args:
            xi: The query point.

        Returns:
            Interpolated value at xi. Clamped to boundary values if out of range.
        """
        var n = self.x.size
        var x_min: Scalar[Self.dtype] = self.x._buf.ptr[0]
        var x_max: Scalar[Self.dtype] = self.x._buf.ptr[n - 1]
        if xi <= x_min:
            return self.y._buf.ptr[0]
        if xi >= x_max:
            return self.y._buf.ptr[n - 1]
        var j: Int = _binary_search(self.x, xi) - 1
        if j < 0:
            j = 0
        if j > n - 2:
            j = n - 2
        var dx = xi - self.x._buf.ptr[j]
        return (
            self.y._buf.ptr[j]
            + self._b._buf.ptr[j] * dx
            + self._c._buf.ptr[j] * dx * dx
            + self._d._buf.ptr[j] * dx * dx * dx
        )

    def __call__(self, xi: NDArray[Self.dtype]) raises -> NDArray[Self.dtype]:
        """Evaluates the spline at an array of points.

        Args:
            xi: Array of query points.

        Returns:
            Array of interpolated values with the same shape as xi.
        """
        var n = self.x.size
        var result: NDArray[Self.dtype] = NDArray[Self.dtype](xi.shape)
        var x_min: Scalar[Self.dtype] = self.x._buf.ptr[0]
        var x_max: Scalar[Self.dtype] = self.x._buf.ptr[n - 1]

        for i in range(xi.size):
            var xi_val: Scalar[Self.dtype] = xi._buf.ptr[i]
            if xi_val <= x_min:
                result._buf.ptr[i] = self.y._buf.ptr[0]
                continue
            if xi_val >= x_max:
                result._buf.ptr[i] = self.y._buf.ptr[n - 1]
                continue
            var j: Int = _binary_search(self.x, xi_val) - 1
            if j < 0:
                j = 0
            if j > n - 2:
                j = n - 2
            var dx = xi_val - self.x._buf.ptr[j]
            result._buf.ptr[i] = (
                self.y._buf.ptr[j]
                + self._b._buf.ptr[j] * dx
                + self._c._buf.ptr[j] * dx * dx
                + self._d._buf.ptr[j] * dx * dx * dx
            )

        return result^


# ===----------------------------------------------------------------------=== #
# Akima1DInterpolator
# ===----------------------------------------------------------------------=== #


struct Akima1DInterpolator[dtype: DType = DType.float64](Copyable, Movable):
    """A callable Akima piecewise cubic interpolator, matching scipy.interpolate.Akima1DInterpolator.

    Uses locally-weighted slopes to build a piecewise cubic Hermite polynomial
    that avoids spurious oscillations near outliers.

    Parameters:
        dtype: The floating-point data type. Defaults to DType.float64.

    Examples:
        ```mojo
        import numojo as nm
        from scijo.interpolate import Akima1DInterpolator

        var x = nm.linspace[nm.f64](0.0, 10.0, 11)
        var y = x * x
        var ak = Akima1DInterpolator(x, y)
        var yi = ak(nm.linspace[nm.f64](0.5, 9.5, 10))
        ```
    """

    var x: NDArray[Self.dtype]
    var y: NDArray[Self.dtype]
    var _t: NDArray[Self.dtype]

    def __init__(
        out self, x: NDArray[Self.dtype], y: NDArray[Self.dtype]
    ) raises:
        """Constructs the Akima interpolator from data points.

        Falls back to cubic spline for fewer than 5 points.

        Args:
            x: Strictly increasing x-coordinates of the data points.
            y: Y-coordinates of the data points, same length as x.

        Raises:
            Error: If inputs are invalid.
        """
        _validate_interpolation_input(x, y)
        var n = x.size
        self.x = x.copy()
        self.y = y.copy()

        var slopes = NDArray[Self.dtype](Shape(n - 1))
        for i in range(n - 1):
            slopes._buf.ptr[i] = (y._buf.ptr[i + 1] - y._buf.ptr[i]) / (
                x._buf.ptr[i + 1] - x._buf.ptr[i]
            )

        var t = NDArray[Self.dtype](Shape(n))
        t._buf.ptr[0] = slopes._buf.ptr[0]
        if n > 1:
            t._buf.ptr[n - 1] = slopes._buf.ptr[n - 2]
        if n > 2:
            t._buf.ptr[1] = (slopes._buf.ptr[0] + slopes._buf.ptr[1]) * 0.5
            t._buf.ptr[n - 2] = (
                slopes._buf.ptr[n - 3] + slopes._buf.ptr[n - 2]
            ) * 0.5

        for i in range(2, n - 2):
            var w1 = abs(slopes._buf.ptr[i + 1] - slopes._buf.ptr[i])
            var w2 = abs(slopes._buf.ptr[i - 1] - slopes._buf.ptr[i - 2])
            if w1 + w2 > 0:
                t._buf.ptr[i] = (
                    w1 * slopes._buf.ptr[i - 1] + w2 * slopes._buf.ptr[i]
                ) / (w1 + w2)
            else:
                t._buf.ptr[i] = (
                    slopes._buf.ptr[i - 1] + slopes._buf.ptr[i]
                ) * 0.5

        self._t = t^

    def __call__(self, xi: Scalar[Self.dtype]) raises -> Scalar[Self.dtype]:
        """Evaluates the Akima interpolant at a single point.

        Args:
            xi: The query point.

        Returns:
            Interpolated value at xi. Clamped to boundary values if out of range.
        """
        var n = self.x.size
        var x_min: Scalar[Self.dtype] = self.x._buf.ptr[0]
        var x_max: Scalar[Self.dtype] = self.x._buf.ptr[n - 1]
        if xi <= x_min:
            return self.y._buf.ptr[0]
        if xi >= x_max:
            return self.y._buf.ptr[n - 1]
        var j: Int = _binary_search(self.x, xi) - 1
        if j < 0:
            j = 0
        if j > n - 2:
            j = n - 2
        var h = self.x._buf.ptr[j + 1] - self.x._buf.ptr[j]
        var s = (xi - self.x._buf.ptr[j]) / h
        var s2 = s * s
        var s3 = s2 * s
        return (
            (2.0 * s3 - 3.0 * s2 + 1.0) * self.y._buf.ptr[j]
            + (s3 - 2.0 * s2 + s) * h * self._t._buf.ptr[j]
            + (-2.0 * s3 + 3.0 * s2) * self.y._buf.ptr[j + 1]
            + (s3 - s2) * h * self._t._buf.ptr[j + 1]
        )

    def __call__(self, xi: NDArray[Self.dtype]) raises -> NDArray[Self.dtype]:
        """Evaluates the Akima interpolant at an array of points.

        Args:
            xi: Array of query points.

        Returns:
            Array of interpolated values with the same shape as xi.
        """
        var n = self.x.size
        var result: NDArray[Self.dtype] = NDArray[Self.dtype](xi.shape)
        var x_min: Scalar[Self.dtype] = self.x._buf.ptr[0]
        var x_max: Scalar[Self.dtype] = self.x._buf.ptr[n - 1]

        for i in range(xi.size):
            var xi_val: Scalar[Self.dtype] = xi._buf.ptr[i]
            if xi_val <= x_min:
                result._buf.ptr[i] = self.y._buf.ptr[0]
                continue
            if xi_val >= x_max:
                result._buf.ptr[i] = self.y._buf.ptr[n - 1]
                continue
            var j: Int = _binary_search(self.x, xi_val) - 1
            if j < 0:
                j = 0
            if j > n - 2:
                j = n - 2
            var h = self.x._buf.ptr[j + 1] - self.x._buf.ptr[j]
            var s = (xi_val - self.x._buf.ptr[j]) / h
            var s2 = s * s
            var s3 = s2 * s
            result._buf.ptr[i] = (
                (2.0 * s3 - 3.0 * s2 + 1.0) * self.y._buf.ptr[j]
                + (s3 - 2.0 * s2 + s) * h * self._t._buf.ptr[j]
                + (-2.0 * s3 + 3.0 * s2) * self.y._buf.ptr[j + 1]
                + (s3 - s2) * h * self._t._buf.ptr[j + 1]
            )

        return result^


# ===----------------------------------------------------------------------=== #
# interp1d (constructor)
# ===----------------------------------------------------------------------=== #


# TODO: Add more interpolation methods like 'quadratic'.
# TODO: Add both interpolate and extrapolate fill methods.
def interp1d[
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
# interp (functional)
# ===----------------------------------------------------------------------=== #


def interp[
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
    """Interpolates y values at query points xi (functional interface).

    Similar to ``numpy.interp``: directly returns interpolated values without
    creating a reusable interpolator object. Use ``interp1d`` when you need a
    callable object for repeated evaluations.

    Parameters:
        dtype: The floating-point data type. Defaults to DType.float64.
        type: The interpolation method. Currently supported: "linear", "cubic", "akima".
        fill_method: Out-of-bounds handling: "interpolate" (clamp to boundary
            values) or "extrapolate" (linear extrapolation).

    Args:
        xi: Array of query points at which to interpolate.
        x: Array of x-coordinates of data points, must be strictly increasing.
        y: Array of y-coordinates of data points, same length as x.

    Raises:
        Error: If inputs are invalid or method/fill_method is unsupported.

    Returns:
        NDArray of interpolated values at the points xi.

    Examples:
        ```mojo
        import numojo as nm
        from scijo.interpolate import interp
        from scijo.prelude import *

        var x = nm.arange[f64](0.0, 1.0, 0.5)  # [0.0, 0.5, 1.0]
        var y = x * x  # y = x^2
        var xq = nm.array[f64]([0.1, 0.5, 0.9])
        var yq = interp[f64, type="linear", fill_method="interpolate"](xq, x, y)
        ```
    """
    _validate_interpolation_input(x, y)

    comptime if type == "linear" and fill_method == "extrapolate":
        return _interp1d_linear_extrapolate(xi, x, y)
    elif type == "linear" and fill_method == "interpolate":
        return _interp1d_linear_interpolate(xi, x, y)
    elif type == "cubic" and fill_method == "interpolate":
        return _interp1d_cubic_interpolate(xi, x, y)
    elif type == "akima" and fill_method == "interpolate":
        return _interp1d_akima_interpolate(xi, x, y)
    else:
        raise Error(
            String(
                "Invalid interpolation method: {} with fill_method: {}."
                " Supported: type='linear' with fill_method='interpolate' or"
                " 'extrapolate', and type='cubic'/'akima' with"
                " fill_method='interpolate'"
            ).format(type, fill_method)
        )


# ===----------------------------------------------------------------------=== #
# Internal linear helpers
# ===----------------------------------------------------------------------=== #


def _interp1d_linear_interpolate[
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


def _interp1d_linear_extrapolate[
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


def _interp1d_cubic_interpolate[
    dtype: DType
](xi: NDArray[dtype], x: NDArray[dtype], y: NDArray[dtype]) raises -> NDArray[
    dtype
]:
    """Natural cubic spline interpolation with boundary clamping.

    Constructs a natural cubic spline (second derivatives at endpoints are 0),
    then evaluates it at query points. Points outside `[x[0], x[-1]]` are
    clamped to boundary values for parity with linear interpolate mode.
    """
    var n = x.size
    var result: NDArray[dtype] = NDArray[dtype](xi.shape)
    var x_min: Scalar[dtype] = x._buf.ptr[0]
    var x_max: Scalar[dtype] = x._buf.ptr[n - 1]

    if n < 3:
        return _interp1d_linear_interpolate(xi, x, y)

    var h = NDArray[dtype](Shape(n - 1))
    for i in range(n - 1):
        h._buf.ptr[i] = x._buf.ptr[i + 1] - x._buf.ptr[i]

    var alpha = NDArray[dtype](Shape(n))
    alpha._buf.ptr[0] = 0.0
    alpha._buf.ptr[n - 1] = 0.0
    for i in range(1, n - 1):
        alpha._buf.ptr[i] = (
            3.0 * (y._buf.ptr[i + 1] - y._buf.ptr[i]) / h._buf.ptr[i]
            - 3.0 * (y._buf.ptr[i] - y._buf.ptr[i - 1]) / h._buf.ptr[i - 1]
        )

    var l = NDArray[dtype](Shape(n))
    var mu = NDArray[dtype](Shape(n))
    var z = NDArray[dtype](Shape(n))
    l._buf.ptr[0] = 1.0
    mu._buf.ptr[0] = 0.0
    z._buf.ptr[0] = 0.0

    for i in range(1, n - 1):
        l._buf.ptr[i] = (
            2.0 * (x._buf.ptr[i + 1] - x._buf.ptr[i - 1])
            - h._buf.ptr[i - 1] * mu._buf.ptr[i - 1]
        )
        mu._buf.ptr[i] = h._buf.ptr[i] / l._buf.ptr[i]
        z._buf.ptr[i] = (
            alpha._buf.ptr[i] - h._buf.ptr[i - 1] * z._buf.ptr[i - 1]
        ) / l._buf.ptr[i]

    l._buf.ptr[n - 1] = 1.0
    z._buf.ptr[n - 1] = 0.0

    var c = NDArray[dtype](Shape(n))
    var b = NDArray[dtype](Shape(n - 1))
    var d = NDArray[dtype](Shape(n - 1))
    c._buf.ptr[n - 1] = 0.0

    for j in range(n - 2, -1, -1):
        c._buf.ptr[j] = z._buf.ptr[j] - mu._buf.ptr[j] * c._buf.ptr[j + 1]
        b._buf.ptr[j] = (y._buf.ptr[j + 1] - y._buf.ptr[j]) / h._buf.ptr[
            j
        ] - h._buf.ptr[j] * (c._buf.ptr[j + 1] + 2.0 * c._buf.ptr[j]) / 3.0
        d._buf.ptr[j] = (c._buf.ptr[j + 1] - c._buf.ptr[j]) / (
            3.0 * h._buf.ptr[j]
        )

    for i in range(xi.size):
        var xi_val: Scalar[dtype] = xi._buf.ptr[i]
        if xi_val <= x_min:
            result._buf.ptr[i] = y._buf.ptr[0]
            continue
        if xi_val >= x_max:
            result._buf.ptr[i] = y._buf.ptr[n - 1]
            continue

        var j: Int = _binary_search(x, xi_val) - 1
        if j < 0:
            j = 0
        if j > n - 2:
            j = n - 2

        var dx = xi_val - x._buf.ptr[j]
        result._buf.ptr[i] = (
            y._buf.ptr[j]
            + b._buf.ptr[j] * dx
            + c._buf.ptr[j] * dx * dx
            + d._buf.ptr[j] * dx * dx * dx
        )

    return result^


def _interp1d_akima_interpolate[
    dtype: DType
](xi: NDArray[dtype], x: NDArray[dtype], y: NDArray[dtype]) raises -> NDArray[
    dtype
]:
    """Akima 1D interpolation with boundary clamping."""
    var n = x.size
    var result: NDArray[dtype] = NDArray[dtype](xi.shape)
    var x_min: Scalar[dtype] = x._buf.ptr[0]
    var x_max: Scalar[dtype] = x._buf.ptr[n - 1]

    if n < 5:
        return _interp1d_cubic_interpolate(xi, x, y)

    var slopes = NDArray[dtype](Shape(n - 1))
    for i in range(n - 1):
        slopes._buf.ptr[i] = (y._buf.ptr[i + 1] - y._buf.ptr[i]) / (
            x._buf.ptr[i + 1] - x._buf.ptr[i]
        )

    var t = NDArray[dtype](Shape(n))
    t._buf.ptr[0] = slopes._buf.ptr[0]
    t._buf.ptr[1] = (slopes._buf.ptr[0] + slopes._buf.ptr[1]) * 0.5
    t._buf.ptr[n - 2] = (slopes._buf.ptr[n - 3] + slopes._buf.ptr[n - 2]) * 0.5
    t._buf.ptr[n - 1] = slopes._buf.ptr[n - 2]

    for i in range(2, n - 2):
        var w1 = abs(slopes._buf.ptr[i + 1] - slopes._buf.ptr[i])
        var w2 = abs(slopes._buf.ptr[i - 1] - slopes._buf.ptr[i - 2])
        if w1 + w2 > 0:
            t._buf.ptr[i] = (
                w1 * slopes._buf.ptr[i - 1] + w2 * slopes._buf.ptr[i]
            ) / (w1 + w2)
        else:
            t._buf.ptr[i] = (slopes._buf.ptr[i - 1] + slopes._buf.ptr[i]) * 0.5

    for i in range(xi.size):
        var xi_val: Scalar[dtype] = xi._buf.ptr[i]
        if xi_val <= x_min:
            result._buf.ptr[i] = y._buf.ptr[0]
            continue
        if xi_val >= x_max:
            result._buf.ptr[i] = y._buf.ptr[n - 1]
            continue

        var j: Int = _binary_search(x, xi_val) - 1
        if j < 0:
            j = 0
        if j > n - 2:
            j = n - 2

        var h = x._buf.ptr[j + 1] - x._buf.ptr[j]
        var s = (xi_val - x._buf.ptr[j]) / h
        var s2 = s * s
        var s3 = s2 * s

        var h00 = 2.0 * s3 - 3.0 * s2 + 1.0
        var h10 = s3 - 2.0 * s2 + s
        var h01 = -2.0 * s3 + 3.0 * s2
        var h11 = s3 - s2

        result._buf.ptr[i] = (
            h00 * y._buf.ptr[j]
            + h10 * h * t._buf.ptr[j]
            + h01 * y._buf.ptr[j + 1]
            + h11 * h * t._buf.ptr[j + 1]
        )

    return result^
