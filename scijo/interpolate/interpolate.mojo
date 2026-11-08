# ===----------------------------------------------------------------------=== #
# SciJo: Interpolate module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Linear Interpolation (`scijo.interpolate.interpolate`)
=========================================================
Interpolation utilities for 1-D and N-D data. Provides reusable
`LinearInterpolator`/`interp1d` for 1-D data, and `RegularGridInterpolator`/
`interpn` for data sampled on a regular (rectilinear) grid in N dimensions.

Examples
--------
    ```mojo
    from scijo.interpolate import interp1d, RegularGridInterpolator

    var x = nm.arange[f64](0.0, 1.0, 0.5)
    var y = nm.array[f64]([0.0, 0.25, 1.0])
    var interp = interp1d(x, y, bounds_error=False, fill_value=0.0)
    var yq1 = interp(Scalar[f64](0.25))
    var yq2 = interp(nm.array[f64]([0.1, 0.5, 0.9]))
    ```
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.utils.numerics import nan

# ===----------------------------------------------------------------------=== #
# External
# ===----------------------------------------------------------------------=== #
from numojo import zeros
from numojo.core import Shape
from numojo.core.ndarray import NDArray

# ===----------------------------------------------------------------------=== #
# SciJo
# ===----------------------------------------------------------------------=== #
from scijo.interpolate.utility import (
    _binary_search,
    _validate_interpolation_input,
)

# ===----------------------------------------------------------------------=== #
# Linear interpolator
# ===----------------------------------------------------------------------=== #


struct LinearInterpolator[dtype: DType = DType.float64](Copyable, Movable):
    """A callable linear interpolation object.

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

        var x_min = self.x.unsafe_load(0)
        var x_max = self.x.unsafe_load(self.x.size - 1)

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
                return self.y.unsafe_load(0)
            return self.y.unsafe_load(self.y.size - 1)

        var j: Int = _binary_search(self.x, xi)

        var x0: Scalar[Self.dtype] = self.x.unsafe_load(j - 1)
        var x1: Scalar[Self.dtype] = self.x.unsafe_load(j)
        var y0: Scalar[Self.dtype] = self.y.unsafe_load(j - 1)
        var y1: Scalar[Self.dtype] = self.y.unsafe_load(j)

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
        var x_min: Scalar[Self.dtype] = self.x.unsafe_load(0)
        var x_max: Scalar[Self.dtype] = self.x.unsafe_load(self.x.size - 1)

        for i in range(xi.size):
            var x_val: Scalar[Self.dtype] = xi.unsafe_load(i)

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
                    result.unsafe_store(i, self.fill_value.value())
                elif x_val < x_min:
                    result.unsafe_store(i, self.y.unsafe_load(0))
                else:
                    result.unsafe_store(i, self.y.unsafe_load(self.y.size - 1))
                continue

            var j: Int = _binary_search(self.x, x_val)

            var x0: Scalar[Self.dtype] = self.x.unsafe_load(j - 1)
            var x1: Scalar[Self.dtype] = self.x.unsafe_load(j)
            var y0: Scalar[Self.dtype] = self.y.unsafe_load(j - 1)
            var y1: Scalar[Self.dtype] = self.y.unsafe_load(j)

            var slope: Scalar[Self.dtype] = (y1 - y0) / (x1 - x0)
            result.unsafe_store(i, y0 + slope * (x_val - x0))

        return result^


# ===----------------------------------------------------------------------=== #
# CubicSpline interpolator
# ===----------------------------------------------------------------------=== #


struct CubicSpline[dtype: DType = DType.float64, bc_type: String = "natural"](
    Copyable, Movable
):
    """A callable natural cubic spline interpolator.

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
            h.unsafe_store(i, x.unsafe_load(i + 1) - x.unsafe_load(i))

        var alpha = NDArray[Self.dtype](Shape(n))
        alpha.unsafe_store(0, 0.0)
        alpha.unsafe_store(n - 1, 0.0)
        for i in range(1, n - 1):
            alpha.unsafe_store(
                i,
                (
                    3.0
                    * (y.unsafe_load(i + 1) - y.unsafe_load(i))
                    / h.unsafe_load(i)
                    - 3.0
                    * (y.unsafe_load(i) - y.unsafe_load(i - 1))
                    / h.unsafe_load(i - 1)
                ),
            )

        var l = NDArray[Self.dtype](Shape(n))
        var mu = NDArray[Self.dtype](Shape(n))
        var z = NDArray[Self.dtype](Shape(n))
        l.unsafe_store(0, 1.0)
        mu.unsafe_store(0, 0.0)
        z.unsafe_store(0, 0.0)

        for i in range(1, n - 1):
            l.unsafe_store(
                i,
                (
                    2.0 * (x.unsafe_load(i + 1) - x.unsafe_load(i - 1))
                    - h.unsafe_load(i - 1) * mu.unsafe_load(i - 1)
                ),
            )
            mu.unsafe_store(i, h.unsafe_load(i) / l.unsafe_load(i))
            z.unsafe_store(
                i,
                (
                    alpha.unsafe_load(i)
                    - h.unsafe_load(i - 1) * z.unsafe_load(i - 1)
                )
                / l.unsafe_load(i),
            )

        l.unsafe_store(n - 1, 1.0)
        z.unsafe_store(n - 1, 0.0)

        var c = NDArray[Self.dtype](Shape(n))
        var b = NDArray[Self.dtype](Shape(n - 1))
        var d = NDArray[Self.dtype](Shape(n - 1))
        c.unsafe_store(n - 1, 0.0)

        for j in range(n - 2, -1, -1):
            c.unsafe_store(
                j, z.unsafe_load(j) - mu.unsafe_load(j) * c.unsafe_load(j + 1)
            )
            b.unsafe_store(
                j,
                (y.unsafe_load(j + 1) - y.unsafe_load(j)) / h.unsafe_load(j)
                - h.unsafe_load(j)
                * (c.unsafe_load(j + 1) + 2.0 * c.unsafe_load(j))
                / 3.0,
            )
            d.unsafe_store(
                j,
                (c.unsafe_load(j + 1) - c.unsafe_load(j))
                / (3.0 * h.unsafe_load(j)),
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
        var x_min: Scalar[Self.dtype] = self.x.unsafe_load(0)
        var x_max: Scalar[Self.dtype] = self.x.unsafe_load(n - 1)
        if xi <= x_min:
            return self.y.unsafe_load(0)
        if xi >= x_max:
            return self.y.unsafe_load(n - 1)
        var j: Int = _binary_search(self.x, xi) - 1
        if j < 0:
            j = 0
        if j > n - 2:
            j = n - 2
        var dx = xi - self.x.unsafe_load(j)
        return (
            self.y.unsafe_load(j)
            + self._b.unsafe_load(j) * dx
            + self._c.unsafe_load(j) * dx * dx
            + self._d.unsafe_load(j) * dx * dx * dx
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
        var x_min: Scalar[Self.dtype] = self.x.unsafe_load(0)
        var x_max: Scalar[Self.dtype] = self.x.unsafe_load(n - 1)

        for i in range(xi.size):
            var xi_val: Scalar[Self.dtype] = xi.unsafe_load(i)
            if xi_val <= x_min:
                result.unsafe_store(i, self.y.unsafe_load(0))
                continue
            if xi_val >= x_max:
                result.unsafe_store(i, self.y.unsafe_load(n - 1))
                continue
            var j: Int = _binary_search(self.x, xi_val) - 1
            if j < 0:
                j = 0
            if j > n - 2:
                j = n - 2
            var dx = xi_val - self.x.unsafe_load(j)
            result.unsafe_store(
                i,
                (
                    self.y.unsafe_load(j)
                    + self._b.unsafe_load(j) * dx
                    + self._c.unsafe_load(j) * dx * dx
                    + self._d.unsafe_load(j) * dx * dx * dx
                ),
            )

        return result^


# ===----------------------------------------------------------------------=== #
# Akima1DInterpolator
# ===----------------------------------------------------------------------=== #


struct Akima1DInterpolator[dtype: DType = DType.float64](Copyable, Movable):
    """A callable Akima piecewise cubic interpolator.

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
            slopes.unsafe_store(
                i,
                (y.unsafe_load(i + 1) - y.unsafe_load(i))
                / (x.unsafe_load(i + 1) - x.unsafe_load(i)),
            )

        if n == 2:
            # Only one slope exists; the interpolant is just that line.
            var t2 = NDArray[Self.dtype](Shape(2))
            t2.unsafe_store(0, slopes.unsafe_load(0))
            t2.unsafe_store(1, slopes.unsafe_load(0))
            self._t = t2^
            return

        # Extend the slope sequence by two quadratically-extrapolated
        # "phantom" slopes on each side (Akima's original prescription),
        # so the same weighted-average formula applies uniformly at every
        # point, including the boundaries, rather than treating the two
        # boundary points and their neighbors as special cases.
        var mm = NDArray[Self.dtype](Shape(n + 3))
        for i in range(n - 1):
            mm.unsafe_store(i + 2, slopes.unsafe_load(i))
        mm.unsafe_store(1, 2.0 * slopes.unsafe_load(0) - slopes.unsafe_load(1))
        mm.unsafe_store(0, 2.0 * mm.unsafe_load(1) - mm.unsafe_load(2))
        mm.unsafe_store(
            n + 1,
            2.0 * slopes.unsafe_load(n - 2) - slopes.unsafe_load(n - 3),
        )
        mm.unsafe_store(n + 2, 2.0 * mm.unsafe_load(n + 1) - mm.unsafe_load(n))

        var t = NDArray[Self.dtype](Shape(n))
        for i in range(n):
            var f1 = abs(mm.unsafe_load(i + 3) - mm.unsafe_load(i + 2))
            var f2 = abs(mm.unsafe_load(i + 1) - mm.unsafe_load(i))
            if f1 + f2 > 0:
                t.unsafe_store(
                    i,
                    (f1 * mm.unsafe_load(i + 1) + f2 * mm.unsafe_load(i + 2))
                    / (f1 + f2),
                )
            else:
                t.unsafe_store(
                    i,
                    (mm.unsafe_load(i + 1) + mm.unsafe_load(i + 2)) * 0.5,
                )

        self._t = t^

    def __call__(self, xi: Scalar[Self.dtype]) raises -> Scalar[Self.dtype]:
        """Evaluates the Akima interpolant at a single point.

        Args:
            xi: The query point.

        Returns:
            Interpolated value at xi. Clamped to boundary values if out of range.
        """
        var n = self.x.size
        var x_min: Scalar[Self.dtype] = self.x.unsafe_load(0)
        var x_max: Scalar[Self.dtype] = self.x.unsafe_load(n - 1)
        if xi <= x_min:
            return self.y.unsafe_load(0)
        if xi >= x_max:
            return self.y.unsafe_load(n - 1)
        var j: Int = _binary_search(self.x, xi) - 1
        if j < 0:
            j = 0
        if j > n - 2:
            j = n - 2
        var h = self.x.unsafe_load(j + 1) - self.x.unsafe_load(j)
        var s = (xi - self.x.unsafe_load(j)) / h
        var s2 = s * s
        var s3 = s2 * s
        return (
            (2.0 * s3 - 3.0 * s2 + 1.0) * self.y.unsafe_load(j)
            + (s3 - 2.0 * s2 + s) * h * self._t.unsafe_load(j)
            + (-2.0 * s3 + 3.0 * s2) * self.y.unsafe_load(j + 1)
            + (s3 - s2) * h * self._t.unsafe_load(j + 1)
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
        var x_min: Scalar[Self.dtype] = self.x.unsafe_load(0)
        var x_max: Scalar[Self.dtype] = self.x.unsafe_load(n - 1)

        for i in range(xi.size):
            var xi_val: Scalar[Self.dtype] = xi.unsafe_load(i)
            if xi_val <= x_min:
                result.unsafe_store(i, self.y.unsafe_load(0))
                continue
            if xi_val >= x_max:
                result.unsafe_store(i, self.y.unsafe_load(n - 1))
                continue
            var j: Int = _binary_search(self.x, xi_val) - 1
            if j < 0:
                j = 0
            if j > n - 2:
                j = n - 2
            var h = self.x.unsafe_load(j + 1) - self.x.unsafe_load(j)
            var s = (xi_val - self.x.unsafe_load(j)) / h
            var s2 = s * s
            var s3 = s2 * s
            result.unsafe_store(
                i,
                (
                    (2.0 * s3 - 3.0 * s2 + 1.0) * self.y.unsafe_load(j)
                    + (s3 - 2.0 * s2 + s) * h * self._t.unsafe_load(j)
                    + (-2.0 * s3 + 3.0 * s2) * self.y.unsafe_load(j + 1)
                    + (s3 - s2) * h * self._t.unsafe_load(j + 1)
                ),
            )

        return result^


# ===----------------------------------------------------------------------=== #
# interp1d (constructor)
# ===----------------------------------------------------------------------=== #


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

    Returns:
        A callable LinearInterpolator object.

    Raises:
        Error: If x and y have different lengths, have fewer than 2 points,
            or x is not strictly increasing.

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

    Returns:
        NDArray of interpolated values at the points xi.

    Raises:
        Error: If inputs are invalid or method/fill_method is unsupported.

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
    var x_min: Scalar[dtype] = x.unsafe_load(0)
    var x_max: Scalar[dtype] = x.unsafe_load(x.size - 1)

    for i in range(xi.size):
        var xi_val: Scalar[dtype] = xi.unsafe_load(i)

        if xi_val <= x_min:
            result.itemset(i, y.unsafe_load(0))
        elif xi_val >= x_max:
            result.itemset(i, y.unsafe_load(y.size - 1))
        else:
            var j: Int = _binary_search(x, xi_val)

            var x0: Scalar[dtype] = x.unsafe_load(j - 1)
            var x1: Scalar[dtype] = x.unsafe_load(j)
            var y0: Scalar[dtype] = y.unsafe_load(j - 1)
            var y1: Scalar[dtype] = y.unsafe_load(j)
            var t: Scalar[dtype] = (xi_val - x0) / (x1 - x0)
            result.unsafe_store(i, y0 + t * (y1 - y0))

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
    var x_min: Scalar[dtype] = x.unsafe_load(0)
    var x_max: Scalar[dtype] = x.unsafe_load(x.size - 1)

    for i in range(xi.size):
        var xi_val: Scalar[dtype] = xi.unsafe_load(i)

        if xi_val < x_min:
            var slope = (y.unsafe_load(1) - y.unsafe_load(0)) / (
                x.unsafe_load(1) - x.unsafe_load(0)
            )
            result.itemset(
                i, y.unsafe_load(0) + slope * (xi_val - x.unsafe_load(0))
            )
        elif xi_val > x_max:
            var slope = (
                y.unsafe_load(y.size - 1) - y.unsafe_load(y.size - 2)
            ) / (x.unsafe_load(x.size - 1) - x.unsafe_load(x.size - 2))
            result.itemset(
                i,
                y.unsafe_load(y.size - 1)
                + slope * (xi_val - x.unsafe_load(x.size - 1)),
            )
        else:
            if xi_val == x_min:
                result.itemset(i, y.unsafe_load(0))
            elif xi_val == x_max:
                result.itemset(i, y.unsafe_load(y.size - 1))
            else:
                var j: Int = _binary_search(x, xi_val)

                var x0: Scalar[dtype] = x.unsafe_load(j - 1)
                var x1: Scalar[dtype] = x.unsafe_load(j)
                var y0: Scalar[dtype] = y.unsafe_load(j - 1)
                var y1: Scalar[dtype] = y.unsafe_load(j)
                var t: Scalar[dtype] = (xi_val - x0) / (x1 - x0)
                result.unsafe_store(i, y0 + t * (y1 - y0))

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
    var x_min: Scalar[dtype] = x.unsafe_load(0)
    var x_max: Scalar[dtype] = x.unsafe_load(n - 1)

    if n < 3:
        return _interp1d_linear_interpolate(xi, x, y)

    var h = NDArray[dtype](Shape(n - 1))
    for i in range(n - 1):
        h.unsafe_store(i, x.unsafe_load(i + 1) - x.unsafe_load(i))

    var alpha = NDArray[dtype](Shape(n))
    alpha.unsafe_store(0, 0.0)
    alpha.unsafe_store(n - 1, 0.0)
    for i in range(1, n - 1):
        alpha.unsafe_store(
            i,
            (
                3.0
                * (y.unsafe_load(i + 1) - y.unsafe_load(i))
                / h.unsafe_load(i)
                - 3.0
                * (y.unsafe_load(i) - y.unsafe_load(i - 1))
                / h.unsafe_load(i - 1)
            ),
        )

    var l = NDArray[dtype](Shape(n))
    var mu = NDArray[dtype](Shape(n))
    var z = NDArray[dtype](Shape(n))
    l.unsafe_store(0, 1.0)
    mu.unsafe_store(0, 0.0)
    z.unsafe_store(0, 0.0)

    for i in range(1, n - 1):
        l.unsafe_store(
            i,
            (
                2.0 * (x.unsafe_load(i + 1) - x.unsafe_load(i - 1))
                - h.unsafe_load(i - 1) * mu.unsafe_load(i - 1)
            ),
        )
        mu.unsafe_store(i, h.unsafe_load(i) / l.unsafe_load(i))
        z.unsafe_store(
            i,
            (alpha.unsafe_load(i) - h.unsafe_load(i - 1) * z.unsafe_load(i - 1))
            / l.unsafe_load(i),
        )

    l.unsafe_store(n - 1, 1.0)
    z.unsafe_store(n - 1, 0.0)

    var c = NDArray[dtype](Shape(n))
    var b = NDArray[dtype](Shape(n - 1))
    var d = NDArray[dtype](Shape(n - 1))
    c.unsafe_store(n - 1, 0.0)

    for j in range(n - 2, -1, -1):
        c.unsafe_store(
            j, z.unsafe_load(j) - mu.unsafe_load(j) * c.unsafe_load(j + 1)
        )
        b.unsafe_store(
            j,
            (y.unsafe_load(j + 1) - y.unsafe_load(j)) / h.unsafe_load(j)
            - h.unsafe_load(j)
            * (c.unsafe_load(j + 1) + 2.0 * c.unsafe_load(j))
            / 3.0,
        )
        d.unsafe_store(
            j,
            (c.unsafe_load(j + 1) - c.unsafe_load(j))
            / (3.0 * h.unsafe_load(j)),
        )

    for i in range(xi.size):
        var xi_val: Scalar[dtype] = xi.unsafe_load(i)
        if xi_val <= x_min:
            result.unsafe_store(i, y.unsafe_load(0))
            continue
        if xi_val >= x_max:
            result.unsafe_store(i, y.unsafe_load(n - 1))
            continue

        var j: Int = _binary_search(x, xi_val) - 1
        if j < 0:
            j = 0
        if j > n - 2:
            j = n - 2

        var dx = xi_val - x.unsafe_load(j)
        result.unsafe_store(
            i,
            (
                y.unsafe_load(j)
                + b.unsafe_load(j) * dx
                + c.unsafe_load(j) * dx * dx
                + d.unsafe_load(j) * dx * dx * dx
            ),
        )

    return result^


def _interp1d_akima_interpolate[
    dtype: DType
](xi: NDArray[dtype], x: NDArray[dtype], y: NDArray[dtype]) raises -> NDArray[
    dtype
]:
    """Akima 1D interpolation with boundary clamping.

    Delegates to `Akima1DInterpolator` so both entry points share one
    implementation of the boundary-extended weighted-slope formula.
    """
    var ak = Akima1DInterpolator[dtype](x, y)
    return ak(xi)


# ===----------------------------------------------------------------------=== #
# RegularGridInterpolator
# ===----------------------------------------------------------------------=== #


def _rgi_grid_interval[
    dtype: DType
](x: NDArray[dtype], value: Scalar[dtype]) raises -> Tuple[Int, Scalar[dtype]]:
    """Locates the grid cell and fractional offset for one axis.

    Example: idx, t = _rgi_grid_interval(x, value)

    Out-of-bounds values resolve to the first or last interval, allowing the
    caller to extrapolate. Both ascending and descending axes are supported.
    A singleton axis resolves to `(0, 0)`.

    Parameters:
        dtype: The floating-point data type.

    Args:
        x: Strictly monotonic 1-D coordinate array for one axis.
        value: The coordinate value to locate.

    Returns:
        A tuple `(idx, t)` where `idx` is the lower grid index and `t` is
        the fractional position between `x[idx]` and `x[idx + 1]`.
    """
    var n = x.size
    if n == 1:
        return (0, 0.0)

    var ascending = x.unsafe_load(n - 1) > x.unsafe_load(0)
    var lo = 0
    var hi = n - 1
    while hi - lo > 1:
        var mid = (lo + hi) // 2
        if ascending:
            if value < x.unsafe_load(mid):
                hi = mid
            else:
                lo = mid
        else:
            if value > x.unsafe_load(mid):
                hi = mid
            else:
                lo = mid

    var x0 = x.unsafe_load(lo)
    var x1 = x.unsafe_load(lo + 1)
    var t: Scalar[dtype] = 0.0
    if x1 != x0:
        t = (value - x0) / (x1 - x0)
    return (lo, t)


def _rgi_in_bounds[
    dtype: DType
](
    points: List[NDArray[dtype]], xi: List[Scalar[dtype]], ndim: Int
) raises -> Bool:
    """Checks whether a query point lies within the grid's domain.

    Example: ok = _rgi_in_bounds(points, xi, ndim)
    """
    for d in range(ndim):
        var first = points[d].unsafe_load(0)
        var last = points[d].unsafe_load(points[d].size - 1)
        var x_min = first
        var x_max = last
        if first > last:
            x_min = last
            x_max = first
        if xi[d] < x_min or xi[d] > x_max:
            return False
    return True


def _rgi_linear_eval[
    dtype: DType
](
    points: List[NDArray[dtype]],
    values: NDArray[dtype],
    xi: List[Scalar[dtype]],
    ndim: Int,
) raises -> Scalar[dtype]:
    """Multilinear interpolation at one query point.

    Example: yq = _rgi_linear_eval(points, values, xi, ndim)

    Locates the hyper-rectangular grid cell containing `xi` along each axis,
    then blends the `2^ndim` surrounding grid values by the product of each
    axis's fractional weight. Out-of-bounds coordinates extrapolate through
    the nearest boundary interval. Singleton axes contribute their sole value.

    Parameters:
        dtype: The floating-point data type.

    Args:
        points: Sorted 1-D coordinate arrays, one per axis.
        values: The grid of function values, one dimension per axis in
            `points`, in the same order.
        xi: The query point coordinates, length `ndim`.
        ndim: The number of grid axes.

    Returns:
        The interpolated value at `xi`.
    """
    var idx = List[Int]()
    var t = List[Scalar[dtype]]()
    for d in range(ndim):
        var pair = _rgi_grid_interval(points[d], xi[d])
        idx.append(pair[0])
        t.append(pair[1])

    var result: Scalar[dtype] = 0.0
    var n_corners = 1 << ndim
    for mask in range(n_corners):
        var weight: Scalar[dtype] = 1.0
        var flat: Int = 0
        for d in range(ndim):
            var bit = (mask >> d) & 1
            if points[d].size == 1:
                if bit == 1:
                    weight = 0.0
                continue
            if bit == 1:
                weight *= t[d]
                flat += (idx[d] + 1) * values.strides[d]
            else:
                weight *= 1.0 - t[d]
                flat += idx[d] * values.strides[d]
        result += weight * values.unsafe_load(flat)

    return result


def _rgi_nearest_eval[
    dtype: DType
](
    points: List[NDArray[dtype]],
    values: NDArray[dtype],
    xi: List[Scalar[dtype]],
    ndim: Int,
) raises -> Scalar[dtype]:
    """Nearest-grid-point lookup at one query point.

    Example: yq = _rgi_nearest_eval(points, values, xi, ndim)
    """
    var flat: Int = 0
    for d in range(ndim):
        var pair = _rgi_grid_interval(points[d], xi[d])
        var nearest = pair[0]
        if pair[1] > 0.5:
            nearest += 1
        flat += nearest * values.strides[d]
    return values.unsafe_load(flat)


struct RegularGridInterpolator[dtype: DType = DType.float64](Copyable, Movable):
    """A callable interpolator for data on a regular (rectilinear) grid.

    Stores per-axis coordinate arrays and the grid of values sampled at
    their outer product, and provides a callable interface that linearly
    (or nearest-neighbor) interpolates at arbitrary points in N dimensions.

    Parameters:
        dtype: The floating-point data type. Defaults to DType.float64.

    Examples:
        ```mojo
        import numojo as nm
        from scijo.interpolate import RegularGridInterpolator
        from scijo.prelude import *

        var x = nm.arange[f64](0.0, 3.0, 1.0)
        var y = nm.arange[f64](0.0, 3.0, 1.0)
        var values = nm.zeros[f64](Shape(3, 3))
        # values[i, j] = x[i] + y[j]
        var points: List[nm.NDArray[f64]] = [x, y]
        var rgi = RegularGridInterpolator[f64](points^, values)
        var yq = rgi([Scalar[f64](0.5), Scalar[f64](1.5)])
        ```
    """

    var points: List[NDArray[Self.dtype]]
    """Sorted 1-D coordinate arrays, one per axis."""
    var values: NDArray[Self.dtype]
    """Grid of function values, one dimension per axis in `points`."""
    var method: String
    """Interpolation method: `"linear"` or `"nearest"`."""
    var bounds_error: Bool
    """If True, raise error when interpolating outside the grid domain."""
    var fill_value: Optional[Scalar[Self.dtype]]
    """Value for out-of-bounds points when bounds_error is False. If None,
    out-of-bounds points are extrapolated."""
    var ndim: Int
    """The number of grid axes."""

    def __init__(
        out self,
        points: List[NDArray[Self.dtype]],
        values: NDArray[Self.dtype],
        method: String = "linear",
        bounds_error: Bool = True,
        fill_value: Optional[Scalar[Self.dtype]] = nan[Self.dtype](),
    ) raises:
        """Initializes the regular grid interpolator.

        Args:
            points: Strictly ascending or descending 1-D coordinate arrays,
                one per axis. Singleton axes are accepted.
            values: The grid of function values. Must have `len(points)`
                dimensions, with `values.shape[d] == points[d].size`.
            method: Interpolation method, `"linear"` or `"nearest"`.
            bounds_error: If True, raise error when interpolating outside
                the grid domain. If False, use fill_value, or extrapolate if
                fill_value is None.
            fill_value: Value to use for points outside the grid domain
                when bounds_error is False. Defaults to NaN. If None,
                extrapolate through the nearest boundary interval.

        Raises:
            Error: If method is unsupported, if points/values are
                inconsistent in shape, or if an axis is not strictly
                monotonic.

        Examples:
            `RegularGridInterpolator(points, values, method="nearest")`
        """
        comptime assert (
            Self.dtype.is_floating_point()
        ), "RegularGridInterpolator requires a floating-point dtype"
        if method != "linear" and method != "nearest":
            raise Error(
                "method must be 'linear' or 'nearest', got '" + method + "'"
            )

        var ndim = len(points)
        if ndim == 0:
            raise Error("points must contain at least one axis")
        if values.ndim != ndim:
            raise Error("values must have one dimension per entry in points")

        for d in range(ndim):
            if points[d].size == 0:
                raise Error("each axis in points must contain at least 1 point")
            if points[d].size != values.shape[d]:
                raise Error(
                    "points["
                    + String(d)
                    + "] length must match values.shape["
                    + String(d)
                    + "]"
                )
            if points[d].size > 1:
                var ascending = points[d].item(1) > points[d].item(0)
                if points[d].item(1) == points[d].item(0):
                    raise Error(
                        "each axis in points must be strictly monotonic"
                    )
                for i in range(2, points[d].size):
                    var previous = points[d].item(i - 1)
                    var current = points[d].item(i)
                    if (ascending and current <= previous) or (
                        not ascending and current >= previous
                    ):
                        raise Error(
                            "each axis in points must be strictly monotonic"
                        )

        self.points = points.copy()
        self.values = values.copy()
        self.method = method
        self.bounds_error = bounds_error
        self.fill_value = fill_value
        self.ndim = ndim

    def __call__(
        self, xi: List[Scalar[Self.dtype]]
    ) raises -> Scalar[Self.dtype]:
        """Interpolates at a single N-dimensional point.

        Example: yq = rgi([Scalar[dtype](0.5), Scalar[dtype](1.5)])

        Args:
            xi: The query point coordinates, one per axis.

        Returns:
            The interpolated value at xi.

        Raises:
            Error: If xi's length doesn't match ndim, or if bounds_error is
                True and xi is outside the grid domain.
        """
        if len(xi) != self.ndim:
            raise Error("xi must have length ndim")

        if not _rgi_in_bounds(self.points, xi, self.ndim):
            if self.bounds_error:
                raise Error("Interpolation point is outside the grid domain")
            if self.fill_value:
                return self.fill_value.value()

        if self.method == "nearest":
            return _rgi_nearest_eval(self.points, self.values, xi, self.ndim)
        return _rgi_linear_eval(self.points, self.values, xi, self.ndim)

    def __call__(self, xi: NDArray[Self.dtype]) raises -> NDArray[Self.dtype]:
        """Interpolates at one or more N-dimensional points.

        Example: yq = rgi(xi_array)  # xi_array has shape (..., ndim)

        Args:
            xi: Query points with shape `(..., ndim)`.

        Returns:
            Interpolated values with shape `xi.shape[:-1]`. A single query
            with shape `(ndim,)` returns shape `(1,)`.

        Raises:
            Error: If xi's last dimension doesn't match `ndim`, or if
                bounds_error is True and any row of xi is outside the grid
                domain.
        """
        if xi.ndim == 0 or xi.shape[xi.ndim - 1] != self.ndim:
            raise Error("xi must have shape (..., ndim)")

        var output_shape = List[Int]()
        for d in range(xi.ndim - 1):
            output_shape.append(xi.shape[d])
        if len(output_shape) == 0:
            output_shape.append(1)

        var query_count = xi.size // self.ndim
        var result = NDArray[Self.dtype](Shape(output_shape))
        for i in range(query_count):
            var point = List[Scalar[Self.dtype]]()
            for d in range(self.ndim):
                point.append(xi.item(i * self.ndim + d))
            result.unsafe_store(i, self(point))

        return result^


def interpn[
    dtype: DType
](
    points: List[NDArray[dtype]],
    values: NDArray[dtype],
    xi: NDArray[dtype],
    method: String = "linear",
    bounds_error: Bool = True,
    fill_value: Optional[Scalar[dtype]] = nan[dtype](),
) raises -> NDArray[dtype]:
    """Interpolates on a regular (rectilinear) grid in N dimensions.

    Example: yq = interpn(points, values, xi, method="linear")

    A one-shot functional interface, equivalent to constructing a
    `RegularGridInterpolator` and calling it on `xi`.

    Parameters:
        dtype: The floating-point data type.

    Args:
        points: Sorted 1-D coordinate arrays, one per axis.
        values: The grid of function values, one dimension per axis.
        xi: Query points with shape `(..., ndim)`.
        method: Interpolation method, `"linear"` or `"nearest"`.
        bounds_error: If True, raise error when interpolating outside the
            grid domain.
        fill_value: Value to use for points outside the grid domain when
            bounds_error is False. Defaults to NaN. If None, extrapolate
            through the nearest boundary interval.

    Returns:
        Interpolated values with shape `xi.shape[:-1]`. A single query with
        shape `(ndim,)` returns shape `(1,)`.

    Raises:
        Error: If method is unsupported, if points/values/xi are
            inconsistent in shape, or if bounds_error is True and any
            query point is outside the grid domain.
    """
    var rgi = RegularGridInterpolator[dtype](
        points, values, method, bounds_error, fill_value
    )
    return rgi(xi)
