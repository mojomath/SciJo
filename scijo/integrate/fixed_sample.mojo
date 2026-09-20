# ===----------------------------------------------------------------------=== #
# SciJo: Integrate module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Fixed Sample Integration Methods (`scijo.integrate.fixed_sample`)
====================================================================
Integration methods for discrete, evenly or unevenly spaced sample data.
Includes the composite trapezoidal rule, Simpson's rule, and Romberg integration.

Examples
--------
    ```mojo
    import numojo as nm
    from scijo.integrate import trapezoid, simpson, romb
    from scijo.prelude import *

    var y = nm.linspace[f64](0.0, 10.0, 100) ** 2
    var area_trap = trapezoid(y, dx=0.1)
    var area_sim = simpson(y, dx=0.1)
    var area_rom = romb(y, dx=0.1)
    ```
"""

# ===----------------------------------------------------------------------=== #
# External
# ===----------------------------------------------------------------------=== #
import numojo as nm
from numojo.core.error import NumojoError
from numojo.core.ndarray import (
    NDArray,
    NDArrayShape,
)

# ===----------------------------------------------------------------------=== #
# Trapezoid
# ===----------------------------------------------------------------------=== #


def trapezoid[
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

    Examples:
        ```mojo
        import numojo as nm
        from scijo.integrate import trapezoid
        from scijo.prelude import *

        var y = nm.linspace[f64](0.0, 10.0, 100) ** 2  # y = x^2 sampled at 100 points from 0 to 10
        var area = trapezoid(y, dx=0.1)
        ```
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


def trapezoid[
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

    Examples:
        ```mojo
        import numojo as nm
        from scijo.integrate import trapezoid
        from scijo.prelude import *

        var x = nm.linspace[f64](0.0, 10.0, 100)
        var y = x * x
        var area = trapezoid(y, x)
        ```
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


# ===----------------------------------------------------------------------=== #
# Simpson
# ===----------------------------------------------------------------------=== #


def simpson[
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

    Examples:
        ```mojo
        import numojo as nm
        from scijo.integrate import simpson
        from scijo.prelude import *

        var y = nm.linspace[f64](0.0, 10.0, 100) ** 2  # y = x^2 sampled at 100 points from 0 to 10
        var area = simpson(y, dx=0.1)
        ```
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
    for i in range(0, y.size - 2, 2):
        integral += (
            multiplier
            * dx
            * 2
            * (y.item(i) + 4.0 * y.item(i + 1) + y.item(i + 2))
        )

    if y.size % 2 == 0:
        var y_n1 = y.item(y.size - 2)
        var y_n = y.item(y.size - 1)
        integral += (y_n1 + y_n) * dx * 0.5

    return integral


def simpson[
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

    Examples:
        ```mojo
        import numojo as nm
        from scijo.integrate import simpson
        from scijo.prelude import *

        var y = nm.linspace[f64](0.0, 10.0, 100) ** 2  # y = x^2 sampled at 100 points from 0 to 10
        var x = nm.linspace[f64](0.0, 10.0, 100)  # x values corresponding to y
        var area = simpson(y, x)
        ```
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


# ===----------------------------------------------------------------------=== #
# Romberg
# ===----------------------------------------------------------------------=== #


def romb[
    dtype: DType
](y: NDArray[dtype], dx: Scalar[dtype] = 1.0, axis: Int = -1) raises -> Scalar[
    dtype
]:
    """Integrates along the given axis using Romberg integration with Richardson extrapolation.

    Requires the input array to have ``2^k + 1`` points for some integer ``k ≥ 1``
    (e.g. 3, 5, 9, 17, 33, 65, …).

    The algorithm fills a Romberg tableau: each column applies one step of
    Richardson extrapolation to cancel the leading error term, achieving
    ``O(h^{2(k+1)})`` accuracy from k+1 trapezoidal estimates.

    Parameters:
        dtype: The floating-point data type.

    Args:
        y: Input array to integrate. Must be 1-D with ``2^k + 1`` elements.
        dx: The spacing between sample points. Defaults to 1.0.
        axis: The axis along which to integrate. Currently only 1-D is supported.

    Returns:
        Best Romberg estimate of the definite integral.

    Raises:
        Error: If y is not 1-D.
        Error: If y.size is not of the form ``2^k + 1`` for integer k ≥ 1.

    Examples:
        ```mojo
        import numojo as nm
        from scijo.integrate import romb
        from scijo.prelude import *

        var y = nm.linspace[f64](0.0, 1.0, 9) ** 2  # 9 = 2^3 + 1 points
        var area = romb(y, dx=0.125)                 # ≈ 0.333...
        ```
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
                location="romb(y, dx=1.0)",
            )
        )

    var n = y.size
    var n_minus_1 = n - 1
    if n < 3 or (n_minus_1 & (n_minus_1 - 1)) != 0:
        raise Error(
            NumojoError(
                category="value",
                message=String(
                    "y.size must be 2^k + 1 for integer k ≥ 1 (e.g. 3, 5, 9,"
                    " 17, 33, …), got {}."
                ).format(n),
                location="romb(y, dx=1.0)",
            )
        )

    var k: Int = 0
    var tmp = n_minus_1
    while tmp > 1:
        tmp >>= 1
        k += 1

    var R_prev = nm.zeros[dtype](nm.Shape(k + 1))
    var R_curr = nm.zeros[dtype](nm.Shape(k + 1))

    R_prev.itemset(
        0,
        Scalar[dtype](0.5)
        * dx
        * (y.item(0) + y.item(n - 1))
        * Scalar[dtype](n_minus_1),
    )

    for i in range(1, k + 1):
        var stride: Int = n_minus_1 >> i  # = 2^(k-i)
        var num_new: Int = 1 << (i - 1)  # = 2^(i-1) new interior points

        var h_i: Scalar[dtype] = dx * Scalar[dtype](stride)
        var s: Scalar[dtype] = 0.0
        for j in range(1, 2 * num_new, 2):
            s += y.item(j * stride)
        R_curr.itemset(0, Scalar[dtype](0.5) * R_prev.item(0) + h_i * s)

        for m in range(1, i + 1):
            var factor: Scalar[dtype] = Scalar[dtype](4.0) ** m
            var val = (factor * R_curr.item(m - 1) - R_prev.item(m - 1)) / (
                factor - Scalar[dtype](1.0)
            )
            R_curr.itemset(m, val)

        for m in range(i + 1):
            R_prev.itemset(m, R_curr.item(m))

    return R_prev.item(k)


# ===----------------------------------------------------------------------=== #
# Cumulative trapezoid
# ===----------------------------------------------------------------------=== #


def cumulative_trapezoid[
    dtype: DType
](
    y: NDArray[dtype],
    dx: Scalar[dtype] = 1.0,
    axis: Int = -1,
    initial: Optional[Scalar[dtype]] = None,
) raises -> NDArray[dtype] where dtype.is_floating_point():
    """Cumulatively integrates y using the composite trapezoidal rule.

    Returns an array of the running integral, matching
    `scipy.integrate.cumulative_trapezoid`. The output has length ``n-1``
    when `initial` is None, or length ``n`` when an `initial` value is given
    (prepended as the first element).

    Parameters:
        dtype: The floating-point data type.

    Args:
        y: Input 1-D array of sample values.
        dx: Spacing between sample points. Defaults to 1.0.
        axis: Axis along which to integrate. Currently only 1-D is supported.
        initial: If provided, prepend this value to the output so the result
                 has the same length as y. Typically 0.0.

    Returns:
        NDArray of cumulative integral values. Shape is ``(n-1,)`` if
        `initial` is None, or ``(n,)`` if `initial` is provided.

    Raises:
        Error: If y is not 1-D or has fewer than 2 elements.

    Examples:
        ```mojo
        import numojo as nm
        from scijo.integrate import cumulative_trapezoid
        from scijo.prelude import *

        var y = nm.array[f64]([1.0, 2.0, 3.0, 4.0])
        var cum = cumulative_trapezoid(y, dx=1.0, initial=0.0)
        # [0.0, 1.5, 4.0, 7.5]
        ```
    """
    if y.ndim != 1:
        raise Error(
            "Scijo [cumulative_trapezoid]: Expected 1-D array, got ndim="
            + String(y.ndim)
        )
    if y.size < 2:
        raise Error(
            "Scijo [cumulative_trapezoid]: Need at least 2 elements, got "
            + String(y.size)
        )

    var n = y.size
    var out_len = n - 1 if not initial else n
    var result = NDArray[dtype](NDArrayShape(out_len))
    var offset = 1 if initial else 0

    if initial:
        result.unsafe_store(0, initial.value())

    var running: Scalar[dtype] = 0.0
    for i in range(n - 1):
        running += (y.unsafe_load(i) + y.unsafe_load(i + 1)) * dx * 0.5
        result.unsafe_store(i + offset, running)

    return result^


def cumulative_trapezoid[
    dtype: DType
](
    y: NDArray[dtype],
    x: NDArray[dtype],
    axis: Int = -1,
    initial: Optional[Scalar[dtype]] = None,
) raises -> NDArray[dtype] where dtype.is_floating_point():
    """Cumulatively integrates y(x) using the composite trapezoidal rule.

    Non-uniform spacing version — spacing is taken from successive differences
    in `x`. Matches `scipy.integrate.cumulative_trapezoid`.

    Parameters:
        dtype: The floating-point data type.

    Args:
        y: Input 1-D array of sample values.
        x: 1-D array of sample points corresponding to y. Must have the same
           length as y and be strictly increasing.
        axis: Axis along which to integrate. Currently only 1-D is supported.
        initial: If provided, prepend this value so the result has the same
                 length as y.

    Returns:
        NDArray of cumulative integral values. Shape ``(n-1,)`` or ``(n,)``.

    Raises:
        Error: If y or x are not 1-D, sizes differ, or fewer than 2 elements.

    Examples:
        ```mojo
        import numojo as nm
        from scijo.integrate import cumulative_trapezoid
        from scijo.prelude import *

        var x = nm.array[f64]([0.0, 1.0, 2.0, 3.0])
        var y = x * x
        var cum = cumulative_trapezoid(y, x, initial=0.0)
        # [0.0, 0.5, 3.0, 8.5]
        ```
    """
    if y.ndim != 1:
        raise Error(
            "Scijo [cumulative_trapezoid]: Expected 1-D y, got ndim="
            + String(y.ndim)
        )
    if x.ndim != 1:
        raise Error(
            "Scijo [cumulative_trapezoid]: Expected 1-D x, got ndim="
            + String(x.ndim)
        )
    if y.size != x.size:
        raise Error(
            "Scijo [cumulative_trapezoid]: y.size ("
            + String(y.size)
            + ") != x.size ("
            + String(x.size)
            + ")"
        )
    if y.size < 2:
        raise Error(
            "Scijo [cumulative_trapezoid]: Need at least 2 elements, got "
            + String(y.size)
        )

    var n = y.size
    var out_len = n - 1 if not initial else n
    var result = NDArray[dtype](NDArrayShape(out_len))
    var offset = 1 if initial else 0

    if initial:
        result.unsafe_store(0, initial.value())

    var running: Scalar[dtype] = 0.0
    for i in range(n - 1):
        var dx_seg = x.unsafe_load(i + 1) - x.unsafe_load(i)
        running += (y.unsafe_load(i) + y.unsafe_load(i + 1)) * dx_seg * 0.5
        result.unsafe_store(i + offset, running)

    return result^


# ===----------------------------------------------------------------------=== #
# Cumulative Simpson
# ===----------------------------------------------------------------------=== #


def cumulative_simpson[
    dtype: DType
](
    y: NDArray[dtype],
    dx: Scalar[dtype] = 1.0,
    axis: Int = -1,
    initial: Optional[Scalar[dtype]] = None,
) raises -> NDArray[dtype] where dtype.is_floating_point():
    """Cumulatively integrates y using Simpson's rule on successive pairs of intervals.

    Each output element i (0-based, before prepending `initial`) is the
    integral over ``y[0..2i+2]`` using composite Simpson's rule applied to
    each consecutive triple ``(y[2i], y[2i+1], y[2i+2])``. Matches
    `scipy.integrate.cumulative_simpson` for even-length inputs.

    Requires an odd number of samples (even number of intervals). For even
    number of samples the last panel falls back to the trapezoidal rule,
    consistent with `simpson`.

    Parameters:
        dtype: The floating-point data type.

    Args:
        y: Input 1-D array of sample values. Should have odd length (≥ 3)
           for pure Simpson; even length falls back for the last interval.
        dx: Spacing between sample points. Defaults to 1.0.
        axis: Axis along which to integrate. Currently only 1-D is supported.
        initial: If provided, prepend this value so the result has the same
                 length as y.

    Returns:
        NDArray of cumulative integral values. Shape ``(n-1,)`` or ``(n,)``.

    Raises:
        Error: If y is not 1-D or has fewer than 3 elements.

    Examples:
        ```mojo
        import numojo as nm
        from scijo.integrate import cumulative_simpson
        from scijo.prelude import *

        var y = nm.array[f64]([1.0, 4.0, 1.0, 4.0, 1.0])
        var cum = cumulative_simpson(y, dx=1.0, initial=0.0)
        # matches scipy.integrate.cumulative_simpson
        ```
    """
    if y.ndim != 1:
        raise Error(
            "Scijo [cumulative_simpson]: Expected 1-D array, got ndim="
            + String(y.ndim)
        )
    if y.size < 3:
        raise Error(
            "Scijo [cumulative_simpson]: Need at least 3 elements, got "
            + String(y.size)
        )

    var n = y.size
    var out_len = n - 1 if not initial else n
    var result = NDArray[dtype](NDArrayShape(out_len))
    var offset = 1 if initial else 0

    if initial:
        result.unsafe_store(0, initial.value())

    var running: Scalar[dtype] = 0.0
    var i = 0
    while i < n - 2:
        var panel = (
            (
                y.unsafe_load(i)
                + 4.0 * y.unsafe_load(i + 1)
                + y.unsafe_load(i + 2)
            )
            * dx
            / 3.0
        )
        var half = (y.unsafe_load(i) + y.unsafe_load(i + 1)) * dx * 0.5
        result.unsafe_store(i + offset, running + half)
        running += panel
        result.unsafe_store(i + 1 + offset, running)
        i += 2

    if i == n - 2:
        var half = (y.unsafe_load(i) + y.unsafe_load(i + 1)) * dx * 0.5
        result.unsafe_store(i + offset, running + half)

    return result^
