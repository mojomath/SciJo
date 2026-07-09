# ===----------------------------------------------------------------------=== #
# SciJo: Interpolate module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Interpolation Utility Functions (`scijo.interpolate.utility`)
===============================================================
Internal utility functions for interpolation, including binary search and
input validation.
"""

# ===----------------------------------------------------------------------=== #
# Binary search
# ===----------------------------------------------------------------------=== #


def _binary_search[
    dtype: DType
](x: NDArray[dtype], value: Scalar[dtype]) raises -> Int:
    """Binary search to find the interval containing the interpolation point.

    Parameters:
        dtype: The floating-point data type.

    Args:
        x: Sorted array of x-coordinates.
        value: The value to search for.

    Returns:
        Index j such that x[j-1] <= value < x[j].
    """
    var left: Int = 0
    var right: Int = x.size - 1

    while right - left > 1:
        var mid = (left + right) // 2
        if x._buf.ptr[mid] <= value:
            left = mid
        else:
            right = mid

    return right


# ===----------------------------------------------------------------------=== #
# Input validation
# ===----------------------------------------------------------------------=== #


def _validate_interpolation_input[
    dtype: DType
](x: NDArray[dtype], y: NDArray[dtype]) raises:
    """Validates input arrays for interpolation.

    Parameters:
        dtype: The floating-point data type.

    Args:
        x: Array of x-coordinates.
        y: Array of y-coordinates.

    Raises:
        Error: If arrays differ in length, have fewer than 2 points, or x is
            not strictly increasing.
    """
    if x.size != y.size:
        raise Error("x and y arrays must have the same length")

    if x.size < 2:
        raise Error("x and y arrays must have at least 2 points")

    for i in range(1, x.size):
        if x.item(i) <= x.item(i - 1):
            raise Error("x array must be strictly increasing")
