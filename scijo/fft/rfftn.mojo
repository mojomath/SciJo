# ===----------------------------------------------------------------------=== #
# SciJo: FFT module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""N-Dimensional Real Fast Fourier Transform (`scijo.fft.rfftn`)
================================================================
Real-input forward/inverse FFT over one or more axes of an N-dimensional
real array: `rfftn`/`irfftn` (general axes) and `rfft2`/`irfft2`
(convenience wrappers for exactly two axes), built on top of the 1-D
`rfft`/`irfft` in `scijo.fft.fastfourier` and the axis-iteration machinery
in `scijo.fft.fftn`.

Matching `numpy.fft.rfftn`'s own algorithm: the real-FFT trick (exploiting
conjugate symmetry) is applied only to the *last* requested axis, while
the input is still real; every other requested axis is then transformed
with the ordinary complex `fft`/`ifft`, operating on the now-complex
intermediate result. `irfftn` reverses this in the opposite order:
complex `ifft` on every axis but the last, then `irfft` on the last axis.

Constraints
-----------
- The length of every transformed axis must be a power of 2 (matching the
  rest of this module's `fft`/`rfft` - non-power-of-2 lengths are padded
  up, matching NumPy's own zero-padding behavior).
"""

# ===----------------------------------------------------------------------=== #
# External
# ===----------------------------------------------------------------------=== #
from numojo.core.complex import ComplexNDArray
from numojo.core.dtype import ComplexDType
from numojo.core.layout import NDArrayShape
from numojo.core.ndarray import NDArray

# ===----------------------------------------------------------------------=== #
# SciJo
# ===----------------------------------------------------------------------=== #
from scijo.fft.fastfourier import (
    irfft,
    rfft,
)
from scijo.fft.fftn import (
    _fft_along_axis,
    _flat_offset,
    _increment_multi_index,
    _resolve_axes,
)

# ===----------------------------------------------------------------------=== #
# Internal helpers
# ===----------------------------------------------------------------------=== #


def _shape_with_axis[
    cdtype: ComplexDType
](
    shape: NDArrayShape, ndim: Int, axis: Int, new_len: Int
) raises -> NDArrayShape:
    """Returns a copy of `shape` with `shape[axis]` replaced by `new_len`."""
    var dims = List[Int](capacity=ndim)
    for d in range(ndim):
        dims.append(Int(shape[d]) if d != axis else new_len)
    return NDArrayShape(dims)


def _rfft_along_axis[
    dtype: DType, cdtype: ComplexDType
](arr: NDArray[dtype], axis: Int) raises -> ComplexNDArray[
    cdtype
] where cdtype.dtype.is_floating_point():
    """Applies the 1-D `rfft` to every line along `axis` of a real array."""
    var ndim = arr.ndim
    var n = arr.shape[axis]

    var n_padded = n
    if (n_padded & (n_padded - 1)) != 0:
        n_padded = 1
        while n_padded < n:
            n_padded <<= 1
    var out_len = n_padded // 2 + 1

    var result = ComplexNDArray[cdtype](
        _shape_with_axis[cdtype](arr.shape, ndim, axis, out_len)
    )
    var num_lines = arr.size // n

    var arr_strides = arr.strides
    var result_strides = result.strides

    var idx = List[Int](length=ndim, fill=0)
    var line = NDArray[dtype](NDArrayShape(n))

    for _ in range(num_lines):
        for k in range(n):
            idx[axis] = k
            var src_offset = _flat_offset(idx, arr_strides, ndim)
            line.unsafe_store(k, arr.unsafe_load(src_offset))

        var transformed = rfft[dtype, cdtype=cdtype](line)

        for k in range(out_len):
            idx[axis] = k
            var dst_offset = _flat_offset(idx, result_strides, ndim)
            result.unsafe_store(dst_offset, transformed.unsafe_load(k))

        idx[axis] = 0
        _increment_multi_index(idx, arr.shape, axis, ndim)

    return result^


def _irfft_along_axis[
    dtype: DType, cdtype: ComplexDType
](arr: ComplexNDArray[cdtype], axis: Int, n: Optional[Int]) raises -> NDArray[
    dtype
] where cdtype.dtype.is_floating_point():
    """Applies the 1-D `irfft` to every line along `axis` of a complex array."""
    var ndim = arr.ndim
    var m = arr.shape[axis]

    var full_n: Int
    if n:
        full_n = n.value()
    else:
        full_n = 2 * (m - 1)
    if (full_n & (full_n - 1)) != 0:
        var padded = 1
        while padded < full_n:
            padded <<= 1
        full_n = padded

    var result = NDArray[dtype](
        _shape_with_axis[cdtype](arr.shape, ndim, axis, full_n)
    )
    var num_lines = arr.size // m

    var arr_strides = arr.strides
    var result_strides = result.strides

    var idx = List[Int](length=ndim, fill=0)
    var line = ComplexNDArray[cdtype](NDArrayShape(m))

    for _ in range(num_lines):
        for k in range(m):
            idx[axis] = k
            var src_offset = _flat_offset(idx, arr_strides, ndim)
            line.unsafe_store(k, arr.unsafe_load(src_offset))

        var transformed = irfft[dtype, cdtype=cdtype](line, n=full_n)

        for k in range(full_n):
            idx[axis] = k
            var dst_offset = _flat_offset(idx, result_strides, ndim)
            result.unsafe_store(dst_offset, transformed.unsafe_load(k))

        idx[axis] = 0
        _increment_multi_index(idx, arr.shape, axis, ndim)

    return result^


# ===----------------------------------------------------------------------=== #
# RFFTN / IRFFTN
# ===----------------------------------------------------------------------=== #


def rfftn[
    dtype: DType = DType.float64,
    *,
    cdtype: ComplexDType = ComplexDType(mlir_value=dtype._mlir_value),
](
    arr: NDArray[dtype], axes: Optional[List[Int]] = None
) raises -> ComplexNDArray[cdtype] where cdtype.dtype.is_floating_point():
    """Computes the N-dimensional real-input Fast Fourier Transform.

    Applies the real-FFT trick (`rfft`) to the last requested axis, then
    the ordinary complex `fft` to every other requested axis.

    Parameters:
        dtype: The floating-point element type of the input array.
        cdtype: The complex data type corresponding to `dtype`.

    Args:
        arr: Real-valued input array. The length of every transformed
            axis is padded up to the next power of 2 if needed.
        axes: Axes over which to compute the transform. Defaults to all
            axes, transformed in order. Negative axes count from the end.

    Returns:
        ComplexNDArray with the same shape as `arr`, except the last
        transformed axis has length `n_padded // 2 + 1`.

    Raises:
        Error: If an axis is out of bounds, or `axes` is empty.

    Examples:
        ```mojo
        import numojo as nm
        from scijo.fft import rfftn, irfftn
        from scijo.prelude import *

        var arr = nm.linspace[f64](0.0, 15.0, num=16).reshape(Shape(4, 4))
        var freq = rfftn(arr)
        var time = irfftn[f64](freq, s=4)
        ```
    """
    var axes_list = _resolve_axes(arr.ndim, axes)
    if len(axes_list) == 0:
        raise Error("Scijo [rfftn]: axes must not be empty")

    var last_axis = axes_list[len(axes_list) - 1]
    var result = _rfft_along_axis[dtype, cdtype](arr, last_axis)
    for i in range(len(axes_list) - 1):
        result = _fft_along_axis[cdtype](result, axes_list[i], False)

    return result^


def irfftn[
    dtype: DType = DType.float64,
    *,
    cdtype: ComplexDType = ComplexDType(mlir_value=dtype._mlir_value),
](
    arr: ComplexNDArray[cdtype],
    s: Optional[Int] = None,
    axes: Optional[List[Int]] = None,
) raises -> NDArray[dtype] where cdtype.dtype.is_floating_point():
    """Computes the N-dimensional inverse real-input Fast Fourier Transform.

    Applies the unnormalized inverse complex `fft` to every requested axis
    but the last, then the real `irfft` to the last axis, normalizing by
    the product of all transformed axes' output lengths.

    Parameters:
        dtype: The floating-point element type of the output array.
        cdtype: The complex data type corresponding to `dtype`.

    Args:
        arr: Complex input array, as produced by `rfftn`.
        s: Output length of the *last* transformed axis. Defaults to
            `2 * (arr.shape[last_axis] - 1)`. Unlike NumPy's `s`, this
            only controls the last axis - a scoped simplification, since
            every other transformed axis's length is already fixed by
            `arr`'s own shape.
        axes: Axes over which to compute the transform. Defaults to all
            axes. Negative axes count from the end.

    Returns:
        NDArray with the same shape as `arr`, except the last transformed
        axis has the output length described above.

    Raises:
        Error: If an axis is out of bounds, or `axes` is empty.

    Examples:
        ```mojo
        import numojo as nm
        from scijo.fft import rfftn, irfftn
        from scijo.prelude import *

        var arr = nm.linspace[f64](0.0, 15.0, num=16).reshape(Shape(4, 4))
        var freq = rfftn(arr)
        var time = irfftn[f64](freq, s=4)
        ```
    """
    var axes_list = _resolve_axes(arr.ndim, axes)
    if len(axes_list) == 0:
        raise Error("Scijo [irfftn]: axes must not be empty")

    var last_axis = axes_list[len(axes_list) - 1]

    var current = ComplexNDArray[cdtype](re=arr._re.copy(), im=arr._im.copy())
    var total_n = 1
    for i in range(len(axes_list) - 1):
        current = _fft_along_axis[cdtype](current, axes_list[i], True)
        total_n *= arr.shape[axes_list[i]]

    var result = _irfft_along_axis[dtype, cdtype](current, last_axis, n=s)

    var inv_n = Scalar[dtype](1.0) / Scalar[dtype](total_n)
    for i in range(result.size):
        result.unsafe_store(i, result.unsafe_load(i) * inv_n)

    return result^


# ===----------------------------------------------------------------------=== #
# RFFT2 / IRFFT2
# ===----------------------------------------------------------------------=== #


def rfft2[
    dtype: DType = DType.float64,
    *,
    cdtype: ComplexDType = ComplexDType(mlir_value=dtype._mlir_value),
](
    arr: NDArray[dtype], axes: Tuple[Int, Int] = (-2, -1)
) raises -> ComplexNDArray[cdtype] where cdtype.dtype.is_floating_point():
    """Computes the 2-dimensional real-input Fast Fourier Transform.

    Convenience wrapper over `rfftn` that transforms exactly two axes.

    Parameters:
        dtype: The floating-point element type of the input array.
        cdtype: The complex data type corresponding to `dtype`.

    Args:
        arr: Real-valued input array. Must have at least 2 dimensions.
        axes: The two axes to transform. Defaults to the last two axes.
            Negative axes count from the end.

    Returns:
        ComplexNDArray with the same shape as `arr`, except the last of
        `axes` has length `n_padded // 2 + 1`.

    Raises:
        Error: If an axis is out of bounds.

    Examples:
        ```mojo
        import numojo as nm
        from scijo.fft import rfft2, irfft2
        from scijo.prelude import *

        var arr = nm.linspace[f64](0.0, 15.0, num=16).reshape(Shape(4, 4))
        var freq = rfft2(arr)
        ```
    """
    var axes_list: List[Int] = [axes[0], axes[1]]
    return rfftn[dtype, cdtype=cdtype](arr, axes_list^)


def irfft2[
    dtype: DType = DType.float64,
    *,
    cdtype: ComplexDType = ComplexDType(mlir_value=dtype._mlir_value),
](
    arr: ComplexNDArray[cdtype],
    s: Optional[Int] = None,
    axes: Tuple[Int, Int] = (-2, -1),
) raises -> NDArray[dtype] where cdtype.dtype.is_floating_point():
    """Computes the 2-dimensional inverse real-input Fast Fourier Transform.

    Convenience wrapper over `irfftn` that transforms exactly two axes.

    Parameters:
        dtype: The floating-point element type of the output array.
        cdtype: The complex data type corresponding to `dtype`.

    Args:
        arr: Complex input array, as produced by `rfft2`.
        s: Output length of the last of `axes`. See `irfftn`.
        axes: The two axes to transform. Defaults to the last two axes.
            Negative axes count from the end.

    Returns:
        NDArray with the same shape as `arr`, except the last of `axes`
        has the output length described in `irfftn`.

    Raises:
        Error: If an axis is out of bounds.

    Examples:
        ```mojo
        import numojo as nm
        from scijo.fft import rfft2, irfft2
        from scijo.prelude import *

        var arr = nm.linspace[f64](0.0, 15.0, num=16).reshape(Shape(4, 4))
        var freq = rfft2(arr)
        var time = irfft2[f64](freq, s=4)
        ```
    """
    var axes_list: List[Int] = [axes[0], axes[1]]
    return irfftn[dtype, cdtype=cdtype](arr, s, axes_list^)
