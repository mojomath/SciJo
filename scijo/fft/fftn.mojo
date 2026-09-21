# ===----------------------------------------------------------------------=== #
# SciJo: FFT module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""N-Dimensional Fast Fourier Transform (`scijo.fft.fftn`)
==========================================================
Forward and inverse FFT over one or more axes of an N-dimensional complex
array, built on top of the 1-D Cooley-Tukey `fft`/`ifft` in
`scijo.fft.fastfourier`. Each requested axis is transformed independently by
applying the 1-D transform to every line along that axis, matching
`scipy.fft.fftn`/`ifftn`/`fft2`/`ifft2`.

Constraints
-----------
- The length of every transformed axis must be a power of 2.

Examples
--------
    ```mojo
    from scijo.fft import fftn, ifftn, fft2, ifft2

    var freq = fftn(arr)              # transform over all axes
    var freq_rows = fftn(arr, axes=[0])  # transform over axis 0 only
    var time = ifftn(freq)
    ```
"""

# ===----------------------------------------------------------------------=== #
# External
# ===----------------------------------------------------------------------=== #
from numojo.core.complex import ComplexNDArray
from numojo.core.dtype import ComplexDType
from numojo.core.layout import (
    NDArrayShape,
    NDArrayStrides,
)
from numojo.core.type_aliases import CScalar

# ===----------------------------------------------------------------------=== #
# SciJo
# ===----------------------------------------------------------------------=== #
from scijo.fft.fastfourier import (
    _ifft_unnormalized,
    fft,
)

# ===----------------------------------------------------------------------=== #
# Internal helpers
# ===----------------------------------------------------------------------=== #


def _normalize_axis(axis: Int, ndim: Int) raises -> Int:
    """Resolves a (possibly negative) axis to a valid index in [0, ndim)."""
    var a = axis
    if a < 0:
        a += ndim
    if a < 0 or a >= ndim:
        raise Error(
            "Scijo [fftn]: axis "
            + String(axis)
            + " is out of bounds for an array of dimension "
            + String(ndim)
        )
    return a


def _resolve_axes(ndim: Int, axes: Optional[List[Int]]) raises -> List[Int]:
    """Normalizes the `axes` argument to an explicit list of valid axes."""
    var axes_list = List[Int]()
    if axes:
        for i in range(len(axes.value())):
            axes_list.append(_normalize_axis(axes.value()[i], ndim))
    else:
        for d in range(ndim):
            axes_list.append(d)
    return axes_list^


def _increment_multi_index(
    mut idx: List[Int], shape: NDArrayShape, skip_axis: Int, ndim: Int
) raises -> None:
    """Advances `idx` to the next combination, skipping `skip_axis`."""
    var d = ndim - 1
    while d >= 0:
        if d == skip_axis:
            d -= 1
            continue
        idx[d] += 1
        if idx[d] < shape[d]:
            return
        idx[d] = 0
        d -= 1


def _flat_offset(idx: List[Int], strides: NDArrayStrides, ndim: Int) -> Int:
    """Computes the flat buffer offset of a multi-index under `strides`."""
    var offset = 0
    for d in range(ndim):
        offset += idx[d] * Int(strides.unsafe_load(d))
    return offset


def _fft_along_axis[
    cdtype: ComplexDType
](
    arr: ComplexNDArray[cdtype], axis: Int, inverse: Bool
) raises -> ComplexNDArray[cdtype] where cdtype.dtype.is_floating_point():
    """Applies the 1-D `fft`/unnormalized-`ifft` to every line along `axis`.

    Uses unsafe, flat-offset loads/stores (no per-element bounds checks or
    `Item` construction) since this is the inner loop of every N-D transform.
    """
    var ndim = arr.ndim
    var n = arr.shape[axis]
    var result = ComplexNDArray[cdtype](arr.shape)
    var num_lines = arr.size // n

    var arr_strides = arr.strides
    var result_strides = result.strides

    var idx = List[Int](length=ndim, fill=0)
    var line = ComplexNDArray[cdtype](NDArrayShape(n))

    for _ in range(num_lines):
        for k in range(n):
            idx[axis] = k
            var src_offset = _flat_offset(idx, arr_strides, ndim)
            line.unsafe_store(k, arr.unsafe_load(src_offset))

        var transformed: ComplexNDArray[cdtype]
        if inverse:
            transformed = _ifft_unnormalized[cdtype](line)
        else:
            transformed = fft[cdtype](line)

        for k in range(n):
            idx[axis] = k
            var dst_offset = _flat_offset(idx, result_strides, ndim)
            result.unsafe_store(dst_offset, transformed.unsafe_load(k))

        idx[axis] = 0
        _increment_multi_index(idx, arr.shape, axis, ndim)

    return result^


# ===----------------------------------------------------------------------=== #
# FFTN / IFFTN
# ===----------------------------------------------------------------------=== #


def fftn[
    cdtype: ComplexDType = ComplexDType.float64
](
    arr: ComplexNDArray[cdtype], axes: Optional[List[Int]] = None
) raises -> ComplexNDArray[cdtype] where cdtype.dtype.is_floating_point():
    """Computes the N-dimensional Fast Fourier Transform.

    Applies the 1-D Cooley-Tukey `fft` independently along each of the
    requested axes, matching `scipy.fft.fftn`.

    Parameters:
        cdtype: The data type of the complex elements (ComplexDType).

    Args:
        arr: Input complex array to transform. The length of every
            transformed axis must be a power of 2.
        axes: Axes over which to compute the transform. Defaults to all
            axes, transformed in order. Negative axes count from the end.

    Returns:
        ComplexNDArray containing the FFT of the input array with the same
        shape and dtype.

    Raises:
        Error: If an axis is out of bounds.
        Error: If a transformed axis's length is not a power of 2.

    Examples:
        ```mojo
        import numojo as nm
        from scijo.fft import fftn
        from scijo.prelude import *

        var arr = nm.linspace[cf64](CScalar[cf64](0, 0), CScalar[cf64](15, 0), num=16).reshape(Shape(4, 4))
        var freq = fftn(arr)             # transform over both axes
        var freq_rows = fftn(arr, axes=[1])  # transform along axis 1 only
        ```
    """
    var axes_list = _resolve_axes(arr.ndim, axes)
    if len(axes_list) == 0:
        return ComplexNDArray[cdtype](re=arr._re.copy(), im=arr._im.copy())

    var result = _fft_along_axis[cdtype](arr, axes_list[0], False)
    for i in range(1, len(axes_list)):
        result = _fft_along_axis[cdtype](result, axes_list[i], False)
    return result^


def ifftn[
    cdtype: ComplexDType = ComplexDType.float64
](
    arr: ComplexNDArray[cdtype], axes: Optional[List[Int]] = None
) raises -> ComplexNDArray[cdtype] where cdtype.dtype.is_floating_point():
    """Computes the N-dimensional Inverse Fast Fourier Transform.

    Applies the unnormalized inverse 1-D `fft` independently along each of
    the requested axes and then normalizes by the product of their lengths,
    matching `scipy.fft.ifftn`.

    Parameters:
        cdtype: The data type of the complex elements (ComplexDType).

    Args:
        arr: Input complex array to transform. The length of every
            transformed axis must be a power of 2.
        axes: Axes over which to compute the transform. Defaults to all
            axes. Negative axes count from the end.

    Returns:
        ComplexNDArray containing the IFFT of the input array with the same
        shape and dtype.

    Raises:
        Error: If an axis is out of bounds.
        Error: If a transformed axis's length is not a power of 2.

    Examples:
        ```mojo
        import numojo as nm
        from scijo.fft import fftn, ifftn
        from scijo.prelude import *

        var arr = nm.linspace[cf64](CScalar[cf64](0, 0), CScalar[cf64](15, 0), num=16).reshape(Shape(4, 4))
        var freq = fftn(arr)
        var time = ifftn(freq)
        ```
    """
    var axes_list = _resolve_axes(arr.ndim, axes)
    if len(axes_list) == 0:
        return ComplexNDArray[cdtype](re=arr._re.copy(), im=arr._im.copy())

    var result = _fft_along_axis[cdtype](arr, axes_list[0], True)
    var total_n = arr.shape[axes_list[0]]
    for i in range(1, len(axes_list)):
        result = _fft_along_axis[cdtype](result, axes_list[i], True)
        total_n *= arr.shape[axes_list[i]]

    var inv_n = CScalar[cdtype](1.0, 1.0) / CScalar[cdtype](
        Scalar[cdtype.dtype](total_n), Scalar[cdtype.dtype](total_n)
    )
    for i in range(result.size):
        result.unsafe_store(i, result.unsafe_load(i) * inv_n)

    return result^


# ===----------------------------------------------------------------------=== #
# FFT2 / IFFT2
# ===----------------------------------------------------------------------=== #


def fft2[
    cdtype: ComplexDType = ComplexDType.float64
](
    arr: ComplexNDArray[cdtype], axes: Tuple[Int, Int] = (-2, -1)
) raises -> ComplexNDArray[cdtype] where cdtype.dtype.is_floating_point():
    """Computes the 2-dimensional Fast Fourier Transform.

    Convenience wrapper over `fftn` that transforms exactly two axes,
    matching `scipy.fft.fft2`.

    Parameters:
        cdtype: The data type of the complex elements (ComplexDType).

    Args:
        arr: Input complex array to transform. Must have at least 2
            dimensions, and the length of each transformed axis must be a
            power of 2.
        axes: The two axes to transform. Defaults to the last two axes.
            Negative axes count from the end.

    Returns:
        ComplexNDArray containing the 2-D FFT of the input array.

    Raises:
        Error: If an axis is out of bounds.
        Error: If a transformed axis's length is not a power of 2.

    Examples:
        ```mojo
        import numojo as nm
        from scijo.fft import fft2
        from scijo.prelude import *

        var arr = nm.linspace[cf64](CScalar[cf64](0, 0), CScalar[cf64](15, 0), num=16).reshape(Shape(4, 4))
        var freq = fft2(arr)
        ```
    """
    var axes_list: List[Int] = [axes[0], axes[1]]
    return fftn[cdtype](arr, axes_list^)


def ifft2[
    cdtype: ComplexDType = ComplexDType.float64
](
    arr: ComplexNDArray[cdtype], axes: Tuple[Int, Int] = (-2, -1)
) raises -> ComplexNDArray[cdtype] where cdtype.dtype.is_floating_point():
    """Computes the 2-dimensional Inverse Fast Fourier Transform.

    Convenience wrapper over `ifftn` that transforms exactly two axes,
    matching `scipy.fft.ifft2`.

    Parameters:
        cdtype: The data type of the complex elements (ComplexDType).

    Args:
        arr: Input complex array to transform. Must have at least 2
            dimensions, and the length of each transformed axis must be a
            power of 2.
        axes: The two axes to transform. Defaults to the last two axes.
            Negative axes count from the end.

    Returns:
        ComplexNDArray containing the 2-D IFFT of the input array.

    Raises:
        Error: If an axis is out of bounds.
        Error: If a transformed axis's length is not a power of 2.

    Examples:
        ```mojo
        import numojo as nm
        from scijo.fft import fft2, ifft2
        from scijo.prelude import *

        var arr = nm.linspace[cf64](CScalar[cf64](0, 0), CScalar[cf64](15, 0), num=16).reshape(Shape(4, 4))
        var freq = fft2(arr)
        var time = ifft2(freq)
        ```
    """
    var axes_list: List[Int] = [axes[0], axes[1]]
    return ifftn[cdtype](arr, axes_list^)
