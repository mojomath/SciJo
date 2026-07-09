# ===----------------------------------------------------------------------=== #
# SciJo: FFT module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Fast Fourier Transform (`scijo.fft.fastfourier`)
===================================================
Forward and inverse Fast Fourier Transform using the Cooley-Tukey radix-2
decimation-in-time algorithm for 1-D complex arrays with power-of-2 lengths.

Constraints
-----------
- Input arrays must be 1-dimensional.
- Array length must be a power of 2.

Examples
--------
    ```mojo
    from scijo.fft import fft, ifft

    var arr = nm.linspace[cf32](CScalar[cf32](0, 0), CScalar[cf32](10, 10), num=10)
    var freq = fft(arr)
    var time = ifft(freq)
    ```
"""

from std.math import sin, cos

from numojo.core.complex import ComplexNDArray, ComplexSIMD
from numojo.core.dtype import ComplexDType
from numojo.core.ndarray import NDArray
from numojo.core.layout import NDArrayShape
from numojo.routines.constants import Constants
from numojo.core.indexing import Item

# ===----------------------------------------------------------------------=== #
# FFT
# ===----------------------------------------------------------------------=== #


def fft[
    cdtype: ComplexDType = ComplexDType.float64
](arr: ComplexNDArray[cdtype]) raises -> ComplexNDArray[
    cdtype
] where cdtype.dtype.is_floating_point():
    """Computes the Fast Fourier Transform using the Cooley-Tukey algorithm.

    Decomposes the DFT computation by recursively breaking down the transform
    of size N into two transforms of size N/2, achieving O(N log N) complexity.

    Parameters:
        cdtype: The data type of the complex elements (ComplexDType).

    Args:
        arr: Input complex array to transform. Must be 1-dimensional with length
             that is a power of 2.

    Raises:
        Error: If the input array is not 1-dimensional.
        Error: If the array length is not a power of 2.

    Returns:
        ComplexNDArray containing the FFT of the input array with the same
        shape and dtype.

    Examples:
        ```mojo
        import numojo as nm
        from scijo.fft import fft
        from scijo.prelude import *

        var arr = nm.linspace[cf32](CScalar[cf32](0, 0), CScalar[cf32](10, 10), num=10)
        var fft_arr = fft(arr)
        ```
    """
    if arr.ndim != 1:
        raise Error("Scijo [fft]: FFT currently only supports 1D arrays")

    var n: Int = arr.shape[0]
    if n <= 1:
        return ComplexNDArray[cdtype](re=arr._re.copy(), im=arr._im.copy())

    if (n & (n - 1)) != 0:
        raise Error(
            "Scijo [fft]: FFT currently only supports arrays with length that"
            " is a power of 2"
        )

    var half_size = n // 2
    var even_indices = ComplexNDArray[cdtype](NDArrayShape(half_size))
    var odd_indices = ComplexNDArray[cdtype](NDArrayShape(half_size))

    for i in range(half_size):
        even_indices[Item(i)] = arr[Item(2 * i)]
        odd_indices[Item(i)] = arr[Item(2 * i + 1)]

    var even_fft = fft[cdtype](even_indices)
    var odd_fft = fft[cdtype](odd_indices)

    var result = ComplexNDArray[cdtype](arr.shape)

    for k in range(half_size):
        var angle = (
            -2.0
            * Constants.pi
            * Scalar[cdtype.dtype](k)
            / Scalar[cdtype.dtype](n)
        )
        var twiddle = ComplexSIMD[cdtype](
            cos(angle).cast[cdtype.dtype](), sin(angle).cast[cdtype.dtype]()
        )

        var twiddle_odd = twiddle * odd_fft[Item(k)]

        result[Item(k)] = even_fft[Item(k)] + twiddle_odd
        result[Item(k + half_size)] = even_fft[Item(k)] - twiddle_odd

    return result^


# ===----------------------------------------------------------------------=== #
# RFFT / IRFFT
# ===----------------------------------------------------------------------=== #


def rfft[
    dtype: DType = DType.float64,
    *,
    cdtype: ComplexDType = ComplexDType(mlir_value=dtype._mlir_value),
](arr: NDArray[dtype]) raises -> ComplexNDArray[
    cdtype
] where cdtype.dtype.is_floating_point():
    """Computes the real Fast Fourier Transform.

    Equivalent to `fft` on a zero-imaginary complex signal, but returns only
    the non-redundant first ``N // 2 + 1`` frequency bins (the upper half is
    the complex conjugate mirror of the lower half for real input).

    Parameters:
        dtype: The floating-point element type of the input array.
        cdtype: The complex data type corresponding to the input dtype. Defaults to
            ComplexDType with the same underlying float type.

    Args:
        arr: Real-valued 1-D input array. If length is not a power of 2, the
            array is zero-padded to the next power of 2 (matching NumPy behaviour).

    Raises:
        Error: If the input array is not 1-dimensional.

    Returns:
        ComplexNDArray of length ``N_padded // 2 + 1`` containing the non-redundant
        frequency components, where ``N_padded`` is the next power of 2 ≥ len(arr).

    Examples:
        ```mojo
        import numojo as nm
        from scijo.fft import rfft, irfft
        from scijo.prelude import *

        var x = nm.linspace[f64](0.0, 1.0, 6)   # length 6, padded to 8
        var freqs = rfft(x)                       # shape: (5,)
        var x_rec = irfft(freqs, 8)               # shape: (8,)
        ```
    """
    if arr.ndim != 1:
        raise Error("Scijo [rfft]: rfft currently only supports 1D arrays")

    var n = arr.shape[0]
    # Zero-pad to next power of 2 if needed
    var n_padded = n
    if (n & (n - 1)) != 0:
        n_padded = 1
        while n_padded < n:
            n_padded <<= 1

    var complex_input = ComplexNDArray[cdtype](NDArrayShape(n_padded))
    for i in range(n):
        complex_input[Item(i)] = ComplexSIMD[cdtype](
            arr._buf.ptr[i].cast[cdtype.dtype](),
            Scalar[cdtype.dtype](0),
        )
    # Remaining elements are already zero-initialized

    var full = fft[cdtype](complex_input)

    var out_len = n_padded // 2 + 1
    var result = ComplexNDArray[cdtype](NDArrayShape(out_len))
    for i in range(out_len):
        result[Item(i)] = full[Item(i)]
    return result^


def irfft[
    dtype: DType = DType.float64,
    *,
    cdtype: ComplexDType = ComplexDType(mlir_value=dtype._mlir_value),
](
    arr: ComplexNDArray[cdtype],
    n: Optional[Int] = None,
) raises -> NDArray[
    dtype
] where cdtype.dtype.is_floating_point():
    """Computes the inverse real Fast Fourier Transform.

    Reconstructs a real-valued signal from the non-redundant frequency bins
    produced by ``rfft``. The output length is ``2 * (M - 1)`` where ``M`` is
    the number of input bins, unless ``n`` is specified explicitly.

    Parameters:
        dtype: The floating-point element type of the output array.
        cdtype: The complex data type corresponding to the output dtype. Defaults to
            ComplexDType with the same underlying float type.

    Args:
        arr: Complex 1-D input array of ``N // 2 + 1`` frequency bins.
        n: Length of the output signal. Defaults to ``2 * (len(arr) - 1)``.
            If not a power of 2, it is rounded up to the next power of 2
            (matching NumPy behaviour).

    Raises:
        Error: If the input array is not 1-dimensional.

    Returns:
        Real-valued NDArray of length ``n_padded`` (next power of 2 ≥ n).

    Examples:
        ```mojo
        import numojo as nm
        from scijo.fft import rfft, irfft
        from scijo.prelude import *

        var x = nm.linspace[f64](0.0, 1.0, 8)
        var freqs = rfft(x)
        var x_rec = irfft[f64](freqs, 8)
        ```
    """
    if arr.ndim != 1:
        raise Error("Scijo [irfft]: irfft currently only supports 1D arrays")

    var m = arr.shape[0]
    var full_n: Int
    if n:
        full_n = n.value()
    else:
        full_n = 2 * (m - 1)

    # Round up to next power of 2
    if (full_n & (full_n - 1)) != 0:
        var padded = 1
        while padded < full_n:
            padded <<= 1
        full_n = padded

    # Reconstruct the full symmetric spectrum
    var full = ComplexNDArray[cdtype](NDArrayShape(full_n))
    for i in range(m):
        full[Item(i)] = arr[Item(i)]
    # Mirror: full[N-k] = conj(full[k]) for k = 1 .. N//2-1
    for k in range(1, full_n // 2):
        var c = arr[Item(k)]
        full[Item(full_n - k)] = ComplexSIMD[cdtype](c.re, -c.im)

    var complex_out = ifft[cdtype](full)

    var result = NDArray[dtype](NDArrayShape(full_n))
    for i in range(full_n):
        result._buf.ptr[i] = complex_out[Item(i)].re.cast[dtype]()
    return result^


def _ifft_unnormalized[
    cdtype: ComplexDType = ComplexDType.float64
](arr: ComplexNDArray[cdtype]) raises -> ComplexNDArray[
    cdtype
] where cdtype.dtype.is_floating_point():
    """Computes the unnormalized inverse FFT using the Cooley-Tukey algorithm.

    This is an internal helper that performs the inverse butterfly operations
    (with positive twiddle exponent) without the 1/N normalization factor.
    The caller is responsible for applying normalization.

    Parameters:
        cdtype: The data type of the complex elements (ComplexDType).

    Args:
        arr: Input complex array to transform. Must be 1-dimensional with length
             that is a power of 2.

    Raises:
        Error: If the input array is not 1-dimensional.
        Error: If the array length is not a power of 2.

    Returns:
        ComplexNDArray containing the unnormalized inverse FFT of the input array.
    """
    if arr.ndim != 1:
        raise Error("Scijo [fft]: FFT currently only supports 1D arrays")

    var n: Int = arr.shape[0]
    if n <= 1:
        return ComplexNDArray[cdtype](re=arr._re.copy(), im=arr._im.copy())

    if (n & (n - 1)) != 0:
        raise Error(
            "Scijo [fft]: FFT currently only supports arrays with length that"
            " is a power of 2"
        )

    var half_size = n // 2
    var even_indices = ComplexNDArray[cdtype](NDArrayShape(half_size))
    var odd_indices = ComplexNDArray[cdtype](NDArrayShape(half_size))

    for i in range(half_size):
        even_indices[Item(i)] = arr[Item(2 * i)]
        odd_indices[Item(i)] = arr[Item(2 * i + 1)]

    var even_ifft = _ifft_unnormalized[cdtype](even_indices)
    var odd_ifft = _ifft_unnormalized[cdtype](odd_indices)

    var result = ComplexNDArray[cdtype](arr.shape)

    for k in range(half_size):
        var angle = (
            2.0
            * Constants.pi
            * Scalar[cdtype.dtype](k)
            / Scalar[cdtype.dtype](n)
        )
        var twiddle = ComplexSIMD[cdtype](
            cos(angle).cast[cdtype.dtype](), sin(angle).cast[cdtype.dtype]()
        )

        var twiddle_odd = twiddle * odd_ifft[Item(k)]

        result[Item(k)] = even_ifft[Item(k)] + twiddle_odd
        result[Item(k + half_size)] = even_ifft[Item(k)] - twiddle_odd

    return result^


# ===----------------------------------------------------------------------=== #
# IFFT
# ===----------------------------------------------------------------------=== #


def ifft[
    cdtype: ComplexDType = ComplexDType.float64
](arr: ComplexNDArray[cdtype]) raises -> ComplexNDArray[
    cdtype
] where cdtype.dtype.is_floating_point():
    """Computes the Inverse Fast Fourier Transform using the Cooley-Tukey algorithm.

    Recovers the original signal from its frequency-domain representation by
    computing the unnormalized inverse FFT and applying 1/N normalization.

    Parameters:
        cdtype: The data type of the complex elements (ComplexDType).

    Args:
        arr: Input complex array to transform. Must be 1-dimensional with length
             that is a power of 2.

    Raises:
        Error: If the input array is not 1-dimensional.
        Error: If the array length is not a power of 2.

    Returns:
        ComplexNDArray containing the IFFT of the input array with the same
        shape and dtype.

    Examples:
        ```mojo
        import numojo as nm
        from scijo.fft import fft, ifft
        from scijo.prelude import *

        var arr = nm.linspace[cf32](CScalar[cf32](0, 0), CScalar[cf32](10, 10), num=10)
        var freq = fft(arr)
        var time = ifft(freq)
        ```
    """
    var n: Int = arr.shape[0]
    var result = _ifft_unnormalized[cdtype](arr)

    var inv_n = CScalar[cdtype](1.0, 1.0) / CScalar[cdtype](
        Scalar[cdtype.dtype](n), Scalar[cdtype.dtype](n)
    )
    for i in range(n):
        result.store[width=1](i, result.load[width=1](i) * inv_n)
    return result^
