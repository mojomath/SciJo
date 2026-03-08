# ===----------------------------------------------------------------------=== #
# Scijo: FFT - Fast Fourier Transform
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""FFT Module - Fast Fourier Transform (scijo.fft.fastfourier)

Forward and inverse Fast Fourier Transform using the Cooley-Tukey radix-2
decimation-in-time algorithm for 1-D complex arrays with power-of-2 lengths.
"""

from numojo.core.complex import ComplexNDArray, ComplexSIMD
from numojo.core.dtype import ComplexDType
from numojo.core.ndarray import NDArray
from numojo.core.layout import NDArrayShape
from numojo.routines.constants import Constants
from numojo.core.indexing import Item

from math import sin, cos


fn fft[
    dtype: ComplexDType = ComplexDType.float64
](arr: ComplexNDArray[dtype]) raises -> ComplexNDArray[
    dtype
] where dtype.dtype.is_floating_point():
    """Computes the Fast Fourier Transform using the Cooley-Tukey algorithm.

    Decomposes the DFT computation by recursively breaking down the transform
    of size N into two transforms of size N/2, achieving O(N log N) complexity.

    Parameters:
        dtype: The data type of the complex elements (ComplexDType).

    Args:
        arr: Input complex array to transform. Must be 1-dimensional with length
             that is a power of 2.

    Returns:
        ComplexNDArray containing the FFT of the input array with the same
        shape and dtype.

    Raises:
        Error: If the input array is not 1-dimensional.
        Error: If the array length is not a power of 2.
    """
    if arr.ndim != 1:
        raise Error("FFT currently only supports 1D arrays")

    var n: Int = arr.shape[0]
    if n <= 1:
        return ComplexNDArray[dtype](re=arr._re.copy(), im=arr._im.copy())

    if (n & (n - 1)) != 0:
        raise Error(
            "FFT currently only supports arrays with length that is a power"
            " of 2"
        )

    var half_size = n // 2
    var even_indices = ComplexNDArray[dtype](NDArrayShape(half_size))
    var odd_indices = ComplexNDArray[dtype](NDArrayShape(half_size))

    for i in range(half_size):
        even_indices[Item(i)] = arr[Item(2 * i)]
        odd_indices[Item(i)] = arr[Item(2 * i + 1)]

    var even_fft = fft[dtype](even_indices)
    var odd_fft = fft[dtype](odd_indices)

    var result = ComplexNDArray[dtype](arr.shape)

    for k in range(half_size):
        var angle = (
            -2.0
            * Constants.pi
            * Scalar[dtype.dtype](k)
            / Scalar[dtype.dtype](n)
        )
        var twiddle = ComplexSIMD[dtype](
            cos(angle).cast[dtype.dtype](), sin(angle).cast[dtype.dtype]()
        )

        var twiddle_odd = twiddle * odd_fft[Item(k)]

        result[Item(k)] = even_fft[Item(k)] + twiddle_odd
        result[Item(k + half_size)] = even_fft[Item(k)] - twiddle_odd

    return result^


fn _ifft_unnormalized[
    dtype: ComplexDType = ComplexDType.float64
](arr: ComplexNDArray[dtype]) raises -> ComplexNDArray[
    dtype
] where dtype.dtype.is_floating_point():
    """Computes the unnormalized inverse FFT using the Cooley-Tukey algorithm.

    This is an internal helper that performs the inverse butterfly operations
    (with positive twiddle exponent) without the 1/N normalization factor.
    The caller is responsible for applying normalization.

    Parameters:
        dtype: The data type of the complex elements (ComplexDType).

    Args:
        arr: Input complex array to transform. Must be 1-dimensional with length
             that is a power of 2.

    Returns:
        ComplexNDArray containing the unnormalized inverse FFT of the input array.

    Raises:
        Error: If the input array is not 1-dimensional.
        Error: If the array length is not a power of 2.
    """
    if arr.ndim != 1:
        raise Error("FFT currently only supports 1D arrays")

    var n: Int = arr.shape[0]
    if n <= 1:
        return ComplexNDArray[dtype](re=arr._re.copy(), im=arr._im.copy())

    if (n & (n - 1)) != 0:
        raise Error(
            "FFT currently only supports arrays with length that is a power"
            " of 2"
        )

    var half_size = n // 2
    var even_indices = ComplexNDArray[dtype](NDArrayShape(half_size))
    var odd_indices = ComplexNDArray[dtype](NDArrayShape(half_size))

    for i in range(half_size):
        even_indices[Item(i)] = arr[Item(2 * i)]
        odd_indices[Item(i)] = arr[Item(2 * i + 1)]

    var even_ifft = _ifft_unnormalized[dtype](even_indices)
    var odd_ifft = _ifft_unnormalized[dtype](odd_indices)

    var result = ComplexNDArray[dtype](arr.shape)

    for k in range(half_size):
        var angle = (
            2.0 * Constants.pi * Scalar[dtype.dtype](k) / Scalar[dtype.dtype](n)
        )
        var twiddle = ComplexSIMD[dtype](
            cos(angle).cast[dtype.dtype](), sin(angle).cast[dtype.dtype]()
        )

        var twiddle_odd = twiddle * odd_ifft[Item(k)]

        result[Item(k)] = even_ifft[Item(k)] + twiddle_odd
        result[Item(k + half_size)] = even_ifft[Item(k)] - twiddle_odd

    return result^


fn ifft[
    dtype: ComplexDType = ComplexDType.float64
](arr: ComplexNDArray[dtype]) raises -> ComplexNDArray[
    dtype
] where dtype.dtype.is_floating_point():
    """Computes the Inverse Fast Fourier Transform using the Cooley-Tukey algorithm.

    Recovers the original signal from its frequency-domain representation by
    computing the unnormalized inverse FFT and applying 1/N normalization.

    Parameters:
        dtype: The data type of the complex elements (ComplexDType).

    Args:
        arr: Input complex array to transform. Must be 1-dimensional with length
             that is a power of 2.

    Returns:
        ComplexNDArray containing the IFFT of the input array with the same
        shape and dtype.

    Raises:
        Error: If the input array is not 1-dimensional.
        Error: If the array length is not a power of 2.
    """
    var n: Int = arr.shape[0]
    var result = _ifft_unnormalized[dtype](arr)

    var inv_n = CScalar[dtype](1.0, 1.0) / CScalar[dtype](
        Scalar[dtype.dtype](n), Scalar[dtype.dtype](n)
    )
    for i in range(n):
        result.store[width=1](i, result.load[width=1](i) * inv_n)
    return result^
