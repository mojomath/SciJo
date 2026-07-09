# ===----------------------------------------------------------------------=== #
# SciJo: FFT module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""FFT Helper Functions (`scijo.fft.helpers`)
=============================================
Utility functions for working with FFT output arrays, matching the
`scipy.fft` helper API.

Examples
--------
    ```mojo
    from scijo.fft import fftfreq, rfftfreq, fftshift, ifftshift, next_fast_len

    var freqs = fftfreq(8, d=0.1)      # frequency bins for 8-point FFT, sample spacing 0.1
    var rfreqs = rfftfreq(8, d=0.1)    # frequency bins for rfft output
    ```
"""

from numojo.core import Shape
from numojo.core.ndarray import NDArray
from std.math import log2, ceil


# ===----------------------------------------------------------------------=== #
# fftfreq
# ===----------------------------------------------------------------------=== #


def fftfreq[
    dtype: DType = DType.float64
](n: Int, d: Scalar[dtype] = 1.0) raises -> NDArray[dtype]:
    """Returns the discrete Fourier transform sample frequencies.

    The returned array contains the frequency bin centres in cycles per unit
    of the sample spacing, matching `scipy.fft.fftfreq`.

    The bins are: ``[0, 1, ..., n//2-1, -n//2, ..., -1] / (d*n)``

    Parameters:
        dtype: The floating-point data type. Defaults to DType.float64.

    Args:
        n: Window length (number of samples).
        d: Sample spacing (inverse of the sampling rate). Defaults to 1.0.

    Returns:
        NDArray of shape (n,) containing the frequency bin centres.

    Raises:
        Error: If n <= 0.

    Examples:
        ```mojo
        from scijo.fft import fftfreq

        var freqs = fftfreq(8)
        # [0.0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125]
        var freqs_d = fftfreq(8, d=0.5)
        # [0.0, 0.25, 0.5, 0.75, -1.0, -0.75, -0.5, -0.25]
        ```
    """
    if n <= 0:
        raise Error("Scijo [fftfreq]: n must be positive, got " + String(n))

    var result = NDArray[dtype](Shape(n))
    var half = n // 2
    var scale = Scalar[dtype](1.0) / (Scalar[dtype](n) * d)

    for i in range(half):
        result._buf.ptr[i] = Scalar[dtype](i) * scale
    for i in range(half, n):
        result._buf.ptr[i] = Scalar[dtype](i - n) * scale

    return result^


# ===----------------------------------------------------------------------=== #
# rfftfreq
# ===----------------------------------------------------------------------=== #


def rfftfreq[
    dtype: DType = DType.float64
](n: Int, d: Scalar[dtype] = 1.0) raises -> NDArray[dtype]:
    """Returns the discrete Fourier transform sample frequencies for rfft output.

    The returned array contains the frequency bin centres for the output of
    `rfft`, matching `scipy.fft.rfftfreq`. Only non-negative frequencies are
    returned since the input is real.

    The bins are: ``[0, 1, ..., n//2] / (d*n)``

    Parameters:
        dtype: The floating-point data type. Defaults to DType.float64.

    Args:
        n: Window length (number of samples in the *original* real signal).
        d: Sample spacing (inverse of the sampling rate). Defaults to 1.0.

    Returns:
        NDArray of shape (n//2 + 1,) containing the non-negative frequency bins.

    Raises:
        Error: If n <= 0.

    Examples:
        ```mojo
        from scijo.fft import rfftfreq

        var freqs = rfftfreq(8)
        # [0.0, 0.125, 0.25, 0.375, 0.5]
        ```
    """
    if n <= 0:
        raise Error("Scijo [rfftfreq]: n must be positive, got " + String(n))

    var out_len = n // 2 + 1
    var result = NDArray[dtype](Shape(out_len))
    var scale = Scalar[dtype](1.0) / (Scalar[dtype](n) * d)

    for i in range(out_len):
        result._buf.ptr[i] = Scalar[dtype](i) * scale

    return result^


# ===----------------------------------------------------------------------=== #
# fftshift
# ===----------------------------------------------------------------------=== #


def fftshift[
    dtype: DType = DType.float64
](x: NDArray[dtype]) raises -> NDArray[dtype]:
    """Shifts the zero-frequency component to the centre of the spectrum.

    Rearranges the output of `fft` by moving the zero-frequency component
    to the middle of the array, matching `scipy.fft.fftshift`.

    For a 1-D array of length n, the element at index k moves to index
    ``(k + n//2) % n``.

    Parameters:
        dtype: The floating-point data type. Defaults to DType.float64.

    Args:
        x: Input 1-D array (typically the output of `fftfreq` or magnitude
           spectrum from `fft`).

    Returns:
        Shifted array with the same shape as x.

    Raises:
        Error: If x is not 1-dimensional.

    Examples:
        ```mojo
        from scijo.fft import fftfreq, fftshift

        var freqs = fftfreq(8)              # [0, 0.125, 0.25, 0.375, -0.5, ...]
        var shifted = fftshift(freqs)       # [-0.5, -0.375, ..., 0, 0.125, ...]
        ```
    """
    if x.ndim != 1:
        raise Error("Scijo [fftshift]: only 1-D arrays are supported")

    var n = x.size
    var result = NDArray[dtype](x.shape)
    var shift = n // 2

    for i in range(n):
        result._buf.ptr[(i + shift) % n] = x._buf.ptr[i]

    return result^


# ===----------------------------------------------------------------------=== #
# ifftshift
# ===----------------------------------------------------------------------=== #


def ifftshift[
    dtype: DType = DType.float64
](x: NDArray[dtype]) raises -> NDArray[dtype]:
    """Inverse of `fftshift`.

    Undoes the shift performed by `fftshift`, restoring the standard FFT
    ordering with zero-frequency at index 0, matching `scipy.fft.ifftshift`.

    For a 1-D array of length n, the element at index k moves to index
    ``(k - n//2) % n`` (equivalently ``(k + (n+1)//2) % n`` for odd n).

    Parameters:
        dtype: The floating-point data type. Defaults to DType.float64.

    Args:
        x: Shifted 1-D array (typically the output of `fftshift`).

    Returns:
        Array with zero-frequency component restored to index 0.

    Raises:
        Error: If x is not 1-dimensional.

    Examples:
        ```mojo
        from scijo.fft import fftshift, ifftshift, fftfreq

        var freqs = fftfreq(8)
        var shifted = fftshift(freqs)
        var restored = ifftshift(shifted)  # equals freqs
        ```
    """
    if x.ndim != 1:
        raise Error("Scijo [ifftshift]: only 1-D arrays are supported")

    var n = x.size
    var result = NDArray[dtype](x.shape)
    var shift = (n + 1) // 2

    for i in range(n):
        result._buf.ptr[(i + shift) % n] = x._buf.ptr[i]

    return result^


# ===----------------------------------------------------------------------=== #
# next_fast_len
# ===----------------------------------------------------------------------=== #


def next_fast_len(n: Int) raises -> Int:
    """Returns the next fast FFT input size >= n.

    Finds the smallest integer >= n that is a power of 2, matching the
    spirit of `scipy.fft.next_fast_len` for the current Cooley-Tukey
    radix-2 implementation (which requires power-of-2 lengths).

    Args:
        n: Minimum desired length.

    Returns:
        Smallest power of 2 that is >= n.

    Raises:
        Error: If n <= 0.

    Examples:
        ```mojo
        from scijo.fft import next_fast_len

        print(next_fast_len(9))    # 16
        print(next_fast_len(16))   # 16
        print(next_fast_len(100))  # 128
        ```
    """
    if n <= 0:
        raise Error(
            "Scijo [next_fast_len]: n must be positive, got " + String(n)
        )
    if n == 1:
        return 1

    var p = 1
    while p < n:
        p <<= 1
    return p
