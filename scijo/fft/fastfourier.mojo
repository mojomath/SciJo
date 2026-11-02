# ===----------------------------------------------------------------------=== #
# SciJo: FFT module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Fast Fourier Transform (`scijo.fft.fastfourier`)
===================================================
Forward and inverse Fast Fourier Transform for 1-D complex arrays of any
length. Power-of-2 lengths use the Cooley-Tukey radix-2 decimation-in-time
algorithm directly; every other length is handled by Bluestein's
algorithm, which rewrites the length-`n` DFT as a length-`m` convolution
(`m` a power of 2, `m >= 2n - 1`) and evaluates that convolution with the
same radix-2 engine - so every length is `O(n log n)`, matching NumPy's
and SciPy's length semantics (no implicit padding).

Constraints
-----------
- Input arrays must be 1-dimensional.

Examples
--------
    ```mojo
    from scijo.fft import fft, ifft

    var arr = nm.linspace[cf32](CScalar[cf32](0, 0), CScalar[cf32](10, 10), num=10)
    var freq = fft(arr)
    var time = ifft(freq)
    ```
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.math import (
    cos,
    sin,
)

# ===----------------------------------------------------------------------=== #
# External
# ===----------------------------------------------------------------------=== #
from numojo.core.complex import (
    ComplexNDArray,
    ComplexSIMD,
)
from numojo.core.dtype import ComplexDType
from numojo.core.layout import NDArrayShape
from numojo.core.ndarray import NDArray
from numojo.core.type_aliases import CScalar
from numojo.routines.constants import Constants

# ===----------------------------------------------------------------------=== #
# FFT
# ===----------------------------------------------------------------------=== #


def fft[
    cdtype: ComplexDType = ComplexDType.float64
](arr: ComplexNDArray[cdtype]) raises -> ComplexNDArray[
    cdtype
] where cdtype.dtype.is_floating_point():
    """Computes the Fast Fourier Transform.

    Power-of-2 lengths use the Cooley-Tukey radix-2 algorithm directly;
    every other length is handled by Bluestein's algorithm (see the module
    docstring), so the full-length transform is computed exactly - no
    length is silently padded or truncated.

    Parameters:
        cdtype: The data type of the complex elements (ComplexDType).

    Args:
        arr: Input complex array to transform. Must be 1-dimensional.

    Returns:
        ComplexNDArray containing the FFT of the input array with the same
        shape and dtype.

    Raises:
        Error: If the input array is not 1-dimensional.

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

    if _is_pow2(n):
        return _fft_radix2[cdtype](arr)
    return _bluestein[cdtype](arr, sign=-1)


def _fft_radix2[
    cdtype: ComplexDType = ComplexDType.float64
](arr: ComplexNDArray[cdtype]) raises -> ComplexNDArray[
    cdtype
] where cdtype.dtype.is_floating_point():
    """Computes the forward FFT via the Cooley-Tukey radix-2 algorithm.

    Decomposes the DFT computation by recursively breaking down the transform
    of size N into two transforms of size N/2, achieving O(N log N) complexity.
    Internal engine: `arr`'s length must already be a power of 2 (checked by
    callers, not here).
    """
    var n: Int = arr.shape[0]
    if n <= 1:
        return ComplexNDArray[cdtype](re=arr._re.copy(), im=arr._im.copy())

    var half_size = n // 2
    var even_indices = ComplexNDArray[cdtype](NDArrayShape(half_size))
    var odd_indices = ComplexNDArray[cdtype](NDArrayShape(half_size))

    for i in range(half_size):
        even_indices.unsafe_store(i, arr.unsafe_load(2 * i))
        odd_indices.unsafe_store(i, arr.unsafe_load(2 * i + 1))

    var even_fft = _fft_radix2[cdtype](even_indices)
    var odd_fft = _fft_radix2[cdtype](odd_indices)

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

        var even_k = even_fft.unsafe_load(k)
        var twiddle_odd = twiddle * odd_fft.unsafe_load(k)

        result.unsafe_store(k, even_k + twiddle_odd)
        result.unsafe_store(k + half_size, even_k - twiddle_odd)

    return result^


# ===----------------------------------------------------------------------=== #
# Arbitrary-length support (Bluestein's algorithm)
# ===----------------------------------------------------------------------=== #


def _is_pow2(n: Int) -> Bool:
    """Whether `n` is a power of 2 (`n >= 1`)."""
    return (n & (n - 1)) == 0


def _next_pow2(n: Int) -> Int:
    """The smallest power of 2 that is `>= n`."""
    var p = 1
    while p < n:
        p <<= 1
    return p


def _chirp[
    cdtype: ComplexDType
](m: Int, n: Int, sign: Int) -> ComplexSIMD[
    cdtype
] where cdtype.dtype.is_floating_point():
    """`exp(sign * i * pi * m**2 / n)`, the Bluestein chirp factor.

    `m` is squared in floating point (not `Int`) so this stays correct for
    `m` large enough that `m * m` would overflow `Int`.
    """
    var mf = Scalar[cdtype.dtype](m)
    var angle = (
        Scalar[cdtype.dtype](sign)
        * Constants.pi
        * mf
        * mf
        / Scalar[cdtype.dtype](n)
    )
    return ComplexSIMD[cdtype](
        cos(angle).cast[cdtype.dtype](), sin(angle).cast[cdtype.dtype]()
    )


def _bluestein[
    cdtype: ComplexDType
](arr: ComplexNDArray[cdtype], sign: Int) raises -> ComplexNDArray[
    cdtype
] where cdtype.dtype.is_floating_point():
    """Computes a length-`n` DFT of arbitrary `n` via Bluestein's algorithm.

    `sign=-1` computes the forward DFT (`exp(-2*pi*i*k*n/N)`); `sign=1`
    computes the unnormalized inverse DFT (`exp(+2*pi*i*k*n/N)`). Rewrites
    the length-`N` transform as a length-`M` circular convolution
    (`M` a power of 2, `M >= 2*N - 1`), evaluated with the radix-2 engine,
    via the identity `2*k*n = k**2 + n**2 - (k - n)**2`.

    References:
        Bluestein, L. I. (1970). "A linear filtering approach to the
        computation of the discrete Fourier transform." IEEE Transactions
        on Audio and Electroacoustics, 18(4), 451-455.
    """
    var n: Int = arr.shape[0]
    var m = _next_pow2(2 * n - 1)

    # b[i] = x[i] * chirp(i); zero-padded to the convolution length m.
    var b = ComplexNDArray[cdtype](NDArrayShape(m))
    for i in range(n):
        b.unsafe_store(i, arr.unsafe_load(i) * _chirp[cdtype](i, n, sign))
    for i in range(n, m):
        b.unsafe_store(i, ComplexSIMD[cdtype](0, 0))

    # c[j] = conj(chirp(j)) for the circular convolution kernel, placed at
    # both +j and -j (mod m); chirp(j) == chirp(-j) since it depends on j**2.
    var c = ComplexNDArray[cdtype](NDArrayShape(m))
    c.unsafe_store(0, _chirp[cdtype](0, n, -sign))
    for j in range(1, n):
        var cj = _chirp[cdtype](j, n, -sign)
        c.unsafe_store(j, cj)
        c.unsafe_store(m - j, cj)
    for j in range(n, m - n + 1):
        c.unsafe_store(j, ComplexSIMD[cdtype](0, 0))

    var B = _fft_radix2[cdtype](b)
    var C = _fft_radix2[cdtype](c)
    var conv_freq = ComplexNDArray[cdtype](NDArrayShape(m))
    for i in range(m):
        conv_freq.unsafe_store(i, B.unsafe_load(i) * C.unsafe_load(i))

    var conv = _ifft_radix2_unnormalized[cdtype](conv_freq)
    var m_scalar = Scalar[cdtype.dtype](m)

    var result = ComplexNDArray[cdtype](arr.shape)
    for k in range(n):
        result.unsafe_store(
            k,
            _chirp[cdtype](k, n, sign) * (conv.unsafe_load(k) / m_scalar),
        )
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
    the complex conjugate mirror of the lower half for real input). The
    transform length is exactly `len(arr)` - matching NumPy/SciPy, no length
    is silently padded.

    Parameters:
        dtype: The floating-point element type of the input array.
        cdtype: The complex data type corresponding to the input dtype. Defaults to
            ComplexDType with the same underlying float type.

    Args:
        arr: Real-valued 1-D input array.

    Returns:
        ComplexNDArray of length ``len(arr) // 2 + 1`` containing the
        non-redundant frequency components.

    Raises:
        Error: If the input array is not 1-dimensional.

    Examples:
        ```mojo
        import numojo as nm
        from scijo.fft import rfft, irfft
        from scijo.prelude import *

        var x = nm.linspace[f64](0.0, 1.0, 6)
        var freqs = rfft(x)                       # shape: (4,)
        var x_rec = irfft(freqs, 6)                # shape: (6,)
        ```
    """
    if arr.ndim != 1:
        raise Error("Scijo [rfft]: rfft currently only supports 1D arrays")

    var n = arr.shape[0]
    var complex_input = ComplexNDArray[cdtype](NDArrayShape(n))
    for i in range(n):
        complex_input.unsafe_store(
            i,
            ComplexSIMD[cdtype](
                arr.unsafe_load(i).cast[cdtype.dtype](),
                Scalar[cdtype.dtype](0),
            ),
        )

    var full = fft[cdtype](complex_input)

    var out_len = n // 2 + 1
    var result = ComplexNDArray[cdtype](NDArrayShape(out_len))
    for i in range(out_len):
        result.unsafe_store(i, full.unsafe_load(i))
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
    the number of input bins, unless ``n`` is specified explicitly - no
    length is rounded or padded, matching NumPy/SciPy.

    Parameters:
        dtype: The floating-point element type of the output array.
        cdtype: The complex data type corresponding to the output dtype. Defaults to
            ComplexDType with the same underlying float type.

    Args:
        arr: Complex 1-D input array of ``N // 2 + 1`` frequency bins.
        n: Length of the output signal. Defaults to ``2 * (len(arr) - 1)``.

    Returns:
        Real-valued NDArray of length ``n`` (or the default above).

    Raises:
        Error: If the input array is not 1-dimensional.

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

    # Reconstruct the full symmetric spectrum.
    var full = ComplexNDArray[cdtype](NDArrayShape(full_n))
    for i in range(m):
        full.unsafe_store(i, arr.unsafe_load(i))
    # Mirror: full[N-k] = conj(full[k]) for every k whose mirror partner
    # N-k differs from k itself (the self-paired Nyquist bin, which only
    # exists when N is even, needs no mirroring).
    for k in range(1, m):
        if full_n - k != k:
            var c = arr.unsafe_load(k)
            full.unsafe_store(full_n - k, ComplexSIMD[cdtype](c.re, -c.im))

    var complex_out = ifft[cdtype](full)

    var result = NDArray[dtype](NDArrayShape(full_n))
    for i in range(full_n):
        result.unsafe_store(i, complex_out.unsafe_load(i).re.cast[dtype]())
    return result^


def _ifft_unnormalized[
    cdtype: ComplexDType = ComplexDType.float64
](arr: ComplexNDArray[cdtype]) raises -> ComplexNDArray[
    cdtype
] where cdtype.dtype.is_floating_point():
    """Computes the unnormalized inverse FFT, for any length.

    This is an internal helper that performs the inverse transform without
    the 1/N normalization factor; the caller is responsible for applying
    it. Power-of-2 lengths use `_ifft_radix2_unnormalized` directly; every
    other length goes through `_bluestein`.

    Parameters:
        cdtype: The data type of the complex elements (ComplexDType).

    Args:
        arr: Input complex array to transform. Must be 1-dimensional.

    Returns:
        ComplexNDArray containing the unnormalized inverse FFT of the input array.

    Raises:
        Error: If the input array is not 1-dimensional.
    """
    if arr.ndim != 1:
        raise Error("Scijo [fft]: FFT currently only supports 1D arrays")

    var n: Int = arr.shape[0]
    if n <= 1:
        return ComplexNDArray[cdtype](re=arr._re.copy(), im=arr._im.copy())

    if _is_pow2(n):
        return _ifft_radix2_unnormalized[cdtype](arr)
    return _bluestein[cdtype](arr, sign=1)


def _ifft_radix2_unnormalized[
    cdtype: ComplexDType = ComplexDType.float64
](arr: ComplexNDArray[cdtype]) raises -> ComplexNDArray[
    cdtype
] where cdtype.dtype.is_floating_point():
    """Computes the unnormalized inverse FFT via the Cooley-Tukey radix-2
    algorithm.

    This is an internal helper that performs the inverse butterfly operations
    (with positive twiddle exponent) without the 1/N normalization factor.
    Internal engine: `arr`'s length must already be a power of 2 (checked
    by callers, not here).
    """
    var n: Int = arr.shape[0]
    if n <= 1:
        return ComplexNDArray[cdtype](re=arr._re.copy(), im=arr._im.copy())

    var half_size = n // 2
    var even_indices = ComplexNDArray[cdtype](NDArrayShape(half_size))
    var odd_indices = ComplexNDArray[cdtype](NDArrayShape(half_size))

    for i in range(half_size):
        even_indices.unsafe_store(i, arr.unsafe_load(2 * i))
        odd_indices.unsafe_store(i, arr.unsafe_load(2 * i + 1))

    var even_ifft = _ifft_radix2_unnormalized[cdtype](even_indices)
    var odd_ifft = _ifft_radix2_unnormalized[cdtype](odd_indices)

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

        var even_k = even_ifft.unsafe_load(k)
        var twiddle_odd = twiddle * odd_ifft.unsafe_load(k)

        result.unsafe_store(k, even_k + twiddle_odd)
        result.unsafe_store(k + half_size, even_k - twiddle_odd)

    return result^


# ===----------------------------------------------------------------------=== #
# IFFT
# ===----------------------------------------------------------------------=== #


def ifft[
    cdtype: ComplexDType = ComplexDType.float64
](arr: ComplexNDArray[cdtype]) raises -> ComplexNDArray[
    cdtype
] where cdtype.dtype.is_floating_point():
    """Computes the Inverse Fast Fourier Transform.

    Recovers the original signal from its frequency-domain representation by
    computing the unnormalized inverse FFT (see `fft` for the power-of-2 vs.
    Bluestein dispatch) and applying 1/N normalization.

    Parameters:
        cdtype: The data type of the complex elements (ComplexDType).

    Args:
        arr: Input complex array to transform. Must be 1-dimensional.

    Returns:
        ComplexNDArray containing the IFFT of the input array with the same
        shape and dtype.

    Raises:
        Error: If the input array is not 1-dimensional.

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
        result.unsafe_store(i, result.unsafe_load(i) * inv_n)
    return result^
