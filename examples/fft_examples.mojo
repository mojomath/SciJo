"""
A tour of `scijo.fft`: forward/inverse FFT on complex arrays, real FFT
(`rfft`/`irfft`), and the frequency-domain helper functions.

Run with (from the repository root, after `pixi run package`):

```bash
mojo run -I . -I tests/ examples/fft_examples.mojo
```
"""

import numojo as nm
from numojo.core import CScalar
from scijo.fft import (
    fft,
    ifft,
    fftn,
    ifftn,
    fft2,
    ifft2,
    rfft,
    irfft,
    fftfreq,
    rfftfreq,
    fftshift,
    ifftshift,
    next_fast_len,
)
from scijo.prelude import *


def main() raises:
    fft_ifft_example()
    fftn_example()
    rfft_irfft_example()
    frequency_helpers()


# ===----------------------------------------------------------------------=== #
# fft / ifft: power-of-2 length (direct radix-2) vs. any other length
# (Bluestein's algorithm) - both are exact, no padding either way.
# ===----------------------------------------------------------------------=== #


def fft_ifft_example() raises:
    print("=" * 80)
    print("FFT / IFFT: complex input, power-of-2 length (radix-2)")
    print("=" * 80)

    var arr = nm.arange[cf64](CScalar[cf64](0), CScalar[cf64](8))
    print("Input (8 complex samples):")
    print(arr)

    var freq = fft(arr)
    print("fft(arr):")
    print(freq)

    var time = ifft(freq)
    print("ifft(fft(arr)) - recovers the input, up to floating-point error:")
    print(time)

    print()
    print("=" * 80)
    print("FFT / IFFT: complex input, length 5 (Bluestein, not a power of 2)")
    print("=" * 80)

    var arr5 = nm.arange[cf64](CScalar[cf64](0), CScalar[cf64](5))
    print("Input (5 complex samples):")
    print(arr5)

    var freq5 = fft(arr5)
    print("fft(arr5) - computed exactly at length 5, no padding:")
    print(freq5)

    var time5 = ifft(freq5)
    print("ifft(fft(arr5)) - recovers the input:")
    print(time5)


# ===----------------------------------------------------------------------=== #
# fftn / ifftn / fft2 / ifft2 on a 2-D array
# ===----------------------------------------------------------------------=== #


def fftn_example() raises:
    print()
    print("=" * 80)
    print("FFTN / IFFTN / FFT2 / IFFT2: 2-D input, shape (4, 4)")
    print("=" * 80)

    var arr = nm.arange[cf64](CScalar[cf64](0), CScalar[cf64](16)).reshape(
        Shape(4, 4)
    )
    print("Input (4x4 complex samples):")
    print(arr)

    var freq = fftn(arr)  # transform over every axis (equivalent to fft2)
    print("fftn(arr) - transform over both axes:")
    print(freq)

    var freq2 = fft2(arr)  # convenience wrapper for the last two axes
    print("fft2(arr) - same result via the fft2 wrapper:")
    print(freq2)

    var axis_1: List[Int] = [1]
    var freq_rows = fftn(arr, axes=axis_1^)  # transform along axis 1 only
    print("fftn(arr, axes=[1]) - transform along axis 1 only:")
    print(freq_rows)

    var time = ifftn(freq)
    print("ifftn(fftn(arr)) - recovers the input, up to floating-point error:")
    print(time)

    var time2 = ifft2(freq2)
    print("ifft2(fft2(arr)) - same roundtrip via the fft2/ifft2 wrappers:")
    print(time2)


# ===----------------------------------------------------------------------=== #
# rfft / irfft on real input
# ===----------------------------------------------------------------------=== #


def rfft_irfft_example() raises:
    print()
    print("=" * 80)
    print("RFFT / IRFFT: real input, any length (no padding)")
    print("=" * 80)

    var x = nm.linspace[f64](0.0, 1.0, 6)  # length 6, not a power of 2
    print("Input (6 real samples):")
    print(x)

    var freqs = rfft(x)  # returns 6 // 2 + 1 = 4 bins, at the exact length
    print("rfft(x) -> 4 non-redundant bins (no padding to length 8):")
    print(freqs)

    var x_rec = irfft(freqs, 6)
    print("irfft(freqs, n=6) -> reconstructs the original length-6 signal:")
    print(x_rec)


# ===----------------------------------------------------------------------=== #
# Frequency-domain helpers
# ===----------------------------------------------------------------------=== #


def frequency_helpers() raises:
    print()
    print("=" * 80)
    print("FREQUENCY HELPERS")
    print("=" * 80)

    var freqs = fftfreq(8, d=0.1)
    print("fftfreq(8, d=0.1):")
    print(freqs)

    var rfreqs = rfftfreq(8, d=0.1)
    print("rfftfreq(8, d=0.1):")
    print(rfreqs)

    var shifted = fftshift(freqs)
    print("fftshift(freqs) - zero-frequency moved to the centre:")
    print(shifted)

    var restored = ifftshift(shifted)
    print("ifftshift(shifted) - equals freqs again:")
    print(restored)

    print("next_fast_len(9)   =", next_fast_len(9))  # 16
    print("next_fast_len(16)  =", next_fast_len(16))  # 16
    print("next_fast_len(100) =", next_fast_len(100))  # 128
