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
    rfft_irfft_example()
    frequency_helpers()


# ===----------------------------------------------------------------------=== #
# fft / ifft on a complex, power-of-2-length array
# ===----------------------------------------------------------------------=== #


def fft_ifft_example() raises:
    print("=" * 80)
    print("FFT / IFFT: complex input, power-of-2 length")
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

    # A length that is not a power of 2 raises.
    var bad = nm.arange[cf64](CScalar[cf64](0), CScalar[cf64](5))
    try:
        _ = fft(bad)
    except e:
        print("fft on a length-5 array raises:")
        print(" ", e)


# ===----------------------------------------------------------------------=== #
# rfft / irfft on real input
# ===----------------------------------------------------------------------=== #


def rfft_irfft_example() raises:
    print()
    print("=" * 80)
    print("RFFT / IRFFT: real input, any length (zero-padded internally)")
    print("=" * 80)

    var x = nm.linspace[f64](0.0, 1.0, 6)  # length 6, not a power of 2
    print("Input (6 real samples), zero-padded to 8 internally:")
    print(x)

    var freqs = rfft(x)  # returns 8 // 2 + 1 = 5 bins
    print("rfft(x) -> 5 non-redundant bins:")
    print(freqs)

    var x_rec = irfft(freqs, 8)
    print("irfft(freqs, n=8) -> reconstructs the padded 8-length signal:")
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
