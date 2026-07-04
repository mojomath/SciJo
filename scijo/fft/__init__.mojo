# ===----------------------------------------------------------------------=== #
# SciJo: FFT module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""FFT Module (`scijo.fft`)
===========================
Provides Fast Fourier Transform operations for complex and real arrays,
along with helper utilities for working with FFT output.

Available Functions
-------------------
- `fft`          — Compute the forward FFT (complex input).
- `ifft`         — Compute the inverse FFT (complex input).
- `rfft`         — Compute the FFT of a real array, returning N//2+1 bins.
- `irfft`        — Compute the inverse FFT returning a real array.
- `fftfreq`      — Frequency bin centres for fft output.
- `rfftfreq`     — Frequency bin centres for rfft output (non-negative only).
- `fftshift`     — Shift zero-frequency component to the centre of the array.
- `ifftshift`    — Inverse of fftshift.
- `next_fast_len`— Next power-of-2 size >= n for efficient FFT.

Examples
--------
    ```mojo
    from scijo.fft import fft, ifft, rfft, irfft, fftfreq, fftshift

    var arr = nm.linspace[cf32](CScalar[cf32](0, 0), CScalar[cf32](10, 10), num=10)
    var fft_arr = fft(arr)
    var time = ifft(fft_arr)

    var freqs = fftfreq(8, d=0.1)
    var shifted = fftshift(freqs)
    ```
"""

from .fastfourier import fft, ifft, rfft, irfft
from .helpers import fftfreq, rfftfreq, fftshift, ifftshift, next_fast_len
