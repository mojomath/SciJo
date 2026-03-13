# ===----------------------------------------------------------------------=== #
# Scijo: FFT
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""FFT Module (scijo.fft)

The `fft` module provides Fast Fourier Transform operations for complex-valued
arrays. It includes forward and inverse FFT using the Cooley-Tukey algorithm.
"""

from .fastfourier import fft, ifft
