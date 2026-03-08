# ===----------------------------------------------------------------------=== #
# Scijo: A Scientific Computation Library for Mojo
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""
SciJo Top-Level Package (`scijo`)
==================================

Welcome to SciJo, a scientific computation library built for the Mojo programming language.

This top-level package exposes the core components of SciJo, including array types, error handling, and type definitions,
as well as a suite of modules for advanced numerical tasks.

Available Modules
-----------------
- `constants`: Common mathematical and physical constants used throughout the library.
- `differentiate`: Tools for numerical differentiation and gradient computation.
- `integrate`: Numerical integration routines for single and multi-dimensional problems.
- `fft`: Fast Fourier Transform algorithms for signal processing and spectral analysis.
- `interpolate`: Interpolation methods for estimating values between data points.
- `optimize`: Optimization algorithms for solving minimization and maximization problems.

Explore the documentation for each module to get started with scientific computing in Mojo using SciJo.
"""

from numojo.prelude import *
from numojo.core.error import NumojoError
