# ===----------------------------------------------------------------------=== #
# SciJo: A Scientific Computation Library for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""SciJo Top-Level Package (`scijo`)
====================================
Welcome to SciJo, a scientific computation library built for the Mojo programming language.

This top-level package exposes the core components of SciJo, including array types, error
handling, and type definitions, as well as a suite of modules for advanced numerical tasks.

Available Modules
-----------------
- `constants`     — Common mathematical and physical constants.
- `differentiate` — Tools for numerical differentiation and gradient computation.
- `integrate`     — Numerical integration routines for single and multi-dimensional problems.
- `fft`           — Fast Fourier Transform algorithms for signal processing.
- `interpolate`   — Interpolation methods for estimating values between data points.
- `optimize`      — Optimization algorithms for minimization and root-finding.

Examples
--------
    ```mojo
    from scijo.constants import pi, c
    from scijo.integrate import quad
    from scijo.differentiate import derivative
    ```
"""

from numojo.prelude import *
from numojo.core.error import NumojoError
