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
- `constants`     - Common mathematical and physical constants.
- `differentiate` - Tools for numerical differentiation and gradient computation.
- `integrate`     - Numerical integration routines for single and multi-dimensional problems.
- `fft`           - Fast Fourier Transform algorithms for signal processing.
- `interpolate`   - Interpolation methods for estimating values between data points.
- `optimize`      - Optimization algorithms for minimization and root-finding.

The common dtype aliases (`f16`, `f32`, `f64`, `i8`, ..., `i64`, `u8`, ...,
`u64`, `boolean`, etc.) are re-exported here for convenience, so `scijo.f64`
works without a separate import. Users who also want NuMojo's array types
(`NDArray`, `Shape`, ...) in scope should import `scijo.prelude` explicitly:

    ```mojo
    from scijo.prelude import *
    ```

Examples
--------
    ```mojo
    from scijo.constants import pi, c
    from scijo.integrate import quad
    from scijo.differentiate import derivative
    import scijo as sj

    var x: Scalar[sj.f64] = 1.0
    ```
"""

# ===----------------------------------------------------------------------=== #
# External
# ===----------------------------------------------------------------------=== #
from numojo.core.dtype import (
    bf16,
    boolean,
    f16,
    f32,
    f64,
    i128,
    i16,
    i256,
    i32,
    i64,
    i8,
    int,
    u128,
    u16,
    u256,
    u32,
    u64,
    u8,
    uint,
)
from numojo.core.error import NumojoError
