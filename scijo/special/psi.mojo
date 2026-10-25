# ===----------------------------------------------------------------------=== #
# SciJo: Digamma function for scijo.special
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Digamma Function (`scijo.special.psi`)
=========================================
`digamma(x) = d/dx[ln(Gamma(x))]`, via the recurrence
`psi(x) = psi(x+1) - 1/x` to shift into the asymptotic-series region
(`x >= 6`), with the reflection formula
`psi(1-x) - psi(x) = pi*cot(pi*x)` for `x <= 0`.

References
----------
- Abramowitz & Stegun, "Handbook of Mathematical Functions", 6.3.5, 6.3.18.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.math import pi
from std.utils.numerics import nan

# ===----------------------------------------------------------------------=== #
# SciJo
# ===----------------------------------------------------------------------=== #
from scijo.types import (
    NumericComparable,
    Primal,
)

# ===----------------------------------------------------------------------=== #
# Digamma
# ===----------------------------------------------------------------------=== #


def digamma[T: NumericComparable](x: T) -> T:
    """The digamma function, `psi(x) = d/dx[ln(Gamma(x))]`.

    Parameters:
        T: A `NumericComparable` conformer (`Primal`, `Dual`, ...).

    Args:
        x: The point at which to evaluate `psi`.

    Returns:
        `psi(x)`. Returns `nan` at the poles (non-positive integers).

    Examples:
        ```mojo
        from scijo.special import digamma
        from scijo.prelude import *

        var y = digamma(Scalar[f64](1.0))  # -0.5772156649... (-gamma_EM)
        ```
    """
    if x <= 0.0:
        if x == x.floor():
            return T.constant(nan[DType.float64]())
        return digamma(1.0 - x) - pi * (pi * x).cos() / (pi * x).sin()

    var y = x
    var result = T.constant(0.0)
    while y < 6.0:
        result = result - 1.0 / y
        y = y + 1.0

    # Asymptotic series (A&S 6.3.18) in powers of 1/y, Bernoulli-number
    # coefficients B2=1/6, B4=-1/30, B6=1/42, B8=-1/30.
    var inv = 1.0 / y
    var inv2 = inv * inv
    result = result + y.log() - 0.5 * inv - inv2 * (
        1.0 / 12.0
        - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0 - inv2 * (1.0 / 240.0)))
    )
    return result


def digamma[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    """The digamma function, on a plain `Scalar`.

    Parameters:
        dtype: The floating-point data type.

    Args:
        x: The point at which to evaluate `psi`.

    Returns:
        `psi(x)`. Returns `nan` at the poles (non-positive integers).
    """
    return digamma(Primal[dtype](x)).v[0]
