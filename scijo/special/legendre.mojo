# ===----------------------------------------------------------------------=== #
# SciJo: Legendre polynomials for scijo.special
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Legendre Polynomials (`scijo.special.legendre`)
==================================================
`legendre(n, x) = P_n(x)`, the degree-`n` Legendre polynomial, via Bonnet's
recurrence `(k+1)*P_{k+1}(x) = (2k+1)*x*P_k(x) - k*P_{k-1}(x)`, starting
from `P_0 = 1`, `P_1 = x`. Forward recurrence is exact and stable for
`P_n` at every `x` (no minimal/dominant-solution split like the Bessel
functions need), so no series/asymptotic branch is needed. Domain:
`n >= 0`; conventionally evaluated on `-1 <= x <= 1`, but the recurrence is
just a polynomial and is exact for any `x`.

References
----------
- Abramowitz & Stegun, "Handbook of Mathematical Functions", 8.5.3.
"""

# ===----------------------------------------------------------------------=== #
# SciJo
# ===----------------------------------------------------------------------=== #
from scijo.types import (
    NumericComparable,
    Primal,
)

# ===----------------------------------------------------------------------=== #
# Legendre polynomials
# ===----------------------------------------------------------------------=== #


def legendre[T: NumericComparable](n: Int, x: T) -> T:
    """The degree-`n` Legendre polynomial, `P_n(x)`.

    Parameters:
        T: A `NumericComparable` conformer (`Primal`, `Dual`, ...).

    Args:
        n: The degree. Must be non-negative.
        x: The point at which to evaluate `P_n`.

    Returns:
        `P_n(x)`.

    Examples:
        ```mojo
        from scijo.special import legendre
        from scijo.prelude import *

        var y = legendre(2, Scalar[f64](0.5))  # (3*0.5^2 - 1)/2 == -0.125
        ```
    """
    if n == 0:
        return T.one()
    if n == 1:
        return x

    var p0 = T.one()
    var p1 = x
    for k in range(1, n):
        var p2 = (
            Float64(2 * k + 1) * x * p1 - Float64(k) * p0
        ) / Float64(k + 1)
        p0 = p1
        p1 = p2
    return p1


def legendre[
    dtype: DType
](n: Int, x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    """The degree-`n` Legendre polynomial, on a plain `Scalar`.

    Parameters:
        dtype: The floating-point data type.

    Args:
        n: The degree. Must be non-negative.
        x: The point at which to evaluate `P_n`.

    Returns:
        `P_n(x)`.
    """
    return legendre(n, Primal[dtype](x)).v[0]
