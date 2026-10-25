# ===----------------------------------------------------------------------=== #
# SciJo: Gamma function for scijo.special
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Gamma Function (`scijo.special.gamma`)
=========================================
`gamma(x) = Gamma(x)` and `lngamma(x) = ln(|Gamma(x)|)`, via the Lanczos
approximation (`g = 7`, `n = 9` terms) for `x >= 0.5`, and the reflection
formula `Gamma(x)*Gamma(1-x) = pi/sin(pi*x)` for `x < 0.5`.

References
----------
- Lanczos, C. (1964). "A Precision Approximation of the Gamma Function."
- https://en.wikipedia.org/wiki/Lanczos_approximation
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.math import (
    log,
    pi,
    sqrt,
)

# ===----------------------------------------------------------------------=== #
# SciJo
# ===----------------------------------------------------------------------=== #
from scijo.types import (
    NumericComparable,
    Primal,
)

# ===----------------------------------------------------------------------=== #
# Lanczos coefficients, g=7, n=9: the standard public-domain set (see
# Numerical Recipes / Wikipedia's Lanczos-approximation article).
# ===----------------------------------------------------------------------=== #
comptime _LANCZOS_G: Float64 = 7.0
comptime _LANCZOS_C0: Float64 = 0.99999999999980993
comptime _LANCZOS_C1: Float64 = 676.5203681218851
comptime _LANCZOS_C2: Float64 = -1259.1392167224028
comptime _LANCZOS_C3: Float64 = 771.32342877765313
comptime _LANCZOS_C4: Float64 = -176.61502916214059
comptime _LANCZOS_C5: Float64 = 12.507343278686905
comptime _LANCZOS_C6: Float64 = -0.13857109526572012
comptime _LANCZOS_C7: Float64 = 9.9843695780195716e-6
comptime _LANCZOS_C8: Float64 = 1.5056327351493116e-7


def _lanczos_sum_and_base[
    T: NumericComparable
](x: T) -> Tuple[T, T]:
    """Shared Lanczos series: returns `(a, t)` for `x >= 0.5`, where
    `Gamma(x) = sqrt(2*pi) * t^(x-0.5) * exp(-t) * a` (with `x` already
    shifted down by 1, as the Lanczos formula expects)."""
    var xm1 = x - 1.0
    var a = T.constant(_LANCZOS_C0)
    a = a + _LANCZOS_C1 / (xm1 + 1.0)
    a = a + _LANCZOS_C2 / (xm1 + 2.0)
    a = a + _LANCZOS_C3 / (xm1 + 3.0)
    a = a + _LANCZOS_C4 / (xm1 + 4.0)
    a = a + _LANCZOS_C5 / (xm1 + 5.0)
    a = a + _LANCZOS_C6 / (xm1 + 6.0)
    a = a + _LANCZOS_C7 / (xm1 + 7.0)
    a = a + _LANCZOS_C8 / (xm1 + 8.0)
    var t = xm1 + (_LANCZOS_G + 0.5)
    return (a^, t^)


def gamma[T: NumericComparable](x: T) -> T:
    """The gamma function, `Gamma(x)`.

    Parameters:
        T: A `NumericComparable` conformer (`Primal`, `Dual`, ...).

    Args:
        x: The point at which to evaluate `Gamma`.

    Returns:
        `Gamma(x)`. Diverges at non-positive integers.

    Examples:
        ```mojo
        from scijo.special import gamma
        from scijo.prelude import *

        var y = gamma(5.0)  # 24.0 == 4!
        ```
    """
    if x < 0.5:
        return pi / ((pi * x).sin() * gamma(1.0 - x))
    var a: T
    var t: T
    a, t = _lanczos_sum_and_base(x)
    var half = x - 0.5
    return sqrt(2.0 * pi) * (half * t.log()).exp() * (-t).exp() * a


def gamma[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    """The gamma function, `Gamma(x)`, on a plain `Scalar`.

    Parameters:
        dtype: The floating-point data type.

    Args:
        x: The point at which to evaluate `Gamma`.

    Returns:
        `Gamma(x)`. Diverges at non-positive integers.
    """
    return gamma(Primal[dtype](x)).v[0]


def lngamma[T: NumericComparable](x: T) -> T:
    """The log of the absolute gamma function, `ln(|Gamma(x)|)`.

    Avoids the overflow `gamma(x)` hits for large `x` (`Gamma(171)`
    already overflows `f64`).

    Parameters:
        T: A `NumericComparable` conformer (`Primal`, `Dual`, ...).

    Args:
        x: The point at which to evaluate `ln(|Gamma|)`.

    Returns:
        `ln(|Gamma(x)|)`.

    Examples:
        ```mojo
        from scijo.special import lngamma
        from scijo.prelude import *

        var y = lngamma(171.0)  # finite, unlike gamma(171)
        ```
    """
    if x < 0.5:
        return log(pi) - (pi * x).sin().abs().log() - lngamma(1.0 - x)
    var a: T
    var t: T
    a, t = _lanczos_sum_and_base(x)
    var half = x - 0.5
    return 0.5 * log(2.0 * pi) + half * t.log() - t + a.log()


def lngamma[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    """The log of the absolute gamma function, `ln(|Gamma(x)|)`, on a plain
    `Scalar`.

    Parameters:
        dtype: The floating-point data type.

    Args:
        x: The point at which to evaluate `ln(|Gamma|)`.

    Returns:
        `ln(|Gamma(x)|)`.
    """
    return lngamma(Primal[dtype](x)).v[0]
