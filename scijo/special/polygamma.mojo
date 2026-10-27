# ===----------------------------------------------------------------------=== #
# SciJo: Polygamma function for scijo.special
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Polygamma Function (`scijo.special.polygamma`)
=================================================
`polygamma(m, x) = psi^(m)(x)`, the `m`-th derivative of `digamma`. `m == 0`
is `digamma` itself; `m >= 1` uses the recurrence
`psi^(m)(x) = (-1)^(m+1)*m!/x^(m+1) + psi^(m)(x+1)` to shift into the
asymptotic-series region (`x >= 15`), then the asymptotic series in powers
of `1/x` with even Bernoulli-number coefficients. Domain: `x > 0`.

References
----------
- Abramowitz & Stegun, "Handbook of Mathematical Functions", 6.4.10.
"""

# ===----------------------------------------------------------------------=== #
# SciJo
# ===----------------------------------------------------------------------=== #
from scijo.special.psi import digamma
from scijo.types import (
    NumericComparable,
    Primal,
)

# ===----------------------------------------------------------------------=== #
# Polygamma
# ===----------------------------------------------------------------------=== #
comptime _POLYGAMMA_SHIFT: Float64 = 15.0
comptime _B2: Float64 = 1.0 / 6.0
comptime _B4: Float64 = -1.0 / 30.0
comptime _B6: Float64 = 1.0 / 42.0
comptime _B8: Float64 = -1.0 / 30.0
comptime _B10: Float64 = 5.0 / 66.0
comptime _B12: Float64 = -691.0 / 2730.0


def _factorial(n: Int) -> Float64:
    """`n!`, for the small non-negative `n` this module ever calls it
    with."""
    var result = 1.0
    for i in range(2, n + 1):
        result = result * Float64(i)
    return result


def _rising_product(start: Int, count: Int) -> Float64:
    """`(start+1) * (start+2) * ... * (start+count)`, the empty product
    (`count == 0`) being `1`."""
    var result = 1.0
    for j in range(1, count + 1):
        result = result * Float64(start + j)
    return result


def _pow_int[T: NumericComparable](x: T, n: Int) -> T:
    """`x^n`, for the small non-negative `n` this module ever calls it
    with."""
    var result = T.one()
    for _ in range(n):
        result = result * x
    return result


def _polygamma_asymptotic[T: NumericComparable](m: Int, x: T) -> T:
    """`psi^(m)(x)` for large `x`, via A&S 6.4.10:
    `psi^(m)(x) ~ (-1)^(m+1) * [(m-1)!/x^m + m!/(2x^(m+1))
    + sum_k B_2k * (2k+m-1)!/(2k)! / x^(2k+m)]`.
    """
    var sign = -1.0 if m % 2 == 0 else 1.0
    var inv_xm = 1.0 / _pow_int(x, m)
    var summ = _factorial(m - 1) * inv_xm + (_factorial(m) / 2.0) * (inv_xm / x)

    var inv_x2 = 1.0 / (x * x)
    var running = inv_xm * inv_x2
    for k in range(1, 7):
        var bern: Float64
        if k == 1:
            bern = _B2
        elif k == 2:
            bern = _B4
        elif k == 3:
            bern = _B6
        elif k == 4:
            bern = _B8
        elif k == 5:
            bern = _B10
        else:
            bern = _B12
        var coeff = bern * _rising_product(2 * k, m - 1)
        summ = summ + coeff * running
        running = running * inv_x2
    return sign * summ


def polygamma[T: NumericComparable](m: Int, x: T) -> T:
    """The polygamma function, `psi^(m)(x)`, the `m`-th derivative of
    `digamma`.

    Parameters:
        T: A `NumericComparable` conformer (`Primal`, `Dual`, ...).

    Args:
        m: The derivative order. `m == 0` is `digamma` itself.
        x: The point at which to evaluate `psi^(m)`. Must be positive.

    Returns:
        `psi^(m)(x)`.

    Examples:
        ```mojo
        from scijo.special import polygamma
        from scijo.prelude import *

        var y = polygamma(1, Scalar[f64](1.0))  # pi^2/6 (trigamma(1))
        ```
    """
    if m == 0:
        return digamma(x)

    var sign = -1.0 if m % 2 == 0 else 1.0
    var fact_m = _factorial(m)
    var y = x
    var correction = T.constant(0.0)
    while y < _POLYGAMMA_SHIFT:
        correction = correction + sign * fact_m / _pow_int(y, m + 1)
        y = y + 1.0
    return correction + _polygamma_asymptotic(m, y)


def polygamma[
    dtype: DType
](m: Int, x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    """The polygamma function, on a plain `Scalar`.

    Parameters:
        dtype: The floating-point data type.

    Args:
        m: The derivative order. `m == 0` is `digamma` itself.
        x: The point at which to evaluate `psi^(m)`. Must be positive.

    Returns:
        `psi^(m)(x)`.
    """
    return polygamma(m, Primal[dtype](x)).v[0]
