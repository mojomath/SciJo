# ===----------------------------------------------------------------------=== #
# SciJo: Error function for scijo.special
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Error Function (`scijo.special.erf`)
=======================================
`erf(x)` and `erfc(x) = 1 - erf(x)`, via a convergent power series for
`|x| < 2` and a continued fraction for `|x| >= 2` (the series converges for
every `x`, but needs more terms than is practical once `x^2` outgrows the
term count; the continued fraction is the standard asymptotic form for the
tail and avoids the cancellation `1 - erf(x)` would hit there).

References
----------
- Abramowitz & Stegun, "Handbook of Mathematical Functions", 7.1.14.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.math import (
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
# Error function
# ===----------------------------------------------------------------------=== #


def _erf_series[T: NumericComparable](x: T) -> T:
    """`erf(x)` for `|x| < 2`, via the everywhere-positive-term series
    `erf(x) = (2x/sqrt(pi)) * exp(-x^2) * sum_n (2x^2)^n / (1*3*5*...*(2n+1))`.
    """
    var x2 = x * x
    var term = T.one()
    var s = T.one()
    for n in range(1, 200):
        term = term * (2.0 * x2 / (2.0 * Float64(n) + 1.0))
        s = s + term
        if term < 1e-17 * s:
            break
    return (2.0 * x / sqrt(pi)) * (-x2).exp() * s


def _erfc_cf[T: NumericComparable](x: T) -> T:
    """`erfc(x)` for `x >= 2`, via the continued fraction
    `erfc(x) = exp(-x^2)/sqrt(pi) / (x + 1/(2x + 2/(x + 3/(2x + ...))))`.
    """
    var f = T.constant(0.0)
    for k in range(60, 0, -1):
        var b = 2.0 * x if k % 2 == 1 else x
        f = Float64(k) / (b + f)
    var d = x + f
    return (-x * x).exp() / (sqrt(pi) * d)


def erf[T: NumericComparable](x: T) -> T:
    """The error function, `erf(x) = (2/sqrt(pi)) * integral_0^x e^(-t^2) dt`.

    Parameters:
        T: A `NumericComparable` conformer (`Primal`, `Dual`, ...).

    Args:
        x: The point at which to evaluate `erf`.

    Returns:
        `erf(x)`, in `(-1, 1)`.

    Examples:
        ```mojo
        from scijo.special import erf
        from scijo.prelude import *

        var y = erf(Scalar[f64](1.0))  # 0.8427007929497149...
        ```
    """
    if x < 0.0:
        return -erf(-x)
    if x < 2.0:
        return _erf_series(x)
    return 1.0 - _erfc_cf(x)


def erf[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    """The error function, on a plain `Scalar`.

    Parameters:
        dtype: The floating-point data type.

    Args:
        x: The point at which to evaluate `erf`.

    Returns:
        `erf(x)`, in `(-1, 1)`.
    """
    return erf(Primal[dtype](x)).v[0]


def erfc[T: NumericComparable](x: T) -> T:
    """The complementary error function, `erfc(x) = 1 - erf(x)`, computed
    directly rather than by subtraction so it stays accurate for large `x`,
    where `erf(x)` is too close to 1 for the subtraction to resolve it.

    Parameters:
        T: A `NumericComparable` conformer (`Primal`, `Dual`, ...).

    Args:
        x: The point at which to evaluate `erfc`.

    Returns:
        `erfc(x)`, in `(0, 2)`.

    Examples:
        ```mojo
        from scijo.special import erfc
        from scijo.prelude import *

        var y = erfc(Scalar[f64](5.0))  # 1.5374597944280347e-12
        ```
    """
    if x < 0.0:
        return 2.0 - erfc(-x)
    if x < 2.0:
        return 1.0 - _erf_series(x)
    return _erfc_cf(x)


def erfc[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    """The complementary error function, on a plain `Scalar`.

    Parameters:
        dtype: The floating-point data type.

    Args:
        x: The point at which to evaluate `erfc`.

    Returns:
        `erfc(x)`, in `(0, 2)`.
    """
    return erfc(Primal[dtype](x)).v[0]
