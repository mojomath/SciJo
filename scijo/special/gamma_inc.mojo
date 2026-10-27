# ===----------------------------------------------------------------------=== #
# SciJo: Regularized incomplete gamma function for scijo.special
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Regularized Incomplete Gamma Function (`scijo.special.gamma_inc`)
====================================================================
`gamma_inc(a, x) = P(a, x)` (the regularized *lower* incomplete gamma
function) and `gamma_incc(a, x) = Q(a, x) = 1 - P(a, x)` (the regularized
*upper* incomplete gamma function), via a power series for `x < a + 1` and a
continued fraction for `x >= a + 1` (the series converges everywhere, but
loses accuracy once `x` outgrows `a`; the continued fraction is the
standard well-conditioned form for that tail). Domain: `a > 0`, `x >= 0`.
"""

# ===----------------------------------------------------------------------=== #
# SciJo
# ===----------------------------------------------------------------------=== #
from scijo.special.gamma import lngamma
from scijo.types import (
    NumericComparable,
    Primal,
)

# ===----------------------------------------------------------------------=== #
# Regularized incomplete gamma function
# ===----------------------------------------------------------------------=== #
comptime _GAMMA_INC_MAXIT: Int = 200
comptime _GAMMA_INC_EPS: Float64 = 3e-16
comptime _GAMMA_INC_FPMIN: Float64 = 1e-300


def _gamma_inc_series[T: NumericComparable](a: T, x: T) -> T:
    """`P(a, x)` for `x < a + 1`, via the series
    `P(a,x) = x^a*exp(-x)/Gamma(a) * sum_n x^n / (a*(a+1)*...*(a+n))`.
    """
    var ap = a
    var term = 1.0 / a
    var summ = term
    for _ in range(1, _GAMMA_INC_MAXIT):
        ap = ap + 1.0
        term = term * (x / ap)
        summ = summ + term
        if term.abs() < summ.abs() * _GAMMA_INC_EPS:
            break
    return summ * (a * x.log() - x - lngamma(a)).exp()


def _gamma_inc_cf[T: NumericComparable](a: T, x: T) -> T:
    """`Q(a, x)` for `x >= a + 1`, via the modified Lentz continued fraction
    `Q(a,x) = x^a*exp(-x)/Gamma(a) / (x+1-a - 1*(1-a)/(x+3-a - 2*(2-a)/(x+5-a - ...)))`.
    """
    var b = x - a + 1.0
    var c = T.constant(1.0 / _GAMMA_INC_FPMIN)
    var d = 1.0 / b
    var h = d
    for i in range(1, _GAMMA_INC_MAXIT):
        var fi = Float64(i)
        var an = (a - fi) * fi
        b = b + 2.0
        d = an * d + b
        if d.abs() < _GAMMA_INC_FPMIN:
            d = T.constant(_GAMMA_INC_FPMIN)
        c = b + an / c
        if c.abs() < _GAMMA_INC_FPMIN:
            c = T.constant(_GAMMA_INC_FPMIN)
        d = 1.0 / d
        var del_ = d * c
        h = h * del_
        if (del_ - 1.0).abs() < _GAMMA_INC_EPS:
            break
    return (a * x.log() - x - lngamma(a)).exp() * h


def gamma_inc[T: NumericComparable](a: T, x: T) -> T:
    """The regularized lower incomplete gamma function, `P(a, x)`.

    Parameters:
        T: A `NumericComparable` conformer (`Primal`, `Dual`, ...).

    Args:
        a: The shape parameter. Must be positive.
        x: The upper integration limit. Must be non-negative.

    Returns:
        `P(a, x)`, in `[0, 1]`.

    Examples:
        ```mojo
        from scijo.special import gamma_inc
        from scijo.prelude import *

        var y = gamma_inc(Scalar[f64](2.0), Scalar[f64](1.0))
        ```
    """
    if x <= 0.0:
        return T.constant(0.0)
    if x < a + 1.0:
        return _gamma_inc_series(a, x)
    return 1.0 - _gamma_inc_cf(a, x)


def gamma_inc[
    dtype: DType
](a: Scalar[dtype], x: Scalar[dtype]) -> Scalar[
    dtype
] where dtype.is_floating_point():
    """The regularized lower incomplete gamma function, on plain `Scalar`s.

    Parameters:
        dtype: The floating-point data type.

    Args:
        a: The shape parameter. Must be positive.
        x: The upper integration limit. Must be non-negative.

    Returns:
        `P(a, x)`, in `[0, 1]`.
    """
    return gamma_inc(Primal[dtype](a), Primal[dtype](x)).v[0]


def gamma_incc[T: NumericComparable](a: T, x: T) -> T:
    """The regularized upper incomplete gamma function,
    `Q(a, x) = 1 - P(a, x)`, computed directly rather than by subtraction
    so it stays accurate where `P(a, x)` is too close to 1 to resolve it.

    Parameters:
        T: A `NumericComparable` conformer (`Primal`, `Dual`, ...).

    Args:
        a: The shape parameter. Must be positive.
        x: The upper integration limit. Must be non-negative.

    Returns:
        `Q(a, x)`, in `[0, 1]`.

    Examples:
        ```mojo
        from scijo.special import gamma_incc
        from scijo.prelude import *

        var y = gamma_incc(Scalar[f64](2.0), Scalar[f64](1.0))
        ```
    """
    if x <= 0.0:
        return T.constant(1.0)
    if x < a + 1.0:
        return 1.0 - _gamma_inc_series(a, x)
    return _gamma_inc_cf(a, x)


def gamma_incc[
    dtype: DType
](a: Scalar[dtype], x: Scalar[dtype]) -> Scalar[
    dtype
] where dtype.is_floating_point():
    """The regularized upper incomplete gamma function, on plain `Scalar`s.

    Parameters:
        dtype: The floating-point data type.

    Args:
        a: The shape parameter. Must be positive.
        x: The upper integration limit. Must be non-negative.

    Returns:
        `Q(a, x)`, in `[0, 1]`.
    """
    return gamma_incc(Primal[dtype](a), Primal[dtype](x)).v[0]
