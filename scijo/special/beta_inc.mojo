# ===----------------------------------------------------------------------=== #
# SciJo: Regularized incomplete beta function for scijo.special
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Regularized Incomplete Beta Function (`scijo.special.beta_inc`)
==================================================================
`beta_inc(a, b, x) = I_x(a, b)`, via Lentz's modified continued fraction
(the standard well-conditioned form, convergent for
`x < (a+1)/(a+b+2)`; swapped to its complement `1 - I_{1-x}(b, a)` above
that threshold, where the same continued fraction converges quickly
instead). Domain: `a > 0`, `b > 0`, `0 <= x <= 1`.
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
# Regularized incomplete beta function
# ===----------------------------------------------------------------------=== #
comptime _BETA_INC_MAXIT: Int = 200
comptime _BETA_INC_EPS: Float64 = 3e-16
comptime _BETA_INC_FPMIN: Float64 = 1e-300


def _beta_inc_cf[T: NumericComparable](a: T, b: T, x: T) -> T:
    """Lentz's continued fraction for `I_x(a, b) / (x^a*(1-x)^b/(a*B(a,b)))`,
    valid for `x < (a+1)/(a+b+2)`."""
    var qab = a + b
    var qap = a + 1.0
    var qam = a - 1.0
    var c = T.constant(1.0)
    var d = 1.0 - qab * x / qap
    if d.abs() < _BETA_INC_FPMIN:
        d = T.constant(_BETA_INC_FPMIN)
    d = 1.0 / d
    var h = d
    for m in range(1, _BETA_INC_MAXIT):
        var fm = Float64(m)
        var m2 = 2.0 * fm
        var aa = fm * (b - fm) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if d.abs() < _BETA_INC_FPMIN:
            d = T.constant(_BETA_INC_FPMIN)
        c = 1.0 + aa / c
        if c.abs() < _BETA_INC_FPMIN:
            c = T.constant(_BETA_INC_FPMIN)
        d = 1.0 / d
        h = h * (d * c)
        aa = -(a + fm) * (qab + fm) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if d.abs() < _BETA_INC_FPMIN:
            d = T.constant(_BETA_INC_FPMIN)
        c = 1.0 + aa / c
        if c.abs() < _BETA_INC_FPMIN:
            c = T.constant(_BETA_INC_FPMIN)
        d = 1.0 / d
        var del_ = d * c
        h = h * del_
        if (del_ - 1.0).abs() < _BETA_INC_EPS:
            break
    return h


def beta_inc[T: NumericComparable](a: T, b: T, x: T) -> T:
    """The regularized incomplete beta function, `I_x(a, b)`.

    Parameters:
        T: A `NumericComparable` conformer (`Primal`, `Dual`, ...).

    Args:
        a: The first shape parameter. Must be positive.
        b: The second shape parameter. Must be positive.
        x: The point at which to evaluate `I_x`. Must be in `[0, 1]`.

    Returns:
        `I_x(a, b)`, in `[0, 1]`.

    Examples:
        ```mojo
        from scijo.special import beta_inc
        from scijo.prelude import *

        var y = beta_inc(Scalar[f64](2.0), Scalar[f64](3.0), Scalar[f64](0.4))
        ```
    """
    if x <= 0.0:
        return T.constant(0.0)
    if not (x < 1.0):
        return T.constant(1.0)
    var bt = (
        lngamma(a + b)
        - lngamma(a)
        - lngamma(b)
        + a * x.log()
        + b * (1.0 - x).log()
    ).exp()
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _beta_inc_cf(a, b, x) / a
    return 1.0 - bt * _beta_inc_cf(b, a, 1.0 - x) / b


def beta_inc[
    dtype: DType
](a: Scalar[dtype], b: Scalar[dtype], x: Scalar[dtype]) -> Scalar[
    dtype
] where dtype.is_floating_point():
    """The regularized incomplete beta function, on plain `Scalar`s.

    Parameters:
        dtype: The floating-point data type.

    Args:
        a: The first shape parameter. Must be positive.
        b: The second shape parameter. Must be positive.
        x: The point at which to evaluate `I_x`. Must be in `[0, 1]`.

    Returns:
        `I_x(a, b)`, in `[0, 1]`.
    """
    return beta_inc(Primal[dtype](a), Primal[dtype](b), Primal[dtype](x)).v[0]
