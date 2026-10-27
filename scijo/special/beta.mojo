# ===----------------------------------------------------------------------=== #
# SciJo: Beta function for scijo.special
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Beta Function (`scijo.special.beta`)
=======================================
`beta(a, b) = Gamma(a)*Gamma(b)/Gamma(a+b)`, built on `lngamma` so it stays
accurate (and finite) for arguments where `Gamma` itself would overflow.
Domain: `a > 0` and `b > 0`.
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
# Beta function
# ===----------------------------------------------------------------------=== #


def lnbeta[T: NumericComparable](a: T, b: T) -> T:
    """The log of the beta function, `ln(B(a, b))`.

    Parameters:
        T: A `NumericComparable` conformer (`Primal`, `Dual`, ...).

    Args:
        a: The first shape parameter. Must be positive.
        b: The second shape parameter. Must be positive.

    Returns:
        `ln(B(a, b))`.

    Examples:
        ```mojo
        from scijo.special import lnbeta
        from scijo.prelude import *

        var y = lnbeta(Scalar[f64](2.0), Scalar[f64](3.0))
        ```
    """
    return lngamma(a) + lngamma(b) - lngamma(a + b)


def lnbeta[
    dtype: DType
](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[
    dtype
] where dtype.is_floating_point():
    """The log of the beta function, on plain `Scalar`s.

    Parameters:
        dtype: The floating-point data type.

    Args:
        a: The first shape parameter. Must be positive.
        b: The second shape parameter. Must be positive.

    Returns:
        `ln(B(a, b))`.
    """
    return lnbeta(Primal[dtype](a), Primal[dtype](b)).v[0]


def beta[T: NumericComparable](a: T, b: T) -> T:
    """The beta function, `B(a, b) = Gamma(a)*Gamma(b)/Gamma(a+b)`.

    Parameters:
        T: A `NumericComparable` conformer (`Primal`, `Dual`, ...).

    Args:
        a: The first shape parameter. Must be positive.
        b: The second shape parameter. Must be positive.

    Returns:
        `B(a, b)`.

    Examples:
        ```mojo
        from scijo.special import beta
        from scijo.prelude import *

        var y = beta(Scalar[f64](2.0), Scalar[f64](3.0))  # 1/12
        ```
    """
    return lnbeta(a, b).exp()


def beta[
    dtype: DType
](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[
    dtype
] where dtype.is_floating_point():
    """The beta function, on plain `Scalar`s.

    Parameters:
        dtype: The floating-point data type.

    Args:
        a: The first shape parameter. Must be positive.
        b: The second shape parameter. Must be positive.

    Returns:
        `B(a, b)`.
    """
    return beta(Primal[dtype](a), Primal[dtype](b)).v[0]
