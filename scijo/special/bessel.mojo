# ===----------------------------------------------------------------------=== #
# SciJo: Bessel functions for scijo.special
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Bessel Functions (`scijo.special.bessel`)
============================================
Order-0 and order-1 Bessel functions of the first kind (`J0`, `J1`), second
kind (`Y0`, `Y1`), and the modified first kind (`I0`, `I1`) and second kind
(`K0`, `K1`). Each uses a convergent power series for small `|x|` and the
shared asymptotic expansion (DLMF 10.17, 10.40) for large `|x|`, where the
series either converges too slowly (`J`, `Y`, `I`) or loses precision to
cancellation (`K`).

References
----------
- DLMF (https://dlmf.nist.gov/), 10.8 (power series), 10.17 and 10.40
  (asymptotic expansions).
- Abramowitz & Stegun, "Handbook of Mathematical Functions", 9.1, 9.6-9.8.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.math import pi

# ===----------------------------------------------------------------------=== #
# SciJo
# ===----------------------------------------------------------------------=== #
from scijo.types import (
    NumericComparable,
    Primal,
)

comptime _EULER_GAMMA: Float64 = 0.5772156649015329

# ===----------------------------------------------------------------------=== #
# Shared asymptotic expansion (DLMF 10.17.3-10.17.4, 10.40.1-10.40.2): both
# families share the same coefficient recurrence
# `a_k(nu)/x^k = a_{k-1}(nu)/x^{k-1} * (4*nu^2 - (2k-1)^2) / (8*k*x)`.
# ===----------------------------------------------------------------------=== #


def _bessel_ak_terms[T: NumericComparable](mu: Float64, x: T) -> List[T]:
    var terms = List[T](capacity=9)
    var a = T.one()
    terms.append(a)
    for k in range(1, 9):
        a = a * (
            (mu - Float64((2 * k - 1) * (2 * k - 1))) / (Float64(k) * 8.0 * x)
        )
        terms.append(a)
    return terms^


def _bessel_jy_asym[
    T: NumericComparable
](nu: Float64, mu: Float64, x: T) -> Tuple[T, T]:
    var terms = _bessel_ak_terms(mu, x)
    var p = terms[0] - terms[2] + terms[4] - terms[6] + terms[8]
    var q = terms[1] - terms[3] + terms[5] - terms[7]
    var omega = x - (nu * pi / 2.0 + pi / 4.0)
    var norm = (2.0 / (pi * x)).sqrt()
    return (
        norm * (p * omega.cos() - q * omega.sin()),
        norm * (p * omega.sin() + q * omega.cos()),
    )


def _bessel_ik_asym[T: NumericComparable](mu: Float64, x: T) -> Tuple[T, T]:
    var terms = _bessel_ak_terms(mu, x)
    var i_sum = (
        terms[0]
        - terms[1]
        + terms[2]
        - terms[3]
        + terms[4]
        - terms[5]
        + terms[6]
        - terms[7]
        + terms[8]
    )
    var k_sum = (
        terms[0]
        + terms[1]
        + terms[2]
        + terms[3]
        + terms[4]
        + terms[5]
        + terms[6]
        + terms[7]
        + terms[8]
    )
    return (
        x.exp() / (2.0 * pi * x).sqrt() * i_sum,
        (pi / (2.0 * x)).sqrt() * (-x).exp() * k_sum,
    )


# ===----------------------------------------------------------------------=== #
# J0, J1 (first kind), Y0, Y1 (second kind)
# ===----------------------------------------------------------------------=== #


def _j0_series[T: NumericComparable](x: T) -> T:
    var x2 = (x / 2.0) * (x / 2.0)
    var term = T.one()
    var s = T.one()
    for k in range(1, 300):
        term = term * (-x2 / Float64(k * k))
        s = s + term
        if k > 5 and term < 1e-18 * s and term > -1e-18 * s:
            break
    return s


def _j1_series_terms[T: NumericComparable](x: T) -> List[T]:
    """`J1`'s series terms, kept (not just summed) because `Y1`'s series
    reuses them weighted by harmonic numbers."""
    var x2 = (x / 2.0) * (x / 2.0)
    var term = x / 2.0
    var terms = List[T](capacity=64)
    terms.append(term)
    var s = term
    for k in range(1, 300):
        term = term * (-x2 / Float64(k * (k + 1)))
        terms.append(term)
        s = s + term
        if k > 5 and term < 1e-18 * s and term > -1e-18 * s:
            break
    return terms^


def bessel_j0[T: NumericComparable](x: T) -> T:
    """The order-0 Bessel function of the first kind, `J0(x)`.

    Parameters:
        T: A `NumericComparable` conformer (`Primal`, `Dual`, ...).

    Args:
        x: The point at which to evaluate `J0`.

    Returns:
        `J0(x)`.
    """
    if x < 0.0:
        return bessel_j0(-x)
    if x < 15.0:
        return _j0_series(x)
    var j: T
    j, _ = _bessel_jy_asym(0.0, 0.0, x)
    return j


def bessel_j0[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    """The order-0 Bessel function of the first kind, on a plain `Scalar`.

    Parameters:
        dtype: The floating-point data type.

    Args:
        x: The point at which to evaluate `J0`.

    Returns:
        `J0(x)`.
    """
    return bessel_j0(Primal[dtype](x)).v[0]


def bessel_j1[T: NumericComparable](x: T) -> T:
    """The order-1 Bessel function of the first kind, `J1(x)`.

    Parameters:
        T: A `NumericComparable` conformer (`Primal`, `Dual`, ...).

    Args:
        x: The point at which to evaluate `J1`.

    Returns:
        `J1(x)`.
    """
    if x < 0.0:
        return -bessel_j1(-x)
    if x < 15.0:
        var s = T.constant(0.0)
        for term in _j1_series_terms(x):
            s = s + term
        return s
    var j: T
    j, _ = _bessel_jy_asym(1.0, 4.0, x)
    return j


def bessel_j1[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    """The order-1 Bessel function of the first kind, on a plain `Scalar`.

    Parameters:
        dtype: The floating-point data type.

    Args:
        x: The point at which to evaluate `J1`.

    Returns:
        `J1(x)`.
    """
    return bessel_j1(Primal[dtype](x)).v[0]


def bessel_y0[T: NumericComparable](x: T) -> T:
    """The order-0 Bessel function of the second kind, `Y0(x)`.

    Parameters:
        T: A `NumericComparable` conformer (`Primal`, `Dual`, ...).

    Args:
        x: The point at which to evaluate `Y0`. Must be positive.

    Returns:
        `Y0(x)`. Diverges as `x -> 0`.
    """
    if x < 15.0:
        var j0 = _j0_series(x)
        var h: Float64 = 0.0
        var s = T.constant(0.0)
        var term = T.one()
        var x2 = (x / 2.0) * (x / 2.0)
        for k in range(1, 100):
            term = term * (-x2 / Float64(k * k))
            h += 1.0 / Float64(k)
            s = s + term * h
        var log_term = (x / 2.0).log() + _EULER_GAMMA
        return (2.0 / pi) * (log_term * j0 - s)
    var y: T
    _, y = _bessel_jy_asym(0.0, 0.0, x)
    return y


def bessel_y0[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    """The order-0 Bessel function of the second kind, on a plain `Scalar`.

    Parameters:
        dtype: The floating-point data type.

    Args:
        x: The point at which to evaluate `Y0`. Must be positive.

    Returns:
        `Y0(x)`. Diverges as `x -> 0`.
    """
    return bessel_y0(Primal[dtype](x)).v[0]


def bessel_y1[T: NumericComparable](x: T) -> T:
    """The order-1 Bessel function of the second kind, `Y1(x)`.

    Parameters:
        T: A `NumericComparable` conformer (`Primal`, `Dual`, ...).

    Args:
        x: The point at which to evaluate `Y1`. Must be positive.

    Returns:
        `Y1(x)`. Diverges as `x -> 0`.
    """
    if x < 15.0:
        var j1_terms = _j1_series_terms(x)
        var j1 = T.constant(0.0)
        for term in j1_terms:
            j1 = j1 + term
        var h: Float64 = 0.0
        var s = T.constant(0.0)
        for k in range(len(j1_terms)):
            var h_k = h
            var h_k1 = h + 1.0 / Float64(k + 1)
            s = s + j1_terms[k] * (h_k + h_k1 - 2.0 * _EULER_GAMMA)
            h = h_k1
        return (2.0 / pi) * ((x / 2.0).log() * j1 - 1.0 / x) - (1.0 / pi) * s
    var y: T
    _, y = _bessel_jy_asym(1.0, 4.0, x)
    return y


def bessel_y1[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    """The order-1 Bessel function of the second kind, on a plain `Scalar`.

    Parameters:
        dtype: The floating-point data type.

    Args:
        x: The point at which to evaluate `Y1`. Must be positive.

    Returns:
        `Y1(x)`. Diverges as `x -> 0`.
    """
    return bessel_y1(Primal[dtype](x)).v[0]


# ===----------------------------------------------------------------------=== #
# I0, I1 (modified, first kind), K0, K1 (modified, second kind)
# ===----------------------------------------------------------------------=== #


def _i0_series[T: NumericComparable](x: T) -> T:
    var x2 = (x / 2.0) * (x / 2.0)
    var term = T.one()
    var s = T.one()
    for k in range(1, 300):
        term = term * (x2 / Float64(k * k))
        s = s + term
        if k > 5 and term < 1e-18 * s:
            break
    return s


def _i1_series_terms[T: NumericComparable](x: T) -> List[T]:
    var x2 = (x / 2.0) * (x / 2.0)
    var term = x / 2.0
    var terms = List[T](capacity=64)
    terms.append(term)
    var s = term
    for k in range(1, 300):
        term = term * (x2 / Float64(k * (k + 1)))
        terms.append(term)
        s = s + term
        if k > 5 and term < 1e-18 * s:
            break
    return terms^


def bessel_i0[T: NumericComparable](x: T) -> T:
    """The order-0 modified Bessel function of the first kind, `I0(x)`.

    Parameters:
        T: A `NumericComparable` conformer (`Primal`, `Dual`, ...).

    Args:
        x: The point at which to evaluate `I0`.

    Returns:
        `I0(x)`.
    """
    if x < 0.0:
        return bessel_i0(-x)
    if x < 15.0:
        return _i0_series(x)
    var i: T
    i, _ = _bessel_ik_asym(0.0, x)
    return i


def bessel_i0[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    """The order-0 modified Bessel function of the first kind, on a plain
    `Scalar`.

    Parameters:
        dtype: The floating-point data type.

    Args:
        x: The point at which to evaluate `I0`.

    Returns:
        `I0(x)`.
    """
    return bessel_i0(Primal[dtype](x)).v[0]


def bessel_i1[T: NumericComparable](x: T) -> T:
    """The order-1 modified Bessel function of the first kind, `I1(x)`.

    Parameters:
        T: A `NumericComparable` conformer (`Primal`, `Dual`, ...).

    Args:
        x: The point at which to evaluate `I1`.

    Returns:
        `I1(x)`.
    """
    if x < 0.0:
        return -bessel_i1(-x)
    if x < 15.0:
        var s = T.constant(0.0)
        for term in _i1_series_terms(x):
            s = s + term
        return s
    var i: T
    i, _ = _bessel_ik_asym(4.0, x)
    return i


def bessel_i1[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    """The order-1 modified Bessel function of the first kind, on a plain
    `Scalar`.

    Parameters:
        dtype: The floating-point data type.

    Args:
        x: The point at which to evaluate `I1`.

    Returns:
        `I1(x)`.
    """
    return bessel_i1(Primal[dtype](x)).v[0]


def bessel_k0[T: NumericComparable](x: T) -> T:
    """The order-0 modified Bessel function of the second kind, `K0(x)`.

    Parameters:
        T: A `NumericComparable` conformer (`Primal`, `Dual`, ...).

    Args:
        x: The point at which to evaluate `K0`. Must be positive.

    Returns:
        `K0(x)`. Diverges as `x -> 0`.
    """
    if x < 8.0:
        var i0 = _i0_series(x)
        var h: Float64 = 0.0
        var s = T.constant(0.0)
        var term = T.one()
        var x2 = (x / 2.0) * (x / 2.0)
        for k in range(1, 60):
            term = term * (x2 / Float64(k * k))
            h += 1.0 / Float64(k)
            s = s + term * h
        var log_term = (x / 2.0).log() + _EULER_GAMMA
        return (-log_term) * i0 + s
    var k: T
    _, k = _bessel_ik_asym(0.0, x)
    return k


def bessel_k0[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    """The order-0 modified Bessel function of the second kind, on a plain
    `Scalar`.

    Parameters:
        dtype: The floating-point data type.

    Args:
        x: The point at which to evaluate `K0`. Must be positive.

    Returns:
        `K0(x)`. Diverges as `x -> 0`.
    """
    return bessel_k0(Primal[dtype](x)).v[0]


def bessel_k1[T: NumericComparable](x: T) -> T:
    """The order-1 modified Bessel function of the second kind, `K1(x)`.

    Parameters:
        T: A `NumericComparable` conformer (`Primal`, `Dual`, ...).

    Args:
        x: The point at which to evaluate `K1`. Must be positive.

    Returns:
        `K1(x)`. Diverges as `x -> 0`.
    """
    if x < 8.0:
        var i1_terms = _i1_series_terms(x)
        var i1 = T.constant(0.0)
        for term in i1_terms:
            i1 = i1 + term
        var h: Float64 = 0.0
        var s = T.constant(0.0)
        for k in range(len(i1_terms)):
            var h_k = h
            var h_k1 = h + 1.0 / Float64(k + 1)
            s = s + i1_terms[k] * (h_k + h_k1)
            h = h_k1
        var log_term = (x / 2.0).log() + _EULER_GAMMA
        return log_term * i1 + 1.0 / x - 0.5 * s
    var k: T
    _, k = _bessel_ik_asym(4.0, x)
    return k


def bessel_k1[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    """The order-1 modified Bessel function of the second kind, on a plain
    `Scalar`.

    Parameters:
        dtype: The floating-point data type.

    Args:
        x: The point at which to evaluate `K1`. Must be positive.

    Returns:
        `K1(x)`. Diverges as `x -> 0`.
    """
    return bessel_k1(Primal[dtype](x)).v[0]
