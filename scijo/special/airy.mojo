# ===----------------------------------------------------------------------=== #
# SciJo: Airy functions for scijo.special
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Airy Functions (`scijo.special.airy`)
========================================
`airy_ai(x) = Ai(x)` and `airy_bi(x) = Bi(x)`, the two independent
solutions of `y'' = x*y`. Uses the convergent power series (A&S 10.4.2-
10.4.3) for `-9 <= x < 6`, and an asymptotic expansion (A&S 10.4.59,
10.4.63-10.4.66, in the decaying/growing-exponential form for `x >= 6` and
the oscillatory form for `x < -9`) elsewhere - the series itself converges
everywhere, but for `x >= 6` resolving `Ai`'s exponential decay out of `f`
and `g`'s exponentially large individual terms loses too much precision to
cancellation, and for `x <= -9` it needs more terms than stay numerically
stable at `f64`.

References
----------
- Abramowitz & Stegun, "Handbook of Mathematical Functions", 10.4.
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

comptime _AIRY_C1: Float64 = 0.35502805388781733
comptime _AIRY_C2: Float64 = 0.25881940379280677
comptime _AIRY_POS_THRESHOLD: Float64 = 6.0
comptime _AIRY_NEG_THRESHOLD: Float64 = -9.0

# ===----------------------------------------------------------------------=== #
# Asymptotic-series coefficients (A&S 10.4.59): `u_0 = 1`,
# `u_k = (2k+1)(2k+3)...(6k-1) / (216^k * k!)`, shared by every branch.
# ===----------------------------------------------------------------------=== #
comptime _AIRY_U0: Float64 = 1.0
comptime _AIRY_U1: Float64 = 0.06944444444444445
comptime _AIRY_U2: Float64 = 0.037133487654320986
comptime _AIRY_U3: Float64 = 0.03799305912780064
comptime _AIRY_U4: Float64 = 0.05764919041266972
comptime _AIRY_U5: Float64 = 0.11609906402551541
comptime _AIRY_U6: Float64 = 0.2915913992307505
comptime _AIRY_U7: Float64 = 0.8776669695100169


def _airy_series_fg[T: NumericComparable](x: T) -> Tuple[T, T]:
    """The two power-series solutions of `y'' = x*y`: `f` (even powers of
    `x`, `f(0) = 1`, `f'(0) = 0`) and `g` (odd powers, `g(0) = 0`,
    `g'(0) = 1`), via the recurrence `a_n = a_{n-3} / (n*(n-1))` the
    differential equation gives their Taylor coefficients.
    """
    var x3 = x * x * x
    var f = T.one()
    var af = T.one()
    var n = 3
    for j in range(1, 200):
        af = af * (x3 / Float64(n * (n - 1)))
        f = f + af
        n += 3
        if j > 5 and af.abs() < 1e-18 * f.abs():
            break

    var g = x
    var ag = x
    n = 4
    for j in range(1, 200):
        ag = ag * (x3 / Float64(n * (n - 1)))
        g = g + ag
        n += 3
        if j > 5 and ag.abs() < 1e-18 * g.abs():
            break
    return (f^, g^)


def _airy_asym_pos[T: NumericComparable](x: T) -> Tuple[T, T]:
    """`(Ai(x), Bi(x))` for `x >= 6`, via A&S 10.4.59: with
    `zeta = (2/3)*x^1.5`, `Ai(x) ~ exp(-zeta)/(2*sqrt(pi)*x^0.25) *
    sum_k (-1)^k u_k/zeta^k` and `Bi(x) ~ exp(zeta)/(sqrt(pi)*x^0.25) *
    sum_k u_k/zeta^k`.
    """
    var zeta = (2.0 / 3.0) * x * x.sqrt()
    var inv_zeta = 1.0 / zeta
    var sum_a = T.constant(_AIRY_U0)
    var sum_b = T.constant(_AIRY_U0)
    var running = T.one()
    var sign = -1.0
    var us = List[Float64](capacity=7)
    us.append(_AIRY_U1)
    us.append(_AIRY_U2)
    us.append(_AIRY_U3)
    us.append(_AIRY_U4)
    us.append(_AIRY_U5)
    us.append(_AIRY_U6)
    us.append(_AIRY_U7)
    for u in us:
        running = running * inv_zeta
        sum_a = sum_a + sign * u * running
        sum_b = sum_b + u * running
        sign = -sign

    var x_quarter = x.sqrt().sqrt()
    var ai = (-zeta).exp() / (2.0 * sqrt(pi) * x_quarter) * sum_a
    var bi = zeta.exp() / (sqrt(pi) * x_quarter) * sum_b
    return (ai^, bi^)


def _airy_asym_neg[T: NumericComparable](x: T) -> Tuple[T, T]:
    """`(Ai(x), Bi(x))` for `x <= -9`, via A&S 10.4.63-10.4.66: with
    `z = -x`, `zeta = (2/3)*z^1.5`, and `phase = zeta - pi/4`,
    `Ai(x) ~ (cos(phase)*S_even + sin(phase)*S_odd) / (sqrt(pi)*z^0.25)`
    and `Bi(x) ~ (-sin(phase)*S_even + cos(phase)*S_odd) / (sqrt(pi)*z^0.25)`,
    where `S_even = sum_k (-1)^k u_2k/zeta^2k` and
    `S_odd = sum_k (-1)^k u_{2k+1}/zeta^{2k+1}`.
    """
    var z = -x
    var zeta = (2.0 / 3.0) * z * z.sqrt()
    var inv_zeta2 = 1.0 / (zeta * zeta)
    var even_us = List[Float64](capacity=4)
    even_us.append(_AIRY_U0)
    even_us.append(_AIRY_U2)
    even_us.append(_AIRY_U4)
    even_us.append(_AIRY_U6)
    var odd_us = List[Float64](capacity=4)
    odd_us.append(_AIRY_U1)
    odd_us.append(_AIRY_U3)
    odd_us.append(_AIRY_U5)
    odd_us.append(_AIRY_U7)
    var s_even = T.constant(0.0)
    var s_odd = T.constant(0.0)
    var running_even = T.one()
    var running_odd = 1.0 / zeta
    var sign = 1.0
    for k in range(4):
        s_even = s_even + sign * even_us[k] * running_even
        s_odd = s_odd + sign * odd_us[k] * running_odd
        running_even = running_even * inv_zeta2
        running_odd = running_odd * inv_zeta2
        sign = -sign

    var z_quarter = z.sqrt().sqrt()
    var phase = zeta - pi / 4.0
    var norm = 1.0 / (sqrt(pi) * z_quarter)
    var ai = (phase.cos() * s_even + phase.sin() * s_odd) * norm
    var bi = (-(phase.sin() * s_even) + phase.cos() * s_odd) * norm
    return (ai^, bi^)


def airy_ai[T: NumericComparable](x: T) -> T:
    """The Airy function of the first kind, `Ai(x)`.

    Parameters:
        T: A `NumericComparable` conformer (`Primal`, `Dual`, ...).

    Args:
        x: The point at which to evaluate `Ai`.

    Returns:
        `Ai(x)`.

    Examples:
        ```mojo
        from scijo.special import airy_ai
        from scijo.prelude import *

        var y = airy_ai(Scalar[f64](0.0))  # 0.3550280538878172...
        ```
    """
    if x < _AIRY_NEG_THRESHOLD:
        var ai: T
        ai, _ = _airy_asym_neg(x)
        return ai
    if x < _AIRY_POS_THRESHOLD:
        var f: T
        var g: T
        f, g = _airy_series_fg(x)
        return _AIRY_C1 * f - _AIRY_C2 * g
    var ai: T
    ai, _ = _airy_asym_pos(x)
    return ai


def airy_ai[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    """The Airy function of the first kind, on a plain `Scalar`.

    Parameters:
        dtype: The floating-point data type.

    Args:
        x: The point at which to evaluate `Ai`.

    Returns:
        `Ai(x)`.
    """
    return airy_ai(Primal[dtype](x)).v[0]


def airy_bi[T: NumericComparable](x: T) -> T:
    """The Airy function of the second kind, `Bi(x)`.

    Parameters:
        T: A `NumericComparable` conformer (`Primal`, `Dual`, ...).

    Args:
        x: The point at which to evaluate `Bi`.

    Returns:
        `Bi(x)`.

    Examples:
        ```mojo
        from scijo.special import airy_bi
        from scijo.prelude import *

        var y = airy_bi(Scalar[f64](0.0))  # 0.6149266274460007...
        ```
    """
    if x < _AIRY_NEG_THRESHOLD:
        var bi: T
        _, bi = _airy_asym_neg(x)
        return bi
    if x < _AIRY_POS_THRESHOLD:
        var f: T
        var g: T
        f, g = _airy_series_fg(x)
        return sqrt(3.0) * (_AIRY_C1 * f + _AIRY_C2 * g)
    var bi: T
    _, bi = _airy_asym_pos(x)
    return bi


def airy_bi[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    """The Airy function of the second kind, on a plain `Scalar`.

    Parameters:
        dtype: The floating-point data type.

    Args:
        x: The point at which to evaluate `Bi`.

    Returns:
        `Bi(x)`.
    """
    return airy_bi(Primal[dtype](x)).v[0]
