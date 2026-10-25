# ===----------------------------------------------------------------------=== #
# SciJo: General-order Bessel functions for scijo.special
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""General-Order Bessel Functions (`scijo.special.bessel_n`)
============================================================
Integer-order Bessel functions built from `scijo.special.bessel`'s order-0
and order-1 functions via the three-term recurrences all four families
share: `F_{n+1}(x) = (2n/x)*F_n(x) -/+ F_{n-1}(x)` (`-` for `J`/`Y`, `+` for
`I`/`K`). Domain: `n >= 0`; `Y`/`K` additionally need `x > 0`.

`Y_n`/`K_n` recur upward from `Y0,Y1`/`K0,K1` directly - stable, because
`Y`/`K` are each family's *dominant* (growing) solution. `J_n`/`I_n` are the
*minimal* (decaying) solution, for which upward recurrence amplifies
rounding error once `n` grows past `x`; both instead use Miller's algorithm
(downward recurrence from a safely higher order, normalized afterward -
`J_n` against the oscillating-sum identity `J0(x) + 2*sum_k J_2k(x) = 1`,
`I_n` against the direct ratio `I0(x)/(unnormalized I0)`), the standard fix
for that instability. `J_n` additionally uses forward recurrence directly
when `x > n`, where it's stable too and the series/asymptotic split in
`bessel_j0`/`bessel_j1` already gives a good starting point.

References
----------
- Numerical Recipes, "Bessel Functions of Integer Order" (the stable-
  recurrence-direction algorithm, including Miller's method and its two
  normalizations, is the standard one; reimplemented from its description
  here, not copied from any existing source).
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.math import sqrt

# ===----------------------------------------------------------------------=== #
# SciJo
# ===----------------------------------------------------------------------=== #
from scijo.special.bessel import (
    bessel_i0,
    bessel_i1,
    bessel_j0,
    bessel_j1,
    bessel_k0,
    bessel_k1,
    bessel_y0,
    bessel_y1,
)
from scijo.types import (
    NumericComparable,
    Primal,
)

comptime _BESSEL_N_ACC: Float64 = 40.0
comptime _BESSEL_N_BIG: Float64 = 1e10
comptime _BESSEL_N_BIGI: Float64 = 1e-10

# ===----------------------------------------------------------------------=== #
# J_n (first kind), Y_n (second kind)
# ===----------------------------------------------------------------------=== #


def bessel_jn[T: NumericComparable](n: Int, x: T) -> T:
    """The order-`n` Bessel function of the first kind, `J_n(x)`.

    Parameters:
        T: A `NumericComparable` conformer (`Primal`, `Dual`, ...).

    Args:
        n: The order. Must be non-negative.
        x: The point at which to evaluate `J_n`.

    Returns:
        `J_n(x)`.
    """
    if n == 0:
        return bessel_j0(x)
    if n == 1:
        return bessel_j1(x)
    if x < 0.0:
        var r = bessel_jn(n, -x)
        return -r if n % 2 == 1 else r
    if x == T.constant(0.0):
        return T.constant(0.0)

    var tox = 2.0 / x
    if x > T.constant(Float64(n)):
        var bjm = bessel_j0(x)
        var bj = bessel_j1(x)
        for j in range(1, n):
            var bjp = Float64(j) * tox * bj - bjm
            bjm = bj
            bj = bjp
        return bj

    var m = 2 * ((n + Int(sqrt(_BESSEL_N_ACC * Float64(n)))) // 2)
    var jsum = False
    var bjp = T.constant(0.0)
    var ans = T.constant(0.0)
    var summ = T.constant(0.0)
    var bj = T.constant(1.0)
    for j in range(m, 0, -1):
        var bjm = Float64(j) * tox * bj - bjp
        bjp = bj
        bj = bjm
        if bj.abs() > T.constant(_BESSEL_N_BIG):
            bj = bj * _BESSEL_N_BIGI
            bjp = bjp * _BESSEL_N_BIGI
            ans = ans * _BESSEL_N_BIGI
            summ = summ * _BESSEL_N_BIGI
        if jsum:
            summ = summ + bj
        jsum = not jsum
        if j == n:
            ans = bjp
    summ = 2.0 * summ - bj
    return ans / summ


def bessel_jn[
    dtype: DType
](n: Int, x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    """The order-`n` Bessel function of the first kind, on a plain
    `Scalar`.

    Parameters:
        dtype: The floating-point data type.

    Args:
        n: The order. Must be non-negative.
        x: The point at which to evaluate `J_n`.

    Returns:
        `J_n(x)`.
    """
    return bessel_jn(n, Primal[dtype](x)).v[0]


def bessel_yn[T: NumericComparable](n: Int, x: T) -> T:
    """The order-`n` Bessel function of the second kind, `Y_n(x)`.

    Parameters:
        T: A `NumericComparable` conformer (`Primal`, `Dual`, ...).

    Args:
        n: The order. Must be non-negative.
        x: The point at which to evaluate `Y_n`. Must be positive.

    Returns:
        `Y_n(x)`. Diverges as `x -> 0`.
    """
    if n == 0:
        return bessel_y0(x)
    if n == 1:
        return bessel_y1(x)

    var tox = 2.0 / x
    var bym = bessel_y0(x)
    var by = bessel_y1(x)
    for j in range(1, n):
        var byp = Float64(j) * tox * by - bym
        bym = by
        by = byp
    return by


def bessel_yn[
    dtype: DType
](n: Int, x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    """The order-`n` Bessel function of the second kind, on a plain
    `Scalar`.

    Parameters:
        dtype: The floating-point data type.

    Args:
        n: The order. Must be non-negative.
        x: The point at which to evaluate `Y_n`. Must be positive.

    Returns:
        `Y_n(x)`. Diverges as `x -> 0`.
    """
    return bessel_yn(n, Primal[dtype](x)).v[0]


# ===----------------------------------------------------------------------=== #
# I_n (modified, first kind), K_n (modified, second kind)
# ===----------------------------------------------------------------------=== #


def bessel_in[T: NumericComparable](n: Int, x: T) -> T:
    """The order-`n` modified Bessel function of the first kind, `I_n(x)`.

    Parameters:
        T: A `NumericComparable` conformer (`Primal`, `Dual`, ...).

    Args:
        n: The order. Must be non-negative.
        x: The point at which to evaluate `I_n`.

    Returns:
        `I_n(x)`.
    """
    if n == 0:
        return bessel_i0(x)
    if n == 1:
        return bessel_i1(x)
    if x < 0.0:
        var r = bessel_in(n, -x)
        return -r if n % 2 == 1 else r
    if x == T.constant(0.0):
        return T.constant(0.0)

    var tox = 2.0 / x
    var m = 2 * (n + Int(sqrt(_BESSEL_N_ACC * Float64(n))))
    var bip = T.constant(0.0)
    var ans = T.constant(0.0)
    var bi = T.constant(1.0)
    for j in range(m, 0, -1):
        var bim = bip + Float64(j) * tox * bi
        bip = bi
        bi = bim
        if bi.abs() > T.constant(_BESSEL_N_BIG):
            ans = ans * _BESSEL_N_BIGI
            bi = bi * _BESSEL_N_BIGI
            bip = bip * _BESSEL_N_BIGI
        if j == n:
            ans = bip
    return ans * (bessel_i0(x) / bi)


def bessel_in[
    dtype: DType
](n: Int, x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    """The order-`n` modified Bessel function of the first kind, on a plain
    `Scalar`.

    Parameters:
        dtype: The floating-point data type.

    Args:
        n: The order. Must be non-negative.
        x: The point at which to evaluate `I_n`.

    Returns:
        `I_n(x)`.
    """
    return bessel_in(n, Primal[dtype](x)).v[0]


def bessel_kn[T: NumericComparable](n: Int, x: T) -> T:
    """The order-`n` modified Bessel function of the second kind, `K_n(x)`.

    Parameters:
        T: A `NumericComparable` conformer (`Primal`, `Dual`, ...).

    Args:
        n: The order. Must be non-negative.
        x: The point at which to evaluate `K_n`. Must be positive.

    Returns:
        `K_n(x)`. Diverges as `x -> 0`.
    """
    if n == 0:
        return bessel_k0(x)
    if n == 1:
        return bessel_k1(x)

    var tox = 2.0 / x
    var bkm = bessel_k0(x)
    var bk = bessel_k1(x)
    for j in range(1, n):
        var bkp = bkm + Float64(j) * tox * bk
        bkm = bk
        bk = bkp
    return bk


def bessel_kn[
    dtype: DType
](n: Int, x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    """The order-`n` modified Bessel function of the second kind, on a
    plain `Scalar`.

    Parameters:
        dtype: The floating-point data type.

    Args:
        n: The order. Must be non-negative.
        x: The point at which to evaluate `K_n`. Must be positive.

    Returns:
        `K_n(x)`. Diverges as `x -> 0`.
    """
    return bessel_kn(n, Primal[dtype](x)).v[0]
