# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.math import exp, log, pi, sqrt
from std.testing import assert_almost_equal, TestSuite

# ===----------------------------------------------------------------------=== #
# SciJo
# ===----------------------------------------------------------------------=== #
from scijo import f64
from scijo.special import (
    airy_ai,
    airy_bi,
    beta,
    beta_inc,
    bessel_i0,
    bessel_i1,
    bessel_in,
    bessel_j0,
    bessel_j1,
    bessel_jn,
    bessel_k0,
    bessel_k1,
    bessel_kn,
    bessel_y0,
    bessel_y1,
    bessel_yn,
    digamma,
    erf,
    erfc,
    gamma,
    gamma_inc,
    gamma_incc,
    legendre,
    lnbeta,
    lngamma,
    polygamma,
)
from scijo.types import Dual, NumericComparable, Primal, variable


# ===----------------------------------------------------------------------=== #
# Level 1 (`scijo.special`, plain `Scalar`): values against the scipy oracle
# (`pixi run -e test python3 -c "from scipy.special import ..."`, recorded
# in each assertion's comment) and against identities that need no oracle.
# ===----------------------------------------------------------------------=== #


def test_gamma_matches_oracle_and_factorial() raises:
    assert_almost_equal(gamma(Scalar[f64](5.0)), 24.0, msg="gamma(5) == 4!")
    assert_almost_equal(
        gamma(Scalar[f64](2.5)), 1.329340388179137, msg="gamma(2.5)"
    )
    assert_almost_equal(gamma(Scalar[f64](0.5)), sqrt(pi), msg="gamma(0.5)")


def test_lngamma_matches_oracle_and_stays_finite() raises:
    assert_almost_equal(
        lngamma(Scalar[f64](171.0)),
        706.5730622457875,
        msg="lngamma(171), where gamma(171) already overflows f64",
    )


def test_digamma_matches_oracle() raises:
    assert_almost_equal(
        digamma(Scalar[f64](1.0)), -0.5772156649015329, msg="digamma(1)"
    )
    assert_almost_equal(
        digamma(Scalar[f64](2.5)), 0.7031566406452432, msg="digamma(2.5)"
    )


def test_erf_matches_oracle_and_odd_symmetry() raises:
    assert_almost_equal(
        erf(Scalar[f64](1.0)), 0.8427007929497148, msg="erf(1)"
    )
    assert_almost_equal(
        erf(Scalar[f64](-1.0)), -0.8427007929497148, msg="erf is odd"
    )
    assert_almost_equal(
        erf(Scalar[f64](1.0)) + erfc(Scalar[f64](1.0)),
        1.0,
        msg="erf(x) + erfc(x) == 1",
    )
    assert_almost_equal(
        erfc(Scalar[f64](5.0)),
        1.5374597944280353e-12,
        atol=1e-20,
        msg="erfc(5), where 1 - erf(5) would lose all its digits",
    )


def test_beta_matches_oracle_and_gamma_relation() raises:
    assert_almost_equal(
        beta(Scalar[f64](2.0), Scalar[f64](3.0)),
        0.08333333333333333,
        msg="beta(2, 3)",
    )
    assert_almost_equal(
        lnbeta(Scalar[f64](2.5), Scalar[f64](4.5)),
        -3.8408317706947397,
        msg="lnbeta(2.5, 4.5)",
    )
    assert_almost_equal(
        beta(Scalar[f64](2.5), Scalar[f64](4.5)),
        beta(Scalar[f64](4.5), Scalar[f64](2.5)),
        msg="beta(a, b) == beta(b, a)",
    )


def test_gamma_inc_matches_oracle_and_boundaries() raises:
    assert_almost_equal(
        gamma_inc(Scalar[f64](2.0), Scalar[f64](1.0)),
        0.2642411176571153,
        msg="gamma_inc(2, 1)",
    )
    assert_almost_equal(
        gamma_incc(Scalar[f64](2.0), Scalar[f64](1.0)),
        0.7357588823428847,
        msg="gamma_incc(2, 1)",
    )
    assert_almost_equal(
        gamma_inc(Scalar[f64](0.5), Scalar[f64](0.5)),
        0.6826894921370859,
        msg="gamma_inc(0.5, 0.5), past the series/CF threshold",
    )
    assert_almost_equal(
        gamma_inc(Scalar[f64](3.0), Scalar[f64](0.0)),
        0.0,
        msg="gamma_inc(a, 0) == 0",
    )
    assert_almost_equal(
        gamma_inc(Scalar[f64](2.5), Scalar[f64](4.0))
        + gamma_incc(Scalar[f64](2.5), Scalar[f64](4.0)),
        1.0,
        msg="P(a, x) + Q(a, x) == 1",
    )
    assert_almost_equal(
        gamma_inc(Scalar[f64](1.0), Scalar[f64](2.0)),
        1.0 - exp(Scalar[f64](-2.0)),
        msg="gamma_inc(1, x) == 1 - exp(-x), the exponential CDF",
    )


def test_beta_inc_matches_oracle_and_boundaries() raises:
    assert_almost_equal(
        beta_inc(Scalar[f64](2.0), Scalar[f64](3.0), Scalar[f64](0.4)),
        0.5247999999999999,
        msg="beta_inc(2, 3, 0.4)",
    )
    assert_almost_equal(
        beta_inc(Scalar[f64](0.5), Scalar[f64](0.5), Scalar[f64](0.3)),
        0.36901011956554536,
        msg="beta_inc(0.5, 0.5, 0.3), past the continued-fraction threshold",
    )
    assert_almost_equal(
        beta_inc(Scalar[f64](2.0), Scalar[f64](3.0), Scalar[f64](0.0)),
        0.0,
        msg="beta_inc(a, b, 0) == 0",
    )
    assert_almost_equal(
        beta_inc(Scalar[f64](2.0), Scalar[f64](3.0), Scalar[f64](1.0)),
        1.0,
        msg="beta_inc(a, b, 1) == 1",
    )
    assert_almost_equal(
        beta_inc(Scalar[f64](2.0), Scalar[f64](3.0), Scalar[f64](0.4))
        + beta_inc(Scalar[f64](3.0), Scalar[f64](2.0), Scalar[f64](0.6)),
        1.0,
        msg="I_x(a, b) == 1 - I_(1-x)(b, a)",
    )
    assert_almost_equal(
        beta_inc(Scalar[f64](1.0), Scalar[f64](1.0), Scalar[f64](0.37)),
        0.37,
        msg="beta_inc(1, 1, x) == x, the uniform CDF",
    )


def test_bessel_matches_oracle() raises:
    assert_almost_equal(bessel_j0(Scalar[f64](2.0)), 0.22389077914123562)
    assert_almost_equal(bessel_j1(Scalar[f64](2.0)), 0.5767248077568734)
    assert_almost_equal(bessel_y0(Scalar[f64](2.0)), 0.5103756726497451)
    assert_almost_equal(bessel_y1(Scalar[f64](2.0)), -0.10703243154093758)
    assert_almost_equal(bessel_i0(Scalar[f64](2.0)), 2.279585302336067)
    assert_almost_equal(bessel_i1(Scalar[f64](2.0)), 1.5906368546373295)
    assert_almost_equal(bessel_k0(Scalar[f64](2.0)), 0.1138938727495334)
    assert_almost_equal(bessel_k1(Scalar[f64](2.0)), 0.13986588181652246)


def test_bessel_wronskian_identity() raises:
    """`J0(x)*Y1(x) - J1(x)*Y0(x) == -2/(pi*x)`, true for every `x`, and
    independent of the series/asymptotic split each side was computed with
    (checked at a series-branch and an asymptotic-branch point)."""
    for x in [2.0, 30.0]:
        var lhs = bessel_j0(Scalar[f64](x)) * bessel_y1(Scalar[f64](x)) - (
            bessel_j1(Scalar[f64](x)) * bessel_y0(Scalar[f64](x))
        )
        var rhs = -2.0 / (pi * x)
        assert_almost_equal(
            lhs, rhs, msg="Bessel Wronskian at x=" + String(x)
        )


def test_bessel_asymptotic_branch_matches_oracle() raises:
    """Past each function's series/asymptotic threshold (`x >= 15` for
    `J`/`Y`/`I`, `x >= 8` for `K`), it's the asymptotic expansion being
    checked here, not the series."""
    assert_almost_equal(
        bessel_j0(Scalar[f64](20.0)), 0.16702466434058313, msg="J0(20)"
    )
    assert_almost_equal(
        bessel_k0(Scalar[f64](10.0)),
        1.778006231616765e-05,
        atol=1e-12,
        msg="K0(10)",
    )


def test_bessel_n_matches_oracle_and_order01() raises:
    """General-order Bessel functions, checked against scipy at both a
    Miller's-algorithm point (`J_n`/`I_n` with `n > x`) and a direct
    forward-recurrence point (`n <= x`), plus agreement with the order-0/1
    functions at `n = 0, 1`."""
    assert_almost_equal(
        bessel_jn(2, Scalar[f64](1.0)), 0.1149034849319005, msg="J2(1), n > x"
    )
    assert_almost_equal(
        bessel_jn(2, Scalar[f64](20.0)),
        -0.16034135192299823,
        msg="J2(20), n <= x",
    )
    assert_almost_equal(
        bessel_jn(10, Scalar[f64](0.5)),
        2.613177360822802e-13,
        atol=1e-20,
        msg="J10(0.5), deep in the Miller's-algorithm regime",
    )
    assert_almost_equal(
        bessel_yn(5, Scalar[f64](20.0)), -0.10003576788953245, msg="Y5(20)"
    )
    assert_almost_equal(
        bessel_in(2, Scalar[f64](5.0)), 17.505614966624236, msg="I2(5)"
    )
    assert_almost_equal(
        bessel_kn(10, Scalar[f64](1.0)),
        180713289.90102947,
        rtol=1e-4,
        msg="K10(1)",
    )
    assert_almost_equal(
        bessel_jn(0, Scalar[f64](3.0)), bessel_j0(Scalar[f64](3.0)), msg="J_0 == J0"
    )
    assert_almost_equal(
        bessel_jn(1, Scalar[f64](3.0)), bessel_j1(Scalar[f64](3.0)), msg="J_1 == J1"
    )


def test_bessel_n_recurrence_identity() raises:
    """`J_{n-1}(x) + J_{n+1}(x) == (2n/x)*J_n(x)`, true for every order and
    independent of which branch (`x > n` or Miller's algorithm) each side
    was computed with."""
    var x = Scalar[f64](5.0)
    var lhs = bessel_jn(2, x) + bessel_jn(4, x)
    var rhs = (2.0 * 3.0 / 5.0) * bessel_jn(3, x)
    assert_almost_equal(lhs, rhs, msg="J2(5) + J4(5) == (6/5)*J3(5)")


# ===----------------------------------------------------------------------=== #
# Level 2 (the generic `scijo.special` overloads, run through `Dual`): every
# derivative here falls out of `Dual`'s own chain-rule arithmetic - there is
# no hand-written derivative formula anywhere in `scijo.special` to call.
# The expected values below are still the closed-form derivative, used only
# to check the *result* is exact, the same "no finite-difference tolerance
# needed" pattern as `tests/test_types.mojo`.
# ===----------------------------------------------------------------------=== #


def erf_fn[T: NumericComparable](x: T) -> T:
    return erf(x)


def gamma_fn[T: NumericComparable](x: T) -> T:
    return gamma(x)


def digamma_fn[T: NumericComparable](x: T) -> T:
    return digamma(x)


def bessel_j0_fn[T: NumericComparable](x: T) -> T:
    return bessel_j0(x)


def bessel_j1_fn[T: NumericComparable](x: T) -> T:
    return bessel_j1(x)


def bessel_y0_fn[T: NumericComparable](x: T) -> T:
    return bessel_y0(x)


def bessel_i1_fn[T: NumericComparable](x: T) -> T:
    return bessel_i1(x)


def bessel_k1_fn[T: NumericComparable](x: T) -> T:
    return bessel_k1(x)


def gamma_inc_fn[T: NumericComparable](a: T, x: T) -> T:
    return gamma_inc(a, x)


def beta_inc_fn[T: NumericComparable](a: T, b: T, x: T) -> T:
    return beta_inc(a, b, x)


def test_dual_erf_derivative() raises:
    var x0 = 0.8
    var x = variable(x0)
    var result = erf_fn(x)
    assert_almost_equal(result.value.v, erf(Scalar[f64](x0)), msg="erf(0.8)")
    var expected_deriv = (2.0 / sqrt(pi)) * exp(-(x0 * x0))
    assert_almost_equal(
        result.deriv.v,
        expected_deriv,
        msg="d/dx erf(x) = (2/sqrt(pi)) * exp(-x^2)",
    )


def test_dual_gamma_derivative() raises:
    """`d/dx Gamma(x) = Gamma(x) * psi(x)`, previously impossible to give
    `Dual` without an unbounded `digamma -> trigamma -> ...` dependency
    chain - no longer an issue, since there is no hand-written derivative
    formula to need one."""
    var x0 = 2.5
    var x = variable(x0)
    var result = gamma_fn(x)
    assert_almost_equal(result.value.v, gamma(Scalar[f64](x0)), msg="gamma(2.5)")
    var expected_deriv = gamma(Scalar[f64](x0)) * digamma(Scalar[f64](x0))
    assert_almost_equal(
        result.deriv.v, expected_deriv, msg="d/dx Gamma(x) = Gamma(x)*psi(x)"
    )


def test_dual_digamma_derivative() raises:
    """`d/dx psi(x) = trigamma(x) = polygamma(1, x)`, checked exactly now
    that `scijo.special` has a `polygamma` oracle - no finite difference
    needed."""
    var x0 = 2.5
    var x = variable(x0)
    var result = digamma_fn(x)
    assert_almost_equal(result.value.v, digamma(Scalar[f64](x0)), msg="digamma(2.5)")
    var expected_deriv = polygamma(1, Scalar[f64](x0))
    assert_almost_equal(
        result.deriv.v, expected_deriv, msg="d/dx psi(x) = trigamma(x)"
    )


def test_polygamma_matches_oracle_and_digamma() raises:
    assert_almost_equal(
        polygamma(1, Scalar[f64](2.5)), 0.4903577561002349, msg="trigamma(2.5)"
    )
    assert_almost_equal(
        polygamma(1, Scalar[f64](1.0)), 1.6449340668482266, msg="trigamma(1) == pi^2/6"
    )
    assert_almost_equal(
        polygamma(2, Scalar[f64](1.5)), -0.8287966442343201, msg="tetragamma(1.5)"
    )
    assert_almost_equal(
        polygamma(3, Scalar[f64](0.7)), 25.879149678427744, msg="polygamma(3, 0.7)"
    )
    assert_almost_equal(
        polygamma(0, Scalar[f64](2.5)),
        digamma(Scalar[f64](2.5)),
        msg="polygamma(0, x) == digamma(x)",
    )


def polygamma1_fn[T: NumericComparable](x: T) -> T:
    return polygamma(1, x)


def test_dual_trigamma_derivative() raises:
    """`d/dx trigamma(x) = tetragamma(x) = polygamma(2, x)`, the same
    free-derivative chain one level deeper."""
    var x0 = 2.5
    var x = variable(x0)
    var result = polygamma1_fn(x)
    assert_almost_equal(result.value.v, polygamma(1, Scalar[f64](x0)), msg="trigamma(2.5)")
    var expected_deriv = polygamma(2, Scalar[f64](x0))
    assert_almost_equal(
        result.deriv.v, expected_deriv, msg="d/dx trigamma(x) = tetragamma(x)"
    )


def test_dual_bessel_j0_derivative() raises:
    """`d/dx J0(x) = -J1(x)`."""
    var x0 = 3.0
    var x = variable(x0)
    var result = bessel_j0_fn(x)
    assert_almost_equal(result.value.v, bessel_j0(Scalar[f64](x0)))
    assert_almost_equal(result.deriv.v, -bessel_j1(Scalar[f64](x0)))


def test_dual_bessel_j1_derivative() raises:
    """`d/dx J1(x) = J0(x) - J1(x)/x`."""
    var x0 = 3.0
    var x = variable(x0)
    var result = bessel_j1_fn(x)
    assert_almost_equal(result.value.v, bessel_j1(Scalar[f64](x0)))
    var expected = bessel_j0(Scalar[f64](x0)) - bessel_j1(
        Scalar[f64](x0)
    ) / x0
    assert_almost_equal(result.deriv.v, expected)


def test_dual_bessel_y0_derivative() raises:
    """`d/dx Y0(x) = -Y1(x)`."""
    var x0 = 3.0
    var x = variable(x0)
    var result = bessel_y0_fn(x)
    assert_almost_equal(result.value.v, bessel_y0(Scalar[f64](x0)))
    assert_almost_equal(result.deriv.v, -bessel_y1(Scalar[f64](x0)))


def test_dual_bessel_i1_derivative() raises:
    """`d/dx I1(x) = I0(x) - I1(x)/x`."""
    var x0 = 2.0
    var x = variable(x0)
    var result = bessel_i1_fn(x)
    assert_almost_equal(result.value.v, bessel_i1(Scalar[f64](x0)))
    var expected = bessel_i0(Scalar[f64](x0)) - bessel_i1(
        Scalar[f64](x0)
    ) / x0
    assert_almost_equal(result.deriv.v, expected)


def test_dual_bessel_k1_derivative() raises:
    """`d/dx K1(x) = -K0(x) - K1(x)/x`."""
    var x0 = 2.0
    var x = variable(x0)
    var result = bessel_k1_fn(x)
    assert_almost_equal(result.value.v, bessel_k1(Scalar[f64](x0)))
    var expected = -bessel_k0(Scalar[f64](x0)) - bessel_k1(
        Scalar[f64](x0)
    ) / x0
    assert_almost_equal(result.deriv.v, expected)


def test_dual_gamma_inc_derivative() raises:
    """`d/dx P(a, x) = x^(a-1) * exp(-x) / Gamma(a)`, the gamma pdf, with
    `a` held fixed as a zero-derivative constant."""
    var a0 = 2.5
    var x0 = 1.3
    var a = Dual[Primal[f64]].constant(a0)
    var x = variable(x0)
    var result = gamma_inc_fn(a, x)
    assert_almost_equal(
        result.value.v, gamma_inc(Scalar[f64](a0), Scalar[f64](x0)),
        msg="gamma_inc(2.5, 1.3)",
    )
    var expected_deriv = exp((a0 - 1.0) * log(x0) - x0 - lngamma(Scalar[f64](a0)))
    assert_almost_equal(
        result.deriv.v, expected_deriv,
        msg="d/dx P(a, x) = x^(a-1)*exp(-x)/Gamma(a)",
    )


def test_dual_beta_inc_derivative() raises:
    """`d/dx I_x(a, b) = x^(a-1) * (1-x)^(b-1) / B(a, b)`, the beta pdf,
    with `a` and `b` held fixed as zero-derivative constants."""
    var a0 = 2.5
    var b0 = 3.1
    var x0 = 0.42
    var a = Dual[Primal[f64]].constant(a0)
    var b = Dual[Primal[f64]].constant(b0)
    var x = variable(x0)
    var result = beta_inc_fn(a, b, x)
    assert_almost_equal(
        result.value.v,
        beta_inc(Scalar[f64](a0), Scalar[f64](b0), Scalar[f64](x0)),
        msg="beta_inc(2.5, 3.1, 0.42)",
    )
    var expected_deriv = exp(
        (a0 - 1.0) * log(x0)
        + (b0 - 1.0) * log(1.0 - x0)
        - lnbeta(Scalar[f64](a0), Scalar[f64](b0))
    )
    assert_almost_equal(
        result.deriv.v, expected_deriv,
        msg="d/dx I_x(a, b) = x^(a-1)*(1-x)^(b-1)/B(a, b)",
    )


def bessel_j3_fn[T: NumericComparable](x: T) -> T:
    return bessel_jn(3, x)


def test_dual_bessel_jn_derivative() raises:
    """`d/dx J_n(x) = J_{n-1}(x) - n*J_n(x)/x`, checked at `n = 3`."""
    var x0 = 4.0
    var x = variable(x0)
    var result = bessel_j3_fn(x)
    assert_almost_equal(result.value.v, bessel_jn(3, Scalar[f64](x0)))
    var expected = bessel_jn(2, Scalar[f64](x0)) - 3.0 * bessel_jn(
        3, Scalar[f64](x0)
    ) / x0
    assert_almost_equal(result.deriv.v, expected)


def test_legendre_matches_oracle_and_order01() raises:
    assert_almost_equal(
        legendre(2, Scalar[f64](0.5)), -0.125, msg="P2(0.5)"
    )
    assert_almost_equal(
        legendre(5, Scalar[f64](0.3)), 0.3453862499999999, msg="P5(0.3)"
    )
    assert_almost_equal(
        legendre(10, Scalar[f64](-0.7)), 0.08580579553164014, msg="P10(-0.7)"
    )
    assert_almost_equal(
        legendre(0, Scalar[f64](0.5)), 1.0, msg="P0(x) == 1"
    )
    assert_almost_equal(
        legendre(1, Scalar[f64](0.5)), 0.5, msg="P1(x) == x"
    )


def legendre3_fn[T: NumericComparable](x: T) -> T:
    return legendre(3, x)


def test_dual_legendre_derivative() raises:
    """`d/dx P_3(x) = (3*x*P_3(x) - 3*P_2(x)) / (x^2 - 1)`, Bonnet's
    derivative recurrence, checked at `n = 3`."""
    var x0 = 0.4
    var x = variable(x0)
    var result = legendre3_fn(x)
    assert_almost_equal(result.value.v, legendre(3, Scalar[f64](x0)))
    var expected = (
        3.0 * x0 * legendre(3, Scalar[f64](x0))
        - 3.0 * legendre(2, Scalar[f64](x0))
    ) / (x0 * x0 - 1.0)
    assert_almost_equal(result.deriv.v, expected)


def test_airy_matches_oracle_across_both_thresholds() raises:
    """Checked on both sides of the series/asymptotic split for each
    branch (`x < -9`, `-9 <= x < 6`, `x >= 6`)."""
    assert_almost_equal(
        airy_ai(Scalar[f64](-20.0)), -0.17640612707798434, msg="Ai(-20), oscillatory asymptotic"
    )
    assert_almost_equal(
        airy_ai(Scalar[f64](-5.0)), 0.3507610090241143, msg="Ai(-5), series"
    )
    assert_almost_equal(
        airy_ai(Scalar[f64](0.0)), 0.3550280538878172, msg="Ai(0)"
    )
    assert_almost_equal(
        airy_ai(Scalar[f64](10.0)),
        1.1047532552898654e-10,
        atol=1e-16,
        msg="Ai(10), decaying asymptotic",
    )
    assert_almost_equal(
        airy_bi(Scalar[f64](-20.0)), -0.20013930932265164, msg="Bi(-20), oscillatory asymptotic"
    )
    assert_almost_equal(
        airy_bi(Scalar[f64](0.0)), 0.6149266274460007, msg="Bi(0)"
    )
    assert_almost_equal(
        airy_bi(Scalar[f64](10.0)),
        455641153.54822654,
        rtol=1e-6,
        msg="Bi(10), growing asymptotic",
    )


def airy_ai_fn[T: NumericComparable](x: T) -> T:
    return airy_ai(x)


def test_dual_airy_ai_derivative() raises:
    """`d/dx Ai(x) = Ai'(x)`, which `scijo.special` has no closed form
    for - checked against the scipy oracle for `Ai'`, the whole point of
    `airy_ai` being generic: it falls out of the same algorithm for free."""
    var x0 = -5.0
    var x = variable(x0)
    var result = airy_ai_fn(x)
    assert_almost_equal(result.value.v, airy_ai(Scalar[f64](x0)), msg="Ai(-5)")
    assert_almost_equal(
        result.deriv.v, 0.3271928185544436, msg="Ai'(-5)"
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
