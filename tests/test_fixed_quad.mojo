"""
Tests for fixed_quad, dblquad, tplquad, and nquad.
"""

from std.python import Python, PythonObject
from std.testing import (
    assert_almost_equal,
    assert_true,
    TestSuite,
)
from std.math import pi, sin

from scijo.integrate import (
    dblquad,
    fixed_quad,
    nquad,
    tplquad,
)
from scijo.integrate.fixed_quad import _gauss_legendre_nodes_weights
import scijo as sj


def test_fixed_quad_polynomial_exact() raises:
    """Fixed_quad is exact for polynomials of degree <= 2n - 1."""

    def cubic[
        dtype: DType
    ](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) capturing -> Scalar[dtype]:
        return x * x * x - 2.0 * x

    # integral of x^3 - 2x from 0 to 2 = [x^4/4 - x^2] = 4 - 4 = 0
    var result = fixed_quad[sj.f64, cubic](0.0, 2.0, n=3)
    assert_almost_equal(result, 0.0, atol=1e-10)


def test_fixed_quad_args_free_overload() raises:
    def quadratic[dtype: DType](x: Scalar[dtype]) capturing -> Scalar[dtype]:
        return x * x

    # integral of x^2 from 0 to 3 = 9
    var result = fixed_quad[sj.f64, quadratic](0.0, 3.0, n=5)
    assert_almost_equal(result, 9.0, atol=1e-10)


def test_fixed_quad_with_args() raises:
    def scaled[
        dtype: DType
    ](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) capturing -> Scalar[dtype]:
        return args.value()[0] * x

    var arglist: List[Scalar[sj.f64]] = [3.0]
    var result = fixed_quad[sj.f64, scaled](0.0, 2.0, args=arglist^, n=2)
    # integral of 3x from 0 to 2 = 6
    assert_almost_equal(result, 6.0, atol=1e-10)


def test_fixed_quad_invalid_n_raises() raises:
    def one[dtype: DType](x: Scalar[dtype]) capturing -> Scalar[dtype]:
        return 1.0

    var raised = False
    try:
        _ = fixed_quad[sj.f64, one](0.0, 1.0, n=0)
    except:
        raised = True
    assert_true(raised, msg="n < 1 should raise")


def test_fixed_quad_sine_half_period() raises:
    def sine[
        dtype: DType
    ](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) capturing -> Scalar[dtype] where dtype.is_floating_point():
        return sin(x)

    # integral of sin(x) from 0 to pi = 2
    var result = fixed_quad[sj.f64, sine](0.0, Scalar[sj.f64](pi), n=10)
    assert_almost_equal(result, 2.0, atol=1e-6)


def test_gauss_legendre_nodes_weights_match_scipy() raises:
    var scipy_special: PythonObject
    try:
        scipy_special = Python.import_module("scipy.special")
    except:
        print("SciPy not available, skipping Gauss-Legendre nodes/weights test")
        return

    var nw = _gauss_legendre_nodes_weights[sj.f64](7)
    ref nodes = nw[0]
    ref weights = nw[1]

    var py_result = scipy_special.roots_legendre(7)
    for i in range(7):
        assert_almost_equal(
            nodes[i],
            Float64(py=py_result[0][i]),
            atol=1e-10,
            msg="Gauss-Legendre node " + String(i) + " should match SciPy",
        )
        assert_almost_equal(
            weights[i],
            Float64(py=py_result[1][i]),
            atol=1e-10,
            msg="Gauss-Legendre weight " + String(i) + " should match SciPy",
        )


def test_dblquad_unit_square() raises:
    """Integral of x*y over the unit square == 1/4."""

    def f[
        dtype: DType
    ](
        y: Scalar[dtype], x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) capturing -> Scalar[dtype]:
        return x * y

    def gfun[dtype: DType](x: Scalar[dtype]) capturing -> Scalar[dtype]:
        return 0.0

    def hfun[dtype: DType](x: Scalar[dtype]) capturing -> Scalar[dtype]:
        return 1.0

    var result = dblquad[sj.f64, f, gfun, hfun](0.0, 1.0, n=10)
    assert_almost_equal(result.integral, 0.25, atol=1e-8)


def test_dblquad_triangular_region() raises:
    """Integral of 1 over {0<=x<=1, 0<=y<=x} == area of the triangle == 1/2."""

    def one[
        dtype: DType
    ](
        y: Scalar[dtype], x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) capturing -> Scalar[dtype]:
        return 1.0

    def gfun[dtype: DType](x: Scalar[dtype]) capturing -> Scalar[dtype]:
        return 0.0

    def hfun[dtype: DType](x: Scalar[dtype]) capturing -> Scalar[dtype]:
        return x

    var result = dblquad[sj.f64, one, gfun, hfun](0.0, 1.0, n=15)
    assert_almost_equal(result.integral, 0.5, atol=1e-8)


def test_dblquad_args_free_overload() raises:
    def f[
        dtype: DType
    ](y: Scalar[dtype], x: Scalar[dtype]) capturing -> Scalar[dtype]:
        return x + y

    def gfun[dtype: DType](x: Scalar[dtype]) capturing -> Scalar[dtype]:
        return 0.0

    def hfun[dtype: DType](x: Scalar[dtype]) capturing -> Scalar[dtype]:
        return 1.0

    # integral over unit square of (x + y) = 1/2 + 1/2 = 1
    var result = dblquad[sj.f64, f, gfun, hfun](0.0, 1.0, n=10)
    assert_almost_equal(result.integral, 1.0, atol=1e-8)


def test_dblquad_variable_upper_bound() raises:
    """Integral of (x^2 + y) over {0<=x<=2, 0<=y<=1+x} == 11 (closed form,
    cross-checked against `scipy.integrate.dblquad`).
    """

    def f[
        dtype: DType
    ](
        y: Scalar[dtype], x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) capturing -> Scalar[dtype]:
        return x * x + y

    def gfun[dtype: DType](x: Scalar[dtype]) capturing -> Scalar[dtype]:
        return 0.0

    def hfun[dtype: DType](x: Scalar[dtype]) capturing -> Scalar[dtype]:
        return 1.0 + x

    var result = dblquad[sj.f64, f, gfun, hfun](0.0, 2.0, n=25)
    assert_almost_equal(result.integral, 11.0, atol=1e-6)


def test_tplquad_unit_cube() raises:
    """Integral of x*y*z over the unit cube == 1/8."""

    def f[
        dtype: DType
    ](
        z: Scalar[dtype],
        y: Scalar[dtype],
        x: Scalar[dtype],
        args: Optional[List[Scalar[dtype]]],
    ) capturing -> Scalar[dtype]:
        return x * y * z

    def qfun[
        dtype: DType
    ](x: Scalar[dtype], y: Scalar[dtype]) capturing -> Scalar[dtype]:
        return 0.0

    def rfun[
        dtype: DType
    ](x: Scalar[dtype], y: Scalar[dtype]) capturing -> Scalar[dtype]:
        return 1.0

    def gfun[dtype: DType](x: Scalar[dtype]) capturing -> Scalar[dtype]:
        return 0.0

    def hfun[dtype: DType](x: Scalar[dtype]) capturing -> Scalar[dtype]:
        return 1.0

    var result = tplquad[sj.f64, f, qfun, rfun, gfun, hfun](0.0, 1.0, n=10)
    assert_almost_equal(result.integral, 0.125, atol=1e-8)


def test_tplquad_args_free_overload() raises:
    def f[
        dtype: DType
    ](z: Scalar[dtype], y: Scalar[dtype], x: Scalar[dtype]) capturing -> Scalar[
        dtype
    ]:
        return 1.0

    def qfun[
        dtype: DType
    ](x: Scalar[dtype], y: Scalar[dtype]) capturing -> Scalar[dtype]:
        return 0.0

    def rfun[
        dtype: DType
    ](x: Scalar[dtype], y: Scalar[dtype]) capturing -> Scalar[dtype]:
        return 1.0

    def gfun[dtype: DType](x: Scalar[dtype]) capturing -> Scalar[dtype]:
        return 0.0

    def hfun[dtype: DType](x: Scalar[dtype]) capturing -> Scalar[dtype]:
        return 1.0

    # volume of the unit cube == 1
    var result = tplquad[sj.f64, f, qfun, rfun, gfun, hfun](0.0, 1.0, n=8)
    assert_almost_equal(result.integral, 1.0, atol=1e-8)


def test_tplquad_sum_over_unit_cube() raises:
    """Integral of (x + y + z) over the unit cube == 3/2 by symmetry (each
    term integrates to 1/2 over the other two unit axes).
    """

    def f[
        dtype: DType
    ](
        z: Scalar[dtype],
        y: Scalar[dtype],
        x: Scalar[dtype],
        args: Optional[List[Scalar[dtype]]],
    ) capturing -> Scalar[dtype]:
        return x + y + z

    def qfun[
        dtype: DType
    ](x: Scalar[dtype], y: Scalar[dtype]) capturing -> Scalar[dtype]:
        return 0.0

    def rfun[
        dtype: DType
    ](x: Scalar[dtype], y: Scalar[dtype]) capturing -> Scalar[dtype]:
        return 1.0

    def gfun[dtype: DType](x: Scalar[dtype]) capturing -> Scalar[dtype]:
        return 0.0

    def hfun[dtype: DType](x: Scalar[dtype]) capturing -> Scalar[dtype]:
        return 1.0

    var result = tplquad[sj.f64, f, qfun, rfun, gfun, hfun](0.0, 1.0, n=10)
    assert_almost_equal(result.integral, 1.5, atol=1e-8)


def test_nquad_2d_matches_dblquad() raises:
    """Nquad over a constant-bounds rectangle should match dblquad."""

    def f2[
        dtype: DType
    ](
        x: List[Scalar[dtype]], args: Optional[List[Scalar[dtype]]]
    ) capturing -> Scalar[dtype]:
        return x[0] * x[0] + x[1]

    var ranges: List[Tuple[Scalar[sj.f64], Scalar[sj.f64]]] = [
        (Scalar[sj.f64](0.0), Scalar[sj.f64](1.0)),
        (Scalar[sj.f64](0.0), Scalar[sj.f64](2.0)),
    ]
    var result = nquad[sj.f64, f2](ranges, n=20)
    # integral over [0,1]x[0,2] of (x^2 + y) dx dy
    # = integral_0^1 x^2 dx * 2 + integral_0^1 dx * (y^2/2 from 0 to 2)
    # = (1/3)*2 + 1*2 = 2/3 + 2 = 8/3
    assert_almost_equal(result.integral, 8.0 / 3.0, atol=1e-6)


def test_nquad_3d_unit_cube() raises:
    """Nquad over the unit cube of x*y*z == 1/8."""

    def f3[
        dtype: DType
    ](
        x: List[Scalar[dtype]], args: Optional[List[Scalar[dtype]]]
    ) capturing -> Scalar[dtype]:
        return x[0] * x[1] * x[2]

    var ranges: List[Tuple[Scalar[sj.f64], Scalar[sj.f64]]] = [
        (Scalar[sj.f64](0.0), Scalar[sj.f64](1.0)),
        (Scalar[sj.f64](0.0), Scalar[sj.f64](1.0)),
        (Scalar[sj.f64](0.0), Scalar[sj.f64](1.0)),
    ]
    var result = nquad[sj.f64, f3](ranges, n=10)
    assert_almost_equal(result.integral, 0.125, atol=1e-8)


def test_nquad_with_args() raises:
    def f[
        dtype: DType
    ](
        x: List[Scalar[dtype]], args: Optional[List[Scalar[dtype]]]
    ) capturing -> Scalar[dtype]:
        return args.value()[0] * x[0]

    var ranges: List[Tuple[Scalar[sj.f64], Scalar[sj.f64]]] = [
        (Scalar[sj.f64](0.0), Scalar[sj.f64](2.0))
    ]
    var arglist: List[Scalar[sj.f64]] = [5.0]
    var result = nquad[sj.f64, f](ranges, args=arglist^, n=10)
    # integral of 5x from 0 to 2 = 10
    assert_almost_equal(result.integral, 10.0, atol=1e-8)


def test_nquad_empty_ranges_raises() raises:
    def f[
        dtype: DType
    ](
        x: List[Scalar[dtype]], args: Optional[List[Scalar[dtype]]]
    ) capturing -> Scalar[dtype]:
        return 0.0

    var ranges = List[Tuple[Scalar[sj.f64], Scalar[sj.f64]]]()
    var raised = False
    try:
        _ = nquad[sj.f64, f](ranges)
    except:
        raised = True
    assert_true(raised, msg="empty ranges should raise")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
