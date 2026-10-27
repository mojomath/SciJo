"""
Tests for initial value problem solving (scijo.integrate.solve_ivp, RK45).

To run: `mojo test tests/test_solve_ivp.mojo -I .` from the project root directory.
"""

from std.testing import (
    assert_almost_equal,
    assert_true,
    TestSuite,
)
from std.math import cos, exp, sin

import numojo as nm

import scijo as sj
from scijo.integrate import solve_ivp


def test_exponential_decay() raises:
    """`dy/dt = -k*y` has the closed-form solution `y0 * exp(-k*t)`."""

    def decay[
        dtype: DType
    ](
        t: Scalar[dtype],
        y: nm.NDArray[dtype],
        args: Optional[List[Scalar[dtype]]],
    ) capturing raises -> nm.NDArray[dtype]:
        var k = args.value()[0]
        return y * (-k)

    var y0 = nm.fromstring[sj.f64]("[2.0]")
    var k_args: List[Scalar[sj.f64]] = [3.0]
    var result = solve_ivp[sj.f64, decay](
        (0.0, 1.0), y0, args=k_args^, atol=1e-10, rtol=1e-10
    )
    assert_true(result.success)
    var n_last = result.t.shape[0] - 1
    assert_almost_equal(result.t.item(n_last), 1.0, atol=1e-10)
    assert_almost_equal(result.y.item(0, n_last), 2.0 * exp(-3.0), atol=1e-6)


def test_shm_energy_conserved() raises:
    """`y'' = -y` as `[theta, omega]' = [omega, -theta]`: exact solution
    `theta(t) = cos(t)`, `omega(t) = -sin(t)`, and `theta^2 + omega^2 == 1`
    (energy conservation) at every point."""

    def shm[
        dtype: DType
    ](t: Scalar[dtype], y: nm.NDArray[dtype]) capturing raises -> nm.NDArray[
        dtype
    ]:
        var out = nm.zeros[dtype](nm.Shape(2))
        out.store(0, val=y.load(1))
        out.store(1, val=-y.load(0))
        return out^

    var y0 = nm.fromstring[sj.f64]("[1.0, 0.0]")
    var t_eval = nm.fromstring[sj.f64]("[0.5, 1.0, 1.5, 2.0]")
    var result = solve_ivp[sj.f64, shm](
        (0.0, 2.0), y0, t_eval=t_eval^, atol=1e-10, rtol=1e-10
    )
    assert_true(result.success)

    for j in range(4):
        var t = result.t.item(j)
        assert_almost_equal(result.y.item(0, j), cos(t), atol=1e-6)
        assert_almost_equal(result.y.item(1, j), -sin(t), atol=1e-6)
        var energy = result.y.item(0, j) * result.y.item(0, j) + result.y.item(
            1, j
        ) * result.y.item(1, j)
        assert_almost_equal(energy, 1.0, atol=1e-6)


def test_linear_system() raises:
    """`dy/dt = diag(-1, -2) @ y` decouples into two independent exponentials.
    """

    def linsys[
        dtype: DType
    ](t: Scalar[dtype], y: nm.NDArray[dtype]) capturing raises -> nm.NDArray[
        dtype
    ]:
        var out = nm.zeros[dtype](nm.Shape(2))
        out.store(0, val=-y.load(0))
        out.store(1, val=-2.0 * y.load(1))
        return out^

    var y0 = nm.fromstring[sj.f64]("[1.0, 1.0]")
    var result = solve_ivp[sj.f64, linsys](
        (0.0, 1.0), y0, atol=1e-10, rtol=1e-10
    )
    assert_true(result.success)
    var n_last = result.t.shape[0] - 1
    assert_almost_equal(result.y.item(0, n_last), exp(-1.0), atol=1e-6)
    assert_almost_equal(result.y.item(1, n_last), exp(-2.0), atol=1e-6)


def test_fast_decay_converges_despite_rejections() raises:
    """A fast-decaying problem forces the adaptive controller to shrink (and
    sometimes reject) steps; the solver should still converge to the exact
    solution."""

    def fast_decay[
        dtype: DType
    ](t: Scalar[dtype], y: nm.NDArray[dtype]) capturing raises -> nm.NDArray[
        dtype
    ]:
        return y * (-50.0)

    var y0 = nm.fromstring[sj.f64]("[1.0]")
    var result = solve_ivp[sj.f64, fast_decay](
        (0.0, 1.0), y0, atol=1e-12, rtol=1e-10
    )
    assert_true(result.success)
    assert_true(result.nsteps > 1, "Expected more than one step")
    var n_last = result.t.shape[0] - 1
    assert_almost_equal(result.y.item(0, n_last), exp(-50.0), atol=1e-9)


def test_t_eval_interpolation() raises:
    """`t_eval` at points off the natural step grid still tracks the
    closed-form solution, within the (looser) interpolation tolerance."""

    def decay[
        dtype: DType
    ](t: Scalar[dtype], y: nm.NDArray[dtype]) capturing raises -> nm.NDArray[
        dtype
    ]:
        return y * (-1.0)

    var y0 = nm.fromstring[sj.f64]("[1.0]")
    var t_eval = nm.fromstring[sj.f64]("[0.3, 0.7, 1.1, 1.5]")
    var result = solve_ivp[sj.f64, decay](
        (0.0, 2.0), y0, t_eval=t_eval^, atol=1e-10, rtol=1e-10
    )
    assert_true(result.success)
    assert_true(result.t.shape[0] == 4)

    for j in range(4):
        var t = result.t.item(j)
        assert_almost_equal(result.y.item(0, j), exp(-t), atol=1e-5)


def test_first_step_and_max_step_respected() raises:
    """An explicit `first_step`/`max_step` still converges to the oracle,
    and no accepted step exceeds `max_step`."""

    def decay[
        dtype: DType
    ](t: Scalar[dtype], y: nm.NDArray[dtype]) capturing raises -> nm.NDArray[
        dtype
    ]:
        return y * (-1.0)

    var y0 = nm.fromstring[sj.f64]("[1.0]")
    var result = solve_ivp[sj.f64, decay](
        (0.0, 1.0), y0, first_step=0.01, max_step=0.05
    )
    assert_true(result.success)

    var n = result.t.shape[0]
    for j in range(1, n):
        var dt = result.t.item(j) - result.t.item(j - 1)
        assert_true(dt <= 0.05 + 1e-9, "Step exceeded max_step")

    var n_last = n - 1
    assert_almost_equal(result.y.item(0, n_last), exp(-1.0), atol=1e-6)


def test_unsupported_method_raises() raises:
    """An unsupported method should raise, not silently do the wrong thing."""

    def decay[
        dtype: DType
    ](t: Scalar[dtype], y: nm.NDArray[dtype]) capturing raises -> nm.NDArray[
        dtype
    ]:
        return y * (-1.0)

    var y0 = nm.fromstring[sj.f64]("[1.0]")
    var raised = False
    try:
        _ = solve_ivp[sj.f64, decay, method="RK23"]((0.0, 1.0), y0)
    except:
        raised = True
    assert_true(raised, "Should raise error for an unsupported method")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
