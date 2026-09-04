"""
A tour of `scijo.integrate`: adaptive quadrature of a function (`quad`) and
fixed-sample integration of discrete data (`trapezoid`, `simpson`, `romb`,
and their cumulative variants).

Run with (from the repository root, after `pixi run package`):

```bash
mojo run -I . -I tests/ examples/integrate_examples.mojo
```
"""

import numojo as nm
from scijo.integrate import (
    quad,
    QAG_GK61,
    trapezoid,
    simpson,
    romb,
    cumulative_trapezoid,
    cumulative_simpson,
)
from scijo.prelude import *


def main() raises:
    quad_example()
    fixed_sample_example()
    romberg_example()
    cumulative_example()


# ===----------------------------------------------------------------------=== #
# quad: integrate a function, x^2 from 0 to 1 -> 1/3
# ===----------------------------------------------------------------------=== #


def integrand[
    dtype: DType
](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) capturing -> Scalar[
    dtype
]:
    return x * x


def quad_example() raises:
    print("=" * 80)
    print("QUAD: adaptive quadrature of x^2 over [0, 1]")
    print("=" * 80)

    var result = quad[f64, integrand](a=0.0, b=1.0)
    print(
        "method='qng' (default):  integral =",
        result.integral,
        " abserr =",
        result.abserr,
    )

    var result_hi = quad[f64, integrand, method="qag", qag_rule=QAG_GK61](
        a=0.0, b=1.0, atol=1e-10, rtol=1e-10
    )
    print("method='qag', GK61:      integral =", result_hi.integral)

    var result_qags = quad[f64, integrand, method="qags"](a=0.0, b=1.0)
    print("method='qags':           integral =", result_qags.integral)


# ===----------------------------------------------------------------------=== #
# Fixed-sample rules on discrete data
# ===----------------------------------------------------------------------=== #


def fixed_sample_example() raises:
    print()
    print("=" * 80)
    print("TRAPEZOID / SIMPSON on discrete samples of y = x^2")
    print("=" * 80)

    var y = nm.linspace[f64](0.0, 10.0, 100) ** 2

    var area_trap = trapezoid(y, dx=10.0 / 99.0)
    var area_sim = simpson(y, dx=10.0 / 99.0)
    print(
        "trapezoid(y, dx):", area_trap
    )  # ~= integral of x^2 from 0 to 10 = 333.33
    print("simpson(y, dx):  ", area_sim)

    # Non-uniform spacing: pass x explicitly instead of dx.
    var x = nm.linspace[f64](0.0, 10.0, 100)
    var area_trap_x = trapezoid(y, x)
    var area_sim_x = simpson(y, x)
    print("trapezoid(y, x): ", area_trap_x)
    print("simpson(y, x):   ", area_sim_x)


def romberg_example() raises:
    print()
    print("=" * 80)
    print("ROMBERG: requires 2^k + 1 points")
    print("=" * 80)

    # 9 = 2^3 + 1 points.
    var y = nm.linspace[f64](0.0, 1.0, 9) ** 2
    var area = romb(y, dx=0.125)
    print("romb(y, dx=0.125), y=x^2 over [0,1]:", area)  # ~= 1/3


# ===----------------------------------------------------------------------=== #
# Cumulative integration
# ===----------------------------------------------------------------------=== #


def cumulative_example() raises:
    print()
    print("=" * 80)
    print("CUMULATIVE TRAPEZOID / SIMPSON")
    print("=" * 80)

    var y = nm.array[f64]([1.0, 2.0, 3.0, 4.0], [4])

    # With `initial`, the output is the same length as y.
    var cum = cumulative_trapezoid(y, dx=1.0, initial=0.0)
    print("cumulative_trapezoid(y, dx=1.0, initial=0.0):")
    print(cum)  # [0.0, 1.5, 4.0, 7.5]

    # Without `initial`, the output is one element shorter.
    var cum_short = cumulative_trapezoid(y, dx=1.0)
    print("cumulative_trapezoid(y, dx=1.0) (no initial):")
    print(cum_short)  # [1.5, 4.0, 7.5]

    var y_odd = nm.array[f64]([1.0, 4.0, 1.0, 4.0, 1.0], [5])
    var cum_simpson = cumulative_simpson(y_odd, dx=1.0, initial=0.0)
    print("cumulative_simpson(y_odd, dx=1.0, initial=0.0):")
    print(cum_simpson)
