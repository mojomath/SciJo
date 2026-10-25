"""
A tour of `scijo.special`: regular users call `erf`/`gamma`/Bessel directly
on a `Scalar`, no `Numeric` required. Power users who want an exact
derivative call the exact same function on a `Dual` instead - there is no
separate derivative formula anywhere in `scijo.special`; it falls out of
`Dual`'s own chain-rule arithmetic automatically (see
`examples/differentiate_examples.mojo` for the finite-difference
alternative to the same derivative, for comparison).

```bash
mojo run -I . -I tests/ examples/special_examples.mojo
```
"""

# ===----------------------------------------------------------------------=== #
# SciJo
# ===----------------------------------------------------------------------=== #
from scijo.special import (
    airy_ai,
    bessel_j0,
    bessel_j1,
    erf,
    gamma,
    gamma_inc,
    legendre,
)
from scijo.types import NumericComparable, variable
from scijo.prelude import *


def main() raises:
    plain_scalar_calls()
    exact_derivative_via_dual()


# ===----------------------------------------------------------------------=== #
# Level 1: plain `Scalar` calls, no `Numeric` in sight.
# ===----------------------------------------------------------------------=== #


def plain_scalar_calls() raises:
    print("=" * 80)
    print("LEVEL 1: scijo.special on a plain Scalar")
    print("=" * 80)

    print("gamma(5.0)       =", gamma(Scalar[f64](5.0)), " (expect 24.0 == 4!)")
    print("erf(1.0)         =", erf(Scalar[f64](1.0)), " (expect 0.8427...)")
    print("bessel_j0(2.0)   =", bessel_j0(Scalar[f64](2.0)), " (expect 0.2239...)")
    print(
        "gamma_inc(2, 1)  =",
        gamma_inc(Scalar[f64](2.0), Scalar[f64](1.0)),
        " (expect 0.2642...)",
    )
    print("airy_ai(0.0)     =", airy_ai(Scalar[f64](0.0)), " (expect 0.3550...)")
    print("legendre(2, 0.5) =", legendre(2, Scalar[f64](0.5)), " (expect -0.125)")


# ===----------------------------------------------------------------------=== #
# Level 2: the same `bessel_j0`, called through `Dual` for a free exact
# derivative - `d/dx J0(x) = -J1(x)`, no finite-difference step size to
# tune.
# ===----------------------------------------------------------------------=== #


def j0_kernel[T: NumericComparable](x: T) -> T:
    return bessel_j0(x)


def exact_derivative_via_dual() raises:
    print()
    print("=" * 80)
    print("LEVEL 2: the same bessel_j0, called through Dual for a free derivative")
    print("=" * 80)

    var x0 = 3.0
    var result = j0_kernel(variable(x0))
    print("J0(3.0)          =", result.value.v)
    print("J0'(3.0)         =", result.deriv.v, " (expect -J1(3.0) =",
        -bessel_j1(Scalar[f64](x0)), ")")
