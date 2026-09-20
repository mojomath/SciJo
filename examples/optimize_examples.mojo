"""
A tour of `scijo.optimize`: scalar root-finding (`root_scalar` and the
individual bisect/newton/secant/brent solvers) and scalar minimization
(`minimize_scalar`).

Run with (from the repository root, after `pixi run package`):

```bash
mojo run -I . -I tests/ examples/optimize_examples.mojo
```
"""

from scijo.optimize import (
    root_scalar,
    bisect,
    brent,
    secant,
    newton,
    minimize_scalar,
)
from scijo.prelude import *


def main() raises:
    root_finding()
    newton_method()
    minimization()


# ===----------------------------------------------------------------------=== #
# Root finding: f(x) = x^2 - 2, root at sqrt(2)
# ===----------------------------------------------------------------------=== #


def f[
    dtype: DType
](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) capturing -> Scalar[
    dtype
]:
    return x * x - 2.0


def fprime[
    dtype: DType
](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) capturing -> Scalar[
    dtype
]:
    return 2.0 * x


def root_finding() raises:
    print("=" * 80)
    print("ROOT_SCALAR: f(x) = x^2 - 2, root at sqrt(2) ~= 1.41421356")
    print("=" * 80)

    var r_bisect = root_scalar[f64, f, method="bisect"](bracket=(0.0, 2.0))
    print("method='bisect':", r_bisect.root, " success =", r_bisect.success)

    var r_brent = root_scalar[f64, f, method="brent"](bracket=(0.0, 2.0))
    print("method='brent': ", r_brent.root, " success =", r_brent.success)

    var r_secant = root_scalar[f64, f, method="secant"](x0=1.0, x1=2.0)
    print("method='secant':", r_secant.root, " success =", r_secant.success)

    # Calling a solver directly, without going through root_scalar.
    var r_direct = bisect[f64, f](None, (0.0, 2.0))
    print("bisect(...) directly:", r_direct.root)

    # A missing method-specific argument raises.
    try:
        _ = root_scalar[f64, f, method="bisect"]()
    except e:
        print("root_scalar without a bracket for method='bisect' raises:")
        print(" ", e)


def newton_method() raises:
    print()
    print("=" * 80)
    print(
        "NEWTON: needs a derivative, so it's a different root_scalar overload"
    )
    print("=" * 80)

    # Newton needs `fprime`, so it goes through the two-function overload.
    var r_newton = root_scalar[f64, f, fprime, method="newton"](x0=1.0)
    print("method='newton':", r_newton.root, " nit =", r_newton.nit)

    var r_newton_direct = newton[f64, f, fprime](
        None, x0=1.0, atol=1e-12, rtol=1e-12
    )
    print("newton(...) directly:", r_newton_direct.root)


# ===----------------------------------------------------------------------=== #
# Minimization: (x - 2)^2 + 1, minimum at x = 2
# ===----------------------------------------------------------------------=== #


def objective[
    dtype: DType
](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) capturing -> Scalar[
    dtype
]:
    return (x - 2.0) * (x - 2.0) + 1.0


def minimization() raises:
    print()
    print("=" * 80)
    print("MINIMIZE_SCALAR: (x - 2)^2 + 1, minimum at x = 2")
    print("=" * 80)

    var r_brent = minimize_scalar[f64, objective, method="Brent"](
        bracket=(0.0, 4.0), atol=1e-8, maxiter=100
    )
    print("method='Brent':")
    print(r_brent)

    var r_golden = minimize_scalar[f64, objective, method="Golden"](
        bracket=(0.0, 4.0)
    )
    print("method='Golden': x =", r_golden.x, " fun =", r_golden.fun)

    # Bounded requires `bounds=`, and never evaluates outside it.
    var r_bounded = minimize_scalar[f64, objective, method="Bounded"](
        bounds=(0.0, 1.5)
    )
    print("method='Bounded' on [0, 1.5] (minimum outside range):")
    print("  x =", r_bounded.x, " fun =", r_bounded.fun)

    # Bounded requires bounds=, not bracket=.
    try:
        _ = minimize_scalar[f64, objective, method="Bounded"](
            bracket=(0.0, 4.0)
        )
    except e:
        print("method='Bounded' with bracket= instead of bounds= raises:")
        print(" ", e)
