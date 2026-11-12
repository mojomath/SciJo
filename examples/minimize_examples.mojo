"""
A tour of `scijo.optimize.minimize`: multi-dimensional minimization of a
scalar-valued function of several variables via the derivative-free
Nelder-Mead simplex or Powell methods, or the gradient-based BFGS, CG, and
L-BFGS-B methods (gradients estimated with finite differences). Powell and
L-BFGS-B also accept simple box constraints via `bounds=(lo, hi)`.

Run with (from the repository root, after `pixi run package`):

```bash
mojo run -I . -I tests/ examples/minimize_examples.mojo
```
"""

# ===----------------------------------------------------------------------=== #
# External
# ===----------------------------------------------------------------------=== #
import numojo as nm

# ===----------------------------------------------------------------------=== #
# SciJo
# ===----------------------------------------------------------------------=== #
from scijo.optimize import minimize
from scijo.prelude import *


def main() raises:
    quadratic_bowl()
    rosenbrock_example()
    with_args_example()
    bfgs_vs_cg_example()
    powell_and_lbfgsb_example()
    bounded_example()


# ===----------------------------------------------------------------------=== #
# Quadratic bowl: f(x, y) = (x - 3)^2 + (y + 1)^2, minimum at (3, -1)
# ===----------------------------------------------------------------------=== #


def quadratic_bowl_f[
    dtype: DType
](x: nm.NDArray[dtype]) capturing raises -> Scalar[dtype]:
    var a = x.item(0) - 3.0
    var b = x.item(1) + 1.0
    return a * a + b * b


def quadratic_bowl() raises:
    print("=" * 80)
    print("MINIMIZE: (x - 3)^2 + (y + 1)^2, minimum at (3, -1)")
    print("=" * 80)

    var x0 = nm.fromstring[f64]("[0, 0]")
    var result = minimize[f64, quadratic_bowl_f](x0)
    print(result)


# ===----------------------------------------------------------------------=== #
# Rosenbrock's banana function, minimum at (1, 1)
# ===----------------------------------------------------------------------=== #


def rosenbrock[
    dtype: DType
](x: nm.NDArray[dtype]) capturing raises -> Scalar[dtype]:
    var a = 1.0 - x.item(0)
    var b = x.item(1) - x.item(0) * x.item(0)
    return a * a + 100.0 * b * b


def rosenbrock_example() raises:
    print()
    print("=" * 80)
    print("MINIMIZE: Rosenbrock's function, minimum at (1, 1)")
    print("=" * 80)

    var x0 = nm.fromstring[f64]("[-1.2, 1.0]")
    var result = minimize[f64, rosenbrock](x0, maxiter=2000)
    print(result)


# ===----------------------------------------------------------------------=== #
# A reusable, module-level function that reads a coefficient from `args`
# ===----------------------------------------------------------------------=== #


def scaled_sphere[
    dtype: DType
](
    x: nm.NDArray[dtype], args: Optional[List[Scalar[dtype]]]
) capturing raises -> Scalar[dtype]:
    var a = args.value()[0]
    return a * (x.item(0) * x.item(0) + x.item(1) * x.item(1))


def with_args_example() raises:
    print()
    print("=" * 80)
    print("MINIMIZE: f(x) = a * ||x||^2, with args=[a]")
    print("=" * 80)

    var x0 = nm.fromstring[f64]("[1, 1]")
    var arglist: List[Scalar[f64]] = [2.0]
    var result = minimize[f64, scaled_sphere](x0, args=arglist^)
    print("With args=[2.0]:", result.x, " fun =", result.fun)


# ===----------------------------------------------------------------------=== #
# BFGS vs. CG on Rosenbrock's function
# ===----------------------------------------------------------------------=== #


def bfgs_vs_cg_example() raises:
    print()
    print("=" * 80)
    print("MINIMIZE: Rosenbrock's function via BFGS and CG")
    print("=" * 80)

    var x0 = nm.fromstring[f64]("[-1.2, 1.0]")

    var r_bfgs = minimize[f64, rosenbrock, method="BFGS"](x0, maxiter=200)
    print("BFGS:", r_bfgs.x, " fun =", r_bfgs.fun, " nit =", r_bfgs.nit)

    var r_cg = minimize[f64, rosenbrock, method="CG"](x0, maxiter=2000)
    print("CG:  ", r_cg.x, " fun =", r_cg.fun, " nit =", r_cg.nit)


# ===----------------------------------------------------------------------=== #
# Powell and L-BFGS-B on Rosenbrock's function
# ===----------------------------------------------------------------------=== #


def powell_and_lbfgsb_example() raises:
    print()
    print("=" * 80)
    print("MINIMIZE: Rosenbrock's function via Powell and L-BFGS-B")
    print("=" * 80)

    var x0 = nm.fromstring[f64]("[-1.2, 1.0]")

    var r_powell = minimize[f64, rosenbrock, method="Powell"](
        x0, maxiter=500
    )
    print(
        "Powell:  ",
        r_powell.x,
        " fun =",
        r_powell.fun,
        " nit =",
        r_powell.nit,
    )

    var r_lbfgsb = minimize[f64, rosenbrock, method="L-BFGS-B"](
        x0, maxiter=2000
    )
    print(
        "L-BFGS-B:",
        r_lbfgsb.x,
        " fun =",
        r_lbfgsb.fun,
        " nit =",
        r_lbfgsb.nit,
    )


# ===----------------------------------------------------------------------=== #
# Box-constrained minimization: the unconstrained minimum lies outside
# the box, so the bounded minimizer settles on the boundary instead.
# ===----------------------------------------------------------------------=== #


def bounded_example() raises:
    print()
    print("=" * 80)
    print("MINIMIZE: (x - 3)^2 + (y + 1)^2 with bounds x in [0, 2]")
    print("=" * 80)

    var x0 = nm.fromstring[f64]("[0, 0]")
    var lo = nm.fromstring[f64]("[0, -5]")
    var hi = nm.fromstring[f64]("[2, 5]")

    var r_powell = minimize[f64, quadratic_bowl_f, method="Powell"](
        x0, bounds=(lo.copy(), hi.copy())
    )
    print("Powell (bounded):  ", r_powell.x, " fun =", r_powell.fun)

    var r_lbfgsb = minimize[f64, quadratic_bowl_f, method="L-BFGS-B"](
        x0, bounds=(lo.copy(), hi.copy())
    )
    print("L-BFGS-B (bounded):", r_lbfgsb.x, " fun =", r_lbfgsb.fun)
