"""
A tour of `scijo.optimize.minimize`: multi-dimensional minimization of a
scalar-valued function of several variables via the Nelder-Mead simplex
method.

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
