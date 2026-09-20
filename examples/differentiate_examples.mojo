"""
A tour of `scijo.differentiate`: numerical derivatives, Jacobians and
Hessians via finite differences.

Run with (from the repository root, after `pixi run package`):

```bash
mojo run -I . -I tests/ examples/differentiate_examples.mojo
```
"""

import numojo as nm
from scijo.differentiate import derivative, jacobian, hessian
from scijo.prelude import *


def main() raises:
    first_derivative()
    step_direction_variants()
    jacobian_example()
    hessian_example()


# ===----------------------------------------------------------------------=== #
# derivative: f(x) = x^2 + 2x + 1, f'(x) = 2x + 2
# ===----------------------------------------------------------------------=== #


def parabola[
    dtype: DType
](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) capturing -> Scalar[
    dtype
]:
    return x * x + 2.0 * x + 1.0


def first_derivative() raises:
    print("=" * 80)
    print("DERIVATIVE: central differences (default)")
    print("=" * 80)

    # f'(1) should be 2*1 + 2 = 4
    var res = derivative[f64, parabola](x0=1.0)
    print("derivative[f64, parabola](x0=1.0):")
    print("  df      =", res.df)
    print("  success =", res.success)
    print("  nit     =", res.nit)
    print("  nfev    =", res.nfev)

    # A function that reads extra coefficients from `args`.
    var args: List[Scalar[f64]] = [3.0]
    var res_args = derivative[f64, scaled_parabola](x0=1.0, args=args^)
    print("With args=[3.0] (f(x) = 3*x^2):", res_args.df)  # 3*2*1 = 6


def scaled_parabola[
    dtype: DType
](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) capturing -> Scalar[
    dtype
]:
    var a = args.value()[0]
    return a * x * x


def step_direction_variants() raises:
    print()
    print("=" * 80)
    print("DERIVATIVE: forward and backward differences")
    print("=" * 80)

    # step_direction is a compile-time, keyword-only parameter: 0 = central
    # (default), 1 = forward, -1 = backward.
    var res_fwd = derivative[f64, parabola, step_direction=1](x0=0.0, order=4)
    print("Forward differences at x0=0:  ", res_fwd.df)  # 2*0 + 2 = 2

    var res_bwd = derivative[f64, parabola, step_direction=-1](x0=5.0, order=4)
    print("Backward differences at x0=5: ", res_bwd.df)  # 2*5 + 2 = 12


# ===----------------------------------------------------------------------=== #
# jacobian: f(x, y) = [x^2, y^2]
# ===----------------------------------------------------------------------=== #


def vector_square[
    dtype: DType
](
    x: NDArray[dtype], args: Optional[List[Scalar[dtype]]]
) capturing raises -> NDArray[dtype]:
    return x * x


def jacobian_example() raises:
    print()
    print("=" * 80)
    print("JACOBIAN")
    print("=" * 80)

    var x = nm.array[f64]([1.0, 2.0], [2])
    var J = jacobian[f64, vector_square](x)
    print("jacobian(f(x)=x*x) at [1, 2] (diagonal should be ~[2, 4]):")
    print(J)


# ===----------------------------------------------------------------------=== #
# hessian: g(x, y) = x^2 + y^2
# ===----------------------------------------------------------------------=== #


def sum_of_squares[
    dtype: DType
](
    x: NDArray[dtype], args: Optional[List[Scalar[dtype]]]
) capturing raises -> Scalar[dtype]:
    return x.item(0) * x.item(0) + x.item(1) * x.item(1)


def hessian_example() raises:
    print()
    print("=" * 80)
    print("HESSIAN")
    print("=" * 80)

    var x = nm.array[f64]([1.0, 2.0], [2])
    var H = hessian[f64, sum_of_squares](x)
    print("hessian(g(x,y)=x^2+y^2) at [1, 2] (should be ~[[2, 0], [0, 2]]):")
    print(H)
