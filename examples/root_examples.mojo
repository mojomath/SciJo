"""
A tour of `scijo.optimize.root`: multi-dimensional root-finding via
Newton's method, with both a finite-difference and an analytic Jacobian.

Run with (from the repository root, after `pixi run package`):

```bash
mojo run -I . -I tests/ examples/root_examples.mojo
```
"""

# ===----------------------------------------------------------------------=== #
# External
# ===----------------------------------------------------------------------=== #
import numojo as nm

# ===----------------------------------------------------------------------=== #
# SciJo
# ===----------------------------------------------------------------------=== #
from scijo.linalg import matmul
from scijo.optimize import root
from scijo.prelude import *


def main() raises:
    linear_system()
    nonlinear_system()
    analytic_jacobian()


# ===----------------------------------------------------------------------=== #
# Linear system: Ax = b, solved as f(x) = Ax - b == 0
# ===----------------------------------------------------------------------=== #


def f_linear[
    dtype: DType
](x: nm.NDArray[dtype]) capturing raises -> nm.NDArray[dtype]:
    var A = nm.fromstring[dtype]("[[3, 1], [1, 2]]")
    var b = nm.fromstring[dtype]("[9, 8]")
    return matmul(A, x) - b


def linear_system() raises:
    var x0 = nm.fromstring[f64]("[0, 0]")
    var result = root[f64, f_linear](x0)
    print(result)


# ===----------------------------------------------------------------------=== #
# Nonlinear system: x^2 + y^2 = 4, x*y = 1
# ===----------------------------------------------------------------------=== #


def f_nonlinear[
    dtype: DType
](x: nm.NDArray[dtype]) capturing raises -> nm.NDArray[dtype]:
    var out = nm.zeros[dtype](nm.Shape(2))
    out.store(0, x.item(0) * x.item(0) + x.item(1) * x.item(1) - 4.0)
    out.store(1, x.item(0) * x.item(1) - 1.0)
    return out^


def nonlinear_system() raises:
    var x0 = nm.fromstring[f64]("[1.5, 1.0]")
    var result = root[f64, f_nonlinear](x0)
    print(result)


# ===----------------------------------------------------------------------=== #
# Same linear system, with an analytic Jacobian supplied
# ===----------------------------------------------------------------------=== #


def jac_linear[
    dtype: DType
](x: nm.NDArray[dtype]) capturing raises -> nm.NDArray[dtype]:
    return nm.fromstring[dtype]("[[3, 1], [1, 2]]")


def analytic_jacobian() raises:
    var x0 = nm.fromstring[f64]("[0, 0]")
    # An analytic Jacobian skips the finite-difference estimate every
    # iteration, and converges in a single Newton step on a linear system.
    var result = root[f64, f_linear, jac_linear](x0)
    print(result)
