"""
A tour of `scijo.optimize.least_squares` and `scijo.optimize.curve_fit`:
nonlinear least squares via Levenberg-Marquardt, both directly (a
residual function `f: R^n -> R^m`) and via the `curve_fit` convenience
wrapper (a model function evaluated against `(xdata, ydata)` data).

Run with (from the repository root, after `pixi run package`):

```bash
mojo run -I . -I tests/ examples/least_squares_examples.mojo
```
"""

# ===----------------------------------------------------------------------=== #
# External
# ===----------------------------------------------------------------------=== #
import numojo as nm

# ===----------------------------------------------------------------------=== #
# SciJo
# ===----------------------------------------------------------------------=== #
from scijo.optimize import curve_fit, least_squares
from scijo.prelude import *


def main() raises:
    overdetermined_linear_system()
    exponential_curve_fit()


# ===----------------------------------------------------------------------=== #
# least_squares: an overdetermined linear system, solved directly
# ===----------------------------------------------------------------------=== #


def residual[
    dtype: DType
](
    params: nm.NDArray[dtype], args: Optional[List[Scalar[dtype]]]
) capturing raises -> nm.NDArray[dtype] where dtype.is_floating_point():
    # Five noisy measurements of a line y = a*x + b (a=2, b=1).
    var xdata = nm.fromstring[dtype]("[0, 1, 2, 3, 4]")
    var ydata = nm.fromstring[dtype]("[1.1, 2.9, 5.1, 6.9, 9.2]")
    var a = params.item(0)
    var b = params.item(1)
    return a * xdata + b - ydata


def overdetermined_linear_system() raises:
    print()
    print("=" * 80)
    print("LEAST_SQUARES: fit y = a*x + b to 5 noisy points (expect a~=2, b~=1)")
    print("=" * 80)

    var p0 = nm.fromstring[f64]("[0, 0]")
    var result = least_squares[f64, residual](p0)
    print(result)


# ===----------------------------------------------------------------------=== #
# curve_fit: an exponential model, fit via the convenience wrapper
# ===----------------------------------------------------------------------=== #


def exponential_model[
    dtype: DType
](
    xdata: nm.NDArray[dtype], params: nm.NDArray[dtype]
) capturing raises -> nm.NDArray[dtype] where dtype.is_floating_point():
    return params.item(0) * nm.exp(params.item(1) * xdata)


def exponential_curve_fit() raises:
    print()
    print("=" * 80)
    print("CURVE_FIT: fit y = a*exp(b*x) (expect a~=2, b~=0.5)")
    print("=" * 80)

    var xdata = nm.fromstring[f64]("[0, 1, 2, 3, 4, 5]")
    var ydata = 2.0 * nm.exp(0.5 * xdata)
    var p0 = nm.fromstring[f64]("[1, 1]")

    var result = curve_fit[f64, exponential_model](xdata, ydata, p0)
    print(result)
