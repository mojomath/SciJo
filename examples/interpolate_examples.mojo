"""
A tour of `scijo.interpolate`: linear, cubic-spline and Akima 1-D
interpolation, both as callable objects and via the functional `interp`.

Run with (from the repository root, after `pixi run package`):

```bash
mojo run -I . -I tests/ examples/interpolate_examples.mojo
```
"""

import numojo as nm
from scijo.interpolate import (
    interp1d,
    LinearInterpolator,
    CubicSpline,
    Akima1DInterpolator,
    interp,
)
from scijo.prelude import *


def main() raises:
    linear_interpolation()
    bounds_behavior()
    cubic_and_akima()
    functional_interp()


# ===----------------------------------------------------------------------=== #
# Linear interpolation
# ===----------------------------------------------------------------------=== #


def linear_interpolation() raises:
    print("=" * 80)
    print("LINEAR INTERPOLATION: interp1d / LinearInterpolator")
    print("=" * 80)

    var x = nm.arange[f64](0.0, 1.5, 0.5)  # [0.0, 0.5, 1.0]
    var y = nm.array[f64]([0.0, 0.25, 1.0], [3])  # samples of y ~= x^2

    var li = interp1d(x, y)
    print("li(0.25) =", li(Scalar[f64](0.25)))  # halfway between 0.0 and 0.25
    print("li([0.1, 0.5, 0.9]) =", li(nm.array[f64]([0.1, 0.5, 0.9], [3])))


def bounds_behavior() raises:
    print()
    print("=" * 80)
    print("OUT-OF-BOUNDS BEHAVIOR: bounds_error and fill_value")
    print("=" * 80)

    var x = nm.arange[f64](0.0, 1.5, 0.5)
    var y = nm.array[f64]([0.0, 0.25, 1.0], [3])

    # Default: bounds_error=True raises outside [x.min(), x.max()].
    var li_strict = interp1d(x, y)
    try:
        _ = li_strict(Scalar[f64](5.0))
    except e:
        print("bounds_error=True on x=5.0 raises:")
        print(" ", e)

    # bounds_error=False, fill_value given: returns the constant fill value.
    var li_fill = interp1d(x, y, bounds_error=False, fill_value=-1.0)
    print(
        "bounds_error=False, fill_value=-1.0, li(5.0) =",
        li_fill(Scalar[f64](5.0)),
    )

    # bounds_error=False, fill_value=None (default): clamps to the boundary y.
    var li_clamp = interp1d(x, y, bounds_error=False)
    print(
        "bounds_error=False, fill_value=None, li(5.0) =",
        li_clamp(Scalar[f64](5.0)),
    )
    print(
        "                                    li(-5.0) =",
        li_clamp(Scalar[f64](-5.0)),
    )


# ===----------------------------------------------------------------------=== #
# CubicSpline / Akima1DInterpolator
# ===----------------------------------------------------------------------=== #


def cubic_and_akima() raises:
    print()
    print("=" * 80)
    print("CUBIC SPLINE / AKIMA")
    print("=" * 80)

    var x = nm.linspace[f64](0.0, 10.0, 11)
    var y = x * x

    var cs = CubicSpline(x, y)
    print("CubicSpline at [0.5 .. 9.5]:")
    print(cs(nm.linspace[f64](0.5, 9.5, 5)))

    var ak = Akima1DInterpolator(x, y)
    print("Akima1DInterpolator at 3.7:", ak(Scalar[f64](3.7)))

    # Both clamp outside the data range instead of raising.
    print("CubicSpline at -5.0 (clamped):", cs(Scalar[f64](-5.0)))


# ===----------------------------------------------------------------------=== #
# Functional interp()
# ===----------------------------------------------------------------------=== #


def functional_interp() raises:
    print()
    print("=" * 80)
    print("FUNCTIONAL interp(): one-shot interpolation")
    print("=" * 80)

    var x = nm.arange[f64](0.0, 1.5, 0.5)
    var y = x * x
    var xq = nm.array[f64]([0.1, 0.5, 0.9], [3])

    var yq_linear = interp[f64, type="linear", fill_method="interpolate"](
        xq, x, y
    )
    print("interp, linear/interpolate:", yq_linear)

    var yq_extrap = interp[f64, type="linear", fill_method="extrapolate"](
        nm.array[f64]([-1.0, 0.5, 5.0], [3]), x, y
    )
    print("interp, linear/extrapolate at [-1, 0.5, 5]:", yq_extrap)

    var yq_cubic = interp[f64, type="cubic"](xq, x, y)
    print("interp, cubic:", yq_cubic)

    var yq_akima = interp[f64, type="akima"](xq, x, y)
    print("interp, akima:", yq_akima)

    # "extrapolate" is only supported for type="linear".
    try:
        _ = interp[f64, type="cubic", fill_method="extrapolate"](xq, x, y)
    except e:
        print("cubic + extrapolate raises:")
        print(" ", e)
