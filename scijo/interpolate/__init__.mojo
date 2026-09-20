# ===----------------------------------------------------------------------=== #
# SciJo: Interpolate module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Interpolate Module (`scijo.interpolate`)
===========================================
Provides tools for interpolating data. It includes linear interpolation
functions and callable interpolator objects for both single-point and
array-based evaluation.

Available Functions
-------------------
- `interp1d`              — Create a callable linear interpolator or interpolate directly.
- `interp`                — Functional interpolation interface (linear, cubic, akima).
- `LinearInterpolator`    — A reusable callable linear interpolation object.
- `CubicSpline`           — A callable natural cubic spline interpolator.
- `Akima1DInterpolator`   — A callable Akima piecewise cubic interpolator.

Examples
--------
    ```mojo
    from scijo.interpolate import interp1d, CubicSpline, Akima1DInterpolator

    var x = nm.arange[f64](0.0, 1.0, 0.5)
    var y = x * x
    var interp = interp1d(x, y, bounds_error=False, fill_value=0.0)
    var yq = interp(Scalar[f64](0.25))

    var cs = CubicSpline(x, y)
    var yq_cs = cs(Scalar[f64](0.25))

    var ak = Akima1DInterpolator(x, y)
    var yq_ak = ak(Scalar[f64](0.25))
    ```
"""

# ===----------------------------------------------------------------------=== #
# SciJo
# ===----------------------------------------------------------------------=== #
from .interpolate import (
    Akima1DInterpolator,
    CubicSpline,
    interp,
    interp1d,
    LinearInterpolator,
)
