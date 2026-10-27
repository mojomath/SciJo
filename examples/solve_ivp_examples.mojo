"""
A tour of `scijo.integrate.solve_ivp`: adaptive-step integration of
`dy/dt = f(t, y)` with RK45 (Dormand-Prince). Two closed-form problems
below double as a correctness check - the printed values should match
the comments next to them.

```bash
mojo run -I . -I tests/ examples/solve_ivp_examples.mojo
```
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.math import cos, exp, sin

# ===----------------------------------------------------------------------=== #
# External
# ===----------------------------------------------------------------------=== #
import numojo as nm

# ===----------------------------------------------------------------------=== #
# SciJo
# ===----------------------------------------------------------------------=== #
from scijo.integrate import solve_ivp
from scijo.prelude import *


def main() raises:
    exponential_decay()
    simple_harmonic_motion()


# ===----------------------------------------------------------------------=== #
# Exponential decay: dy/dt = -y, y(0) = 1, exact solution y(t) = exp(-t).
# ===----------------------------------------------------------------------=== #


def decay[
    dtype: DType
](t: Scalar[dtype], y: nm.NDArray[dtype]) raises -> nm.NDArray[dtype]:
    return y * (-1.0)


def exponential_decay() raises:
    print("=" * 80)
    print("Exponential decay: dy/dt = -y, y(0) = 1")
    print("=" * 80)

    var y0 = nm.fromstring[f64]("[1.0]")
    var result = solve_ivp[f64, decay]((0.0, 2.0), y0)

    var n_last = result.t.shape[0] - 1
    print("steps taken      =", result.nsteps)
    print(
        "y(2.0)           =",
        result.y.item(0, n_last),
        " (expect",
        exp(-2.0),
        ")",
    )


# ===----------------------------------------------------------------------=== #
# Simple harmonic motion: y'' = -y as [theta, omega]' = [omega, -theta].
# Exact solution theta(t) = cos(t), omega(t) = -sin(t); theta^2 + omega^2
# stays exactly 1 (energy conservation).
# ===----------------------------------------------------------------------=== #


def shm[
    dtype: DType
](t: Scalar[dtype], y: nm.NDArray[dtype]) raises -> nm.NDArray[dtype]:
    var out = nm.zeros[dtype](nm.Shape(2))
    out.store(0, val=y.load(1))
    out.store(1, val=-y.load(0))
    return out^


def simple_harmonic_motion() raises:
    print("=" * 80)
    print("Simple harmonic motion: y'' = -y, y(0) = 1, y'(0) = 0")
    print("=" * 80)

    var y0 = nm.fromstring[f64]("[1.0, 0.0]")
    var t_eval = nm.fromstring[f64]("[0.5, 1.0, 1.5, 2.0]")
    var result = solve_ivp[f64, shm]((0.0, 2.0), y0, t_eval=t_eval^)

    for j in range(result.t.shape[0]):
        var t = result.t.item(j)
        var theta = result.y.item(0, j)
        var omega = result.y.item(1, j)
        print(
            "t =",
            t,
            " theta =",
            theta,
            "(expect",
            cos(t),
            ") omega =",
            omega,
            "(expect",
            -sin(t),
            ") theta^2+omega^2 =",
            theta * theta + omega * omega,
            "(expect 1.0)",
        )
