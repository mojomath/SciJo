# ===----------------------------------------------------------------------=== #
# SciJo: Optimize module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Optimize Module (`scijo.optimize`)
=====================================
Provides tools for numerical optimization and root-finding. It includes scalar
root-finding methods such as bisection, Newton-Raphson, and the secant method,
as well as scalar minimization using Brent's method, golden section search,
and bounded minimization.

Available Functions
-------------------
- `root_scalar`       — Find a root of a scalar function.
- `newton`            — Newton-Raphson root-finding method.
- `bisect`            — Bisection root-finding method.
- `secant`            — Secant root-finding method.
- `brent`             — Brent root-finding method.
- `minimize_scalar`   — Minimize a scalar function.

Examples
--------
    ```mojo
    from scijo.optimize import root_scalar, minimize_scalar

    def f[dtype: DType](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[dtype]:
        return x * x - 2

    var root = root_scalar[f64, f](bracket=(1.0, 2.0), method="bisect")
    ```
"""

# ===----------------------------------------------------------------------=== #
# SciJo
# ===----------------------------------------------------------------------=== #
from .min_scalar import (
    minimize_scalar,
    OptimizeResult,
)
from .root_scalar import (
    bisect,
    brent,
    newton,
    root_scalar,
    secant,
)
from .utility import RootResult
