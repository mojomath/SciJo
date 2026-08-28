# ===----------------------------------------------------------------------=== #
# SciJo: Differentiate module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Differentiate Module (`scijo.differentiate`)
===============================================
Provides tools for numerical differentiation and gradient computation.
It includes functions for calculating derivatives, Jacobians, with much more
to come in the future.

Available Functions
-------------------
- `derivative` — Compute first-order derivatives using finite differences.
- `jacobian`   — Compute the Jacobian matrix of a vector-valued function.
- `hessian`    — Compute the Hessian matrix of a scalar-valued function.

Examples
--------
    ```mojo
    from scijo.differentiate import derivative

    def f[dtype: DType](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[dtype]:
        return x * x

    var res = derivative[f64, f](1.0)
    ```
"""

# ===----------------------------------------------------------------------=== #
# SciJo
# ===----------------------------------------------------------------------=== #
from .deriv import derivative
from .jacob import (
    hessian,
    jacobian,
)
