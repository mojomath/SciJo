# ===----------------------------------------------------------------------=== #
# SciJo: Integrate module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Integrate Module (`scijo.integrate`)
=======================================
Provides tools for numerical integration and quadrature. It includes adaptive
and non-adaptive methods for computing definite integrals, as well as
fixed-sample integration rules for discrete data.

Available Functions
-------------------
- `quad`        — General-purpose adaptive quadrature (Gauss-Kronrod).
- `trapezoid`   — Composite trapezoidal rule for discrete data.
- `simpson`     — Simpson's rule for discrete data.
- `romb`        — Romberg integration with Richardson extrapolation.

QAG Rule Constants (use as `qag_rule=` when `method="qag"`)
------------------------------------------------------------
- `QAG_GK15`    — 7-pt Gauss / 15-pt Kronrod
- `QAG_GK21`    — 10-pt Gauss / 21-pt Kronrod (default)
- `QAG_GK31`    — 15-pt Gauss / 31-pt Kronrod
- `QAG_GK41`    — 20-pt Gauss / 41-pt Kronrod
- `QAG_GK51`    — 25-pt Gauss / 51-pt Kronrod
- `QAG_GK61`    — 30-pt Gauss / 61-pt Kronrod

Examples
--------
    ```mojo
    from scijo.integrate import quad, trapezoid, QAG_GK61
    from scijo.prelude import *

    def integrand[dtype: DType](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[dtype]:
        return x * x

    var result = quad[f64, integrand](0.0, 1.0, None)
    var result_hi = quad[f64, integrand, method="qag", qag_rule=QAG_GK61](0.0, 1.0, None)
    ```
"""

from .quadrature import quad, QAG_GK15, QAG_GK21, QAG_GK31, QAG_GK41, QAG_GK51, QAG_GK61
from .fixed_sample import (
    trapezoid,
    simpson,
    romb,
    cumulative_trapezoid,
    cumulative_simpson,
)
