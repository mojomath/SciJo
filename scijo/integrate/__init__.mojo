# ===----------------------------------------------------------------------=== #
# Scijo: Integrate
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Integrate Module (scijo.integrate)

The `integrate` module provides tools for numerical integration and quadrature.
It includes adaptive and non-adaptive methods for computing definite integrals,
as well as fixed-sample integration rules for discrete data.

Examples:
    ```mojo
    from scijo.integrate import quad, trapezoid
    ```
"""

from .quadrature import quad
from .fixed_sample import trapezoid, simpson, romb
