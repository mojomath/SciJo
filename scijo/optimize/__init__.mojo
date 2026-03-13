# ===----------------------------------------------------------------------=== #
# Scijo: Optimize
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Optimize Module (scijo.optimize)

The `optimize` module provides tools for numerical optimization and root-finding.
It includes scalar root-finding methods such as bisection, Newton-Raphson, and
the secant method.
"""

from .root_scalar import root_scalar, newton, bisect, secant
from .min_scalar import minimize_scalar
