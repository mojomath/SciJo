# ===----------------------------------------------------------------------=== #
# Scijo: Differentiate
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Differentiate Module (scijo.differentiate)

The `differentiate` module provides tools for numerical differentiation and gradient computation.
It includes functions for calculating derivatives, Jacobians, with much more to come in the future.
"""

from .deriv import derivative
from .jacob import jacobian
