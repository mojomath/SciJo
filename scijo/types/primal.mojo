# ===----------------------------------------------------------------------=== #
# SciJo: Primal conformer for scijo.types
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Primal (`scijo.types.primal`)
================================
A thin, zero-cost wrapper over `SIMD[dtype, width]` conforming to
`Numeric`.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.math import cos, exp, log, sin, sqrt

# ===----------------------------------------------------------------------=== #
# SciJo
# ===----------------------------------------------------------------------=== #
from .numeric import Numeric


@fieldwise_init
struct Primal[dtype: DType, width: Int = 1](
    Copyable,
    Deinitable,
    Movable,
    Numeric where dtype.is_floating_point(),
):
    """A `SIMD[dtype, width]` value, exposed through `Numeric`."""

    var v: SIMD[Self.dtype, Self.width]

    @implicit
    def __init__(out self, v: Scalar[Self.dtype]):
        """Builds a `Primal` scalar directly from a `Scalar[dtype]` literal.

        Args:
            v: The scalar value to wrap.
        """
        self.v = SIMD[Self.dtype, Self.width](v)

    @staticmethod
    def one() -> Self:
        """The multiplicative identity, `1`.

        Returns:
            `Self` equal to one.
        """
        return Self(SIMD[Self.dtype, Self.width](1))

    @staticmethod
    def constant(v: Float64) -> Self:
        """Wraps a literal `Float64` as a `Primal`.

        Args:
            v: The literal value.

        Returns:
            `Self` representing `v`.
        """
        return Self(SIMD[Self.dtype, Self.width](v))

    def __add__(self, rhs: Self) -> Self:
        """`self + rhs`.

        Args:
            rhs: The value to add.

        Returns:
            The sum.
        """
        return Self(self.v + rhs.v)

    def __mul__(self, rhs: Self) -> Self:
        """`self * rhs`.

        Args:
            rhs: The value to multiply by.

        Returns:
            The product.
        """
        return Self(self.v * rhs.v)

    def __neg__(self) -> Self:
        """`-self`.

        Returns:
            The negation.
        """
        return Self(-self.v)

    def __truediv__(self, rhs: Self) -> Self:
        """`self / rhs`.

        Args:
            rhs: The value to divide by.

        Returns:
            The quotient.
        """
        return Self(self.v / rhs.v)

    def exp(self) -> Self where Self.dtype.is_floating_point():
        """`e^self`.

        Returns:
            The exponential of `self`.
        """
        return Self(exp(self.v))

    def log(self) -> Self where Self.dtype.is_floating_point():
        """The natural logarithm of `self`.

        Returns:
            The natural log of `self`.
        """
        return Self(log(self.v))

    def sqrt(self) -> Self where Self.dtype.is_floating_point():
        """The non-negative square root of `self`.

        Returns:
            The square root of `self`.
        """
        return Self(sqrt(self.v))

    def sin(self) -> Self where Self.dtype.is_floating_point():
        """`sin(self)`, in radians.

        Returns:
            The sine of `self`.
        """
        return Self(sin(self.v))

    def cos(self) -> Self where Self.dtype.is_floating_point():
        """`cos(self)`, in radians.

        Returns:
            The cosine of `self`.
        """
        return Self(cos(self.v))
