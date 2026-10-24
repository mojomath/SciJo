# ===----------------------------------------------------------------------=== #
# SciJo: Numeric trait for scijo.types
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Numeric Trait (`scijo.types.numeric`)
========================================
The minimal arithmetic and transcendental operations a differentiable
scalar kernel needs, independent of any concrete dtype or representation.
"""


trait Numeric(Copyable, Deinitable, Movable):
    """The minimal arithmetic a differentiable scalar kernel needs."""

    @staticmethod
    def one() -> Self:
        """The multiplicative identity, with any auxiliary component zero.

        Returns:
            `Self` equal to one.
        """
        ...

    @staticmethod
    def constant(v: Float64) -> Self:
        """Embeds a literal coefficient as `Self`.

        Args:
            v: The literal value to embed.

        Returns:
            `Self` representing `v`.
        """
        ...

    def __add__(self, rhs: Self) -> Self:
        """`self + rhs`.

        Args:
            rhs: The value to add.

        Returns:
            The sum.
        """
        ...

    def __mul__(self, rhs: Self) -> Self:
        """`self * rhs`.

        Args:
            rhs: The value to multiply by.

        Returns:
            The product.
        """
        ...

    def __neg__(self) -> Self:
        """`-self`.

        Returns:
            The negation.
        """
        ...

    def __sub__(self, rhs: Self) -> Self:
        """`self - rhs`, implemented as `self + (-rhs)` so no conformer has
        to implement subtraction separately.

        Args:
            rhs: The value to subtract.

        Returns:
            The difference.
        """
        return self + (-rhs)

    def __truediv__(self, rhs: Self) -> Self:
        """`self / rhs`.

        Args:
            rhs: The value to divide by.

        Returns:
            The quotient.
        """
        ...

    def __add__(self, rhs: Float64) -> Self:
        """`self + rhs`, treating the literal `rhs` as `Self.constant(rhs)`.

        Args:
            rhs: The literal to add.

        Returns:
            The sum.
        """
        return self + Self.constant(rhs)

    def __radd__(self, lhs: Float64) -> Self:
        """`lhs + self`, for a literal written on the left.

        Args:
            lhs: The literal to add.

        Returns:
            The sum.
        """
        return Self.constant(lhs) + self

    def __mul__(self, rhs: Float64) -> Self:
        """`self * rhs`, treating the literal `rhs` as `Self.constant(rhs)`.

        Args:
            rhs: The literal to multiply by.

        Returns:
            The product.
        """
        return self * Self.constant(rhs)

    def __rmul__(self, lhs: Float64) -> Self:
        """`lhs * self`, for a literal written on the left.

        Args:
            lhs: The literal to multiply by.

        Returns:
            The product.
        """
        return Self.constant(lhs) * self

    def __sub__(self, rhs: Float64) -> Self:
        """`self - rhs`, treating the literal `rhs` as `Self.constant(rhs)`.

        Args:
            rhs: The literal to subtract.

        Returns:
            The difference.
        """
        return self - Self.constant(rhs)

    def __rsub__(self, lhs: Float64) -> Self:
        """`lhs - self`, for a literal written on the left.

        Args:
            lhs: The literal to subtract from.

        Returns:
            The difference.
        """
        return Self.constant(lhs) - self

    def __truediv__(self, rhs: Float64) -> Self:
        """`self / rhs`, treating the literal `rhs` as `Self.constant(rhs)`.

        Args:
            rhs: The literal to divide by.

        Returns:
            The quotient.
        """
        return self / Self.constant(rhs)

    def __rtruediv__(self, lhs: Float64) -> Self:
        """`lhs / self`, for a literal written on the left.

        Args:
            lhs: The literal to divide.

        Returns:
            The quotient.
        """
        return Self.constant(lhs) / self

    def exp(self) -> Self:
        """`e^self`.

        Returns:
            The exponential of `self`.
        """
        ...

    def log(self) -> Self:
        """The natural logarithm. Only meaningful for `self > 0`.

        Returns:
            The natural log of `self`.
        """
        ...

    def sqrt(self) -> Self:
        """The non-negative square root. Only meaningful for `self >= 0`.

        Returns:
            The square root of `self`.
        """
        ...

    def sin(self) -> Self:
        """`sin(self)`, in radians.

        Returns:
            The sine of `self`.
        """
        ...

    def cos(self) -> Self:
        """`cos(self)`, in radians.

        Returns:
            The cosine of `self`.
        """
        ...
