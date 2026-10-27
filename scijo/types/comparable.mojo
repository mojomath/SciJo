# ===----------------------------------------------------------------------=== #
# SciJo: NumericComparable trait for scijo.types
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""NumericComparable Trait (`scijo.types.comparable`)
=====================================================
`Numeric` plus the ordering a branchy algorithm needs to pick a series vs.
an asymptotic expansion, a reflection formula, or a convergence cutoff -
without ever looking at a conformer's derivative. For `Dual`, every method
here compares (or floors) the primal `value` only: branch selection must
not depend on `deriv`, or the chain rule stops applying to whichever branch
runs.
"""

# ===----------------------------------------------------------------------=== #
# SciJo
# ===----------------------------------------------------------------------=== #
from scijo.types.numeric import Numeric


trait NumericComparable(Numeric):
    """`Numeric` plus the comparisons a branchy algorithm needs."""

    def __lt__(self, other: Self) -> Bool:
        """`self < other`.

        Args:
            other: The value to compare against.

        Returns:
            Whether `self` is less than `other`.
        """
        ...

    def __lt__(self, other: Float64) -> Bool:
        """`self < other`, against a literal.

        Args:
            other: The literal to compare against.

        Returns:
            Whether `self` is less than `other`.
        """
        ...

    def __le__(self, other: Float64) -> Bool:
        """`self <= other`, against a literal.

        Args:
            other: The literal to compare against.

        Returns:
            Whether `self` is at most `other`.
        """
        ...

    def __gt__(self, other: Self) -> Bool:
        """`self > other`.

        Args:
            other: The value to compare against.

        Returns:
            Whether `self` is greater than `other`.
        """
        ...

    def __eq__(self, other: Self) -> Bool:
        """`self == other`.

        Args:
            other: The value to compare against.

        Returns:
            Whether `self` equals `other`.
        """
        ...

    def floor(self) -> Self:
        """The largest integer value not greater than `self`.

        Returns:
            `floor(self)`.
        """
        ...

    def abs(self) -> Self:
        """The absolute value of `self`.

        Returns:
            `self` if `self >= 0`, else `-self`.
        """
        return -self if self < 0.0 else self
