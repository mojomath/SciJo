# ===----------------------------------------------------------------------=== #
# SciJo: Dual conformer for scijo.types
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Dual (`scijo.types.dual`)
============================
Forward-mode automatic differentiation: `Dual[Inner]` pairs a value with
its derivative, both of type `Inner: Numeric`. Nesting
`Dual[Dual[Primal[f64]]]` tracks a second derivative using the same chain
rule applied twice.
"""

# ===----------------------------------------------------------------------=== #
# SciJo
# ===----------------------------------------------------------------------=== #
from scijo.types.comparable import NumericComparable
from scijo.types.numeric import Numeric
from scijo.types.primal import Primal


@fieldwise_init
struct Dual[Inner: Numeric](
    Copyable,
    Deinitable,
    ImplicitlyCopyable,
    Movable,
    Numeric,
    NumericComparable where conforms_to(Inner, NumericComparable),
):
    """`value` is `f(x)`; `deriv` is `df/dx` at the same point."""

    var value: Self.Inner
    var deriv: Self.Inner

    @staticmethod
    def one() -> Self:
        """The constant `1`, with a zero derivative.

        Returns:
            `Self` equal to one, with derivative zero.
        """
        return Self(Self.Inner.one(), Self.Inner.constant(0.0))

    @staticmethod
    def constant(v: Float64) -> Self:
        """A literal, with a zero derivative.

        Args:
            v: The literal value.

        Returns:
            `Self` representing `v`, with derivative zero.
        """
        return Self(Self.Inner.constant(v), Self.Inner.constant(0.0))

    @staticmethod
    def variable(x: Float64) -> Self:
        """`x`, with its derivative seeded to `1` - the variable being
        differentiated with respect to. A constant should use
        `Dual[...].constant(c)` instead, which keeps a zero derivative.

        Args:
            x: The point at which to seed the derivative.

        Returns:
            `Self` with value `x` and derivative `1`.
        """
        return Self(Self.Inner.constant(x), Self.Inner.one())

    def __add__(self, rhs: Self) -> Self:
        """Sum rule: `(f + g, f' + g')`.

        Args:
            rhs: The value to add.

        Returns:
            The sum, with derivative computed via the sum rule.
        """
        return Self(self.value + rhs.value, self.deriv + rhs.deriv)

    def __mul__(self, rhs: Self) -> Self:
        """Product rule: `(f*g, f'*g + f*g')`.

        Args:
            rhs: The value to multiply by.

        Returns:
            The product, with derivative computed via the product rule.
        """
        return Self(
            self.value * rhs.value,
            self.deriv * rhs.value + self.value * rhs.deriv,
        )

    def __neg__(self) -> Self:
        """`-self`.

        Returns:
            The negation, negating both value and derivative.
        """
        return Self(-self.value, -self.deriv)

    def __truediv__(self, rhs: Self) -> Self:
        """Quotient rule: `(f/g, (f'*g - f*g') / g^2)`.

        Args:
            rhs: The value to divide by.

        Returns:
            The quotient, with derivative computed via the quotient rule.
        """
        var value = self.value / rhs.value
        var deriv = (self.deriv * rhs.value - self.value * rhs.deriv) / (
            rhs.value * rhs.value
        )
        return Self(value^, deriv^)

    def exp(self) -> Self:
        """Chain rule: `d/dx[exp(f)] = f' * exp(f)`.

        Returns:
            `exp(self)`, with derivative via the chain rule.
        """
        var e = self.value.exp()
        var d = self.deriv * e.copy()
        return Self(e^, d^)

    def log(self) -> Self:
        """Chain rule: `d/dx[log(f)] = f' / f`.

        Returns:
            `log(self)`, with derivative via the chain rule.
        """
        return Self(self.value.log(), self.deriv / self.value)

    def sqrt(self) -> Self:
        """Chain rule: `d/dx[sqrt(f)] = f' / (2*sqrt(f))`.

        Returns:
            `sqrt(self)`, with derivative via the chain rule.
        """
        var v = self.value.sqrt()
        var two = Self.Inner.constant(2.0)
        var d = self.deriv / (two * v.copy())
        return Self(v^, d^)

    def sin(self) -> Self:
        """Chain rule: `d/dx[sin(f)] = cos(f) * f'`.

        Returns:
            `sin(self)`, with derivative via the chain rule.
        """
        return Self(self.value.sin(), self.value.cos() * self.deriv)

    def cos(self) -> Self:
        """Chain rule: `d/dx[cos(f)] = -sin(f) * f'`.

        Returns:
            `cos(self)`, with derivative via the chain rule.
        """
        return Self(self.value.cos(), -(self.value.sin() * self.deriv))

    def __lt__(
        self, other: Self
    ) -> Bool where conforms_to(Self.Inner, NumericComparable):
        """`self < other`, comparing `value` only - a branch decision must
        not depend on `deriv`, or the chain rule stops applying to
        whichever branch runs.

        Args:
            other: The value to compare against.

        Returns:
            Whether `self.value` is less than `other.value`.
        """
        return self.value < other.value

    def __lt__(
        self, other: Float64
    ) -> Bool where conforms_to(Self.Inner, NumericComparable):
        """`self < other`, against a literal, comparing `value` only.

        Args:
            other: The literal to compare against.

        Returns:
            Whether `self.value` is less than `other`.
        """
        return self.value < other

    def __le__(
        self, other: Float64
    ) -> Bool where conforms_to(Self.Inner, NumericComparable):
        """`self <= other`, against a literal, comparing `value` only.

        Args:
            other: The literal to compare against.

        Returns:
            Whether `self.value` is at most `other`.
        """
        return self.value <= other

    def __gt__(
        self, other: Self
    ) -> Bool where conforms_to(Self.Inner, NumericComparable):
        """`self > other`, comparing `value` only.

        Args:
            other: The value to compare against.

        Returns:
            Whether `self.value` is greater than `other.value`.
        """
        return self.value > other.value

    def __eq__(
        self, other: Self
    ) -> Bool where conforms_to(Self.Inner, NumericComparable):
        """`self == other`, comparing `value` only.

        Args:
            other: The value to compare against.

        Returns:
            Whether `self.value` equals `other.value`.
        """
        return self.value == other.value

    def floor(self) -> Self where conforms_to(Self.Inner, NumericComparable):
        """The largest integer value not greater than `self.value`, with a
        zero derivative (only ever used to decide a branch, never to carry
        a gradient).

        Returns:
            `floor(self.value)`, with derivative zero.
        """
        return Self(self.value.floor(), Self.Inner.constant(0.0))


# ===----------------------------------------------------------------------=== #
# Free functions
# ===----------------------------------------------------------------------=== #


def variable(x: Float64) -> Dual[Primal[DType.float64]]:
    """`x`, ready to track a first derivative.

    Shorthand for `Dual[Primal[f64]].variable(x)`.

    Args:
        x: The point at which to seed the derivative.

    Returns:
        `x` wrapped as a `Dual[Primal[f64]]` with derivative `1`.
    """
    return Dual[Primal[DType.float64]].variable(x)


def variable2(x: Float64) -> Dual[Dual[Primal[DType.float64]]]:
    """`x`, ready to track both the first and second derivative in one call.

    Shorthand for `Dual[Dual[Primal[f64]]]`: seeds the inner `Dual` at `x`
    and the outer `Dual` at `1`, so `f(variable2(x)).deriv.deriv` is
    `f''(x)` with no manual wiring.

    Args:
        x: The point at which to seed the derivatives.

    Returns:
        `x` wrapped as a nested `Dual[Dual[Primal[f64]]]`.
    """
    var inner = Dual[Primal[DType.float64]].variable(x)
    var outer_deriv = Dual[Primal[DType.float64]].one()
    return Dual[Dual[Primal[DType.float64]]](inner^, outer_deriv^)
