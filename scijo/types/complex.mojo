# ===----------------------------------------------------------------------=== #
# SciJo: Complex conformer for scijo.types
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Complex (`scijo.types.complex`)
==================================
`Complex[dtype, width]` pairs a real and an imaginary `SIMD[dtype, width]`
component and conforms to `Numeric`, so any function written once against
`Numeric` can be called with a complex argument with no change.

Notes:
`log` and `sqrt` use the principal branch (the same convention as
`numpy`/`std.math`): `log`'s imaginary part is `atan2(im, re)`, in
`(-pi, pi]`.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.math import atan2, copysign, cos, cosh, exp, log, sin, sinh, sqrt

# ===----------------------------------------------------------------------=== #
# SciJo
# ===----------------------------------------------------------------------=== #
from .numeric import Numeric


@fieldwise_init
struct Complex[dtype: DType, width: Int = 1](
    Copyable,
    Deinitable,
    Movable,
    Numeric where dtype.is_floating_point(),
):
    """`re + im*i`, exposed through `Numeric`."""

    var re: SIMD[Self.dtype, Self.width]
    var im: SIMD[Self.dtype, Self.width]

    @implicit
    def __init__(out self, re: Scalar[Self.dtype]):
        """Builds a purely-real `Complex` from a `Scalar[dtype]` literal.

        Args:
            re: The real part; the imaginary part is zero.
        """
        self.re = SIMD[Self.dtype, Self.width](re)
        self.im = SIMD[Self.dtype, Self.width](0)

    @staticmethod
    def one() -> Self:
        """The multiplicative identity, `1 + 0i`.

        Returns:
            `Self` equal to one.
        """
        return Self(SIMD[Self.dtype, Self.width](1), SIMD[Self.dtype, Self.width](0))

    @staticmethod
    def constant(v: Float64) -> Self:
        """Wraps a literal `Float64` as a purely-real `Complex`.

        Args:
            v: The literal value.

        Returns:
            `Self` representing `v + 0i`.
        """
        return Self(SIMD[Self.dtype, Self.width](v), SIMD[Self.dtype, Self.width](0))

    def __add__(self, rhs: Self) -> Self:
        """`self + rhs`, added component-wise.

        Args:
            rhs: The value to add.

        Returns:
            The sum.
        """
        return Self(self.re + rhs.re, self.im + rhs.im)

    def __mul__(self, rhs: Self) -> Self:
        """`self * rhs`, via `(a+bi)(c+di) = (ac-bd) + (ad+bc)i`.

        Args:
            rhs: The value to multiply by.

        Returns:
            The product.
        """
        return Self(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )

    def __neg__(self) -> Self:
        """`-self`.

        Returns:
            The negation, negating both components.
        """
        return Self(-self.re, -self.im)

    def __truediv__(self, rhs: Self) -> Self:
        """`self / rhs`, via multiplying by `conj(rhs) / |rhs|^2`.

        Args:
            rhs: The value to divide by.

        Returns:
            The quotient.
        """
        var denom = rhs.re * rhs.re + rhs.im * rhs.im
        return Self(
            (self.re * rhs.re + self.im * rhs.im) / denom,
            (self.im * rhs.re - self.re * rhs.im) / denom,
        )

    def exp(self) -> Self where Self.dtype.is_floating_point():
        """`e^self`, via `e^a * (cos(b) + i*sin(b))`.

        Returns:
            The exponential of `self`.
        """
        var r = exp(self.re)
        return Self(r * cos(self.im), r * sin(self.im))

    def log(self) -> Self where Self.dtype.is_floating_point():
        """The principal natural logarithm of `self`.

        Returns:
            `log(|self|) + i*atan2(im, re)`.
        """
        var magnitude_sq = self.re * self.re + self.im * self.im
        return Self(0.5 * log(magnitude_sq), atan2(self.im, self.re))

    def sqrt(self) -> Self where Self.dtype.is_floating_point():
        """The principal square root of `self`.

        Returns:
            `w` such that `w * w == self` and `w.re >= 0`.
        """
        var r = sqrt(self.re * self.re + self.im * self.im)
        var re = sqrt((r + self.re) * 0.5)
        var im = sqrt((r - self.re) * 0.5)
        return Self(re, copysign(im, self.im))

    def sin(self) -> Self where Self.dtype.is_floating_point():
        """`sin(self)`, via `sin(a)*cosh(b) + i*cos(a)*sinh(b)`.

        Returns:
            The sine of `self`.
        """
        return Self(sin(self.re) * cosh(self.im), cos(self.re) * sinh(self.im))

    def cos(self) -> Self where Self.dtype.is_floating_point():
        """`cos(self)`, via `cos(a)*cosh(b) - i*sin(a)*sinh(b)`.

        Returns:
            The cosine of `self`.
        """
        return Self(cos(self.re) * cosh(self.im), -sin(self.re) * sinh(self.im))
