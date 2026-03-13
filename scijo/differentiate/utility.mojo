# ===----------------------------------------------------------------------=== #
# Scijo: Differentiate - Utility
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Differentiate Module - Utility Functions (scijo.differentiate.utility)

Utility functions and data structures for numerical differentiation using finite
difference methods. Implements central, forward, and backward finite difference
coefficient tables.

References:
    - Wikipedia: Finite difference coefficient
      https://en.wikipedia.org/wiki/Finite_difference_coefficient
    - Fornberg, B. (1988). Generation of Finite Difference Formulas on
      Arbitrarily Spaced Grids. Mathematics of Computation, 51(184), 699-706.
"""

# ===----------------------------------------------------------------------=== #
# Result types
# ===----------------------------------------------------------------------=== #


struct DiffResult[dtype: DType](ImplicitlyCopyable, Writable):
    """Result structure for numerical differentiation operations.

    Encapsulates the results of derivative computations, including the computed
    derivative value, convergence information, and diagnostic data.

    Parameters:
        dtype: The floating-point data type (e.g., DType.float32, DType.float64).
    """

    var success: Bool
    """Whether the computation converged successfully."""
    var df: Scalar[Self.dtype]
    """The computed derivative value."""
    var error: Scalar[Self.dtype]
    """Estimated error or convergence tolerance achieved."""
    var nit: Int
    """Number of iterations performed."""
    var nfev: Int
    """Number of function evaluations used."""
    var x: Scalar[Self.dtype]
    """The point at which the derivative was evaluated."""

    fn __init__(
        out self,
        success: Bool,
        df: Scalar[Self.dtype],
        error: Scalar[Self.dtype],
        nit: Int,
        nfev: Int,
        x: Scalar[Self.dtype],
    ):
        self.success = success
        self.df = df
        self.error = error
        self.nit = nit
        self.nfev = nfev
        self.x = x

    fn __str__(self) raises -> String:
        return String(
            "Result(success={}, df={}, error={:.2e}, nit={}, nfev={}, x={})"
        ).format(self.success, self.df, self.error, self.nit, self.nfev, self.x)

    fn write_to[W: Writer](self, mut writer: W):
        try:
            writer.write(
                String(
                    "\n================================\n"
                    + "Status      : {}\n"
                    + "Derivative  : {}\n"
                    + "Error Est.  : {}\n"
                    + "Iterations  : {}\n"
                    + "Func Evals  : {}\n"
                    + "Point (x)   : {}\n"
                    + "================================\n"
                ).format(
                    "SUCCESS" if self.success else "FAILED",
                    self.df,
                    self.error,
                    self.nit,
                    self.nfev,
                    self.x,
                )
            )
        except e:
            writer.write("Error displaying Result: " + String(e) + "\n")


# ===----------------------------------------------------------------------=== #
# Finite difference tables
# ===----------------------------------------------------------------------=== #


@parameter
fn generate_central_finite_difference_table[
    dtype: DType
]() -> Dict[Int, List[Scalar[dtype]]]:
    """Generates central finite difference coefficients for first-order derivatives.

    Creates a lookup table of coefficients for central difference approximations
    using symmetric stencils: f'(x) ≈ Σ(c_i * f(x + i*h)) / h.

    Parameters:
        dtype: The floating-point data type for the coefficients.

    Returns:
        Coefficient arrays indexed by accuracy order.
        Available orders: 2, 4, 6, 8 with truncation errors O(h²), O(h⁴), O(h⁶), O(h⁸).
    """
    var coefficients = Dict[Int, List[Scalar[dtype]]]()

    # Order 2: points [-1, 0, 1]
    coefficients[2]: List[Scalar[dtype]] = [-0.5, 0.0, 0.5]

    # Order 4: points [-2, -1, 0, 1, 2]
    coefficients[4]: List[Scalar[dtype]] = [
        1.0 / 12.0,
        -2.0 / 3.0,
        0.0,
        2.0 / 3.0,
        -1.0 / 12.0,
    ]

    # Order 6: points [-3, -2, -1, 0, 1, 2, 3]
    coefficients[6]: List[Scalar[dtype]] = [
        -1.0 / 60.0,
        3.0 / 20.0,
        -3.0 / 4.0,
        0.0,
        3.0 / 4.0,
        -3.0 / 20.0,
        1.0 / 60.0,
    ]

    # Order 8: points [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    coefficients[8]: List[Scalar[dtype]] = [
        1.0 / 280.0,
        -4.0 / 105.0,
        1.0 / 5.0,
        -4.0 / 5.0,
        0.0,
        4.0 / 5.0,
        -1.0 / 5.0,
        4.0 / 105.0,
        -1.0 / 280.0,
    ]

    return coefficients^


@parameter
fn generate_forward_finite_difference_table[
    dtype: DType
]() -> Dict[Int, List[Scalar[dtype]]]:
    """Generates forward finite difference coefficients for first-order derivatives.

    Creates a lookup table of coefficients for forward difference approximations
    using one-sided stencils: f'(x) ≈ Σ(c_i * f(x + i*h)) / h.

    Forward differences are necessary at left domain boundaries or when backward
    evaluations are not feasible.

    Parameters:
        dtype: The floating-point data type for the coefficients.

    Returns:
        Coefficient arrays indexed by accuracy order.
        Available orders: 1, 2, 3, 4, 5, 6 with truncation errors O(h) through O(h⁶).
    """
    var coefficients = Dict[Int, List[Scalar[dtype]]]()

    # Order 1: points [0, 1]
    coefficients[1]: List[Scalar[dtype]] = [-1.0, 1.0]

    # Order 2: points [0, 1, 2]
    coefficients[2]: List[Scalar[dtype]] = [-3.0 / 2.0, 2.0, -1.0 / 2.0]

    # Order 3: points [0, 1, 2, 3]
    coefficients[3]: List[Scalar[dtype]] = [
        -11.0 / 6.0,
        3.0,
        -3.0 / 2.0,
        1.0 / 3.0,
    ]

    # Order 4: points [0, 1, 2, 3, 4]
    coefficients[4]: List[Scalar[dtype]] = [
        -25.0 / 12.0,
        4.0,
        -3.0,
        4.0 / 3.0,
        -1.0 / 4.0,
    ]

    # Order 5: points [0, 1, 2, 3, 4, 5]
    coefficients[5]: List[Scalar[dtype]] = [
        -137.0 / 60.0,
        5.0,
        -5.0,
        10.0 / 3.0,
        -5.0 / 4.0,
        1.0 / 5.0,
    ]

    # Order 6: points [0, 1, 2, 3, 4, 5, 6]
    coefficients[6]: List[Scalar[dtype]] = [
        -49.0 / 20.0,
        6.0,
        -15.0 / 2.0,
        20.0 / 3.0,
        -15.0 / 4.0,
        6.0 / 5.0,
        -1.0 / 6.0,
    ]

    return coefficients^


@parameter
fn generate_backward_finite_difference_table[
    dtype: DType
]() -> Dict[Int, List[Scalar[dtype]]]:
    """Generates backward finite difference coefficients for first-order derivatives.

    Creates a lookup table of coefficients for backward difference approximations
    using one-sided stencils: f'(x) ≈ Σ(c_i * f(x - i*h)) / h.

    Backward differences are derived from forward differences by reversing the
    stencil and adjusting signs. They are necessary at right domain boundaries
    or when forward evaluations are not feasible.

    Parameters:
        dtype: The floating-point data type for the coefficients.

    Returns:
        Coefficient arrays indexed by accuracy order.
        Available orders: 1, 2, 3, 4, 5, 6 with truncation errors O(h) through O(h⁶).
    """
    var coefficients = Dict[Int, List[Scalar[dtype]]]()

    # Order 1: points [-1, 0]
    coefficients[1]: List[Scalar[dtype]] = [-1.0, 1.0]

    # Order 2: points [-2, -1, 0]
    coefficients[2]: List[Scalar[dtype]] = [1.0 / 2.0, -2.0, 3.0 / 2.0]

    # Order 3: points [-3, -2, -1, 0]
    coefficients[3]: List[Scalar[dtype]] = [
        -1.0 / 3.0,
        3.0 / 2.0,
        -3.0,
        11.0 / 6.0,
    ]

    # Order 4: points [-4, -3, -2, -1, 0]
    coefficients[4]: List[Scalar[dtype]] = [
        1.0 / 4.0,
        -4.0 / 3.0,
        3.0,
        -4.0,
        25.0 / 12.0,
    ]

    # Order 5: points [-5, -4, -3, -2, -1, 0]
    coefficients[5]: List[Scalar[dtype]] = [
        -1.0 / 5.0,
        5.0 / 4.0,
        -10.0 / 3.0,
        5.0,
        -5.0,
        137.0 / 60.0,
    ]

    # Order 6: points [-6, -5, -4, -3, -2, -1, 0]
    coefficients[6]: List[Scalar[dtype]] = [
        1.0 / 6.0,
        -6.0 / 5.0,
        15.0 / 4.0,
        -20.0 / 3.0,
        15.0 / 2.0,
        -6.0,
        49.0 / 20.0,
    ]

    return coefficients^
