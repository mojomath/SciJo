# ===----------------------------------------------------------------------=== #
# Scijo: Integrate - Quadrature
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Integrate Module - Quadrature (scijo.integrate.quadrature)

General-purpose numerical integration using adaptive quadrature methods based on
the QUADPACK library. Currently implements the non-adaptive Gauss-Kronrod-Patterson
(QNG) algorithm.

Examples:
    ```mojo
    from scijo.integrate import quad
    ```

References:
    - SciPy quad documentation:
      https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html
    - Netlib QUADPACK: https://www.netlib.org/quadpack/
    - Advanpix G10K21 coefficients:
      https://www.advanpix.com/2011/11/07/gauss-kronrod-quadrature-nodes-weights/
"""

from math import sqrt
from builtin.math import min, max
from utils import StaticTuple

from .utility import (
    IntegralResult,
    machine_epsilon,
    get_quad_error_message,
    smallest_positive_dtype,
    largest_positive_dtype,
    x1_nodes,
    w10_gauss_weights,
    x2_nodes,
    w21a_kronrod_weights,
    w21b_kronrod_weights,
    x3_nodes,
    w43a_kronrod_weights,
    w43b_kronrod_weights,
    x4_nodes,
    w87a_kronrod_weights,
    w87b_kronrod_weights,
)


# ===----------------------------------------------------------------------=== #
# Quad
# ===----------------------------------------------------------------------=== #

fn quad[
    dtype: DType,
    func: fn[dtype: DType](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) -> Scalar[dtype],
    *,
    method: String = "qng",
](
    a: Scalar[dtype],
    b: Scalar[dtype],
    args: Optional[List[Scalar[dtype]]],
    epsabs: Scalar[dtype] = 1.49e-8,
    epsrel: Scalar[dtype] = 1.49e-8,
) raises -> IntegralResult[dtype] where dtype.is_floating_point():
    """Computes the definite integral of a scalar function over [a, b].

    Dispatches to the appropriate quadrature algorithm based on the `method`
    parameter.

    Parameters:
        dtype: The floating-point data type.
        func: Integrand function with signature fn(x, args) -> Scalar[dtype].
        method: Quadrature algorithm to use. Currently supported: "qng". Keyword-only.

    Args:
        a: Lower integration limit.
        b: Upper integration limit.
        args: Optional arguments to pass to the integrand.
        epsabs: Absolute error tolerance.
        epsrel: Relative error tolerance.

    Raises:
        Error: If the specified method is not supported.

    Returns:
        IntegralResult[dtype] containing the integral value, absolute error
        estimate, function evaluation count, and status code.

    Examples:
        ```mojo
        import numojo as nm
        from scijo.integrate import quad
        from scijo.prelude import *

        fn integrand[dtype: DType](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[dtype]:
            return x * x  # Example: f(x) = x^2

        fn main() raises:
            var result = quad[f64, integrand](0.0, 1.0, None)
            print("Integral:", result.integral)  # Should be close to 1/3
            print("Estimated error:", result.abserr)
        ```
    """

    @parameter
    if method == "qng":
        return _qng[dtype, func](a, b, args, epsabs, epsrel)
    else:
        raise Error(
            "Unsupported quad method: "
            + String(method)
            + ". Supported methods: 'qng'."
        )


fn _qng[
    dtype: DType,
    func: fn[dtype: DType](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) -> Scalar[dtype],
](
    a: Scalar[dtype],
    b: Scalar[dtype],
    args: Optional[List[Scalar[dtype]]],
    epsabs: Scalar[dtype] = 1.49e-8,
    epsrel: Scalar[dtype] = 1.49e-8,
) -> IntegralResult[dtype] where dtype.is_floating_point():
    """Non-adaptive Gauss-Kronrod-Patterson integration (QUADPACK QNG).

    Attempts integration using progressively higher-order rules until the
    requested tolerance is met:
        1. 10-point Gauss / 21-point Kronrod
        2. 21-point GK / 43-point Kronrod extension
        3. 43-point GK / 87-point Kronrod extension

    Function evaluations are reused between successive rules for efficiency.

    Parameters:
        dtype: The floating-point data type.
        func: Integrand function with signature fn(x, args) -> Scalar[dtype].

    Args:
        a: Lower integration limit.
        b: Upper integration limit.
        args: Optional arguments to pass to the integrand.
        epsabs: Absolute error tolerance.
        epsrel: Relative error tolerance.

    Returns:
        IntegralResult[dtype] containing the integral value, error estimate,
        function evaluation count, and status code (ier).
    """
    comptime epsilon_mach: Scalar[dtype] = Scalar[dtype](
        machine_epsilon[dtype]()
    )
    comptime under_flow: Scalar[dtype] = smallest_positive_dtype[dtype]

    if a == b:
        return IntegralResult(
            integral=Scalar[dtype](0),
            abserr=Scalar[dtype](0),
            neval=0,
            ier=0,
        )

    if epsabs <= 0 and epsrel < max(
        Scalar[dtype](0.5e-14), Scalar[dtype](50.0) * epsilon_mach
    ):
        return IntegralResult[dtype](
            integral=Scalar[dtype](0),
            abserr=Scalar[dtype](0),
            neval=0,
            ier=6,
        )

    var half_length: Scalar[dtype] = Scalar[dtype](0.5) * (b - a)
    var abs_half_length: Scalar[dtype] = abs(half_length)
    var center: Scalar[dtype] = Scalar[dtype](0.5) * (b + a)
    var f_center: Scalar[dtype] = func(center, args)
    var neval: Int = 1

    # 10-point Gauss / 21-point Kronrod Rule
    var result_10: Scalar[dtype] = Scalar[dtype](0)
    var result_21: Scalar[dtype] = (
        Scalar[dtype](w21b_kronrod_weights[5]) * f_center
    )
    var result_abs: Scalar[dtype] = Scalar[dtype](
        w21b_kronrod_weights[5]
    ) * abs(f_center)

    var saved_fvalues: StaticTuple[Scalar[dtype], 21] = StaticTuple[
        Scalar[dtype], 21
    ](0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    var fv1: StaticTuple[Scalar[dtype], 5] = StaticTuple[Scalar[dtype], 5](
        0, 0, 0, 0, 0
    )
    var fv2: StaticTuple[Scalar[dtype], 5] = StaticTuple[Scalar[dtype], 5](
        0, 0, 0, 0, 0
    )
    for k in range(5):
        var abscissa: Scalar[dtype] = half_length * Scalar[dtype](x1_nodes[k])
        var fval1: Scalar[dtype] = func(center + abscissa, args)
        var fval2: Scalar[dtype] = func(center - abscissa, args)
        var fval: Scalar[dtype] = fval1 + fval2
        result_10 += Scalar[dtype](w10_gauss_weights[k]) * fval
        result_21 += Scalar[dtype](w21a_kronrod_weights[k]) * fval
        result_abs += Scalar[dtype](w21a_kronrod_weights[k]) * (
            abs(fval1) + abs(fval2)
        )
        saved_fvalues[k] = fval
        fv1[k] = fval1
        fv2[k] = fval2
    neval += 10

    var index: Int = 5
    var fv3: StaticTuple[Scalar[dtype], 5] = StaticTuple[Scalar[dtype], 5](
        0, 0, 0, 0, 0
    )
    var fv4: StaticTuple[Scalar[dtype], 5] = StaticTuple[Scalar[dtype], 5](
        0, 0, 0, 0, 0
    )
    for k in range(5):
        var abscissa: Scalar[dtype] = half_length * Scalar[dtype](x2_nodes[k])
        var fval1: Scalar[dtype] = func(center + abscissa, args)
        var fval2: Scalar[dtype] = func(center - abscissa, args)
        var fval: Scalar[dtype] = fval1 + fval2
        result_21 += Scalar[dtype](w21b_kronrod_weights[k]) * fval
        result_abs += Scalar[dtype](w21b_kronrod_weights[k]) * (
            abs(fval1) + abs(fval2)
        )
        saved_fvalues[index] = fval
        fv3[k] = fval1
        fv4[k] = fval2
        index += 1
    neval += 10

    var result: Scalar[dtype] = result_21 * half_length
    result_abs = result_abs * abs_half_length
    var result_mean: Scalar[dtype] = Scalar[dtype](0.5) * result_21
    var result_asc: Scalar[dtype] = Scalar[dtype](
        w21b_kronrod_weights[5]
    ) * abs(f_center - result_mean)
    for k in range(5):
        result_asc += Scalar[dtype](w21a_kronrod_weights[k]) * (
            abs(fv1[k] - result_mean) + abs(fv2[k] - result_mean)
        )
        result_asc += Scalar[dtype](w21b_kronrod_weights[k]) * (
            abs(fv3[k] - result_mean) + abs(fv4[k] - result_mean)
        )
    result_asc = result_asc * abs_half_length

    var abs_error: Scalar[dtype] = abs((result_21 - result_10) * half_length)
    if result_asc != Scalar[dtype](0) and abs_error != Scalar[dtype](0):
        abs_error = result_asc * min(
            Scalar[dtype](1),
            (Scalar[dtype](200) * abs_error / result_asc) ** Scalar[dtype](1.5),
        )
    if result_abs > under_flow / (Scalar[dtype](50) * epsilon_mach):
        abs_error = max(
            (epsilon_mach * Scalar[dtype](50)) * result_abs, abs_error
        )

    if abs_error <= max(epsabs, epsrel * abs(result)):
        return IntegralResult[dtype](
            integral=result,
            abserr=abs_error,
            neval=neval,
            ier=0,
        )

    # 21-point GK / 43-point Kronrod Rule
    var result_43: Scalar[dtype] = (
        Scalar[dtype](w43b_kronrod_weights[11]) * f_center
    )

    for k in range(10):
        result_43 += saved_fvalues[k] * Scalar[dtype](w43a_kronrod_weights[k])

    for k in range(11):
        var abscissa: Scalar[dtype] = half_length * Scalar[dtype](x3_nodes[k])
        var fval: Scalar[dtype] = func(center + abscissa, args) + func(
            center - abscissa, args
        )
        result_43 += Scalar[dtype](w43b_kronrod_weights[k]) * fval
        saved_fvalues[index] = fval
        index += 1
    neval += 22

    result = result_43 * half_length
    abs_error = abs((result_43 - result_21) * half_length)
    if result_asc != Scalar[dtype](0) and abs_error != Scalar[dtype](0):
        abs_error = result_asc * min(
            Scalar[dtype](1),
            (Scalar[dtype](200) * abs_error / result_asc) ** Scalar[dtype](1.5),
        )
    if result_abs > under_flow / (Scalar[dtype](50) * epsilon_mach):
        abs_error = max(
            (epsilon_mach * Scalar[dtype](50)) * result_abs, abs_error
        )

    if abs_error <= max(epsabs, epsrel * abs(result)):
        return IntegralResult[dtype](
            integral=result,
            abserr=abs_error,
            neval=neval,
            ier=0,
        )

    # 43-point GK / 87-point Kronrod Rule
    var result_87: Scalar[dtype] = (
        Scalar[dtype](w87b_kronrod_weights[22]) * f_center
    )

    for k in range(21):
        result_87 += saved_fvalues[k] * Scalar[dtype](w87a_kronrod_weights[k])

    for k in range(22):
        var abscissa: Scalar[dtype] = half_length * Scalar[dtype](x4_nodes[k])
        result_87 += Scalar[dtype](w87b_kronrod_weights[k]) * (
            func(center + abscissa, args) + func(center - abscissa, args)
        )
    neval += 44

    result = result_87 * half_length
    abs_error = abs((result_87 - result_43) * half_length)
    if result_asc != Scalar[dtype](0) and abs_error != Scalar[dtype](0):
        abs_error = result_asc * min(
            Scalar[dtype](1),
            (Scalar[dtype](200) * abs_error / result_asc) ** Scalar[dtype](1.5),
        )
    if result_abs > under_flow / (Scalar[dtype](50) * epsilon_mach):
        abs_error = max(
            (epsilon_mach * Scalar[dtype](50)) * result_abs, abs_error
        )

    if abs_error <= max(epsabs, epsrel * abs(result)):
        return IntegralResult[dtype](
            integral=result,
            abserr=abs_error,
            neval=neval,
            ier=0,
        )

    return IntegralResult[dtype](
        integral=result,
        abserr=abs_error,
        neval=neval,
        ier=1,
    )
