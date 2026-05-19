# ===----------------------------------------------------------------------=== #
# SciJo: Integrate module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Quadrature Integration (`scijo.integrate.quadrature`)
=======================================================
General-purpose numerical integration using a SciPy-like frontend signature.
This module currently exposes `quad` with QNG quadrature support.

References:
    - SciPy quad documentation:
      https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html
    - Netlib QUADPACK:
      https://www.netlib.org/quadpack/
"""

from msl import qng_integrate as msl_qng_integrate

from .utility import IntegralResult


def quad[
    dtype: DType,
    integrand_func: def[dtype: DType](
        x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
    ) capturing -> Scalar[dtype],
    *,
    method: String = "qng",
](
    a: Scalar[dtype],
    b: Scalar[dtype],
    args: Optional[List[Scalar[dtype]]] = None,
    atol: Scalar[dtype] = 1.49e-8,
    rtol: Scalar[dtype] = 1.49e-8,
) raises -> IntegralResult[dtype] where dtype.is_floating_point():
    """Computes the definite integral of a scalar function over [a, b].

    Parameters:
        dtype: The floating-point data type.
        integrand_func: Integrand function with signature
            `def(x, args) -> Scalar[dtype]`.
        method: Quadrature method name (compile-time). Currently supports
            `"qng"`.

    Args:
        a: Lower integration limit.
        b: Upper integration limit.
        args: Optional arguments to pass to the integrand function.
        atol: Absolute error tolerance.
        rtol: Relative error tolerance.

    Returns:
        IntegralResult[dtype] containing integral value, absolute error
        estimate, and status fields.

    Raises:
        Error: If an unsupported integration method is requested.

    NOTES:
        Method `"qng"` performs non-adaptive Gauss-Kronrod-Patterson
        quadrature.
    """

    comptime if method != "qng":
        raise Error(
            "Unsupported quad method: "
            + String(method)
            + ". Supported methods: 'qng'."
        )
    else:
        @parameter
        def wrapped_fn(x: Float64) -> Float64:
            return Float64(integrand_func(Scalar[dtype](x), args))

        var result = msl_qng_integrate[wrapped_fn](
            Float64(a),
            Float64(b),
            epsabs=Float64(atol),
            epsrel=Float64(rtol),
        )

        return IntegralResult[dtype](
            integral=Scalar[dtype](result.val),
            abserr=Scalar[dtype](result.err),
            nfev=0,
            ier=0,
        )
