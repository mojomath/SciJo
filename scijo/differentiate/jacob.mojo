# ===----------------------------------------------------------------------=== #
# Scijo: Differentiate - Jacobian
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Differentiate Module - Jacobian Matrix (scijo.differentiate.jacobian)

Computes the Jacobian matrix of a vector-valued function using central finite
differences with parallelized column evaluation.
"""

from numojo.routines.creation import zeros, full
from numojo.core import NDArray, Shape

from algorithm.functional import parallelize


fn jacobian[
    f: fn(x: NDArray[f64], args: Optional[List[Scalar[f64]]]) raises -> NDArray[
        f64
    ],
](
    x: NDArray[f64],
    args: Optional[List[Scalar[f64]]] = None,
    tolerances: Dict[String, Scalar[f64]] = {"abs": 1e-5, "rel": 1e-3},
    maxiter: Int = 10,
) raises -> NDArray[f64]:
    """Computes the Jacobian matrix of a vector-valued function using central finite differences.

    Evaluates J[i, j] = ∂f_i/∂x_j using the central difference formula
    (f(x + h*e_j) - f(x - h*e_j)) / (2h), where e_j is the j-th unit vector.
    Each column of the Jacobian is computed in parallel.

    Parameters:
        f: Vector-valued function with signature fn(x, args) -> NDArray[f64].

    Args:
        x: Input vector of shape (n,) at which to evaluate the Jacobian.
        args: Optional arguments to pass to the function.
        tolerances: Tolerance dictionary with "abs" and "rel" keys (reserved for future use).
        maxiter: Maximum iterations (reserved for future use).

    Raises:
        Error: If function evaluation fails for any perturbation.

    Returns:
        NDArray[f64] of shape (m, n) representing the Jacobian matrix,
        where m is the output dimension and n is the input dimension.

    Examples:
        ```mojo
        import numojo as nm
        from scijo.differentiate import jacobian
        from scijo.prelude import *

        fn f(x: NDArray[f64], args: Optional[List[Scalar[f64]]]) raises -> NDArray[f64]:
            return x * x

        var x = nm.array[f64]([1.0, 2.0])
        var J = jacobian[f](x)
        ```
    """
    var n: Int = len(x)
    var f0: NDArray[f64] = f(x, args)
    var m: Int = len(f0)

    var jacob: NDArray[f64] = zeros[f64](Shape(m, n))
    var step: NDArray[f64] = full[f64](Shape(n), fill_value=0.5)

    @parameter
    fn closure(j: Int):
        try:
            var x_plus = x.deep_copy()
            var x_minus = x.deep_copy()
            var hj = step.load(j)
            x_plus.store(j, val=x.load(j) + hj)
            x_minus.store(j, val=x.load(j) - hj)
            var f_plus = f(x_plus, args)
            var f_minus = f(x_minus, args)

            var col: NDArray[f64] = (f_plus - f_minus) / (2.0 * hj)

            for i in range(m):
                var val: Scalar[f64] = col.load(i)
                var flat_idx: Int = i * n + j
                jacob.store(flat_idx, val=val)
        except:
            print("Error in computing Jacobian column for j =", j)

    parallelize[closure](n, num_workers=n)

    return jacob^
