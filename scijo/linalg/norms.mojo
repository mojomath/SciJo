# ===----------------------------------------------------------------------=== #
# SciJo: Linalg module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Norms (`scijo.linalg.norms`)
===============================
SciPy-style vector/matrix norms.

Examples
--------
    ```mojo
    from scijo.linalg import norm
    import numojo as nm

    var x = nm.fromstring[nm.f64]("[3, 4]")
    print(norm(x))  # 5.0 (Euclidean norm)
    ```
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.math import sqrt

# ===----------------------------------------------------------------------=== #
# External
# ===----------------------------------------------------------------------=== #
from numojo.core.error import NumojoError
from numojo.core.ndarray import NDArray


def norm[
    dtype: DType
](x: NDArray[dtype], ord: String = "") raises -> Scalar[dtype]:
    """Compute a vector or matrix norm, matching `scipy.linalg.norm`.

    For a 1-D `x`, supported `ord` values are `"1"` (sum of absolute
    values), `"2"` or `""` (Euclidean norm, default), and `"inf"`
    (maximum absolute value).

    For a 2-D `x`, supported `ord` values are `"fro"` or `""`
    (Frobenius norm, default), `"1"` (maximum absolute column sum), and
    `"inf"` (maximum absolute row sum). The induced 2-norm (largest
    singular value) is not yet supported, as it requires an SVD.

    Parameters:
        dtype: Data type of the input array and the returned norm.

    Args:
        x: A 1-D vector or 2-D matrix.
        ord: Which norm to compute. See above for accepted values.

    Returns:
        The computed norm, as a scalar.

    Raises:
        NumojoError: If `x` is neither 1-D nor 2-D, or if `ord` is not a
            supported value for the rank of `x`.
    """

    if x.ndim == 1:
        var n = x.shape[0]
        if ord == "1":
            var total: Scalar[dtype] = 0
            for i in range(n):
                total += abs(x.item(i))
            return total
        elif ord == "inf":
            var m: Scalar[dtype] = 0
            for i in range(n):
                var v = abs(x.item(i))
                if v > m:
                    m = v
            return m
        elif ord == "2" or ord == "":
            var total: Scalar[dtype] = 0
            for i in range(n):
                total += x.item(i) * x.item(i)
            return sqrt(total)
        else:
            raise Error(
                NumojoError(
                    category="value",
                    message="Unsupported ord '"
                    + ord
                    + "' for a vector; use '1', '2', or 'inf'.",
                    location="norm",
                )
            )
    elif x.ndim == 2:
        var rows = x.shape[0]
        var cols = x.shape[1]
        if ord == "fro" or ord == "":
            var total: Scalar[dtype] = 0
            for i in range(rows):
                for j in range(cols):
                    var v = x.item(i, j)
                    total += v * v
            return sqrt(total)
        elif ord == "1":
            var best: Scalar[dtype] = 0
            for j in range(cols):
                var col_sum: Scalar[dtype] = 0
                for i in range(rows):
                    col_sum += abs(x.item(i, j))
                if col_sum > best:
                    best = col_sum
            return best
        elif ord == "inf":
            var best: Scalar[dtype] = 0
            for i in range(rows):
                var row_sum: Scalar[dtype] = 0
                for j in range(cols):
                    row_sum += abs(x.item(i, j))
                if row_sum > best:
                    best = row_sum
            return best
        else:
            raise Error(
                NumojoError(
                    category="value",
                    message="Unsupported ord '"
                    + ord
                    + "' for a matrix; use 'fro', '1', or 'inf'.",
                    location="norm",
                )
            )
    else:
        raise Error(
            NumojoError(
                category="shape",
                message="norm only supports 1-D or 2-D arrays.",
                location="norm",
            )
        )
