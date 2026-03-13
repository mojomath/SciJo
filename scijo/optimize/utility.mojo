# ===----------------------------------------------------------------------=== #
# Scijo: Optimize - Utility
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Optimize Module - Utility Functions (scijo.optimize.utility)

Data structures for returning results from optimization and root-finding routines.
"""

# ===----------------------------------------------------------------------=== #
# RootResults
# ===----------------------------------------------------------------------=== #

struct RootResults[dtype: DType = DType.float64]():
    """Result structure for scalar root-finding operations.

    Encapsulates the computed root, convergence status, and diagnostic
    information returned by root-finding methods.

    Parameters:
        dtype: The floating-point data type. Defaults to DType.float64.
    """

    var root: Scalar[Self.dtype]
    """The estimated root value."""
    var iterations: Int
    """Number of iterations performed."""
    var function_calls: Int
    """Number of function evaluations used."""
    var converged: Bool
    """Whether the algorithm converged within tolerances."""
    var flag: String
    """Human-readable status message."""
    var method: String
    """Name of the method used."""

    fn __init__(
        out self,
        root: Scalar[Self.dtype],
        iterations: Int,
        function_calls: Int,
        converged: Bool,
        flag: String,
        method: String,
    ):
        self.root = root
        self.iterations = iterations
        self.function_calls = function_calls
        self.converged = converged
        self.flag = flag
        self.method = method

    fn __str__(self) raises -> String:
        return String(
            "RootResults(root={}, iterations={}, function_calls={}, "
            "converged={}, flag='{}', method='{}')"
        ).format(
            self.root,
            self.iterations,
            self.function_calls,
            self.converged,
            self.flag,
            self.method,
        )

    fn write_to[W: Writer](self, mut writer: W):
        try:
            writer.write(
                String(
                    "Root Results\n"
                    "============\n"
                    "Root          : {}\n"
                    "Iterations    : {}\n"
                    "Function Calls: {}\n"
                    "Converged     : {}\n"
                    "Flag          : {}\n"
                    "Method        : {}\n"
                ).format(
                    self.root,
                    self.iterations,
                    self.function_calls,
                    self.converged,
                    self.flag,
                    self.method,
                )
            )
        except e:
            writer.write("Error displaying RootResults: " + String(e) + "\n")
