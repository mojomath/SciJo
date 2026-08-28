# ===----------------------------------------------------------------------=== #
# SciJo: Optimize module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Optimization Utility Functions (`scijo.optimize.utility`)
============================================================
Data structures for returning results from optimization and root-finding routines.
"""

# ===----------------------------------------------------------------------=== #
# RootResults
# ===----------------------------------------------------------------------=== #


struct RootResult[dtype: DType = DType.float64]():
    """Result structure for scalar root-finding operations.

    Encapsulates the computed root, convergence status, and diagnostic
    information returned by root-finding methods.

    Parameters:
        dtype: The floating-point data type. Defaults to DType.float64.
    """

    var root: Scalar[Self.dtype]
    """The estimated root value."""
    var nit: Int
    """Number of iterations performed."""
    var nfev: Int
    """Number of function evaluations used."""
    var success: Bool
    """Whether the algorithm converged within tolerances."""
    var message: String
    """Human-readable status message."""
    var method: String
    """Name of the method used."""

    def __init__(
        out self,
        root: Scalar[Self.dtype],
        nit: Int,
        nfev: Int,
        success: Bool,
        message: String,
        method: String,
    ):
        """Constructs a RootResult from the outcome of a root-finding run.

        Args:
            root: The estimated root value.
            nit: Number of iterations performed.
            nfev: Number of function evaluations used.
            success: Whether the algorithm converged within tolerances.
            message: Human-readable status message.
            method: Name of the method used.
        """
        self.root = root
        self.nit = nit
        self.nfev = nfev
        self.success = success
        self.message = message
        self.method = method

    def __str__(self) raises -> String:
        """Returns a single-line summary of the result.

        Returns:
            A compact string representation of this RootResult.
        """
        return String(
            "RootResult(root={}, nit={}, nfev={}, "
            "success={}, message='{}', method='{}')"
        ).format(
            self.root,
            self.nit,
            self.nfev,
            self.success,
            self.message,
            self.method,
        )

    def write_to[W: Writer](self, mut writer: W):
        """Writes a formatted, multi-line report of the result.

        Parameters:
            W: The writer type.

        Args:
            writer: The writer to write the report to.
        """
        try:
            writer.write(
                String(
                    "Root Result\n"
                    "===========\n"
                    "Root    : {}\n"
                    "Iters   : {}\n"
                    "Evals   : {}\n"
                    "Success : {}\n"
                    "Message : {}\n"
                    "Method  : {}\n"
                ).format(
                    self.root,
                    self.nit,
                    self.nfev,
                    self.success,
                    self.message,
                    self.method,
                )
            )
        except e:
            writer.write("Error displaying RootResult: " + String(e) + "\n")
