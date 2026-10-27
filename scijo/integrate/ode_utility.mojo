# ===----------------------------------------------------------------------=== #
# SciJo: Integrate module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""ODE Solver Utility (`scijo.integrate.ode_utility`)
=====================================================
Dormand-Prince (RK45) Butcher tableau coefficients, the dense-output
(interpolation) helper used by `scijo.integrate.ivp.solve_ivp` for `t_eval`,
and `OdeResult`, the result structure returned by `solve_ivp`.

References
----------
- Dormand, J. R.; Prince, P. J. (1980). "A family of embedded Runge-Kutta
  formulae." Journal of Computational and Applied Mathematics, 6(1), 19-26.
- Hairer, E.; Norsett, S. P.; Wanner, G. (1993). "Solving Ordinary
  Differential Equations I: Nonstiff Problems", 2nd ed., section II.4
  (initial step-size selection).
"""

# ===----------------------------------------------------------------------=== #
# External
# ===----------------------------------------------------------------------=== #
from numojo.core import Shape
from numojo.core.ndarray import NDArray
from numojo.routines.creation import zeros

# ===----------------------------------------------------------------------=== #
# Dormand-Prince (RK45) Butcher tableau
# ===----------------------------------------------------------------------=== #
# 7-stage, FSAL (first-same-as-last): the 7th stage derivative at the
# accepted point is exactly the derivative needed as stage 1 of the next
# step, so it is reused rather than recomputed.

comptime RK45_C2: Float64 = 1.0 / 5.0
comptime RK45_C3: Float64 = 3.0 / 10.0
comptime RK45_C4: Float64 = 4.0 / 5.0
comptime RK45_C5: Float64 = 8.0 / 9.0
comptime RK45_C6: Float64 = 1.0
comptime RK45_C7: Float64 = 1.0

comptime RK45_A21: Float64 = 1.0 / 5.0

comptime RK45_A31: Float64 = 3.0 / 40.0
comptime RK45_A32: Float64 = 9.0 / 40.0

comptime RK45_A41: Float64 = 44.0 / 45.0
comptime RK45_A42: Float64 = -56.0 / 15.0
comptime RK45_A43: Float64 = 32.0 / 9.0

comptime RK45_A51: Float64 = 19372.0 / 6561.0
comptime RK45_A52: Float64 = -25360.0 / 2187.0
comptime RK45_A53: Float64 = 64448.0 / 6561.0
comptime RK45_A54: Float64 = -212.0 / 729.0

comptime RK45_A61: Float64 = 9017.0 / 3168.0
comptime RK45_A62: Float64 = -355.0 / 33.0
comptime RK45_A63: Float64 = 46732.0 / 5247.0
comptime RK45_A64: Float64 = 49.0 / 176.0
comptime RK45_A65: Float64 = -5103.0 / 18656.0

comptime RK45_A71: Float64 = 35.0 / 384.0
comptime RK45_A72: Float64 = 0.0
comptime RK45_A73: Float64 = 500.0 / 1113.0
comptime RK45_A74: Float64 = 125.0 / 192.0
comptime RK45_A75: Float64 = -2187.0 / 6784.0
comptime RK45_A76: Float64 = 11.0 / 84.0

# 5th-order solution weights (identical to row 7 of A, by construction).
comptime RK45_B1: Float64 = RK45_A71
comptime RK45_B2: Float64 = RK45_A72
comptime RK45_B3: Float64 = RK45_A73
comptime RK45_B4: Float64 = RK45_A74
comptime RK45_B5: Float64 = RK45_A75
comptime RK45_B6: Float64 = RK45_A76

# Local error weights, `e = b - b*` (b* is the embedded 4th-order solution).
comptime RK45_E1: Float64 = 71.0 / 57600.0
comptime RK45_E2: Float64 = 0.0
comptime RK45_E3: Float64 = -71.0 / 16695.0
comptime RK45_E4: Float64 = 71.0 / 1920.0
comptime RK45_E5: Float64 = -17253.0 / 339200.0
comptime RK45_E6: Float64 = 22.0 / 525.0
comptime RK45_E7: Float64 = -1.0 / 40.0

# Step-size controller constants (Dormand-Prince error order is 4; SciPy's
# own `_ivp/rk.py` uses the same safety/clamp values).
comptime RK45_ERROR_EXPONENT: Float64 = -1.0 / 5.0
comptime RK45_SAFETY: Float64 = 0.9
comptime RK45_MIN_FACTOR: Float64 = 0.2
comptime RK45_MAX_FACTOR: Float64 = 10.0


# ===----------------------------------------------------------------------=== #
# Dense output (cubic Hermite interpolation for `t_eval`)
# ===----------------------------------------------------------------------=== #


def hermite_interp[
    dtype: DType
](
    theta: Scalar[dtype],
    h: Scalar[dtype],
    y_old: NDArray[dtype],
    y_new: NDArray[dtype],
    f_old: NDArray[dtype],
    f_new: NDArray[dtype],
) raises -> NDArray[dtype] where dtype.is_floating_point():
    """Interpolates within an accepted RK45 step using a cubic Hermite
    polynomial matching `y`/`y'` at both endpoints of the step.

    This is a cheaper, lower-order stand-in for the degree-4 dense-output
    polynomial RK45 can build from all 7 stage derivatives: it only costs
    the two endpoint derivatives already on hand (`f_old`, the step's `k1`,
    and `f_new`, its FSAL `k7`), at the cost of one order of local accuracy
    relative to the step itself. This keeps `t_eval` cheap without forcing
    the stepper to shrink its steps to land exactly on requested times.

    Parameters:
        dtype: The floating-point data type.

    Args:
        theta: Fractional position within the step, in `[0, 1]`.
        h: The step size (`t_new - t_old`).
        y_old: Solution at the start of the step.
        y_new: Solution at the end of the step.
        f_old: `f(t_old, y_old)`.
        f_new: `f(t_new, y_new)`.

    Returns:
        The interpolated solution at `t_old + theta * h`.
    """
    var n = len(y_old)
    var out = zeros[dtype](Shape(n))
    var t2 = theta * theta
    var t3 = t2 * theta
    var h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    var h10 = t3 - 2.0 * t2 + theta
    var h01 = -2.0 * t3 + 3.0 * t2
    var h11 = t3 - t2
    for i in range(n):
        out.store(
            i,
            val=h00 * y_old.load(i)
            + h10 * h * f_old.load(i)
            + h01 * y_new.load(i)
            + h11 * h * f_new.load(i),
        )
    return out^


# ===----------------------------------------------------------------------=== #
# Result type
# ===----------------------------------------------------------------------=== #


def get_ode_error_message(status: Int) -> String:
    """Returns the status message corresponding to a `solve_ivp` status code.

    Args:
        status: Solver status code (`0` = success, `<0` = failure).

    Returns:
        A human-readable description of the status.
    """
    if status == 0:
        return String(
            "The solver successfully reached the end of the integration"
            " interval."
        )
    elif status == -1:
        return String(
            "The solver could not keep local error within tolerance without"
            " the step size underflowing. The problem may be stiffer than"
            " RK45 can handle, or the tolerances may be too tight."
        )
    elif status == -2:
        return String(
            "The maximum number of steps was reached before the end of the"
            " integration interval. Increase `maxiter` or loosen tolerances."
        )
    else:
        return String("Unknown status code.")


struct OdeResult[dtype: DType = DType.float64](Copyable, Movable, Writable):
    """Result structure for initial value problem (ODE) solves.

    Encapsulates the solution trajectory and diagnostic information
    returned by `scijo.integrate.ivp.solve_ivp`.

    Parameters:
        dtype: The floating-point data type. Defaults to DType.float64.
    """

    var t: NDArray[Self.dtype]
    """Output times, of shape `(n_points,)`."""
    var y: NDArray[Self.dtype]
    """Solution values, of shape `(n_states, n_points)`: row `i` is the
    full trajectory of state `i`, column `j` is the state vector at
    `t[j]`."""
    var nfev: Int
    """Number of right-hand-side evaluations used."""
    var nsteps: Int
    """Number of steps taken (accepted and rejected)."""
    var success: Bool
    """Whether the solver reached the end of the integration interval."""
    var status: Int
    """Solver status code (`0` = success, `<0` = failure)."""
    var message: String
    """Human-readable status message."""
    var method: String
    """Name of the method used."""

    def __init__(
        out self,
        var t: NDArray[Self.dtype],
        var y: NDArray[Self.dtype],
        nfev: Int,
        nsteps: Int,
        success: Bool,
        status: Int,
        message: String,
        method: String,
    ):
        """Constructs an OdeResult from the outcome of an IVP solve.

        Args:
            t: Output times.
            y: Solution values.
            nfev: Number of right-hand-side evaluations used.
            nsteps: Number of steps taken (accepted and rejected).
            success: Whether the solver reached the end of the interval.
            status: Solver status code.
            message: Human-readable status message.
            method: Name of the method used.
        """
        self.t = t^
        self.y = y^
        self.nfev = nfev
        self.nsteps = nsteps
        self.success = success
        self.status = status
        self.message = message
        self.method = method

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
                    "ODE Solve Result\n"
                    "=================\n"
                    "Status  : {}\n"
                    "Message : {}\n"
                    "Method  : {}\n"
                    "Evals   : {}\n"
                    "Steps   : {}\n"
                    "t shape : ({},)\n"
                    "y shape : ({}, {})\n"
                ).format(
                    self.status,
                    self.message,
                    self.method,
                    self.nfev,
                    self.nsteps,
                    self.t.shape[0],
                    self.y.shape[0],
                    self.y.shape[1],
                )
            )
        except e:
            writer.write("Error displaying OdeResult: " + String(e) + "\n")
