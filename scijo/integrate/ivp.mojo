# ===----------------------------------------------------------------------=== #
# SciJo: Integrate module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Initial Value Problems (`scijo.integrate.ivp`)
=================================================
`solve_ivp`: adaptive-step numerical integration of `dy/dt = f(t, y)`.

Currently implements `method="RK45"` (Dormand-Prince 5(4), the same pair
SciPy's own `RK45` class uses) for non-stiff systems; other methods
(RK23, DOP853, Radau, BDF, LSODA), event functions, and a standalone
queryable dense-output object are not yet implemented.

References
----------
- Dormand, J. R.; Prince, P. J. (1980). "A family of embedded Runge-Kutta
  formulae." Journal of Computational and Applied Mathematics, 6(1), 19-26.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.math import (
    pow,
    sqrt,
)

# ===----------------------------------------------------------------------=== #
# External
# ===----------------------------------------------------------------------=== #
from numojo.core import Shape
from numojo.core.ndarray import NDArray
from numojo.routines.creation import zeros

# ===----------------------------------------------------------------------=== #
# SciJo
# ===----------------------------------------------------------------------=== #
from scijo.integrate.ode_utility import (
    get_ode_error_message,
    hermite_interp,
    OdeResult,
    RK45_A21,
    RK45_A31,
    RK45_A32,
    RK45_A41,
    RK45_A42,
    RK45_A43,
    RK45_A51,
    RK45_A52,
    RK45_A53,
    RK45_A54,
    RK45_A61,
    RK45_A62,
    RK45_A63,
    RK45_A64,
    RK45_A65,
    RK45_B1,
    RK45_B2,
    RK45_B3,
    RK45_B4,
    RK45_B5,
    RK45_B6,
    RK45_C2,
    RK45_C3,
    RK45_C4,
    RK45_C5,
    RK45_E1,
    RK45_E2,
    RK45_E3,
    RK45_E4,
    RK45_E5,
    RK45_E6,
    RK45_E7,
    RK45_ERROR_EXPONENT,
    RK45_MAX_FACTOR,
    RK45_MIN_FACTOR,
    RK45_SAFETY,
)
from scijo.integrate.utility import largest_positive_dtype

# ===----------------------------------------------------------------------=== #
# Public API
# ===----------------------------------------------------------------------=== #


def solve_ivp[
    dtype: DType,
    rhs_func: def[dtype: DType](
        t: Scalar[dtype], y: NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) raises thin capturing[_] -> NDArray[
        dtype
    ] where dtype.is_floating_point(),
    *,
    method: String = "RK45",
](
    t_span: Tuple[Scalar[dtype], Scalar[dtype]],
    y0: NDArray[dtype],
    args: Optional[List[Scalar[dtype]]] = None,
    t_eval: Optional[NDArray[dtype]] = None,
    atol: Scalar[dtype] = 1e-6,
    rtol: Scalar[dtype] = 1e-3,
    max_step: Scalar[dtype] = largest_positive_dtype[dtype],
    first_step: Optional[Scalar[dtype]] = None,
    maxiter: Int = 10000,
) raises -> OdeResult[dtype] where dtype.is_floating_point():
    """Solves the initial value problem `dy/dt = f(t, y)`, `y(t_span[0]) = y0`.

    With `method="RK45"` (the only method currently supported), the
    solution is advanced with the 7-stage Dormand-Prince pair: a 5th-order
    solution propagated each step, checked against an embedded 4th-order
    estimate to adapt the step size (shrinking and retrying a step whose
    estimated local error exceeds `atol`/`rtol`, growing the step otherwise).
    FSAL (first-same-as-last) means the last of the 7 stage derivatives
    computed for an accepted step is exactly the derivative needed to start
    the next one, so it costs one fewer `rhs_func` call than a naive 7-stage
    implementation.

    If `t_eval` is given, the returned trajectory holds only those times
    (interpolated within whichever accepted step contains each one, via a
    cheap cubic Hermite polynomial - see `scijo.integrate.ode_utility.hermite_interp`)
    instead of the solver's own adaptive step points.

    Parameters:
        dtype: The floating-point data type.
        rhs_func: The right-hand side, with signature `def(t, y, args) -> NDArray[dtype]`.
        method: Integration method. Only `"RK45"` is currently supported.

    Args:
        t_span: The integration interval `(t0, tf)`. `tf` may be less than
            `t0` to integrate backward in time.
        y0: Initial state, of shape `(n,)`.
        args: Optional arguments forwarded to `rhs_func`.
        t_eval: Times at which to report the solution, instead of the
            solver's own step points. Must lie within `t_span` and be
            sorted in the direction of integration.
        atol: Absolute error tolerance (elementwise).
        rtol: Relative error tolerance (elementwise).
        max_step: Maximum step size allowed.
        first_step: Initial step size. If not given, it is estimated
            automatically from `rhs_func`'s behavior at `t_span[0]`.
        maxiter: Maximum number of steps (accepted and rejected) before
            giving up.

    Returns:
        OdeResult[dtype] with the solution trajectory and diagnostics.

    Raises:
        Error: If an unsupported method is requested.

    Examples:
        ```mojo
        from scijo.integrate import solve_ivp
        from scijo.prelude import *
        import numojo as nm

        def decay[dtype: DType](t: Scalar[dtype], y: nm.NDArray[dtype], args: Optional[List[Scalar[dtype]]]) raises -> nm.NDArray[dtype]:
            return y * (-1.0)

        var y0 = nm.fromstring[f64]("[1.0]")
        var result = solve_ivp[f64, decay]((0.0, 1.0), y0)
        ```
    """
    comptime if method == "RK45":
        return _rk45_impl[dtype, rhs_func](
            t_span, y0, args, t_eval, atol, rtol, max_step, first_step, maxiter
        )
    else:
        raise Error("Scijo [solve_ivp]: Unsupported method: " + String(method))


def solve_ivp[
    dtype: DType,
    rhs_func: def[dtype: DType](
        t: Scalar[dtype], y: NDArray[dtype]
    ) raises thin capturing[_] -> NDArray[
        dtype
    ] where dtype.is_floating_point(),
    *,
    method: String = "RK45",
](
    t_span: Tuple[Scalar[dtype], Scalar[dtype]],
    y0: NDArray[dtype],
    t_eval: Optional[NDArray[dtype]] = None,
    atol: Scalar[dtype] = 1e-6,
    rtol: Scalar[dtype] = 1e-3,
    max_step: Scalar[dtype] = largest_positive_dtype[dtype],
    first_step: Optional[Scalar[dtype]] = None,
    maxiter: Int = 10000,
) raises -> OdeResult[dtype] where dtype.is_floating_point():
    """Solves an initial value problem (args-free overload).

    For right-hand sides that do not need `args`: `rhs_func` takes only
    `t` and `y`, so extra parameters (if any) are captured directly from
    the enclosing scope instead of threaded through `args`. See the
    `args`-taking overload `solve_ivp[dtype, rhs_func](..., args=...)` for
    a right-hand side reused across call sites with different `args`.

    Parameters:
        dtype: The floating-point data type.
        rhs_func: The right-hand side, with signature `def(t, y) -> NDArray[dtype]`.
        method: Integration method. Only `"RK45"` is currently supported.

    Args:
        t_span: The integration interval `(t0, tf)`.
        y0: Initial state, of shape `(n,)`.
        t_eval: Times at which to report the solution.
        atol: Absolute error tolerance (elementwise).
        rtol: Relative error tolerance (elementwise).
        max_step: Maximum step size allowed.
        first_step: Initial step size, auto-estimated if not given.
        maxiter: Maximum number of steps before giving up.

    Returns:
        OdeResult[dtype] with the solution trajectory and diagnostics.

    Raises:
        Error: If an unsupported method is requested.
    """

    def _with_args[
        dtype2: DType
    ](
        t: Scalar[dtype2],
        y: NDArray[dtype2],
        args: Optional[List[Scalar[dtype2]]],
    ) capturing raises -> NDArray[dtype2] where dtype2.is_floating_point():
        return rhs_func(t, y)

    return solve_ivp[dtype, _with_args, method=method](
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        atol=atol,
        rtol=rtol,
        max_step=max_step,
        first_step=first_step,
        maxiter=maxiter,
    )


# ===----------------------------------------------------------------------=== #
# RK45 implementation
# ===----------------------------------------------------------------------=== #


struct _RK45StepResult[dtype: DType](Movable):
    """Outcome of one adaptive RK45 step (possibly after internal retries)."""

    var accepted: Bool
    var y_new: NDArray[Self.dtype]
    var f_new: NDArray[Self.dtype]
    """`f(t + h_used, y_new)`, reused as the next step's first stage (FSAL)."""
    var h_used: Scalar[Self.dtype]
    var h_next: Scalar[Self.dtype]
    var nfev: Int

    def __init__(
        out self,
        accepted: Bool,
        var y_new: NDArray[Self.dtype],
        var f_new: NDArray[Self.dtype],
        h_used: Scalar[Self.dtype],
        h_next: Scalar[Self.dtype],
        nfev: Int,
    ):
        """Constructs an _RK45StepResult from the outcome of one step attempt.

        Args:
            accepted: Whether the step met the error tolerance.
            y_new: The candidate solution at `t + h_used`.
            f_new: `f(t + h_used, y_new)`, reused as the next step's `k1`.
            h_used: The step size actually taken (or attempted, if rejected).
            h_next: The step size to use next (grown or shrunk).
            nfev: Number of right-hand-side evaluations used by this attempt.
        """
        self.accepted = accepted
        self.y_new = y_new^
        self.f_new = f_new^
        self.h_used = h_used
        self.h_next = h_next
        self.nfev = nfev


def _rk45_attempt_step[
    dtype: DType,
    rhs_func: def[dtype: DType](
        t: Scalar[dtype], y: NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) raises thin capturing[_] -> NDArray[
        dtype
    ] where dtype.is_floating_point(),
](
    t: Scalar[dtype],
    y: NDArray[dtype],
    k1: NDArray[dtype],
    h: Scalar[dtype],
    args: Optional[List[Scalar[dtype]]],
    atol: Scalar[dtype],
    rtol: Scalar[dtype],
) raises -> _RK45StepResult[dtype] where dtype.is_floating_point():
    """Evaluates a single candidate RK45 step and its error norm (no
    retrying - the caller decides whether to accept it)."""
    var n = len(y)

    var y2 = y + (h * Scalar[dtype](RK45_A21)) * k1
    var k2 = rhs_func(t + Scalar[dtype](RK45_C2) * h, y2, args)

    var y3 = y + h * (
        Scalar[dtype](RK45_A31) * k1 + Scalar[dtype](RK45_A32) * k2
    )
    var k3 = rhs_func(t + Scalar[dtype](RK45_C3) * h, y3, args)

    var y4 = y + h * (
        Scalar[dtype](RK45_A41) * k1
        + Scalar[dtype](RK45_A42) * k2
        + Scalar[dtype](RK45_A43) * k3
    )
    var k4 = rhs_func(t + Scalar[dtype](RK45_C4) * h, y4, args)

    var y5 = y + h * (
        Scalar[dtype](RK45_A51) * k1
        + Scalar[dtype](RK45_A52) * k2
        + Scalar[dtype](RK45_A53) * k3
        + Scalar[dtype](RK45_A54) * k4
    )
    var k5 = rhs_func(t + Scalar[dtype](RK45_C5) * h, y5, args)

    var y6 = y + h * (
        Scalar[dtype](RK45_A61) * k1
        + Scalar[dtype](RK45_A62) * k2
        + Scalar[dtype](RK45_A63) * k3
        + Scalar[dtype](RK45_A64) * k4
        + Scalar[dtype](RK45_A65) * k5
    )
    var k6 = rhs_func(t + h, y6, args)

    var y_new = y + h * (
        Scalar[dtype](RK45_B1) * k1
        + Scalar[dtype](RK45_B2) * k2
        + Scalar[dtype](RK45_B3) * k3
        + Scalar[dtype](RK45_B4) * k4
        + Scalar[dtype](RK45_B5) * k5
        + Scalar[dtype](RK45_B6) * k6
    )
    var k7 = rhs_func(t + h, y_new, args)

    var y_err = h * (
        Scalar[dtype](RK45_E1) * k1
        + Scalar[dtype](RK45_E2) * k2
        + Scalar[dtype](RK45_E3) * k3
        + Scalar[dtype](RK45_E4) * k4
        + Scalar[dtype](RK45_E5) * k5
        + Scalar[dtype](RK45_E6) * k6
        + Scalar[dtype](RK45_E7) * k7
    )

    var sq_sum: Scalar[dtype] = 0.0
    for i in range(n):
        var yi = abs(y.load(i))
        var yni = abs(y_new.load(i))
        var scale_i = atol + rtol * (yi if yi > yni else yni)
        var r = y_err.load(i) / scale_i
        sq_sum += r * r
    var err_norm = sqrt(sq_sum / Scalar[dtype](n))

    var factor: Scalar[dtype]
    var accepted: Bool
    if err_norm <= 1.0:
        accepted = True
        if err_norm == 0.0:
            factor = Scalar[dtype](RK45_MAX_FACTOR)
        else:
            var raw = Scalar[dtype](RK45_SAFETY) * pow(
                err_norm, Scalar[dtype](RK45_ERROR_EXPONENT)
            )
            factor = raw if raw < Scalar[dtype](RK45_MAX_FACTOR) else Scalar[
                dtype
            ](RK45_MAX_FACTOR)
    else:
        accepted = False
        var raw = Scalar[dtype](RK45_SAFETY) * pow(
            err_norm, Scalar[dtype](RK45_ERROR_EXPONENT)
        )
        factor = raw if raw > Scalar[dtype](RK45_MIN_FACTOR) else Scalar[dtype](
            RK45_MIN_FACTOR
        )

    return _RK45StepResult[dtype](accepted, y_new^, k7^, h, h * factor, 6)


def _rk45_step[
    dtype: DType,
    rhs_func: def[dtype: DType](
        t: Scalar[dtype], y: NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) raises thin capturing[_] -> NDArray[
        dtype
    ] where dtype.is_floating_point(),
](
    t: Scalar[dtype],
    y: NDArray[dtype],
    k1: NDArray[dtype],
    h_init: Scalar[dtype],
    args: Optional[List[Scalar[dtype]]],
    atol: Scalar[dtype],
    rtol: Scalar[dtype],
) raises -> _RK45StepResult[dtype] where dtype.is_floating_point():
    """Takes one RK45 step at `t`, retrying with a shrunk step size on
    rejection (bounded, so a problem RK45 genuinely cannot handle fails
    fast instead of looping forever)."""
    var h = h_init
    var nfev = 0
    var attempt = 0
    while attempt < 50:
        attempt += 1
        var result = _rk45_attempt_step[dtype, rhs_func](
            t, y, k1, h, args, atol, rtol
        )
        var accepted = result.accepted
        var h_used = result.h_used
        var h_next = result.h_next
        var result_nfev = result.nfev
        var y_candidate = result.y_new.copy()
        var f_candidate = result.f_new.copy()
        nfev += result_nfev
        if accepted:
            return _RK45StepResult[dtype](
                True, y_candidate^, f_candidate^, h_used, h_next, nfev
            )
        h = h_next
        if abs(h) < 1e-13 * (abs(t) + 1.0):
            return _RK45StepResult[dtype](
                False, y_candidate^, f_candidate^, h, h, nfev
            )
    var failed = _rk45_attempt_step[dtype, rhs_func](
        t, y, k1, h, args, atol, rtol
    )
    var failed_nfev = failed.nfev
    var failed_y = failed.y_new.copy()
    var failed_f = failed.f_new.copy()
    nfev += failed_nfev
    return _RK45StepResult[dtype](False, failed_y^, failed_f^, h, h, nfev)


def _select_initial_step[
    dtype: DType,
    rhs_func: def[dtype: DType](
        t: Scalar[dtype], y: NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) raises thin capturing[_] -> NDArray[
        dtype
    ] where dtype.is_floating_point(),
](
    t0: Scalar[dtype],
    y0: NDArray[dtype],
    f0: NDArray[dtype],
    direction: Scalar[dtype],
    args: Optional[List[Scalar[dtype]]],
    atol: Scalar[dtype],
    rtol: Scalar[dtype],
) raises -> Scalar[dtype] where dtype.is_floating_point():
    """Estimates an initial step size (Hairer, Norsett & Wanner's
    algorithm, section II.4 - the same one SciPy's `select_initial_step`
    implements)."""
    var n = len(y0)

    var d0_sq: Scalar[dtype] = 0.0
    var d1_sq: Scalar[dtype] = 0.0
    for i in range(n):
        var scale_i = atol + rtol * abs(y0.load(i))
        var r0 = y0.load(i) / scale_i
        var r1 = f0.load(i) / scale_i
        d0_sq += r0 * r0
        d1_sq += r1 * r1
    var d0 = sqrt(d0_sq / Scalar[dtype](n))
    var d1 = sqrt(d1_sq / Scalar[dtype](n))

    var h0: Scalar[dtype]
    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1

    var y1 = y0 + (direction * h0) * f0
    var f1 = rhs_func(t0 + direction * h0, y1, args)

    var d2_sq: Scalar[dtype] = 0.0
    for i in range(n):
        var scale_i = atol + rtol * abs(y0.load(i))
        var r2 = (f1.load(i) - f0.load(i)) / scale_i
        d2_sq += r2 * r2
    var d2 = sqrt(d2_sq / Scalar[dtype](n)) / h0

    var h1: Scalar[dtype]
    if d1 <= 1e-15 and d2 <= 1e-15:
        var floor_h = h0 * 1e-3
        var floor_lit = Scalar[dtype](1e-6)
        h1 = floor_lit if floor_lit > floor_h else floor_h
    else:
        var dmax = d1 if d1 > d2 else d2
        h1 = pow(0.01 / dmax, Scalar[dtype](0.2))

    var h0_100 = 100.0 * h0
    var h_final = h0_100 if h0_100 < h1 else h1
    return direction * h_final


def _rk45_impl[
    dtype: DType,
    rhs_func: def[dtype: DType](
        t: Scalar[dtype], y: NDArray[dtype], args: Optional[List[Scalar[dtype]]]
    ) raises thin capturing[_] -> NDArray[
        dtype
    ] where dtype.is_floating_point(),
](
    t_span: Tuple[Scalar[dtype], Scalar[dtype]],
    y0: NDArray[dtype],
    args: Optional[List[Scalar[dtype]]],
    t_eval: Optional[NDArray[dtype]],
    atol: Scalar[dtype],
    rtol: Scalar[dtype],
    max_step: Scalar[dtype],
    first_step: Optional[Scalar[dtype]],
    maxiter: Int,
) raises -> OdeResult[dtype] where dtype.is_floating_point():
    var t0 = t_span[0]
    var tf = t_span[1]
    var direction: Scalar[dtype] = 1.0 if tf >= t0 else -1.0
    var n = len(y0)

    var t = t0
    var y = y0.copy()
    var f = rhs_func(t, y, args)
    var nfev = 1

    var h: Scalar[dtype]
    if first_step:
        h = direction * abs(first_step.value())
    else:
        h = _select_initial_step[dtype, rhs_func](
            t, y, f, direction, args, atol, rtol
        )
        nfev += 1

    var t_list = List[Scalar[dtype]]()
    var y_list = List[NDArray[dtype]]()
    t_list.append(t)
    y_list.append(y.copy())

    var n_eval = 0
    if t_eval:
        n_eval = len(t_eval.value())
    var eval_idx = 0
    var eval_t_list = List[Scalar[dtype]]()
    var eval_y_list = List[NDArray[dtype]]()

    var nsteps = 0
    var status = 0
    var message = get_ode_error_message(0)

    while (t - tf) * direction < 0:
        if nsteps >= maxiter:
            status = -2
            message = get_ode_error_message(-2)
            break

        var h_try = h
        if abs(h_try) > max_step:
            h_try = direction * max_step
        if (t + h_try - tf) * direction > 0:
            h_try = tf - t

        var step = _rk45_step[dtype, rhs_func](t, y, f, h_try, args, atol, rtol)
        var step_accepted = step.accepted
        var step_h_used = step.h_used
        var step_h_next = step.h_next
        var step_nfev = step.nfev
        var step_y_new = step.y_new.copy()
        var step_f_new = step.f_new.copy()
        nsteps += 1
        nfev += step_nfev

        if not step_accepted:
            status = -1
            message = get_ode_error_message(-1)
            break

        var t_new = t + step_h_used

        if t_eval:
            var te = t_eval.value().copy()
            while eval_idx < n_eval and (
                (te.item(eval_idx) - t_new) * direction <= 0
            ):
                var theta = (te.item(eval_idx) - t) / step_h_used
                var y_interp = hermite_interp[dtype](
                    theta, step_h_used, y, step_y_new, f, step_f_new
                )
                eval_t_list.append(te.item(eval_idx))
                eval_y_list.append(y_interp^)
                eval_idx += 1

        t = t_new
        y = step_y_new^
        f = step_f_new^
        h = step_h_next

        t_list.append(t)
        y_list.append(y.copy())

    var success = status == 0

    if t_eval:
        var n_out = len(eval_t_list)
        var t_out = zeros[dtype](Shape(n_out))
        var y_out = zeros[dtype](Shape(n, n_out))
        for j in range(n_out):
            t_out.store(j, val=eval_t_list[j])
            for i in range(n):
                y_out.store(i * n_out + j, val=eval_y_list[j].load(i))
        return OdeResult[dtype](
            t_out^, y_out^, nfev, nsteps, success, status, message, "RK45"
        )
    else:
        var n_out = len(t_list)
        var t_out = zeros[dtype](Shape(n_out))
        var y_out = zeros[dtype](Shape(n, n_out))
        for j in range(n_out):
            t_out.store(j, val=t_list[j])
            for i in range(n):
                y_out.store(i * n_out + j, val=y_list[j].load(i))
        return OdeResult[dtype](
            t_out^, y_out^, nfev, nsteps, success, status, message, "RK45"
        )
