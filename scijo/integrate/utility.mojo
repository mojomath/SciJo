# ===----------------------------------------------------------------------=== #
# SciJo: Integrate module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Integration Utility Functions (`scijo.integrate.utility`)
===========================================================
Utility functions, data structures, and Gauss-Kronrod quadrature tables for
numerical integration. Includes result types, priority queue for adaptive
subdivision, machine epsilon computation, and precomputed quadrature nodes
and weights from Netlib QUADPACK.

References
----------
- Netlib QUADPACK: https://www.netlib.org/quadpack/
- Advanpix G10K21 coefficients:
  https://www.advanpix.com/2011/11/07/gauss-kronrod-quadrature-nodes-weights/
"""

from std.utils import StaticTuple
from std.utils.numerics import min_finite, max_finite

comptime smallest_positive_dtype[dtype: DType] = min_finite[dtype]()
comptime largest_positive_dtype[dtype: DType] = max_finite[dtype]()

# TODO: Remove predefined messages in IntegralResult and add custom result according to the result.


# ===----------------------------------------------------------------------=== #
# Helpers
# ===----------------------------------------------------------------------=== #


def machine_epsilon[dtype: DType]() -> Float64 where dtype.is_floating_point():
    """Returns the machine epsilon for the given floating-point dtype.

    Parameters:
        dtype: The floating-point data type (float16, float32, or float64).

    Returns:
        The machine epsilon as a Float64 value.
    """

    # TODO: Check if these values are correct lol
    comptime if dtype == DType.float16:
        return Float64(0.0009765625)  # 2**-10
    elif dtype == DType.float32:
        return Float64(1.1920928955078125e-07)  # 2**-23
    else:
        return Float64(2.220446049250313e-16)  # 2**-52


# ===----------------------------------------------------------------------=== #
# Adaptive intervals
# ===----------------------------------------------------------------------=== #


struct QAGSInterval[dtype: DType](ImplicitlyCopyable, Movable):
    """Represents an integration subinterval with error estimate for adaptive subdivision.

    Parameters:
        dtype: The floating-point data type.
    """

    var a: Float64
    """Left endpoint."""
    var b: Float64
    """Right endpoint."""
    var integral: Float64
    """Integral estimate for this interval."""
    var error: Float64
    """Error estimate for this interval."""
    var level: Int
    """Subdivision level."""

    def __init__(
        out self,
        a: Float64,
        b: Float64,
        integral: Float64,
        error: Float64,
        level: Int = 0,
    ):
        self.a = a
        self.b = b
        self.integral = integral
        self.error = error
        self.level = level


struct QAGSPriorityQueue[dtype: DType]:
    """Max-heap priority queue for QAGS intervals, ordered by error estimate.

    Parameters:
        dtype: The floating-point data type.
    """

    var intervals: List[QAGSInterval[Self.dtype]]
    """Heap-ordered list of integration intervals."""

    def __init__(out self):
        self.intervals = List[QAGSInterval[Self.dtype]]()

    def __len__(self) -> Int:
        return len(self.intervals)

    def is_empty(self) -> Bool:
        """Returns True if the queue contains no intervals."""
        return len(self.intervals) == 0

    def _parent(self, i: Int) -> Int:
        return (i - 1) // 2

    def _left_child(self, i: Int) -> Int:
        return 2 * i + 1

    def _right_child(self, i: Int) -> Int:
        return 2 * i + 2

    def _swap(mut self, i: Int, j: Int):
        var temp = self.intervals[i]
        self.intervals[i] = self.intervals[j]
        self.intervals[j] = temp

    def _heapify_up(mut self, index: Int):
        if index == 0:
            return

        var parent_idx = self._parent(index)
        if self.intervals[index].error > self.intervals[parent_idx].error:
            self._swap(index, parent_idx)
            self._heapify_up(parent_idx)

    def _heapify_down(mut self, index: Int):
        var largest = index
        var left = self._left_child(index)
        var right = self._right_child(index)
        var size = len(self.intervals)

        if (
            left < size
            and self.intervals[left].error > self.intervals[largest].error
        ):
            largest = left

        if (
            right < size
            and self.intervals[right].error > self.intervals[largest].error
        ):
            largest = right

        if largest != index:
            self._swap(index, largest)
            self._heapify_down(largest)

    def push(mut self, interval: QAGSInterval[Self.dtype]):
        """Adds an interval to the priority queue.

        Args:
            interval: The integration interval to insert.
        """
        self.intervals.append(interval)
        self._heapify_up(len(self.intervals) - 1)

    def pop(mut self) -> QAGSInterval[Self.dtype]:
        """Removes and returns the interval with the largest error estimate.

        Returns:
            The interval with maximum error. Returns a zero-valued interval
            if the queue is empty.
        """
        if self.is_empty():
            return QAGSInterval[Self.dtype](
                Float64(0),
                Float64(0),
                Float64(0),
                Float64(0),
            )

        var max_interval = self.intervals[0]
        var last_interval = self.intervals.pop()

        if not self.is_empty():
            self.intervals[0] = last_interval
            self._heapify_down(0)

        return max_interval

    def peek(self) -> QAGSInterval[Self.dtype]:
        """Returns the interval with the largest error estimate without removing it.

        Returns:
            The interval at the top of the heap.
        """
        return self.intervals[0]


def get_quad_error_message(ier: Int) -> String:
    """Returns the error message corresponding to a QUADPACK integration error code.

    Args:
        ier: Integration error code (0 = success, >0 = error type).

    Returns:
        A human-readable description of the error condition.
    """
    if ier == 0:
        return String("The integral converged successfully.")
    elif ier == 1:
        return String(
            "Maximum subdivisions reached. The integrand may have singularities"
            " or discontinuities. Try splitting the integration interval at"
            " problem points or using a specialized integrator."
        )
    elif ier == 2:
        return String(
            "Roundoff error prevents achieving the requested tolerance. The"
            " error estimate may be inaccurate."
        )
    elif ier == 3:
        return String(
            "The function has problematic behavior at some points in the"
            " integration interval."
        )
    elif ier == 4:
        return String(
            "Algorithm failed to converge due to roundoff errors. The result"
            " may be the best possible approximation."
        )
    elif ier == 5:
        return String(
            "The integral is likely divergent or converges very slowly."
        )
    elif ier == 6:
        return String(
            "Invalid input parameters: tolerance values or limits are out of"
            " valid range."
        )
    else:
        return String("Unknown error code.")


# ===----------------------------------------------------------------------=== #
# Result types
# ===----------------------------------------------------------------------=== #


struct IntegralResult[dtype: DType](Copyable, Movable, Writable):
    """Result structure for numerical integration operations.

    Encapsulates the computed integral value, error estimate, function evaluation
    count, and status code returned by quadrature routines.

    Parameters:
        dtype: The floating-point data type (e.g., DType.float32, DType.float64).
    """

    var integral: Scalar[Self.dtype]
    """The computed integral value."""
    var abserr: Scalar[Self.dtype]
    """Absolute error estimate."""
    var nfev: Int
    """Number of function evaluations used."""
    var ier: Int
    """Integration error code (0 = success, >0 = error type)."""

    def __init__(
        out self,
        integral: Scalar[Self.dtype] = 0,
        abserr: Scalar[Self.dtype] = 0,
        nfev: Int = 0,
        ier: Int = 0,
    ):
        self.integral = integral
        self.abserr = abserr
        self.nfev = nfev
        self.ier = ier

    def success(self) -> Bool:
        """Returns True if integration converged successfully (ier == 0)."""
        return self.ier == 0

    def message(self) -> String:
        """Returns a human-readable status message for this result."""
        return get_quad_error_message(self.ier)

    def __str__(self) raises -> String:
        var status = "SUCCESS" if self.success() else "ERROR"
        return String(
            "QuadResult(status={}, message='{}', integral={}, abserr={:.2e},"
            " nfev={})"
        ).format(
            status,
            self.message(),
            self.integral,
            self.abserr,
            self.nfev,
        )

    def write_to[W: Writer](self, mut writer: W):
        try:
            var status = "SUCCESS" if self.success() else "ERROR"
            writer.write(
                String(
                    "Numerical Integration Result\n"
                    + "============================\n"
                    + "Status      : {}\n"
                    + "Message     : {}\n"
                    + "Integral    : {}\n"
                    + "Error Est.  : {}\n"
                    + "Func Evals  : {}\n"
                ).format(
                    status,
                    self.message(),
                    self.integral,
                    self.abserr,
                    self.nfev,
                )
            )
        except e:
            writer.write("Error displaying QuadResult: " + String(e) + "\n")


# ===----------------------------------------------------------------------=== #
# Gauss-Kronrod Quadrature Rules (from Netlib QUADPACK qng.f)
# ===----------------------------------------------------------------------=== #
# These rules use symmetric nodes about 0, so we only store positive nodes
# and weights. During integration, we evaluate f(a+x) + f(a-x) for each node.

# -----------------------------------------------------------------------------
# G5K10: 10-point Kronrod rule with embedded 5-point Gauss rule
# x1 in QUADPACK: nodes common to 10-, 21-, 43-, and 87-point rules
# w10 in QUADPACK: weights of the 10-point formula
# -----------------------------------------------------------------------------
comptime x1_nodes: StaticTuple[Float64, 5] = StaticTuple[Float64, 5](
    0.9739065285171717,  # ± (Gauss node)
    0.8650633666889845,  # ± (Gauss node)
    0.6794095682990244,  # ± (Gauss node)
    0.4333953941292472,  # ± (Gauss node)
    0.1488743389816312,  # ± (Gauss node)
)

comptime w10_gauss_weights: StaticTuple[Float64, 5] = StaticTuple[Float64, 5](
    0.0666713443086881,  # ±0.9739065285171717
    0.1494513491505806,  # ±0.8650633666889845
    0.2190863625159820,  # ±0.6794095682990244
    0.2692667193099964,  # ±0.4333953941292472
    0.2955242247147529,  # ±0.1488743389816312
)

# -----------------------------------------------------------------------------
# G10K21: 21-point Kronrod rule with embedded 10-point Gauss rule
# x1 + x2 in QUADPACK
# w21a: weights for x1 nodes, w21b: weights for x2 nodes (+ center)
# -----------------------------------------------------------------------------
comptime x2_nodes: StaticTuple[Float64, 5] = StaticTuple[Float64, 5](
    0.9956571630258081,  # ± (Kronrod only)
    0.9301574913557082,  # ± (Kronrod only)
    0.7808177265864169,  # ± (Kronrod only)
    0.5627571346686047,  # ± (Kronrod only)
    0.2943928627014602,  # ± (Kronrod only)
)

comptime w21a_kronrod_weights: StaticTuple[Float64, 5] = StaticTuple[
    Float64, 5
](
    0.0325816230796473,  # for x1[0] (Gauss node)
    0.0750396748109200,  # for x1[1] (Gauss node)
    0.1093871588022976,  # for x1[2] (Gauss node)
    0.1347092173114733,  # for x1[3] (Gauss node)
    0.1477391049013385,  # for x1[4] (Gauss node)
)

comptime w21b_kronrod_weights: StaticTuple[Float64, 6] = StaticTuple[
    Float64, 6
](
    0.0116946388673187,  # for x2[0] (Kronrod only)
    0.0547558965743520,  # for x2[1] (Kronrod only)
    0.0931254545836976,  # for x2[2] (Kronrod only)
    0.1234919762620659,  # for x2[3] (Kronrod only)
    0.1427759385770601,  # for x2[4] (Kronrod only)
    0.1494455540029169,  # center point (x=0)
)

# -----------------------------------------------------------------------------
# G21K43: 43-point Kronrod rule with embedded 21-point Gauss-Kronrod rule
# x1 + x2 + x3 in QUADPACK
# w43a: weights for x1, x3 nodes; w43b: weights for x3 nodes (+ center)
# -----------------------------------------------------------------------------
comptime x3_nodes: StaticTuple[Float64, 11] = StaticTuple[Float64, 11](
    0.9993333609019321,  # ± (Kronrod only)
    0.9874334029080889,  # ± (Kronrod only)
    0.9548079348142663,  # ± (Kronrod only)
    0.9001486957483283,  # ± (Kronrod only)
    0.8251983149831142,  # ± (Kronrod only)
    0.7321483889893050,  # ± (Kronrod only)
    0.6228479705377252,  # ± (Kronrod only)
    0.4994795740710565,  # ± (Kronrod only)
    0.3649016613465808,  # ± (Kronrod only)
    0.2222549197766013,  # ± (Kronrod only)
    0.0746506174613833,  # ± (Kronrod only)
)

comptime w43a_kronrod_weights: StaticTuple[Float64, 10] = StaticTuple[
    Float64, 10
](
    0.0162967342896666,  # for x1[0]
    0.0375228761208695,  # for x1[1]
    0.0546949020582554,  # for x1[2]
    0.0673554146094781,  # for x1[3]
    0.0738701996323940,  # for x1[4]
    0.0057685560597698,  # for x2[0]
    0.0273718905932488,  # for x2[1]
    0.0465608269104288,  # for x2[2]
    0.0617449952014426,  # for x2[3]
    0.0713872672686934,  # for x2[4]
)

comptime w43b_kronrod_weights: StaticTuple[Float64, 12] = StaticTuple[
    Float64, 12
](
    0.0018444776402124,  # for x3[0]
    0.0107986895858917,  # for x3[1]
    0.0218953638677954,  # for x3[2]
    0.0325974639753457,  # for x3[3]
    0.0421631379351918,  # for x3[4]
    0.0507419396001846,  # for x3[5]
    0.0583793955426193,  # for x3[6]
    0.0647464049514459,  # for x3[7]
    0.0695661979123565,  # for x3[8]
    0.0728244414718332,  # for x3[9]
    0.0745077510141751,  # for x3[10]
    0.0747221475174030,  # center point (x=0)
)

# -----------------------------------------------------------------------------
# G43K87: 87-point Kronrod rule with embedded 43-point Gauss-Kronrod rule
# x1 + x2 + x3 + x4 in QUADPACK
# w87a: weights for x1, x2, x3 nodes; w87b: weights for x4 nodes (+ center)
# -----------------------------------------------------------------------------
comptime x4_nodes: StaticTuple[Float64, 22] = StaticTuple[Float64, 22](
    0.9999029772627292,  # ± (Kronrod only)
    0.9979898959866787,  # ± (Kronrod only)
    0.9921754978606872,  # ± (Kronrod only)
    0.9813581635727128,  # ± (Kronrod only)
    0.9650576238583846,  # ± (Kronrod only)
    0.9431676131336706,  # ± (Kronrod only)
    0.9158064146855072,  # ± (Kronrod only)
    0.8832216577713165,  # ± (Kronrod only)
    0.8457107484624157,  # ± (Kronrod only)
    0.8035576580352310,  # ± (Kronrod only)
    0.7570057306854956,  # ± (Kronrod only)
    0.7062732097873218,  # ± (Kronrod only)
    0.6515894665011779,  # ± (Kronrod only)
    0.5932233740579611,  # ± (Kronrod only)
    0.5314936059708319,  # ± (Kronrod only)
    0.4667636230420228,  # ± (Kronrod only)
    0.3994248478592188,  # ± (Kronrod only)
    0.3298748771061883,  # ± (Kronrod only)
    0.2585035592021616,  # ± (Kronrod only)
    0.1856953965683467,  # ± (Kronrod only)
    0.1118422131799075,  # ± (Kronrod only)
    0.0373521233946199,  # ± (Kronrod only)
)

comptime w87a_kronrod_weights: StaticTuple[Float64, 21] = StaticTuple[
    Float64, 21
](
    0.0081483773841492,  # for x1[0]
    0.0187614382015628,  # for x1[1]
    0.0273474510500523,  # for x1[2]
    0.0336777073116379,  # for x1[3]
    0.0369350998204279,  # for x1[4]
    0.0028848724302115,  # for x2[0]
    0.0136859460227127,  # for x2[1]
    0.0232804135028883,  # for x2[2]
    0.0308724976117134,  # for x2[3]
    0.0356936336394188,  # for x2[4]
    0.0009152833452022,  # for x3[0]
    0.0053992802193005,  # for x3[1]
    0.0109476796011189,  # for x3[2]
    0.0162987316967873,  # for x3[3]
    0.0210815688892038,  # for x3[4]
    0.0253709697692538,  # for x3[5]
    0.0291896977564758,  # for x3[6]
    0.0323732024672028,  # for x3[7]
    0.0347830989503651,  # for x3[8]
    0.0364122207313518,  # for x3[9]
    0.0372538755030477,  # for x3[10]
)

comptime w87b_kronrod_weights: StaticTuple[Float64, 23] = StaticTuple[
    Float64, 23
](
    0.0002741455637621,  # for x4[0]
    0.0018071241550579,  # for x4[1]
    0.0040968692827592,  # for x4[2]
    0.0067582900518474,  # for x4[3]
    0.0095499576722016,  # for x4[4]
    0.0123294476522449,  # for x4[5]
    0.0150104473463890,  # for x4[6]
    0.0175489679862432,  # for x4[7]
    0.0199380377864409,  # for x4[8]
    0.0221949359610123,  # for x4[9]
    0.0243391471260008,  # for x4[10]
    0.0263745054148392,  # for x4[11]
    0.0282869107887712,  # for x4[12]
    0.0300525811280927,  # for x4[13]
    0.0316467513714399,  # for x4[14]
    0.0330504134199785,  # for x4[15]
    0.0342550997042261,  # for x4[16]
    0.0352624126701567,  # for x4[17]
    0.0360769896228887,  # for x4[18]
    0.0366986044984561,  # for x4[19]
    0.0371205492698326,  # for x4[20]
    0.0373342287519350,  # for x4[21]
    0.0373610737626790,  # center point (x=0)
)


# ===----------------------------------------------------------------------=== #
# Complete Gauss-Kronrod Rules (for QAGS algo, this is the better format)
# ===----------------------------------------------------------------------=== #

# -----------------------------------------------------------------------------
# GK15: 15-point Gauss-Kronrod (7-point Gauss + 8 Kronrod extensions)
# Standard rule used in QAGS/QAG
# -----------------------------------------------------------------------------
comptime gk15_nodes: StaticTuple[Float64, 8] = StaticTuple[Float64, 8](
    0.0000000000000000,  # 0: center (Gauss)
    0.2077849550078985,  # 1: ± (Kronrod)
    0.4058451513773972,  # 2: ± (Gauss)
    0.5860872354676911,  # 3: ± (Kronrod)
    0.7415311855993944,  # 4: ± (Gauss)
    0.8648644233597691,  # 5: ± (Kronrod)
    0.9491079123427585,  # 6: ± (Gauss)
    0.9914553711208126,  # 7: ± (Kronrod)
)

comptime gk15_kronrod_weights: StaticTuple[Float64, 8] = StaticTuple[
    Float64, 8
](
    0.2094821410847278,  # 0: center
    0.2044329400752989,  # 1: ±0.2077849550078985
    0.1903505780647854,  # 2: ±0.4058451513773972 (Gauss)
    0.1690047266392679,  # 3: ±0.5860872354676911
    0.1406532597155259,  # 4: ±0.7415311855993944 (Gauss)
    0.1047900103222502,  # 5: ±0.8648644233597691
    0.0630920926299786,  # 6: ±0.9491079123427585 (Gauss)
    0.0229353220105292,  # 7: ±0.9914553711208126
)

comptime gk15_gauss_weights: StaticTuple[Float64, 4] = StaticTuple[Float64, 4](
    0.4179591836734694,  # 0: center (0.0) - note: Gauss uses double weight
    0.3818300505051189,  # 2: ±0.4058451513773972
    0.2797053914892767,  # 4: ±0.7415311855993944
    0.1294849661688697,  # 6: ±0.9491079123427585
)

# -----------------------------------------------------------------------------
# GK21: 21-point Gauss-Kronrod (10-point Gauss + 11 Kronrod extensions)
# -----------------------------------------------------------------------------
comptime gk21_nodes: StaticTuple[Float64, 11] = StaticTuple[Float64, 11](
    0.0000000000000000,  # 0: center (Gauss)
    0.1488743389816312,  # 1: ± (Gauss)
    0.2943928627014602,  # 2: ± (Kronrod)
    0.4333953941292472,  # 3: ± (Gauss)
    0.5627571346686047,  # 4: ± (Kronrod)
    0.6794095682990244,  # 5: ± (Gauss)
    0.7808177265864169,  # 6: ± (Kronrod)
    0.8650633666889845,  # 7: ± (Gauss)
    0.9301574913557082,  # 8: ± (Kronrod)
    0.9739065285171717,  # 9: ± (Gauss)
    0.9956571630258081,  # 10: ± (Kronrod)
)

comptime gk21_kronrod_weights: StaticTuple[Float64, 11] = StaticTuple[
    Float64, 11
](
    0.1494455540029169,  # 0: center
    0.1477391049013385,  # 1: ±0.1488743389816312 (Gauss)
    0.1427759385770601,  # 2: ±0.2943928627014602
    0.1347092173114733,  # 3: ±0.4333953941292472 (Gauss)
    0.1234919762620659,  # 4: ±0.5627571346686047
    0.1093871588022976,  # 5: ±0.6794095682990244 (Gauss)
    0.0931254545836976,  # 6: ±0.7808177265864169
    0.0750396748109200,  # 7: ±0.8650633666889845 (Gauss)
    0.0547558965743520,  # 8: ±0.9301574913557082
    0.0325816230796473,  # 9: ±0.9739065285171717 (Gauss)
    0.0116946388673187,  # 10: ±0.9956571630258081
)

comptime gk21_gauss_weights: StaticTuple[Float64, 5] = StaticTuple[Float64, 5](
    0.2955242247147529,  # 1: ±0.1488743389816312
    0.2692667193099964,  # 3: ±0.4333953941292472
    0.2190863625159820,  # 5: ±0.6794095682990244
    0.1494513491505806,  # 7: ±0.8650633666889845
    0.0666713443086881,  # 9: ±0.9739065285171717
)

# -----------------------------------------------------------------------------
# GK43: 43-point Gauss-Kronrod (21-point GK + 22 Kronrod extensions)
# -----------------------------------------------------------------------------
comptime gk43_nodes: StaticTuple[Float64, 22] = StaticTuple[Float64, 22](
    0.0000000000000000,  # 0: center
    0.0746506174613833,  # 1: ± (Kronrod)
    0.1488743389816312,  # 2: ± (from GK21)
    0.2222549197766013,  # 3: ± (Kronrod)
    0.2943928627014602,  # 4: ± (from GK21)
    0.3649016613465808,  # 5: ± (Kronrod)
    0.4333953941292472,  # 6: ± (from GK21)
    0.4994795740710565,  # 7: ± (Kronrod)
    0.5627571346686047,  # 8: ± (from GK21)
    0.6228479705377252,  # 9: ± (Kronrod)
    0.6794095682990244,  # 10: ± (from GK21)
    0.7321483889893050,  # 11: ± (Kronrod)
    0.7808177265864169,  # 12: ± (from GK21)
    0.8251983149831142,  # 13: ± (Kronrod)
    0.8650633666889845,  # 14: ± (from GK21)
    0.9001486957483283,  # 15: ± (Kronrod)
    0.9301574913557082,  # 16: ± (from GK21)
    0.9548079348142663,  # 17: ± (Kronrod)
    0.9739065285171717,  # 18: ± (from GK21)
    0.9874334029080889,  # 19: ± (Kronrod)
    0.9956571630258081,  # 20: ± (from GK21)
    0.9993333609019321,  # 21: ± (Kronrod)
)

comptime gk43_kronrod_weights: StaticTuple[Float64, 22] = StaticTuple[
    Float64, 22
](
    0.0747221475174030,  # 0: center
    0.0745077510141751,  # 1: ±0.0746506174613833
    0.0738701996323940,  # 2: ±0.1488743389816312 (from GK21)
    0.0728244414718332,  # 3: ±0.2222549197766013
    0.0713872672686934,  # 4: ±0.2943928627014602 (from GK21)
    0.0695661979123565,  # 5: ±0.3649016613465808
    0.0673554146094781,  # 6: ±0.4333953941292472 (from GK21)
    0.0647464049514459,  # 7: ±0.4994795740710565
    0.0617449952014426,  # 8: ±0.5627571346686047 (from GK21)
    0.0583793955426193,  # 9: ±0.6228479705377252
    0.0546949020582554,  # 10: ±0.6794095682990244 (from GK21)
    0.0507419396001846,  # 11: ±0.7321483889893050
    0.0465608269104288,  # 12: ±0.7808177265864169 (from GK21)
    0.0421631379351918,  # 13: ±0.8251983149831142
    0.0375228761208695,  # 14: ±0.8650633666889845 (from GK21)
    0.0325974639753457,  # 15: ±0.9001486957483283
    0.0273718905932488,  # 16: ±0.9301574913557082 (from GK21)
    0.0218953638677954,  # 17: ±0.9548079348142663
    0.0162967342896666,  # 18: ±0.9739065285171717 (from GK21)
    0.0107986895858917,  # 19: ±0.9874334029080889
    0.0057685560597698,  # 20: ±0.9956571630258081 (from GK21)
    0.0018444776402124,  # 21: ±0.9993333609019321
)

comptime gk43_gauss21_weights: StaticTuple[Float64, 11] = StaticTuple[
    Float64, 11
](
    0.1494455540029169,  # 0: center (0.0)
    0.1477391049013385,  # 2: ±0.1488743389816312
    0.1427759385770601,  # 4: ±0.2943928627014602
    0.1347092173114733,  # 6: ±0.4333953941292472
    0.1234919762620659,  # 8: ±0.5627571346686047
    0.1093871588022976,  # 10: ±0.6794095682990244
    0.0931254545836976,  # 12: ±0.7808177265864169
    0.0750396748109200,  # 14: ±0.8650633666889845
    0.0547558965743520,  # 16: ±0.9301574913557082
    0.0325816230796473,  # 18: ±0.9739065285171717
    0.0116946388673187,  # 20: ±0.9956571630258081
)

# -----------------------------------------------------------------------------
# GK87: 87-point Gauss-Kronrod (43-point GK + 44 Kronrod extensions)
# -----------------------------------------------------------------------------
comptime gk87_nodes: StaticTuple[Float64, 44] = StaticTuple[Float64, 44](
    0.0000000000000000,  # 0: center
    0.0373521233946199,  # 1: ± (Kronrod)
    0.0746506174613833,  # 2: ± (from GK43)
    0.1118422131799075,  # 3: ± (Kronrod)
    0.1488743389816312,  # 4: ± (from GK43)
    0.1856953965683467,  # 5: ± (Kronrod)
    0.2222549197766013,  # 6: ± (from GK43)
    0.2585035592021616,  # 7: ± (Kronrod)
    0.2943928627014602,  # 8: ± (from GK43)
    0.3298748771061883,  # 9: ± (Kronrod)
    0.3649016613465808,  # 10: ± (from GK43)
    0.3994248478592188,  # 11: ± (Kronrod)
    0.4333953941292472,  # 12: ± (from GK43)
    0.4667636230420228,  # 13: ± (Kronrod)
    0.4994795740710565,  # 14: ± (from GK43)
    0.5314936059708319,  # 15: ± (Kronrod)
    0.5627571346686047,  # 16: ± (from GK43)
    0.5932233740579611,  # 17: ± (Kronrod)
    0.6228479705377252,  # 18: ± (from GK43)
    0.6515894665011779,  # 19: ± (Kronrod)
    0.6794095682990244,  # 20: ± (from GK43)
    0.7062732097873218,  # 21: ± (Kronrod)
    0.7321483889893050,  # 22: ± (from GK43)
    0.7570057306854956,  # 23: ± (Kronrod)
    0.7808177265864169,  # 24: ± (from GK43)
    0.8035576580352310,  # 25: ± (Kronrod)
    0.8251983149831142,  # 26: ± (from GK43)
    0.8457107484624157,  # 27: ± (Kronrod)
    0.8650633666889845,  # 28: ± (from GK43)
    0.8832216577713165,  # 29: ± (Kronrod)
    0.9001486957483283,  # 30: ± (from GK43)
    0.9158064146855072,  # 31: ± (Kronrod)
    0.9301574913557082,  # 32: ± (from GK43)
    0.9431676131336706,  # 33: ± (Kronrod)
    0.9548079348142663,  # 34: ± (from GK43)
    0.9650576238583846,  # 35: ± (Kronrod)
    0.9739065285171717,  # 36: ± (from GK43)
    0.9813581635727128,  # 37: ± (Kronrod)
    0.9874334029080889,  # 38: ± (from GK43)
    0.9921754978606872,  # 39: ± (Kronrod)
    0.9956571630258081,  # 40: ± (from GK43)
    0.9979898959866787,  # 41: ± (Kronrod)
    0.9993333609019321,  # 42: ± (from GK43)
    0.9999029772627292,  # 43: ± (Kronrod)
)

comptime gk87_kronrod_weights: StaticTuple[Float64, 44] = StaticTuple[
    Float64, 44
](
    0.0373610737626790,  # 0: center
    0.0373342287519350,  # 1: ±0.0373521233946199
    0.0372538755030477,  # 2: ±0.0746506174613833 (from GK43)
    0.0371205492698326,  # 3: ±0.1118422131799075
    0.0369350998204279,  # 4: ±0.1488743389816312 (from GK43)
    0.0366986044984561,  # 5: ±0.1856953965683467
    0.0364122207313518,  # 6: ±0.2222549197766013 (from GK43)
    0.0360769896228887,  # 7: ±0.2585035592021616
    0.0356936336394188,  # 8: ±0.2943928627014602 (from GK43)
    0.0352624126701567,  # 9: ±0.3298748771061883
    0.0347830989503651,  # 10: ±0.3649016613465808 (from GK43)
    0.0342550997042261,  # 11: ±0.3994248478592188
    0.0336777073116379,  # 12: ±0.4333953941292472 (from GK43)
    0.0330504134199785,  # 13: ±0.4667636230420228
    0.0323732024672028,  # 14: ±0.4994795740710565 (from GK43)
    0.0316467513714399,  # 15: ±0.5314936059708319
    0.0308724976117134,  # 16: ±0.5627571346686047 (from GK43)
    0.0300525811280927,  # 17: ±0.5932233740579611
    0.0291896977564758,  # 18: ±0.6228479705377252 (from GK43)
    0.0282869107887712,  # 19: ±0.6515894665011779
    0.0273474510500523,  # 20: ±0.6794095682990244 (from GK43)
    0.0263745054148392,  # 21: ±0.7062732097873218
    0.0253709697692538,  # 22: ±0.7321483889893050 (from GK43)
    0.0243391471260008,  # 23: ±0.7570057306854956
    0.0232804135028883,  # 24: ±0.7808177265864169 (from GK43)
    0.0221949359610123,  # 25: ±0.8035576580352310
    0.0210815688892038,  # 26: ±0.8251983149831142 (from GK43)
    0.0199380377864409,  # 27: ±0.8457107484624157
    0.0187614382015628,  # 28: ±0.8650633666889845 (from GK43)
    0.0175489679862432,  # 29: ±0.8832216577713165
    0.0162987316967873,  # 30: ±0.9001486957483283 (from GK43)
    0.0150104473463890,  # 31: ±0.9158064146855072
    0.0136859460227127,  # 32: ±0.9301574913557082 (from GK43)
    0.0123294476522449,  # 33: ±0.9431676131336706
    0.0109476796011189,  # 34: ±0.9548079348142663 (from GK43)
    0.0095499576722016,  # 35: ±0.9650576238583846
    0.0081483773841492,  # 36: ±0.9739065285171717 (from GK43)
    0.0067582900518474,  # 37: ±0.9813581635727128
    0.0053992802193005,  # 38: ±0.9874334029080889 (from GK43)
    0.0040968692827592,  # 39: ±0.9921754978606872
    0.0028848724302115,  # 40: ±0.9956571630258081 (from GK43)
    0.0018071241550579,  # 41: ±0.9979898959866787
    0.0009152833452022,  # 42: ±0.9993333609019321 (from GK43)
    0.0002741455637621,  # 43: ±0.9999029772627292
)

comptime gk87_gauss43_weights: StaticTuple[Float64, 22] = StaticTuple[
    Float64, 22
](
    0.0747221475174030,  # 0: center (0.0)
    0.0745077510141751,  # 2: ±0.0746506174613833
    0.0738701996323940,  # 4: ±0.1488743389816312
    0.0728244414718332,  # 6: ±0.2222549197766013
    0.0713872672686934,  # 8: ±0.2943928627014602
    0.0695661979123565,  # 10: ±0.3649016613465808
    0.0673554146094781,  # 12: ±0.4333953941292472
    0.0647464049514459,  # 14: ±0.4994795740710565
    0.0617449952014426,  # 16: ±0.5627571346686047
    0.0583793955426193,  # 18: ±0.6228479705377252
    0.0546949020582554,  # 20: ±0.6794095682990244
    0.0507419396001846,  # 22: ±0.7321483889893050
    0.0465608269104288,  # 24: ±0.7808177265864169
    0.0421631379351918,  # 26: ±0.8251983149831142
    0.0375228761208695,  # 28: ±0.8650633666889845
    0.0325974639753457,  # 30: ±0.9001486957483283
    0.0273718905932488,  # 32: ±0.9301574913557082
    0.0218953638677954,  # 34: ±0.9548079348142663
    0.0162967342896666,  # 36: ±0.9739065285171717
    0.0107986895858917,  # 38: ±0.9874334029080889
    0.0057685560597698,  # 40: ±0.9956571630258081
    0.0018444776402124,  # 42: ±0.9993333609019321
)
