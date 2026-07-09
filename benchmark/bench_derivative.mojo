from scijo.differentiate.derivative import derivative
import scijo as sj
from utils import StaticTuple
import benchmark
from benchmark.compiler import keep
from numojo.prelude import *
from algorithm.functional import vectorize
from sys.info import simd_width_of
from python import Python, PythonObject
from math import sin

comptime dtype: DType = DType.float64
comptime SIZE: Int = 50
comptime coeffs: List[Scalar[dtype]] = [
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0,
    11.0,
    12.0,
    13.0,
    14.0,
    15.0,
    16.0,
    17.0,
    18.0,
    19.0,
    20.0,
    21.0,
    22.0,
    23.0,
    24.0,
    25.0,
    26.0,
    27.0,
    28.0,
    29.0,
    30.0,
    31.0,
    32.0,
    33.0,
    34.0,
    35.0,
    36.0,
    37.0,
    38.0,
    39.0,
    40.0,
    41.0,
    42.0,
    43.0,
    44.0,
    45.0,
    46.0,
    47.0,
    48.0,
    49.0,
    50.0,
]
comptime simd_width: Int = simd_width_of[dtype]()


def poly_func[
    dtype: DType
](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[dtype]:
    var result: Scalar[dtype] = 0.0
    var coeff = materialize[coeffs]()

    var coeff_ptr = coeff.unsafe_ptr()

    # test only for 10 coeffs.
    @parameter
    def closure[wid: Int](i: Int) unified {mut result, read coeff_ptr, read x}:
        result += SIMD[dtype, wid](
            coeff_ptr.load[width=wid](i)
        ).reduce_add() * (x**i)

    vectorize[simd_width](SIZE, closure)

    # the speed is not so different with native loop.
    # for i in range(SIZE):
    #     result = result + Scalar[dtype](coeff[i]) * x ** i
    return result


def sin_func[
    dtype: DType
](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) -> Scalar[
    dtype
] where dtype.is_floating_point():
    return sin(x)


comptime x0 = 1.0


def numeric_derivative_1point[order: Int]() raises:
    var result = derivative[dtype, poly_func, step_direction=0](
        x0=x0,
        args=None,
        order=order,
    )
    keep(result.df)


from scijo.differentiate.utility import DiffResult

comptime npoints: Int = 10


def numeric_derivative_npoints[order: Int]() raises:
    var xs: NDArray[dtype] = nm.linspace[dtype](-1.0, 1.0, npoints)

    @parameter
    for i in range(npoints):
        var result = derivative[dtype, poly_func, step_direction=0](
            x0=xs._buf.ptr[i],
            args=None,
            order=order,
        )
        keep(result.df)
        keep(result.error)
        keep(result.success)
        keep(result.x)
    # var xs: NDArray[dtype] = nm.linspace[dtype](-1.0, 1.0, 5)
    # var result: Scalar[dtype] = 0.0
    # for i in range(5):
    #     result = derivative[dtype, poly_func, step_direction=0](
    #         x0=xs._buf.ptr[i],
    #         args=None,
    #         order=order,
    #     ).df
    # keep(result)


# def numeric_derivative():
#     var np = Python.import_module("numpy")
#     var scipy = Python.import_module("scipy.differentiate")
#     x = 1.0
#     order = 8
#     # for i in range(10):
#     result = scipy.derivative(np.sin, x, order=order)

# def test_python_derivative():
#     var np = Python.import_module("numpy")
#     var scipy = Python.import_module("scipy.differentiate")
#     print(scipy.derivative(np.sin, 1.0, order=8))
#     benchmark.run[numeric_derivative]().print()

# def main():
#     test_python_derivative()

from time import perf_counter_ns


def main() raises:
    # testing only order 8 for now.
    comptime ord2 = 8
    var report2 = benchmark.run[numeric_derivative_1point[ord2]](2, 100)
    print("Derivative Benchmark - 1 Point Evaluation")
    print("order=: ", ord2)
    report2.print()

    var report3 = benchmark.run[numeric_derivative_npoints[ord2]](2, 100)
    print("Derivative Benchmark - N Points Evaluation")
    print("order=: ", ord2)
    report3.print()

    var xs: NDArray[dtype] = nm.linspace[dtype](-1.0, 1.0, npoints)
    var start = perf_counter_ns()
    # numeric_derivative_npoints[8]()
    var result = 0.0
    for i in range(npoints):
        result = derivative[dtype, sin_func, step_direction=0](
            # x0=1.0,
            x0=xs._buf.ptr[i],
            args=None,
            order=8,
        ).df

    var end = perf_counter_ns()
    var duration = end - start
    print("Time taken (ns): ", duration / 1e9)
    print("Computed derivative: ", result)
