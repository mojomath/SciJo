from scijo.fft.fastfourier import fft, ifft
import scijo as sj
import numojo as nm
from python import Python, PythonObject
from testing import assert_almost_equal, assert_equal, assert_true
from testing import TestSuite
from numojo.core.complex import ComplexNDArray, ComplexSIMD
from numojo.core import CScalar
from numojo.core.ndarray import NDArray
from numojo.core.layout import NDArrayShape


fn compare_complex_arrays[
    dtype: nm.ComplexDType
](
    arr: ComplexNDArray[dtype],
    np_result: PythonObject,
    msg: String,
    atol: Float64 = 1e-10,
) raises:
    """Compare complex arrays element by element with NumPy results."""
    var np = Python.import_module("numpy")

    for i in range(arr.shape[0]):
        var mojo_real = Float64(arr._re[i])
        var numpy_real = Float64(py=np_result.real[i])
        var real_diff = abs(mojo_real - numpy_real)
        if real_diff > atol:
            raise Error(
                String(msg)
                + " - Real part mismatch at index "
                + String(i)
                + ": Mojo="
                + String(mojo_real)
                + ", NumPy="
                + String(numpy_real)
                + ", diff="
                + String(real_diff)
            )

    for i in range(arr.shape[0]):
        var mojo_imag = Float64(arr._im[i])
        var numpy_imag = Float64(py=np_result.imag[i])
        var imag_diff = abs(mojo_imag - numpy_imag)
        if imag_diff > atol:
            raise Error(
                String(msg)
                + " - Imaginary part mismatch at index "
                + String(i)
                + ": Mojo="
                + String(mojo_imag)
                + ", NumPy="
                + String(numpy_imag)
                + ", diff="
                + String(imag_diff)
            )
    print(msg + " - PASSED")


fn compare_real_arrays[
    dtype: DType
](
    arr: NDArray[dtype],
    np_result: PythonObject,
    msg: String,
    atol: Float64 = 1e-10,
) raises:
    """Compare real arrays element by element with NumPy results."""
    for i in range(arr.shape[0]):
        var mojo_val = Float64(arr[i])
        var numpy_val = Float64(py=np_result[i])
        var diff = abs(mojo_val - numpy_val)
        if diff > atol:
            raise Error(
                String(msg)
                + " - Value mismatch at index "
                + String(i)
                + ": Mojo="
                + String(mojo_val)
                + ", NumPy="
                + String(numpy_val)
                + ", diff="
                + String(diff)
            )
    print(msg + " - PASSED")


fn test_fft_basic() raises:
    """Test basic FFT with simple input."""
    var np = Python.import_module("numpy")

    # Test 1: Simple constant array [1, 1, 1, 1]
    var arr1 = ComplexNDArray[nm.cf64](NDArrayShape(4))
    for i in range(4):
        arr1[nm.Item(i)] = ComplexSIMD[nm.cf64](1.0, 0.0)

    var result1 = fft[nm.cf64](arr1)

    var np_arr1 = np.ones(4, dtype=np.complex64)
    var np_result1 = np.fft.fft(np_arr1)

    compare_complex_arrays[nm.cf64](
        result1, np_result1, "FFT test: constant array"
    )

    # Test 2: Simple impulse [1, 0, 0, 0, 0, 0, 0, 0]
    var arr2 = ComplexNDArray[nm.cf64](NDArrayShape(8))
    arr2[nm.Item(0)] = ComplexSIMD[nm.cf64](1.0, 0.0)
    for i in range(1, 8):
        arr2[nm.Item(i)] = ComplexSIMD[nm.cf64](0.0, 0.0)

    var result2 = fft[nm.cf64](arr2)

    var np_arr2 = np.zeros(8, dtype=np.complex64)
    np_arr2[0] = 1.0
    var np_result2 = np.fft.fft(np_arr2)

    compare_complex_arrays[nm.cf64](result2, np_result2, "FFT test: impulse")


fn test_fft_sequential() raises:
    """Test FFT with sequential input."""
    var np = Python.import_module("numpy")

    # Test with sequential values [0, 1, 2, 3]
    var arr = ComplexNDArray[nm.cf64](NDArrayShape(4))
    for i in range(4):
        arr[nm.Item(i)] = ComplexSIMD[nm.cf64](Float64(i), 0.0)

    var result = fft[nm.cf64](arr)

    var np_arr = np.arange(4, dtype=np.complex64)
    var np_result = np.fft.fft(np_arr)

    compare_complex_arrays[nm.cf64](
        result, np_result, "FFT test: sequential [0,1,2,3]"
    )


fn test_fft_complex_input() raises:
    """Test FFT with complex input."""
    var np = Python.import_module("numpy")

    var arr = ComplexNDArray[nm.cf64](NDArrayShape(4))
    arr[nm.Item(0)] = ComplexSIMD[nm.cf64](1.0, 0.0)
    arr[nm.Item(1)] = ComplexSIMD[nm.cf64](2.0, 1.0)
    arr[nm.Item(2)] = ComplexSIMD[nm.cf64](3.0, 2.0)
    arr[nm.Item(3)] = ComplexSIMD[nm.cf64](4.0, 3.0)

    var result = fft[nm.cf64](arr)

    var real = np.array(Python.tuple(1.0, 2.0, 3.0, 4.0), dtype=np.float64)
    var imag = np.array(Python.tuple(0.0, 1.0, 2.0, 3.0), dtype=np.float64)
    var np_arr = np.array(real, dtype=np.complex64)
    np_arr.imag = imag
    var np_result = np.fft.fft(np_arr)

    compare_complex_arrays[nm.cf64](
        result, np_result, "FFT test: complex input"
    )


# make it larger later and test performance.
fn test_fft_larger_size() raises:
    """Test FFT with larger size (16 elements)."""
    var np = Python.import_module("numpy")

    var arr = nm.zeros[nm.cf64](nm.Shape(16))
    for i in range(16):
        arr.store(i, val=CScalar[nm.cf64](Float64(i) / 4.0, 0.0))

    var np_arr = np.arange(16, dtype=np.complex64) / 4
    var np_result = np.fft.fft(np_arr)

    var result = fft[nm.cf64](arr)

    compare_complex_arrays[nm.cf64](
        result, np_result, "FFT test: 16 elements", atol=1e-5
    )


fn test_ifft_basic() raises:
    """Test inverse FFT."""
    var np = Python.import_module("numpy")

    var arr = ComplexNDArray[nm.cf64](NDArrayShape(8))
    for i in range(8):
        arr[nm.Item(i)] = ComplexSIMD[nm.cf64](Float64(i), Float64(i) * 0.5)

    var fft_result = fft[nm.cf64](arr)
    var ifft_result = ifft[nm.cf64](fft_result)

    var real = np.arange(8, dtype=np.float64)
    var imag = np.arange(8, dtype=np.float64) * 0.5
    var np_arr = np.array(real, dtype=np.complex64)
    np_arr.imag = imag
    var np_fft = np.fft.fft(np_arr)
    var np_ifft = np.fft.ifft(np_fft)

    compare_complex_arrays[nm.cf64](
        ifft_result, np_ifft, "IFFT test: roundtrip", atol=1e-5
    )


fn test_ifft_standalone() raises:
    """Test inverse FFT with known frequency domain input."""
    var np = Python.import_module("numpy")

    var arr = ComplexNDArray[nm.cf64](NDArrayShape(4))
    arr[nm.Item(0)] = ComplexSIMD[nm.cf64](10.0, 0.0)
    arr[nm.Item(1)] = ComplexSIMD[nm.cf64](-2.0, 2.0)
    arr[nm.Item(2)] = ComplexSIMD[nm.cf64](-2.0, 0.0)
    arr[nm.Item(3)] = ComplexSIMD[nm.cf64](-2.0, -2.0)
    var result = ifft[nm.cf64](arr)

    var real = np.array(Python.tuple(10.0, -2.0, -2.0, -2.0), dtype=np.float64)
    var imag = np.array(Python.tuple(0.0, 2.0, 0.0, -2.0), dtype=np.float64)
    var np_arr = np.array(real, dtype=np.complex64)
    np_arr.imag = imag
    var np_result = np.fft.ifft(np_arr)

    compare_complex_arrays[nm.cf64](
        result, np_result, "IFFT test: frequency domain", atol=1e-5
    )


fn test_edge_cases() raises:
    """Test edge cases and special conditions."""
    var np = Python.import_module("numpy")

    var arr1 = ComplexNDArray[nm.cf64](NDArrayShape(1))
    arr1[nm.Item(0)] = ComplexSIMD[nm.cf64](5.0, 3.0)
    var result1 = fft[nm.cf64](arr1)

    assert_almost_equal(
        Float64(result1._re[0]), 5.0, msg="Single element FFT real part"
    )
    assert_almost_equal(
        Float64(result1._im[0]), 3.0, msg="Single element FFT imag part"
    )

    var arr2 = ComplexNDArray[nm.cf64](NDArrayShape(2))
    arr2[nm.Item(0)] = ComplexSIMD[nm.cf64](1.0, 0.0)
    arr2[nm.Item(1)] = ComplexSIMD[nm.cf64](2.0, 0.0)
    var result2 = fft[nm.cf64](arr2)

    var np_arr2 = np.array(Python.tuple(1.0, 2.0), dtype=np.complex64)
    var np_result2 = np.fft.fft(np_arr2)

    compare_complex_arrays[nm.cf64](result2, np_result2, "FFT test: 2 elements")


fn test_error_conditions() raises:
    """Test error conditions."""
    var np = Python.import_module("numpy")

    var arr_bad = ComplexNDArray[nm.cf64](NDArrayShape(5))
    for i in range(5):
        arr_bad[nm.Item(i)] = ComplexSIMD[nm.cf64](Float64(i), 0.0)

    try:
        var _ = fft[nm.cf64](arr_bad)
    except:
        print("Non-power-of-2 error handling - PASSED")


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
