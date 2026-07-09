import timeit

import numpy as np
from scipy.differentiate import derivative

# Use the same coefficients as the Mojo test: 1..20 (Mojo used 1.0..20.0)
coeffs = [i + 1 for i in range(10)]


# def poly_func(x):
#     return sum(c * x**i for i, c in enumerate(coeffs))


def sin_func(x):
    return np.sin(x)


def numeric_derivative(x, order=8):
    return derivative(sin_func, x, order=order)


def benchmark(func, points, repeat=5):
    def wrapper():
        return [func(x) for x in points]

    timer = timeit.Timer(wrapper)
    timings = timer.repeat(repeat, number=1)
    avg_time = sum(timings) / len(timings)
    outputs = wrapper()
    return avg_time, np.array(outputs)


if __name__ == "__main__":
    # print("At x0 = 1.0: ", numeric_derivative(1.0, order=4))

    # xs = np.linspace(-1.0, 1.0, 20)
    xs = [1.0]
    orders = [8]  # only one for comparision.
    repeat = 10

    # _ = poly_func(1.0)
    _ = sin_func(1.0)
    _ = numeric_derivative(1.0, order=6)

    for ord in orders:
        numeric_time, numeric_vals = benchmark(
            lambda x: numeric_derivative(x, order=ord), xs, repeat=repeat
        )
        print(f"  Numeric derivative (scipy), order={ord} : {numeric_time:.6f} s")

    # this result is to be compared with mojo's `benchmark.run[numeric_derivative_npoints[ord2]]()` since we are differentiating at many points.
