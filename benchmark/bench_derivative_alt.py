import gc
import time

from scipy.differentiate import derivative  # or appropriate import

# Testing only order = 8 derivative on a polynomial with 10 coefficients
coeffs = [i + 1 for i in range(10)]


def f(x):
    return sum(c * x**i for i, c in enumerate(coeffs))


x0 = 1.0

for i in range(10):
    _ = derivative(f, x0, order=8)

gc.disable()
t0 = time.perf_counter()
N = 10000
for i in range(N):
    _ = derivative(f, x0, order=8)
t1 = time.perf_counter()
gc.enable()
total = t1 - t0
print("Avg time per call: ", total / N)
# this result is to be compared with mojo's `benchmark.run[numeric_derivative_1point[ord2]]()` since we are differentiating at only one point.
