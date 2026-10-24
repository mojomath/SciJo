# ===----------------------------------------------------------------------=== #
# SciJo: Types module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Types Module (`scijo.types`)
===============================
A `Numeric` trait and the numeric types that conform to it,
so a single SciJo kernel can be called with a Primal value, a Complex value, or with a value
that also carries its own derivative.


Available Types
---------------
- `Numeric`  - the trait: `+`, `-`, `*`, `/`, `exp`, `log`, `sqrt`, `sin`,
  `cos`, and the `one()`/`constant()` factories every conformer needs. Every
  operator also accepts a bare `Float64` on either side, so a kernel can
  write `2.0 * x + 1.0` instead of `T.constant(2.0) * x + T.constant(1.0)`.
- `Primal`   - a thin, zero-cost wrapper over `SIMD[dtype, width]`.
- `Complex`  - `re + im*i`, with the usual complex arithmetic and
  transcendentals (principal branch for `log`/`sqrt`).
- `Dual`     - forward-mode automatic differentiation: `(value, derivative)`.
  `Dual[Complex[f64]]` differentiates complex-valued kernels the same way.
- `variable`, `variable2` - free functions that build a `Dual`/nested `Dual`
  seeded at a point, without having to name the nested generic type.

Examples
--------
    ```mojo
    from scijo.types import Numeric, Primal, variable

    def f[T: Numeric](x: T) -> T:
        return x * x + 2.0 * x

    def main() raises:
        print(f(Primal[f64](3.0)).v)         # 15.0
        print(f(variable(3.0)).deriv.v)      # f'(3) = 2*3 + 2 = 8.0
    ```
"""

# ===----------------------------------------------------------------------=== #
# SciJo
# ===----------------------------------------------------------------------=== #
from .complex import Complex
from .dual import Dual, variable, variable2
from .numeric import Numeric
from .primal import Primal
