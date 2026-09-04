# SciJo User Manual

A working guide to the library: what the functions are, what they expect, and
why a few things are spelled the way they are.

This manual is written for someone who already knows SciPy and is meeting
Mojo's idioms for the first time. Most of SciJo can be guessed at from
`scipy.optimize`, `scipy.integrate` and friends - the module names, the
argument names and the algorithms are deliberately familiar. What is not
obvious from SciPy alone is how a callback is spelled in Mojo, which is a
recurring shape across every module here and is worth reading once, in
[Passing a function to SciJo](#passing-a-function-to-scijo), before the rest.

The per-symbol reference lives in the docstrings on every public function -
each one documents its `Parameters:`, `Args:`, `Returns:` and `Raises:` in
full. This manual is the prose half: the shape of the API and the gotchas,
not an enumeration of every signature.

- [SciJo User Manual](#scijo-user-manual)
  - [Getting started](#getting-started)
    - [Passing a function to SciJo](#passing-a-function-to-scijo)
  - [Differentiation (`scijo.differentiate`)](#differentiation-scijodifferentiate)
    - [`derivative`](#derivative)
    - [`jacobian` and `hessian`](#jacobian-and-hessian)
  - [Integration (`scijo.integrate`)](#integration-scijointegrate)
    - [`quad` - adaptive quadrature of a function](#quad--adaptive-quadrature-of-a-function)
    - [Fixed-sample rules for discrete data](#fixed-sample-rules-for-discrete-data)
    - [Cumulative integration](#cumulative-integration)
  - [Interpolation (`scijo.interpolate`)](#interpolation-scijointerpolate)
    - [`interp1d` and `LinearInterpolator`](#interp1d-and-linearinterpolator)
    - [`CubicSpline` and `Akima1DInterpolator`](#cubicspline-and-akima1dinterpolator)
    - [The functional `interp`](#the-functional-interp)
  - [FFT (`scijo.fft`)](#fft-scijofft)
    - [`fft` and `ifft`](#fft-and-ifft)
    - [`rfft` and `irfft`](#rfft-and-irfft)
    - [Frequency helpers](#frequency-helpers)
  - [Constants (`scijo.constants`)](#constants-scijoconstants)
    - [Named constants vs. the CODATA table](#named-constants-vs-the-codata-table)
    - [Temperature and optics helpers](#temperature-and-optics-helpers)
  - [Optimization (`scijo.optimize`)](#optimization-scijooptimize)
    - [`root_scalar` and the individual solvers](#root_scalar-and-the-individual-solvers)
    - [`minimize_scalar`](#minimize_scalar)
  - [Errors](#errors)
  - [Appendix: NuMojo is the array backend](#appendix-numojo-is-the-array-backend)
  - [Appendix: what is not here yet](#appendix-what-is-not-here-yet)

---

## Getting started

SciJo is built on [NuMojo](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo)
and targets Mojo `>=1.0.0,<2`. Inside a checkout, pixi builds and packages it:

```bash
pixi install
pixi run package     # mojo precompile scijo && cp scijo.mojoc tests/
```

A program compiles against the source tree, or the precompiled `.mojoc`,
with the package directory on the import path:

```bash
mojo run -I . my_program.mojo
```

One import gets you a module's public names:

```mojo
from scijo.differentiate import derivative
from scijo.integrate import quad, trapezoid
```

Everything that also needs a NuMojo array type or dtype alias - `NDArray`,
`f64`, `nm.linspace`, and so on - should additionally import the SciJo
prelude, which simply re-exports NuMojo's own prelude:

```mojo
from scijo.prelude import *    # f64, i32, NDArray, ... come from numojo.prelude
```

**`scijo`'s own top-level `__init__.mojo` does not export `f64` or any other
dtype alias itself** - it only re-exports `NumojoError`. A call like
`sj.f64` after a plain `import scijo as sj` does not compile; the alias has
to come from `scijo.prelude` (or directly from `numojo.prelude`). This
manual's examples therefore always pair a module import with
`from scijo.prelude import *` when they need `f64` or `NDArray`, exactly as
SciJo's own module docstrings do.

A first program:

```mojo
from scijo.differentiate import derivative
from scijo.prelude import *

def f[dtype: DType](
    x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
) capturing -> Scalar[dtype]:
    return x * x

def main() raises:
    var result = derivative[f64, f](1.0)
    print(result.df)      # 2.0
```

### Passing a function to SciJo

Every callback-based routine in SciJo - `derivative`, `jacobian`, `hessian`,
`quad`, `root_scalar` and its individual solvers, `minimize_scalar` - takes
the function to operate on as a **compile-time parameter**, not a runtime
argument, and expects one exact parameter shape:

```mojo
def[dtype: DType](
    x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
) capturing -> Scalar[dtype]
```

Read that as: a `def` (not `fn`) taking the evaluation point and an optional
list of extra parameters, both generic over the same `dtype`, returning a
scalar of that dtype, and `capturing` so it may close over outer variables.
`jacobian` and `hessian` take the vector-valued version of the same shape,
operating on `NDArray[dtype]` instead of `Scalar[dtype]`:

```mojo
def[dtype: DType](
    x: NDArray[dtype], args: Optional[List[Scalar[dtype]]]
) capturing raises -> NDArray[dtype]        # jacobian
def[dtype: DType](
    x: NDArray[dtype], args: Optional[List[Scalar[dtype]]]
) capturing raises -> Scalar[dtype]         # hessian
```

Two consequences follow directly from the shape:

**`args` is always present in the signature, even when you never use it.**
It exists so a function can be parameterised without a closure - pass extra
coefficients through `args=List[Scalar[dtype]](2.0, 3.0)` and read them back
with `args.value()[0]`, `args.value()[1]`, ... A function that ignores the
argument simply never touches it:

```mojo
def parabola[dtype: DType](
    x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
) capturing -> Scalar[dtype]:
    var a = args.value()[0]
    return a * x * x
```

**The function is a parameter, so the call site names it in brackets, next
to `dtype`.** This is what makes `derivative[f64, f](1.0)` read the way it
does - `f64` and `f` both go where NumPy would only ever see a runtime
argument:

```mojo
var d  = derivative[f64, parabola](x0=1.0, args=List[Scalar[f64]](2.0))
var r  = root_scalar[f64, my_f](bracket=(0.0, 2.0), method="bisect")
var m  = minimize_scalar[f64, my_f, method="Brent"](bracket=(0.0, 4.0))
```

Because the function is a compile-time parameter, SciJo can specialise the
whole call at compile time - there is no function-pointer indirection at
the call site - but it also means a function used with two different
`dtype`s needs to *be* generic over `dtype`, as every example in this manual
is. A concrete `fn(x: Float64) -> Float64` will not satisfy the parameter.

---

## Differentiation (`scijo.differentiate`)

`scijo.differentiate` computes derivatives of scalar and vector-valued
functions using finite differences - there is no symbolic or automatic
differentiation here, only numerical approximation with adaptive step
control.

### `derivative`

`derivative` computes the first derivative of a scalar function at a point,
using central, forward, or backward finite differences, refining the step
size until two successive estimates agree within tolerance (or `max_iter`
is reached).

```mojo
from scijo.differentiate import derivative
from scijo.prelude import *

def f[dtype: DType](
    x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
) capturing -> Scalar[dtype]:
    return x * x + 2.0 * x + 1.0

def main() raises:
    # Central differences (the default), order 8.
    var res = derivative[f64, f](x0=1.0)
    print(res.df, res.success, res.nit, res.nfev)

    # Forward differences, for a left boundary point.
    var res_fwd = derivative[f64, f, step_direction=1](x0=0.0, order=4)

    # Backward differences, for a right boundary point.
    var res_bwd = derivative[f64, f, step_direction=-1](x0=5.0, order=4)
```

`step_direction` is a keyword-only compile-time parameter: `0` for central
(the default), `1` for forward, `-1` for backward. It is not a runtime
argument, so it is written in the brackets alongside `dtype` and the
function, after a `*,` in the signature - `derivative[f64, f, step_direction=1](...)`.

The return value is a `DiffResult[dtype]`, with fields `success`, `df`
(the derivative), `error` (the last change between iterations), `nit`
(iterations used) and `nfev` (function evaluations used). `success=False`
means `max_iter` was reached without the tolerance being met - `df` still
holds the last estimate, which is often usable, but check `success` if the
distinction matters.

**`order` is restricted, and differently for each direction.** Central
differences accept `order ∈ {2, 4, 6, 8}`; forward and backward accept
`order ∈ {1, 2, 3, 4, 5, 6}`. Any other value raises `Error`, naming the
valid set. Higher orders converge faster (fewer iterations to reach a given
tolerance) but cost more function evaluations per iteration, since more
stencil points are needed.

`atol`, `rtol`, `max_iter`, `initial_step` and `step_factor` all have
defaults and are validated eagerly: a negative tolerance, a non-positive
`initial_step`, a `step_factor <= 1`, or a non-positive `max_iter` all raise
before any function evaluation happens.

### `jacobian` and `hessian`

`jacobian` computes the Jacobian of a vector-valued function using central
differences, one column per input dimension, evaluated in parallel:

```mojo
import numojo as nm
from scijo.differentiate import jacobian
from scijo.prelude import *

def f[dtype: DType](
    x: NDArray[dtype], args: Optional[List[Scalar[dtype]]]
) capturing raises -> NDArray[dtype]:
    return x * x

def main() raises:
    var x = nm.array[f64]([1.0, 2.0])
    var J = jacobian[f64, f](x)     # 2x2, diagonal ≈ [2.0, 4.0]
    print(J)
```

`jacobian` takes a single `step` argument (default `0.5`) rather than the
adaptive tolerance loop `derivative` uses - there is no Richardson
refinement here, so a step that is too large will show visibly in the
result on a curved function. `hessian` computes the full second-derivative
matrix of a scalar-valued function, using the standard central formula on
the diagonal and the mixed-partial four-point stencil off it:

```mojo
def g[dtype: DType](
    x: NDArray[dtype], args: Optional[List[Scalar[dtype]]]
) capturing raises -> Scalar[dtype]:
    return x.item(0) * x.item(0) + x.item(1) * x.item(1)

var x = nm.array[f64]([1.0, 2.0])
var H = hessian[f64, g](x)          # ≈ [[2, 0], [0, 2]]
```

`hessian`'s default `step` is `1e-5`, much smaller than `jacobian`'s `0.5`
- a second-derivative stencil divides by `step * step`, so the truncation
error scales differently and the right step size is not the same one.

Both functions raise if any column's (or entry's) evaluation raises inside
the callback; `jacobian` collects errors from its parallel workers and
re-raises the first one it finds, prefixed `"SciJo [jacobian]: "`.

---

## Integration (`scijo.integrate`)

`scijo.integrate` has two distinct kinds of routine: `quad`, which
integrates a *function* you supply (adaptive quadrature, like
`scipy.integrate.quad`), and the fixed-sample family - `trapezoid`,
`simpson`, `romb`, and their cumulative variants - which integrate
*discrete data* you already have, as `NDArray` samples (like
`scipy.integrate.trapezoid`).

### `quad` - adaptive quadrature of a function

```mojo
from scijo.integrate import quad, QAG_GK61
from scijo.prelude import *

def integrand[dtype: DType](
    x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
) capturing -> Scalar[dtype]:
    return x * x

def main() raises:
    var result = quad[f64, integrand](a=0.0, b=1.0)
    print(result.integral, result.abserr)        # ≈ 0.3333..., tiny

    # A higher-order Gauss-Kronrod rule under the adaptive "qag" method.
    var result_hi = quad[f64, integrand, method="qag", qag_rule=QAG_GK61](
        a=0.0, b=1.0, atol=1e-10, rtol=1e-10
    )
```

`method` (a compile-time, keyword-only parameter defaulting to `"qng"`)
selects the backend algorithm, all from MSL's QUADPACK-derived routines:

| `method` | What it does |
| --- | --- |
| `"qng"` | Non-adaptive Gauss-Kronrod-Patterson quadrature (10/21/43/87-point rules); the default. |
| `"qag"` | Adaptive Gauss-Kronrod, refining subintervals until `limit` is reached or tolerance is met. `qag_rule=` selects the rule: `QAG_GK15`, `QAG_GK21` (default), `QAG_GK31`, `QAG_GK41`, `QAG_GK51`, `QAG_GK61` - larger numbers mean more points per subinterval and faster convergence on smooth integrands. |
| `"qags"` | Adaptive integration plus Wynn epsilon extrapolation, for integrands with endpoint singularities or slow convergence. |

The return value is an `IntegralResult[dtype]` with fields `integral`,
`abserr`, `nfev` and `ier` (SciJo does not currently populate `nfev`/`ier`
from the backend - they read `0` on success). `limit <= 0` raises before any
integration is attempted, and an unrecognised `method` string raises naming
the three supported values.

### Fixed-sample rules for discrete data

`trapezoid`, `simpson` and `romb` all take an already-sampled `NDArray` and
integrate it - either against a uniform spacing `dx`, or against an
explicit array of sample points `x` (only `trapezoid` and `simpson` have
the `x`-taking overload; `romb` requires uniform spacing):

```mojo
import numojo as nm
from scijo.integrate import trapezoid, simpson, romb
from scijo.prelude import *

def main() raises:
    var y = nm.linspace[f64](0.0, 10.0, 100) ** 2

    var area_trap = trapezoid(y, dx=0.1)
    var area_sim  = simpson(y, dx=0.1)

    # romb requires 2^k + 1 points for some integer k >= 1.
    var y_romb = nm.linspace[f64](0.0, 1.0, 9) ** 2     # 9 = 2^3 + 1
    var area_rom = romb(y_romb, dx=0.125)

    # Non-uniform spacing: pass x explicitly instead of dx.
    var x = nm.linspace[f64](0.0, 10.0, 100)
    var area_trap_x = trapezoid(y, x)
    var area_sim_x  = simpson(y, x)
```

All three currently require **1-D input** - every routine raises (via
`NumojoError`) if `y.ndim != 1`, there is no `axis` argument that does
anything yet despite the parameter existing in the signature for
SciPy-compatibility. `trapezoid` and `simpson` on an empty or single-point
array return `0.0` rather than raising (for `trapezoid`); `romb` raises if
`y.size` is not of the form `2^k + 1`, naming the size it received.

`simpson` on an **even-length** array (an odd number of intervals) applies
Simpson's rule to as many interior pairs as it can and falls back to a
trapezoidal estimate on the final leftover interval - this matches
`scipy.integrate.simpson`'s behaviour for even-length input, but it means
the accuracy on that last panel drops to `O(h^2)` from `O(h^4)`.

### Cumulative integration

`cumulative_trapezoid` and `cumulative_simpson` return the running integral
as an array rather than a single number, matching
`scipy.integrate.cumulative_trapezoid` / `cumulative_simpson`:

```mojo
import numojo as nm
from scijo.integrate import cumulative_trapezoid, cumulative_simpson
from scijo.prelude import *

def main() raises:
    var y = nm.array[f64]([1.0, 2.0, 3.0, 4.0])

    var cum = cumulative_trapezoid(y, dx=1.0, initial=0.0)
    print(cum)     # [0.0, 1.5, 4.0, 7.5]

    # Without `initial`, the result is one element shorter than `y`.
    var cum_short = cumulative_trapezoid(y, dx=1.0)
    print(cum_short)  # [1.5, 4.0, 7.5]
```

**The output length depends on whether `initial` is given.** With
`initial=None` (the default) the result has `n - 1` elements - there is no
"integral up to the first point" to report. Passing `initial=0.0` (or any
other starting value) prepends it, so the result has the same length `n` as
the input and lines up index-for-index with `y`. Both cumulative routines
also accept an `x` array in place of `dx` for non-uniform spacing, the same
way `trapezoid` and `simpson` do.

---

## Interpolation (`scijo.interpolate`)

`scijo.interpolate` provides 1-D interpolation only - there is no
multi-dimensional interpolation yet. Three methods are available: linear,
natural cubic spline, and Akima; each comes as both a reusable callable
object and, for linear, a one-shot functional form.

### `interp1d` and `LinearInterpolator`

```mojo
import numojo as nm
from scijo.interpolate import interp1d
from scijo.prelude import *

def main() raises:
    var x = nm.arange[f64](0.0, 1.0, 0.5)          # [0.0, 0.5, 1.0]
    var y = nm.array[f64]([0.0, 0.25, 1.0])

    var li = interp1d(x, y, bounds_error=False, fill_value=0.0)
    print(li(Scalar[f64](0.25)))                    # a single point
    print(li(nm.array[f64]([0.1, 0.5, 0.9])))        # an array of points
```

`interp1d(x, y, ...)` is a convenience constructor for `LinearInterpolator`
- both take the same arguments and `interp1d` just forwards to
`LinearInterpolator[dtype](x, y, bounds_error, fill_value)`. Once built, the
object is callable on either a `Scalar[dtype]` or an `NDArray[dtype]`;
`x` must be strictly increasing and have the same length as `y`, checked at
construction time.

**Out-of-bounds behaviour is controlled by `bounds_error` and
`fill_value`, and the two interact in a specific order:**

| `bounds_error` | `fill_value` | Behaviour outside `[x.min(), x.max()]` |
| --- | --- | --- |
| `True` (default) | (ignored) | Raises `Error` naming the point and the valid range. |
| `False` | a value | Returns that constant value. |
| `False` | `None` (default) | Clamps to the nearest boundary `y` value - **not** linear extrapolation. |

This differs from plain `scipy.interpolate.interp1d`, whose
`fill_value=None` extrapolates by default; here, `fill_value=None` with
`bounds_error=False` clamps. Ask for extrapolation explicitly through the
functional `interp` (below) if you want it.

### `CubicSpline` and `Akima1DInterpolator`

```mojo
import numojo as nm
from scijo.interpolate import CubicSpline, Akima1DInterpolator
from scijo.prelude import *

def main() raises:
    var x = nm.linspace[f64](0.0, 10.0, 11)
    var y = x * x

    var cs = CubicSpline(x, y)
    print(cs(nm.linspace[f64](0.5, 9.5, 5)))

    var ak = Akima1DInterpolator(x, y)
    print(ak(Scalar[f64](3.7)))
```

`CubicSpline` builds a natural cubic spline - second derivatives at both
endpoints are pinned to zero - matching `scipy.interpolate.CubicSpline`
with `bc_type="natural"`. It is the only boundary condition SciJo currently
supports: `CubicSpline[bc_type="not-a-knot"](...)` compiles (the parameter
exists for SciPy-compatibility) but raises at construction, since only
`"natural"` has an implementation. `Akima1DInterpolator` builds the
Akima piecewise-cubic interpolant, which resists the oscillation near
outliers that a global cubic spline can show; with fewer than 5 points it
falls back to `CubicSpline` internally rather than raising.

Both objects clamp to the boundary `y` value outside `[x.min(), x.max()]`
rather than raising or extrapolating - there is no `bounds_error` option on
these two types, unlike `LinearInterpolator`.

### The functional `interp`

`interp` is the one-shot equivalent of building an interpolator object and
calling it once, similar to `numpy.interp` but generalised to more than
linear:

```mojo
import numojo as nm
from scijo.interpolate import interp
from scijo.prelude import *

def main() raises:
    var x = nm.arange[f64](0.0, 1.0, 0.5)
    var y = x * x
    var xq = nm.array[f64]([0.1, 0.5, 0.9])

    var yq = interp[f64, type="linear", fill_method="interpolate"](xq, x, y)
    var yq_extrap = interp[f64, type="linear", fill_method="extrapolate"](xq, x, y)
    var yq_cubic = interp[f64, type="cubic"](xq, x, y)
    var yq_akima = interp[f64, type="akima"](xq, x, y)
```

`type` and `fill_method` are both compile-time, keyword-only parameters.
**`fill_method="extrapolate"` is only implemented for `type="linear"`** -
combining it with `"cubic"` or `"akima"` raises `Error` at call time naming
the unsupported combination; those two methods only support
`fill_method="interpolate"` (boundary clamping). Reach for a
`LinearInterpolator` directly, rather than `interp`, when you need to
evaluate the same data repeatedly - building the interpolator once and
calling it many times avoids repeating the input validation on every call.

---

## FFT (`scijo.fft`)

`scijo.fft` implements the Cooley-Tukey radix-2 decimation-in-time FFT.
**Every transform in this module requires a 1-D input whose length is a
power of two** (`rfft` and `irfft` zero-pad to the next power of two
automatically; `fft` and `ifft` do not - they raise if the length is not
already a power of two). There is no 2-D FFT yet; see
[Appendix: what is not here yet](#appendix-what-is-not-here-yet).

### `fft` and `ifft`

```mojo
import numojo as nm
from scijo.fft import fft, ifft
from scijo.prelude import *

def main() raises:
    var arr = nm.arange[cf64](CScalar[cf64](0), CScalar[cf64](8))

    var freq = fft(arr)      # forward transform
    var time = ifft(freq)    # inverse - recovers arr, up to floating-point error
```

`fft` and `ifft` both operate on `ComplexNDArray[cdtype]`, parameterised on
a `ComplexDType` (default `ComplexDType.float64`) rather than a plain
`DType` - a real-valued signal has to be lifted into a complex array with a
zero imaginary part before calling `fft` directly, which is what `rfft`
does for you. `ifft` applies the `1/N` normalization internally, so
`ifft(fft(x))` recovers `x` without any manual rescaling.

**Length must already be a power of 2** for both - `fft`/`ifft` on a
5-element array raises `Error`, it does not round up for you. `n <= 1` is
accepted trivially and returns a copy of the input unchanged (there is
nothing to transform).

### `rfft` and `irfft`

`rfft` computes the FFT of a real-valued array and returns only the
non-redundant half of the spectrum - `N // 2 + 1` bins - since the rest is
the complex-conjugate mirror of what is returned, exactly like
`scipy.fft.rfft` / `numpy.fft.rfft`:

```mojo
import numojo as nm
from scijo.fft import rfft, irfft
from scijo.prelude import *

def main() raises:
    var x = nm.linspace[f64](0.0, 1.0, 6)   # length 6, not a power of 2
    var freqs = rfft(x)                      # zero-padded to 8, returns 5 bins
    var x_rec = irfft(freqs, 8)              # reconstructs the padded 8-length signal
```

**Unlike `fft`, `rfft` accepts any length and zero-pads to the next power
of two itself** - matching NumPy's zero-padding behaviour, but meaning the
output corresponds to the *padded* length, not the original one, so
`x_rec` above has 8 samples, not 6. `irfft`'s `n` argument names the
desired output length; if omitted it defaults to `2 * (len(arr) - 1)`, and
either way the result is rounded up to a power of two if it is not one
already.

### Frequency helpers

```mojo
from scijo.fft import fftfreq, rfftfreq, fftshift, ifftshift, next_fast_len

var freqs = fftfreq(8, d=0.1)      # bin centres for an 8-point fft, spacing 0.1
var rfreqs = rfftfreq(8, d=0.1)    # bin centres for an 8-point rfft (non-negative only)
var shifted = fftshift(freqs)      # zero-frequency moved to the centre
var restored = ifftshift(shifted)  # equals freqs again

print(next_fast_len(100))          # 128 - smallest power of 2 >= 100
```

`next_fast_len` reports the smallest power of two `>= n`, which for this
implementation *is* the fast length - unlike SciPy's version, which also
accepts other highly-composite sizes, SciJo's Cooley-Tukey implementation
only ever runs efficiently at a power of two, so that is the only answer
`next_fast_len` gives. `fftshift` and `ifftshift` are each other's inverse
and both require 1-D input.

---

## Constants (`scijo.constants`)

`scijo.constants` mirrors `scipy.constants`: a set of named module-level
constants for the ones you reach for constantly (`pi`, `c`, `h`, ...), plus
a CODATA 2022 lookup table reachable by string key for everything else.

### Named constants vs. the CODATA table

```mojo
from scijo.constants import pi, c, h, hbar, golden, N_A, k
from scijo.constants import value, unit, precision, find

def main() raises:
    print(pi)                                    # 3.141592653589793
    print(c)                                      # 299792458.0 (speed of light)

    # The same constants, and hundreds more, via the CODATA table:
    print(value("speed_of_light_in_vacuum"))      # 299792458.0
    print(unit("Planck_constant"))                # "J Hz^-1"
    print(precision("Planck_constant"))            # relative uncertainty

    var names = find("electron")                  # every key containing "electron"
```

The named constants (`pi`, `c`, `h`, `hbar`, `G`, `g`, `e`, `R`, `alpha`,
`N_A`, `k`, `sigma`, `Wien`, `Rydberg`, `golden`, and more) are `comptime`
aliases in `scijo.constants.constants` - plain compile-time values, cheap
to use, with both a short SciPy-style name (`h`) and a long descriptive one
(`Planck`) for most of them. `value`, `unit`, `precision`, `find`,
`get_constant_tuple` and `list_all_constants` instead go through
`physical_constants`, a `Dict[String, PhysicalConstant[f64]]` keyed by the
CODATA name (`"speed_of_light_in_vacuum"`, not `"c"`) - the two spellings
are not interchangeable, and a lookup by the short name in the table raises
`Error: Unknown physical constant`.

`precision(key)` returns `uncertainty / value`, matching
`scipy.constants.precision`; it returns `0.0` rather than dividing by zero
for the handful of constants (like `c` itself) that are exact by
definition and so carry zero uncertainty.

### Temperature and optics helpers

```mojo
from scijo.constants import convert_temperature, lambdanu, nulambda

def main() raises:
    var k_temp = convert_temperature["Celsius", "Kelvin"](0.0)       # 273.15
    var c_temp = convert_temperature["Fahrenheit", "Celsius"](32.0)  # 0.0

    var wavelength = lambdanu(5e14)     # c / frequency
    var frequency  = nulambda(500e-9)   # c / wavelength
```

`convert_temperature` takes the two scales as compile-time string
parameters (`"Celsius"`, `"Fahrenheit"`, `"Kelvin"`) rather than runtime
arguments - an unsupported scale name raises `Error` listing the three
valid ones. `lambdanu` and `nulambda` are literally the same formula,
`c / x`, offered under both names because the physics reads either
direction depending on which quantity you have in hand.

---

## Optimization (`scijo.optimize`)

`scijo.optimize` covers scalar root-finding and scalar minimization -
there is no multi-dimensional optimization yet (see
[Appendix: what is not here yet](#appendix-what-is-not-here-yet)). Every
solver follows the callback shape from
[Passing a function to SciJo](#passing-a-function-to-scijo).

### `root_scalar` and the individual solvers

```mojo
from scijo.optimize import root_scalar
from scijo.prelude import *

def f[dtype: DType](
    x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
) capturing -> Scalar[dtype]:
    return x * x - 2.0

def fprime[dtype: DType](
    x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
) capturing -> Scalar[dtype]:
    return 2.0 * x

def main() raises:
    var r_bisect = root_scalar[f64, f, method="bisect"](bracket=(0.0, 2.0))
    var r_brent  = root_scalar[f64, f, method="brent"](bracket=(0.0, 2.0))
    var r_secant = root_scalar[f64, f, method="secant"](x0=1.0, x1=2.0)

    # Newton needs a derivative, so it is a different overload of root_scalar,
    # naming fprime as a second function parameter:
    var r_newton = root_scalar[f64, f, fprime, method="newton"](x0=1.0)

    print(r_bisect.root, r_bisect.success, r_bisect.method)
```

**`method` is a compile-time, keyword-only parameter, not a runtime
argument** - like `step_direction` on `derivative`, it is written in the
brackets next to `dtype` and the function(s), after the `*,` in the
signature, not among the parentheses. `root_scalar[dtype, f]` (one function
parameter) supports `method` values `"bisect"`, `"brent"` and `"secant"`;
asking it for `"newton"` raises, directing you to the two-function-parameter
overload `root_scalar[dtype, f, fprime]`, which additionally supports
`"newton"` (alongside the same three). **Which runtime arguments are
required depends on the method** - `bracket=(a, b)` for bisect/brent, `x0=`
and `x1=` for secant, `x0=` alone for newton - and passing the wrong ones
raises `Error` naming what was missing, rather than silently picking a
default. `bisect`, `brent`, `secant` and `newton` are also directly
importable and callable without going through `root_scalar` at all, if you
already know which method you want.

Every solver returns a `RootResult[dtype]`: `root`, `nit`, `nfev`,
`success`, `message` (a short human-readable status, e.g. `"converged"` or
`"maximum iterations exceeded"`) and `method` (the name of the algorithm
that ran).

### `minimize_scalar`

```mojo
from scijo.optimize import minimize_scalar
from scijo.prelude import *

def objective[dtype: DType](
    x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]
) capturing -> Scalar[dtype]:
    return (x - 2.0) * (x - 2.0) + 1.0

def main() raises:
    var result = minimize_scalar[f64, objective, method="Brent"](
        bracket=(0.0, 4.0), atol=1e-8, maxiter=100
    )
    print(result)          # multi-line report via write_to
    print(result.x, result.fun, result.success)
```

`method` (default `"Brent"`) accepts `"Brent"`, `"Golden"` or `"Bounded"`,
case-insensitively (`"brent"` and `"Brent"` both work). Brent and Golden
take a `bracket=(a, b)` **or** a `bounds=(a, b)` - either is accepted and
used the same way, since both methods bracket the minimum themselves -
while Bounded requires `bounds=` specifically and searches strictly within
`[a, b]`, never evaluating outside it. Omitting the argument the chosen
method needs raises `Error` naming it. The result is an
`OptimizeResult[dtype]` with `x`, `fun` (the value at `x`), `success`,
`message`, `nit` and `nfev`; printing it directly (`print(result)`) runs
its `write_to`, producing a multi-line report rather than the compact
`__str__` form.

---

## Errors

SciJo functions are `raises` almost everywhere, since the checks that
matter - array shape, dtype floating-point-ness, bracket validity, method
names - are runtime facts. Two error-construction styles appear across the
codebase:

A plain `Error` with a descriptive message, often multi-line and including
the value that was rejected:

```console
Error: SciJo Derivative (Central): Invalid accuracy order specified.
  Expected: order ∈ {2, 4, 6, 8}
  Got: order = 3
  ...
```

And, in modules that integrate more closely with NuMojo's own array
checking (`trapezoid`, `simpson`, `romb`), an `Error` built from
`NumojoError`, which carries a `category`, a `message` and a `location`:

```mojo
raise Error(
    NumojoError(
        category="shape",
        message="Expected y to be 1-D, received ndim=2.",
        location="trapezoid(y, dx=1.0)",
    )
)
```

Both print as an ordinary Mojo `Error` when caught with `except e:` - there
are no distinct exception types to match on, only the message text. Common
raise conditions across the library: an unsupported `order` (differentiate),
a non-power-of-2 length (`fft`/`ifft`), wrong-dimensional or mismatched-size
arrays (integrate, interpolate), a missing method-specific argument
(optimize), and an unrecognised `method`/`type` string (`quad`, `interp`,
`root_scalar`, `minimize_scalar`). Validate the obviously-checkable things
- array shapes, bracket ordering, whether you actually have the arguments a
given `method` needs - before the call if you want to avoid the exception
path in a hot loop.

---

## Appendix: NuMojo is the array backend

SciJo does not define its own array type. Every `NDArray[dtype]`,
`ComplexNDArray[cdtype]`, `Shape`, and every array-construction routine
(`nm.linspace`, `nm.arange`, `nm.array`, `nm.zeros`, ...) that appears in
this manual and in SciJo's source comes directly from
[NuMojo](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo).
SciJo's `scijo.prelude` module is nothing more than `from numojo.prelude
import *`, re-exported under SciJo's own name for convenience - there is
no SciJo-specific array API layered on top, and NuMojo's own documentation
is the reference for everything array-shaped: indexing, slicing, dtypes,
complex arrays, and array construction.

What SciJo adds on top is the numerical routines themselves - the finite
differences, the quadrature, the interpolants, the FFT, the solvers - all
written to take and return NuMojo's own types.

---

## Appendix: what is not here yet

This manual documents what exists. The README's roadmap tracks what does
not; as of this writing, the notable gaps are:

- **FFT**: 2-D FFT support. Every transform in `scijo.fft` is 1-D only.
- **Optimization**: multi-dimensional root-finding and minimization -
  everything in `scijo.optimize` today is scalar-only.
- **Differentiation**: higher-order Jacobians, and Hessians for
  vector-valued (rather than scalar-valued) functions.
- **Signal processing, sparse matrices, and further linear algebra** are
  listed as future directions but have no code in the repository yet.

None of these are silently half-implemented - a 2-D array handed to `fft`,
or a bracket-based multi-dimensional call to `root_scalar`, simply does not
compile or raises, rather than producing a plausible-looking wrong answer.
