# Changelog

All notable changes to this project will be documented in this file.

The format is based on "Keep a Changelog" and follows Semantic Versioning.

## [Unreleased]
### Added
- **`scijo.integrate.solve_ivp`**, an initial value problem solver:
  - `method="RK45"` (Dormand-Prince 5(4)), with adaptive step-size control
    and FSAL (first-same-as-last) stage reuse.
  - `t_eval` support via cheap cubic-Hermite interpolation within each
    accepted step, instead of forcing the stepper to land on exact times.
  - `first_step` auto-estimation (Hairer, Norsett & Wanner's heuristic) and
    an explicit `max_step` cap.
  - `OdeResult` (`scijo.integrate.ode_utility`), the returned result type.
  - Other methods (RK23, DOP853, Radau, BDF, LSODA), event functions, and
    a standalone queryable dense-output object are not yet implemented.

## [v0.4.0] - 2026-10-26
### Added
- **Types module** (`scijo.types`), an opt-in trait-based layer used by
  `scijo.special` (below) and available for any future module that wants
  the same pattern:
  - `Numeric` - the minimal arithmetic/transcendental trait
    (`+`/`-`/`*`/`/`/`exp`/`log`/`sqrt`/`sin`/`cos`, `one()`/`constant()`)
    a differentiable scalar kernel needs.
  - `NumericComparable` - `Numeric` plus `<`/`<=`/`>`/`==`/`floor()`, for
    algorithms that branch by magnitude (series vs. asymptotic expansion,
    reflection formulas, convergence cutoffs) without the branch ever
    depending on a conformer's derivative.
  - `Primal` - a zero-cost wrapper over `SIMD[dtype, width]` conforming to
    both traits.
  - `Complex` - `re + im*i` with the usual complex arithmetic and
    transcendentals (principal branch for `log`/`sqrt`).
  - `Dual` - forward-mode automatic differentiation: a `(value, derivative)`
    pair that differentiates any kernel written generically over `Numeric`
    via its own chain-rule arithmetic, with nesting
    (`Dual[Dual[Primal[f64]]]`) for higher-order derivatives and
    `Dual[Complex[f64]]` for complex-valued kernels. `variable`/`variable2`
    build a seeded `Dual`/nested `Dual` without naming the nested generic
    type.
  - `examples/types_examples.mojo` and `tests/test_types.mojo`.
- **Special functions module** (`scijo.special`), the first consumer of
  the pattern above: every function has a concrete `Scalar[dtype]`
  overload for ordinary use, backed by a single generic implementation
  over `NumericComparable` - so calling that same generic function on a
  `Dual` instead of a `Primal` gives an exact derivative for free, with no
  separate hand-written derivative formula anywhere in the module:
  - `gamma` / `lngamma` - the gamma function and its log, via the Lanczos
    approximation (`x >= 0.5`) and the reflection formula (`x < 0.5`).
  - `erf` / `erfc` - the error function and its complement, via a power
    series (`|x| < 2`) and a continued fraction (`|x| >= 2`).
  - `beta` / `lnbeta` - `Gamma(a)*Gamma(b)/Gamma(a+b)` and its log, built
    on `lngamma`.
  - `digamma` - `d/dx[ln(Gamma(x))]`, via the shift recurrence into an
    asymptotic series, with the reflection formula for `x <= 0`.
  - `gamma_inc` / `gamma_incc` - the regularized lower/upper incomplete
    gamma functions `P(a, x)`/`Q(a, x)`, via a power series and a
    continued fraction depending on how `x` compares to `a`.
  - `beta_inc` - the regularized incomplete beta function `I_x(a, b)`, via
    Lentz's modified continued fraction.
  - `bessel_j0`/`bessel_j1`/`bessel_y0`/`bessel_y1` (first/second kind) and
    `bessel_i0`/`bessel_i1`/`bessel_k0`/`bessel_k1` (modified first/second
    kind) - order-0/1 Bessel functions, via a convergent power series for
    small `|x|` and a shared asymptotic expansion for large `|x|`.
  - `examples/special_examples.mojo` and `tests/test_special.mojo`.
- **Developer guide**: new "Progressive Disclosure" section
  (`docs/developer_guide.md`) documenting the concrete-`Scalar`-first,
  `Numeric`-generic-second pattern `scijo.types`/`scijo.special` establish,
  as the template for any future module.
- README and Manual sections for both new modules.

### Changed
- Bumped the `mojo` dependency to `>=1.2.0` (from `>=1.1.0`) and `numojo`
  to the `v0.12.0` tag (from `v0.11.0`).

## [v0.3.0] - 2026-10-22
### Added
- **Linear algebra module** (`scijo.linalg`), built on NuMojo 0.11.0's linalg primitives (LU/QR/Cholesky decompositions, `solve`/`lstsq`/`inv`):
  - `lu` - pivoted LU decomposition, scipy-style `(P, L, U)` with `A == P @ L @ U`.
  - `qr` - reduced QR decomposition (thin wrapper over NuMojo's Householder implementation).
  - `cholesky` - Cholesky decomposition, with a `lower` flag for the lower/upper factor.
  - `solve` - solve `Ax = b`, accepting a 1-D or 2-D right-hand side.
  - `solve_triangular` - forward/back substitution for triangular systems.
  - `inv` - matrix inverse (thin wrapper over NuMojo's `inv`).
  - `pinv` - Moore-Penrose pseudo-inverse for full-rank matrices (via least-squares).
  - `lstsq` - least-squares solution to `Ax = b` (thin wrapper over NuMojo's `lstsq`).
  - `norm` - vector (`1`/`2`/`inf`) and matrix (`fro`/`1`/`inf`) norms.
  - `det`, `trace` - re-exported from NuMojo.
  - `matmul`, `dot`, `outer`, `kron`, `tensordot`, `cross`, `diagonal` - re-exported from NuMojo for convenience.
  - The general (non-symmetric) eigenvalue problem, where eigenvalues may be complex, is not yet implemented (tracked as future work).
  - `examples/linalg_examples.mojo` and `tests/test_linalg.mojo` covering the new module.
- **Eigenvalues and SVD** (`scijo.linalg`):
  - `eigh` / `eigvalsh` - eigenvalues and eigenvectors of a real symmetric matrix, via the classic cyclic Jacobi eigenvalue algorithm, iterating to a convergence tolerance (not a fixed sweep count).
  - `svd` / `svdvals` - singular value decomposition via one-sided Jacobi column orthogonalization, supporting rectangular matrices.
  - `matrix_rank`, `cond` - built on `svdvals`.
  - `norm`'s matrix `ord="2"` (induced 2-norm / largest singular value) is now implemented, using `svdvals`.
  - `pinv` rewritten on top of `svd`: now a proper rank-revealing Moore-Penrose pseudo-inverse (`V @ diag(1/s) @ U.T`, zeroing singular values below `rcond` times the largest), rather than full-rank-only.
  - `examples/linalg_examples.mojo` and `tests/test_linalg.mojo` extended to cover the new functions, including comparisons against NumPy's `eigh`/`svd`.
- **Multi-dimensional root finding** (`scijo.optimize.root`), for `f: R^n -> R^n`, matching `scipy.optimize.root`'s frontend signature:
  - `method="newton"` (default): `root[dtype, f](x0, ...)` estimates the Jacobian with `scijo.differentiate.jacobian` at every iteration; `root[dtype, f, jac](x0, ...)` takes an analytic Jacobian instead and skips that cost. Solves the Newton linear system at each step with `scijo.linalg.solve`.
  - `method="broyden1"`: Broyden's "good" method, derivative-free - maintains a running approximation of the inverse Jacobian, starting at the identity and refined after every step by a rank-1 (Sherman-Morrison) correction, so `f` is the only function ever evaluated (no finite-difference Jacobian columns, no analytic derivative).
  - `RootResultVector` - the vector-valued analog of `RootResult` (`x`, `fun`, `nit`, `nfev`, `success`, `message`, `method`).
  - `examples/root_examples.mojo` and `tests/test_root.mojo` covering both methods, including a comparison against `scipy.optimize.root(method="broyden1")` on a linear system with a single, unambiguous root.
- **Args-free callback overloads** for every callback-taking public function: `derivative`, `jacobian`, `hessian`, `quad`, `root_scalar`, `bisect`, `brent`, `secant`, `newton`, `root`, `minimize_scalar`. Each now has a `def(x) -> ...` overload alongside the existing `def(x, args) -> ...` one, so a callback that doesn't need extra parameters no longer has to carry an unused `args` argument - capture what you need from the enclosing scope instead. The `args`-taking overloads are unchanged and still the right tool for a function reused across call sites with different parameter values (a module-level `def` can't `capturing`-capture, since it has no enclosing scope).
- **N-dimensional FFT** (`scijo.fft`):
  - `fftn` / `ifftn` - forward/inverse FFT over one or more axes of an N-dimensional `ComplexNDArray`, matching `scipy.fft.fftn`/`ifftn`. Applies the existing 1-D Cooley-Tukey `fft`/`ifft` to every line along each requested axis; every transformed axis's length must be a power of 2.
  - `fft2` / `ifft2` - convenience wrappers over `fftn`/`ifftn` defaulting to the last two axes, matching `scipy.fft.fft2`/`ifft2`.
  - `examples/fft_examples.mojo` extended and `tests/test_fftn.mojo` added covering the new functions.
- **Multi-dimensional minimization** (`scijo.optimize`):
  - `minimize` - minimize a scalar-valued function of several variables (`f: R^n -> R`), matching `scipy.optimize.minimize`'s frontend signature. Three methods:
    - `method="Nelder-Mead"` (default): derivative-free simplex method. Never evaluates a gradient or Hessian; convergence checked on both the simplex's point-to-point spread (`xatol`) and function-value spread (`fatol`), matching SciPy's own criteria.
    - `method="BFGS"`: quasi-Newton method maintaining a running inverse-Hessian approximation (BFGS secant update) and a backtracking line search satisfying the Armijo condition. Gradient estimated with central finite differences (`fd_step`); converges once `||grad f(x)|| <= gtol`.
    - `method="CG"`: nonlinear conjugate gradient (Polak-Ribiere, with periodic restarts), same gradient/line-search machinery as BFGS but O(n) memory per iteration instead of O(n^2).
  - `OptimizeResultVector` - the vector-valued analog of `OptimizeResult` (`x`, `fun`, `nit`, `nfev`, `success`, `message`, `method`).
  - `examples/minimize_examples.mojo` and `tests/test_minimize.mojo` covering all three methods.
- **Hessian for vector-valued functions** (`scijo.differentiate`):
  - `hessian` overload for `f: R^n -> R^m` (a callback returning `NDArray`
    instead of `Scalar`): applies the existing central-difference stencil
    to every output component and stacks the results into an `(m, n, n)`
    tensor, `H[k, i, j] = ∂²f_k/∂x_i∂x_j`. Args-taking and args-free
    variants, matching the existing scalar-valued `hessian`.
  - `examples/differentiate_examples.mojo` and `tests/test_differentiate.mojo`
    extended to cover the new overloads.

### Changed
- **Removed the MSL dependency**: `scijo.integrate.quad` (`qng`/`qag`/`qags`)
  and `scijo.optimize.root_scalar`'s `bisect`/`brent`/`newton`/`secant` were
  the last functions backed by the external MSL library; both are now native:
  - `quad` is built on the Gauss-Kronrod rules and adaptive-subdivision
    priority queue already present in `scijo.integrate.utility` (15/21/43/87
    point rules, QUADPACK-style). `qng` runs the same accuracy cascade as
    before; `qag` adaptively subdivides using the requested `qag_rule` (the
    31/41/51/61-point requests are served by the next larger tabulated rule,
    43 or 87); `qags` adds Wynn epsilon extrapolation on the running integral
    estimate.
  - `bisect`, `brent` (root-finding), `newton`, and `secant` in
    `scijo.optimize.root_scalar` are now independent implementations of the
    standard algorithms, matching the existing native `minimize_scalar`
    solvers in style.
  - SciJo now depends only on NuMojo; `msl` is removed from `pixi.toml`.
- The `numojo` dependency, temporarily pinned earlier in development to a feature branch adding `qr`/`cholesky`/`lstsq`, now targets the NuMojo v0.11.0 release.

### Fixed
- **`rfft` zero-padding for non-power-of-two inputs**: explicitly initialize
  the padded tail before transforming. Previously those elements contained
  uninitialized data, producing an incorrect spectrum and inverse round trip.
- **Closure-capture miscompile with generic callback parameters**, across every public callback-taking function (`scijo.differentiate.derivative`/`jacobian`/`hessian`, `scijo.integrate.quad`, `scijo.optimize.root_scalar`/`minimize_scalar`/`root`/`minimize`): a closure capturing outer-scope runtime state, passed as a generic `capturing`-effect closure-type parameter and called from inside the receiving generic function (or a further generic helper it forwards to), was silently computing garbage (e.g. a Jacobian of all zeros, or a BFGS gradient of NaN) on the current Mojo toolchain - no error raised. Closures that only read their own `args` parameter (the common, recommended pattern) were unaffected; this only bit a closure that actually captured a variable from its enclosing scope. Fixed everywhere by declaring those closure-type parameters with the parameterized `capturing[_]` effect instead of bare `capturing` (ordering: `raises thin capturing[_] -> ReturnType where ...`, or `raises capturing[_] -> ReturnType` where no `where`-constraint is needed), which lets the compiler track the closure's origin per call site instead of miscomputing it.
- **Test suite correctness audit**: several test files wrapped their entire
  body - including the actual `assert_*` comparisons against SciPy/NumPy -
  in a single bare `try: ... except: print("SciPy not available, skipping
  ...")`. A genuine assertion failure inside that block was indistinguishable
  from a missing import and was silently swallowed, so the test still
  reported PASS. Restructured every such function (`test_fft.mojo`,
  `test_fftn.mojo`, `test_interpolate.mojo`, `test_trapezoid.mojo`) so only
  the import call is protected (with an early `return` on failure) and the
  assertions run unprotected. Two "expect this to raise" tests
  (`test_differentiate.mojo::test_error_conditions`,
  `test_root_scalar.mojo::test_bisect_invalid_bracket_raises`) had the same
  flaw in miniature - a failing `assert_false`/`assert_equal` placed *inside*
  the `try` was itself caught by the following bare `except: pass` - fixed to
  the standard raised-flag-checked-after-the-try pattern already used
  elsewhere in the suite. Restructuring surfaced four real bugs that had been
  masked this way (see below); a fifth (`test_trapezoid.mojo`'s
  `test_scipy_compatibility`) turned out to be a bug in the test itself
  (`python.list(...)` called on the `numpy` module object, which has no such
  attribute - fixed to `Python.list(...)`).
- **`fftfreq`** (`scijo.fft`): wrong split point between positive and
  negative frequencies for odd `n` (used `n // 2` for both parities; the
  positive/zero block needs `(n + 1) // 2` entries, since index `n // 2`
  itself is the last positive frequency, not the first negative one - e.g.
  `fftfreq(7)` returned `[0, 1, 2, -4, -3, -2, -1]` instead of
  `[0, 1, 2, 3, -3, -2, -1]`). `fftshift`/`ifftshift` inherited this via their
  own odd-`n` tests, which reused `fftfreq`.
- **`cumulative_simpson`** and **`simpson`** (`scijo.integrate`): the
  even-length fallback for the last, unpaired interval used a plain
  trapezoid estimate instead of the correct sub-panel integral of the
  quadratic interpolant through the last three points, undercounting that
  interval's curvature (e.g. `simpson([1,4,1,4], dx=1)` returned `8.5`
  instead of scipy's `8.0`). Fixed both the `dx`- and `x`-based `simpson`
  overloads; the `x`-based (non-uniform spacing) overload additionally had
  the main-loop panel formula silently assume the two intervals within each
  Simpson panel were equal length, which is not guaranteed for arbitrary
  `x` - replaced with the general unequal-spacing formula, verified against
  `scipy.integrate.simpson` for several non-uniform cases including spacing
  that varies within a single panel.
- **Golden-section search** (`scijo.optimize.minimize_scalar`,
  `method="Golden"`): the interval-update branches in `_golden_minimize` had
  `a`/`b` assigned to the wrong side, so the search could converge to the
  wrong point outright (for a bracket symmetric about the minimum, the
  minimum was discarded from *both* candidate sub-intervals on the very
  first step). Found while validating the native `root_scalar` solvers above,
  since `"Brent"` and `"Bounded"` were unaffected.
- **`Akima1DInterpolator`** (`scijo.interpolate`), and the functional
  `interp[..., type="akima"]` entry point (previously a second,
  independently-buggy copy of the same logic, now delegating to
  `Akima1DInterpolator`): the weighted-slope formula was only applied to
  interior points; the two points nearest each boundary used ad hoc
  formulas (the raw edge slope, and a plain average of the two nearest
  slopes) instead of Akima's actual boundary rule - extending the slope
  sequence at each end with two quadratically-extrapolated "phantom"
  slopes and applying the same weighted formula everywhere. Verified against
  `scipy.interpolate.Akima1DInterpolator`'s derivative values at every knot
  for several data sets (up to a few `1e-15`).
- Data-entry typo in `scijo.integrate.utility`'s 21-point Gauss-Kronrod
  weight table (`0.0325816230796473` should have been `0.0325581623079647`
  - a digit transposition), found while validating the native `quad`
  above against known integrals; this table backs both
  `gk21_kronrod_weights`/`gk21_gauss_weights` and (via the shared
  embedded-rule values) `gk43_gauss21_weights`.
- **Release packaging** (`pixi.toml`, `pixi.lock`): `pixi.lock` had drifted
  from `pixi.toml` (still locked to a local `../numojo` path checkout from
  development, rather than the `numojo` git dependency `pixi.toml` declares)
  - regenerated via `pixi lock`. The `feature.test` `python` pin
  (`>=3.15.0,<3.16`) was narrowed to `>=3.13,<3.15`: conda-forge's current
  `cpython 3.15.0` build for `osx-arm64` ships no shared `libpython`,
  which crashes every test that imports `numpy`/`scipy` for comparison
  (`dlopen failed`) rather than raising a catchable Mojo error.
- `tests/test_linalg.mojo::test_trace`: `trace` (re-exported from NuMojo)
  now returns a 0-D array; indexing it with `.item(0)` raises
  (`Cannot index a 0-D array ... Call .item() with no arguments`) on the
  NuMojo v0.11.0 release. Fixed to `.item()`.

## [v0.2.0] - 2026-09-04
### Added
- **Optimization module** (`scijo.optimize`):
  - `root_scalar` - Unified scalar root-finding interface with method dispatch.
  - `newton` - Newton-Raphson root-finding method.
  - `bisect` - Bisection method for bracketed roots.
  - `secant` - Derivative-free secant method.
  - `minimize_scalar` - Scalar function minimization (Brent's method, golden section, bounded).
  - `OptimizeResult` struct for minimization results.
  - `RootResults` struct for root-finding results.
- **Interpolation** (`scijo.interpolate`):
  - `CubicSpline` - callable natural cubic spline interpolator struct, matching `scipy.interpolate.CubicSpline` API (compile-time `bc_type` param, scalar and array `__call__`).
  - `Akima1DInterpolator` - callable Akima piecewise cubic interpolator struct, matching `scipy.interpolate.Akima1DInterpolator` API (scalar and array `__call__`).
  - `interp[..., type="cubic"]` and `interp[..., type="akima"]` - functional interface for cubic spline and Akima interpolation.
  - Both callable structs exported from `scijo.interpolate` alongside `LinearInterpolator`.
- **Integration** (`scijo.integrate`):
  - `quad[..., method="qag"]` and `quad[..., method="qags"]` - adaptive Gauss-Kronrod integration via MSL backend.
  - `QAG_GK15`, `QAG_GK21`, `QAG_GK31`, `QAG_GK41`, `QAG_GK51`, `QAG_GK61` - compile-time rule constants for the `qag_rule` parameter, mirroring MSL's `MSL_INTEG_GAUSS*` values. All exported from `scijo.integrate`.
- **Optimization** (`scijo.optimize`):
  - `brent` - direct Brent bracketed root-finding wrapper (MSL backend).
  - `root_scalar[..., method="brent"]` - Brent method via unified `root_scalar` interface.
- **Differentiation**:
  - `jacobian` - Jacobian matrix computation for vector-valued functions with parallelized column evaluation.
- **Integration**:
  - `simpson` - Simpson's rule for discrete data (uniform and non-uniform grids).
  - `romb` - Romberg integration with Richardson extrapolation.
- **Constants**:
  - `precision()` - Get relative precision of a physical constant.
  - `find()` - Search constants by substring.
  - `list_all_constants()` - List all available constant names.
  - `get_constant_tuple()` - Get (value, unit, uncertainty) tuple.
  - `convert_temperature()` - Convert between Celsius, Fahrenheit, and Kelvin.
  - `lambdanu()` / `nulambda()` - Frequency-wavelength conversions.
- Developer guide (`docs/developer_guide.md`) covering file headers, docstrings, naming, testing, and contribution conventions.
- Module-level docstrings to all `__init__.mojo` and implementation files.
- Per-field docstrings to all result structs (`DiffResult`, `IntegralResult`, `RootResults`, `OptimizeResult`).
- **Documentation**: `docs/MANUAL.md`, a full prose tour of every module (getting
  started, per-module sections with gotchas, an Errors section, and appendices
  on the NuMojo backend and unimplemented features).
- **Examples**: an `examples/` directory with one runnable `.mojo` file per
  module (`differentiate`, `integrate`, `interpolate`, `fft`, `constants`,
  `optimize`) plus `examples/run_all.sh` to run them all.
- Re-exported NuMojo's dtype aliases (`f16`, `f32`, `f64`, `i8`...`i256`,
  `u8`...`u256`, `int`, `uint`, `boolean`, `bf16`) from the top-level `scijo`
  package, so `scijo.f64` (or `sj.f64` under `import scijo as sj`) works
  without a separate `numojo` import.

### Changed
- Standardized license header block across all files (`# SciJo: <module> module for Mojo`).
- Rewrote all function and struct docstrings to follow the Mojo docstring style guide (consistent `Parameters:`, `Args:`, `Returns:`, `Raises:`, `Constraints:`, `Examples:` sections).
- Standardized `Args:` label everywhere (replaced `Arguments:` in `integrate.fixed_sample`).
- Standardized parameter descriptions (e.g., `"The floating-point data type."` across all modules).
- Renamed misleading `central_diff` variable to `diff_estimate` in forward/backward derivative functions.
- Updated README with new modules (Optimization, Jacobian, Simpson, Romberg) and quick-start examples.
- Updated roadmap to mark completed items.
- Bumped Mojo requirement to the stable `1.0.0` release (from `1.0.0b2`).
- Pinned the NuMojo dependency to its `main` branch (from `pre-0.10`).
- Reorganized and standardized file headers and imports across the codebase.
- Linked the new Manual and Examples from the README.

### Fixed
- Fixed `simpson` loop bounds in `fixed_sample.mojo` to prevent out-of-bounds access for even-sized arrays; added trapezoidal fallback.
- Fixed error messages in `root_scalar.mojo` incorrectly referencing "newton" in `bisect` and `secant` functions.
- Fixed backward difference loop counter in `deriv.mojo` from `Scalar[dtype]` to `Int`.
- Fixed `Scalar[Self.dtype]` typo in `generate_backward_finite_difference_table` return docstring.
- Removed duplicate `value` import in `constants/__init__.mojo`.
- Fixed coefficient table creation in `generate_central_finite_difference_table` and related functions for Dict literal syntax.
- Fixed README installation examples: corrected GitHub URLs to `mojomath/SciJo`.
- Fixed docstring format in `convert_temperature()`: removed `Parameters:` section, added `Examples:` with concrete conversions.
- Fixed README code examples: removed undefined `sj` alias, standardized dtype notation from `Float64` to `f64`.
- Replaced all direct `._buf.ptr[i]` NDArray buffer access with `.unsafe_load(i)` /
  `.unsafe_store(i, value)` throughout `scijo.interpolate`, `scijo.integrate`,
  and `scijo.fft`, eliminating the deprecation warnings raised by `pixi run package`.
- Fixed `sj.f64`-style calls failing to compile in the test suite - `scijo` did
  not re-export any dtype alias (see Added, above).
- Fixed every code sample in the README Quick Start: `fn` was removed from Mojo
  in favor of `def`; `method=` on `root_scalar`/`minimize_scalar` is a
  compile-time keyword parameter and must go in the brackets, not the call
  arguments; `minimize_scalar`'s bracket argument is `bracket=`, not `Bracket=`;
  `derivative`/`quad` take `atol=`/`rtol=` directly, not a `tolerance=` dict;
  variadic-style `List[Scalar[f64]](2.0)` construction was replaced with the
  list-literal form; the Integration sample's import path was corrected from
  `scijo.integrate.quad` to `scijo.integrate`. All six samples now compile and
  run as written.
- Fixed callback-taking public functions (`derivative`, `jacobian`, `hessian`,
  `root_scalar` and its individual solvers, `minimize_scalar`, `root`) so a
  callback that itself requires `where dtype.is_floating_point()` (e.g. one
  that calls `sin`/`cos`/`exp`) can be passed to them. Previously their
  generic callback-parameter type lacked that constraint, so the
  supplied-closure's type and the expected parameter type disagreed and the
  call failed to type-check, even though the constraint holds trivially once
  the outer `dtype` is fixed to a concrete floating-point type.
- Removed stray "matching `scipy.X`"-style references scattered through
  docstrings and module headers across `scijo/`; genuinely useful
  attributions (e.g. the CODATA data source, the Akima/Simpson algorithmic
  rationale) were kept, reworded to describe behavior directly rather than
  by comparison.

### Removed
- No removals in this release.

### Security
- No security-related changes in this release.

## [v0.1.0] - initial release
- Initial public release (baseline for v0.2)
