# Changelog

All notable changes to this project will be documented in this file.

The format is based on "Keep a Changelog" and follows Semantic Versioning.

## [v0.2.0] - 2026-07-09
### Added
- **Optimization module** (`scijo.optimize`):
  - `root_scalar` — Unified scalar root-finding interface with method dispatch.
  - `newton` — Newton-Raphson root-finding method.
  - `bisect` — Bisection method for bracketed roots.
  - `secant` — Derivative-free secant method.
  - `minimize_scalar` — Scalar function minimization (Brent's method, golden section, bounded).
  - `OptimizeResult` struct for minimization results.
  - `RootResults` struct for root-finding results.
- **Interpolation** (`scijo.interpolate`):
  - `CubicSpline` — callable natural cubic spline interpolator struct, matching `scipy.interpolate.CubicSpline` API (compile-time `bc_type` param, scalar and array `__call__`).
  - `Akima1DInterpolator` — callable Akima piecewise cubic interpolator struct, matching `scipy.interpolate.Akima1DInterpolator` API (scalar and array `__call__`).
  - `interp[..., type="cubic"]` and `interp[..., type="akima"]` — functional interface for cubic spline and Akima interpolation.
  - Both callable structs exported from `scijo.interpolate` alongside `LinearInterpolator`.
- **Integration** (`scijo.integrate`):
  - `quad[..., method="qag"]` and `quad[..., method="qags"]` — adaptive Gauss-Kronrod integration via MSL backend.
  - `QAG_GK15`, `QAG_GK21`, `QAG_GK31`, `QAG_GK41`, `QAG_GK51`, `QAG_GK61` — compile-time rule constants for the `qag_rule` parameter, mirroring MSL's `MSL_INTEG_GAUSS*` values. All exported from `scijo.integrate`.
- **Optimization** (`scijo.optimize`):
  - `brent` — direct Brent bracketed root-finding wrapper (MSL backend).
  - `root_scalar[..., method="brent"]` — Brent method via unified `root_scalar` interface.
- **Differentiation**:
  - `jacobian` — Jacobian matrix computation for vector-valued functions with parallelized column evaluation.
- **Integration**:
  - `simpson` — Simpson's rule for discrete data (uniform and non-uniform grids).
  - `romb` — Romberg integration with Richardson extrapolation.
- **Constants**:
  - `precision()` — Get relative precision of a physical constant.
  - `find()` — Search constants by substring.
  - `list_all_constants()` — List all available constant names.
  - `get_constant_tuple()` — Get (value, unit, uncertainty) tuple.
  - `convert_temperature()` — Convert between Celsius, Fahrenheit, and Kelvin.
  - `lambdanu()` / `nulambda()` — Frequency-wavelength conversions.
- Developer guide (`docs/developer_guide.md`) covering file headers, docstrings, naming, testing, and contribution conventions.
- Module-level docstrings to all `__init__.mojo` and implementation files.
- Per-field docstrings to all result structs (`DiffResult`, `IntegralResult`, `RootResults`, `OptimizeResult`).

### Changed
- Bumped package version to v0.2.
- Standardized license header block across all files (`# SciJo: <module> module for Mojo`).
- Rewrote all function and struct docstrings to follow the Mojo docstring style guide (consistent `Parameters:`, `Args:`, `Returns:`, `Raises:`, `Constraints:`, `Examples:` sections).
- Standardized `Args:` label everywhere (replaced `Arguments:` in `integrate.fixed_sample`).
- Standardized parameter descriptions (e.g., `"The floating-point data type."` across all modules).
- Renamed misleading `central_diff` variable to `diff_estimate` in forward/backward derivative functions.
- Updated README with new modules (Optimization, Jacobian, Simpson, Romberg) and quick-start examples.
- Updated roadmap to mark completed items.

### Fixed
- Fixed `simpson` loop bounds in `fixed_sample.mojo` to prevent out-of-bounds access for even-sized arrays; added trapezoidal fallback.
- Fixed error messages in `root_scalar.mojo` incorrectly referencing "newton" in `bisect` and `secant` functions.
- Fixed backward difference loop counter in `deriv.mojo` from `Scalar[dtype]` to `Int`.
- Fixed `Scalar[Self.dtype]` typo in `generate_backward_finite_difference_table` return docstring.
- Removed duplicate `value` import in `constants/__init__.mojo`.
- Fixed coefficient table creation in `generate_central_finite_difference_table` and related functions for Dict literal syntax.
- Fixed README installation examples: corrected GitHub URLs to `mojomath/SciJo`, updated Mojo version constraint to `>=1.0.0b2,<2`.
- Fixed docstring format in `convert_temperature()`: removed `Parameters:` section, added `Examples:` with concrete conversions.
- Fixed README code examples: removed undefined `sj` alias, standardized dtype notation from `Float64` to `f64`.

### Removed
- No removals in this release.

### Security
- No security-related changes in this release.

## [v0.1.0] - initial release
- Initial public release (baseline for v0.2)
