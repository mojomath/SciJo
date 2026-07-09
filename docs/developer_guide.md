# SciJo Developer Guide

This guide covers the conventions and standards used across the SciJo codebase. Follow these when contributing new code or modifying existing modules.

---

## Project Structure

```
scijo/
├── __init__.mojo              # Top-level package exports and type aliases
├── constants/                 # Mathematical and physical constants
├── differentiate/             # Numerical differentiation and gradients
├── fft/                       # Fast Fourier Transform algorithms
├── integrate/                 # Numerical integration and quadrature
├── interpolate/               # Interpolation methods
└── optimize/                  # Optimization and root-finding
tests/
├── test_all.sh                # Runs all test files
├── test_differentiate.mojo
├── test_fft.mojo
├── ...
```

Each submodule follows the same internal layout:

- `__init__.mojo` — Public API re-exports.
- One or more implementation files (e.g., `derivative.mojo`, `jacobian.mojo`).
- `utility.mojo` — Internal helpers, result structs, and constants.

---

## File Header

Every `.mojo` file **must** start with the standard license block followed by a module docstring.

```
# ===----------------------------------------------------------------------=== #
# Scijo: <Submodule> - <File Description>
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
```

Immediately after the header, include a module-level docstring:

```
"""<Submodule> Module - <Topic> (scijo.<submodule>.<file>)

Brief description of what this file provides. Keep it to 1–3 lines.

References:
    - <URL or citation>
"""
```

For `__init__.mojo` files, the docstring describes the submodule as a whole:

```
"""<Submodule> Module (scijo.<submodule>)

The `<submodule>` module provides ...
"""
```

---

## Docstring Format

Follow the [Mojo docstring style guide](https://docs.modular.com/mojo/manual/docstrings). Sections appear in this order (include only those that apply):
1. **Summary** — First line, imperative mood ("Computes...", "Returns...", "Finds...").
2. **Parameters:** — Compile-time parameters (dtype, func, etc.).
3. **Args:** — Runtime arguments. Use `Args:`, never `Arguments:`.
4. **Constraints:** — Compile-time constraints if any.
5. **Returns:** — What the function returns.
6. **Raises:** — Errors that can be raised.
7. **NOTES:** - Any notes for the user. 
8. **References:** — Only in module-level docstrings (indent with 4 spaces).
9. **Examples:** - Example usage of code wrapped within a code block (```mojo)

### Function docstring example

```mojo
fn derivative[
    dtype: DType,
    func: fn[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype],
](x0: Scalar[dtype], order: Int = 8) raises -> DiffResult[dtype]:
    """Computes the first derivative of a scalar function using finite differences.

    Provides a unified interface for computing first-order derivatives using
    central, forward, or backward finite difference methods.

    Parameters:
        dtype: The floating-point data type.
        func: Function to differentiate.

    Args:
        x0: Point at which to evaluate the derivative.
        order: Accuracy order for finite differences.

    Returns:
        DiffResult[dtype] containing the derivative and convergence information.

    Raises:
        Error: If the specified order is not supported.
    """
```

### Struct docstring example

```mojo
struct DiffResult[dtype: DType](ImplicitlyCopyable, Writable):
    """Result structure for numerical differentiation operations.

    Encapsulates the computed derivative value, convergence information,
    and diagnostic data.

    Parameters:
        dtype: The floating-point data type (e.g., DType.float32, DType.float64).
    """

    var success: Bool
    """Whether the computation converged successfully."""
    var df: Scalar[Self.dtype]
    """The computed derivative value."""
```

Key rules:

- Struct fields get their own single-line docstring directly below the declaration.
- Use `Parameters:` for the struct-level type parameters (not `Type Parameters:`).
- Do not add `Fields:` or `Usage:` sections — per-field docstrings replace these.

---

## Naming Conventions

Refer to Mojo style guide. 

---

## Error Handling

- Use descriptive, structured error messages that include context:

```mojo
raise Error(
    "SciJo Derivative (Central): Invalid accuracy order specified.\n"
    "  Expected: order ∈ {2, 4, 6, 8}\n"
    "  Got: order = " + String(order)
)
```

- For modules that use NuMojo's error types, use `NumojoError` with a `category` and `location`:

```mojo
raise Error(
    NumojoError(
        category="shape",
        message="Expected y to be 1-D, received ndim=" + String(y.ndim),
        location="trapezoid(y, dx=1.0)",
    )
)
```

- Validate inputs early — check tolerances, step sizes, array shapes, etc. at the top of the function before doing any computation.

---

## Result Structs

Each submodule that returns structured results should define a result struct in its `utility.mojo`. These follow a common pattern:

- Implement `ImplicitlyCopyable`, `Movable`, and `Writable` traits.
- Include a `write_to` method for pretty-printed output.
- Name it descriptively: `DiffResult`, `IntegralResult`, `RootResults`.

---

## Comments

- **Do not write unnecessary comments.** The code and docstrings should be self-explanatory.
- Use `# TODO:` for planned work. These are preserved across cleanups.
- Do not leave `# !` debug notes, `# HACK`, or commented-out code blocks in the codebase. Remove them before committing.
- For inline section separators within large files (e.g., quadrature tables), use:

```
# ===----------------------------------------------------------------------=== #
# Section Title
# ===----------------------------------------------------------------------=== #
```

---

## Testing

Tests live in the `tests/` directory. Each submodule has a corresponding test file (e.g., `test_differentiate.mojo`).

- Use Mojo's built-in `testing` module (`assert_almost_equal`, `assert_equal`, `assert_true`) and other test functions in utility file. 
- Define test helper functions (e.g., mathematical functions for differentiation tests) at the top of the test file.
- Test functions should cover: basic correctness, edge cases, and error conditions.
- Run all tests with:

```sh
pixi run tests
```

This packages the library and runs `tests/test_all.sh`.

---

## Common Commands

| Task | Command |
|---|---|
| Format all code | `pixi run format` |
| Package the library | `pixi run package` |
| Run all tests | `pixi run tests` |

---

## Checklist Before Submitting

- [ ] Every new file has the standard license header.
- [ ] Every new file has a module-level docstring.
- [ ] All public functions and structs have docstrings with appropriate sections.
- [ ] `Args:` is used (not `Arguments:`), and `Parameters:` descriptions say "The floating-point data type." (not "The datatype" or "Data type of the scalar function").
- [ ] Struct fields have per-field docstrings.
- [ ] No commented-out code, `# !` notes, or stale `# HACK` comments.
- [ ] `# TODO:` comments are only for genuine planned work.
- [ ] `__init__.mojo` re-exports all public symbols from the submodule.
- [ ] Tests exist and pass for new functionality.
- [ ] `pixi run format` has been run.
- [ ] `pixi run package` has been run and the package compiles succesfully. 
- [ ] `pixi run tests` has been run and all the tests pass.
