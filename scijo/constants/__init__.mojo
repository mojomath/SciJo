# ===----------------------------------------------------------------------=== #
# SciJo: Constants module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Constants Module (`scijo.constants`)
=======================================
Provides a collection of physical and mathematical constants, CODATA 2022
recommended values, and utility functions for accessing constant properties.

Available Functions
-------------------
- `value(key)`         — Get the numerical value of a physical constant.
- `unit(key)`          — Get the unit string of a physical constant.
- `precision(key)`     — Get the relative precision of a physical constant.
- `find(substring)`    — Find constants whose names contain a substring.
- `list_all_constants` — List all available constant names.
- `get_constant_tuple` — Get (value, unit, uncertainty) for a constant.
- `convert_temperature`— Convert between Celsius, Fahrenheit, and Kelvin.
- `lambdanu(frequency)`— Compute wavelength from frequency.
- `nulambda(wavelength)`— Compute frequency from wavelength.

Examples
--------
    ```mojo
    from scijo.constants import value, pi, c

    print(pi)                            # 3.141592653589793
    print(value("speed_of_light_in_vacuum"))  # 299792458.0
    ```
"""

from .codata import (
    physical_constants,
)
from .constants import *
from .utils import (
    value,
    list_all_constants,
    get_constant_tuple,
    find,
    unit,
    precision,
    nulambda,
    lambdanu,
    convert_temperature,
)
