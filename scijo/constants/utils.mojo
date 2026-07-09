# ===----------------------------------------------------------------------=== #
# SciJo: Constants module for Mojo
# Distributed under the Apache 2.0 License.
# ===----------------------------------------------------------------------=== #
"""Utility Functions for Constants (`scijo.constants.utils`)
===========================================================
Includes temperature conversions, frequency-wavelength calculations, and
functions for accessing physical constant properties.
"""

from std.builtin.value import materialize

from scijo.constants.constants import c
from scijo.constants.codata import physical_constants

# ===----------------------------------------------------------------------=== #
# Functions to access physical constants
# ===----------------------------------------------------------------------=== #


def value(key: String) raises -> Scalar[DType.float64]:
    """Get the value of a physical constant.

    Args:
        key: Name of the physical constant.

    Returns:
        The numerical value of the constant.

    Raises:
        Error: If the constant name is not found.
    """
    var physical_constants = materialize[physical_constants]()
    if key in physical_constants:
        return physical_constants[key].value
    raise Error("Unknown physical constant: '" + key + "'")


def unit(key: String) raises -> String:
    """Get the unit of a physical constant.

    Args:
        key: Name of the physical constant.

    Returns:
        The unit string of the constant.

    Raises:
        Error: If the constant name is not found.
    """
    var physical_constants = materialize[physical_constants]()
    if key in physical_constants:
        return physical_constants[key].unit
    raise Error("Unknown physical constant: '" + key + "'")


def precision(key: String) raises -> Scalar[DType.float64]:
    """Get the relative precision (uncertainty/value) of a physical constant.

    Args:
        key: Name of the physical constant.

    Returns:
        The relative precision of the constant.

    Raises:
        Error: If the constant name is not found.
    """
    var physical_constants = materialize[physical_constants]()
    if key in physical_constants:
        var constant = physical_constants[key]
        if constant.value != 0.0:
            return constant.uncertainty / constant.value
        else:
            return 0.0
    raise Error("Unknown physical constant: '" + key + "'")


def find(substring: String = "") raises -> List[String]:
    """
    Find physical constants containing a substring in their name.

    Args:
        substring: Substring to search for (empty returns all constants).

    Returns:
        List of constant names containing the substring.
    """
    var physical_constants = materialize[physical_constants]()
    var result = List[String]()

    for item in physical_constants.items():
        var key = item.key
        if substring == "" or substring in key:
            result.append(key)

    return result^


# Additional helper functions for common access patterns
def get_constant_tuple(
    key: String,
) raises -> Tuple[Scalar[DType.float64], String, Scalar[DType.float64]]:
    """
    Get a physical constant as a tuple (value, unit, uncertainty).

    Args:
        key: Name of the physical constant.

    Returns:
        Tuple containing (value, unit, uncertainty).
    """
    var physical_constants = materialize[physical_constants]()
    if key in physical_constants:
        var constant = physical_constants[key]
        return (constant.value, constant.unit, constant.uncertainty)
    raise Error("Unknown physical constant: '" + key + "'")


def list_all_constants() raises -> List[String]:
    """
    Get a list of all available physical constant names.

    Returns:
        List of all constant names in the database.
    """
    var physical_constants = materialize[physical_constants]()
    var result = List[String]()
    for item in physical_constants.items():
        result.append(item.key)
    return result^


# ===----------------------------------------------------------------------=== #
# Temperature conversion
# ===----------------------------------------------------------------------=== #


def convert_temperature[
    old_scalar: String, new_scalar: String
](value: Scalar[f64]) raises -> Scalar[f64]:
    """Converts a temperature value between Celsius, Fahrenheit, and Kelvin.

    Compile-time parameters `old_scalar` and `new_scalar` specify the source and
    target temperature scales. Supported scales: "Celsius", "Fahrenheit", "Kelvin".

    Args:
        value: The temperature value to be converted.

    Returns:
        The converted temperature value in the target scale.

    Examples:
        ```mojo
        from scijo.constants import convert_temperature

        # Convert 0 Celsius to Kelvin
        var temp_k = convert_temperature["Celsius", "Kelvin"](0.0)  # 273.15

        # Convert 32 Fahrenheit to Celsius
        var temp_c = convert_temperature["Fahrenheit", "Celsius"](32.0)  # 0.0
        ```
    """

    comptime if old_scalar == "Celsius" and new_scalar == "Fahrenheit":
        return (value * 9.0 / 5.0) + 32.0
    elif old_scalar == "Celsius" and new_scalar == "Kelvin":
        return value + 273.15
    elif old_scalar == "Fahrenheit" and new_scalar == "Celsius":
        return (value - 32.0) * 5.0 / 9.0
    elif old_scalar == "Fahrenheit" and new_scalar == "Kelvin":
        return ((value - 32.0) * 5.0 / 9.0) + 273.15
    elif old_scalar == "Kelvin" and new_scalar == "Celsius":
        return value - 273.15
    elif old_scalar == "Kelvin" and new_scalar == "Fahrenheit":
        return ((value - 273.15) * 9.0 / 5.0) + 32.0
    else:
        raise Error(
            "Invalid temperature scales provided. Supported scales are:"
            " Celsius, Fahrenheit, Kelvin."
        )


# ===----------------------------------------------------------------------=== #
# Optics
# ===----------------------------------------------------------------------=== #


def lambdanu(frequency: Scalar[f64]) -> Scalar[f64]:
    """
    Calculates the wavelength (lambda) from the frequency (nu) using the speed of light.

    Args:
        frequency: The frequency of the wave in Hz.

    Returns:
        The wavelength in meters.
    """
    return c / frequency


def nulambda(wavelength: Scalar[f64]) -> Scalar[f64]:
    """
    Calculates the frequency (nu) from the wavelength (lambda) using the speed of light.

    Args:
        wavelength: The wavelength of the wave in meters.

    Returns:
        The frequency in Hz.
    """
    return c / wavelength
