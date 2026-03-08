# ===----------------------------------------------------------------------=== #
# Scijo: Utils for Constants
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Utility functions for physical constants.

Includes temperature conversions and frequency-wavelength calculations.
"""

from builtin.value import materialize

from .constants import c
from .codata import physical_constants

# ===----------------------------------------------------------------------=== #
# Functions to access physical constants
# ===----------------------------------------------------------------------=== #


fn value(key: String) raises -> Scalar[DType.float64]:
    """
    Get the value of a physical constant.

    Args:
        key: Name of the physical constant.

    Returns:
        The numerical value of the constant.
    """
    var physical_constants = materialize[physical_constants]()
    if key in physical_constants:
        return physical_constants[key].value
    else:
        print("Warning: Unknown constant '" + key + "'")
        return 0.0


fn unit(key: String) raises -> String:
    """
    Get the unit of a physical constant.

    Args:
        key: Name of the physical constant.

    Returns:
        The unit string of the constant.
    """
    var physical_constants = materialize[physical_constants]()
    if key in physical_constants:
        return physical_constants[key].unit
    else:
        print("Warning: Unknown constant '" + key + "'")
        return ""


fn precision(key: String) raises -> Scalar[DType.float64]:
    """
    Get the relative precision (uncertainty/value) of a physical constant.

    Args:
        key: Name of the physical constant.

    Returns:
        The relative precision of the constant.
    """
    var physical_constants = materialize[physical_constants]()
    if key in physical_constants:
        var constant = physical_constants[key]
        if constant.value != 0.0:
            return constant.uncertainty / constant.value
        else:
            return 0.0
    else:
        print("Warning: Unknown constant '" + key + "'")
        return 0.0


fn find(substring: String = "") raises -> List[String]:
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
fn get_constant_tuple(
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
    else:
        print("Warning: Unknown constant '" + key + "'")
        return (0.0, "", 0.0)


fn list_all_constants() raises -> List[String]:
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


fn convert_temperature[
    old_scalar: String, new_scalar: String
](value: Scalar[f64]) raises -> Scalar[f64]:
    """Converts a temperature value from one scalar to another.

    Parameters:
        old_scalar: The original temperature scale (e.g., "Celsius", "Fahrenheit", "Kelvin").
        new_scalar: The target temperature scale (e.g., "Celsius", "Fahrenheit", "Kelvin").

    Args:
        value: The temperature value to be converted.

    Returns:
        The converted temperature value in the new scalar.
    """

    @parameter
    if old_scalar == "Celsius" and new_scalar == "Fahrenheit":
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


fn lambdanu(frequency: Scalar[f64]) -> Scalar[f64]:
    """
    Calculates the wavelength (lambda) from the frequency (nu) using the speed of light.

    Args:
        frequency: The frequency of the wave in Hz.

    Returns:
        The wavelength in meters.
    """
    return c / frequency


fn nulambda(wavelength: Scalar[f64]) -> Scalar[f64]:
    """
    Calculates the frequency (nu) from the wavelength (lambda) using the speed of light.

    Args:
        wavelength: The wavelength of the wave in meters.

    Returns:
        The frequency in Hz.
    """
    return c / wavelength
