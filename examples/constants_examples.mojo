"""
A tour of `scijo.constants`: named mathematical/physical constants, the
CODATA lookup table, and the temperature/optics helper functions.

Run with (from the repository root, after `pixi run package`):

```bash
mojo run -I . -I tests/ examples/constants_examples.mojo
```
"""

from scijo.constants import (
    pi,
    c,
    h,
    hbar,
    golden,
    N_A,
    k,
    value,
    unit,
    precision,
    find,
    get_constant_tuple,
    convert_temperature,
    lambdanu,
    nulambda,
)


def main() raises:
    named_constants()
    codata_lookup()
    temperature_and_optics()


# ===----------------------------------------------------------------------=== #
# Named constants
# ===----------------------------------------------------------------------=== #


def named_constants() raises:
    print("=" * 80)
    print("NAMED CONSTANTS")
    print("=" * 80)

    print("pi     =", pi)
    print("c      =", c, "m/s")
    print("h      =", h, "J Hz^-1")
    print("hbar   =", hbar, "J s")
    print("golden =", golden)
    print("N_A    =", N_A, "mol^-1")
    print("k      =", k, "J K^-1 (Boltzmann constant)")


# ===----------------------------------------------------------------------=== #
# CODATA lookup table, by string key
# ===----------------------------------------------------------------------=== #


def codata_lookup() raises:
    print()
    print("=" * 80)
    print("CODATA LOOKUP: value / unit / precision / find")
    print("=" * 80)

    print(
        "value('speed_of_light_in_vacuum') =", value("speed_of_light_in_vacuum")
    )
    print("unit('Planck_constant')           =", unit("Planck_constant"))
    print("precision('Planck_constant')      =", precision("Planck_constant"))

    var tup = get_constant_tuple("Boltzmann_constant")
    print(
        "get_constant_tuple('Boltzmann_constant') = (",
        tup[0],
        ",",
        tup[1],
        ",",
        tup[2],
        ")",
    )

    var names = find("electron")
    print("find('electron') matches", len(names), "constant names")

    # The short name ("c") is not a key in the CODATA table - only the long
    # descriptive name is.
    try:
        _ = value("c")
    except e:
        print("value('c') raises (short names are not CODATA keys):")
        print(" ", e)


# ===----------------------------------------------------------------------=== #
# Temperature conversion and optics helpers
# ===----------------------------------------------------------------------=== #


def temperature_and_optics() raises:
    print()
    print("=" * 80)
    print("TEMPERATURE CONVERSION / OPTICS HELPERS")
    print("=" * 80)

    var k_temp = convert_temperature["Celsius", "Kelvin"](0.0)
    print("0 C in Kelvin:    ", k_temp)  # 273.15

    var c_temp = convert_temperature["Fahrenheit", "Celsius"](32.0)
    print("32 F in Celsius:  ", c_temp)  # 0.0

    var wavelength = lambdanu(5e14)
    print("lambdanu(5e14 Hz) =", wavelength, "m")

    var frequency = nulambda(500e-9)
    print("nulambda(500e-9 m) =", frequency, "Hz")
