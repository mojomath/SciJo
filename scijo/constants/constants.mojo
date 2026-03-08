# ===----------------------------------------------------------------------=== #
# Scijo: Constants
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Constants Module (scijo.constants.constants)

Provides commonly used scientific constants, primarily in SI units.
Values are based on the SciPy constants module.

The `constants` module includes:
- Mathematical constants (e.g., pi, golden ratio).
- Physical constants (e.g., speed of light, Planck constant).
- SI prefixes (e.g., kilo, milli).
- Binary prefixes (e.g., kibi, mebi).
- Units of mass, length, time, energy, power, and force in SI and imperial systems.

Examples:
    ```mojo
    import scijo.constants as const
    print(const.pi)  # 3.141592653589793
    print(const.c)   # 299792458.0
    print(const.kilo) # 1000.0
    print(const.inch) # 0.0254
    ```
"""

# ===----------------------------------------------------------------------=== #
# MATHEMATICAL CONSTANTS
# ===----------------------------------------------------------------------=== #

comptime pi = 3.141592653589793
"""Pi [dimensionless]."""
comptime golden = 1.618033988749895
"""Golden ratio = (1 + sqrt(5)) / 2  [dimensionless]."""
comptime golden_ratio = 1.618033988749895
"""Golden ratio = (1 + sqrt(5)) / 2  [dimensionless]."""

# ===----------------------------------------------------------------------=== #
# Physical constants (SI units)
# ===----------------------------------------------------------------------=== #

# Speed of light
comptime c = 299792458.0
"""Speed of light in vacuum [m s^-1]."""
comptime speed_of_light: Scalar[f64] = 299792458.0
"""Speed of light in vacuum [m s^-1]."""

# Magnetic and electric constants
comptime mu_0: Scalar[f64] = 1.25663706127e-06
"""Vacuum magnetic permeability [N A^-2]."""
comptime epsilon_0: Scalar[f64] = 8.8541878188e-12
"""Vacuum electric permittivity [F m^-1]."""

# Planck constants
comptime h = 6.62607015e-34
"""Planck constant [J Hz^-1]."""
comptime Planck = 6.62607015e-34
"""Planck constant [J Hz^-1]."""
comptime hbar: Scalar[f64] = 1.0545718176461565e-34
"""Reduced Planck constant [J s]."""

# Gravitational constants
comptime G: Scalar[f64] = 6.6743e-11
"""Newtonian constant of gravitation [m^3 kg^-1 s^-2]."""
comptime gravitational_constant: Scalar[f64] = 6.6743e-11
"""Newtonian constant of gravitation [m^3 kg^-1 s^-2]."""
comptime g = 9.80665
"""Standard acceleration of gravity [m s^-2]."""

# Elementary charge
comptime e = 1.602176634e-19
"""Elementary charge [C]."""
comptime elementary_charge: Scalar[f64] = 1.602176634e-19
"""Elementary charge [C]."""

# Gas constant
comptime R = 8.31446261815324
"""Molar gas constant [J mol^-1 K^-1]."""
comptime gas_constant: Scalar[f64] = 8.31446261815324
"""Molar gas constant [J mol^-1 K^-1]."""

# Fine structure constant
comptime alpha: Scalar[f64] = 0.0072973525643
"""Fine-structure constant [dimensionless]."""
comptime fine_structure: Scalar[f64] = 0.0072973525643
"""Fine-structure constant [dimensionless]."""

# Avogadro constant
comptime N_A = 6.02214076e23
"""Avogadro constant [mol^-1]."""
comptime Avogadro = 6.02214076e23
"""Avogadro constant [mol^-1]."""

# Boltzmann constant
comptime k = 1.380649e-23
"""Boltzmann constant [J K^-1]."""
comptime Boltzmann = 1.380649e-23
"""Boltzmann constant [J K^-1]."""

# Stefan-Boltzmann constant
comptime sigma: Scalar[f64] = 5.6703744191844314e-08
"""Stefan-Boltzmann constant [W m^-2 K^-4]."""
comptime Stefan_Boltzmann: Scalar[f64] = 5.6703744191844314e-08
"""Stefan-Boltzmann constant [W m^-2 K^-4]."""

# Wien displacement law constant
comptime Wien: Scalar[f64] = 0.0028977719551851727
"""Wien wavelength displacement law constant [m K]."""

# Rydberg constant
comptime Rydberg = 10973731.568157
"""Rydberg constant [m^-1]."""

# Masses
comptime m_e = 9.1093837139e-31
"""Electron mass [kg]."""
comptime electron_mass = 9.1093837139e-31
"""Electron mass [kg]."""
comptime m_p = 1.67262192595e-27
"""Proton mass [kg]."""
comptime proton_mass = 1.67262192595e-27
"""Proton mass [kg]."""
comptime m_n = 1.67492750056e-27
"""Neutron mass [kg]."""
comptime neutron_mass = 1.67492750056e-27
"""Neutron mass [kg]."""

# ===----------------------------------------------------------------------=== #
# SI PREFIXES
# ===----------------------------------------------------------------------=== #

# Large prefixes
comptime quetta = 1e30
"""Large prefixes."""
comptime ronna = 1e27
"""Large prefixes."""
comptime yotta = 1e24
"""Large prefixes."""
comptime zetta = 1e21
"""Large prefixes."""
comptime exa = 1e18
"""Large prefixes."""
comptime peta = 1e15
"""Large prefixes."""
comptime tera = 1e12
"""Large prefixes."""
comptime giga = 1e9
"""Large prefixes."""
comptime mega = 1e6
"""Large prefixes."""
comptime kilo = 1e3
"""Large prefixes."""
comptime hecto = 1e2
"""Large prefixes."""
comptime deka = 1e1
"""Large prefixes."""

# Small prefixes
comptime deci = 1e-1
"""Small prefixes."""
comptime centi = 1e-2
"""Small prefixes."""
comptime milli = 1e-3
"""Small prefixes."""
comptime micro = 1e-6
"""Small prefixes."""
comptime nano = 1e-9
"""Small prefixes."""
comptime pico = 1e-12
"""Small prefixes."""
comptime femto = 1e-15
"""Small prefixes."""
comptime atto = 1e-18
"""Small prefixes."""
comptime zepto = 1e-21
"""Small prefixes."""
comptime yocto = 1e-24
"""Small prefixes."""
comptime ronto = 1e-27
"""Small prefixes."""
comptime quecto = 1e-30
"""Small prefixes."""

# ===----------------------------------------------------------------------=== #
# BINARY PREFIXES
# ===----------------------------------------------------------------------=== #

comptime kibi = 1024.0
"""2^10."""
comptime mebi = 1048576.0
"""2^20."""
comptime gibi = 1073741824.0
"""2^30."""
comptime tebi = 1099511627776.0
"""2^40."""
comptime pebi = 1125899906842624.0
"""2^50."""
comptime exbi = 1152921504606846976.0
"""2^60."""
comptime zebi = 1180591620717411303424.0
"""2^70."""
comptime yobi = 1208925819614629174706176.0
"""2^80."""

# ===----------------------------------------------------------------------=== #
# MASS IN KG
# ===----------------------------------------------------------------------=== #

# Base mass units
comptime gram = 1e-3
"""Gram [kg]."""
comptime metric_ton = 1e3
"""Metric ton [kg]."""

# Imperial/US mass units
comptime grain = 64.79891e-6
"""Grain [kg]."""
comptime lb = 0.45359237
"""Pound (avoirdupois) [kg]."""
comptime pound = 0.45359237
"""Pound (avoirdupois) [kg]."""
comptime oz = 0.028349523125
"""Ounce [kg]:Scalar[f64] = pound/16."""
comptime ounce: Scalar[f64] = 0.028349523125
"""Ounce [kg]:Scalar[f64] = pound/16."""
comptime stone = 6.35029318
"""Stone [kg]:Scalar[f64] = 14*pound."""
comptime long_ton: Scalar[f64] = 1016.0469088
"""Long ton [kg]:Scalar[f64] = 2240*pound."""
comptime short_ton: Scalar[f64] = 907.18474
"""Short ton [kg]:Scalar[f64] = 2000*pound."""

# Specialized mass units
comptime troy_ounce: Scalar[f64] = 0.0311034768
"""Troy ounce [kg]:Scalar[f64] = 480*grain."""
comptime troy_pound: Scalar[f64] = 0.3732417216
"""Troy pound [kg]:Scalar[f64] = 12*troy_ounce."""
comptime carat = 0.0002
"""Carat [kg]:Scalar[f64] = 200e-6."""
comptime blob: Scalar[f64] = 175.126835246
"""Blob [kg]:Scalar[f64] = pound*g/0.0254."""
comptime slinch: Scalar[f64] = 175.126835246
"""Slinch [kg]:Scalar[f64] = pound*g/0.0254."""
comptime slug = 14.593902937
"""Slug [kg]:Scalar[f64] = blob/12."""

# Particle masses (2022 CODATA values)
comptime m_u = 1.66053906892e-27
"""Atomic mass constant [kg]."""
comptime u = 1.66053906892e-27
"""Atomic mass constant [kg]."""
comptime atomic_mass: Scalar[f64] = 1.66053906892e-27
"""Atomic mass constant [kg]."""

# ===----------------------------------------------------------------------=== #
# ANGLE IN RADIANS
# ===----------------------------------------------------------------------=== #

comptime degree: Scalar[f64] = 0.017453292519943295
"""Degree [rad]:Scalar[f64] = pi/180."""
comptime arcmin: Scalar[f64] = 0.00029088820866572158
"""Arcminute [rad]:Scalar[f64] = degree/60."""
comptime arcminute: Scalar[f64] = 0.00029088820866572158
"""Arcminute [rad]:Scalar[f64] = degree/60."""
comptime arcsec: Scalar[f64] = 4.8481368110953599e-06
"""Arcsecond [rad]:Scalar[f64] = arcmin/60."""
comptime arcsecond: Scalar[f64] = 4.8481368110953599e-06
"""Arcsecond [rad]:Scalar[f64] = arcmin/60."""

# ===----------------------------------------------------------------------=== #
# TIME IN SECONDS
# ===----------------------------------------------------------------------=== #

comptime minute = 60.0
"""Minute [s]."""
comptime hour = 3600.0
"""Hour [s]:Scalar[f64] = 60*minute."""
comptime day = 86400.0
"""Day [s]:Scalar[f64] = 24*hour."""
comptime week = 604800.0
"""Week [s]:Scalar[f64] = 7*day."""
comptime year = 31536000.0
"""Year [s]:Scalar[f64] = 365*day."""
comptime Julian_year: Scalar[f64] = 31557600.0
"""Julian year [s]:Scalar[f64] = 365.25*day."""

# ===----------------------------------------------------------------------=== #
# LENGTH IN METERS
# ===----------------------------------------------------------------------=== #

# Basic length units
comptime inch = 0.0254
"""Inch [m]."""
comptime foot = 0.3048
"""Foot [m]:Scalar[f64] = 12*inch."""
comptime yard = 0.9144
"""Yard [m]:Scalar[f64] = 3*foot."""
comptime mile = 1609.344
"""Mile [m]:Scalar[f64] = 1760*yard."""
comptime mil = 2.54e-05
"""Mil [m]:Scalar[f64] = inch/1000."""
comptime pt: Scalar[f64] = 0.00035277777777777776
"""Point [m]:Scalar[f64] = inch/72."""
comptime point: Scalar[f64] = 0.00035277777777777776
"""Point [m]:Scalar[f64] = inch/72."""

# Survey units
comptime survey_foot: Scalar[f64] = 0.30480060960121924
"""Survey foot [m]:Scalar[f64] = 1200.0/3937."""
comptime survey_mile: Scalar[f64] = 1609.3472186944375
"""Survey mile [m]:Scalar[f64] = 5280*survey_foot."""

# Maritime and scientific units
comptime nautical_mile = 1852.0
"""Nautical mile [m]."""
comptime fermi = 1e-15
"""Fermi [m]."""
comptime angstrom = 1e-10
"""Angstrom [m]."""
comptime micron = 1e-06
"""Micron [m]."""

# Astronomical units
comptime au = 149597870700.0
"""Astronomical unit [m]."""
comptime astronomical_unit: Scalar[f64] = 149597870700.0
"""Astronomical unit [m]."""
comptime light_year: Scalar[f64] = 9460730472580800.0
"""Light year [m]:Scalar[f64] = Julian_year*c."""
comptime parsec: Scalar[f64] = 3.0856775814913673e16
"""Parsec [m]:Scalar[f64] = au/arcsec."""

# ===----------------------------------------------------------------------=== #
# PRESSURE IN PASCALS
# ===----------------------------------------------------------------------=== #

comptime atm = 101325.0
"""Standard atmosphere [Pa] (2022 CODATA)."""
comptime atmosphere = 101325.0
"""Standard atmosphere [Pa]."""
comptime bar = 100000.0
"""Bar [Pa]:Scalar[f64] = 1e5."""
comptime torr: Scalar[f64] = 133.32236842105263
"""Torr [Pa]:Scalar[f64] = atm/760."""
comptime mmHg: Scalar[f64] = 133.32236842105263
"""MmHg [Pa]:Scalar[f64] = atm/760."""
comptime psi: Scalar[f64] = 6894.757293168361
"""Psi [Pa]:Scalar[f64] = pound*g/(inch*inch)."""

# ===----------------------------------------------------------------------=== #
# AREA IN SQUARE METERS
# ===----------------------------------------------------------------------=== #

comptime hectare = 10000.0
"""Hectare [m^2]:Scalar[f64] = 1e4."""
comptime acre: Scalar[f64] = 4046.8564224
"""Acre [m^2]:Scalar[f64] = 43560*foot^2."""

# ===----------------------------------------------------------------------=== #
# VOLUME IN CUBIC METERS
# ===----------------------------------------------------------------------=== #

# Metric volume
comptime liter = 0.001
"""Liter [m^3]:Scalar[f64] = 1e-3."""
comptime litre = 0.001
"""Litre [m^3]:Scalar[f64] = 1e-3."""

# US volume
comptime gallon: Scalar[f64] = 0.003785411784
"""Gallon (US) [m^3]:Scalar[f64] = 231*inch^3."""
comptime gallon_US: Scalar[f64] = 0.003785411784
"""Gallon (US) [m^3]:Scalar[f64] = 231*inch^3."""
comptime fluid_ounce: Scalar[f64] = 2.95735295625e-05
"""Fluid ounce (US) [m^3]:Scalar[f64] = gallon_US/128."""
comptime fluid_ounce_US: Scalar[f64] = 2.95735295625e-05
"""Fluid ounce (US) [m^3]:Scalar[f64] = gallon_US/128."""
comptime bbl: Scalar[f64] = 0.158987294928
"""Barrel [m^3]:Scalar[f64] = 42*gallon_US."""
comptime barrel: Scalar[f64] = 0.158987294928
"""Barrel [m^3]:Scalar[f64] = 42*gallon_US."""

# UK volume
comptime gallon_imp: Scalar[f64] = 0.00454609
"""Gallon (UK) [m^3]:Scalar[f64] = 4.54609e-3."""
comptime fluid_ounce_imp: Scalar[f64] = 2.8413062499999997e-05
"""Fluid ounce (UK) [m^3]:Scalar[f64] = gallon_imp/160."""

# ===----------------------------------------------------------------------=== #
# SPEED IN METERS PER SECOND
# ===----------------------------------------------------------------------=== #

comptime kmh: Scalar[f64] = 0.2777777777777778
"""Km/h [m s^-1]:Scalar[f64] = 1e3/hour."""
comptime mph = 0.44704
"""Mph [m s^-1]:Scalar[f64] = mile/hour."""
comptime mach = 340.5
"""Mach [m s^-1] (approx at 15°C, 1 atm)."""
comptime speed_of_sound: Scalar[f64] = 340.5
"""Speed of sound [m s^-1] (approx at 15°C, 1 atm)."""
comptime knot: Scalar[f64] = 0.5144444444444445
"""Knot [m s^-1]:Scalar[f64] = nautical_mile/hour."""

# ===----------------------------------------------------------------------=== #
# TEMPERATURE IN KELVIN
# ===----------------------------------------------------------------------=== #

comptime zero_Celsius = 273.15
"""Zero Celsius [K]."""
comptime degree_Fahrenheit: Scalar[f64] = 0.5555555555555556
"""Degree Fahrenheit [K] (for differences only):Scalar[f64] = 1/1.8."""

# ===----------------------------------------------------------------------=== #
# ENERGY IN JOULES
# ===----------------------------------------------------------------------=== #

# Basic energy units
comptime eV: Scalar[f64] = 1.602176634e-19
"""Electron volt [J]:Scalar[f64] = elementary_charge."""
comptime electron_volt: Scalar[f64] = 1.602176634e-19
"""Electron volt [J]:Scalar[f64] = elementary_charge."""

# Thermal energy units
comptime calorie = 4.184
"""Calorie (thermochemical) [J]."""
comptime calorie_th = 4.184
"""Calorie (thermochemical) [J]."""
comptime calorie_IT: Scalar[f64] = 4.1868
"""Calorie (International Steam Table) [J]."""
comptime erg = 1e-07
"""Erg [J]:Scalar[f64] = 1e-7."""

# British thermal units
comptime Btu_th: Scalar[f64] = 1054.3502644888888
"""BTU (thermochemical) [J]:Scalar[f64] = pound*degree_Fahrenheit*calorie_th/gram."""
comptime Btu: Scalar[f64] = 1055.05585262
"""BTU (International Steam Table) [J]:Scalar[f64] = pound*degree_Fahrenheit*calorie_IT/gram."""
comptime Btu_IT: Scalar[f64] = 1055.05585262
"""BTU (International Steam Table) [J]."""

# Explosive energy
comptime ton_TNT: Scalar[f64] = 4184000000.0
"""Ton of TNT [J]:Scalar[f64] = 1e9*calorie_th."""

# ===----------------------------------------------------------------------=== #
# POWER IN WATTS
# ===----------------------------------------------------------------------=== #

comptime hp: Scalar[f64] = 745.6998715822702
"""Horsepower [W]:Scalar[f64] = 550*foot*pound*g."""
comptime horsepower: Scalar[f64] = 745.6998715822702
"""Horsepower [W]:Scalar[f64] = 550*foot*pound*g."""

# ===----------------------------------------------------------------------=== #
# FORCE IN NEWTONS
# ===----------------------------------------------------------------------=== #

comptime dyn = 1e-05
"""Dyne [N]:Scalar[f64] = 1e-5."""
comptime dyne = 1e-05
"""Dyne [N]:Scalar[f64] = 1e-5."""
comptime lbf: Scalar[f64] = 4.4482216152605
"""Pound force [N]:Scalar[f64] = pound*g."""
comptime pound_force: Scalar[f64] = 4.4482216152605
"""Pound force [N]:Scalar[f64] = pound*g."""
comptime kgf = 9.80665
"""Kilogram force [N]:Scalar[f64] = g."""
comptime kilogram_force: Scalar[f64] = 9.80665
"""Kilogram force [N]:Scalar[f64] = g."""
