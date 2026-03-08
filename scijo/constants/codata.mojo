# ===----------------------------------------------------------------------=== #
# Scijo: CODATA Physical Constants
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""
CODATA Physical Constants Module

Provides access to the CODATA 2022 recommended values for fundamental
physical constants. Constants are stored in a global dictionary, similar to
SciPy's constants module.

Examples:
    ```mojo
    import scijo.constants as const

    var c = const.value("speed_of_light_in_vacuum")
    var h = const.value("Planck_constant")
    var e = const.value("elementary_charge")
    ```

References:
    - CODATA 2022 Recommended Values
    - https://github.com/scipy/scipy/blob/main/scipy/constants/_codata.py

Based on the official CODATA 2022 adjustment for maximum accuracy.
"""

# ===----------------------------------------------------------------------=== #
# Data structure for physical constants
# ===----------------------------------------------------------------------=== #


struct PhysicalConstant[dtype: DType = DType.float64](
    ImplicitlyCopyable, Movable, Writable
):
    """Physical constant data containing value, unit, and uncertainty."""

    var value: Scalar[Self.dtype]
    var unit: String
    var uncertainty: Scalar[Self.dtype]

    fn __init__(
        out self,
        value: Scalar[Self.dtype],
        unit: String,
        uncertainty: Scalar[Self.dtype],
    ):
        self.value = value
        self.unit = unit
        self.uncertainty = uncertainty

    fn __str__(self) raises -> String:
        return String("{} {} ± {}").format(
            self.value, self.unit, self.uncertainty
        )

    fn write_to[W: Writer](self, mut writer: W):
        """
        Writes the array to a writer.

        Args:
            writer: The writer to write the array to.
        """
        try:
            writer.write(
                String("{} {} ± {}\n").format(
                    self.value, self.unit, self.uncertainty
                )
            )
        except e:
            print("Error writing to writer: ", e)


# ===----------------------------------------------------------------------=== #
# Global dictionary of physical constants (CODATA 2022 recommended values)
# ===----------------------------------------------------------------------=== #

comptime physical_constants: Dict[String, PhysicalConstant[f64]] = {
    "speed_of_light_in_vacuum": PhysicalConstant[f64](
        299792458.0, "m s^-1", 0.0
    ),
    "elementary_charge": PhysicalConstant[f64](1.602176634e-19, "C", 0.0),
    "Planck_constant": PhysicalConstant[f64](6.62607015e-34, "J Hz^-1", 0.0),
    "reduced_Planck_constant": PhysicalConstant[f64](
        1.0545718176461565e-34, "J s", 0.0
    ),
    "Boltzmann_constant": PhysicalConstant[f64](1.380649e-23, "J K^-1", 0.0),
    "Avogadro_constant": PhysicalConstant[f64](6.02214076e23, "mol^-1", 0.0),
    "molar_gas_constant": PhysicalConstant[f64](
        8.314462618, "J mol^-1 K^-1", 0.0
    ),
    # Electromagnetic constants
    "vacuum_electric_permittivity": PhysicalConstant[f64](
        8.8541878188e-12, "F m^-1", 1.3e-21
    ),
    "vacuum_mag_permeability": PhysicalConstant[f64](
        1.25663706127e-06, "N A^-2", 1.9e-16
    ),
    "characteristic_impedance_of_vacuum": PhysicalConstant[f64](
        376.730313412, "ohm", 5.9e-8
    ),
    # Gravitational constants
    "Newtonian_constant_of_gravitation": PhysicalConstant[f64](
        6.6743e-11, "m^3 kg^-1 s^-2", 1.5e-15
    ),
    "standard_acceleration_of_gravity": PhysicalConstant[f64](
        9.80665, "m s^-2", 0.0
    ),
    # Fine structure constants
    "fine_structure_constant": PhysicalConstant[f64](
        0.0072973525643, "", 1.1e-12
    ),
    "inverse_fine_structure_constant": PhysicalConstant[f64](
        137.035999177, "", 2.1e-8
    ),
    # Particle masses and properties
    "electron_mass": PhysicalConstant[f64](9.1093837139e-31, "kg", 2.8e-40),
    "proton_mass": PhysicalConstant[f64](1.67262192595e-27, "kg", 5.2e-37),
    "neutron_mass": PhysicalConstant[f64](1.67492750056e-27, "kg", 8.5e-37),
    "atomic_mass_constant": PhysicalConstant[f64](
        1.66053906892e-27, "kg", 5.2e-37
    ),
    "unified_atomic_mass_unit": PhysicalConstant[f64](
        1.66053906892e-27, "kg", 5.2e-37
    ),
    # Additional particle masses (from CODATA 2022),
    "muon_mass": PhysicalConstant[f64](1.883531627e-28, "kg", 4.2e-37),
    "tau_mass": PhysicalConstant[f64](3.16754e-27, "kg", 2.1e-31),
    "alpha_particle_mass": PhysicalConstant[f64](
        6.6446573450e-27, "kg", 2.1e-36
    ),
    "deuteron_mass": PhysicalConstant[f64](3.3435837768e-27, "kg", 1.0e-36),
    "helion_mass": PhysicalConstant[f64](5.0064127862e-27, "kg", 1.6e-36),
    "triton_mass": PhysicalConstant[f64](5.0073567512e-27, "kg", 1.6e-36),
    # Atomic units and radii
    "Bohr_radius": PhysicalConstant[f64](5.29177210544e-11, "m", 8.2e-21),
    "classical_electron_radius": PhysicalConstant[f64](
        2.8179403205e-15, "m", 1.3e-24
    ),
    "Thomson_cross_section": PhysicalConstant[f64](
        6.6524587051e-29, "m^2", 6.2e-38
    ),
    # Additional atomic units (from CODATA 2022),
    "atomic_unit_of_action": PhysicalConstant[f64](
        1.0545718176461565e-34, "J s", 0.0  # exact
    ),
    "atomic_unit_of_charge": PhysicalConstant[f64](
        1.602176634e-19, "C", 0.0  # exact
    ),
    "atomic_unit_of_mass": PhysicalConstant[f64](
        9.1093837139e-31, "kg", 2.8e-40
    ),
    "atomic_unit_of_current": PhysicalConstant[f64](
        6.6236182375082e-3, "A", 7.2e-18
    ),
    "atomic_unit_of_electric_potential": PhysicalConstant[f64](
        27.211386245981, "V", 3.0e-11
    ),
    "atomic_unit_of_electric_field": PhysicalConstant[f64](
        5.14220675112e11, "V m^-1", 8.0e4
    ),
    "atomic_unit_of_electric_dipole_moment": PhysicalConstant[f64](
        8.4783536198e-30, "C m", 1.3e-39
    ),
    "atomic_unit_of_electric_polarizability": PhysicalConstant[f64](
        1.64877727212e-41, "C^2 m^2 J^-1", 5.1e-52
    ),
    "atomic_unit_of_force": PhysicalConstant[f64](
        8.2387235038e-8, "N", 1.3e-15
    ),
    "atomic_unit_of_velocity": PhysicalConstant[f64](
        2.18769126216e6, "m s^-1", 3.4e-1
    ),
    "atomic_unit_of_momentum": PhysicalConstant[f64](
        1.99285191545e-24, "kg m s^-1", 3.1e-34
    ),
    # Magnetic moments and related constants
    "Bohr_magneton": PhysicalConstant[f64](9.2740100657e-24, "J T^-1", 2.9e-33),
    "nuclear_magneton": PhysicalConstant[f64](
        5.0507837393e-27, "J T^-1", 1.6e-36
    ),
    "electron_mag_mom": PhysicalConstant[f64](
        -9.2847646917e-24, "J T^-1", 2.9e-33
    ),
    "proton_mag_mom": PhysicalConstant[f64](
        1.41060679545e-26, "J T^-1", 6.0e-36
    ),
    # Additional magnetic moments (from CODATA 2022),
    "neutron_magnetic_moment": PhysicalConstant[f64](
        -9.6623653e-27, "J T^-1", 2.3e-33
    ),
    "muon_magnetic_moment": PhysicalConstant[f64](
        -4.49044830e-26, "J T^-1", 1.0e-33
    ),
    "deuteron_magnetic_moment": PhysicalConstant[f64](
        4.330735087e-27, "J T^-1", 1.1e-35
    ),
    "helion_magnetic_moment": PhysicalConstant[f64](
        -1.07461755198e-26, "J T^-1", 9.3e-35
    ),
    "triton_magnetic_moment": PhysicalConstant[f64](
        1.5046095178e-26, "J T^-1", 3.0e-35
    ),
    # Gyromagnetic ratios
    "electron_gyromagnetic_ratio": PhysicalConstant[f64](
        1.76085962784e11, "s^-1 T^-1", 5.5e4
    ),
    "proton_gyromagnetic_ratio": PhysicalConstant[f64](
        2.6752218708e8, "s^-1 T^-1", 1.1e1
    ),
    "neutron_gyromagnetic_ratio": PhysicalConstant[f64](
        1.83247174e8, "s^-1 T^-1", 4.3e1
    ),
    # Atomic and molecular constants
    "Rydberg_constant": PhysicalConstant[f64](10973731.568157, "m^-1", 1.2e-5),
    "Hartree_energy": PhysicalConstant[f64](4.3597447222060e-18, "J", 4.8e-28),
    "atomic_unit_of_length": PhysicalConstant[f64](
        5.29177210544e-11, "m", 8.2e-21
    ),
    "atomic_unit_of_energy": PhysicalConstant[f64](
        4.3597447222060e-18, "J", 4.8e-28
    ),
    "atomic_unit_of_time": PhysicalConstant[f64](
        2.4188843265864e-17, "s", 2.6e-27
    ),
    # Compton wavelengths
    "compton_wavelength": PhysicalConstant[f64](
        2.42631023538e-12, "m", 7.6e-22
    ),
    "electron_compton_wavelength": PhysicalConstant[f64](
        2.42631023538e-12, "m", 7.6e-22
    ),
    "proton_compton_wavelength": PhysicalConstant[f64](
        1.32140985360e-15, "m", 4.1e-25
    ),
    "neutron_compton_wavelength": PhysicalConstant[f64](
        1.31959090382e-15, "m", 6.7e-25
    ),
    # Thermodynamic constants
    "stefan_boltzmann_constant": PhysicalConstant[f64](
        5.670374419e-08, "W m^-2 K^-4", 0.0
    ),
    "wien_wavelength_displacement_law_constant": PhysicalConstant[f64](
        0.002897771955, "m K", 0.0
    ),
    "wien_frequency_displacement_law_constant": PhysicalConstant[f64](
        5.878925757e10, "Hz K^-1", 0.0
    ),
    # Radiation constants
    "first_radiation_constant": PhysicalConstant[f64](
        3.741771852e-16, "W m^2", 0.0
    ),
    "second_radiation_constant": PhysicalConstant[f64](
        0.01438776877, "m K", 0.0
    ),
    # Quantum electrical constants
    "conductance_quantum": PhysicalConstant[f64](7.748091729e-05, "S", 0.0),
    "inverse_of_conductance_quantum": PhysicalConstant[f64](
        12906.40372, "ohm", 0.0
    ),
    "magnetic_flux_quantum": PhysicalConstant[f64](2.067833848e-15, "Wb", 0.0),
    "Josephson_constant": PhysicalConstant[f64](4.835978484e14, "Hz V^-1", 0.0),
    "von_klitzing_constant": PhysicalConstant[f64](25812.80745, "ohm", 0.0),
    # Additional quantum constants (from CODATA 2022),
    "quantum_of_circulation": PhysicalConstant[f64](
        3.6369475467e-4, "m^2 s^-1", 1.1e-13
    ),
    "Faraday_constant": PhysicalConstant[f64](
        96485.33212, "C mol^-1", 0.0  # exact
    ),
    # Natural units
    "natural_unit_of_length": PhysicalConstant[f64](
        3.8615926744e-13, "m", 1.2e-22
    ),
    "natural_unit_of_energy": PhysicalConstant[f64](
        8.1871057880e-14, "J", 2.6e-23
    ),
    "natural_unit_of_time": PhysicalConstant[f64](
        1.28808866644e-21, "s", 4.0e-31
    ),
    "natural_unit_of_velocity": PhysicalConstant[f64](
        299792458.0, "m s^-1", 0.0  # exact (speed of light),
    ),
    # Mass ratios and conversion factors
    "proton_electron_mass_ratio": PhysicalConstant[f64](
        1836.152673426, "", 3.2e-8
    ),
    "neutron_electron_mass_ratio": PhysicalConstant[f64](
        1838.68366200, "", 7.4e-7
    ),
    "deuteron_electron_mass_ratio": PhysicalConstant[f64](
        3670.482967655, "", 6.3e-8
    ),
    "alpha_particle_electron_mass_ratio": PhysicalConstant[f64](
        7294.29954144, "", 2.4e-7
    ),
    # Additional mass ratios (from CODATA 2022),
    "muon_electron_mass_ratio": PhysicalConstant[f64](206.7682827, "", 4.6e-6),
    "tau_electron_mass_ratio": PhysicalConstant[f64](3477.23, "", 2.3e-1),
    "proton_neutron_mass_ratio": PhysicalConstant[f64](
        0.99862347797, "", 4.0e-10
    ),
    "deuteron_proton_mass_ratio": PhysicalConstant[f64](
        1.9990075012699, "", 8.4e-12
    ),
    "helion_proton_mass_ratio": PhysicalConstant[f64](
        2.993152671552, "", 7.0e-11
    ),
    # Energy conversion factors
    "electron_volt": PhysicalConstant[f64](1.602176634e-19, "J", 0.0),
    "atomic_mass_unit_electron_volt_relationship": PhysicalConstant[f64](
        9.3149410372e8, "eV", 2.9e-2
    ),
    "electron_volt_joule_relationship": PhysicalConstant[f64](
        1.602176634e-19, "J", 0.0
    ),
    "electron_volt_kelvin_relationship": PhysicalConstant[f64](
        1.160451812e4, "K", 0.0
    ),
    # Additional conversion factors (from CODATA 2022),
    "hartree_electron_volt_relationship": PhysicalConstant[f64](
        27.211386245981, "eV", 3.0e-11
    ),
    "hartree_joule_relationship": PhysicalConstant[f64](
        4.3597447222060e-18, "J", 4.8e-28
    ),
    "electron_volt_hertz_relationship": PhysicalConstant[f64](
        2.417989242e14, "Hz", 0.0  # exact
    ),
    "electron_volt_inverse_meter_relationship": PhysicalConstant[f64](
        8.065543937e5, "m^-1", 0.0  # exact
    ),
    # Planck units
    "planck_length": PhysicalConstant[f64](1.616255e-35, "m", 1.8e-40),
    "planck_mass": PhysicalConstant[f64](2.176434e-8, "kg", 2.4e-13),
    "planck_time": PhysicalConstant[f64](5.391247e-44, "s", 6.0e-49),
    "planck_temperature": PhysicalConstant[f64](1.416784e32, "K", 1.6e27),
    # Pressure and atmosphere
    "standard_atmosphere": PhysicalConstant[f64](101325.0, "Pa", 0.0),
    "standard_state_pressure": PhysicalConstant[f64](100000.0, "Pa", 0.0),
    # Molar properties and additional constants
    "molar_mass_constant": PhysicalConstant[f64](
        1.00000000105e-3, "kg mol^-1", 3.1e-13
    ),
    "molar_mass_of_carbon_12": PhysicalConstant[f64](
        1.20000000126e-2, "kg mol^-1", 3.7e-12
    ),
    "molar_volume_of_ideal_gas_(273.15 K, 100 kPa),": PhysicalConstant[f64](
        2.271095464e-2, "m^3 mol^-1", 0.0
    ),
    "molar_volume_of_ideal_gas_(273.15 K, 101.325 kPa),": PhysicalConstant[f64](
        2.241396954e-2, "m^3 mol^-1", 0.0
    ),
    # Additional fundamental constants (from CODATA 2022),
    "hyperfine_transition_frequency_of_Cs_133": PhysicalConstant[f64](
        9192631770.0, "Hz", 0.0  # exact
    ),
    "luminous_efficacy": PhysicalConstant[f64](683.0, "lm W^-1", 0.0),  # exact
    "Loschmidt_constant_273_15_K_100_kPa": PhysicalConstant[f64](
        2.651645804e25, "m^-3", 0.0  # exact
    ),
    "Loschmidt_constant_273_15_K_101_325_kPa": PhysicalConstant[f64](
        2.686780111e25, "m^-3", 0.0  # exact
    ),
    # Additional Boltzmann constant relationships
    "Boltzmann_constant_in_eV_K": PhysicalConstant[f64](
        8.617333262e-5, "eV K^-1", 0.0  # exact
    ),
    "Boltzmann_constant_in_Hz_K": PhysicalConstant[f64](
        2.083661912e10, "Hz K^-1", 0.0  # exact
    ),
    # Lattice and crystallographic constants
    "lattice_parameter_of_silicon": PhysicalConstant[f64](
        5.431020511e-10, "m", 8.9e-18
    ),
    "lattice_spacing_of_ideal_Si_220": PhysicalConstant[f64](
        1.920155716e-10, "m", 3.2e-18
    ),
}
