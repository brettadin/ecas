"""Stores the Radtrans object.
"""
import copy
import os
import sys
import warnings
from typing import Any, Callable

import h5py
import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d

from petitRADTRANS import physical_constants as cst
from petitRADTRANS.__file_conversion import rebin_ck_line_opacities
from petitRADTRANS.config import petitradtrans_config_parser
# noinspection PyUnresolvedReferences
from petitRADTRANS.fortran_inputs import fortran_inputs as finput
# noinspection PyUnresolvedReferences
from petitRADTRANS.fortran_radtrans_core import fortran_radtrans_core as fcore
from petitRADTRANS.opacities.opacities import CIAOpacity, CloudOpacity, CorrelatedKOpacity, LineByLineOpacity, Opacity
from petitRADTRANS.physics import (
    flux_hz2flux_cm, frequency2wavelength, hz2um, rebin_spectrum, um2hz, wavelength2frequency
)
from petitRADTRANS.utils import LockedDict, intersection_indices


class Radtrans:
    r"""Calculate spectra using a given set of opacities.

    Args:
        pressures (Optional):
            (bar) Array defining the pressure grid to be used for the spectral calculations.
        wavelength_boundaries (Optional):
            List containing left and right border of wavelength region to be considered, in micron. If nothing else
            is specified, it will be equal to ``[0.05, 300]``, hence using the full petitRADTRANS wavelength range
            (0.11 to 250 microns for ``'c-k'`` mode, 0.3 to 30 microns for the ``'lbl'`` mode). The larger the
            range the longer the computation time.
        line_species (Optional):
            List of strings, denoting which line absorber species to include.
        gas_continuum_contributors (Optional):
            List of strings, denoting which continuum absorber species to include.
        rayleigh_species (Optional):
            List of strings, denoting which Rayleigh scattering species to include.
        cloud_species (Optional):
            List of strings, denoting which cloud opacity species to include.
        line_opacity_mode (Optional[string]):
            If equal to ``'c-k'``: use low-resolution mode, at :math:`\\lambda/\\Delta \\lambda = 1000`, with the
            correlated-k assumption. if equal to ``'lbl'``: use high-resolution mode, at
            :math:`\\lambda/\\Delta \\lambda = 10^6`, with a line-by-line treatment.
        line_by_line_opacity_sampling (Optional[int]):
            If ``mode = 'lbl'``, then this will only load every line_by_line_opacity_sampling-nth point of the
            high-resolution opacities. This can be used to save time and RAM.
            The user should verify whether this leads to solutions which are identical to the results of the
            non down-sampled :math:`10^6` resolution.
        scattering_in_emission (Optional[bool]):
            Will be ``False`` by default.
            If ``True`` scattering will be included in the emission spectral calculations. Note that this increases
            the runtime of pRT!
        emission_angle_grid (Optional):
            Array defining the cosines of the angle grid to be used for the emission spectrum calculations,
            and their weights. The array is of shape (2, n_angles), with emission_angle_grid[0] being the cosines of
            the angles, and emission_angle_grid[1] being the weights.
            A dictionary of array with keys 'cos_angles' and 'weights' can also be used.
            If None, a default set of values and weights are used.
        anisotropic_cloud_scattering (Optional[bool, str]):
            If True, anisotropic cloud scattering opacities are used for the spectral calculations.
            If False, isotropic cloud scattering opacities are used for the spectral calculations.
            If 'auto' (recommended), anisotropic_cloud_scattering is set to True for emission spectrum
            calculations, and to False for transmission spectrum calculations.
        path_input_data (Optional[str]):
            Path to the input_data folder, containing the files to be loaded by petitRADTRANS.
    """
    # TODO replace all name mangled functions (starting with __) with "private" functions (starting with _) in v. 4.0.0
    __dat_opacity_files_warning_message = (
        "loading opacities from .dat files is discouraged, the HDF5 format offer better performances at for a lower "
        "memory usage\n\n"
        "Converting petitRADTRANS .dat opacity files into HDF5 can be done by executing:\n"
        ">>> from petitRADTRANS.__file_conversion import convert_all\n"
        ">>> convert_all()\n\n"
        "Alternatively, the petitRADTRANS HDF5 files can be downloaded "
        "(see https://petitradtrans.readthedocs.io/en/latest/content/available_opacities.html)"
    )

    __line_opacity_property_setting_warning_message = (
        "setting a Radtrans line opacity property should be avoided\n"
        "These properties are loaded from the opacity data in the input_data directory and are inter-dependent "
        "(they need to be updated for consistency)\n"
        "It is recommended to create a new Radtrans instance instead"
    )

    __property_setting_warning_message = (
        "setting a Radtrans property directly is not recommended\n"
        "Create a new Radtrans instance (recommended) "
        "or re-do all the setup steps necessary for the modification to be taken into account"
    )

    _frequency_grid_misalignment_tolerance: float = 2e-3

    def __init__(
        self,
        pressures: npt.NDArray[np.floating] = None,
        wavelength_boundaries: npt.NDArray[np.floating] = None,
        line_species: list[str] = None,
        gas_continuum_contributors: list[str] = None,
        rayleigh_species: list[str] = None,
        cloud_species: list[str] = None,
        line_opacity_mode: str = 'c-k',
        line_by_line_opacity_sampling: int = 1,
        scattering_in_emission: bool = False,
        emission_angle_grid: npt.NDArray[np.floating] = None,
        anisotropic_cloud_scattering: bool = 'auto',
        path_input_data: str = None
    ):
        if path_input_data is None:
            path_input_data = petitradtrans_config_parser.get_input_data_path()

        # Inputs checks
        self.__check_line_opacity_mode(line_opacity_mode)
        self.__check_anisotropic_cloud_scattering(anisotropic_cloud_scattering)

        # Initialize properties
        if pressures is None:
            warnings.warn("pressure was not set, initializing one layer at 1 bar")
            pressures: npt.NDArray[np.floating] = np.array([1.0])  # bar

        self._pressures = pressures * 1e6  # bar to cgs  # TODO pressure could be spectral function argument

        self.__check_pressures(pressures)

        if line_species is None:
            self._line_species = []
        else:
            self._line_species = line_species

        if gas_continuum_contributors is None:
            self._gas_continuum_contributors = []
        else:
            self._gas_continuum_contributors = gas_continuum_contributors

        if rayleigh_species is None:
            self._rayleigh_species = []
        else:
            self._rayleigh_species = rayleigh_species

        if cloud_species is None:
            self._cloud_species = []
        else:
            self._cloud_species = cloud_species

        self._line_opacity_mode = line_opacity_mode
        self._line_by_line_opacity_sampling = line_by_line_opacity_sampling
        self._scattering_in_emission = scattering_in_emission

        if wavelength_boundaries is None:
            self._wavelength_boundaries = np.array([0.05, 300.])  # um
        else:
            self.__check_wavelength_boundaries(wavelength_boundaries)
            self._wavelength_boundaries = wavelength_boundaries

        self._anisotropic_cloud_scattering = anisotropic_cloud_scattering

        self._path_input_data = path_input_data

        # Initialize non-settable properties
        self.__scattering_in_transmission = False
        self.__sum_opacities = False
        self.__absorber_present = not (
                len(self._line_species) == 0
                and len(self._gas_continuum_contributors) == 0
                and len(self._rayleigh_species) == 0
                and len(self._cloud_species) == 0
        )
        self.__use_smart_clear_opacities = False

        # Initialize line parameters
        self._frequencies, self._frequency_bins_edges = self._init_frequency_grid()

        # Initialize loaded line opacities variables
        self._lines_loaded_opacities = LockedDict.build_and_lock(
            {
                'temperature_pressure_grid': {},
                'temperature_grid_size': {},
                'pressure_grid_size': {},
                'opacity_grid': {},
                'g_gauss': np.array(np.ones(1), dtype='d', order='F'),
                'weights_gauss': np.array(np.ones(1), dtype='d', order='F')
            }
        )

        # Initialize cia opacities
        self._cias_loaded_opacities = self._init_cia_loaded_opacities(
            cia_contributors=self.__get_cia_contributors(self._gas_continuum_contributors)
        )

        # Initialize loaded cloud opacities variables
        self._clouds_loaded_opacities = LockedDict.build_and_lock(
            {
                'wavelengths': None,
                'absorption_opacities': None,
                'scattering_opacities': None,
                'particles_radii': None,
                'particles_radii_bins': None,
                'particles_densities': None,
                'particles_asymmetry_parameters': None,
            }
        )

        # Initialize the angle (mu) grid for the emission spectral calculations
        self._emission_angle_grid = LockedDict.build_and_lock(
            {
                'cos_angles': None,
                'weights': None
            }
        )

        if emission_angle_grid is None:
            self._emission_angle_grid['cos_angles'] = np.array([
                0.1127016654,
                0.5,
                0.8872983346
            ])

            self._emission_angle_grid['weights'] = np.array([
                0.2777777778,
                0.4444444444,
                0.2777777778
            ])
        else:
            if isinstance(emission_angle_grid, dict):
                self._emission_angle_grid['cos_angles'] = copy.deepcopy(emission_angle_grid['cos_angles'])
                self._emission_angle_grid['weights'] = copy.deepcopy(emission_angle_grid['weights'])
            elif hasattr(emission_angle_grid, '__iter__'):
                self._emission_angle_grid['cos_angles'] = np.array(emission_angle_grid[0], copy=True)
                self._emission_angle_grid['weights'] = np.array(emission_angle_grid[1], copy=True)

                if self._emission_angle_grid['cos_angles'].size != self._emission_angle_grid['weights'].size:
                    raise ValueError(f"emission_cos_angle_grid_weights must be an array of the same size than "
                                     f"emission_cos_angle_grid ({self._emission_angle_grid['cos_angles'].size}), "
                                     f"but is of size {self._emission_angle_grid['weights'].size}")
            else:
                raise ValueError(f"emission_angle_grid must be a dictionary or an iterable of shape (2, n_angles), but "
                                 f"is of type {type(emission_angle_grid)}")

        # Load all opacities
        self.load_all_opacities()

    def __getattr__(self, name):
        """Override of the object base __getattr__ method, in order to hint towards pRT3 names when pRT2 names are used.
        """
        # TODO remove in version 4.0.0
        base_message = f"'{self.__class__.__name__}' object has no attribute '{name}'"

        def __handle_deprecated_attributes(suggested_name):
            raise AttributeError(f"{base_message}. Maybe you meant {suggested_name}?")

        if name == 'calc_flux':
            __handle_deprecated_attributes(suggested_name=self.calculate_flux.__name__)
        elif name == 'calc_transm':
            __handle_deprecated_attributes(suggested_name=self.calculate_transit_radii.__name__)

        return super().__getattribute__(name)

    @property
    def anisotropic_cloud_scattering(self):
        return self._anisotropic_cloud_scattering

    @anisotropic_cloud_scattering.setter
    def anisotropic_cloud_scattering(self, mode: str):
        warnings.warn(self.__property_setting_warning_message)
        self.__check_anisotropic_cloud_scattering(mode)
        self._anisotropic_cloud_scattering = mode

    @property
    def cias_loaded_opacities(self):
        return self._cias_loaded_opacities

    @cias_loaded_opacities.setter
    def cias_loaded_opacities(self, dictionary: dict[str, dict[str, npt.NDArray[np.floating]]]):
        warnings.warn(
            "setting the Radtrans CIA opacity property should be avoided\n"
            "These properties are loaded from the opacity data in the input_data directory and are inter-dependent "
            "(they need to be updated for consistency)\n"
            "It is recommended to create a new Radtrans instance instead"
        )
        for key1, value1 in dictionary.items():
            for key2, value2 in value1.items():
                self._cias_loaded_opacities[key1][key2] = value2

    @property
    def clouds_loaded_opacities(self):
        return self._clouds_loaded_opacities

    @clouds_loaded_opacities.setter
    def clouds_loaded_opacities(self, dictionary: dict[str, npt.NDArray[np.floating]]):
        warnings.warn(
            "setting the Radtrans cloud opacity property should be avoided\n"
            "These properties are loaded from the opacity data in the input_data directory and are inter-dependent "
            "(they need to be updated for consistency)\n"
            "It is recommended to create a new Radtrans instance instead"
        )
        for key, value in dictionary.items():
            self._clouds_loaded_opacities[key] = value

    @property
    def cloud_species(self):
        return self._cloud_species

    @cloud_species.setter
    def cloud_species(self, species: list):
        warnings.warn(self.__property_setting_warning_message)
        self._cloud_species = species

    @property
    def emission_angle_grid(self):
        return self._emission_angle_grid

    @emission_angle_grid.setter
    def emission_angle_grid(self, dictionary: dict[str, npt.NDArray[np.floating]]):
        warnings.warn(self.__property_setting_warning_message)

        for key, value in dictionary.items():
            self._emission_angle_grid[key] = value

    @property
    def frequencies(self):
        return self._frequencies

    @frequencies.setter
    def frequencies(self, array: npt.NDArray[np.floating]):
        warnings.warn(
            "setting frequencies directly should be avoided\n"
            "This property is loaded from the opacity data in the input_data directory "
            "and is inter-dependent with other line opacities parameters, that also need to be updated\n"
            "It is recommended to create a new Radtrans instance with e.g. different wavelength boundaries instead"
        )
        self._frequencies = array

    @property
    def frequency_bins_edges(self):
        return self._frequency_bins_edges

    @frequency_bins_edges.setter
    def frequency_bins_edges(self, array: npt.NDArray[np.floating]):
        warnings.warn(self.__line_opacity_property_setting_warning_message)
        self._frequency_bins_edges = array

    @property
    def gas_continuum_contributors(self):
        return self._gas_continuum_contributors

    @gas_continuum_contributors.setter
    def gas_continuum_contributors(self, species: list[str]):
        warnings.warn(self.__property_setting_warning_message)
        self._gas_continuum_contributors = species

    @property
    def line_by_line_opacity_sampling(self):
        return self._line_by_line_opacity_sampling

    @line_by_line_opacity_sampling.setter
    def line_by_line_opacity_sampling(self, value: int):
        warnings.warn(self.__property_setting_warning_message)
        self._line_by_line_opacity_sampling = value

    @property
    def lines_loaded_opacities(self):
        return self._lines_loaded_opacities

    @lines_loaded_opacities.setter
    def lines_loaded_opacities(self, dictionary: dict[str, npt.NDArray[np.floating]]):
        warnings.warn(self.__line_opacity_property_setting_warning_message)

        for key, value in dictionary.items():
            self._lines_loaded_opacities[key] = value

    @property
    def line_opacity_mode(self):
        return self._line_opacity_mode

    @line_opacity_mode.setter
    def line_opacity_mode(self, mode):
        warnings.warn(self.__property_setting_warning_message)
        self.__check_line_opacity_mode(mode)
        self._line_opacity_mode = mode

    @property
    def line_species(self):
        return self._line_species

    @line_species.setter
    def line_species(self, species: list):
        warnings.warn(self.__property_setting_warning_message)
        self._line_species = species

    @property
    def path_input_data(self):
        return self._path_input_data

    @path_input_data.setter
    def path_input_data(self, path: str):
        warnings.warn(self.__property_setting_warning_message)
        self.__check_path_input_data(path)
        self._path_input_data = path

    @property
    def pressures(self) -> npt.NDArray[np.floating]:
        # TODO pressures doesn't need to be a property
        return self._pressures

    @pressures.setter
    def pressures(self, array):
        warnings.warn(self.__property_setting_warning_message)
        self.__check_pressures(array)
        self._pressures = array

    @property
    def rayleigh_species(self):
        return self._rayleigh_species

    @rayleigh_species.setter
    def rayleigh_species(self, species: list):
        warnings.warn(self.__property_setting_warning_message)
        self._rayleigh_species = species

    @property
    def scattering_in_emission(self):
        return self._scattering_in_emission

    @scattering_in_emission.setter
    def scattering_in_emission(self, value: bool):
        warnings.warn(self.__property_setting_warning_message)
        self._scattering_in_emission = value

    @property
    def wavelength_boundaries(self):
        return self._wavelength_boundaries

    @wavelength_boundaries.setter
    def wavelength_boundaries(self, array: npt.NDArray[np.floating]):
        warnings.warn(self.__property_setting_warning_message)
        self.__check_wavelength_boundaries(array)
        self._wavelength_boundaries = array

    @staticmethod
    def __check_anisotropic_cloud_scattering(mode):
        if mode not in ['auto', True, False]:
            raise ValueError(f"anisotropic cloud scattering must be 'auto'|True|False, but was '{mode}'")

    @staticmethod
    def __check_line_opacity_mode(mode):
        if mode not in ['c-k', 'lbl']:
            raise ValueError(f"opacity mode must be 'c-k'|'lbl', but was '{mode}'")

    @staticmethod
    def __check_input_data_file_existence(path):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"no such file or directory: '{path}'\n"
                f"This may be caused by an incorrect input_data path, outdated file formatting, or a missing file\n\n"
                f"To set the input_data path, execute: \n"
                f">>> from petitRADTRANS.config.configuration import petitradtrans_config_parser\n"
                f">>> petitradtrans_config_parser.set_input_data_path('path/to/input_data')\n"
                f"replacing 'path/to/' with the path to the input_data directory (setting "
                f"pRT_input_data_path by an environment variable is no longer supported, "
                f"this environment variable can be removed safely from your setup)\n\n"
                f"To update the outdated .dat files to HDF5, execute:\n"
                f">>> from petitRADTRANS.__file_conversion import convert_all\n"
                f">>> convert_all()\n\n"
                f"To download the missing file, "
                f"see https://petitradtrans.readthedocs.io/en/latest/content/installation.html"
            )

    @staticmethod
    def __check_path_input_data(path):
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"no such directory: '{path}'\n"
                f"This may be caused by an incorrect input_data path, or a missing basic input_data directory\n\n"
                f"To set the input_data path, execute: \n"
                f">>> from petitRADTRANS.config.configuration import petitradtrans_config_parser\n"
                f">>> petitradtrans_config_parser.set_input_data_path('path/to/input_data')\n"
                f"replacing 'path/to/' with the path to the input_data directory (setting "
                f"pRT_input_data_path by an environment variable is no longer supported, "
                f"this environment variable can be removed safely from your setup)\n\n"
                f"To download the basic input_data directory, "
                f"see https://petitradtrans.readthedocs.io/en/latest/content/installation.html"
            )

    @staticmethod
    def __check_pressures(pressures):
        if not np.all(np.diff(pressures) > 0):
            raise ValueError("pressures must be an array in increasing order")

    @staticmethod
    def __check_return_clear_spectrum_relevance(
        return_clear_spectrum, clouds_have_effect,
        opaque_cloud_top_pressure, cloud_fraction
    ):
        if return_clear_spectrum and not clouds_have_effect and opaque_cloud_top_pressure is None:
            warnings.warn(
                "requested to return a clear spectrum, but clouds have no effect with this setup\n"
                "To remove this warning, add non-zero cloud mass fractions, set 'cloud_fraction' to a value > 0, "
                "add an opaque cloud layer, or set 'return_clear_spectrum' to False."
            )

        if return_clear_spectrum and cloud_fraction < 1:
            warnings.warn(
                "both the clear and cloudy spectra will be returned, but the returned cloudy spectrum will have "
                "a partial cloud coverage\n"
                "To remove this warning, set 'cloud_fraction' to 1, or set 'return_clear_spectrum' to False."
            )

    @staticmethod
    def __check_wavelength_boundaries(boundaries):
        if np.size(boundaries) != 2:
            raise ValueError(f"wavelengths boundaries must be an array of 2 floats, but was {boundaries}")

        if not boundaries[0] < boundaries[1]:
            raise ValueError(f"wavelengths boundaries must be an array of 2 floats in increasing order, "
                             f"but was {boundaries}")

    def __clouds_have_effect(self, mass_fractions, cloud_fraction):
        """Check if the clouds have any effect, i.e. if the cloud species MMR is greater than 0.

        Args:
            mass_fractions: atmospheric mass mixing ratios
            cloud_fraction: cloud coverage fraction
        """
        if cloud_fraction == 0:
            return False

        if len(self._cloud_species) > 0:
            for i_spec in range(len(self._cloud_species)):
                if np.any(mass_fractions[self._cloud_species[i_spec]] > 0):
                    return True  # add cloud opacity only if there are actually clouds

        return False

    @staticmethod
    def __compute_additional_continuum_opacities(
        frequencies,
        pressures,
        additional_absorption_opacities_function,
        additional_scattering_opacities_function,
        cloud_photosphere_median_optical_depth
    ):
        additional_absorption_opacities = None
        additional_scattering_opacities = None
        cloud_absorption_opacities = None
        cloud_scattering_opacities = None

        # Add optional absorption opacity from outside
        if additional_absorption_opacities_function is None:
            if cloud_photosphere_median_optical_depth is not None:
                cloud_absorption_opacities = np.zeros((frequencies.size, pressures.size))
        else:
            additional_absorption_opacities = additional_absorption_opacities_function(
                hz2um(frequencies),
                pressures * 1e-6  # cgs to bar
            )

            if cloud_photosphere_median_optical_depth is not None:  # the additional opacities are treated as clouds
                # This assumes a single cloud model that is given by the parametrized opacities from
                # give_absorption_opacity and give_scattering_opacity
                cloud_absorption_opacities = additional_absorption_opacities

        # Add optional scattering opacity from outside
        if additional_scattering_opacities_function is None:
            if cloud_photosphere_median_optical_depth is not None:
                cloud_scattering_opacities = np.zeros((frequencies.size, pressures.size))
        else:
            additional_scattering_opacities = additional_scattering_opacities_function(
                hz2um(frequencies),
                pressures * 1e-6  # cgs to bar
            )

            if cloud_photosphere_median_optical_depth is not None:  # the additional opacities are treated as clouds
                # This assumes a single cloud model that is given by the parametrized opacities from
                # give_absorption_opacity and give_scattering_opacity
                cloud_scattering_opacities = additional_scattering_opacities

        return (
            additional_absorption_opacities, additional_scattering_opacities,
            cloud_absorption_opacities, cloud_scattering_opacities
        )

    @staticmethod
    def __compute_cloud_continuum_opacities(
        frequencies,
        frequency_bins_edges,
        pressures,
        temperatures,
        mass_fractions,
        mean_molar_masses,
        reference_gravity,
        scattering_in_transmission,
        sum_opacities,
        opaque_cloud_top_pressure,
        power_law_opacity_350nm,
        power_law_opacity_coefficient,
        clouds_have_effect,
        cloud_species,
        clouds_particles_porosity_factor,
        clouds_loaded_opacities,
        anisotropic_cloud_scattering,
        cloud_particles_mean_radii,
        cloud_particle_radius_distribution_std,
        cloud_f_sed,
        eddy_diffusion_coefficients,
        cloud_particles_radius_distribution,
        cloud_hansen_a,
        cloud_hansen_b,
        cloud_particle_number_density_grid,
        cloud_photosphere_median_optical_depth,
        complete_coverage_clouds,
        return_cloud_contribution,
        target_particle_radii,
        target_pressure
    ):
        # Initialization
        cloud_absorption_opacities = None
        cloud_scattering_opacities = None
        cloud_opacities = None
        _cloud_particles_mean_radii = None

        cloud_continuum_opacities = np.zeros((frequencies.size, pressures.size))
        cloud_continuum_opacities_scattering = 0

        # Add opaque cloud deck opacity
        if opaque_cloud_top_pressure is not None:
            cloud_continuum_opacities[:, pressures > opaque_cloud_top_pressure * 1e6] += 1e99  # TODO why '+=' and not '='?  # noqa E501

        # Add power law opacity
        if power_law_opacity_350nm is not None and power_law_opacity_coefficient is not None:
            scattering_in_transmission = True
            cloud_continuum_opacities_scattering += (
                Radtrans._compute_power_law_opacities(
                    power_law_opacity_350nm=power_law_opacity_350nm,
                    power_law_opacity_coefficient=power_law_opacity_coefficient,
                    frequencies=frequencies,
                    n_layers=pressures.size
                )
            )
        elif not (power_law_opacity_350nm is None and power_law_opacity_coefficient is None):
            if power_law_opacity_350nm is None:
                none_parameter = 'power_law_opacity_350nm'
            elif power_law_opacity_coefficient is None:
                none_parameter = 'power_law_opacity_coefficient'
            else:
                raise RuntimeError(
                    f"one of the two power law opacity parameters must be None, "
                    f"but 'power_law_opacity_350nm' was {power_law_opacity_350nm} "
                    f"and 'power_law_opacity_coefficient' was {power_law_opacity_coefficient}"
                )

            warnings.warn(
                f"to include a power law opacity, "
                f"both parameters 'power_law_opacity_350nm' "
                f"and 'power_law_opacity_coefficient' "
                f"must be not None, but '{none_parameter}' is None\n"
                f"To remove this warning, set '{none_parameter}' with a float, or set both parameters to None\n"
                f"Power law opacity will not be included"
            )

        # Calculate cloud opacities
        if clouds_have_effect or len(complete_coverage_clouds) > 0:  # add cloud opacities only if there are clouds
            cloud_species_mass_fractions = {
                species: mass_fraction for species, mass_fraction in mass_fractions.items()
                if species in cloud_species
            }

            if len(cloud_species_mass_fractions) != len(cloud_species):
                raise ValueError(
                    f"the number of cloud species mass fractions ({len(cloud_species_mass_fractions)}) "
                    f"does not match the number of cloud species ({len(cloud_species)}); "
                    f"check for inconsistencies between mass_fractions keys "
                    f"({list(mass_fractions.keys())}) "
                    f"and cloud_species keys ({cloud_species})"
                )

            scattering_in_transmission = True

            clouds_particles_porosity_factor = Radtrans.__init_clouds_particles_porosity_factor(
                clouds_particles_porosity_factor=clouds_particles_porosity_factor,
                cloud_species=cloud_species
            )

            # Opacities with all clouds
            (cloud_continuum_opacities, cloud_continuum_opacities_scattering,
             cloud_scattering_opacities, cloud_absorption_opacities,
             cloud_opacities,  # scattering + absorption, used for cloud contribution
             _cloud_particles_mean_radii) = Radtrans._compute_cloud_opacities(
                pressures=pressures,
                temperatures=temperatures,
                frequency_bins_edges=frequency_bins_edges,
                cloud_species_mass_fractions=cloud_species_mass_fractions,
                mean_molar_masses=mean_molar_masses,
                reference_gravity=reference_gravity,
                cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
                clouds_loaded_opacities=clouds_loaded_opacities,
                sum_opacities=sum_opacities,
                anisotropic_cloud_scattering=anisotropic_cloud_scattering,
                cloud_f_sed=cloud_f_sed,
                eddy_diffusion_coefficients=eddy_diffusion_coefficients,
                cloud_particles_mean_radii=cloud_particles_mean_radii,
                cloud_particles_radius_distribution=cloud_particles_radius_distribution,
                cloud_hansen_a=cloud_hansen_a,
                cloud_hansen_b=cloud_hansen_b,
                cloud_particle_number_density_grid=cloud_particle_number_density_grid,
                clouds_particles_porosity_factor=clouds_particles_porosity_factor,
                photospheric_cloud_optical_depths=cloud_photosphere_median_optical_depth,
                continuum_opacities=cloud_continuum_opacities,
                return_cloud_contribution=return_cloud_contribution,
                target_particle_radii=target_particle_radii,
                target_pressure=target_pressure
            )

            # Opacities with only the clouds with complete coverage (will be treated as standard continuum opacities)
            if len(complete_coverage_clouds) > 0:
                clear_cloud_species_mass_fractions = copy.deepcopy(cloud_species_mass_fractions)

                for species in clear_cloud_species_mass_fractions:
                    if species not in complete_coverage_clouds:
                        clear_cloud_species_mass_fractions[species] *= 0

                cloud_continuum_opacities = [cloud_continuum_opacities, np.zeros((frequencies.size, pressures.size))]
                cloud_continuum_opacities_scattering = [cloud_continuum_opacities_scattering, None]
                cloud_scattering_opacities = [cloud_scattering_opacities, None]
                cloud_absorption_opacities = [cloud_absorption_opacities, None]
                cloud_opacities = [cloud_opacities, None]

                if opaque_cloud_top_pressure is not None:
                    cloud_continuum_opacities[1][:, pressures > opaque_cloud_top_pressure * 1e6] += 1e99  # TODO why '+=' and not '='?  # noqa E501

                (cloud_continuum_opacities[1], cloud_continuum_opacities_scattering[1],
                 cloud_scattering_opacities[1], cloud_absorption_opacities[1],
                 _, _) = Radtrans._compute_cloud_opacities(
                    pressures=pressures,
                    temperatures=temperatures,
                    frequency_bins_edges=frequency_bins_edges,
                    cloud_species_mass_fractions=clear_cloud_species_mass_fractions,
                    mean_molar_masses=mean_molar_masses,
                    reference_gravity=reference_gravity,
                    cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
                    clouds_loaded_opacities=clouds_loaded_opacities,
                    sum_opacities=sum_opacities,
                    anisotropic_cloud_scattering=anisotropic_cloud_scattering,
                    cloud_f_sed=cloud_f_sed,
                    eddy_diffusion_coefficients=eddy_diffusion_coefficients,
                    cloud_particles_mean_radii=cloud_particles_mean_radii,
                    cloud_particles_radius_distribution=cloud_particles_radius_distribution,
                    cloud_hansen_a=cloud_hansen_a,
                    cloud_hansen_b=cloud_hansen_b,
                    cloud_particle_number_density_grid=cloud_particle_number_density_grid,
                    clouds_particles_porosity_factor=clouds_particles_porosity_factor,
                    photospheric_cloud_optical_depths=cloud_photosphere_median_optical_depth,
                    continuum_opacities=cloud_continuum_opacities[1],
                    return_cloud_contribution=return_cloud_contribution,
                    target_particle_radii=target_particle_radii,
                    target_pressure=target_pressure
                )

        return (
            cloud_continuum_opacities, cloud_continuum_opacities_scattering,
            cloud_scattering_opacities, cloud_absorption_opacities,
            cloud_opacities, _cloud_particles_mean_radii,
            scattering_in_transmission
        )

    @staticmethod
    def __compute_continuum_opacities(
        frequencies, frequency_bins_edges,
        pressures, temperatures, mass_fractions, mean_molar_masses,
        cias_loaded_opacities, gas_continuum_contributors, gray_opacity
    ):
        # Reset
        continuum_opacities = np.zeros((frequencies.size, pressures.size), dtype='d', order='F')

        # Add CIA opacities
        continuum_opacities += (
            Radtrans._compute_cia_opacities(
                cia_dicts=cias_loaded_opacities,
                mass_fractions=mass_fractions,
                pressures=pressures,
                temperatures=temperatures,
                frequencies=frequencies,
                mean_molar_masses=mean_molar_masses
            )
        )

        # Add other gas continuum opacities
        continuum_opacities += (
            Radtrans._compute_non_cia_gas_induced_continuum_opacities(
                gas_continuum_contributors=gas_continuum_contributors,
                mass_fractions=mass_fractions,
                pressures=pressures,
                temperatures=temperatures,
                frequencies=frequencies,
                frequency_bins_edges=frequency_bins_edges,
                mean_molar_masses=mean_molar_masses
            )
        )

        # Add gray opacity
        if gray_opacity is not None:
            continuum_opacities += gray_opacity

        return continuum_opacities

    @staticmethod
    def __compute_continuum_scattering_opacities(
        frequencies,
        pressures, temperatures, mass_fractions, mean_molar_masses,
        rayleigh_species, haze_factor
    ):
        # Reset
        continuum_opacities_scattering = np.zeros((frequencies.size, pressures.size), dtype='d', order='F')
        scattering_in_transmission = False

        # Add rayleigh scattering opacities
        if len(rayleigh_species) > 0:
            scattering_in_transmission = True

            continuum_opacities_scattering += (
                Radtrans._compute_rayleigh_scattering_opacities(
                    rayleigh_species=rayleigh_species,
                    pressures=pressures,
                    temperatures=temperatures,
                    mass_fractions=mass_fractions,
                    mean_molar_masses=mean_molar_masses,
                    frequencies=frequencies,
                    haze_factor=haze_factor
                )
            )

        return continuum_opacities_scattering, scattering_in_transmission

    @staticmethod
    def __compute_combined_opacities(
        opacities, continuum_opacities,
        line_species, line_species_mass_fractions, line_opacity_mode,
        sum_opacities, g_gauss, weights_gauss
    ):
        # Combine line opacities with continuum opacities, apply line opacity mass fraction weighting
        opacities = Radtrans._combine_opacities(
            line_species_mass_fractions=line_species_mass_fractions,
            opacities=opacities,
            continuum_opacities=continuum_opacities  # added to the first species slot
        )

        if line_opacity_mode == 'c-k' and sum_opacities:
            # Similar to the line-by-line case below, if sum_opacities is True, we will put the total opacity
            # into the first species slot and then carry the remaining radiative transfer steps only over that 0 index
            # In c-k mode, sum_opacities is True only in emission if scattering_in_emission is True
            opacities = Radtrans._combine_ck_opacities(
                opacities=opacities,
                g_gauss=g_gauss,
                weights_gauss=weights_gauss
            )
        elif line_opacity_mode == 'lbl' and len(line_species) > 1:
            # In the line-by-line case we can simply add the opacities of different species in frequency space
            # All opacities are stored in the first species index slot
            opacities[:, :, 0, :] = np.sum(opacities, axis=2)

        return opacities

    @staticmethod
    def __get_base_species_mass_fractions(mass_fractions):
        base_species_mass_fractions = {}

        for species, mass_fraction in mass_fractions.items():
            species_basename = Opacity.get_species_base_name(species)

            if species_basename not in base_species_mass_fractions:
                base_species_mass_fractions[species_basename] = copy.deepcopy(mass_fraction)
            else:
                base_species_mass_fractions[species_basename] += mass_fraction

        return base_species_mass_fractions

    @staticmethod
    def __get_cia_contributors(gas_continuum_contributors):
        cia = []

        for gas_continuum_contributor in gas_continuum_contributors:
            if gas_continuum_contributor in Radtrans.__get_non_cia_gas_continuum_contributions():
                continue

            cia.append(gas_continuum_contributor)

        return cia

    @staticmethod
    def __get_base_opacities(
        use_smart_clear_opacities, complete_coverage_clouds,
        opacities, continuum_opacities_scattering,
        cloud_anisotropic_scattering_opacities=None, cloud_absorption_opacities=None
    ):
        """Return the opacities that should be returned when not using cloud coverage fraction."""
        if use_smart_clear_opacities:  # return the cloudy opacities
            # Sanity checks for later clear/cloudy swaps
            if continuum_opacities_scattering.ndim != 3:
                raise ValueError(
                    f"when using cloud coverage fractions, 'continuum_opacities_scattering' must have 3 dimensions, "
                    f"but had {continuum_opacities_scattering.ndim}\n"
                    f"This is likely a programming error and not a user error."
                )

            if not isinstance(opacities, list):
                raise ValueError(
                    f"when using cloud coverage fractions, 'opacities' must be a list, "
                    f"but was {type(opacities)}\n"
                    f"This is likely a programming error and not a user error."
                )

            # Use cloudy values
            _continuum_opacities_scattering = continuum_opacities_scattering[0]
            _opacities = opacities[0]
        else:  # return the cloudy (cloud coverage fraction = 1) or the cloudless (= 0) opacities
            _continuum_opacities_scattering = continuum_opacities_scattering
            _opacities = opacities

        _cloud_anisotropic_scattering_opacities = None
        _cloud_absorption_opacities = None

        if cloud_anisotropic_scattering_opacities is not None and cloud_absorption_opacities is not None:
            if len(complete_coverage_clouds) > 0:  # return the opacities with all the clouds
                _cloud_anisotropic_scattering_opacities = cloud_anisotropic_scattering_opacities[0]
                _cloud_absorption_opacities = cloud_absorption_opacities[0]
        elif cloud_anisotropic_scattering_opacities is None and cloud_absorption_opacities is None:
            pass
        else:
            raise ValueError(
                f"cloud_anisotropic_scattering_opacities and cloud_absorption_opacities must either both be not None "
                f"or both be None, "
                f"but were {cloud_anisotropic_scattering_opacities} and {cloud_absorption_opacities}\n"
                f"This is likely a programming error and not a user error."
            )

        return (
            _opacities, _continuum_opacities_scattering,
            _cloud_anisotropic_scattering_opacities, _cloud_absorption_opacities
        )

    @staticmethod
    def __get_cia_aliases(name: str) -> str:
        # TODO remove in version 4.0.0
        # Try to match name with aliases
        cia_aliases = {
            'H2--H2': ['H2H2', 'H2-H2', 'H2--H2'],
            'H2--He': ['H2He', 'H2-He', 'H2--He', 'HeH2', 'He-H2', 'He--H2'],
            'H2O--H2O': ['H2OH2O', 'H2O-H2O', 'H2O--H2O'],
            'H2O--N2': ['H2ON2', 'H2O-N2', 'H2O--N2', 'N2H2O', 'N2-H2O', 'N2--H2O'],
            'N2--H2': ['N2H2', 'N2-H2', 'N2--H2', 'H2N2', 'H2-N2', 'H2--N2'],
            'N2--He': ['N2He', 'N2-He', 'N2--He', 'HeN2', 'He-N2', 'He--N2'],
            'N2--N2': ['N2N2', 'N2-N2', 'N2--N2'],
            'O2--O2': ['O2O2', 'O2-O2', 'O2--O2'],
            'N2--O2': ['N2O2', 'N2-O2', 'N2--O2', 'O2N2', 'O2-N2', 'O2--N2'],
            'CO2--CO2': ['CO2CO2', 'CO2-CO2', 'CO2--CO2']
        }

        for cia, aliases in cia_aliases.items():
            if name in aliases:
                if name != cia:
                    warnings.warn(
                        "usage of CIA aliases is deprecated and will be removed in a future update.\n"
                        "Use the CIA actual name, with pRT's CIA separator ('--') between colliding species, instead",
                        FutureWarning
                    )

                return cia

        # Name does not match a directory, a directory shortcut, or an alias
        return name

    @staticmethod
    def __get_frequency_grids_intersection_indices(
        frequencies: npt.NDArray[np.floating],
        file_frequencies: npt.NDArray[np.floating],
        tolerate_grid_misalignment: bool = False
    ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
        """Get the indices to fill frequencies from a file within the Radtrans frequency grid.

        For example, if the Radtrans frequency grid is [0.1, ..., 0.3, ..., 3] and some loaded opacity frequency grid
        is in the interval [0.3, ..., 3, ..., 28], then the output indices will be the indices corresponding to
        [0.3, ..., 3] on the Radtrans frequency grid, and on the file frequency grid.

        Args:
            frequencies: the Radtrans frequency grid
            file_frequencies: the file frequency grid
            tolerate_grid_misalignment: if True, do not raise an error if the intersection indices sizes differ by at
            most 1.

        Returns:
            The indices of the intersection between both grids, for the Radtrans grid and the file grid0
        """
        indices_opacities, indices_file = intersection_indices(
            array1=frequencies,
            array2=file_frequencies
        )

        if np.nonzero(indices_opacities)[0].size != np.nonzero(indices_file)[0].size:
            if (
                not tolerate_grid_misalignment
                or np.abs(np.nonzero(indices_opacities)[0].size - np.nonzero(indices_file)[0].size) != 1
            ):
                raise ValueError(
                    f"frequencies size mismatch: value frequency array of size {np.nonzero(indices_opacities)[0].size} "
                    f"cannot be broadcast to indexing result of size {np.nonzero(indices_file)[0].size}\n"
                    f"This may be caused by loading opacities of different resolving power")

        return indices_opacities, indices_file

    @staticmethod
    def __get_line_opacity_file(path_input_data, species, category):
        hdf5_file = None

        # Handle bin-down of correlated-k opacities
        if category == CorrelatedKOpacity.get_default_category():
            # Get the targeted resolving power
            _species, _ = Opacity.split_species_spectral_info(species)
            target_resolving_power, range_filename = Opacity.find_spectral_information(species)

            if 'R' in target_resolving_power:
                target_resolving_power = CorrelatedKOpacity.get_resolving_power_from_string(target_resolving_power)

            if target_resolving_power == '':
                target_resolving_power = int(CorrelatedKOpacity.get_default_resolving_power())

            # Bin down the opacities if necessary
            if target_resolving_power < int(CorrelatedKOpacity.get_default_resolving_power()):
                # Try to find the binned-down file first
                _hdf5_file = CorrelatedKOpacity.find(
                    path_input_data=path_input_data,
                    category=category,
                    species=species,
                    find_all=True,  # output an empty list if no matching file found
                    search_online=False  # don't search online since this is not the standard resolving power
                )

                if len(_hdf5_file) == 0:  # no matching file found, make the file using exo_k, then find it
                    print("Binned-down file not found, starting rebinning...")

                    # Rebin from the file with the default resolving power, which should always exist
                    default_spectral_sampling = CorrelatedKOpacity.get_resolving_power_string(
                        resolving_power=int(CorrelatedKOpacity.get_default_resolving_power())
                    )

                    # Build a "default" species using the provided information and the default spectral sampling
                    default_species = Opacity.join_species_all_info(
                        species_name=_species,  # contains already all non-spectral info
                        natural_abundance='',
                        charge='',
                        cloud_info='',
                        source='',
                        spectral_info='',
                        spectral_sampling=default_spectral_sampling,
                        wavelength_range=range_filename
                    )

                    # Find the HDF5 file of the species with the default spectral sampling
                    print(
                        f"Searching for the file of species '{_species}' "
                        f"with default spectral sampling ({default_spectral_sampling})...",
                    )
                    default_hdf5_file = CorrelatedKOpacity.find(
                        path_input_data=path_input_data,
                        category=category,
                        species=default_species,
                        find_all=False,
                        search_online=True
                    )
                    print(
                        " File found.",
                    )

                    state = rebin_ck_line_opacities(
                        input_file=default_hdf5_file,
                        target_resolving_power=target_resolving_power,
                        wavenumber_grid=None,
                        rewrite=False
                    )

                    if state == -1:
                        raise RuntimeError("unable to perform binning down, please install exo_k")

                    hdf5_file = None
                elif isinstance(_hdf5_file, str):  # exactly one matching file found, use it
                    hdf5_file = _hdf5_file
                else:  # more than one file exists, try to find the correct file
                    hdf5_file = None

        if hdf5_file is None:
            hdf5_file = Opacity.find(
                path_input_data=path_input_data,
                category=category,
                species=species,
                find_all=False,
                search_online=True
            )

        return hdf5_file

    @staticmethod
    def __get_non_cia_gas_continuum_contributions():
        return {
            'H-': Radtrans._compute_h_minus_opacities
        }

    @staticmethod
    def __get_opacity_sources_dict():
        return LockedDict.build_and_lock({
            'init': LockedDict.build_and_lock({
                'line_species': [],
                'gas_continuum_contributors': [],
                'rayleigh_species': [],
                'cloud_species': []
            }),
            'runtime': LockedDict.build_and_lock({
                'opaque_cloud_top_pressure': None,
                'power_law': LockedDict.build_and_lock({
                    'power_law_opacity_350nm': None,
                    'power_law_opacity_coefficient': None,
                }),
                'gray_opacity': None,
                'cloud_photosphere_median_optical_depth': None,
                'additional_absorption_opacities_function': None,
                'additional_scattering_opacities_function': None,
                'stellar_intensities': None
            })
        })

    @staticmethod
    def __init_clouds_particles_porosity_factor(clouds_particles_porosity_factor, cloud_species):
        if clouds_particles_porosity_factor is None:
            clouds_particles_porosity_factor = {species: -np.inf for species in cloud_species}

        for species in cloud_species:
            _, _, _, _, source, _ = Opacity.split_species_all_info(CloudOpacity.get_aliases(species))

            if source == 'DHS':
                if clouds_particles_porosity_factor[species] == 0:
                    warnings.warn(f"cloud species '{species}' opacity calculation method is using the DHS method, "
                                  f"the porosity factor should be > 0 (0.25 is used by default)")
                elif not np.isfinite(clouds_particles_porosity_factor[species]):
                    clouds_particles_porosity_factor[species] = 0.25
            else:
                if not np.isfinite(clouds_particles_porosity_factor[species]):
                    clouds_particles_porosity_factor[species] = 0

            if clouds_particles_porosity_factor[species] < 0 or clouds_particles_porosity_factor[species] >= 1:
                raise ValueError(f"clouds particles porosity factor "
                                 f"must be >= 0 (full particles) and < 1 (empty particles), "
                                 f"but is {clouds_particles_porosity_factor[species]} for species '{species}'")

        return clouds_particles_porosity_factor

    def __init_spectral_function(
        self, is_emission, reference_gravity, complete_coverage_clouds, return_clear_spectrum,
        mass_fractions, cloud_fraction, opaque_cloud_top_pressure
    ):
        """Initializations and checks common to calculate_flux and calculate_transit_radii."""
        if reference_gravity <= 0:
            raise ValueError(f"reference gravity must be > 0, but was {reference_gravity}")

        if complete_coverage_clouds is None:
            complete_coverage_clouds = []

        spectrum_clear = None

        self.__check_return_clear_spectrum_relevance(
            return_clear_spectrum=return_clear_spectrum,
            clouds_have_effect=self.__clouds_have_effect(mass_fractions, cloud_fraction),
            opaque_cloud_top_pressure=opaque_cloud_top_pressure,
            cloud_fraction=cloud_fraction,
        )

        self.__set_sum_opacities(emission=is_emission)
        self.__set_use_smart_clear_opacities(
            cloud_fraction=cloud_fraction,
            complete_coverage_clouds=complete_coverage_clouds,
            return_clear_spectrum=return_clear_spectrum
        )

        auto_anisotropic_cloud_scattering = False

        if self._anisotropic_cloud_scattering == 'auto':
            self._anisotropic_cloud_scattering = is_emission
            auto_anisotropic_cloud_scattering = True
        elif not self._anisotropic_cloud_scattering and is_emission:
            warnings.warn(f"anisotropic cloud scattering is recommended for emission spectra, "
                          f"but 'anisotropic_cloud_scattering' was set to {self._anisotropic_cloud_scattering}; "
                          f"set it to True or 'auto' to disable this warning")
        elif self._anisotropic_cloud_scattering and not is_emission:
            warnings.warn(f"anisotropic cloud scattering is not recommended for transmission spectra, "
                          f"but 'anisotropic_cloud_scattering' was set to {self._anisotropic_cloud_scattering}; "
                          f"set it to False or 'auto' to disable this warning")

        return auto_anisotropic_cloud_scattering, complete_coverage_clouds, spectrum_clear

    def __set_use_smart_clear_opacities(self, cloud_fraction, complete_coverage_clouds, return_clear_spectrum):
        self.__use_smart_clear_opacities = False

        if 0 < cloud_fraction < 1 or len(complete_coverage_clouds) > 0 or return_clear_spectrum:
            self.__use_smart_clear_opacities = True

    def __set_sum_opacities(self, emission):
        self.__sum_opacities = False  # False in c-k emission if no scattering, and always False in c-k transmission

        if self._line_opacity_mode == 'c-k':
            if emission and self._scattering_in_emission:
                self.__sum_opacities = True
        elif self._line_opacity_mode == 'lbl':  # always True in lbl mode
            self.__sum_opacities = True

    def _calculate_flux(
        self, temperatures, reference_gravity, opacities, continuum_opacities_scattering,
        emission_geometry, star_irradiation_cos_angle, stellar_intensity, reflectances, emissivities,
        cloud_f_sed,
        photospheric_cloud_optical_depths, cloud_photosphere_wavelength_boundaries,
        cloud_anisotropic_scattering_opacities, cloud_absorption_opacities,
        adaptive_feautrier_iterations, return_contribution=False, return_rosseland_opacities=False
    ):
        """Calculate the flux.
        TODO complete docstring

        Args:
            temperatures:
            reference_gravity:
            opacities:
            continuum_opacities_scattering:
            emission_geometry:
            star_irradiation_cos_angle:
            stellar_intensity:
            reflectances:
            emissivities:
            cloud_f_sed:
            photospheric_cloud_optical_depths:
            cloud_photosphere_wavelength_boundaries:
            cloud_anisotropic_scattering_opacities:
            cloud_absorption_opacities:
            adaptive_feautrier_iterations:
            return_contribution:
            return_rosseland_opacities:

        Returns:

        """
        optical_depths, photon_destruction_probabilities, relative_cloud_scaling_factor = (
            self._compute_optical_depths_wrapper(
                pressures=self._pressures,
                reference_gravity=reference_gravity,
                opacities=opacities,
                continuum_opacities_scattering=continuum_opacities_scattering,
                sum_opacities=self.__sum_opacities,
                scattering_in_emission=self._scattering_in_emission,
                absorber_present=self.__absorber_present,
                # Custom cloud parameters
                frequencies=self._frequencies,
                weights_gauss=self._lines_loaded_opacities['weights_gauss'],
                cloud_wavelengths=cloud_photosphere_wavelength_boundaries,
                cloud_f_sed=cloud_f_sed,
                cloud_anisotropic_scattering_opacities=cloud_anisotropic_scattering_opacities,
                cloud_absorption_opacities=cloud_absorption_opacities,
                photospheric_cloud_optical_depths=photospheric_cloud_optical_depths
            )
        )

        opacities_rosseland = None

        if self._scattering_in_emission:
            # TODO investigate bug with scattering and low VMR near surface
            # Only use 0 index for species because for lbl or test_ck_shuffle_comp = True
            # everything has been moved into the 0th index
            flux, emission_contribution = self._compute_feautrier_radiative_transfer(
                frequency_bins_edges=self._frequency_bins_edges,
                temperatures=temperatures,
                weights_gauss=self._lines_loaded_opacities['weights_gauss'],
                emission_cos_angle_grid=self._emission_angle_grid['cos_angles'],
                emission_cos_angle_grid_weights=self._emission_angle_grid['weights'],
                optical_depths=optical_depths[:, :, 0, :],
                photon_destruction_probabilities=photon_destruction_probabilities,
                emission_geometry=emission_geometry,
                stellar_intensity=stellar_intensity,
                star_irradiation_cos_angle=star_irradiation_cos_angle,
                reflectances=reflectances,
                emissivities=emissivities,
                adaptive_feautrier_iterations=adaptive_feautrier_iterations,
                return_contribution=return_contribution
            )

            if return_rosseland_opacities:
                opacities_rosseland = \
                    self._compute_rosseland_opacities(
                        frequency_bins_edges=self._frequency_bins_edges,
                        temperatures=temperatures,
                        weights_gauss=self._lines_loaded_opacities['weights_gauss'],
                        opacities=opacities[:, :, 0, :],
                        continuum_opacities_scattering=continuum_opacities_scattering,
                        scattering_in_emission=self._scattering_in_emission
                    )
        else:
            if self._line_opacity_mode == 'lbl' and len(self._line_species) > 1:
                _optical_depths = optical_depths[:, :, :1, :]
            else:
                _optical_depths = optical_depths.view()

            flux, emission_contribution = self._compute_ck_flux(
                frequencies=self._frequencies,
                temperatures=temperatures,
                weights_gauss=self._lines_loaded_opacities['weights_gauss'],
                emission_cos_angle_grid=self._emission_angle_grid['cos_angles'],
                emission_cos_angle_grid_weights=self._emission_angle_grid['weights'],
                optical_depths=_optical_depths,
                return_contribution=return_contribution
            )

        return flux, emission_contribution, optical_depths, opacities_rosseland, relative_cloud_scaling_factor

    def _calculate_opacities(
        self, temperatures, mass_fractions, mean_molar_masses, reference_gravity,
        opaque_cloud_top_pressure=None,
        cloud_particles_mean_radii=None, cloud_particle_radius_distribution_std=None,
        cloud_particles_radius_distribution="lognormal",
        cloud_particle_number_density_grid=None,
        cloud_hansen_a=None, cloud_hansen_b=None,
        clouds_particles_porosity_factor=None,
        cloud_f_sed=None, eddy_diffusion_coefficients=None,
        haze_factor=1.0, power_law_opacity_350nm=None, power_law_opacity_coefficient=None,
        gray_opacity=None, cloud_photosphere_median_optical_depth=None,
        cloud_fraction=1.0, complete_coverage_clouds=None,
        return_cloud_contribution=False,
        additional_absorption_opacities_function=None,
        additional_scattering_opacities_function=None,
        target_particle_radii=None,
        target_pressure=None
    ):
        """Combine total line opacities, according to mass fractions (abundances), also add continuum opacities,
        i.e. clouds, CIA...
        TODO complete docstring

        Args:
            temperatures:
            mass_fractions:
            mean_molar_masses:
            reference_gravity:
            cloud_particle_radius_distribution_std:
            cloud_f_sed:
            eddy_diffusion_coefficients:
            cloud_particles_mean_radii:
            cloud_particles_radius_distribution:
            cloud_hansen_a:
            cloud_hansen_b:
            cloud_particle_number_density_grid:
            return_cloud_contribution:
            additional_absorption_opacities_function:
            additional_scattering_opacities_function:

        Returns:

        """
        if np.any(np.less_equal(temperatures, 0)):
            warnings.warn(f"temperature array ('temperatures') contains negative or null values ({temperatures})")

        for species, mass_fraction in mass_fractions.items():
            if np.any(np.less(mass_fraction, 0)):
                warnings.warn(f"mass fractions for species '{species}' ('mass_fractions') "
                              f"contains negative values ({mass_fraction})")

        if np.any(np.less(mean_molar_masses, 0)):
            warnings.warn(f"mean molar mass array ('mean_molar_masses') contains negative values ({mean_molar_masses})")

        if cloud_fraction < 0 or cloud_fraction > 1:
            raise ValueError(f"cloud coverage fraction must be between 0 and 1, but was {cloud_fraction}")

        if complete_coverage_clouds is None:
            complete_coverage_clouds = []

        # Initialization
        if len(self._line_species) > 0:
            line_species_mass_fractions = np.zeros(
                (self._pressures.size, len(self._line_species)), dtype='d', order='F'
            )
        else:
            # If there are no specified line species then we need at least an array to contain the continuum opacities
            line_species_mass_fractions = np.zeros(
                (self._pressures.size, 1), dtype='d', order='F'
            )

        # Fill line mass fraction dictionary with provided mass fraction dictionary
        for i_spec in range(len(self._line_species)):
            # Check if user provided the detailed line absorber name or if line absorber name should be matched exactly
            if self._line_species[i_spec] in mass_fractions:
                line_species_mass_fractions[:, i_spec] = mass_fractions[self._line_species[i_spec]]
            else:
                default_species, spectral_info = Opacity.split_species_spectral_info(self._line_species[i_spec])

                if default_species in mass_fractions:
                    line_species_mass_fractions[:, i_spec] = mass_fractions[default_species]
                else:
                    raise ValueError(f"line species '{self._line_species[i_spec]}' not found in mass fractions dict "
                                     f"(listed species: {list(mass_fractions.keys())}")

        # Continuum opacities
        continuum_opacities = self.__compute_continuum_opacities(
            frequencies=self._frequencies,
            frequency_bins_edges=self._frequency_bins_edges,
            pressures=self._pressures,
            temperatures=temperatures,
            mass_fractions=mass_fractions,
            mean_molar_masses=mean_molar_masses,
            cias_loaded_opacities=self._cias_loaded_opacities,
            gas_continuum_contributors=self._gas_continuum_contributors,
            gray_opacity=gray_opacity
        )

        continuum_opacities_scattering, self.__scattering_in_transmission = (
            self.__compute_continuum_scattering_opacities(
                frequencies=self._frequencies,
                pressures=self._pressures,
                temperatures=temperatures,
                mass_fractions=mass_fractions,
                mean_molar_masses=mean_molar_masses,
                rayleigh_species=self._rayleigh_species,
                haze_factor=haze_factor
            )
        )

        # Cloud opacities
        (cloud_continuum_opacities, cloud_continuum_opacities_scattering,
         cloud_scattering_opacities, cloud_absorption_opacities,
         cloud_opacities, _cloud_particles_mean_radii,
         self.__scattering_in_transmission) = self.__compute_cloud_continuum_opacities(
            frequencies=self._frequencies,
            frequency_bins_edges=self._frequency_bins_edges,
            pressures=self._pressures,
            temperatures=temperatures,
            mass_fractions=mass_fractions,
            mean_molar_masses=mean_molar_masses,
            reference_gravity=reference_gravity,
            scattering_in_transmission=self.__scattering_in_transmission,
            sum_opacities=self.__sum_opacities,
            opaque_cloud_top_pressure=opaque_cloud_top_pressure,
            power_law_opacity_350nm=power_law_opacity_350nm,
            power_law_opacity_coefficient=power_law_opacity_coefficient,
            clouds_have_effect=self.__clouds_have_effect(mass_fractions, cloud_fraction),
            cloud_species=self._cloud_species,
            clouds_particles_porosity_factor=clouds_particles_porosity_factor,
            clouds_loaded_opacities=self._clouds_loaded_opacities,
            anisotropic_cloud_scattering=self._anisotropic_cloud_scattering,
            cloud_particles_mean_radii=cloud_particles_mean_radii,
            cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
            cloud_particle_number_density_grid=cloud_particle_number_density_grid,
            cloud_f_sed=cloud_f_sed,
            eddy_diffusion_coefficients=eddy_diffusion_coefficients,
            cloud_particles_radius_distribution=cloud_particles_radius_distribution,
            cloud_hansen_a=cloud_hansen_a,
            cloud_hansen_b=cloud_hansen_b,
            cloud_photosphere_median_optical_depth=cloud_photosphere_median_optical_depth,
            complete_coverage_clouds=complete_coverage_clouds,
            return_cloud_contribution=return_cloud_contribution,
            target_particle_radii=target_particle_radii,
            target_pressure=target_pressure
        )

        # Check if photospheric_cloud_optical_depths is used with a single cloud model
        # Combining cloud opacities from different models is currently not supported with the
        # photospheric_cloud_optical_depths parameter
        if len(self._cloud_species) > 0 and cloud_photosphere_median_optical_depth is not None:
            if (additional_absorption_opacities_function is not None
                    or additional_scattering_opacities_function is not None):
                raise ValueError(
                    "The photospheric_cloud_optical_depths can only be used in combination with "
                    "a single cloud model. "
                    "Either use a physical cloud model by choosing cloud_species "
                    "or use parametrized cloud opacities with "
                    "the give_absorption_opacity and give_scattering_opacity parameters."
                )

        # Additional opacities
        if not self.__clouds_have_effect(mass_fractions, cloud_fraction):
            (additional_absorption_opacities, additional_scattering_opacities,
             _cloud_absorption_opacities, _cloud_scattering_opacities) = self.__compute_additional_continuum_opacities(
                frequencies=self._frequencies,
                pressures=self._pressures,
                additional_absorption_opacities_function=additional_absorption_opacities_function,
                additional_scattering_opacities_function=additional_scattering_opacities_function,
                cloud_photosphere_median_optical_depth=cloud_photosphere_median_optical_depth
            )
        else:
            additional_absorption_opacities = None
            additional_scattering_opacities = None
            _cloud_absorption_opacities = None
            _cloud_scattering_opacities = None

        if len(complete_coverage_clouds) > 0:
            cloud_absorption_opacities[1] = _cloud_absorption_opacities
            cloud_scattering_opacities[1] = _cloud_scattering_opacities
        else:
            cloud_absorption_opacities = _cloud_absorption_opacities
            cloud_scattering_opacities = _cloud_scattering_opacities

        del _cloud_absorption_opacities
        del _cloud_scattering_opacities

        _opacities_first_species = None

        # Line opacities (calculating them just before calculating the combined opacities reduces peak memory usage)
        opacities = self._interpolate_species_opacities(
            pressures=self._pressures,
            temperatures=temperatures,
            n_g=self._lines_loaded_opacities['g_gauss'].size,
            n_frequencies=self._frequencies.size,
            line_opacities_grid=self._lines_loaded_opacities['opacity_grid'],
            line_opacities_temperature_pressure_grid=self._lines_loaded_opacities['temperature_pressure_grid'],
            line_opacities_temperature_grid_size=self._lines_loaded_opacities['temperature_grid_size'],
            line_opacities_pressure_grid_size=self._lines_loaded_opacities['pressure_grid_size']
        )

        # Add cloud opacities
        if cloud_fraction == 1 and not self.__use_smart_clear_opacities:
            continuum_opacities += cloud_continuum_opacities
            continuum_opacities_scattering += cloud_continuum_opacities_scattering

            if additional_absorption_opacities is not None:
                continuum_opacities += additional_absorption_opacities

            if additional_scattering_opacities is not None:
                continuum_opacities_scattering += additional_scattering_opacities
        elif self.__use_smart_clear_opacities:
            if len(complete_coverage_clouds) > 0:
                # Treat cloud opacities with complete coverage clouds as standard continuum opacities
                continuum_opacities += cloud_continuum_opacities[1]
                continuum_opacities_scattering += cloud_continuum_opacities_scattering[1]

                # The difference of all the clouds opacities with the opacities of the complete coverage clouds only
                # is the opacity of the clouds with partial coverage only
                cloud_continuum_opacities = (
                    cloud_continuum_opacities[0] - cloud_continuum_opacities[1]
                )
                cloud_continuum_opacities_scattering = (
                    cloud_continuum_opacities_scattering[0] - cloud_continuum_opacities_scattering[1]
                )

            # Store clear continuum_opacities_scattering in new axis
            continuum_opacities_scattering = continuum_opacities_scattering[np.newaxis]
            continuum_opacities_scattering = np.repeat(continuum_opacities_scattering, 2, axis=0)

            # Store the opacities of the first species
            if self.__sum_opacities:
                _opacities_first_species = copy.deepcopy(opacities[:, :, 0])

        # Combine line opacities with continuum opacities, combined opacities are stored in the 1st species slot
        opacities = self.__compute_combined_opacities(
            opacities=opacities,
            continuum_opacities=continuum_opacities,
            line_species=self._line_species,
            line_species_mass_fractions=line_species_mass_fractions,
            line_opacity_mode=self._line_opacity_mode,
            sum_opacities=self.__sum_opacities,
            g_gauss=self._lines_loaded_opacities['g_gauss'],
            weights_gauss=self._lines_loaded_opacities['weights_gauss']
        )

        if self.__use_smart_clear_opacities:
            # Store the clear opacities
            _opacities_clear = copy.deepcopy(opacities[:, :, 0])  # opacities of 1st species times MMR + clear continuum

            # Add cloud scattering opacities
            continuum_opacities_scattering[0] += cloud_continuum_opacities_scattering

            if additional_scattering_opacities is not None:
                continuum_opacities_scattering[0] += additional_scattering_opacities

            if self.__sum_opacities:
                # Set opacities values back to before combination
                opacities[:, :, 0] = _opacities_first_species
                opacities[:, :, 1:] /= line_species_mass_fractions.T[1:]

                del _opacities_first_species

                # Add cloud continuum opacities
                continuum_opacities += cloud_continuum_opacities

                if additional_absorption_opacities is not None:
                    continuum_opacities += additional_absorption_opacities

                # Re-calculate opacities in the cloudy case
                # Combined opacities in the cloudy case are stored in the 1st species slot
                # Combined opacities in the clear case are stored in the 2nd species slot
                opacities = self.__compute_combined_opacities(
                    opacities=opacities,
                    continuum_opacities=continuum_opacities,
                    line_species=self._line_species,
                    line_species_mass_fractions=line_species_mass_fractions,
                    line_opacity_mode=self._line_opacity_mode,
                    sum_opacities=self.__sum_opacities,
                    g_gauss=self._lines_loaded_opacities['g_gauss'],
                    weights_gauss=self._lines_loaded_opacities['weights_gauss']
                )
            else:
                if additional_absorption_opacities is not None:
                    cloud_continuum_opacities += additional_absorption_opacities

                opacities = np.swapaxes(opacities, 1, 2)  # (g, species, frequencies, layers)
                opacities[:, 0] += cloud_continuum_opacities
                opacities = np.swapaxes(opacities, 2, 1)  # (g, frequencies, species, layers)

            opacities = [opacities, _opacities_clear]

        del continuum_opacities  # delete unnecessary variable

        return (
            opacities, continuum_opacities_scattering,
            cloud_scattering_opacities, cloud_absorption_opacities, cloud_opacities,
            _cloud_particles_mean_radii
        )

    def _calculate_transit_radii(
        self, temperatures, mean_molar_masses, reference_gravity,
        reference_pressure, planet_radius, variable_gravity,
        opacities, continuum_opacities_scattering, copy_opacities, return_contributions
    ):
        """Calculate the transit radii.
        TODO complete docstring

        Args:
            temperatures:
            mean_molar_masses:
            reference_gravity:
            reference_pressure:
            planet_radius:
            variable_gravity:
            opacities:
            continuum_opacities_scattering:
            copy_opacities:

        Returns:

        """
        if copy_opacities:
            transmission_contribution = np.zeros(
                (np.size(self._pressures), self._frequencies.size), dtype='d', order='F'
            )
        else:
            transmission_contribution = None

        # Calculate the transmission spectrum
        if self.__sum_opacities and len(self._line_species) > 1:
            transit_radii, radius_hydrostatic_equilibrium = self._compute_transit_radii(
                opacities=opacities,
                continuum_opacities_scattering=continuum_opacities_scattering,
                pressures=self._pressures * 1e-6,  # cgs to bar
                temperatures=temperatures,
                weights_gauss=self._lines_loaded_opacities['weights_gauss'],
                mean_molar_masses=mean_molar_masses,
                reference_gravity=reference_gravity,
                reference_pressure=reference_pressure,
                planet_radius=planet_radius,
                variable_gravity=variable_gravity,
                summed_species_opacities=True,
                copy_opacities=copy_opacities
            )

            # TODO: contribution function calculation with python-only implementation
            if return_contributions:
                transmission_contribution, radius_hydrostatic_equilibrium = (
                    self._compute_transmission_spectrum_contribution(
                        transit_radii=transit_radii,
                        opacities=opacities[:, :, :1, :],
                        temperatures=temperatures,
                        pressures=self._pressures,
                        reference_gravity=reference_gravity,
                        mean_molar_masses=mean_molar_masses,
                        reference_pressure=reference_pressure,
                        planet_radius=planet_radius,
                        weights_gauss=self._lines_loaded_opacities['weights_gauss'],
                        scattering_in_transmission=self.__scattering_in_transmission,
                        continuum_opacities_scattering=continuum_opacities_scattering,
                        variable_gravity=variable_gravity
                    )
                )
        else:
            transit_radii, radius_hydrostatic_equilibrium = self._compute_transit_radii(
                opacities=opacities,
                continuum_opacities_scattering=continuum_opacities_scattering,
                pressures=self._pressures * 1e-6,  # cgs to bar
                temperatures=temperatures,
                weights_gauss=self._lines_loaded_opacities['weights_gauss'],
                mean_molar_masses=mean_molar_masses,
                reference_gravity=reference_gravity,
                reference_pressure=reference_pressure,
                planet_radius=planet_radius,
                variable_gravity=variable_gravity,
                summed_species_opacities=False,
                copy_opacities=copy_opacities
            )

            # TODO: contribution function calculation with python-only implementation
            if return_contributions:
                transmission_contribution, radius_hydrostatic_equilibrium = (
                    self._compute_transmission_spectrum_contribution(
                        transit_radii=transit_radii,
                        opacities=opacities,
                        temperatures=temperatures,
                        pressures=self._pressures,
                        reference_gravity=reference_gravity,
                        mean_molar_masses=mean_molar_masses,
                        reference_pressure=reference_pressure,
                        planet_radius=planet_radius,
                        weights_gauss=self._lines_loaded_opacities['weights_gauss'],
                        scattering_in_transmission=self.__scattering_in_transmission,
                        continuum_opacities_scattering=continuum_opacities_scattering,
                        variable_gravity=variable_gravity
                    )
                )

        return transit_radii, radius_hydrostatic_equilibrium, transmission_contribution

    @staticmethod
    def _combine_ck_opacities(opacities, g_gauss, weights_gauss):
        opacities[:, :, 0, :] = fcore.combine_ck_opacities(
            opacities,
            g_gauss,
            weights_gauss
        )

        return opacities

    @staticmethod
    def _combine_opacities(line_species_mass_fractions, opacities, continuum_opacities):
        opacities *= line_species_mass_fractions.T  # (g, frequencies, species, layers)
        opacities = np.swapaxes(opacities, 1, 2)  # (g, species, frequencies, layers)
        opacities[:, 0] += continuum_opacities  # quirkiness: put total opacities in the opacities of species 0
        opacities = np.swapaxes(opacities, 2, 1)  # back to default shape (g, frequencies, species, layers)

        return opacities

    @staticmethod
    def _compute_cia_opacities(cia_dicts, mass_fractions, pressures, temperatures, frequencies, mean_molar_masses):
        """Wrapper to _interpolate_cia, calculating each collision's combined mass fraction."""
        # Add mass fractions sharing the same basename
        base_species_mass_fractions = Radtrans.__get_base_species_mass_fractions(mass_fractions)

        # Calculate CIA opacities
        cia_opacities = 0

        for collision, collision_dict in cia_dicts.items():
            # Calculate combined mass fraction
            combined_mass_fraction = 1

            for collision_species in collision_dict['molecules']:
                if collision_species in base_species_mass_fractions:
                    combined_mass_fraction = combined_mass_fraction * base_species_mass_fractions[collision_species]
                else:
                    raise ValueError(
                        f"species '{collision_species}' of CIA '{collision}' "
                        f"not found in mass fractions dict "
                        f"(listed species: {list(mass_fractions.keys())}, "
                        f"listed species base names {list(base_species_mass_fractions.keys())}:)"
                    )

            # Add CIA opacities
            cia_opacities += (
                Radtrans._interpolate_cia(
                    collision_dict=collision_dict,
                    combined_mass_fractions=combined_mass_fraction,
                    pressures=pressures,
                    temperatures=temperatures,
                    frequencies=frequencies,
                    mean_molar_masses=mean_molar_masses
                )
            )

        return cia_opacities

    @staticmethod
    def _compute_ck_flux(
        frequencies, temperatures, weights_gauss,
        emission_cos_angle_grid, emission_cos_angle_grid_weights, optical_depths, return_contribution
    ):
        flux, emission_contribution = fcore.compute_ck_flux(
            frequencies,
            temperatures,
            weights_gauss,
            emission_cos_angle_grid,
            emission_cos_angle_grid_weights,
            optical_depths,
            return_contribution
        )

        return flux, emission_contribution

    @staticmethod
    def _integrate_cloud_opacities_from_dn_dr(
            dn_dr,
            atmosphere_densities,
            clouds_particles_densities,
            cloud_particles_radii_bins,
            cloud_particles_radii,
            clouds_absorption_opacities,
            clouds_scattering_opacities,
            clouds_particles_asymmetry_parameters,
            pressures,
            target_particle_radii,
            target_pressure,
    ):
        """
        Shared integration backend for both cloud opacity methods.
        Operates purely on dn/dr and handles all integration logic.
        """

        # Scale factor
        integrand_scale = (
                (4.0 * np.pi / 3.0)
                * cloud_particles_radii[:, None, None] ** 3
                * clouds_particles_densities
                * dn_dr
        )

        # Opacities
        integrand_absorption = integrand_scale[:, None] * clouds_absorption_opacities[:, :, None]
        integrand_scattering = integrand_scale[:, None] * clouds_scattering_opacities[:, :, None]
        integrand_anisotropy = integrand_scattering * (1.0 - clouds_particles_asymmetry_parameters[:, :, None])

        # Selectively disable bins
        if target_pressure is not None:
            target_pressure = [tp * 1e6 for tp in target_pressure]  # bar→cgs
            target_particle_radii = [tr * 1e-4 for tr in target_particle_radii]  # micron→cm

            disabled_bins = {(np.abs(cloud_particles_radii_bins - tr)).argmin()
                             for tr in target_particle_radii}
            in_layer = {(np.abs(pressures - tp)).argmin()
                        for tp in target_pressure}

            block = np.ones_like(integrand_scale)
            for bi in disabled_bins:
                for li in in_layer:
                    block[bi, li, :] = 0

            integrand_scale *= block
            integrand_absorption *= block[:, None]
            integrand_scattering *= block[:, None]
            integrand_anisotropy *= block[:, None]

        # Integrate over radius
        widths = np.diff(cloud_particles_radii_bins)[:, None, None, None]

        clouds_absorption_opacities = np.sum(integrand_absorption * widths, axis=(0, 3))
        clouds_scattering_opacities = np.sum(integrand_scattering * widths, axis=(0, 3))
        cloud_one_minus_g = np.sum(integrand_anisotropy * widths, axis=(0, 3))

        cloud_one_minus_g = np.true_divide(
            cloud_one_minus_g, clouds_scattering_opacities,
            out=np.zeros_like(clouds_scattering_opacities),
            where=clouds_scattering_opacities > 1e-200
        )

        # Divide by atmosphere density
        clouds_absorption_opacities /= atmosphere_densities
        clouds_scattering_opacities /= atmosphere_densities

        return clouds_absorption_opacities, clouds_scattering_opacities, cloud_one_minus_g

    @staticmethod
    def _compute_cloud_log_normal_particles_distribution_opacities(
        atmosphere_densities,
        clouds_particles_densities,
        clouds_mass_fractions,
        cloud_particles_mean_radii,
        cloud_particles_distribution_std,
        cloud_particles_radii_bins,
        cloud_particles_radii,
        clouds_absorption_opacities,
        clouds_scattering_opacities,
        clouds_particles_asymmetry_parameters,
        pressures,
        target_particle_radii,
        target_pressure
    ):
        r"""This function reimplements calc_cloud_opas from fortran_radtrans_core.f90.
        For some reason it runs faster in python than in fortran, so we'll use this from now on.
        This function integrates the cloud opacity through the different layers of the atmosphere to get the total
        optical depth, scattering and anisotropic fraction.
        # TODO optical depth or opacity?
        # TODO complete docstring and vectorize on/off code

        Author: Francois Rozet

        Args:
            atmosphere_densities:
                Density of the atmosphere at each of its layer
            clouds_particles_densities:
                Density of each cloud particles
            clouds_mass_fractions:
                Mass fractions of each cloud at each atmospheric layer
            cloud_particles_mean_radii:
                Mean radius of each cloud particles at each atmospheric layer
            cloud_particles_distribution_std:
                Standard deviation of the log-normal cloud particles distribution
            cloud_particles_radii_bins:
                Bins of the particles cloud radii grid
            cloud_particles_radii:
                Particles cloud radii grid
            clouds_absorption_opacities:
                Cloud absorption opacities (radius grid, wavelength grid, clouds)
            clouds_scattering_opacities:
                Cloud scattering opacities (radius grid, wavelength grid, clouds)
            clouds_particles_asymmetry_parameters:
                Cloud particles asymmetry parameters (radius grid, wavelength grid, clouds)

        Returns:

        """
        # Compute dn/dr from log-normal distribution
        n = (
                3.0 * clouds_mass_fractions * atmosphere_densities[:, None] /
                (4.0 * np.pi * clouds_particles_densities * cloud_particles_mean_radii ** 3)
                * np.exp(-4.5 * np.log(cloud_particles_distribution_std) ** 2)
        )

        diff = np.log(cloud_particles_radii[:, None, None]) - np.log(cloud_particles_mean_radii)

        dn_dr = (
                n / (cloud_particles_radii[:, None, None] * np.sqrt(2.0 * np.pi)
                     * np.log(cloud_particles_distribution_std))
                * np.exp(-diff ** 2 / (2.0 * np.log(cloud_particles_distribution_std) ** 2))
                )

        # Delegate to shared integrator
        _cloud_absorption_opacities, _cloud_scattering_opacities, cloud_one_minus_g =\
            Radtrans._integrate_cloud_opacities_from_dn_dr(dn_dr,
                                                           atmosphere_densities,
                                                           clouds_particles_densities,
                                                           cloud_particles_radii_bins,
                                                           cloud_particles_radii,
                                                           clouds_absorption_opacities,
                                                           clouds_scattering_opacities,
                                                           clouds_particles_asymmetry_parameters,
                                                           pressures,
                                                           target_particle_radii,
                                                           target_pressure
                                                           )

        return _cloud_absorption_opacities, _cloud_scattering_opacities, cloud_one_minus_g

    @staticmethod
    def _compute_cloud_opacities_using_particle_size_grid(
            atmosphere_densities,
            clouds_mass_fractions,
            cloud_particle_number_density_grid,
            clouds_particles_densities,
            cloud_particles_radii_bins,
            cloud_particles_radii,
            clouds_absorption_opacities,
            clouds_scattering_opacities,
            clouds_particles_asymmetry_parameters,
            pressures,
            target_particle_radii,
            target_pressure
    ):
        r"""
        # TODO vectorize on/off code
        Args:
            atmosphere_densities:
                Density of the atmosphere at each of its layer
            clouds_mass_fractions:
                Mass fractions of each cloud at each atmospheric layer
            cloud_particle_number_density_grid:
                Output from cloud lognormal EXPERIMENTAL
            clouds_particles_densities:
                Density of each cloud particles
            cloud_particles_radii_bins:
                Bins of the particles cloud radii grid
            cloud_particles_radii:
                Particles cloud radii grid
            clouds_absorption_opacities:
                Cloud absorption opacities (radius grid, wavelength grid, clouds)
            clouds_scattering_opacities:
                Cloud scattering opacities (radius grid, wavelength grid, clouds)
            clouds_particles_asymmetry_parameters:
                Cloud particles asymmetry parameters (radius grid, wavelength grid, clouds)
            pressures:
                Pressure array
            target_particle_radii:
                Target particle radii array
            target_pressure:
                Target pressure array

        Returns:

        """
        # Extract precomputed dn/dr
        dn_dr = None  # prevent eventual reference before assignment

        if isinstance(cloud_particle_number_density_grid, dict):
            # map cloud names to the correct entry
            for i_spec, cloud in enumerate(clouds_mass_fractions):
                dn_dr = cloud_particle_number_density_grid[cloud]
        else:
            dn_dr = cloud_particle_number_density_grid

        _cloud_absorption_opacities, _cloud_scattering_opacities, cloud_one_minus_g = \
            Radtrans._integrate_cloud_opacities_from_dn_dr(
                dn_dr,
                atmosphere_densities,
                clouds_particles_densities,
                cloud_particles_radii_bins,
                cloud_particles_radii,
                clouds_absorption_opacities,
                clouds_scattering_opacities,
                clouds_particles_asymmetry_parameters,
                pressures,
                target_particle_radii,
                target_pressure
            )

        return _cloud_absorption_opacities, _cloud_scattering_opacities, cloud_one_minus_g

    @staticmethod
    def _compute_cloud_opacities(
        pressures, temperatures, frequency_bins_edges,
        cloud_species_mass_fractions, mean_molar_masses, reference_gravity,
        cloud_particle_radius_distribution_std, clouds_loaded_opacities,
        sum_opacities, anisotropic_cloud_scattering,
        cloud_f_sed=None, eddy_diffusion_coefficients=None,
        cloud_particles_mean_radii=None,
        cloud_particles_radius_distribution="lognormal",
        cloud_hansen_a=None, cloud_hansen_b=None,
        cloud_particle_number_density_grid=None,
        clouds_particles_porosity_factor=None,
        photospheric_cloud_optical_depths=None, continuum_opacities=None,
        return_cloud_contribution=False,
        target_particle_radii=None,
        target_pressure=None
    ):
        """Calculate cloud opacities for a defined atmospheric structure.
        # TODO complete docstring

        Args:
            temperatures:
            cloud_species_mass_fractions:
            mean_molar_masses:
            reference_gravity:
            cloud_particle_radius_distribution_std:
            cloud_f_sed:
            eddy_diffusion_coefficients:
            cloud_particles_mean_radii:
            cloud_particles_radius_distribution:
            cloud_hansen_a:
            cloud_hansen_b:
            cloud_particle_number_density_grid:
            return_cloud_contribution:

        Returns:

        """
        # Initialization
        n_clouds = len(cloud_species_mass_fractions)
        _cloud_species_mass_fractions = np.zeros((pressures.size, n_clouds), dtype='d', order='F')
        cloud_absorption_opacities = None
        cloud_anisotropic_scattering_opacities = None
        _cloud_particles_mean_radii = np.zeros((pressures.size, n_clouds), dtype='d', order='F')
        _cloud_particles_density = np.zeros(n_clouds, dtype='d', order='F')
        atmospheric_densities = pressures / cst.kB / temperatures * mean_molar_masses * cst.amu

        if continuum_opacities is None:
            continuum_opacities = np.zeros((frequency_bins_edges.size - 1, pressures.size))

        # Initialize Hansen's b coefficient
        if "hansen" in cloud_particles_radius_distribution.lower():
            if isinstance(cloud_hansen_b, np.ndarray):
                if not cloud_hansen_b.shape == (pressures.size, n_clouds):
                    raise ValueError(
                        "cloud_hansen_b must be a float, a dictionary with arrays for each cloud species, "
                        f"or a numpy array with shape {(pressures.shape[0], n_clouds)}, "
                        f"but was of shape {np.shape(cloud_hansen_b)}"
                    )
            elif isinstance(cloud_hansen_b, dict):
                cloud_hansen_b = np.array(list(cloud_hansen_b.values()), dtype='d', order='F').T
            elif isinstance(cloud_hansen_b, float):
                cloud_hansen_b = np.array(
                    np.tile(cloud_hansen_b * np.ones_like(pressures), (n_clouds, 1)),
                    dtype='d',
                    order='F'
                ).T
            else:
                raise ValueError(f"The Hansen distribution width (cloud_hansen_b) must be an array, a dict, "
                                 f"or a float, but is of type '{type(cloud_hansen_b)}' ({cloud_hansen_b})")

        # Initialize cloud species mass fractions, densities, and mean radii
        for i_spec, cloud_name in enumerate(cloud_species_mass_fractions):
            _cloud_species_mass_fractions[:, i_spec] = cloud_species_mass_fractions[cloud_name]

            if clouds_particles_porosity_factor is not None:
                _cloud_particles_density[i_spec] = (
                        clouds_loaded_opacities['particles_densities'][i_spec]
                        * (1 - clouds_particles_porosity_factor[cloud_name])
                )
            else:
                _cloud_particles_density[i_spec] = clouds_loaded_opacities['particles_densities']

            if cloud_particles_mean_radii is not None:
                _cloud_particles_mean_radii[:, i_spec] = cloud_particles_mean_radii[cloud_name]
            elif cloud_hansen_a is not None:
                _cloud_particles_mean_radii[:, i_spec] = cloud_hansen_a[cloud_name]

        # Calculate cloud opacities
        if (
            cloud_particles_mean_radii is not None
            or cloud_hansen_a is not None
            or cloud_particle_number_density_grid is not None
        ):
            if cloud_particles_radius_distribution == "lognormal":
                (
                    clouds_total_absorption_opacities,
                    clouds_total_scattering_opacities,
                    cloud_scattering_reduction_factor
                ) = (
                    Radtrans._compute_cloud_log_normal_particles_distribution_opacities(
                        atmosphere_densities=atmospheric_densities,
                        clouds_particles_densities=_cloud_particles_density,
                        clouds_mass_fractions=_cloud_species_mass_fractions,
                        cloud_particles_mean_radii=_cloud_particles_mean_radii,
                        cloud_particles_distribution_std=cloud_particle_radius_distribution_std,
                        cloud_particles_radii_bins=clouds_loaded_opacities['particles_radii_bins'],
                        cloud_particles_radii=clouds_loaded_opacities['particles_radii'],
                        clouds_absorption_opacities=clouds_loaded_opacities['absorption_opacities'],
                        clouds_scattering_opacities=clouds_loaded_opacities['scattering_opacities'],
                        clouds_particles_asymmetry_parameters=clouds_loaded_opacities['particles_asymmetry_parameters'],
                        pressures=pressures,
                        target_particle_radii=target_particle_radii,
                        target_pressure=target_pressure
                    )
                )
            elif cloud_particles_radius_distribution == "hansen":
                (
                    clouds_total_absorption_opacities,
                    clouds_total_scattering_opacities,
                    cloud_scattering_reduction_factor
                ) = (
                    fcore.compute_cloud_hansen_opacities(
                        atmospheric_densities,
                        _cloud_particles_density,
                        _cloud_species_mass_fractions,
                        _cloud_particles_mean_radii,
                        cloud_hansen_b,
                        clouds_loaded_opacities['particles_radii_bins'],
                        clouds_loaded_opacities['particles_radii'],
                        clouds_loaded_opacities['absorption_opacities'],
                        clouds_loaded_opacities['scattering_opacities'],
                        clouds_loaded_opacities['particles_asymmetry_parameters']
                    )
                )
            elif cloud_particles_radius_distribution == "custom":
                (
                    clouds_total_absorption_opacities,
                    clouds_total_scattering_opacities,
                    cloud_scattering_reduction_factor
                ) = (
                    Radtrans._compute_cloud_opacities_using_particle_size_grid(
                        atmosphere_densities=atmospheric_densities,
                        clouds_mass_fractions=cloud_species_mass_fractions,
                        cloud_particle_number_density_grid=cloud_particle_number_density_grid,
                        clouds_particles_densities=_cloud_particles_density,
                        cloud_particles_radii_bins=clouds_loaded_opacities['particles_radii_bins'],
                        cloud_particles_radii=clouds_loaded_opacities['particles_radii'],
                        clouds_absorption_opacities=clouds_loaded_opacities['absorption_opacities'],
                        clouds_scattering_opacities=clouds_loaded_opacities['scattering_opacities'],
                        clouds_particles_asymmetry_parameters=clouds_loaded_opacities['particles_asymmetry_parameters'],
                        pressures=pressures,
                        target_particle_radii=target_particle_radii,
                        target_pressure=target_pressure
                    )
                )
            else:
                raise ValueError(
                    f"cloud_particles_radius_distribution must be 'lognormal', 'hansen', or 'custom', "
                    f"but is '{cloud_particles_radius_distribution}'"
                )
        else:
            missing_arguments = []

            if cloud_particle_radius_distribution_std is None and cloud_particles_radius_distribution == "lognormal":
                missing_arguments.append("'cloud_particle_radius_distribution_std'")

            if cloud_f_sed is None:
                missing_arguments.append("'cloud_f_sed'")

            if eddy_diffusion_coefficients is None:
                missing_arguments.append("'eddy_diffusion_coefficients'")

            if cloud_particles_radius_distribution == "custom" and cloud_particle_number_density_grid is None:
                missing_arguments.append("'cloud_particle_number_density_grid'")

            if len(missing_arguments) > 0:
                raise ValueError(
                    f"missing necessary arguments to calculate cloud particle radii: {', '.join(missing_arguments)} "
                    f"(got 'None')\n"
                    f"Set the missing arguments, "
                    f"or directly set cloud particle radii "
                    f"through the arguments 'cloud_particles_mean_radii' or 'cloud_hansen_a', "
                    f"and argument 'cloud_particle_radius_distribution_std'.\n"
                    f"The cloud particle radii are necessary to calculate the cloud opacities."
                )

            # Initialize f_seds
            f_seds = np.array(np.zeros((len(pressures), n_clouds)), dtype='d', order='F')

            for i_spec, cloud in enumerate(cloud_species_mass_fractions):
                if isinstance(cloud_f_sed, dict):  # is either dictionary
                    f_seds[:, i_spec] = cloud_f_sed[cloud]
                else:  # or array with length of number of atmospheric layers, or scalar.
                    f_seds[:, i_spec] = cloud_f_sed

            # Calculate cloud_particles_mean_radii then cloud opacities
            if cloud_particles_radius_distribution == "lognormal":
                _cloud_particles_mean_radii = fcore.compute_cloud_particles_mean_radius(
                    reference_gravity,
                    atmospheric_densities,
                    _cloud_particles_density,
                    temperatures,
                    mean_molar_masses,
                    f_seds,
                    cloud_particle_radius_distribution_std,
                    eddy_diffusion_coefficients
                )

                (clouds_total_absorption_opacities, clouds_total_scattering_opacities,
                 cloud_scattering_reduction_factor) = \
                    Radtrans._compute_cloud_log_normal_particles_distribution_opacities(
                        atmosphere_densities=atmospheric_densities,
                        clouds_particles_densities=_cloud_particles_density,
                        clouds_mass_fractions=_cloud_species_mass_fractions,
                        cloud_particles_mean_radii=_cloud_particles_mean_radii,
                        cloud_particles_distribution_std=cloud_particle_radius_distribution_std,
                        cloud_particles_radii_bins=clouds_loaded_opacities['particles_radii_bins'],
                        cloud_particles_radii=clouds_loaded_opacities['particles_radii'],
                        clouds_absorption_opacities=clouds_loaded_opacities['absorption_opacities'],
                        clouds_scattering_opacities=clouds_loaded_opacities['scattering_opacities'],
                        clouds_particles_asymmetry_parameters=clouds_loaded_opacities[
                            'particles_asymmetry_parameters'],
                        pressures=pressures,
                        target_particle_radii=target_particle_radii,
                        target_pressure=target_pressure
                    )
            elif cloud_particles_radius_distribution == "hansen":
                _cloud_particles_mean_radii = fcore.compute_cloud_particles_mean_radius_hansen(
                    reference_gravity,
                    atmospheric_densities,
                    _cloud_particles_density,
                    temperatures,
                    mean_molar_masses,
                    f_seds,
                    cloud_hansen_b,
                    eddy_diffusion_coefficients
                )

                (clouds_total_absorption_opacities, clouds_total_scattering_opacities,
                 cloud_scattering_reduction_factor) = \
                    fcore.compute_cloud_hansen_opacities(
                        atmospheric_densities,
                        _cloud_particles_density,
                        _cloud_species_mass_fractions,
                        _cloud_particles_mean_radii,
                        cloud_hansen_b,
                        clouds_loaded_opacities['particles_radii_bins'],
                        clouds_loaded_opacities['particles_radii'],
                        clouds_loaded_opacities['absorption_opacities'],
                        clouds_loaded_opacities['scattering_opacities'],
                        clouds_loaded_opacities['particles_asymmetry_parameters']
                    )
            elif cloud_particles_radius_distribution == "custom":
                (clouds_total_absorption_opacities, clouds_total_scattering_opacities,
                 cloud_scattering_reduction_factor) = Radtrans._compute_cloud_opacities_using_particle_size_grid(
                    atmosphere_densities=atmospheric_densities,
                    clouds_mass_fractions=cloud_species_mass_fractions,
                    cloud_particle_number_density_grid=cloud_particle_number_density_grid,
                    clouds_particles_densities=_cloud_particles_density,
                    cloud_particles_radii_bins=clouds_loaded_opacities['particles_radii_bins'],
                    cloud_particles_radii=clouds_loaded_opacities['particles_radii'],
                    clouds_absorption_opacities=clouds_loaded_opacities['absorption_opacities'],
                    clouds_scattering_opacities=clouds_loaded_opacities['scattering_opacities'],
                    clouds_particles_asymmetry_parameters=clouds_loaded_opacities['particles_asymmetry_parameters'],
                    pressures=pressures,
                    target_particle_radii=target_particle_radii,
                    target_pressure=target_pressure
                )
            else:
                raise ValueError(
                    f"cloud_particles_radius_distribution must be 'lognormal', 'hansen', or 'custom', "
                    f"but is '{cloud_particles_radius_distribution}'"
                )

        # Take into account anisotropy (= 1 - asymmetry_parameter)
        (cloud_final_absorption_opacities, cloud_anisotropic_extinctions,
         cloud_scattering_reduction_factor, cloud_isotropic_extinctions) = \
            fcore.interpolate_cloud_opacities(
                clouds_total_absorption_opacities,
                clouds_total_scattering_opacities,
                cloud_scattering_reduction_factor,
                clouds_loaded_opacities['wavelengths'],
                frequency_bins_edges
            )

        if anisotropic_cloud_scattering:
            continuum_opacities_scattering = cloud_anisotropic_extinctions - cloud_final_absorption_opacities
        else:
            continuum_opacities_scattering = cloud_isotropic_extinctions - cloud_final_absorption_opacities

        if sum_opacities and photospheric_cloud_optical_depths is not None:
            cloud_anisotropic_scattering_opacities = (
                    cloud_anisotropic_extinctions - cloud_final_absorption_opacities
            )
            cloud_absorption_opacities = cloud_final_absorption_opacities

        continuum_opacities += cloud_final_absorption_opacities

        # This included scattering plus absorption
        if return_cloud_contribution:
            opacity_shape = (1, frequency_bins_edges.size - 1, 1, pressures.size)
            cloud_opacities = cloud_anisotropic_extinctions.reshape(opacity_shape)
        else:
            cloud_opacities = None

        return (continuum_opacities, continuum_opacities_scattering,
                cloud_anisotropic_scattering_opacities, cloud_absorption_opacities, cloud_opacities,
                _cloud_particles_mean_radii)

    @staticmethod
    def _compute_species_optical_depths(reference_gravity, pressures, cloud_opacities):
        """Calculate the optical depth of the clouds as function of
        frequency and pressure. The array with the optical depths is set to the
        ``tau_cloud`` attribute. The optical depth is calculated from the top of
        the atmosphere (i.e. the smallest pressure). Therefore, below the cloud
        base, the optical depth is constant and equal to the value at the cloud
        base.
        # TODO complete docstring

            Args:
                reference_gravity (float):
                    Surface gravity in cgs. Vertically constant for emission
                    spectra.
        """
        return fcore.compute_species_optical_depths(reference_gravity, pressures, cloud_opacities)

    @staticmethod
    def _compute_feautrier_radiative_transfer(
        frequency_bins_edges, temperatures, weights_gauss,
        emission_cos_angle_grid, emission_cos_angle_grid_weights,
        optical_depths, photon_destruction_probabilities,
        emission_geometry, stellar_intensity, star_irradiation_cos_angle,
        reflectances, emissivities, adaptive_feautrier_iterations, return_contribution
    ):
        flux, emission_contribution = fcore.compute_feautrier_radiative_transfer(
            frequency_bins_edges,
            temperatures,
            weights_gauss,
            emission_cos_angle_grid,
            emission_cos_angle_grid_weights,
            optical_depths,
            photon_destruction_probabilities,
            emission_geometry,
            stellar_intensity,
            star_irradiation_cos_angle,
            reflectances,
            emissivities,
            adaptive_feautrier_iterations,
            return_contribution
        )

        return flux, emission_contribution

    @staticmethod
    def _compute_h_minus_free_free_xsec(wavelengths, temperatures, electron_partial_pressure):
        """Calculate the H- free-free cross-section in units of cm^2 per H per e- pressure (in cgs).
        Source: "The Observation and Analysis of Stellar Photospheres" by David F. Gray, p. 156
        # TODO complete docstring
        Args:
            wavelengths: (angstrom)
            temperatures:
            electron_partial_pressure:

        Returns:

        """

        index = (wavelengths >= 2600.) & (wavelengths <= 113900.)
        lamb_use = wavelengths[index]

        if temperatures >= 2500.:
            # Convert to Angstrom (from cgs)
            theta = 5040. / temperatures

            f0 = (
                -2.2763
                - 1.6850 * np.log10(lamb_use)
                + 0.76661 * np.log10(lamb_use) ** 2.
                - 0.053346 * np.log10(lamb_use) ** 3.
            )
            f1 = (
                15.2827
                - 9.2846 * np.log10(lamb_use)
                + 1.99381 * np.log10(lamb_use) ** 2.
                - 0.142631 * np.log10(lamb_use) ** 3.
            )
            f2 = (
                -197.789
                + 190.266 * np.log10(lamb_use)
                - 67.9775 * np.log10(lamb_use) ** 2.
                + 10.6913 * np.log10(lamb_use) ** 3.
                - 0.625151 * np.log10(lamb_use) ** 4.
            )

            ret_val = np.zeros_like(wavelengths)
            ret_val[index] = (
                1e-26 * electron_partial_pressure * 10 ** (
                    f0 + f1 * np.log10(theta) + f2 * np.log10(theta) ** 2.
                )
            )
            return ret_val

        else:

            return np.zeros_like(wavelengths)

    @staticmethod
    def _compute_h_minus_bound_free_xsec(wavelengths_bin_edges):
        """ Calculate the H- bound-free cross-section in units of cm^2 per H-, as defined on page 155 of
        "The Observation and Analysis of Stellar Photospheres" by David F. Gray
        # TODO complete docstring
        Args:
            wavelengths_bin_edges: (angstrom)

        Returns:

        """

        left = wavelengths_bin_edges[:-1]
        right = wavelengths_bin_edges[1:]
        diff = np.diff(wavelengths_bin_edges)

        a = [
            1.99654,
            -1.18267e-5,
            2.64243e-6,
            -4.40524e-10,
            3.23992e-14,
            -1.39568e-18,
            2.78701e-23
        ]

        ret_val = np.zeros_like(wavelengths_bin_edges[1:])

        index = right <= 1.64e4

        for i_a in range(len(a)):
            ret_val[index] += a[i_a] * (
                    right[index] ** (i_a + 1) - left[index] ** (i_a + 1)
            ) / (i_a + 1)

        index_bracket = (left < 1.64e4) & (right > 1.64e4)
        for i_a in range(len(a)):
            ret_val[index_bracket] += a[i_a] * (1.64e4 ** (i_a + 1) -
                                                left[index_bracket] ** (i_a + 1)) / (i_a + 1)

        index = (left + right) / 2. > 1.64e4
        ret_val[index] = 0.
        index = ret_val < 0.
        ret_val[index] = 0.

        return ret_val * 1e-18 / diff

    @staticmethod
    def _compute_h_minus_opacities(
        mass_fractions, pressures, temperatures, frequencies, frequency_bins_edges,
        mean_molar_masses, **kwargs
    ):
        """Calculate the H- opacity."""
        _ = kwargs  # kwargs is not used in this function, this is intended

        wavelengths = frequency2wavelength(frequencies) * 1e8  # cm to Angstrom
        wavelengths_bin_edges = frequency2wavelength(frequency_bins_edges) * 1e8  # cm to Angstrom

        ret_val = np.array(np.zeros(len(wavelengths) * len(pressures)).reshape(
            len(wavelengths),
            len(pressures)), dtype='d', order='F')

        # Calculate electron number fraction
        m_e = cst.e_molar_mass  # (AMU)
        n_e = mean_molar_masses / m_e * mass_fractions['e-']

        # Calculate electron partial pressure
        p_e = pressures * n_e

        opacities_h_minus_bf = Radtrans._compute_h_minus_bound_free_xsec(wavelengths_bin_edges) / cst.amu

        for i_struct in range(len(n_e)):
            opacities_h_minus_ff = Radtrans._compute_h_minus_free_free_xsec(
                wavelengths,
                temperatures[i_struct],
                p_e[i_struct]
            ) / cst.amu * mass_fractions['H'][i_struct]

            ret_val[:, i_struct] = opacities_h_minus_bf * mass_fractions['H-'][i_struct] + opacities_h_minus_ff

        return ret_val

    @staticmethod
    def _compute_non_cia_gas_induced_continuum_opacities(
        gas_continuum_contributors, mass_fractions,
        pressures, temperatures, frequencies, frequency_bins_edges,
        mean_molar_masses, **kwargs
    ):
        continuum_opacities = 0

        for gas_continuum_contributor, contribution_function in (
                Radtrans.__get_non_cia_gas_continuum_contributions().items()):
            if gas_continuum_contributor in gas_continuum_contributors:
                continuum_opacities += (
                    contribution_function(
                        mass_fractions=mass_fractions,
                        pressures=pressures,
                        temperatures=temperatures,
                        frequencies=frequencies,
                        frequency_bins_edges=frequency_bins_edges,
                        mean_molar_masses=mean_molar_masses,
                        **kwargs
                    )
                )

        return continuum_opacities

    @staticmethod
    def _compute_optical_depths(
        pressures, reference_gravity, opacities, continuum_opacities_scattering, scattering_in_emission
    ):
        optical_depths, photon_destruction_probabilities = \
            fcore.compute_optical_depths(
                reference_gravity,
                pressures,
                opacities,
                scattering_in_emission,
                continuum_opacities_scattering
            )

        return optical_depths, photon_destruction_probabilities

    @staticmethod
    def _compute_optical_depths_wrapper(
        pressures,
        reference_gravity,
        opacities,
        continuum_opacities_scattering,
        scattering_in_emission,
        sum_opacities,
        photospheric_cloud_optical_depths=None,
        absorber_present=True,
        **custom_cloud_parameters
    ):
        optical_depths = np.zeros(opacities.shape, dtype='d', order='F')
        photon_destruction_probabilities = None
        relative_cloud_scaling_factor = None

        # Calculate optical depth for the total opacity
        if sum_opacities:
            if photospheric_cloud_optical_depths is not None:
                optical_depths, photon_destruction_probabilities, relative_cloud_scaling_factor = (
                    Radtrans._compute_optical_depths_with_photospheric_cloud(
                        pressures=pressures,
                        reference_gravity=reference_gravity,
                        opacities=opacities,
                        continuum_opacities_scattering=continuum_opacities_scattering,
                        **custom_cloud_parameters
                    )
                )
            else:
                if scattering_in_emission:
                    _continuum_opacities_scattering = continuum_opacities_scattering
                else:
                    _continuum_opacities_scattering = np.zeros(
                        continuum_opacities_scattering.shape, dtype='d', order='F'
                    )

                optical_depths[:, :, :1, :], photon_destruction_probabilities = (  # the :1 is necessary to get +1 ndim
                    Radtrans._compute_optical_depths(
                        pressures=pressures,
                        reference_gravity=reference_gravity,
                        opacities=opacities[:, :, :1, :],
                        continuum_opacities_scattering=_continuum_opacities_scattering,
                        scattering_in_emission=scattering_in_emission
                    )
                )
            # Handle cases without any absorbers, where opacities are zero
            if not absorber_present:
                print('No absorbers present, setting the photon destruction probability in the atmosphere to 1.')
                photon_destruction_probabilities[np.isnan(photon_destruction_probabilities)] = 1.

            # To handle cases when tau_cloud_at_Phot_clear = 0,
            # therefore cloud_scaling_factor = inf,
            # continuum_opacities_scattering_emission will contain nans and infs,
            # and photon_destruction_prob contains only nans
            if len(photon_destruction_probabilities[np.isnan(photon_destruction_probabilities)]) > 0:
                print('Region of zero opacity detected, '
                      'setting the photon destruction probability in this spectral range to 1.')
                photon_destruction_probabilities[np.isnan(photon_destruction_probabilities)] = 1.
        else:
            optical_depths = Radtrans._compute_species_optical_depths(
                reference_gravity=reference_gravity,
                pressures=pressures,
                cloud_opacities=opacities
            )

        return optical_depths, photon_destruction_probabilities, relative_cloud_scaling_factor

    @staticmethod
    def _compute_optical_depths_with_photospheric_cloud(
        pressures, reference_gravity,
        opacities, continuum_opacities_scattering,
        frequencies, weights_gauss, cloud_wavelengths, cloud_f_sed,
        cloud_anisotropic_scattering_opacities,
        cloud_absorption_opacities,
        photospheric_cloud_optical_depths, scattering_in_emission
    ):
        # TODO rename cloud_wavelengths into cloud_photosphere_wavelength_boundaries in version 4.0.0
        optical_depths = np.zeros(opacities.shape, dtype='d', order='F')
        relative_cloud_scaling_factor = None

        if scattering_in_emission:
            continuum_opacities_scattering_emission = copy.deepcopy(continuum_opacities_scattering)
        else:
            continuum_opacities_scattering_emission = np.zeros(continuum_opacities_scattering.shape)

        n_species = opacities.shape[2]

        _mass_fractions_1 = np.ones((pressures.size, n_species))

        if cloud_anisotropic_scattering_opacities is not None and cloud_absorption_opacities is not None:
            # Calculate continuum scattering opacity without clouds
            continuum_opacities_scattering_emission -= cloud_anisotropic_scattering_opacities

            opacities = Radtrans._combine_opacities(
                line_species_mass_fractions=_mass_fractions_1,
                opacities=opacities,
                continuum_opacities=-cloud_absorption_opacities
            )

        # Calculate optical depths without clouds
        optical_depths[:, :, :1, :], photon_destruction_probabilities = (
            Radtrans._compute_optical_depths(
                pressures=pressures,
                reference_gravity=reference_gravity,
                opacities=opacities[:, :, :1, :],
                continuum_opacities_scattering=continuum_opacities_scattering_emission,
                scattering_in_emission=scattering_in_emission
            )
        )

        # If there are no cloud opacities, return the optical depths without clouds
        if cloud_anisotropic_scattering_opacities is None or cloud_absorption_opacities is None:
            return optical_depths, photon_destruction_probabilities, None

        # Calculate optical depths for cloud only
        total_tau_cloud = np.zeros_like(optical_depths)

        # Reduce total (absorption) line opacity by continuum absorption opacity
        # (those two were added in before)
        mock_line_cloud_continuum_only = np.zeros_like(opacities)

        mock_line_cloud_continuum_only = Radtrans._combine_opacities(
            line_species_mass_fractions=_mass_fractions_1,
            opacities=mock_line_cloud_continuum_only,
            continuum_opacities=cloud_absorption_opacities
        )

        # Calculate optical depth of cloud only
        total_tau_cloud[:, :, :1, :], photon_destruction_prob_cloud = (
            Radtrans._compute_optical_depths(
                pressures=pressures,
                reference_gravity=reference_gravity,
                opacities=mock_line_cloud_continuum_only[:, :, :1, :],
                continuum_opacities_scattering=cloud_anisotropic_scattering_opacities,
                scattering_in_emission=scattering_in_emission
            )
        )

        # Calculate photospheric position of atmo without cloud, determine cloud optical depth there, compare to
        # photospheric_cloud_optical_depths, calculate scaling ratio
        median = True

        if cloud_wavelengths is None:
            # Use the full wavelength range for calculating the median optical depth of the clouds
            wavelengths_select = np.ones(frequencies.shape[0], dtype=bool)
        else:
            # Use a smaller wavelength range for the median optical depth
            # The units of cloud_wavelengths are converted from micron to cm
            # Unit conversion is done multiple times to preserve memory
            wavelengths_select = (np.logical_and(
                np.greater_equal(frequency2wavelength(frequencies), cloud_wavelengths[0] * 1e-4),
                np.less_equal(frequency2wavelength(frequencies), cloud_wavelengths[1] * 1e-4)
            ))

        # Calculate the cloud-free optical depth per wavelength
        w_gauss_photosphere = weights_gauss[..., np.newaxis, np.newaxis]
        optical_depth = np.sum(w_gauss_photosphere * optical_depths[:, :, 0, :], axis=0)

        if median:
            optical_depth_integral = np.median(optical_depth[wavelengths_select, :], axis=0)
        else:
            optical_depth_integral = np.sum(
                (optical_depth[1:, :] + optical_depth[:-1, :]) * np.diff(frequencies)[..., np.newaxis],
                axis=0) / (frequencies[-1] - frequencies[0]) / 2.

        optical_depth_cloud = np.sum(w_gauss_photosphere * total_tau_cloud[:, :, 0, :], axis=0)

        if median:
            optical_depth_cloud_integral = np.median(optical_depth_cloud[wavelengths_select, :], axis=0)
        else:
            optical_depth_cloud_integral = np.sum(
                (optical_depth_cloud[1:, :] + optical_depth_cloud[:-1, :]) * np.diff(frequencies)[
                    ..., np.newaxis], axis=0) / \
                                           (frequencies[-1] - frequencies[0]) / 2.

        # Interpolate the pressure where the optical depth of cloud-free atmosphere is 1.0
        if np.min(optical_depth_integral) < 1 < np.max(optical_depth_integral):
            press_bol_clear = interp1d(optical_depth_integral, pressures)
            p_phot_clear = press_bol_clear(1.)
        else:
            p_phot_clear = pressures[-1]

        # Interpolate the optical depth of the cloud-only atmosphere at the pressure of the cloud-free photosphere
        tau_bol_cloud = interp1d(pressures, optical_depth_cloud_integral)
        tau_cloud_at_phot_clear = tau_bol_cloud(p_phot_clear)

        # Apply cloud scaling
        cloud_scaling_factor = photospheric_cloud_optical_depths / tau_cloud_at_phot_clear

        if len(cloud_f_sed) > 0:
            max_rescaling = 1e100

            for f in cloud_f_sed.keys():
                if isinstance(cloud_f_sed[f], npt.NDArray):
                    raise ValueError("Only scalar cloud_f_sed values are allowed for cloud rescaling!")

                _max_rescaling = 2. * (cloud_f_sed[f] + 1.)
                max_rescaling = min(max_rescaling, _max_rescaling)

            relative_cloud_scaling_factor = cloud_scaling_factor / max_rescaling
            print(f"Relative cloud scaling factor: {relative_cloud_scaling_factor}")

        # Get continuum scattering opacity, including clouds:
        continuum_opacities_scattering_emission = \
            (continuum_opacities_scattering_emission
             + cloud_scaling_factor * cloud_anisotropic_scattering_opacities)

        opacities = Radtrans._combine_opacities(
            line_species_mass_fractions=_mass_fractions_1,
            opacities=opacities,
            continuum_opacities=cloud_scaling_factor * cloud_absorption_opacities
        )

        # Calculate total optical depth, including clouds
        optical_depths[:, :, :1, :], photon_destruction_probabilities = (
            Radtrans._compute_optical_depths(
                pressures=pressures,
                reference_gravity=reference_gravity,
                opacities=opacities[:, :, :1, :],
                continuum_opacities_scattering=continuum_opacities_scattering_emission,
                scattering_in_emission=scattering_in_emission
            )
        )

        return optical_depths, photon_destruction_probabilities, relative_cloud_scaling_factor

    @staticmethod
    def _compute_power_law_opacities(
        power_law_opacity_350nm, power_law_opacity_coefficient, frequencies, n_layers
    ):
        wavelengths = hz2um(frequencies)
        power_law_opacities = power_law_opacity_350nm * (wavelengths / 0.35) ** power_law_opacity_coefficient

        return np.tile(power_law_opacities, (n_layers, 1)).T  # (p, wavelengths)

    @staticmethod
    def _compute_radius_hydrostatic_equilibrium(
        pressures, temperatures, mean_molar_masses, reference_gravity, reference_pressure, planet_radius,
        variable_gravity=True
    ):
        pressures = pressures * 1e6  # bar to cgs
        reference_pressure = reference_pressure * 1e6  # bar to cgs

        atmospheric_densities = pressures * mean_molar_masses * cst.amu / cst.kB / temperatures
        radius_hydrostatic_equilibrium = fcore.compute_radius_hydrostatic_equilibrium(
            pressures,
            reference_gravity,
            atmospheric_densities,
            reference_pressure,
            planet_radius,
            variable_gravity
        )

        return radius_hydrostatic_equilibrium

    @staticmethod
    def _compute_rayleigh_scattering_opacities(
        rayleigh_species, pressures, temperatures, mass_fractions, mean_molar_masses, frequencies, haze_factor=1.0
    ):
        """Add Rayleigh scattering opacities to scattering continuum opacities.

        Args:
            temperatures: temperatures in each atmospheric layer
            mass_fractions: dictionary of the Rayleigh scattering species mass fractions
        """
        wavelengths_angstrom = np.array(frequency2wavelength(frequencies) * 1e8, dtype='d', order='F')
        rayleigh_scattering_opacities = np.zeros((frequencies.size, pressures.size), dtype='d', order='F')

        base_species_mass_fractions = Radtrans.__get_base_species_mass_fractions(mass_fractions)

        for species in rayleigh_species:
            if species not in base_species_mass_fractions:
                raise ValueError(f"Rayleigh species '{species}' not found in mass fractions dict "
                                 f"(listed species: {list(mass_fractions.keys())}, "
                                 f"listed species base names {list(base_species_mass_fractions.keys())}:)")

            rayleigh_scattering_opacities += haze_factor * fcore.compute_rayleigh_scattering_opacities(
                species,
                base_species_mass_fractions[species],
                wavelengths_angstrom,
                mean_molar_masses,
                temperatures,
                pressures
            )

        return rayleigh_scattering_opacities

    @staticmethod
    def _compute_rosseland_opacities(
        frequency_bins_edges, temperatures, weights_gauss, opacities, continuum_opacities_scattering,
        scattering_in_emission
    ):
        opacities_rosseland = fcore.compute_rosseland_opacities(
            opacities[:, :, 0, :],
            temperatures,
            weights_gauss,
            frequency_bins_edges,
            scattering_in_emission,
            continuum_opacities_scattering
        )

        return opacities_rosseland

    @staticmethod
    def _compute_transit_radii(
        opacities, continuum_opacities_scattering, pressures, temperatures, weights_gauss, mean_molar_masses,
        reference_gravity, reference_pressure, planet_radius, variable_gravity, summed_species_opacities,
        copy_opacities
    ):
        """Calculate the planetary transmission spectrum.
            # TODO complete docstring
            Args:
                opacities:
                continuum_opacities_scattering:
                pressures:
                temperatures:
                weights_gauss:
                mean_molar_masses:
                    Mean molecular weight in units of amu.
                    (1-d numpy array, same length as pressure array).
                reference_gravity (float):
                    Atmospheric gravitational acceleration at reference pressure and radius in units of
                    dyne/cm^2
                reference_pressure (float):
                    Reference pressure in bar.
                planet_radius (float):
                    Planet radius in cm.
                variable_gravity (bool):
                    If True, gravity in the atmosphere will vary proportional to 1/r^2, where r is the planet
                    radius.
                summed_species_opacities (bool):
                    If True, it is assumed that the given opacities contains the summed opacities of all species.
                    This is the case in line-by-line.
                copy_opacities (bool):
                    If True, a new array will be created for the transparencies, distinct from the opacities. This
                    increases memory usage.

            Returns:
                * transmission radius in cm (1-d numpy array, as many elements as wavelengths)
                * planet radius as function of atmospheric pressure (1-d numpy array, as many elements as atmospheric
                layers)
        """
        n_frequencies = np.size(opacities, axis=1)

        # Calculate planetary radius in hydrostatic equilibrium, using the atmospheric structure
        # (temperature, pressure, mmw), gravity, reference pressure and radius.
        radius_hydrostatic_equilibrium = Radtrans._compute_radius_hydrostatic_equilibrium(
            pressures=pressures,
            temperatures=temperatures,
            mean_molar_masses=mean_molar_masses,
            reference_gravity=reference_gravity,
            reference_pressure=reference_pressure,
            planet_radius=planet_radius,
            variable_gravity=variable_gravity
        )

        radius_hydrostatic_equilibrium = np.array(radius_hydrostatic_equilibrium, dtype='d', order='F')
        neg_rad = radius_hydrostatic_equilibrium < 0.
        radius_hydrostatic_equilibrium[neg_rad] = radius_hydrostatic_equilibrium[~neg_rad][0]

        # Calculate the density
        # TODO: replace values here with cst.amu and cst.kB.
        # Currently it is kept at the values of the Fortran implementation, such that
        # unit tests are still being passed.
        #                           cst.amu        # cst.kB  # the Fortran values are different from the cst values
        rho = (pressures * 1e6  # bar to cgs
               * mean_molar_masses * cst.amu / cst.kB / temperatures)

        # Bring continuum scattering opacities in right shape for matrix operations later.
        # Reminder: when calling this function, continuum absorption opacities have already
        # been added to line_struc_kappas.
        continuum_opa_scat_reshaped = continuum_opacities_scattering.reshape((1, n_frequencies, 1, pressures.size))

        # Calculate the inverse mean free paths (g, species, frequencies, levels)
        if copy_opacities:
            transparencies = copy.deepcopy(opacities)  # opacities will not be modified, but memory usage is increased
        else:
            transparencies = opacities  # opacities will be modified

        if summed_species_opacities:
            transparencies = transparencies[:, :, :1, :]
            transparencies *= rho
            transparencies += continuum_opa_scat_reshaped * rho
        else:
            transparencies *= rho
            transparencies[:, :, :1, :] += continuum_opa_scat_reshaped * rho

        # Calculate average mean free path between neighboring layers for later integration
        # Factor 1/2 is omitted because it cancels with effective planet area integration below.
        transparencies[:, :, :, 1:] += transparencies[:, :, :, :-1]

        # Prepare matrix for delta path lengths during optical depth integration
        diff_s = np.zeros((pressures.size, pressures.size))

        # Calculate matrix of delta path lengths
        r_ik = (radius_hydrostatic_equilibrium.reshape(1, pressures.size) ** 2.
                - radius_hydrostatic_equilibrium.reshape(pressures.size, 1) ** 2.)
        r_ik[r_ik < 0.] = 0.
        r_ik = np.sqrt(r_ik)
        diff_s[1:, 1:] = - r_ik[1:, 1:] + r_ik[1:, :-1]

        # Calculate optical depths (out is used for in-place calculations to reduce memory usage)
        transparencies = np.einsum('ijkl,ml', transparencies, diff_s, out=transparencies, optimize=True)

        # Delete unnecessary intermediate array
        del diff_s

        # Calculate transmittances (exp(-optical_depths), out is used for in-place calculations to reduce memory usage)
        transparencies *= -1
        transparencies = np.exp(transparencies, out=transparencies)

        if weights_gauss.size == 1:
            # Get rid of the g-dimension
            transparencies = transparencies[0]  # (frequencies, species, levels)
        else:
            # Integrate over correlated-k's g-coordinate
            transparencies = np.einsum('ijkl,i', transparencies, weights_gauss, optimize=True)

        transparencies = np.swapaxes(transparencies, 1, 0)  # (species, frequencies, levels)

        if transparencies.shape[0] == 1:
            # Get rid of the species dimension
            transparencies = transparencies[0]  # (frequencies, levels)
        else:
            # Multiply transmittances of all absorber species
            transparencies = np.prod(transparencies, axis=0)  # (frequencies, levels)

        # Calculate transparencies (1 - transmittances)
        transparencies *= -1
        transparencies += 1

        # Annulus radius increments
        annulus_radius_increments = -np.diff(radius_hydrostatic_equilibrium)

        # Integrate effective area, 2 pi omitted:
        # - 2 cancels with 1/2 of average inverse mean free path above
        # - pi cancels when calculating the radius from the area below
        transparencies *= radius_hydrostatic_equilibrium
        transparencies[:, 1:] += transparencies[:, :-1]
        transparencies = transparencies[:, 1:]  # get rid of unnecessary layer
        transparencies *= annulus_radius_increments
        transit_radii = np.sum(transparencies, axis=1)  # (frequencies)

        # Transform area to transmission radius
        transit_radii = np.sqrt(transit_radii + radius_hydrostatic_equilibrium[-1] ** 2.)

        return transit_radii, radius_hydrostatic_equilibrium

    @staticmethod
    def _compute_transmission_spectrum_contribution(
        transit_radii, pressures, temperatures, mean_molar_masses, reference_gravity, reference_pressure, planet_radius,
        weights_gauss, opacities, continuum_opacities_scattering, scattering_in_transmission, variable_gravity
    ):
        transmission_contribution, radius_hydrostatic_equilibrium = (
            fcore.compute_transmission_spectrum_contribution(
                opacities,
                temperatures,
                pressures,
                reference_gravity,
                mean_molar_masses,
                reference_pressure,
                planet_radius,
                weights_gauss,
                transit_radii ** 2,
                scattering_in_transmission,
                continuum_opacities_scattering,
                variable_gravity
            )
        )

        return transmission_contribution, radius_hydrostatic_equilibrium

    @staticmethod
    def _handle_grid_misalignment(
        indices: npt.NDArray[np.bool_],
        frequencies_reference: npt.NDArray[np.floating],
        frequencies_test: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.bool_]:
        _indices: npt.NDArray[np.signedinteger] = np.nonzero(indices)[0]

        if (
                np.abs(frequencies_test[indices][0] - frequencies_reference[0])
                < np.abs(frequencies_test[indices][0] - frequencies_reference[1])
        ):  # right misalignment
            # Situation is as follows:
            # 0   1   2   3
            # |...|...|...|  (ref.)
            #  |...|...|     (test)
            #  0   1   2
            # Include one more frequency to the right
            # 0   1   2   3
            # |...|...|...|  (ref.)
            #  |...|...|...| (test)
            #  0   1   2   3
            index_include = _indices[-1] + 1
        else:  # left misalignment
            # Situation is as follows:
            #  0   1   2   3
            #  |...|...|...|  (ref.)
            #     |...|...|   (test)
            #     1   2   3
            # Include one more frequency to the right
            #  0   1   2   3
            #  |...|...|...|  (ref.)
            # |...|...|...|   (test)
            # 0   1   2   3
            index_include = _indices[0] - 1

        if index_include < 0 or index_include == frequencies_test.size:  # handle out-of-bounds corrections
            _wavelengths = 1e4 * cst.c / frequencies_test
            wavelengths = 1e4 * cst.c / frequencies_reference

            raise ValueError(
                f"unrecoverable frequency grid misalignment: "
                f"reference frequency array is of size {frequencies_reference.size}, "
                f"but corresponding frequency array "
                f"is of size {_indices.size}; "
                f"a grid realignment was attempted, but required a frequency interval larger than stored in "
                f"the opacity file ({_wavelengths[0]} -- {_wavelengths[-1]} um in file, "
                f"{wavelengths[0]} -- {wavelengths[-1]} um in model)\n"
                f"Use a slightly smaller wavelength range, or use different opacity files"
            )

        indices[index_include] = True

        return indices

    @staticmethod
    def _init_cia_loaded_opacities(cia_contributors):
        tmp_collision_dict = LockedDict.build_and_lock({
            'molecules': None,
            'weight': None,
            'lambda': None,
            'temperature': None,
            'alpha': None
        })

        cia_loaded_opacities = LockedDict()

        for cia in cia_contributors:
            cia_loaded_opacities[cia] = copy.deepcopy(tmp_collision_dict)

        cia_loaded_opacities.lock()

        return cia_loaded_opacities

    def _init_frequency_grid(self):
        """Initialize the Radtrans frequency grid, used to calculate the spectra.
        The frequency grid comes from the requested opacity files, in the following priority order:
            1. lines,
            2. CIA,
            3. clouds.
        If not opacities are provided, the mean of the wavelength boundaries is used. The frequency grid in that case
        has only 1 element.

        Returns:
            frequencies:
                (Hz) frequencies (center of bin) of the line opacities, also use for spectral calculations, of size N
            frequency_bins_edges:
                (Hz) edges of the frequencies bins, of size N+1
                for correlated-k only, number of points used to sample the g-space (1 in the case lbl is used)
        """
        if len(self._line_species) > 0:
            frequencies, frequency_bins_edges = self._init_frequency_grid_from_lines()
        elif len(self._gas_continuum_contributors) > 0:
            hdf5_file = self.__get_cia_aliases(self._gas_continuum_contributors[0])
            hdf5_file = CIAOpacity.find(
                path_input_data=self._path_input_data,
                species=hdf5_file
            )

            with h5py.File(hdf5_file, 'r') as f:
                frequency_grid = cst.c * f['wavenumbers'][:]  # cm-1 to Hz

            frequencies, frequency_bins_edges = self._init_frequency_grid_from_frequency_grid(
                frequency_grid=frequency_grid,
                wavelength_boundaries=self._wavelength_boundaries,
                sampling=1
            )
        elif len(self._cloud_species) > 0:
            hdf5_file = CloudOpacity.get_aliases(self._cloud_species[0])
            hdf5_file = CloudOpacity.find(
                path_input_data=self._path_input_data,
                species=hdf5_file
            )

            with h5py.File(hdf5_file, 'r') as f:
                frequency_grid = cst.c * f['wavenumbers'][:]  # cm-1 to Hz

            frequencies, frequency_bins_edges = self._init_frequency_grid_from_frequency_grid(
                frequency_grid=frequency_grid,
                wavelength_boundaries=self._wavelength_boundaries,
                sampling=1
            )
        else:
            warnings.warn("no opacity source given (lines, CIA, or clouds), "
                          "setting frequency grid using the mean of the frequency boundaries (1 element)")
            frequency_bins_edges = np.zeros(2)
            frequency_bins_edges[0] = wavelength2frequency(self._wavelength_boundaries[1] * 1e-4)  # um to cm
            frequency_bins_edges[1] = wavelength2frequency(self._wavelength_boundaries[0] * 1e-4)  # um to cm
            frequencies = np.mean(frequency_bins_edges)

        return frequencies, frequency_bins_edges

    @staticmethod
    def _init_frequency_grid_from_frequency_grid(frequency_grid, wavelength_boundaries, sampling=1):
        # Get frequency boundaries
        frequency_min = um2hz(wavelength_boundaries[1])
        frequency_max = um2hz(wavelength_boundaries[0])

        # Check if the requested wavelengths boundaries are within the file boundaries
        bad_boundaries = False

        if frequency_min < frequency_grid[0]:
            bad_boundaries = True

        if frequency_max > frequency_grid[-1]:
            bad_boundaries = True

        if bad_boundaries:
            raise ValueError(f"Requested wavelength interval "
                             f"({wavelength_boundaries[0]}--{wavelength_boundaries[1]}) "
                             f"is out of opacities table wavelength grid "
                             f"({hz2um(frequency_grid[-1])}--{hz2um(frequency_grid[0])})")

        # Get the freq. corresponding to the requested boundaries, with the request fully within the selection
        selection = np.nonzero(np.logical_and(
            np.greater_equal(frequency_grid, frequency_min),
            np.less_equal(frequency_grid, frequency_max)
        ))[0]
        selection = np.array([selection[0], selection[-1]])

        if frequency_grid[selection[0]] > frequency_min:
            selection[0] -= 1

        if frequency_grid[selection[-1]] < frequency_max:
            selection[-1] += 1

        if sampling > 1:
            # Ensure that down-sampled wavelength upper bound >= requested wavelength upper bound
            selection[0] -= sampling - 1

        frequencies = frequency_grid[selection[0]:selection[-1] + 1]
        frequencies = frequencies[::-1]

        # Down-sample frequency grid in lbl mode if requested
        if sampling > 1:
            frequencies = frequencies[::sampling]

        frequency_bins_edges = np.array(
            wavelength2frequency(Radtrans.compute_bins_edges(frequency2wavelength(frequencies))), dtype='d', order='F'
        )

        return frequencies, frequency_bins_edges

    def _init_frequency_grid_from_lines(self):
        if self._line_opacity_mode == 'c-k':  # correlated-k
            # Get dimensions of molecular opacity arrays for a given P-T point, they define the resolution
            # Use the first entry of self.line_species for this, if given
            opacities_file = self.__get_line_opacity_file(
                path_input_data=self._path_input_data,
                species=self._line_species[0],
                category='correlated_k_opacities'
            )

            with h5py.File(opacities_file, 'r') as f:
                frequency_bins_edges = cst.c * f['bin_edges'][:][::-1]

            wavelengths = hz2um(frequency_bins_edges)
        elif self._line_opacity_mode == 'lbl':  # line-by-line
            # Load the wavelength grid
            opacities_file = LineByLineOpacity.find(
                path_input_data=self._path_input_data,
                species=self._line_species[0]
            )

            with h5py.File(opacities_file, 'r') as f:
                frequency_grid = cst.c * f['bin_edges'][:]  # cm-1 to Hz

            if frequency_grid[0] == frequency_grid[1]:  # handle duplicated last wavenumber
                frequency_grid = frequency_grid[1:]

            wavelengths = hz2um(frequency_grid[::-1])
        else:
            raise ValueError(f"line opacity mode must be 'c-k' or 'lbl', but was '{self._line_opacity_mode}'")

        # Extend the wavelength range if user requests larger range than what first line opa species contains
        if wavelengths[-1] < self._wavelength_boundaries[1]:
            delta_log_wavelength = np.diff(np.log10(wavelengths))[-1]

            add_high = 1e1 ** np.arange(
                np.log10(wavelengths[-1]),
                np.log10(self._wavelength_boundaries[-1]) + delta_log_wavelength,
                delta_log_wavelength
            )[1:]
            wavelengths = np.concatenate((wavelengths, add_high))

        if wavelengths[0] > self._wavelength_boundaries[0]:
            delta_log_wavelength = np.diff(np.log10(wavelengths))[0]

            if delta_log_wavelength == 0:  # handle duplicate first wavelength
                delta_log_wavelength = np.diff(np.log10(wavelengths))[1]

            add_low = 1e1 ** (-np.arange(
                -np.log10(wavelengths[0]),
                -np.log10(self._wavelength_boundaries[0]) + delta_log_wavelength,
                delta_log_wavelength
            )[1:][::-1])
            wavelengths = np.concatenate((add_low, wavelengths))

        # Get back to frequencies
        if self._line_opacity_mode == 'c-k':
            frequency_bins_edges = um2hz(wavelengths)
            frequencies = (frequency_bins_edges[1:] + frequency_bins_edges[:-1]) * 0.5

            # Cut the wavelength range if user requests smaller range than what first line opa species contains
            indices_within_boundaries = np.nonzero(np.logical_and(
                np.greater(frequency2wavelength(frequencies), self._wavelength_boundaries[0] * 1e-4),  # um to cm
                np.less(frequency2wavelength(frequencies), self._wavelength_boundaries[1] * 1e-4)  # um to cm
            ))[0]

            frequencies = np.array(frequencies[indices_within_boundaries], dtype='d', order='F')

            # Get the corresponding frequencies bin edges, +2 is to catch the upper bin edge
            frequency_bins_edges = np.array(
                frequency_bins_edges[indices_within_boundaries[0]:indices_within_boundaries[-1] + 2],
                dtype='d',
                order='F'
            )
        elif self._line_opacity_mode == 'lbl':
            frequency_grid = um2hz(wavelengths[::-1])
            frequencies, frequency_bins_edges = self._init_frequency_grid_from_frequency_grid(
                frequency_grid=frequency_grid,
                wavelength_boundaries=self._wavelength_boundaries,
                sampling=self._line_by_line_opacity_sampling
            )
        else:
            raise ValueError(f"line opacity mode must be 'c-k' or 'lbl', but was '{self._line_opacity_mode}'")

        return frequencies, frequency_bins_edges

    @staticmethod
    def _interpolate_cia(
        collision_dict, combined_mass_fractions, pressures, temperatures, frequencies, mean_molar_masses
    ):
        """Interpolate CIA cross-sections onto the Radtrans (wavelength, temperature) grid and convert it into
        opacities.

        Args:
            combined_mass_fractions: combined mass fractions of the colliding species
                e.g., for H2-He and an atmosphere with H2 and He MMR of respectively 0.74 and 0.24,
                combined_mas_fractions = 0.74 * 0.24
                combined_mas_fractions is divided by the combined weight (e.g. for H2 and He, 2 * 4 AMU^2), so there is
                no units issue.

        Returns:
            A (wavelength, temperature) array containing the CIA opacities.
        """
        factor = (
            combined_mass_fractions / collision_dict['weight']
            * mean_molar_masses / cst.amu / (cst.L0 ** 2) * pressures / cst.kB / temperatures
        )

        log10_alpha = np.log10(collision_dict['alpha'])

        if collision_dict['temperature'].shape[0] > 1:
            # Interpolation on temperatures for each wavelength point
            interpolating_function = interp1d(
                x=collision_dict['temperature'],
                y=log10_alpha,
                kind='linear',
                bounds_error=False,
                fill_value=(log10_alpha[:, 0], log10_alpha[:, -1]), axis=1
            )

            cia_opacities = np.exp(interpolating_function(temperatures) * np.log(10))

            interpolating_function = interp1d(
                x=collision_dict['lambda'],
                y=cia_opacities,
                kind='linear',
                bounds_error=False,
                fill_value=sys.float_info.min,
                axis=0
            )

            cia_opacities = interpolating_function(frequency2wavelength(frequencies))
            cia_opacities[cia_opacities < sys.float_info.min] = 0

            return cia_opacities * factor
        else:
            raise ValueError(f"petitRADTRANS require a rectangular CIA table, "
                             f"table shape was {collision_dict['temperature'].shape}")

    @staticmethod
    def _interpolate_species_opacities(
        pressures, temperatures, n_g, n_frequencies, line_opacities_grid, line_opacities_temperature_pressure_grid,
        line_opacities_temperature_grid_size, line_opacities_pressure_grid_size
    ):
        # Interpolate line opacities to given temperature structure.
        n_layers = pressures.size
        n_line_species = len(line_opacities_grid)

        if n_line_species > 0:
            line_opacities = np.zeros(
                (n_g, n_frequencies, n_line_species, n_layers), dtype='d', order='F'
            )

            for i, species in enumerate(line_opacities_grid):
                line_opacities[:, :, i, :] = finput.interpolate_line_opacities(
                    pressures,
                    temperatures,
                    line_opacities_temperature_pressure_grid[species],
                    True,  # always assume custom PT grid with new format
                    line_opacities_temperature_grid_size[species],
                    line_opacities_pressure_grid_size[species],
                    line_opacities_grid[species]
                )
        else:
            line_opacities = np.zeros(
                (n_g, n_frequencies, 1, n_layers), dtype='d', order='F'
            )

        return line_opacities

    @staticmethod
    def compute_bins_edges(middle_bin_points: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Calculate bin edges for middle bin points.

        Args:
            middle_bin_points: array of size N containing the middle bin points to calculate the bin edges of

        Returns:
            array of size N+1 containing the bins edges
        """
        bin_edges = [middle_bin_points[0] - (middle_bin_points[1] - middle_bin_points[0]) / 2]

        for i in range(int(len(middle_bin_points)) - 1):
            bin_edges.append(middle_bin_points[i] + (middle_bin_points[i + 1] - middle_bin_points[i]) / 2)

        bin_edges.append(
            middle_bin_points[len(middle_bin_points) - 1]
            + (middle_bin_points[len(middle_bin_points) - 1] - middle_bin_points[len(middle_bin_points) - 2]) / 2
        )

        return np.array(bin_edges)

    def calculate_contribution_spectra(
        self, mode,
        opaque_cloud_top_pressure=None,
        power_law_opacity_350nm=None,
        power_law_opacity_coefficient=None,
        gray_opacity=None,
        cloud_photosphere_median_optical_depth=None,
        additional_absorption_opacities_function=None,
        additional_scattering_opacities_function=None,
        stellar_intensities=None,
        **kwargs
    ):
        # Store opacity sources values
        opacity_sources = self.__get_opacity_sources_dict()
        opacity_sources.update({
            'init': {
                'line_species': copy.deepcopy(self.line_species),
                'gas_continuum_contributors': copy.deepcopy(self.gas_continuum_contributors),
                'cloud_species': copy.deepcopy(self.cloud_species),
                'rayleigh_species': copy.deepcopy(self.rayleigh_species)
            },
            'runtime': {
                'opaque_cloud_top_pressure': opaque_cloud_top_pressure,
                'power_law': {
                    'power_law_opacity_350nm': power_law_opacity_350nm,
                    'power_law_opacity_coefficient': power_law_opacity_coefficient,
                },
                'gray_opacity': gray_opacity,
                'cloud_photosphere_median_optical_depth': cloud_photosphere_median_optical_depth,
                'additional_absorption_opacities_function': additional_absorption_opacities_function,
                'additional_scattering_opacities_function': additional_scattering_opacities_function,
                'stellar_intensities': stellar_intensities
            }
        })

        # Default values for opacity sources
        opacity_source_default = self.__get_opacity_sources_dict()

        # Initialize spectral mode
        if mode == 'emission':
            spectral_function = 'calculate_flux'
        elif mode == 'transmission':
            spectral_function = 'calculate_transit_radii'

            # Remove spectral parameters that are not implemented in transmission
            del opacity_sources['runtime']['cloud_photosphere_median_optical_depth']
            del opacity_sources['runtime']['stellar_intensities']
        else:
            raise ValueError(f"mode '{mode}' is not implemented, "
                             f"available modes are 'emission'|'transmission'")

        # Initialize opacity contributions dict
        opacity_contributions = {}

        # Total spectrum
        spectral_parameters = {
            copy.deepcopy(k): copy.deepcopy(v)
            for k, v in kwargs.items()
        }

        for key, value in opacity_sources['runtime'].items():
            if key == 'power_law':
                spectral_parameters['power_law_opacity_350nm'] = copy.deepcopy(value['power_law_opacity_350nm'])
                spectral_parameters['power_law_opacity_coefficient'] = copy.deepcopy(
                    value['power_law_opacity_coefficient'])
            else:
                spectral_parameters[key] = copy.deepcopy(value)

        opacity_contributions['Total'] = self.__getattr__(spectral_function)(
            **spectral_parameters
        )

        # Contribution spectra
        _default_radtrans = None
        _default_mass_fractions = None

        for opacity_type, opacity_source in opacity_sources.items():
            if opacity_type == 'init':  # fixed (at instantiation) opacity sources
                for opacity, species_list in opacity_source.items():  # all fixed opacity sources are lists
                    opacity_contributions[opacity] = {}

                    # Init a temporary radtrans object with only the relevant opacity source
                    for species in species_list:
                        _user_cloud_species = None

                        # Use full names for CIA and clouds to have a clean legend when plotting the opacities
                        if opacity == 'gas_continuum_contributors':
                            species = self.__get_cia_aliases(species)
                        elif opacity == 'cloud_species':
                            _user_cloud_species = copy.deepcopy(species)
                            species = CloudOpacity.get_aliases(species)

                        fixed_opacity_sources = copy.deepcopy(opacity_source_default[opacity_type])
                        fixed_opacity_sources[opacity] = [species]

                        print(f"Generating temporary Radtrans object with '{opacity}': '{species}'")

                        # Instantiate a default Radtrans object that will be used for runtime opacity sources
                        if _default_radtrans is None:
                            _default_radtrans = Radtrans(
                                pressures=copy.deepcopy(self.pressures) * 1e-6,  # cgs to bar
                                wavelength_boundaries=copy.deepcopy(self.wavelength_boundaries),
                                line_opacity_mode=copy.deepcopy(self.line_opacity_mode),
                                line_by_line_opacity_sampling=copy.deepcopy(
                                    self.line_by_line_opacity_sampling),
                                scattering_in_emission=copy.deepcopy(self.scattering_in_emission),
                                emission_angle_grid=copy.deepcopy(self.emission_angle_grid),
                                anisotropic_cloud_scattering=copy.deepcopy(
                                    self.anisotropic_cloud_scattering),
                                path_input_data=copy.deepcopy(self.path_input_data),
                                **fixed_opacity_sources
                            )

                        if _default_mass_fractions is None:
                            _default_mass_fractions = {species: np.zeros(self.pressures.size)}

                        # Use the first line species in order to keep the resolving power consistent in all spectra
                        if opacity != 'line_species':
                            fixed_opacity_sources['line_species'] = copy.deepcopy(_default_radtrans.line_species)

                        _radtrans = Radtrans(
                            pressures=copy.deepcopy(self.pressures) * 1e-6,  # cgs to bar
                            wavelength_boundaries=copy.deepcopy(self.wavelength_boundaries),
                            line_opacity_mode=copy.deepcopy(self.line_opacity_mode),
                            line_by_line_opacity_sampling=copy.deepcopy(self.line_by_line_opacity_sampling),
                            scattering_in_emission=copy.deepcopy(self.scattering_in_emission),
                            emission_angle_grid=copy.deepcopy(self.emission_angle_grid),
                            anisotropic_cloud_scattering=copy.deepcopy(self.anisotropic_cloud_scattering),
                            path_input_data=copy.deepcopy(self.path_input_data),
                            **fixed_opacity_sources
                        )

                        # Generate the spectrum containing only the relevant fixed opacity source
                        spectral_parameters = {}

                        # Fill the spectral parameters with default values for all runtime opacity sources
                        for _key, _value in kwargs.items():
                            if _key in opacity_source_default['runtime']:
                                spectral_parameters[_key] = copy.deepcopy(opacity_source_default['runtime'][_key])
                            elif _key == 'mass_fractions':
                                spectral_parameters[_key] = copy.deepcopy(_value)

                                # Fixes the first line species MMR to 0 so that it does not impact the spectrum
                                if species not in _default_mass_fractions:
                                    spectral_parameters[_key].update(_default_mass_fractions)
                            else:  # non-opacity spectral parameters
                                spectral_parameters[_key] = copy.deepcopy(_value)

                        # Update the user's cloud name to the full one everywhere it is needed to prevent crashes
                        if opacity == 'cloud_species':
                            if _user_cloud_species != species:
                                for _key, _value in spectral_parameters.items():
                                    if isinstance(_value, dict):
                                        if _user_cloud_species in _value:
                                            _value[species] = copy.deepcopy(_value[_user_cloud_species])
                                            del _value[_user_cloud_species]

                        opacity_contributions[opacity][species] = _radtrans.__getattr__(spectral_function)(
                            **spectral_parameters
                        )

                        del _radtrans  # free up some memory
            else:  # variable (at runtime) opacity sources
                for opacity, opacity_value in opacity_source.items():
                    if opacity_value is None:
                        opacity_contributions[opacity] = None
                        continue

                    spectral_parameters = {}

                    # Handle power law dual parameters special case
                    if opacity == 'power_law':
                        for key, value in opacity_value.items():
                            spectral_parameters[key] = copy.deepcopy(value)

                        if np.any(np.equal(list(spectral_parameters.values()), None)):
                            if not np.all(np.equal(list(spectral_parameters.values()), None)):
                                missing = []

                                for k, v in spectral_parameters.items():
                                    if v is None:
                                        missing.append(k)

                                missing = "'" + ", '".join(missing) + "'"

                                warnings.warn(f"opacity '{opacity}' requires {len(spectral_parameters)} parameters, "
                                              f"but lacks value for parameters {missing}")

                            opacity_contributions[opacity] = None
                            continue
                    else:
                        spectral_parameters[opacity] = copy.deepcopy(opacity_value)

                    # Fill the spectral parameters with default values for non-relevant opacity sources
                    for key, value in kwargs.items():
                        if key in opacity_source_default['runtime']:
                            spectral_parameters[key] = copy.deepcopy(opacity_source_default['runtime'][key])
                        elif key == 'mass_fractions':
                            spectral_parameters[key] = copy.deepcopy(_default_mass_fractions)
                        else:  # non-opacity spectral parameters
                            spectral_parameters[key] = copy.deepcopy(value)

                    opacity_contributions[opacity] = _default_radtrans.__getattr__(spectral_function)(
                        **spectral_parameters
                    )

        return opacity_contributions

    def calculate_flux(
        self,
        temperatures: npt.NDArray[np.floating],
        mass_fractions: dict[str, npt.NDArray[np.floating]],
        mean_molar_masses: npt.NDArray[np.floating],
        reference_gravity: float,
        planet_radius: float = None,
        opaque_cloud_top_pressure: float = None,
        cloud_particles_mean_radii: dict[str, npt.NDArray[np.floating]] = None,
        cloud_particle_radius_distribution_std: float = None,
        cloud_particles_radius_distribution: str = 'lognormal',
        cloud_hansen_a: dict[str, npt.NDArray[np.floating]] = None,
        cloud_hansen_b: dict[str, npt.NDArray[np.floating]] = None,
        cloud_particle_number_density_grid: dict[str, npt.NDArray[np.floating]] = None,
        clouds_particles_porosity_factor: dict[str, float] = None,
        cloud_f_sed: float = None,
        eddy_diffusion_coefficients: npt.NDArray[np.floating] = None,
        haze_factor: float = 1.0,
        power_law_opacity_350nm: float = None,
        power_law_opacity_coefficient: float = None,
        gray_opacity: float = None,
        cloud_photosphere_wavelength_boundaries: npt.NDArray[np.floating] = None,
        cloud_photosphere_median_optical_depth: float = None,
        cloud_fraction: float = 1.0,
        complete_coverage_clouds: list[str] = None,
        irradiation_geometry: str = 'dayside_ave',
        emission_geometry: str = None,  # TODO deprecated, replace with irradiation_geometry everywhere in the code
        stellar_intensities: npt.NDArray[np.floating] = None,
        star_effective_temperature: float = None,
        star_radius: float = None,
        orbit_semi_major_axis: float = None,
        star_irradiation_angle: float = 0.0,
        reflectances: npt.NDArray[np.floating] = None,
        emissivities: npt.NDArray[np.floating] = None,
        additional_absorption_opacities_function: Callable = None,
        additional_scattering_opacities_function: Callable = None,
        adaptive_feautrier_iterations: bool = False,
        frequencies_to_wavelengths: bool = True,
        return_contribution: bool = False,
        return_clear_spectrum: bool = False,
        return_photosphere_radius: bool = False,
        return_rosseland_optical_depths: bool = False,
        return_cloud_contribution: bool = False,
        return_opacities: bool = False,
        return_abundances: bool = False,
        return_particles_radii_bins: bool = False,
        return_particles_radii: bool = False,
        target_particle_radii: npt.NDArray[np.floating] = None,
        target_pressure: npt.NDArray[np.floating] = None
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], dict[str, Any]]:
        """Method to calculate the atmosphere's emitted flux (emission spectrum).

            Args:
                temperatures:
                    The atmospheric temperature in K, at each atmospheric layer
                    (1-d numpy array, same length as pressure array).
                mass_fractions:
                    Dictionary of mass fractions for all atmospheric absorbers.
                    Dictionary keys are the species names. Every mass fraction array has same length as pressure array.
                mean_molar_masses:
                    The atmospheric mean molecular weight in amu, at each atmospheric layer
                    (1-d numpy array, same length as pressure array).
                reference_gravity (float):
                    Surface gravity in cgs. Vertically constant for emission spectra.
                planet_radius: planet radius at maximum pressure in cm. Only used to calculate the planet's changing
                    Photospheric radius as function of wavelength, if return_photosphere_radius is True.
                opaque_cloud_top_pressure (Optional[float]):
                    Pressure, in bar, where opaque cloud deck is added to the absorption opacity.
                cloud_particles_mean_radii (Optional):
                    Dictionary of mean particle radii for all cloud species.
                    Dictionary keys are the cloud species names. Every radius array has same length as pressure array.
                cloud_particle_radius_distribution_std (Optional[float]):
                    Width of the log-normal cloud particle size distribution
                cloud_particles_radius_distribution (Optional[string]):
                    The cloud particle size distribution to use.
                    Can be either 'lognormal' (default) or 'hansen'.
                    If hansen, the cloud_hansen_b parameters must be used.
                cloud_hansen_a (Optional[dict]):
                    A dictionary of the 'a' parameter values for each included cloud species and for each atmospheric
                    layer, formatted as the kzz argument. Equivalent to cloud_particles_mean_radii.
                    If cloud_hansen_a is not included and dist is "hansen", then it will be computed using Kzz and fsed
                    (recommended).
                cloud_hansen_b (Optional[dict]):
                    A dictionary of the 'b' parameter values for each included cloud species and for each atmospheric
                    layer, formatted as the kzz argument. This is the width of the hansen distribution normalized by
                    the particle area (1/cloud_hansen_a^2)
                cloud_particle_number_density_grid (Optional[dict]):
                    EXPERIMENTAL
                clouds_particles_porosity_factor (Optional[dict]):
                    A dictionary of porosity factors depending on the cloud species. This can be useful when opacities
                    are calculated using the Distribution of Hollow Spheres (DHS) method.
                cloud_f_sed (Optional[float]):
                    Cloud settling parameter.
                eddy_diffusion_coefficients (Optional[float]):
                    The atmospheric eddy diffusion coefficient in cgs (i.e. :math:`\\rm cm^2/s`), at each atmospheric
                    layer (1-d numpy array, same length as pressure array).
                haze_factor (Optional[float]):
                    Scalar factor, increasing the gas Rayleigh scattering cross-section.
                power_law_opacity_350nm (Optional[float]):
                    Scattering opacity at 0.35 micron, in cgs units (cm^2/g).
                power_law_opacity_coefficient (Optional[float]):
                    Has to be given if kappa_zero is defined, this is the
                    wavelength powerlaw index of the parametrized scattering
                    opacity.
                gray_opacity (Optional[float]):
                    Gray opacity value, to be added to the opacity at all pressures and wavelengths
                    (units :math:`\\rm cm^2/g`)
                cloud_photosphere_median_optical_depth (Optional[float]):
                    Median optical depth (across ``wavelength_boundaries``) of the clouds from the top of the
                    atmosphere down to the gas-only photosphere. This parameter can be used for enforcing the presence
                    of clouds in the photospheric region.
                cloud_photosphere_wavelength_boundaries (Optional[NDArray]):
                    Min and max wavelength boundaries when calculating the median of the photospheric cloud optical
                    depth. By default, the whole spectral range is used.
                cloud_fraction:
                    Mix a column without cloud with a column with cloud to the requested fraction
                    (0 = clear, 1 = full cloud cover). By default, assume a full cloud cover.
                complete_coverage_clouds:
                    List of clouds that have full coverage, regardless of the value of cloud_fraction.
                irradiation_geometry (Optional[string]):
                    if equal to ``'dayside_ave'``: use the dayside average geometry.
                    If equal to ``'planetary_ave'``: use the planetary average geometry.
                    If equal to ``'non-isotropic'``: use the non-isotropic geometry.
                emission_geometry (Optional[string]):
                    Deprecated, same as irradiation_geometry.
                stellar_intensities (Optional[array]):
                    The stellar intensity to use. If None, it will be calculated using a PHOENIX model.
                star_effective_temperature (Optional[float]):
                    The temperature of the host star in K, used only if the
                    scattering is considered. If not specified, the direct light contribution is not calculated.
                star_radius (Optional[float]):
                    The radius of the star in cm. If specified, used to scale the to scale the stellar flux,
                    otherwise it uses PHOENIX radius.
                orbit_semi_major_axis (Optional[float]):
                    The distance of the planet from the star. Used to scale the stellar flux when the scattering of the
                    direct light is considered.
                star_irradiation_angle (Optional[float]):
                    Inclination angle of the direct light with respect to the normal to the atmosphere. Used only in
                    the non-isotropic geometry scenario.
                reflectances (Optional):
                    Reflectances of the surface (layer with the highest pressure).
                emissivities (Optional):
                    Emissivities of the surface (layer with the highest pressure).
                additional_absorption_opacities_function (Optional[function]):
                    A python function that takes wavelength arrays in microns and pressure arrays in bars
                    as input, and returns an absorption opacity matrix in units of cm^2/g, in the shape of
                    number of wavelength points x number of pressure points.
                    This opacity will then be added to the atmospheric absorption opacity.
                    This must not be used to add atomic / molecular line opacities in low-resolution mode (c-k),
                    because line opacities require a proper correlated-k treatment.
                    It may be used to add simple cloud absorption laws, for example, which
                    have opacities that vary only slowly with wavelength, such that the current
                    model resolution is sufficient to resolve any variations.
                additional_scattering_opacities_function (Optional[function]):
                    A python function that takes wavelength arrays in microns and pressure arrays in bars
                    as input, and returns an isotropic scattering opacity matrix in units of cm^2/g, in the shape of
                    number of wavelength points x number of pressure points.
                    This opacity will then be added to the atmospheric absorption opacity.
                    It may be used to add simple cloud absorption laws, for example, which
                    have opacities that vary only slowly with wavelength, such that the current
                    model resolution is sufficient to resolve any variations.
                adaptive_feautrier_iterations (Optional[bool]):
                    If True, adapt the number of iterations of the Feautrier scattering-method
                    (scattering_in_emission=True) based on the photon destruction probability.
                frequencies_to_wavelengths (Optional[bool]):
                    if True, convert the frequencies (Hz) output to wavelengths (cm),
                    and the flux per frequency output (erg.s-1.cm-2/Hz) to flux per wavelength (erg.s-1.cm-2/cm)
                return_contribution (Optional[bool]):
                    If True, calculate the emission contribution function.
                return_clear_spectrum (Optional[bool]):
                    If True, return the clear spectrum in addition to a cloudy spectrum.
                return_photosphere_radius (Optional[bool]):
                    If True, calculate and return the photosphere radius.
                return_rosseland_optical_depths (Optional[bool]):
                    If True, calculate and return the Rosseland opacities and optical depths.
                return_cloud_contribution (Optional[bool]):
                    If True, calculate the cloud contribution.
                return_opacities (Optional[bool]):
                    If True, return the absorption opacities and scattering opacities for species and clouds, as well
                    as optical depths
                return_abundances (Optional[bool]):
                    If True, return the mass fractions.
                return_particles_radii_bins (Optional[bool]):
                    If True, return the particle radii bins.
                return_particles_radii (Optional[bool]):
                    If True, return the particle radii.
                target_particle_radii (Optional[npt.NDArray[np.floating]]):
                    # TODO complete docstring
                target_pressure (Optional[npt.NDArray[np.floating]]):
                    # TODO complete docstring
        """
        if emission_geometry is not None:  # TODO [4.0.0] remove
            irradiation_geometry = emission_geometry

            warnings.warn(
                "'emission_geometry' is deprecated and will be removed in a future update. "
                "Use 'irradiation_geometry' instead",
                FutureWarning
            )

        auto_anisotropic_cloud_scattering, complete_coverage_clouds, flux_clear = (
            self.__init_spectral_function(
                is_emission=True,
                reference_gravity=reference_gravity,
                complete_coverage_clouds=complete_coverage_clouds,
                return_clear_spectrum=return_clear_spectrum,
                mass_fractions=mass_fractions,
                cloud_fraction=cloud_fraction,
                opaque_cloud_top_pressure=opaque_cloud_top_pressure
            )
        )

        if opaque_cloud_top_pressure is not None and self._scattering_in_emission:
            warnings.warn("the use of opaque_cloud_top_pressure in conjunction with scattering for emission spectrum "
                          "is not recommended")

        star_irradiation_cos_angle = np.cos(np.deg2rad(star_irradiation_angle))  # flux

        if star_irradiation_cos_angle <= 0.:
            star_irradiation_cos_angle = 1e-8

        if stellar_intensities is None:
            if star_effective_temperature is not None and orbit_semi_major_axis is not None:
                stellar_intensities = self.compute_star_spectrum(
                    star_effective_temperature=star_effective_temperature,
                    orbit_semi_major_axis=orbit_semi_major_axis,
                    frequencies=self._frequencies,
                    star_radius=star_radius
                )
            else:
                stellar_intensities = np.zeros_like(self._frequencies)

        if reflectances is None:
            reflectances = np.zeros_like(self._frequencies)
        elif np.ndim(reflectances) == 0:
            reflectances = reflectances * np.ones_like(self._frequencies)
        elif np.size(reflectances) != self._frequencies.size:
            raise ValueError(f"reflectance must be a scalar "
                             f"or of the same size than frequencies ({self._frequencies.size}), "
                             f"but is of size {np.size(reflectances)}")

        if emissivities is None:
            emissivities = np.ones_like(self._frequencies)
        elif np.ndim(emissivities) == 0:
            emissivities = emissivities * np.ones_like(self._frequencies)
        elif np.size(emissivities) != self._frequencies.size:
            raise ValueError(f"emissivity must be a scalar "
                             f"or of the same size than frequencies ({self._frequencies.size}), "
                             f"but is of size {np.size(emissivities)}")

        optical_depths_rosseland = None
        photosphere_radius = None
        flux_clear = None

        (opacities, continuum_opacities_scattering, cloud_anisotropic_scattering_opacities, cloud_absorption_opacities,
         cloud_opacities, cloud_particles_mean_radii) = (
            self._calculate_opacities(
                temperatures=temperatures,
                mass_fractions=mass_fractions,
                mean_molar_masses=mean_molar_masses,
                reference_gravity=reference_gravity,
                opaque_cloud_top_pressure=opaque_cloud_top_pressure,
                cloud_particles_mean_radii=cloud_particles_mean_radii,
                cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
                cloud_particles_radius_distribution=cloud_particles_radius_distribution,
                cloud_hansen_a=cloud_hansen_a,
                cloud_hansen_b=cloud_hansen_b,
                cloud_particle_number_density_grid=cloud_particle_number_density_grid,
                clouds_particles_porosity_factor=clouds_particles_porosity_factor,
                cloud_f_sed=cloud_f_sed,
                eddy_diffusion_coefficients=eddy_diffusion_coefficients,
                haze_factor=haze_factor,
                power_law_opacity_350nm=power_law_opacity_350nm,
                power_law_opacity_coefficient=power_law_opacity_coefficient,
                gray_opacity=gray_opacity,
                cloud_photosphere_median_optical_depth=cloud_photosphere_median_optical_depth,
                cloud_fraction=cloud_fraction,
                complete_coverage_clouds=complete_coverage_clouds,
                return_cloud_contribution=return_cloud_contribution,
                additional_absorption_opacities_function=additional_absorption_opacities_function,
                additional_scattering_opacities_function=additional_scattering_opacities_function,
                target_particle_radii=target_particle_radii,
                target_pressure=target_pressure
            )
        )

        if auto_anisotropic_cloud_scattering:
            self._anisotropic_cloud_scattering = 'auto'

        (_opacities, _continuum_opacities_scattering,
         _cloud_anisotropic_scattering_opacities, _cloud_absorption_opacities) = self.__get_base_opacities(
            use_smart_clear_opacities=self.__use_smart_clear_opacities,
            complete_coverage_clouds=complete_coverage_clouds,
            opacities=opacities,
            continuum_opacities_scattering=continuum_opacities_scattering,
            cloud_anisotropic_scattering_opacities=cloud_anisotropic_scattering_opacities,
            cloud_absorption_opacities=cloud_absorption_opacities
        )

        flux, emission_contribution, optical_depths, opacities_rosseland, relative_cloud_scaling_factor = (
            self._calculate_flux(
                temperatures=temperatures,
                reference_gravity=reference_gravity,
                opacities=_opacities,
                continuum_opacities_scattering=_continuum_opacities_scattering,
                emission_geometry=irradiation_geometry,
                star_irradiation_cos_angle=star_irradiation_cos_angle,
                stellar_intensity=stellar_intensities,
                reflectances=reflectances,
                emissivities=emissivities,
                return_contribution=return_contribution,
                cloud_f_sed=cloud_f_sed,
                photospheric_cloud_optical_depths=cloud_photosphere_median_optical_depth,
                cloud_photosphere_wavelength_boundaries=cloud_photosphere_wavelength_boundaries,
                cloud_anisotropic_scattering_opacities=_cloud_anisotropic_scattering_opacities,
                cloud_absorption_opacities=_cloud_absorption_opacities,
                adaptive_feautrier_iterations=adaptive_feautrier_iterations,
                return_rosseland_opacities=return_rosseland_optical_depths
            )
        )

        if self.__use_smart_clear_opacities:
            # Swap cloudy opacities with clear opacities
            opacities[0][:, :, 0], opacities[1] = opacities[1], opacities[0][:, :, 0]

            if not return_opacities and not return_photosphere_radius:  # get rid of the now useless cloudy case
                opacities = opacities[0]
                _opacities = opacities
            else:
                _opacities = opacities[0]

            _continuum_opacities_scattering = continuum_opacities_scattering[1]  # use the clear scattering opacities

            if len(complete_coverage_clouds) > 0:  # use the opacities with only the clouds with complete coverage
                _cloud_anisotropic_scattering_opacities = cloud_anisotropic_scattering_opacities[1]
                _cloud_absorption_opacities = cloud_absorption_opacities[1]

            flux_clear, _, _, _, _ = (
                self._calculate_flux(
                    temperatures=temperatures,
                    reference_gravity=reference_gravity,
                    opacities=_opacities,
                    continuum_opacities_scattering=_continuum_opacities_scattering,
                    emission_geometry=irradiation_geometry,
                    star_irradiation_cos_angle=star_irradiation_cos_angle,
                    stellar_intensity=stellar_intensities,
                    reflectances=reflectances,
                    emissivities=emissivities,
                    return_contribution=return_contribution,
                    cloud_f_sed=cloud_f_sed,
                    cloud_photosphere_wavelength_boundaries=cloud_photosphere_wavelength_boundaries,
                    photospheric_cloud_optical_depths=cloud_photosphere_median_optical_depth,
                    cloud_anisotropic_scattering_opacities=cloud_anisotropic_scattering_opacities,
                    cloud_absorption_opacities=cloud_absorption_opacities,
                    adaptive_feautrier_iterations=adaptive_feautrier_iterations,
                    return_rosseland_opacities=return_rosseland_optical_depths
                )
            )

            flux = (
                cloud_fraction * flux
                + (1 - cloud_fraction) * flux_clear
            )

            if not return_clear_spectrum:
                flux_clear = None

            if return_opacities or return_photosphere_radius:
                # Swap clear opacities with cloudy opacities
                opacities[0][:, :, 0], opacities[1] = opacities[1], opacities[0][:, :, 0]
                _opacities = opacities[0]
                _continuum_opacities_scattering = continuum_opacities_scattering[0]

                if len(complete_coverage_clouds) > 0:
                    _cloud_anisotropic_scattering_opacities = cloud_anisotropic_scattering_opacities[0]
                    _cloud_absorption_opacities = cloud_absorption_opacities[0]

        if planet_radius is not None and return_photosphere_radius:
            photosphere_radius = self.calculate_photosphere_radius(
                temperatures=temperatures,
                mean_molar_masses=mean_molar_masses,
                reference_gravity=reference_gravity,
                planet_radius=planet_radius,
                opacities=_opacities,
                continuum_opacities_scattering=_continuum_opacities_scattering,
                cloud_f_sed=cloud_f_sed,
                cloud_photosphere_median_optical_depth=cloud_photosphere_median_optical_depth,
                cloud_photosphere_wavelength_boundaries=cloud_photosphere_wavelength_boundaries,
                cloud_anisotropic_scattering_opacities=_cloud_anisotropic_scattering_opacities,
                cloud_absorption_opacities=_cloud_absorption_opacities,
                optical_depths=optical_depths
            )

        if self.__clouds_have_effect(mass_fractions, cloud_fraction) and return_cloud_contribution:
            cloud_contribution = self._compute_species_optical_depths(
                reference_gravity=reference_gravity,
                pressures=self._pressures,
                cloud_opacities=cloud_opacities
            )
        else:
            cloud_contribution = None

        if self.__sum_opacities and len(self._line_species) > 1:
            if self.__sum_opacities and opacities_rosseland is not None:
                optical_depths_rosseland = self._compute_species_optical_depths(
                    reference_gravity=reference_gravity,
                    pressures=self._pressures,
                    cloud_opacities=opacities_rosseland.reshape(1, 1, 1, len(self._pressures))
                ).reshape(len(self._pressures))

        additional_outputs = {}

        if cloud_particles_mean_radii is not None:
            additional_outputs['cloud_particles_mean_radii'] = cloud_particles_mean_radii

        if return_particles_radii_bins:
            additional_outputs['particles_radii_bins'] = self._clouds_loaded_opacities['particles_radii_bins']

        if return_particles_radii:
            additional_outputs['particles_radii'] = self._clouds_loaded_opacities['particles_radii']

        if stellar_intensities is not None:
            additional_outputs['stellar_intensities'] = stellar_intensities

        if return_contribution:
            additional_outputs['emission_contribution'] = emission_contribution

        if return_clear_spectrum:
            if frequencies_to_wavelengths:
                additional_outputs['clear_spectrum'] = flux_hz2flux_cm(
                    flux_hz=flux_clear,
                    frequency=self._frequencies
                )
            else:
                additional_outputs['clear_spectrum'] = flux_clear

        if return_photosphere_radius:
            additional_outputs['photosphere_radius'] = photosphere_radius

        if return_rosseland_optical_depths:
            additional_outputs['opacities_rosseland'] = opacities_rosseland
            additional_outputs['optical_depths_rosseland'] = optical_depths_rosseland

        if return_cloud_contribution:
            additional_outputs['cloud_contribution'] = cloud_contribution

        if return_opacities:
            additional_outputs['opacities'] = opacities
            additional_outputs['continuum_opacities_scattering'] = continuum_opacities_scattering
            additional_outputs['cloud_absorption_opacities'] = cloud_absorption_opacities
            additional_outputs['cloud_anisotropic_scattering_opacities'] = cloud_anisotropic_scattering_opacities
            additional_outputs['optical_depths'] = optical_depths
        if return_abundances:
            additional_outputs['abundances'] = mass_fractions
        if relative_cloud_scaling_factor is not None:
            additional_outputs['relative_cloud_scaling_factor'] = relative_cloud_scaling_factor

        if frequencies_to_wavelengths:
            return (
                frequency2wavelength(self._frequencies),
                flux_hz2flux_cm(
                    flux_hz=flux,
                    frequency=self._frequencies
                ),
                additional_outputs
            )
        else:
            return self._frequencies, flux, additional_outputs

    def calculate_photosphere_radius(
        self,
        temperatures: npt.NDArray[np.floating],
        mean_molar_masses: npt.NDArray[np.floating],
        reference_gravity: float,
        planet_radius: float,
        opacities: npt.NDArray[np.floating],
        continuum_opacities_scattering: npt.NDArray[np.floating],
        cloud_f_sed: float,
        cloud_photosphere_median_optical_depth: float,
        cloud_anisotropic_scattering_opacities: npt.NDArray[np.floating],
        cloud_absorption_opacities: npt.NDArray[np.floating],
        optical_depths: npt.NDArray[np.floating] = None,
        cloud_photosphere_wavelength_boundaries: npt.NDArray[np.floating] = None,
    ) -> npt.NDArray[np.floating]:
        """Calculate the photosphere radius.
        TODO complete docstring
        Args:
            temperatures:
            mean_molar_masses:
            reference_gravity:
            planet_radius:
            opacities:
            continuum_opacities_scattering:
            cloud_f_sed:
            cloud_photosphere_wavelength_boundaries:
            cloud_photosphere_median_optical_depth:
            cloud_anisotropic_scattering_opacities:
            cloud_absorption_opacities:
            optical_depths:
        Returns:

        """
        self.__set_sum_opacities(emission=True)

        radius_hydrostatic_equilibrium = self._compute_radius_hydrostatic_equilibrium(
            pressures=self._pressures * 1e-6,
            temperatures=temperatures,
            mean_molar_masses=mean_molar_masses,
            reference_gravity=reference_gravity,
            planet_radius=planet_radius,
            reference_pressure=self._pressures[-1] * 1e-6,
        )

        radius_interp = interp1d(self._pressures, radius_hydrostatic_equilibrium)

        photosphere_radius = np.zeros(self._frequencies.size, dtype=float)

        if self.__sum_opacities:
            if optical_depths is None:
                optical_depths, _, _ = self._compute_optical_depths_wrapper(
                    pressures=self._pressures,
                    reference_gravity=reference_gravity,
                    opacities=opacities,
                    continuum_opacities_scattering=continuum_opacities_scattering,
                    scattering_in_emission=self._scattering_in_emission,
                    sum_opacities=self.__sum_opacities,
                    absorber_present=self.__absorber_present,
                    # Custom cloud parameters
                    frequencies=self._frequencies,
                    weights_gauss=self._lines_loaded_opacities['weights_gauss'],
                    cloud_wavelengths=cloud_photosphere_wavelength_boundaries,
                    cloud_f_sed=cloud_f_sed,
                    cloud_anisotropic_scattering_opacities=cloud_anisotropic_scattering_opacities,
                    cloud_absorption_opacities=cloud_absorption_opacities,
                    photospheric_cloud_optical_depths=cloud_photosphere_median_optical_depth
                )

            weights_gauss_reshape = self._lines_loaded_opacities['weights_gauss'].reshape(
                len(self._lines_loaded_opacities['weights_gauss']), 1
            )

            for i_freq in range(self._frequencies.size):
                tau_p = np.sum(weights_gauss_reshape * optical_depths[:, i_freq, 0, :], axis=0)
                pressures_tau_p = interp1d(tau_p, self._pressures)
                if np.max(tau_p) > 2. / 3.:
                    photosphere_radius[i_freq] = radius_interp(pressures_tau_p(2 / 3))
                else:
                    warnings.warn('Atmosphere is too transparent for calculating the "photospheric '
                                  'radius" at tau = 2/3. Returning the radius at maximum pressure instead!')
                    photosphere_radius[i_freq] = radius_hydrostatic_equilibrium[-1]

        return photosphere_radius

    def calculate_rosseland_planck_opacities(
        self,
        temperatures: npt.NDArray[np.floating],
        mass_fractions: dict[str, npt.NDArray[np.floating]],
        mean_molar_masses: npt.NDArray[np.floating],
        reference_gravity: float,
        opaque_cloud_top_pressure: float = None,
        cloud_particles_mean_radii: dict[str, npt.NDArray[np.floating]] = None,
        cloud_particle_radius_distribution_std: float = None,
        cloud_particles_radius_distribution: str = 'lognormal',
        cloud_hansen_a: float = None,
        cloud_hansen_b: float = None,
        clouds_particles_porosity_factor: dict[str, float] = None,
        cloud_f_sed: float = None,
        eddy_diffusion_coefficients: float = None,
        haze_factor: float = 1.0,
        power_law_opacity_350nm: float = None,
        power_law_opacity_coefficient: float = None,
        gray_opacity: float = None
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], dict[str, Any]]:
        """ Method to calculate the atmosphere's Rosseland and Planck mean opacities.

            Args:
                temperatures:
                    the atmospheric temperature in K, at each atmospheric layer
                    (1-d numpy array, same length as pressure array).
                mass_fractions:
                    dictionary of mass fractions for all atmospheric absorbers.
                    Dictionary keys are the species names.
                    Every mass fraction array
                    has same length as pressure array.
                mean_molar_masses:
                    the atmospheric mean molecular weight in amu,
                    at each atmospheric layer
                    (1-d numpy array, same length as pressure array).
                reference_gravity (float):
                    Surface gravity in cgs. Vertically constant for emission
                    spectra.
                opaque_cloud_top_pressure (Optional[float]):
                    Pressure, in bar, where opaque cloud deck is added to the
                    absorption opacity.
                cloud_particles_mean_radii (Optional):
                    dictionary of mean particle radii for all cloud species.
                    Dictionary keys are the cloud species names.
                    Every radius array has same length as pressure array.
                cloud_particle_radius_distribution_std (Optional[float]):
                    width of the log-normal cloud particle size distribution
                cloud_particles_radius_distribution (Optional[string]):
                    The cloud particle size distribution to use.
                    Can be either 'lognormal' (default) or 'hansen'.
                    If hansen, the cloud_hansen_b parameters must be used.
                cloud_hansen_a (Optional[dict]):
                    A dictionary of the 'a' parameter values for each
                    included cloud species and for each atmospheric layer,
                    formatted as the kzz argument. Equivalent to radius arg.
                    If cloud_hansen_a is not included and dist is "hansen", then it will
                    be computed using Kzz and fsed (recommended).
                cloud_hansen_b (Optional[dict]):
                    A dictionary of the 'b' parameter values for each
                    included cloud species and for each atmospheric layer,
                    formatted as the kzz argument. This is the width of the hansen
                    distribution normalized by the particle area (1/cloud_hansen_a^2)
                clouds_particles_porosity_factor (Optional[dict]):
                    A dictionary containing the particle porosity factor for each cloud species.
                cloud_f_sed (Optional[float]):
                    cloud settling parameter
                eddy_diffusion_coefficients (Optional):
                    the atmospheric eddy diffusion coefficient in cgs
                    (i.e. :math:`\\rm cm^2/s`),
                    at each atmospheric layer
                    (1-d numpy array, same length as pressure array).
                haze_factor (Optional[float]):
                    Scalar factor, increasing the gas Rayleigh scattering
                    cross-section.
                power_law_opacity_350nm (Optional[float]):
                    Scattering opacity at 0.35 micron, in cgs units (cm^2/g).
                power_law_opacity_coefficient (Optional[float]):
                    Has to be given if kappa_zero is defined, this is the
                    wavelength powerlaw index of the parametrized scattering
                    opacity.
                gray_opacity (Optional[float]):
                    Gray opacity value, to be added to the opacity at all
                    pressures and wavelengths (units :math:`\\rm cm^2/g`)
        """
        # TODO should be 2 separated functions
        if not self._scattering_in_emission:
            raise ValueError(
                "pRT must run in scattering_in_emission = True mode to calculate kappa_Rosseland and kappa_Planck'"
            )

        auto_anisotropic_cloud_scattering = False

        if self._anisotropic_cloud_scattering == 'auto':
            self._anisotropic_cloud_scattering = True
            auto_anisotropic_cloud_scattering = True
        elif not self._anisotropic_cloud_scattering:
            warnings.warn(f"anisotropic cloud scattering is recommended for emission spectra, "
                          f"but 'anisotropic_cloud_scattering' was set to {self._anisotropic_cloud_scattering}; "
                          f"set it to 'auto' to disable this warning")

        # No photospheric cloud in Rosseland or Planck opacities
        opacities, continuum_opacities_scattering, _, _, _, cloud_particles_mean_radii = (
            self._calculate_opacities(
                temperatures=temperatures,
                mass_fractions=mass_fractions,
                mean_molar_masses=mean_molar_masses,
                reference_gravity=reference_gravity,
                gray_opacity=gray_opacity,
                haze_factor=haze_factor,
                opaque_cloud_top_pressure=opaque_cloud_top_pressure,
                power_law_opacity_350nm=power_law_opacity_350nm,
                power_law_opacity_coefficient=power_law_opacity_coefficient,
                cloud_photosphere_median_optical_depth=None,
                cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
                cloud_f_sed=cloud_f_sed,
                eddy_diffusion_coefficients=eddy_diffusion_coefficients,
                cloud_particles_mean_radii=cloud_particles_mean_radii,
                cloud_particles_radius_distribution=cloud_particles_radius_distribution,
                cloud_hansen_a=cloud_hansen_a,
                cloud_hansen_b=cloud_hansen_b,
                clouds_particles_porosity_factor=clouds_particles_porosity_factor
            )
        )

        if auto_anisotropic_cloud_scattering:
            self._anisotropic_cloud_scattering = 'auto'

        opacities_rosseland = (
            self._compute_rosseland_opacities(
                frequency_bins_edges=self._frequency_bins_edges,
                temperatures=temperatures,
                weights_gauss=self._lines_loaded_opacities['weights_gauss'],
                opacities=opacities[:, :, :1, :],
                continuum_opacities_scattering=continuum_opacities_scattering,
                scattering_in_emission=self._scattering_in_emission
            )
        )

        opacities_planck = (
            fcore.compute_planck_opacities(
                opacities[:, :, :1, :],
                temperatures,
                self._lines_loaded_opacities['weights_gauss'],
                self._frequency_bins_edges,
                self._scattering_in_emission,
                continuum_opacities_scattering
            )
        )

        return opacities_rosseland, opacities_planck, cloud_particles_mean_radii

    def calculate_transit_radii(
        self,
        temperatures: npt.NDArray[np.floating],
        mass_fractions: dict[str, npt.NDArray[np.floating]],
        mean_molar_masses: npt.NDArray[np.floating],
        reference_gravity: float,
        reference_pressure: float,
        planet_radius: float,
        variable_gravity: bool = True,
        opaque_cloud_top_pressure: float = None,
        cloud_particles_mean_radii: dict[str, npt.NDArray[np.floating]] = None,
        cloud_particle_radius_distribution_std: float = None,
        cloud_particles_radius_distribution: str = 'lognormal',
        cloud_hansen_a: float = None,
        cloud_hansen_b: float = None,
        cloud_particle_number_density_grid: dict[str, npt.NDArray[np.floating]] = None,
        clouds_particles_porosity_factor: dict[str, float] = None,
        cloud_f_sed: float = None,
        eddy_diffusion_coefficients: float = None,
        haze_factor: float = 1.0,
        power_law_opacity_350nm: float = None,
        power_law_opacity_coefficient: float = None,
        gray_opacity: float = None,
        cloud_fraction: float = 1.0,
        complete_coverage_clouds: list[str] = None,
        additional_absorption_opacities_function: Callable = None,
        additional_scattering_opacities_function: Callable = None,
        frequencies_to_wavelengths: bool = True,
        return_contribution: bool = False,
        return_clear_spectrum: bool = False,
        return_cloud_contribution: bool = False,
        return_radius_hydrostatic_equilibrium: bool = False,
        return_opacities: bool = False,
        return_abundances: bool = False,
        return_particles_radii_bins: bool = False,
        return_particles_radii: bool = False,
        return_particles_densities: bool = False,
        target_particle_radii: npt.NDArray[np.floating] = None,
        target_pressure: npt.NDArray[np.floating] = None,
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], dict[str, Any]]:
        """Method to calculate the atmosphere's transmission radius
        (for the transmission spectrum).

            Args:
                temperatures:
                    the atmospheric temperature in K, at each atmospheric layer
                    (1-d numpy array, same length as pressure array).
                mass_fractions:
                    dictionary of mass fractions for all atmospheric absorbers.
                    Dictionary keys are the species names.
                    Every mass fraction array
                    has same length as pressure array.
                mean_molar_masses:
                    the atmospheric mean molecular weight in amu,
                    at each atmospheric layer
                    (1-d numpy array, same length as pressure array).
                reference_gravity (float):
                    Surface gravity in cgs at reference radius and pressure.
                reference_pressure (float):
                    Reference pressure P0 in bar where R(P=P0) = R_pl,
                    where R_pl is the reference radius (parameter of this
                    method), and g(P=P0) = gravity, where gravity is the
                    reference gravity (parameter of this method)
                planet_radius (float):
                    Reference radius R_pl, in cm.
                variable_gravity (Optional[bool]):
                    Standard is ``True``. If ``False`` the gravity will be
                    constant as a function of pressure, during the transmission
                    radius calculation.
                opaque_cloud_top_pressure (Optional[float]):
                    Pressure, in bar, where opaque cloud deck is added to the
                    absorption opacity.
                cloud_particles_mean_radii (Optional):
                    dictionary of mean particle radii for all cloud species.
                    Dictionary keys are the cloud species names.
                    Every radius array has same length as pressure array.
                cloud_particle_radius_distribution_std (Optional[float]):
                    width of the log-normal cloud particle size distribution
                cloud_particles_radius_distribution (Optional[string]):
                    The cloud particle size distribution to use.
                    Can be either 'lognormal' (default) or 'hansen'.
                    If hansen, the cloud_hansen_b parameters must be used.
                cloud_hansen_a (Optional[dict]):
                    A dictionary of the 'a' parameter values for each
                    included cloud species and for each atmospheric layer,
                    formatted as the kzz argument. Equivalent to radius arg.
                    If cloud_hansen_a is not included and dist is "hansen", then it will
                    be computed using Kzz and fsed (recommended).
                cloud_hansen_b (Optional[dict]):
                    A dictionary of the 'b' parameter values for each
                    included cloud species and for each atmospheric layer,
                    formatted as the kzz argument. This is the width of the hansen
                    distribution normalized by the particle area (1/cloud_hansen_a^2)
                cloud_particle_number_density_grid (Optional[dict]):
                    EXPERIMENTAL
                clouds_particles_porosity_factor (Optional[dict]):
                    A dictionary of porosity factors depending on the cloud species. This can be useful when opacities
                    are calculated using the Distribution of Hollow Spheres (DHS) method.
                cloud_f_sed (Optional[float]):
                    cloud settling parameter
                eddy_diffusion_coefficients (Optional):
                    the atmospheric eddy diffusion coefficient in cgs
                    (i.e. :math:`\\rm cm^2/s`),
                    at each atmospheric layer
                    (1-d numpy array, same length as pressure array).
                haze_factor (Optional[float]):
                    Scalar factor, increasing the gas Rayleigh scattering
                    cross-section.
                power_law_opacity_350nm (Optional[float]):
                    Scattering opacity at 0.35 micron, in cgs units (cm^2/g).
                power_law_opacity_coefficient (Optional[float]):
                    Has to be given if kappa_zero is defined, this is the
                    wavelength powerlaw index of the parametrized scattering
                    opacity.
                gray_opacity (Optional[float]):
                    Gray opacity value, to be added to the opacity at all
                    pressures and wavelengths (units :math:`\\rm cm^2/g`)
                cloud_fraction:
                    Mix a column without cloud with a column with cloud to the requested fraction
                    (0 = clear, 1 = full cloud cover). By default, assume a full cloud cover.
                complete_coverage_clouds:
                    List of clouds that have full coverage, regardless of the value of cloud_fraction.
                additional_absorption_opacities_function (Optional[function]):
                    A python function that takes wavelength arrays in microns and pressure arrays in bars
                    as input, and returns an absorption opacity matrix in units of cm^2/g, in the shape of
                    number of wavelength points x number of pressure points.
                    This opacity will then be added to the atmospheric absorption opacity.
                    This must not be used to add atomic / molecular line opacities in low-resolution mode (c-k),
                    because line opacities require a proper correlated-k treatment.
                    It may be used to add simple cloud absorption laws, for example, which
                    have opacities that vary only slowly with wavelength, such that the current
                    model resolution is sufficient to resolve any variations.
                additional_scattering_opacities_function (Optional[function]):
                    A python function that takes wavelength arrays in microns and pressure arrays in bars
                    as input, and returns an isotropic scattering opacity matrix in units of cm^2/g, in the shape of
                    number of wavelength points x number of pressure points.
                    This opacity will then be added to the atmospheric absorption opacity.
                    It may be used to add simple cloud absorption laws, for example, which
                    have opacities that vary only slowly with wavelength, such that the current
                    model resolution is sufficient to resolve any variations.
                frequencies_to_wavelengths (Optional[bool]):
                    If True, convert the frequencies (Hz) output to wavelengths (cm).
                return_contribution (Optional[bool]):
                    If True the transmission and emission contribution function is calculated.
                return_clear_spectrum (Optional[bool]):
                    If True, return the clear spectrum in addition to a cloudy spectrum.
                return_cloud_contribution (Optional[bool]):
                    If True, calculate and return the cloud contribution.
                return_radius_hydrostatic_equilibrium (Optional[bool]):
                    If True, return the radius at hydrostatic equilibrium of the planet.
                return_opacities (Optional[bool]):
                    If True, return the absorption opacities and scattering opacities.
                return_abundances (Optional[bool]):
                    If True, return the mass fractions.
                return_particles_radii_bins (Optional[bool]):
                    If True, return the particles radii bins.
                return_particles_radii (Optional[bool]):
                    If True, return the particles radii.
                return_particles_densities (Optional[bool]):
                    If True, return the particles densities.
                target_particle_radii (Optional[npt.NDArray[np.floating]]):
                    # TODO complete docstring
                target_pressure (Optional[npt.NDArray[np.floating]]):
                    # TODO complete docstring
        """
        auto_anisotropic_cloud_scattering, complete_coverage_clouds, transit_radii_clear = (
            self.__init_spectral_function(
                is_emission=False,
                reference_gravity=reference_gravity,
                complete_coverage_clouds=complete_coverage_clouds,
                return_clear_spectrum=return_clear_spectrum,
                mass_fractions=mass_fractions,
                cloud_fraction=cloud_fraction,
                opaque_cloud_top_pressure=opaque_cloud_top_pressure
            )
        )
        copy_opacities = False

        if return_contribution:
            copy_opacities = True  # opacities need to be copied only if they are used for contribution
        elif self.__use_smart_clear_opacities and not self.__sum_opacities:
            copy_opacities = True  # opacities need to be copied when using cloud coverage in c-k mode

        # No photospheric clouds in transmission
        opacities, continuum_opacities_scattering, _, _, _, cloud_particles_mean_radii = self._calculate_opacities(
            temperatures=temperatures,
            mass_fractions=mass_fractions,
            mean_molar_masses=mean_molar_masses,
            reference_gravity=reference_gravity,
            gray_opacity=gray_opacity,
            haze_factor=haze_factor,
            opaque_cloud_top_pressure=opaque_cloud_top_pressure,
            power_law_opacity_350nm=power_law_opacity_350nm,
            power_law_opacity_coefficient=power_law_opacity_coefficient,
            cloud_photosphere_median_optical_depth=None,
            cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
            cloud_f_sed=cloud_f_sed,
            eddy_diffusion_coefficients=eddy_diffusion_coefficients,
            cloud_particles_mean_radii=cloud_particles_mean_radii,
            cloud_particles_radius_distribution=cloud_particles_radius_distribution,
            cloud_hansen_a=cloud_hansen_a,
            cloud_hansen_b=cloud_hansen_b,
            cloud_particle_number_density_grid=cloud_particle_number_density_grid,
            clouds_particles_porosity_factor=clouds_particles_porosity_factor,
            cloud_fraction=cloud_fraction,
            complete_coverage_clouds=complete_coverage_clouds,
            return_cloud_contribution=return_cloud_contribution,
            additional_absorption_opacities_function=additional_absorption_opacities_function,
            additional_scattering_opacities_function=additional_scattering_opacities_function,
            target_particle_radii=target_particle_radii,
            target_pressure=target_pressure
        )

        # Start filling additional outputs here to get rid of unnecessary variables ASAP
        additional_outputs = {}

        if return_opacities:
            additional_outputs['opacities'] = copy.deepcopy(opacities)
            additional_outputs['continuum_opacities_scattering'] = copy.deepcopy(continuum_opacities_scattering)

        if cloud_particles_mean_radii is not None:
            additional_outputs['cloud_particles_mean_radii'] = cloud_particles_mean_radii

        if return_particles_radii_bins:
            additional_outputs['particles_radii_bins'] = self._clouds_loaded_opacities['particles_radii_bins']

        if return_particles_radii:
            additional_outputs['particles_radii'] = self._clouds_loaded_opacities['particles_radii']

        if return_particles_densities:
            additional_outputs['particles_densities'] = self._clouds_loaded_opacities['particles_densities']

        del cloud_particles_mean_radii

        if auto_anisotropic_cloud_scattering:
            self._anisotropic_cloud_scattering = 'auto'

        _opacities, _continuum_opacities_scattering, _, _ = self.__get_base_opacities(
            use_smart_clear_opacities=self.__use_smart_clear_opacities,
            complete_coverage_clouds=complete_coverage_clouds,
            opacities=opacities,
            continuum_opacities_scattering=continuum_opacities_scattering,
            cloud_anisotropic_scattering_opacities=None,
            cloud_absorption_opacities=None
        )

        # Calculate the transit radii
        transit_radii, radius_hydrostatic_equilibrium, transmission_contribution = self._calculate_transit_radii(
            temperatures=temperatures,
            mean_molar_masses=mean_molar_masses,
            reference_gravity=reference_gravity,
            reference_pressure=reference_pressure,
            planet_radius=planet_radius,
            variable_gravity=variable_gravity,
            opacities=_opacities,
            continuum_opacities_scattering=_continuum_opacities_scattering,
            copy_opacities=copy_opacities,
            return_contributions=return_contribution
        )

        if self.__use_smart_clear_opacities:
            # Swap cloudy opacities with clear opacities
            opacities[0][:, :, 0] = opacities[1]
            opacities = opacities[0]  # get rid of the now useless cloudy case
            continuum_opacities_scattering = continuum_opacities_scattering[1]  # use the clear scattering opacities

            transit_radii_clear, _, _ = self._calculate_transit_radii(
                temperatures=temperatures,
                mean_molar_masses=mean_molar_masses,
                reference_gravity=reference_gravity,
                reference_pressure=reference_pressure,
                planet_radius=planet_radius,
                variable_gravity=variable_gravity,
                opacities=opacities,
                continuum_opacities_scattering=continuum_opacities_scattering,
                copy_opacities=False,
                return_contributions=False
            )

            transit_radii = (
                    cloud_fraction * transit_radii
                    + (1 - cloud_fraction) * transit_radii_clear
            )

            if not return_clear_spectrum:
                transit_radii_clear = None

        del opacities

        # Fill last additional parameters
        if return_contribution:
            additional_outputs['transmission_contribution'] = transmission_contribution

        if return_clear_spectrum:
            if frequencies_to_wavelengths:
                additional_outputs['clear_spectrum'] = flux_hz2flux_cm(
                    flux_hz=transit_radii_clear,
                    frequency=self._frequencies
                )
            else:
                additional_outputs['clear_spectrum'] = transit_radii_clear

        if return_radius_hydrostatic_equilibrium:
            additional_outputs['radius_hydrostatic_equilibrium'] = radius_hydrostatic_equilibrium

        if return_abundances:
            additional_outputs['abundances'] = mass_fractions  # TODO [4.0.0] rename 'abundances' to 'mass_fractions' for consistency # noqa: E501

        if frequencies_to_wavelengths:
            return (
                frequency2wavelength(self._frequencies),
                transit_radii,
                additional_outputs
            )
        else:
            return self._frequencies, transit_radii, additional_outputs

    @staticmethod
    def compute_pressure_hydrostatic_equilibrium(
            mean_molar_masses, reference_gravity, planet_radius, p0, temperature, radii, rk4=True):
        # TODO is it used?
        p0 = p0 * 1e6
        vs = 1. / radii
        r_pl_sq = planet_radius ** 2

        def integrand(press):
            temp = temperature(press)
            mu = mean_molar_masses(press / 1e6, temp)
            if not np.isscalar(mu):
                mu = mu[0]

            integral = mu * cst.amu * reference_gravity * r_pl_sq / cst.kB / temp
            return integral

        pressures = [p0]
        dvs = np.diff(vs)
        press1 = p0
        chi1 = np.log(p0)
        for dv in dvs:
            k1 = integrand(press1)

            if rk4:
                chi2 = chi1 + 0.5 * dv * k1
                press2 = np.exp(chi2)
                k2 = integrand(press2)
                chi3 = chi1 + 0.5 * dv * k2
                press3 = np.exp(chi3)
                k3 = integrand(press3)
                chi4 = chi1 + dv * k3
                press4 = np.exp(chi4)
                k4 = integrand(press4)
                chi1 = chi1 + 1. / 6. * (k1 + 2 * k2 + 2 * k3 + k4) * dv
            else:
                chi1 = chi1 + dv * k1

            press1 = np.exp(chi1)
            pressures.append(press1)

        return np.array(pressures) / 1e6

    @staticmethod
    def compute_star_spectrum(star_effective_temperature, orbit_semi_major_axis, frequencies, star_radius=None):
        """Method to get the PHOENIX spectrum of the star and rebin it
        to the wavelength points. If t_star is not explicitly written, the
        spectrum will be 0. If the distance is not explicitly written,
        the code will raise an error and break to urge the user to
        specify the value.

            Args:
                star_effective_temperature (float):
                    the stellar temperature in K.
                orbit_semi_major_axis (float):
                    the semi-major axis of the planet in cm.
                frequencies (float):
                    the frequencies on which to interpolate the spectrum in Hz.
                star_radius (float):
                    if specified, uses this radius in cm
                    to scale the flux, otherwise it uses PHOENIX radius.
        """
        from petitRADTRANS.stellar_spectra.phoenix import phoenix_star_table

        if star_radius is not None:
            star_spectrum, _ = phoenix_star_table.compute_spectrum(star_effective_temperature)
            _star_radius = star_radius
        else:
            star_spectrum, _star_radius = phoenix_star_table.compute_spectrum(star_effective_temperature)

        stellar_intensity = Radtrans.rebin_star_spectrum(
            star_spectrum=star_spectrum[:, 1],
            star_wavelengths=star_spectrum[:, 0],
            wavelengths=frequency2wavelength(frequencies)
        )

        stellar_intensity = stellar_intensity / np.pi * (star_radius / orbit_semi_major_axis) ** 2

        return stellar_intensity

    def get_wavelengths(self):
        """Convert the frequency grid into wavelengths in centimeter."""
        # Not a property because the child SpectralModel already have a 'wavelengths' attribute that behaves differently
        return frequency2wavelength(self._frequencies)

    def load_all_opacities(self):
        print("Loading Radtrans opacities...")

        # Load line opacities
        self.load_line_opacities(self._path_input_data)

        # Read continuum opacities
        # CIA
        if len(self._gas_continuum_contributors) > 0:
            self.load_cia_opacities(self._path_input_data)

        # Clouds
        if len(self._cloud_species) > 0:
            # Inherited from ReadOpacities in _read_opacities.py
            self.load_cloud_opacities(self._path_input_data)

        print("Successfully loaded all opacities")

    def load_cia_opacities(self, path_input_data):
        for collision in self._cias_loaded_opacities:
            if collision in Radtrans.__get_non_cia_gas_continuum_contributions():
                continue

            hdf5_file = self.__get_cia_aliases(collision)
            hdf5_file = CIAOpacity.find(
                path_input_data=path_input_data,
                species=hdf5_file
            )

            print(f" Loading CIA opacities for {collision} from file '{hdf5_file}'...", end='')

            (
                self._cias_loaded_opacities[collision]['molecules'],
                self._cias_loaded_opacities[collision]['weight'],
                self._cias_loaded_opacities[collision]['lambda'],
                self._cias_loaded_opacities[collision]['temperature'],
                self._cias_loaded_opacities[collision]['alpha']
            ) = self.load_cia_opacity(hdf5_file)

            print(" Done.")

        print(" Successfully loaded all CIA opacities")

    @staticmethod
    def load_cia_opacity(file_path_hdf5, return_radtrans_opacities=True):
        """Load a CIA opacity file."""
        with h5py.File(file_path_hdf5, 'r') as f:
            if not return_radtrans_opacities:
                return (
                    f['DOI'][()][0].decode('utf-8'),
                    f['mol_name'][:],
                    f['wavenumbers'][:],
                    f['alpha'][:],
                    f['t'][:]
                )

            wavelengths = 1 / f['wavenumbers'][:]  # cm-1 to cm
            wavelengths = wavelengths[::-1]  # correct ordering

            species = f['mol_name'][:]
            species = [s.decode('utf-8') for s in species]

            return (
                species,
                np.prod(f['mol_mass'][:]),  # weight
                wavelengths,
                f['t'][:],  # temperatures
                np.transpose(f['alpha'][:])[::-1, :]  # alpha (wavelength, temperature), correct ordering
            )

    def load_cloud_opacities(self, path_input_data):
        """ Load the cloud opacities.

        Args:
            path_input_data: path to petitRADTRANS input_data directory
        """
        hdf5_files = []
        internal_structures = []
        scattering_methods = []

        for i in range(len(self._cloud_species)):
            hdf5_file = CloudOpacity.get_aliases(self._cloud_species[i])

            hdf5_file = CloudOpacity.find(
                path_input_data=path_input_data,
                species=hdf5_file
            )

            internal_structures.append(hdf5_file.rsplit(')_', 1)[1].split('__', 1)[0])
            scattering_methods.append(hdf5_file.rsplit('__', 1)[1].split('.', 1)[0])

            hdf5_files.append(hdf5_file)

        clouds_particles_densities = np.zeros(len(hdf5_files))
        clouds_absorption_opacities = None
        clouds_scattering_opacities = None
        clouds_asymmetry_parameters = None
        cloud_wavelengths = None
        clouds_particles_radii_bins = None
        clouds_particles_radii = None

        for i, hdf5_file in enumerate(hdf5_files):
            print(f" Loading opacities of cloud species '{self._cloud_species[i]}' from file '{hdf5_file}' "
                  f"({internal_structures[i]}, using {scattering_methods[i]} scattering)...", end='')

            (
                _cloud_wavelengths,
                _cloud_particles_radii_bins,
                _cloud_particles_radii,
                _cloud_particles_densities,
                _cloud_absorption_opacities,
                _cloud_scattering_opacities,
                _cloud_asymmetry_parameters
            ) = self.load_cloud_opacity(hdf5_file)

            if i == 0:
                # Initialize cloud arrays
                cloud_wavelengths = _cloud_wavelengths
                clouds_particles_radii_bins = _cloud_particles_radii_bins
                clouds_particles_radii = _cloud_particles_radii

                clouds_absorption_opacities = np.zeros(
                    (clouds_particles_radii.size, cloud_wavelengths.size, len(hdf5_files))
                )
                clouds_scattering_opacities = np.zeros(
                    (clouds_particles_radii.size, cloud_wavelengths.size, len(hdf5_files))
                )
                clouds_asymmetry_parameters = np.zeros(
                    (clouds_particles_radii.size, cloud_wavelengths.size, len(hdf5_files))
                )

            clouds_particles_densities[i] = _cloud_particles_densities
            clouds_absorption_opacities[:, :, i] = _cloud_absorption_opacities
            clouds_scattering_opacities[:, :, i] = _cloud_scattering_opacities
            clouds_asymmetry_parameters[:, :, i] = _cloud_asymmetry_parameters

            print(" Done.")

        # Flip wavelengths/wavenumbers axis to match wavelengths ordering
        clouds_absorption_opacities = clouds_absorption_opacities[:, ::-1, :]
        clouds_scattering_opacities = clouds_scattering_opacities[:, ::-1, :]
        clouds_asymmetry_parameters = clouds_asymmetry_parameters[:, ::-1, :]

        clouds_absorption_opacities[clouds_absorption_opacities < 0.] = 0.
        clouds_scattering_opacities[clouds_scattering_opacities < 0.] = 0.

        self._clouds_loaded_opacities['particles_densities'] = (
            np.array(clouds_particles_densities, dtype='d', order='F')
        )
        self._clouds_loaded_opacities['absorption_opacities'] = (
            np.array(clouds_absorption_opacities, dtype='d', order='F')
        )
        self._clouds_loaded_opacities['scattering_opacities'] = (
            np.array(clouds_scattering_opacities, dtype='d', order='F')
        )
        self._clouds_loaded_opacities['particles_asymmetry_parameters'] = (
            np.array(clouds_asymmetry_parameters, dtype='d', order='F')
        )
        self._clouds_loaded_opacities['wavelengths'] = (
            np.array(cloud_wavelengths, dtype='d', order='F')
        )
        self._clouds_loaded_opacities['particles_radii_bins'] = (
            np.array(clouds_particles_radii_bins, dtype='d', order='F')
        )
        self._clouds_loaded_opacities['particles_radii'] = (
            np.array(clouds_particles_radii, dtype='d', order='F')
        )

        print(" Successfully loaded all clouds opacities")

    @staticmethod
    def load_cloud_opacity(file_path_hdf5, return_radtrans_opacities=True):
        """Load a cloud opacity file."""
        with h5py.File(file_path_hdf5, 'r') as f:
            if not return_radtrans_opacities:
                return (
                    f['wavenumbers'][:],
                    f['particle_radius_bins'][:],
                    f['particles_radii'][:],
                    f['particles_density'][()],
                    f['absorption_opacities'][:],
                    f['scattering_opacities'][:],
                    f['asymmetry_parameters'][:]
                )

            cloud_wavelengths = 1 / f['wavenumbers'][:]  # cm-1 to cm
            cloud_wavelengths = cloud_wavelengths[::-1]  # correct ordering

            return (
                cloud_wavelengths,
                f['particle_radius_bins'][:],
                f['particles_radii'][:],
                f['particles_density'][()],
                f['absorption_opacities'][:],
                f['scattering_opacities'][:],
                f['asymmetry_parameters'][:]
            )

    @classmethod
    def load_hdf5_ktables(
        cls, file_path_hdf5, frequencies, g_size, temperature_pressure_grid_size, return_radtrans_opacities=True
    ):
        """Load k-coefficient tables in HDF5 format, based on the ExoMol setup."""
        with h5py.File(file_path_hdf5, 'r') as f:
            k_table = np.array(f['kcoeff'])

            if not return_radtrans_opacities:
                return (
                    f['DOI'][()][0].decode('utf-8'),
                    f['bin_centers'][:],
                    f['bin_edges'][:],
                    k_table,
                    f['mol_mass'][()][0],
                    f['mol_name'][()][0].decode('utf-8'),
                    f['p'][:],
                    f['t'][:],
                    f['samples'][:],
                    f['weights'][:]
                )

            n_wavelengths = len(f['bin_centers'][:])
            _frequencies = cst.c * f['bin_centers'][:][::-1]
            n_temperatures = len(f['t'][:])
            n_pressures = len(f['p'][:])

            # Reshape k-table to petitRADTRANS format
            k_table = np.swapaxes(k_table, 0, 1)
            k_table = k_table.reshape((n_pressures * n_temperatures, n_wavelengths, 16))
            k_table = np.swapaxes(k_table, 0, 2)
            k_table = k_table[:, ::-1, :]

            indices_opacities, indices_file = Radtrans.__get_frequency_grids_intersection_indices(
                frequencies=frequencies,
                file_frequencies=_frequencies,
                tolerate_grid_misalignment=True
            )

            if np.nonzero(indices_file)[0].size != np.nonzero(indices_opacities)[0].size:
                grid_misalignment: bool = False

                if np.nonzero(indices_file)[0].size == frequencies.size - 1:
                    indices_file = cls._handle_grid_misalignment(
                        indices=indices_file,
                        frequencies_reference=frequencies,
                        frequencies_test=_frequencies
                    )
                    grid_misalignment = True
                elif np.nonzero(indices_opacities)[0].size == np.nonzero(indices_file)[0].size - 1:
                    indices_opacities = cls._handle_grid_misalignment(
                        indices=indices_opacities,
                        frequencies_reference=_frequencies,
                        frequencies_test=frequencies
                    )
                    grid_misalignment = True

                if grid_misalignment:
                    max_relative_difference: npt.NDArray[np.floating] = np.max(
                        np.abs(frequencies[indices_opacities] / _frequencies[indices_file] - 1)
                    )
                    spectral_info = Opacity.split_species_spectral_info(os.path.basename(file_path_hdf5))[1]
                    resolving_power = Opacity.split_spectral_info(spectral_info)[0]
                    resolving_power = Opacity.get_resolving_power_from_string(resolving_power)

                    if max_relative_difference > cls._frequency_grid_misalignment_tolerance / resolving_power:
                        warnings.warn(
                            f"maximum frequency grid misalignment from file ({max_relative_difference}) "
                            f"is larger than tolerated "
                            f"({cls._frequency_grid_misalignment_tolerance / resolving_power}); "
                            f"expect inaccuracies in spectral features location\n"
                            f"To remove this warning, ensure that your correlated-k opacity files "
                            f"have similar wavenumber grids"
                        )

            # Fill opacity array
            opacities = np.zeros((g_size, frequencies.size, 1, temperature_pressure_grid_size))
            opacities[:, indices_opacities, 0, :] = k_table[:, indices_file, :]
            opacities[opacities < 0.] = 0.

            # Divide by mass to convert cross-sections to opacities
            mol_mass_inv = 1 / (f['mol_mass'][()] * cst.amu)

        line_opacities_grid = opacities * mol_mass_inv

        return line_opacities_grid

    @staticmethod
    def load_hdf5_line_opacity_table(
        file_path_hdf5, frequencies, line_by_line_opacity_sampling=1, return_radtrans_opacities=True
    ):
        """Load opacities (cm2.g-1) tables in HDF5 format, based on petitRADTRANS pseudo-ExoMol setup."""
        frequencies_relative_tolerance = 1e-12  # allow for a small relative deviation from the wavenumber grid

        with h5py.File(file_path_hdf5, 'r') as f:
            if not return_radtrans_opacities:
                return (
                    f['DOI'][()][0].decode('utf-8'),
                    f['bin_edges'][:],
                    f['xsecarr'][:],
                    f['mol_mass'][()][0],
                    f['mol_name'][()][0].decode('utf-8'),
                    f['p'][:],
                    f['t'][:]
                )

            frequency_grid = cst.c * f['bin_edges'][:]  # cm-1 to s-1

            selection = np.nonzero(np.logical_and(
                np.greater_equal(frequency_grid, np.min(frequencies) * (1 - frequencies_relative_tolerance)),
                np.less_equal(frequency_grid, np.max(frequencies) * (1 + frequencies_relative_tolerance))
            ))[0]
            selection = np.array([selection[0], selection[-1]])

            if line_by_line_opacity_sampling > 1:
                # Ensure that down-sampled wavelength upper bound >= requested wavelength upper bound
                selection[0] -= line_by_line_opacity_sampling - 1  # array is ordered by increasing wavenumber

            line_opacities_grid = f['xsecarr'][:, :, selection[0]:selection[-1] + 1]

            # Divide by mass to convert cross-sections to opacities
            mol_mass_inv = 1 / (f['mol_mass'][()] * cst.amu)
            line_opacities_grid *= mol_mass_inv

        line_opacities_grid = line_opacities_grid[:, :, ::-1]

        if line_by_line_opacity_sampling > 1:
            line_opacities_grid = line_opacities_grid[:, :, ::line_by_line_opacity_sampling]

        if line_opacities_grid.shape[-1] != frequencies.size:
            frequency_grid = frequency_grid[selection[0]:selection[-1] + 1]
            frequency_grid = frequency_grid[::-1]

            if line_by_line_opacity_sampling > 1:
                frequency_grid = frequency_grid[::line_by_line_opacity_sampling]

            if frequency_grid[-1] == frequency_grid[-2]:  # handle duplicated last wavenumber
                frequency_grid = frequency_grid[:-1]
                line_opacities_grid = line_opacities_grid[:, :, :-1]

            indices_opacities, indices_file = Radtrans.__get_frequency_grids_intersection_indices(
                frequencies=frequencies,
                file_frequencies=frequency_grid,
                tolerate_grid_misalignment=False
            )

            # Fill opacity array
            _line_opacities_grid = np.zeros(
                (line_opacities_grid.shape[0], line_opacities_grid.shape[1], frequencies.size)
            )
            _line_opacities_grid[:, :, indices_opacities] = line_opacities_grid[:, :, indices_file]
            del line_opacities_grid
        else:
            _line_opacities_grid = line_opacities_grid

        _line_opacities_grid = np.swapaxes(_line_opacities_grid, 0, 1)  # (t, p, wvl)
        _line_opacities_grid = _line_opacities_grid.reshape(
            (_line_opacities_grid.shape[0] * _line_opacities_grid.shape[1], _line_opacities_grid.shape[2])
        )  # (tp, wvl)
        _line_opacities_grid = np.swapaxes(_line_opacities_grid, 0, 1)  # (wvl, tp)
        _line_opacities_grid = _line_opacities_grid[np.newaxis, :, np.newaxis, :]  # (g, wvl, species, tp)

        return _line_opacities_grid

    def load_line_opacities(self, path_input_data):
        """Load the line opacities for spectral calculation.
        The default pressure-temperature grid is a log-uniform (10, 13) grid.

        Args:
            path_input_data: path to petitRADTRANS input_data directory
        """
        # TODO currently all the pressure-temperature grid is loaded, it could be more memory efficient to provide a T an p range at init and only load the relevant parts of the grid # noqa: E501
        if self._line_opacity_mode == 'c-k':
            category = 'correlated_k_opacities'
        elif self._line_opacity_mode == 'lbl':
            category = 'line_by_line_opacities'
        else:
            raise ValueError(f"invalid line opacity mode: '{self._line_opacity_mode}' (must be 'c-k'|'lbl')")

        # Read opacities grid
        if len(self._line_species) > 0:
            for i, species in enumerate(self._line_species):
                hdf5_file = self.__get_line_opacity_file(
                    path_input_data=path_input_data,
                    species=species,
                    category=category
                )

                # Load g grid for correlated-k
                if self._line_opacity_mode == 'c-k' and i == 0:
                    with h5py.File(hdf5_file, 'r') as f:
                        self._lines_loaded_opacities['g_gauss'] = f['samples'][:]
                        self._lines_loaded_opacities['weights_gauss'] = f['weights'][:]

                    # Convert into F-ordered array for more efficient processing in the Fortran modules
                    self._lines_loaded_opacities['g_gauss'] = np.array(
                        self._lines_loaded_opacities['g_gauss'], dtype='d', order='F'
                    )
                    self._lines_loaded_opacities['weights_gauss'] = np.array(
                        self._lines_loaded_opacities['weights_gauss'], dtype='d', order='F'
                    )

                # Load temperature-pressure grid
                (
                    self._lines_loaded_opacities['temperature_pressure_grid'][species],
                    self._lines_loaded_opacities['temperature_grid_size'][species],
                    self._lines_loaded_opacities['pressure_grid_size'][species]
                ) = self.load_line_opacities_pressure_temperature_grid(
                    hdf5_file=hdf5_file
                )
                # Load the opacities
                print(f" Loading line opacities of species '{species}' from file '{hdf5_file}'...", end='')

                if self._line_opacity_mode == 'c-k':
                    self._lines_loaded_opacities['opacity_grid'][species] = self.load_hdf5_ktables(
                        file_path_hdf5=hdf5_file,
                        frequencies=self._frequencies,
                        g_size=self._lines_loaded_opacities['g_gauss'].size,
                        temperature_pressure_grid_size=self._lines_loaded_opacities['temperature_pressure_grid'][
                            species].shape[0]
                    )
                elif self._line_opacity_mode == 'lbl':
                    self._lines_loaded_opacities['opacity_grid'][species] = self.load_hdf5_line_opacity_table(
                        file_path_hdf5=hdf5_file,
                        frequencies=self._frequencies,
                        line_by_line_opacity_sampling=self._line_by_line_opacity_sampling
                    )

                print(" Done.")

                # Convert into F-ordered array for more efficient processing in the Fortran modules
                self._lines_loaded_opacities['opacity_grid'][species] = \
                    np.array(self._lines_loaded_opacities['opacity_grid'][species][:, :, 0, :], dtype='d', order='F')

            print(" Successfully loaded all line opacities")

    @staticmethod
    def load_line_opacities_pressure_temperature_grid(hdf5_file):
        """Load line opacities temperature grids."""
        with h5py.File(hdf5_file, 'r') as f:
            pressure_grid = f['p'][:]
            temperature_grid = f['t'][:]

        ret_val = np.zeros((temperature_grid.size * pressure_grid.size, 2))

        for i_t in range(temperature_grid.size):
            for i_p in range(pressure_grid.size):
                ret_val[i_t * pressure_grid.size + i_p, 1] = pressure_grid[i_p] * 1e6  # bar to cgs
                ret_val[i_t * pressure_grid.size + i_p, 0] = temperature_grid[i_t]

        line_opacities_temperature_pressure_grid = ret_val
        line_opacities_temperature_grid_size = temperature_grid.size
        line_opacities_pressure_grid_size = pressure_grid.size

        return (line_opacities_temperature_pressure_grid, line_opacities_temperature_grid_size,
                line_opacities_pressure_grid_size)

    @staticmethod
    def rebin_star_spectrum(star_spectrum, star_wavelengths, wavelengths):
        add_stellar_flux = np.zeros(100)
        add_wavelengths = np.logspace(np.log10(1.0000002e-02), 2, 100)

        wavelengths_interp = np.append(star_wavelengths, add_wavelengths)
        fluxes_interp = np.append(star_spectrum, add_stellar_flux)

        stellar_intensity = rebin_spectrum(wavelengths_interp, fluxes_interp, wavelengths)

        return stellar_intensity
