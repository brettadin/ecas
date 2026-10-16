"""SpectralModel object and related.
The argument kwargs is not always used in SpectralModel functions, this is by design (see __init__ docstring).
"""
import copy
import inspect
import os
import sys
import warnings
from typing import Any, Callable

import h5py
import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter1d as scipy_gaussian_filter1d

from petitRADTRANS import physical_constants as cst
from petitRADTRANS.chemistry.pre_calculated_chemistry import pre_calculated_equilibrium_chemistry_table
from petitRADTRANS.chemistry.utils import (
    compute_mean_molar_masses, fill_atmosphere, mass_fractions2volume_mixing_ratios, simplify_species_list
)
from petitRADTRANS.config.configuration import petitradtrans_config_parser
from petitRADTRANS.math import gaussian_weights_running, resolving_space
from petitRADTRANS.physics import (
    doppler_shift,
    flux2irradiance,
    flux_hz2flux_cm,
    frequency2wavelength,
    hz2um,
    rebin_spectrum,
    rebin_spectrum_bin,
    temperature_profile_function_guillot_metallic
)
from petitRADTRANS.planet import Planet
from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS.retrieval.data import Data
from petitRADTRANS.retrieval.parameter import RetrievalParameter
from petitRADTRANS.retrieval.preparing import polyfit
from petitRADTRANS.retrieval.retrieval import Retrieval
from petitRADTRANS.retrieval.retrieval_config import RetrievalConfig
from petitRADTRANS.retrieval.utils import get_pymultinest_sample_dict
from petitRADTRANS.stellar_spectra.phoenix import phoenix_star_table
from petitRADTRANS.utils import (
    LockedDict,
    dict2hdf5,
    fill_object,
    hdf52dict,
    list_str2str,
    remove_mask,
    topological_sort
)


class SpectralModel(Radtrans):
    r"""Extension of the Radtrans object. Used to construct custom spectral models.

    The parameters required for the Radtrans spectral functions are calculated from "model functions", which require
    "model parameters". All parameters are calculated in order thanks to a topological sorting algorithm. Looping
    dependencies are not supported by default.

    Spectra can be generated with the calculate_spectrum function. Several spectral modification functions are
    available.

    A Data object can be initialized with the init_data function, which can then be used to generate a Retrieval object
    with the Retrieval.from_data function.

    Some features are detailed below.

    Automatic wavelength boundaries calculation:
        The model wavelength boundaries can be determined from the rebinned_wavelengths model parameter, taking into
        account Doppler shift if necessary, using the following model parameters (see below for a description):
            - relative_velocities
            - system_observer_radial_velocities
            - is_orbiting
            - orbital_longitudes
            - orbital_phases
            - rest_frame_velocity_shift
            - radial_velocity_semi_amplitude
        The size of the arrays is used to generate multiple observations of the spectrum. For example, if n_phases
        orbital phases are given, the generated spectrum of size n_wavelengths is shifted according to the relative
        velocity at each orbital phase. This generates a time-dependent spectrum of shape (n_phases, n_wavelengths).

    Custom functions and kwargs:
        The compute_* functions can be rewritten in scripts if necessary. Functions calculate_* should not be
        rewritten. Custom functions *must* include the **kwargs argument, even if it is not used. This is by design,
        to allow for any model parameter to be used with an arbitrarily complex custom function.

    Default model parameters:
        Default model parameters used when calling the SpectralModel built-in functions. This list may change when
        using custom functions.
        Radtrans spectral parameters can also be used as model parameters (see next section).

        airmass:
            Airmass of the observations, must be an array with one element per exposure.
            Used to calculate telluric transmittances and in ground-based high-resolution data preparation.
        apply_telluric_lines_removal:
            If True, apply the telluric lines removal correction of the retrieval.preparing.polyfit function.
        apply_throughput_removal:
            If True, apply the instrumental throughput removal correction of the retrieval.preparing.polyfit function.
        atmospheric_mixing:
            Scaling factor [0, 1] representing how well metals are mixed in the atmosphere.
            Used to calculate the atmospheric metallicity when the metallicity model parameter is None.
        carbon_pressure_quench:
            (bar) Pressure where the carbon species are quenched, used with equilibrium chemistry.
        co_ratio:
            Desired carbon to oxygen ratios, obtained by increasing the amount of oxygen. Can be a scalar or an array
            with one element per layer.
            Used to calculate mass fractions when equilibrium chemistry is activated
            (``use_equilibrium_chemistry=True``).
        constance_tolerance:
            Relative tolerance on input resolving power to apply constant or running convolutions.
        convolve_resolving_power:
            Resolving power of the convolved spectrum, used when the convolve argument is True.
        correct_uncertainties:
            If True, the uncertainties of the prepared spectrum are corrected from the preparing steps.
        filling_species:
            Dictionary with filling species as keys and the weights of the mass fractions as values. Unweighted
            filling species are represented with None. Filling species are used to ensure that the MMR at each layer is
            1. See chemistry.utils.fill_atmospheric_layer for more details.
        guillot_temperature_profile_gamma:
            Ratio between visual and infrared opacity.
            Used by the Guillot temperature profile (``temperature_profile_mode='guillot'``).
        guillot_temperature_profile_infrared_mean_opacity_solar_metallicity:
            (cm2.s-1) infrared mean opacity for a solar metallicity (Z = 1) atmosphere.
            The effective infrared opacity is obtained from this parameter times the metallicity.
            Used with the Guillot temperature profile (``temperature_profile_mode='guillot'``).
        imposed_mass_fractions:
            Mass fractions imposed when calculating the mass fractions. See SpectralModel.compute_mass_fractions for
            more details.
        intrinsic_temperature:
            (K) The planet's intrinsic temperature.
            Used by the Guillot temperature profile (``temperature_profile_mode='guillot'``).
        is_around_star:
            If True, and if the spectral emission mode is 'emission', model parameter star_flux can be calculated. It
            is then used to calculate stellar_intensities (for emission spectra), and when scaling the spectrum
            (``scale=True``).
        is_observed:
            If True, and if the spectral calculation mode is ``'emission'``, calculates the spectral flux received by
            the observer, taking into account the distance between the target and the observer.
        is_orbiting:
            If True, model parameter ``radial_velocity`` is calculated from orbital parameters (see previous section).
        metallicity:
            Ratio between heavy elements and H2 + He compared to solar.
            Used with equilibrium chemistry.
        metallicity_mass_coefficient:
            Power of the mass-metallicity relationship.
            Used to calculate the metallicity.
        metallicity_mass_scaling:
            Scaling factor of the mass-metallicity relationship.
            Used to calculate the metallicity.
        mid_transit_time:
            (s) Mid-transit time of the planet's primary transit.
            Used to calculate the transit effect (``use_transit_light_loss=True``).
        modification_arguments:
            Dictionary with the SpectralModel modification flags as keys and their status as values.
            The modification parameters are:
                - scale,
                - shift,
                - use_transit_light_loss,
                - convolve,
                - rebin,
                - prepare.
        orbit_semi_major_axis:
            (cm) Semi-major axis of the planet's orbit.
            Used to calculate the radial velocities, transit effect, and stellar intensities.
        orbital_inclination:
            (deg) Angle between the normal of the planet orbital plane and the axis of observation,
            i.e. 90 degree: edge view, 0 degree: top view.
            Used to calculate the radial velocities, and transit effect.
        orbital_longitudes:
            (deg) Orbital longitudes of the observation's exposures.
            Used to calculate the radial velocities, and transit effect.
            Can also be calculated from model parameters ``times`` or ``orbital_phases``.
        orbital_period:
            (s) Period of the planet's orbit.
            Used to calculate the orbital longitudes and the transit effect.
        planet_mass:
            (g) Mass of the planet.
            Can be used to calculate the metallicity, the planet radius, or the reference gravity.
        polynomial_fit_degree:
            Degree of the polynomial fit used to prepare the spectrum, in the retrieval.preparing.polyfit function
        radial_velocity_semi_amplitude:
            (cm.s-1) Radial orbital velocity semi-amplitude of the planet (Kp).
            Used to calculate the planet's relative velocities. Can be automatically calculated.
        rebin_range_margin_power:
            Relative wavelength margin to use for the automatic wavelength boundary calculation.
        rebinned_wavelengths:
            (cm) Wavelengths at which to rebin the spectrum (``rebin=True``).
        relative_velocities:
            (cm.s-1) Array of relative velocities between the target and the observer.
            Can be automatically calculated.
            Used when shifting the spectrum (``shift=True``).
        rest_frame_velocity_shift:
            (cm.s-1) Shift in the planet's rest frame velocity.
            Used when shifting the spectrum (``shift=True``).
        star_effective_temperature:
            (K) Effective temperature of the planet's star.
            Used to calculate the stellar intensities.
        star_mass:
            (g) Mass of the star.
            Used to calculate the relative velocities.
        star_metallicity:
             Metallicity of the star in solar metallicity.
             Used to calculate the metallicity.
        star_radius:
            (cm) Radius of the planet's star.
            Used to calculate the stellar intensities, the scaled spectrum, and the transit effect.
        system_distance:
            (cm) Distance between the target and the observer.
            Used to calculate the emission spectrum when ``is_observed`` is True.
        system_observer_radial_velocities:
            (cm.s-1) Array of velocities between the system and the observer.
        tellurics_mask_threshold:
            Mask wavelengths where the atmospheric transmittance estimate is below this value.
            Used when preparing the spectrum with retrieval.preparing.polyfit function.
        temperature:
            (K) When ``temperature_profile_mode='isothermal'``, the value of the temperature.
            When ``temperature_profile_mode='guillot'``, it corresponds to the equilibrium temperature.
        temperature_profile_mode:
            If ``'isothermal'``, the temperature profile is isothermal
            If ``guillot``, the temperature profile follows the Guillot 2010 temperature profile.
        times:
            (s) Array of size n_exposures.
            Used to calculate the orbital longitudes.
        transit_duration:
            (s) Duration of the planet total transit (T14)
            Used to calculate the transit effect.
        uncertainties:
            Spectrum uncertainties.
            Used when preparing the spectrum with retrieval.preparing.polyfit function.
        uncertainties_as_weights:
            If True, the uncertainties are used as fit weights when preparing the spectrum.
            Used when preparing the spectrum with retrieval.preparing.polyfit function.
        use_equilibrium_chemistry:
            If True, use pRT equilibrium chemistry module to calculate the mass fractions. Does not override the
            imposed mass fractions.
        verbose:
            If True, displays additional information when running some SpectralModel functions.

    Radtrans model parameters:
        Model parameters used when calling the Radtrans spectral functions.

        cloud_f_sed:
            Cloud settling parameter.
        cloud_hansen_a:
            A dictionary of the 'a' parameter values for each included cloud species and for each atmospheric
            layer, formatted as the kzz argument. Equivalent to cloud_particles_mean_radii.
            If cloud_hansen_a is not included and dist is "hansen", then it will be computed using Kzz and fsed
            (recommended).
        cloud_hansen_b:
            A dictionary of the 'b' parameter values for each included cloud species and for each atmospheric
            layer, formatted as the kzz argument. This is the width of the hansen distribution normalized by
            the particle area (1/cloud_hansen_a^2).
        cloud_particle_number_density_grid:
            A dictionary of cloud particle number densities for all cloud species.
        cloud_particle_radius_distribution_std:
            Width of the log-normal cloud particle size distribution.
        cloud_particles_mean_radii:
            Dictionary of mean particle radii for all cloud species.
            Dictionary keys are the cloud species names. Every radius array has same length as pressure array.
        cloud_particles_radius_distribution:
            The cloud particle size distribution to use.
            Can be either 'lognormal' (default) or 'hansen'.
            If hansen, the cloud_hansen_b parameters must be used.
        cloud_photosphere_median_optical_depth:
            Median optical depth (across ``wavelength_boundaries``) of the clouds from the top of the
            atmosphere down to the gas-only photosphere. This parameter can be used for enforcing the presence
            of clouds in the photospheric region.
        clouds_particles_porosity_factor:
            A dictionary of porosity factors depending on the cloud species. This can be useful when opacities
            are calculated using the Distribution of Hollow Spheres (DHS) method.
        eddy_diffusion_coefficients:
            The atmospheric eddy diffusion coefficient in cgs
            (i.e. :math:`\\rm cm^2/s`),
            at each atmospheric layer
            (1-d numpy array, same length as pressure array).
        emissivities:
            Emissivities of the surface (layer with the highest pressure).
        frequencies_to_wavelengths:
            If True, convert the frequencies (Hz) output to wavelengths (cm)
        gray_opacity:
            Gray opacity value, to be added to the opacity at all pressures and wavelengths
            (units :math:`\\rm cm^2/g`).
        haze_factor:
            Scalar factor, increasing the gas Rayleigh scattering cross-section.
        irradiation_geometry:
            If equal to ``'dayside_ave'``: use the dayside average geometry.
            If equal to ``'planetary_ave'``: use the planetary average geometry.
            If equal to ``'non-isotropic'``: use the non-isotropic geometry.
        opaque_cloud_top_pressure:
            (bar) Pressure where an opaque cloud deck is added to the absorption opacity.
        planet_radius:
            (cm) Radius of the planet.
        power_law_opacity_350nm:
            Scattering opacity at 0.35 micron, in cgs units (cm^2/g).
        power_law_opacity_coefficient:
            Has to be given if kappa_zero is defined, this is the
            wavelength powerlaw index of the parametrized scattering
            opacity.
        reference_gravity:
            (cm.s-2) Gravity at the planet's reference pressure.
        reference_pressure:
            (bar) Reference pressure used to set the planet's radius.
        reflectances:
            Reflectances of the surface (layer with the highest pressure).
        return_cloud_contribution:
            If True, the cloud contribution is calculated.
        return_contribution:
            If True the spectral contribution function is calculated.
        return_photosphere_radius:
            If True, the photosphere radius is calculated and returned.
            Only for emission spectra.
        return_radius_hydrostatic_equilibrium:
            If True, the radius at hydrostatic equilibrium of the planet is returned.
            Only for transmission spectra.
        return_rosseland_optical_depths:
            If True, the Rosseland opacities and optical depths are calculated and returned.
            Only for emission spectra.
        star_irradiation_angle:
            Inclination angle of the direct light with respect to the normal to the atmosphere. Used only in
            the non-isotropic geometry scenario.
        stellar_intensities:
            (W.m-2/um) The stellar intensity to use. If None, it is calculated using a PHOENIX model.
        variable_gravity:
            Standard is ``True``. If ``False`` the gravity will be
            constant as a function of pressure, during the transmission
            radius calculation.

    Args:
        pressures:
            (bar) array containing the pressures of the model, from lowest to highest.
        line_species:
            list of strings, denoting which line absorber species to include; must match the opacity file names.
        rayleigh_species:
            list of strings, denoting which Rayleigh scattering species to include; must match the opacity file
            names.
        continuum_opacities:
            list of strings, denoting which continuum absorber species to include; must match the opacity file
            names.
        cloud_species:
            list of strings, denoting which cloud opacity species to include; must match the opacity file names.
        line_opacity_mode:
            if equal to ``'c-k'``: use low-resolution mode, at
            :math:`\\lambda/\\Delta \\lambda = 1000`, with the correlated-k
            assumption. if equal to ``'lbl'``: use high-resolution mode, at
            :math:`\\lambda/\\Delta \\lambda = 10^6`, with a line-by-line
            treatment.
        scattering_in_emission:
            Will be ``False`` by default.
            If ``True`` scattering will be included in the emission spectral
            calculations. Note that this increases the runtime of pRT!
        line_by_line_opacity_sampling:
            Will be ``None`` by default. If integer positive value, and if
            ``mode == 'lbl'`` is ``True``, then this will only consider every
            line_by_line_opacity_sampling-nth point of the high-resolution opacities.
            This may be desired in the case where medium-resolution spectra are
            required with a :math:`\\lambda/\\Delta \\lambda > 1000`, but much smaller than
            :math:`10^6`, which is the resolution of the ``lbl`` mode. In this case it
            may make sense to carry out the calculations with line_by_line_opacity_sampling = 10,
            for example, and then re-binning to the final desired resolution:
            this may save time! The user should verify whether this leads to
            solutions which are identical to the re-binned results of the fiducial
            :math:`10^6` resolution. If not, this parameter must not be used.
        temperatures:
            array containing the temperatures of the model, at each pressure.
        mass_fractions:
            dictionary containing the mass mixing ratios of the model, at each pressure, for every species.
        mean_molar_masses:
            dictionary containing the mean_molar_masses of the model, at each pressure.
        wavelength_boundaries:
            (um) list containing the min and max wavelength of the model. Can be automatically determined from
        wavelengths:
            (um) wavelengths of the model.
        transit_radii:
            transit radii of the model.
        fluxes:
            (erg.s-1.cm-2.sr-1/cm) spectral radiosities of the spectrum.
        **model_parameters:
            dictionary of parameters. The keys can match arguments of functions used to generate the model.
    """
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
            path_input_data: str = None,
            radial_velocity_semi_amplitude_function: Callable = None,
            radial_velocities_function: Callable = None,
            relative_velocities_function: Callable = None,
            orbital_longitudes_function: Callable = None,
            temperatures=None, mass_fractions=None, mean_molar_masses=None,
            wavelengths=None, transit_radii=None, fluxes=None,
            model_functions_map=None, spectral_modification_functions_map=None,
            **model_parameters
    ):
        if radial_velocity_semi_amplitude_function is None:
            radial_velocity_semi_amplitude_function = self.compute_radial_velocity_semi_amplitude
        else:
            self.compute_radial_velocity_semi_amplitude = radial_velocity_semi_amplitude_function

        if radial_velocities_function is None:
            radial_velocities_function = self.compute_radial_velocities
        else:
            self.compute_radial_velocity = radial_velocities_function

        if relative_velocities_function is None:
            relative_velocities_function = self.compute_relative_velocities
        else:
            self.compute_relative_velocities = relative_velocities_function

        if orbital_longitudes_function is None:
            orbital_longitudes_function = self.compute_orbital_longitudes
        else:
            self.compute_orbital_longitudes = orbital_longitudes_function

        if model_functions_map is None:
            model_functions_map = {}

        # TODO (v.4.0.0) if spectrum generation parameters are not None, change functions to get them so that they return the initialised value # noqa: E501
        # Spectrum generation base parameters
        self.temperatures = temperatures
        self.mass_fractions = mass_fractions
        self.mean_molar_masses = mean_molar_masses

        # Spectrum parameters
        self.wavelengths = wavelengths
        self.transit_radii = transit_radii
        self.fluxes = fluxes

        # Other model parameters
        self.model_parameters = {}

        for model_parameter, value in model_parameters.items():
            self.model_parameters[model_parameter] = value

        # Velocities
        self.model_parameters['relative_velocities'], \
            self.model_parameters['system_observer_radial_velocities'], \
            self.model_parameters['radial_velocity_semi_amplitude'], \
            self.model_parameters['radial_velocities'], \
            self.model_parameters['orbital_longitudes'], \
            self.model_parameters['is_orbiting'] = \
            self.__init_velocities(
                radial_velocity_semi_amplitude_function=radial_velocity_semi_amplitude_function,
                radial_velocities_function=radial_velocities_function,
                relative_velocities_function=relative_velocities_function,
                orbital_longitudes_function=orbital_longitudes_function,
                **self.model_parameters
            )

        # Wavelength boundaries
        if wavelength_boundaries is None:  # calculate the optimal wavelength boundaries
            if self.wavelengths is not None:
                self.model_parameters['rebinned_wavelengths'] = copy.deepcopy(self.wavelengths)
            elif 'rebinned_wavelengths' not in model_parameters:
                raise TypeError(
                    "missing required argument "
                    "'wavelength_boundaries', add this argument to manually set the boundaries or "
                    "add model parameter 'rebinned_wavelengths' to set the boundaries automatically"
                )
            elif self.model_parameters['rebinned_wavelengths'] is None:
                raise TypeError(
                    "missing required argument "
                    "'wavelength_boundaries', and model parameter 'rebinned_wavelengths' was not set, "
                    "add model parameter 'wavelength_boundaries' to manually set the boundaries or "
                    "add model parameter 'rebinned_wavelengths' to set the boundaries automatically"
                )

            wavelength_boundaries = self.calculate_optimal_wavelength_boundaries()

        # Radtrans parameters
        super().__init__(
            pressures=pressures,
            wavelength_boundaries=wavelength_boundaries,
            line_species=line_species,
            gas_continuum_contributors=gas_continuum_contributors,
            rayleigh_species=rayleigh_species,
            cloud_species=cloud_species,
            line_opacity_mode=line_opacity_mode,
            line_by_line_opacity_sampling=line_by_line_opacity_sampling,
            scattering_in_emission=scattering_in_emission,
            emission_angle_grid=emission_angle_grid,
            anisotropic_cloud_scattering=anisotropic_cloud_scattering,
            path_input_data=path_input_data
        )

        self.spectral_modification_functions_map = self._get_spectral_modification_functions_map(
            custom_map=spectral_modification_functions_map
        )

        self.model_functions_map = self._get_model_functions_map(
            update_model_parameters=True,
            extra_model_parameters=self.model_parameters,
            custom_map=model_functions_map
        )

        self.__check_model_functions_map()

        self._fixed_model_parameters = set()

        for model_parameter, value in model_parameters.items():
            if self.model_functions_map[model_parameter] is not None:
                self._fixed_model_parameters.add(model_parameter)

        self.__check_model_parameters_relevance()

    def __check_model_parameters_relevance(self):
        default_parameters = self.get_default_parameters(
            sort=False,
            spectral_model_specific=False,
            from_maps=True
        )

        for parameter in self.model_parameters:
            if 'log10_' in parameter:
                parameter = parameter.split('log10_', 1)[1]

            if parameter not in default_parameters and parameter != 'modification_parameters':  # TODO modification_parameters is deprecated, remove in version 4 # noqa: E501
                warnings.warn(f"model parameter '{parameter}' is not used by any function of this "
                              f"SpectralModel\n"
                              f"To remove this warning, add a custom function using this parameter, "
                              f"or remove it from your model")

    def __check_model_functions_map(self):
        for parameter, model_function in self.model_functions_map.items():
            if (
                    model_function is None
                    and parameter not in self.model_parameters
                    and parameter not in ['pressures', 'wavelengths', 'mode']
                    and '_function' not in parameter
            ):
                raise ValueError(f"'{parameter}' has no model function and is not a model parameter")

    @staticmethod
    def __check_none_model_parameters(explanation_message_=None, **kwargs):
        missing = []

        for parameter_name, value in kwargs.items():
            if value is None:
                missing.append(parameter_name)

        if len(missing) > 0:
            joint = "', '".join(missing)

            base_error_message = f"missing {len(missing)} required model parameters: '{joint}'"

            raise TypeError(SpectralModel._explained_error(base_error_message, explanation_message_))

    @staticmethod
    def __init_velocities(radial_velocity_semi_amplitude_function, radial_velocities_function,
                          relative_velocities_function, orbital_longitudes_function,
                          relative_velocities=None, system_observer_radial_velocities=None,
                          radial_velocities=None,
                          radial_velocity_semi_amplitude=None, rest_frame_velocity_shift=0.0,
                          orbital_longitudes=None, orbital_phases=None, orbital_period=None,
                          times=None, mid_transit_time=None, is_orbiting=None, **kwargs):
        if system_observer_radial_velocities is None:
            system_observer_radial_velocities = np.zeros(1)

        # Determine if the planet is orbiting or not from the given information
        if mid_transit_time is not None or orbital_period is not None:
            if is_orbiting is None:
                is_orbiting = True
        elif orbital_longitudes is not None or orbital_phases is not None:
            if orbital_longitudes is None and orbital_phases is not None:
                orbital_longitudes = np.rad2deg(2 * np.pi * orbital_phases)

            if is_orbiting is None:
                is_orbiting = True
        elif is_orbiting is None:
            is_orbiting = False

        # Handle case where orbital longitudes are required but not given
        if is_orbiting \
                and relative_velocities is None \
                and orbital_longitudes is None \
                and (times is None or mid_transit_time is None or orbital_period is None):
            warnings.warn("Modelled object is orbiting but no orbital position information were provided, "
                          "assumed an orbital longitude of 0; "
                          "ensure that model parameters 'times', 'mid_transit_time' and 'orbital_period' are set; "
                          "alternatively, set model parameter 'orbital_longitudes' or 'orbital_phases', "
                          "or set model parameter 'is_orbiting' to False")

            orbital_longitudes = np.zeros(1)

        # Calculate relative velocities if needed
        if relative_velocities is None:
            relative_velocities, radial_velocities, radial_velocity_semi_amplitude, orbital_longitudes = \
                SpectralModel._compute_relative_velocities_wrap(
                    radial_velocity_semi_amplitude_function=radial_velocity_semi_amplitude_function,
                    radial_velocities_function=radial_velocities_function,
                    relative_velocities_function=relative_velocities_function,
                    orbital_longitudes_function=orbital_longitudes_function,
                    system_observer_radial_velocities=system_observer_radial_velocities,
                    rest_frame_velocity_shift=rest_frame_velocity_shift,
                    orbital_period=orbital_period,
                    times=times,
                    mid_transit_time=mid_transit_time,
                    orbital_longitudes=orbital_longitudes,
                    is_orbiting=is_orbiting,
                    radial_velocity_semi_amplitude=radial_velocity_semi_amplitude,
                    **kwargs
                )

        return relative_velocities, system_observer_radial_velocities, radial_velocity_semi_amplitude, \
            radial_velocities, orbital_longitudes, is_orbiting

    @staticmethod
    def __retrieval_parameters_interface(retrieved_parameters, fixed_parameters, line_species, n_layers):
        # TODO (v4.0.0) Change model generating function template to not include pt_plot_mode
        if fixed_parameters is None:
            fixed_parameters = {}

        def __special_parameters_interface(_parameter):
            if _parameter in fixed_parameters:
                return None

            __parameter = None

            for k, v in parameters.items():
                if k in fixed_parameters:
                    continue

                k = k.split(_parameter, 1)[0]

                if k == '':
                    __parameter = copy.deepcopy(v)
                    break
                elif k == '_log10':
                    __parameter = copy.deepcopy(10 ** v)
                    break

            return __parameter

        # Build model parameters
        parameters = {}
        special_retrieval_parameters = Retrieval.get_special_parameters()

        # Add fixed parameters
        for key, value in fixed_parameters.items():
            parameters[key] = value

        # Add free parameters, convert from Parameter object to dictionary
        for key, value in retrieved_parameters.items():
            if key not in fixed_parameters:
                parameters[key] = copy.deepcopy(value)

        for key, value in parameters.items():
            if key[0] == '_':
                is_special_parameter = False

                for parameter in special_retrieval_parameters:
                    if key.split(parameter, 1)[0] == '' or key.split(parameter, 1)[0] == '_log10':
                        is_special_parameter = True
                        break

                if not is_special_parameter:
                    available = list_str2str(special_retrieval_parameters)
                    raise ValueError(
                        f"parameter '{key}' is not a special retrieval parameter\n"
                        f"Available special retrieval parameters are {available}."
                    )

            if hasattr(value, 'value'):
                parameters[key] = parameters[key].value

        # Handle special parameters
        # TODO remove beta handling in version 4.0.0
        uncertainty_scaling_in_parameters = False

        for parameter in parameters:
            split = parameter.split('_uncertainty_scaling', 1)

            if split[0] == '' or split[0] == '_log10':
                uncertainty_scaling_in_parameters = True

        if 'beta' in parameters and not uncertainty_scaling_in_parameters:
            uncertainty_scaling = __special_parameters_interface('beta')
            warnings.warn(  # TODO remove in version 4.0.0
                "special retrieved parameter 'beta' is deprecated and will be removed in a future update\n"
                "Use '_uncertainty_scaling' instead.",
                FutureWarning
            )
        elif 'beta' in parameters and uncertainty_scaling_in_parameters:
            raise ValueError(
                "special retrieved parameter '_uncertainty_scaling' with alias 'beta' declared twice\n"
                "Declare only '_uncertainty_scaling', and remove 'beta' from the retrieved parameters."
            )
        elif uncertainty_scaling_in_parameters:
            uncertainty_scaling = __special_parameters_interface('_uncertainty_scaling')
        else:
            uncertainty_scaling = None

        spectrum_scaling = __special_parameters_interface('_spectrum_scaling')
        spectrum_offset = __special_parameters_interface('_spectrum_offset')

        # Put retrieved species into imposed mass fractions
        imposed_mass_fractions = {}

        for species in line_species:
            if species in parameters:
                imposed_mass_fractions[species] = 10 ** parameters[species] * np.ones(n_layers)

                del parameters[species]

        for species in parameters['imposed_mass_fractions']:
            if species in parameters and species not in line_species:
                imposed_mass_fractions[species] = 10 ** parameters[species] * np.ones(n_layers)

                del parameters[species]

        for key, value in imposed_mass_fractions.items():
            parameters['imposed_mass_fractions'][key] = value

        return parameters, uncertainty_scaling, spectrum_scaling, spectrum_offset

    @staticmethod
    def _compute_planet_star_centers_distance(orbit_semi_major_axis, orbital_inclination,
                                              planet_radius_normalized, star_radius,
                                              orbital_longitudes, transit_duration, orbital_period,
                                              **kwargs):
        """Calculate the sky-projected distance between the centers of a star and a planet.
        This equation is valid if the eccentricity of the planet orbit is low.

        Source: Csizmadia 2020 (https://doi.org/10.1093/mnras/staa349)

        Args:
            orbit_semi_major_axis: planet orbit semi-major axis
            orbital_inclination: planet orbital inclination
            planet_radius_normalized: planet radius over its star radius
            star_radius: radius of the planet star
            orbital_longitudes: orbital longitudes of the observation exposures
            transit_duration: duration of the planet total transit (T14)
            orbital_period: period of the planet orbit
        """
        _ = kwargs  # kwargs is not used in this function, this is intended

        impact_parameter_squared = Planet.compute_impact_parameter(
            orbit_semi_major_axis=orbit_semi_major_axis,
            orbital_inclination=orbital_inclination,
            star_radius=star_radius
        ) ** 2

        # Get the orbital longitude corresponding to the absolute value of the orbital longitude at the beginning of T14
        # phi14 = T_14 / P is the phase length of T14, we want half this value: the transit occurs between +/- phi14 / 2
        # phi14 is converted into longitude by multiplying by 2 * pi; 2 * pi / 2 = pi, hence the pi factor
        transit_longitude_half_length = np.rad2deg(transit_duration / orbital_period * np.pi)

        # Move axis to (wavelength, ..., exposure) to be able to multiply with orbital longitudes
        planet_radius_normalized_squared = np.moveaxis((1 + planet_radius_normalized) ** 2, -1, 0)

        if np.any(np.greater(impact_parameter_squared, planet_radius_normalized_squared)):
            raise ValueError(f"planet do not transit: "
                             f"minimal distance to star center ({np.sqrt(impact_parameter_squared)}) "
                             f"is greater than the minimum sum of the star and planet radii "
                             f"({np.sqrt(np.min(planet_radius_normalized_squared))})")

        planet_star_centers_distance = np.sqrt(
            impact_parameter_squared + (planet_radius_normalized_squared - impact_parameter_squared)
            * (orbital_longitudes / transit_longitude_half_length) ** 2
        )

        # Move axis back to (..., exposure, wavelength)
        return np.moveaxis(planet_star_centers_distance, 0, -1)

    @staticmethod
    def _compute_relative_velocities_wrap(radial_velocity_semi_amplitude_function, radial_velocities_function,
                                          relative_velocities_function, orbital_longitudes_function,
                                          system_observer_radial_velocities, rest_frame_velocity_shift,
                                          orbital_period=None, times=None, mid_transit_time=None,
                                          orbital_longitudes=None, is_orbiting=False, radial_velocities=None,
                                          **kwargs):
        """Calculate the planet radial velocities if needed."""
        if is_orbiting:
            if orbital_period is None or mid_transit_time is None:
                if orbital_longitudes is None:
                    raise TypeError("missing model parameter 'orbital_longitude' "
                                    "required to calculate planet radial velocities; "
                                    "add this parameter, "
                                    "or add the model parameters 'orbital_period' and 'mid_transit_time'; "
                                    "alternatively, set 'is_orbiting' to False")
                else:
                    kwargs['orbital_longitudes'] = orbital_longitudes
            else:
                if times is not None and mid_transit_time is not None and orbital_period is not None:
                    orbital_longitudes = orbital_longitudes_function(
                        times=times,
                        mid_transit_time=mid_transit_time,
                        orbital_period=orbital_period,
                        **kwargs
                    )
                    kwargs['orbital_longitudes'] = orbital_longitudes
                else:
                    kwargs['orbital_longitudes'] = orbital_longitudes

            if 'radial_velocity_semi_amplitude' not in kwargs:
                radial_velocity_semi_amplitude = radial_velocity_semi_amplitude_function(
                    **kwargs
                )
                kwargs['radial_velocity_semi_amplitude'] = radial_velocity_semi_amplitude
            else:
                if kwargs['radial_velocity_semi_amplitude'] is None:
                    radial_velocity_semi_amplitude = radial_velocity_semi_amplitude_function(
                        **kwargs
                    )
                    kwargs['radial_velocity_semi_amplitude'] = radial_velocity_semi_amplitude
                else:
                    radial_velocity_semi_amplitude = kwargs['radial_velocity_semi_amplitude']

            # Use the above calculated orbital longitudes and planet radial velocity amplitude
            radial_velocities = radial_velocities_function(
                **kwargs
            )
        else:
            # Object is not orbiting, it has 0 orbit radial velocity
            orbital_longitudes = None
            radial_velocity_semi_amplitude = None

            if radial_velocities is None:
                radial_velocities = np.zeros(1)

        # Calculate the relative velocities from the above calculated orbit radial velocity
        relative_velocities = relative_velocities_function(
            radial_velocities=radial_velocities,
            system_observer_radial_velocities=system_observer_radial_velocities,
            rest_frame_velocity_shift=rest_frame_velocity_shift,
            **kwargs
        )

        return relative_velocities, radial_velocities, radial_velocity_semi_amplitude, orbital_longitudes

    @staticmethod
    def _compute_transit_fractional_light_loss_uniform(planet_radius_normalized, planet_star_centers_distance,
                                                       planet_radius_normalized_squared=None, **kwargs):
        """Calculate the fractional light loss observed when a planet transit a star.
        This equation neglects the effect of limb-darkening, assuming that the source is uniform.

        Source: Mandel & Agol 2002 (https://iopscience.iop.org/article/10.1086/345520)

        Args:
            planet_radius_normalized: planet radius over its star radius
            planet_radius_normalized_squared: planet radius over its star radius, squared
            planet_star_centers_distance: sky-projected distance between the centers of the planet and the star,
                normalized over the radius of the star
        """
        _ = kwargs  # kwargs is not used in this function, this is intended

        transit_fractional_light_loss = np.zeros(np.shape(planet_star_centers_distance))

        if planet_radius_normalized_squared is None:
            planet_radius_normalized_squared = planet_radius_normalized ** 2

        partial_transit = np.nonzero(
            np.logical_and(
                np.less(planet_star_centers_distance, 1 + planet_radius_normalized),
                np.greater(planet_star_centers_distance, np.abs(1 - planet_radius_normalized))
            )
        )
        full_transit = np.nonzero(np.less_equal(planet_star_centers_distance, 1 - planet_radius_normalized))
        star_totally_eclipsed = np.nonzero(np.less_equal(planet_star_centers_distance, planet_radius_normalized - 1))

        if partial_transit[0].size > 0:
            z = planet_star_centers_distance[partial_transit]
            z_squared = z ** 2

            kappa_0 = np.arccos(
                (planet_radius_normalized_squared[partial_transit] + z_squared - 1)
                / (2 * planet_radius_normalized[partial_transit] * z)
            )

            kappa_1 = np.arccos(
                (1 - planet_radius_normalized_squared[partial_transit] + z_squared)
                / (2 * z)
            )

            square_root = np.sqrt(
                (
                    4 * z_squared
                    - (1 + z_squared - planet_radius_normalized_squared[partial_transit]) ** 2
                ) / 4
            )

            transit_fractional_light_loss[partial_transit] = \
                (planet_radius_normalized_squared[partial_transit] * kappa_0 + kappa_1 - square_root) / np.pi

        transit_fractional_light_loss[full_transit] = planet_radius_normalized_squared[full_transit]
        transit_fractional_light_loss[star_totally_eclipsed] = 1

        return transit_fractional_light_loss

    @staticmethod
    def _convolve_constant(input_wavelengths, input_spectrum, convolve_resolving_power, input_resolving_power=None,
                           **kwargs):
        """Convolve a spectrum to a new resolving power with a Gaussian filter.
        The original spectrum must have a resolving power very large compared to the target resolving power.
        The convolution resolving power is given in that case by:
            convolve_resolving_power = input_wavelengths / FWHM_LSF  (FWHM_LSF <=> "Delta_lambda" in Wikipedia)
        Therefore, the full width half maximum (FWHM) of the target line spread function (LSF) is given by:
            FWHM_LSF = input_wavelengths / convolve_resolving_power
        This FWHM is converted in terms of wavelength steps by:
            FWHM_LSF_Delta = FWHM_LSF / Delta_input_wavelengths
        where Delta_input_wavelengths is the difference between the edges of the bin.
        And converted into a Gaussian standard deviation by:
            sigma = FWHM_LSF_Delta / 2 * sqrt(2 * ln(2))

        Args:
            input_wavelengths: (cm) wavelengths of the input spectrum
            input_spectrum: input spectrum
            convolve_resolving_power: resolving power of output spectrum

        Returns:
            convolved_spectrum: the convolved spectrum at the new resolving power
        """
        _ = kwargs  # kwargs is not used in this function, this is intended

        # Get input wavelengths over input wavelength steps
        if input_resolving_power is None:
            input_resolving_power = np.mean(SpectralModel.compute_bins_resolving_power(input_wavelengths))

        # Calculate the sigma to be used in the Gaussian filter in units of input wavelength bins
        # Conversion from FWHM to Gaussian sigma
        sigma_lsf_gauss_filter = input_resolving_power / convolve_resolving_power / (2 * np.sqrt(2 * np.log(2)))

        convolved_spectrum = scipy_gaussian_filter1d(
            input=input_spectrum,
            sigma=sigma_lsf_gauss_filter,
            mode='reflect'
        )

        return convolved_spectrum

    @staticmethod
    def _convolve_running(input_wavelengths, input_spectrum, convolve_resolving_power, input_resolving_power=None,
                          **kwargs):
        """Convolve a spectrum to a new resolving power.
        The spectrum is convolved using Gaussian filters with a standard deviation
            std_dev = R_in(lambda) / R_new(lambda) * input_wavelengths_bins.
        Both the input resolving power and output resolving power can vary with wavelength.
        The input resolving power is given by:
            lambda / Delta_lambda
        where lambda is the center of a wavelength bin and Delta_lambda is the difference between the edges of the bin.

        The weights of the convolution are stored in a (N, M) matrix, with N being the size of the input, and M the size
        of the convolution kernels.
        To speed-up calculations, a matrix A of shape (N, M) is built from the inputs such as:
            A[i, :] = s[i - M/2], s[i - M/2 + 1], ..., s[i - M/2 + M],
        with s the input spectrum.
        The definition of the convolution C of s by constant weights with wavelength is:
            C[i] = sum_{j=0}^{j=M-1} s[i - M/2 + j] * weights[j].
        Thus, the convolution of s by weights at index i is:
            C[i] = sum_{j=0}^{j=M-1} A[i, j] * weights[i, j].

        Args:
            input_wavelengths: (cm) wavelengths of the input spectrum
            input_spectrum: input spectrum
            convolve_resolving_power: resolving power of output spectrum
            input_resolving_power: if not None, skip its calculation using input_wavelengths

        Returns:
            convolved_spectrum: the convolved spectrum at the new resolving power
        """
        _ = kwargs  # kwargs is not used in this function, this is intended

        if input_resolving_power is None:
            input_resolving_power = SpectralModel.compute_bins_resolving_power(input_wavelengths)

        sigma_lsf_gauss_filter = input_resolving_power / convolve_resolving_power / (2 * np.sqrt(2 * np.log(2)))
        weights = gaussian_weights_running(sigma_lsf_gauss_filter)

        input_length = weights.shape[1]
        central_index = int(input_length / 2)

        # Create a matrix
        input_matrix = np.moveaxis(
            np.array([np.roll(input_spectrum, i - central_index, axis=-1) for i in range(input_length)]),
            0,
            -1
        )

        convolved_spectrum = np.sum(input_matrix * weights, axis=-1)
        n_dims_non_wavelength = len(input_spectrum.shape[:-1])

        # Replace non-valid convolved values by non-convolved values (inaccurate but better than 'reflect' or 0 padding)
        for i in range(input_length):
            if i - central_index < 0:
                ind = np.arange(0, central_index - i, dtype=int)

                for j in range(n_dims_non_wavelength):
                    ind = np.expand_dims(ind, axis=0)

                np.put_along_axis(
                    convolved_spectrum,
                    ind,
                    np.take_along_axis(
                        input_spectrum,
                        ind,
                        axis=-1
                    ),
                    axis=-1
                )
            elif i - central_index > 0:
                ind = -np.arange(1, i - central_index + 1, dtype=int)

                for j in range(n_dims_non_wavelength):
                    ind = np.expand_dims(ind, axis=0)

                np.put_along_axis(
                    convolved_spectrum,
                    ind,
                    np.take_along_axis(
                        input_spectrum,
                        ind,
                        axis=-1
                    ),
                    axis=-1
                )

        return convolved_spectrum

    @staticmethod
    def _convolve_wrap(wavelengths, convolve_function, spectrum, **kwargs):
        """Convolution wrapping function."""
        if np.ndim(wavelengths) <= 1:
            spectrum = convolve_function(
                input_wavelengths=wavelengths,
                input_spectrum=spectrum,
                **kwargs
            )
        else:
            spectrum = np.array([convolve_function(
                input_wavelengths=wavelength,
                input_spectrum=spectrum[i],
                **kwargs
            ) for i, wavelength in enumerate(wavelengths)])

        return spectrum

    @staticmethod
    def _explained_error(base_error_message, explanation_message):
        """Print an error with an explanation."""
        # TODO deprecated in Python 3.11
        if explanation_message is None:
            explanation_message = ''
        else:
            explanation_message = '\n' + explanation_message

        return str(base_error_message) + explanation_message

    def _get_model_functions_map(self, update_model_parameters=True, extra_model_parameters=None, custom_map=None):
        """Get the model function map.
        First, get the spectral parameters, find the spectral find the spectral functions from them, then the model
        functions, then the model parameters. Then, determine the dependencies of each parameter, then  topologically
        sort the model and spectral parameters.
        """
        def __build_dependencies_dict(_parameter, _model_functions_parameters, _map=None, _parameter0=None):
            if _map is None:
                _map: dict[str, set] = {}

            if _parameter0 is None:
                _parameter0 = _parameter

            if _parameter in ['self', 'cls', 'args', 'kwargs']:
                return _map

            if _parameter in ['pressures', 'wavelengths', 'mode']:
                _map[_parameter] = {None}
                return _map

            if _parameter not in _map:
                _map[_parameter] = set()

            if not isinstance(_map[_parameter], set):
                warnings.warn(f'parameter {_parameter} must be a set, but was {_map[_parameter]}')
                _map[_parameter] = set(_map[_parameter])

            if _parameter in _model_functions_parameters:
                _model_function = _model_functions_parameters[_parameter]

                if _model_function is None:
                    _map[_parameter] = {None}
                    return _map

                if isinstance(_model_function, str):
                    _model_function = getattr(self, _model_function)

                signature = inspect.signature(_model_function)

                for _model_function_parameter in signature.parameters:
                    if _model_function_parameter == 'kwargs':
                        continue

                    # Ignore size-0 cycles (i.e. x -> f(x) -> x...)
                    if _parameter == _model_function_parameter:
                        _map[_parameter].add(None)
                        continue

                    # Handle cycles (f_y(x) -> y -> f_x(y) -> x -> f_y(x)...)
                    if _model_function_parameter == _parameter0:
                        raise ValueError(
                            f"parameter '{_parameter0}' has a circular dependency "
                            f"with '{_model_function}'\n"
                            f"Add '{_parameter0}' to model parameters, or fix '{_model_function}'"
                        )

                    _map[_parameter].add(_model_function_parameter)
                    _map = __build_dependencies_dict(
                        _model_function_parameter, _model_functions_parameters, _map, _parameter0
                    )
            else:
                _map[_parameter].add(None)

            return _map

        spectral_parameters = self._get_spectral_parameters(
            remove_flag=True,
            remove_branching=True
        )

        # Check that all spectral parameters have a model parameter or a model function to generate them
        # Initialization
        parameter_dependencies = {}

        # Get the spectral parameters default function names
        for spectral_parameter in spectral_parameters:
            if '_function' not in spectral_parameter:  # special functions are model parameters
                parameter_dependencies[spectral_parameter] = f'compute_{spectral_parameter}'

        # Get the actual model functions
        model_functions = self.get_model_functions(
            from_maps=False
        )

        # Get the parameter-function pairs of the model functions
        model_functions_parameters = {
            model_function.__name__.split('compute_', 1)[1]: model_function
            for model_function in model_functions
        }

        # Also add the parameters of the spectral functions
        model_functions_parameters['emission_spectrum'] = self.calculate_emission_spectrum
        model_functions_parameters['transmission_spectrum'] = self.calculate_transmission_spectrum

        if custom_map is not None:
            if not isinstance(custom_map, dict):
                raise TypeError(f"custom model function map must be a dict, but is '{type(custom_map)}'")

            model_functions_parameters.update(custom_map)

        model_parameters = self.get_model_parameters(
            check_relevance=False,
            model_functions=list(model_functions_parameters.values())
        )

        if extra_model_parameters is not None:
            for key, value in extra_model_parameters.items():
                model_parameters[key] = extra_model_parameters[key]

        if update_model_parameters:
            self.model_parameters = model_parameters

        # Check if every spectral parameter is calculated by a function
        for spectral_parameter, spectral_parameter_function in parameter_dependencies.items():
            if spectral_parameter not in list(model_functions_parameters.keys()):
                if spectral_parameter not in model_parameters:
                    warnings.warn(f"no function to calculate spectral parameter '{spectral_parameter}'\n"
                                  f"Add function '{spectral_parameter_function}', "
                                  f"or add '{spectral_parameter}' as a model parameter")
                    parameter_dependencies[spectral_parameter] = None

        # Build the parameter dependencies dict
        parameter_dependencies = {}

        for spectral_parameter in spectral_parameters:
            if '_function' not in spectral_parameter:  # special functions are model parameters
                parameter_dependencies = __build_dependencies_dict(
                    spectral_parameter,
                    model_functions_parameters,
                    parameter_dependencies
                )
            else:
                parameter_dependencies[spectral_parameter] = {None}

        for parameter in model_parameters:
            parameter_dependencies = __build_dependencies_dict(
                parameter,
                model_functions_parameters,
                parameter_dependencies
            )

        # Build the model functions map using topological sort to execute the model functions in the correct order
        parameter_dependencies = list(topological_sort(parameter_dependencies))

        model_functions_map = {}

        for parameter in parameter_dependencies:
            if parameter in model_functions_parameters:
                model_functions_map[parameter] = model_functions_parameters[parameter]
            elif parameter is None:
                continue
            else:
                model_functions_map[parameter] = None

        return model_functions_map

    def _get_spectral_modification_functions_map(self, custom_map=None):
        """Get the spectral modification functions (scale, shift, etc.) map."""
        default_map = LockedDict.build_and_lock({
            'scale': self.scale_spectrum,
            'shift': self.shift_wavelengths,
            'use_transit_light_loss': self.compute_transit_fractional_light_loss,
            'convolve': self.convolve,
            'rebin': self.rebin_spectrum,
            'prepare': self.preparing_pipeline,
            'modify': self.modify_spectrum
        })

        if custom_map is not None:
            if not isinstance(custom_map, dict):
                raise TypeError(f"custom model function map must be a dict, but is '{type(custom_map)}'")

            default_map.update(custom_map)

        return default_map

    def _get_spectral_parameters(self, remove_flag=False, remove_branching=False):
        """Get the spectral parameters by checking the arguments of the spectral functions."""
        spectral_functions = [
            self.calculate_flux,
            self.calculate_transit_radii
        ]

        spectral_parameters = {}

        for spectral_function in spectral_functions:
            signature = inspect.signature(spectral_function)

            for parameter, value in signature.parameters.items():
                if (parameter in ['self', 'args', 'kwargs']  # remove generic placeholder arguments
                        or (isinstance(value.default, bool) and remove_flag)  # remove flag parameters
                        or (isinstance(value.default, str) and remove_branching)):  # remove branching parameters
                    continue

                if value.default is not inspect.Parameter.empty:
                    spectral_parameters[parameter] = value.default
                else:
                    spectral_parameters[parameter] = None

        return spectral_parameters

    @staticmethod
    def _rebin_wrap(wavelengths, spectrum: npt.NDArray, rebin_spectrum_function, **kwargs):
        """Wrapper for the rebin functions."""
        if np.ndim(wavelengths) <= 1:
            wavelengths_tmp, spectrum = rebin_spectrum_function(
                input_wavelengths=wavelengths,
                input_spectrum=spectrum,
                **kwargs
            )

            if spectrum.dtype == 'O' and np.ndim(spectrum) >= 2:
                spectrum = np.moveaxis(spectrum, 0, 1)
        elif np.ndim(wavelengths) == 2:
            spectrum_tmp = []  # create a list to handle detectors with varying number of wavelengths
            wavelengths_tmp = None

            if np.ndim(spectrum) == 0:
                pass
            elif np.ndim(spectrum) == 1:
                for i, wavelength_shift in enumerate(wavelengths):
                    spectrum_tmp.append([])
                    wavelengths_tmp, spectrum_tmp[-1] = rebin_spectrum_function(
                        input_wavelengths=wavelength_shift,
                        input_spectrum=spectrum,
                        **kwargs
                    )
            elif np.ndim(spectrum) == 2:
                for i, wavelength_shift in enumerate(wavelengths):
                    spectrum_tmp.append([])
                    wavelengths_tmp, spectrum_tmp[-1] = rebin_spectrum_function(
                        input_wavelengths=wavelength_shift,
                        input_spectrum=spectrum[i],
                        **kwargs
                    )
            else:
                raise ValueError(f"spectrum must have at most 2 dimensions, but has {np.ndim(spectrum)}")

            spectrum = np.array(spectrum_tmp)

            if np.ndim(spectrum) == 3 or spectrum.dtype == 'O':
                spectrum = np.moveaxis(spectrum, 0, 1)
            elif np.ndim(spectrum) > 3:
                raise ValueError(f"output spectrum must have at most 3 dimensions, but has {np.ndim(spectrum)}")
        else:
            raise ValueError(f"argument 'wavelength' must have at most 2 dimensions, "
                             f"but has {np.ndim(wavelengths)}")

        return wavelengths_tmp, spectrum

    def calculate_contribution_spectra(self, mode=None, **kwargs):
        """Calculate the contribution spectra.

        Args:
            mode:
                Can be 'emission' or 'transmission'.

        Returns:
            The contribution spectra.
        """
        if mode is None and isinstance(self.model_parameters['modification_arguments'], dict):
            modification_arguments: dict = self.model_parameters['modification_arguments']

            if modification_arguments['mode'] is not None:
                mode = copy.deepcopy(modification_arguments['mode'])

        if mode == 'emission':
            function = 'calculate_flux'
        elif mode == 'transmission':
            function = 'calculate_transit_radii'
        else:
            raise ValueError(f"mode must be 'emission' or 'transmission', not '{mode}'")

        signature = inspect.signature(self.__getattr__(function))

        parameters = {}

        for parameter in signature.parameters:
            if parameter in self.model_parameters:
                parameters[parameter] = copy.deepcopy(self.model_parameters[parameter])

        self.update_spectral_calculation_parameters(mode=mode, **self.model_parameters)

        return super().calculate_contribution_spectra(
            mode=mode,
            temperatures=self.temperatures,
            mass_fractions=self.mass_fractions,
            mean_molar_masses=self.mean_molar_masses,
            **parameters
        )

    def calculate_emission_spectrum(
            self,
            reference_gravity: float,
            planet_radius: float = None,
            opaque_cloud_top_pressure: float = None,
            cloud_particles_mean_radii: dict[str, npt.NDArray[np.floating]] = None,
            cloud_particle_radius_distribution_std: float = None,
            cloud_particles_radius_distribution: str = 'lognormal',
            cloud_hansen_a: dict[str, npt.NDArray[np.floating]] = None,
            cloud_hansen_b: dict[str, npt.NDArray[np.floating]] = None,
            cloud_particle_number_density_grid: dict[str, npt.NDArray[np.floating]] = None,
            cloud_f_sed: float = None,
            eddy_diffusion_coefficients: npt.NDArray[np.floating] = None,
            haze_factor: float = 1.0,
            power_law_opacity_350nm: float = None,
            power_law_opacity_coefficient: float = None,
            gray_opacity: float = None,
            cloud_photosphere_median_optical_depth: float = None,
            cloud_photosphere_wavelength_boundaries: float = None,
            cloud_fraction: float = 1.0,
            complete_coverage_clouds: list[str] = None,
            irradiation_geometry: str = 'dayside_ave',
            emission_geometry: str = None,
            stellar_intensities: npt.NDArray[np.floating] = None,
            star_effective_temperature: float = None,
            star_radius: float = None,
            orbit_semi_major_axis: float = None,
            star_irradiation_angle: float = 0.0,
            reflectances: npt.NDArray[np.floating] = None,
            emissivities: npt.NDArray[np.floating] = None,
            additional_absorption_opacities_function: Callable = None,
            additional_scattering_opacities_function: Callable = None,
            frequencies_to_wavelengths: bool = True,
            return_contribution: bool = False,
            return_clear_spectrum: bool = False,
            return_photosphere_radius: bool = False,
            return_rosseland_optical_depths: bool = False,
            return_cloud_contribution: bool = False,
            target_particle_radii: npt.NDArray[np.floating] = None,
            target_pressure: npt.NDArray[np.floating] = None,
            **kwargs
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], dict[str, Any]]:
        """Wrapper for the Radtrans calculate_flux function."""
        _ = kwargs  # kwargs is not used in this function, this is intended

        if emission_geometry is not None:
            # Deprecation warning is handled in Radtrans
            irradiation_geometry = emission_geometry  # TODO remove when emission_geometry is removed

        self.wavelengths, self.fluxes, additional_outputs = self.calculate_flux(
            temperatures=self.temperatures,
            mass_fractions=self.mass_fractions,
            mean_molar_masses=self.mean_molar_masses,
            reference_gravity=reference_gravity,
            planet_radius=planet_radius,
            opaque_cloud_top_pressure=opaque_cloud_top_pressure,
            cloud_particles_mean_radii=cloud_particles_mean_radii,
            cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
            cloud_particles_radius_distribution=cloud_particles_radius_distribution,
            cloud_hansen_a=cloud_hansen_a,
            cloud_hansen_b=cloud_hansen_b,
            cloud_particle_number_density_grid=cloud_particle_number_density_grid,
            cloud_f_sed=cloud_f_sed,
            eddy_diffusion_coefficients=eddy_diffusion_coefficients,
            haze_factor=haze_factor,
            power_law_opacity_350nm=power_law_opacity_350nm,
            power_law_opacity_coefficient=power_law_opacity_coefficient,
            gray_opacity=gray_opacity,
            cloud_photosphere_median_optical_depth=cloud_photosphere_median_optical_depth,
            cloud_photosphere_wavelength_boundaries=cloud_photosphere_wavelength_boundaries,
            cloud_fraction=cloud_fraction,
            complete_coverage_clouds=complete_coverage_clouds,
            irradiation_geometry=irradiation_geometry,
            emission_geometry=emission_geometry,  # TODO remove when emission_geometry is removed
            stellar_intensities=stellar_intensities,
            star_effective_temperature=star_effective_temperature,
            star_radius=star_radius,
            orbit_semi_major_axis=orbit_semi_major_axis,
            star_irradiation_angle=star_irradiation_angle,
            reflectances=reflectances,
            emissivities=emissivities,
            additional_absorption_opacities_function=additional_absorption_opacities_function,
            additional_scattering_opacities_function=additional_scattering_opacities_function,
            frequencies_to_wavelengths=frequencies_to_wavelengths,
            return_contribution=return_contribution,
            return_clear_spectrum=return_clear_spectrum,
            return_photosphere_radius=return_photosphere_radius,
            return_rosseland_optical_depths=return_rosseland_optical_depths,
            return_cloud_contribution=return_cloud_contribution,
            target_particle_radii=target_particle_radii,
            target_pressure=target_pressure
        )

        return self.wavelengths, self.fluxes, additional_outputs

    def calculate_optimal_wavelength_boundaries(self, rebinned_wavelengths=None, relative_velocities=None):
        """Return the optimal wavelength boundaries for rebin on output wavelengths.
        This minimizes the number of wavelengths to load and over which to calculate the spectra.
        Doppler shifting is also taken into account.

        The SpectralModel must have in its model_parameters keys:
            -  'rebinned_wavelengths': (um) the wavelengths to rebin to

        The SpectralModel can have in its model_parameters keys:
            - 'relative_velocities' (cm.s-1) the velocities of the source relative to the observer, in that case the
                wavelength range is increased to take into account Doppler shifting

        Returns:
            optimal_wavelengths_boundaries: (um) the optimal wavelengths boundaries for the spectrum
        """
        if rebinned_wavelengths is None:
            rebinned_wavelengths = self.model_parameters['rebinned_wavelengths']

        if relative_velocities is None and 'relative_velocities' in self.model_parameters:
            relative_velocities = copy.deepcopy(self.model_parameters['relative_velocities'])

        # Prevent multiple definitions and give priority to the function argument
        if 'rebinned_wavelengths' in self.model_parameters:
            rebinned_wavelengths_tmp = copy.deepcopy(self.model_parameters['rebinned_wavelengths'])
            del self.model_parameters['rebinned_wavelengths']
            save_rebinned_wavelengths = True
        else:
            rebinned_wavelengths_tmp = None
            save_rebinned_wavelengths = False

        if 'relative_velocities' in self.model_parameters:
            relative_velocities_tmp = copy.deepcopy(self.model_parameters['relative_velocities'])
            del self.model_parameters['relative_velocities']
            save_relative_velocities = True
        else:
            relative_velocities_tmp = None
            save_relative_velocities = False

        optimal_wavelengths_boundaries = self.compute_optimal_wavelength_boundaries(
            rebinned_wavelengths=rebinned_wavelengths,
            shift_wavelengths_function=self.shift_wavelengths,
            relative_velocities=relative_velocities,
            **self.model_parameters
        )

        if save_rebinned_wavelengths:
            self.model_parameters['rebinned_wavelengths'] = rebinned_wavelengths_tmp

        if save_relative_velocities:
            self.model_parameters['relative_velocities'] = relative_velocities_tmp

        return optimal_wavelengths_boundaries

    def calculate_spectral_parameters(self, pressures=None, wavelengths=None, **kwargs):
        """Initialize the temperature profile, mass mixing ratios and mean molar mass of a model.

        Args:
            pressures:
            wavelengths:

        Returns:

        """
        if pressures is None:
            pressures = self.pressures  # coming from Radtrans

        if wavelengths is None:
            wavelengths = self.wavelengths  # coming from Radtrans

        functions_dict = {
            parameter: function
            for parameter, function in self.model_functions_map.items()
            if function is not None and parameter not in self._fixed_model_parameters
        }

        # Include properties to model parameters
        for key in self.__dict__:
            if key in ['_line_species', '_gas_continuum_contributors', '_rayleigh_species', '_cloud_species']:
                kwargs[key[1:]] = copy.deepcopy(self.__dict__[key])

        # Include all used functions arguments default value into the model parameters
        spectral_model_attributes = list(self.__dict__.keys())

        for function in functions_dict.values():
            signature = inspect.signature(function)

            for parameter, value in signature.parameters.items():
                if parameter not in spectral_model_attributes \
                        and parameter not in functions_dict \
                        and parameter not in kwargs \
                        and value.default is not inspect.Parameter.empty:
                    kwargs[parameter] = value.default

        # Include spectral parameters
        spectral_parameters = self._get_spectral_parameters(
            remove_flag=False,
            remove_branching=False
        )

        for spectral_parameter, value in spectral_parameters.items():
            if spectral_parameter not in kwargs:
                kwargs[spectral_parameter] = value

        # Temporarily put mode as a model parameter, as it is required by some functions
        if 'modification_arguments' in kwargs:
            if 'mode' in kwargs['modification_arguments']:
                kwargs['mode'] = kwargs['modification_arguments']['mode']

        temperatures, mass_fractions, mean_molar_mass, model_parameters = self.compute_spectral_parameters(
            model_functions_map=functions_dict,
            pressures=pressures,
            wavelengths=wavelengths,
            **kwargs
        )

        _model_parameters = {}

        for _model_parameter, value in model_parameters.items():
            if _model_parameter in self.model_parameters:
                _model_parameters[_model_parameter] = value
            elif _model_parameter in self.model_parameters['modification_arguments']:
                _model_parameters['modification_arguments'][_model_parameter] = value

        return temperatures, mass_fractions, mean_molar_mass, _model_parameters

    def calculate_spectrum(self, mode='emission', parameters=None, update_parameters=True,
                           telluric_transmittances_wavelengths=None, telluric_transmittances=None,
                           instrumental_deformations=None, noise_matrix=None,
                           scale=False, shift=False, use_transit_light_loss=False, convolve=False, rebin=False,
                           prepare=False, return_additional_outputs=False):
        """Calculate a transmission or emission spectrum using the SpectralModel model parameters.
        The model parameters and spectral parameters can be (re)calculated. The spectrum is calculated using the
        these parameters with the corresponding Radtrans function, then can be modified by several parameters.

        Args:
            mode:
                Can be 'emission' for emission spectrum, or 'transmission' for transmission spectrum.
            parameters:
                Parameters used to calculate the spectrum. If None, the model parameters of the instance are used.
            update_parameters:
                If True, the model parameters are (re)calculated.
            telluric_transmittances_wavelengths:
                (cm) Wavelengths of the telluric transmittances.
            telluric_transmittances:
                Telluric transmittances (between 0 and 1). If the 'airmass' model parameter is provided, the telluric
                transmittances at each exposure is modified as ``T = exp(log(T) * airmass)``.
            instrumental_deformations:
                Array that is multiplied to the spectrum after it has been modified. Intended to simulate various
                instrumental effects such as the blaze function, throughput variation, etc.
            noise_matrix:
                Array that is added to the spectrum after it has been modified. Intended to be used to simulate noise.
            scale:
                If True, modify the spectrum according to the ``scale_spectrum`` function.
                By default, scale with respect to ``star_observed_spectrum`` (in emission), or ``star_radius``
                (in transmission).
            shift:
                If True, modify the spectrum according to the ``shift_wavelengths`` function.
                By default, Doppler-shift the spectrum according to ``relative_velocities``
            use_transit_light_loss:
                If True, modify the spectrum according to the ``compute_transit_fractional_light_loss`` function.
                By default, apply Eq. 1 of Mandel & Agol, 2002 (https://www.doi.org/10.1086/345520). Limb darkening is
                not taken into account.
            convolve:
                If True, modify the spectrum according to the ``convolve`` function.
                By default, convolve the function with a Gaussian kernel of FWHM determined by
                ``convolve_resolving_power``.
            rebin:
                If True, modify the spectrum according to the ``rebin_spectrum`` function.
                By default, rebin the spectrum from its native wavelengths to ``rebinned_wavelengths``.
            prepare:
                If True, modify the spectrum according to the ``preparing_pipeline```function.
                By default, prepare the spectrum using the ``polyfit`` preparing pipeline. Intended for
                ground-based high-resolution forward models.
            return_additional_outputs:
                If True, return the ``Radtrans`` additional outputs.
        Returns:
            Wavelengths (cm) and spectra (cm in transmission, erg.cm-2.s-1/cm in emission), for each order, exposure,
            and wavelength. If ``return_additional_outputs`` is True, also return a dict containing additional outputs.
        """
        # Initialize parameters
        if parameters is None:
            parameters = self.model_parameters

        # Add modification parameters to model parameters
        modification_arguments = inspect.signature(self.calculate_spectrum).parameters
        local_variables = locals()

        if 'modification_arguments' not in parameters:
            parameters['modification_arguments'] = {}

        for parameter in modification_arguments:
            if parameter not in ['self', 'parameters']:
                self.model_parameters['modification_arguments'][parameter] = local_variables[parameter]
                parameters['modification_arguments'][parameter] = local_variables[parameter]

        # Update parameters
        if update_parameters:
            self.update_spectral_calculation_parameters(
                **parameters
            )

            parameters = self.model_parameters

        # Calculate base spectrum
        if mode == 'emission':
            self.wavelengths, self.fluxes, additional_outputs = (
                self.calculate_emission_spectrum(
                    **parameters
                )
            )
            spectrum = copy.copy(self.fluxes)
        elif mode == 'transmission':
            self.wavelengths, self.transit_radii, additional_outputs = (
                self.calculate_transmission_spectrum(
                    **parameters
                )
            )
            spectrum = copy.copy(self.transit_radii)
        else:
            raise ValueError(f"mode must be 'emission' or 'transmission', not '{mode}'")

        # Modify the spectrum
        wavelengths, spectrum, star_observed_spectrum = self.modify_spectrum(
            wavelengths=copy.copy(self.wavelengths),
            spectrum=spectrum,
            scale_function=self.spectral_modification_functions_map['scale'],
            shift_wavelengths_function=self.spectral_modification_functions_map['shift'],
            transit_fractional_light_loss_function=self.spectral_modification_functions_map['use_transit_light_loss'],
            convolve_function=self.spectral_modification_functions_map['convolve'],
            rebin_spectrum_function=self.spectral_modification_functions_map['rebin'],
            **parameters['modification_arguments'],
            **parameters
        )

        # Store star observed spectrum for debug
        if star_observed_spectrum is not None:
            if 'star_observed_spectrum' not in parameters or update_parameters:
                parameters['star_observed_spectrum'] = star_observed_spectrum

        # Prepare the spectrum
        if prepare:
            spectrum, parameters['preparation_matrix'], parameters['prepared_uncertainties'] = \
                self.spectral_modification_functions_map['prepare'](
                    spectrum=spectrum,
                    wavelengths=wavelengths,
                    **parameters
                )
        else:
            parameters['preparation_matrix'] = np.ones(spectrum.shape)

            if 'data_uncertainties' in parameters:
                parameters['prepared_uncertainties'] = \
                    copy.deepcopy(parameters['data_uncertainties'])
            else:
                parameters['prepared_uncertainties'] = None

        if return_additional_outputs:
            return wavelengths, spectrum, additional_outputs

        return wavelengths, spectrum

    def calculate_transmission_spectrum(
            self,
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
            target_particle_radii: npt.NDArray[np.floating] = None,
            target_pressure: npt.NDArray[np.floating] = None,
            **kwargs
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], dict[str, Any]]:
        """Wrapper for the Radtrans calculate_transit_radii function."""
        _ = kwargs  # kwargs is not used in this function, this is intended

        self.wavelengths, self.transit_radii, additional_outputs = self.calculate_transit_radii(
            temperatures=self.temperatures,
            mass_fractions=self.mass_fractions,
            mean_molar_masses=self.mean_molar_masses,
            reference_gravity=reference_gravity,
            reference_pressure=reference_pressure,
            planet_radius=planet_radius,
            variable_gravity=variable_gravity,
            opaque_cloud_top_pressure=opaque_cloud_top_pressure,
            cloud_particles_mean_radii=cloud_particles_mean_radii,
            cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
            cloud_particles_radius_distribution=cloud_particles_radius_distribution,
            cloud_hansen_a=cloud_hansen_a,
            cloud_hansen_b=cloud_hansen_b,
            cloud_particle_number_density_grid=cloud_particle_number_density_grid,
            cloud_f_sed=cloud_f_sed,
            eddy_diffusion_coefficients=eddy_diffusion_coefficients,
            haze_factor=haze_factor,
            power_law_opacity_350nm=power_law_opacity_350nm,
            power_law_opacity_coefficient=power_law_opacity_coefficient,
            gray_opacity=gray_opacity,
            cloud_fraction=cloud_fraction,
            complete_coverage_clouds=complete_coverage_clouds,
            additional_absorption_opacities_function=additional_absorption_opacities_function,
            additional_scattering_opacities_function=additional_scattering_opacities_function,
            frequencies_to_wavelengths=frequencies_to_wavelengths,
            return_contribution=return_contribution,
            return_clear_spectrum=return_clear_spectrum,
            return_cloud_contribution=return_cloud_contribution,
            return_radius_hydrostatic_equilibrium=return_radius_hydrostatic_equilibrium,
            target_particle_radii=target_particle_radii,
            target_pressure=target_pressure
        )

        return self.wavelengths, self.transit_radii, additional_outputs

    @staticmethod
    def compute_bins_resolving_power(wavelengths):
        """Calculate the resolving power of wavelengths bins.
        The "resolving power" of the bins is defined here as:
            R = wavelengths / wavelength_steps
        This is different from the "true" (/spectral) resolving power:
            R = wavelengths / FWHM_LSF
        where FWHM_LSF is the full width half maximum of the line spread function (aka Delta lambda)

        Args:
            wavelengths: wavelengths at the center of the bins

        Returns:
            The resolving power for each bins
        """
        input_wavelengths_half_diff = np.diff(wavelengths) / 2

        # Take second to before-last element on the last axis of wavelengths
        wavelengths_shape = wavelengths.shape[-1]
        taken_wavelengths = np.take(wavelengths, np.arange(1, wavelengths_shape - 1, 1), axis=-1)
        taken_diffs_m = np.take(input_wavelengths_half_diff, np.arange(0, wavelengths_shape - 2, 1), axis=-1)
        taken_diffs_p = np.take(input_wavelengths_half_diff, np.arange(1, wavelengths_shape - 1, 1), axis=-1)

        resolving_power = taken_wavelengths / (taken_diffs_m + taken_diffs_p)
        resolving_power = np.concatenate((
            [resolving_power[0]],
            resolving_power,
            [resolving_power[-1]],
        ))

        return resolving_power

    @staticmethod
    def compute_cloud_f_sed(cloud_f_sed: float = None, **kwargs):
        """Get the cloud_f_sed parameter. No calculation performed."""
        _ = kwargs  # kwargs is not used in this function, this is intended

        return cloud_f_sed

    @staticmethod
    def compute_cloud_particles_mean_radii(
        cloud_particles_mean_radii: dict[str, npt.NDArray[np.floating]] = None,
        **kwargs
    ):
        """Get the cloud_particles_mean_radii parameter. No calculation performed."""
        _ = kwargs  # kwargs is not used in this function, this is intended

        return cloud_particles_mean_radii

    @staticmethod
    def compute_cloud_particle_radius_distribution_std(cloud_particle_radius_distribution_std: float = None, **kwargs):
        """Get the cloud_particle_radius_distribution_std parameter. No calculation performed."""
        _ = kwargs  # kwargs is not used in this function, this is intended

        return cloud_particle_radius_distribution_std

    @staticmethod
    def compute_clouds_particles_porosity_factor(clouds_particles_porosity_factor: dict[str, float] = None, **kwargs):
        """Get the clouds_particles_porosity_factor parameter. No calculation performed."""
        _ = kwargs  # kwargs is not used in this function, this is intended

        return clouds_particles_porosity_factor

    @staticmethod
    def compute_cloud_hansen_a(cloud_hansen_a: dict[str, npt.NDArray[np.floating]] = None, **kwargs):
        """Get the cloud_hansen_a parameter. No calculation performed."""
        _ = kwargs  # kwargs is not used in this function, this is intended

        return cloud_hansen_a

    @staticmethod
    def compute_cloud_hansen_b(cloud_hansen_b: dict[str, npt.NDArray[np.floating]] = None, **kwargs):
        """Get the cloud_hansen_b parameter. No calculation performed."""
        _ = kwargs  # kwargs is not used in this function, this is intended

        return cloud_hansen_b

    @staticmethod
    def compute_cloud_photosphere_median_optical_depth(cloud_photosphere_median_optical_depth: float = None, **kwargs):
        """Get the cloud_photosphere_median_optical_depth parameter. No calculation performed."""
        _ = kwargs  # kwargs is not used in this function, this is intended

        return cloud_photosphere_median_optical_depth

    @staticmethod
    def compute_emissivities(emissivities: npt.NDArray[np.floating] = None, **kwargs):
        """Get the 'emissivities' parameter. No calculation performed."""
        _ = kwargs  # kwargs is not used in this function, this is intended

        return emissivities

    @staticmethod
    def compute_eddy_diffusion_coefficients(eddy_diffusion_coefficients: npt.NDArray[np.floating] = None, **kwargs):
        """Get the eddy_diffusion_coefficients parameter. No calculation performed."""
        _ = kwargs  # kwargs is not used in this function, this is intended

        return eddy_diffusion_coefficients

    @staticmethod
    def compute_equilibrium_mass_fractions(pressures, temperatures, co_ratio, metallicity,
                                           carbon_pressure_quench=None):
        """Use the per-calculated chemical table to calculate major species mass fractions at equilibrium.

        Args:
            co_ratio:
                Desired carbon to oxygen ratios, obtained by increasing the amount of oxygen.
            metallicity:
                The desired metallitcities.
            temperatures:
                (K) desired temperatures.
            pressures:
                (bar) desired pressures.
            carbon_pressure_quench:
                (bar) pressure at which to put a simplistic carbon-bearing species quenching.

        Returns:
            A dict with species names as keys and mass fractions as values.
        """
        if np.size(co_ratio) == 1:
            co_ratios = np.ones_like(pressures) * co_ratio
        else:
            co_ratios = co_ratio

        if np.size(metallicity) == 1:
            log10_metallicities = np.ones_like(pressures) * metallicity
        else:
            log10_metallicities = metallicity

        log10_metallicities = np.log10(log10_metallicities)

        equilibrium_mass_fractions = (
            pre_calculated_equilibrium_chemistry_table.interpolate_mass_fractions(
                co_ratios=co_ratios,
                log10_metallicities=log10_metallicities,
                temperatures=temperatures,
                pressures=pressures,
                carbon_pressure_quench=carbon_pressure_quench,
                full=False  # no need for nabla_adiabatic or the mean molar mass
            )
        )

        return equilibrium_mass_fractions

    @staticmethod
    def compute_gray_opacity(gray_opacity: float = None, **kwargs):
        """Get the gray_opacity parameter. No calculation performed."""
        _ = kwargs  # kwargs is not used in this function, this is intended

        return gray_opacity

    @staticmethod
    def compute_haze_factor(haze_factor: float = 1.0, **kwargs):
        """Get the haze_factor parameter. No calculation performed."""
        _ = kwargs  # kwargs is not used in this function, this is intended

        return haze_factor

    @staticmethod
    def compute_mass_fractions(pressures, temperatures=None, imposed_mass_fractions=None, line_species=None,
                               cloud_species=None, filling_species=None, use_equilibrium_chemistry=False,
                               metallicity=None, co_ratio=0.55, carbon_pressure_quench=None,
                               verbose=False, **kwargs):
        """Initialize a model mass mixing ratios.
        Ensure that in any case, the sum of mass mixing ratios is equal to 1. Imposed mass mixing ratios are kept to
        their imposed value as long as the sum of the imposed values is lower or equal to 1. H2 and He are used as
        filling gases.
        The different possible cases are dealt with as follows:
            - Sum of imposed mass mixing ratios > 1: the mass mixing ratios are scaled down, conserving the ratio
            between them. Non-imposed mass mixing ratios are set to 0.
            - Sum of imposed mass mixing ratio of all imposed species < 1: if equilibrium chemistry is used or if H2 and
            He are imposed species, the atmosphere will be filled with H2 and He respecting the imposed H2/He ratio.
            Otherwise, the heh2_ratio parameter is used.
            - Sum of imposed and non-imposed mass mixing ratios > 1: the non-imposed mass mixing ratios are scaled down,
            conserving the ratios between them. Imposed mass mixing ratios are unchanged.
            - Sum of imposed and non-imposed mass mixing ratios < 1: if equilibrium chemistry is used or if H2 and
            He are imposed species, the atmosphere will be filled with H2 and He respecting the imposed H2/He ratio.
            Otherwise, the heh2_ratio parameter is used.

        When using equilibrium chemistry with imposed mass mixing ratios, imposed mass mixing ratios are set to their
        imposed value regardless of chemical equilibrium consistency.

        Args:
            pressures: (cgs) pressures of the mass mixing ratios
            line_species: list of line species, required to manage naming differences between opacities and chemistry
            cloud_species: list of cloud species, required to not count cloud species in the gas mass fractions.
            temperatures: (K) temperatures of the mass mixing ratios, used with equilibrium chemistry
            co_ratio: carbon over oxygen ratios of the model, used with equilibrium chemistry
            metallicity: ratio between heavy elements and H2 + He compared to solar, used with equilibrium chemistry
            carbon_pressure_quench: (bar) pressure where the carbon species are quenched, used with equilibrium
                chemistry
            imposed_mass_fractions: imposed mass mixing ratios
            use_equilibrium_chemistry: if True, use pRT equilibrium chemistry module
            filling_species:
                Dictionary with the filling species as keys and the weights of the mass fractions as values. Unweighted
                filling species are represented with None.
            verbose: if True, print additional information

        Returns:
            A dictionary containing the mass mixing ratios.
        """
        _ = kwargs  # kwargs is not used in this function, this is intended

        # Initialization
        if line_species is None:
            line_species = []

        if hasattr(filling_species, '__iter__'):
            # Facilitate the building of H2 and He-only filling species dict
            if len(filling_species) == 2 and not isinstance(filling_species, dict):
                if 'H2' in filling_species and 'He' in filling_species:
                    if use_equilibrium_chemistry:
                        filling_species = {
                            'H2': None,
                            'He': None
                        }
                    else:
                        filling_species = {
                            'H2': 37 * np.ones(pressures.size),
                            'He': 12 * np.ones(pressures.size)
                        }

        # Handle string and non-dict case
        if not isinstance(filling_species, dict):
            if isinstance(filling_species, str):
                filling_species = {
                    filling_species: np.ones(pressures.size)
                }
            elif hasattr(filling_species, '__iter__'):
                filling_species = {species: np.ones(pressures.size) for species in filling_species}
            elif filling_species is None:
                filling_species = {}

            if not isinstance(filling_species, dict):
                raise ValueError(f"filling_species must be a dictionary, but is {filling_species}")

        mass_fractions = {}

        # Initialize imposed mass fractions
        if imposed_mass_fractions is not None:
            for species, mass_fraction in imposed_mass_fractions.items():
                if np.size(mass_fraction) == 1:
                    imposed_mass_fractions[species] = np.ones(np.shape(pressures)) * mass_fraction
                elif np.size(mass_fraction) != np.size(pressures):
                    raise ValueError(f"mass mixing ratio for species '{species}' must be a scalar or an array of the"
                                     f"size of the pressure array ({np.size(pressures)}), "
                                     f"but is of size ({np.size(mass_fraction)})")
        else:
            # Nothing is imposed
            imposed_mass_fractions = {}

        # Remove imposed species from filling species
        for species in imposed_mass_fractions:
            if species in filling_species:
                warnings.warn(f"filling species '{species}' has also imposed mass fractions, "
                              f"the imposed mass fraction will be used")
                del filling_species[species]

        # Add imposed mass fractions to mass fractions
        m_sum_imposed_species = np.zeros(pressures.shape)

        for species, imposed_mass_fraction in imposed_mass_fractions.items():
            mass_fractions[species] = imposed_mass_fraction

            # Do not count clouds
            if species not in cloud_species:
                m_sum_imposed_species += imposed_mass_fraction

        # Ensure that the sum of imposed mass fractions is <= 1
        for i in range(np.size(m_sum_imposed_species)):
            if m_sum_imposed_species[i] > 1:
                if verbose:
                    warnings.warn(f"sum of mass mixing ratios of imposed species ({m_sum_imposed_species}) is > 1, "
                                  f"correcting...")

                for species in imposed_mass_fractions:
                    if species not in cloud_species:  # Do not count clouds
                        mass_fractions[species][i] /= m_sum_imposed_species[i]

        m_sum_imposed_species = np.sum(list(mass_fractions.values()), axis=0)

        # Initialize chemical equilibrium mass fractions
        mass_fractions_chemical_table = None

        if use_equilibrium_chemistry:
            missing = {}

            if metallicity is None:
                missing['metallicity'] = metallicity

            if co_ratio is None:
                missing['co_ratio'] = co_ratio

            if len(missing) > 0:
                names = ', '.join(list(missing.keys()))
                values = list(missing.values())

                raise ValueError(f"equilibrium chemistry function missing {names} values (were {values})\n"
                                 f"Add 'metallicity' as a model parameter, check your metallicity model function, "
                                 f"or disable equilibrium chemistry with 'use_equilibrium_chemistry=False'")

            # Interpolate chemical equilibrium
            mass_fractions_chemical_table = SpectralModel.compute_equilibrium_mass_fractions(
                pressures=pressures,
                temperatures=temperatures,
                co_ratio=co_ratio,
                metallicity=metallicity,
                carbon_pressure_quench=carbon_pressure_quench
            )

        # Add non-imposed mass fractions to mass fractions
        m_sum_chemical_table_species = np.zeros(pressures.shape)

        if mass_fractions_chemical_table is not None:
            # Convert chemical table species names to line species names
            line_species_simple = simplify_species_list(line_species)

            for i, simple_species in enumerate(line_species_simple):
                if simple_species in mass_fractions_chemical_table and simple_species != line_species[i]:
                    mass_fractions_chemical_table[line_species[i]] = copy.deepcopy(
                        mass_fractions_chemical_table[simple_species]
                    )
                    del mass_fractions_chemical_table[simple_species]

            # Remove imposed mass fractions names from chemical table species names
            imposed_species_simple = simplify_species_list(list(imposed_mass_fractions.keys()))

            for simple_species in imposed_species_simple:
                if simple_species in mass_fractions_chemical_table:
                    del mass_fractions_chemical_table[simple_species]

            # Get the sum of mass fractions of non-imposed species
            for species in mass_fractions_chemical_table:
                if species not in imposed_mass_fractions:
                    if species in filling_species:
                        if filling_species[species] is not None:
                            continue

                    mass_fractions[species] = mass_fractions_chemical_table[species]
                    m_sum_chemical_table_species += mass_fractions_chemical_table[species]

        # Ensure that all line species are in mass_fractions
        for species in line_species:
            if species not in mass_fractions:
                warnings.warn(
                    f"line species '{species}' initialised to {sys.float_info.min} ; "
                    f"to remove this warning set use_equilibrium_chemistry to True "
                    f"or add '{species}' and the desired mass mixing ratio to imposed_mass_fractions"
                )

                mass_fractions[species] = sys.float_info.min * np.ones(pressures.size)

        # Ensure that filling species are initialized as an array
        for species in filling_species:
            if species not in mass_fractions:
                mass_fractions[species] = np.zeros(pressures.size)

        # Ensure that the sum of mass mixing ratios of all species is = 1
        m_sum_total = m_sum_chemical_table_species + m_sum_imposed_species

        for i, m_sum in enumerate(m_sum_total):
            if 0 < m_sum < 1:
                if len(filling_species) > 0:  # fill the atmosphere using the filling species
                    mass_fractions_i = fill_atmosphere(
                        mass_fractions=mass_fractions,
                        filling_species=filling_species,
                        fill_layer=i
                    )

                    for species in mass_fractions:
                        mass_fractions[species][i] = mass_fractions_i[species][0]
                else:
                    warnings.warn(f"the sum of mass mixing ratios at level {i} is lower than 1 ({m_sum_total[i]}). "
                                  f"Set filling_species to automatically fill the atmosphere "
                                  f"or manually adjust the imposed mass fractions")
            elif m_sum > 1:  # scale down the mass fractions, no filling species needed
                if verbose:
                    warnings.warn(
                        f"sum of species mass fraction "
                        f"({m_sum_chemical_table_species[i]} + {m_sum_imposed_species[i]}) "
                        f"is > 1, correcting..."
                    )

                for species in mass_fractions:
                    if species not in imposed_mass_fractions:
                        if m_sum_chemical_table_species[i] > 0:
                            mass_fractions[species][i] = (
                                mass_fractions[species][i]
                                * (1 - m_sum_imposed_species[i]) / m_sum_chemical_table_species[i]
                            )
                        else:
                            mass_fractions[species][i] = mass_fractions[species][i] / m_sum
            elif m_sum <= 0:
                raise ValueError(f"invalid total mass fraction at pressure level {i}: "
                                 f"mass fraction must be > 0, but is {m_sum}\n"
                                 f"If the sum is 0, add at least one species with non-zero imposed mass fractions "
                                 f"or set equilibrium chemistry to True\n"
                                 f"If the sum is < 0, check your imposed mass fractions and chemical model")

        return mass_fractions

    @staticmethod
    def compute_mean_molar_masses(mass_fractions, **kwargs):
        """Calculate the mean molar masses.

        Args:
            mass_fractions: dictionary of the mass fractions of the model
            **kwargs: used to store unnecessary parameters

        Returns:

        """
        _ = kwargs  # kwargs is not used in this function, this is intended

        return compute_mean_molar_masses(mass_fractions)

    @staticmethod
    def compute_metallicity(planet_mass=None,
                            star_metallicity=1.0, atmospheric_mixing=1.0, metallicity_mass_coefficient=-0.68,
                            metallicity_mass_scaling=7.2, verbose=False, metallicity=None, **kwargs):
        _ = kwargs  # kwargs is not used in this function, this is intended

        if metallicity is not None:
            return metallicity

        if planet_mass is None:
            return None

        if verbose:
            print("metallicity set to None, calculating it using scaled metallicity...")

        metallicity = SpectralModel.compute_scaled_metallicity(
            planet_mass=planet_mass,
            star_metallicity=star_metallicity,
            atmospheric_mixing=atmospheric_mixing,
            metallicity_mass_coefficient=metallicity_mass_coefficient,
            metallicity_mass_scaling=metallicity_mass_scaling
        )

        if metallicity <= 0:
            warnings.warn(f"non-physical metallicity ({metallicity}), setting its value to near 0")
            metallicity = sys.float_info.min

        return metallicity

    @staticmethod
    def compute_opaque_cloud_top_pressure(opaque_cloud_top_pressure: float = None, **kwargs):
        """Get the opaque_cloud_top_pressure parameter. No calculation performed."""
        _ = kwargs  # kwargs is not used in this function, this is intended

        return opaque_cloud_top_pressure

    @staticmethod
    def compute_optimal_wavelength_boundaries(rebinned_wavelengths, shift_wavelengths_function=None,
                                              relative_velocities=None, rebin_range_margin_power=15, **kwargs
                                              ) -> npt.NDArray[np.floating]:
        # Re-bin requirement is an interval half a bin larger than re-binning interval
        if hasattr(rebinned_wavelengths, 'dtype'):
            if rebinned_wavelengths.dtype != 'O':
                wavelengths_flat = rebinned_wavelengths.flatten()
            else:
                wavelengths_flat = np.concatenate(rebinned_wavelengths)
        else:
            wavelengths_flat = np.concatenate(rebinned_wavelengths)

        if np.ndim(wavelengths_flat) > 1:
            wavelengths_flat = np.concatenate(wavelengths_flat)

        rebin_required_interval = np.array([
            wavelengths_flat[0]
            - (wavelengths_flat[1] - wavelengths_flat[0]) / 2,
            wavelengths_flat[-1]
            + (wavelengths_flat[-1] - wavelengths_flat[-2]) / 2,
        ])

        # Take Doppler shifting into account
        rebin_required_interval_shifted = copy.copy(rebin_required_interval)

        if shift_wavelengths_function is not None:
            if relative_velocities is None:
                warnings.warn("wavelength shifting function provided but no relative velocities, "
                              "assuming relative_velocities = 0 cm.s-1")
                relative_velocities = np.zeros(1)

            rebin_required_interval_shifted[0] = shift_wavelengths_function(
                wavelengths_rest=np.array([rebin_required_interval[0]]),
                relative_velocities=np.array([
                    -np.max(relative_velocities)
                ]),
                **kwargs
            )[0][0]

            rebin_required_interval_shifted[1] = shift_wavelengths_function(
                wavelengths_rest=np.array([rebin_required_interval[1]]),
                relative_velocities=np.array([
                    -np.min(relative_velocities)
                ]),
                **kwargs
            )[0][0]
        elif relative_velocities is not None:
            raise TypeError(f"missing argument 'shift_wavelengths_function' "
                            f"to take into account relative velocities ({relative_velocities})")

        # Ensure that non-shifted spectrum can still be re-binned
        rebin_required_interval[0] = np.min((rebin_required_interval_shifted[0], rebin_required_interval[0]))
        rebin_required_interval[1] = np.max((rebin_required_interval_shifted[1], rebin_required_interval[1]))

        # Satisfy re-bin requirement by increasing the range by the smallest possible significant value
        rebin_required_interval[0] -= 10 ** (np.floor(np.log10(rebin_required_interval[0])) - rebin_range_margin_power)
        rebin_required_interval[1] += 10 ** (np.floor(np.log10(rebin_required_interval[1])) - rebin_range_margin_power)

        return np.array(rebin_required_interval) * 1e4  # cm to um

    @staticmethod
    def compute_orbit_semi_major_axis(orbit_semi_major_axis: float = None, **kwargs):
        """Get the orbit_semi_major_axis parameter. No calculation performed."""
        _ = kwargs  # kwargs is not used in this function, this is intended

        return orbit_semi_major_axis

    @staticmethod
    def compute_orbital_longitudes(times: npt.NDArray[np.floating] = None, mid_transit_time: float = None,
                                   orbital_period: float = None, **kwargs):
        _ = kwargs  # kwargs is not used in this function, this is intended

        if times is None and mid_transit_time is None and orbital_period is None:
            return None

        return Planet.compute_orbital_longitudes(
            times=times - mid_transit_time,
            orbital_period=orbital_period,
            longitude_start=0,
            rad2deg=True
        )

    @staticmethod
    def compute_power_law_opacity_coefficient(power_law_opacity_coefficient: float = None, **kwargs):
        """Get the power_law_opacity_coefficient parameter. No calculation performed."""
        _ = kwargs  # kwargs is not used in this function, this is intended

        return power_law_opacity_coefficient

    @staticmethod
    def compute_power_law_opacity_350nm(power_law_opacity_350nm: float = None, **kwargs):
        """Get the power_law_opacity_350nm parameter. No calculation performed."""
        _ = kwargs  # kwargs is not used in this function, this is intended

        return power_law_opacity_350nm

    @staticmethod
    def compute_planet_radius(planet_radius: float = None, reference_gravity: float = None, planet_mass: float = None,
                              **kwargs):
        _ = kwargs  # kwargs is not used in this function, this is intended

        if planet_radius is not None:
            return planet_radius
        elif reference_gravity is None or planet_mass is None:
            return planet_radius

        return Planet.reference_gravity2radius(
            reference_gravity=reference_gravity,
            mass=planet_mass,
            reference_gravity_error_upper=0,
            reference_gravity_error_lower=0,
            mass_error_upper=0,
            mass_error_lower=0
        )

    @staticmethod
    def compute_radial_velocities(orbital_longitudes, radial_velocity_semi_amplitude, **kwargs):
        _ = kwargs  # kwargs is not used in this function, this is intended

        if orbital_longitudes is not None:
            radial_velocity = Planet.compute_radial_velocity(
                radial_velocity_semi_amplitude=radial_velocity_semi_amplitude,
                orbital_longitude=orbital_longitudes
            )
        else:
            radial_velocity = np.zeros(1)

        return radial_velocity

    @staticmethod
    def compute_radial_velocity_semi_amplitude(star_mass: float = None, orbit_semi_major_axis: float = None, **kwargs):
        """
        Calculate the planet orbital radial velocity semi-amplitude (aka K_p).

        Args:
            star_mass: (g) mass of the star
            orbit_semi_major_axis: (cm) orbit semi major axis
            **kwargs: used to store unnecessary parameters

        Returns:
            (cm.s-1) the planet orbital radial velocity semi-amplitude
        """
        _ = kwargs  # kwargs is not used in this function, this is intended

        if star_mass is not None and orbit_semi_major_axis is not None:
            radial_velocity_semi_amplitude = Planet.compute_orbital_velocity(
                star_mass=star_mass,
                orbit_semi_major_axis=orbit_semi_major_axis
            )
        else:
            radial_velocity_semi_amplitude = 0.0

        return radial_velocity_semi_amplitude

    @staticmethod
    def compute_reference_gravity(reference_gravity: float = None, **kwargs):
        """Get the reference_gravity parameter. No calculation performed."""
        _ = kwargs  # kwargs is not used in this function, this is intended

        return reference_gravity

    @staticmethod
    def compute_reference_pressure(reference_pressure: float = None, **kwargs):
        """Get the reference_pressure parameter. No calculation performed."""
        _ = kwargs  # kwargs is not used in this function, this is intended

        return reference_pressure

    @staticmethod
    def compute_reflectances(reflectances: npt.NDArray[np.floating] = None, **kwargs):
        """Get the 'reflectances' parameter. No calculation performed."""
        _ = kwargs  # kwargs is not used in this function, this is intended

        return reflectances

    @staticmethod
    def compute_relative_velocities(radial_velocities, system_observer_radial_velocities=0.0,
                                    rest_frame_velocity_shift=0.0, **kwargs):
        _ = kwargs  # kwargs is not used in this function, this is intended

        return radial_velocities + system_observer_radial_velocities + rest_frame_velocity_shift

    @staticmethod
    def compute_scaled_metallicity(planet_mass, star_metallicity=1.0, atmospheric_mixing=1.0,
                                   metallicity_mass_coefficient=-0.68, metallicity_mass_scaling=7.2):
        """Calculate the scaled metallicity of a planet.
        The relation used is a power law. Default parameters come from the source.

        Source: Mordasini et al. 2014 (https://www.aanda.org/articles/aa/pdf/2014/06/aa21479-13.pdf)

        Args:
            planet_mass: (g) mass of the planet
            star_metallicity: metallicity of the star in solar metallicity
            atmospheric_mixing: scaling factor [0, 1] representing how well metals are mixed in the atmosphere
            metallicity_mass_coefficient: power of the relation
            metallicity_mass_scaling: scaling factor of the relation

        Returns:
            An estimation of the planet atmospheric metallicity in solar metallicity.
        """
        return (metallicity_mass_scaling * (planet_mass / cst.m_jup) ** metallicity_mass_coefficient
                * star_metallicity * atmospheric_mixing)

    @staticmethod
    def compute_spectral_parameters(model_functions_map, **kwargs):
        for model_parameter, function in model_functions_map.items():
            kwargs[model_parameter] = function(**kwargs)

        return kwargs['temperatures'], kwargs['mass_fractions'], kwargs['mean_molar_masses'], kwargs

    @staticmethod
    def compute_star_effective_temperature(star_effective_temperature: float = None, **kwargs):
        """Get the star_effective_temperature parameter. No calculation performed."""
        _ = kwargs  # kwargs is not used in this function, this is intended

        return star_effective_temperature

    @staticmethod
    def compute_star_irradiation_angle(star_irradiation_angle: float = 0.0, **kwargs):
        """Get the star_irradiation_angle parameter. No calculation performed."""
        _ = kwargs  # kwargs is not used in this function, this is intended

        return star_irradiation_angle

    @staticmethod
    def compute_star_radius(star_radius: float = None, **kwargs):
        """Get the star_radius parameter. No calculation performed."""
        _ = kwargs  # kwargs is not used in this function, this is intended

        return star_radius

    @staticmethod
    def compute_star_flux(star_effective_temperature, mode, is_around_star, **kwargs):
        _ = kwargs  # kwargs is not used in this function, this is intended

        if mode == 'emission' and is_around_star:
            star_data, _ = phoenix_star_table.compute_spectrum(star_effective_temperature)

            star_spectral_radiosities = star_data[:, 1]
            star_spectrum_wavelengths = star_data[:, 0]
        else:
            star_spectrum_wavelengths = None
            star_spectral_radiosities = None

        return np.array([star_spectrum_wavelengths, star_spectral_radiosities])

    @staticmethod
    def compute_stellar_intensities(star_flux, star_radius, orbit_semi_major_axis, mode, is_around_star,
                                    wavelengths=None, **kwargs):
        _ = kwargs  # kwargs is not used in this function, this is intended

        if mode == 'emission' and is_around_star:
            if star_flux[1] is None:
                raise ValueError(f"star flux when is_around_star is True must be defined, but is {star_flux[1]}")

            planet_star_spectral_irradiances = flux2irradiance(
                flux=star_flux[1],
                source_radius=star_radius,
                target_distance=orbit_semi_major_axis
            )  # ingoing radiosity of the star on the planet

            stellar_intensities = planet_star_spectral_irradiances / np.pi  # erg.s-1.cm-2/Hz to erg.s-1.cm-2.sr-1/Hz

            if star_flux[0] is not None:  # otherwise, the star spectral radiosities are re-binned
                stellar_intensities = rebin_spectrum(
                    input_wavelengths=star_flux[0] * 1e4,  # cm to um
                    input_spectrum=stellar_intensities,
                    rebinned_wavelengths=wavelengths
                )
        else:
            stellar_intensities = None

        return stellar_intensities

    @staticmethod
    def compute_temperatures(pressures, temperature_profile_mode='isothermal', temperature=None,
                             intrinsic_temperature=None, reference_gravity=None, metallicity=None,
                             guillot_temperature_profile_gamma=0.4,
                             guillot_temperature_profile_infrared_mean_opacity_solar_metallicity=0.01,
                             guillot_temperature_profile_kappa_ir_z0=None,
                             **kwargs):
        # TODO deprecated: remove guillot_temperature_profile_kappa_ir_z0 in version 4
        _ = kwargs  # kwargs is not used in this function, this is intended

        if guillot_temperature_profile_kappa_ir_z0 is not None:
            warnings.warn(
                "'guillot_temperature_profile_kappa_ir_z0' is deprecated "
                "and will be removed in a future update. "
                "Use 'guillot_temperature_profile_infrared_mean_opacity_solar_metallicity' instead",
                FutureWarning
            )

            if (guillot_temperature_profile_infrared_mean_opacity_solar_metallicity != 0.01
                    and guillot_temperature_profile_kappa_ir_z0 !=
                    guillot_temperature_profile_infrared_mean_opacity_solar_metallicity):
                raise SyntaxError(
                    f"different values for alias keywords: "
                    f"guillot_temperature_profile_infrared_mean_opacity_solar_metallicity "
                    f"({guillot_temperature_profile_infrared_mean_opacity_solar_metallicity}), "
                    f"guillot_temperature_profile_kappa_ir_z0 ({guillot_temperature_profile_kappa_ir_z0})"
                )

            guillot_temperature_profile_infrared_mean_opacity_solar_metallicity = (
                guillot_temperature_profile_kappa_ir_z0)

        SpectralModel.__check_none_model_parameters(
            explanation_message_="Required for calculating the temperature profile",
            temperature=temperature
        )

        if temperature_profile_mode == 'isothermal':
            if np.size(temperature) == 1:
                temperatures = np.ones(np.shape(pressures)) * temperature
            elif np.size(temperature) == np.size(pressures):
                temperatures = np.asarray(temperature)
            else:
                raise ValueError(f"could not initialize isothermal temperature profile ; "
                                 f"possible inputs are a scalar, "
                                 f"or a 1-D array of the same size of parameter 'pressures' ({np.size(pressures)}), "
                                 f"but was: {temperature}")
        elif temperature_profile_mode == 'guillot':
            temperatures = temperature_profile_function_guillot_metallic(
                pressures=pressures,  # TODO (v4.0.0) change TP pressure to CGS
                gamma=guillot_temperature_profile_gamma,
                reference_gravity=reference_gravity,
                intrinsic_temperature=intrinsic_temperature,
                equilibrium_temperature=temperature,
                infrared_mean_opacity_solar_metallicity=guillot_temperature_profile_infrared_mean_opacity_solar_metallicity,  # noqa: E501
                metallicity=metallicity
            )
        else:
            raise ValueError(f"mode must be 'isothermal' or 'guillot', but was '{temperature_profile_mode}'")

        return temperatures

    @staticmethod
    def compute_transit_duration(transit_duration: float = None, **kwargs):
        """Get the transit_duration parameter. No calculation performed."""
        _ = kwargs  # kwargs is not used in this function, this is intended

        return transit_duration

    @staticmethod
    def compute_transit_fractional_light_loss(spectrum, orbit_semi_major_axis, orbital_inclination,
                                              star_radius, orbital_longitudes, transit_duration, orbital_period,
                                              **kwargs):
        """Calculate the transit depth taking into account the transit fractional light loss.
        Spectrum must be scaled.

        Args:
            spectrum:
                The scaled spectrum.
            orbit_semi_major_axis:
                (cm) Semi-major axis of the planet's orbit.
            orbital_inclination:
                (deg) Orbital inclination of the planet's orbit.
            star_radius:
                (cm) Radius of the planet's star.
            orbital_longitudes:
                Orbital longitudes of the observed planet's transit.
            transit_duration:
                (s) Duration of the planet's transit.
            orbital_period:
                (s) Period of the planet's orbit.

        Returns:
            The scaled spectrum, taking into account the transit light loss.
        """
        _ = kwargs  # kwargs is not used in this function, this is intended

        planet_radius_normalized_squared = 1 - spectrum
        planet_radius_normalized = np.sqrt(planet_radius_normalized_squared)

        planet_star_centers_distance = SpectralModel._compute_planet_star_centers_distance(
            orbit_semi_major_axis=orbit_semi_major_axis,
            orbital_inclination=orbital_inclination,
            planet_radius_normalized=planet_radius_normalized,
            star_radius=star_radius,
            orbital_longitudes=orbital_longitudes,
            transit_duration=transit_duration,
            orbital_period=orbital_period
        )

        spectrum_transit_fractional_light_loss = SpectralModel._compute_transit_fractional_light_loss_uniform(
            planet_radius_normalized=planet_radius_normalized,
            planet_radius_normalized_squared=planet_radius_normalized_squared,
            planet_star_centers_distance=planet_star_centers_distance
        )

        return 1 - spectrum_transit_fractional_light_loss

    @staticmethod
    def compute_velocity_range(radial_velocity_semi_amplitude_range,
                               rest_frame_velocity_shift_range,
                               mid_transit_times_range,
                               system_observer_radial_velocities=None, orbital_period=None,
                               orbital_inclination=None,
                               radial_velocity_semi_amplitude_function=None, radial_velocities_function=None,
                               relative_velocities_function=None, orbital_longitudes_function=None,
                               times=None,
                               **kwargs):
        if radial_velocity_semi_amplitude_range is None:
            radial_velocity_semi_amplitude_range = 0

        if rest_frame_velocity_shift_range is None:
            rest_frame_velocity_shift_range = 0

        if mid_transit_times_range is None:
            mid_transit_times_range = 0

        if radial_velocity_semi_amplitude_function is None:
            radial_velocity_semi_amplitude_function = SpectralModel.compute_radial_velocity_semi_amplitude

        if radial_velocities_function is None:
            radial_velocities_function = SpectralModel.compute_radial_velocities

        if relative_velocities_function is None:
            relative_velocities_function = SpectralModel.compute_relative_velocities

        if orbital_longitudes_function is None:
            orbital_longitudes_function = SpectralModel.compute_orbital_longitudes

        velocities, _, _, _ = SpectralModel._compute_relative_velocities_wrap(
            radial_velocity_semi_amplitude_function=radial_velocity_semi_amplitude_function,
            radial_velocities_function=radial_velocities_function,
            relative_velocities_function=relative_velocities_function,
            orbital_longitudes_function=orbital_longitudes_function,
            system_observer_radial_velocities=system_observer_radial_velocities,
            rest_frame_velocity_shift=np.min(rest_frame_velocity_shift_range),
            radial_velocity_semi_amplitude=np.max(np.abs(radial_velocity_semi_amplitude_range)),
            orbital_period=orbital_period,
            times=times,
            mid_transit_time=np.min(mid_transit_times_range),  # the transit happens sooner, spectrum is more r-shifted
            orbital_inclination=orbital_inclination,
            is_orbiting=True,
            radial_velocities=None,
            **kwargs
        )
        velocity_min = np.min(velocities)

        velocities, _, _, _ = SpectralModel._compute_relative_velocities_wrap(
            radial_velocity_semi_amplitude_function=radial_velocity_semi_amplitude_function,
            radial_velocities_function=radial_velocities_function,
            relative_velocities_function=relative_velocities_function,
            orbital_longitudes_function=orbital_longitudes_function,
            system_observer_radial_velocities=system_observer_radial_velocities,
            rest_frame_velocity_shift=np.max(rest_frame_velocity_shift_range),
            radial_velocity_semi_amplitude=np.max(np.abs(radial_velocity_semi_amplitude_range)),
            orbital_period=orbital_period,
            times=times,
            mid_transit_time=np.max(mid_transit_times_range),  # the transit happens later, spectrum is more b-shifted
            is_orbiting=True,
            radial_velocities=None,
            **kwargs
        )
        velocity_max = np.max(velocities)

        return np.array([velocity_min, velocity_max])

    @staticmethod
    def convolve(input_wavelengths, input_spectrum, convolve_resolving_power, constance_tolerance=1e-6, **kwargs):
        """
        Args:
            input_wavelengths: (cm) wavelengths of the input spectrum
            input_spectrum: input spectrum
            convolve_resolving_power: resolving power of output spectrum
            constance_tolerance: relative tolerance on input resolving power to apply constant or running convolutions

        Returns:
            convolved_spectrum: the convolved spectrum at the new resolving power
        """
        input_resolving_powers = SpectralModel.compute_bins_resolving_power(input_wavelengths)

        if np.allclose(input_resolving_powers, np.mean(input_resolving_powers), atol=0.0, rtol=constance_tolerance) \
                and np.size(convolve_resolving_power) <= 1:
            convolved_spectrum = SpectralModel._convolve_constant(
                input_wavelengths=input_wavelengths,
                input_spectrum=input_spectrum,
                convolve_resolving_power=convolve_resolving_power,
                input_resolving_power=input_resolving_powers[0],
                **kwargs
            )
        else:
            convolved_spectrum = SpectralModel._convolve_running(
                input_wavelengths=input_wavelengths,
                input_spectrum=input_spectrum,
                convolve_resolving_power=convolve_resolving_power,
                input_resolving_power=input_resolving_powers,
                **kwargs
            )

        return convolved_spectrum

    @classmethod
    def from_radtrans(
            cls,
            radtrans,
            copy_radtrans=False,
            temperatures=None, mass_fractions=None, mean_molar_masses=None,
            wavelengths=None, transit_radii=None, fluxes=None,
            model_functions_map=None, spectral_modification_functions_map=None,
            **model_parameters
    ):
        with warnings.catch_warnings():
            # Expect UserWarnings caused by pressure not set and no opacity source given
            warnings.filterwarnings("ignore", category=UserWarning)

            empty_radtrans = Radtrans()

            new_spectral_model = cls(
                wavelength_boundaries=radtrans.wavelength_boundaries,
                temperatures=temperatures,
                mass_fractions=mass_fractions,
                mean_molar_masses=mean_molar_masses,
                wavelengths=wavelengths,
                transit_radii=transit_radii,
                fluxes=fluxes,
                model_functions_map=model_functions_map,
                spectral_modification_functions_map=spectral_modification_functions_map,
                **model_parameters
            )

        for k, v in radtrans.__dict__.items():
            if k in new_spectral_model.__dict__ and k in empty_radtrans.__dict__:
                if copy_radtrans:
                    new_spectral_model.__dict__[copy.deepcopy(k)] = copy.deepcopy(v)
                else:
                    new_spectral_model.__dict__[k] = v

        return new_spectral_model

    @classmethod
    def from_retrieval(
            cls,
            retrieval_directory: str,
            sample_extraction_method: Callable = 'median',
            sample_extraction_method_parameters: dict[str, Any] = None,
            model_file: str = None,
            retrieval_name: str = None,
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
            path_input_data: str = None,
            radial_velocity_semi_amplitude_function: Callable = None,
            radial_velocities_function: Callable = None,
            relative_velocities_function: Callable = None,
            orbital_longitudes_function: Callable = None,
            temperatures=None, mass_fractions=None, mean_molar_masses=None,
            wavelengths=None, transit_radii=None, fluxes=None, **model_parameters
    ):
        def __get_median_samples(_samples, **kwargs):
            _ = kwargs  # kwargs is not used in this function, this is intended
            return {_key: np.median(_value) for _key, _value in _samples.items()}

        def __get_best_fit_samples(_samples, **kwargs):
            _ = kwargs  # kwargs is not used in this function, this is intended
            min_log_likelihood_index = np.argmin(_samples['log_likelihood'])
            return {_key: value[min_log_likelihood_index] for _key, _value in _samples.items()}

        available_extraction_methods = {
            'median': __get_median_samples,
            'best': __get_best_fit_samples
        }

        if isinstance(sample_extraction_method, str):
            if sample_extraction_method not in available_extraction_methods:
                quote = "'"
                raise ValueError(f"sample extraction method '{sample_extraction_method}' is not available, "
                                 f"available methods are {quote + ', '.join(available_extraction_methods) + quote}, "
                                 f"alternatively, use a function directly")
            else:
                sample_extraction_method = available_extraction_methods[sample_extraction_method]

        if sample_extraction_method_parameters is None:
            sample_extraction_method_parameters = {}

        if model_file is not None:
            print(f"Loading model from file '{model_file}'...")
            new_spectral_model = cls.load(
                file=model_file,
                path_input_data=path_input_data
            )
        else:
            new_spectral_model = cls(
                pressures=pressures,
                wavelength_boundaries=wavelength_boundaries,
                line_species=line_species,
                gas_continuum_contributors=gas_continuum_contributors,
                rayleigh_species=rayleigh_species,
                cloud_species=cloud_species,
                line_opacity_mode=line_opacity_mode,
                line_by_line_opacity_sampling=line_by_line_opacity_sampling,
                scattering_in_emission=scattering_in_emission,
                emission_cos_angle_grid=emission_angle_grid,
                anisotropic_cloud_scattering=anisotropic_cloud_scattering,
                path_input_data=path_input_data,
                radial_velocity_semi_amplitude_function=radial_velocity_semi_amplitude_function,
                radial_velocities_function=radial_velocities_function,
                relative_velocities_function=relative_velocities_function,
                orbital_longitudes_function=orbital_longitudes_function,
                temperatures=temperatures,
                mass_fractions=mass_fractions,
                mean_molar_masses=mean_molar_masses,
                wavelengths=wavelengths,
                transit_radii=transit_radii,
                fluxes=fluxes,
                **model_parameters
            )

        samples_dict = get_pymultinest_sample_dict(
            output_dir=retrieval_directory,
            name=retrieval_name,
            add_log_likelihood=True,
            add_stats=False
        )

        sample_selection = sample_extraction_method(
            samples_dict,
            **sample_extraction_method_parameters
        )

        # Get fixed parameters by filtering out the retrieved parameters
        fixed_parameters = {}

        for parameter, value in new_spectral_model.model_parameters.items():
            if parameter not in sample_selection and 'log10_' + parameter not in sample_selection:
                fixed_parameters[parameter] = copy.deepcopy(value)

        parameters, _, _, _ = new_spectral_model.__retrieval_parameters_interface(
            retrieved_parameters=sample_selection,
            fixed_parameters=fixed_parameters,
            line_species=new_spectral_model.line_species,
            n_layers=new_spectral_model.pressures.size
        )

        new_spectral_model.model_parameters.update(parameters)

        return new_spectral_model

    def get_default_parameters(self, sort=True, spectral_model_specific=True, from_maps=True):
        """Get all the default SpectralModel parameters."""
        def __list_arguments(_object, exclude_functions=None, include_functions=None, _from_maps=False):
            if exclude_functions is None:
                exclude_functions = []

            if include_functions is None:
                include_functions = []

            _arguments = set()

            if _from_maps:
                functions = (
                    list(_object.model_functions_map.values())
                    + list(_object.spectral_modification_functions_map.values())
                    + include_functions
                )

                for function in functions:
                    if isinstance(function, str):
                        function = getattr(_object, function)

                    if (
                            callable(function)
                            and function not in exclude_functions
                    ):
                        signature = inspect.signature(function)

                        for parameter in signature.parameters:
                            _arguments.add(parameter)
            else:
                for function in dir(_object):
                    if (
                            callable(getattr(_object, function))
                            and (function[0] != '_' or getattr(_object, function) in include_functions)
                            and getattr(_object, function) not in exclude_functions
                    ):
                        signature = inspect.signature(getattr(_object, function))

                        for parameter in signature.parameters:
                            _arguments.add(parameter)

            return _arguments

        default_parameters = __list_arguments(
            self,
            exclude_functions=[
                self.get_default_parameters,
                self.init_data,
                self.init_retrieval,
                self.load,
                self.get_true_parameters,
                self.get_telluric_transmittances,
                self.remove_mask,
                self.retrieval_model_generating_function,
                self.run_retrieval,
                self.save,
                self.save_parameters
            ],
            include_functions=[
                self.__init_velocities,
                self.compute_optimal_wavelength_boundaries,
                self.compute_scaled_metallicity,
                self.calculate_emission_spectrum,
                self.calculate_transmission_spectrum,
                self.with_velocity_range
            ],
            _from_maps=from_maps
        )

        default_parameters.add('modification_arguments')
        default_parameters.add('preparation_matrix')
        default_parameters.add('prepared_uncertainties')

        if spectral_model_specific:
            radtrans_parameters = np.array(list(__list_arguments(Radtrans)))

            default_parameters = set(
                parameter
                for parameter in default_parameters
                if parameter not in radtrans_parameters
            )

        default_parameters = np.array(list(default_parameters))

        if sort:
            default_parameters = np.sort(default_parameters)

        return default_parameters

    def get_model_functions(self, include_functions=None, exclude_functions=None,
                            from_maps=True, model_functions_map=None, spectral_modification_functions_map=None):
        if include_functions is None:
            include_functions = []

        if exclude_functions is None:
            exclude_functions = []

        radtrans_functions = set()

        for function in dir(Radtrans):
            if callable(getattr(Radtrans, function)):
                radtrans_functions.add(function)

        _model_functions = set()

        for function in include_functions:
            if function in exclude_functions:
                raise ValueError(f"function '{function}' is both included and excluded")

            _model_functions.add(function)

        if from_maps:
            if model_functions_map is None:
                model_functions_map = self.model_functions_map

            if spectral_modification_functions_map is None:
                spectral_modification_functions_map = self.spectral_modification_functions_map

            for function in model_functions_map.values():
                if function is None or function in exclude_functions:
                    continue

                _model_functions.add(function)

            for function in spectral_modification_functions_map.values():
                if function is None or function in exclude_functions:
                    continue

                _model_functions.add(function)
        else:
            for function in SpectralModel.__dict__:
                if (callable(getattr(self, function))
                        and 'compute_' in function
                        and function not in exclude_functions
                        and function[0] != '_'
                        and function not in radtrans_functions):
                    _model_functions.add(function)

            for function in self.__dict__:
                if (callable(getattr(self, function))
                        and 'compute_' in function
                        and function not in exclude_functions
                        and function[0] != '_'
                        and function not in radtrans_functions):
                    _model_functions.add(function)

        model_functions = set()

        for function in _model_functions:
            if isinstance(function, str):
                _function = getattr(self, function)
                _function.__name__ = function

                model_functions.add(_function)
            else:
                model_functions.add(function)

        return model_functions

    def get_model_parameters(self, check_relevance=True, from_function_maps=True,
                             model_functions_map=None, spectral_modification_functions_map=None, model_functions=None):
        if model_functions is None:
            model_functions = self.get_model_functions(
                from_maps=from_function_maps,
                model_functions_map=model_functions_map,
                spectral_modification_functions_map=spectral_modification_functions_map
            )

        model_parameters = {}

        # Include all used functions arguments default value into the model parameters
        spectral_model_attributes = list(SpectralModel.__dict__.keys())

        for function in model_functions:
            if function is None:
                continue

            if isinstance(function, str):
                function = getattr(self, function)

            signature = inspect.signature(function)

            for parameter, value in signature.parameters.items():
                if (parameter not in spectral_model_attributes
                        and '_function' not in parameter and parameter not in [
                            'wavelengths', 'pressures', 'temperatures', 'mass_fractions', 'spectrum', 'kwargs', 'mode'
                        ]):  # TODO in v4.0.0, remove these attributes
                    if value.default is not inspect.Parameter.empty:
                        model_parameters[parameter] = value.default
                    else:
                        model_parameters[parameter] = None

        # Include or update instantiated model parameters
        if self.model_parameters is not None:
            for parameter in self.model_parameters:
                model_parameters[parameter] = self.model_parameters[parameter]

        # Initialize the modification parameters dict
        model_parameters['modification_arguments'] = {}
        model_parameters['modification_parameters'] = model_parameters['modification_arguments']  # TODO deprecated, remove in version 4 # noqa: E501

        # Check relevance of model parameters
        if check_relevance:
            self.__check_model_parameters_relevance()

        return model_parameters

    def get_telluric_transmittances(self, file, relative_velocities=None, rewrite=False, tellurics_resolving_power=1e6,
                                    **kwargs):
        from petitRADTRANS.cli.eso_skycalc_cli import get_tellurics_npz

        if relative_velocities is None:
            relative_velocities = self.model_parameters['relative_velocities']

        wavelengths_range = self.compute_optimal_wavelength_boundaries(
            rebinned_wavelengths=self.wavelengths,
            shift_wavelengths_function=self.shift_wavelengths,
            relative_velocities=-relative_velocities,
            **kwargs
        )

        # Add one step at each end to guarantee that at least the required range is fetched for
        wavelengths_range[0] -= wavelengths_range[0] / tellurics_resolving_power
        wavelengths_range[1] += wavelengths_range[1] / tellurics_resolving_power

        return get_tellurics_npz(
            file=file,
            wavelength_range=wavelengths_range,
            rewrite=rewrite,
            **kwargs
        )

    def get_true_parameters(self, retrieved_parameters: dict) -> dict[str, float]:
        """Get the true value of retrieved parameters.
        Intended to be used with plot_result_corner, for retrieval on simulated data.

        Args:
            retrieved_parameters: parameters retrieved during the retrieval

        Returns:
            The true value of the retrieved parameters, in the correct format for plot_result_corner.
        """
        true_parameters = {}

        for parameter in retrieved_parameters:
            if parameter not in self.model_parameters and 'log10_' not in parameter:
                true_parameters[parameter] = (
                    np.mean(np.log10(self.model_parameters['imposed_mass_fractions'][parameter]))
                )
            elif parameter not in self.model_parameters and 'log10_' in parameter:
                _parameter = copy.deepcopy(parameter)
                parameter = parameter.split('log10_', 1)[1]
                true_parameters[_parameter] = np.mean(np.log10(self.model_parameters[parameter]))
            else:
                true_parameters[parameter] = np.mean(self.model_parameters[parameter])

        return true_parameters

    def get_volume_mixing_ratios(self):
        return mass_fractions2volume_mixing_ratios(
            mass_fractions=self.mass_fractions,
            mean_molar_masses=self.mean_molar_masses
        )

    def init_data(self, data_spectrum: npt.NDArray[np.floating],
                  data_wavelengths: npt.NDArray[np.floating],
                  data_uncertainties: npt.NDArray[np.floating],
                  data_name: str = 'data',
                  retrieved_parameters: dict[str, dict[str, Any]] = None, model_parameters: dict[str, Any] = None,
                  fixed_special_parameters: dict[str, Any] = None,
                  mode: str = 'emission', update_parameters: bool = True,
                  telluric_transmittances: npt.NDArray[np.floating] = None,
                  instrumental_deformations: npt.NDArray[np.floating] = None,
                  noise_matrix: npt.NDArray[np.floating] = None,
                  scale: bool = False, shift: bool = False, use_transit_light_loss: bool = False,
                  convolve: bool = False, rebin: bool = False, prepare: bool = False) -> Data:
        """Initialize a Data object using a SpectralModel object.
        Automatically handle the fixed parameters, data mask, and model generating function.
        Fixed parameters, i.e. model parameters that are not retrieved, are stored within the model generating function.

        Args:
            data_spectrum:
                The data spectrum.
            data_wavelengths:
                The data wavelengths.
            data_uncertainties:
                The data uncertainties.
            data_name:
                The data name.
            retrieved_parameters:
                A dictionary with retrieved parameter names as keys and dictionaries as values. Those sub-dictionaries
                must have keys 'prior_parameters' and 'prior_type'. This can also be a list of RetrievalParameter
                objects.
            model_parameters:
                Model parameters to use. Should be None by default.
            fixed_special_parameters:
                In the context of retrieving multiple Data objects, special parameters to be fixed for this Data object,
                while it may be retrieved for other Data objects of the same retrieval.
            mode:
                Radtrans spectral mode. Can be "emission" or "transmission".
            update_parameters:
                If True, update the model parameters.
            telluric_transmittances:
                Telluric transmittances of the model.
            instrumental_deformations:
                Instrumental deformations of the model.
            noise_matrix:
                Noise matrix of the model.
            scale:
                If True, the spectrum is scaled.
            shift:
                If True, the spectrum is Doppler-shifted.
            use_transit_light_loss:
                If True, the transit light loss effect is taken into account to calculate the spectrum.
            convolve:
                If True, the spectrum is convolved.
            rebin:
                If True, the spectrum is re-binned.
            prepare:
                If True, the spectrum is prepared.

        Returns:
            A new Data object instance.
        """
        if retrieved_parameters is None:
            raise TypeError("missing required argument: 'retrieved_parameters'")

        if model_parameters is None:
            model_parameters = copy.deepcopy(self.model_parameters)

        if fixed_special_parameters is None:
            fixed_special_parameters = {}

        # Identify retrieved parameters
        if isinstance(retrieved_parameters, dict):
            retrieved_parameters = RetrievalParameter.from_dict(retrieved_parameters)

        retrieved_parameters_names = [retrieved_parameter.name for retrieved_parameter in retrieved_parameters]

        # Get fixed parameters by filtering out the retrieved parameters
        fixed_parameters: dict[str, Any] = {}

        for parameter, value in model_parameters.items():
            if parameter not in retrieved_parameters_names and 'log10_' + parameter not in retrieved_parameters_names:
                fixed_parameters[parameter] = copy.deepcopy(value)

        special_parameters = Retrieval.get_special_parameters()

        for parameter, value in fixed_special_parameters.items():
            is_special_parameter = False

            for special_parameter in special_parameters:
                _parameter = parameter.split(special_parameter, 1)[0]

                if _parameter == '' or _parameter == '_log10':
                    is_special_parameter = True
                    break

            if not is_special_parameter:
                available = list_str2str(special_parameters)
                raise ValueError(
                    f"parameter '{parameter}' is not a special retrieval parameter\n"
                    f"Available special retrieval parameters are {available}."
                )

            if parameter not in retrieved_parameters_names:
                warnings.warn(
                    f"fixed special parameter '{parameter}' is not a retrieved parameters, so is fixed by default\n"
                    f"To remove this warning, add '{parameter}' to the retrieved parameters, "
                    f"or remove '{parameter}' from 'fixed_special_parameters'.\n"
                    f"Note that 'fixed_special_parameters' is intended to be used in the case that "
                    f"multiple Data objects are retrieved."
                )

            fixed_parameters[parameter] = value

        # Set the model generating function
        def model_generating_function(
            prt_object, parameters, pt_plot_mode=None, amr=False
        ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
            # A special function is needed due to the specificity of the Retrieval object
            return self.retrieval_model_generating_function(
                prt_object=prt_object,
                parameters=parameters,
                fixed_parameters=fixed_parameters,
                pt_plot_mode=pt_plot_mode,
                amr=amr,
                mode=mode,
                update_parameters=update_parameters,
                telluric_transmittances=telluric_transmittances,
                instrumental_deformations=instrumental_deformations,
                noise_matrix=noise_matrix,
                scale=scale,
                shift=shift,
                use_transit_light_loss=use_transit_light_loss,
                convolve=convolve,
                rebin=rebin,
                prepare=prepare
            )

        # Remove data masked values if necessary
        if hasattr(data_spectrum, 'mask'):
            data_spectrum, data_uncertainties, data_mask = SpectralModel.remove_mask(
                data=data_spectrum,
                data_uncertainties=data_uncertainties
            )
        else:
            data_mask = fill_object(copy.deepcopy(data_spectrum), False)

        return Data(
            name=data_name,
            spectrum=data_spectrum,
            wavelengths=data_wavelengths,
            uncertainties=data_uncertainties,
            mask=data_mask,
            radtrans_object=self,
            model_generating_function=model_generating_function
        )

    def init_retrieval(self, data, data_wavelengths, data_uncertainties, retrieval_directory,
                       retrieved_parameters, model_parameters=None, retrieval_name='retrieval',
                       mode='emission', uncertainties_mode='default', update_parameters=False,
                       telluric_transmittances=None, instrumental_deformations=None, noise_matrix=None,
                       scale=False, shift=False, use_transit_light_loss=False, convolve=False, rebin=False,
                       prepare=False,
                       run_mode='retrieval', amr=False, scattering_in_emission=False, pressures=None,
                       dataset_name='data', **kwargs) -> Retrieval:
        if pressures is None:
            pressures = copy.copy(self.pressures)

        if model_parameters is None:
            model_parameters = copy.deepcopy(self.model_parameters)

        retrieval_configuration = RetrievalConfig(
            retrieval_name=retrieval_name,
            run_mode=run_mode,
            amr=amr,
            scattering_in_emission=scattering_in_emission,
            pressures=pressures
        )

        # Retrieved parameters
        if isinstance(retrieved_parameters, dict):
            retrieved_parameters = RetrievalParameter.from_dict(retrieved_parameters)

        for parameter in retrieved_parameters:
            if not hasattr(parameter, 'prior_function'):
                raise AttributeError(
                    f"'{type(parameter)}' object has no attribute 'prior_function': "
                    f"usage of dictionary or a '{type(RetrievalParameter)}' instance is recommended"
                )

            retrieval_configuration.add_parameter(
                name=parameter.name,
                free=True,
                value=None,
                transform_prior_cube_coordinate=parameter.prior_function
            )

        # Fixed parameters
        retrieved_parameters_names = [retrieved_parameter.name for retrieved_parameter in retrieved_parameters]

        for parameter in model_parameters:
            if parameter not in retrieved_parameters_names and 'log10_' + parameter not in retrieved_parameters_names:
                retrieval_configuration.add_parameter(
                    name=parameter,
                    free=False,
                    value=model_parameters[parameter],
                    transform_prior_cube_coordinate=None
                )

        # Remove masked values if necessary
        if hasattr(data, 'mask'):
            data, data_uncertainties, data_mask = SpectralModel.remove_mask(
                data=data,
                data_uncertainties=data_uncertainties
            )
        else:
            data_mask = fill_object(copy.deepcopy(data), False)

        # Set model generating function
        def model_generating_function(prt_object, parameters, pt_plot_mode=None, _amr=False):
            # A special function is needed due to the specificity of the Retrieval object
            return self.retrieval_model_generating_function(
                prt_object=prt_object,
                parameters=parameters,
                pt_plot_mode=pt_plot_mode,
                amr=_amr,
                mode=mode,
                update_parameters=update_parameters,
                telluric_transmittances=telluric_transmittances,
                instrumental_deformations=instrumental_deformations,
                noise_matrix=noise_matrix,
                scale=scale,
                shift=shift,
                use_transit_light_loss=use_transit_light_loss,
                convolve=convolve,
                rebin=rebin,
                prepare=prepare
            )

        # Set Data object
        retrieval_configuration.add_data(
            name=dataset_name,
            path_to_observations=None,
            model_generating_function=model_generating_function,
            radtrans_object=self,
            wavelengths=data_wavelengths,
            spectrum=data,
            uncertainties=data_uncertainties,
            mask=data_mask
        )

        return Retrieval(
            configuration=retrieval_configuration,
            output_directory=retrieval_directory,
            uncertainties_mode=uncertainties_mode,
            **kwargs
        )

    @classmethod
    def load(cls, file, path_input_data=None, overwrite=False):
        if path_input_data is None:
            path_input_data = petitradtrans_config_parser.get_input_data_path()

        # Update the SpectralModel attributes from the file
        with h5py.File(file, 'r') as f:
            parameters = hdf52dict(f)

        del parameters['units']

        radtrans_attributes = Radtrans.__dict__

        discarded_radtrans_attributes = {
            '_cias_loaded_opacities': None,
            '_clouds_loaded_opacities': None,
            '_lines_loaded_opacities': None,
            '_frequency_bins_edges': None,
            '_frequencies': None
        }

        fixed_model_parameters = set()

        # Convert Radtrans properties to input names
        for parameter in list(parameters.keys()):
            if parameter[0] == '_' and parameter[1] != '_':
                if parameter[1:] in radtrans_attributes:
                    if parameter == '_path_input_data':
                        parameters[parameter] = path_input_data
                    elif parameter in discarded_radtrans_attributes:
                        # Remove hidden and opacity-related parameters
                        discarded_radtrans_attributes[parameter] = parameters.pop(parameter)
                        continue

                    if parameter == '_pressures':
                        print("Converting pressures from CGS to bar for Radtrans input...")
                        parameters[parameter[1:]] = parameters.pop(parameter) * 1e-6  # cgs to bar
                    else:
                        parameters[parameter[1:]] = parameters.pop(parameter)
                elif '_Radtrans__' in parameter:
                    del parameters[parameter]
                elif parameter == '_fixed_model_parameters':
                    fixed_model_parameters = set(parameters.pop(parameter))

        # Convert model parameters to instantiation arguments
        for key, value in parameters['model_parameters'].items():
            if key not in parameters:  # prevent Radtrans parameters overwrite
                parameters[key] = value

        del parameters['model_parameters']

        # Remove custom maps
        if 'model_functions_map' in parameters:
            del parameters['model_functions_map']

        if 'spectral_modification_functions_map' in parameters:
            del parameters['spectral_modification_functions_map']

        # Generate an empty SpectralModel
        new_spectrum_model = cls(
            **parameters
        )

        new_spectrum_model._fixed_model_parameters = fixed_model_parameters

        def __check_new_parameters(loaded_parameters: Any, new_parameters, dataset='.'):
            if isinstance(loaded_parameters, dict):
                for _key in loaded_parameters:
                    __check_new_parameters(
                        loaded_parameters[_key],
                        new_parameters[_key],
                        f"{dataset}/{_key}"
                    )
            elif loaded_parameters is None:
                return
            elif np.array(loaded_parameters).dtype == '<U2':
                return
            elif not np.allclose(
                loaded_parameters,
                new_parameters,
                atol=0,
                rtol=10 ** -(sys.float_info.dig + 1)
            ):
                warnings.warn(
                    f"parameter '{dataset}' in loaded SpectralModel is different from the re-generated "
                    f"parameter\n"
                    f"This may be due to the use of different (default) opacities"
                )

                return

        for parameter in discarded_radtrans_attributes:
            if not overwrite:
                __check_new_parameters(
                    loaded_parameters=discarded_radtrans_attributes[parameter],
                    new_parameters=new_spectrum_model.__dict__[parameter],
                    dataset=parameter
                )
            else:
                print(f"Overwriting new parameter '{parameter}' with loaded parameter...")
                new_spectrum_model.__dict__[parameter] = discarded_radtrans_attributes[parameter]

        return new_spectrum_model

    @staticmethod
    def mass2reference_gravity(planet_mass, planet_radius, verbose=False, **kwargs):
        _ = kwargs  # kwargs is not used in this function, this is intended

        if verbose:
            print("reference_gravity set to None, calculating it using surface gravity and radius...")

        if planet_radius is None or planet_mass is None:
            raise ValueError(f"both planet radius ({planet_radius}) "
                             f"and planet mass ({planet_mass}) "
                             f"are required to calculate planet surface gravity")
        elif planet_radius <= 0:
            raise ValueError("cannot calculate surface gravity from planet mass with a radius <= 0")

        return Planet.mass2reference_gravity(
            mass=planet_mass,
            radius=planet_radius
        )[0]

    @staticmethod
    def modify_spectrum(wavelengths, spectrum, mode,
                        scale=False, shift=False, use_transit_light_loss=False, convolve=False, rebin=False,
                        telluric_transmittances_wavelengths=None, telluric_transmittances=None, airmass=None,
                        instrumental_deformations=None, noise_matrix=None,
                        rebinned_wavelengths=None, relative_velocities=None, radial_velocities=None,
                        planet_radius=None,
                        star_flux=None, star_observed_spectrum=None,
                        is_observed=False, star_radius=None, system_distance=None,
                        scale_function=None, shift_wavelengths_function=None,
                        transit_fractional_light_loss_function=None, convolve_function=None,
                        rebin_spectrum_function=None,
                        **kwargs):
        # TODO star observed spectrum will always be re-calculated in this configuration
        # Initialization
        if scale_function is None:
            scale_function = SpectralModel.scale_spectrum

        if shift_wavelengths_function is None:
            shift_wavelengths_function = SpectralModel.shift_wavelengths

        if transit_fractional_light_loss_function is None:
            transit_fractional_light_loss_function = SpectralModel.compute_transit_fractional_light_loss

        if convolve_function is None:
            convolve_function = SpectralModel.convolve

        if rebin_spectrum_function is None:
            rebin_spectrum_function = SpectralModel.rebin_spectrum

        if rebinned_wavelengths is not None:
            if np.ndim(rebinned_wavelengths) <= 1:
                rebinned_wavelengths = np.array([rebinned_wavelengths])

        if star_flux is None:
            star_flux = (None, None)

        star_spectrum_wavelengths = star_flux[0]
        star_spectrum = star_flux[1]  # erg.s-1.cm-2/Hz

        if star_spectrum_wavelengths is not None and star_flux[1] is not None:
            star_spectrum = flux_hz2flux_cm(  # erg.s-1.cm-2/Hz to erg.s-1.cm-2/cm
                star_spectrum,
                frequency2wavelength(star_spectrum_wavelengths)
            )

        if star_observed_spectrum is not None:
            star_observed_spectrum = None  # reset star_observed_spectrum

        if rebin and telluric_transmittances is not None:
            wavelengths_0 = copy.deepcopy(rebinned_wavelengths)
        elif telluric_transmittances is not None:
            wavelengths_0 = copy.deepcopy(wavelengths)
        else:
            wavelengths_0 = None

        # Shift from the planet rest frame to the star system rest frame
        if shift and star_spectrum is not None and mode == 'emission':
            wavelengths_shift_system = shift_wavelengths_function(
                wavelengths_rest=wavelengths,
                relative_velocities=radial_velocities,
                **kwargs
            )

            _star_spectrum = np.zeros(wavelengths_shift_system.shape)

            # Get a star spectrum for each exposure
            for i, wavelength_shift in enumerate(wavelengths_shift_system):
                _, _star_spectrum[i] = SpectralModel._rebin_wrap(
                    wavelengths=star_spectrum_wavelengths,
                    spectrum=star_spectrum,
                    rebinned_wavelengths=wavelength_shift,
                    rebin_spectrum_function=rebin_spectrum_function,
                    **kwargs
                )

            star_spectrum = _star_spectrum

        # Calculate flux received by the observer
        if is_observed and mode == 'emission':
            # Calculate planet radiosity + star radiosity
            if star_spectrum is not None:
                star_spectrum = Radtrans.rebin_star_spectrum(
                    star_spectrum=star_spectrum,
                    star_wavelengths=star_spectrum_wavelengths,
                    wavelengths=wavelengths
                )

                star_observed_spectrum = flux2irradiance(
                    flux=star_spectrum,
                    source_radius=star_radius,
                    target_distance=system_distance
                )

            spectrum = flux2irradiance(
                flux=spectrum,
                source_radius=planet_radius,
                target_distance=system_distance
            )
        else:
            star_observed_spectrum = star_spectrum

        # Scale the spectrum
        if scale:
            spectrum = scale_function(
                spectrum=spectrum,
                star_radius=star_radius,
                star_observed_spectrum=star_observed_spectrum,
                mode=mode,
                **kwargs
            )

        # Shift from the planet rest frame to the observer rest frame
        if shift:
            wavelengths = shift_wavelengths_function(
                wavelengths_rest=wavelengths,
                relative_velocities=relative_velocities,
                **kwargs
            )

            if np.ndim(spectrum) <= 1:  # generate 2D spectrum
                spectrum = np.tile(spectrum, (wavelengths.shape[0], 1))
        else:
            wavelengths = np.array([wavelengths])

            if np.ndim(spectrum) <= 1:  # generate 2D spectrum
                spectrum = np.array([spectrum])

        if use_transit_light_loss:
            if scale and shift:
                # TODO fix _calculate_transit_fractional_light_loss_uniform not working when shift is False
                spectrum = transit_fractional_light_loss_function(
                    spectrum=spectrum,
                    star_radius=star_radius,  # star_radius is not in kwargs
                    **kwargs
                )
            else:
                warnings.warn("'scale' and 'shift' must be True to calculate transit light loss, skipping step...")

        # Add telluric transmittance
        if telluric_transmittances is not None:
            # Rebin spectra on an intermediate wavelength grid, only if re-binning is needed
            '''
            Another way is to rebin the telluric transmittances on the current wavelengths, however this can lead to
            inaccuracies if the rebin function is not precise enough.
            '''
            if shift and rebin:
                # Get the mean resolving power of the current shifted wavelengths grids:
                #   R = lambda / d_lambda
                # With np.diff we get R from "one side" of the bin, to get the resolving power inside the bins:
                #   R = (lambda[:-1] + d_lambda / 2) / d_lambda = lambda / d_lambda + 0.5
                current_resolving_power = np.mean(wavelengths[:, :-1] / np.diff(wavelengths) + 0.5)

                # Get the intermediate wavelength grid at the same resolving power than the current shifted grids
                diff = np.max(np.diff(rebinned_wavelengths))
                _rebinned_wavelengths = resolving_space(
                    start=np.min(rebinned_wavelengths) - diff,
                    stop=np.max(rebinned_wavelengths) + diff,
                    resolving_power=current_resolving_power
                )

                _, spectrum = SpectralModel._rebin_wrap(
                    wavelengths=wavelengths,
                    spectrum=spectrum,
                    rebinned_wavelengths=_rebinned_wavelengths,
                    rebin_spectrum_function=rebin_spectrum_function,
                    **kwargs
                )

                wavelengths = np.tile(_rebinned_wavelengths, (spectrum.shape[0], 1))

            # Initialize arrays
            telluric_transmittances_rebin = np.zeros(spectrum.shape)

            if telluric_transmittances_wavelengths is None:
                telluric_transmittances_wavelengths = wavelengths_0

            if np.ndim(wavelengths) == 1:
                _rebinned_wavelengths = np.array([wavelengths])
            else:
                _rebinned_wavelengths = wavelengths

            # Get a telluric transmittance for each exposure
            if np.ndim(telluric_transmittances) == 1:
                for i, wavelength_shift in enumerate(_rebinned_wavelengths):
                    _, telluric_transmittances_rebin[i] = SpectralModel._rebin_wrap(
                        wavelengths=telluric_transmittances_wavelengths,
                        spectrum=telluric_transmittances,
                        rebinned_wavelengths=wavelength_shift,
                        rebin_spectrum_function=rebin_spectrum_function,
                        **kwargs
                    )

                # Add airmass effect
                if airmass is not None:
                    telluric_transmittances_rebin = np.moveaxis(
                        np.ma.exp(np.ma.log(np.moveaxis(telluric_transmittances_rebin, 0, 1)) * airmass),
                        1, 0
                    )
            elif np.ndim(telluric_transmittances) == 2:
                warnings.warn("using 2D telluric transmittances is not recommended for precise modelling, "
                              "consider using 1D telluric transmittances and providing for airmass at each exposure")

                for i, wavelength_shift in enumerate(rebinned_wavelengths):
                    _, telluric_transmittances_rebin[i] = SpectralModel._rebin_wrap(
                        wavelengths=telluric_transmittances_wavelengths[i],
                        spectrum=telluric_transmittances[i],
                        rebinned_wavelengths=wavelength_shift,
                        rebin_spectrum_function=rebin_spectrum_function,
                        **kwargs
                    )
            else:
                raise ValueError(f"telluric transmittances must have 1 or 2 dimensions, "
                                 f"but have {np.ndim(telluric_transmittances)}")

            spectrum = spectrum * telluric_transmittances_rebin

        # Convolve the spectrum
        if convolve:
            spectrum = SpectralModel._convolve_wrap(
                wavelengths=wavelengths,
                convolve_function=convolve_function,
                spectrum=spectrum,
                **kwargs
            )

        # Rebin the spectrum
        if rebin:
            wavelengths, spectrum = SpectralModel._rebin_wrap(
                wavelengths=wavelengths,
                spectrum=spectrum,
                rebinned_wavelengths=rebinned_wavelengths,
                rebin_spectrum_function=rebin_spectrum_function,
                **kwargs
            )

        # Add instrumental deformations
        if instrumental_deformations is not None:
            spectrum = spectrum * instrumental_deformations

        # Add noise
        if noise_matrix is not None:
            spectrum = spectrum + noise_matrix

        return wavelengths, spectrum, star_observed_spectrum

    @staticmethod
    def preparing_pipeline(spectrum, uncertainties=None,
                           wavelengths=None, airmass=None, tellurics_mask_threshold=0.1, polynomial_fit_degree=1,
                           apply_throughput_removal=True, apply_telluric_lines_removal=True, correct_uncertainties=True,
                           uncertainties_as_weights=False, **kwargs):
        """Interface with retrieval.preparing.preparing_pipeline."""
        # simple_pipeline interface
        if not hasattr(spectrum, 'mask'):
            spectrum = np.ma.masked_array(spectrum)

        if 'uncertainties' in kwargs:  # ensure that spectrum and uncertainties share the same mask
            if hasattr(kwargs['uncertainties'], 'mask'):
                spectrum = np.ma.masked_where(kwargs['uncertainties'].mask, spectrum)

        if np.ndim(spectrum.mask) == 0:
            spectrum.mask = np.zeros(spectrum.shape, dtype=bool)

        return polyfit(
            spectrum=spectrum,
            uncertainties=uncertainties,
            wavelengths=wavelengths,
            airmass=airmass,
            tellurics_mask_threshold=tellurics_mask_threshold,
            polynomial_fit_degree=polynomial_fit_degree,
            apply_throughput_removal=apply_throughput_removal,
            apply_telluric_lines_removal=apply_telluric_lines_removal,
            correct_uncertainties=correct_uncertainties,
            uncertainties_as_weights=uncertainties_as_weights,
            full=True,
            **kwargs
        )

    @staticmethod
    def rebin_spectrum(input_wavelengths, input_spectrum, rebinned_wavelengths, bin_widths=None, **kwargs):
        _ = kwargs  # kwargs is not used in this function, this is intended

        arguments = {
            'input_wavelengths': input_wavelengths,
            'input_spectrum': input_spectrum,
            'rebinned_wavelengths': rebinned_wavelengths,
        }

        if bin_widths is not None:
            arguments['bin_widths'] = bin_widths
            rebin_function = rebin_spectrum_bin
        else:
            rebin_function = rebin_spectrum

        if np.ndim(rebinned_wavelengths) <= 1 and isinstance(rebinned_wavelengths, np.ndarray):
            rebinned_spectrum = rebin_function(**arguments)

            return rebinned_wavelengths, rebinned_spectrum
        else:
            if np.ndim(rebinned_wavelengths) == 2 and isinstance(rebinned_wavelengths, np.ndarray):
                spectra = []
                lengths = []

                for i, wavelengths in enumerate(rebinned_wavelengths):
                    arguments['rebinned_wavelengths'] = wavelengths

                    if bin_widths is not None and np.ndim(bin_widths) == 2:
                        arguments['bin_widths'] = bin_widths[i]

                    rebinned_spectrum = rebin_function(**arguments)

                    spectra.append(rebinned_spectrum)
                    lengths.append(spectra[-1].size)

                if np.all(np.array(lengths) == lengths[0]):
                    spectra = np.array(spectra)
                else:
                    spectra = np.array(spectra, dtype=object)

                return rebinned_wavelengths, spectra
            else:
                raise ValueError(f"parameter 'rebinned_wavelengths' must have at most 2 dimensions, "
                                 f"but has {np.ndim(rebinned_wavelengths)}")

    @staticmethod
    def remove_mask(data, data_uncertainties):
        print('Taking care of mask...')

        return remove_mask(
            data=data,
            data_uncertainties=data_uncertainties
        )

    @staticmethod
    def retrieval_model_generating_function(
        prt_object: Radtrans,
        parameters: dict[str, Any],
        pt_plot_mode: None = None,
        amr: bool = False,
        fixed_parameters: dict[str, Any] | None = None,
        mode: str = 'emission',
        update_parameters: bool = True,
        telluric_transmittances_wavelengths: npt.NDArray[np.floating] | None = None,
        telluric_transmittances: npt.NDArray[np.floating] | None = None,
        instrumental_deformations: npt.NDArray[np.floating] | None = None,
        noise_matrix: npt.NDArray[np.floating] | None = None,
        scale: bool = False,
        shift: bool = False,
        use_transit_light_loss: bool = False,
        convolve: bool = False,
        rebin: bool = False,
        prepare: bool = False
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        _ = pt_plot_mode  # pt_plot_mode is not used in this function, this is intended for compatibility with Retrieval
        _ = amr  # amr is not used in this function, this is intended for compatibility with Retrieval

        _parameters, uncertainty_scaling, spectrum_scaling, spectrum_offset = (
            SpectralModel.__retrieval_parameters_interface(
                retrieved_parameters=parameters,
                fixed_parameters=fixed_parameters,
                line_species=prt_object.line_species,
                n_layers=prt_object.pressures.size
            )
        )

        # Calculate the spectrum
        wavelengths, model = prt_object.calculate_spectrum(
            mode=mode,
            parameters=_parameters,
            update_parameters=update_parameters,
            telluric_transmittances_wavelengths=telluric_transmittances_wavelengths,
            telluric_transmittances=telluric_transmittances,
            instrumental_deformations=instrumental_deformations,
            noise_matrix=noise_matrix,
            scale=scale,
            shift=shift,
            use_transit_light_loss=use_transit_light_loss,
            convolve=convolve,
            rebin=rebin,
            prepare=prepare
        )

        if spectrum_scaling is not None:
            model *= spectrum_scaling

        if spectrum_offset is not None:
            model += spectrum_offset

        return wavelengths, model, uncertainty_scaling

    @staticmethod
    def run_retrieval(retrieval_configuration: RetrievalConfig, retrieval_directory, uncertainties_mode='default',
                      n_live_points=100, resume=False,
                      sampling_efficiency=0.8,
                      const_efficiency_mode=False, log_z_convergence=0.5, n_iter_before_update=50, max_iterations=0,
                      save=True, filename='retrieval_parameters', rank=0, **kwargs):
        retrieval = Retrieval(
            configuration=retrieval_configuration,
            output_directory=retrieval_directory,
            uncertainties_mode=uncertainties_mode,
            **kwargs
        )

        if save:
            parameter_dict = {}  # copy.deepcopy(retrieval.__dict__)  # TODO fix issues with objects not stored in HDF5 (long and difficult, some kind of parser) # noqa: E501

            for arg_name, arg_value in kwargs.items():
                parameter_dict[arg_name] = arg_value

            for argument, value in locals().items():
                if argument in parameter_dict and argument != 'self' and argument != 'kwargs':
                    del parameter_dict[argument]

            if rank == 0:
                SpectralModel.save_parameters(
                    file=os.path.join(retrieval.output_directory, filename + '.h5'),
                    n_live_points=n_live_points,
                    const_efficiency_mode=const_efficiency_mode,
                    log_z_convergence=log_z_convergence,
                    n_iter_before_update=n_iter_before_update,
                    max_iters=max_iterations,
                    **parameter_dict
                )

        retrieval.run(
            sampling_efficiency=sampling_efficiency,
            const_efficiency_mode=const_efficiency_mode,
            n_live_points=n_live_points,
            log_z_convergence=log_z_convergence,
            n_iter_before_update=n_iter_before_update,
            max_iters=max_iterations,
            resume=resume,
            **kwargs
        )

    def save(self, file, save_opacities=False):
        if save_opacities:
            attribute_dictionary = copy.deepcopy(self.__dict__)
        else:
            attribute_dictionary = {
                copy.deepcopy(key): copy.deepcopy(value)
                for key, value in self.__dict__.items()
                if key not in ['_loaded_opacities', 'modification_parameters']
            }

        def __check_map(_map_name, _functions_label):
            for parameter, function in attribute_dictionary[_map_name].items():
                if not isinstance(function, str) and function is not None:
                    attribute_dictionary[_map_name][parameter] = function.__name__

                    if not hasattr(self, attribute_dictionary[_map_name][parameter]):
                        function_name = attribute_dictionary[_map_name][parameter]
                        warnings.warn(
                            f"{_functions_label} '{function_name}' is not a default SpectralModel function\n"
                            f"Only the function's name will be saved"
                        )

        if 'model_functions_map' in attribute_dictionary:
            __check_map('model_functions_map', 'model function')
        else:
            warnings.warn("saved SpectralModel object have no model functions map")

        if 'spectral_modification_functions_map' in attribute_dictionary:
            __check_map('spectral_modification_functions_map', 'spectral modification function')
        else:
            warnings.warn("saved SpectralModel object have no spectral modification functions map")

        self.save_parameters(
            file=file,
            **attribute_dictionary
        )

    @staticmethod
    def save_parameters(file, **kwargs):
        with h5py.File(file, 'w') as f:
            dict2hdf5(
                dictionary=kwargs,  # TODO (long term) units as attributes of dataset
                hdf5_file=f
            )
            f.create_dataset(
                name='units',
                data='all units are in CGS'
            )

    @staticmethod
    def scale_spectrum(spectrum, star_radius, star_observed_spectrum=None,
                       mode='emission', **kwargs):
        _ = kwargs  # kwargs is not used in this function, this is intended

        if mode == 'emission':
            if star_observed_spectrum is None:
                missing = []

                if star_observed_spectrum is None:
                    missing.append('star_observed_spectrum')

                joint = "', '".join(missing)

                raise TypeError(f"missing {len(missing)} positional arguments: '{joint}'")

            return spectrum / star_observed_spectrum
        elif mode == 'transmission':
            return 1 - (spectrum / star_radius) ** 2
        else:
            raise ValueError(f"mode must be 'emission' or 'transmission', not '{mode}'")

    @staticmethod
    def shift_wavelengths(wavelengths_rest, relative_velocities, **kwargs):
        _ = kwargs  # kwargs is not used in this function, this is intended

        wavelengths_shift = np.zeros((relative_velocities.size, wavelengths_rest.size))

        for i, relative_velocity in enumerate(relative_velocities):
            wavelengths_shift[i] = doppler_shift(wavelengths_rest, relative_velocity)

        return wavelengths_shift

    @staticmethod
    def reference_gravity2mass(reference_gravity, planet_radius, verbose=False, **kwargs):
        _ = kwargs  # kwargs is not used in this function, this is intended

        if verbose:
            print("planet_mass set to None, calculating it using surface gravity and radius...")

        if planet_radius is None or reference_gravity is None:
            raise ValueError(f"both planet radius ({planet_radius}) "
                             f"and surface gravity ({reference_gravity}) "
                             f"are required to calculate planet mass")
        elif planet_radius <= 0:
            raise ValueError("cannot calculate planet mass from surface gravity with a radius <= 0")

        return Planet.reference_gravity2mass(
            reference_gravity=reference_gravity,
            radius=planet_radius
        )[0]

    def update_model_functions_map(self, update_model_parameters=True):
        self.model_functions_map = self._get_model_functions_map(update_model_parameters=update_model_parameters)

    def update_spectral_calculation_parameters(self, **parameters):
        pressures = self.pressures * 1e-6  # cgs to bar

        kwargs = {'imposed_mass_fractions': {}}

        for parameter, value in parameters.items():
            if 'log10_' in parameter and value is not None:
                parameter_name = parameter.split('log10_', 1)[-1]

                if parameter_name in kwargs:
                    raise TypeError(f"got multiple values for parameter '{parameter_name}'; "
                                    f"this may be caused by "
                                    f"giving both '{parameter_name}' and e.g. 'log10_{parameter_name}'")

                kwargs[parameter.split('log10_', 1)[-1]] = 10 ** value
            else:
                if parameter == 'imposed_mass_fractions':
                    if value is None:
                        imposed_mass_fractions = {}
                    else:
                        imposed_mass_fractions = parameters[parameter]

                    for species, mass_fractions in imposed_mass_fractions.items():
                        if species not in kwargs[parameter]:
                            kwargs[parameter][species] = copy.copy(mass_fractions)
                elif parameter in kwargs:
                    raise TypeError(f"got multiple values for parameter '{parameter}'; "
                                    f"this may be caused by "
                                    f"giving both '{parameter}' and e.g. 'log10_{parameter}'")
                else:
                    kwargs[parameter] = copy.copy(value)

        self.temperatures, self.mass_fractions, self.mean_molar_masses, self.model_parameters = \
            self.calculate_spectral_parameters(
                pressures=pressures,
                wavelengths=hz2um(self.frequencies),
                **kwargs
            )

        # Adapt chemical names to line species names, as required by Retrieval
        for species in self.line_species:
            spec = species.split('_', 1)[0]

            if spec in self.mass_fractions:
                if species not in self.mass_fractions:
                    self.mass_fractions[species] = self.mass_fractions[spec]
                    del self.mass_fractions[spec]

    @classmethod
    def with_velocity_range(
            cls,
            radial_velocity_semi_amplitude_range: npt.NDArray[np.floating] = None,
            rest_frame_velocity_shift_range: npt.NDArray[np.floating] = None,
            mid_transit_times_range: npt.NDArray[np.floating] = None,
            times: npt.NDArray[np.floating] = None,
            system_observer_radial_velocities: npt.NDArray[np.floating] = None,
            orbital_period: float = None,
            star_mass: float = None,
            orbit_semi_major_axis: float = None,
            rebinned_wavelengths: npt.NDArray[np.floating] = None,
            orbital_inclination: float = 90.0,
            mid_transit_time: float = None,
            radial_velocity_semi_amplitude: float = None,
            rest_frame_velocity_shift: float = None,
            shift_wavelengths_function: Callable = None,
            pressures: npt.NDArray[np.floating] = None,
            line_species: list[str] = None,
            gas_continuum_contributors: list[str] = None,
            rayleigh_species: list[str] = None,
            cloud_species: list[str] = None,
            line_opacity_mode: str = 'c-k',
            line_by_line_opacity_sampling: int = 1,
            scattering_in_emission: bool = False,
            emission_angle_grid: npt.NDArray[np.floating] = None,
            anisotropic_cloud_scattering: bool = 'auto',
            path_input_data: str = petitradtrans_config_parser.get_input_data_path(),
            radial_velocity_semi_amplitude_function: Callable = None,
            radial_velocities_function: Callable = None,
            relative_velocities_function: Callable = None,
            orbital_longitudes_function: Callable = None,
            temperatures=None, mass_fractions=None, mean_molar_masses=None,
            wavelengths=None, transit_radii=None, spectral_radiosities=None, **model_parameters
    ):
        # Initialization
        if shift_wavelengths_function is None:
            shift_wavelengths_function = SpectralModel.shift_wavelengths

        if times is not None:
            if mid_transit_time is None:
                mid_transit_time = np.mean(mid_transit_times_range)
            elif mid_transit_times_range is not None:
                if (mid_transit_time > np.max(mid_transit_times_range)
                        or mid_transit_time < np.min(mid_transit_times_range)):
                    raise ValueError(f"mid_transit_time must be within mid_transit_times_range "
                                     f"({mid_transit_times_range}), "
                                     f"but was {mid_transit_time}")
        else:
            if 'orbital_longitudes' in model_parameters or 'orbital_phases' in model_parameters:
                if 'orbital_longitudes' not in model_parameters:
                    model_parameters['orbital_longitudes'] = np.rad2deg(2 * np.pi * model_parameters['orbital_phases'])
                elif model_parameters['orbital_longitudes'] is None and model_parameters['orbital_phases'] is not None:
                    model_parameters['orbital_longitudes'] = np.rad2deg(2 * np.pi * model_parameters['orbital_phases'])

        if radial_velocity_semi_amplitude is None:
            radial_velocity_semi_amplitude = np.mean(radial_velocity_semi_amplitude_range)
        else:
            if (radial_velocity_semi_amplitude > np.max(radial_velocity_semi_amplitude_range)
                    or radial_velocity_semi_amplitude < np.min(radial_velocity_semi_amplitude_range)):
                raise ValueError(f"radial_velocity_semi_amplitude must be within "
                                 f"radial_velocity_semi_amplitude_range "
                                 f"({radial_velocity_semi_amplitude_range}), "
                                 f"but was {radial_velocity_semi_amplitude}")

        if rest_frame_velocity_shift is None:
            rest_frame_velocity_shift = np.mean(rest_frame_velocity_shift_range)
        else:
            if (rest_frame_velocity_shift > np.max(rest_frame_velocity_shift_range)
                    or rest_frame_velocity_shift < np.min(rest_frame_velocity_shift_range)):
                raise ValueError(f"rest_frame_velocity_shift must be within "
                                 f"rest_frame_velocity_shift_range "
                                 f"({rest_frame_velocity_shift_range}), "
                                 f"but was {rest_frame_velocity_shift}")

        # Get the velocity range
        retrieval_velocities = SpectralModel.compute_velocity_range(
            radial_velocity_semi_amplitude_range=radial_velocity_semi_amplitude_range,
            rest_frame_velocity_shift_range=rest_frame_velocity_shift_range,
            mid_transit_times_range=mid_transit_times_range,
            system_observer_radial_velocities=system_observer_radial_velocities,
            orbital_period=orbital_period,
            orbital_inclination=orbital_inclination,
            star_mass=star_mass,
            orbit_semi_major_axis=orbit_semi_major_axis,
            times=times,
            radial_velocity_semi_amplitude_function=radial_velocity_semi_amplitude_function,
            radial_velocities_function=radial_velocities_function,
            relative_velocities_function=relative_velocities_function,
            orbital_longitudes_function=orbital_longitudes_function,
            **model_parameters
        )

        # Get the wavelengths boundaries from the velocity range
        wavelength_boundaries = SpectralModel.compute_optimal_wavelength_boundaries(
            rebinned_wavelengths=rebinned_wavelengths,
            shift_wavelengths_function=shift_wavelengths_function,
            relative_velocities=retrieval_velocities,
            **model_parameters
        )

        # Generate a SpectralModel using the calculated wavelength boundaries
        new_spectral_model = cls(
            pressures=pressures,
            wavelength_boundaries=wavelength_boundaries,
            line_species=line_species,
            gas_continuum_contributors=gas_continuum_contributors,
            rayleigh_species=rayleigh_species,
            cloud_species=cloud_species,
            line_opacity_mode=line_opacity_mode,
            line_by_line_opacity_sampling=line_by_line_opacity_sampling,
            scattering_in_emission=scattering_in_emission,
            emission_angle_grid=emission_angle_grid,
            anisotropic_cloud_scattering=anisotropic_cloud_scattering,
            path_input_data=path_input_data,
            radial_velocity_semi_amplitude_function=radial_velocity_semi_amplitude_function,
            radial_velocities_function=radial_velocities_function,
            relative_velocities_function=relative_velocities_function,
            orbital_longitudes_function=orbital_longitudes_function,
            temperatures=temperatures,
            mass_fractions=mass_fractions,
            mean_molar_masses=mean_molar_masses,
            wavelengths=wavelengths,
            transit_radii=transit_radii,
            fluxes=spectral_radiosities,
            orbital_period=orbital_period,
            system_observer_radial_velocities=system_observer_radial_velocities,
            orbital_inclination=orbital_inclination,
            star_mass=star_mass,
            orbit_semi_major_axis=orbit_semi_major_axis,
            times=times,
            mid_transit_time=mid_transit_time,
            radial_velocity_semi_amplitude=radial_velocity_semi_amplitude,
            rest_frame_velocity_shift=rest_frame_velocity_shift,
            rebinned_wavelengths=rebinned_wavelengths,
            **model_parameters
        )

        # Save the shift function used
        new_spectral_model.shift_wavelengths = shift_wavelengths_function

        return new_spectral_model
