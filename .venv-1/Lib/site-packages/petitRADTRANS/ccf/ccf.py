"""Useful functions to analyze data using cross-correlation.
The most useful functions are cross_correlate_data_model, and more importantly ccf_analysis, they provide an easy way to
calculate the CCF of data against models.
Function get_co_added_ccf_peak_properties is especially useful to quickly analyze the CCF.
"""

import copy

import numpy as np

from petitRADTRANS.ccf.ccf_core import cross_correlate_matrices, co_add_cross_correlation
from petitRADTRANS.planet import Planet
from petitRADTRANS.physics import doppler_shift, rebin_spectrum


def _check_spectrum_wavelengths_rules(wavelengths, spectrum, wavelength_name='wavelengths', spectra_name='spectra'):
    """Check if wavelengths and spectrum obey the ruling shape.
    The spectrum must have at least 1 dimension, with the last 3 corresponding to: (CCD, exposure, wavelength).
    The wavelengths of the spectrum must have one of the following shapes:
        - 1 dimension (wavelength) but only if the data have 1 or 2 dimensions,
        - 2 dimensions (CCD, wavelength) but only if the data have at least 3 dimensions,
        - the same shape as data.

    Args:
        wavelengths: wavelengths of the spectrum
        spectrum: spectrum
        wavelength_name: variable name of wavelengths, for error message
        spectra_name: variable name of spectra, for error message
    """
    # Check data shapes
    spectrum_shape = np.shape(spectrum)
    spectrum_n_dimensions = np.ndim(spectrum)

    wavelengths_shape = np.shape(wavelengths)
    wavelengths_n_dimensions = np.ndim(wavelengths)

    if spectrum_n_dimensions < 1:
        raise ValueError(f"{spectra_name} must have at least 1 dimension")
    elif spectrum_n_dimensions == 1:
        if wavelengths_shape != spectrum_shape:
            raise ValueError(f"{spectra_name} has 1 dimension; {spectra_name} and {wavelength_name} must have the "
                             f"same shape, but have shape {spectrum_shape} and {wavelengths_shape} ")
    elif spectrum_n_dimensions == 2:
        n_exposures, n_wavelengths = spectrum_shape[-2:]

        if wavelengths_n_dimensions == 1:
            wavelengths_test_shape = (n_exposures, wavelengths_shape[0])
        else:
            wavelengths_test_shape = wavelengths_shape

        if wavelengths_test_shape != spectrum_shape:
            raise ValueError(f"{spectra_name} has 2 dimensions (exposure, wavelength); "
                             f"allowed shapes for {wavelength_name} are: "
                             f"({n_wavelengths}) or {(n_exposures, n_wavelengths)}, "
                             f"but it has shape {wavelengths_shape}")
    else:  # 3D+ spectrum case
        n_ccd, n_exposures, n_wavelengths = spectrum_shape[-3:]
        spectra_shape_others = spectrum_shape[:-3]

        if wavelengths_n_dimensions == 2:
            wavelengths_test_shape = spectra_shape_others + (wavelengths_shape[0], n_exposures, wavelengths_shape[1])
        else:
            wavelengths_test_shape = wavelengths_shape

        if wavelengths_test_shape != spectrum_shape:
            raise ValueError(
                f"{spectra_name} has 3 dimensions or more (..., CCD, exposure, wavelength); "
                f"allowed shapes for {wavelength_name} are: "
                f"({n_ccd}, {n_wavelengths}) or {spectrum_shape}, "
                f"but it has shape {wavelengths_shape}")


def _check_data_model_rules(data, model):
    """Check if data and model obey the ruling shape.
    The model must have one of the following shapes:
        - 1 dimension (wavelength),
        - 2 dimensions (exposure, wavelength) but only if the data have at least 2 dimensions,
        - one dimension less (CCD) than the data (..., exposure, wavelength).
    The model cannot have the CCD dimension, otherwise the re-binning will fail due to insufficient wavelength range,
    taking into account the Doppler shift from the CCF rest velocities.

    Args:
        data: wavelengths of the spectra
        model: variable name of wavelengths, for error message
    """
    data_shape = np.shape(data)
    data_n_dimensions = np.ndim(data)

    model_shape = np.shape(model)
    model_n_dimensions = np.ndim(model)

    if data_n_dimensions < 1:
        raise ValueError("data must have at least 1 dimension")
    elif data_n_dimensions == 1:
        if model_n_dimensions != data_n_dimensions:
            raise ValueError(f"data has 1 dimension; data and model must have the "
                             f"same number of dimensions, but have shape {data_shape} and {model_shape} ")
    elif data_n_dimensions == 2:
        n_exposures, n_wavelengths = data_shape[-2:]

        if model_n_dimensions == 1:
            model_test_shape = (n_exposures, model_shape[0])
        else:
            model_test_shape = model_shape

        if model_test_shape[:-1] != data_shape[:-1]:
            raise ValueError(f"data has 2 dimensions (exposure, wavelength); "
                             f"allowed shapes for model are: "
                             f"({model_shape[-1]}) or {(n_exposures, model_shape[-1])}, "
                             f"but it has shape {model_shape}")
    else:  # 3D+ spectrum case
        n_ccd, n_exposures, n_wavelengths = data_shape[-3:]
        spectra_shape_others = data_shape[:-3]

        if model_n_dimensions == 1:
            model_test_shape = spectra_shape_others + (n_ccd, n_exposures, model_shape[0])
        elif model_n_dimensions == 2:
            model_test_shape = spectra_shape_others + (n_ccd, model_shape[0], model_shape[1])
        else:
            model_test_shape = model_shape[:-2] + (n_ccd, model_shape[-2], model_shape[-1])

        if model_test_shape[:-1] != data_shape[:-1]:
            raise ValueError(
                f"data has 3 dimensions or more (..., CCD, exposure, wavelength); "
                f"allowed shapes for model are: "
                f"({n_exposures}, {model_shape[-1]}) or {data_shape[:-3] + (n_exposures, model_shape[-1])}, "
                f"but it has shape {model_shape}")


def calculate_co_added_ccf_snr(co_added_cross_correlation, rest_velocities, vr_peak_width):
    """Calculate the "signal" to "noise" ratio of each point of a co-added cross correlation map.
    The "SNR" is calculated as the value of the point of the CCF map divided by the standard deviation around that point
    over rest velocities.
    The standard deviation calculation excludes points at a distance vr_peak_width of the evaluated point.

    Args:
        co_added_cross_correlation: co-added cross correlation map
        rest_velocities: 1D array containing the rest velocities of the co-added CCF map
        vr_peak_width: expected width of the peak, in rest velocities units
    """
    co_added_cross_correlation_snr = np.zeros(co_added_cross_correlation.shape)

    for i, ccf in enumerate(co_added_cross_correlation):
        for j, ccf_kp in enumerate(ccf):
            # Find the maximum of the co-added CCF at a given Kp
            index_max = np.argmax(ccf_kp)

            # Select co-added CCF points around the "detected peak", hopefully far enough to not include the peak itself
            noise_indices_lower = np.where(rest_velocities < (rest_velocities[index_max] - vr_peak_width))[0]
            noise_indices_greater = np.where(rest_velocities > (rest_velocities[index_max] + vr_peak_width))[0]
            noise_indices = np.concatenate((noise_indices_lower, noise_indices_greater))

            # Calculate the "signal" to "noise" ratio
            co_added_cross_correlation_snr[i, j, :] = ccf_kp / np.std(ccf_kp[noise_indices])

    return co_added_cross_correlation_snr


def ccf_analysis(wavelengths_data, data, wavelengths_model, model, velocities_ccf=None,
                 model_velocities=None, normalize_ccf=True, calculate_ccf_snr=True, ccf_sum_axes=None,
                 radial_velocity_semi_amplitude=None, system_observer_radial_velocities=None, orbital_longitudes=None,
                 line_spread_function_fwhm=None, pixels_per_resolution_element=2,
                 co_added_ccf_peak_width=None,
                 velocity_interval_extension_factor=0.25, kp_factor=2.0, n_kp=None, n_vr=None,
                 radial_velocity_function=None, **kwargs):
    """Calculate the co-added CCF map of the data against the provided models.
    If velocities_ccf is not provided, one will be calculated using planetary and instrumental parameters.

    Args:
        wavelengths_data: wavelengths of the data
        data: data
        wavelengths_model: wavelengths of the model
        model: model
        velocities_ccf: (cm.s-1) 1D array containing the values of the CCF rest velocities
        model_velocities: (cm.s-1) rest frame velocities of the model, one per model exposure
        normalize_ccf: if True, normalize the data and CCF models by subtracting their respective mean
        calculate_ccf_snr: if True, output the signal-to-noise of the CCF map, based on the standard deviation
        ccf_sum_axes: iterable listing the axes on which to sum the CCF before computing the co-added CCF
        system_observer_radial_velocities: (cm.s-1) array of velocities between the system and the observer
        orbital_longitudes: 1D array containing the orbital longitudes of the data
        radial_velocity_semi_amplitude: (cm.s-1) radial orbital velocity semi-amplitude of the planet (Kp)
        line_spread_function_fwhm: (cm.s-1) Full Width at Half-Maximum of the instrument line spread function (LSF)
        co_added_ccf_peak_width: (cm.s-1) width of the CCF peak, for the SNR, 3 times the LSF px size by default
        pixels_per_resolution_element: number of spectral pixels per resolution element for the instrument
        velocity_interval_extension_factor: extension to the calculated velocities_ccf interval, set to 0 for no
            extension
        kp_factor: used to set the boundaries of the Kp space relative to the planet Kp (2 -> boundaries at 2 times Kp)
        n_kp: length of the Kp space, same as the size of velocities_ccf by default
        n_vr: length of the rest velocity space, same as the size of velocities_ccf by default
        radial_velocity_function: function to calculate the planet radial velocity with respect to the observer,
            must have at least the following arguments:
                - radial_velocity_semi_amplitude,
                - planet_orbital_inclination,
                - orbital_longitude=orbital_longitudes
            the Planet.calculate_radial_velocity function is used by default

    Returns:
        co_added_cross_correlations_snr: only if calculate_ccf_snr is True, the SNR of the co-added CCF map
        co_added_cross_correlations: the co-added CCF map of the data against the models
        v_rest: the rest velocities of the co-added CCF map
        kps: the orbit radial velocity semi-amplitudes of the co-added CCF map
        ccf_sum: the sum of CCFs used to calculate the co-added CCF
        ccfs: the CCFs of the data against the models
        velocities_ccf: the rest velocities of the CCF
        ccf_model: the models used to calculate the CCFs
        ccf_model_wavelengths: the wavelengths of the models used to calculate the CCFs
    """
    # Initialization
    if velocities_ccf is None:
        velocities_ccf = get_ccf_velocity_space(
            system_observer_radial_velocities=system_observer_radial_velocities,
            radial_velocity_semi_amplitude=radial_velocity_semi_amplitude,
            line_spread_function_fwhm=line_spread_function_fwhm,
            pixels_per_resolution_element=pixels_per_resolution_element,
            velocity_interval_extension_factor=velocity_interval_extension_factor,
        )

    if co_added_ccf_peak_width is None:
        if line_spread_function_fwhm is None:
            raise ValueError("co_added_ccf_peak_width must be a scalar or determined from line_spread_function_fwhm, "
                             "but both are None")

        co_added_ccf_peak_width = 3 * line_spread_function_fwhm / pixels_per_resolution_element

    ccfs, ccf_models, ccf_model_wavelengths = cross_correlate_data_model(
        wavelengths_data=wavelengths_data,
        data=data,
        wavelengths_model=wavelengths_model,
        model=model,
        velocities_ccf=velocities_ccf,
        model_velocities=model_velocities,
        normalize_ccf=normalize_ccf
    )

    ccf_sum = copy.deepcopy(ccfs)

    if ccf_sum_axes is not None:
        if not hasattr(ccf_sum_axes, '__iter__'):
            ccf_sum_axes = (ccf_sum_axes,)

        for i, axis in enumerate(ccf_sum_axes):
            ccf_sum = np.sum(ccf_sum, axis=axis - i)

    if np.ndim(ccf_sum) >= 3:
        ccf_sum = ccf_sum.reshape((-1, ccf_sum.shape[-2], ccf_sum.shape[-1]))  # flatten to get number of co-additions
    elif np.ndim(ccf_sum) == 2:
        ccf_sum = np.array([ccf_sum])  # 1 co-addition
    else:
        raise ValueError(f"CCF must have at least 2 dimensions, but has shape {ccf_sum.shape}; something went wrong "
                         f"(CCF sum axes were {ccf_sum_axes} on CCF of shape {ccfs.shape})")

    co_added_velocities, kps, v_rest = get_co_added_ccf_velocity_space(
        radial_velocity_semi_amplitude=radial_velocity_semi_amplitude,
        velocities_ccf=velocities_ccf,
        system_observer_radial_velocities=system_observer_radial_velocities,
        orbital_longitudes=orbital_longitudes,
        kp_factor=kp_factor,
        n_kp=n_kp,
        n_vr=n_vr,
        radial_velocity_function=radial_velocity_function,
        **kwargs
    )

    co_added_cross_correlations = np.zeros((ccf_sum.shape[0], kps.size, v_rest.size))

    for i, ccf_ in enumerate(ccf_sum):
        co_added_cross_correlations[i] = co_add_cross_correlation(
            cross_correlation=ccf_,
            velocities_ccf=velocities_ccf,
            co_added_velocities=co_added_velocities
        )

    if calculate_ccf_snr:
        co_added_cross_correlations_snr = calculate_co_added_ccf_snr(
            co_added_cross_correlation=co_added_cross_correlations,
            rest_velocities=v_rest,
            vr_peak_width=co_added_ccf_peak_width
        )

        return co_added_cross_correlations_snr, co_added_cross_correlations, \
            v_rest, kps, ccf_sum, ccfs, velocities_ccf, ccf_models, ccf_model_wavelengths
    else:
        return co_added_cross_correlations, \
            v_rest, kps, ccf_sum, ccfs, velocities_ccf, ccf_models, ccf_model_wavelengths


def cross_correlate_data_model(wavelengths_data, data, wavelengths_model, model, velocities_ccf=None,
                               model_velocities=None, normalize_ccf=True):
    """Cross correlate a model with data.
    The data must have at least 2 dimensions, with the last 3 corresponding to: (..., CCD, exposure, wavelength).
    The wavelengths of the data must have one of the following shapes:
        - 1 dimension (wavelength) but only if the data have 2 dimensions,
        - 2 dimensions (exposure, wavelength) but only if the data have 2 dimensions,
        - 2 dimensions (CCD, wavelength) but only if the data have at least 3 dimensions,
        - the same shape as data.
    The model must have 1 (wavelength), 2 (exposure, wavelength), or one dimension less than the data
    (..., exposure, wavelength). The model must have no CCD axis.
    The wavelengths of the model must follow the same rules as for the wavelengths of the data.

    Args:
        wavelengths_data: wavelengths of the data
        data: data
        wavelengths_model: wavelengths of the model
        model: model
        velocities_ccf: (cm.s-1) 1D array containing the values of the CCF rest velocities
        model_velocities: (cm.s-1) rest frame velocities of the model, one per model exposure
        normalize_ccf: if True, normalize the data and CCF models by subtracting their respective mean

    Returns:
        The CCF (..., exposure, CCF rest velocities), the CCF models and CCF models wavelengths
    """
    # Input checks
    _check_spectrum_wavelengths_rules(
        wavelengths=wavelengths_data,
        spectrum=data,
        wavelength_name='wavelength_data',
        spectra_name='data'
    )

    _check_spectrum_wavelengths_rules(
        wavelengths=wavelengths_model,
        spectrum=model,
        wavelength_name='wavelength_model',
        spectra_name='model'
    )

    # Initialization
    if model_velocities is None:
        model_velocities = 0

    # Check velocities shape
    if np.ndim(model) >= 2 and (np.ndim(model_velocities) == 0
                                or (np.ndim(model_velocities) > 0
                                    and np.size(model_velocities) != np.shape(model)[-2])):
        raise ValueError(f"model has at least 2 dimensions; "
                         f"relative_velocities must be of shape ({np.shape(model)[-2]}), "
                         f"but have shape {np.shape(model_velocities)}")

    if np.ndim(velocities_ccf) != 1:
        raise ValueError(f"velocities_ccf must have 1 dimension, "
                         f"but have shape {np.shape(velocities_ccf)}")

    # Doppler-shift the model to the CCF rest velocities
    ccf_model_wavelengths = get_ccf_model_wavelengths(
        wavelengths_model=wavelengths_model,
        velocities_ccf=velocities_ccf,
        relative_velocities=model_velocities
    )

    ccf_models = get_ccf_models(
        data=data,
        wavelengths_data=wavelengths_data,
        model=model,
        ccf_model_wavelengths=ccf_model_wavelengths
    )

    # Move velocity axis as first axis
    ccf_model_wavelengths = np.moveaxis(ccf_model_wavelengths, -2, 0)
    ccf_models = np.moveaxis(ccf_models, -2, 0)

    # Calculate CCF
    ccfs = np.zeros((np.size(velocities_ccf),) + np.shape(data)[:-1])

    if normalize_ccf:
        # Calculate the mean of the matrices over wavelengths, moveaxis is used to enable numpy array operation
        data = np.moveaxis(np.moveaxis(data, -1, 0) - np.mean(data, axis=-1), 0, -1)
        ccf_models = np.moveaxis(np.moveaxis(ccf_models, -1, 0) - np.mean(ccf_models, axis=-1), 0, -1)

    for i in range(np.size(velocities_ccf)):
        ccfs[i] = cross_correlate_matrices(data, ccf_models[i])

    ccfs = np.moveaxis(ccfs, 0, -1)  # move velocity axis to the last axis

    return ccfs, ccf_models, ccf_model_wavelengths


def get_ccf_models(data, wavelengths_data, model, ccf_model_wavelengths):
    """Get the models that will be used for the CCF analysis using the wavelengths shifted according to the CCF rest
    velocity axis.
    The model is re-binned to the wavelengths of the data to obtain a matrix with the same shape as the data, plus one
    dimension for the velocity axis.

    Args:
        data: data
        wavelengths_data: wavelengths of the data
        model: variable name of wavelengths, for error message
        ccf_model_wavelengths: wavelengths of the model Doppler-shifted to the CCF rest velocities
    """
    _check_data_model_rules(data, model)

    # Initialization
    model_shape = np.shape(model)

    n_wavelengths = np.shape(wavelengths_data)[-1]

    if np.ndim(data) >= 2:
        n_exposures = np.shape(data)[-2]

        if (np.ndim(data) >= 3 and np.ndim(wavelengths_data) == 2) or np.ndim(wavelengths_data) >= 3:
            n_ccd = np.shape(data)[-3]
        else:
            n_ccd = 1
    else:
        n_exposures = 1
        n_ccd = 1

    if 1 < np.ndim(data) == np.ndim(wavelengths_data):
        exposures_in_data_wavelengths = True
    else:
        exposures_in_data_wavelengths = False

    size_rest_velocity_grid = np.shape(ccf_model_wavelengths)[-2]

    if np.ndim(model) == 1:
        model = np.array([model])  # add exposure dimension to be consistent with ccf_wavelengths_model
        models_view = model.reshape(
            (-1, 1, model_shape[-1])
        )
        ccf_model_wavelengths_view = ccf_model_wavelengths.reshape(
            (-1, 1, size_rest_velocity_grid, ccf_model_wavelengths.shape[-1])
        )

        exposures_in_model = False
    else:
        models_view = model.reshape(
            (-1, n_exposures, model_shape[-1])
        )
        ccf_model_wavelengths_view = ccf_model_wavelengths.reshape(
            (-1, n_exposures, size_rest_velocity_grid, ccf_model_wavelengths.shape[-1])
        )

        exposures_in_model = True

    n_others = ccf_model_wavelengths_view.shape[0]

    ccf_models = np.zeros(np.shape(data)[:-3] + (n_ccd, n_exposures, size_rest_velocity_grid, n_wavelengths))
    ccf_models_view = ccf_models.reshape(
        (-1, n_ccd, n_exposures, size_rest_velocity_grid, n_wavelengths)
    )

    # Ensure that there is always a CCD dimension
    n_dimensions_wavelengths_data = np.ndim(wavelengths_data)
    other_dimensions_in_wavelengths_data = False

    if n_dimensions_wavelengths_data == 1:
        wavelengths_data_view = np.array([wavelengths_data])  # add one CCD
    elif n_dimensions_wavelengths_data == np.ndim(data):  # there is always an exposure dimension in that case
        if n_dimensions_wavelengths_data == 2:
            wavelengths_data = np.array([wavelengths_data])  # add one CCD
        else:
            other_dimensions_in_wavelengths_data = True

        wavelengths_data_view = wavelengths_data.reshape(
            (-1, n_ccd, n_exposures, np.shape(wavelengths_data)[-1])  # exposures are always in before-last dimension
        )
    else:  # data wavelengths have dimension (CCD, wavelength)
        wavelengths_data_view = wavelengths_data  # no need to reshape

    # Slightly dodgy way to deal with the changes in dimensionality
    if exposures_in_data_wavelengths:
        if other_dimensions_in_wavelengths_data:
            def __id_data_wavelengths(others, ccd, exposure):
                return others, ccd, exposure
        else:
            def __id_data_wavelengths(_, ccd, exposure):
                return ccd, exposure
    else:
        def __id_data_wavelengths(_, ccd, __):
            return ccd

    if exposures_in_model:
        def __id_model(others, exposure):
            return others, exposure

        def __id_wavelengths_model(others, exposure, velocity):
            return others, exposure, velocity
    else:
        def __id_model(others, _):
            return others, 0

        def __id_wavelengths_model(others, _, velocity):
            return others, 0, velocity

    # Re-binning
    for i in range(n_others):
        for c in range(n_ccd):
            for j in range(n_exposures):
                for k in range(size_rest_velocity_grid):
                    ccf_models_view[i, c, j, k] = rebin_spectrum(
                        input_wavelengths=ccf_model_wavelengths_view[__id_wavelengths_model(i, j, k)],
                        input_spectrum=models_view[__id_model(i, j)],
                        rebinned_wavelengths=wavelengths_data_view[__id_data_wavelengths(i, c, j)]
                    )

    '''
    ccf_models and ccf_models_view have the same address in memory, so ccf_models is already modified
    '''

    return ccf_models


def get_ccf_model_wavelengths(wavelengths_model, velocities_ccf, relative_velocities=None):
    """Get the CCF wavelengths for the models that will be used for the CCF analysis.
    The wavelengths are Doppler-shifted to the rest velocities values of the CCF.

    Args:
        wavelengths_model: wavelengths of the model
        velocities_ccf: rest velocities used for the CCF
        relative_velocities: velocities used to build the model
    """
    # Initialization
    if relative_velocities is None:
        relative_velocities = 0

    shape_others = ()

    if np.ndim(wavelengths_model) >= 2:
        n_exposures = np.shape(wavelengths_model)[-2]

        if np.ndim(wavelengths_model) >= 3:
            shape_others = np.shape(wavelengths_model)[:-1]

        if np.ndim(relative_velocities) == 0:
            relative_velocities = np.ones(n_exposures) * relative_velocities

        if np.size(relative_velocities) != n_exposures:
            raise ValueError(f"wavelengths of model has 2 or more dimensions {np.shape(wavelengths_model)}; "
                             f"relative velocities must be a scalar "
                             f"or a 1D array of size {n_exposures}, "
                             f"but has shape {np.shape(relative_velocities)}")

    elif np.ndim(relative_velocities) == 1:  # 2D+ model
        n_exposures = np.size(relative_velocities)
    else:  # 1D model
        n_exposures = 1

        if np.ndim(relative_velocities) == 0:
            relative_velocities = np.ones(n_exposures) * relative_velocities

    n_velocities = np.size(velocities_ccf)

    # Insert velocity dimension
    wavelengths_model_shifted = np.zeros(
        shape_others + (n_exposures, n_velocities, np.shape(wavelengths_model)[-1])
    )

    # Create views for efficient operation
    wavelengths_model_shifted_view = wavelengths_model_shifted.reshape(
        (-1, n_exposures, n_velocities, np.shape(wavelengths_model)[-1])
    )

    if np.ndim(wavelengths_model) >= 2:
        exposures_in_wavelengths = True
        wavelengths_model_view = wavelengths_model.reshape(
            (-1, n_exposures, np.shape(wavelengths_model)[-1])
        )
    else:
        exposures_in_wavelengths = False
        wavelengths_model_view = wavelengths_model.reshape(
            (-1, np.shape(wavelengths_model)[-1])
        )

    # Shift model wavelengths by desired velocities
    if exposures_in_wavelengths:
        for i in range(wavelengths_model_shifted_view.shape[0]):
            for j in range(n_exposures):
                for k in range(n_velocities):
                    wavelengths_model_shifted_view[i, j, k] = doppler_shift(
                        wavelength_0=wavelengths_model_view[i, j],
                        velocity=velocities_ccf[k] - relative_velocities[j]
                    )
    else:
        for i in range(wavelengths_model_shifted_view.shape[0]):
            for j in range(n_exposures):
                for k in range(n_velocities):
                    wavelengths_model_shifted_view[i, j, k] = doppler_shift(
                        wavelength_0=wavelengths_model_view[i],
                        velocity=velocities_ccf[k] - relative_velocities[j]
                    )

    '''
    wavelengths_model_shifted and wavelengths_model_shifted_view have the same address in memory, so
    wavelengths_model_shifted is already modified
    '''

    return wavelengths_model_shifted


def get_ccf_velocity_space(system_observer_radial_velocities, radial_velocity_semi_amplitude,
                           line_spread_function_fwhm, pixels_per_resolution_element,
                           velocity_interval_extension_factor=0.25):
    """Get a velocity space of for CCF calculations."""
    # Get min and max velocities based on the planet parameters
    velocity_min = np.min(system_observer_radial_velocities) - radial_velocity_semi_amplitude
    velocity_max = np.max(system_observer_radial_velocities) + radial_velocity_semi_amplitude

    # Add a margin to the boundaries
    velocity_interval = velocity_max - velocity_min
    velocity_min -= velocity_interval_extension_factor * velocity_interval
    velocity_max += velocity_interval_extension_factor * velocity_interval

    # Set the velocity step as the equivalent to the instrument spectral resolution
    # A finer step won't be fully resolved by the instrument
    velocity_step = line_spread_function_fwhm / pixels_per_resolution_element

    # Get the velocities
    n_low = int(np.floor(velocity_min / velocity_step))
    n_high = int(np.ceil(velocity_max / velocity_step))

    velocities = np.linspace(
        n_low * velocity_step,
        n_high * velocity_step,
        n_high - n_low + 1
    )

    return velocities


def get_co_added_ccf_peak_properties(co_added_cross_correlation, kp_space, vr_space, peak_cutoff):
    """Get the location of the co-added cross-correlation ("SNR") peak and the number of points around the peak.

    Args:
        co_added_cross_correlation: the co-added CCF map to get the peak from
        kp_space: 1D array containing the orbital radial velocity semi-amplitudes of the co-added CCF map
        vr_space: 1D array containing the rest velocities of the co-added CCF map
        peak_cutoff: all points below this factor of the maximum are not considered part of the peak

    Returns:
        The value of the CCF peak, the Kp and rest velocity of the peak, and the number of points of the peak
    """
    ccf_tot_max = np.max(co_added_cross_correlation)

    max_coord = np.nonzero(co_added_cross_correlation == ccf_tot_max)
    max_kp = kp_space[max_coord[0]][0]
    max_v_rest = vr_space[max_coord[1]][0]

    n_around_peak = np.size(np.nonzero(co_added_cross_correlation >= peak_cutoff * ccf_tot_max))

    return ccf_tot_max, max_kp, max_v_rest, n_around_peak


def get_co_added_ccf_velocity_space(radial_velocity_semi_amplitude, velocities_ccf,
                                    system_observer_radial_velocities, orbital_longitudes,
                                    kp_factor=2.0,
                                    n_kp=None, n_vr=None, radial_velocity_function=None, **_):
    # Initializations
    if n_kp is None:
        n_kp = np.size(velocities_ccf)

    if n_vr is None:
        n_vr = np.size(velocities_ccf)

    if radial_velocity_function is None:
        radial_velocity_function = Planet.compute_radial_velocity

    n_exposures = np.size(orbital_longitudes)

    # Get Kp space
    kps = np.linspace(
        -radial_velocity_semi_amplitude * kp_factor,
        radial_velocity_semi_amplitude * kp_factor,
        n_kp
    )

    # Calculate the planet relative velocities in the Kp space
    planet_observer_radial_velocities = system_observer_radial_velocities + np.array([
        radial_velocity_function(
            radial_velocity_semi_amplitude=kp,
            orbital_longitude=orbital_longitudes,  # phase to longitude (deg)
        ) for kp in kps
    ])

    # Set rest velocity space boundaries taking the CCF velocities into account to prevent out-of-bound interpolation
    v_planet_min = np.min(planet_observer_radial_velocities)
    v_planet_max = np.max(planet_observer_radial_velocities)

    v_rest_min = np.max((
        -radial_velocity_semi_amplitude * kp_factor,
        np.min(velocities_ccf) - v_planet_min
    ))
    v_rest_max = np.min((
        radial_velocity_semi_amplitude * kp_factor,
        np.max(velocities_ccf) - v_planet_max
    ))
    v_rest = np.linspace(v_rest_min, v_rest_max, n_vr)

    # Get velocity space
    velocities = np.zeros((n_kp, n_exposures, n_vr))

    for i in range(n_kp):
        for j in range(n_exposures):
            velocities[i, j] = v_rest + planet_observer_radial_velocities[i, j]

    return velocities, kps, v_rest
