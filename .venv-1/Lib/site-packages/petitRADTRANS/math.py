"""Stores useful mathematical functions.
"""
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from scipy.ndimage import gaussian_filter
from scipy.optimize import lsq_linear
from scipy.special import erf, erfinv, lambertw

# noinspection PyUnresolvedReferences
from petitRADTRANS.fortran_convolve import fortran_convolve as fconvolve


def bayes_factor2sigma(bayes_factor: float) -> float | npt.NDArray[np.floating]:
    """
    Convert a Bayes factor, or "evidence", into a sigma significance. For Bayes factor higher than exp(25), the function
    is approximated with a square root function.
    Note: sometimes algorithms return the "log-evidence", or ln(z). The Bayes factor is z.
    Source: Benneke et al. 2013 https://iopscience.iop.org/article/10.1088/0004-637X/778/2/153
    Molliere part by Paul Mollière.
    :param bayes_factor: Bayes factor (aka "evidence")
    :return: sigma significance
    """
    molliere_threshold = 72004899337  # ~exp(25), roughly where numerical errors start to be significant
    is_scalar = False

    if not hasattr(bayes_factor, '__iter__'):
        is_scalar = True
        bayes_factor = np.array([bayes_factor])
    elif not isinstance(bayes_factor, np.ndarray):
        bayes_factor = np.array(bayes_factor)

    benneke_part = np.nonzero(np.less_equal(bayes_factor, molliere_threshold))
    molliere_part = np.nonzero(np.greater(bayes_factor, molliere_threshold))

    sigma = np.zeros(np.shape(bayes_factor))

    if np.size(benneke_part) > 0:
        a = -1 / (np.exp(1) * bayes_factor[benneke_part])
        rho = np.real(np.exp(lambertw(a, -1)))  # the solution to x * log(x) = a is the lambert W function of a
        sigma[benneke_part] = np.sqrt(2) * erfinv(1 - rho)

    if np.size(molliere_part) > 0:
        a = -1 / (np.exp(1) * molliere_threshold)
        rho = np.real(np.exp(lambertw(a, -1)))  # the solution to x * log(x) = a is the lambert W function of a
        offset = np.sqrt(2) * erfinv(1 - rho) - np.sqrt(2 * np.log(molliere_threshold))

        sigma[molliere_part] = np.sqrt(2 * np.log(bayes_factor[molliere_part])) + offset

    if is_scalar:
        return sigma[0]
    else:
        return sigma


def box_car_conv(array: npt.NDArray, points: npt.NDArray) -> npt.NDArray:
    res = np.zeros_like(array)
    len_arr = len(array)

    for i in range(len(array)):
        if (i - points / 2 >= 0) and (i + points / 2 <= len_arr + 1):
            smooth_val = array[i - points / 2:i + points / 2]
            res[i] = np.sum(smooth_val) / len(smooth_val)
        elif i + points / 2 > len_arr + 1:
            len_use = len_arr + 1 - i
            smooth_val = array[i - len_use:i + len_use]
            res[i] = np.sum(smooth_val) / len(smooth_val)
        elif i - points / 2 < 0:
            smooth_val = array[:max(2 * i, 1)]
            res[i] = np.sum(smooth_val) / len(smooth_val)

    return res


def calculate_chi2(
    data: float | npt.NDArray,
    model: float | npt.NDArray,
    uncertainties: float | npt.NDArray
) -> float:
    return float(np.sum(((data - model) / uncertainties) ** 2))


def calculate_reduced_chi2(
    data: float | npt.NDArray,
    model: float | npt.NDArray,
    uncertainties: float | npt.NDArray,
    degrees_of_freedom: int = 0
) -> float | npt.NDArray:
    return calculate_chi2(data, model, uncertainties) / (np.size(data) - degrees_of_freedom)


def calculate_uncertainty(
    derivatives: npt.NDArray,
    uncertainties: npt.NDArray,
    covariance_matrix: npt.NDArray = None
) -> npt.NDArray | None:
    """
    Calculate the uncertainty of a function f(x, y, ...) with uncertainties on x, y, ... and Pearson's correlation
    coefficients between x, y, ...
    The function must be (approximately) linear with its variables within the uncertainties of said variables.
    For independent variables, set the covariance matrix to identity.
    Uncertainties can be asymmetric, in that case for N variables, use a (N, 2) array for the uncertainties.
    Asymmetric uncertainties are handled **the wrong way** (see source 2), but it is better than nothing.

    Sources:
        1. https://en.wikipedia.org/wiki/Propagation_of_uncertainty
        2. https://phas.ubc.ca/~oser/p509/Lec_10.pdf
        3. http://math.jacobs-university.de/oliver/teaching/jacobs/fall2015/esm106/handouts/error-propagation.pdf
    Args:
        derivatives:
            Partial derivatives of the function with respect to each variable (df/dx, df/dy, ...)
        uncertainties:
            Uncertainties of each variable (either a 1D-array or a 2D-array containing - and + unc.)
        covariance_matrix:
            Covariance matrix between the variables, by default set to the identity matrix

    Returns:
        A size-2 array containing the - and + uncertainties of the function
    """
    if covariance_matrix is None:
        covariance_matrix = np.identity(np.size(derivatives))

    if np.ndim(uncertainties) == 1:
        sigmas = derivatives * uncertainties

        return np.sqrt(np.matmul(sigmas, np.matmul(covariance_matrix, np.transpose(sigmas))))
    elif np.ndim(uncertainties) == 2:
        sigma_less = derivatives * uncertainties[:, 0]
        sigma_more = derivatives * uncertainties[:, 1]

        return np.sqrt(np.array([  # beware, this is not strictly correct
            np.matmul(sigma_less, np.matmul(covariance_matrix, np.transpose(sigma_less))),
            np.matmul(sigma_more, np.matmul(covariance_matrix, np.transpose(sigma_more)))
        ]))

    return None


def compute_resolving_power(array: npt.NDArray[np.floating]) -> float:
    """Compute the mean resolving power of an array.

    Args:
        array:
            A 1-D array.

    Returns:
        The mean resolving power of the array.
    """
    return np.mean(
        array[:-1] / np.diff(array) + 0.5,
        dtype=float
    )


def convolve(input_wavelength, input_flux, instrument_res):
    r"""
    This function convolves a model spectrum to the instrumental wavelength
    using the provided data_resolution
    Args:
        input_wavelength : numpy.ndarray
            The wavelength grid of the model spectrum
        input_flux : numpy.ndarray
            The flux as computed by the model
        instrument_res : float
            :math:`\\lambda/\\Delta \\lambda`, the width of the gaussian kernel to convolve with the model spectrum.

    Returns:
        flux_lsf
            The convolved spectrum.
    """
    if isinstance(instrument_res, np.ndarray):
        return fconvolve.variable_width_convolution(input_wavelength, input_flux, instrument_res)

    # From talking to Ignas: delta lambda of resolution element
    # is FWHM of the LSF's standard deviation, hence:
    sigma_lsf: float = 1 / instrument_res / (2 * np.sqrt(2 * np.log(2)))

    # The input spacing of petitRADTRANS is 1e3, but just compute
    # it to be sure, or more versatile in the future.
    # Also, we have a log-spaced grid, so the spacing is constant
    # as a function of wavelength
    spacing: float = np.mean(2 * np.diff(input_wavelength) / (input_wavelength[1:] + input_wavelength[:-1]))

    # Calculate the sigma to be used in the gauss filter in units
    # of input wavelength bins
    sigma_lsf_gauss_filter = sigma_lsf / spacing

    flux_lsf = gaussian_filter(
        input_flux,
        sigma=sigma_lsf_gauss_filter,
        mode='nearest'
    )

    return flux_lsf


def convolve_and_sample_variable_resolution_breads(
        wavelengths,
        resolutions,
        model_wavelengths,
        model_fluxes,
        num_sigma=3
):
    """
    Simulate the observations of a model.
    Convolves the model with a variable Gaussian LSF, sampled at each desired spectral channel.

    From Jerry Xuan circa 2024, https://github.com/jruffio/breads

    Args:
        wavelengths: np.ndarray
            The wavelengths desired (length of N_output)
        resolutions: np.ndarray
            The spectral resolving power at each wavelength (length of N_output)
        model_wavelengths: np.ndarray
            The wavelengths of the model (length of N_model)
        model_fluxes: np.ndarray
            The fluxes of the model (length of N_model)
        num_sigma: float
            Number of +/- sigmas to evaluate the LSF to.

    Returns:
        output_model: np.ndarray
            the fluxes in each of the wavelength channels (length of N_output)
    """

    # JX added to use function with input R, instead of LSF FWHM.
    # first get FWHM of LSF from lambda / R. Then convert FWHM to stddev of Gaussian
    sigmas_wvs = wavelengths / (resolutions * 2 * np.sqrt(2 * np.log(2)))

    model_in_range = np.where((model_wavelengths >= np.min(wavelengths)) & (model_wavelengths < np.max(wavelengths)))
    dwv_model = np.abs(model_wavelengths[model_in_range] - np.roll(model_wavelengths[model_in_range], 1))
    dwv_model[0] = dwv_model[1]

    filter_size = int(np.ceil(np.max((2 * num_sigma * sigmas_wvs)/np.min(dwv_model))))
    filter_coords = np.linspace(-num_sigma, num_sigma, filter_size)

    # Commenting out try-except block from breads, doesn't seem to do anything.
    # try:
    filter_coords = np.tile(filter_coords, [wavelengths.shape[0], 1])  # shape of (N_output, filter_size)
    # except Exception as e:
    # print(e)
    # print('reached exception for ' + str(filter_size))

    filter_wv_coords = filter_coords * sigmas_wvs[:, None] + wavelengths[:, None]  # model wavelengths we want
    lsf = np.exp(-filter_coords**2/2)/np.sqrt(2*np.pi)

    model_interp = interp1d(model_wavelengths, model_fluxes, kind='cubic', bounds_error=False)
    filter_model = model_interp(filter_wv_coords)

    output_model = np.nansum(filter_model * lsf, axis=1)/np.sum(lsf, axis=1)

    return output_model


def feature_scaling(array: npt.NDArray, min_value: float = 0.0, max_value: float = 1.0) -> npt.NDArray:
    """Bring all values of array between a min and max value.

    Args:
        array: array to normalize
        min_value: target minimum value
        max_value: target maximum value

    Returns:
        The normalized array with values between min_value and max_value.
    """
    return min_value + ((array - np.min(array)) * (max_value - min_value)) / (np.max(array) - np.min(array))


def filter_spectrum_with_spline(wavelengths, fluxes, uncertainties=None, x_nodes=None, m_spline=None):
    """
    Use spline interpolation produce a high-pass filtered version of the input spectrum.

    From BREADS, BSD 3-Clause License
    Copyright (c) 2024, jruffio
    https://github.com/jruffio/breads

    Args:
        wavelengths: np.ndarray
            Wavelengths of the input spectrum
        fluxes: np.ndarray
            Fluxes of the input spectrum
        uncertainties: np.ndarray
            Uncertainties of the input spectrum
        x_nodes: np.ndarray
            Nodes for the spline interpolation as np.ndarray in the same units as wavelengths.
            x_nodes can also be a list of ndarrays/list to model discontinous functions.
        m_spline: np.ndarray
            Pre-computed spline matrix. If None, it will be computed from x_nodes and wavelengths.
    Returns:
        high_pass_filtered_fluxes: np.ndarray
            High-pass filtered fluxes of the input spectrum
    """
    if uncertainties is None:
        uncertainties = np.ones(fluxes.shape)

    if m_spline is None:
        m_spline = linear_spline_interpolation(x_nodes, wavelengths, spline_degree=3)

    normalised_spline_matrix = m_spline/uncertainties[:, None]
    weighted_fluxes = fluxes/uncertainties
    where_finite = np.where(np.isfinite(weighted_fluxes))
    normalised_spline_matrix = normalised_spline_matrix[where_finite[0], :]
    weighted_fluxes = weighted_fluxes[where_finite]

    paras = lsq_linear(normalised_spline_matrix, weighted_fluxes).x
    spline_model = np.dot(normalised_spline_matrix, paras)
    residuals = weighted_fluxes - spline_model

    # LPF_fluxes = np.zeros(fluxes.shape)+np.nan
    high_pass_filtered_fluxes = np.zeros(fluxes.shape)+np.nan
    # LPF_fluxes[where_finite] = m*uncertainties[where_finite]
    high_pass_filtered_fluxes[where_finite] = residuals*uncertainties[where_finite]

    return high_pass_filtered_fluxes  # ,LPF_spec


def gaussian_weights1d(sigma: float, truncate: float = 4.0) -> npt.NDArray:
    """Compute a 1D Gaussian convolution kernel.
    To be used with scipy.ndimage.convolve1d.

    Based on scipy.ndimage gaussian_filter1d and _gaussian_kernel1d.

    Args:
        sigma:
            Standard deviation for Gaussian kernel.
        truncate:
            Truncate the filter at this many standard deviations.

    Returns:

    """
    # Make the radius of the filter equal to truncate standard deviations
    radius = int(truncate * sigma + 0.5)

    x = np.arange(-radius, radius + 1)
    phi_x = np.exp(-0.5 / sigma ** 2 * x ** 2)

    return phi_x / phi_x.sum()


def gaussian_weights_running(sigmas: npt.NDArray, truncate: float = 4.0) -> npt.NDArray:
    """Compute 1D Gaussian convolution kernels for an array of standard deviations.

    Based on scipy.ndimage gaussian_filter1d and _gaussian_kernel1d.

    Args:
        sigmas:
            Standard deviations for Gaussian kernel.
        truncate:
            Truncate the filter at this many standard deviations.

    Returns:

    """
    # Make the radius of the filter equal to truncate standard deviations
    radius = int(truncate * np.max(sigmas) + 0.5)

    x = np.arange(-radius, radius + 1)
    sd = np.tile(sigmas, (x.size, 1)).T

    phi_x = np.exp(-0.5 / sd ** 2 * x ** 2)

    return np.transpose(phi_x.T / phi_x.sum(axis=1))


def linear_spline_interpolation(
    x_knots: npt.NDArray[np.floating],
    x_samples: npt.NDArray[np.floating],
    spline_degree: Literal[1, 2, 3, 4, 5] = 3
):
    """Compute a spline based linear model.

    If Y = [y1, y2, ...] are the values of the function at the location of the node [x1, x2, ...].
    np.dot(M,Y) is the interpolated spline corresponding to the sampling of the x-axis (x_samples)

    From BREADS, BSD 3-Clause License
    Copyright (c) 2024, jruffio
    https://github.com/jruffio/breads

    Args:
        x_knots: np.ndarray
            List of nodes for the spline interpolation as np.ndarray in the same units as x_samples.
            x_knots can also be a list of ndarrays/list to model discontinous functions.
        x_samples: np.ndarray
            Vector of x values. ie, the sampling of the data.
        spline_degree: int
            Degree of the spline interpolation (default: 3).
            if np.size(x_knots) <= spline_degree, then spline_degree = np.size(x_knots)-1

    Returns:
        M: Matrix of shape (D, N) with D the size of x_samples and N the total number of nodes.
    """
    if type(x_knots[0]) is list or type(x_knots[0]) is np.ndarray:
        x_knots_list = x_knots
    else:
        x_knots_list = [x_knots]

    if np.size(x_knots_list) <= 1:
        return np.ones((np.size(x_samples), 1))

    if np.size(x_knots_list) <= spline_degree:
        spline_degree: int = np.size(x_knots) - 1

    if spline_degree > 5:
        raise ValueError(
            f"spline degree cannot be greater than 5, but is ({spline_degree})"
        )

    spline_degree: Literal[1, 2, 3, 4, 5]
    spline_matrix: list[npt.NDArray[np.floating]] = []

    for nodes in x_knots_list:
        _spline_matrix = np.zeros((np.size(x_samples), np.size(nodes)))

        in_bounds = np.where((np.min(nodes) < x_samples) & (x_samples < np.max(nodes)))
        _x = x_samples[in_bounds]

        for chunk in range(np.size(nodes)):
            # TODO @Evert add comment on what is going on with y
            y = np.zeros(np.size(nodes))
            y[chunk] = 1

            spline = InterpolatedUnivariateSpline(
                x=nodes,
                y=y,
                k=spline_degree,
                ext=0
            )
            _spline_matrix[in_bounds[0], chunk] = spline(_x)

        spline_matrix.append(_spline_matrix)

    return np.concatenate(spline_matrix, axis=1)


def longitude2phase(longitude: float):
    return longitude / (2 * np.pi)


def mean_uncertainty(uncertainties: npt.NDArray) -> float:
    """Calculate the uncertainty of the mean of an array.

    Args:
        uncertainties: individual uncertainties of the averaged array

    Returns:
        The uncertainty of the mean of the array
    """
    return np.sqrt(np.sum(uncertainties ** 2)) / np.size(uncertainties)


def median_uncertainties(uncertainties: npt.NDArray) -> float:
    """Calculate the uncertainty of the median of an array.

    Demonstration:
        uncertainty ~ standard deviation = sqrt(variance) = sqrt(V)
        V_mean / V_median = 2 * (N - 1) / (pi * N); (see source)
        => V_median = V_mean * pi * N / (2 * (N - 1))
        => uncertainty_median = uncertainty_mean * sqrt(pi * N / (2 * (N - 1)))

    Source:
        https://mathworld.wolfram.com/StatisticalMedian.html

    Args:
        uncertainties: individual uncertainties of the median of the array

    Returns:
        The uncertainty of the median of the array
    """
    return mean_uncertainty(uncertainties) \
        * np.sqrt(np.pi * np.size(uncertainties) / (2 * (np.size(uncertainties) - 1)))


def normalize(array: npt.NDArray, axis: int = None) -> npt.NDArray:
    return (array - np.min(array, axis=axis)) / (np.max(array, axis=axis) - np.min(array, axis=axis))


def phase2longitude(phase: float, rad2deg: bool = False):
    longitude = phase * 2 * np.pi

    if rad2deg:
        longitude = np.rad2deg(longitude)

    return longitude


def prt_resolving_space(start, stop, resolving_power) -> npt.NDArray:
    """Return numbers evenly spaced at the specified resolving power.

    Args:
        start:
            The starting value of the sequence.
        stop:
            The end value of the sequence.
        resolving_power:
            Resolving power of the sample

    Returns:
        Samples spaced following the specified resolving power.

    """
    # Check for inputs validity
    if start > stop:
        raise ValueError(f"start ({start}) must be lower than stop {stop}")

    if resolving_power <= 0:
        raise ValueError(f"resolving power ({resolving_power}) must be strictly positive")

    inverse_resolving_power = 1 / resolving_power

    # Get maximum space length (much higher than required)
    size_max = int(np.ceil((stop - start) / (start / resolving_power)))

    # Start generating space
    samples = [start]
    i = 0

    for i in range(size_max):
        samples.append(samples[-1] * np.exp(inverse_resolving_power))

        if samples[-1] >= stop:
            break

    if i == size_max - 1 and samples[-1] < stop:
        raise ValueError(f"maximum size ({size_max}) reached before reaching stop ({samples[-1]} < {stop})")

    return np.array(samples)


def resolving_space(start, stop, resolving_power):
    # Check for inputs validity
    if start > stop:
        raise ValueError(f"start ({start}) must be lower than stop {stop}")

    if resolving_power <= 0:
        raise ValueError(f"resolving power ({resolving_power}) must be strictly positive")

    # Get maximum space length (much higher than required)
    size_max = int(np.ceil((stop - start) / (start / resolving_power)))

    if not np.isfinite(size_max) or size_max < 0:
        raise ValueError(f"invalid maximum size ({size_max})")

    # Start generating space
    samples = [start]
    i = 0

    for i in range(size_max):
        if samples[-1] >= stop:
            break

        samples.append(samples[-1] + samples[-1] / resolving_power)

    if i == size_max - 1 and samples[-1] < stop:
        raise ValueError(
            f"maximum size ({size_max}) reached before reaching stop ({samples[-1]} < {stop})"
            )
    elif samples[-1] > stop:
        del samples[-1]  # ensure that the space is within the [start, stop] interval

    return np.array(samples)


def running_mean(x: npt.NDArray, n: int) -> npt.NDArray:
    cum_sum = np.cumsum(np.insert(x, 0, 0))

    return (cum_sum[n:] - cum_sum[:-n]) / float(n)


def sigma2bayes_factor(sigma: float) -> float:
    """
    Convert a sigma significance into a Bayes factor, or "evidence".
    Note: sometimes algorithms return the "log-evidence", or ln(z). The Bayes factor is z.
    Source: Benneke et al. 2013 https://iopscience.iop.org/article/10.1088/0004-637X/778/2/153
    :param sigma: sigma significance
    :return: Bayes factor (aka "evidence")
    """
    rho = 1 - erf(sigma / np.sqrt(2))
    return - 1 / (np.exp(1) * rho * np.log(rho))
