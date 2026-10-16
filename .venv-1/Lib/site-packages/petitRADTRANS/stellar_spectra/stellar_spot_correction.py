"""Stellar spot correction.
"""
import numpy as np
import numpy.typing as npt
# [IDE] PyCharm does not recognise Fortran modules
# noinspection PyUnresolvedReferences
from petitRADTRANS.fortran_rebin import fortran_rebin as frebin


def rackham_stellar_spot_correction(
        wavelength_bin_edges: npt.NDArray[np.floating],
        star_effective_temperature: float,
        star_spot_effective_temperature: float,
        spot_coverage: float,
        stellar_model: npt.NDArray[np.floating] = None,
        spot_model: npt.NDArray[np.floating] = None
) -> npt.NDArray[np.floating]:
    """Compute the stellar spot correction based on stellar models with different temperatures.

    Reference:
    - Rackham et al. (2018). The Transit Light Source Effect: False Spectral Features and Incorrect Densities for
      M-dwarf Transiting Planets.
      The Astrophysical Journal, Volume 853, Issue 2, article id. 122, 18 pp.
      DOI: https://doi.org/10.3847/1538-4357/aaa08c.

    Args:
        wavelength_bin_edges: numpy.array
            (cm) The edges of the wavelength bins for which the correction factor should be computed.
        star_effective_temperature: float
            (K) The temperature of the unspotted stellar photosphere.
        star_spot_effective_temperature: float
            (K) The temperature of the spots.
        spot_coverage: float
            The fraction of the stellar surface covered by spots (between 0 and 1).
        stellar_model: numpy.ndarray, optional
            (cm, erg/cm^2/s/Hz) A 2D array of shape (n_wavelengths, 2), where the first column contains the wavelengths
            in cm and the second column contains the corresponding stellar fluxes. If None, a Phoenix model will be
            used.
        spot_model: numpy.ndarray, optional
            (cm, erg/cm^2/s/Hz) A 2D array of shape (n_wavelengths, 2), where the first column contains the wavelengths
            in cm and the second column contains the corresponding spot fluxes. If None, a Phoenix model will be used.

    Returns:
        correction_factor: numpy.ndarray
            The correction factor to be applied to the transit depth in the dimension of the wavelength.
    """

    if stellar_model is not None:
        if spot_model is None:
            raise ValueError("if an individual stellar model should be used, a spot model must also be provided.")

    wavelength_bin_edges = wavelength_bin_edges * 1e4  # cm to um
    wavelength_bin_widths = np.diff(wavelength_bin_edges)
    wavelengths = (wavelength_bin_edges[1:] + wavelength_bin_edges[:-1]) * 0.5

    # Compute the stellar spectra
    if stellar_model is None:
        from petitRADTRANS.stellar_spectra.phoenix import phoenix_star_table
        star = phoenix_star_table
        star_data, _ = star.compute_spectrum(star_effective_temperature)
    else:
        star_data = stellar_model

    wavelengths_star = star_data[:, 0] * 1e4  # cm to um
    star_flux = star_data[:, 1]

    # Rebin the stellar spectrum to the prt_object resolution
    bin_flux_star = frebin.rebin_spectrum_bin(
        input_wavelengths=wavelengths_star,
        input_spectrum=star_flux,
        rebinned_wavelengths=wavelengths,
        bin_widths=wavelength_bin_widths
    )

    # Compute the spot spectrum
    if spot_model is None:
        from petitRADTRANS.stellar_spectra.phoenix import phoenix_star_table
        star = phoenix_star_table
        spot_data, _ = star.compute_spectrum(star_spot_effective_temperature)
    else:
        spot_data = spot_model

    wavelengths_spots = spot_data[:, 0] * 1e4  # cm to um
    star_spot_flux = spot_data[:, 1]

    # Rebin the spot spectrum to the prt_object resolution
    bin_flux_spot = frebin.rebin_spectrum_bin(
        input_wavelengths=wavelengths_spots,
        input_spectrum=star_spot_flux,
        rebinned_wavelengths=wavelengths,
        bin_widths=wavelength_bin_widths
    )

    # Compute the correction factor
    correction_factor = 1 - spot_coverage * (1 - bin_flux_spot / bin_flux_star)

    return correction_factor
