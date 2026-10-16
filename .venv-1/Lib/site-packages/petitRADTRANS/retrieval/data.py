import copy
import os
import warnings
from typing import Callable

import numpy as np
import numpy.typing as npt
from astropy.io import fits

import petitRADTRANS.physical_constants as cst
# [IDE] PyCharm does not recognise Fortran modules
# noinspection PyUnresolvedReferences
from petitRADTRANS.fortran_rebin import fortran_rebin as frebin
from petitRADTRANS.math import convolve, convolve_and_sample_variable_resolution_breads, filter_spectrum_with_spline


class Data:
    r"""
    This class stores the spectral data to be retrieved from a single instrument or observation.

    Each dataset is associated with an instance of petitRadTrans and an atmospheric model.
    The pRT instance can be overwritten, and associated with an existing pRT instance with the
    external_pRT_reference parameter.
    This setup allows for joint or independent retrievals on multiple datasets.
    # TODO @Evert complete docstring
    Args:
        name : str
            Identifier for this data set.
        path_to_observations : str
            Path to observations file, including filename. This can be a txt or dat file
            containing the wavelength, flux, transit depth and error, or a fits file
            containing the wavelength, spectrum and covariance matrix.
            Alternatively, the data information can be directly given by the wavelengths, spectrum, uncertainties, and
            mask attributes.
        data_resolution : float or np.ndarray
            Spectral resolution of the instrument. Optional, allows convolution of model to
            instrumental line width. If the data_resolution is an array, the resolution can
            vary as as a function of wavelength. The array should have the same shape as
            the input wavelength array, and should specify the spectral resolution at each
            wavelength bin.
        model_resolution : float
            Will be ``None`` by default.  The resolution of the c-k opacity tables in pRT.
            This will generate a new c-k table using exo-k. The default (and maximum)
            correlated k resolution in pRT is :math:`\\lambda/\\Delta \\lambda > 1000` (R=500).
            Lowering the resolution will speed up the computation.
            If integer positive value, and if ``opacities == 'lbl'`` is ``True``, then this
            will sample the high-resolution opacities at the specified resolution.
            This may be desired in the case where medium-resolution spectra are
            required with a :math:`\\lambda/\\Delta \\lambda > 1000`, but much smaller than
            :math:`10^6`, which is the resolution of the ``lbl`` mode. In this case it
            may make sense to carry out the calculations with line_by_line_opacity_sampling = 10e5,
            for example, and then re-binning to the final desired resolution:
            this may save time! The user should verify whether this leads to
            solutions which are identical to the re-binned results of the fiducial
            :math:`10^6` resolution. If not, this parameter must not be used.
            Note the difference between this parameter and the line_by_line_opacity_sampling
            parameter in the RadTrans class - the actual desired resolution should
            be set here.
        system_distance : float
            The distance to the object in CGS units. Defaults to a 10pc normalized distance.
        external_radtrans_reference : object
            An existing RadTrans object. Leave as none unless you're sure of what you're doing.
        model_generating_function : method
            A function, typically defined in run_definition.py that returns the model wavelength and spectrum
            (emission or transmission).
            This is the function that contains the physics of the model, and calls pRT in order to compute the
            spectrum.
        wavelength_boundaries : tuple,list
            Set the wavelength range of the pRT object. Defaults to a range +/-5% greater than that of the data.
            Must at least be equal to the range of the data.
        scale : bool
            Turn on or off scaling the data by a constant factor. Set to True if scaling the data during the
            retrieval.
        scale_err:
            # TODO @Evert complete docstring
        offset_bool:
            # TODO @Evert complete docstring
        wavelength_bin_widths : numpy.ndarray
            Set the wavelength bin width to bin the Radtrans object to the data. Defaults to the data bins.
        photometry : bool
            Set to True if using photometric data.
        photometric_transformation_function : method
            Transform the photometry (account for filter transmission etc.).
            This function must take in the wavelength and flux of a spectrum,
            and output a single photometric point (and optionally flux error).
        photometric_bin_edges : Tuple, numpy.ndarray
            The edges of the photometric bin in micron. [low,high]
        line_opacity_mode : str
            Should the retrieval be run using correlated-k opacities (default, 'c-k'),
            or line by line ('lbl') opacities? If 'lbl' is selected, it is HIGHLY
            recommended to set the model_resolution parameter. In general,
            'c-k' mode is recommended for retrievals of everything other than
            high-resolution (R>40000) spectra.
        radtrans_grid: bool
            Set to true if data has been binned to a pRT c-k grid.
        concatenate_flux_epochs_variability: bool
            Set to true if data concatenation treatment for variability is to be used.
        atmospheric_column_flux_mixer: method
            Function that mixes model fluxes of atmospheric columns in variability retrievals.
        variability_atmospheric_column_model_flux_return_mode: bool
            Set to true if the forward model should returns the fluxes of the individual atmospheric
            columns. This is useful if external_radtrans_reference is True, but the master (reference) object
            should return the column fluxes for mixing, not the combined column flux. In this case a column
            mixing function needs to be handed to the data constructor.
        radtrans_object:
            An instance of Radtrans object to be used to generate model spectra in retrievals.
        wavelengths:
            (um) Wavelengths of the data.
        spectrum:
            Spectrum of the data.
        uncertainties:
            Uncertainties of the data, in the same units as the spectrum.
        covariance:
            Covariance matrix of the data, in the same units as the spectrum squared.
        mask:
            Mask of the data.
    """
    resolving_power_str = ".R"

    def __init__(
        self,
        name: str,
        path_to_observations: str | None = None,
        data_resolution: float | None = None,
        model_resolution: float | None = None,
        system_distance: float | None = None,
        external_radtrans_reference: object | None = None,
        model_generating_function: Callable[..., tuple[npt.NDArray, npt.NDArray, npt.NDArray]] | None = None,
        wavelength_boundaries: tuple[float, float] | None = None,
        scale: bool = False,
        scale_err: bool = False,
        offset_bool: bool = False,
        resample: bool = False,
        subtract_continuum: bool = False,
        wavelength_bin_widths: npt.NDArray[np.floating] | None = None,
        photometry: bool = False,
        photometric_transformation_function: Callable[[npt.NDArray, npt.NDArray], tuple[float, float]] | None = None,
        photometric_bin_edges: tuple[float, float] | None = None,
        line_opacity_mode: str = 'c-k',
        radtrans_grid: bool = False,
        concatenate_flux_epochs_variability: bool = False,
        atmospheric_column_flux_mixer: Callable | None = None,
        variability_atmospheric_column_model_flux_return_mode: bool = False,
        radtrans_object: object = None,
        wavelengths: npt.NDArray[np.floating] | None = None,
        spectrum: npt.NDArray[np.floating] | None = None,
        uncertainties: npt.NDArray[np.floating] | None = None,
        covariance: npt.NDArray[np.floating] | None = None,
        mask: npt.NDArray[np.bool_] | None = None
    ):
        self.name: str = name
        self.path_to_observations: str | None = path_to_observations

        # To be filled later
        self.radtrans_object: object = radtrans_object
        self.wavelengths: npt.NDArray[np.floating] | None = wavelengths  #: The wavelength bin centers
        self.spectrum: npt.NDArray[np.floating] | None = spectrum  #: The flux or transit depth
        self.uncertainties: npt.NDArray[np.floating] | None = uncertainties  #: The error on the flux or transit depth

        # Add a mask with that will be used in retrievals
        self.mask: npt.NDArray[np.bool_]

        if mask is None:
            self.mask = np.zeros(np.shape(self.spectrum), dtype=bool)
        else:
            self.mask = mask

        self.system_distance: float

        if system_distance is None:
            self.system_distance = 10 * cst.pc
        else:
            self.system_distance = system_distance

        # Sanity check system distance
        if self.system_distance < cst.pc:
            warnings.warn(
                f"system distance ({self.system_distance}) is less than 1 pc, make sure that its units are in CGS"
            )

        self.data_resolution: float | None = data_resolution
        self.data_resolution_array_model: npt.NDArray[np.floating] | None = None

        self.model_resolution: float | None = model_resolution
        self.external_radtrans_reference: object = external_radtrans_reference
        self.model_generating_function: Callable = model_generating_function
        self.line_opacity_mode: str = line_opacity_mode

        if line_opacity_mode not in ['c-k', 'lbl']:
            raise ValueError(f"line_opacity_mode must be either 'c-k' or 'lbl', but is '{line_opacity_mode}'")

        # Sanity check model function
        if model_generating_function is None and external_radtrans_reference is None:
            raise ValueError(
                "any of parameters 'model_generating_function' or 'external_radtrans_reference' must be set, "
                "but both were None"
            )

        if model_resolution is not None:
            if line_opacity_mode == 'c_k' and model_resolution > 1000:
                warnings.warn(
                    f"opacity resolving power ({model_resolution}) is above the maximum possible in c-k mode (1000), "
                    f"resetting the resolving power to its default value..."
                )
                self.model_resolution = None
            if line_opacity_mode == 'lbl' and model_resolution < 1000:
                warnings.warn(
                    f"opacity resolving power ({model_resolution}) is below the maximum possible in c-k mode (1000), "
                    f"but current mode is 'lbl', which is inefficient\n"
                    f"To remove this warning, switch the line opacity mode to 'c-k'."
                )

        # Optional, covariance and scaling
        self.covariance: npt.NDArray[np.floating] | None = covariance
        self.inv_cov: npt.NDArray[np.floating] | None = None
        self.log_covariance_determinant: float | None = None

        if covariance is not None:
            self.inv_cov = np.linalg.inv(covariance)
            sign, self.log_covariance_determinant = np.linalg.slogdet(2.0 * np.pi * covariance)

        if covariance is not None and uncertainties is None:
            self.uncertainties = np.sqrt(np.diagonal(covariance))

        self.scale: bool = scale
        self.scale_err: bool = scale_err
        self.offset_bool: bool = offset_bool
        self.resample: bool = resample
        self.subtract_continuum: bool = subtract_continuum
        self.scale_factor: float = 1.0
        self.offset: float = 0.0
        self.bval: float = -np.inf

        # Bins and photometry
        self.wavelength_boundaries: tuple[float, float] | None = None
        self.wavelength_bin_widths: npt.NDArray[np.floating] | float | None = wavelength_bin_widths
        self.photometry = photometry
        self.photometric_transformation_function = \
            photometric_transformation_function

        if photometry:
            missing = []

            if photometric_transformation_function is None:
                missing.append("'photometric_transformation_function'")

            if photometric_bin_edges is None:
                missing.append("'photometric_bin_edges'")

            if len(missing) > 0:
                ', '.join(missing)
                raise ValueError(f"missing photometric arguments for photometric data '{name}': {missing}")

        self.photometry_range = wavelength_boundaries
        self.photometric_bin_edges: tuple[float, float] | None = photometric_bin_edges

        self.radtrans_grid = radtrans_grid
        self.concatenate_flux_epochs_variability = concatenate_flux_epochs_variability
        self.variability_atmospheric_column_model_flux_return_mode = (
            variability_atmospheric_column_model_flux_return_mode)
        self.atmospheric_column_flux_mixer = atmospheric_column_flux_mixer

        # Read in data
        if path_to_observations is not None:
            # Check if data exists
            if not os.path.exists(path_to_observations):
                raise FileNotFoundError(f"data file '{path_to_observations}' does not exist")

            if not photometry:
                if path_to_observations.endswith("_x1d.fits"):
                    self.load_jwst(path_to_observations)
                elif path_to_observations.endswith('.fits'):
                    self.loadfits(path_to_observations)
                else:
                    self.loadtxt(path_to_observations)

                if wavelength_boundaries is not None:
                    self.wavelength_boundaries = wavelength_boundaries
                else:
                    self.wavelength_boundaries = (
                        0.95 * self.wavelengths[0],
                        1.05 * self.wavelengths[-1]
                    )

                if self.wavelength_bin_widths is None:
                    if wavelength_bin_widths is not None:
                        self.wavelength_bin_widths = wavelength_bin_widths
                    else:
                        self.wavelength_bin_widths = np.zeros_like(self.wavelengths)
                        self.wavelength_bin_widths[:-1] = np.diff(self.wavelengths)
                        self.wavelength_bin_widths[-1] = self.wavelength_bin_widths[-2]
            else:
                if self.photometric_bin_edges is None:
                    raise ValueError(
                        f"'photometric_bin_edges' must be a tuple of two floats in photometric data ('{self.name}'), "
                        f"but is None"
                    )

                if wavelength_boundaries is not None:
                    self.wavelength_boundaries = wavelength_boundaries
                else:
                    self.wavelength_boundaries = (
                        0.95 * self.photometric_bin_edges[0],
                        1.05 * self.photometric_bin_edges[1]
                    )

                # For binning later
                # TODO @Evert why using a float instead of an array of size 1? The attribute would be more predictable
                self.wavelength_bin_widths = self.photometric_bin_edges[1] - self.photometric_bin_edges[0]

                if self.data_resolution is None:
                    self.data_resolution = np.array(self.photometric_bin_edges).mean() / self.wavelength_bin_widths

    # TODO [4.0.0] This should be a classmethod instantiating a Data object, also fix function name => from_txt()
    def loadtxt(self, path: str, delimiter: str = ',', comments: str = '#') -> None:
        """
        Read a TXT or DAT file containing a header above 3 or 4 data columns representing a spectrum.
        Headers should start with the symbol given in the 'comment' argument.
        The 4 possible data columns are, from left to right:
            1. The wavelengths in microns,
            2. [Optional] The wavelengths bin widths, in microns.
            3. The flux or transit depths,
            4. The error on each data point.

        Alternatively, the wavelength column
        Checks will be performed to determine the correct delimiter, but the recommended format is to use a CSV file
        with columns for wavelength, flux and error.

        Args:
            path : str
                Directory and filename of the data.
            delimiter : string, int
                The string used to separate values. By default, commas act as delimiter.
                An integer or sequence of integers can also be provided as width(s) of each field.
            comments : string
                The character used to indicate the start of a comment.
                All the characters occurring on a line after a comment are discarded
        """

        if self.photometry:
            return

        # Data shape: (n_samples, n_columns)
        data: npt.NDArray[np.floating] = np.genfromtxt(path, delimiter=delimiter, comments=comments)

        # Input sanity checks
        # TODO [4.0.0] this is inefficient (the file is loaded up to 4 times!), change behaviour by putting more respon-
        #  -sibility on the user and loading the file only once
        if np.isnan(data).any():  # try to load the file with the default settings
            data = np.genfromtxt(path)

        if len(data.shape) < 2:  # a columns in missing, try loading with no delimiter
            data = np.genfromtxt(path, comments=comments)

        if data.shape[1] == 4:
            self.wavelengths = data[:, 0]
            self.wavelength_bin_widths = data[:, 1]
            self.spectrum = data[:, 2]
            self.uncertainties = data[:, 3]
            return
        elif data.shape[1] != 3:
            data = np.genfromtxt(path)

        # Warnings and errors
        if data.shape[1] < 3:
            raise ValueError(
                f"data file '{path}' must contain at least 3 columns (wavelength, flux, flux error), "
                f"but has {data.shape[1]} column(s)"
            )
        elif data.shape[1] > 4:
            warnings.warn(
                f"data file '{path}' should contain at most 4 columns "
                f"(wavelength, wavelength bins (optional), flux, flux error), "
                f"but has {data.shape[1]} column(s)\n"
                f"Additional columns will be ignored"
            )

        if np.isnan(data).any():
            warnings.warn(f"NANs present in data file '{path}'")

        self.wavelengths = data[:, 0]
        self.spectrum = data[:, 1]
        self.uncertainties = data[:, 2]

    # TODO [4.0.0] This should be a classmethod instantiating a Data object, named "from_jwst_data"
    def load_jwst(self, path):
        """
        Load in a x1d fits file as produced by the STSci JWST pipeline.
        Expects units of Jy for the flux and micron for the wavelength.

        Args:
            path : str
                Directory and filename of the data.
        """
        hdul = fits.open(path)
        self.wavelengths = hdul["EXTRACT1D"].data["WAVELENGTH"]
        self.spectrum = hdul["EXTRACT1D"].data["FLUX"]
        self.uncertainties = hdul["EXTRACT1D"].data["FLUX_ERROR"]

        # Convert from Jy to W/m^2/micron
        self.spectrum = 1e-26 * 2.99792458e14 * self.spectrum / self.wavelengths ** 2
        self.uncertainties = 1e-26 * 2.99792458e14 * self.uncertainties / self.wavelengths ** 2

    # TODO [4.0.0] This should be a classmethod instantiating a Data object, named "from_fits"
    def loadfits(self, path) -> None:
        """
        Load a Radtrans-formatted fits file.
        Must include extension SPECTRUM with fields WAVELENGTH, FLUX
        and COVARIANCE (or ERROR).

        Args:
            path : str
                Directory and filename of the data.
        """

        if self.photometry:
            return

        data = np.array(fits.getdata(path, 'SPECTRUM'))

        if not isinstance(data, np.ndarray):  # TODO @Evert, right now the condition is never met
            self.wavelengths = data.field("WAVELENGTH")
            self.spectrum = data.field("FLUX")

            if "COVARIANCE" in data.columns.names:
                self.covariance = data.field("COVARIANCE")
                self.inv_cov = np.linalg.inv(self.covariance)
                sign, self.log_covariance_determinant = np.linalg.slogdet(2.0 * np.pi * self.covariance)
                self.uncertainties = np.sqrt(self.covariance.diagonal())
            elif "ERROR" in data.columns.names:
                # Note that this will only be the uncorrelated error.
                # Dot with the correlation matrix (if available) to get
                # the full error.
                self.uncertainties = data.field("ERROR")
                self.covariance = np.diag(self.uncertainties ** 2)
                self.inv_cov = np.linalg.inv(self.covariance)
        else:
            hdul = fits.open(path)
            self.wavelengths = hdul[1].data['WAVELENGTH'].astype(np.float64)
            self.spectrum = hdul[1].data['FLUX'].astype(np.float64)

            if 'FLUX_STD' in hdul[1].data.columns.names:
                self.uncertainties = hdul[1].data['FLUX_STD'].astype(np.float64)
            elif 'FLUX_ERROR' in hdul[1].data.columns.names:
                self.uncertainties = hdul[1].data['FLUX_ERROR'].astype(np.float64)

            if 'FLUX_COV' in hdul[1].data.columns.names:
                self.covariance = hdul[1].data['FLUX_COV'].astype(np.float64)
            elif 'COVARIANCE' in hdul[1].data.columns.names:
                self.covariance = hdul[1].data['COVARIANCE'].astype(np.float64)
            self.inv_cov = np.linalg.inv(self.covariance)

        if self.uncertainties is None and self.covariance is not None:
            self.uncertainties = np.sqrt(self.covariance.diagonal())

        sign, self.log_covariance_determinant = np.linalg.slogdet(2.0 * np.pi * self.covariance)

    # TODO [4.0.0] remove this function, or rename "system_distance" to "_system_distance" and use a property
    def set_distance(self, distance) -> float:
        """Set the system distance attribute.

        This does not rescale the flux to the new distance.
        In order to rescale the flux and error, use the scale_to_distance method.

        Args:
            distance : float
                The distance to the object in CGS units.
        """

        self.system_distance = distance
        return self.system_distance

    def initialise_data_resolution(self, wavelengths_model: npt.NDArray[np.floating]) -> None:
        # TODO [4.0.0] this function should raise an error when the interpolation is not possible
        #  As it is now, it can do nothing, which can be confusing
        if isinstance(self.data_resolution, np.ndarray):
            self.data_resolution_array_model = np.interp(wavelengths_model, self.wavelengths, self.data_resolution)

    # TODO [4.0.0] rename wlens => wavelengths (also, maybe it can be removed?)
    def update_bins(self, wlens: npt.NDArray[np.floating]) -> None:
        # TODO @Evert add docstrings
        self.wavelength_bin_widths = np.zeros_like(wlens)
        self.wavelength_bin_widths[:-1] = np.diff(wlens)
        self.wavelength_bin_widths[-1] = self.wavelength_bin_widths[-2]

    # TODO [4.0.0] rename new_dist => new_distance
    def scale_to_distance(self, new_dist: float) -> float:
        """Update the distance variable in the data class.

        This will rescale the flux to the new distance.

        Args:
            new_dist : float
                The distance to the object in CGS units.
        """

        scale: float = (self.system_distance / new_dist) ** 2
        self.spectrum *= scale

        if self.covariance is not None:
            self.covariance *= scale ** 2
            self.inv_cov = np.linalg.inv(self.covariance)
            sign, self.log_covariance_determinant = np.linalg.slogdet(2.0 * np.pi * self.covariance)

            self.uncertainties = np.sqrt(self.covariance.diagonal())
        else:
            self.uncertainties *= scale
            self.covariance = np.diag(self.uncertainties)
            self.inv_cov = np.linalg.inv(self.covariance)
            sign, self.log_covariance_determinant = np.linalg.slogdet(2.0 * np.pi * self.covariance)

        self.system_distance = new_dist

        return scale

    # TODO [4.0.0] should be removed as it is redundant with log_likelihood, other functionalities should be split a-
    #  -mong several function
    def get_chisq(
        self,
        wlen_model,
        spectrum_model,
        plotting,
        parameters=None,
        per_datapoint=False,
        atmospheric_model_column_fluxes=None,
        generate_mock_data=False
    ) -> float:
        """
        Calculate the chi square between the model and the data.

        Args:
            wlen_model : numpy.ndarray
                The wavelengths of the model
            spectrum_model : numpy.ndarray
                The model flux in the same units as the data.
            plotting : bool
                Show test plots.
            parameters :
                # TODO @Evert complete docstring
            per_datapoint : bool
                # TODO @Evert complete docstring
            atmospheric_model_column_fluxes : numpy.ndarray
                The fluxes of individual atmospheric columns in case the retrieval is run in the associated
                column_flux_return mode.
            generate_mock_data : bool
                Generate mock data for input = output tests. This is done in get_chisq because here the actual data
                and the forward models are brought to the same shape (resolution convolved, rebinned, column mixed).
        Returns:
            logL : float
                The log likelihood of the model given the data.
        """
        # TODO merge with SpectralModel: the chi2 calculation function should only calculate the chi2
        # Convolve to data resolution
        flux_rebinned = None

        if not self.photometry:
            if self.radtrans_grid:
                # TODO remove the concatenate_flux_epochs_variability treatment: this should now be handled by atmospheric_column_flux_mixer # noqa E501
                # TODO make atmospheric_column_flux_mixer accessible also to self.radtrans_grid = False data.
                if self.concatenate_flux_epochs_variability:
                    flux_rebinned = spectrum_model
                else:
                    index = (wlen_model >= self.wavelengths[0] * 0.99999999) & \
                            (wlen_model <= self.wavelengths[-1] * 1.00000001)
                    if not self.variability_atmospheric_column_model_flux_return_mode:
                        flux_rebinned = spectrum_model[index]
                    elif self.atmospheric_column_flux_mixer is not None:
                        flux_rebinned = self.atmospheric_column_flux_mixer(atmospheric_model_column_fluxes,
                                                                           parameters,
                                                                           self.name)
                        flux_rebinned = flux_rebinned[index]
            elif self.resample:
                resolution_slope = parameters[self.name + "_R_slope"].value
                resolution_intersect = parameters[self.name + "_R_int"].value
                resolution_array = (self.wavelengths*resolution_slope)+resolution_intersect
                # TODO: interpolate resolution_array and use standard convolve function.
                # TODO: compare different convolution and binning methods
                flux_rebinned = convolve_and_sample_variable_resolution_breads(
                    self.wavelengths,
                    resolution_array,
                    self.wavelengths,
                    spectrum_model
                )
            else:
                model_spectra = []
                column_rebinned_spectra = []
                if self.atmospheric_column_flux_mixer is not None:
                    for i_column_flux in range(np.shape(atmospheric_model_column_fluxes)[0]):
                        model_spectra.append(atmospheric_model_column_fluxes[i_column_flux, :])
                else:
                    model_spectra.append(spectrum_model)

                for spectrum_model in model_spectra:
                    if self.data_resolution_array_model is not None:
                        spectrum_model = convolve(
                            wlen_model,
                            spectrum_model,
                            self.data_resolution_array_model
                        )
                    elif self.data_resolution is not None:
                        spectrum_model = convolve(
                            wlen_model,
                            spectrum_model,
                            self.data_resolution
                        )

                    # Rebin to model observation
                    rebin = True
                    if np.size(wlen_model) == np.size(self.wavelengths) and np.all(wlen_model == self.wavelengths):
                        flux_rebinned = copy.deepcopy(spectrum_model)
                        rebin = False

                    if self.name + "_radial_velocity" in parameters.keys():
                        # RV in km/s -> multiply by 1e5 to cm/s
                        # wlen_model in micron
                        # cst.c in cm/s
                        radial_velocity = parameters[self.name + "_radial_velocity"].value * 1e5
                        wlen_model *= np.sqrt((1 + radial_velocity/cst.c)/(1 - radial_velocity/cst.c))
                    elif "system_radial_velocity" in parameters.keys():
                        radial_velocity = parameters["system_radial_velocity"].value * 1e5
                        wlen_model *= np.sqrt((1 + radial_velocity/cst.c)/(1 - radial_velocity/cst.c))

                    if rebin:
                        flux_rebinned = frebin.rebin_spectrum_bin(
                            wlen_model,
                            spectrum_model,
                            self.wavelengths,
                            self.wavelength_bin_widths
                        )

                    if self.atmospheric_column_flux_mixer is None:
                        break
                    else:
                        column_rebinned_spectra.append(flux_rebinned)

                if self.atmospheric_column_flux_mixer is not None:
                    column_rebinned_spectra = np.array(column_rebinned_spectra)
                    flux_rebinned = self.atmospheric_column_flux_mixer(column_rebinned_spectra,
                                                                       parameters,
                                                                       self.name)
        else:
            flux_rebinned = \
                self.photometric_transformation_function(wlen_model,
                                                         spectrum_model)
            # species spectrum_to_flux functions return (flux,error)
            if isinstance(flux_rebinned, (tuple, list)):
                flux_rebinned = flux_rebinned[0]

        if self.subtract_continuum:
            x_nodes = None
            if self.name + "_nodes" in parameters.keys():
                nodes = parameters[self.name + "_nodes"].value
                x_nodes = np.linspace(self.wavelengths[0], self.wavelengths[-1], nodes)

            if self.name + "_node_array" in parameters.keys():
                x_nodes = parameters[self.name + "_node_array"].value
            flux_rebinned = filter_spectrum_with_spline(self.wavelengths, flux_rebinned, x_nodes=x_nodes)

        if self.scale:
            diff = (flux_rebinned - self.spectrum * parameters[self.name + "_scale_factor"].value) + self.offset
        else:
            diff = (flux_rebinned - self.spectrum) + self.offset

        f_err = self.uncertainties
        b_val = None

        if f"{self.name}_b" in parameters.keys():
            b_val = parameters[self.name + "_b"].value
        elif f"{self.name.rsplit('_', 1)[0]}_b" in parameters.keys():
            b_val = parameters[f"{self.name.rsplit('_', 1)[0]}_b"].value
        elif "uncertainty_scaling_b" in parameters.keys():
            b_val = parameters["uncertainty_scaling_b"].value

        if b_val is not None:
            f_err = np.sqrt(f_err ** 2 + 10 ** b_val)

        if self.scale_err:
            f_err = f_err * parameters[self.name + "_scale_factor"].value

        log_l = 0.0
        log_l_per_datapoint = None

        if self.covariance is not None:
            inv_cov = self.inv_cov
            log_covariance_determinant = self.log_covariance_determinant

            if self.scale_err:
                cov = self.scale_factor ** 2 * self.covariance
                inv_cov = np.linalg.inv(cov)
                _, log_covariance_determinant = np.linalg.slogdet(2 * np.pi * cov)

            if b_val is not None:
                cov = np.diag(np.diag(self.covariance) + 10 ** b_val)
                inv_cov = np.linalg.inv(cov)
                _, log_covariance_determinant = np.linalg.slogdet(2 * np.pi * cov)

            log_l += -0.5 * np.dot(diff, np.dot(inv_cov, diff))
            log_l += -0.5 * log_covariance_determinant

            if per_datapoint:
                # Following Buerkner et al. (2020) to handle
                # off-diagonal covariance elements
                g_i = np.dot(inv_cov, diff)
                sigma_bar_ii = np.diag(inv_cov)

                sigma_tilde_i = 1 / sigma_bar_ii
                mu_tilde_i = flux_rebinned - g_i / sigma_bar_ii

                log_l_per_datapoint = (
                        -0.5 * np.log(2 * np.pi * sigma_tilde_i)
                        - 0.5 * (flux_rebinned - mu_tilde_i) ** 2 / sigma_tilde_i
                )
        else:
            log_l += -0.5 * np.sum((diff / f_err) ** 2)
            log_l += -0.5 * np.sum(np.log(2.0 * np.pi * f_err ** 2))
            if per_datapoint:  # TODO @Evert is there a point calculating log_l if only log_l_per_datapoint is returned?
                # Only diagonal covariance elements
                log_l_per_datapoint = -0.5 * np.log(2 * np.pi * f_err ** 2) - 0.5 * (diff / f_err) ** 2

        if plotting:
            import matplotlib.pyplot as plt

            if not self.photometry:
                plt.clf()
                plt.title(self.name)
                plt.plot(self.wavelengths, flux_rebinned)
                plt.errorbar(self.wavelengths,
                             self.spectrum * self.scale_factor,
                             yerr=f_err,
                             fmt='+')
                plt.show()

        if generate_mock_data:
            # Check if the mock data folder exists, if not create it:
            if not os.path.exists("mock_data"):
                os.makedirs("mock_data")

            np.savetxt("mock_data/" + self.name + "_mock_data.dat",
                       np.column_stack((self.wavelengths, flux_rebinned, self.uncertainties)))

        if per_datapoint:
            return log_l_per_datapoint

        return log_l

    def get_log_likelihood(self, spectrum_model):
        """Calculate the log-likelihood between the model and the data.

        The spectrum model must be on the same wavelength grid than the data.

        Args:
            spectrum_model: numpy.ndarray
                The model flux in the same units as the data.

        Returns:
            logL : float
                The log likelihood of the model given the data.
        """
        return self.log_likelihood(
            model=spectrum_model,
            data=self.spectrum,
            uncertainties=self.uncertainties,
            beta=self.scale_factor
        )

    # TODO: do we want to pass the whole parameter dict,
    #  or just set a class variable for b in the likelihood function?
    def line_b_uncertainty_scaling(self, parameters):
        """
        This function implements the 10^b scaling from Line 2015, which allows
        for us to account for underestimated uncertainties:

        We modify the standard error on the data point by the factor 10^b to account for
        underestimated uncertainties and/or unknown missing forward model physics
        (Foreman-Mackey et al. 2013, Hogg et al. 2010, Tremain et al. 2002), e.g., imperfect fits.
        This results in a more generous estimate of the parameter uncertainties. Note that this
        is similar to inflating the error bars post-facto in order to achieve reduced chi-squares
        of unity, except that this approach is more formal because uncertainties in this parameter
        are properly marginalized into the other relevant parameters. Generally, the factor 10^b
        takes on values that fall between the minimum and maximum of the square of the data uncertainties.

        Args:
            parameters: Dict
                Dictionary of Parameters, should contain key 'uncertainty_scaling_b'.
                This can be done for all data sets, or specified with a tag at the end of
                the key to apply different factors to different datasets.
        Returns:
            b: float
                10**b error bar scaling factor.
        """
        b_val = -np.inf
        if parameters is not None:
            if f'{self.name}_b' in parameters.keys():
                b_val = parameters[f'{self.name}_b'].value
            elif 'uncertainty_scaling_b' in parameters.keys():
                b_val = parameters['uncertainty_scaling_b'].value
        return b_val

    @staticmethod
    def log_likelihood(
        model: npt.NDArray[np.floating],
        data: npt.NDArray[np.floating],
        uncertainties: npt.NDArray[np.floating],
        beta: float | None = None,
        beta_mode: str = 'multiply'
    ) -> float:
        """Calculate the log-likelihood between the model and the data.

        The spectrum model must be on the same wavelength grid than the data.

        From Gibson et al. 2020 (https://doi.org/10.1093/mnras/staa228). Constant terms are dropped and constant
        coefficients are set to 1.
        The 'add' beta mode comes from Line et al. 2015 (DOI 10.1088/0004-637X/807/2/183).

        Set:
            chi2(A, B) = sum(((f_i - A * m_i) / (B * sig_i)) ** 2),  implicit sum on i
        with f_i the data, m_i the model, and sig_i the uncertainty; where "i" denotes wavelength/time variation.
        Starting from (implicit product on i):
            L        = prod(1 / sqrt(2 * pi * B * sig_i ** 2) * exp(-1/2 * sum(((f_i - A * m_i) / (B * sig_i)) ** 2))),
            => L     = prod( 1 / sqrt(2 * pi * B * sig_i ** 2) * exp(-1/2 * chi2(A, B)) ),
            => ln(L) = -N/2 * ln(2 * pi) - N * ln(B) - sum(ln(sig_i)) - 1/2 * chi2(A, B).
        Dropping constant terms:
            ln(L)^* = - N * ln(B) - 1/2 * chi2(A, B).

        B can be automatically optimized by "nulling" the ln(L) partial derivative with respect to B.
        Using the best estimator of B instead of the true value:
            d_ln(L) / d_B = - N / B + 1 / B ** 3 * chi2(A, B=1) = 0,
            => B = sqrt( 1/N *  chi2(A, B=1)).
        Replacing:
            ln(L)^**= - N * ln(B) - 1/2 * chi2(A, B),
                    = - N * ln(sqrt(1/N * chi2(A, B=1))) - 1/2 * sum(((f_i - A * m_i) / (B * sig_i)) ** 2),
                    = - N/2 * ln(1/N * chi2(A, B=1)) - 1/2 / B ** 2 * sum(((f_i - A * m_i) / sig_i) ** 2),  B cst with i
                    = - N/2 * ln(1/N * chi2(A, B=1)) - 1/2 * N / chi2(A, B=1) * chi2(A, B=1),
                    = - N/2 * ln(1/N * chi2(A, B=1)) - N/2.
        Dropping constant terms:
            ln(L)^*** = - N/2 * ln(1/N * chi2(A, B=1)).

        Args:
            model: numpy.ndarray
                The model flux in the same units as the data.
            data: numpy.ndarray
                The data.
            uncertainties: numpy.ndarray
                The uncertainties on the data.
            beta: float, optional
                Noise scaling coefficient. If None, the noise scaling is "automatically optimised".
            beta_mode: string, optional


        Returns:
            logL : float
                The log likelihood of the model given the data.
        """
        chi2: float | npt.NDArray[np.floating]

        if beta is None:
            # "Automatically optimize" for beta
            chi2 = data - model
            chi2 /= uncertainties
            chi2 *= chi2
            chi2 = chi2.sum()

            return - 0.5 * data.size * np.log(chi2 / data.size)
        else:
            penalty_term: float

            # Classical log-likelihood
            if beta_mode == 'multiply':
                uncertainties = uncertainties * beta
                penalty_term = - data.size * np.log(beta)
            elif beta_mode == 'add':
                uncertainties = uncertainties + beta
                penalty_term = - np.sum(np.log(uncertainties))
            else:
                raise ValueError(f"beta mode must be 'multiply'|'add', but was '{beta_mode}'")

            chi2 = data - model
            chi2 /= uncertainties
            chi2 *= chi2
            chi2 = chi2.sum()

            return - 0.5 * chi2 + penalty_term

    @staticmethod
    def log_likelihood2chi2(log_likelihood: float) -> float:
        """Convert a log-likelihood value into its equivalent chi2 value."""
        return -2 * log_likelihood
