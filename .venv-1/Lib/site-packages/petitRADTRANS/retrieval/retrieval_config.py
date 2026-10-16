import logging
import os
from typing import Callable

import numpy as np

from petitRADTRANS.config.configuration import petitradtrans_config_parser
from petitRADTRANS.retrieval.data import Data
from petitRADTRANS.retrieval.parameter import Parameter
from petitRADTRANS.opacities.opacities import Opacity, CloudOpacity

# MPI Multiprocessing
rank = 0
comm = None

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
except ImportError:
    MPI = None


class RetrievalConfig:
    """Contain all the data and model level information necessary to run a petitRADTRANS retrieval.

    The name of the class will be used to name outputs.
    This class is passed to the Retrieval, which runs the actual pymultinest retrieval and produces the outputs.

    The general usage of this class is to define it, add the parameters and their priors, add the opacity sources,
    the data together with a model for each dataset, and then configure a few plotting arguments.

    Args:
        retrieval_name : str
            Name of this retrieval. Make it informative so that you can keep track of the outputs!
        run_mode : str
            Can be either 'retrieval', which runs the retrieval normally using pymultinest,
            or 'evaluate', which produces plots from the best fit parameters stored in the
            output post_equal_weights file.
        amr : bool
            Use an adaptive high resolution pressure grid around the location of cloud condensation.
            This will increase the size of the pressure grid by a constant factor that can be adjusted
            in the setup_pres function.
        scattering_in_emission : bool
            If using emission spectra, turn scattering on or off.
        pressures : numpy.array
            A log-spaced array of pressures over which to retrieve. 100 points is standard, between
            10^-6 and 10^3.
    """

    def __init__(self,
                 retrieval_name="retrieval_name",
                 run_mode="retrieval",
                 amr=False,
                 scattering_in_emission=False,
                 pressures=None):

        self.retrieval_name = retrieval_name

        if run_mode == 'retrieve':
            run_mode = 'retrieval'  # TODO please no.

        self.run_mode = run_mode

        if self.run_mode != 'retrieval' and self.run_mode != 'evaluate':
            raise ValueError(f"run_mode must be 'retrieval' or 'evaluate', but was '{self.run_mode}'")

        self.amr = amr

        if pressures is not None:
            self.pressures = pressures
        else:
            self.pressures = np.logspace(-6, 3, 100)

        self.scattering_in_emission = scattering_in_emission
        self.parameters = {}  #: Dictionary of the parameters passed to the model generating function
        self.data = {}  #: Dictionary of the datasets used in the retrieval.
        self.instruments = []
        self.line_species = []
        self.cloud_species = []
        self.rayleigh_species = []
        self.continuum_opacities = []
        self.plot_kwargs = {}

        self._plot_defaults()

        self.add_parameter("pressure_scaling", False, value=1)
        self.add_parameter("pressure_width", False, value=1)
        self.add_parameter("pressure_simple", False, value=self.pressures.shape[0])

    def _plot_defaults(self):
        ##################################################################
        # Define axis properties of spectral plot if run_mode == 'evaluate'
        ##################################################################
        self.plot_kwargs["spec_xlabel"] = 'Wavelength [micron]'
        self.plot_kwargs["spec_ylabel"] = "Flux [W/m2/micron]"
        self.plot_kwargs["y_axis_scaling"] = 1.0
        self.plot_kwargs["xscale"] = 'log'
        self.plot_kwargs["yscale"] = 'linear'
        self.plot_kwargs["resolution"] = 1500.
        self.plot_kwargs["nsample"] = 10.

        ##################################################################
        # Define from which observation object to take P-T
        # in evaluation mode (if run_mode == 'evaluate'),
        # add PT-envelope plotting options
        ##################################################################
        self.plot_kwargs["temp_limits"] = [150, 3000]
        self.plot_kwargs["press_limits"] = [1e2, 1e-5]

    def _setup_pres(self, scaling=10, width=3):
        """
        This converts the standard pressure grid into the correct length
        for the AMR pressure grid. The scaling adjusts the resolution of the
        high resolution grid, while the width determines the size of the high
        pressure region. This function is automatically called in
        Retrieval.setupData().

        Args:
            scaling : int
                A multiplicative factor that determines the size of the full high resolution pressure grid,
                which will have length self.p_global.shape[0] * scaling.
            width : int
                The number of cells in the low pressure grid to replace with the high resolution grid.
        """

        print("Setting up AMR pressure grid.")
        self.scaling = scaling
        self.width = width
        nclouds = len(self.cloud_species)
        if nclouds == 0:
            print("WARNING: there are no clouds in the retrieval, please add cloud species before setting up AMR")
        new_len = self.pressures.shape[0] + nclouds * width * (scaling - 1)
        self.amr_pressure = np.logspace(np.log10(self.pressures[0]), np.log10(self.pressures[-1]), new_len)
        self.add_parameter("pressure_scaling", False, value=scaling)
        self.add_parameter("pressure_width", False, value=width)
        self.add_parameter("pressure_simple", False, value=self.pressures.shape[0])

        return self.amr_pressure

    def add_parameter(self, name, free, value=None, transform_prior_cube_coordinate=None):
        """
        This function adds a Parameter (see parameter.py) to the dictionary of parameters. A Parameter
        has a name and a boolean parameter to set whether it is a free or fixed parameter during the retrieval.
        In addition, a value can be set, or a prior function can be given that transforms a random variable in
        [0,1] to the physical dimensions of the Parameter.

        Args:
            name : str
                The name of the parameter. Must match the name used in the model function for the retrieval.
            free : bool
                True if the parameter is a free parameter in the retrieval, false if it is fixed.
            value : float
                The value of the parameter in the units used by the model function.
            transform_prior_cube_coordinate : method
                A function that transforms the unit interval to the physical units of the parameter.
                Typically given as a lambda function.
        """

        self.parameters[name] = Parameter(name, free, value,
                                          transform_prior_cube_coordinate=transform_prior_cube_coordinate)

    @staticmethod
    def list_available_line_species():
        """
        List the currently installed opacity tables that are available for species that contribute to the line opacity.
        """

        prt_path = petitradtrans_config_parser.get_input_data_path()

        files = [f[0].split('/')[-1] for f in os.walk(prt_path + "/opacities/lines/correlated_k/")]
        files = set([f.split('.R')[0] for f in files])
        print("\ncorrelated-k opacities")

        for f in files:
            print(f)

        lbl = [f[0].split('/')[-1] for f in os.walk(prt_path + "/opacities/lines/line_by_line/")]
        lbl = set(lbl)
        print("\nline-by-line opacities")

        for f in lbl:
            print(f)

        return files.union(lbl)

    @staticmethod
    def list_available_cloud_species():
        """
        List the currently installed opacity tables that are available for cloud species.
        """

        prt_path = petitradtrans_config_parser.get_input_data_path()

        files = [f[0].split('/')[-1] for f in os.walk(prt_path + "/opacities/continuum/clouds/")]
        files = set(files)

        for f in files:
            print(f)

        return files

    @staticmethod
    def list_available_cia_species():
        """
        List the currently installed opacity tables that are available for CIA species.
        """

        prt_path = petitradtrans_config_parser.get_input_data_path()

        files = [f[0].split('/')[-1] for f in os.walk(prt_path + "/opacities/continuum/cia/")]
        files = set(files)

        for f in files:
            print(f)

        return files

    def set_line_species(self, linelist, eq=False, abund_lim=(-6.0, -0.5)):
        """
        Set RadTrans.line_species

        This function adds a list of species to the pRT object that will define the line
        opacities of the model. The values in the list are strings, with the names matching
        the pRT opacity names, which vary between the c-k line opacities and the line-by-line opacities.

        NOTE: As of pRT version 2.4.9, the behavior of this function has changed. In previous versions the
        abundance limits were set from abund_lim[0] to (abund_lim[0] + abund_lim[1]). This has been changed
        so that the limits of the prior range are from abund_lim[0] to abund_lim[1] (ie the actual boundaries).

        Args:
            linelist : List(str)
                The list of species to include in the retrieval
            eq : bool
                If false, the retrieval should use free chemistry, and Parameters for the abundance of each
                species in the linelist will be added to the retrieval. Otherwise, equilibrium chemistry will
                be used. If you need fine control species, use the add_line_species and set up each species
                individually.
            abund_lim : Tuple(float,float)
                If free is True, this sets the boundaries of the uniform prior that will be applied for
                each species in linelist. The range of the prior goes from abund_lim[0] to abund_lim[1].
                The abundance limits must be given in log10 units of the mass fraction.
        """
        if abund_lim[1] > 0.0:
            raise ValueError(
                f"upper limit must be <= 0.0 (was {abund_lim})! Please set abundance limits as (low, high)"
            )

        self.line_species = linelist

        if not eq:
            for species in self.line_species:
                Opacity.check_name(species)
                species_opacity = Opacity([species])
                species_full_name = species_opacity.get_full_name().split('.R')[0]
                print(species, species_full_name)
                self.parameters[species_full_name] = Parameter(
                    species_full_name,
                    True,
                    transform_prior_cube_coordinate=lambda x: abund_lim[0] + (abund_lim[1] - abund_lim[0]) * x
                )

    def set_rayleigh_species(self, linelist):
        """
        Set the list of species that contribute to the rayleigh scattering in the pRT object.

        Args:
            linelist : List(str)
                A list of species that contribute to the rayleigh opacity.
        """

        self.rayleigh_species = linelist

    def set_continuum_opacities(self, linelist):
        """
        Set the list of species that contribute to the continuum opacity in the pRT object.

        Args:
            linelist : List(str)
                A list of species that contribute to the continuum opacity.
        """

        self.continuum_opacities = linelist

    def add_line_species(self, species, eq=False, abund_lim=(-7.0, 0.0), fixed_abund=None):
        """
        This function adds a single species to the pRT object that will define the line opacities of the model.
        The name must match the pRT opacity name, which vary between the c-k line opacities and the line-by-line
        opacities.

        NOTE: As of pRT version 2.4.9, the behavior of this function has changed. In previous versions the
        abundance limits were set from abund_lim[0] to (abund_lim[0] + abund_lim[1]). This has been changed
        so that the limits of the prior range are from abund_lim[0] to abund_lim[1] (ie the actual boundaries).

        Args:
            species : str
                The species to include in the retrieval
            eq : bool
                If False, the retrieval should use free chemistry, and Parameters for the abundance of the
                species will be added to the retrieval. Otherwise, (dis)equilibrium chemistry will be used.
            abund_lim : Tuple(float,float)
                If free is True, this sets the boundaries of the uniform prior that will be applied the species given.
                The range of the prior goes from abund_lim[0] to abund_lim[1]
                The abundance limits must be given in log10 units of the mass fraction.
            fixed_abund : float
                The log-mass fraction abundance of the species. Currently only supports vertically constant
                abundances. If this is set, then the species will not be a free parameter in the retrieval.
        """

        # parameter passed through loglike is log10 abundance
        if abund_lim[1] > 0.0:
            raise ValueError(
                f"upper limit must be <= 0.0 (was {abund_lim})! Please set abundance limits as (low, high)"
            )

        Opacity.check_name(species)
        species_opacity = Opacity([species])
        species_full_name = species_opacity.get_full_name().split('.R')[0]

        self.line_species.append(species_full_name)
        if not eq:
            if fixed_abund is not None:
                self.parameters[species_full_name] = Parameter(
                    species_full_name,
                    is_free_parameter=False,
                    value=fixed_abund)
            else:
                self.parameters[species_full_name] = Parameter(
                    species_full_name,
                    is_free_parameter=True,
                    transform_prior_cube_coordinate=lambda x: abund_lim[0] + (abund_lim[1] - abund_lim[0]) * x
                )

    def add_pressure_varying_line_species(
            self,
            species,
            mode='linear',
            pressure_spacing='relative',
            n_nodes=3,
            abund_lim=(-7.0, 0.0),
            log_pressure_range_prior=(0, 9),
            fixed_pressure_node_species=None
            ):
        """
        This function adds a single species to the Radtrans object that will define the line opacities of the model.
        The name must match the pRT opacity name, which vary between the c-k line opacities and the line-by-line
        opacities. This species will have and abundance that varies with pressure, defined by the retrieved
        abundance at each of the provided abundance nodes.

        This function adds a set of parameters to the retrieval to define the abundance profile of the species.
         - {species}_{mode}_abundance_profile defines which profile will be used (fixed parameter)
         - {species}_n_abundance_nodes sets the number of abundance nodes (fixed parameter)
         - {species}_n_pressure_nodes sets the number of pressure nodes (n_nodes - 2) (fixed parameter)
         - {species}_interpolation_mode sets the pressure spacing mode (fixed parameter)
         - {species}_pressure_node_{i} for i in 0 to n_nodes-2, the pressure at each node (free parameter)
         - {species}_abundance_node_{i} for i in 0 to n_nodes, the abundance at each node (free parameter)

        Args:
            species: str
                The species to include in the retrieval
            mode: str
                One of 'linear', 'cubic', or 'stepped' - determines the interpolation method between abundance nodes.
            pressure_spacing: str
                One of 'relative', 'absolute', or 'fixed' - determines whether the pressure nodes are spaced
                relative to the top of the atmosphere pressure, retrieved in absolute log pressue, or fixed
                to the pressure nodes set by fixed_pressure_node_species.
            n_nodes: int
                The number of abundance nodes to use in the retrieval, including top and bottom nodes.
            abund_lim : Tuple(float,float)
                If free is True, this sets the boundaries of the uniform prior that will be applied the species given.
                The range of the prior goes from abund_lim[0] to abund_lim[1]
                The abundance limits must be given in log10 units of the mass fraction.
            log_pressure_range_prior: Tuple(float,float)
                The prior range on the log pressure of the pressure nodes in log bar, only used if pressure_spacing is
                'absolute' or 'relative'.
            fixed_pressure_node_species: str
                If pressure_spacing is 'fixed', this species pressure nodes will be used as the pressure nodes.
                Note that this species must already have been added to the retrieval with
                add_pressure_varying_line_species.
        """
        Opacity.check_name(species)
        species_opacity = Opacity([species])
        species_full_name = species_opacity.get_full_name().split('.R')[0]
        # parameter passed through loglike is log10 abundance
        if abund_lim[1] > 0.0:
            raise ValueError(
                f"upper limit must be <= 0.0 (was {abund_lim})! Please set abundance limits as (low, high)"
            )

        self.parameters[f"{species_full_name}_{mode}_abundance_profile"] = Parameter(
            name=f"{species_full_name}_linear_abundance_profile",
            is_free_parameter=False,
            value=True
        )
        self.parameters[f"{species_full_name}_n_abundance_nodes"] = Parameter(
            name=f"{species_full_name}_n_abundance_nodes",
            is_free_parameter=False,
            value=n_nodes
        )
        self.parameters[f"{species_full_name}_n_pressure_nodes"] = Parameter(
            name=f"{species_full_name}_n_pressure_nodes",
            is_free_parameter=False,
            value=n_nodes-2
        )
        self.parameters[f"{species_full_name}_interpolation_mode"] = Parameter(
            name=f"{species_full_name}_interpolation_mode",
            is_free_parameter=False,
            value=pressure_spacing
        )

        self.line_species.append(species_full_name)

        for i in range(n_nodes - 2):
            if not pressure_spacing == 'fixed':
                self.parameters[f"{species_full_name}_pressure_node_{i}"] = Parameter(
                    name=f"{species_full_name}_pressure_node_{i}",
                    is_free_parameter=True,
                    transform_prior_cube_coordinate=lambda x: log_pressure_range_prior[0] +
                    log_pressure_range_prior[1] * x
                )
            else:
                self.parameters[f"{species_full_name}_pressure_node_{i}"] = Parameter(
                    name=f"{fixed_pressure_node_species}_pressure_node_{i}",
                    is_free_parameter=False,
                    value=self.parameters[f"{fixed_pressure_node_species}_pressure_node_{i}"].value
                )

        for i in range(n_nodes):
            self.parameters[f"{species_full_name}_abundance_node_{i}"] = Parameter(
                name=f"{species_full_name}_abundance_node_{i}",
                is_free_parameter=True,
                transform_prior_cube_coordinate=lambda x: abund_lim[0] +
                (abund_lim[1] - abund_lim[0]) * x
            )

    def remove_species_lines(self, species, free=False):
        """
        This function removes a species from the pRT line list, and if using a free chemistry retrieval,
        removes the associated Parameter of the species.

        Args:
            species : str
                The species to remove from the retrieval
            free : bool
                If true, the retrieval should use free chemistry, and Parameters for the abundance of the
                species will be removed to the retrieval
        """
        remove_opacity = Opacity([species])
        species_full_name = remove_opacity.get_full_name().split('.R')[0]
        if species in self.line_species:
            self.line_species.remove(species)
        elif species_full_name in self.line_species:
            self.line_species.remove(species_full_name)
        if free:
            self.parameters.pop(species, None)
            self.parameters.pop(species_full_name, None)

    def add_cloud_species(self, species, eq=True, abund_lim=(-3.5, 1.5), p_base_lim=None, fixed_abund=None,
                          scaling_factor=None, fixed_base=None):
        """
        This function adds a single cloud species to the list of species. Optionally,
        it will add parameters to allow for a retrieval using an ackermann-marley model.
        If an equilibrium condensation model is used in th retrieval model function (eq=True),
        then a parameter is added that scales the equilibrium cloud abundance, as in Molliere (2020).
        If eq is false, two parameters are added, the cloud abundnace and the cloud base pressure.
        The limits set the prior ranges, both on a log scale.

        NOTE: As of pRT version 2.4.9, the behaviour of this function has changed. In previous versions the
        abundance limits were set from abund_lim[0] to (abund_lim[0] + abund_lim[1]). This has been changed
        so that the limits of the prior range are from abund_lim[0] to abund_lim[1] (ie the actual boundaries).
        The same is true for PBase_lim.

        Args:
            species : str
                Name of the pRT cloud species, including the cloud shape tag.
            eq : bool
                Does the retrieval model use an equilibrium cloud model. This restricts the available species!
            abund_lim : tuple(float,float)
                If eq is True, this sets the scaling factor for the equilibrium condensate abundance, typical
                range would be (-3,1). If eq is false, this sets the range on the actual cloud abundance,
                with a typical range being (-5,0).
            p_base_lim : tuple(float,float)
                Only used if not using an equilibrium model. Sets the limits on the log of the cloud base pressure.
                Obsolete.
            fixed_abund : Optional(float)
                A vertically constant log mass fraction abundance for the cloud species. If set, this will not be
                a free parameter in the retrieval. Only compatible with non-equilibrium clouds.
            fixed_base : Optional(float)
                The log cloud base pressure. If set, fixes this parameter to a constant value, and it will not be
                a free parameter in the retrieval. Only compatible with non-equilibrium clouds. Not yet compatible
                with most built in pRT models.
            scaling_factor :
                # TODO complete docstring
        """

        if species.endswith("(c)"):
            logging.warning("Ensure you set the cloud particle shape, typically with the _cd tag!")
            logging.warning(species + " was not added to the list of cloud species")
            return
        cloud_opacity = CloudOpacity([species], natural_abundance=False)
        species_full_name = cloud_opacity.species_full_name
        cloud_name = species_full_name.split('_')[0]

        # print(species, species_full_name, cloud_opacity.species_isotopologue_name, cloud_name)
        self.cloud_species.append(cloud_opacity.species_full_name)
        if scaling_factor is not None:
            self.parameters['eq_scaling_' + cloud_name] = Parameter(
                'eq_scaling_' + cloud_name, True,
                transform_prior_cube_coordinate=lambda x: scaling_factor[0] + (
                   scaling_factor[1] - scaling_factor[
                    0]
                ) * x
            )
        if not eq:
            if abund_lim[1] > 0.0:
                raise ValueError(
                    f"upper limit must be <= 0.0 (was {abund_lim})! Please set abundance limits as (low, high)"
                )

            if fixed_abund is None:
                self.parameters['log_X_cb_' + cloud_name] = Parameter(
                    'log_X_cb_' + cloud_name,
                    True,
                    transform_prior_cube_coordinate=lambda x: abund_lim[0] + (abund_lim[1] - abund_lim[0]) * x
                )
            else:
                self.parameters['log_X_cb_' + cloud_name] = Parameter(
                    'log_X_cb_' + cloud_name,
                    False,
                    value=fixed_abund
                )

        if p_base_lim is not None or fixed_base is not None:
            if fixed_base is None:
                self.parameters['log_Pbase_' + cloud_name] = Parameter(
                    'log_Pbase_' + cloud_name,
                    True,
                    transform_prior_cube_coordinate=lambda x: p_base_lim[0] + (p_base_lim[1] - p_base_lim[0]) * x
                )
            else:
                self.parameters['log_Pbase_' + cloud_name] = Parameter(
                    'log_Pbase_' + cloud_name,
                    False,
                    value=fixed_base
                )

    # TODO @Evert complete docstring
    # TODO [4.0.0] remove radvel as it is not used
    def add_data(self,
                 name,
                 path_to_observations,
                 model_generating_function,
                 data_resolution=None,
                 model_resolution=None,
                 system_distance=None,
                 scale=False,
                 scale_err=False,
                 offset_bool=False,
                 resample=False,
                 subtract_continuum=False,
                 radvel=False,
                 photometry=False,
                 photometric_transformation_function=None,
                 photometric_bin_edges=None,
                 wavelength_boundaries=None,
                 external_radtrans_reference=None,
                 line_opacity_mode='c-k',
                 wavelength_bin_widths=None,
                 radtrans_grid=False,
                 radtrans_object=None,
                 wavelengths=None,
                 spectrum=None,
                 uncertainties=None,
                 covariance=None,
                 mask=None,
                 concatenate_flux_epochs_variability=False,
                 variability_atmospheric_column_model_flux_return_mode=False,
                 atmospheric_column_flux_mixer=None):
        """
        Create a Data class object.

        Args:
            name : str
                Identifier for this data set.
            path_to_observations : str
                Path to observations file, including filename. This can be a txt or dat file containing the wavelength,
                flux, transit depth and error, or a fits file containing the wavelength, spectrum and covariance matrix.
            model_generating_function : fnc
                A function, typically defined in run_definition.py that returns the model wavelength and
                spectrum (emission or transmission). This is the function that contains the physics
                of the model, and calls pRT in order to compute the spectrum.
            data_resolution : float
                Spectral resolution of the instrument. Optional, allows convolution of model to instrumental line width.
            model_resolution : float
                Spectral resolution of the model, allowing for low resolution correlated k tables from exo-k.
            system_distance : float
                The distance to the object in cgs units. Defaults to a 10pc normalized distance. All data must
                be scaled to the same distance before running the retrieval, which can be done using the
                scale_to_distance method in the Data class.
            scale : bool
                Turn on or off scaling the data by a constant factor.
            wavelength_boundaries : Tuple
                A pair of wavelengths in units of micron that determine the lower and upper boundaries of the
                model computation.
            external_radtrans_reference : str
                The name of an existing Data object. This object's prt_object will be used to calculate the chi squared
                of the new Data object. This is useful when two datasets overlap, as only one model computation is
                required to compute the log likelihood of both datasets.
            line_opacity_mode : str
                Should the retrieval be run using correlated-k opacities (default, 'c-k'),
                or line by line ('lbl') opacities? If 'lbl' is selected, it is HIGHLY
                recommended to set the model_resolution parameter.
            radtrans_grid: bool
                Set to true if data has been binned to a pRT opacity grid, exactly.
        """
        self.data[name] = Data(
            name=name,
            path_to_observations=path_to_observations,
            model_generating_function=model_generating_function,
            data_resolution=data_resolution,
            model_resolution=model_resolution,
            system_distance=system_distance,
            scale=scale,
            scale_err=scale_err,
            offset_bool=offset_bool,
            resample=resample,
            subtract_continuum=subtract_continuum,
            photometry=photometry,
            photometric_transformation_function=photometric_transformation_function,
            photometric_bin_edges=photometric_bin_edges,
            wavelength_boundaries=wavelength_boundaries,
            external_radtrans_reference=external_radtrans_reference,
            line_opacity_mode=line_opacity_mode,
            wavelength_bin_widths=wavelength_bin_widths,
            radtrans_grid=radtrans_grid,
            radtrans_object=radtrans_object,
            wavelengths=wavelengths,
            spectrum=spectrum,
            uncertainties=uncertainties,
            covariance=covariance,
            mask=mask,
            concatenate_flux_epochs_variability=concatenate_flux_epochs_variability,
            variability_atmospheric_column_model_flux_return_mode=variability_atmospheric_column_model_flux_return_mode,
            atmospheric_column_flux_mixer=atmospheric_column_flux_mixer
        )

    def add_photometry(
        self,
        path: str,
        model_generating_function: Callable,
        model_resolution: float = 10.0,
        distance: float | None = None,
        scale: bool = False,
        wlen_range_micron: tuple[float, float] = None,
        photometric_transformation_function: Callable = None,
        external_prt_reference: object = None,
        opacity_mode: str = 'c-k'
    ) -> None:
        """
        Create a Data class object for each photometric point in a photometry file.
        The photometry file must be a csv file and have the following structure:
        name, lower wavelength bound [um], upper wavelength boundary[um], flux [W/m2/micron], flux error [W/m2/micron]

        Photometric data requires a transformation function to convert a spectrum into synthetic photometry.
        You must provide this function yourself, or have the species package installed.
        If using species, the name in the data file must be of the format instrument/filter.

        Args:
            model_generating_function : str
                Identifier for this data set.
            path : str
                Path to observations file, including filename.
            model_resolution : float
                Spectral resolution of the model, allowing for low resolution correlated k tables from exo-k.
            scale : bool
                Turn on or off scaling the data by a constant factor. Currently only set up to scale all
                photometric data in a given file.
            distance : float
                The distance to the object in cgs units. Defaults to a 10pc normalized distance. All data must
                be scaled to the same distance before running the retrieval, which can be done using the
                scale_to_distance method in the Data class.
            wlen_range_micron : Tuple
                A pair of wavelengths in units of micron that determine the lower and upper boundaries of
                the model computation.
            external_prt_reference : str
                The name of an existing Data object. This object's prt_object will be used to calculate the
                chi squared of the new Data object. This is useful when two datasets overlap, as only
                one model computation is required to compute the log likelihood of both datasets.
            photometric_transformation_function : method
                A function that will transform a spectrum into an average synthetic photometric point,
                typically accounting for filter transmission.
            opacity_mode: str
                Opacity mode.
        """
        with open(path) as photometry:
            if photometric_transformation_function is None:
                try:
                    import species as sp
                except ModuleNotFoundError as e:
                    e.add_note(
                        "Please provide a function to transform a spectrum into photometry, or pip install species"
                    )
                    raise e

                from species.phot.syn_phot import SyntheticPhotometry

                sp.SpeciesInit()
            else:
                # Prevent potential reference before assignment
                class SyntheticPhotometry:
                    ...

            for line in photometry:
                # # must be the comment character
                if line[0] == '#':
                    continue

                columns = line.split(',')
                name = columns[0]
                wavelengths_low = float(columns[1])
                wavelengths_high = float(columns[2])
                flux = float(columns[3])
                uncertainties = float(columns[4])

                if photometric_transformation_function is None:
                    if comm is not None and comm.Get_size() > 1:
                        if rank == 0:
                            transform = SyntheticPhotometry(name).spectrum_to_flux
                        else:
                            transform = None  # transform still needs to exist in the other processes
                        transform = comm.bcast(transform, 0)
                    else:
                        transform = SyntheticPhotometry(name).spectrum_to_flux
                else:
                    transform = photometric_transformation_function

                if wlen_range_micron is None:
                    wbins = [0.95 * wavelengths_low, 1.05 * wavelengths_high]
                else:
                    wbins = wlen_range_micron

                if opacity_mode == 'lbl':
                    logging.warning("Are you sure you want a high resolution model for photometry?")

                data_resolution: float = (
                    np.array(
                        [wavelengths_low, wavelengths_high]
                    ).mean()
                    / (wavelengths_high - wavelengths_low)
                )

                self.data[name] = Data(
                    name=name,
                    path_to_observations=path,
                    model_generating_function=model_generating_function,
                    system_distance=distance,
                    photometry=True,
                    wavelength_boundaries=wbins,
                    photometric_bin_edges=(wavelengths_low, wavelengths_high),
                    data_resolution=data_resolution,
                    model_resolution=model_resolution,
                    scale=scale,
                    photometric_transformation_function=transform,
                    external_radtrans_reference=external_prt_reference,
                    line_opacity_mode=opacity_mode
                )
                self.data[name].wavelengths = np.mean(np.vstack((wavelengths_low, wavelengths_high)), axis=0)
                self.data[name].spectrum = flux
                self.data[name].uncertainties = uncertainties
