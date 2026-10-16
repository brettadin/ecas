import copy

import numpy as np

try:
    from petitRADTRANS.chemistry.prt_easychem import get_exoatmos_abundances
except ModuleNotFoundError:
    get_exoatmos_abundances = None

from petitRADTRANS.chemistry import clouds as fc
from petitRADTRANS.chemistry.pre_calculated_chemistry import pre_calculated_equilibrium_chemistry_table
from petitRADTRANS.chemistry.utils import (
    compute_mean_molar_masses,
    cubic_spline_profile,
    define_abundance_node_list,
    define_pressure_node_list,
    fixed_length_amr,
    linear_spline_profile,
    stepped_profile,
)
from petitRADTRANS.opacities.opacities import Opacity, CloudOpacity


def get_abundances(pressures, temperatures, line_species, cloud_species, parameters, amr=False):
    """
    This function takes in the C/O ratio, metallicity, and quench pressures and uses them
    to compute the gas phase and equilibrium condensate abundances from an interpolated table.
    Alternatively, a "free chemistry" approach can be used to set abundances.
    This function assumes a hydrogen-helium dominated atmosphere, and enforces <100% trace gas
    abundance by mass.

    Args:
        pressures : numpy.ndarray
            A log spaced pressure array. If AMR is on it should be the full high resolution grid.
        temperatures : numpy.ndarray
            A temperature array with the same shape as pressures
        line_species : List(str)
            A list of gas species that will contribute to the line-by-line opacity of the pRT atmosphere.
        cloud_species : List(str)
            A list of condensate species that will contribute to the cloud opacity of the pRT atmosphere.
        parameters : dict
            A dictionary of model parameters, in particular it must contain the names C/O, Fe/H and
            log_pquench. Additionally, the cloud parameters log_X_cb_Fe(c) and MgSiO3(c) must be present.
        amr : bool
            Turn the adaptive mesh grid on or off. See fixed_length_amr for implementation.

    Returns:
        abundances : dict
            Mass fraction abundances of all atmospheric species
        MMW : numpy.ndarray
            Array of the mean molecular weights in each pressure bin
        pressure_indices : numpy.ndarray
            The indices of the high resolution grid to use to define the adaptive grid.
        PBases : dict
            A dictionary of the cloud base pressures, either computed from equilibrium
            condensation or set by the user.
    """
    # TODO replace by SpectralModel function
    # Free Chemistry
    abundances_interp = {}
    mmw = None

    if 'use_easychem' in parameters.keys():
        if get_exoatmos_abundances is None:
            raise ModuleNotFoundError("module 'easychem' is not installed, "
                                      "remove key 'use_easychem from the parameters'")

        # Actual equilibrium chemistry
        # Can retrieve atomic abundances

        # Calling it abundances_interp to be consistent with poor man's
        abundances_interp = get_exoatmos_abundances(pressures,
                                                    temperatures,
                                                    parameters)
        mmw = abundances_interp['MMW']
    elif "C/O" in parameters.keys():
        # Check C/O AFTER easychem check -> need to use poor man's

        # Interpolated Equilibrium chemistry
        # Make the abundance profile
        pquench_c = None

        if 'log_pquench' in parameters.keys():
            pquench_c = 10 ** parameters['log_pquench'].value

        abundances_interp, mmw, _ = (
            pre_calculated_equilibrium_chemistry_table.interpolate_mass_fractions(
                parameters['C/O'].value * np.ones_like(pressures),
                parameters['Fe/H'].value * np.ones_like(pressures),
                temperatures,
                pressures,
                carbon_pressure_quench=pquench_c,
                full=True
            )
        )
    # Free chemistry abundances
    msum = np.zeros_like(pressures)

    if abundances_interp:
        for key, val in abundances_interp.items():
            msum += val

    # Free chemistry species
    for species in line_species:
        species_opacity = Opacity([species])
        species_full_name = species_opacity.get_full_name().split('.R')[0]
        species_basename = Opacity.get_species_base_name(species)
        # Vertically constant abundance
        if species_full_name in abundances_interp.keys():
            msum -= abundances_interp[species_full_name]
        if species_basename in abundances_interp.keys():
            msum -= abundances_interp[species_basename]

        if species_full_name in parameters.keys():
            abund = 10 ** parameters[species_full_name].value
            abundances_interp[species_full_name] = abund * np.ones_like(pressures)
            msum += abundances_interp[species_full_name]
            continue
        elif species_basename in parameters.keys():
            abund = 10 ** parameters[species_basename].value
            abundances_interp[species_basename] = abund * np.ones_like(pressures)
            msum += abundances_interp[species_basename]
            continue

        # Non-vertically constant abundances
        pressure_interpolation_nodes = define_pressure_node_list(
            pressures,
            species_basename,
            parameters
        )
        abundance_nodes = define_abundance_node_list(species_basename, parameters)

        # Stepped abundance profile
        if f"{species_basename}_stepped_abundance_profile" in parameters.keys():
            if species_basename in abundances_interp.keys():
                msum -= abundances_interp[species_basename]
            abundances_interp[species_basename] = stepped_profile(
                pressures,
                pressure_interpolation_nodes,
                abundance_nodes)
            msum += abundances_interp[species_basename]

        # Linear spline interpolation
        if f"{species_basename}_linear_abundance_profile" in parameters.keys():
            if species_basename in abundances_interp.keys():
                msum -= abundances_interp[species_basename]
            abundances_interp[species_basename], _ = linear_spline_profile(
                pressures,
                pressure_interpolation_nodes,
                abundance_nodes,
                gamma=0.04,
                nnodes=len(pressure_interpolation_nodes))
            msum += abundances_interp[species_basename]

        # Cubic spline interpolation
        if f"{species_basename}_cubic_abundance_profile" in parameters.keys():
            if species_basename in abundances_interp.keys():
                msum -= abundances_interp[species_basename]
            abundances_interp[species_basename], _ = cubic_spline_profile(
                pressures,
                pressure_interpolation_nodes,
                abundance_nodes,
                gamma=0.04,
                nnodes=len(pressure_interpolation_nodes))
            msum += abundances_interp[species_basename]

    # For free chemistry, need to fill with background gas (H2-He)
    # TODO use arbitrary background gas
    if "Fe/H" not in parameters.keys():
        # Check to make sure we're using free chemistry
        # Whatever's left is H2 and He
        if 'H2' in parameters.keys():
            abundances_interp['H2'] = 10 ** parameters['H2'].value * np.ones_like(pressures)
        else:
            abundances_interp['H2'] = 0.766 * (1.0 - msum)

        if 'He' in parameters.keys():
            abundances_interp['He'] = 10 ** parameters['He'].value * np.ones_like(pressures)
        else:
            abundances_interp['He'] = 0.234 * (1.0 - msum)

        # Imposing strict limit on msum to ensure H2 dominated composition
        if np.max(msum) > 1.0:
            print(f"Abundance sum > 1.0, msum={np.max(msum):.2f}")
            return None, None, None, None

        mmw = compute_mean_molar_masses(abundances_interp)

    # Prior check all input params
    clouds = {}
    p_bases = {}
    abundances = {}

    for cloud in cloud_species:
        cloud_opacity = CloudOpacity([cloud], natural_abundance=False)
        species_full_name = cloud_opacity.species_full_name
        cloud_name = species_full_name.split('_')[0]

        if 'use_easychem' in parameters.keys():
            # AMR CANNOT BE USED WITH EASYCHEM RIGHT NOW
            clouds[cloud_name] = abundances_interp[cloud_name]
            continue

        if "eq_scaling_" + cloud_name in parameters.keys():
            # equilibrium cloud abundance
            x_cloud = fc.return_cloud_mass_fraction(cloud, parameters['Fe/H'].value, parameters['C/O'].value)
            # Scaled by a constant factor
            clouds[cloud_name] = 10 ** parameters['eq_scaling_' + cloud_name].value * x_cloud
        else:
            # Free cloud abundance
            clouds[cloud_name] = 10 ** parameters['log_X_cb_' + cloud_name].value

        # Free cloud bases
        if 'log_Pbase_' + cloud_name in parameters.keys():
            p_bases[cloud_name] = 10 ** parameters['log_Pbase_' + cloud_name].value
        elif 'Pbase_' + cloud_name in parameters.keys():
            p_bases[cloud_name] = parameters['Pbase_' + cloud_name].value
        # Equilibrium locations
        elif 'Fe/H' in parameters.keys():
            p_bases[cloud_name] = fc.simple_cdf(
                cloud_name,
                pressures,
                temperatures,
                parameters['Fe/H'].value,
                parameters['C/O'].value,
                np.mean(mmw)
            )
        else:
            p_bases[cloud_name] = fc.simple_cdf_free(
                cloud_name,
                pressures,
                temperatures,
                10 ** parameters['log_X_cb_' + cloud_name].value,
                mmw[0]
            )

    # Find high resolution pressure grid and indices
    if amr:
        press_use, pressure_indices = fixed_length_amr(
            np.array(list(p_bases.values())),
            pressures,
            parameters['pressure_scaling'].value,
            parameters['pressure_width'].value)
    else:
        pressure_indices = np.array(list(range(pressures.shape[0])))

    fseds = {}

    if 'use_easychem' not in parameters.keys():
        for cloud in copy.copy(cloud_species):
            cloud_opacity = CloudOpacity([cloud], natural_abundance=False)
            species_full_name = cloud_opacity.species_full_name
            cloud_name = species_full_name.split('_')[0]

            # Set up fseds per-cloud
            if 'fsed_' + cloud_name in parameters.keys():
                fseds[cloud_name] = parameters['fsed_' + cloud_name].value
            else:
                fseds[cloud_name] = parameters['fsed'].value

            abundances[species_full_name] = np.zeros_like(temperatures)
            abundances[species_full_name][pressures < p_bases[cloud_name]] = \
                clouds[cloud_name] * (
                        pressures[pressures <= p_bases[cloud_name]] / p_bases[cloud_name]
                ) ** fseds[cloud_name]

            abundances[species_full_name] = abundances[species_full_name][pressure_indices]

    for species in line_species:
        species_opacity = Opacity([species])
        species_full_name = species_opacity.get_full_name().split('.R')[0]
        species_basename = Opacity.get_species_base_name(species)

        # Depending on easychem vs interpolated and different versions of pRT
        # C2H2 is named differently.
        if species_basename == "C2H2":
            # might be',acetylene'
            not_found = True
            for key in abundances_interp.keys():
                if species_basename in key:
                    species_basename = key
                    not_found = False
                    break

            if not_found:
                continue

        if 'FeH' in species_basename:
            # Magic factor for FeH opacity - off by factor of 2
            abunds_change_rainout = copy.copy(abundances_interp[species_basename] / 2.)

            if 'Fe(c)' in p_bases.keys() and 'use_easychem' not in parameters.keys():
                index_ro = pressures < p_bases['Fe(c)']  # Must have iron cloud
                abunds_change_rainout[index_ro] = 0.

            abundances[species_full_name] = abunds_change_rainout[pressure_indices]

        if species_basename in abundances_interp.keys():
            abundances[species_full_name] = abundances_interp[species_basename][pressure_indices]
        else:
            abundances[species_full_name] = abundances_interp[species_full_name][pressure_indices]

    abundances['H2'] = abundances_interp['H2'][pressure_indices]
    abundances['He'] = abundances_interp['He'][pressure_indices]

    return abundances, mmw, pressure_indices, p_bases
