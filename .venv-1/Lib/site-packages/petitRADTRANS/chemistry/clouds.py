"""This file allows the calculation of equilibrium cloud abundances and base pressures"""
# TODO make a better cloud module
# TODO add/replace with Exo-REM condensation curves
import copy
import warnings

import numpy as np
from scipy.interpolate import interp1d
from petitRADTRANS.opacities.opacities import CloudOpacity
from petitRADTRANS.chemistry.prt_molmass import get_species_molar_mass

# metal species
__metals = ['C', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Ti', 'V', 'Fe', 'Ni']

# solar abundances, [Fe/H] = 0, from Asplund+ 2009
__elemental_abundances = {  # TODO replace with chemistry.utils._solar_elemental_abundances
    'H': 0.9207539305,
    'He': 0.0783688694,
    'C': 0.0002478241,
    'N': 6.22506056949881e-05,
    'O': 0.0004509658,
    'Na': 1.60008694353205e-06,
    'Mg': 3.66558742055362e-05,
    'Al': 2.595e-06,
    'Si': 2.9795e-05,
    'P': 2.36670201997668e-07,
    'S': 1.2137900734604e-05,
    'Cl': 2.91167958499589e-07,
    'K': 9.86605611925677e-08,
    'Ca': 2.01439011429255e-06,
    'Ti': 8.20622804366359e-08,
    'V': 7.83688694089992e-09,
    'Fe': 2.91167958499589e-05,
    'Ni': 1.52807116806281e-06
}

# Get molar masses ahead of time to gain speed
__molar_masses = {
    'H': get_species_molar_mass('H'),
    'He': get_species_molar_mass('He'),
    'C': get_species_molar_mass('C'),
    'N': get_species_molar_mass('N'),
    'O': get_species_molar_mass('O'),
    'Na': get_species_molar_mass('Na'),
    'Mg': get_species_molar_mass('Mg'),
    'Al': get_species_molar_mass('Al'),
    'Si': get_species_molar_mass('Si'),
    'P': get_species_molar_mass('P'),
    'S': get_species_molar_mass('S'),
    'Cl': get_species_molar_mass('Cl'),
    'K': get_species_molar_mass('K'),
    'Ca': get_species_molar_mass('Ca'),
    'Ti': get_species_molar_mass('Ti'),
    'V': get_species_molar_mass('V'),
    'Fe': get_species_molar_mass('Fe'),
    'Ni': get_species_molar_mass('Ni')
}


def __get_species_molar_mass(species):
    if species in __molar_masses:
        return __molar_masses[species]
    else:
        return get_species_molar_mass(species)


def setup_clouds(pressures, parameters, cloud_species):
    """
    This function provides the set of cloud parameters used in
    petitRADTRANS. This will be some combination of atmospheric
    parameters (fsed and Kzz), distribution descriptions (log normal or hansen)
    and the cloud particle radius. Fsed and the particle radii can be provided
    on a per-cloud basis.

    Args:
        pressures : np.ndarray
            The pressure array used to provide the atmospheric grid
        parameters : dict
            The dictionary of parameters passed to the model function. Should contain:
                 *  fsed : sedimentation parameter - can be unique to each cloud type
                One of:
                  *  sigma_lnorm : Width of cloud particle size distribution (log normal)
                  *  b_hans : Width of cloud particle size distribution (hansen)
                One of:
                  *  log_cloud_radius_* : Central particle radius (typically computed with fsed and Kzz)
                  *  log_kzz : Vertical mixing parameter
        cloud_species : list
            A list of the names of each of the cloud species used in the atmosphere.

    Returns:
        sigma_lnorm : float, None
            The width of a log normal particle size distribution.
        fseds : dict, None
            The sedimentation fraction for each cloud species in the atmosphere.
        kzz : np.ndarray, None
            The vertical mixing parameter.
        b_hans : float, None
            The width of a hansen particle size distribution.
        radii : dict, None
            The central radius of the particle size distribution.
        distribution : string
            Either "lognormal" or "hansen" - tells pRT which distribution to use.
    """
    sigma_lnorm = None
    b_hans = None
    distribution = "lognormal"
    kzz = None

    # Setup distribution shape
    if "sigma_lnorm" in parameters.keys():
        sigma_lnorm = parameters['sigma_lnorm'].value
    elif "b_hans" in parameters.keys():
        b_hans = get_bhans(parameters, cloud_species, pressures.shape[0])
        distribution = "hansen"

    # Are we retrieving the particle radii?
    radii = {}

    for cloud in cloud_species:
        cloud_opacity = CloudOpacity([cloud], natural_abundance=False)
        species_full_name = cloud_opacity.species_full_name
        cloud_name = species_full_name.split('_')[0]
        if 'log_cloud_radius_' + cloud_name in parameters.keys():
            radii[cloud] = 10 ** parameters['log_cloud_radius_' + cloud_name].value * np.ones_like(pressures)

    if not radii:
        radii = None

    # per-cloud species fseds
    fseds = get_fseds(parameters, cloud_species)

    if "log_kzz" in parameters.keys():
        kzz = 10 ** parameters["log_kzz"].value * np.ones_like(pressures)

    cloud_fraction = 1.0
    complete_coverage_clouds = None

    if 'patchiness' in parameters.keys():
        cloud_fraction = parameters['patchiness'].value

    if 'cloud_fraction' in parameters.keys():
        cloud_fraction = parameters['cloud_fraction'].value

    if 'complete_coverage_clouds' in parameters.keys():
        complete_coverage_clouds = parameters['complete_coverage_clouds'].value

    return sigma_lnorm, fseds, kzz, b_hans, radii, cloud_fraction, complete_coverage_clouds, distribution


def setup_simple_clouds_hazes(parameters):
    """
    Setup clouds for transmission spectrum

    Args:
        parameters (dict): dictionary of atmospheric parameters
    """
    pcloud = None
    power_law_opacity_coefficient = None
    haze_factor = 1.0
    power_law_opacity_350nm = None

    # Grey clouds
    if 'log_Pcloud' in parameters.keys():  # TODO should be log10 to be consistent with SpecModel
        pcloud = 10 ** parameters['log_Pcloud'].value
    elif 'Pcloud' in parameters.keys():
        pcloud = parameters['Pcloud'].value
    elif 'log_opaque_cloud_top_pressure' in parameters.keys():  # TODO should be log10 to be consistent with SpecModel
        pcloud = 10 ** parameters['log_opaque_cloud_top_pressure'].value
    elif 'opaque_cloud_top_pressure' in parameters.keys():
        pcloud = parameters['opaque_cloud_top_pressure'].value

    # Hazes and Rayleigh
    if "power_law_opacity_350nm" in parameters.keys():
        power_law_opacity_350nm = parameters["power_law_opacity_350nm"].value
    if "power_law_opacity_coefficient" in parameters.keys():
        power_law_opacity_coefficient = parameters["power_law_opacity_coefficient"].value
    if "log_haze_factor" in parameters.keys():  # TODO should be log10 to be consistent with SpectralModel
        # TODO Document. Should be taken care of in prior.
        haze_factor = 10 ** parameters["log_haze_factor"].value
    elif "haze_factor" in parameters.keys():
        haze_factor = parameters["haze_factor"].value

    # Compatibility
    if "gamma_scat" in parameters.keys():
        # TODO deprecated naming
        power_law_opacity_coefficient = parameters["gamma_scat"].value
    if "kappa_0" in parameters.keys():
        # TODO deprecated naming
        power_law_opacity_350nm = 10 ** parameters["kappa_0"].value
    return pcloud, power_law_opacity_coefficient, haze_factor, power_law_opacity_350nm


def cloud_dict(parameters, parameter_name, cloud_species, shape=0):
    """
    This is a generic method to create a dictionary of
    parameters values for a given cloud parameterization, testing if
    the parameter should be filled on a per-species basis or
    if each cloud species should have the same value.
    """
    output_dictionary = {}

    for cloud in cloud_species:
        cloud_opacity = CloudOpacity([cloud], natural_abundance=False)
        species_full_name = cloud_opacity.species_full_name
        cloud_name = species_full_name.split('_')[0]
        output = None

        if parameter_name + "_" + cloud_name in parameters.keys():
            output = parameters[parameter_name + "_" + cloud_name].value
        elif parameter_name in parameters.keys():
            output = parameters[parameter_name].value

        if shape > 0:
            output = output * np.ones(shape)

        output_dictionary[cloud] = output

    if not output_dictionary:
        output_dictionary = None

    return output_dictionary


def get_fseds(parameters, cloud_species):
    """
    This function checks to see if the fsed values are input on a per-cloud basis
    or only as a single value, and returns the dictionary providing the fsed values
    for each cloud, or None, if no cloud is used.
    """
    return cloud_dict(parameters, "fsed", cloud_species)


def get_bhans(parameters, cloud_species, shape=0):
    """
    This function checks to see if the bhans values are input on a per-cloud basis
    or only as a single value, and returns the dictionary providing the fsed values
    for each cloud, or None, if no cloud is used.
    """
    return cloud_dict(parameters, "b_hans", cloud_species, shape=shape)


def return_cloud_mass_fraction(name, metallicity, co_ratio):
    if "Fe(s)" in name or "Fe(l)" in name:
        return return_x_fe(metallicity, co_ratio)
    elif "MgSiO3(s)" in name or "MgSiO3(l)" in name:
        return return_x_mgsio3(metallicity, co_ratio)
    elif "Mg2SiO4(s)" in name or "Mg2SiO4(l)" in name:
        return return_x_mg2sio4(metallicity, co_ratio)
    elif "Na2S(s)" in name or "Na2S(l)" in name:
        return return_x_na2s(metallicity, co_ratio)
    elif "KCL(s)" in name or "KCL(l)" in name:
        return return_x_kcl(metallicity, co_ratio)
    elif "MgFeSiO4(s)" in name:
        return return_x_mgfesio4(metallicity, co_ratio)
    elif "SiO(s)" in name:
        return return_x_sio(metallicity, co_ratio)
    else:
        warnings.warn(f"The cloud {name} is not currently implemented.")
        return np.zeros_like(metallicity)


def simple_cdf(name, press, temp, metallicity, co_ratio, mmw=2.33):
    if "Fe(s)" in name or "Fe(l)" in name:
        return simple_cdf_fe(press, temp, metallicity, co_ratio, mmw)
    elif "MgSiO3(s)" in name or "MgSiO3(l)" in name:
        return simple_cdf_mgsio3(press, temp, metallicity, co_ratio, mmw)
    elif "Mg2SiO4(s)" in name or "Mg2SiO4(l)" in name:
        return simple_cdf_mg2sio4(press, temp, metallicity, co_ratio, mmw)
    elif "Na2S(s)" in name or "Na2S(l)" in name:
        return simple_cdf_na2s(press, temp, metallicity, co_ratio, mmw)
    elif "KCL(s)" in name or "KCL(l)" in name:
        return simple_cdf_kcl(press, temp, metallicity, co_ratio, mmw)
    elif "SiO(s)":
        return simple_cdf_sio(press, temp, metallicity, co_ratio, mmw)
    else:
        warnings.warn(f"The cloud {name} is not currently implemented.")

        return np.zeros_like(metallicity)


def simple_cdf_free(name, press, temp, metallicity, mfrac, mmw=2.33):
    if "Fe(s)" in name or "Fe(l)" in name:
        return simple_cdf_fe_free(press, temp, mfrac, mmw)
    elif "MgSiO3(s)" in name or "MgSiO3(l)" in name:
        return simple_cdf_mgsio3_free(press, temp, mfrac, mmw)
    elif "Mg2SiO4(s)" in name or "Mg2SiO4(l)" in name:
        return simple_cdf_mg2sio4_free(press, temp, mfrac, mmw)
    elif "Na2S(s)" in name or "Na2S(l)" in name:
        return simple_cdf_na2s_free(press, temp, mfrac, mmw)
    elif "KCL(s)" in name or "KCL(l)" in name:
        return simple_cdf_kcl_free(press, temp, mfrac, mmw)
    elif "SiO(s)" in name:
        return simple_cdf_sio_free(press, temp, mfrac, mmw)
    else:
        warnings.warn(f"The cloud {name} is not currently implemented.")

        return np.zeros_like(metallicity)


def return_x_fe(metallicity, co_ratio):
    nfracs_use = copy.copy(__elemental_abundances)

    for spec in __elemental_abundances.keys():
        if (spec != 'H') and (spec != 'He'):
            nfracs_use[spec] = __elemental_abundances[spec] * 1e1 ** metallicity

    nfracs_use['O'] = nfracs_use['C'] / co_ratio

    x_fe = __get_species_molar_mass('Fe') * nfracs_use['Fe']
    add = 0.

    for spec in nfracs_use.keys():
        add += __get_species_molar_mass(spec) * nfracs_use[spec]

    x_fe = x_fe / add

    return x_fe


def return_x_mgsio3(metallicity, co_ratio):
    nfracs_use = copy.copy(__elemental_abundances)

    for spec in __elemental_abundances.keys():
        if (spec != 'H') and (spec != 'He'):
            nfracs_use[spec] = __elemental_abundances[spec] * 1e1 ** metallicity

    nfracs_use['O'] = nfracs_use['C'] / co_ratio

    nfracs_mgsio3 = np.min([nfracs_use['Mg'],
                            nfracs_use['Si'],
                            nfracs_use['O'] / 3.])
    masses_mgsio3 = __get_species_molar_mass('Mg') \
        + __get_species_molar_mass('Si') \
        + 3. * __get_species_molar_mass('O')

    xmgsio3 = masses_mgsio3 * nfracs_mgsio3
    add = 0.

    for spec in nfracs_use.keys():
        add += __get_species_molar_mass(spec) * nfracs_use[spec]

    xmgsio3 = xmgsio3 / add

    return xmgsio3


def return_x_mg2sio4(metallicity, co_ratio):
    nfracs_use = copy.copy(__elemental_abundances)

    for spec in __elemental_abundances.keys():
        if (spec != 'H') and (spec != 'He'):
            nfracs_use[spec] = __elemental_abundances[spec] * 1e1 ** metallicity

    nfracs_use['O'] = nfracs_use['C'] / co_ratio

    nfracs_mg2sio4 = np.min([
        nfracs_use['Mg'] / 2.,
        nfracs_use['Si'],
        nfracs_use['O'] / 4.]
    )
    masses_mg2sio4 = 2 * __get_species_molar_mass('Mg') \
        + __get_species_molar_mass('Si') \
        + 4. * __get_species_molar_mass('O')

    x_mg2sio4 = masses_mg2sio4 * nfracs_mg2sio4

    add = 0.

    for spec in nfracs_use.keys():
        add += __get_species_molar_mass(spec) * nfracs_use[spec]

    x_mg2sio4 = x_mg2sio4 / add

    return x_mg2sio4


def return_x_mgfesio4(metallicity, co_ratio):
    nfracs_use = copy.copy(__elemental_abundances)

    for spec in __elemental_abundances.keys():
        if (spec != 'H') and (spec != 'He'):
            nfracs_use[spec] = __elemental_abundances[spec] * 1e1 ** metallicity

    nfracs_use['O'] = nfracs_use['C'] / co_ratio

    nfracs_mgfesio4 = np.min([
        nfracs_use['Mg'],
        nfracs_use['Fe'],
        nfracs_use['Si'],
        nfracs_use['O'] / 4.]
    )
    masses_mgfesio4 = __get_species_molar_mass('Mg') \
        + __get_species_molar_mass('Fe') \
        + __get_species_molar_mass('Si') \
        + 4. * __get_species_molar_mass('O')

    x_mgfesio4 = masses_mgfesio4 * nfracs_mgfesio4

    add = 0.

    for spec in nfracs_use.keys():
        add += __get_species_molar_mass(spec) * nfracs_use[spec]

    x_mgfesio4 = x_mgfesio4 / add

    return x_mgfesio4


def return_x_sio(metallicity, co_ratio):
    nfracs_use = copy.copy(__elemental_abundances)

    for spec in __elemental_abundances.keys():
        if (spec != 'H') and (spec != 'He'):
            nfracs_use[spec] = __elemental_abundances[spec] * 1e1 ** metallicity

    nfracs_use['O'] = nfracs_use['C'] / co_ratio

    nfracs_sio = np.min([
        nfracs_use['Si'],
        nfracs_use['O']]
    )
    masses_sio = __get_species_molar_mass('Si') \
        + __get_species_molar_mass('O')

    x_sio = masses_sio * nfracs_sio

    add = 0.

    for spec in nfracs_use.keys():
        add += __get_species_molar_mass(spec) * nfracs_use[spec]

    x_sio = x_sio / add

    return x_sio


def return_x_na2s(metallicity, co_ratio):
    nfracs_use = copy.copy(__elemental_abundances)

    for spec in __elemental_abundances.keys():
        if (spec != 'H') and (spec != 'He'):
            nfracs_use[spec] = __elemental_abundances[spec] * 1e1 ** metallicity

    nfracs_use['O'] = nfracs_use['C'] / co_ratio

    nfracs_na2s = np.min([nfracs_use['Na'] / 2.,
                          nfracs_use['S']])
    masses_na2s = 2. * __get_species_molar_mass('Na') \
        + __get_species_molar_mass('S')

    xna2s = masses_na2s * nfracs_na2s
    add = 0.

    for spec in nfracs_use.keys():
        add += __get_species_molar_mass(spec) * nfracs_use[spec]

    xna2s = xna2s / add

    return xna2s


def return_x_kcl(metallicity, co_ratio):
    nfracs_use = copy.copy(__elemental_abundances)

    for spec in __elemental_abundances.keys():
        if (spec != 'H') and (spec != 'He'):
            nfracs_use[spec] = __elemental_abundances[spec] * 1e1 ** metallicity

    nfracs_use['O'] = nfracs_use['C'] / co_ratio

    nfracs_kcl = np.min([nfracs_use['K'],
                         nfracs_use['Cl']])
    masses_kcl = __get_species_molar_mass('K') \
        + __get_species_molar_mass('Cl')

    xkcl = masses_kcl * nfracs_kcl
    add = 0.

    for spec in nfracs_use.keys():
        add += __get_species_molar_mass(spec) * nfracs_use[spec]

    xkcl = xkcl / add

    return xkcl


#############################################################
# Fe saturation pressure, from Ackerman & Marley (2001), including erratum (P_vap is in bar, not cgs!)
#############################################################

def return_t_cond_fe(metallicity, co_ratio, mmw=2.33):
    t = np.linspace(100., 10000., 1000)

    # Taken from Ackerman & Marley (2001) including their erratum
    def p_vap(x):
        return np.exp(15.71 - 47664. / x)

    x_fe = return_x_fe(metallicity, co_ratio)

    return p_vap(t) / (x_fe * mmw / __get_species_molar_mass('Fe')), t


def return_t_cond_fe_l(metallicity, co_ratio, mmw=2.33):
    t = np.linspace(100., 10000., 1000)

    # Taken from Ackerman & Marley (2001) including their erratum
    def p_vap(x):
        return np.exp(9.86 - 37120. / x)

    x_fe = return_x_fe(metallicity, co_ratio)

    return p_vap(t) / (x_fe * mmw / __get_species_molar_mass('Fe')), t


def return_t_cond_fe_comb(metallicity, co_ratio, mmw=2.33):
    p1, t1 = return_t_cond_fe(metallicity, co_ratio, mmw)
    p2, t2 = return_t_cond_fe_l(metallicity, co_ratio, mmw)

    ret_p = np.zeros_like(p1)
    index = p1 < p2
    ret_p[index] = p1[index]
    ret_p[~index] = p2[~index]

    return ret_p, t2


def return_t_cond_fe_free(x_fe, mmw=2.33):
    t = np.linspace(100., 12000., 1200)

    # Taken from Ackerman & Marley (2001) including their erratum
    def p_vap(x):
        return np.exp(15.71 - 47664. / x)

    return p_vap(t) / (x_fe * mmw / __get_species_molar_mass('Fe')), t


def return_t_cond_fe_l_free(x_fe, mmw=2.33):
    t = np.linspace(100., 12000., 1200)

    # Taken from Ackerman & Marley (2001) including their erratum
    def p_vap(x):
        return np.exp(9.86 - 37120. / x)

    return p_vap(t) / (x_fe * mmw / __get_species_molar_mass('Fe')), t


def return_t_cond_fe_comb_free(x_fe, mmw=2.33):
    p1, t1 = return_t_cond_fe_free(x_fe, mmw)
    p2, t2 = return_t_cond_fe_l_free(x_fe, mmw)
    ret_p = np.zeros_like(p1)
    index = p1 < p2
    ret_p[index] = p1[index]
    ret_p[~index] = p2[~index]

    return ret_p, t2


def return_t_cond_mgsio3(metallicity, co_ratio, mmw=2.33):
    t = np.linspace(100., 10000., 1000)

    # Taken from Ackerman & Marley (2001) including their erratum
    def p_vap(x):
        return np.exp(25.37 - 58663. / x)

    xmgsio3 = return_x_mgsio3(metallicity, co_ratio)

    m_mgsio3 = __get_species_molar_mass('Mg') \
        + __get_species_molar_mass('Si') \
        + 3. * __get_species_molar_mass('O')

    return p_vap(t) / (xmgsio3 * mmw / m_mgsio3), t


def return_t_cond_mg2sio4(metallicity, co_ratio, mmw=2.33):
    t = np.linspace(100., 10000., 1000)

    # Visscher 2010 condensation curve
    def p_vap(x):
        """Total pressure of condensation, not the species' vapor pressure like for most of the other species."""
        return 10 ** (15.92 - 1.97 * metallicity - 27027.03 / x)

    return p_vap(t), t


def return_t_cond_mgsio3_free(x_mgsio3, mmw=2.33):
    t = np.linspace(100., 10000., 1000)

    # Taken from Ackerman & Marley (2001) including their erratum

    def p_vap(x):
        return np.exp(25.37 - 58663. / x)

    m_mgsio3 = __get_species_molar_mass('Mg') \
        + __get_species_molar_mass('Si') \
        + 3. * __get_species_molar_mass('O')

    return p_vap(t) / (x_mgsio3 * mmw / m_mgsio3), t


def return_t_cond_mg2sio4_free(x_mg2sio4, mmw=2.33):
    t = np.linspace(100., 10000., 1000)

    # Taken from Ackerman & Marley (2001) including their erratum
    def p_vap(x):
        return np.exp(25.37 - 58663. / x)

    m_mg2sio4 = 2 * __get_species_molar_mass('Mg') \
        + __get_species_molar_mass('Si') \
        + 4. * __get_species_molar_mass('O')

    return p_vap(t) / (x_mg2sio4 * mmw / m_mg2sio4), t


def return_t_cond_na2s(metallicity, co_ratio, mmw=2.33):
    # Taken from Charnay+2018
    t = np.linspace(100., 10000., 1000)

    # This is the partial pressure of Na, so /2 to get the partial pressure of the hypothetical Na2S gas particles
    # This is OK: there are more S than Na atoms at solar abundance ratios
    def p_vap(x):
        return 1e1 ** (8.55 - 13889. / x - 0.5 * metallicity) / 2.

    xna2s = return_x_na2s(metallicity, co_ratio)

    m_na2s = 2. * __get_species_molar_mass('Na') \
        + __get_species_molar_mass('S')

    return p_vap(t) / (xna2s * mmw / m_na2s), t


def return_t_cond_na2s_free(x_na2s, mmw=2.33):
    # Taken from Charnay+2018
    t = np.linspace(100., 10000., 1000)

    # This is the partial pressure of Na, so /2 to get the partial pressure of the hypothetical Na2S gas particles
    # This is OK: there are more S than Na atoms at solar abundance ratios

    # We're also using [Na/H] as a proxy for [Fe/H]
    # Definitely not strictly correct, but should be good enough for ~ solar compositions. [+- 1 for Fe/H]
    # Assumes constant vertical abundance
    def p_vap(x):
        return 1e1 ** (8.55 - 13889. / x - 0.5 * (np.log10(2 * x_na2s * mmw / m_na2s) + 5.7)) / 2

    m_na2s = 2. * __get_species_molar_mass('Na') \
        + __get_species_molar_mass('S')

    return p_vap(t) / (x_na2s * mmw / m_na2s), t


def return_t_cond_kcl(metallicity, co_ratio, mmw=2.33):
    # Taken from Charnay+2018
    t = np.linspace(100., 10000., 1000)

    def p_vap(x):
        return 1e1 ** (7.611 - 11382. / x)  # TODO check if this p_vap is alright

    xkcl = return_x_kcl(metallicity, co_ratio)

    m_kcl = __get_species_molar_mass('K') \
        + __get_species_molar_mass('Cl')

    return p_vap(t) / (xkcl * mmw / m_kcl), t


def return_t_cond_kcl_free(x_kcl, mmw=2.33):
    # Taken from Charnay+2018
    t = np.linspace(100., 10000., 1000)

    def p_vap(x):
        return 1e1 ** (7.611 - 11382. / x)  # TODO check if this p_vap is alright

    m_kcl = __get_species_molar_mass('K') \
        + __get_species_molar_mass('Cl')

    return p_vap(t) / (x_kcl * mmw / m_kcl), t


def return_t_cond_sio(metallicity, co_ratio, mmw=2.33):
    # Taken from Wetzel+2013
    t = np.linspace(100., 10000., 1000)

    def p_vap(x):                        # conversion to bar
        return np.exp(33.848 - 42648./x)*1e-6

    xsio = return_x_sio(metallicity, co_ratio)

    m_sio = __get_species_molar_mass('Si') \
        + __get_species_molar_mass('O')

    return p_vap(t) / (xsio * mmw / m_sio), t


def return_t_cond_sio_free(x_sio, mmw=2.33):
    # Taken from Wetzel+2013
    t = np.linspace(100., 10000., 1000)

    def p_vap(x):                        # conversion to bar
        return np.exp(33.848 - 42648./x)*1e-6

    m_sio = __get_species_molar_mass('Si') \
        + __get_species_molar_mass('O')

    return p_vap(t) / (x_sio * mmw / m_sio), t


def simple_cdf_fe(press, temp, metallicity, co_ratio, mmw=2.33):
    pc, tc = return_t_cond_fe_comb(metallicity, co_ratio, mmw)
    index = (pc > 1e-8) & (pc < 1e5)
    pc, tc = pc[index], tc[index]
    tcond_p = interp1d(pc, tc)
    tcond_on_input_grid = tcond_p(press)

    tdiff = tcond_on_input_grid - temp
    diff_vec = tdiff[1:] * tdiff[:-1]
    ind_cdf = (diff_vec < 0.)

    if len(diff_vec[ind_cdf]) > 0:
        p_clouds = (press[1:] + press[:-1])[ind_cdf] / 2.
        p_cloud = p_clouds[-1]
    else:
        p_cloud = np.min(press)

    return p_cloud


def simple_cdf_fe_free(press, temp, x_fe, mmw=2.33):
    pc, tc = return_t_cond_fe_comb_free(x_fe, mmw)
    index = (pc > 1e-8) & (pc < 1e5)
    pc, tc = pc[index], tc[index]
    tcond_p = interp1d(pc, tc)

    try:  # TODO replace try/except with proper conditional handling of potential error
        tcond_on_input_grid = tcond_p(press)
    except ValueError:
        print(pc)
        return np.min(press)

    t_diff = tcond_on_input_grid - temp
    diff_vec = t_diff[1:] * t_diff[:-1]
    ind_cdf = (diff_vec < 0.)

    if len(diff_vec[ind_cdf]) > 0:
        p_clouds = (press[1:] + press[:-1])[ind_cdf] / 2.
        p_cloud = p_clouds[-1]
    else:
        p_cloud = np.min(press)

    return p_cloud


def simple_cdf_mgsio3(press, temp, metallicity, co_ratio, mmw=2.33):
    pc, tc = return_t_cond_mgsio3(metallicity, co_ratio, mmw)
    index = (pc > 1e-8) & (pc < 1e5)
    pc, tc = pc[index], tc[index]

    try:  # TODO replace try/except with proper conditional handling of potential error
        tcond_p = interp1d(pc, tc)
    except ValueError:
        warnings.warn("Could not interpolate pressures and temperatures!")
        return np.min(press)

    tcond_on_input_grid = tcond_p(press)

    tdiff = tcond_on_input_grid - temp
    diff_vec = tdiff[1:] * tdiff[:-1]
    ind_cdf = (diff_vec < 0.)

    if len(diff_vec[ind_cdf]) > 0:
        p_clouds = (press[1:] + press[:-1])[ind_cdf] / 2.
        p_cloud = p_clouds[-1]
    else:
        p_cloud = np.min(press)

    return p_cloud


def simple_cdf_mgsio3_free(press, temp, x_mgsio3, mmw=2.33):
    pc, tc = return_t_cond_mgsio3_free(x_mgsio3, mmw)
    index = (pc > 1e-8) & (pc < 1e5)
    pc, tc = pc[index], tc[index]
    tcond_p = interp1d(pc, tc)
    tcond_on_input_grid = tcond_p(press)

    tdiff = tcond_on_input_grid - temp
    diff_vec = tdiff[1:] * tdiff[:-1]
    ind_cdf = (diff_vec < 0.)

    if len(diff_vec[ind_cdf]) > 0:
        p_clouds = (press[1:] + press[:-1])[ind_cdf] / 2.
        p_cloud = p_clouds[-1]
    else:
        p_cloud = np.min(press)

    return p_cloud


def simple_cdf_mg2sio4(press, temp, metallicity, co_ratio, mmw=2.33):
    p_c, t_c = return_t_cond_mg2sio4(metallicity, co_ratio, mmw)
    index = (p_c > 1e-8) & (p_c < 1e5)
    p_c, t_c = p_c[index], t_c[index]

    try:  # TODO replace try/except with proper conditional handling of potential error
        tcond_p = interp1d(p_c, t_c)
    except ValueError:
        warnings.warn("Could not interpolate pressures and temperatures!")
        return np.min(press)

    t_cond_on_input_grid = tcond_p(press)

    t_diff = t_cond_on_input_grid - temp
    diff_vec = t_diff[1:] * t_diff[:-1]
    ind_cdf = (diff_vec < 0.)

    if len(diff_vec[ind_cdf]) > 0:
        p_clouds = (press[1:] + press[:-1])[ind_cdf] / 2.
        p_cloud = p_clouds[-1]
    else:
        p_cloud = np.min(press)

    return p_cloud


def simple_cdf_mg2sio4_free(press, temp, x_mg2sio4, mmw=2.33):
    p_c, t_c = return_t_cond_mg2sio4_free(x_mg2sio4, mmw)
    index = (p_c > 1e-8) & (p_c < 1e5)
    p_c, t_c = p_c[index], t_c[index]
    tcond_p = interp1d(p_c, t_c)
    t_cond_on_input_grid = tcond_p(press)

    t_diff = t_cond_on_input_grid - temp
    diff_vec = t_diff[1:] * t_diff[:-1]
    ind_cdf = (diff_vec < 0.)

    if len(diff_vec[ind_cdf]) > 0:
        p_clouds = (press[1:] + press[:-1])[ind_cdf] / 2.
        p_cloud = p_clouds[-1]
    else:
        p_cloud = np.min(press)

    return p_cloud


def simple_cdf_na2s(press, temp, metallicity, co_ratio, mmw=2.33):
    pc, tc = return_t_cond_na2s(metallicity, co_ratio, mmw)
    index = (pc > 1e-8) & (pc < 1e5)
    pc, tc = pc[index], tc[index]
    tcond_p = interp1d(pc, tc)
    tcond_on_input_grid = tcond_p(press)

    tdiff = tcond_on_input_grid - temp
    diff_vec = tdiff[1:] * tdiff[:-1]
    ind_cdf = (diff_vec < 0.)

    if len(diff_vec[ind_cdf]) > 0:
        p_clouds = (press[1:] + press[:-1])[ind_cdf] / 2.
        p_cloud = p_clouds[-1]
    else:
        p_cloud = np.min(press)

    return p_cloud


def simple_cdf_na2s_free(press, temp, x_na2s, mmw=2.33):
    pc, tc = return_t_cond_na2s_free(x_na2s, mmw)
    index = (pc > 1e-8) & (pc < 1e5)

    pc, tc = pc[index], tc[index]
    tcond_p = interp1d(pc, tc)
    tcond_on_input_grid = tcond_p(press)

    tdiff = tcond_on_input_grid - temp
    diff_vec = tdiff[1:] * tdiff[:-1]
    ind_cdf = (diff_vec < 0.)

    if len(diff_vec[ind_cdf]) > 0:
        p_clouds = (press[1:] + press[:-1])[ind_cdf] / 2.
        p_cloud = p_clouds[-1]
    else:
        p_cloud = np.min(press)

    return p_cloud


def simple_cdf_kcl(press, temp, metallicity, co_ratio, mmw=2.33):
    pc, tc = return_t_cond_kcl(metallicity, co_ratio, mmw)
    index = (pc > 1e-8) & (pc < 1e5)
    pc, tc = pc[index], tc[index]
    tcond_p = interp1d(pc, tc)
    tcond_on_input_grid = tcond_p(press)

    tdiff = tcond_on_input_grid - temp
    diff_vec = tdiff[1:] * tdiff[:-1]
    ind_cdf = (diff_vec < 0.)

    if len(diff_vec[ind_cdf]) > 0:
        p_clouds = (press[1:] + press[:-1])[ind_cdf] / 2.
        p_cloud = p_clouds[-1]
    else:
        p_cloud = np.min(press)

    return p_cloud


def simple_cdf_kcl_free(press, temp, x_kcl, mmw=2.33):
    pc, tc = return_t_cond_kcl_free(x_kcl, mmw)
    index = (pc > 1e-8) & (pc < 1e5)
    pc, tc = pc[index], tc[index]
    tcond_p = interp1d(pc, tc)

    tcond_on_input_grid = tcond_p(press)

    tdiff = tcond_on_input_grid - temp
    diff_vec = tdiff[1:] * tdiff[:-1]
    ind_cdf = (diff_vec < 0.)

    if len(diff_vec[ind_cdf]) > 0:
        p_clouds = (press[1:] + press[:-1])[ind_cdf] / 2.
        p_cloud = p_clouds[-1]
    else:
        p_cloud = np.min(press)

    return p_cloud


def simple_cdf_sio(press, temp, metallicity, co_ratio, mmw=2.33):
    pc, tc = return_t_cond_sio(metallicity, co_ratio, mmw)
    index = (pc > 1e-8) & (pc < 1e5)
    pc, tc = pc[index], tc[index]
    tcond_p = interp1d(pc, tc)
    tcond_on_input_grid = tcond_p(press)

    tdiff = tcond_on_input_grid - temp
    diff_vec = tdiff[1:] * tdiff[:-1]
    ind_cdf = (diff_vec < 0.)

    if len(diff_vec[ind_cdf]) > 0:
        p_clouds = (press[1:] + press[:-1])[ind_cdf] / 2.
        p_cloud = p_clouds[-1]
    else:
        p_cloud = np.min(press)

    return p_cloud


def simple_cdf_sio_free(press, temp, x_sio, mmw=2.33):
    pc, tc = return_t_cond_sio_free(x_sio, mmw)
    index = (pc > 1e-8) & (pc < 1e5)
    pc, tc = pc[index], tc[index]
    tcond_p = interp1d(pc, tc)

    tcond_on_input_grid = tcond_p(press)

    tdiff = tcond_on_input_grid - temp
    diff_vec = tdiff[1:] * tdiff[:-1]
    ind_cdf = (diff_vec < 0.)

    if len(diff_vec[ind_cdf]) > 0:
        p_clouds = (press[1:] + press[:-1])[ind_cdf] / 2.
        p_cloud = p_clouds[-1]
    else:
        p_cloud = np.min(press)

    return p_cloud
