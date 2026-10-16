import easychem as ec
import numpy as np


def get_easychem_abundances(pressures, temperatures, line_species, cloud_species, parameters, AMR=False):
    metallicity = None
    carbon_to_oxygen = None

    if "Fe/H" in parameters.keys():
        metallicity = parameters["Fe/H"].value
    if "C/O" in parameters.keys():
        carbon_to_oxygen = parameters["C/O"].value

    default_remove_condensates = ['Mg2SiO4(c)',
                                  'Mg2SiO4(L)',
                                  'MgAl2O4(c)',
                                  'FeO(c)',
                                  'Fe2SiO4(c)',
                                  'TiO2(c)',
                                  'TiO2(L)',
                                  'H3PO4(c)',
                                  'H3PO4(L)',
                                  'H2O(L)',
                                  'H2O(c)', ]

    exo = ec.ExoAtmos(atoms=None,
                      reactants=None,
                      atomAbunds=None,
                      thermofpath=None,
                      feh=metallicity,
                      co=carbon_to_oxygen)

    reactants = exo.reactants.copy()

    for condensate in default_remove_condensates:
        reactants = np.delete(reactants, np.argwhere(reactants == condensate))
    exo.updateReactants(reactants)

    exo.feh = metallicity
    exo._updateFEH()

    # Set atomic abundances from parameters
    for key, val in parameters.items():
        if key not in exo.atoms:
            continue
        itemindex = np.where(exo.atoms == key)
        exo.atomAbunds[itemindex] = 10 ** val.value

    exo.solve(pressures, temperatures)
    abundances = exo.result_mass()
    pressures_goal = pressures.reshape(-1)
    if 'log_pquench' in parameters.keys():
        p_quench_carbon = 10 ** parameters['log_pquench'].value

        if p_quench_carbon > np.min(pressures_goal):
            q_index = min(np.searchsorted(pressures_goal, p_quench_carbon),
                          int(len(pressures_goal)) - 1)

            methane_abb = abundances['CH4']
            methane_abb[pressures_goal < p_quench_carbon] = \
                abundances['CH4'][q_index]
            abundances['CH4'] = methane_abb

            co_abb = abundances['CO']
            co_abb[pressures_goal < p_quench_carbon] = \
                abundances['CO'][q_index]
            abundances['CO'] = co_abb

            h2o_abb = abundances['H2O']
            h2o_abb[pressures_goal < p_quench_carbon] = \
                abundances['H2O'][q_index]
            abundances['H2O'] = h2o_abb

    abundances["MMW"] = exo.mmw

    return abundances


def update_atom_abundances(atom_abundances_list, atom_name_list, parameters):
    modif_abundances = atom_abundances_list.copy()
    for i_spec, name in enumerate(atom_name_list):
        if name in parameters.keys():
            modif_abundances[i_spec] = 10 ** parameters[name].value
    return modif_abundances


def get_exoatmos_abundances(pressures, temperatures, parameters, AMR=False):
    exo = parameters['exoatmos'].value
    metallicity = None

    if "Fe/H" in parameters.keys():
        metallicity = parameters["Fe/H"].value
        exo.Z = metallicity

    # Set atomic abundances from parameters
    atom_abundances = exo._atomAbunds.copy()
    update_abunds = update_atom_abundances(atom_abundances, exo.atoms, parameters)
    exo.updateAtomAbunds(update_abunds)
    exo.solve(pressures, temperatures)
    abundances = exo.result_mass()

    pressures_goal = pressures.reshape(-1)
    if 'log_pquench' in parameters.keys():
        p_quench_carbon = 10 ** parameters['log_pquench'].value

        if p_quench_carbon > np.min(pressures_goal):
            q_index = min(np.searchsorted(pressures_goal, p_quench_carbon),
                          int(len(pressures_goal)) - 1)

            methane_abb = abundances['CH4']
            methane_abb[pressures_goal < p_quench_carbon] = \
                abundances['CH4'][q_index]
            abundances['CH4'] = methane_abb

            co_abb = abundances['CO']
            co_abb[pressures_goal < p_quench_carbon] = \
                abundances['CO'][q_index]
            abundances['CO'] = co_abb

            h2o_abb = abundances['H2O']
            h2o_abb[pressures_goal < p_quench_carbon] = \
                abundances['H2O'][q_index]
            abundances['H2O'] = h2o_abb

    abundances["MMW"] = exo.mmw

    return abundances


def setup_exoatmos():
    exo = ec.ExoAtmos()
    # In this version, we initialize the abundances to solar
    # and fit in terms of [C/H], [O/H], and [M/H] where M is everything besides C, O, H, and He.
    exo.Z = 0.0
    exo.co = 0.55

    reactants = exo.reactants.copy()
    default_remove_condensates = ['Mg2SiO4(c)',
                                  'TiO(c)',
                                  'TiO(L)',
                                  'MgAl2O4(c)',
                                  'FeO(c)',
                                  'Fe2SiO4(c)',
                                  'TiO2(c)',
                                  'TiO2(L)',
                                  'H3PO4(c)',
                                  'H3PO4(L)']

    for condensate in default_remove_condensates:
        reactants = np.delete(reactants, np.argwhere(reactants == condensate))
    exo.updateReactants(reactants)

    # add FeH
    # reactants = np.append(reactants,'FeH')
    exo.updateReactants(reactants)
    return exo
