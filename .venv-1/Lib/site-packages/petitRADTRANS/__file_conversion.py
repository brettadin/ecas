"""Stores functions that convert files from a format to another.

The functions in this module are stored for the sake of keeping trace of changes made to files. They are intended to be
used only once.
"""
import copy
import datetime
import glob
import os
import shutil
import warnings

import h5py
import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d

import petitRADTRANS
# noinspection PyUnresolvedReferences
from petitRADTRANS.fortran_inputs import fortran_inputs as finput
import petitRADTRANS.physical_constants as cst
from petitRADTRANS.chemistry.prt_molmass import get_species_molar_mass
from petitRADTRANS.config.configuration import get_input_data_subpaths, petitradtrans_config_parser
from petitRADTRANS.math import prt_resolving_space
from petitRADTRANS.opacities.opacities import CIAOpacity, CloudOpacity, CorrelatedKOpacity, LineByLineOpacity, Opacity
from petitRADTRANS.utils import LockedDict

# MPI Multiprocessing
prt_emcee_mode = os.environ.get("pRT_emcee_mode")
load_mpi = True

if prt_emcee_mode == 'True':
    load_mpi = False

MPI = None
rank = 0
comm = None

if load_mpi:
    # MPI Multiprocessing
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    except ImportError:
        MPI = None
        comm = None
        rank = 0


def __get_prt2_input_data_subpaths() -> dict[str, str]:
    old_input_data_subpaths = LockedDict()
    old_input_data_subpaths.update(get_input_data_subpaths())
    old_input_data_subpaths.lock()
    old_input_data_subpaths.update(
        {
            "cia_opacities": os.path.join("opacities", "continuum", "CIA"),
            "clouds_opacities": os.path.join("opacities", "continuum", "clouds"),
            "correlated_k_opacities": os.path.join("opacities", "lines", "corr_k"),
            "line_by_line_opacities": os.path.join("opacities", "lines", "line_by_line"),
            "planet_data": "planet_data",
            "pre_calculated_chemistry": "abundance_files",
            "stellar_spectra": "stellar_specs"
        }
    )

    return old_input_data_subpaths


def __remove_files(old_files):
    if isinstance(old_files, str):
        old_files = [old_files]

    for old_file in old_files:
        if os.path.isfile(old_file):
            print(f" Removing old file '{old_file}'...")
            os.remove(old_file)
        elif os.path.isdir(old_file):
            print(f" Removing old directory '{old_file}'...")
            shutil.rmtree(old_file)
        else:
            print(f" No such file or directory '{old_file}', it probably was already removed")


def _clean_input_data_mac_junk_files(path_input_data=petitradtrans_config_parser.get_input_data_path()):
    print("Removing Mac junk files...")
    mac_junk_files = []

    for (root, directories, files) in os.walk(path_input_data):
        if root.rsplit(os.path.sep, 1)[1] != 'PaxHeader':
            for directory in directories:
                if directory == 'PaxHeader':
                    mac_junk_files.append(os.path.join(root, directory))

            for file in files:
                if file[:2] == '._' or file == '.DS_Store':
                    mac_junk_files.append(os.path.join(root, file))

    __remove_files(mac_junk_files)
    print(f"Successfully removed {len(mac_junk_files)} Mac junk files.")


def __print_missing_data_file_message(obj, object_name, directory):
    print(f"Data for {obj} '{object_name}' not found (path '{directory}' does not exist), skipping...")


def __print_skipping_message(hdf5_opacity_file):
    print(f"File '{hdf5_opacity_file}' already exists, skipping conversion...")


def _chemical_table_dat2h5(path_input_data=petitradtrans_config_parser.get_input_data_path(), rewrite=False,
                           old_paths=False, clean=False):
    # noinspection PyUnresolvedReferences
    from petitRADTRANS.fortran_chemistry import fortran_chemistry as fchem

    # Read in parameters of chemistry grid
    if old_paths:
        path = str(os.path.join(
            path_input_data, __get_prt2_input_data_subpaths()['pre_calculated_chemistry'],
        ))
    else:
        path = str(os.path.join(
            path_input_data, get_input_data_subpaths()['pre_calculated_chemistry'], 'equilibrium_chemistry'
        ))

    hdf5_file = os.path.join(path, 'equilibrium_chemistry.chemtable.petitRADTRANS.h5')

    if os.path.isfile(hdf5_file) and not rewrite:
        __print_skipping_message(hdf5_file)
        return

    dat_files = {
        'FeH': os.path.join(path, "FEHs.dat"),
        'C/O': os.path.join(path, "COs.dat"),
        'T': os.path.join(path, "temps.dat"),
        'P': os.path.join(path, "pressures.dat"),
        'table': os.path.join(path, "species.dat")
    }

    if not os.path.isfile(dat_files['FeH']):
        warnings.warn('missing chemistry file, skipping conversion')
        return

    log10_metallicities = np.genfromtxt(dat_files['FeH'])
    co_ratios = np.genfromtxt(dat_files['C/O'])
    temperature = np.genfromtxt(dat_files['T'])
    pressure = np.genfromtxt(dat_files['P'])

    with open(dat_files['table'], 'r') as f:
        species_name = f.readlines()

    for i in range(len(species_name)):
        species_name[i] = species_name[i][:-1]  # remove the line break character

        # Fix C2H2 special name
        if species_name[i] == 'C2H2,acetylene':
            species_name[i] = 'C2H2'

    chemistry_table = fchem.read_dat_chemical_table(
        path,
        int(len(log10_metallicities)), int(len(co_ratios)), int(len(temperature)), int(len(pressure)),
        int(len(species_name))
    )

    chemistry_table = np.array(chemistry_table, dtype='d', order='F')

    # Remove nabla_ad and mmw from the "species"
    mean_molar_masses = chemistry_table[-2]
    nabla_adiabatic = chemistry_table[-1]
    chemistry_table = chemistry_table[:-2]
    species_name = species_name[:-2]

    if not os.path.isdir(path):
        os.makedirs(path)

    with h5py.File(hdf5_file, 'w') as f:
        dataset = f.create_dataset(
            name='co_ratios',
            data=co_ratios
        )
        dataset.attrs['long_name'] = 'Elemental abundance of Carbon over Oxygen (C/O) grid'
        dataset.attrs['units'] = 'None'

        dataset = f.create_dataset(
            name='log10_metallicities',
            data=log10_metallicities
        )
        dataset.attrs['long_name'] = (
            'Base 10 logarithm of the metallicity (Z/H with respect to the solar value) grid'
        )
        dataset.attrs['units'] = 'dex'

        dataset = f.create_dataset(
            name='mass_fractions',
            data=chemistry_table
        )
        dataset.attrs['long_name'] = 'Mass fraction table, with axes (species, temperature, pressure, C/O, Z/H)'
        dataset.attrs['units'] = 'None'

        dataset = f.create_dataset(
            name='mean_molar_masses',
            data=mean_molar_masses
        )
        dataset.attrs['long_name'] = 'Mean molar mass table, with axes (temperature, pressure, C/O, Z/H)'
        dataset.attrs['units'] = 'AMU'

        dataset = f.create_dataset(
            name='nabla_adiabatic',
            data=nabla_adiabatic
        )
        dataset.attrs['long_name'] = ('Table of the logarithmic derivative of temperature with respect to pressure, '
                                      'with axes (temperature, pressure, C/O, Z/H)')
        dataset.attrs['units'] = 'None'

        dataset = f.create_dataset(
            name='pressures',
            data=pressure
        )
        dataset.attrs['long_name'] = 'Pressure grid'
        dataset.attrs['units'] = 'bar'

        dataset = f.create_dataset(
            name='species',
            data=species_name
        )
        dataset.attrs['long_name'] = 'Species grid'
        dataset.attrs['units'] = 'N/A'

        dataset = f.create_dataset(
            name='temperatures',
            data=temperature
        )
        dataset.attrs['long_name'] = 'Temperature grid'
        dataset.attrs['units'] = 'K'

    print("Successfully converted chemical tables")

    if clean:
        __remove_files(list(dat_files.values()))
        __remove_files([os.path.join(path, 'abunds_python.dat')])


def _continuum_cia_dat2h5(path_input_data=petitradtrans_config_parser.get_input_data_path(),
                          rewrite=False, output_directory=None, old_paths=False, clean=False):
    """Using ExoMol units for HDF5 files."""
    # Initialize infos
    molliere2019_doi = '10.1051/0004-6361/201935470'

    if old_paths:
        doi_dict = _get_prt2_cia_names()
        description_dict = copy.deepcopy(doi_dict)

        doi_dict.update({
            'H2-H2': molliere2019_doi,
            'H2-He': molliere2019_doi,
            'H2O-H2O': 'unknown',
            'H2O-N2': 'unknown',
            'N2-H2': 'unknown',
            'N2-He': 'unknown',
            'N2-N2': 'unknown',
            'O2-O2': 'unknown',
            'N2-O2': 'unknown',
            'CO2-CO2': 'unknown'
        })

        description_dict.update({
            'H2-H2': 'None',
            'H2-He': 'None',
            'H2O-H2O': 'None',
            'H2O-N2': 'None',
            'N2-H2': 'None',
            'N2-He': 'None',
            'N2-N2': 'None',
            'O2-O2': 'None',
            'N2-O2': 'None',
            'CO2-CO2': 'None'
        })
    else:
        doi_dict = _get_base_cia_names()
        description_dict = copy.deepcopy(doi_dict)

        doi_dict.update({
            'H2--H2': molliere2019_doi,
            'H2--He': molliere2019_doi,
            'H2O--H2O': 'unknown',
            'H2O--N2': 'unknown',
            'N2--H2': 'unknown',
            'N2--He': 'unknown',
            'N2--N2': 'unknown',
            'O2--O2': 'unknown',
            'N2--O2': 'unknown',
            'CO2--CO2': 'unknown'
        })

        description_dict.update({
            'H2--H2': 'None',
            'H2--He': 'None',
            'H2O--H2O': 'None',
            'H2O--N2': 'None',
            'N2--H2': 'None',
            'N2--He': 'None',
            'N2--N2': 'None',
            'O2--O2': 'None',
            'N2--O2': 'None',
            'CO2--CO2': 'None'
        })

    # Get only existing directories
    if old_paths:
        input_directory = str(os.path.join(path_input_data, __get_prt2_input_data_subpaths()['cia_opacities']))
    else:
        input_directory = str(os.path.join(path_input_data, get_input_data_subpaths()['cia_opacities']))

    # Save each clouds data into HDF5 file
    if output_directory is None:
        output_directory_ref = input_directory
    else:
        output_directory_ref = copy.deepcopy(output_directory)

    # Loop over CIAs
    for i, key in enumerate(doi_dict):
        # Check if data directory exists
        if old_paths:
            cia_dir = os.path.join(str(input_directory), key)
        else:
            cia_dir = os.path.join(
                str(input_directory), key, CIAOpacity.get_species_isotopologue_name(_get_base_cia_names()[key])
            )

        if not os.path.isdir(cia_dir):
            __print_missing_data_file_message('CIA', key, cia_dir)
            continue

        # Get HDF5 file name
        output_directory = output_directory_ref
        hdf5_cia_file = os.path.join(
            cia_dir, _get_base_cia_names()[key] + '.ciatable.petitRADTRANS.h5'
        )

        if os.path.isfile(hdf5_cia_file) and not rewrite:
            __print_skipping_message(hdf5_cia_file)
            continue

        # Check if current key is in all information dicts
        not_in_dict = False

        if key not in description_dict:
            warnings.warn(f"CIA '{key}' was not in contributor dict; "
                          f"add key '{key}' to the script contributor_dict to run this conversion")
            not_in_dict = True

        if not_in_dict:
            print(" Skipping due to missing species in supplementary info dict...")
            continue

        # Read the dat files
        colliding_species = key.split('--')

        print(f"  Read CIA opacities for {key}...")
        cia_directory = os.path.join(input_directory, key)

        if not os.path.isdir(cia_directory):
            raise FileNotFoundError(f"CIA directory '{cia_directory}' do not exists")

        if not os.path.isdir(cia_dir):
            raise FileNotFoundError(f"CIA isotopologue directory '{cia_dir}' do not exists")

        cia_wavelength_grid, cia_temperature_grid, cia_alpha_grid, \
            cia_temp_dims, cia_lambda_dims = finput.load_cia_opacities(key.replace('--', '-'), cia_dir)
        cia_alpha_grid = np.array(cia_alpha_grid, dtype='d', order='F')
        cia_temperature_grid = cia_temperature_grid[:cia_temp_dims]
        cia_wavelength_grid = cia_wavelength_grid[:cia_lambda_dims]
        cia_alpha_grid = cia_alpha_grid[:cia_lambda_dims, :cia_temp_dims]

        weight = 1

        for species in colliding_species:
            weight = weight * get_species_molar_mass(species)

        cia_dict = {
            'id': key,
            'molecules': colliding_species,
            'weight': weight,
            'lambda': cia_wavelength_grid,
            'temperature': cia_temperature_grid,
            'alpha': cia_alpha_grid
        }

        wavenumbers = 1 / cia_dict['lambda'][::-1]  # cm to cm-1, with correct ordering

        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)

        # Write HDF5 file
        print(f" Writing file '{hdf5_cia_file}'...", end=' ')

        write_cia_opacities(
            hdf5_cia_file=hdf5_cia_file,
            molecules=cia_dict['molecules'],
            wavenumbers=wavenumbers,
            wavelengths=cia_dict['lambda'],
            alpha=np.transpose(cia_dict['alpha'])[:, ::-1],  # (temperature, wavenumber) wavenumbers ordering
            temperatures=cia_dict['temperature'],
            doi=doi_dict[key],
            description=description_dict[key]
        )

        print("Done.")

        if clean:
            files = glob.glob(os.path.join(cia_dir, '*.dat'))
            __remove_files(files)

    print("Successfully converted CIA opacities")


def _continuum_clouds_opacities_dat2h5(path_input_data=petitradtrans_config_parser.get_input_data_path(),
                                       rewrite=False, old_paths=False, clean=False):
    """Using ExoMol units for HDF5 files."""
    # Initialize infos
    molliere2019_doi = '10.1051/0004-6361/201935470'

    new_cloud_files = _get_base_cloud_names()

    doi_dict = _get_prt2_cloud_names()
    description_dict = copy.deepcopy(doi_dict)
    molmass_dict = copy.deepcopy(doi_dict)

    doi_dict.update({
        'Al2O3(c)_cm': molliere2019_doi,
        'Al2O3(c)_cd': molliere2019_doi,
        'Fe(c)_am': molliere2019_doi,
        'Fe(c)_ad': molliere2019_doi,
        'Fe(c)_cm': molliere2019_doi,
        'Fe(c)_cd': molliere2019_doi,
        'H2O(c)_cm': molliere2019_doi,
        'H2O(c)_cd': molliere2019_doi,
        'H2OL(c)_am': molliere2019_doi,  # TODO not in docs
        'H2OSO425(c)_am': molliere2019_doi,  # TODO not in docs
        'H2OSO450(c)_am': molliere2019_doi,  # TODO not in docs
        'H2OSO475(c)_am': molliere2019_doi,  # TODO not in docs
        'H2OSO484(c)_am': molliere2019_doi,  # TODO not in docs
        'H2OSO495(c)_am': molliere2019_doi,  # TODO not in docs
        'KCL(c)_cm': molliere2019_doi,
        'KCL(c)_cd': molliere2019_doi,
        'Mg05Fe05SiO3(c)_am': molliere2019_doi,
        'Mg05Fe05SiO3(c)_ad': molliere2019_doi,
        'Mg2SiO4(c)_am': molliere2019_doi,
        'Mg2SiO4(c)_ad': molliere2019_doi,
        'Mg2SiO4(c)_cm': molliere2019_doi,
        'Mg2SiO4(c)_cd': molliere2019_doi,
        'MgAl2O4(c)_cm': molliere2019_doi,
        'MgAl2O4(c)_cd': molliere2019_doi,
        'MgFeSiO4(c)_am': molliere2019_doi,
        'MgFeSiO4(c)_ad': molliere2019_doi,
        'MgSiO3(c)_am': molliere2019_doi,
        'MgSiO3(c)_ad': molliere2019_doi,
        'MgSiO3(c)_cm': molliere2019_doi,
        'MgSiO3(c)_cd': molliere2019_doi,
        'Na2S(c)_cm': molliere2019_doi,
        'Na2S(c)_cd': molliere2019_doi,
        'SiC(c)_cm': molliere2019_doi,
        'SiC(c)_cd': molliere2019_doi
    })

    description_dict.update({
        'Al2O3(c)_cm': '',
        'Al2O3(c)_cd': '',
        'Fe(c)_am': '',
        'Fe(c)_ad': '',
        'Fe(c)_cm': '',
        'Fe(c)_cd': '',
        'H2O(c)_cm': '',
        'H2O(c)_cd': '',
        'H2OL(c)_am': '',  # TODO not in docs
        'H2OSO425(c)_am': '',  # TODO not in docs
        'H2OSO450(c)_am': '',  # TODO not in docs
        'H2OSO475(c)_am': '',  # TODO not in docs
        'H2OSO484(c)_am': '',  # TODO not in docs
        'H2OSO495(c)_am': '',  # TODO not in docs
        'KCL(c)_cm': '',
        'KCL(c)_cd': '',
        'Mg05Fe05SiO3(c)_am': '',
        'Mg05Fe05SiO3(c)_ad': '',
        'Mg2SiO4(c)_am': '',
        'Mg2SiO4(c)_ad': '',
        'Mg2SiO4(c)_cm': '',
        'Mg2SiO4(c)_cd': '',
        'MgAl2O4(c)_cm': '',
        'MgAl2O4(c)_cd': '',
        'MgFeSiO4(c)_am': '',
        'MgFeSiO4(c)_ad': '',
        'MgSiO3(c)_am': '',
        'MgSiO3(c)_ad': '',
        'MgSiO3(c)_cm': '',
        'MgSiO3(c)_cd': '',
        'Na2S(c)_cm': '',
        'Na2S(c)_cd': '',
        'SiC(c)_cm': '',
        'SiC(c)_cd': ''
    })

    for key in description_dict:
        particle_mode = key.rsplit('_', 1)[1]
        particle_mode_description = ''

        if particle_mode[0] == 'c':
            particle_mode_description += 'Crystalline, '
        elif particle_mode[0] == 'a':
            particle_mode_description += 'Amorphous, '
        else:
            raise ValueError(f"invalid particle name '{key}': "
                             f"particle internal structure must be either crystalline ('c') or amorphous ('a'), "
                             f"but was '{particle_mode[0]}'")

        if particle_mode[1] == 'm':
            particle_mode_description += 'Mie scattering (spherical)'
        elif particle_mode[1] == 'd':
            particle_mode_description += 'DHS (irregular shape)'
        else:
            raise ValueError(f"invalid particle name '{key}': "
                             f"particle shape must be either calculated using Mie scattering ('m') "
                             f"or using the distribution of hollow spheres method ('d'), "
                             f"but was '{particle_mode[1]}'")

        description_dict[key] = particle_mode_description

    molmass_dict.update({
        'Al2O3(c)_cm': get_species_molar_mass('Al2O3'),
        'Al2O3(c)_cd': get_species_molar_mass('Al2O3'),
        'Fe(c)_am': get_species_molar_mass('Fe'),
        'Fe(c)_ad': get_species_molar_mass('Fe'),
        'Fe(c)_cm': get_species_molar_mass('Fe'),
        'Fe(c)_cd': get_species_molar_mass('Fe'),
        'H2O(c)_cm': get_species_molar_mass('H2O'),
        'H2O(c)_cd': get_species_molar_mass('H2O'),
        'H2OL(c)_am': get_species_molar_mass('H2O'),
        'H2OSO425(c)_am': get_species_molar_mass('H2O') * 0.75 + get_species_molar_mass('SO4') * 0.25,
        'H2OSO450(c)_am': get_species_molar_mass('H2O') * 0.50 + get_species_molar_mass('SO4') * 0.50,
        'H2OSO475(c)_am': get_species_molar_mass('H2O') * 0.25 + get_species_molar_mass('SO4') * 0.75,
        'H2OSO484(c)_am': get_species_molar_mass('H2O') * 0.16 + get_species_molar_mass('SO4') * 0.84,
        'H2OSO495(c)_am': get_species_molar_mass('H2O') * 0.05 + get_species_molar_mass('SO4') * 0.95,
        'KCL(c)_cm': get_species_molar_mass('H2O'),
        'KCL(c)_cd': get_species_molar_mass('H2O'),
        'Mg05Fe05SiO3(c)_am': (
                0.5 * get_species_molar_mass('Mg') + 0.5 * get_species_molar_mass('Fe') + get_species_molar_mass('SiO3')
        ),
        'Mg05Fe05SiO3(c)_ad': (
                0.5 * get_species_molar_mass('Mg') + 0.5 * get_species_molar_mass('Fe') + get_species_molar_mass('SiO3')
        ),
        'Mg2SiO4(c)_am': get_species_molar_mass('Mg2SiO4'),
        'Mg2SiO4(c)_ad': get_species_molar_mass('Mg2SiO4'),
        'Mg2SiO4(c)_cm': get_species_molar_mass('Mg2SiO4'),
        'Mg2SiO4(c)_cd': get_species_molar_mass('Mg2SiO4'),
        'MgAl2O4(c)_cm': get_species_molar_mass('MgAl2O4'),
        'MgAl2O4(c)_cd': get_species_molar_mass('MgAl2O4'),
        'MgFeSiO4(c)_am': get_species_molar_mass('MgFeSiO4'),
        'MgFeSiO4(c)_ad': get_species_molar_mass('MgFeSiO4'),
        'MgSiO3(c)_am': get_species_molar_mass('MgSiO3'),
        'MgSiO3(c)_ad': get_species_molar_mass('MgSiO3'),
        'MgSiO3(c)_cm': get_species_molar_mass('MgSiO3'),
        'MgSiO3(c)_cd': get_species_molar_mass('MgSiO3'),
        'Na2S(c)_cm': get_species_molar_mass('Na2S'),
        'Na2S(c)_cd': get_species_molar_mass('Na2S'),
        'SiC(c)_cm': get_species_molar_mass('SiC'),
        'SiC(c)_cd': get_species_molar_mass('SiC')
    })

    # Get only existing directories
    if old_paths:
        input_directory = os.path.join(path_input_data, __get_prt2_input_data_subpaths()['clouds_opacities'])
    else:
        input_directory = os.path.join(path_input_data, get_input_data_subpaths()['clouds_opacities'])

    input_directory = str(input_directory)

    bad_keys = []

    for key in doi_dict:
        species = key.split('(', 1)[0]

        if old_paths:
            species_dir = os.path.join(input_directory, species + '_c')
        else:
            k = _get_prt2_cloud_names()[key]

            if k not in _get_base_cloud_names():
                warnings.warn(f"pRT2 species name '{k}' has no default pRT3 name, skipping...")
                bad_keys.append(key)

                continue

            species_dir = CloudOpacity.get_species_base_name(
                species_full_name=_get_base_cloud_names()[k],
                join=True
            )
            iso_dir = CloudOpacity.get_species_isotopologue_name(_get_base_cloud_names()[k], join=True)
            species_dir = os.path.join(input_directory, species_dir, iso_dir)

        if not os.path.isdir(species_dir):
            __print_missing_data_file_message('cloud', key, species_dir)
            bad_keys.append(key)
            continue

        particle_mode = key.rsplit('_', 1)[1]

        particle_mode_dir = None

        if old_paths:
            if particle_mode[0] == 'c':
                particle_mode_dir = os.path.join(species_dir, 'crystalline')
            elif particle_mode[0] == 'a':
                particle_mode_dir = os.path.join(species_dir, 'amorphous')

            if not os.path.isdir(particle_mode_dir):
                __print_missing_data_file_message('cloud', key, particle_mode_dir)
                bad_keys.append(key)
                continue

            if particle_mode[1] == 'm':
                particle_mode_dir = os.path.join(particle_mode_dir, 'mie')
            elif particle_mode[1] == 'd':
                particle_mode_dir = os.path.join(particle_mode_dir, 'DHS')
        else:
            if particle_mode[1] == 'm':
                particle_mode_dir = os.path.join(species_dir, 'mie')
            elif particle_mode[1] == 'd':
                particle_mode_dir = os.path.join(species_dir, 'DHS')

        if not os.path.isdir(particle_mode_dir):
            print(__print_missing_data_file_message('cloud', key, particle_mode_dir))
            bad_keys.append(key)
            continue

    for key in bad_keys:
        del doi_dict[key]

    if len(doi_dict) == 0:
        print("No cloud opacities conversion is necessary or possible")
        return

    # Prepare single strings delimited by ':' which are then put into Fortran routines
    cloud_species_modes = []
    cloud_species = []
    cloud_isos = []

    for key in doi_dict:
        cloud_species_ = key.rsplit('_', 1)
        cloud_species_modes.append(cloud_species_[1])

        if old_paths:
            cloud_species.append(cloud_species_[0])

            if cloud_species_modes[-1][0] == 'c':
                cloud_isos.append('crystalline')
            elif cloud_species_modes[-1][0] == 'a':
                cloud_isos.append('amorphous')
            else:
                raise ValueError(f"invalid cloud mode '{cloud_species_modes[-1]}' for key '{key}'")
        else:
            k = _get_prt2_cloud_names()[key]

            if 'KCl(s)' in k:
                basename = 'KCL'
            else:
                basename = CloudOpacity.get_species_base_name(_get_base_cloud_names()[k])

            cloud_species.append(basename + '(c)')

            species_dir = CloudOpacity.get_species_base_name(
                species_full_name=_get_base_cloud_names()[k],
                join=True
            )
            iso_dir = CloudOpacity.get_species_isotopologue_name(_get_base_cloud_names()[k], join=True)
            cloud_isos.append(os.path.join(species_dir, iso_dir))

    all_cloud_species = ''

    for cloud_species_ in cloud_species:
        all_cloud_species = all_cloud_species + cloud_species_ + ','

    all_cloud_species_mode = ''

    for cloud_species_mode in cloud_species_modes:
        all_cloud_species_mode = all_cloud_species_mode + cloud_species_mode + ','

    all_cloud_isos = ''

    for cloud_iso in cloud_isos:
        all_cloud_isos = all_cloud_isos + cloud_iso + ','

    if old_paths:
        reference_file = str(os.path.join(
            path_input_data, __get_prt2_input_data_subpaths()['clouds_opacities'],
            'MgSiO3_c', 'amorphous', 'mie', 'opa_0001.dat'
        ))
    else:
        reference_file = str(os.path.join(
            path_input_data,
            get_input_data_subpaths()['clouds_opacities'],
            'MgSiO3(s)_amorphous', 'Mg-Si-O3-NatAbund(s)_amorphous', 'mie', 'opa_0001.dat'
        ))

    if not os.path.isfile(reference_file):
        raise FileNotFoundError(
            f"reference file for loading .dat cloud opacities ('{reference_file}') not found, "
            f"it must be downloaded "
            f"(see https://petitradtrans.readthedocs.io/en/latest/content/available_opacities.html)"
        )

    n_cloud_wavelength_bins = int(len(np.genfromtxt(reference_file)[:, 0]))

    if old_paths:
        cloud_path = str(os.path.join(path_input_data, __get_prt2_input_data_subpaths()['clouds_opacities']))
        path_reference_files = os.path.join(cloud_path, 'MgSiO3_c', 'amorphous', 'mie')
    else:
        cloud_path = str(os.path.join(path_input_data, get_input_data_subpaths()['clouds_opacities']))
        path_reference_files = os.path.join(cloud_path, 'MgSiO3(s)_amorphous', 'Mg-Si-O3-NatAbund(s)_amorphous', 'mie')

    path_input_files = os.path.join(path_input_data, 'opa_input_files')

    # Load .dat files
    print("Loading dat files...")
    cloud_particles_densities, cloud_absorption_opacities, cloud_scattering_opacities, \
        cloud_asymmetry_parameter, cloud_wavelengths, cloud_particles_radius_bins, cloud_particles_radii \
        = finput.load_cloud_opacities(
            cloud_path, path_input_files, path_reference_files,
            all_cloud_species, all_cloud_isos, all_cloud_species_mode,
            len(doi_dict), n_cloud_wavelength_bins
        )

    wavenumbers = 1 / cloud_wavelengths[::-1]  # cm to cm-1

    # Save each clouds data into HDF5 file
    for i, key in enumerate(doi_dict):
        # Check if current key is in all information dicts
        not_in_dict = False

        if key not in description_dict:
            warnings.warn(f"cloud '{key}' was not in contributor dict; "
                          f"add key '{key}' to the script contributor_dict to run this conversion")
            not_in_dict = True

        if key not in molmass_dict:
            warnings.warn(f"cloud '{key}' was not in molar mass dict; "
                          f"add key '{key}' to the script molmass_dict to run this conversion")
            not_in_dict = True

        if not_in_dict:
            print(" Skipping due to missing species in supplementary info dict...")
            continue

        # Get HDF5 file name
        particle_mode = key.rsplit('_', 1)[1]

        if key == 'H2OL(c)_am':
            new_key = 'H2O(l)'
        elif key in ['KCL(c)_cm', 'KCL(c)_cd']:
            new_key = 'KCl(s)_crystalline'
        elif key in ['H2OSO425(c)_am', 'H2OSO450(c)_am', 'H2OSO475(c)_am', 'H2OSO484(c)_am', 'H2OSO495(c)_am']:
            new_key = 'H2OSO4(l)'
        else:
            new_key = key.replace('(c)', '(s)')

            if particle_mode[0] == 'a':
                new_key = new_key.replace(particle_mode, 'amorphous')
            elif particle_mode[0] == 'c':
                new_key = new_key.replace(particle_mode, 'crystalline')
            else:
                raise ValueError(f"invalid particle mode '{particle_mode}'")

        output_directory = os.path.join(input_directory, cloud_isos[i])

        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)

        if particle_mode[1] == 'd':
            method = 'DHS'
        elif particle_mode[1] == 'm':
            method = 'Mie'
        else:
            raise ValueError(f"invalid particle mode '{particle_mode}'")

        if key in ['H2OSO425(c)_am', 'H2OSO450(c)_am', 'H2OSO475(c)_am', 'H2OSO484(c)_am', 'H2OSO495(c)_am']:
            percentage = key.split('(', 1)[0][-2:]
            method = f"aq{percentage}" + method

        new_cloud_file = new_cloud_files[new_key]
        species, spectral_info = new_cloud_file.split('.', 1)

        hdf5_opacity_file = os.path.join(
            output_directory, f"{species}__{method}.{spectral_info}.cotable.petitRADTRANS.h5"
        )

        if os.path.isfile(hdf5_opacity_file) and not rewrite:
            __print_skipping_message(hdf5_opacity_file)
            continue

        # Write HDF5 file
        print(f" Writing file '{hdf5_opacity_file}'...", end=' ')

        with h5py.File(hdf5_opacity_file, "w") as fh5:
            dataset = fh5.create_dataset(
                name='DOI',
                shape=(1,),
                data=doi_dict[key]
            )
            dataset.attrs['long_name'] = 'Data object identifier linked to the data'
            dataset.attrs['additional_description'] = description_dict[key]

            dataset = fh5.create_dataset(
                name='Date_ID',
                shape=(1,),
                data=f'petitRADTRANS-v{petitRADTRANS.__version__}'
                     f'_{datetime.datetime.now(datetime.timezone.utc).isoformat()}'
            )
            dataset.attrs['long_name'] = 'ISO 8601 UTC time (https://docs.python.org/3/library/datetime.html) ' \
                                         'at which the table has been created, ' \
                                         'along with the version of petitRADTRANS'

            dataset = fh5.create_dataset(
                name='wavenumbers',
                data=wavenumbers
            )
            dataset.attrs['long_name'] = 'Opacities wavenumbers'
            dataset.attrs['units'] = 'cm^-1'

            dataset = fh5.create_dataset(
                name='absorption_opacities',
                data=cloud_absorption_opacities[:, ::-1, i]
            )
            dataset.attrs['long_name'] = 'Table of the absorption opacities with axes (particle radius, wavenumber)'
            dataset.attrs['units'] = 'cm^2.g^-1'

            dataset = fh5.create_dataset(
                name='scattering_opacities',
                data=cloud_scattering_opacities[:, ::-1, i]
            )
            dataset.attrs['long_name'] = 'Table of the scattering opacities with axes (particle radius, wavenumber)'
            dataset.attrs['units'] = 'cm^2.g^-1'

            dataset = fh5.create_dataset(
                name='asymmetry_parameters',
                data=cloud_asymmetry_parameter[:, ::-1, i]
            )
            dataset.attrs['long_name'] = 'Table of the asymmetry parameters with axes (particle radius, wavenumber)'
            dataset.attrs['units'] = 'None'

            dataset = fh5.create_dataset(
                name='particles_density',
                data=cloud_particles_densities[i]
            )
            dataset.attrs['long_name'] = 'Average density of the cloud particles'
            dataset.attrs['units'] = 'g.cm^-3'

            dataset = fh5.create_dataset(
                name='mol_name',
                shape=(1,),
                data=cloud_species[i]
            )
            dataset.attrs['long_name'] = 'Name of the species described, "(c)" indicates that it has condensed'

            dataset = fh5.create_dataset(
                name='particles_radii',
                data=cloud_particles_radii
            )
            dataset.attrs['long_name'] = 'Particles average radius grid'
            dataset.attrs['units'] = 'cm'

            dataset = fh5.create_dataset(
                name='particle_radius_bins',
                data=cloud_particles_radius_bins
            )
            dataset.attrs['long_name'] = 'Particles average radius grid bins'
            dataset.attrs['units'] = 'cm'

            dataset = fh5.create_dataset(
                name='wlrange',
                data=np.array([cloud_wavelengths.min(), cloud_wavelengths.max()]) * 1e4  # cm to um
            )
            dataset.attrs['long_name'] = 'Wavelength range covered'
            dataset.attrs['units'] = 'µm'

            dataset = fh5.create_dataset(
                name='wnrange',
                data=np.array([wavenumbers.min(), wavenumbers.max()])
            )
            dataset.attrs['long_name'] = 'Wavenumber range covered'
            dataset.attrs['units'] = 'cm^-1'

        print("Done.")

        if clean:
            if method == 'Mie':
                method = 'mie'

            __remove_files([os.path.join(output_directory, method)])

    print("Successfully converted cloud opacities")


def _correlated_k_opacities_dat2h5(path_input_data=petitradtrans_config_parser.get_input_data_path(),
                                   rewrite=False, old_paths=False, clean=False,
                                   external_single_species=False,
                                   path_to_external_species_opacity_folder=None,
                                   external_species_longname=None,
                                   external_species_doi=None,
                                   external_species_contributor=None,
                                   external_species_description=None,
                                   external_species_molmass=None):
    # Initialize information
    kurucz_website = 'http://kurucz.harvard.edu/'
    molliere2019_doi = '10.1051/0004-6361/201935470'
    burrows2003_doi = '10.1086/345412'
    mckemmish2019_doi = '10.1093/mnras/stz1818'
    hitemp_doi = '10.1016/j.jqsrt.2010.05.001'
    vald_website = 'http://vald.astro.uu.se/'
    hitran_doi = '10.1016/j.jqsrt.2013.07.002'

    exomolop_description = '10.1051/0004-6361/202038350'

    molaverdikhani_email = 'karan.molaverdikhani@colorado.edu'

    kurucz_description = 'gamma_nat + V dW, sigma_therm'
    exomolop_link = 'https://www.exomol.com/data/data-types/opacity/'

    if not external_single_species:
        names = _get_prt2_correlated_k_names()

        for key, value in names.items():
            if value is None:
                names[key] = _get_base_correlated_k_names()[key]
    else:
        try:
            spec = path_to_external_species_opacity_folder.rsplit(os.path.sep, 1)[1]
        except IndexError:  # Case where one gives the external target folder path locally (so not absolute path).
            spec = path_to_external_species_opacity_folder
        spec = spec.rsplit('_def', 1)[0]
        names = {spec: external_species_longname}

    # None is used for already referenced HDF5 files
    doi_dict = _get_prt2_correlated_k_names()
    contributor_dict = copy.deepcopy(doi_dict)
    description_dict = copy.deepcopy(doi_dict)
    molmass_dict = copy.deepcopy(doi_dict)

    doi_dict.update({
        'Al': kurucz_website,
        'Al+': kurucz_website,
        'AlH': None,
        'AlO': None,
        'C2H2': None,
        'C2H4': None,
        'Ca': kurucz_website,
        'Ca+': kurucz_website,
        'CaH': None,
        'CH4': None,
        'CO_12_HITEMP': hitemp_doi,
        'CO_13_HITEMP': hitemp_doi,
        'CO_13_Chubb': None,
        'CO_all_iso_Chubb': None,
        'CO_all_iso_HITEMP': hitemp_doi,
        'CO2': None,
        'CrH': None,
        'Fe': kurucz_website,
        'Fe+': kurucz_website,
        'FeH': None,
        'H2O_Exomol': None,
        'H2O_HITEMP': hitemp_doi,
        'H2S': None,
        'HCN': None,
        'K_allard': molliere2019_doi,
        'K_burrows': burrows2003_doi,
        'K_lor_cut': vald_website,
        'Li': kurucz_website,
        'Mg': kurucz_website,
        'Mg+': kurucz_website,
        'MgH': None,
        'MgO': None,
        'Na_allard': '10.1051/0004-6361/201935593',
        'Na_burrows': burrows2003_doi,
        'Na_lor_cut': vald_website,
        'NaH': None,
        'NH3': None,
        'O': kurucz_website,
        'O+': kurucz_website,  # TODO not in the docs
        'O2': None,
        'O3': hitran_doi,
        'OH': None,
        'PH3': None,
        'SH': None,
        'Si': kurucz_website,
        'Si+': kurucz_website,
        'SiO': None,
        'SiO2': None,
        'Ti': kurucz_website,
        'Ti+': kurucz_website,
        'TiO_48_Exomol': mckemmish2019_doi,
        'TiO_48_Plez': molliere2019_doi,
        'TiO_all_Exomol': mckemmish2019_doi,
        'TiO_all_Plez': molliere2019_doi,
        'V': kurucz_website,
        'V+': kurucz_website,
        'VO': None,
        'VO_Plez': molliere2019_doi
    })
    contributor_dict.update({
        'Al': molaverdikhani_email,
        'Al+': molaverdikhani_email,
        'AlH': exomolop_link,
        'AlO': exomolop_link,
        'C2H2': exomolop_link,
        'C2H4': exomolop_link,
        'Ca': molaverdikhani_email,
        'Ca+': molaverdikhani_email,
        'CaH': exomolop_link,
        'CH4': exomolop_link,
        'CO_12_HITEMP': 'None',
        'CO_13_HITEMP': 'None',
        'CO_13_Chubb': exomolop_link,
        'CO_all_iso_Chubb': exomolop_link,
        'CO_all_iso_HITEMP': 'None',
        'CO2': exomolop_link,
        'CrH': exomolop_link,
        'Fe': molaverdikhani_email,
        'Fe+': molaverdikhani_email,
        'FeH': exomolop_link,
        'H2O_Exomol': exomolop_link,
        'H2O_HITEMP': 'None',
        'H2S': exomolop_link,
        'HCN': exomolop_link,
        'K_allard': 'None',
        'K_burrows': 'None',
        'K_lor_cut': 'None',
        'Li': molaverdikhani_email,
        'Mg': molaverdikhani_email,
        'Mg+': molaverdikhani_email,
        'MgH': exomolop_link,
        'MgO': exomolop_link,
        'Na_allard': 'None',
        'Na_burrows': 'None',
        'Na_lor_cut': 'None',
        'NaH': exomolop_link,
        'NH3': exomolop_link,
        'O': molaverdikhani_email,
        'O+': molaverdikhani_email,  # TODO not in the docs
        'O2': exomolop_link,
        'O3': 'None',
        'OH': exomolop_link,
        'PH3': exomolop_link,
        'SH': exomolop_link,
        'Si': molaverdikhani_email,
        'Si+': molaverdikhani_email,
        'SiO': exomolop_link,
        'SiO2': exomolop_link,
        'Ti': molaverdikhani_email,
        'Ti+': molaverdikhani_email,
        'TiO_48_Exomol': exomolop_link,
        'TiO_48_Plez': 'None',
        'TiO_all_Exomol': exomolop_link,
        'TiO_all_Plez': 'None',
        'V': molaverdikhani_email,
        'V+': molaverdikhani_email,
        'VO': exomolop_link,
        'VO_Plez': 'None'
    })
    description_dict.update({
        'Al': kurucz_description,
        'Al+': kurucz_description,
        'AlH': 'Main isotopologue, ' + exomolop_description,
        'AlO': 'Main isotopologue, ' + exomolop_description,
        'C2H2': 'Main isotopologue, ' + exomolop_description,
        'C2H4': 'Main isotopologue, ' + exomolop_description,
        'Ca': kurucz_description,
        'Ca+': kurucz_description,
        'CaH': 'Main isotopologue, ' + exomolop_description,
        'CH4': 'Main isotopologue, ' + exomolop_description,
        'CO_12_HITEMP': "Using HITEMP's air broadening prescription.",
        'CO_13_HITEMP': "Using HITEMP's air broadening prescription.",
        'CO_13_Chubb': '13C-16O, ' + exomolop_description,
        'CO_all_iso_Chubb': 'All isotopologues, ' + exomolop_description,
        'CO_all_iso_HITEMP': "Using HITEMP's air broadening prescription.",
        'CO2': 'Main isotopologue, ' + exomolop_description,
        'CrH': 'Main isotopologue, ' + exomolop_description,
        'Fe': kurucz_description,
        'Fe+': kurucz_description,
        'FeH': 'Main isotopologue, ' + exomolop_description,
        'H2O_Exomol': 'Main isotopologue, ' + exomolop_description,
        'H2O_HITEMP': "Using HITEMP's air broadening prescription.",
        'H2S': 'Main isotopologue, ' + exomolop_description,
        'HCN': 'Main isotopologue, ' + exomolop_description,
        'K_allard': 'Allard wings',
        'K_burrows': 'Burrows wings',
        'K_lor_cut': 'Lorentzian wings',
        'Li': kurucz_description,
        'Mg': kurucz_description,
        'Mg+': kurucz_description,
        'MgH': 'Main isotopologue, ' + exomolop_description,
        'MgO': 'Main isotopologue, ' + exomolop_description,
        'Na_allard': 'new Allard wings',  # TODO difference with "old" Allard wings?
        'Na_burrows': 'Burrows wings',
        'Na_lor_cut': 'Lorentzian wings',
        'NaH': 'Main isotopologue, ' + exomolop_description,
        'NH3': 'Main isotopologue, ' + exomolop_description,
        'O': kurucz_description,
        'O+': kurucz_description,  # TODO not in the docs
        'O2': 'Main isotopologue, ' + exomolop_description,
        'O3': "Using HITRAN's air broadening prescription.",
        'OH': 'Main isotopologue, ' + exomolop_description,
        'PH3': 'Main isotopologue, ' + exomolop_description,
        'SH': 'Main isotopologue, ' + exomolop_description,
        'Si': kurucz_description,
        'Si+': kurucz_description,
        'SiO': 'Main isotopologue, ' + exomolop_description,
        'SiO2': 'Main isotopologue, ' + exomolop_description,
        'Ti': kurucz_description,
        'Ti+': kurucz_description,
        'TiO_48_Exomol': 'Using Sharp & Burrows, Eq. 15 for pressure broadening.',
        'TiO_48_Plez': 'Using Sharp & Burrows, Eq. 15 for pressure broadening.',
        'TiO_all_Exomol': 'Using Sharp & Burrows, Eq. 15 for pressure broadening.',
        'TiO_all_Plez': 'Using Sharp & Burrows, Eq. 15 for pressure broadening.',
        'V': kurucz_description,
        'V+': kurucz_description,
        'VO': 'Main isotopologue, ' + exomolop_description,
        'VO_Plez': 'Using Sharp & Burrows, Eq. 15 for pressure broadening.'
    })
    molmass_dict.update({
        'Al': get_species_molar_mass('Al'),
        'Al+': get_species_molar_mass('Al') - get_species_molar_mass('e-'),
        'AlH': get_species_molar_mass('AlH'),
        'AlO': get_species_molar_mass('AlO'),
        'C2H2': get_species_molar_mass('C2H2'),
        'C2H4': get_species_molar_mass('C2H4'),
        'Ca': get_species_molar_mass('Ca'),
        'Ca+': get_species_molar_mass('Ca') - get_species_molar_mass('e-'),
        'CaH': get_species_molar_mass('CaH'),
        'CH4': get_species_molar_mass('CH4'),
        'CO_12_HITEMP': get_species_molar_mass('12CO'),
        'CO_13_HITEMP': get_species_molar_mass('13CO'),
        'CO_13_Chubb': get_species_molar_mass('13CO'),
        'CO_all_iso_Chubb': get_species_molar_mass('CO'),
        'CO_all_iso_HITEMP': get_species_molar_mass('CO'),
        'CO2': get_species_molar_mass('CO2'),
        'CrH': get_species_molar_mass('CrH'),
        'Fe': get_species_molar_mass('Fe'),
        'Fe+': get_species_molar_mass('Fe+'),
        'FeH': get_species_molar_mass('FeH'),
        'H2O_Exomol': get_species_molar_mass('H2O'),
        'H2O_HITEMP': get_species_molar_mass('H2O'),
        'H2S': get_species_molar_mass('H2S'),
        'HCN': get_species_molar_mass('HCN'),
        'K_allard': get_species_molar_mass('K'),
        'K_burrows': get_species_molar_mass('K'),
        'K_lor_cut': get_species_molar_mass('K'),
        'Li': get_species_molar_mass('Li'),
        'Mg': get_species_molar_mass('Mg'),
        'Mg+': get_species_molar_mass('Mg') - get_species_molar_mass('e-'),
        'MgH': get_species_molar_mass('MgH'),
        'MgO': get_species_molar_mass('MgO'),
        'Na_allard': get_species_molar_mass('Na'),
        'Na_burrows': get_species_molar_mass('Na'),
        'Na_lor_cut': get_species_molar_mass('Na'),
        'NaH': get_species_molar_mass('NaH'),
        'NH3': get_species_molar_mass('NH3'),
        'O': get_species_molar_mass('O'),
        'O+': get_species_molar_mass('O') - get_species_molar_mass('e-'),  # TODO not in the docs
        'O2': get_species_molar_mass('O2'),
        'O3': get_species_molar_mass('O3'),
        'OH': get_species_molar_mass('OH'),
        'PH3': get_species_molar_mass('PH3'),
        'SH': get_species_molar_mass('SH'),
        'Si': get_species_molar_mass('Si'),
        'Si+': get_species_molar_mass('Si') - get_species_molar_mass('e-'),
        'SiO': get_species_molar_mass('SiO'),
        'SiO2': get_species_molar_mass('SiO2'),
        'Ti': get_species_molar_mass('Ti'),
        'Ti+': get_species_molar_mass('Ti') - get_species_molar_mass('e-'),
        'TiO_48_Exomol': get_species_molar_mass('48TiO'),
        'TiO_48_Plez': get_species_molar_mass('48TiO'),
        'TiO_all_Exomol': get_species_molar_mass('TiO'),
        'TiO_all_Plez': get_species_molar_mass('TiO'),
        'V': get_species_molar_mass('V'),
        'V+': get_species_molar_mass('V') - get_species_molar_mass('e-'),
        'VO': get_species_molar_mass('VO'),
        'VO_Plez': get_species_molar_mass('VO')
    })

    # Loading
    print("Loading default files names...")
    line_paths = np.loadtxt(os.path.join(path_input_data, 'opa_input_files', 'opa_filenames.txt'), dtype=str)

    print("Loading default PT grid...")

    opacities_temperature_profile_grid = np.genfromtxt(
        os.path.join(path_input_data, 'opa_input_files', 'opa_PT_grid.dat')
    )

    opacities_temperature_profile_grid = np.flip(opacities_temperature_profile_grid, axis=1)

    opacities_temperatures = np.unique(opacities_temperature_profile_grid[:, 0])
    opacities_pressures = np.unique(opacities_temperature_profile_grid[:, 1])  # grid is already in bar

    print("Loading correlated-k g grid...")
    buffer = np.genfromtxt(
        os.path.join(path_input_data, 'opa_input_files', 'g_comb_grid.dat')
    )
    g_gauss = np.array(buffer[:, 0], dtype='d', order='F')
    weights_gauss = np.array(buffer[:, 1], dtype='d', order='F')

    if not external_single_species:
        if old_paths:
            input_directory = str(
                os.path.join(path_input_data, __get_prt2_input_data_subpaths()['correlated_k_opacities'])
            )

            directories = [
                os.path.join(input_directory, d) for d in os.listdir(input_directory)
                if os.path.isdir(d) and d != 'PaxHeader'
            ]
        else:
            input_directory = str(
                os.path.join(path_input_data, get_input_data_subpaths()['correlated_k_opacities'])
            )

            directories = []

            for species_dir in os.listdir(input_directory):
                species_dir = os.path.join(input_directory, species_dir)

                if os.path.isdir(species_dir) and species_dir.rsplit(os.path.sep, 1)[1] != 'PaxHeader':
                    for iso_dir in os.listdir(species_dir):
                        iso_dir = os.path.join(species_dir, iso_dir)

                        if os.path.isdir(iso_dir) and iso_dir.rsplit(os.path.sep, 1)[1] != 'PaxHeader':
                            for d in os.listdir(iso_dir):
                                d = os.path.join(iso_dir, d)

                                if os.path.isdir(d) and d.rsplit(os.path.sep, 1)[1] != 'PaxHeader':
                                    directories.append(d)
    else:
        directories = [path_to_external_species_opacity_folder]

    for directory in directories:
        try:
            species = directory.rsplit(os.path.sep, 1)[1]
        except IndexError:  # Case where one gives the external target folder path locally (so not absolute path).
            species = directory

        species = species.rsplit('_def', 1)[0]
        not_in_dict = False

        if external_single_species:
            doi_dict.unlock()
            doi_dict[species] = external_species_doi
            doi_dict.lock()
            contributor_dict.unlock()
            contributor_dict[species] = external_species_contributor
            contributor_dict.lock()
            description_dict.unlock()
            description_dict[species] = external_species_description
            description_dict.lock()
            molmass_dict.unlock()
            molmass_dict[species] = external_species_molmass
            molmass_dict.lock()

        # Check information availability
        if species not in doi_dict:
            warnings.warn(f"species '{species}' was not in DOI dict; "
                          f"add key '{species}' to the script doi_ict to run this conversion")
            not_in_dict = True

        if species not in contributor_dict:
            warnings.warn(f"species '{species}' was not in contributor dict; "
                          f"add key '{species}' to the script contributor_dict to run this conversion")
            not_in_dict = True

        if species not in description_dict:
            warnings.warn(f"species '{species}' was not in contributor dict; "
                          f"add key '{species}' to the script contributor_dict to run this conversion")
            not_in_dict = True

        if species not in molmass_dict:
            warnings.warn(f"species '{species}' was not in molar mass dict; "
                          f"add key '{species}' to the script molmass_dict to run this conversion")
            not_in_dict = True

        if not_in_dict:
            if external_single_species:
                raise ValueError(f" Please add all required information for the conversion of species '{species}'.")
            else:
                print(f" Skipping species '{species}' due to missing species in supplementary info dict...")
                continue

        if doi_dict[species] is None:
            file = glob.glob(os.path.join(directory, '*.h5'))

            if len(file) == 0:
                print(f" HDF5 file for species '{species}' was already moved...")
                continue
            elif len(file) > 1:
                raise FileExistsError(f"more than one HDF5 file in '{directory}' ({file})")

            file = file[0]
            f = file.rsplit(os.path.sep, 1)[1]

            if not external_single_species:
                if f != _get_prt2_correlated_k_names()[species] and _get_prt2_correlated_k_names()[species] is not None:
                    f = _get_prt2_correlated_k_names()[species]
            else:
                f = external_species_longname

            if '.ktable.petitRADTRANS.h5' not in f:
                f += '.ktable.petitRADTRANS.h5'

            new_file = os.path.abspath(os.path.join(directory, '..', f))
            print(f"Moving HDF5 file '{file}' to '{new_file}'...")
            os.rename(file, new_file)

            if clean:
                __remove_files([directory])

            continue

        # Check output directory
        output_directory = os.path.abspath(os.path.join(directory, '..'))

        # Check HDF5 file existence
        hdf5_opacity_file = os.path.join(output_directory, names[species] + '.ktable.petitRADTRANS.h5')

        if os.path.isfile(hdf5_opacity_file) and not rewrite:
            __print_skipping_message(hdf5_opacity_file)
            continue

        # Read dat file
        print(f"Converting opacities in '{directory}'...")

        custom_pt_grid_file = os.path.join(directory, 'PTpaths.ls')
        has_custom_grid = False
        opacities_temperature_profile_grid_ = None

        if os.path.isfile(custom_pt_grid_file):
            print(" Found custom PT grid")
            has_custom_grid = True

            # _sort_opa_pt_grid converts bar into cgs
            custom_grid_data = _sort_pressure_temperature_grid(custom_pt_grid_file)

            opacities_temperature_profile_grid_ = custom_grid_data[0]
            opacities_temperatures_ = np.unique(opacities_temperature_profile_grid_[:, 0])
            opacities_pressures_ = np.unique(opacities_temperature_profile_grid_[:, 1])
            opacities_pressures_ *= 1e-6  # cgs to bar
            line_paths_ = custom_grid_data[1]

            for i, line_path in enumerate(line_paths_):
                line_paths_[i] = line_path

            line_paths_ = np.array(line_paths_)
        else:
            print(" Using default PT grid")

            opacities_temperatures_ = copy.deepcopy(opacities_temperatures)
            opacities_pressures_ = copy.deepcopy(opacities_pressures)

            line_paths_ = []

            for f_ in os.scandir(directory):
                if f_.is_file():
                    line_paths_.append(f_.path)

            f_ = []

            for ref_path in line_paths:
                for i, line_path in enumerate(line_paths_):
                    if ref_path in line_path:
                        f_.append(line_paths_.pop(i))

                        break

            line_paths_ = np.array(f_)

            if line_paths_.size != line_paths.size:
                warnings.warn(f"number of opacity files founds in '{directory}' ({line_paths_.size}) "
                              f"does not match the expected number of files ({line_paths.size})")

        custom_file_names = ''

        if has_custom_grid:
            size_tp_grid = opacities_temperature_profile_grid_.shape[0]

            for i_TP in range(size_tp_grid):
                custom_file_names = custom_file_names + line_paths_[i_TP] + ':'
        else:
            size_tp_grid = opacities_pressures_.size * opacities_temperatures_.size

        # Convert units and shape
        _n_frequencies, _n_g = finput.load_frequencies_g_sizes(directory)
        _frequencies, frequency_bins_edges = finput.load_frequencies(directory, _n_frequencies)
        wavenumbers = _frequencies[::-1] / cst.c  # Hz to cm-1
        wavenumbers_bins_edges = frequency_bins_edges[::-1] / cst.c  # Hz to cm-1
        wavelengths = 1 / wavenumbers

        opacities = finput.load_line_opacity_grid(
            os.path.join(path_input_data, 'opa_input_files'),
            directory,
            species + ':',
            _n_frequencies,
            _n_g,
            1,
            size_tp_grid,
            'c-k',
            1,  # lbl start index, unused in this case
            has_custom_grid,
            custom_file_names
        )
        # Opacities are divided by isotopic ratio in loading function, there is no need to store it or use it
        cross_sections = opacities * molmass_dict[species] * cst.amu  # opacities to cross-sections

        print(" Reshaping...")
        # Exo-Mol axis order (pressures, temperatures, wavenumbers, g)
        cross_sections = cross_sections[:, :, 0, :]  # get rid of useless dimension
        cross_sections = cross_sections.reshape(
            (_n_g, _n_frequencies, opacities_temperatures_.size, opacities_pressures_.size)
        )
        cross_sections = np.swapaxes(cross_sections, 0, -1)
        cross_sections = np.swapaxes(cross_sections, 1, -2)
        cross_sections = cross_sections[:, :, ::-1, :]  # match the wavenumber order

        # Write converted file
        print(f" Writing file '{hdf5_opacity_file}'...", end=' ')

        write_correlated_k(
            file=hdf5_opacity_file,
            doi=doi_dict[species],
            wavenumbers=wavenumbers,
            wavenumbers_bins_edges=wavenumbers_bins_edges,
            cross_sections=cross_sections,
            mol_mass=molmass_dict[species],
            species=species,
            opacities_pressures=opacities_pressures_,
            opacities_temperatures=opacities_temperatures_,
            g_gauss=g_gauss,
            weights_gauss=weights_gauss,
            wavelengths=wavelengths,
            n_g=_n_g,
            contributor=contributor_dict[species],
            description=description_dict[species]
        )

        print("Done.")

        if clean:
            __remove_files([directory])

    print("Successfully converted correlated-k line opacities")


def _correlated_k_opacities_dat2h5_external_species(path_to_species_opacity_folder,
                                                    path_prt2_input_data,
                                                    longname,
                                                    doi=None,
                                                    contributor=None,
                                                    description=None,
                                                    molmass=None):
    _correlated_k_opacities_dat2h5(path_input_data=path_prt2_input_data,
                                   external_single_species=True,
                                   path_to_external_species_opacity_folder=path_to_species_opacity_folder,
                                   external_species_longname=longname,
                                   external_species_doi=doi,
                                   external_species_contributor=contributor,
                                   external_species_description=description,
                                   external_species_molmass=molmass)


def _get_base_cia_names():
    return LockedDict.build_and_lock({
        'H2--H2': 'H2--H2-NatAbund__BoRi.R831_0.6-250mu',
        'H2--He': 'H2--He-NatAbund__BoRi.DeltaWavenumber2_0.5-500mu',
        'H2O--H2O': 'H2O--H2O-NatAbund',
        'H2O--N2': 'H2O--N2-NatAbund',
        'N2--H2': 'N2--H2-NatAbund.DeltaWavenumber1_5.3-909mu',
        'N2--He': 'N2--He-NatAbund.DeltaWavenumber1_10-909mu',
        'N2--N2': 'N2--N2-NatAbund.DeltaWavelength1e-6_2-100mu',
        'O2--O2': 'O2--O2-NatAbund.DeltaWavelength1e-6_0.34-8.7mu',
        'N2--O2': 'N2--O2-NatAbund.DeltaWavelength1e-6_0.72-5.4mu',
        'CO2--CO2': 'CO2--CO2-NatAbund.DeltaWavelength1e-6_3-100mu',
    })


def _get_base_cloud_names():
    return LockedDict.build_and_lock({
        'Al2O3(s)_crystalline': 'Al2-O3-NatAbund(s)_crystalline_000.R39_0.1-250mu',
        'Fe(s)_amorphous': 'Fe-NatAbund(s)_amorphous.R39_0.1-250mu',
        'Fe(s)_crystalline': 'Fe-NatAbund(s)_crystalline_000.R39_0.1-250mu',
        'H2O(s)_crystalline': 'H2-O-NatAbund(s)_crystalline_000.R39_0.1-250mu',
        'H2O(l)': 'H2-O-NatAbund(l).R39_0.1-250mu',
        'H2OSO4(l)': 'H2-O-S-O4-NatAbund(l).R39_0.1-250mu',
        'KCl(s)_crystalline': 'K-Cl-NatAbund(s)_crystalline_000.R39_0.1-250mu',
        'Mg2SiO4(s)_amorphous': 'Mg2-Si-O4-NatAbund(s)_amorphous.R39_0.1-250mu',
        'Mg2SiO4(s)_crystalline': 'Mg2-Si-O4-NatAbund(s)_crystalline_000.R39_0.1-250mu',
        'Mg05Fe05SiO3(s)_amorphous': 'Mg05-Fe05-Si-O3-NatAbund(s)_amorphous.R39_0.1-250mu',
        'MgAl2O4(s)_crystalline': 'Mg-Al2-O4-NatAbund(s)_amorphous.R39_0.1-250mu',
        'MgFeSiO4(s)_amorphous': 'Mg-Fe-Si-O4-NatAbund(s)_amorphous.R39_0.2-250mu',
        'MgSiO3(s)_amorphous': 'Mg-Si-O3-NatAbund(s)_amorphous.R39_0.1-250mu',
        'MgSiO3(s)_crystalline': 'Mg-Si-O3-NatAbund(s)_crystalline_000.R39_0.1-250mu',
        'Na2S(s)_crystalline': 'Na2-S-NatAbund(s)_crystalline_000.R39_0.1-250mu',
        'SiC(s)_crystalline': 'Si-C-NatAbund(s)_crystalline_000.R39_0.1-250mu'
    })


def _get_base_correlated_k_names():
    """Only used for file conversion from pRT2 to pRT3."""
    return LockedDict.build_and_lock({
        'Al': '27Al__Kurucz.R1000_0.1-250mu',
        'Al+': '27Al_p__Kurucz.R1000_0.1-250mu',
        'AlH': '27Al-1H__AlHambra.R1000_0.3-50mu',
        'AlO': '27Al-16O__ATP.R1000_0.3-50mu',
        'C2H2': '12C2-1H2__aCeTY.R1000_0.3-50mu',
        'C2H4': '12C2-1H4__MaYTY.R1000_0.3-50mu',
        'Ca': '40Ca__Kurucz.R1000_0.1-250mu',
        'Ca+': '40Ca_p__Kurucz.R1000_0.1-250mu',
        'CaH': '40Ca-1H__MoLLIST.R1000_0.3-50mu',
        'CH4': '12C-1H4__YT34to10.R1000_0.3-50mu',
        'CO': '12C-16O__HITEMP.R1000_0.1-250mu',
        '13CO': '13C-16O__HITEMP.R1000_0.1-250mu',
        'CO_all_iso': 'C-O-NatAbund__HITEMP.R1000_0.1-250mu',
        'CO2': '12C-16O2__UCL-4000.R1000_0.3-50mu',
        'CrH': '52Cr-1H__MoLLIST.R1000_0.3-50mu',
        'Fe': '56Fe__Kurucz.R1000_0.1-250mu',
        'Fe+': '56Fe_p__Kurucz.R1000_0.1-250mu',
        'FeH': '56Fe-1H__MoLLIST.R1000_0.3-50mu',
        'H2O': '1H2-16O__HITEMP.R1000_0.1-250mu',
        'H2S': '1H2-32S__AYT2.R1000_0.3-50mu',
        'HCN': '1H-12C-14N__Harris.R1000_0.3-50mu',
        'K': '39K__Allard.R1000_0.1-250mu',
        'Li': '7Li__Kurucz.R1000_0.1-250mu',
        'Mg': '24Mg__Kurucz.R1000_0.1-250mu',
        'Mg+': '24Mg_p__Kurucz.R1000_0.1-250mu',
        'MgH': '24Mg-1H__MoLLIST.R1000_0.3-50mu',
        'MgO': '24Mg-16O__LiTY.R1000_0.3-50mu',
        'Na': '23Na__Allard.R1000_0.1-250mu',
        'NaH': '23Na-1H__Rivlin.R1000_0.3-50mu',
        'NH3': '14N-1H3__CoYuTe.R1000_0.3-50mu',
        'O': '16O__Kurucz.R1000_0.1-250mu',
        'O+': '16O_p__Kurucz.R1000_0.1-250mu',  # TODO not in the docs
        'O2': '16O2__HITRAN.R1000_0.3-50mu',
        'O3': '16O3__HITRAN.R1000_0.1-250mu',
        'OH': '16O-1H__HITEMP.R1000_0.3-50mu',
        'PH3': '31P-1H3__SAlTY.R1000_0.3-50mu',
        'SH': '32S-1H__GYT.R1000_0.3-50mu',
        'Si': '28Si__Kurucz.R1000_0.1-250mu',
        'Si+': '28Si_p__Kurucz.R1000_0.1-250mu',
        'SiO': '28Si-16O__EBJT.R1000_0.3-50mu',
        'SiO2': '28Si-16O2__OYT3.R1000_0.3-50mu',
        'Ti': '48Ti__Kurucz.R1000_0.1-250mu',
        'Ti+': '48Ti_p__Kurucz.R1000_0.1-250mu',
        '48TiO': '48Ti-16O__Plez.R1000_0.1-250mu',
        'TiO_all_iso': 'Ti-O-NatAbund__Plez.R1000_0.1-250mu',
        'V': '51V__Kurucz.R1000_0.1-250mu',
        'V+': '51V_p__Kurucz.R1000_0.1-250mu',
        'VO': '51V-16O__VOMYT.R1000_0.3-50mu'
    })


def _get_base_line_by_line_names():
    """Only used for file conversion from pRT2 to pRT3."""
    return LockedDict.build_and_lock({
        'Al': '27Al__Kurucz.R1e6_0.3-28mu',
        'B': '11B__Kurucz.R1e6_0.3-28mu',
        'Be': '9Be__Kurucz.R1e6_0.3-28mu',
        'C2H2': '12C2-1H2__HITRAN.R1e6_0.3-28mu',
        'Ca': '40Ca__Kurucz.R1e6_0.3-28mu',
        'Ca+': '40Ca_p__Kurucz.R1e6_0.3-28mu',
        'CaH': '40Ca-1H__MoLLIST.R1e6_0.3-28mu',  # TODO not in docs
        'CH3D': '12C-1H3-2H__HITRAN.R1e6_0.3-28mu',
        'CH4': '12C-1H4__Hargreaves.R1e6_0.3-28mu',
        'CO2': '12C-16O2__HITEMP.R1e6_0.3-28mu',
        'C-17O': '12C-17O__HITRAN.R1e6_0.3-28mu',
        'C-18O': '12C-18O__HITRAN.R1e6_0.3-28mu',
        '13CO': '13C-16O__HITRAN.R1e6_0.3-28mu',
        '13C-17O': '13C-17O__HITRAN.R1e6_0.3-28mu',
        '13C-18O': '13C-18O__HITRAN.R1e6_0.3-28mu',
        'CO_all_iso': 'C-O-NatAbund__HITEMP.R1e6_0.3-28mu',
        'CO': '12C-16O__HITEMP.R1e6_0.3-28mu',
        'Cr': '52Cr__Kurucz.R1e6_0.3-28mu',
        'Fe': '56Fe__Kurucz.R1e6_0.3-28mu',
        'Fe+': '56Fe_p__Kurucz.R1e6_0.3-28mu',
        'FeH': '56Fe-1H__MoLLIST.R1e6_0.3-28mu',
        'HD': '1H-2H__HITRAN.R1e6_0.3-28mu',
        'H2': '1H2__HITRAN.R1e6_0.3-28mu',
        'HDO': '1H-2H-16O__HITEMP.R1e6_0.3-28mu',
        'H2-17O': '1H2-17O__HITEMP.R1e6_0.3-28mu',
        'HD-17O': '1H-2H-17O__HITEMP.R1e6_0.3-28mu',
        'H2-18O': '1H2-18O__HITEMP.R1e6_0.3-28mu',
        'HD-18O': '1H-2H-18O__HITEMP.R1e6_0.3-28mu',
        'H2O': '1H2-16O__HITEMP.R1e6_0.3-28mu',
        'H2S_main_iso': '1H2-32S__HITRAN.R1e6_0.3-28mu',
        'HCN_main_iso': '1H-12C-14N__Harris.R1e6_0.3-28mu',
        'K': '39K__Allard.R1e6_0.3-28mu',
        'Li': '7Li__Kurucz.R1e6_0.3-28mu',
        'Mg': '24Mg__Kurucz.R1e6_0.3-28mu',
        'Mg+': '24Mg_p__Kurucz.R1e6_0.3-28mu',
        'N': '14N__Kurucz.R1e6_0.3-28mu',
        'Na': '23Na__Allard.R1e6_0.3-28mu',
        'NH3': '14N-1H3__BYTe.R1e6_0.3-28mu',
        'O3': '16O3__HITRAN.R1e6_0.3-28mu',
        'OH': '16O-1H__HITEMP.R1e6_0.3-28mu',
        'PH3': '31P-1H3__SAlTY.R1e6_0.3-28mu',
        'Si': '28Si__Kurucz.R1e6_0.3-28mu',
        'SiO': '28Si-16O__EBJT.R1e6_0.3-28mu',
        'Ti': '48Ti__Kurucz.R1e6_0.3-28mu',
        '46TiO': '46Ti-16O__Plez.R1e6_0.3-28mu',
        '47TiO': '47Ti-16O__Plez.R1e6_0.3-28mu',
        'TiO': '48Ti-16O__Plez.R1e6_0.3-28mu',
        '49TiO': '49Ti-16O__Plez.R1e6_0.3-28mu',
        '50TiO': '50Ti-16O__Plez.R1e6_0.3-28mu',
        'TiO_all_iso': 'Ti-O-NatAbund__Plez.R1e6_0.3-28mu',
        'V': '51V__Kurucz.R1e6_0.3-28mu',
        'V+': '51V_p__Kurucz.R1e6_0.3-28mu',
        'VO': '51V-16O__Plez.R1e6_0.3-28mu',
        'Y': '89Y__Kurucz.R1e6_0.3-28mu'
    })


def _get_default_rebinning_wavelength_range():
    return np.array([0.1, 251.0])  # um


def _get_prt2_cia_names():
    return LockedDict.build_and_lock({
        'H2-H2': 'H2--H2',
        'H2-He': 'H2--He',
        'H2O-H2O': 'H2O--H2O',  # TODO not in default input_data
        'H2O-N2': 'H2O--N2',  # TODO not in default input_data
        'N2-H2': 'N2--H2',
        'N2-He': 'N2--He',
        'N2-N2': 'N2--N2',
        'O2-O2': 'O2--O2',
        'N2-O2': 'N2--O2',
        'CO2-CO2': 'CO2--CO2',
    })


def _get_prt2_cloud_names():
    return LockedDict.build_and_lock({
        'Al2O3(c)_cm': 'Al2O3(s)_crystalline',
        'Al2O3(c)_cd': 'Al2O3(s)_crystalline',
        'Fe(c)_am': 'Fe(s)_amorphous',
        'Fe(c)_ad': 'Fe(s)_amorphous',
        'Fe(c)_cm': 'Fe(s)_crystalline',
        'Fe(c)_cd': 'Fe(s)_crystalline',
        'H2O(c)_cm': 'H2O(s)_crystalline',  # TODO not in the docs
        'H2O(c)_cd': 'H2O(s)_crystalline',  # TODO not in the docs
        'H2OL(c)_am': 'H2O(l)',  # TODO not in the docs
        'H2OSO425(c)_am': 'H2OSO425(l)',  # TODO not in the docs
        'H2OSO450(c)_am': 'H2OSO450(l)',  # TODO not in the docs
        'H2OSO475(c)_am': 'H2OSO475(l)',  # TODO not in the docs
        'H2OSO484(c)_am': 'H2OSO484(l)',  # TODO not in the docs
        'H2OSO495(c)_am': 'H2OSO495(l)',  # TODO not in the docs
        'KCL(c)_cm': 'KCl(s)_crystalline',
        'KCL(c)_cd': 'KCl(s)_crystalline',
        'Mg2SiO4(c)_am': 'Mg2SiO4(s)_amorphous',
        'Mg2SiO4(c)_ad': 'Mg2SiO4(s)_amorphous',
        'Mg2SiO4(c)_cm': 'Mg2SiO4(s)_crystalline',
        'Mg2SiO4(c)_cd': 'Mg2SiO4(s)_crystalline',
        'Mg05Fe05SiO3(c)_am': 'Mg05Fe05SiO3(s)_amorphous',
        'Mg05Fe05SiO3(c)_ad': 'Mg05Fe05SiO3(s)_amorphous',
        'MgAl2O4(c)_cm': 'MgAl2O4(s)_crystalline',
        'MgAl2O4(c)_cd': 'MgAl2O4(s)_crystalline',
        'MgFeSiO4(c)_am': 'MgFeSiO4(s)_amorphous',
        'MgFeSiO4(c)_ad': 'MgFeSiO4(s)_amorphous',
        'MgSiO3(c)_am': 'MgSiO3(s)_amorphous',
        'MgSiO3(c)_ad': 'MgSiO3(s)_amorphous',
        'MgSiO3(c)_cm': 'MgSiO3(s)_crystalline',
        'MgSiO3(c)_cd': 'MgSiO3(s)_crystalline',
        'Na2S(c)_cm': 'Na2S(s)_crystalline',
        'Na2S(c)_cd': 'Na2S(s)_crystalline',
        'SiC(c)_cm': 'SiC(s)_crystalline',
        'SiC(c)_cd': 'SiC(s)_crystalline'
    })


def _get_prt2_correlated_k_names():
    return LockedDict.build_and_lock({
        'Al': None,
        'Al+': None,
        'AlH': None,
        'AlO': None,
        'C2H2': None,
        'C2H4': None,
        'Ca': None,
        'Ca+': None,
        'CaH': None,
        'CH4': None,
        'CO_12_HITEMP': '12C-16O__HITEMP.R1000_0.1-250mu',
        'CO_13_HITEMP': '13C-16O__HITEMP.R1000_0.1-250mu',
        'CO_13_Chubb': '13C-16O__Li2015.R1000_0.3-50mu',
        'CO_all_iso_Chubb': 'C-O-NatAbund__Chubb.R1000_0.3-50mu',
        'CO_all_iso_HITEMP': 'C-O-NatAbund__HITEMP.R1000_0.1-250mu',
        'CO2': None,
        'CrH': None,
        'Fe': None,
        'Fe+': None,
        'FeH': None,
        'H2O_Exomol': '1H2-16O__POKAZATEL.R1000_0.3-50mu',
        'H2O_HITEMP': '1H2-16O__HITEMP.R1000_0.1-250mu',
        'H2S': None,
        'HCN': None,
        'K_allard': '39K__Allard.R1000_0.1-250mu',
        'K_burrows': '39K__Burrows.R1000_0.1-250mu',
        'K_lor_cut': '39K__LorCut.R1000_0.1-250mu',
        'Li': None,
        'Mg': None,
        'Mg+': None,
        'MgH': None,
        'MgO': None,
        'Na_allard': '23Na__Allard.R1000_0.1-250mu',
        'Na_burrows': '23Na__Burrows.R1000_0.1-250mu',
        'Na_lor_cut': '23Na__LorCut.R1000_0.1-250mu',
        'NaH': None,
        'NH3': None,
        'O': None,
        'O+': None,  # TODO not in the docs
        'O2': None,
        'O3': None,
        'OH': None,
        'PH3': None,
        'SH': None,
        'Si': None,
        'Si+': None,
        'SiO': None,
        'SiO2': None,
        'Ti': None,
        'Ti+': None,
        'TiO_48_Exomol': '48Ti-16O__McKemmish.R1000_0.1-250mu',
        'TiO_48_Plez': '48Ti-16O__Plez.R1000_0.1-250mu',
        'TiO_all_Exomol': 'Ti-O-NatAbund__McKemmish.R1000_0.1-250mu',
        'TiO_all_Plez': 'Ti-O-NatAbund__Plez.R1000_0.1-250mu',
        'V': None,
        'V+': None,
        'VO': None,
        'VO_Plez': '51V-16O__Plez.R1000_0.1-250mu'
    })


def _get_prt2_line_by_line_names():
    return LockedDict.build_and_lock({
        '13CH4': '13C-1H4__HITRAN.R1e6_0.3-28mu',
        '15NH3': '15N-1H3__HITRAN.R1e6_0.3-28mu',
        'Al': None,
        'B': None,
        'Be': None,
        'C2H2_main_iso': '12C2-1H2__HITRAN.R1e6_0.3-28mu',
        'Ca': None,
        'Ca+': None,
        'CaH': None,
        'CH4_212': '12C-1H3-2H__HITRAN.R1e6_0.3-28mu',
        'CH4_Hargreaves_main_iso': '12C-1H4__Hargreaves.R1e6_0.3-28mu',
        'CH4_main_iso': '12C-1H4__Molliere.R1e6_0.3-28mu',  # TODO not in docs
        'CO2_main_iso': '12C-16O2__HITEMP.R1e6_0.3-28mu',
        'CO_27': '12C-17O__HITRAN.R1e6_0.3-28mu',
        'CO_28': '12C-18O__HITRAN.R1e6_0.3-28mu',
        'CO_36': '13C-16O__HITRAN.R1e6_0.3-28mu',
        'CO_37': '13C-17O__HITRAN.R1e6_0.3-28mu',
        'CO_38': '13C-18O__HITRAN.R1e6_0.3-28mu',
        'CO_all_iso': None,
        'CO_main_iso': '12C-16O__HITEMP.R1e6_0.3-28mu',
        'Cr': None,
        'Fe': None,
        'Fe+': None,
        'FeH_main_iso': '56Fe-1H__MoLLIST.R1e6_0.3-28mu',
        'H2_12': '1H-2H__HITRAN.R1e6_0.3-28mu',
        'H2_main_iso': '1H2__HITRAN.R1e6_0.3-28mu',
        'H217O_HITRAN': '1H2-17O__HITRAN.R1e6_0.3-28mu',
        'H2O_162': '1H-2H-16O__HITEMP.R1e6_0.3-28mu',
        'H2O_171': '1H2-17O__HITEMP.R1e6_0.3-28mu',
        'H2O_172': '1H-2H-17O__HITEMP.R1e6_0.3-28mu',
        'H2O_181': '1H2-18O__HITEMP.R1e6_0.3-28mu',
        'H2O_182': '1H-2H-18O__HITEMP.R1e6_0.3-28mu',
        'H2O_main_iso': '1H2-16O__HITEMP.R1e6_0.3-28mu',
        'H2O_pokazatel_main_iso': '1H2-16O__POKAZATEL.R1e6_0.3-28mu',
        'H2S_main_iso': '1H2-32S__HITRAN.R1e6_0.3-28mu',
        'HCN_main_iso': '1H-12C-14N__Harris.R1e6_0.3-28mu',
        'K': '39K__Allard.R1e6_0.3-28mu',
        'K_allard_cold': '39K__Allard.R1e6_0.3-28mu',
        'K_burrows': '39K__Burrows.R1e6_0.3-28mu',
        'K_lor_cut': '39K__LorCut.R1e6_0.3-28mu',
        'Li': None,
        'Mg': None,
        'Mg+': None,
        'N': None,
        'Na_allard': '23Na__AllardOld.R1e6_0.3-28mu',
        'Na_allard_new': '23Na__Allard.R1e6_0.3-28mu',
        'Na_burrows': '23Na__Burrows.R1e6_0.3-28mu',
        'Na_lor_cut': '23Na__LorCut.R1e6_0.3-28mu',
        'NH3_main_iso': '14N-1H3__BYTe.R1e6_0.3-28mu',
        'NH3_main_iso_HITRAN': '14N-1H3__HITRAN.R1e6_0.3-28mu',
        'NH3_Coles_main_iso': '14N-1H3__CoYuTe.R1e6_0.3-28mu',
        'O3_main_iso': None,
        'OH_main_iso': None,
        'PH3_SAlTY': '31P-1H3__SAlTY.R1e6_0.3-28mu',
        'PH3_main_iso': None,
        'Si': None,
        'SiO_main_iso': None,
        'Ti': None,
        'TiO_46_Exomol_McKemmish': '46Ti-16O__Toto.R1e6_0.3-28mu',
        'TiO_46_Plez': '46Ti-16O__Plez.R1e6_0.3-28mu',
        'TiO_47_Exomol_McKemmish': '47Ti-16O__Toto.R1e6_0.3-28mu',
        'TiO_47_Plez': '47Ti-16O__Plez.R1e6_0.3-28mu',
        'TiO_48_Exomol_McKemmish': '48Ti-16O__Toto.R1e6_0.3-28mu',
        'TiO_48_Plez': '48Ti-16O__Plez.R1e6_0.3-28mu',
        'TiO_49_Exomol_McKemmish': '49Ti-16O__Toto.R1e6_0.3-28mu',
        'TiO_49_Plez': '49Ti-16O__Plez.R1e6_0.3-28mu',
        'TiO_50_Exomol_McKemmish': '50Ti-16O__Toto.R1e6_0.3-28mu',
        'TiO_50_Plez': '50Ti-16O__Plez.R1e6_0.3-28mu',
        'TiO_all_iso_Exomol_McKemmish': 'Ti-O-NatAbund__TotoMcKemmish.R1e6_0.3-28mu',
        'TiO_all_iso_Plez': 'Ti-O-NatAbund__Plez.R1e6_0.3-28mu',
        'TiO_all_iso_exo': 'Ti-O-NatAbund__Toto.R1e6_0.3-28mu',
        'V': None,
        'V+': None,
        'VO': '51V-16O__Plez.R1e6_0.3-28mu',
        'VO_ExoMol_McKemmish': '51V-16O__VOMYT.R1e6_0.3-28mu',
        'VO_ExoMol_Specific_Transitions': '51V-16O__VOMYTSpe.R1e6_0.3-28mu',
        'Y': None
    })


def _line_by_line_opacities_dat2h5(path_input_data=petitradtrans_config_parser.get_input_data_path(),
                                   memory_map_mode=False, rewrite=False, old_paths=False, clean=False):
    """Using ExoMol units for HDF5 files."""

    # Initialize infos
    kurucz_website = 'http://kurucz.harvard.edu/'
    molliere2019_doi = '10.1051/0004-6361/201935470'
    burrows2003_doi = '10.1086/345412'
    vald_website = 'http://vald.astro.uu.se/'

    molaverdikhani_email = 'karan.molaverdikhani@colorado.edu'

    kurucz_description = 'gamma_nat + V dW, sigma_therm'

    doi_dict = _get_prt2_line_by_line_names()
    contributor_dict = copy.deepcopy(doi_dict)
    description_dict = copy.deepcopy(doi_dict)
    molmass_dict = copy.deepcopy(doi_dict)

    doi_dict.update({
        '13CH4': '10.1016/j.jqsrt.2021.107949',
        '15NH3': '10.1016/j.jqsrt.2021.107949',
        'Al': kurucz_website,
        'B': kurucz_website,
        'Be': kurucz_website,
        'C2H2_main_iso': molliere2019_doi,
        'Ca': kurucz_website,
        'Ca+': kurucz_website,
        'CaH': 'unknown',  # TODO not in the docs
        'CH4_212': molliere2019_doi,  # TODO not in the referenced paper
        'CH4_Hargreaves_main_iso': '10.3847/1538-4365/ab7a1a',
        'CH4_main_iso': 'unknown',  # TODO not in the docs (outdated)
        'CO2_main_iso': molliere2019_doi,
        'CO_27': molliere2019_doi,  # TODO not in the referenced paper
        'CO_28': molliere2019_doi,  # TODO not in the referenced paper
        'CO_36': molliere2019_doi,  # TODO not in the referenced paper
        'CO_37': molliere2019_doi,  # TODO not in the referenced paper
        'CO_38': molliere2019_doi,  # TODO not in the referenced paper
        'CO_all_iso': molliere2019_doi,
        'CO_main_iso': molliere2019_doi,  # TODO not clear in the referenced paper nor in the docs
        'Cr': kurucz_website,
        'Fe': kurucz_website,
        'Fe+': kurucz_website,
        'FeH_main_iso': molliere2019_doi,  # TODO not in the docs
        'H2_12': molliere2019_doi,  # TODO not in the referenced paper
        'H2_main_iso': molliere2019_doi,
        'H217O_HITRAN': '10.1016/j.jqsrt.2013.07.002',
        'H2O_162': '10.1016/j.jqsrt.2013.07.002',  # TODO not in the referenced paper
        'H2O_171': '10.1016/j.jqsrt.2013.07.002',  # TODO not in the referenced paper
        'H2O_172': '10.1016/j.jqsrt.2013.07.002',  # TODO not in the referenced paper
        'H2O_181': '10.1016/j.jqsrt.2013.07.002',  # TODO not in the referenced paper
        'H2O_182': '10.1016/j.jqsrt.2013.07.002',  # TODO not in the referenced paper
        'H2O_main_iso': molliere2019_doi,
        'H2O_pokazatel_main_iso': '10.1093/mnras/sty1877',
        'H2S_main_iso': molliere2019_doi,
        'HCN_main_iso': molliere2019_doi,
        'K': molliere2019_doi,
        'K_allard_cold': molliere2019_doi,  # TODO not in the referenced paper nor in the docs
        'K_lor_cut': vald_website,  # TODO not in the referenced paper nor in the docs
        'K_burrows': burrows2003_doi,
        'Li': kurucz_website,
        'Mg': kurucz_website,
        'Mg+': kurucz_website,
        'N': kurucz_website,
        'Na_allard': '10.1051/0004-6361/201935593',  # incorrect mass
        'Na_allard_new': '10.1051/0004-6361/201935593',  # same as above but with the correct mass
        'Na_lor_cut': vald_website,
        'Na_burrows': burrows2003_doi,
        'NH3_main_iso': 'unknown',  # TODO referenced twice in the docs! Which one is it?
        'NH3_main_iso_HITRAN': '10.1016/j.jqsrt.2013.07.002',
        'NH3_Coles_main_iso': '10.1093/mnras/stz2778',
        'O3_main_iso': molliere2019_doi,
        'OH_main_iso': molliere2019_doi,
        'PH3_main_iso': '10.1093/mnras/stu2246',
        'Si': kurucz_website,
        'SiO_main_iso': '10.1093/mnras/stt1105',
        'Ti': kurucz_website,
        'TiO_46_Exomol_McKemmish': '10.1093/mnras/stz1818',
        'TiO_46_Plez': molliere2019_doi,
        'TiO_47_Exomol_McKemmish': '10.1093/mnras/stz1818',
        'TiO_47_Plez': molliere2019_doi,
        'TiO_48_Exomol_McKemmish': '10.1093/mnras/stz1818',
        'TiO_48_Plez': molliere2019_doi,
        'TiO_49_Exomol_McKemmish': '10.1093/mnras/stz1818',
        'TiO_49_Plez': molliere2019_doi,
        'TiO_50_Exomol_McKemmish': '10.1093/mnras/stz1818',
        'TiO_50_Plez': molliere2019_doi,
        'TiO_all_iso_Exomol_McKemmish': '10.1093/mnras/stz1818',
        'TiO_all_iso_Plez': molliere2019_doi,
        'TiO_all_iso_exo': molliere2019_doi,
        'V': kurucz_website,
        'V+': kurucz_website,
        'VO': molliere2019_doi,
        'VO_ExoMol_McKemmish': '10.1093/mnras/stw1969',
        'VO_ExoMol_Specific_Transitions': '10.1093/mnras/stw1969',  # TODO difference unclear with "default" version
        'Y': kurucz_website
    })
    contributor_dict.update({
        '13CH4': 'None',
        '15NH3': 'None',
        'Al': molaverdikhani_email,
        'B': molaverdikhani_email,
        'Be': molaverdikhani_email,
        'C2H2_main_iso': 'None',
        'Ca': molaverdikhani_email,
        'Ca+': molaverdikhani_email,
        'CaH': 'None',
        'CH4_212': 'None',
        'CH4_Hargreaves_main_iso': 'None',
        'CH4_main_iso': 'None',
        'CO2_main_iso': 'None',
        'CO_27': 'None',
        'CO_28': 'None',
        'CO_36': 'None',
        'CO_37': 'None',
        'CO_38': 'None',
        'CO_all_iso': 'None',
        'CO_main_iso': 'None',
        'Cr': molaverdikhani_email,
        'Fe': molaverdikhani_email,
        'Fe+': molaverdikhani_email,
        'FeH_main_iso': 'None',
        'H2_12': 'None',
        'H2_main_iso': 'None',
        'H217O_HITRAN': 'None',
        'H2O_162': 'None',
        'H2O_171': 'None',
        'H2O_172': 'None',
        'H2O_181': 'None',
        'H2O_182': 'None',
        'H2O_main_iso': 'None',
        'H2O_pokazatel_main_iso': 'gandhi@strw.leidenuniv.nl',
        'H2S_main_iso': 'None',
        'HCN_main_iso': 'None',
        'K': 'None',
        'K_allard_cold': 'None',
        'K_burrows': 'None',
        'K_lor_cut': 'None',
        'Li': molaverdikhani_email,
        'Mg': molaverdikhani_email,
        'Mg+': molaverdikhani_email,
        'N': molaverdikhani_email,
        'Na_allard': 'None',
        'Na_allard_new': 'None',
        'Na_burrows': 'None',
        'Na_lor_cut': 'None',
        'NH3_main_iso': 'None',
        'NH3_main_iso_HITRAN': 'None',
        'NH3_Coles_main_iso': 'gandhi@strw.leidenuniv.nl',
        'O3_main_iso': 'None',
        'OH_main_iso': 'None',
        'PH3_main_iso': 'adriano.miceli@stud.unifi.it',
        'Si': molaverdikhani_email,
        'SiO_main_iso': 'None',
        'Ti': molaverdikhani_email,
        'TiO_46_Exomol_McKemmish': 'None',
        'TiO_46_Plez': 'None',
        'TiO_47_Exomol_McKemmish': 'None',
        'TiO_47_Plez': 'None',
        'TiO_48_Exomol_McKemmish': 'None',
        'TiO_48_Plez': 'None',
        'TiO_49_Exomol_McKemmish': 'None',
        'TiO_49_Plez': 'None',
        'TiO_50_Exomol_McKemmish': 'None',
        'TiO_50_Plez': 'None',
        'TiO_all_iso_Exomol_McKemmish': 'None',
        'TiO_all_iso_Plez': 'None',
        'TiO_all_iso_exo': 'None',
        'V': molaverdikhani_email,
        'V+': molaverdikhani_email,
        'VO': 'None',
        'VO_ExoMol_McKemmish': 'regt@strw.leidenuniv.nl',
        'VO_ExoMol_Specific_Transitions': 'regt@strw.leidenuniv.nl',
        'Y': molaverdikhani_email
    })
    description_dict.update({
        '13CH4': 'None',
        '15NH3': 'None',
        'Al': kurucz_description,
        'B': kurucz_description,
        'Be': kurucz_description,
        'C2H2_main_iso': 'None',
        'Ca': kurucz_description,
        'Ca+': kurucz_description,
        'CaH': 'None',
        'CH4_212': 'None',
        'CH4_Hargreaves_main_iso': 'None',
        'CH4_main_iso': 'None',
        'CO2_main_iso': 'None',
        'CO_27': 'None',
        'CO_28': 'None',
        'CO_36': 'None',
        'CO_37': 'None',
        'CO_38': 'None',
        'CO_all_iso': 'None',
        'CO_main_iso': 'None',
        'Cr': kurucz_description,
        'Fe': kurucz_description,
        'Fe+': kurucz_description,
        'FeH_main_iso': 'None',
        'H2_12': 'None',
        'H2_main_iso': 'None',
        'H217O_HITRAN': "Using HITRAN's air broadening prescription",
        'H2O_162': "Using HITRAN's air broadening prescription",
        'H2O_171': "Using HITRAN's air broadening prescription",
        'H2O_172': "Using HITRAN's air broadening prescription",
        'H2O_181': "Using HITRAN's air broadening prescription",
        'H2O_182': "Using HITRAN's air broadening prescription",
        'H2O_main_iso': 'None',
        'H2O_pokazatel_main_iso': 'None',
        'H2S_main_iso': 'None',
        'HCN_main_iso': 'None',
        'K': 'None',
        'K_allard_cold': "Using Allard wings",
        'K_burrows': 'None',
        'K_lor_cut': 'Using Voigt wings with cutoff at 4500 cm^-1',
        'Li': kurucz_description,
        'Mg': kurucz_description,
        'Mg+': kurucz_description,
        'N': kurucz_description,
        'Na_allard': 'None',
        'Na_allard_new': 'Using Allard wings',
        'Na_burrows': 'None',
        'Na_lor_cut': 'Using Voigt wings with cutoff at 4500 cm^-1',
        'NH3_main_iso': 'None',
        'NH3_main_iso_HITRAN': 'None',
        'NH3_Coles_main_iso': 'None',
        'O3_main_iso': 'None',
        'OH_main_iso': 'None',
        'PH3_main_iso': 'None',
        'Si': kurucz_description,
        'SiO_main_iso': 'None',
        'Ti': kurucz_description,
        'TiO_46_Exomol_McKemmish': 'None',
        'TiO_46_Plez': 'None',
        'TiO_47_Exomol_McKemmish': 'None',
        'TiO_47_Plez': 'None',
        'TiO_48_Exomol_McKemmish': 'None',
        'TiO_48_Plez': 'None',
        'TiO_49_Exomol_McKemmish': 'None',
        'TiO_49_Plez': 'None',
        'TiO_50_Exomol_McKemmish': 'None',
        'TiO_50_Plez': 'None',
        'TiO_all_iso_Exomol_McKemmish': 'None',
        'TiO_all_iso_Plez': 'None',
        'TiO_all_iso_exo': 'None',
        'V': kurucz_description,
        'V+': kurucz_description,
        'VO': 'None',
        'VO_ExoMol_McKemmish': 'None',
        'VO_ExoMol_Specific_Transitions': 'Most accurate transitions from McKemmish et al. (2016)',
        'Y': kurucz_description
    })
    molmass_dict.update({
        '13CH4': get_species_molar_mass('13C') + get_species_molar_mass('1H4'),
        '15NH3': get_species_molar_mass('15N') + get_species_molar_mass('1H3'),
        'Al': get_species_molar_mass('Al'),
        'B': get_species_molar_mass('B'),
        'Be': get_species_molar_mass('Be'),
        'C2H2_main_iso': get_species_molar_mass('C2H2'),
        'Ca': get_species_molar_mass('Ca'),
        'Ca+': get_species_molar_mass('Ca') - get_species_molar_mass('e-'),
        'CaH': get_species_molar_mass('CaH'),
        'CH4_212': get_species_molar_mass('CH3D'),
        'CH4_Hargreaves_main_iso': get_species_molar_mass('CH4'),
        'CH4_main_iso': get_species_molar_mass('CH4'),
        'CO2_main_iso': get_species_molar_mass('CO2'),
        'CO_27': get_species_molar_mass('12C') + get_species_molar_mass('17O'),
        'CO_28': get_species_molar_mass('12C') + get_species_molar_mass('18O'),
        'CO_36': get_species_molar_mass('13C') + get_species_molar_mass('16O'),
        'CO_37': get_species_molar_mass('13C') + get_species_molar_mass('17O'),
        'CO_38': get_species_molar_mass('13C') + get_species_molar_mass('18O'),
        'CO_all_iso': get_species_molar_mass('CO_all_iso'),
        'CO_main_iso': get_species_molar_mass('CO'),
        'Cr': get_species_molar_mass('Cr'),
        'Fe': get_species_molar_mass('Fe'),
        'Fe+': get_species_molar_mass('Fe') - get_species_molar_mass('e-'),
        'FeH_main_iso': get_species_molar_mass('FeH'),
        'H2_12': get_species_molar_mass('HD'),
        'H2_main_iso': get_species_molar_mass('H2'),
        'H217O_HITRAN': get_species_molar_mass('1H') + get_species_molar_mass('17O') + get_species_molar_mass('1H'),
        'H2O_162': get_species_molar_mass('1H') + get_species_molar_mass('16O') + get_species_molar_mass('2H'),
        'H2O_171': get_species_molar_mass('1H') + get_species_molar_mass('17O') + get_species_molar_mass('1H'),
        'H2O_172': get_species_molar_mass('1H') + get_species_molar_mass('17O') + get_species_molar_mass('2H'),
        'H2O_181': get_species_molar_mass('1H') + get_species_molar_mass('18O') + get_species_molar_mass('1H'),
        'H2O_182': get_species_molar_mass('1H') + get_species_molar_mass('18O') + get_species_molar_mass('2H'),
        'H2O_main_iso': get_species_molar_mass('H2O'),
        'H2O_pokazatel_main_iso': get_species_molar_mass('H2O'),
        'H2S_main_iso': get_species_molar_mass('H2S'),
        'HCN_main_iso': get_species_molar_mass('HCN'),
        'K': get_species_molar_mass('K'),
        'K_allard_cold': get_species_molar_mass('K'),
        'K_burrows': get_species_molar_mass('K'),
        'K_lor_cut': get_species_molar_mass('K'),
        'Li': get_species_molar_mass('Li'),
        'Mg': get_species_molar_mass('Mg'),
        'Mg+': get_species_molar_mass('Mg') - get_species_molar_mass('e-'),
        'N': get_species_molar_mass('N'),
        'Na_allard': get_species_molar_mass('Na'),
        'Na_allard_new': get_species_molar_mass('Na'),
        'Na_burrows': get_species_molar_mass('Na'),
        'Na_lor_cut': get_species_molar_mass('Na'),
        'NH3_main_iso': get_species_molar_mass('NH3'),
        'NH3_main_iso_HITRAN': get_species_molar_mass('NH3'),
        'NH3_Coles_main_iso': get_species_molar_mass('NH3'),
        'O3_main_iso': get_species_molar_mass('O3'),
        'OH_main_iso': get_species_molar_mass('OH'),
        'PH3_main_iso': get_species_molar_mass('PH3'),
        'Si': get_species_molar_mass('Si'),
        'SiO_main_iso': get_species_molar_mass('SiO'),
        'Ti': get_species_molar_mass('Ti'),
        'TiO_46_Exomol_McKemmish': get_species_molar_mass('46Ti') + get_species_molar_mass('16O'),
        'TiO_46_Plez': get_species_molar_mass('46Ti') + get_species_molar_mass('16O'),
        'TiO_47_Exomol_McKemmish': get_species_molar_mass('47Ti') + get_species_molar_mass('16O'),
        'TiO_47_Plez': get_species_molar_mass('47Ti') + get_species_molar_mass('16O'),
        'TiO_48_Exomol_McKemmish': get_species_molar_mass('48Ti') + get_species_molar_mass('16O'),
        'TiO_48_Plez': get_species_molar_mass('48Ti') + get_species_molar_mass('16O'),
        'TiO_49_Exomol_McKemmish': get_species_molar_mass('49Ti') + get_species_molar_mass('16O'),
        'TiO_49_Plez': get_species_molar_mass('49Ti') + get_species_molar_mass('16O'),
        'TiO_50_Exomol_McKemmish': get_species_molar_mass('50Ti') + get_species_molar_mass('16O'),
        'TiO_50_Plez': get_species_molar_mass('50Ti') + get_species_molar_mass('16O'),
        'TiO_all_iso_Exomol_McKemmish': get_species_molar_mass('TiO-NatAbund'),
        'TiO_all_iso_Plez': get_species_molar_mass('TiO-NatAbund'),
        'TiO_all_iso_exo': get_species_molar_mass('TiO-NatAbund'),
        'V': get_species_molar_mass('V'),
        'V+': get_species_molar_mass('V') - get_species_molar_mass('e-'),
        'VO': get_species_molar_mass('VO'),
        'VO_ExoMol_McKemmish': get_species_molar_mass('VO'),
        'VO_ExoMol_Specific_Transitions': get_species_molar_mass('VO'),
        'Y': get_species_molar_mass('Y')
    })

    names = _get_prt2_line_by_line_names()

    for key, value in names.items():
        if value is None:
            if '_main_iso' in key:
                k = key.split('_main_iso', 1)[0]
            else:
                k = key

            names[key] = _get_base_line_by_line_names()[k]

    # Loading
    print("Loading default PT grid...")

    opacities_temperature_profile_grid = np.genfromtxt(
        os.path.join(path_input_data, 'opa_input_files', 'opa_PT_grid.dat')
    )

    opacities_temperature_profile_grid = np.flip(opacities_temperature_profile_grid, axis=1)

    opacities_temperatures = np.unique(opacities_temperature_profile_grid[:, 0])
    opacities_pressures = np.unique(opacities_temperature_profile_grid[:, 1])  # grid is already in bar

    print("Loading default files names...")
    line_paths = np.loadtxt(os.path.join(path_input_data, 'opa_input_files', 'opa_filenames.txt'), dtype=str)

    # Conversion
    if old_paths:
        input_directory = str(os.path.join(path_input_data, __get_prt2_input_data_subpaths()['line_by_line_opacities']))

        directories = [
            os.path.join(input_directory, d) for d in os.listdir(input_directory)
            if os.path.isdir(d) and d != 'PaxHeader'
        ]
    else:
        input_directory = str(os.path.join(path_input_data, get_input_data_subpaths()['line_by_line_opacities']))
        directories = []

        for species_dir in os.listdir(input_directory):
            species_dir = os.path.join(input_directory, species_dir)

            if os.path.isdir(species_dir) and species_dir.rsplit(os.path.sep, 1)[1] != 'PaxHeader':
                for iso_dir in os.listdir(species_dir):
                    iso_dir = os.path.join(species_dir, iso_dir)

                    if os.path.isdir(iso_dir) and iso_dir.rsplit(os.path.sep, 1)[1] != 'PaxHeader':
                        for d in os.listdir(iso_dir):
                            d = os.path.join(iso_dir, d)

                            if os.path.isdir(d) and d.rsplit(os.path.sep, 1)[1] != 'PaxHeader':
                                directories.append(d)

    for directory in directories:
        species = directory.rsplit(os.path.sep, 1)[1]
        species = species.rsplit('_def', 1)[0]
        not_in_dict = False

        # Check information availability
        if species not in doi_dict:
            warnings.warn(f"species '{species}' was not in DOI dict; "
                          f"add key '{species}' to the script doi_ict to run this conversion")
            not_in_dict = True

        if species not in contributor_dict:
            warnings.warn(f"species '{species}' was not in contributor dict; "
                          f"add key '{species}' to the script contributor_dict to run this conversion")
            not_in_dict = True

        if species not in description_dict:
            warnings.warn(f"species '{species}' was not in contributor dict; "
                          f"add key '{species}' to the script contributor_dict to run this conversion")
            not_in_dict = True

        if species not in molmass_dict:
            warnings.warn(f"species '{species}' was not in molar mass dict; "
                          f"add key '{species}' to the script molmass_dict to run this conversion")
            not_in_dict = True

        if not_in_dict:
            print(f" Skipping species '{species}' due to missing species in supplementary info dict...")
            continue

        line_by_line_opacities_dat2h5(
            directory=directory,
            output_name=names[species],
            molmass=molmass_dict[species],
            doi=doi_dict[species],
            path_input_data=path_input_data,
            contributor=contributor_dict[species],
            description=description_dict[species],
            opacities_pressures=opacities_pressures,
            opacities_temperatures=opacities_temperatures,
            line_paths=line_paths,
            memory_map_mode=memory_map_mode,
            rewrite=rewrite,
            clean=clean
        )

    print("Successfully converted line-by-line line opacities")


def _phoenix_spec_dat2h5(path_input_data=petitradtrans_config_parser.get_input_data_path(), rewrite=False,
                         old_paths=False, clean=False):
    """
    Convert a PHOENIX stellar spectrum in .dat format to HDF5 format.
    """
    from petitRADTRANS.stellar_spectra.phoenix import phoenix_star_table
    # Load the stellar parameters
    if old_paths:
        path = str(os.path.join(path_input_data, __get_prt2_input_data_subpaths()['stellar_spectra']))
    else:
        path = str(os.path.join(path_input_data, get_input_data_subpaths()['stellar_spectra'], 'phoenix'))

    hdf5_file = phoenix_star_table.get_default_file(path_input_data=path_input_data)

    if os.path.isfile(hdf5_file) and not rewrite:
        __print_skipping_message(hdf5_file)
        return

    dat_file = os.path.join(path, 'stellar_params.dat')

    if not os.path.isfile(dat_file):
        __print_missing_data_file_message(
            'stellar spectrum', 'stellar_params.dat', dat_file.rsplit(os.path.sep, 1)[0]
        )
        return

    old_files = [os.path.join(path, 'stellar_params.dat')]
    description = np.genfromtxt(old_files[0])

    # Initialize the grids
    log_temp_grid = description[:, 0]
    star_rad_grid = description[:, 1]

    # Load the corresponding numbered spectral files
    spec_dats = []

    for spec_num in range(len(log_temp_grid)):
        old_files.append(os.path.join(
            path,
            'spec_' + str(int(spec_num)).zfill(2) + '.dat')
        )
        spec_dats.append(np.genfromtxt(old_files[-1]))

    # Write the HDF5 file
    with h5py.File(hdf5_file, 'w') as f:
        t_eff = f.create_dataset(
            name='log10_effective_temperatures',
            data=log_temp_grid
        )
        t_eff.attrs['units'] = 'log10(K)'

        radius = f.create_dataset(
            name='radii',
            data=star_rad_grid
        )
        radius.attrs['units'] = 'R_sun'

        mass = f.create_dataset(
            name='masses',
            data=description[:, 2]
        )
        mass.attrs['units'] = 'M_sun'

        spectral_type = f.create_dataset(
            name='spectral_types',
            data=description[:, -1]
        )
        spectral_type.attrs['units'] = 'None'

        wavelength = f.create_dataset(
            name='wavelengths',
            data=np.asarray(spec_dats)[0, :, 0]
        )
        wavelength.attrs['units'] = 'cm'

        spectral_radiosity = f.create_dataset(
            name='fluxes',
            data=np.asarray(spec_dats)[:, :, 1]
        )
        spectral_radiosity.attrs['units'] = 'erg/s/cm^2/Hz'

    print("Successfully converted stellar spectra")

    if clean:
        __remove_files(old_files)


def _refactor_input_data_folder(path_input_data=petitradtrans_config_parser.get_input_data_path()):
    old_input_data_subpaths = __get_prt2_input_data_subpaths()
    directories_to_create = copy.deepcopy(old_input_data_subpaths)

    directories_to_create.update(
        {
            "cia_opacities": None,
            "clouds_opacities": None,
            "correlated_k_opacities": None,
            "line_by_line_opacities": None,
            "planet_data": None,
            "pre_calculated_chemistry": "equilibrium_chemistry",
            "stellar_spectra": "phoenix"
        }
    )

    for key, value in get_input_data_subpaths().items():
        old_subpath = str(os.path.join(path_input_data, old_input_data_subpaths[key]))
        new_subpath = str(os.path.join(path_input_data, value))

        if key == 'planet_data':
            if not os.path.isdir(old_subpath):
                print("No planet yet, skipping planet_data refactor...")
                os.mkdir(new_subpath)
                continue
        else:
            if not os.path.isdir(old_subpath) and not os.path.isdir(new_subpath):
                print(f"Incomplete input_data dict, skipping {old_input_data_subpaths[key]} refactor...")
                os.makedirs(new_subpath)
                continue

        old_subpath_is_dir = os.path.isdir(old_subpath)
        rename = True

        if os.path.isdir(new_subpath):
            if not old_subpath_is_dir:
                print(f"No need to rename input_data path subpath '{value}'")
            elif old_input_data_subpaths[key] != value:
                warnings.warn(f"old and new input_data subpaths ('{old_input_data_subpaths[key]}' and '{value}') "
                              f"coexists; the old directory can be removed")

            rename = False

        if old_subpath_is_dir and rename:
            print(f"Renaming directory '{old_subpath}' to '{new_subpath}'...")
            os.rename(old_subpath, new_subpath)

        files = [f for f in os.listdir(new_subpath) if os.path.isfile(os.path.join(new_subpath, f))]
        directories = [d for d in os.listdir(new_subpath) if os.path.isdir(os.path.join(new_subpath, d))]

        if directories_to_create[key] is not None:
            new_directory = os.path.join(new_subpath, directories_to_create[key])

            if os.path.isdir(new_directory):
                print(f"No need to create already existing directory '{new_directory}'")
            else:
                print(f"Making directory '{new_directory}'...")
                os.mkdir(new_directory)

            if len(files) > 0:
                for file in files:
                    if file[:2] == '._' or file == '.DS_Store':
                        os.remove(os.path.join(new_subpath, file))
                        continue

                    print(f"Moving file '{file}' to directory '{new_directory}'...")
                    _file = os.path.join(new_subpath, file)
                    os.rename(_file, os.path.join(new_directory, file))
        else:
            move_dirs = False
            multi = False

            if key == 'cia_opacities':
                d_prt2 = {v: k for k, v in _get_prt2_cia_names().items()}  # invert keys and values
                d_prt3 = _get_base_cia_names()
            elif key == 'clouds_opacities':
                d_prt2 = {v: k for k, v in _get_prt2_cloud_names().items()}  # invert keys and values
                d_prt3 = _get_base_cloud_names()
                move_dirs = True
            elif key == 'correlated_k_opacities':
                d_prt2 = _get_prt2_correlated_k_names()
                d_prt3 = _get_base_correlated_k_names()
                d_prt3.unlock()
                multi = True

                for k, v in d_prt2.items():
                    if v is None:
                        d_prt2[k] = k

                    if k not in d_prt3:
                        d_prt3[k] = v
                        d_prt2[k] = k

                d_prt3.lock()
            elif key == 'line_by_line_opacities':
                d_prt2 = _get_prt2_line_by_line_names()
                d_prt3 = _get_base_line_by_line_names()
                d_prt2.unlock()
                d_prt3.unlock()
                new_d_prt2 = copy.deepcopy(d_prt2)
                multi = True

                for k, v in d_prt2.items():
                    if v is None or k in d_prt3:
                        new_d_prt2[k] = k

                    if k not in d_prt3:
                        if '_main_iso' in k and 'Hargreaves' not in k and 'pokazatel' not in k and 'Coles' not in k:
                            _k = k.split('_main_iso', 1)[0]

                            if _k not in d_prt3:
                                raise KeyError(f"'{_k}' not found in pRT3 lbl names")

                            new_d_prt2[_k] = k

                            if v is not None:
                                d_prt3[_k] = v
                        else:
                            new_d_prt2[k] = k

                            if v is None:
                                raise ValueError(f"old non-standard species name '{k}' must be given a new name")

                            d_prt3[k] = v

                new_d_prt2.lock()
                d_prt2 = new_d_prt2
                d_prt3.lock()
            else:
                print(f"Skipping key '{key}'")
                continue

            for _key, filename in d_prt3.items():
                if multi and _key not in d_prt2:
                    print(f"Skipping multi-defined key '{_key}'...")
                    continue

                # If folder only contains a part of the opacities that are usually converted (e.g., only c-k).
                if _key not in d_prt2:
                    print('Species ' + _key + ' not found in old input data folder. Skipping...')
                    continue
                else:
                    old_directory = d_prt2[_key]

                if move_dirs:
                    spec = copy.deepcopy(old_directory)
                    old_directory = old_directory.split('(', 1)[0]
                    old_directory += '_c'
                else:
                    spec = None

                base_dirname = Opacity.get_species_base_name(filename, join=True)
                iso_dirname = Opacity.get_species_isotopologue_name(filename, join=True)

                if old_directory in directories:
                    base_dirname = os.path.join(new_subpath, base_dirname)
                    iso_dirname = os.path.join(new_subpath, base_dirname, iso_dirname)

                    if not move_dirs:
                        if multi:
                            old_directory = os.path.join(new_subpath, old_directory)

                            if os.path.isdir(iso_dirname) and base_dirname == old_directory:
                                print(f"Skipping already existing directory '{iso_dirname}'")
                                continue

                            last_dir = old_directory.rsplit(os.path.sep, 1)[-1]
                            _iso_dirname = os.path.join(iso_dirname, last_dir)

                            if os.path.isdir(_iso_dirname) or os.path.isdir(_iso_dirname + '_def'):
                                print(f"Skipping already existing directory '{_iso_dirname}'")
                                continue

                            default_dir = base_dirname + '_def'

                            if not os.path.isdir(base_dirname):
                                os.mkdir(base_dirname)
                            else:
                                if base_dirname == old_directory:
                                    if not os.path.isdir(default_dir):
                                        os.rename(old_directory, default_dir)

                                    old_directory = default_dir

                                    if not os.path.isdir(base_dirname):
                                        os.mkdir(base_dirname)

                            if os.path.isdir(base_dirname + '_def'):
                                old_directory = default_dir
                        else:
                            os.rename(os.path.join(new_subpath, old_directory), base_dirname)
                    else:
                        old_directory = os.path.join(new_subpath, old_directory)

                        if not os.path.isdir(base_dirname):
                            os.mkdir(base_dirname)

                    if not os.path.isdir(iso_dirname) and not move_dirs:
                        os.mkdir(iso_dirname)

                    if not move_dirs:
                        if multi:
                            last_dir = old_directory.rsplit(os.path.sep, 1)[-1]
                            iso_dirname = os.path.join(iso_dirname, last_dir)
                            _files = [old_directory]
                        else:
                            _files = [
                                f for f in os.listdir(base_dirname) if os.path.isfile(os.path.join(base_dirname, f))
                            ]
                    else:
                        if not os.path.isdir(old_directory):
                            print(f"Skipping not found directory '{old_directory}'")
                            continue

                        _files = [
                            f for f in os.listdir(old_directory)
                            if os.path.isdir(os.path.join(old_directory, f))
                        ]

                        if spec[-2] == 'c' and 'crystalline' in _files:
                            _files = ['crystalline']
                        elif spec[-2] == 'a' and 'amorphous' in _files:
                            _files = ['amorphous']
                        elif len(_files) == 0:
                            os.removedirs(old_directory)
                        elif len(_files) == 1 and _files[0] == 'PaxHeader':
                            shutil.rmtree(old_directory)
                        else:
                            raise ValueError(f"incorrect cloud directories '{_files}' for species '{spec}'")

                    for _file in _files:
                        if _file[:2] == '._' or _file == '.DS_Store' or _file == 'PaxHeader':
                            if not move_dirs:
                                os.remove(os.path.join(base_dirname, _file))
                            else:
                                if os.path.isdir(old_directory):
                                    os.remove(os.path.join(old_directory, _file))

                            continue

                        print(f"Moving file '{_file}' to directory '{iso_dirname}'...")

                        if not move_dirs:
                            if multi:
                                os.rename(_file, iso_dirname)
                            else:
                                __file = os.path.join(base_dirname, _file)
                                os.rename(__file, os.path.join(iso_dirname, _file))
                        else:
                            __file = os.path.join(old_directory, _file)
                            os.rename(__file, iso_dirname)

                    if move_dirs:
                        if os.path.isdir(old_directory):
                            _files = [
                                f for f in os.listdir(old_directory)
                                if os.path.isdir(os.path.join(old_directory, f))
                            ]

                            if len(_files) == 0 or len(_files) == 1 and _files[0] == 'PaxHeader':
                                shutil.rmtree(old_directory)
                else:
                    print(f"Skipping not found directory '{old_directory}'")
                    continue


def _sort_pressure_temperature_grid(pressure_temperature_grid_file):
    # Read the Ps and Ts
    pressure_temperature_grid = np.genfromtxt(pressure_temperature_grid_file)

    # Read the file names
    with open(pressure_temperature_grid_file, 'r') as f:
        lines = f.readlines()

    n_lines = len(lines)

    # Prepare the array to contain the pressures, temperatures, indices in the unsorted list.
    # Also prepare the list of unsorted names
    sorted_grid = np.ones((n_lines, 3))
    names = []

    # Fill the array and name list
    for i in range(n_lines):
        columns = lines[i].split(' ')

        sorted_grid[i, 0] = pressure_temperature_grid[i, 0]
        sorted_grid[i, 1] = pressure_temperature_grid[i, 1]
        sorted_grid[i, 2] = i

        if columns[-1][-1] == '\n':
            names.append(columns[-1][:-1])
        else:
            names.append(columns[-1])

    # Sort the array by temperature
    sorted_indices = np.argsort(sorted_grid[:, 1])
    sorted_grid = sorted_grid[sorted_indices, :]

    # Sort the array entries with constant temperatures by pressure
    n_pressures = 0

    for i in range(n_lines):
        if np.abs(sorted_grid[i, 1] - sorted_grid[0, 1]) > 1e-10:
            break

        n_pressures = n_pressures + 1

    n_temperatures = int(n_lines / n_pressures)

    for i in range(n_temperatures):
        sorted_grid_ = sorted_grid[i * n_pressures:(i + 1) * n_pressures, :]
        sorted_indices = np.argsort(sorted_grid_[:, 0])
        sorted_grid_ = sorted_grid_[sorted_indices, :]
        sorted_grid[i * n_pressures:(i + 1) * n_pressures, :] = sorted_grid_

    names_sorted = []

    for i in range(n_lines):
        names_sorted.append(names[int(sorted_grid[i, 2] + 0.01)])

    # Convert from bar to cgs
    sorted_grid[:, 0] = sorted_grid[:, 0] * 1e6

    return [sorted_grid[:, :-1][:, ::-1], names_sorted, n_temperatures, n_pressures]


def bin_species_exok(species: list[str], resolution: float):
    """
    This function uses exo-k to bin the c-k table of a
    multiple species to a desired (lower) spectral resolution.

    Args:
        species : string
            The name of the species
        resolution : int
            The desired spectral resolving power.
    """
    from petitRADTRANS.config import petitradtrans_config_parser

    prt_path = petitradtrans_config_parser.get_input_data_path()

    ck_paths = []

    print(f"Resolving power: {resolution}")

    for s in species:
        ck_paths.append(CorrelatedKOpacity.find(
            species=s,
            path_input_data=prt_path
        ))

        print(f" Re-binned opacities: '{ck_paths[-1]}'")

    rebin_multiple_ck_line_opacities(
        target_resolving_power=int(resolution),
        paths=ck_paths,
        species=species
    )


def continuum_clouds_opacities_dat2h5(input_directory, output_name, cloud_species, doi,
                                      cloud_species_mode=None,
                                      path_input_data=petitradtrans_config_parser.get_input_data_path(),
                                      description=None,
                                      cloud_path=None, path_input_files=None, path_reference_files=None,
                                      rewrite=False, clean=False):
    """Using ExoMol units for HDF5 files."""
    CloudOpacity.check_name(output_name)

    if cloud_species_mode is None:
        cloud_species_mode = ''

    all_cloud_species = cloud_species + ','
    all_cloud_isos = input_directory.rsplit(os.path.sep, 2)[1] + ','
    all_cloud_species_mode = cloud_species_mode + ','

    reference_file = str(os.path.join(
        path_input_data,
        get_input_data_subpaths()['clouds_opacities'],
        'MgSiO3(s)_amorphous', 'Mg-Si-O3-NatAbund(s)_amorphous', 'mie', 'opa_0001.dat'
    ))

    if not os.path.isfile(reference_file):
        raise FileNotFoundError(
            f"reference file for loading .dat cloud opacities ('{reference_file}') not found, "
            f"it must be downloaded "
            f"(see https://petitradtrans.readthedocs.io/en/latest/content/available_opacities.html)"
        )

    n_cloud_wavelength_bins = len(np.genfromtxt(reference_file)[:, 0])

    if cloud_path is None:
        cloud_path = os.path.join(path_input_data, get_input_data_subpaths()['clouds_opacities'])

    if path_input_files is None:
        path_input_files = os.path.join(path_input_data, 'opa_input_files')

    if path_reference_files is None:
        path_reference_files = os.path.join(cloud_path, 'MgSiO3(s)_amorphous', 'Mg-Si-O3-NatAbund(s)_amorphous', 'mie')

    # Load .dat files
    print("Loading dat file...")
    cloud_particles_densities, cloud_absorption_opacities, cloud_scattering_opacities, \
        cloud_asymmetry_parameter, cloud_wavelengths, cloud_particles_radius_bins, cloud_particles_radii \
        = finput.load_cloud_opacities(
            cloud_path, path_input_files, path_reference_files,
            all_cloud_species, all_cloud_isos, all_cloud_species_mode,
            1, n_cloud_wavelength_bins
        )

    wavenumbers = 1 / cloud_wavelengths[::-1]  # cm to cm-1

    output_directory = os.path.join(input_directory, '..')

    hdf5_opacity_file = os.path.join(
        output_directory, f"{output_name}.cotable.petitRADTRANS.h5"
    )

    if os.path.isfile(hdf5_opacity_file) and not rewrite:
        __print_skipping_message(hdf5_opacity_file)
        return

    # Write HDF5 file
    print(f" Writing file '{hdf5_opacity_file}'...", end=' ')

    i = 0  # only one cloud

    write_cloud_opacities(
        hdf5_opacity_file=hdf5_opacity_file,
        cloud_species=cloud_species[i],
        wavenumbers=wavenumbers,
        cloud_wavelengths=cloud_wavelengths,
        cloud_absorption_opacities=cloud_absorption_opacities[:, ::-1, i],
        cloud_scattering_opacities=cloud_scattering_opacities[:, ::-1, i],
        cloud_asymmetry_parameter=cloud_asymmetry_parameter[:, ::-1, i],
        cloud_particles_densities=cloud_particles_densities[i],
        cloud_particles_radii=cloud_particles_radii,
        cloud_particles_radius_bins=cloud_particles_radius_bins,
        doi=doi,
        description=description
    )

    print("Done.")

    if clean:
        __remove_files([input_directory])


def continuum_clouds_opacities_dat2h5_external_species(path_to_species_opacity_folder,
                                                       longname,
                                                       cloud_material_density,
                                                       save_folder='converted_cloud_opacities',
                                                       doi=None,
                                                       description=None,
                                                       wavelength_limit=None):
    # noinspection PyUnresolvedReferences
    from petitRADTRANS.fortran_inputs import fortran_inputs as finput

    n_cloud_wavelength_bins = len(np.genfromtxt(os.path.join(path_to_species_opacity_folder, 'opa_0001.dat'))[:, 0])

    (cloud_absorption_opacities, cloud_scattering_opacities, cloud_asymmetry_parameter, cloud_wavelengths,
     cloud_particles_radius_bins, cloud_particles_radii) = finput.load_cloud_opacities_external(
        path_to_species_opacity_folder,
        1,
        n_cloud_wavelength_bins
    )

    if wavelength_limit is not None:
        index_bad = (cloud_wavelengths < wavelength_limit[0] * 1e-4) | (cloud_wavelengths > wavelength_limit[1] * 1e-4)
        cloud_absorption_opacities[:, index_bad, :] = 0
        cloud_scattering_opacities[:, index_bad, :] = 0

    wavenumbers = 1 / cloud_wavelengths[::-1]  # cm to cm-1

    output_directory = os.path.join(save_folder, longname.split('__')[0].replace('-NatAbund', '').replace('-', ''))

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    output_directory = os.path.join(output_directory, longname.split('__')[0])

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    hdf5_opacity_file = os.path.join(
        output_directory, longname
    )

    # Write HDF5 file
    print(f" Writing file '{hdf5_opacity_file}'...", end=' ')

    i = 0  # only one cloud

    with h5py.File(hdf5_opacity_file, "w") as fh5:
        dataset = fh5.create_dataset(
            name='DOI',
            shape=(1,),
            data=doi
        )
        dataset.attrs['long_name'] = 'Data object identifier linked to the data'
        dataset.attrs['additional_description'] = str(description)

        dataset = fh5.create_dataset(
            name='Date_ID',
            shape=(1,),
            data=f'petitRADTRANS-v{petitRADTRANS.__version__}'
                 f'_{datetime.datetime.now(datetime.timezone.utc).isoformat()}'
        )
        dataset.attrs['long_name'] = 'ISO 8601 UTC time (https://docs.python.org/3/library/datetime.html) ' \
                                     'at which the table has been created, ' \
                                     'along with the version of petitRADTRANS'

        dataset = fh5.create_dataset(
            name='wavenumbers',
            data=wavenumbers
        )
        dataset.attrs['long_name'] = 'Opacities wavenumbers'
        dataset.attrs['units'] = 'cm^-1'

        dataset = fh5.create_dataset(
            name='absorption_opacities',
            data=cloud_absorption_opacities[:, ::-1, i]
        )
        dataset.attrs['long_name'] = 'Table of the absorption opacities with axes (particle radius, wavenumber)'
        dataset.attrs['units'] = 'cm^2.g^-1'

        dataset = fh5.create_dataset(
            name='scattering_opacities',
            data=cloud_scattering_opacities[:, ::-1, i]
        )
        dataset.attrs['long_name'] = 'Table of the scattering opacities with axes (particle radius, wavenumber)'
        dataset.attrs['units'] = 'cm^2.g^-1'

        dataset = fh5.create_dataset(
            name='asymmetry_parameters',
            data=cloud_asymmetry_parameter[:, ::-1, i]
        )
        dataset.attrs['long_name'] = 'Table of the asymmetry parameters with axes (particle radius, wavenumber)'
        dataset.attrs['units'] = 'None'

        dataset = fh5.create_dataset(
            name='particles_density',
            data=cloud_material_density
        )
        dataset.attrs['long_name'] = 'Average density of the cloud particles'
        dataset.attrs['units'] = 'g.cm^-3'

        dataset = fh5.create_dataset(
            name='mol_name',
            shape=(1,),
            data=longname.replace('-NatAbund', '').replace('-', '')
        )
        dataset.attrs['long_name'] = 'Name of the species described.'

        dataset = fh5.create_dataset(
            name='particles_radii',
            data=cloud_particles_radii
        )
        dataset.attrs['long_name'] = 'Particles average radius grid'
        dataset.attrs['units'] = 'cm'

        dataset = fh5.create_dataset(
            name='particle_radius_bins',
            data=cloud_particles_radius_bins
        )
        dataset.attrs['long_name'] = 'Particles average radius grid bins'
        dataset.attrs['units'] = 'cm'

        dataset = fh5.create_dataset(
            name='wlrange',
            data=np.array([cloud_wavelengths.min(), cloud_wavelengths.max()]) * 1e4  # cm to um
        )
        dataset.attrs['long_name'] = 'Wavelength range covered'
        dataset.attrs['units'] = 'µm'

        dataset = fh5.create_dataset(
            name='wnrange',
            data=np.array([wavenumbers.min(), wavenumbers.max()])
        )
        dataset.attrs['long_name'] = 'Wavenumber range covered'
        dataset.attrs['units'] = 'cm^-1'

    print("Done.")


def fits_output(wavelength, spectrum, covariance, object_name, output_dir="",
                correlation=None):
    """
    Generate a fits file that can be used as an input to a pRT retrieval.

    Args:
        wavelength : numpy.ndarray
            The wavelength bin centers in micron. dim(N)
        spectrum : numpy.ndarray
            The flux density in W/m2/micron at each wavelength bin. dim(N)
        covariance : numpy.ndarray
            The covariance of the flux in (W/m2/micron)^2 dim(N,N)
        object_name : string
            The name of the object, used for file naming.
        output_dir : string
            The parent directory of the output file.
        correlation : numpy.ndarray
            The correlation matrix of the flux points (See Brogi & Line 2018, https://arxiv.org/pdf/1811.01681.pdf)

    Returns:
        hdul : astropy.fits.HDUlist
            The HDUlist object storing the spectrum.
    """

    from astropy.io import fits

    primary_hdu = fits.PrimaryHDU([])
    primary_hdu.header['OBJECT'] = object_name

    c1 = fits.Column(name="WAVELENGTH", array=wavelength, format='D', unit="micron")
    c2 = fits.Column(name="FLUX", array=spectrum, format='D', unit="W/m2/micron")
    c3 = fits.Column(name="COVARIANCE", array=covariance, format=str(covariance.shape[0]) + 'D', unit="[W/m2/micron]^2")

    if correlation is not None:
        c4 = fits.Column(name="CORRELATION", array=correlation, format=str(correlation.shape[0]) + 'D', unit=" - ")
    else:
        c4 = None

    columns = [c1, c2, c3, c4]

    table_hdu = fits.BinTableHDU.from_columns(columns, name='SPECTRUM')
    hdul = fits.HDUList([primary_hdu, table_hdu])
    outstring = os.path.join(output_dir, object_name + "_spectrum.fits")
    hdul.writeto(outstring, overwrite=True, checksum=True, output_verify='exception')

    return hdul


def format2petitradtrans(load_function, opacities_directory: str, natural_abundance: bool,
                         source: str, doi: str, species: str,
                         charge: str = '', cloud_info: str = '', contributor: str = None, description: str = None,
                         opacity_files_extension: str = None, spectral_dimension_file: str = None,
                         path_input_data: str = petitradtrans_config_parser.get_input_data_path(), rebin: bool = True,
                         save_correlated_k: bool = True, correlated_k_resolving_power: float = 1000,
                         samples: np.ndarray = None, weights: np.ndarray = None,
                         save_line_by_line: bool = True, line_by_line_wavelength_boundaries: np.ndarray = None,
                         standard_line_by_line_wavelength_boundaries: np.ndarray = None, rewrite: bool = False,
                         use_legacy_correlated_k_wavenumbers_sampling: bool = False):
    """Convert opacities loaded with the specified load function into petitRADTRANS line-by-line and correlated-k
    opacities.

    Args:
        load_function:
            String ('dace' or 'exocross'), or function loading the opacities.
            If 'dace', the DACE (https://dace.unige.ch/opacityDatabase/) opacity built-in loading function is used.
            If 'exocross', the ExoCross (https://github.com/Trovemaster/exocross) opacity built-in loading function is
            used.
            If a function is used, it must have the following arguments, in that order:
                - file: the file name of a file containing the opacities (automatically taken from opacities_directory)
                - file_extension: see opacity_files_extension below
                - molmass: molar mass of the species (automatically set)
                - wavelength_file: see spectral_dimension_file below
                - wavenumbers_petitradtrans_line_by_line: the petitRADTRANS wavenumber grid for line-by-line opacities
                  (automatically set)
                - save_line_by_line: see save_line_by_line below
                - rebin: see rebin below
                - selection: indices corresponding to the wavenumbers to be extracted
            Not all of the above arguments have to be used.
            The function must output the following, in that order:
                - cross_sections: (cm2/molecule) the cross-sections
                - cross_sections_line_by_line: (cm2/molecule), the cross-sections,
                    interpolated to wavenumbers_petitradtrans_line_by_line
                - wavenumbers: (cm-1) the wavenumbers corresponding to cross-sections
                - pressure: (bar) the pressure of the cross-sections
                - temperature: (K) the temperature of the cross-sections
            The cross_sections, cross_sections_line_by_line and wavenumbers must be returned in increasing wavenumber
            order.
        opacities_directory:
            Directory in which the opacity files are stored.
        natural_abundance:
            If True, the opacities are considered coming from the natural (Earth's) occurring isotopologue mix, instead
            of from a single isotope.
        source:
            Name of the opacities' source (e.g., 'POKAZATEL').
        doi:
            DOI of the opacities' source
        species:
            Chemical formula of the species (e.g. 'H2O')
        charge:
            Charge of the species (e.g. '2+')
        cloud_info:
            Cloud additional info (physical state, internal structure, space group). See the petitRADTRANS cloud
            filename convention for more information.
        contributor:
            Contributor that helped obtained the opacities.
        description:
            Additional description on the opacities.
        opacity_files_extension:
            The extension of the opacity files.
        spectral_dimension_file:
            File containing the opacities' wavelengths.
        path_input_data:
            Path to the input data directory.
        rebin:
            If true, rebin the opacities to the petitRADTRANS standard wavenumber grid. Should be True in most cases.
        save_correlated_k:
            If True, convert and save the opacities in correlated-k ('c-k') format.
        correlated_k_resolving_power:
            Resolving power at which to convert the opacities, for the correlated-k case.
        samples:
            Array containing the correlated-k samples.
        weights:
            Array containing the correlated-k weights.
        save_line_by_line:
            If True, convert and save the opacities in line-by-line ('lbl') format.
        line_by_line_wavelength_boundaries:
            Wavelength boundaries to use for the line-by-line conversion.
        standard_line_by_line_wavelength_boundaries:
            Wavelength boundaries to use when generating the lbl standard petitRADTRANS wavelengths.
        rewrite:
            if True, rewrite existing converted files
        use_legacy_correlated_k_wavenumbers_sampling:
            If True, use the legacy (pRT2) way to sample the correlated-k wavenumbers.
    """
    if load_function == 'exocross':
        load_function = load_exocross

        if rebin:
            opacity_files_extension = '.out.xsec'
        else:
            opacity_files_extension = '.dat'

        spectral_dimension_file = os.path.join(opacities_directory, 'wavelength.dat')
    elif load_function == 'dace':
        load_function = load_dace
        opacity_files_extension = '.bin'
        spectral_dimension_file = None  # wavenumber points are obtained from the file names
    elif not callable(load_function):
        raise ValueError(f"load_function must be 'exocross', 'dace', or a function, but is {load_function}")

    if opacity_files_extension is None:
        raise TypeError("missing 1 required argument: 'opacity_files_extension'")

    if line_by_line_wavelength_boundaries is None:
        line_by_line_wavelength_boundaries = np.array([0.3, 28])  # um

    if standard_line_by_line_wavelength_boundaries is None:
        standard_line_by_line_wavelength_boundaries = np.array([1.1e-1, 250])  # um

    # Read the fiducial petitRADTRANS wavelength grid
    wavenumbers_petitradtrans_file = os.path.join(
        path_input_data, 'opacities', 'lines', 'line_by_line', 'wavenumber_grid.petitRADTRANS.h5'
    )

    wavenumbers_petitradtrans_line_by_line = np.empty(0)

    if not os.path.isfile(wavenumbers_petitradtrans_file):
        print("Generating petitRADTRANS wavenumber grid... ", end='')

        wavenumbers_petitradtrans_line_by_line = prt_resolving_space(
            start=standard_line_by_line_wavelength_boundaries[0],
            stop=standard_line_by_line_wavelength_boundaries[-1],
            resolving_power=1e6
        )  # (um)

        wavenumbers_petitradtrans_line_by_line = 1e4 / wavenumbers_petitradtrans_line_by_line[::-1]   # um to cm-1
    else:
        print("Loading petitRADTRANS wavenumber grid... ", end='')

        with h5py.File(wavenumbers_petitradtrans_file, 'r') as f:
            wavenumbers_petitradtrans_line_by_line = f['bin_edges'][:]

    print("Done.")

    selection = np.nonzero(
        np.logical_and(
            np.greater_equal(wavenumbers_petitradtrans_line_by_line, 1e4 / line_by_line_wavelength_boundaries[1]),
            np.less_equal(wavenumbers_petitradtrans_line_by_line, 1e4 / line_by_line_wavelength_boundaries[0])
        )
    )[0]
    selection = np.array([selection[0] - 1, selection[-1] + 2])  # ensure that the selection contains the bounds

    wavenumbers_line_by_line_selected = wavenumbers_petitradtrans_line_by_line[selection[0]:selection[1]]

    if rebin:
        print(f"Reading opacity files in directory '{opacities_directory}'...")

        opacity_files = glob.glob(os.path.join(opacities_directory, f'*{opacity_files_extension}'))
    else:
        print(f"Reading re-binned opacity files in directory '{opacities_directory}'...")

        opacity_files = glob.glob(os.path.join(opacities_directory, f'*bar{opacity_files_extension}'))

    if len(opacity_files) == 0:
        raise FileNotFoundError(f"no file to convert found in directory '{opacities_directory}'")
    elif len(opacity_files) < 130:
        warnings.warn(f"directory '{opacities_directory}' contains only {len(opacity_files)} files, "
                      f"the standard petitRADTRANS temperature-pressure grid size is 130; a finer temperature-pressure "
                      f"grid allows for a more accurate interpolation of the cross-sections")

    species_isotopologue_name = copy.deepcopy(species)

    if natural_abundance:
        species_isotopologue_name = species_isotopologue_name + '-NatAbund'
        natural_abundance = 'NatAbund'
    else:
        natural_abundance = ''

    species_isotopologue_name = Opacity.get_species_isotopologue_name(species_isotopologue_name)

    molmass = get_species_molar_mass(species)

    wavenumbers = None
    __wavenumbers = None  # used to check wavenumbers consistency
    pressures = []
    temperatures = []
    sigmas = []
    sigmas_line_by_line = []

    print(f"Starting conversion of directory '{opacities_directory}'...")

    for i, opacity_file in enumerate(opacity_files):
        print(f" Reading file '{opacity_file}' ({i + 1}/{len(opacity_files)})... ", end='')

        cross_sections, cross_sections_line_by_line, wavenumbers, pressure, temperature = load_function(
            file=opacity_file,
            file_extension=opacity_files_extension,
            wavelength_file=spectral_dimension_file,
            molmass=molmass,
            wavenumbers_petitradtrans_line_by_line=wavenumbers_petitradtrans_line_by_line,  # use the unselected wvn
            save_line_by_line=save_line_by_line,
            rebin=rebin,
            selection=selection  # wavenumbers selection
        )

        # Raise error if the wavenumbers are not sorted in increasing order
        if not np.all(np.diff(wavenumbers) > 0):
            raise ValueError("wavenumbers (and cross-sections) returned by external opacity loading function must be "
                             "sorted in increasing order")

        if i == 0:
            __wavenumbers = wavenumbers
        else:
            if wavenumbers.size != __wavenumbers.size:
                raise ValueError(f"all opacity files must have the same wavenumbers, "
                                 f"but the wavenumbers size of file '{opacity_files[0]}' is {wavenumbers.size}, "
                                 f"and the wavenumbers size of file '{opacity_file}' is {wavenumbers.size}")
            elif not np.allclose(wavenumbers, __wavenumbers, atol=0, rtol=1e-6):
                raise ValueError(f"all opacity files must have the same wavenumbers, "
                                 f"but the wavenumbers of file '{opacity_file}' "
                                 f"is different from the wavenumbers of file '{opacity_files[0]}'")

        temperatures.append(temperature)
        pressures.append(pressure)
        sigmas.append(cross_sections)
        sigmas_line_by_line.append(cross_sections_line_by_line)

        print("Done.")

    pressures = np.array(pressures)
    temperatures = np.array(temperatures)

    unique_temperatures = np.unique(temperatures)
    unique_pressures = np.unique(pressures)

    if len(opacity_files) != unique_temperatures.size * unique_pressures.size:
        raise ValueError(f"temperature-pressure grid used in directory '{opacities_directory}' must be rectangular, "
                         f"but {len(opacity_files)} files were found instead of "
                         f"{unique_temperatures.size * unique_pressures.size}\n"
                         f"Unique temperatures found: {unique_temperatures}\n"
                         f"Unique pressures found: {unique_pressures.size}")

    pressure_temperature_grid_sorted_id = np.lexsort((temperatures, pressures))
    resolving_power_line_by_line: float | None = None

    print("Preparation successful")

    if save_line_by_line:
        print(f"Starting line-by-line conversion (boundaries: {line_by_line_wavelength_boundaries})...")
        sigmas_line_by_line = np.array(sigmas_line_by_line)[pressure_temperature_grid_sorted_id]
        sigmas_line_by_line = sigmas_line_by_line.reshape(
            (unique_pressures.size, unique_temperatures.size, wavenumbers_line_by_line_selected.size)
        )

        output_directory = LineByLineOpacity.get_species_directory(
            species=species,
            category=None,  # use default
            path_input_data=path_input_data
        )

        if resolving_power_line_by_line is None:
            resolving_power_line_by_line = np.mean(
                wavenumbers_petitradtrans_line_by_line[:-1] / np.diff(wavenumbers_petitradtrans_line_by_line) + 0.5,
                dtype=float
            )

        filename = get_opacity_filename(
            resolving_power=resolving_power_line_by_line,
            wavelength_boundaries=line_by_line_wavelength_boundaries,
            species_isotopologue_name=species_isotopologue_name,
            source=source,
            natural_abundance=natural_abundance,
            charge=charge,
            cloud_info=cloud_info
        )

        hdf5_opacity_file = os.path.join(
            output_directory,
            filename + '.xsec.petitRADTRANS.h5'
        )

        if not os.path.isdir(output_directory):
            print(f"Creating directory '{output_directory}'")
            os.makedirs(output_directory)

        if os.path.isfile(hdf5_opacity_file) and not rewrite:
            raise FileExistsError(f"file '{hdf5_opacity_file}' already exists, "
                                  f"set rewrite to True to rewrite the file")

        print(f" Writing line-by-line file '{hdf5_opacity_file}'...")

        write_line_by_line(
            file=hdf5_opacity_file,
            doi=doi,
            wavenumbers=wavenumbers_line_by_line_selected,
            opacities=sigmas_line_by_line,
            mol_mass=molmass,
            species=species,
            opacities_pressures=unique_pressures,
            opacities_temperatures=unique_temperatures,
            contributor=str(contributor),
            description=str(description)
        )

        print(f"Successfully converted files in '{opacities_directory}' to line-by-line pRT files")

    if save_correlated_k:
        print(f"Starting correlated-k conversion (R = {correlated_k_resolving_power})...")
        # Initialize the samples grids if necessary
        if samples is None:
            samples = np.array([
                0.0178695646,
                0.0915000852,
                0.2135104155,
                0.3674544109,
                0.5325455891,
                0.6864895845,
                0.8084999148,
                0.8821304354,
                0.9019855072,
                0.9101666761,
                0.9237233795,
                0.9408282679,
                0.9591717321,
                0.9762766205,
                0.9898333239,
                0.9980144928
            ])

        if weights is None:
            weights = np.array([
                0.0455528413,
                0.1000714655,
                0.1411679906,
                0.1632077025,
                0.1632077025,
                0.1411679906,
                0.1000714655,
                0.0455528413,
                0.0050614268,
                0.0111190517,
                0.0156853323,
                0.0181341892,
                0.0181341892,
                0.0156853323,
                0.0111190517,
                0.0050614268
            ])

        if samples.size != weights.size:
            raise ValueError(f"correlated-k g and weight arrays must have the same size, "
                             f"but have sizes {samples.size} and {weights.size}")

        print(" Reshaping...")
        sigmas = np.array(sigmas)[pressure_temperature_grid_sorted_id]
        sigmas = sigmas.reshape(
            (unique_pressures.size, unique_temperatures.size, wavenumbers.size)
        )

        print(" Initializing correlated-k parameters...")
        samples_edges = np.zeros(samples.size + 1)
        samples_edges[-1] = 1.0

        for i in range(samples.size - 1):
            samples_edges[i + 1] = (samples[i + 1] + samples[i]) / 2

        if resolving_power_line_by_line is None:
            resolving_power_line_by_line = np.mean(
                wavenumbers_petitradtrans_line_by_line[:-1] / np.diff(wavenumbers_petitradtrans_line_by_line) + 0.5,
                dtype=float
            )

        downsampling = int(resolving_power_line_by_line / correlated_k_resolving_power)

        if downsampling > wavenumbers.size:
            raise ValueError(f"unable to make correlate-k resolving power {correlated_k_resolving_power}"
                             f"from source with resolving power {resolving_power_line_by_line}")

        if use_legacy_correlated_k_wavenumbers_sampling:
            starting_index = 999
        else:
            starting_index = 0

        bin_edges: npt.NDArray[np.floating] = wavenumbers_petitradtrans_line_by_line[
            ::-1
        ][starting_index::downsampling][::-1]

        if use_legacy_correlated_k_wavenumbers_sampling:
            max_extra_edges = 2

            # Append an extra edge until the source maximum wavenumber is reached
            for i in range(max_extra_edges):
                resolving_power_line_by_line = bin_edges[-1] / np.diff(bin_edges[-2:]) - 1
                bin_edges = np.append(bin_edges, bin_edges[-1] * (1 + 1 / resolving_power_line_by_line))

                if bin_edges[-1] > wavenumbers[-1]:
                    warnings.warn(f"legacy maximum bin edges ({bin_edges[-1]} cm-1) "
                                  f"is greater than the opacity source maximum wavenumber ({wavenumbers[-1]} cm-1)\n"
                                  f"Do not use legacy mode, "
                                  f"or use an opacity source with wavenumbers that match exactly "
                                  f"the legacy mode downsampled wavenumber grid")
                    break
                elif bin_edges[-1] == wavenumbers[-1]:
                    break

            if bin_edges[-1] < wavenumbers[-1]:
                raise ValueError(f"could not reach the source maximum wavenumber ({wavenumbers[-1]} cm-1) "
                                 f"after adding {max_extra_edges} extra edges ({bin_edges[-1]} cm-1)\n"
                                 f"This is likely a code issue, but can be an input issue\n"
                                 f"This issue can only happen in legacy mode")

        bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2

        correlated_k = np.zeros((
            unique_pressures.size,
            unique_temperatures.size,
            bin_edges.size - 1,
            samples.size
        ))

        min_selection_size: int = int(np.ceil(1 / np.min(np.diff(samples)))) * 2

        for i in range(bin_edges.size - 1):
            print(f" Calculating correlated-k ({i + 1}/{bin_edges.size - 1})...", end='\r')
            selection = np.nonzero(
                np.logical_and(
                    np.greater_equal(wavenumbers, bin_edges[i]),
                    np.less(wavenumbers, bin_edges[i + 1])
                )
            )[0]

            if selection.size < 2:  # happens when resolving power is too small
                selection_low = np.nonzero(np.less(wavenumbers, bin_edges[i + 1]))[0]
                selection_high = np.nonzero(np.greater_equal(wavenumbers, bin_edges[i]))[0]

                if selection_low.size == 0:
                    selection_low = 0
                else:
                    selection_low = selection_low[-1]

                if selection_high.size == 0:
                    selection_high = wavenumbers.size - 1
                else:
                    selection_high = selection_high[0]

                if selection_low == selection_high:
                    selection_low = max((selection_low - 1, 0))
                    selection_high = min((selection_high + 1, wavenumbers.size - 1))

                if selection_low > wavenumbers.size - 2:
                    selection_low = wavenumbers.size - 2

                if selection_high < 1:
                    selection_high = 1
                elif selection_high == wavenumbers.size - 1:
                    selection_high = None

                selection = [selection_low, selection_high]

            _sigmas = sigmas[:, :, selection[0]:selection[-1]]

            # If selection size is too small, interpolate the cross-sections so that there is enough samples for c-k
            if np.size(selection) < min_selection_size:
                _wavenumbers = wavenumbers[selection[0]:selection[-1]]
                _wavenumbers = np.linspace(_wavenumbers[0], _wavenumbers[-1], min_selection_size)
                _sigmas = interp1d(wavenumbers[selection[0]:selection[-1]], _sigmas, axis=-1)
                _sigmas = _sigmas(_wavenumbers)

                selection_size = min_selection_size
            else:
                selection_size = selection.size

            _sigmas = np.sort(_sigmas, axis=-1)
            sigmas_g_mean = np.zeros((unique_pressures.size, unique_temperatures.size, samples.size))

            g_sort = np.linspace(0, 1, selection_size)

            for j in range(samples.size):
                g_selection = np.nonzero(
                    np.logical_and(
                        np.greater_equal(g_sort, samples_edges[j]),
                        np.less_equal(g_sort, samples_edges[j + 1])
                    )
                )[0]

                sigmas_g_mean[:, :, j] = np.mean(_sigmas[:, :, g_selection[0]:g_selection[-1]], axis=-1)

            correlated_k[:, :, i, :] = sigmas_g_mean

        print(" Done.")

        correlated_k_wavelength_boundaries = np.array([1e4 / bin_edges[-1], int(np.ceil(1e4 / bin_edges[0]))])

        filename = get_opacity_filename(
            resolving_power=correlated_k_resolving_power,
            wavelength_boundaries=correlated_k_wavelength_boundaries,
            species_isotopologue_name=species_isotopologue_name,
            source=source,
            natural_abundance=natural_abundance,
            charge=charge,
            cloud_info=cloud_info
        )

        output_directory = CorrelatedKOpacity.get_species_directory(
            species=species,
            category=None,  # use default
            path_input_data=path_input_data
        )

        hdf5_opacity_file = os.path.join(
            output_directory,
            filename + '.ktable.petitRADTRANS.h5'
        )

        if not os.path.isdir(output_directory):
            print(f"Creating directory '{output_directory}'")
            os.makedirs(output_directory)

        if os.path.isfile(hdf5_opacity_file) and not rewrite:
            raise FileExistsError(f"file '{hdf5_opacity_file}' already exists, "
                                  f"set rewrite to True to rewrite the file")

        print(f" Writing file '{hdf5_opacity_file}'...")

        write_correlated_k(
            file=hdf5_opacity_file,
            doi=doi,
            wavenumbers=bin_centers,
            wavenumbers_bins_edges=bin_edges,
            cross_sections=correlated_k,
            mol_mass=molmass,
            species=species,
            opacities_pressures=unique_pressures,
            opacities_temperatures=unique_temperatures,
            g_gauss=samples,
            weights_gauss=weights,
            wavelengths=None,
            n_g=None,
            contributor=contributor,
            description=description
        )

        print(f"Successfully converted files in '{opacities_directory}' to correlated-k pRT files")


def get_opacity_filename(resolving_power, wavelength_boundaries, species_isotopologue_name,
                         source, natural_abundance='', charge='', cloud_info=''):
    if resolving_power < 1e6:
        resolving_power = f"{resolving_power:.0f}"
    else:
        decimals = np.mod(resolving_power / 10 ** np.floor(np.log10(resolving_power)), 1)

        if decimals >= 1e-3:
            resolving_power = f"{resolving_power:.3e}"
        else:
            resolving_power = f"{resolving_power:.0e}"

    spectral_info = (f"R{resolving_power}_"
                     f"{wavelength_boundaries[0]:.1f}-{wavelength_boundaries[1]:.1f}mu")
    spectral_info = spectral_info.replace('e+0', 'e').replace('e-0', 'e-')
    spectral_info = spectral_info.replace('.0-', '-').replace('.0mu', 'mu')

    return Opacity.join_species_all_info(
        species_name=species_isotopologue_name.replace('-NatAbund', ''),
        natural_abundance=natural_abundance,
        charge=charge,
        cloud_info=cloud_info,
        source=source,
        spectral_info=spectral_info
    )


def line_by_line_opacities_dat2h5(directory, output_name, molmass, doi,
                                  path_input_data=petitradtrans_config_parser.get_input_data_path(),
                                  contributor=None, description=None,
                                  opacities_pressures=None, opacities_temperatures=None, line_paths=None,
                                  memory_map_mode=False, rewrite=False, clean=False):
    """Using ExoMol units for HDF5 files."""
    LineByLineOpacity.check_name(output_name)

    if output_name is None:
        output_name = directory.rsplit(os.path.sep, 1)[1]

    if opacities_pressures is None or opacities_temperatures is None:
        print("Loading default PT grid...")

        opacities_temperature_profile_grid = np.genfromtxt(
            os.path.join(path_input_data, 'opa_input_files', 'opa_PT_grid.dat')
        )

        opacities_temperature_profile_grid = np.flip(opacities_temperature_profile_grid, axis=1)

        opacities_temperatures = np.unique(opacities_temperature_profile_grid[:, 0])
        opacities_pressures = np.unique(opacities_temperature_profile_grid[:, 1])  # grid is already in bar

    if line_paths is None:
        print("Loading default files names...")
        line_paths = np.loadtxt(os.path.join(path_input_data, 'opa_input_files', 'opa_filenames.txt'), dtype=str)

    species = directory.rsplit(os.path.sep, 1)[1]
    species = species.rsplit('_def', 1)[0]

    # Check HDF5 file existence
    hdf5_opacity_file = os.path.abspath(os.path.join(str(directory), '..', output_name + '.xsec.petitRADTRANS.h5'))

    if os.path.isfile(hdf5_opacity_file) and not rewrite:
        __print_skipping_message(hdf5_opacity_file)
        return

    # Read dat file
    print(f"Converting opacities in '{directory}'...")

    custom_pt_grid_file = os.path.join(directory, 'PTpaths.ls')

    if os.path.isfile(custom_pt_grid_file):
        print(" Found custom PT grid")

        # _sort_opa_pt_grid converts bar into cgs
        custom_grid_data = _sort_pressure_temperature_grid(custom_pt_grid_file)

        opacities_temperature_profile_grid_ = custom_grid_data[0]
        opacities_temperatures_ = np.unique(opacities_temperature_profile_grid_[:, 0])
        opacities_pressures_ = np.unique(opacities_temperature_profile_grid_[:, 1])
        opacities_pressures_ *= 1e-6  # cgs to bar
        line_paths_ = custom_grid_data[1]

        for i, line_path in enumerate(line_paths_):
            line_paths_[i] = directory + os.path.sep + line_path

        line_paths_ = np.array(line_paths_)
    else:
        print(" Using default PT grid")

        opacities_temperatures_ = copy.deepcopy(opacities_temperatures)
        opacities_pressures_ = copy.deepcopy(opacities_pressures)

        line_paths_ = []

        for f_ in os.scandir(directory):
            if f_.is_file():
                line_paths_.append(f_.path)

        f_ = []

        for ref_path in line_paths:
            for i, line_path in enumerate(line_paths_):
                if ref_path in line_path:
                    f_.append(line_paths_.pop(i))

                    break

        line_paths_ = np.array(f_)

        if line_paths_.size != line_paths.size:
            warnings.warn(f"number of opacity files founds in '{directory}' ({line_paths_.size}) "
                          f"does not match the expected number of files ({line_paths.size})")

    molparam_file = os.path.join(directory, 'molparam_id.txt')

    # Convert units and shape
    if os.path.isfile(molparam_file):
        print(" Loading isotopic ratio...")
        with open(molparam_file, 'r') as f2:
            isotopic_ratio = float(f2.readlines()[-1])
    else:
        raise FileNotFoundError(f"file '{molparam_file}' not found: unable to load isotopic ratio")

    n_lines = finput.count_file_line_number(os.path.join(directory, 'wlen.dat'))
    wavelengths = finput.load_all_line_by_line_opacities(os.path.join(directory, 'wlen.dat'), n_lines)
    wavenumbers = 1 / wavelengths[::-1]  # cm to cm-1

    if not memory_map_mode:
        memory_map_file = None
        opacities = np.zeros((line_paths_.size, wavelengths.size))
    else:  # thanks to Luke Finnerty for this fix
        memory_map_file = 'temp.memmap'
        opacities = np.memmap(memory_map_file, dtype='float32', mode='w+', shape=(line_paths_.size, wavelengths.size))

    for i, line_path in enumerate(line_paths_):
        if not os.path.isfile(str(line_path)):
            raise FileNotFoundError(f"file '{line_path}' does not exists")

        print(f" Loading file '{line_path}' ({i + 1}/{line_paths_.size})...")

        opacities[i] = finput.load_all_line_by_line_opacities(line_path, n_lines)

    print(" Reshaping...")
    opacities = opacities.reshape((opacities_temperatures_.size, opacities_pressures_.size, wavelengths.size))
    # Exo-Mol axis order (pressures, temperatures, wavenumbers, g)
    opacities = np.moveaxis(opacities, 0, 1)
    opacities = opacities[:, :, ::-1]  # match the wavenumber order
    opacities *= 1 / isotopic_ratio * molmass * cst.amu

    # Write converted file
    print(f" Writing file '{hdf5_opacity_file}'...", end=' ')

    write_line_by_line(
        file=hdf5_opacity_file,
        doi=doi,
        wavenumbers=wavenumbers,
        opacities=opacities,
        mol_mass=molmass,
        species=species,
        opacities_pressures=opacities_pressures_,
        opacities_temperatures=opacities_temperatures_,
        wavelengths=wavelengths,
        contributor=contributor,
        description=description
    )

    del opacities

    if memory_map_mode:
        os.remove(memory_map_file)

    print("Done.")

    if clean:
        __remove_files([directory])


def load_dace(file, file_extension, molmass, wavelength_file=None, wavenumbers_petitradtrans_line_by_line=None,
              save_line_by_line=False, rebin=True, selection=None):
    """Read a DACE opacity file."""
    import struct

    if wavenumbers_petitradtrans_line_by_line is None:
        raise TypeError("missing 1 required argument: 'wavenumbers_petitradtrans_line_by_line'")

    if wavelength_file is not None:
        warnings.warn("Dace opacities does not require a wavelength file\n"
                      "Set 'wavelength_file' to None to remove this warning")

    if not rebin:
        raise NotImplementedError("using pre-rebinned files for Dace opacities is not implemented")

    # Open file
    with open(file, mode='rb') as f:
        # Read content
        data = f.read()

    # The number of bytes per entry is 4
    # Get the number of datapoints
    n_points = int(len(data) * 0.25)
    # Create array of the appropriate length
    cross_sections = np.ones(n_points)

    # Read the binary data into the array
    for i in range(int(n_points)):
        value = struct.unpack('f', data[i * 4:(i + 1) * 4])
        cross_sections[i] = value[0]

    cross_sections *= cst.amu * molmass  # cm2.g-1 to cm2.molecule-1

    filename = file.rsplit(os.path.sep, 1)[-1]

    temperature = int(filename.split('_')[3])
    pressure = str(
        filename.split('_')[4].split(file_extension)[0].replace('n', '-').replace('p', ' ')
    )

    pressure = pressure[:2] + '.' + pressure[2:]
    pressure = np.round(1e1 ** float(pressure), 10)

    wavenumber_start = filename.split('_', 3)[1:3]
    wavenumber_end = int(wavenumber_start[-1])
    wavenumber_start = int(wavenumber_start[0])
    wavenumbers = np.linspace(wavenumber_start, wavenumber_end, cross_sections.size)

    # Handle insufficient wavelength coverage
    d_wavenumbers = float(np.mean(np.diff(wavenumbers)))

    if wavenumber_start > wavenumbers_petitradtrans_line_by_line[selection[0]]:
        warnings.warn(
          f"wavenumber coverage of converted opacities does not extend to "
          f"the requested wavelength lower value "
          f"({wavenumber_start}  cm-1 > {wavenumbers_petitradtrans_line_by_line[selection[0]]} cm-1)"
          f"\nOpacities set to 0 within {wavenumbers_petitradtrans_line_by_line[selection[0]]}--{wavenumber_start} cm-1"
        )
        wavenumbers = np.insert(
            wavenumbers,
            0,
            [
                wavenumbers_petitradtrans_line_by_line[selection[0]],
                wavenumber_start - d_wavenumbers
            ]
        )
        cross_sections = np.insert(cross_sections, 0, [0, 0])

    if wavenumber_end < wavenumbers_petitradtrans_line_by_line[selection[1]]:
        warnings.warn(
            f"wavenumber coverage of converted opacities does not extend to "
            f"the requested wavelength higher value "
            f"({wavenumber_end} cm-1 < {wavenumbers_petitradtrans_line_by_line[selection[1]]} cm-1)"
            f"\nOpacities set to 0 within {wavenumber_end}--{wavenumbers_petitradtrans_line_by_line[selection[1]]} cm-1"
        )
        wavenumbers = np.concatenate(
            [wavenumbers, [wavenumber_end + d_wavenumbers, wavenumbers_petitradtrans_line_by_line[selection[1]]]]
        )
        cross_sections = np.concatenate([cross_sections, [0, 0]])

    # Interpolate the Dace cross-sections to pRT's line-by-line grid
    sig_interp = interp1d(wavenumbers, cross_sections)
    sigmas_prt = sig_interp(wavenumbers_petitradtrans_line_by_line[selection[0]:selection[1]])

    # Check if interp values are below 0 or NaN
    if np.any(np.less(sigmas_prt, 0)):
        raise ValueError("interpolated opacity has negative values")

    if np.any(~np.isfinite(sigmas_prt)):
        raise ValueError("interpolated opacity has invalid values")

    if np.all(np.equal(sigmas_prt, 0)):
        raise ValueError

    if save_line_by_line:
        cross_sections_line_by_line = sigmas_prt
    else:
        cross_sections_line_by_line = None
        warnings.warn("Dace opacities do not extend enough "
                      "to be on the standard petitRADTRANS correlated-k wavelength grid")

    return cross_sections, cross_sections_line_by_line, wavenumbers, pressure, temperature


def load_exocross(file, file_extension, molmass=None, wavelength_file=None, wavenumbers_petitradtrans_line_by_line=None,
                  save_line_by_line=False, rebin=True, selection=None):
    if wavelength_file is None:
        raise TypeError("missing required argument 'wavelength_file'")

    if wavenumbers_petitradtrans_line_by_line is None:
        raise TypeError("missing required argument 'wavenumbers_petitradtrans_line_by_line'")

    if molmass is not None:
        pass  # silent warning

    if selection is None:
        selection = np.array([0, -1])

    file_name = file.rsplit(os.path.sep, 1)[-1]

    if file_name.count('_') < 2:
        raise ValueError(f"ExoCross filename must contains, in that order, the species, temperature, and pressure "
                         f"of the cross-sections, separated by '_' (e.g., 'NaH_1000K_1em5bar.out.xsec'), but was "
                         f"'{file_name}'")

    species, temperature, pressure = file_name.rsplit('_', 2)
    pressure = pressure.rsplit(file_extension, 1)[0]

    if temperature[-1] != 'K':
        raise ValueError(f"the temperature of the ExoCross filename '{file_name}' must be in K, "
                         f"but was '{temperature}'")

    temperature = float(temperature[:-1])  # pop unit and convert to float

    if pressure[-3:] != 'bar':
        raise ValueError(f"the pressure of the ExoCross filename '{file_name}' must be in bar, "
                         f"but was '{pressure}'")

    pressure = pressure[:-3]  # pop unit
    pressure = pressure.replace('m', '-').replace('p', '+')  # convert exponential notation for float conversion
    pressure = float(pressure)

    if rebin:
        data = np.genfromtxt(file)

        wavenumbers = data[:, 0]
        cross_sections = data[:, 1]

        # Interpolate the ExoCross calculation to that grid
        print(" Interpolating...")
        sig_interp = interp1d(wavenumbers, cross_sections)
        sigmas_prt = sig_interp(wavenumbers_petitradtrans_line_by_line[selection[0]:selection[1]])
    else:
        n_lines = finput.count_file_line_number_sequential(file)
        sigma = finput.load_all_line_by_line_opacities_sequential(file, n_lines)
        wavenumbers = finput.load_all_line_by_line_opacities_sequential(
            wavelength_file, n_lines
        )

        sigma = sigma[::-1]
        sigmas_prt = copy.deepcopy(sigma)
        cross_sections = sigma
        wavenumbers = 1 / wavenumbers[::-1]

    if save_line_by_line:
        cross_sections_line_by_line = sigmas_prt[selection[0]:selection[1]]
    else:
        cross_sections_line_by_line = None

    return cross_sections, cross_sections_line_by_line, wavenumbers, pressure, temperature


def rebin_ck_line_opacities(input_file, target_resolving_power, wavenumber_grid=None, rewrite=False):
    try:
        import exo_k
    except ImportError:
        # Only raise a warning to give a chance to download the binned-down file
        warnings.warn("binning down of opacities requires exo_k to be installed, no binning down has been performed")
        return -1

    if rank == 0:  # prevent race condition when writing the binned down file during multi-processes execution
        if wavenumber_grid is None:
            # Define own wavenumber grid, make sure that log spacing is constant everywhere
            wavelengths_boundaries = _get_default_rebinning_wavelength_range()

            n_spectral_points = int(
                target_resolving_power * np.log(wavelengths_boundaries[1] / wavelengths_boundaries[0]) + 1
            )

            wavenumber_grid = np.logspace(
                np.log10(1 / wavelengths_boundaries[1] * 1e4),
                np.log10(1 / wavelengths_boundaries[0] * 1e4),
                n_spectral_points
            )

        # Output files
        output_file = input_file.replace(
            CorrelatedKOpacity.join_species_all_info(
                '',
                spectral_info=CorrelatedKOpacity.get_resolving_power_string(
                    CorrelatedKOpacity.get_default_resolving_power()
                )
            ),
            CorrelatedKOpacity.join_species_all_info(
                '', spectral_info=CorrelatedKOpacity.get_resolving_power_string(target_resolving_power)
            )
        )

        if os.path.isfile(output_file) and not rewrite:
            print(f"File '{output_file}' already exists, skipping re-binning...")
        else:
            print(f"Rebinning file '{input_file}' to R = {target_resolving_power}...")
            # Use Exo-k to rebin to low-res
            tab = exo_k.Ktable(filename=input_file, remove_zeros=True)
            tab.bin_down(wavenumber_grid)
            print(" Done.")

            print(f" Writing binned down file '{output_file}'...", end=' ')
            tab.write_hdf5(output_file)
            print("Done.")

            print(f"Successfully binned down k-table into '{output_file}' (R = {target_resolving_power})")
    elif not isinstance(rank, int):
        raise TypeError(
            f"MPI rank must be of type '{int.__name__}', not {type(rank)}\n"
            f"This is likely a code error and not a user error, you may report this to "
            f"https://gitlab.com/mauricemolli/petitRADTRANS/-/issues"
        )

    if comm is not None:  # wait for the main process to finish the binning down
        comm.barrier()

    return 0  # execution without error


def rebin_multiple_ck_line_opacities(target_resolving_power, paths=None, species=None, rewrite=False):
    if species is None:
        species = []

    if paths is None:
        paths = []

    # Define own wavenumber grid, make sure that log spacing is constant everywhere
    wavelengths_boundaries = _get_default_rebinning_wavelength_range()
    n_spectral_points = int(
        target_resolving_power * np.log(wavelengths_boundaries[1] / wavelengths_boundaries[0]) + 1
    )

    wavenumber_grid = np.logspace(
        np.log10(1 / wavelengths_boundaries[1] * 1e4),
        np.log10(1 / wavelengths_boundaries[0] * 1e4),
        n_spectral_points
    )

    success = False

    # Do the rebinning, loop through species
    for i, s in enumerate(species):
        # Output files
        hdf5_opacity_file_input = paths[i]

        state = rebin_ck_line_opacities(
            input_file=hdf5_opacity_file_input,
            target_resolving_power=target_resolving_power,
            wavenumber_grid=wavenumber_grid,
            rewrite=rewrite
        )

        if state is None:
            success = True
        elif state == -1:
            success = False
            break

    if success:
        print("Successfully binned down all k-tables\n")


def write_cia_opacities(hdf5_cia_file, molecules, wavenumbers, wavelengths, alpha, temperatures,
                        doi='', description=''):
    with h5py.File(hdf5_cia_file, "w") as fh5:
        dataset = fh5.create_dataset(
            name='DOI',
            shape=(1,),
            data=doi
        )
        dataset.attrs['long_name'] = 'Data object identifier linked to the data'
        dataset.attrs['additional_description'] = description

        dataset = fh5.create_dataset(
            name='Date_ID',
            shape=(1,),
            data=f'petitRADTRANS-v{petitRADTRANS.__version__}_'
                 f'{datetime.datetime.now(datetime.timezone.utc).isoformat()}'
        )
        dataset.attrs['long_name'] = 'ISO 8601 UTC time (https://docs.python.org/3/library/datetime.html) ' \
                                     'at which the table has been created, ' \
                                     'along with the version of petitRADTRANS'

        dataset = fh5.create_dataset(
            name='wavenumbers',
            data=wavenumbers
        )
        dataset.attrs['long_name'] = 'CIA wavenumbers'
        dataset.attrs['units'] = 'cm^-1'

        dataset = fh5.create_dataset(
            name='alpha',
            data=alpha
        )
        dataset.attrs['long_name'] = 'Table of monochromatic absorption with axes (temperature, wavenumber)'
        dataset.attrs['units'] = 'cm^-1'

        dataset = fh5.create_dataset(
            name='t',
            data=temperatures
        )
        dataset.attrs['long_name'] = 'Temperature grid'
        dataset.attrs['units'] = 'K'

        dataset = fh5.create_dataset(
            name='mol_mass',
            data=np.array([float(get_species_molar_mass(species)) for species in molecules])
        )
        dataset.attrs['long_name'] = 'Masses of the colliding species'
        dataset.attrs['units'] = 'AMU'

        dataset = fh5.create_dataset(
            name='mol_name',
            data=molecules
        )
        dataset.attrs['long_name'] = 'Names of the colliding species described'

        dataset = fh5.create_dataset(
            name='wlrange',
            data=np.array([wavelengths.min(), wavelengths.max()]) * 1e4  # cm to um
        )
        dataset.attrs['long_name'] = 'Wavelength range covered'
        dataset.attrs['units'] = 'µm'

        dataset = fh5.create_dataset(
            name='wnrange',
            data=np.array([wavenumbers.min(), wavenumbers.max()])
        )
        dataset.attrs['long_name'] = 'Wavenumber range covered'
        dataset.attrs['units'] = 'cm^-1'


def write_cloud_opacities(hdf5_opacity_file, cloud_species, wavenumbers, cloud_wavelengths,
                          cloud_absorption_opacities, cloud_scattering_opacities, cloud_asymmetry_parameter,
                          cloud_particles_densities, cloud_particles_radii, cloud_particles_radius_bins,
                          doi='', description=''):
    with h5py.File(hdf5_opacity_file, "w") as fh5:
        dataset = fh5.create_dataset(
            name='DOI',
            shape=(1,),
            data=doi
        )
        dataset.attrs['long_name'] = 'Data object identifier linked to the data'
        dataset.attrs['additional_description'] = str(description)

        dataset = fh5.create_dataset(
            name='Date_ID',
            shape=(1,),
            data=f'petitRADTRANS-v{petitRADTRANS.__version__}'
                 f'_{datetime.datetime.now(datetime.timezone.utc).isoformat()}'
        )
        dataset.attrs['long_name'] = 'ISO 8601 UTC time (https://docs.python.org/3/library/datetime.html) ' \
                                     'at which the table has been created, ' \
                                     'along with the version of petitRADTRANS'

        dataset = fh5.create_dataset(
            name='wavenumbers',
            data=wavenumbers
        )
        dataset.attrs['long_name'] = 'Opacities wavenumbers'
        dataset.attrs['units'] = 'cm^-1'

        dataset = fh5.create_dataset(
            name='absorption_opacities',
            data=cloud_absorption_opacities
        )
        dataset.attrs['long_name'] = 'Table of the absorption opacities with axes (particle radius, wavenumber)'
        dataset.attrs['units'] = 'cm^2.g^-1'

        dataset = fh5.create_dataset(
            name='scattering_opacities',
            data=cloud_scattering_opacities
        )
        dataset.attrs['long_name'] = 'Table of the scattering opacities with axes (particle radius, wavenumber)'
        dataset.attrs['units'] = 'cm^2.g^-1'

        dataset = fh5.create_dataset(
            name='asymmetry_parameters',
            data=cloud_asymmetry_parameter
        )
        dataset.attrs['long_name'] = 'Table of the asymmetry parameters with axes (particle radius, wavenumber)'
        dataset.attrs['units'] = 'None'

        dataset = fh5.create_dataset(
            name='particles_density',
            data=cloud_particles_densities
        )
        dataset.attrs['long_name'] = 'Average density of the cloud particles'
        dataset.attrs['units'] = 'g.cm^-3'

        dataset = fh5.create_dataset(
            name='mol_name',
            shape=(1,),
            data=cloud_species
        )
        dataset.attrs['long_name'] = 'Name of the species described, "(c)" indicates that it has condensed'

        dataset = fh5.create_dataset(
            name='particles_radii',
            data=cloud_particles_radii
        )
        dataset.attrs['long_name'] = 'Particles average radius grid'
        dataset.attrs['units'] = 'cm'

        dataset = fh5.create_dataset(
            name='particle_radius_bins',
            data=cloud_particles_radius_bins
        )
        dataset.attrs['long_name'] = 'Particles average radius grid bins'
        dataset.attrs['units'] = 'cm'

        dataset = fh5.create_dataset(
            name='wlrange',
            data=np.array([cloud_wavelengths.min(), cloud_wavelengths.max()]) * 1e4  # cm to um
        )
        dataset.attrs['long_name'] = 'Wavelength range covered'
        dataset.attrs['units'] = 'µm'

        dataset = fh5.create_dataset(
            name='wnrange',
            data=np.array([wavenumbers.min(), wavenumbers.max()])
        )
        dataset.attrs['long_name'] = 'Wavenumber range covered'
        dataset.attrs['units'] = 'cm^-1'


def write_correlated_k(file, doi, wavenumbers, wavenumbers_bins_edges, cross_sections, mol_mass, species,
                       opacities_pressures, opacities_temperatures, g_gauss, weights_gauss,
                       wavelengths=None, n_g=None,
                       contributor=None, description=None):
    if wavelengths is None:
        wavelengths = np.array([1 / wavenumbers[0], 1 / wavenumbers[-1]])

    if n_g is None:
        n_g = g_gauss.size

    with h5py.File(file, "w") as fh5:
        dataset = fh5.create_dataset(
            name='DOI',
            shape=(1,),
            data=doi
        )
        dataset.attrs['long_name'] = 'Data object identifier linked to the data'
        dataset.attrs['contributor'] = str(contributor)
        dataset.attrs['additional_description'] = str(description)

        dataset = fh5.create_dataset(
            name='Date_ID',
            shape=(1,),
            data=f'petitRADTRANS-v{petitRADTRANS.__version__}'
                 f'_{datetime.datetime.now(datetime.timezone.utc).isoformat()}'
        )
        dataset.attrs['long_name'] = 'ISO 8601 UTC time (https://docs.python.org/3/library/datetime.html) ' \
                                     'at which the table has been created, ' \
                                     'along with the version of petitRADTRANS'

        dataset = fh5.create_dataset(
            name='bin_centers',
            data=wavenumbers
        )
        dataset.attrs['long_name'] = 'Centers of the wavenumber bins'
        dataset.attrs['units'] = 'cm^-1'

        dataset = fh5.create_dataset(
            name='bin_edges',
            data=wavenumbers_bins_edges
        )
        dataset.attrs['long_name'] = 'Separations between the wavenumber bins'
        dataset.attrs['units'] = 'cm^-1'

        dataset = fh5.create_dataset(
            name='kcoeff',
            data=cross_sections
        )
        dataset.attrs['long_name'] = ('Table of the k-coefficients with axes '
                                      '(pressure, temperature, wavenumber, g space)')
        dataset.attrs['units'] = 'cm^2/molecule'

        dataset = fh5.create_dataset(
            name='method',
            shape=(1,),
            data='petit_samples'
        )
        dataset.attrs['long_name'] = 'Name of the method used to sample g-space'

        dataset = fh5.create_dataset(
            name='mol_mass',
            shape=(1,),
            data=float(mol_mass)
        )
        dataset.attrs['long_name'] = 'Mass of the species'
        dataset.attrs['units'] = 'AMU'

        dataset = fh5.create_dataset(
            name='mol_name',
            shape=(1,),
            data=species.split('_', 1)[0]
        )
        dataset.attrs['long_name'] = 'Name of the species described'

        dataset = fh5.create_dataset(
            name='ngauss',
            data=n_g
        )
        dataset.attrs['long_name'] = 'Number of points used to sample the g-space'

        dataset = fh5.create_dataset(
            name='p',
            data=opacities_pressures
        )
        dataset.attrs['long_name'] = 'Pressure grid'
        dataset.attrs['units'] = 'bar'

        dataset = fh5.create_dataset(
            name='samples',
            data=g_gauss
        )
        dataset.attrs['long_name'] = 'Abscissas used to sample the k-coefficients in g-space'

        dataset = fh5.create_dataset(
            name='t',
            data=opacities_temperatures
        )
        dataset.attrs['long_name'] = 'Temperature grid'
        dataset.attrs['units'] = 'K'

        dataset = fh5.create_dataset(
            name='temperature_grid_type',
            shape=(1,),
            data='regular'
        )
        dataset.attrs['long_name'] = 'Whether the temperature grid is "regular" ' \
                                     '(same temperatures for all pressures) or "pressure-dependent"'

        dataset = fh5.create_dataset(
            name='weights',
            data=weights_gauss
        )
        dataset.attrs['long_name'] = 'Weights used in the g-space quadrature'

        dataset = fh5.create_dataset(
            name='wlrange',
            data=np.array([wavelengths.min(), wavelengths.max()]) * 1e4  # cm to um
        )
        dataset.attrs['long_name'] = 'Wavelength range covered'
        dataset.attrs['units'] = 'µm'

        dataset = fh5.create_dataset(
            name='wnrange',
            data=np.array([wavenumbers.min(), wavenumbers.max()])
        )
        dataset.attrs['long_name'] = 'Wavenumber range covered'
        dataset.attrs['units'] = 'cm^-1'


def write_line_by_line(file, doi, wavenumbers, opacities, mol_mass, species,
                       opacities_pressures, opacities_temperatures, wavelengths=None,
                       contributor=None, description=None):
    if wavelengths is None:
        wavelengths = np.array([1 / wavenumbers[0], 1 / wavenumbers[-1]])

    with h5py.File(file, "w") as fh5:
        dataset = fh5.create_dataset(
            name='DOI',
            shape=(1,),
            data=doi
        )
        dataset.attrs['long_name'] = 'Data object identifier linked to the data'
        dataset.attrs['contributor'] = str(contributor)
        dataset.attrs['additional_description'] = str(description)

        dataset = fh5.create_dataset(
            name='Date_ID',
            shape=(1,),
            data=f'petitRADTRANS-v{petitRADTRANS.__version__}'
                 f'_{datetime.datetime.now(datetime.timezone.utc).isoformat()}'
        )
        dataset.attrs['long_name'] = 'ISO 8601 UTC time (https://docs.python.org/3/library/datetime.html) ' \
                                     'at which the table has been created, ' \
                                     'along with the version of petitRADTRANS'

        dataset = fh5.create_dataset(
            name='bin_edges',
            data=wavenumbers
        )
        dataset.attrs['long_name'] = 'Wavenumber grid'
        dataset.attrs['units'] = 'cm^-1'

        dataset = fh5.create_dataset(
            name='xsecarr',
            data=opacities
        )
        dataset.attrs['long_name'] = 'Table of the cross-sections with axes (pressure, temperature, wavenumber)'
        dataset.attrs['units'] = 'cm^2/molecule'

        dataset = fh5.create_dataset(
            name='mol_mass',
            shape=(1,),
            data=float(mol_mass)
        )
        dataset.attrs['long_name'] = 'Mass of the species'
        dataset.attrs['units'] = 'AMU'

        dataset = fh5.create_dataset(
            name='mol_name',
            shape=(1,),
            data=species.split('_', 1)[0]
        )
        dataset.attrs['long_name'] = 'Name of the species described'

        dataset = fh5.create_dataset(
            name='p',
            data=opacities_pressures
        )
        dataset.attrs['long_name'] = 'Pressure grid'
        dataset.attrs['units'] = 'bar'

        dataset = fh5.create_dataset(
            name='t',
            data=opacities_temperatures
        )
        dataset.attrs['long_name'] = 'Temperature grid'
        dataset.attrs['units'] = 'K'

        dataset = fh5.create_dataset(
            name='temperature_grid_type',
            shape=(1,),
            data='regular'
        )
        dataset.attrs['long_name'] = 'Whether the temperature grid is "regular" ' \
                                     '(same temperatures for all pressures) or "pressure-dependent"'

        dataset = fh5.create_dataset(
            name='wlrange',
            data=np.array([wavelengths.min(), wavelengths.max()]) * 1e4  # cm to um
        )
        dataset.attrs['long_name'] = 'Wavelength range covered'
        dataset.attrs['units'] = 'µm'

        dataset = fh5.create_dataset(
            name='wnrange',
            data=np.array([wavenumbers.min(), wavenumbers.max()])
        )
        dataset.attrs['long_name'] = 'Wavenumber range covered'
        dataset.attrs['units'] = 'cm^-1'


def convert_all(path_input_data=petitradtrans_config_parser.get_input_data_path(),
                memory_map_mode=False, rewrite=False, old_paths=False, clean=False):
    path_input_data = os.path.abspath(path_input_data)

    if not old_paths:
        print("Refactoring input data folder...")
        _refactor_input_data_folder(path_input_data=path_input_data)
        print("Refactoring done\n----")

    print("Starting all conversions...")

    print("Stellar spectra...")
    _phoenix_spec_dat2h5(path_input_data=path_input_data, rewrite=rewrite, old_paths=old_paths, clean=clean)

    print("Chemical tables...")
    _chemical_table_dat2h5(path_input_data=path_input_data, rewrite=rewrite, old_paths=old_paths, clean=clean)

    print("CIA...")
    _continuum_cia_dat2h5(path_input_data=path_input_data, rewrite=rewrite, old_paths=old_paths, clean=clean)

    print("Clouds continuum...")
    _continuum_clouds_opacities_dat2h5(path_input_data=path_input_data, rewrite=rewrite,
                                       old_paths=old_paths, clean=clean)

    print("Correlated-k opacities...")
    _correlated_k_opacities_dat2h5(path_input_data=path_input_data, rewrite=rewrite, old_paths=old_paths, clean=clean)

    print("Line-by-line opacities...")
    _line_by_line_opacities_dat2h5(
        path_input_data=path_input_data,
        memory_map_mode=memory_map_mode, rewrite=rewrite, old_paths=old_paths, clean=clean
    )

    print("Successfully converted all .dat files into HDF5")

    if clean:
        print("Starting final cleaning...")
        _clean_input_data_mac_junk_files(path_input_data)
        __remove_files([os.path.join(path_input_data, 'opa_input_files')])
        print("Successfully removed opa_input_files directory")
        print("Final cleaning complete")
