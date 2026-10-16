"""This module contains a set of useful functions that don't really fit anywhere
else. This includes flux conversions, prior functions, mean molecular weight
calculations, transforms from mass to number fractions, and fits file output.
"""
import json
import os
import warnings

import numpy as np
from scipy.special import erfcinv, gamma

SQRT2 = np.sqrt(2)


# Prior Functions
# Stolen from https://github.com/JohannesBuchner/MultiNest/blob/master/src/priors.f90
def log_prior(cube, lx1, lx2):
    return 10 ** (lx1 + cube * (lx2 - lx1))


def uniform_prior(cube, x1, x2):
    return x1 + cube * (x2 - x1)


def gaussian_prior(cube, mu, sigma):
    return mu + sigma * SQRT2 * erfcinv(2.0 * (1.0 - cube))
    # return -(((cube-mu)/sigma)**2.)/2.


def log_gaussian_prior(cube, mu, sigma):
    bracket = sigma * sigma + sigma * SQRT2 * erfcinv(2.0 * cube)
    return mu * np.exp(bracket)


def delta_prior(cube, x1, x2):
    return x1


def inverse_gamma_prior(cube, a, b):
    return ((b ** a) / gamma(a)) * (1 / cube) ** (a + 1) * np.exp(-b / cube)


# Sanity checks on parameter ranges
def b_range(x, b):
    if x > b:
        return -np.inf
    else:
        return 0.


def a_b_range(x, a, b):
    if x < a:
        return -np.inf
    elif x > b:
        return -np.inf
    else:
        return 0.


# File Formatting
def get_pymultinest_sample_dict(output_dir, name=None, add_log_likelihood=False, add_stats=False):
    if name is None:
        name = output_dir.rsplit(os.sep, 1)[1]

    if not os.path.isdir(output_dir):
        raise NotADirectoryError(f"output directory '{output_dir}' does not exist")

    sample_file = os.path.join(output_dir, 'out_PMN', name + '_post_equal_weights.dat')
    parameter_names_file = os.path.join(output_dir, 'out_PMN', name + '_params.json')

    if not os.path.isfile(sample_file):
        raise FileNotFoundError(f"sample file '{sample_file}' does not exist")

    if not os.path.isfile(parameter_names_file):
        raise FileNotFoundError(f"parameter names file '{parameter_names_file}' does not exist")

    samples = np.genfromtxt(str(os.path.join(output_dir, 'out_PMN', name + '_post_equal_weights.dat')))

    with open(parameter_names_file, 'r') as f:
        parameters_read = json.load(f)

    samples_dict = {}

    for i, key in enumerate(parameters_read):
        if np.ndim(samples) == 1:
            warnings.warn(
                f"samples in '{output_dir}' has only one live point\n"
                f"Ensure that the retrieval ran correctly and that it had enough live points "
                f"(>~ the number of free parameters)."
            )
            samples = samples[np.newaxis, :]

        samples_dict[key] = samples[:, i]

    if add_log_likelihood:
        samples_dict['log_likelihood'] = samples[:, -1]

    if add_stats:
        with open(os.path.join(output_dir, 'out_PMN', name + '_stats.json'), 'r') as f:
            parameters_read = json.load(f)

        samples_dict['stats'] = parameters_read

    return samples_dict


def get_calculate_flux_return_values(parameters):
    return_contribution = False
    return_opacities = False
    return_photosphere_radius = False
    return_rosseland_optical_depths = False
    return_radius_hydrostatic_equilibrium = False
    return_cloud_contribution = False
    return_abundances = False
    return_any = False
    if "contribution" in parameters.keys():
        return_contribution = parameters["contribution"].value
        return_any = True
    if "return_contribution" in parameters.keys():
        return_contribution = parameters["return_contribution"].value
        return_any = True
    if "return_opacities" in parameters.keys():
        return_opacities = parameters["return_opacities"].value
        return_any = True
    if "return_photosphere_radius" in parameters.keys():
        return_photosphere_radius = parameters["return_photosphere_radius"].value
        return_any = True
    if "return_abundances" in parameters.keys():
        return_abundances = parameters["return_abundances"].value
        return_any = True
    if "return_radius_hydrostatic_equilibrium" in parameters.keys():
        return_radius_hydrostatic_equilibrium = parameters["return_radius_hydrostatic_equilibrium"].value
        return_any = True
    if "return_rosseland_optical_depths" in parameters.keys():
        return_rosseland_optical_depths = parameters["return_rosseland_optical_depths"].value
        return_any = True
    if "return_cloud_contribution" in parameters.keys():
        return_cloud_contribution = parameters["return_cloud_contribution"].value
        return_any = True
    return (return_contribution,
            return_opacities,
            return_photosphere_radius,
            return_rosseland_optical_depths,
            return_radius_hydrostatic_equilibrium,
            return_cloud_contribution,
            return_abundances,
            return_any)
