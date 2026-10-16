"""SLURM run useful functions.
"""
import numpy as np


def fair_share(array, n_entities, append_within_existing=True):
    """Split the elements of an array into a given number of entities.

    Examples:
        fair_share([1, 2, 3], 2) => [[1, 3], [2]]
        fair_share([1, 2, 3], 3) => [[1], [2], [3]]
        fair_share([1, 2, 3, 4], 2) => [[1, 2], [3, 4]]
        fair_share([1, 2, 3, 4, 5], 2) => [[1, 2, 5], [3, 4]]
        fair_share([1, 2, 3, 4, 5], 2, False) => [[1, 2], [3, 4], [5]]

    Args:
        array: a numpy array
        n_entities: the number of entities
        append_within_existing: if True, leftover elements will be shared within the entities; otherwise, leftover
            elements will be added as an extra entity

    Returns:
        list with n_entities elements if append_within_existing is True, n_entities + 1 elements otherwise, each
        sharing elements of the original array.
    """
    elements_per_entities = int(np.floor(array.size / n_entities))
    n_leftover_elements = (array.size - elements_per_entities * n_entities)

    if array.size > n_leftover_elements:
        shared_array = list(
            array[:array.size - n_leftover_elements].reshape(
                n_entities, elements_per_entities
            )
        )
        leftover_elements = array[array.size - n_leftover_elements:]
    else:
        shared_array = [array]
        leftover_elements = np.array([])

    if leftover_elements.size > 0:
        if append_within_existing:
            for i, leftover_element in enumerate(leftover_elements):
                shared_array[np.mod(i, n_entities)] = np.append(shared_array[i], leftover_element)
        else:
            if array.size - n_leftover_elements <= n_entities:
                shared_array[0] = np.append(shared_array[0], leftover_elements)
            else:
                shared_array.append(leftover_elements)

    return shared_array


def load_dat(file, **kwargs):
    """
    Load a data file.

    Args:
        file: data file
        **kwargs: keywords arguments for numpy.loadtxt()

    Returns:
        data_dict: a dictionary containing the data
    """
    with open(file, 'r') as f:
        header = f.readline()
        unit_line = f.readline()

    header_keys = header.rsplit('!')[0].split('#')[-1].split()
    units = unit_line.split('#')[-1].split()

    data = np.loadtxt(file, **kwargs)
    data_dict = {}

    for i, key in enumerate(header_keys):
        data_dict[key] = data[:, i]

    data_dict['units'] = units

    return data_dict
