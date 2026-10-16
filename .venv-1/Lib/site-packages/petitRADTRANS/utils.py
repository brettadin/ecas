"""Stores useful generic functions.
"""
import copy
import csv
import sys
import warnings

import h5py
import numpy as np
import numpy.typing as npt


class LockedDict(dict):
    """Derivative of dict with a lock.
    Can be used to ensure that no new key is added once the lock is on, to prevent errors due to key typos.
    """

    def __init__(self):
        self._locked = False
        super().__init__()

    def __copy__(self):
        """Override the copy.copy method. Necessary to allow locked LockedDict to be copied."""
        cls = self.__class__
        result = cls.__new__(cls)

        result.unlock()  # force initialization of _locked

        # First copy the keys in the new object
        for key, value in self.items():
            result[key] = value

        # Then copy the attributes to prevent the effect of the lock
        result.__dict__.update(self.__dict__)

        return result

    def __deepcopy__(self, memo):
        """Override the copy.deepcopy method. Necessary to allow locked LockedDict to be copied."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        result.unlock()  # force initialization of _locked

        # First copy the keys in the new object
        for key, value in self.items():
            key = copy.deepcopy(key, memo)
            value = copy.deepcopy(value, memo)
            result[key] = value

        # Then copy the attributes to prevent the effect of the lock
        for key, value in self.__dict__.items():
            setattr(result, key, copy.deepcopy(value, memo))

        return result

    def __setitem__(self, key, value):
        """Prevent a key to be added if the lock is on."""
        if not hasattr(self, '_locked'):
            self._locked = False

        if key not in self and self._locked:
            raise KeyError(f"'{key}' not in locked LockedDict, unlock the LockedDict to add new keys")
        else:
            if key in self:  # handle nested LockedDicts
                if isinstance(self[key], LockedDict) and isinstance(value, dict):
                    self[key].update(value)
                    value = self[key]

            super().__setitem__(key, value)

    @classmethod
    def build_and_lock(cls, dictionary=None, **kwargs):
        """Instantiate a LockedDict and lock it."""
        new_cls = cls()
        new_cls.unlock()  # ensure the dictionary can be updated
        new_cls.update(dictionary, **kwargs)
        new_cls.lock()

        return new_cls

    def lock(self):
        self._locked = True

    def update(self, __m=None, **kwargs) -> None:
        """Ensure that update takes the lock into account and do not remove keys."""
        __tmp = dict(self)

        if __m is None:
            __tmp.update(**kwargs)
        else:
            __tmp.update(__m, **kwargs)

        # Ensure that no keys are removed during update
        if len(__tmp.keys()) < len(self.keys()) and self._locked:
            raise KeyError(
                f"locked LockedDict has {len(self.keys())} items "
                f"but the update has {len(__tmp.keys())}, "
                f"unlock the LockedDict to change the number of keys during an update"
            )

        for key, value in __tmp.items():
            self.__setitem__(key, value)

    def unlock(self):
        self._locked = False


def check_all_close(a, b, **kwargs):
    if isinstance(a, dict):
        if len(a) != len(b):  # check if both dict have the same number of keys
            raise AssertionError(f"a and b have a different number of keys ({len(a)} and {len(b)})\n"
                                 f"{a=}\n"
                                 f"{b=}")

        for key, value in a.items():
            # Since there is the same number of keys, an error will be raised if a key in 'a' is not in 'b'
            check_all_close(value, b[str(key)], **kwargs)
    elif isinstance(a, str):
        if a != b:
            raise AssertionError(f"'{a}' != '{b}'")
    elif not isinstance(a, np.ndarray) and hasattr(a, '__iter__'):
        for i, value in enumerate(a):
            check_all_close(value, b[i], **kwargs)
    elif a is None:
        if b is not None:
            raise AssertionError(f"a is None but b is {b}")
    elif isinstance(b, list):
        if isinstance(b[0], str):
            for i, value in enumerate(a):
                if value != b[0]:
                    raise AssertionError(f"'{a}' != '{b}'")
        elif not np.allclose(a, b, **kwargs):
            raise AssertionError(f"a and b are not close enough\n"
                                 f"{a=}\n"
                                 f"{b=}\n"
                                 f"{kwargs}")
    else:
        if not np.allclose(a, b, **kwargs):
            raise AssertionError(f"a and b are not close enough\n"
                                 f"{a=}\n"
                                 f"{b=}\n"
                                 f"{kwargs}")


def class_init_args2class_args(string):
    """Convenience code-writing function to convert a series of arguments into lines of initialization for a class.
    Useful to quickly write the __init__ function of a class from its arguments.
    Example:
        >>> s = "arg1, arg2=0.3, arg3='a'"
        >>> print(class_init_args2class_args(s))
        output:
            self.arg1 = arg1
            self.arg2 = arg2
            self.arg3 = arg3
    """
    arguments = string.split(',')
    out_string = ''

    for argument in arguments:
        arg = argument.strip().rsplit('=', 1)[0]
        out_string += f"self.{arg} = {arg}\n"

    return out_string


def class_init_args2dict(string):
    """Convenience code-writing function to convert a series of arguments into a dictionary.
    Useful to quickly write a dictionary from a long list of arguments.
    Example:
        >>> s = "arg1, arg2=0.3, arg3='a'"
        >>> print(class_init_args2class_args(s))
        output:
            {
                'arg1': ,
                'arg2': ,
                'arg3': ,
            }
    """
    arguments = string.split(',')
    out_string = '{\n'

    for argument in arguments:
        arg = argument.strip().rsplit('=', 1)[0]
        out_string += f"    '{arg}': ,\n"

    out_string += '}'

    return out_string


def class2hdf5(obj, filename=None):
    """Convert an instance of a class into a HDF5 dataset."""
    with h5py.File(filename, 'w') as f:
        dict2hdf5(
            dictionary=obj.__dict__,
            hdf5_file=f
        )


def dataset2obj(obj):
    """Convert a HDF5 dataset into a list of objects (float, int or str)."""
    if hasattr(obj, '__iter__') and not isinstance(obj, bytes):
        new_obj = []

        for o in obj:
            new_obj.append(dataset2obj(o))

        return np.array(new_obj)
    elif isinstance(obj, bytes):
        obj = str(obj, 'utf-8')

        if obj == 'None':
            obj = None

        return obj
    else:
        return obj


def dict2hdf5(dictionary, hdf5_file, group='/'):
    """Convert a dictionary into a HDF5 dataset."""
    if len(dictionary) == 0:
        hdf5_file.create_dataset(
            name=group + '__EMPTY_DICT__',
            data=np.nan
        )

    for key in dictionary:
        if isinstance(dictionary[key], dict):  # create a new group for the dictionary
            new_group = group + str(key) + '/'
            dict2hdf5(dictionary[key], hdf5_file, new_group)
        elif callable(dictionary[key]):
            print(f"Skipping callable '{key}': dtype('O') has no native HDF5 equivalent")
        else:
            if dictionary[key] is None:
                data = 'None'
            elif isinstance(dictionary[key], set):
                data = list(dictionary[key])
            elif hasattr(dictionary[key], 'dtype'):
                if dictionary[key].dtype == 'O':
                    data = flatten_object(dictionary[key])
                elif isinstance(dictionary[key].dtype, np.dtypes.StrDType):
                    data = [str(value) for value in dictionary[key]]
                else:
                    data = dictionary[key]
            else:
                data = dictionary[key]

            hdf5_file.create_dataset(
                name=group + key,
                data=data
            )


def fill_object(array, value):
    """Fill a numpy object array with a value."""
    if array.dtype == 'O':
        for i, dim in enumerate(array):
            array[i] = fill_object(dim, value)
    elif array.dtype == type(value):
        array[:] = value
    else:
        array = np.ones(array.shape, dtype=type(value)) * value

    return array


def flatten_object(array):
    """Flatten a numpy object array."""
    if array.dtype == 'O':
        if np.ndim(array) <= 1:
            for i, element in enumerate(array):
                if element is None or isinstance(element, str):
                    array[i] = str(element)
                else:
                    raise ValueError(f"element '{element}' is not a number nor None, flattening is not possible")

            return array

        array = flatten_object(np.concatenate(array))
    else:
        if np.ndim(array) <= 1:
            return array

        array = np.concatenate(array)

    return array


def hdf52dict(hdf5_file):
    dictionary = {}

    for key in hdf5_file:
        if key == '__EMPTY_DICT__':
            if np.isnan(hdf5_file[key]):
                return {}

        if isinstance(hdf5_file[key], h5py.Dataset):
            dictionary[key] = dataset2obj(hdf5_file[key][()])
        elif isinstance(hdf5_file[key], h5py.Group):
            dictionary[key] = hdf52dict(hdf5_file[key])
        else:
            warnings.warn(f"Ignoring '{key}' of type '{type(hdf5_file[key])} in HDF5 file: "
                          f"hdf52dict() can only handle types 'Dataset' and 'Group'")

    return dictionary


def intersection_indices(array1: npt.NDArray, array2: npt.NDArray):
    """Get the indices of the intersection between two sorted arrays in increasing order.

    For example, if array1 is [0.1, ..., 0.3, ..., 3] and array2 is interval [0.3, ..., 3, ..., 28], then the output
    indices will be the indices corresponding to [0.3, ..., 3] in both arrays.

    Args:
        array1: the first array
        array2: the second array

    Returns:
        The indices of the intersection between both arrays, for the first and the second array.
    """
    tiny = 10 ** -sys.float_info.dig

    indices1 = np.logical_and(
        np.less_equal(array1, array2[0] * (1. + tiny)),
        np.greater_equal(array1, array2[-1] * (1. - tiny))
    )
    indices2 = np.logical_and(
        np.less_equal(array2, array1[0] * (1. + tiny)),
        np.greater_equal(array2, array1[-1] * (1. - tiny))
    )

    return indices1, indices2


def list_str2str(list_str, separator: str = ", "):
    """Convert a list of strings into a string in which each input string is quoted and separated by 'separator'.
    Example:
        >>> list_str2str(('AA', 'BBB', 'C'))
        >>> "'AA', 'BBB', 'C'"
    """
    separator = "'" + separator + "'"

    return "'" + separator.join(list_str) + "'"


def load_csv(file, **kwargs):
    data = {}
    header_read = False

    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, **kwargs)

        for row in csv_reader:
            if not header_read:
                column_names = copy.deepcopy(row)

                for column_name in column_names:
                    if '# ' in column_name:
                        column_name = column_name.split('# ', 1)[1]

                    data[column_name] = []

                header_read = True
            else:
                for i, column_name in enumerate(data):
                    data[column_name].append(float(row[i]))

    for column_name, value in data.items():
        data[column_name] = np.array(value)

    return data


def remove_mask(data, data_uncertainties):
    """Remove masked values of 3D data and linked uncertainties. TODO generalize this
    An array of objects is created if the resulting array is jagged.

    Args:
        data: 3D masked array
        data_uncertainties: 3D masked array

    Returns:
        The data and errors without the data masked values, and the mask of the original data array.
    """
    data_ = []
    error_ = []
    mask_ = copy.copy(data.mask)
    lengths = []

    for i in range(data.shape[0]):
        data_.append([])
        error_.append([])

        for j in range(data.shape[1]):
            data_[i].append(np.array(
                data[i, j, ~mask_[i, j, :]]
            ))
            error_[i].append(np.array(data_uncertainties[i, j, ~mask_[i, j, :]]))
            lengths.append(data_[i][j].size)

    # Handle jagged arrays
    if np.all(np.array(lengths) == lengths[0]):
        data_ = np.array(data_)
        error_ = np.array(error_)
    else:
        print("Array is jagged, generating object array...")
        data_ = np.array(data_, dtype=object)
        error_ = np.array(error_, dtype=object)

    return data_, error_, mask_


def savez_compressed_record(file, numpy_record_array):
    """Apply numpy.savez_compressed on a record array."""
    data_dict = {key: numpy_record_array[key] for key in numpy_record_array.dtype.names}
    np.savez_compressed(file, **data_dict)


def topological_sort(source):
    """Perform topological sort on a dictionary.

    Source: https://stackoverflow.com/questions/11557241/python-sorting-a-dependency-list

    Args:
        source: dictionary of {name: [list of dependencies]} pairs

    Returns:
        list of names, with dependencies listed first
    """
    pending = [(name, set(deps)) for name, deps in source.items()]  # copy deps so we can modify set in-place

    if None not in source:
        pending.append((None, set()))  # append None "dependency"

    emitted = []

    while pending:
        next_pending = []
        next_emitted = []

        for entry in pending:
            name, deps = entry
            deps.difference_update(emitted)  # remove deps we emitted last pass

            if deps:  # still has deps? Recheck during next pass
                next_pending.append(entry)
            else:  # no more deps? Time to emit
                yield name
                emitted.append(name)  # not required, but preserves original ordering
                next_emitted.append(name)  # remember what we emitted for difference_update() in next pass

        if not next_emitted:  # all entries have unmet deps, a dependency is missing or is cyclic
            raise ValueError(f"cyclic or missing dependency detected: {next_pending}")

        pending = next_pending
        emitted = next_emitted


def user_input(
    introduction_message: str,
    input_message: str,
    failure_message: str,
    cancel_message: str,
    mode: str,
    max_attempts: int = 5,
    list_length: int = None
) -> str | int | None:
    available_modes = ['list', 'y/n']

    if mode not in available_modes:
        quote = "'"
        raise ValueError(f"user input mode '{mode}' is not available, "
                         f"available modes are {quote + ', '.join(available_modes) + quote}")

    if mode == 'list' and list_length is None:
        raise TypeError("'list' mode missing required argument 'list_size'")

    print(introduction_message)

    for i in range(max_attempts):
        if mode == 'y/n':
            selection = input(
                f"{input_message} ('y'/'n'; 'cancel')"
            )
        elif mode == 'list':
            selection = input(
                f"{input_message} (1-{list_length}; 'cancel')"
            )
        else:
            quote = "'"
            raise ValueError(f"user input mode '{mode}' is not available, "
                             f"available modes are {quote + ', '.join(available_modes) + quote}")

        if selection == 'cancel':
            print(cancel_message)
            return None

        if mode == 'y/n':
            selection = selection.lower()

            if selection == 'y':
                selection = True
            elif selection == 'n':
                selection = False
            else:
                print(f"Unclear input '{selection}', please enter 'y' or 'n'")
                continue
        elif mode == 'list':
            if not selection.isdigit():
                print(f"'{selection}' is not an integer, please enter an integer within 1-{list_length}")
                continue

            selection = int(selection)

            if selection < 1 or selection > list_length:
                print(f"{selection} is not within the range 1-{list_length}, "
                      f"please enter an integer within 1-{list_length}")
                continue

        return selection

    raise ValueError(f"{failure_message} after {max_attempts} attempts")
