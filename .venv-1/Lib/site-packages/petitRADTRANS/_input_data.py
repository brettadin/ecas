import os
import warnings
from typing import Callable

from petitRADTRANS.cli.prt_cli import download_input_data, get_keeper_files_url_paths
from petitRADTRANS.config.configuration import petitradtrans_config_parser
from petitRADTRANS.utils import list_str2str, user_input


def _get_input_data_file_not_found_error_message(file: str) -> str:
    return (
        f"no matching file found in path '{file}'\n"
        f"This may be caused by an incorrect input_data path, outdated file formatting, or a missing file\n\n"
        f"To set the input_data path, execute: \n"
        f">>> from petitRADTRANS.config.configuration import petitradtrans_config_parser\n"
        f">>> petitradtrans_config_parser.set_input_data_path('path/to/input_data')\n"
        f"replacing 'path/to/' with the path to the input_data directory\n\n"
        f"To update the outdated files, execute:\n"
        f">>> from petitRADTRANS.__file_conversion import convert_all\n"
        f">>> convert_all()\n\n"
        f"To download the missing file, "
        f"see https://petitradtrans.readthedocs.io/en/latest/content/installation.html"
    )


def _get_input_file_from_keeper(full_path, path_input_data=None, sub_path=None, filename=None, match_function=None,
                                find_all=False,
                                ext='h5', timeout=3, url_input_data=None):
    if match_function is None:
        match_function = match_function_default

    if path_input_data is None:
        path_input_data = petitradtrans_config_parser.get_input_data_path()

        if path_input_data not in full_path:
            raise ValueError(f"full path '{full_path}' not within default input_data path '{path_input_data}'\n "
                             f"Set the path_input_data argument in accordance with full_path, "
                             f"or correct full_path")

    if sub_path is None:
        sub_path = full_path.split(path_input_data, 1)[1]

    url_paths = get_keeper_files_url_paths(
        path=full_path,
        ext=ext,
        timeout=timeout,
        path_input_data=path_input_data,
        url_input_data=url_input_data
    )

    matches = match_function(
        path_input_data=path_input_data,
        sub_path=sub_path,
        files=list(url_paths.keys()),
        filename=filename,
        expect_default_file_exists=False,
        find_all=find_all,
        display_other_files=True
    )

    if len(matches) == 0 and not isinstance(matches, str):
        _ = match_function(
            path_input_data=path_input_data,
            sub_path=sub_path,
            files=None,
            filename=filename,
            expect_default_file_exists=True,
            find_all=find_all,
            display_other_files=True
        )
    elif len(matches) == 1 or isinstance(matches, str):
        download_input_data(
            destination=os.path.join(full_path, matches),
            source=url_paths[matches],
            rewrite=False,
            path_input_data=path_input_data,
            url_input_data=url_input_data
        )
    else:
        files_str = [f" {i + 1}: {file}" for i, file in enumerate(matches)]
        files_str = "\n".join(files_str)

        introduction_message = (
            f"Multiple matching files found in the Keeper library, and no default file set for this path "
            f"in petitRADTRANS' configuration\n"
            f"List of matching files:\n"
            f"{files_str}"
        )

        download_all = user_input(
            introduction_message=introduction_message,
            input_message=f"Download all of the {len(matches)} matching files?",
            failure_message="unclear answer",
            cancel_message="Cancelling...",
            mode='y/n',
            list_length=len(url_paths)
        )

        if download_all is None:
            raise ValueError("Keeper file download cancelled")

        if download_all:
            for match in matches:
                download_input_data(
                    destination=os.path.join(full_path, match),
                    source=url_paths[match],
                    rewrite=False,
                    path_input_data=path_input_data,
                    url_input_data=url_input_data
                )
        else:
            new_default_file = select_default_file(
                files=tuple(url_paths.keys()),
                full_path="the Keeper library",
                sub_path=sub_path
            )

            petitradtrans_config_parser.set_default_file(
                file=os.path.join(full_path, new_default_file),
                path_input_data=path_input_data
            )

            download_input_data(
                destination=os.path.join(full_path, new_default_file),
                source=url_paths[new_default_file],
                rewrite=False,
                path_input_data=path_input_data,
                url_input_data=url_input_data
            )

    return matches


def default_file_selection(files: tuple[str, ...] | list[str], full_path: str, sub_path: str) -> str:
    files_str = [f" {i + 1}: {file}" for i, file in enumerate(files)]
    files_str = "\n".join(files_str)

    introduction_message = (
        f"More than one file detected in '{full_path}', and no default file set for this path "
        f"in petitRADTRANS' configuration\n"
        f"Please select one of the files in the list below by typing the corresponding integer:\n"
        f"{files_str}"
    )

    new_default_file: int | None = user_input(
        introduction_message=introduction_message,
        input_message=f"Select which file to set as the default file for '{sub_path}'",
        failure_message=f"failure to enter new default file for '{sub_path}'",
        cancel_message="Cancelling default file selection...",
        mode='list',
        list_length=len(files)
    )

    if new_default_file is None:
        raise ValueError(f"no default file selected for path '{sub_path}'")

    new_default_file -= 1
    new_default_file: str = files[new_default_file]

    return new_default_file


def find_input_file(file: str, path_input_data: str, sub_path: str = None,
                    match_function: Callable = None, find_all: bool = False, search_online: bool = True):
    if match_function is None:
        match_function = match_function_default

    if sub_path is None:
        full_path = os.path.dirname(file)
        _, sub_path, file = split_input_data_path(file, path_input_data)
    else:
        full_path = os.path.abspath(os.path.join(path_input_data, sub_path))

    if not os.path.isdir(full_path):  # search even if search_online is False
        print(f"No such directory '{full_path}'\n"
              f"Searching in the Keeper library...")

        matches = _get_input_file_from_keeper(
            full_path=full_path,
            path_input_data=path_input_data,
            sub_path=sub_path,
            filename=file,
            find_all=find_all
        )
    else:
        matches = match_function(
            path_input_data=path_input_data,
            sub_path=sub_path,
            filename=file,
            expect_default_file_exists=True,
            find_all=find_all
        )

        if len(matches) == 0 and search_online:
            print(f"No file matching name '{file}' found in directory '{full_path}'\n"
                  f"Searching in the Keeper library...")

            matches = _get_input_file_from_keeper(
                full_path=full_path,
                path_input_data=path_input_data,
                sub_path=sub_path,
                filename=file,
                find_all=find_all
            )

    if len(matches) == 0 and not find_all:
        raise FileNotFoundError(_get_input_data_file_not_found_error_message(full_path))

    if hasattr(matches, '__iter__') and not isinstance(matches, str):
        matches = [os.path.join(full_path, m) for m in matches]
    else:
        matches = os.path.join(full_path, matches)

    return matches


def match_function_default(path_input_data: str, sub_path: str,
                           files: list[str] = None, filename: str = None,
                           expect_default_file_exists: bool = True,
                           find_all: bool = False, display_other_files: bool = False):
    full_path: str = os.path.join(path_input_data, sub_path)

    if not os.path.isdir(full_path) and files is None:  # directory does not exist, return empty list
        return []

    if files is None:
        files = [f for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f))]

    if len(files) == 0:  # no file in path, return empty list
        return []
    else:  # at least one file detected in path
        matching_files: list[str] = []

        if filename is not None:  # check if one of the files matches the given filename
            matching_files = [
                file for file in files if file.startswith(filename)
            ]

            if len(matching_files) == 0:
                if display_other_files:
                    files_str = "\n".join(files)
                    warnings.warn(f"no file matching name '{filename}' found in directory '{full_path}'\n"
                                  f"Available files are:\n"
                                  f"{files_str}")

                return []
            elif len(matching_files) == 1:
                return matching_files[0]
            elif find_all:
                return matching_files
        elif len(files) == 1:  # no filename given and only one file is in path, return it
            return files[0]

        # More than one matching file or more than one file in path
        if sub_path in petitradtrans_config_parser['Default files']:  # check for a default file in configuration
            default_file = os.path.join(
                path_input_data,
                sub_path,
                petitradtrans_config_parser['Default files'][sub_path]
            )

            # Check if the default file exists
            if os.path.isfile(default_file):
                # Check if the requested file matches the default file
                if filename is not None:
                    if not default_file.startswith(filename):
                        # TODO make the example more robust by moving this function in opacities in 4.0.0
                        example_species: str = os.path.basename(matching_files[0]).rsplit('.', 3)[0]
                        separator: str = '\n\t- '

                        raise ValueError(
                            f"more than one file matching species '{filename}':\n"
                            f"\t- {list_str2str(list_str=matching_files, separator=separator)}\n"
                            f"Use a less ambiguous species name by adding isotopes, source, sampling "
                            f"and/or spectral range (e.g. '{example_species}')."
                        )

                return default_file
            elif not expect_default_file_exists:
                return os.path.split(default_file)[-1]
            else:
                raise FileNotFoundError(
                    f"no such file: '{default_file}'\n"
                    f"Update the 'Default file' entry for '{sub_path}' in petitRADTRANS' configuration by executing:\n"
                    f">>> from petitRADTRANS.config.configuration import petitradtrans_config_parser\n"
                    f">>> petitradtrans_config_parser.set_default_file(<new_default_file>)\n"
                    f"Or download the missing file."
                )
        else:  # make the user enter the default file
            new_default_file = select_default_file(
                files=tuple(files),
                full_path=full_path,
                sub_path=sub_path
            )

            petitradtrans_config_parser.set_default_file(
                file=os.path.join(full_path, new_default_file),
                path_input_data=path_input_data
            )

            return new_default_file


def select_default_file(files: tuple[str, ...], full_path: str, sub_path: str) -> str:
    files_str = [f" {i + 1}: {file}" for i, file in enumerate(files)]
    files_str = "\n".join(files_str)

    introduction_message = (
        f"More than one file detected in '{full_path}', and no default file set for this path "
        f"in petitRADTRANS' configuration\n"
        f"Please select one of the files in the list below by typing the corresponding integer:\n"
        f"{files_str}"
    )

    new_default_file: int | None = user_input(
        introduction_message=introduction_message,
        input_message=f"Select which file to set as the default file for '{sub_path}'",
        failure_message=f"failure to enter new default file for '{sub_path}'",
        cancel_message="Cancelling default file selection...",
        mode='list',
        list_length=len(files)
    )

    if new_default_file is None:
        raise ValueError(f"no default file selected for path '{sub_path}'")

    new_default_file -= 1
    new_default_file: str = files[new_default_file]

    return new_default_file


def split_input_data_path(path: str, path_input_data: str):
    if path_input_data not in path:
        raise ValueError(f"path '{path}' does not contains the input data path ('{path_input_data}')")

    sub_path = path.split(path_input_data + os.path.sep, 1)[-1]
    file = os.path.basename(sub_path)
    sub_path = os.path.dirname(sub_path)

    return path_input_data, sub_path, file
