"""
Manages the configuration files.
"""
import configparser
import os
import warnings
from pathlib import Path

from petitRADTRANS.utils import LockedDict


class PetitradtransConfigParser(configparser.ConfigParser):
    """Used to manage petitRADTRANS's configuration."""
    _instance = None

    _directory = os.path.join(str(Path.home()), '.petitradtrans')
    _config_file = os.path.join(_directory, 'petitradtrans_config_file.ini')
    _default_config = {
        'Default files': {

        },
        'Paths': {
            'pRT_input_data_path': os.path.join(str(Path.home()), 'petitRADTRANS', 'input_data'),
            'pRT_outputs_path': os.path.join(str(Path.home()), 'petitRADTRANS', 'outputs')
        },
        'URLs': {
            'pRT_input_data_url': 'https://keeper.mpdl.mpg.de/d/ccf25082fda448c8a0d0/?p='
        }
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def _make(cls):
        """
        Get the petitRADTRANS configuration directory, where the configuration file is stored.

        Returns:
            The configuration directory.
        """
        print(f"Generating configuration file '{cls._config_file}'...")

        config = cls.init_default()

        if not os.path.isdir(cls._directory):
            print(f"Creating directory '{cls._directory}'...")
            os.makedirs(cls._directory)

        config.save()

    @property
    def config_file(self):
        """
        Get the full path to the petitRADTRANS configuration file.

        Returns:
            The configuration filename.
        """
        return self._config_file

    @property
    def default_config(self):
        return self._default_config

    @property
    def directory(self):
        return self._directory

    def get_input_data_path(self) -> str:
        return self['Paths']['prt_input_data_path']

    def get_input_data_subpath(self, path, path_input_data=None, strict=True):
        if path_input_data is None:
            path_input_data = self.get_input_data_path()

        if self.get_input_data_path() not in path and strict:
            warnings.warn(f"path '{path}' is not within "
                          f"the configured input_data path ('{self.get_input_data_path()}')\n"
                          f"Check your path, "
                          f"or, if you are targeting an alternate input_data path, "
                          f"set strict to False to supress this warning")

        path = path.rsplit(path_input_data)[1][1:]  # remove leading '/'

        if len(path) == 0:
            raise ValueError(f"path must be within the input_data path, but is the input_data path itself ('{path}')")

        is_subpath = False

        for subpath in get_input_data_subpaths().values():
            if subpath in path:
                is_subpath = True
                break

        if not is_subpath:
            subpaths_str = "\n".join(get_input_data_subpaths().values())
            raise ValueError(f"path '{path}' must be within an input_data subpath\n"
                             f"Valid subpaths are:\n"
                             f"{subpaths_str}")

        return path

    @classmethod
    def init_default(cls):
        config = cls()
        config.update(cls._default_config)

        return config

    def load(self):
        """
        Load the petitRADTRANS configuration file.
        """
        self.read(PetitradtransConfigParser._config_file)

    def power_load(self):
        """
        Load the petitRADTRANS configuration file. Generate it if necessary.
        """
        if not os.path.isfile(PetitradtransConfigParser._config_file):
            PetitradtransConfigParser._make()  # TODO find a better, safer way to do generate the configuration file?

        self.read(PetitradtransConfigParser._config_file)
        self.repair()

    def power_set(self, section, option, value):
        self.set(section, option, value)
        self.repair()
        self.save()

    def power_update(self, new_config, repair_update=True):
        self.update(new_config)

        if repair_update:
            self.repair()

        self.save()

        print("Configuration updated")

    def save(self):
        with open(self._config_file, 'w') as configfile:
            self.write(configfile)

    def set_input_data_path(self, path: str):
        """
        Update the configuration file of petitRADTRANS.

        Args:
            path: path to the petitRADTRANS input data file
        """
        path_tail = path.rsplit(os.path.sep, 1)[1]

        if path_tail != 'input_data':
            path = os.path.join(path, 'input_data')

        # Ensure that we are in the home directory when '~' is used
        if not os.path.isdir(os.path.abspath(path)) and '~' + os.path.sep in path:
            path = os.path.abspath(path).rsplit('~' + os.path.sep, 1)[1]
            path = os.path.join(os.path.abspath(str(Path.home())), path)

        self.power_set('Paths', 'pRT_input_data_path', os.path.abspath(path))

        print(f"Input data path changed to '{os.path.abspath(path)}'")

        self.load()

    def set_default_file(self, file: str, path_input_data=None):
        if path_input_data is None:
            path_input_data = self.get_input_data_path()

        file = os.path.abspath(file)
        path, filename = file.rsplit(os.path.sep, 1)

        if path_input_data != self.get_input_data_path():
            print(f"Setting default side for '{path_input_data}', "
                  f"outside of the configured input_data path ('{self.get_input_data_path()}')")

        sub_path = self.get_input_data_subpath(
            path=path,
            path_input_data=path_input_data,
            strict=False
        )

        print(f"Setting new default file '{file}'")
        self.power_set('Default files', sub_path, filename)

        self.load()

    def repair(self):
        repaired = False

        for section in self._default_config:
            if not self.has_section(section):
                print(f"Adding missing section '{section}'...")
                repaired = True
                self.add_section(section)

                for option, value in self._default_config[section].items():
                    self.set(section, option, value)
            else:
                for option, value in self._default_config[section].items():
                    if option not in self[section]:
                        print(f"Adding missing option '{option}' in section '{section}'...")
                        repaired = True
                        self.set(section, option, value)

        if repaired:
            print(f"Repairing configuration file ('{self._config_file}')...")
            self.save()


def get_input_data_subpaths() -> LockedDict:
    return LockedDict.build_and_lock({
        "cia_opacities": os.path.join("opacities", "continuum", "collision_induced_absorptions"),
        "clouds_opacities": os.path.join("opacities", "continuum", "clouds"),
        "correlated_k_opacities": os.path.join("opacities", "lines", "correlated_k"),
        "line_by_line_opacities": os.path.join("opacities", "lines", "line_by_line"),
        "planet_data": "planet_data",
        "pre_calculated_chemistry": "pre_calculated_chemistry",
        "stellar_spectra": "stellar_spectra",
        "unknown_opacities": os.path.join("opacities", "unknown"),
    })


petitradtrans_config_parser = PetitradtransConfigParser()
petitradtrans_config_parser.power_load()
