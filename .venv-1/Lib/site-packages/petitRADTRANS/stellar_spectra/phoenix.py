"""Manages the Phoenix star table.
"""
import os

import h5py
import numpy as np

import petitRADTRANS.physical_constants as cst
from petitRADTRANS._input_data import find_input_file
from petitRADTRANS.config.configuration import get_input_data_subpaths, petitradtrans_config_parser
from petitRADTRANS.physics import wavelength2frequency


class PhoenixStarTable:
    """Used to store petitRADTRANS's PHOENIX star spectrum models.

    The compute_spectrum function can be used to get a star spectrum at a given temperature.
    """
    def __init__(self):
        self._loaded = False

        self.log10_effective_temperature_grid = None
        self.radius_grid = None
        self.flux_grid = None
        self.wavelength_grid = None

    def compute_spectrum(self, temperature):
        """
        Returns a matrix where the first column is the wavelength in cm
        and the second is the stellar flux :math:`F_\\nu` in units of
        :math:`\\rm erg/cm^2/s/Hz`, at the surface of the star.
        The spectra are PHOENIX models from (Husser et al. 2013), the spectral
        grid used here was described in van Boekel et al. (2012).

        Args:
            temperature (float):
                stellar effective temperature in K.
        """
        if not self._loaded:
            self.load()

        log_temp = np.log10(temperature)
        interpolation_index = np.searchsorted(self.log10_effective_temperature_grid, log_temp)

        if interpolation_index == 0:
            spec_dat = self.flux_grid[0]
            radius = self.radius_grid[0]
            print('Warning, input temperature is lower than minimum grid temperature.')
            print('Taking F = F_grid(minimum grid temperature), normalized to desired')
            print('input temperature.')

        elif interpolation_index == len(self.log10_effective_temperature_grid):
            spec_dat = self.flux_grid[int(len(self.log10_effective_temperature_grid) - 1)]
            radius = self.radius_grid[int(len(self.log10_effective_temperature_grid) - 1)]
            print('Warning, input temperature is higher than maximum grid temperature.')
            print('Taking F = F_grid(maximum grid temperature), normalized to desired')
            print('input temperature.')

        else:
            weight_high = (
                    (log_temp - self.log10_effective_temperature_grid[interpolation_index - 1])
                    / (self.log10_effective_temperature_grid[interpolation_index]
                       - self.log10_effective_temperature_grid[interpolation_index - 1])
            )

            weight_low = 1. - weight_high

            spec_dat_low = self.flux_grid[int(interpolation_index - 1)]

            spec_dat_high = self.flux_grid[int(interpolation_index)]

            spec_dat = (
                weight_low * spec_dat_low
                + weight_high * spec_dat_high
            )

            radius = (
                weight_low * self.radius_grid[int(interpolation_index - 1)]
                + weight_high * self.radius_grid[int(interpolation_index)]
            )

        frequency_grid = wavelength2frequency(self.wavelength_grid)
        flux = spec_dat
        norm = -np.sum((flux[1:] + flux[:-1]) * np.diff(frequency_grid)) / 2.

        spec_dat = flux / norm * cst.sigma * temperature ** 4.

        spec_dat = np.transpose(np.stack((self.wavelength_grid, spec_dat)))

        return spec_dat, radius

    @staticmethod
    def get_default_file(path_input_data=None):
        if path_input_data is None:
            path_input_data = petitradtrans_config_parser.get_input_data_path()

        return os.path.join(
            path_input_data,
            get_input_data_subpaths()["stellar_spectra"],
            "phoenix",
            "phoenix.startable.petitRADTRANS.h5"
        )

    def load(self, file=None, path_input_data=None, search_online=True):
        if file is None:
            file = self.get_default_file(path_input_data)

        if path_input_data is None:
            path_input_data = petitradtrans_config_parser.get_input_data_path()

        if not os.path.isfile(file):
            file = find_input_file(
                file=file,
                path_input_data=path_input_data,
                sub_path=None,
                find_all=False,
                search_online=search_online
            )

        print(f"Loading PHOENIX star table in file '{file}'... ", end='')

        with h5py.File(file, "r") as f:
            self.log10_effective_temperature_grid = f['log10_effective_temperatures'][()]
            self.radius_grid = f['radii'][()]
            self.flux_grid = f['fluxes'][()]
            self.wavelength_grid = f['wavelengths'][()]

        self._loaded = True

        print("Done.")


phoenix_star_table = PhoenixStarTable()
