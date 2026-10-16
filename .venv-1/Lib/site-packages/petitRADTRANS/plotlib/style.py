"""This module creates a default plotting style for all pRT plots

All of these can be changed when calling most plotting functions.
This will affect the matplotlib rcParams, which can be reset to
the default values after pRT is finished.
"""
import os

import matplotlib
import matplotlib.pyplot as plt

__tiny_figure_font_size = 40  # 0.5 text width 16/9
__small_figure_font_size = 22  # 0.25 text width
__medium_figure_font_size = 16  # 0.5 text width
__large_figure_font_size = 22  # 1.0 text width

wavenumber_units = r'cm$^{-1}$'
wavelength_units = r'cm'
flux_units = r'erg$\cdot$s$^{-1}\cdot$cm${-2}$/cm$^{-1}$'


default_color = 'magenta'


__species_color = {
    'CH4': 'C7',
    'CO': 'C3',
    'CO2': 'C5',
    'FeH': 'C4',
    'H2O': 'C0',
    'H2S': 'olive',
    'HCN': 'darkblue',
    'K': 'C8',
    'Na': 'gold',
    'NH3': 'C9',
    'PH3': 'C1',
    'TiO': 'C2',
    'VO': 'darkgreen',
}

__other_gases_color = {
    'Al': 'C7',
    'Ar': 'violet',
    'AsH3': 'm',
    'Ca': 'peru',
    'Co': 'aliceblue',
    'Cr': 'skyblue',
    'Cu': 'tan',
    'Fe': 'C4',
    'GeH4': 'olivedrab',
    'H': 'dimgray',
    'H2': 'k',
    'HCl': 'palegreen',
    'HF': 'y',
    'He': 'r',
    'KCl': 'darkolivegreen',
    'Kr': 'lightgrey',
    'Li': 'c',
    'Mg': 'darkorange',
    'Mn': 'olive',
    'N2': 'b',
    'NaCl': 'yellowgreen',
    'Ne': 'brown',
    'Ni': 'lightcoral',
    'P': 'wheat',
    'P2': 'navajowhite',
    'PH2': 'papayawhip',
    'PO': 'sandybrown',
    'SiH4': 'plum',
    'SiO': 'darkred',
    'Ti': 'lime',
    'TiO2': 'mediumseagreen',
    'V': 'forestgreen',
    'VO2': 'seagreen',
    'Xe': 'dodgerblue',
    'Zn': 'salmon'
}

__cloud_color = {
    'NH4SH': 'C1',
    'NH4Cl': 'C6',
    'H3PO4': 'wheat',
    'ZnS': 'C3',
    'KCl': 'C8',
    'Na2S': 'gold',
    'MnS': 'olive',
    'Cr2O3': 'deepskyblue',
    'MgSiO3': 'darkorange',
    'Mg2SiO4': 'C5',
    'SiO2': 'darkred',
    'TiN': 'lime',
    'CaTiO3': 'peru',
    'Al2O3': 'C7',
}


def get_species_color(species, implemented_only=False):
    if species in __species_color:
        return __species_color[species]
    elif species in __cloud_color:
        return __cloud_color[species]
    elif species in __other_gases_color:
        return __other_gases_color[species]
    elif implemented_only:
        available_species = []

        for k in __species_color:
            available_species.append(k)

        for k in __cloud_color:
            available_species.append(k)

        for k in __other_gases_color:
            available_species.append(k)

        available_species = [f"'{species}'" for species in available_species]
        available_species.sort()
        available_species = ', '.join(available_species)

        raise ValueError(f"'{species}' has no pre-defined color; "
                         f"available species are {available_species}")
    else:
        return default_color


def set_petitradtrans_plot_style():
    have_display = bool(os.environ.get('DISPLAY', None))

    if not have_display:
        matplotlib.use('Agg')

    prt_colours = ['#009FB8', '#FF695C', '#70FF92', '#FFBB33', '#6171FF', "#FF1F69", "#52AC25", '#E574FF', "#FF261D",
                   "#B429FF"]
    font = {'family': 'serif',
            'size': 24}
    lines = {'markeredgecolor': 'k',
             'markersize': 8}

    xtick = {'top': True,
             'bottom': True,
             'direction': 'in',
             'labelsize': 20}

    ytick = {'left': True,
             'right': True,
             'direction': 'in',
             'labelsize': 20}
    xmin = {'size': 5,
            'visible': True}
    ymin = {'size': 5,
            'visible': True}
    xmaj = {'size': 10}
    ymaj = {'size': 10}
    axes = {'labelsize': 26,
            'prop_cycle': matplotlib.cycler(  # type: ignore
                color=prt_colours
            ),
            'titlesize': 32,
            'linewidth': 3}
    figure = {'titlesize': 32,
              'figsize': (16, 10),
              'autolayout': True}
    legend = {'fancybox': True,
              'fontsize': 24}
    scatter = {'marker': 'o',
               'edgecolors': 'k'}

    print("Using pRT Plotting style!")
    plt.rc('font', **font)
    plt.rc('lines', **lines)
    plt.rc('xtick', **xtick)
    plt.rc('xtick.minor', **xmin)
    plt.rc('xtick.major', **xmaj)
    plt.rc('ytick', **ytick)
    plt.rc('ytick.minor', **ymin)
    plt.rc('ytick.major', **ymaj)
    plt.rc('axes', **axes)
    plt.rc('figure', **figure)
    plt.rc('legend', **legend)
    plt.rc('scatter', **scatter)


def update_figure_font_size(font_size):
    """
    Update the figure font size in a nice way.
    :param font_size: new font size
    """
    if font_size == 'tiny':
        font_size = __tiny_figure_font_size
    elif font_size == 'small':
        font_size = __small_figure_font_size
    elif font_size == 'medium':
        font_size = __medium_figure_font_size
    elif font_size == 'large':
        font_size = __large_figure_font_size

    plt.rc('font', size=font_size)  # controls default text sizes
    plt.rc('axes', titlesize=font_size)  # fontsize of the axes title
    plt.rc('axes', labelsize=font_size)  # fontsize of the x and y labels
    plt.rc('axes.formatter', use_mathtext=True)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_size)  # fontsize of the tick labels
    plt.rc('xtick', direction='in')  # fontsize of the tick labels
    plt.rc('xtick.major', width=font_size / 10 * 0.8, size=font_size / 10 * 3.5)  # fontsize of the tick labels
    plt.rc('xtick.minor', width=font_size / 10 * 0.6, size=font_size / 10 * 2)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_size)  # fontsize of the tick labels
    plt.rc('ytick', direction='in')  # fontsize of the tick labels
    plt.rc('ytick.major', width=font_size / 10 * 0.8, size=font_size / 10 * 3.5)  # fontsize of the tick labels
    plt.rc('ytick.minor', width=font_size / 10 * 0.6, size=font_size / 10 * 2)  # fontsize of the tick labels
    plt.rc('legend', fontsize=font_size)  # legend fontsize
    plt.rc('figure', titlesize=font_size)  # fontsize of the figure title
