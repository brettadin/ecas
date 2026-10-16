"""Stores useful plot function.
"""
import copy
import os

import corner
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.lines import Line2D
from typing import Any, Callable
from scipy.stats import binned_statistic

import petitRADTRANS.physical_constants as cst
from petitRADTRANS.chemistry.clouds import (
    return_t_cond_fe,
    return_t_cond_fe_comb,
    return_t_cond_fe_l,
    return_t_cond_kcl,
    return_t_cond_mgsio3,
    return_t_cond_na2s,
    simple_cdf_fe,
    simple_cdf_kcl,
    simple_cdf_mgsio3,
    simple_cdf_na2s
)
from petitRADTRANS.chemistry.pre_calculated_chemistry import pre_calculated_equilibrium_chemistry_table
from petitRADTRANS.opacities.opacities import Opacity
from petitRADTRANS.physics import frequency2wavelength
from petitRADTRANS.planet import Planet
from petitRADTRANS.plotlib.style import default_color, get_species_color, update_figure_font_size
from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS.retrieval.utils import get_pymultinest_sample_dict
from petitRADTRANS.spectral_model import SpectralModel
from petitRADTRANS.utils import LockedDict


def _corner_wrap(
        data_list,
        title_kwargs,
        labels_list,
        label_kwargs,
        range_list,
        color,
        truths_list,
        contour_kwargs,
        hist_kwargs,
        bins=20,
        axes_scale="linear",
        weights=None,
        hist_bin_factor: npt.NDArray[np.floating] = 1.0,
        smooth=None,
        smooth1d=None,
        titles=None,
        show_titles=True,
        title_quantiles=None,
        title_fmt=None,
        truth_color='r',
        scale_hist=False,
        quantiles=None,
        fig=None,
        max_n_ticks=5,
        top_ticks=False,
        use_math_text=False,
        reverse=False,
        labelpad=0.0,
        # 2-D histogram args
        plot_contours=True,
        hist2d_levels=None,
        **hist2d_kwargs
):
    if quantiles is None:
        quantiles = (0.16, 0.5, 0.84)  # using default title_quantiles, the 1-sigma quantile is actually erf(1)

    if hist2d_levels is None:
        hist2d_levels = (1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-4.5))  # 1, 2 and 3 sigma
    else:
        hist2d_levels = [1 - np.exp(-sigma ** 2 / 2) for sigma in hist2d_levels]
        hist2d_levels = [sigma for sigma in hist2d_levels]

    fig = corner.corner(
        data=np.array(data_list).T,
        bins=bins,
        range=range_list,
        axes_scale=axes_scale,
        weights=weights,
        color=color,
        hist_bin_factor=hist_bin_factor,
        smooth=smooth,
        smooth1d=smooth1d,
        labels=labels_list,
        label_kwargs=label_kwargs,
        titles=titles,
        show_titles=show_titles,
        title_quantiles=title_quantiles,
        title_fmt=title_fmt,
        title_kwargs=title_kwargs,
        truths=truths_list,
        truth_color=truth_color,
        scale_hist=scale_hist,
        quantiles=quantiles,
        fig=fig,
        hist_kwargs=hist_kwargs,
        max_n_ticks=max_n_ticks,
        top_ticks=top_ticks,
        use_math_text=use_math_text,
        reverse=reverse,
        labelpad=labelpad,
        **hist2d_kwargs,
        plot_contours=plot_contours,
        contour_kwargs=contour_kwargs,
        levels=hist2d_levels
    )

    return fig


def _get_parameter_range(sample_dict, retrieved_parameters, spectral_model=None):
    parameter_ranges = []
    parameter_names = []
    parameter_titles = []
    parameter_labels = []
    coefficients = []
    offsets = []

    ordered_id = []
    sd_keys = list(sample_dict.keys())
    sd_non_parameter_keys = []

    if 'log_likelihood' in sample_dict:
        sd_non_parameter_keys.append('log_likelihood')

    if 'stats' in sample_dict:
        sd_non_parameter_keys.append('stats')

    for key in sample_dict:
        if key not in retrieved_parameters:
            if key not in sd_non_parameter_keys:
                raise KeyError(f"Key '{key}' not in retrieved parameters")
            else:
                continue

    for i, key in enumerate(retrieved_parameters):
        if key not in sample_dict:
            continue

        ordered_id.append(sd_keys.index(key))
        dictionary = retrieved_parameters[key]

        prior_parameters = np.array(dictionary['prior_parameters'])

        # pRT corner range
        mean = np.mean(sample_dict[key])
        std = np.std(sample_dict[key])
        low_ref = mean - 4 * std
        high_ref = mean + 4 * std

        if 'figure_coefficient' in dictionary:
            if spectral_model is not None:
                if key == 'radial_velocity_semi_amplitude':
                    figure_coefficient = spectral_model.model_parameters['radial_velocity_semi_amplitude']
                elif key == 'planet_radius':
                    figure_coefficient = spectral_model.model_parameters['planet_radius']
                else:
                    figure_coefficient = 1
            else:
                figure_coefficient = 1

            figure_coefficient *= dictionary['figure_coefficient']

            coefficients.append(dictionary['figure_coefficient'])
            low_ref *= dictionary['figure_coefficient']
            high_ref *= dictionary['figure_coefficient']
        else:
            figure_coefficient = 1
            coefficients.append(1)

        if 'figure_offset' in dictionary:
            figure_offset = dictionary['figure_offset']

            offsets.append(dictionary['figure_offset'])
            low_ref += dictionary['figure_offset']
            high_ref += dictionary['figure_offset']
        else:
            figure_offset = 0
            offsets.append(0)

        low, high = prior_parameters * figure_coefficient + figure_offset
        low = np.max((low_ref, low))
        high = np.min((high_ref, high))

        parameter_ranges.append([low, high])
        parameter_names.append(key)

        if 'figure_label' in dictionary:
            parameter_labels.append(dictionary['figure_label'])
        else:
            parameter_labels.append(key)

        if 'figure_title' in dictionary:
            parameter_titles.append(dictionary['figure_title'])
        else:
            parameter_titles.append(None)

    sd_tmp = {}

    for i in ordered_id:
        sd_tmp[sd_keys[i]] = copy.deepcopy(sample_dict[sd_keys[i]])

    for k in sd_non_parameter_keys:
        sd_tmp[k] = copy.deepcopy(sample_dict[k])

    sample_dict = copy.deepcopy(sd_tmp)

    return parameter_ranges, parameter_names, parameter_labels, parameter_titles, \
        np.array(coefficients), np.array(offsets), sample_dict


def _prepare_multiple_retrievals_plot(result_directory, retrieved_parameters, true_values,
                                      retrieval_name=None, spectral_model=None):
    # Initialisation
    parameter_names_dict = {}
    parameter_plot_indices_dict = {}
    parameter_ranges_dict = {}
    true_values_dict = {}
    fig_titles_dict = {}
    fig_labels_dict = {}
    parameter_names_ref = None

    # Get all samples
    if np.ndim(result_directory) > 0:
        sd = []
        sample_dict: dict[str, Any] = {}

        for i, directory in enumerate(result_directory):
            sd.append(get_pymultinest_sample_dict(directory, name=retrieval_name))
            sample_dict[f'{i}'] = None
    else:
        sd = [get_pymultinest_sample_dict(result_directory, name=retrieval_name)]
        sample_dict = {'': None}

    samples = list(sample_dict.keys())
    coefficients = 1
    offsets = 0

    # Get all useful figure information from all retrievals
    for i, sample in enumerate(samples):
        parameter_ranges, parameter_names, fig_labels, fig_titles, coefficients, offsets, sd[i] = \
            _get_parameter_range(sd[i], retrieved_parameters, spectral_model)

        sample_dict[sample] = np.array(list(sd[i].values())).T

        sample_dict[sample] = sample_dict[sample] * coefficients + offsets
        parameter_names_dict[sample] = parameter_names
        fig_titles_dict[sample] = fig_titles
        fig_labels_dict[sample] = fig_labels

        # Ensure that all parameter ranges share the widest ranges among all retrievals
        if i == 0:
            parameter_plot_indices_dict[sample] = np.arange(0, len(parameter_ranges))
            parameter_names_ref = np.array(parameter_names_dict[sample])
            parameter_ranges_dict[sample] = parameter_ranges
        else:
            parameter_ranges_dict[sample] = copy.deepcopy(parameter_ranges_dict[samples[0]])

            for j, range_ in enumerate(parameter_ranges):
                if parameter_names_dict[sample][j] not in parameter_names_ref:
                    raise KeyError(f"parameter '{parameter_names_dict[sample][j]}' was not in first sample")

                # Store the widest parameter ranges in the first sample, which contains all possible parameters
                for k, parameter_name in enumerate(parameter_names_dict[samples[0]]):
                    if parameter_name == parameter_names_dict[sample][j]:
                        if range_[0] < parameter_ranges_dict[samples[0]][k][0]:
                            parameter_ranges_dict[samples[0]][k][0] = range_[0]

                        if range_[1] > parameter_ranges_dict[samples[0]][k][1]:
                            parameter_ranges_dict[samples[0]][k][1] = range_[1]

                        break

            # Ensure that all parameters with the same name across retrievals have the same plot index
            parameter_plot_indices_dict[sample] = np.zeros(np.size(parameter_names_dict[sample]), dtype=int)

            for j, parameter_name in enumerate(parameter_names_dict[sample]):
                if parameter_name not in parameter_names_ref:
                    raise KeyError(f"key '{parameter_name}' "
                                   f"of sample '{sample}' not in sample '{parameter_names_ref}'")

                parameter_plot_indices_dict[sample][j] = (
                    np.array(parameter_names_ref == parameter_name).nonzero()[0][0]
                )

    # Broadcast the widest parameter ranges stored in the first sample to all the other samples
    for sample in samples[1:]:
        parameter_ranges_dict[sample] = parameter_ranges_dict[samples[0]]

        parameter_ranges_dict_tmp = copy.deepcopy(parameter_ranges_dict[sample])

        for j, plot_index in enumerate(parameter_plot_indices_dict[sample]):
            parameter_ranges_dict_tmp[plot_index] = parameter_ranges_dict[sample][j]

        parameter_ranges_dict[sample] = copy.deepcopy(parameter_ranges_dict_tmp)

    # Add true values
    if true_values is not None:
        if isinstance(true_values, dict):
            if list(true_values.keys())[0] != '':
                true_values = np.array([true_values[key] for key in sd[0]])

        true_values = true_values * coefficients + offsets

        for sample in sample_dict:
            true_values_dict[sample] = true_values
    else:
        true_values_dict = None

    return sample_dict, parameter_names_dict, parameter_plot_indices_dict, parameter_ranges_dict, true_values_dict, \
        fig_titles_dict, fig_labels_dict


def contour_corner(
        sampledict: dict[str, Any],
        parameter_names: dict[str, list[str]],
        output_file: str = None,
        parameter_ranges: dict[str, npt.NDArray[np.floating]] = None,
        parameter_plot_indices: dict[str, npt.NDArray[np.integer]] = None,
        true_values: dict[str, npt.NDArray[np.floating]] = None,
        short_name: dict[str, str] = None,
        quantiles: list[float] = None,
        hist2d_levels: list[float] = None,
        legend: bool = False,
        prt_plot_style: bool = True,
        plot_best_fit: bool = False,
        color_list: list[str] = None,
        bins: int = 20,
        axes_scale: str = "linear",
        weights: npt.NDArray[np.floating] = None,
        hist_bin_factor: npt.NDArray[np.floating] = 1.0,
        smooth: float = None,
        smooth1d: float = None,
        titles: list = None,
        show_titles: bool = True,
        title_quantiles: list[float] = None,
        title_fmt: str = ".2f",
        truth_color: str = 'r',
        scale_hist: bool = False,
        fig: matplotlib.axes.Axes = None,
        max_n_ticks: int = 5,
        top_ticks: bool = False,
        use_math_text: bool = False,
        reverse: bool = False,
        labelpad: float = 0.0,
        plot_contours: bool = True,
        label_kwargs: dict = None,
        title_kwargs: dict = None,
        hist_kwargs: dict = None,
        hist2d_kwargs: dict = None,
        contour_kwargs: dict = None,
        **_
):
    """
    Use the corner package to plot the posterior distributions produced by pymultinest.
    # TODO @Evert complete docstring
    Args:
        sampledict : dict
            A dictionary of samples, each sample has shape (N_Samples,N_params). The keys of the
            dictionary correspond to the names of each retrieval, and are the prefixes to the
            post_equal_weights.dat files. These are passed as arguments to retrieve.py.
            By default, this is only the current retrieval, and plots the posteriors for a single
            retrieval. If multiple names are passed, they are overplotted on the same figure.
        parameter_names : dict
            A dictionary with keys for each retrieval name, as in sampledict. Each value of the
            dictionary is the names of the parameters to be plotted, as set in the
            run_definition file.
        output_file : str
            Output file name
        parameter_ranges : dict
            A dictionary with keys for each retrieval name as in sampledict. Each value
            contains the ranges of parameters that have a range set with corner_range in the
            parameter class. Otherwise, the range is +/- 4 sigma
        parameter_plot_indices : dict
            A dictionary with keys for each retrieval name as in sampledict. Each value
            contains the indices of the sample to plotlib, as set by the plot_in_corner
            parameter of the parameter class
        true_values : dict
            A dictionary with keys for each retrieval name as in sampledict. Each value
            contains the known values of the parameters.
        short_name : dict
            A dictionary with keys for each retrieval name as in sampledict. Each value
            contains the names to be plotted in the corner plotlib legend. If non, uses the
            retrieval names used as keys for sampledict
        quantiles : list
            A list with the quantiles to plotlib over the 1D histograms.
            Note: the conversion from sigma to quantile is:
                quantile_m = (1 - erf(sigma / sqrt(2))) / 2
                quantile_p = 1 - quantile_m
        hist2d_levels :  list
            A list with the sigmas-level to plotlib over the 2D histograms. The sigmas are converted into their
            corresponding level-value following the formula for 2D normal distribution
            (see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4358977/pdf/pone.0118537.pdf).
        legend : bool
            Turn the legend on or off
        prt_plot_style : bool
            Use the prt plotlib style, changes the colour scheme and fonts to match the rest of
            the prt plots.
        plot_best_fit : bool
            Plot the maximum likelihood values as vertical lines for each parameter.
        color_list : list
            List of colours for plotting multiple retrievals.
        bins : int or array_like[ndim,]
            The number of bins to use in histograms, either as a fixed value for all dimensions, or as a list of
            integers for each dimension.
        axes_scale : str or iterable (ndim,)
            Scale (``"linear"``, ``"log"``) to use for each data dimension. If only one scale is specified, use that
            for all dimensions.
        weights : array_like[nsamples,]
            The weight of each sample. If `None` (default), samples are given equal weight.
        hist_bin_factor : float or array_like[ndim,]
            This is a factor (or list of factors, one for each dimension) that will multiply the bin specifications
            when making the 1-D histograms.
            This is generally used to increase the number of bins in the 1-D plots to provide more resolution.
        smooth : float
           The standard deviation for Gaussian kernel passed to scipy.ndimage.gaussian_filter` to smooth the 2-D
           histogram. If `None` (default), no smoothing is applied.
        smooth1d : float
           The standard deviation for Gaussian kernel passed to `scipy.ndimage.gaussian_filter` to smooth the 1-D
           histogram. If `None` (default), no smoothing is applied.
        titles : iterable (ndim,)
            A list of titles for the dimensions. If `None` (default), uses labels as titles.
        show_titles : bool
            Displays a title above each 1-D histogram showing the 0.5 quantile with the upper and lower errors supplied
            by the quantiles argument.
        title_quantiles : iterable
            A list of 3 fractional quantiles to show as the upper and lower errors. If `None` (default), inherit
            the values from quantiles, unless quantiles is `None`, in which case it defaults to [0.16, 0.5, 0.84].
        title_fmt : string
            The format string for the quantiles given in titles. If you explicitly set ``show_titles=True`` and
            ``title_fmt=None``, the labels will be shown as the titles. (default: ``.2f``).
        truth_color : str
            A ``matplotlib`` style colour for the ``truths`` makers.
        scale_hist : bool
            Should the 1-D histograms be scaled in such a way that the zero line is visible?
        fig : `~matplotlib.figure.Figure`
            Overplot onto the provided figure object, which must either have no axes yet, or ``ndim * ndim`` axes
            already present.  If not set, the plot will be drawn on a newly created figure.
        max_n_ticks: int
            Maximum number of ticks to try to use.
        top_ticks : bool
            If true, label the top ticks of each axis.
        use_math_text : bool
            If true, then axis tick labels for very large or small exponents will be displayed as powers of 10 rather
            than using `e`.
        reverse : bool
            If true, plot the corner plot starting in the upper-right corner instead of the usual bottom-left corner.
        labelpad : float
            Padding between the axis and the x- and y-labels in units of the fraction of the axis from the lower left.
        plot_contours : bool
            Draw contours for dense regions of the plot.
        kwargs : dict
            Each kwarg can be one of the kwargs used in corner.corner. These can be used to adjust
            the title_kwargs,label_kwargs,hist_kwargs, hist2d_kwargs or the contour kwargs. Each
            kwarg must be a dictionary with the arguments as keys and values as the values.
    """
    if parameter_ranges is None:
        parameter_ranges = {}

    if parameter_plot_indices is None:
        parameter_plot_indices = {}

    if hist2d_kwargs is None:
        hist2d_kwargs = {}

    if prt_plot_style:
        import matplotlib as mpl

        mpl.rcParams.update(mpl.rcParamsDefault)
        font = {'family': 'serif'}
        xtick = {'top': True,
                 'bottom': True,
                 'direction': 'in'}

        ytick = {'left': True,
                 'right': True,
                 'direction': 'in'}
        xmin = {'visible': True}
        ymin = {'visible': True}
        mpl.rc('xtick', **xtick)
        mpl.rc('xtick.minor', **xmin)
        mpl.rc('ytick', **ytick)
        mpl.rc('ytick.minor', **ymin)
        mpl.rc('font', **font)

        if color_list is None:
            color_list = [
                '#009FB8', '#FF695C', '#70FF92', '#FFBB33', '#6171FF', "#FF1F69", "#52AC25", '#E574FF', "#FF261D",
                "#B429FF"
            ]
    elif color_list is None:
        color_list = [f'C{i}' for i in range(8)]  # standard matplotlib colour cycle

    handles = []
    range_list = []
    count = 0

    for key, samples in sampledict.items():
        if prt_plot_style and count > len(color_list):
            print("Not enough colors to continue plotting. Please add to the list.")
            print("Outputting first " + str(count) + " retrievals.")
            break

        n_samples = len(samples)
        s = n_samples

        if key not in parameter_plot_indices:  # TODO @Evert use linspace/arange instead of range()
            parameter_plot_indices[key] = np.array(list(range(len(parameter_names[key]))))
        elif parameter_plot_indices[key] is None:  # same as in the case the key doesn't exist
            parameter_plot_indices[key] = np.array(list(range(len(parameter_names[key]))))

        if key not in parameter_ranges:
            parameter_ranges[key] = [None] * (max(parameter_plot_indices[key]) + 1)

        data_list = []
        labels_list = []

        if plot_best_fit:
            best_fit = []
            best_fit_ind = np.argmax(samples[:, -1])

            for i in parameter_plot_indices[key]:
                best_fit.append(samples[best_fit_ind][i])

        for range_i, i in enumerate(parameter_plot_indices[key]):
            data_list.append(samples[len(samples) - s:, i])
            labels_list.append(parameter_names[key][i])

            if parameter_ranges[key][i] is None:
                range_mean = np.mean(samples[len(samples) - s:, i])
                range_std = np.std(samples[len(samples) - s:, i])
                low = range_mean - 4 * range_std
                high = range_mean + 4 * range_std

                if count > 0:
                    if low > range_list[range_i][0]:
                        low = range_list[range_i][0]
                    if high < range_list[range_i][1]:
                        high = range_list[range_i][1]
                    range_take = (low, high)
                    range_list[range_i] = range_take
                else:
                    range_list.append((low, high))
            else:
                range_take = (parameter_ranges[key][i][0], parameter_ranges[key][i][1])
                range_list.append(range_take)

        if parameter_plot_indices is not None and true_values is not None:
            truths_list = []

            if plot_best_fit:
                best_fit_ind = np.argmax(samples[:, -1])

                for i in parameter_plot_indices[key]:
                    truths_list.append(samples[best_fit_ind][i])
            else:
                for i in parameter_plot_indices[key]:
                    truths_list.append(true_values[key][i])
        else:
            truths_list = None

        if short_name is None:
            label = key
        else:
            label = short_name[key]

        handles.append(Line2D([0], [0], marker='o', color=color_list[count], label=label, markersize=15))

        if count == 0:
            fig = _corner_wrap(
                data_list=data_list,
                title_kwargs=title_kwargs,
                labels_list=labels_list,
                label_kwargs=label_kwargs,
                range_list=range_list,
                color=color_list[count],
                truths_list=truths_list,
                contour_kwargs=contour_kwargs,
                hist_kwargs=hist_kwargs,
                bins=bins,
                axes_scale=axes_scale,
                weights=weights,
                hist_bin_factor=hist_bin_factor,
                smooth=smooth,  # smoothing can be useful when using a few live points
                smooth1d=smooth1d,
                titles=titles,
                show_titles=show_titles,
                title_quantiles=title_quantiles,
                title_fmt=title_fmt,
                truth_color=truth_color,
                scale_hist=scale_hist,
                quantiles=quantiles,
                fig=fig,  # None by default, in that case a new figure is created
                max_n_ticks=max_n_ticks,
                top_ticks=top_ticks,
                use_math_text=use_math_text,
                reverse=reverse,
                labelpad=labelpad,
                plot_contours=plot_contours,
                hist2d_levels=hist2d_levels,
                **hist2d_kwargs,
            )
        else:
            _ = _corner_wrap(
                data_list=data_list,
                title_kwargs=title_kwargs,
                labels_list=labels_list,
                label_kwargs=label_kwargs,
                range_list=range_list,
                color=color_list[count],
                truths_list=truths_list,
                contour_kwargs=contour_kwargs,
                hist_kwargs=hist_kwargs,
                bins=bins,
                axes_scale=axes_scale,
                weights=weights,
                hist_bin_factor=hist_bin_factor,
                smooth=smooth,
                smooth1d=smooth1d,
                titles=titles,
                show_titles=False,  # only show titles (median +1sigma -1sigma) for the first sample
                title_quantiles=None,
                title_fmt=None,
                truth_color=truth_color,
                scale_hist=scale_hist,
                quantiles=quantiles,
                fig=fig,  # overplot on the existing figure
                max_n_ticks=max_n_ticks,
                top_ticks=top_ticks,
                use_math_text=use_math_text,
                reverse=reverse,
                labelpad=labelpad,
                plot_contours=plot_contours,
                hist2d_levels=hist2d_levels,
                **hist2d_kwargs,
            )
        count += 1

    if legend:
        fig.get_axes()[2].legend(handles=handles,
                                 loc='upper right')

    if output_file is not None:
        plt.savefig(output_file, bbox_inches='tight')

    return fig


def nice_corner(samples,
                parameter_names,
                output_file,
                n_samples=None,
                parameter_ranges=None,
                parameter_plot_indices=None,
                true_values=None,
                max_val_ratio=None):
    """
    Paul's custom hex grid corner plots.
    Won't work with sampledict setup in retrieve.py!
    """
    import seaborn as sns  # TODO is seaborn really that useful?

    font = {'family': 'serif',
            'weight': 'normal',
            'size': int(23 * 5. / len(parameter_plot_indices))}

    plt.rc('font', **font)
    plt.rc('text', usetex=True)

    if n_samples is not None:
        s = n_samples
    else:
        s = len(samples)

    try:
        if parameter_plot_indices is None:
            parameter_plot_indices = np.linspace(0, len(parameter_names) - 1,
                                                 len(parameter_names) - 1).astype('int')
    except Exception:  # TODO @Evert replace bare exception
        pass

    if max_val_ratio is None:
        max_val_ratio = 5.

    data_list = []
    labels_list = []
    range_list: list[tuple[int, int]] = []

    for i in parameter_plot_indices:

        data_list.append(samples[len(samples) - s:, i])
        labels_list.append(parameter_names[i])

        try:
            if parameter_ranges[i] is None:
                range_mean = np.mean(samples[len(samples) - s:, i])
                range_std = np.std(samples[len(samples) - s:, i])
                range_take = (range_mean - 4 * range_std, range_mean + 4 * range_std)
                range_list.append(range_take)
            else:
                range_list.append(parameter_ranges[i])
        except Exception:  # TODO @Evert replace bare exception
            range_mean = np.mean(samples[len(samples) - s:, i])
            range_std = np.std(samples[len(samples) - s:, i])
            range_take = (range_mean - 4 * range_std, range_mean + 4 * range_std)
            range_list.append(range_take)

    try:
        truths_list = []

        for i in parameter_plot_indices:
            truths_list.append(true_values[i])
    except Exception:  # TODO @Evert replace bare exception
        truths_list = None

    dimensions = len(parameter_plot_indices)

    # Set up the matplotlib figure
    f, axes = plt.subplots(dimensions, dimensions, figsize=(13, 13),
                           sharex='none', sharey='none')
    i_col = 0
    i_lin = 0

    try:
        ax_array = axes.flat
    except Exception:  # TODO @Evert replace bare exception
        ax_array = [plt.gca()]

    for ax in ax_array:

        # print(i_col, i_lin, dimensions, len(axes.flat))
        if i_col < i_lin:
            x = copy.copy(data_list[i_col])
            y = copy.copy(data_list[i_lin])
            indexx = (x >= range_list[i_col][0]) & \
                     (x <= range_list[i_col][1])
            indexy = (y >= range_list[i_lin][0]) & \
                     (y <= range_list[i_lin][1])

            x = x[indexx & indexy]
            y = y[indexx & indexy]

            gridsize = 22
            ax.hexbin(x, y, cmap='bone_r', gridsize=gridsize,
                      vmax=int(max_val_ratio * s / gridsize ** 2.),
                      rasterized=True)

            ax.set_xlim((range_list[i_col][0], range_list[i_col][1]))
            ax.set_ylim((range_list[i_lin][0], range_list[i_lin][1]))

            try:
                ax.axhline(truths_list[i_lin], color='red',
                           linestyle='--', linewidth=2.5)
                ax.axvline(truths_list[i_col], color='red',
                           linestyle='--', linewidth=2.5)
            except Exception:  # TODO @Evert replace bare exception
                pass

            if i_col > 0:
                ax.get_yaxis().set_visible(False)

        elif i_col == i_lin:

            med = np.median(data_list[i_col])
            up = str(np.round(np.percentile(data_list[i_col], 84) - med, 2))
            do = str(np.round(med - np.percentile(data_list[i_col], 16), 2))
            med = str(np.round(med, 2))
            med = med.split('.')[0] + '.' + med.split('.')[1].ljust(2, '0')
            up = up.split('.')[0] + '.' + up.split('.')[1].ljust(2, '0')
            do = do.split('.')[0] + '.' + do.split('.')[1].ljust(2, '0')

            ax.set_title(med + r'$^{+' + up + '}_{-' + do + '}$',
                         fontdict=font, fontsize=int(22 * 5. / len(parameter_plot_indices)))

            use_data = copy.copy(data_list[i_col])
            index = (use_data >= range_list[i_col][0]) & \
                    (use_data <= range_list[i_col][1])
            use_data = use_data[index]
            sns.distplot(use_data, bins=22, kde=False,
                         rug=False, ax=ax, color='gray')

            ax.set_xlim((range_list[i_col][0], range_list[i_col][1]))
            ax.get_yaxis().set_visible(False)
            try:
                ax.axvline(truths_list[i_col], color='red',
                           linestyle='--', linewidth=2.5)
            except Exception:  # TODO @Evert replace bare exception
                pass

            ax.axvline(float(med) + float(up), color='black',
                       linestyle=':', linewidth=1.0)
            ax.axvline(float(med), color='black',
                       linestyle=':', linewidth=1.0)
            ax.axvline(float(med) - float(do), color='black',
                       linestyle=':', linewidth=1.0)

        else:
            ax.axis('off')

        i_col += 1
        if i_col % dimensions == 0:
            i_col = 0
            i_lin += 1

    for i_col in range(dimensions):
        for i_lin in range(dimensions):
            try:
                plt.sca(axes[i_lin, i_col])
            except Exception:  # TODO @Evert replace bare exception
                pass
            range_use = np.linspace(range_list[i_col][0],
                                    range_list[i_col][1], 5)[1:-1]

            labels_use = []
            for r in range_use:
                labels_use.append(str(np.round(r, 1)))
            plt.xticks(range_use[::2], labels_use[::2])
            plt.xlabel(labels_list[i_col])

    for i_lin in range(dimensions):
        try:
            plt.sca(axes[i_lin, 0])
        except Exception:  # TODO @Evert replace bare exception
            pass
        plt.ylabel(labels_list[i_lin])

    plt.subplots_adjust(wspace=0, hspace=0)
    if dimensions == 1:
        plt.tight_layout(h_pad=0, w_pad=0)
    plt.savefig(output_file)
    # plt.show()
    # plt.clf()


def plot_cloud_condensation_curves(metallicities, co_ratios, pressures=None, temperatures=None):
    for metallicity in metallicities:
        for co_ratio in co_ratios:
            p, t = return_t_cond_fe(metallicity, co_ratio)
            plt.plot(t, p, label='Fe(c), [Fe/H] = ' + str(metallicity) + ', C/O = ' + str(co_ratio), color='black')
            p, t = return_t_cond_fe_l(metallicity, co_ratio)
            plt.plot(t, p, '--', label='Fe(l), [Fe/H] = ' + str(metallicity) + ', C/O = ' + str(co_ratio))
            p, t = return_t_cond_fe_comb(metallicity, co_ratio)
            plt.plot(t, p, ':', label='Fe(c+l), [Fe/H] = ' + str(metallicity) + ', C/O = ' + str(co_ratio))
            p, t = return_t_cond_mgsio3(metallicity, co_ratio)
            plt.plot(t, p, label='MgSiO3, [Fe/H] = ' + str(metallicity) + ', C/O = ' + str(co_ratio))
            p, t = return_t_cond_na2s(metallicity, co_ratio)
            plt.plot(t, p, label='Na2S, [Fe/H] = ' + str(metallicity) + ', C/O = ' + str(co_ratio))
            p, t = return_t_cond_kcl(metallicity, co_ratio)
            plt.plot(t, p, label='KCL, [Fe/H] = ' + str(metallicity) + ', C/O = ' + str(co_ratio))

    plt.yscale('log')
    plt.xlim([0., 2000.])
    plt.ylim([1e2, 1e-3])
    plt.legend(loc='best', frameon=False)
    plt.show()

    if pressures is not None or temperatures is not None:
        simple_cdf_fe(pressures, temperatures, 0., 0.55)
        simple_cdf_mgsio3(pressures, temperatures, 0., 0.55)
        simple_cdf_na2s(pressures, temperatures, 0., 0.55)
        simple_cdf_kcl(pressures, temperatures, 0., 0.55)


def plot_data(fig, ax, data, resolution=None, scaling=1.0):
    if not data.photometry:
        try:
            # Sometimes this fails, I'm not super sure why.
            resolution_data = np.mean(data.wavelengths[1:] / np.diff(data.wavelengths))
            ratio = resolution_data / resolution
            if int(ratio) > 1:
                flux, edges, _ = binned_statistic(data.wavelengths, data.spectrum, 'mean',
                                                  data.wavelengths.shape[0] / ratio)
                error, _, _ = np.array(binned_statistic(data.wavelengths, data.uncertainties,
                                                        'mean', data.wavelengths.shape[0] / ratio)) / np.sqrt(ratio)
                wlen = np.array([(edges[i] + edges[i + 1]) / 2.0 for i in range(edges.shape[0] - 1)])
            else:
                wlen = data.wavelengths
                # flux/error: commented since it was not used
                # error = data.uncertainties
                # flux = data.spectrum
        except Exception:  # TODO find what is the error expected here
            wlen = data.wavelengths
            # error = data.uncertainties
            # flux = data.spectrum
    else:
        wlen = np.mean(data.photometric_bin_edges)
        # flux = data.spectrum
        # error = data.uncertainties

    marker = 'o'
    scale = 1.0
    errscale = 1.0
    flux = data.spectrum
    error = data.uncertainties

    if data.scale:
        scale = data.scale_factor

    if data.scale_err:
        errscale = data.scale_factor
        error = error * errscale

    offset = 0.0

    if data.offset_bool:
        offset = data.offset

    flux = (flux * scale) - offset
    if data.photometry:
        marker = 's'

    if not data.photometry:
        ax.errorbar(wlen,
                    flux * scaling,
                    yerr=error * scaling * errscale,
                    marker=marker, markeredgecolor='k', linewidth=0, elinewidth=2,
                    label=data.name, zorder=10, alpha=0.9)
    else:
        ax.errorbar(wlen,
                    flux * scaling,
                    yerr=error * scaling * errscale,
                    xerr=data.wavelength_bin_widths / 2., linewidth=0, elinewidth=2,
                    marker=marker, markeredgecolor='k', color='grey', zorder=10,
                    label=None, alpha=0.6)

    return fig, ax


def plot_multiple_posteriors(result_directory, retrieved_parameters, log_evidences=None, true_values=None,
                             add_rectangle=None,
                             parameter_names_ref=None, bins=15, color='C0', figure_font_size=11, use_titles=True,
                             save=False, figure_directory='./', figure_name='fig', image_format='png', fig_size=19.2):
    if isinstance(result_directory, dict):
        result_names = list(result_directory.keys())
        result_directory = list(result_directory.values())

        if log_evidences is not None:
            result_names = [result_name + f' ({log_evidences[i]:.2f})' for i, result_name in enumerate(result_names)]
    else:
        result_names = None

    sample_dict, parameter_names_dict, parameter_plot_indices_dict, parameter_ranges_dict, true_values_dict, \
        fig_titles, _ = _prepare_multiple_retrievals_plot(result_directory, retrieved_parameters, true_values)

    lengths = [len(names) for names in parameter_names_dict.values()]

    nrows = len(sample_dict)
    ncols = np.max(lengths)

    fig_titles_ref = []
    id_ref = np.zeros(np.max(lengths), dtype=int)

    if parameter_names_ref is None:
        parameter_names_ref = parameter_names_dict[str(np.argmax(lengths))]
        id_ref = np.arange(0, np.max(lengths))

        if use_titles:
            fig_titles_ref = fig_titles[str(np.argmax(lengths))]
        else:
            fig_titles_ref = parameter_names_dict[str(np.argmax(lengths))]
    else:
        for sample_id, parameter_name in enumerate(parameter_names_ref):
            if parameter_name not in parameter_names_dict[str(np.argmax(lengths))]:
                raise ValueError(f"unknown parameter '{parameter_name}'")

            for j, pn in enumerate(parameter_names_dict[str(np.argmax(lengths))]):
                if pn == parameter_name:
                    id_ref[sample_id] = j

                    if use_titles:
                        fig_titles_ref.append(fig_titles[str(np.argmax(lengths))][j])
                    else:
                        fig_titles_ref.append(parameter_names_dict[str(np.argmax(lengths))][j])

                    break

    update_figure_font_size(figure_font_size)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_size, fig_size / ncols * nrows), sharex='col')

    if np.ndim(axes) == 1:
        axes = np.array([axes])

    max_col = 0

    # Put last key at the end so that the largest sample is at the bottom of the figure, showing all x-axis tick labels
    _sample_dict = {key: sample_dict[key] for key in list(sample_dict.keys())[1:]}
    _sample_dict[list(sample_dict.keys())[0]] = sample_dict[list(sample_dict.keys())[0]]
    i = 0

    for sample_id in _sample_dict:
        if isinstance(color, dict):
            if result_names[int(sample_id)] in color:
                c = color[result_names[int(sample_id)]]
            else:
                raise ValueError(f"Title {result_names[int(sample_id)]} not found in color")
        else:
            c = color

        for j in range(ncols):
            axes[i, j].set_xlim(parameter_ranges_dict[list(sample_dict.keys())[0]][id_ref[j]])  # type: ignore

            if parameter_names_ref[j] not in parameter_names_dict[sample_id]:
                axes[i, j].axis('off')  # type: ignore
            else:
                if int(sample_id) == add_rectangle:
                    max_col = j

                for k, parameter_name in enumerate(parameter_names_dict[sample_id]):
                    if parameter_name not in parameter_names_ref:
                        raise KeyError(f"unknown parameter '{parameter_name}'")
                    elif parameter_name == parameter_names_ref[j]:
                        if parameter_name == 'new_resolving_power':
                            fmt = '.0f'
                        else:
                            fmt = '.2f'

                        plot_posterior(
                            _sample_dict[sample_id].T[k],
                            label=fig_titles_ref[j],
                            true_value=true_values,
                            cmp=None,
                            bins=bins,
                            color=c,
                            axe=axes[i, j],  # type: ignore
                            y_label=None,
                            tight_layout=False,
                            fmt=fmt
                        )
                        axes[i, j].set_yticks([])  # type: ignore

                        if parameter_name == 'new_resolving_power' and i == len(sample_dict) - 1:
                            x_ticks = axes[i, j].get_xticks()  # type: ignore

                            if len(x_ticks) > 2:
                                x_ticks = x_ticks[::2]
                                axes[i, j].set_xticks(x_ticks[::2])  # type: ignore

                        break

        if result_names is not None:
            axes[i, 0].set_ylabel(result_names[int(sample_id)])  # type: ignore

        i += 1

    fig.tight_layout(rect=(0, 0, 0.99, 1))

    if add_rectangle is not None:
        bbox0 = axes[add_rectangle, 0].get_tightbbox(fig.canvas.get_renderer())  # type: ignore
        bbox1 = axes[add_rectangle, max_col].get_tightbbox(fig.canvas.get_renderer())  # type: ignore
        x0, y0, width0, height = bbox0.transformed(fig.transFigure.inverted()).bounds
        x1, y1, width1, _ = bbox1.transformed(fig.transFigure.inverted()).bounds

        width = x1 - x0 + width1 / 4
        print(width, width0, width1, height, x1, x0)

        # slightly increase the very tight bounds:
        xpad0 = np.min((0.05 * width, x0 * 0.80))
        xpad1 = np.min((0.05 * width, 0.05 / 2))
        ypad = 0.05 * height
        fig.add_artist(
            plt.Rectangle(
                (x0 - xpad0, y0 - ypad),
                width + 2 * xpad1,
                height + 2 * ypad,
                edgecolor='k', linewidth=2, fill=False
            )
        )

    if save:
        plt.savefig(os.path.join(figure_directory, figure_name + '.' + image_format))


def plot_opacity_contributions(
    radtrans_object: Radtrans,
    mode: str,
    include: list = 'all',
    exclude: list = None,
    opacity_contributions: dict = None,
    colors: dict = None,
    line_styles: dict = None,
    fill_below: bool = False,
    fill_alpha: float = 0.5,
    x_axis_scale: str = 'linear',
    y_axis_scale: str = 'linear',
    fig_size: tuple = (12.8, 4.8),
    opaque_cloud_top_pressure: float = None,
    power_law_opacity_350nm: float = None,
    power_law_opacity_coefficient: float = None,
    gray_opacity: float = None,
    cloud_photosphere_median_optical_depth: float = None,
    additional_absorption_opacities_function: Callable = None,
    additional_scattering_opacities_function: Callable = None,
    stellar_intensities: npt.NDArray[np.floating] = None,
    star_radius: float = None,
    **kwargs
) -> dict[str, npt.NDArray[np.floating]]:
    if exclude is None:
        exclude = []

    spectrum_factor = 1
    y_label = 'Flux (A.U.)'

    if mode == 'transmission':
        if star_radius is not None:
            y_label = 'Transit depth (ppm)'
            spectrum_factor = 1e-6  # unity to ppm
        else:
            y_label = 'Transit radius (m)'
            spectrum_factor = 1e-2  # cm to m
    elif mode == 'emission':
        y_label = r'Flux (W$\cdot$m-2/$\mu$m)'
        spectrum_factor = 1e-1  # erg.s-1.cm-2/cm to W.m-2/um

    opacity_sources_colors = LockedDict.build_and_lock({
        'line_species': None,
        'gas_continuum_contributors': None,
        'cloud_species': None,
        'rayleigh_species': None,
        'opaque_cloud_top_pressure': 'darkgray',
        'power_law': 'red',
        'gray_opacity': 'gray',
        'cloud_photosphere_median_optical_depth': 'gold',
        'additional_absorption_opacities_function': 'darkviolet',
        'additional_scattering_opacities_function': 'magenta',
        'stellar_intensities': 'yellow',
        'Total': 'k'
    })

    opacity_sources_linestyles = LockedDict.build_and_lock({
        'line_species': '-',
        'gas_continuum_contributors': '--',
        'cloud_species': '-.',
        'rayleigh_species': (0, (1, 0.66)),  # densely dotted
        'opaque_cloud_top_pressure': '-.',
        'power_law': ':',
        'gray_opacity': ':',
        'cloud_photosphere_median_optical_depth': '-.',
        'additional_absorption_opacities_function': ':',
        'additional_scattering_opacities_function': ':',
        'stellar_intensities': '-',
        'Total': '-'
    })

    if colors is None:
        colors = {}

    if line_styles is None:
        line_styles = {}

    opacity_sources_colors.update(colors)
    opacity_sources_linestyles.update(line_styles)

    # Get the contribution spectra
    if opacity_contributions is None:
        if isinstance(radtrans_object, SpectralModel):
            local_variables = locals()

            _radtrans_object = radtrans_object

            for local_variable, value in local_variables.items():
                if local_variable in radtrans_object.model_parameters and local_variable is not None:
                    if _radtrans_object is radtrans_object:
                        _radtrans_object = copy.deepcopy(radtrans_object)

                    _radtrans_object.model_parameters[local_variable] = value

            opacity_contributions = _radtrans_object.calculate_contribution_spectra(mode=mode)
        else:
            opacity_contributions = radtrans_object.calculate_contribution_spectra(
                mode=mode,
                opaque_cloud_top_pressure=opaque_cloud_top_pressure,
                power_law_opacity_350nm=power_law_opacity_350nm,
                power_law_opacity_coefficient=power_law_opacity_coefficient,
                gray_opacity=gray_opacity,
                cloud_photosphere_median_optical_depth=cloud_photosphere_median_optical_depth,
                additional_absorption_opacities_function=additional_absorption_opacities_function,
                additional_scattering_opacities_function=additional_scattering_opacities_function,
                stellar_intensities=stellar_intensities,
                **kwargs
            )

    # Setup species colors
    default_species_color_index = (i for i in range(1001))

    for opacity_source, species_list in opacity_contributions.items():
        if not isinstance(species_list, dict):
            continue

        if opacity_sources_colors[opacity_source] is None:
            opacity_sources_colors[opacity_source] = {}

        for species in species_list:
            if opacity_source == 'gas_continuum_contributors':
                _species = species.split('--', 1)

                if _species[0] == _species[1]:
                    _species = _species[0]
                elif 'H2' in _species:
                    if _species[0] == 'H2':
                        _species = _species[1]
                    else:
                        _species = _species[0]
                else:
                    _species = _species[0]
            else:
                _species = species

            if species not in opacity_sources_colors[opacity_source]:
                species_color = get_species_color(
                    Opacity.get_species_base_name(_species),
                    implemented_only=False
                )

                if species_color == default_color:
                    i = next(default_species_color_index)

                    if i == 1000:
                        raise ValueError("cannot support more than 1000 species")

                    species_color = f"C{i}"

                opacity_sources_colors[opacity_source][species] = species_color

    # Plot
    fig, axe = plt.subplots(1, 1)
    fig.set_size_inches(fig_size)

    if fill_below:
        function = 'fill_between'
        alpha = fill_alpha
    else:
        function = 'plot'
        alpha = 1

    min_y = np.inf
    max_y = -np.inf

    for opacity_type, opacity_source in opacity_contributions.items():
        if isinstance(opacity_source, tuple):
            _opacity_type = opacity_type.replace('_', ' ')

            if include != 'all' and _opacity_type not in include:
                continue

            if opacity_type in exclude or opacity_source is None:
                continue

            if opacity_type == 'Total':
                z_order = 10
                function = 'plot'
                alpha = 1
            else:
                z_order = 1

            _spectrum = opacity_source[1] * spectrum_factor

            getattr(axe, function)(
                opacity_source[0] * 1e-2, _spectrum,
                label=_opacity_type, color=opacity_sources_colors[opacity_type],
                linestyle=opacity_sources_linestyles[opacity_type],
                zorder=z_order,
                alpha=alpha
            )

            if opacity_type == 'Total' and fill_below:
                function = 'fill_between'
                alpha = fill_alpha

            if np.min(_spectrum) < min_y:
                min_y = np.min(_spectrum)

            if np.max(_spectrum) > max_y:
                max_y = np.max(_spectrum)
        elif opacity_source is None:
            continue
        else:
            for species, spectrum in opacity_source.items():
                _species = Opacity.get_species_scientific_name(species)

                if opacity_type == 'rayleigh_species':
                    _species += ' (Rayleigh)'
                    __species = species + ' (Rayleigh)'
                elif opacity_type == 'gas_continuum_contributors':
                    __species = species.rsplit('-NatAbund', 1)[0]
                else:
                    __species = species

                if include != 'all' and __species not in include:
                    continue

                if __species in exclude or opacity_source is None:
                    continue

                _spectrum = spectrum[1] * spectrum_factor

                getattr(axe, function)(
                    spectrum[0] * 1e-2, spectrum[1] * spectrum_factor,
                    label=_species,
                    color=opacity_sources_colors[opacity_type][species],
                    linestyle=opacity_sources_linestyles[opacity_type],
                    alpha=alpha
                )

                if np.min(_spectrum) < min_y:
                    min_y = np.min(_spectrum)

                if np.max(_spectrum) > max_y:
                    max_y = np.max(_spectrum)

    axe.set_xlabel('Wavelength (m)')

    if mode == 'transmission':
        axe.set_ylabel(y_label)

    axe.set_xlim([radtrans_object.wavelength_boundaries[0] * 1e-6, radtrans_object.wavelength_boundaries[1] * 1e-6])

    delta_y = max_y - min_y
    axe.set_ylim([min_y - delta_y * 0.05, max_y + delta_y * 0.05])

    axe.set_xscale(x_axis_scale)
    axe.set_yscale(y_axis_scale)

    fig.tight_layout()
    fig.legend()

    return opacity_contributions


def plot_planet_context(planet_name: str, mass_radius_uncertainty_tolerance: float = 0.15,
                        fig_size: tuple[float, float] = (6.4, 5.6), figure_font_size: float = 12,
                        plot_annotations: bool = True, plot_planet_references: bool = True, tight_layout: bool = True,
                        save: bool = False, figure_directory: str = './', figure_name: str = 'result_corner',
                        image_format: str = 'png') -> None:
    """Plot an exoplanet mass-radius scatter plot highlighting a planet using the NASA Exoplanet Archive.

    Args:
        planet_name:
            Name of the planet to highlight.
        mass_radius_uncertainty_tolerance:
            Plot only planets with a maximum relative mass and radius uncertainty lower to this value.
        fig_size:
            Size of the figure.
        figure_font_size:
            Base size of the figure font.
        plot_annotations:
            If True, plot the mass limits between rocky planets, sub-neptunes, ice giants and gas giants.
        plot_planet_references:
            If True, place Earth, Neptune and Jupiter on the scatter plot.
        tight_layout:
            If True, apply tight layout on the figure.
        save:
            If True, save the figure.
        figure_directory:
            Directory in which to save the figure.
        figure_name:
            Name of the figure.
        image_format:
            Image format of the saved figure.
    """
    # Get planet data
    print("Fetching for Nasa Exoplanet Archive Planetary Systems Composite Parameters Table...")
    composite_astro_table = Planet.download_from_nasa_exoplanet_archive(
        search_request="select pl_name, pl_bmasse, pl_bmasseerr1, pl_bmasseerr2, pl_rade, pl_radeerr1, pl_radeerr2 "
                       "from pscomppars"
    )
    planet = Planet.get(planet_name)
    planet_mass = planet.mass / cst.m_earth
    planet_radius = planet.radius / cst.r_earth

    jupiter_radius = cst.r_jup / cst.r_earth
    jupiter_mass = cst.m_jup / cst.m_earth

    neptune_radius = 24622e5 / cst.r_earth  # https://doi.org/10.1007/s10569-007-9072-y
    neptune_mass = 1.0243e29 / cst.m_earth  # https://web.archive.org/web/20100701192119/http://nssdc.gsfc.nasa.gov/planetary/factsheet/neptunefact.html  # noqa E501

    # Select the planets to plot
    selected_scatter_planets = []
    selected_radii = []
    selected_masses = []

    print("Selecting planets...")
    for row in composite_astro_table.as_array():
        row = [column for column in row]
        row = row[1:]  # discard planet name (requested in composite_astro_table for debug)

        # Replace masks with nan to prevent warning that masks have been automatically replaced by nan
        for i, column in enumerate(row):
            if np.ma.is_masked(column):
                row[i] = np.nan

        row = np.ma.masked_invalid(row)

        # Calculate normalized mass and radius uncertainties
        mass_uncertainty = np.ma.max(np.ma.abs(row[1:3]))
        radius_uncertainty = np.ma.max(np.ma.abs(row[4:]))

        if np.ma.is_masked(mass_uncertainty) or np.ma.is_masked(row[0]):
            mass_uncertainty = np.nan
        elif row[0] <= 0:
            mass_uncertainty = np.nan
        else:
            mass_uncertainty /= row[0]

        if np.ma.is_masked(radius_uncertainty) or np.ma.is_masked(row[3]):
            radius_uncertainty = np.nan
        elif row[3] <= 0:
            radius_uncertainty = np.nan
        else:
            radius_uncertainty /= row[3]

        # Get the maximum uncertainty
        max_uncertainty = np.max(np.array((mass_uncertainty, radius_uncertainty)))

        # For the mass histogram, ignore the planets for which the mass is unknown
        if not np.ma.is_masked(row[0]):
            selected_masses.append(row[0])

        # For the radius histogram, ignore the planets for which the radius is unknown
        if not np.ma.is_masked(row[3]):
            selected_radii.append(row[3])

        # For the scatter plot, ignore the planets for which max uncertainty is too large or unknown
        if np.isnan(max_uncertainty) or max_uncertainty > mass_radius_uncertainty_tolerance:
            continue

        selected_scatter_planets.append(np.ma.abs(row))

    print(f" Selection statistics:\n"
          f"  - Planets in table: {np.size(composite_astro_table.as_array())}\n"
          f"  - Planets in mass histogram: {len(selected_masses)}\n"
          f"  - Planets in radius histogram: {len(selected_radii)}\n"
          f"  - Planets in scatter plot: {len(selected_scatter_planets)}")

    # Plot the figure
    print("Drawing plot...")
    fig = plt.figure(figsize=fig_size)

    update_figure_font_size(figure_font_size)

    # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between the size of the marginal axes and the
    # main axes in both directions
    # Also adjust the subplot parameters for a square plot
    gs = fig.add_gridspec(2, 2,  width_ratios=(1, 4), height_ratios=(4, 1),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.00, hspace=0.00)
    # Create the Axes
    ax = fig.add_subplot(gs[0, 1])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(direction='in', top=False, right=False, labelleft=False, labelbottom=False, which='both')
    ax_histx = fig.add_subplot(gs[1, 1], sharex=ax)
    ax_histx.set_yscale('linear')
    ax_histx.tick_params(direction='in', top=False, right=False, which='both')
    ax_histx.minorticks_on()
    ax_histy = fig.add_subplot(gs[0, 0], sharey=ax)
    ax_histy.set_xscale('linear')
    ax_histy.tick_params(direction='in', top=False, right=False, which='both')
    ax_histy.minorticks_on()

    # Draw the scatter plot
    ax.errorbar(
        x=np.array(selected_scatter_planets)[:, 0],
        y=np.array(selected_scatter_planets)[:, 3],
        xerr=np.array(selected_scatter_planets)[:, 1:3].T,
        yerr=np.array(selected_scatter_planets)[:, 4:].T,
        color='darkgrey',
        linestyle='',
        marker='+'
    )

    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    # Annotations
    if plot_annotations:
        mass_limit_rocky = 1.9  # super-earth lower limit (https://doi.org/10.1038/nature08679)
        mass_limit_sub_neptune = 10  # super-earth higher limit (https://doi.org/10.1038/nature08679)
        mass_limit_ice_giant = 0.41 * cst.m_jup / cst.m_earth  # self-cmpompression limit (https://doi.org/10.3847/1538-4357/834/1/17)  # noqa E501

        ax.vlines(x=mass_limit_rocky, ymin=y_lim[0], ymax=y_lim[1], color='darkgrey', ls='--', zorder=3)
        ax.text(x=mass_limit_rocky, y=y_lim[1], c='darkgrey', s='Rocky planets ', ha='right', va='top',
                fontsize=figure_font_size * 0.8, zorder=4, rotation='vertical')

        ax.vlines(x=mass_limit_sub_neptune, ymin=y_lim[0], ymax=y_lim[1] * 0.8, color='darkgrey', ls=':', zorder=3)
        ax.text(x=mass_limit_sub_neptune, y=y_lim[1] * 0.8, c='darkgrey', s='?', ha='center', va='bottom',
                fontsize=figure_font_size * 0.8, zorder=4, rotation='horizontal')
        ax.annotate(
            '', xy=(mass_limit_rocky, y_lim[1] * 0.8),
            xytext=(mass_limit_sub_neptune - mass_limit_rocky, y_lim[1] * 0.8),
            arrowprops=dict(arrowstyle="<|-|>", color='darkgrey')
        )
        ax.text(
            x=(mass_limit_sub_neptune + mass_limit_rocky) / 2, y=y_lim[1] * 0.8,
            c='darkgrey', s='Super-earths/ \nSub-Neptunes ', ha='right', va='top',
            fontsize=figure_font_size * 0.8, zorder=4, rotation='vertical'
        )

        # Self-compression limit (https://doi.org/10.3847/1538-4357/834/1/17)
        ax.vlines(x=mass_limit_ice_giant, ymin=y_lim[0], ymax=y_lim[1], color='darkgrey', ls='--', zorder=3)
        ax.text(x=mass_limit_ice_giant, y=y_lim[1], c='darkgrey', s='Ice giants ', ha='right', va='top',
                fontsize=figure_font_size * 0.8, zorder=4, rotation='horizontal')
        ax.text(x=mass_limit_ice_giant, y=y_lim[1], c='darkgrey', s=' Gas giants', ha='left', va='top',
                fontsize=figure_font_size * 0.8, zorder=4, rotation='horizontal')

    # Planet references
    if plot_planet_references:
        # Earth
        ax.scatter(x=[1], y=[1], c='k', marker='o', zorder=3)
        ax.text(x=1, y=1, c='k', s='Earth', ha='left', va='top',
                fontsize=figure_font_size, zorder=4)

        # Neptune
        ax.scatter(x=[neptune_mass], y=[neptune_radius], c='b', marker='o', zorder=3)
        ax.text(x=neptune_mass, y=neptune_radius, c='b', s='Neptune', ha='left', va='top',
                fontsize=figure_font_size, zorder=4)

        # Jupiter
        ax.scatter(x=[jupiter_mass], y=[jupiter_radius], c='C1', marker='o', zorder=3)
        ax.text(x=jupiter_mass, y=jupiter_radius, c='C1', s='Jupiter', ha='left', va='top',
                fontsize=figure_font_size, zorder=4)

    # Highlight the planet
    ax.scatter(x=[planet_mass], y=[planet_radius], c='r', marker='o', zorder=3)
    ax.text(x=planet_mass, y=planet_radius, c='r', s=planet.name, ha='left', va='bottom',
            fontsize=figure_font_size, zorder=4)
    ax.hlines(y=planet_radius, xmin=x_lim[0], xmax=planet_mass, color='r', ls=':', zorder=3)
    ax.vlines(x=planet_mass, ymin=y_lim[0], ymax=planet_radius, color='r', ls=':', zorder=3)

    # Radius histogram
    logbins = np.logspace(
        np.log10(np.min(selected_radii)), np.log10(np.max(selected_radii)), 50
    )
    ax_histy.hist(
        np.array(selected_radii), color='w', ec='k',
        histtype='bar', orientation='horizontal', bins=logbins
    )
    ax_histy.set_ylim(y_lim)
    ax_y_xlim = ax_histy.get_xlim()
    ax_histy.hlines(y=planet_radius, xmin=0, xmax=ax_y_xlim[1], color='r', ls=':', zorder=3)
    ax_histy.set_xlim(ax_y_xlim)
    ax_histy.set_xlabel('Count    ')
    ax_histy.set_ylabel(r'Radius (R$_\oplus$)')
    ax_histy_xticks = ax_histy.get_xticks()

    # Mass histogram
    logbins = np.logspace(
        np.log10(np.min(selected_masses)), np.log10(np.max(selected_masses)), 50
    )
    ax_histx.hist(
        np.array(selected_masses), color='w', ec='k',
        histtype='bar', orientation='vertical', bins=logbins
    )
    ax_histx.set_xlim(x_lim)
    ax_x_ylim = ax_histx.get_ylim()
    ax_histx.vlines(x=planet_mass, ymin=0, ymax=ax_x_ylim[1], color='r', ls=':', zorder=3)
    ax_histx.set_ylim(ax_x_ylim)
    ax_histx.set_xlabel(r'Mass (M$_\oplus$)')
    ax_histx.set_yticks(np.array(ax_histy_xticks[:-1]) / 2)

    if tight_layout:
        fig.tight_layout()

    if save:
        fig.savefig(os.path.join(figure_directory, figure_name + '.' + image_format))


def plot_posterior(data, label=None, true_value=None, cmp=None, bins=15, color='C0',
                   axe=None, y_max=None, y_label=None, tight_layout=True, fmt='.2f'):
    if axe is None:
        fig, axe = plt.subplots(1, 1)

    median = np.median(data)
    sm = np.quantile(data, 0.16)
    sp = np.quantile(data, 0.84)

    c = axe.hist(data, bins=bins, histtype='step', color=color, density=True)

    if y_max is None:
        y_max = np.max(c[0]) * 1.1

    axe.vlines(median, 0, y_max, color=color, ls='--')
    axe.vlines(sm, 0, y_max, color=color, ls='--')
    axe.vlines(sp, 0, y_max, color=color, ls='--')

    if true_value is not None:
        axe.vlines(true_value, 0, y_max, color='r', ls='-')
        ts = f' ({true_value:.2f})'
    else:
        ts = ''

    if cmp is not None:
        plt.errorbar(true_value, y_max * 0.1, xerr=np.array([[cmp[0]], [cmp[1]]]), color='C1', capsize=2, marker='o')

    fmt = "{{0:{0}}}".format(fmt).format
    title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
    title = title.format(fmt(median), fmt(median - sm), fmt(sp - median))

    if label is not None:
        axe.set_xlabel(label + ' = ' + title + ts)

    if y_label is not None:
        axe.set_ylabel('Probability density')

    axe.set_ylim([0, y_max])

    if tight_layout:
        plt.tight_layout()

    return axe


def plot_radtrans_opacities(radtrans, species, temperature, pressure_bar, mass_fractions=None, co_ratio=0.55,
                            log10_metallicity=0., return_opacities=False, **kwargs):
    def __compute_opacities(_pressures, _temperatures):
        """ Method to calculate and return the line opacities (assuming an abundance
        of 100% for the individual species) of the Radtrans object. This method
        updates the line_struc_kappas attribute within the Radtrans class. For the
        low resolution (`c-k`) mode, the wavelength-mean within every frequency bin
        is returned.

            Args:
                _temperatures:
                    the atmospheric temperature in K, at each atmospheric layer
                    (1-d numpy array, same length as pressure array).

            Returns:
                * wavelength in cm (1-d numpy array)
                * dictionary of opacities, keys are the names of the line_species
                  dictionary, entries are 2-d numpy arrays, with the shape
                  being (number of frequencies, number of atmospheric layers).
                  Units are cm^2/g, assuming an absorber abundance of 100 % for all
                  respective species.

        """

        # Function to calc flux, called from outside
        # TODO [4.0.0] _interpolate_species_opacities should not be private if it is used here
        _opacities = radtrans._interpolate_species_opacities(
            pressures=_pressures,
            temperatures=_temperatures,
            n_g=radtrans.lines_loaded_opacities['g_gauss'].size,
            n_frequencies=radtrans.frequencies.size,
            line_opacities_grid=radtrans.lines_loaded_opacities['opacity_grid'],
            line_opacities_temperature_pressure_grid=radtrans.lines_loaded_opacities['temperature_pressure_grid'],
            line_opacities_temperature_grid_size=radtrans.lines_loaded_opacities['temperature_grid_size'],
            line_opacities_pressure_grid_size=radtrans.lines_loaded_opacities['pressure_grid_size']
        )

        _opacities_dict = {}

        weights_gauss = radtrans.lines_loaded_opacities['weights_gauss'].reshape(
            (len(radtrans.lines_loaded_opacities['weights_gauss']), 1, 1)
        )

        for i, _species in enumerate(radtrans.line_species):
            _opacities_dict[_species] = np.sum(_opacities[:, :, i, :] * weights_gauss, axis=0)

        return frequency2wavelength(radtrans.frequencies), _opacities_dict

    temperatures = np.array(temperature)
    pressure_bar = np.array(pressure_bar)

    temperatures = temperatures.reshape(1)
    pressure_bar = pressure_bar.reshape(1)

    pressures = pressure_bar * 1e6

    wavelengths, opacities = __compute_opacities(pressures, temperatures)
    wavelengths *= 1e4  # cm to um

    opacities_weights = {}

    if mass_fractions is None:
        for s in species:
            opacities_weights[s] = 1.
    elif mass_fractions == 'eq':
        mass_fractions = pre_calculated_equilibrium_chemistry_table.interpolate_mass_fractions(
            co_ratios=co_ratio * np.ones_like(temperatures),
            log10_metallicities=log10_metallicity * np.ones_like(temperatures),
            temperatures=temperatures,
            pressures=pressure_bar,
            full=False
        )

        for s in species:
            if s in mass_fractions:
                opacities_weights[s] = mass_fractions[s]
            else:
                # Try to remove opacity information
                chem_spec = s.split('.', 1)[0].split('_', 1)[0].split('-', 1)[0]

                if chem_spec not in mass_fractions:
                    raise KeyError(
                        f"line species '{s}' is not a species in the chemical table\n"
                        f"Try to remove this line species, or do not set mass_fractions to 'eq'."
                    )

                opacities_weights[s] = mass_fractions[chem_spec]
    else:
        for s in species:
            opacities_weights[s] = mass_fractions[s]

    if return_opacities:
        opacities_dict = {}

        for s in species:
            opacities_dict[s] = [
                wavelengths,
                opacities_weights[s] * opacities[s]
            ]

        return opacities_dict
    else:
        for s in species:
            plt.plot(
                wavelengths,
                opacities_weights[s] * opacities[s],
                label=s,
                **kwargs
            )

    return None


def plot_result_corner(retrieval_directory: str, retrieved_parameters: dict, retrieval_name=None,
                       true_values: dict[str, float] = None, spectral_model: SpectralModel = None, smooth: float = 1.0,
                       figure_font_size: float = 8, save: bool = True,
                       figure_directory: str = './', figure_name: str = 'result_corner', image_format: str = 'png',
                       **kwargs):
    """Plot the posteriors of one or multiple retrievals in the same corner plot.

    Args:
        retrieval_directory:
            String or list of string containing the retrieval directories.
        retrieved_parameters:
            Dictionary containing all the retrieved parameters and their prior.
            # TODO retrieved parameters should be automatically stored in retrievals
        retrieval_name:
            Name of the retrieval. If None, the name is extracted from the retrieval directory.
        true_values:
            True values for the parameters.
        spectral_model:
            SpectralModel used to make the retrievals.
        smooth : float
           The standard deviation for Gaussian kernel passed to scipy.ndimage.gaussian_filter` to smooth the 2-D
           histogram. If `None` (default), no smoothing is applied.
        figure_font_size:
            Size of the font in the figure.
        save:
            If True, save the figure.
        figure_directory:
            Directory to save the figure.
        figure_name:
            Name of the figure.
        image_format:
            Format (extension) of the figure.
        **kwargs:
            contour_corner keyword arguments.
    """
    (sample_dict, parameter_names_dict, parameter_plot_indices_dict, parameter_ranges_dict, true_values_dict,
        fig_titles, fig_labels) = _prepare_multiple_retrievals_plot(
        result_directory=retrieval_directory,
        retrieval_name=retrieval_name,
        retrieved_parameters=retrieved_parameters,
        true_values=true_values,
        spectral_model=spectral_model
    )

    if 'titles' not in kwargs:
        kwargs['titles'] = list(fig_titles.values())[0]

    update_figure_font_size(figure_font_size)

    contour_corner(
        sampledict=sample_dict,
        parameter_names=fig_labels,
        output_file=None,
        parameter_plot_indices=parameter_plot_indices_dict,
        parameter_ranges=parameter_ranges_dict,
        true_values=true_values_dict,
        prt_plot_style=False,
        smooth=smooth,
        **kwargs
    )

    figure_size = plt.gcf().get_size_inches()

    if np.max(figure_size) > 19.2:
        plt.gcf().set_size_inches(19.2, 19.2)

    if save:
        plt.savefig(os.path.join(figure_directory, figure_name + '.' + image_format))
