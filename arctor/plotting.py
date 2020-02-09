# https://stackoverflow.com/questions/45786714/custom-marker-edge-style-in-manual-legend

from astropy.modeling.models import Planar2D, Linear1D
from astropy.modeling.fitting import LinearLSQFitter
from astropy.visualization import simple_norm
from matplotlib import pyplot as plt
from photutils import RectangularAperture
from tqdm import tqdm

from scipy.special import erf
from sklearn.neighbors import KernelDensity
from statsmodels.robust import scale as sc

import numpy as np
import pandas as pd

from arctor.utils import find_flux_stddev, print_flux_stddev
from arctor.utils import get_flux_idx_from_df, create_sub_sect
from arctor.utils import get_map_results_models


def debug_message(message, end='\n'):
    print(f'[DEBUG] {message}', end=end)


def warning_message(message, end='\n'):
    print(f'[WARNING] {message}', end=end)


def info_message(message, end='\n'):
    print(f'[INFO] {message}', end=end)


def plot_32_subplots_for_each_feature(aper_widths,
                                      aper_heights,
                                      res_std_ppm, sdnr_apers, chisq_apers,
                                      aic_apers, bic_apers,
                                      idx_split, use_xcenters,
                                      use_ycenters, use_trace_angles,
                                      use_trace_lengths, focus='aic',
                                      one_fig=False):

    if one_fig:
        fig, axs = plt.subplots(nrows=4, ncols=8)

    n_options = len(aper_widths)
    counter = 0
    for idx_split_ in [True, False]:
        for use_xcenters_ in [True, False]:
            for use_ycenters_ in [True, False]:
                for use_trace_angles_ in [True, False]:
                    for use_trace_lengths_ in [True, False]:
                        if one_fig:
                            ax = axs.flatten()[counter]
                            counter = counter + 1
                        else:
                            fig, ax = plt.subplots()

                        plot_aper_grid_per_feature(ax, n_options,
                                                   idx_split,
                                                   use_xcenters,
                                                   use_ycenters,
                                                   use_trace_angles,
                                                   use_trace_lengths,
                                                   res_std_ppm,
                                                   sdnr_apers,
                                                   chisq_apers,
                                                   aic_apers,
                                                   bic_apers,
                                                   aper_widths,
                                                   aper_heights,
                                                   idx_split_,
                                                   use_xcenters_,
                                                   use_ycenters_,
                                                   use_trace_angles_,
                                                   use_trace_lengths_,
                                                   focus=focus,
                                                   one_fig=one_fig)


def plot_map_model(times, phots, uncs, model, t0_guess):
    fig, ax = plt.subplots()

    ax.errorbar(times - t0_guess, phots, uncs,
                fmt='o', ms=10, label='WASP43b UVIS')
    ax.plot(times - t0_guess, model, 'k--', lw=3, label='MAP Model')

    ax.legend(loc=0, fontsize=15)

    ax.set_xlim(None, None)
    ax.set_ylim(None, None)
    ax.set_ylabel('Normalized Flux [ppm]')
    ax.set_xlabel('Hours from Eclipse', fontsize=20)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fonstize(15)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fonstize(15)


def plot_feature_vs_res_diff(idx_split, use_xcenters, use_ycenters,
                             use_trace_angles, use_trace_lengths,
                             res_diff_ppm, width=1e-3):
    plt.scatter(idx_split[~idx_split] * width + 0 * width,
                res_diff_ppm[~idx_split],
                alpha=0.1, edgecolors='None')
    plt.scatter(idx_split[idx_split] * width + 0 * width,
                res_diff_ppm[idx_split],
                alpha=0.1, edgecolors='None')

    plt.scatter(use_xcenters[~use_xcenters] * width + 1 * width,
                res_diff_ppm[~use_xcenters],
                alpha=0.1, edgecolors='None')
    plt.scatter(use_xcenters[use_xcenters] * width + 1 * width,
                res_diff_ppm[use_xcenters],
                alpha=0.1, edgecolors='None')

    plt.scatter(use_ycenters[~use_ycenters] * width + 2 * width,
                res_diff_ppm[~use_ycenters],
                alpha=0.1, edgecolors='None')
    plt.scatter(use_ycenters[use_ycenters] * width + 2 * width,
                res_diff_ppm[use_ycenters],
                alpha=0.1, edgecolors='None')

    plt.scatter(use_trace_angles[use_trace_angles] * width + 3 * width,
                res_diff_ppm[~use_trace_angles],
                alpha=0.1, edgecolors='None')
    plt.scatter(use_trace_angles[use_trace_angles] * width + 3 * width,
                res_diff_ppm[use_trace_angles],
                alpha=0.1, edgecolors='None')

    plt.scatter(use_trace_lengths[~use_trace_lengths] * width + 4 * width,
                res_diff_ppm[~use_trace_lengths],
                alpha=0.1, edgecolors='None')
    plt.scatter(use_trace_lengths[use_trace_lengths] * width + 4 * width,
                res_diff_ppm[use_trace_lengths],
                alpha=0.1, edgecolors='None')

    plt.scatter([], [],
                label='IDX Split False', edgecolors='None', color='C0')
    plt.scatter([], [],
                label='IDX Split True', edgecolors='None', color='C1')
    plt.scatter([], [],
                label='Use Xcenters False', edgecolors='None', color='C2')
    plt.scatter([], [],
                label='Use Xcenters True', edgecolors='None', color='C3')
    plt.scatter([], [],
                label='Use Ycenters False', edgecolors='None', color='C4')
    plt.scatter([], [],
                label='Use Ycenters True', edgecolors='None', color='C5')
    plt.scatter([], [],
                label='Use Trace Angles False', edgecolors='None', color='C6')
    plt.scatter([], [],
                label='Use Trace Angles True', edgecolors='None', color='C7')
    plt.scatter([], [],
                label='Use Trace Lengths False', edgecolors='None', color='C8')
    plt.scatter([], [],
                label='Use Trace Lengths True', edgecolors='None', color='C9')

    plt.xlim(-width, None)
    plt.legend(loc=0, fontsize=15)


def plot_aper_width_grid():
    rand0 = np.random.normal(0, 0.1, 3200)

    plt.scatter(res_std_ppm[res_diff_ppm > 0],
                res_diff_ppm[res_diff_ppm > 0],
                c=res_diff_ppm[res_diff_ppm > 0])

    plt.scatter((aper_widths + 0.25 * use_xcenters)[use_xcenters],
                (aper_heights + 0.25 * use_ycenters)[~use_ycenters],
                c=res_std_ppm, alpha=0.25, label='x:True y:False', marker='o')

    plt.scatter((aper_widths + 0.25 * use_xcenters)[use_xcenters],
                (aper_heights + 0.25 * use_ycenters)[use_ycenters],
                c=res_std_ppm, alpha=0.25, label='x:True y:True', marker='s')

    plt.scatter((aper_widths + 0.25 * use_xcenters)[~use_xcenters],
                (aper_heights + 0.25 * use_ycenters)[use_ycenters],
                c=res_std_ppm, alpha=0.25, label='x:False y:True', marker='*')

    plt.scatter((aper_widths + 0.25 * use_xcenters)[~use_xcenters],
                (aper_heights + 0.25 * use_ycenters)[~use_ycenters],
                c=res_std_ppm[~use_xcenters], alpha=0.25,
                label='x:False y:False', marker='^')

    plt.legend(loc=0, fontsize=15)


def plot_aper_grid_per_feature(ax, n_options, idx_split, use_xcenters,
                               use_ycenters, use_trace_angles,
                               use_trace_lengths, res_std_ppm,
                               sdnr_apers, chisq_apers, aic_apers, bic_apers,
                               aper_widths, aper_heights,
                               idx_split_, use_xcenters_, use_ycenters_,
                               use_trace_angles_, use_trace_lengths_,
                               focus='aic', one_fig=False, fig=None,
                               hspace=0.5):

    foci = {}
    foci['std'] = res_std_ppm
    foci['aic'] = aic_apers
    foci['bic'] = bic_apers
    foci['chisq'] = chisq_apers
    foci['sdnr'] = sdnr_apers

    focus_ = foci[focus]

    sub_sect = create_sub_sect(n_options,
                               idx_split,
                               use_xcenters,
                               use_ycenters,
                               use_trace_angles,
                               use_trace_lengths,
                               idx_split_,
                               use_xcenters_,
                               use_ycenters_,
                               use_trace_angles_,
                               use_trace_lengths_)
    if one_fig:
        size = 100000
    else:
        size = 200

    out = ax.scatter(aper_widths[sub_sect], aper_heights[sub_sect],
                     c=focus_[sub_sect],
                     marker='s', s=200)

    min_aic_sub = focus_[sub_sect].min()
    argmin_aic_sub = focus_[sub_sect].argmin()
    manual_argmin = np.where(focus_[sub_sect] == min_aic_sub)[0][0]
    assert(manual_argmin == argmin_aic_sub), \
        f'{manual_argmin}, {argmin_aic_sub}'

    best_ppm = sdnr_apers[sub_sect][argmin_aic_sub]
    width_best = aper_widths[sub_sect][argmin_aic_sub]
    height_best = aper_heights[sub_sect][argmin_aic_sub]

    txt = f'{np.round(best_ppm):0.0f} ppm\n[{width_best}x{height_best}]'
    ax.plot(width_best - 0.5, height_best - 0.5, 'o',
            color='C1', ms=10)

    ax.annotate(txt,
                (width_best - 0.5 + 0.1, height_best - 0.5 + 0.1),
                # xycoords='axes fraction',
                xytext=(width_best - 0.5 + 0.1, height_best - 0.5 + 0.1),
                # textcoords='offset points',
                ha='left',
                va='bottom',
                fontsize=12,
                color='C1',
                weight='bold')

    features = ''
    if idx_split_:
        features = features + 'Fwd/Rev Split\n'

    if use_xcenters_:
        features = features + 'Fit Xcenters\n'

    if use_ycenters_:
        features = features + 'Fit Ycenters\n'

    if use_trace_angles_:
        features = features + 'Fit Trace Angles\n'

    if use_trace_lengths_:
        features = features + 'Fit Trace Lengths'

    if features == '':
        features = 'Null Hypothesis'

    annotate_loc = (0.1, 0.9)
    ax.annotate(features,
                annotate_loc,
                xycoords='axes fraction',
                xytext=annotate_loc,
                textcoords='offset points',
                ha='left',
                va='top',
                fontsize=10,
                color='black',
                weight='bold')

    title = f'{focus.upper()}: {np.int(np.round(min_aic_sub)):0.0f}'
    ax.set_title(title)

    if one_fig:
        plt.subplots_adjust(hspace=hspace)

    if not one_fig:
        if fig is None:
            fig = plt.gcf()

        left, bottom, width, height = ax.get_position().bounds
        cbaxes = fig.add_axes([left + width, bottom, 0.025, height])
        cb = plt.colorbar(out, cax=cbaxes)


def plot_aperture_edges_with_angle(planet, img_id=42):
    image = planet.image_stack[img_id]
    y_center = planet.trace_ycenters[img_id]
    x_left = planet.x_left
    x_right = planet.x_right
    trace_width = x_right - x_left

    positions = np.transpose([planet.trace_xcenters, planet.trace_ycenters])
    thetas = planet.trace_angles

    aper_tilt = RectangularAperture(
        positions[img_id], trace_width, 2, thetas[img_id])
    aper_flat = RectangularAperture(positions[img_id], trace_width, 2, 0)

    fig, axs = plt.subplots(nrows=2, ncols=2)
    plt.subplots_adjust(bottom=0, left=0, right=1,
                        top=0.95, hspace=0.01, wspace=0.01)

    [ax.axis('off') for axrow in axs for ax in axrow]
    [ax.imshow(image) for axrow in axs for ax in axrow]
    [ax.set_ylim(y_center - 5, y_center + 5) for axrow in axs for ax in axrow]

    aper_tilt.plot(axes=axs[0][0], color='white')
    aper_tilt.plot(axes=axs[0][1], color='white')
    aper_flat.plot(axes=axs[1][0], color='red')
    aper_flat.plot(axes=axs[1][1], color='red')

    axs[0][0].set_xlim(x_left - 10, x_left + 10)
    axs[1][0].set_xlim(x_left - 10, x_left + 10)
    axs[0][1].set_xlim(x_right - 10, x_right + 10)
    axs[1][1].set_xlim(x_right - 10, x_right + 10)

    axs[0][0].annotate('With Calculated Tilt',
                       (0, 0),
                       xycoords='axes fraction',
                       xytext=(5, 5),
                       textcoords='offset points',
                       ha='left',
                       va='bottom',
                       fontsize=20,
                       color='white')

    axs[0][1].annotate('With Calculated Tilt',
                       (0, 0),
                       xycoords='axes fraction',
                       xytext=(5, 5),
                       textcoords='offset points',
                       ha='left',
                       va='bottom',
                       fontsize=20,
                       color='white')

    axs[1][0].annotate('Without Calculated Tilt',
                       (0, 0),
                       xycoords='axes fraction',
                       xytext=(5, 5),
                       textcoords='offset points',
                       ha='left',
                       va='bottom',
                       fontsize=20,
                       color='red')

    axs[1][1].annotate('Without Calculated Tilt',
                       (0, 0),
                       xycoords='axes fraction',
                       xytext=(5, 5),
                       textcoords='offset points',
                       ha='left',
                       va='bottom',
                       fontsize=20,
                       color='red')

    fig.suptitle('Example Aperture With and Without Rotation', fontsize=20)


def uniform_scatter_plot(planet, arr1, arr2, include_orbits=True,
                         arr1_center=0, use_time_sort=True, size=50,
                         title='', xlabel='', ylabel='', ax=None):
    # https://stackoverflow.com/questions/45786714/custom-marker-edge-style-in-manual-legend
    # marker=u'$\u25CC$'

    if ax is None:
        fig, ax = plt.subplots()

    ax.clear()

    if use_time_sort:
        time_sort = np.argsort(planet.times)
    else:
        time_sort = np.arange(len(arr1))

    idx_orbit1 = np.arange(18)  # by eye
    idx_orbit2 = np.arange(18, 37)  # by eye
    idx_eclipse = np.arange(37, 56)  # by eye
    idx_orbit4 = np.arange(56, len(arr1))  # by eye

    ax.scatter(arr1[planet.idx_fwd] - arr1_center,
               arr2[planet.idx_fwd], s=size,
               color='C0', label='Forward Scans')

    ax.scatter(arr1[planet.idx_rev] - arr1_center,
               arr2[planet.idx_rev], s=size,
               color='C1', label='Reverse Scans')

    ax.set_title(title, fontsize=20)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)

    if not include_orbits:
        ax.legend(loc=0, fontsize=20)
        return ax

    # By hand values from looking at the light curve
    ax.plot(arr1[time_sort][idx_orbit1] - arr1_center,
            arr2[time_sort][idx_orbit1], 'o',
            ms=20, mew=2, color='none', mec='black',
            label='First Orbit', zorder=0, marker=u'$\u25CC$')

    ax.plot(arr1[time_sort][idx_orbit2] - arr1_center,
            arr2[time_sort][idx_orbit2], 'o',
            ms=20, mew=2, color='none', mec='lightgreen',
            label='Second Orbit', zorder=0)  # , marker=u'$\u25CC$')

    ax.plot(arr1[time_sort][idx_eclipse] - arr1_center,
            arr2[time_sort][idx_eclipse], 'o',
            ms=20, mew=2, color='none', mec='pink',
            label='Third Orbit (eclipse)', zorder=0)  # , marker=u'$\u25CC$')

    ax.plot(arr1[time_sort][idx_orbit4] - arr1_center,
            arr2[time_sort][idx_orbit4], 'o',
            ms=20, mew=2, color='none', mec='indigo',
            label='Fourth Orbit', zorder=0)  # , marker=u'$\u25CC$')

    ax.legend(loc=0, fontsize=20)
    return ax


def plot_center_position_vs_scan_and_orbit(planet, t0_base=0, size=50,
                                           include_orbits=True, ax=None):
    title = 'Center Positions of the Trace in Forward and Reverse Scanning'
    xlabel = 'X-Center [pixels]'
    ylabel = 'Y-Center [pixels]'

    ax = uniform_scatter_plot(planet,
                              planet.trace_xcenters,
                              planet.trace_ycenters,
                              arr1_center=t0_base,
                              title=title,
                              xlabel=xlabel,
                              ylabel=ylabel,
                              size=size,
                              include_orbits=include_orbits,
                              ax=ax)

    return ax


def plot_ycenter_vs_time(planet, t0_base=None, size=50,
                         include_orbits=True, ax=None):
    title = 'Y-Positions vs Time of the Trace'
    xlabel = 'Time Since Mean Time [days]'
    ylabel = 'Y-Center [pixels]'

    ax = uniform_scatter_plot(planet,
                              planet.times,
                              planet.trace_ycenters,
                              arr1_center=t0_base,
                              title=title,
                              xlabel=xlabel,
                              ylabel=ylabel,
                              size=size,
                              include_orbits=include_orbits,
                              ax=ax)

    return ax


def plot_xcenter_vs_time(planet, t0_base=None, size=50,
                         include_orbits=True, ax=None):
    title = 'X-Positions vs Time of the Trace'
    xlabel = 'Time Since Mean Time [days]'
    ylabel = 'X-Center [pixels]'

    ax = uniform_scatter_plot(planet,
                              planet.times,
                              planet.trace_xcenters,
                              arr1_center=t0_base,
                              title=title,
                              xlabel=xlabel,
                              ylabel=ylabel,
                              size=size,
                              include_orbits=include_orbits,
                              ax=ax)

    return ax


def plot_trace_angle_vs_time(planet, nSig=0.5, size=50, t0_base=None,
                             include_orbits=True, ax=None):
    title = 'Trace Angle vs Time of the Trace'
    xlabel = 'Time Since Mean Time [days]'
    ylabel = 'Trace Angle [degrees]'

    trace_angles = planet.trace_angles * 180 / np.pi

    ax = uniform_scatter_plot(planet,
                              planet.times,
                              trace_angles,
                              arr1_center=t0_base,
                              title=title,
                              xlabel=xlabel,
                              ylabel=ylabel,
                              size=size,
                              include_orbits=include_orbits,
                              ax=ax)

    y_min, y_max = np.percentile(trace_angles, [0.1, 99.9])
    inliers = (trace_angles > y_min) * (trace_angles < y_max)
    angles_std = trace_angles[inliers].std()

    ax.set_ylim(y_min - nSig * angles_std,
                y_max + nSig * angles_std)

    return ax


def plot_trace_length_vs_time(planet, t0_base=None, size=50,
                              include_orbits=True, ax=None):
    title = 'Trace Lengths vs Time of the Trace'
    xlabel = 'Time Since Mean Time [days]'
    ylabel = 'Trace Length [pixels]'

    ax = uniform_scatter_plot(planet,
                              planet.times,
                              planet.trace_lengths,
                              arr1_center=t0_base,
                              title=title,
                              xlabel=xlabel,
                              ylabel=ylabel,
                              size=size,
                              include_orbits=include_orbits,
                              ax=ax)

    return ax


def plot_columwise_background_vs_time(planet,  # aper_width, aper_height,
                                      t0_base=None, size=50,
                                      include_orbits=True, ax=None):

    title = 'Sky Background vs Time of the Trace'
    xlabel = 'Time Since Mean Time [days]'
    ylabel = 'Sky Background [electrons]'

    # aper_column = f'aperture_sum_{aper_width}x{aper_height}'

    sky_background = np.median(planet.sky_bg_columnwise, axis=1)
    # sky_background = sky_background / planet.photometry_df[aper_column]

    ax = uniform_scatter_plot(planet,
                              planet.times,
                              sky_background,
                              arr1_center=t0_base,
                              title=title,
                              xlabel=xlabel,
                              ylabel=ylabel,
                              size=size,
                              include_orbits=include_orbits,
                              ax=ax)

    return ax


def plot_aperture_background_vs_time(planet, t0_base=None, size=50,
                                     include_orbits=True, ax=None):
    title = 'Sky Background vs Time of the Trace'
    xlabel = 'Time Since Mean Time [days]'
    ylabel = 'Sky Background [electrons]'

    ax = uniform_scatter_plot(planet,
                              planet.times,
                              planet.sky_bgs,
                              arr1_center=t0_base,
                              title=title,
                              xlabel=xlabel,
                              ylabel=ylabel,
                              size=size,
                              include_orbits=include_orbits,
                              ax=ax)

    return ax


def plot_ycenter_vs_flux(planet, aper_width, aper_height,
                         use_time_sort=False, nSig=0.5, size=50,
                         t0_base=0, ax=None, include_orbits=True):

    ppm = 1e6

    aper_column = f'aperture_sum_{aper_width}x{aper_height}'
    fluxes = planet.normed_photometry_df[aper_column]

    title = 'Flux vs Y-Center Positions'
    xlabel = 'Y-Center [pixels]'
    ylabel = 'Flux [ppm]'

    med_flux = np.median(fluxes)
    fluxes = (fluxes - med_flux) * ppm

    min_flux, max_flux = np.percentile(fluxes, [0.01, 99.99])
    std_flux = fluxes[(fluxes >= min_flux) * (fluxes <= max_flux)].std()

    ax = uniform_scatter_plot(planet,
                              planet.trace_ycenters,
                              fluxes,
                              arr1_center=0,
                              title=title,
                              xlabel=xlabel,
                              ylabel=ylabel,
                              use_time_sort=use_time_sort,
                              size=size,
                              include_orbits=include_orbits,
                              ax=ax)

    ax.set_ylim(min_flux - nSig * std_flux,
                max_flux + nSig * std_flux)

    return ax


def plot_2D_fit_time_vs_other(times, flux, other, idx_fwd, idx_rev,
                              xytext=(15, 15), n_sig=5, varname='Other',
                              n_spaces=[10, 10], convert_to_ppm=True,
                              lw=3, fontsize=10, xlim=None, fig=None, ax=None):
    ppm = 1e6

    if fig is None and ax is None:
        fig, ax = plt.subplots()

    if ax is None:
        ax = fig.add_subplot(111)

    if ax is None:
        fig, ax = plt.subplots()

    inliers = np.sqrt((other - np.median(other))**2 + (flux - np.median(flux))
                      ** 2) < n_sig * np.sqrt(np.var(other) + np.var(flux))

    fitter_o = LinearLSQFitter()
    fitter_t = LinearLSQFitter()

    model_o = Linear1D(slope=1e-6, intercept=np.median(flux))
    model_t = Linear1D(slope=-1e-3, intercept=0)

    times_med = np.median(times[inliers])
    times_std = np.median(times[inliers])
    times_normed = (times - times_med) / times_std

    other_med = np.median(other[inliers])
    other_std = np.median(other[inliers])
    other_normed = (other - other_med) / other_std

    flux_med = np.median(flux[inliers])
    flux_std = np.median(flux[inliers])
    flux_normed = (flux - flux_med) / flux_std

    fit_t = fitter_t(model_t, times_normed[inliers], flux_normed[inliers])

    flux_corrected = flux_normed - fit_t(times_normed)
    fit_o = fitter_o(model_o, other_normed[inliers], flux_corrected[inliers])

    model_comb = Planar2D(slope_x=fit_o.slope,
                          slope_y=fit_t.slope,
                          intercept=fit_t.intercept)

    fit_comb = fitter_t(model_comb,
                        other_normed[inliers],
                        times_normed[inliers],
                        flux_normed[inliers])

    # annotation = (f'o_slope:{fit_o.slope.value:0.2e}\n'
    #               f't_slope:{fit_t.slope.value:0.2e}\n'
    #               f'c_slope_o:{fit_comb.slope_x.value:0.2e}\n'
    #               f'c_slope_t:{fit_comb.slope_y.value:0.2e}\n'
    #               f'o_intcpt:{fit_o.intercept.value:0.2e}\n'
    #               f't_intcpt:{fit_t.intercept.value:0.2e}\n'
    #               f'c_intcpt:{fit_comb.intercept.value:0.2e}'
    #               )

    n_sp0, n_sp1 = n_spaces
    annotation = (f'2D Slope {varname}: {fit_comb.slope_x.value:0.2e}\n'
                  f'2D Slope Time:{" "*n_sp0}{fit_comb.slope_y.value:0.2f}\n'
                  f'2D Intercept:{" "*(n_sp1)}'
                  f'{fit_comb.intercept.value * flux_std * ppm:0.2f} [ppm]'
                  )

    min_o = other_normed.min()
    max_o = other_normed.max()
    min_t = times_normed.min()
    max_t = times_normed.max()

    ax.plot(other_normed[idx_fwd] * other_std,
            flux_normed[idx_fwd] * flux_std * ppm,
            'o', label='Forward Scan')
    ax.plot(other_normed[idx_rev] * other_std,
            flux_normed[idx_rev] * flux_std * ppm,
            'o', label='Reverse Scan')

    other_normed_th = np.linspace(2 * min_o, 2 * max_o, 100)
    times_normed_th = np.linspace(2 * min_t, 2 * max_t, 100)

    best_model = fit_comb(other_normed_th, times_normed_th)
    ax.plot(other_normed_th * other_std, best_model * flux_std * ppm,
            lw=lw, zorder=0)

    ax.set_title(f'{varname} + Time 2D Fit to Flux')
    ax.annotate(annotation,
                (0, 0),
                xycoords="axes fraction",
                xytext=xytext,
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=12,
                )

    if xlim is None:
        ax.set_xlim(1.05 * min_o * other_std, 1.05 * max_o * other_std)
    else:
        assert(len(xlim) == 2), "`xlim` must be a 2-tuple"
        ax.set_xlim(xlim)

    ax.set_ylabel('Flux [ppm]')
    ax.set_xlabel(f'{varname} [Median Subtracted]')
    ax.legend(loc=0, fontsize=fontsize)

    return fig, ax


def plot_kde_with_BCR_annotation(mcmc_samples_df, min_edepth=0, max_edepth=100,
                                 n_edepths=1000, kernel='gaussian', lw=5,
                                 bins=50, verbose=False, ax=None,
                                 hist_color='C4', kde_color='C0',
                                 kde_alpha=1.0, include_hist=True):

    # spelled Frenchy on purpose
    bleus = ('#1f77b4', '#52aae7', '#85ddff')

    if ax is None:
        fig, ax = plt.subplots()

    ax.clear()

    ppm = 1e6

    edepths = mcmc_samples_df['edepth'].values

    edepths_fake = np.r_[edepths,
                         -(edepths - np.min(edepths)) + np.min(edepths)]

    sigmas = erf(np.arange(1, 6) / np.sqrt(2))
    percentiles = np.percentile(edepths * ppm, sigmas * 100)

    if verbose:
        for k, (sigma_, perc_) in enumerate(zip(sigmas, percentiles)):
            print(f'{k+1}-sigma: {sigma_*100:0.5f}% - {perc_*1e6:0.1f} ppm')

    edepths_med = np.median(edepths)
    edepths_th_real = np.linspace(min_edepth, max_edepth, n_edepths)

    # instantiate and fit the KDE model
    kde = KernelDensity(bandwidth=1.0, kernel=kernel)
    kde.fit((edepths_fake * ppm)[:, None])

    # score_samples returns the log of the probability density
    logprob = kde.score_samples(edepths_th_real[:, None])
    kde_edepths_real_vals = np.exp(logprob)
    max_kde_edepth = np.max(kde_edepths_real_vals)
    edepths_mode = edepths_th_real[np.argmax(kde_edepths_real_vals)]

    lbl_base = 'MCMC Posterior'
    if include_hist:
        yhist, _, _ = ax.hist(edepths * ppm, bins=bins, density=True,
                              color=hist_color, zorder=-1, alpha=0.7,
                              label=f'{kernel.capitalize()} {lbl_base}'
                              ' Histogram')
    else:
        yhist, _ = np.histogram(edepths * ppm, bins=bins, density=True)

        plt.fill_between(edepths_th_real, 100 * kde_edepths_real_vals * 2,
                         color='darkgrey')  # , alpha=kde_alpha)

    last_perc = min_edepth
    for color_, perc_ in zip(bleus, percentiles):
        edepths_th_ = np.linspace(
            last_perc, perc_, n_edepths // len(bleus))

        logprob_ = kde.score_samples(edepths_th_[:, None])
        kde_edepths_ = np.exp(logprob_)
        plt.fill_between(edepths_th_, 100 * kde_edepths_ * 2,
                         color=color_, alpha=kde_alpha)

        last_perc = perc_

    plt.fill_between([], [], color=bleus[0], alpha=kde_alpha,
                     label=f'{kernel.capitalize()} {lbl_base} KDE')

    ax.axvline(edepths_mode, color='C1', ls='--', lw=lw)
    annotation = f'{kernel.capitalize()} KDE Mode: {edepths_mode:0.0f} ppm'
    ax.annotate(annotation,
                xy=(edepths_mode + 1.0, 100 * 0.25 * max_kde_edepth),
                rotation=90, color='C1', fontsize=30)

    ax.plot([], [], color='C1', ls='-', lw=lw,
            label=f'Mode {kernel.capitalize()} KDE-MCMC')

    latex_sigma = r"$\sigma$"
    for k, (sigma_, perc_) in enumerate(zip(sigmas, percentiles)):
        # sigma_str = f'{k+1}-{latex_sigma}'
        annotation = f'{perc_:0.0f} ppm'
        ax.axvline(perc_, color='#555555', ls='--', lw=2)
        ax.annotate(annotation, xy=(perc_ + 0.5, 100 * 0.8 * max_kde_edepth),
                    rotation=90, color='#555555', fontsize=30)

    ax.set_ylim(0, 100 * yhist.max() * 1.02)
    ax.set_xlim(min_edepth, max_edepth)
    ax.set_xlabel('Eclipse Depth [ppm]', fontsize=20)
    ax.set_ylabel('Marginalized Posterior Probability [%]', fontsize=20)

    ax.legend(loc=9, fontsize=20)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)

    for key, val in ax.spines.items():
        val.set_visible(False)

    return ax


def plot_xcenter_vs_flux(planet, aper_width, aper_height,
                         use_time_sort=False, nSig=0.5, t0_base=0, size=50,
                         include_orbits=True, ax=None):

    ppm = 1e6

    aper_column = f'aperture_sum_{aper_width}x{aper_height}'
    fluxes = planet.normed_photometry_df[aper_column]

    title = 'Flux vs X-Center Positions'
    xlabel = 'X-Center [pixels]'
    ylabel = 'Flux [ppm]'

    med_flux = np.median(fluxes)
    fluxes = (fluxes - med_flux) * ppm

    min_flux, max_flux = np.percentile(fluxes, [0.01, 99.99])
    std_flux = fluxes[(fluxes >= min_flux) * (fluxes <= max_flux)].std()

    ax = uniform_scatter_plot(planet,
                              planet.trace_xcenters,
                              fluxes,
                              arr1_center=0,
                              title=title,
                              xlabel=xlabel,
                              ylabel=ylabel,
                              use_time_sort=use_time_sort,
                              include_orbits=include_orbits,
                              size=size,
                              ax=ax)

    ax.set_ylim(min_flux - nSig * std_flux,
                max_flux + nSig * std_flux)

    return ax


def plot_best_aic_light_curve(planet, map_solns,
                              decor_results_df,  # mcmc_samples_df,
                              aic_apers,  keys_list,
                              aic_thresh=2, t0_base=0,
                              plot_many=False, plot_raw=False,
                              ax=None):
    ppm = 1e6

    idx_fwd = planet.idx_fwd
    idx_rev = planet.idx_rev
    times = planet.times - t0_base

    aic_argmin = decor_results_df['aic_sub_min'].values.argmin()
    aic_min = decor_results_df['aic_sub_min'].iloc[aic_argmin]

    width_best = decor_results_df['width_best'].iloc[aic_argmin]
    height_best = decor_results_df['height_best'].iloc[aic_argmin]
    idx_split = decor_results_df['idx_split'].iloc[aic_argmin]
    use_xcenters = decor_results_df['xcenters'].iloc[aic_argmin]
    use_ycenters = decor_results_df['xcenters'].iloc[aic_argmin]
    use_trace_angles = decor_results_df['trace_angles'].iloc[aic_argmin]
    use_trace_lengths = decor_results_df['trace_lengths'].iloc[aic_argmin]

    aper_column = f'aperture_sum_{width_best}x{height_best}'

    map_soln_key = (f'aper_column:{aper_column}-'
                    f'idx_split:{idx_split}-'
                    f'_use_xcenters:{use_xcenters}-'
                    f'_use_ycenters:{use_ycenters}-'
                    f'_use_trace_angles:{use_trace_angles}-'
                    f'_use_trace_lengths:{use_trace_lengths}')

    map_soln = map_solns[map_soln_key]
    map_model, line_model = get_map_results_models(times, map_soln,
                                                   idx_fwd, idx_rev)

    if ax is None:
        fig, ax = plt.subplots()

    ax.clear()

    phots = planet.normed_photometry_df[aper_column]
    uncs = planet.normed_uncertainty_df[aper_column]

    phots_corrected = (phots - line_model)

    ax.errorbar(times[idx_fwd],
                phots_corrected[idx_fwd] * ppm,
                uncs[idx_fwd] * ppm, label='Forward Scan',
                fmt='o', color='C0', ms=10, zorder=10)

    ax.errorbar(times[idx_rev],
                phots_corrected[idx_rev] * ppm,
                uncs[idx_rev] * ppm, label='Reverse Scan',
                fmt='o', color='C1', ms=10, zorder=10)

    ax.plot(times[times.argsort()], map_model[times.argsort()] * ppm,
            label='Best Fit Model', color='C7', lw=3, zorder=5)

    ax.axhline(0.0, ls='--', color='k', lw=2,
               zorder=-1, label='Null Hypothesis')

    if plot_raw:
        phots_med = np.median(phots)
        phots_med_sub = phots - phots_med
        ax.plot(times[idx_fwd], phots_med_sub[idx_fwd] * ppm, 'o',
                color='darkblue', ms=10, zorder=0, alpha=0.2, mew=0)
        ax.plot(times[idx_rev], phots_med_sub[idx_rev] * ppm, 'o',
                color='darkorange', ms=10, zorder=0, alpha=0.2, mew=0)

    ax.set_xlabel('Time [Days from Eclipse]', fontsize=20)
    ax.set_ylabel('Normalized Flux [ppm]', fontsize=20)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(15)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)

    ax.legend(loc=0, fontsize=15)
    if not plot_many:
        plt.show()
        return ax

    idx_split_vals = decor_results_df['idx_split'].values
    use_xcenters_vals = decor_results_df['xcenters'].values
    use_ycenters_vals = decor_results_df['xcenters'].values
    use_trace_angles_vals = decor_results_df['trace_angles'].values
    use_trace_lengths_vals = decor_results_df['trace_lengths'].values

    for aper_column in tqdm(planet.normed_photometry_df.columns):
        if 'aperture_sum_' not in aper_column:
            continue

        for entry in zip(idx_split_vals,
                         use_xcenters_vals,
                         use_ycenters_vals,
                         use_trace_angles_vals,
                         use_trace_lengths_vals):

            idx_split = entry[0]
            use_xcenters = entry[1]
            use_xcenters = entry[2]
            use_trace_angles = entry[3]
            use_trace_lengths = entry[4]

            phots = planet.normed_photometry_df[aper_column]
            uncs = planet.normed_uncertainty_df[aper_column]

            map_soln_key = (f'aper_column:{aper_column}-'
                            f'idx_split:{idx_split}-'
                            f'_use_xcenters:{use_xcenters}-'
                            f'_use_ycenters:{use_ycenters}-'
                            f'_use_trace_angles:{use_trace_angles}-'
                            f'_use_trace_lengths:{use_trace_lengths}')

            idx_ = keys_list == map_soln_key
            aic_ = aic_apers[idx_][0]

            if abs(aic_ - aic_min) > aic_thresh:
                continue

            map_soln = map_solns[map_soln_key]
            map_model, line_model = get_map_results_models(times,
                                                           map_soln,
                                                           idx_fwd,
                                                           idx_rev)

            ax.plot(times, (phots - line_model) * ppm, 'o',
                    color='lightgrey', alpha=0.05, mew=None, zorder=-1)
            ax.plot(times[times.argsort()], map_model[times.argsort()] * ppm,
                    color='pink', lw=3, alpha=0.05, zorder=-1)

    plt.show()

    return ax


def plot_raw_light_curve(planet, aper_width, aper_height,
                         lw=3, t0_base=0, ax=None):
    ppm = 1e6

    idx_fwd = planet.idx_fwd
    idx_rev = planet.idx_rev
    times = planet.times - t0_base

    aper_column = f'aperture_sum_{aper_width}x{aper_height}'

    if ax is None:
        fig, ax = plt.subplots()

    ax.clear()

    phots = planet.normed_photometry_df[aper_column]
    uncs = planet.normed_uncertainty_df[aper_column]

    phots_med = np.median(phots)
    phots_med_sub = phots - phots_med
    ax.errorbar(times[idx_fwd],
                phots_med_sub[idx_fwd] * ppm,
                uncs[idx_fwd] * ppm, label='Forward Scan',
                fmt='o', color='C0', ms=10, zorder=1, mew=0)

    ax.errorbar(times[idx_rev],
                phots_med_sub[idx_rev] * ppm,
                uncs[idx_rev] * ppm, label='Reverse Scan',
                fmt='o', color='C1', ms=10, zorder=1, mew=0)

    ax.axhline(0.0, ls='--', color='darkgrey', lw=lw,
               zorder=0, label='Null Hypothesis')

    ax.set_xlabel('Time [Days from Eclipse]', fontsize=20)
    ax.set_ylabel('Normalized Flux [ppm]', fontsize=20)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(15)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)

    ax.legend(loc=0, fontsize=15)
    plt.show()

    return ax


def plot_lightcurve(planet, aper_width, aper_height,
                    size=50, t0_base=None, include_orbits=True):

    flux_id = get_flux_idx_from_df(planet, aper_width, aper_height)
    fluxes = planet.photometry_df[f'aperture_sum_{flux_id}']
    fluxes = fluxes / np.median(fluxes)

    min_flux, max_flux = np.percentile(fluxes, [0.1, 99.9])
    # info_message(f'Fluxes Scatter: {np.std(fluxes)*1e6:0.0f} ppm')
    title = 'Flux vs Time [days]'
    xlabel = 'Time Since Mean Time [days]'
    ylabel = 'Flux [ppm]'

    ax = uniform_scatter_plot(planet,
                              planet.times,
                              fluxes,
                              arr1_center=t0_base,
                              title=title,
                              xlabel=xlabel,
                              ylabel=ylabel,
                              size=size,
                              include_orbits=include_orbits,
                              ax=ax)

    plt.ylim(min_flux, max_flux)

    return ax


def plot_apertures(image, aperture,
                   inner_annular=None, outer_annular=None):
    norm = simple_norm(image, 'sqrt', percent=99)

    plt.imshow(image, norm=norm)

    aperture.plot(color='white', lw=2)

    if inner_annular is not None:
        inner_annular.plot(color='red', lw=2)

    if outer_annular is not None:
        outer_annular.plot(color='violet', lw=2)

    plt.axis('off')
    plt.tight_layout()
    plt.waitforbuttonpress()


def plot_trace_peaks(planet, image_id):
    image = planet.image_stack[image_id]
    image_shape = image.shape
    gauss_means = np.zeros(image_shape[1])
    for key, val in planet.center_traces[image_id].items():
        gauss_means[key] = val['results'].mean.value

    norm = simple_norm(image, 'sqrt', percent=99)
    plt.imshow(image, norm=norm)
    plt.plot(np.arange(image_shape[1]), gauss_means,
             'o', color='C1', ms=1)

    plt.xlim(0, image_shape[1])
    plt.ylim(0, image_shape[0])
    plt.tight_layout()
    plt.axis('off')
    plt.waitforbuttonpress()


def plot_errorbars(planet, id_=None):

    id_ = list(planet.fluxes['apertures'].keys())[0] if id_ is None else id_

    fluxes_ = planet.fluxes['fluxes'][id_]
    fwd_fluxes_ = fluxes_[planet.idx_fwd]
    rev_fluxes_ = fluxes_[planet.idx_rev]

    med_flux = np.median(fluxes_)
    fwd_scatter = np.std(fwd_fluxes_ / np.median(fwd_fluxes_)) * 1e6
    rev_scatter = np.std(rev_fluxes_ / np.median(rev_fluxes_)) * 1e6

    fwd_annotate = f'Forward Scatter: {fwd_scatter:0.0f}ppm'
    rev_annotate = f'Reverse Scatter: {rev_scatter:0.0f}ppm'
    info_message(fwd_annotate)
    info_message(rev_annotate)

    fluxes_normed = fluxes_ / med_flux
    errors_normed = np.sqrt(fluxes_) / med_flux

    plt.errorbar(planet.times[planet.idx_fwd],
                 fluxes_normed[planet.idx_fwd],
                 errors_normed[planet.idx_fwd],
                 fmt='o', color='C0')

    plt.errorbar(planet.times[planet.idx_rev],
                 fluxes_normed[planet.idx_rev],
                 errors_normed[planet.idx_rev],
                 fmt='o', color='C3')

    plt.axhline(1.0, ls='--', color='C2')
    plt.title('WASP-43 HST/UVIS Observation Initial Draft Photometry')
    plt.xlabel('Time [MJD]')
    plt.ylabel('Normalized Flux')

    plt.annotate(fwd_annotate,
                 (0, 0),
                 xycoords='axes fraction',
                 xytext=(5, 5),
                 textcoords='offset points',
                 ha='left',
                 va='bottom',
                 fontsize=12,
                 color='C0',
                 weight='bold')

    plt.annotate(rev_annotate,
                 (0, 0.025),
                 xycoords='axes fraction',
                 xytext=(5, 5),
                 textcoords='offset points',
                 ha='left',
                 va='bottom',
                 fontsize=12,
                 color='C3',
                 weight='bold')

    plt.tight_layout()
    plt.show()


def convert_photometry_df_columns_standard(existing_phot_df, trace_lengths):
    # existing_phot_df = planet_savedict_backup_221019['photometry_df']
    # existing_phot_df = planet.photometry_df.copy()

    aperture_columns = [colname
                        for colname in existing_phot_df.columns
                        if 'aperture_sum_' in colname]

    aperwidth_columns = [colname
                         for colname in existing_phot_df.columns
                         if 'aper_width_' in colname]

    aperheight_columns = [colname
                          for colname in existing_phot_df.columns
                          if 'aper_height_' in colname]

    # Why do i need to add +0.1 ??
    med_trace_length_ = np.median(trace_lengths) + 0.1

    mesh_widths_ = []
    for colname in aperwidth_columns:
        aper_width_ = np.int32(existing_phot_df[colname] - med_trace_length_)
        mesh_widths_.append(np.median(aper_width_))

    mesh_heights_ = []
    for colname in aperheight_columns:
        aper_height_ = np.int32(existing_phot_df[colname])
        mesh_heights_.append(np.median(aper_height_))

    mesh_widths_ = np.array(mesh_widths_)
    mesh_heights_ = np.array(mesh_heights_)

    photometry_df = pd.DataFrame([])
    for colname in aperture_columns:
        aper_id = int(colname.replace('aperture_sum_', ''))
        aper_width_ = mesh_widths_[aper_id].astype(int)
        aper_height_ = mesh_heights_[aper_id].astype(int)
        newname = f'aperture_sum_{aper_width_}x{aper_height_}'

        photometry_df[newname] = existing_phot_df[colname]

    photometry_df['xcenter'] = existing_phot_df['xcenter']
    photometry_df['ycenter'] = existing_phot_df['ycenter']

    return photometry_df


def plot_2D_stddev(planet, signal_max=235):
    ppm = 1e6
    photometry_df = planet.photometry_df

    phot_columns = [colname
                    for colname in photometry_df.columns
                    if 'aperture_sum' in colname]

    new_standard = True
    for colname in phot_columns:
        if 'x' not in colname:
            new_standard = False

    if not new_standard:
        warning_message(
            'Converting Photomery Format from '
            '`aperture_sum_0` to aperture_sum_{width}x{heights}\n'
            'Run `convert_photometry_df_columns_standard` to make '
            'this permanent')

        photometry_df = convert_photometry_df_columns_standard(
            photometry_df, planet.trace_lengths)
        phot_columns = [colname
                        for colname in photometry_df.columns
                        if 'aperture_sum' in colname]

    info_message(f'\n{photometry_df}')

    aper_heights = np.zeros(len(phot_columns), dtype=int)
    aper_widths = np.zeros(len(phot_columns), dtype=int)

    for k, colname in enumerate(phot_columns):
        example = 'aperture_sum_{width}x{heights}'
        assert('x' in colname), \
            f'You need to update your column names as {example}'

        # aperture_sum_{aper_width_}x{aper_height_}
        vals = colname.split('_')[-1]
        aper_widths[k], aper_heights[k] = np.int32(vals.split('x'))

    aper_widths = np.unique(aper_widths)
    aper_heights = np.unique(aper_heights)

    meshgrid = np.meshgrid(aper_widths, aper_heights)

    phot_vals = photometry_df[phot_columns].values
    lc_std_rev = phot_vals[planet.idx_rev].std(axis=0)
    lc_std_fwd = phot_vals[planet.idx_fwd].std(axis=0)

    lc_med_rev = np.median(phot_vals[planet.idx_rev], axis=0)
    lc_med_fwd = np.median(phot_vals[planet.idx_rev], axis=0)

    lc_std = np.mean([lc_std_rev, lc_std_fwd], axis=0)
    lc_med = np.mean([lc_med_rev, lc_med_fwd], axis=0)

    signal = lc_std / lc_med * ppm
    good = signal < signal_max  # ppm
    sig_min, sig_max = np.percentile(signal[good], [0.1, 99.9])

    max_widths = np.max(meshgrid[0].ravel()[good])
    min_widths = np.min(meshgrid[0].ravel()[good])

    max_height = np.max(meshgrid[1].ravel()[good])
    min_height = np.min(meshgrid[1].ravel()[good])

    idx_best = signal.argmin()
    width_best = meshgrid[0].ravel()[idx_best]
    height_best = meshgrid[1].ravel()[idx_best]
    best_ppm = signal[idx_best]

    plt.scatter(meshgrid[0].ravel()[good], meshgrid[1].ravel()[
                good], c=(lc_std / lc_med)[good] * ppm, marker='s', s=1260)

    cbar = plt.colorbar()
    # cbar.ax.set_yticklabels()
    cbar.set_label('Raw Light Curve Std-Dev [ppm]', rotation=270)
    cbar.ax.get_yaxis().labelpad = 30

    plt.plot(width_best, height_best, 'o', color='C1', ms=10)
    plt.annotate(f'{best_ppm:.0f} ppm [{width_best}x{height_best}]',
                 (width_best + 1, height_best + 1),
                 # xycoords='axes fraction',
                 xytext=(width_best + 1, height_best + 1),
                 # textcoords='offset points',
                 ha='left',
                 va='bottom',
                 fontsize=12,
                 color='C1',
                 weight='bold')

    plt.xlabel('Aperture Width Outside Trace', fontsize=20)
    plt.ylabel('Aperture Height Above Trace', fontsize=20)

    plt.xlim(min_widths - 2, max_widths + 2)
    plt.ylim(min_height - 5, max_height + 5)

    plt.title(
        'Raw Lightcurve Normalized Std-Dev over Height x Width of Aperture', fontsize=20)
