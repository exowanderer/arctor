# https://stackoverflow.com/questions/45786714/custom-marker-edge-style-in-manual-legend

from matplotlib import pyplot as plt
from photutils import RectangularAperture
from tqdm import tqdm
from statsmodels.robust import scale as sc
import numpy as np


def info_message(message, end='\n'):
    print(f'[INFO] {message}', end=end)


def get_flux_idx_from_df(wasp43, aper_width, aper_height):
    # There *must* be a faster way!
    aperwidth_columns = [colname
                         for colname in wasp43.photometry_df.columns
                         if 'aper_width' in colname]

    aperheight_columns = [colname
                          for colname in wasp43.photometry_df.columns
                          if 'aper_height' in colname]

    trace_length = np.median(wasp43.trace_lengths) - 0.1

    aperwidths_df = (wasp43.photometry_df[aperwidth_columns] - trace_length)
    aperwidths_df = aperwidths_df.astype(int)

    aperheight_df = wasp43.photometry_df[aperheight_columns].astype(int)
    aperwidth_flag = aperwidths_df.values[0] == aper_width
    aperheight_flag = aperheight_df.values[0] == aper_height

    return np.where(aperwidth_flag * aperheight_flag)[0][0]


def print_flux_stddev(wasp43, aper_width, aper_height):
    # There *must* be a faster way!
    flux_id = get_flux_idx_from_df(wasp43, aper_width, aper_height)
    fluxes = wasp43.photometry_df[f'aperture_sum_{flux_id}']
    fluxes = fluxes / np.median(fluxes)

    info_message(f'{aper_width}x{aper_height}: {np.std(fluxes)*1e6:0.0f} ppm')


def find_flux_stddev(wasp43, flux_std, aper_widths, aper_heights):
    # There *must* be a faster way!
    for aper_width in tqdm(aper_widths):
        for aper_height in tqdm(aper_heights):
            flux_id = get_flux_idx_from_df(wasp43, aper_width, aper_height)
            fluxes = wasp43.photometry_df[f'aperture_sum_{flux_id}']
            fluxes = fluxes / np.median(fluxes)

            if np.std(fluxes) * 1e6 < flux_std:
                info_message(f'{aper_width}x{aper_height}: '
                             f'{np.std(fluxes)*1e6:0.0f} ppm')


def plot_aperture_edges_with_angle(wasp43, img_id=42):
    image = wasp43.image_stack[img_id]
    y_center = wasp43.trace_ycenters[img_id]
    x_left = wasp43.x_left
    x_right = wasp43.x_right
    trace_width = x_right - x_left

    positions = np.transpose([wasp43.trace_xcenters, wasp43.trace_ycenters])
    thetas = wasp43.trace_angles

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


def uniform_scatter_plot(wasp43, arr1, arr2,
                         arr1_center=0,
                         title='', xlabel='', ylabel=''):
    fig, axs = plt.subplots()
    time_sort = np.argsort(wasp43.times)

    idx_orbit1 = np.arange(18)  # by eye
    idx_orbit2 = np.arange(18, 38)  # by eye
    idx_eclipse = np.arange(38, 56)  # by eye
    idx_orbit4 = np.arange(56, len(arr1))  # by eye

    plt.scatter(arr1[wasp43.idx_fwd] - arr1_center, arr2[wasp43.idx_fwd],
                color='C0', label='Forward Scans')
    plt.scatter(arr1[wasp43.idx_rev] - arr1_center, arr2[wasp43.idx_rev],
                color='C1', label='Reverse Scans')

    # By hand values from looking at the light curve
    plt.plot(arr1[time_sort][idx_orbit1] - arr1_center,
             arr2[time_sort][idx_orbit1], 'o',
             ms=20, mew=2, color='none', mec='black',
             label='First Orbit', zorder=0, marker=u'$\u25CC$')

    plt.plot(arr1[time_sort][idx_orbit2] - arr1_center,
             arr2[time_sort][idx_orbit2], 'o',
             ms=20, mew=2, color='none', mec='lightgreen',
             label='Second Orbit', zorder=0)  # , marker=u'$\u25CC$')

    plt.plot(arr1[time_sort][idx_eclipse] - arr1_center,
             arr2[time_sort][idx_eclipse], 'o',
             ms=20, mew=2, color='none', mec='pink',
             label='Third Orbit (eclipse)', zorder=0)  # , marker=u'$\u25CC$')

    plt.plot(arr1[time_sort][idx_orbit4] - arr1_center,
             arr2[time_sort][idx_orbit4], 'o',
             ms=20, mew=2, color='none', mec='indigo',
             label='Fourth Orbit', zorder=0)  # , marker=u'$\u25CC$')

    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.legend(loc=0, fontsize=20)


def plot_center_position_vs_scan_and_orbit(wasp43):
    title = 'Center Positions of the Trace in Forward and Reverse Scanning'
    xlabel = 'X-Center [pixels]'
    ylabel = 'Y-Center [pixels]'

    uniform_scatter_plot(wasp43,
                         wasp43.trace_ycenters,
                         wasp43.trace_xcenters,
                         arr1_center=0,
                         title=title,
                         xlabel=xlabel,
                         ylabel=ylabel)


def plot_ycenter_vs_time(wasp43):
    title = 'Y-Positions vs Time of the Trace'
    xlabel = 'Time Since Mean Time [days]'
    ylabel = 'Y-Center [pixels]'

    uniform_scatter_plot(wasp43,
                         wasp43.times,
                         wasp43.trace_ycenters,
                         arr1_center=wasp43.times.mean(),
                         title=title,
                         xlabel=xlabel,
                         ylabel=ylabel)


def plot_xcenter_vs_time(wasp43):
    title = 'X-Positions vs Time of the Trace'
    xlabel = 'Time Since Mean Time [days]'
    ylabel = 'X-Center [pixels]'

    uniform_scatter_plot(wasp43,
                         wasp43.times,
                         wasp43.trace_xcenters,
                         arr1_center=wasp43.times.mean(),
                         title=title,
                         xlabel=xlabel,
                         ylabel=ylabel)


def plot_ycenter_vs_flux(wasp43, aper_width, aper_height):
    flux_id = get_flux_idx_from_df(wasp43, aper_width, aper_height)
    fluxes = wasp43.photometry_df[f'aperture_sum_{flux_id}']
    fluxes = fluxes / np.median(fluxes)

    min_flux, max_flux = np.percentile(fluxes, [0.1, 99.9])
    # info_message(f'Fluxes Scatter: {np.std(fluxes)*1e6:0.0f} ppm')
    title = 'Flux vs Y-Center Positions'
    xlabel = 'Y-Center [pixels]'
    ylabel = 'Flux [ppm]'

    uniform_scatter_plot(wasp43,
                         wasp43.trace_ycenters,
                         fluxes,
                         arr1_center=0,
                         title=title,
                         xlabel=xlabel,
                         ylabel=ylabel)

    plt.ylim(min_flux, max_flux)


def plot_xcenter_vs_flux(wasp43, aper_width, aper_height):
    flux_id = get_flux_idx_from_df(wasp43, aper_width, aper_height)
    fluxes = wasp43.photometry_df[f'aperture_sum_{flux_id}']
    fluxes = fluxes / np.median(fluxes)

    min_flux, max_flux = np.percentile(fluxes, [0.1, 99.9])

    title = 'Flux vs X-Center Positions'
    xlabel = 'X-Center [pixels]'
    ylabel = 'Flux [ppm]'

    uniform_scatter_plot(wasp43,
                         wasp43.trace_xcenters,
                         fluxes,
                         arr1_center=0,
                         title=title,
                         xlabel=xlabel,
                         ylabel=ylabel)

    plt.ylim(min_flux, max_flux)


def plot_lightcurve(wasp43, aper_width, aper_height):

    flux_id = get_flux_idx_from_df(wasp43, aper_width, aper_height)
    fluxes = wasp43.photometry_df[f'aperture_sum_{flux_id}']
    fluxes = fluxes / np.median(fluxes)

    min_flux, max_flux = np.percentile(fluxes, [0.1, 99.9])
    # info_message(f'Fluxes Scatter: {np.std(fluxes)*1e6:0.0f} ppm')
    title = 'Flux vs Time [days]'
    xlabel = 'Time Since Mean Time [days]'
    ylabel = 'Flux [ppm]'

    uniform_scatter_plot(wasp43,
                         wasp43.times,
                         fluxes,
                         arr1_center=wasp43.times.mean(),
                         title=title,
                         xlabel=xlabel,
                         ylabel=ylabel)

    plt.ylim(min_flux, max_flux)


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


def plot_trace_peaks(wasp43, image):
    gauss_means = np.zeros(image_shape[1])
    for key, val in wasp43.center_traces.items():
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


def plot_errorbars(wasp43, id_=None):

    id_ = list(wasp43.fluxes['apertures'].keys())[0] if id_ is None else id_

    fluxes_ = wasp43.fluxes['fluxes'][id_]
    fwd_fluxes_ = fluxes_[wasp43.idx_fwd]
    rev_fluxes_ = fluxes_[wasp43.idx_rev]

    med_flux = np.median(fluxes_)
    fwd_scatter = np.std(fwd_fluxes_ / np.median(fwd_fluxes_)) * 1e6
    rev_scatter = np.std(rev_fluxes_ / np.median(rev_fluxes_)) * 1e6

    fwd_annotate = f'Forward Scatter: {fwd_scatter:0.0f}ppm'
    rev_annotate = f'Reverse Scatter: {rev_scatter:0.0f}ppm'
    info_message(fwd_annotate)
    info_message(rev_annotate)

    fluxes_normed = fluxes_ / med_flux
    errors_normed = np.sqrt(fluxes_) / med_flux

    plt.errorbar(wasp43.times[wasp43.idx_fwd],
                 fluxes_normed[wasp43.idx_fwd],
                 errors_normed[wasp43.idx_fwd],
                 fmt='o', color='C0')

    plt.errorbar(wasp43.times[wasp43.idx_rev],
                 fluxes_normed[wasp43.idx_rev],
                 errors_normed[wasp43.idx_rev],
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


def plot_2D_stddev(wasp43, aper_widths, aper_heights, signal_max=500):
    photometry_df = wasp43.photometry_df
    ppm = 1e6
    meshgrid = np.meshgrid(aper_widths, aper_heights)

    # trace_width = wasp43.x_right - wasp43.x_left
    # area_grid = (meshgrid[0] + trace_width) * meshgrid[0]
    # sky_bg = np.array([skbg * area_grid.flatten() for skbg in wasp43.sky_bgs])
    # - sky_bg
    phot_columns = [colname
                    for colname in photometry_df.columns
                    if 'aperture_sum' in colname]

    phot_vals = photometry_df[phot_columns].values
    med_phots = np.median(phot_vals, axis=0)
    phot_vals = phot_vals / med_phots

    # lc_std_rev = phot_vals[wasp43.idx_rev].std(axis=0)
    # lc_std_fwd = phot_vals[wasp43.idx_fwd].std(axis=0)

    mad2std = 1.4815034297  # integration of ERF into M.A.D.
    lc_std_rev = sc.mad(phot_vals[wasp43.idx_rev]) * mad2std
    lc_std_fwd = sc.mad(phot_vals[wasp43.idx_fwd]) * mad2std

    lc_med_rev = np.median(phot_vals[wasp43.idx_rev], axis=0)
    lc_med_fwd = np.median(phot_vals[wasp43.idx_rev], axis=0)

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
    # plt.tight_layout()
