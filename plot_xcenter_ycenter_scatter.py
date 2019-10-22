from tqdm import tqdm


def info_message(message, end='\n'):
    print(f'[INFO] {message}', end=end)


def plot_aperture_edges_with_angle(wasp43, img_id=42):
    plt.close('all')

    y_center = wasp43.trace_ycenters[img_id]
    x_left = wasp43.x_left
    x_right = wasp43.x_right
    trace_width = x_right - x_left

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


def plot_center_position_vs_scan_and_orbit(wasp43):
    # Came from initial eclipse fitting, isolation where lc < median
    # idx_eclipse = np.array([37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    #                         50, 51, 52, 53, 54, 55])
    plt.clf()
    plt.scatter(wasp43.trace_xcenters[wasp43.idx_fwd],
                wasp43.trace_ycenters[wasp43.idx_fwd],
                color='C0', label='Forward Scans')
    plt.scatter(wasp43.trace_xcenters[wasp43.idx_rev],
                wasp43.trace_ycenters[wasp43.idx_rev],
                color='C1', label='Reverse Scans')
    # plt.plot(wasp43.trace_xcenters[idx_eclipse],
    #          wasp43.trace_ycenters[idx_eclipse], 'o',
    #          ms=10, mew=2, color='none', mec='black',
    #          label='During Eclipse')

    # # By hand values from looking at the light curve
    # plt.plot(wasp43.trace_xcenters[:18],
    #          wasp43.trace_ycenters[:18], 'o',
    #          ms=15, mew=2, color='none', mec='red',
    #          label='First Orbit')

    # plt.plot(wasp43.trace_xcenters[18:38],
    #          wasp43.trace_ycenters[18:38], 'o',
    #          ms=20, mew=2, color='none', mec='green',
    #          label='Second Orbit')

    # plt.plot(wasp43.trace_xcenters[56:],
    #          wasp43.trace_ycenters[56:], 'o',
    #          ms=25, mew=2, color='none', mec='purple',
    #          label='Fourth Orbit')

    plt.title(
        'Center Positions of the Trace in Forward and Reverse Scanning',
        fontsize=20)
    plt.xlabel('X-Center [pixels]', fontsize=20)
    plt.ylabel('Y-Center [pixels]', fontsize=20)
    plt.legend(loc=0, fontsize=20)


def plot_ycenter_vs_time(wasp43):
    # Came from initial eclipse fitting, isolation where lc < median
    # idx_eclipse = np.array([37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    #                         50, 51, 52, 53, 54, 55])
    plt.clf()
    plt.scatter(wasp43.times[wasp43.idx_fwd] - wasp43.times.mean(),
                wasp43.trace_ycenters[wasp43.idx_fwd],
                color='C0', label='Forward Scans')
    plt.scatter(wasp43.times[wasp43.idx_rev] - wasp43.times.mean(),
                wasp43.trace_ycenters[wasp43.idx_rev],
                color='C1', label='Reverse Scans')
    # plt.plot(wasp43.times[idx_eclipse] - wasp43.times.mean(),
    #          wasp43.trace_ycenters[idx_eclipse], 'o',
    #          ms=10, mew=2, color='none', mec='black',
    #          label='During Eclipse')

    # # By hand values from looking at the light curve
    # plt.plot(wasp43.times[:18] - wasp43.times.mean(),
    #          wasp43.trace_ycenters[:18], 'o',
    #          ms=15, mew=2, color='none', mec='red',
    #          label='First Orbit')

    # plt.plot(wasp43.times[18:38] - wasp43.times.mean(),
    #          wasp43.trace_ycenters[18:38], 'o',
    #          ms=20, mew=2, color='none', mec='green',
    #          label='Second Orbit')

    # plt.plot(wasp43.times[56:] - wasp43.times.mean(),
    #          wasp43.trace_ycenters[56:], 'o',
    #          ms=25, mew=2, color='none', mec='purple',
    #          label='Fourth Orbit')

    plt.title(
        'Center Positions vs Time of the Trace',
        fontsize=20)
    plt.xlabel('time from Mean Time [days]', fontsize=20)
    plt.ylabel('Y-Center [pixels]', fontsize=20)
    plt.legend(loc=0, fontsize=20)


def plot_xcenter_vs_time(wasp43):
    # Came from initial eclipse fitting, isolation where lc < median
    # idx_eclipse = np.array([37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    #                         50, 51, 52, 53, 54, 55])
    plt.clf()
    plt.scatter(wasp43.times[wasp43.idx_fwd] - wasp43.times.mean(),
                wasp43.trace_xcenters[wasp43.idx_fwd],
                color='C0', label='Forward Scans')
    plt.scatter(wasp43.times[wasp43.idx_rev] - wasp43.times.mean(),
                wasp43.trace_xcenters[wasp43.idx_rev],
                color='C1', label='Reverse Scans')
    # plt.plot(wasp43.times[idx_eclipse] - wasp43.times.mean(),
    #          wasp43.trace_xcenters[idx_eclipse], 'o',
    #          ms=10, mew=2, color='none', mec='black',
    #          label='During Eclipse')

    # # By hand values from looking at the light curve
    # plt.plot(wasp43.times[:18] - wasp43.times.mean(),
    #          wasp43.trace_xcenters[:18], 'o',
    #          ms=15, mew=2, color='none', mec='red',
    #          label='First Orbit')

    # plt.plot(wasp43.times[18:38] - wasp43.times.mean(),
    #          wasp43.trace_xcenters[18:38], 'o',
    #          ms=20, mew=2, color='none', mec='green',
    #          label='Second Orbit')

    # plt.plot(wasp43.times[56:] - wasp43.times.mean(),
    #          wasp43.trace_xcenters[56:], 'o',
    #          ms=25, mew=2, color='none', mec='purple',
    #          label='Fourth Orbit')

    plt.title(
        'Center Positions vs Time of the Trace',
        fontsize=20)
    plt.xlabel('time from Mean Time [days]', fontsize=20)
    plt.ylabel('X-Center [pixels]', fontsize=20)
    plt.legend(loc=0, fontsize=20)


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


def plot_ycenter_vs_flux(wasp43, aper_width, aper_height):
    flux_id = get_flux_idx_from_df(wasp43, aper_width, aper_height)
    fluxes = wasp43.photometry_df[f'aperture_sum_{flux_id}']
    fluxes = fluxes / np.median(fluxes)

    min_flux, max_flux = np.percentile(fluxes, [0.1, 99.9])
    # info_message(f'Fluxes Scatter: {np.std(fluxes)*1e6:0.0f} ppm')
    y_centers = wasp43.trace_ycenters
    plt.clf()
    plt.scatter(y_centers[wasp43.idx_fwd] - y_centers.mean(),
                fluxes[wasp43.idx_fwd],
                color='C0', label='Forward Scans')

    plt.scatter(y_centers[wasp43.idx_rev] - y_centers.mean(),
                fluxes[wasp43.idx_rev],
                color='C1', label='Reverse Scans')

    plt.title('Flux vs Y-Center Positions', fontsize=20)
    plt.xlabel('Y-Center [pixels]', fontsize=20)
    plt.ylabel('Flux [ppm]', fontsize=20)
    plt.ylim(min_flux, max_flux)
    plt.legend(loc=0, fontsize=20)


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


def plot_xcenter_vs_flux(wasp43, aper_width, aper_height):
    flux_id = get_flux_idx_from_df(wasp43, aper_width, aper_height)
    fluxes = wasp43.photometry_df[f'aperture_sum_{flux_id}']
    fluxes = fluxes / np.median(fluxes)

    min_flux, max_flux = np.percentile(fluxes, [0.1, 99.9])
    # info_message(f'Fluxes Scatter: {np.std(fluxes)*1e6:0.0f} ppm')
    x_centers = wasp43.trace_xcenters
    plt.clf()
    plt.scatter(x_centers[wasp43.idx_fwd] - x_centers.mean(),
                fluxes[wasp43.idx_fwd],
                color='C0', label='Forward Scans')

    plt.scatter(x_centers[wasp43.idx_rev] - x_centers.mean(),
                fluxes[wasp43.idx_rev],
                color='C1', label='Reverse Scans')

    plt.title('Flux vs X-Center Positions', fontsize=20)
    plt.xlabel('X-Center [pixels]', fontsize=20)
    plt.ylabel('Flux [ppm]', fontsize=20)
    plt.ylim(min_flux, max_flux)
    plt.legend(loc=0, fontsize=20)
