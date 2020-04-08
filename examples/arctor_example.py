import joblib
import numpy as np
import os

from arctor import Arctor, instantiate_arctor, create_raw_lc_stddev
from arctor import debug_message, warning_message, info_message

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument('--planet_name', type=str, default=None)
    parser.add_argument('--file_type', type=str, default='flt.fits')
    parser.add_argument('--save_now', action='store_true')
    parser.add_argument('--fit_model_flag', action='store_true')
    parser.add_argument('--plot_verbose', action='store_true')

    clargs = parser.parse_args()

    # clargs.plot_verbose = True
    # clargs.save_now = True

    # if clargs.planet_name is None:
    #     planet_name = 'WASP43'
    # else:
    planet_name = clargs.planet_name
    planet_name = 'HD80606'  # because I am unimaginative
    assert(planet_name is not None), \
        "Must provide planet name as --planet_name or -pn"

    file_type = clargs.file_type

    # HOME = os.environ['HOME']
    base_dir = os.path.join('/', 'bigData', 'Research', 'Planets', planet_name)
    data_dir = os.path.join(base_dir, 'data', 'UVIS', 'HST', 'FLTs')
    working_dir = os.path.join(base_dir, 'github_analysis')
    save_dir = os.path.join(base_dir, 'savefiles')

    assert(os.path.exists(base_dir)), (
        '[WARNING] Please either edit the `base_dir` '
        'or configure the directory structure as above')
    assert(os.path.exists(data_dir)), (
        '[WARNING] Please either edit the `data_dir` '
        'or configure the directory structure as above')

    if not os.path.exists(working_dir):
        info_message(f'Creating {working_dir}')
        os.mkdir(working_dir)
    if not os.path.exists(save_dir):
        info_message(f'Creating {save_dir}')
        os.mkdir(save_dir)

    planet = instantiate_arctor(planet_name, data_dir, working_dir, file_type)

    if not hasattr(planet, 'gaussian_centers'):
        planet.clean_cosmic_rays()
        planet.calibration_trace_location()
        planet.identify_trace_direction()
        planet.simple_phots()
        planet.center_all_traces()
        planet.fit_trace_slopes()
        planet.compute_sky_background(subpixels=32)
        planet.compute_columnwise_sky_background()

    # Set up the list of aperture widths and heights to search
    min_aper_width = 1
    max_aper_width = 100
    min_aper_height = 1
    max_aper_height = 100

    aper_widths = np.arange(min_aper_width, max_aper_width + 2, 5)
    aper_heights = np.arange(min_aper_height, max_aper_height + 2, 5)

    planet.do_multi_phot(aper_widths, aper_heights)

    snr_lightcurves = create_raw_lc_stddev(planet)
    min_snr_colname = planet.photometry_df.columns[snr_lightcurves.argmin()]
    min_snr_col = planet.normed_photometry_df[min_snr_colname]
    temp = min_snr_colname.split('_')[-1].split('x')
    min_snr_aper_width, min_snr_aper_height = np.int32(temp)

    fine_buffer = 10
    fine_aper_widths = np.arange(min_snr_aper_width - fine_buffer,
                                 min_snr_aper_width + fine_buffer)

    fine_aper_heights = np.arange(min_snr_aper_height - fine_buffer,
                                  min_snr_aper_height + fine_buffer)

    planet.do_multi_phot(fine_aper_widths, fine_aper_heights)

    from plotting import plot_2D_stddev, plot_center_position_vs_scan_and_orbit
    plot_2D_stddev(planet, aper_widths, aper_heights, signal_max=235)
    plot_center_position_vs_scan_and_orbit(planet)

    if clargs.plot_verbose:
        planet.plot_errorbars()

    if clargs.fit_model_flag:
        res = planet.do_fit(times, fluxes, errors,
                            init_params=[], static_params={})

    if clargs.save_now:
        # joblib_filename = f'{planet_name}_savedict_206ppm_100x100_finescale_columnwiseSkyBG.save'
        # csv_filename = f'{clargs.planet_name}_photometry.csv'
        joblib_filename = f'{planet_name}_savedict_NNN_NNNxNNN.joblib.save'

        # csv_filename = f'{working_dir}/{csv_filename}'
        joblib_filename = f'{working_dir}/{joblib_filename}'

        # planet.save_text_file(csv_filename)
        planet.save_dict(joblib_filename)

    from plotting import plot_aperture_edges_with_angle, plot_lightcurve
    from plotting import plot_xcenter_vs_time, plot_ycenter_vs_time
    from plotting import plot_xcenter_vs_flux, plot_ycenter_vs_flux
    from plotting import plot_center_position_vs_scan_and_orbit, plot_2D_stddev

    plot_lightcurve(planet, aper_width, aper_height)
    plot_ycenter_vs_flux(planet, aper_width, aper_height)
    plot_xcenter_vs_flux(planet, aper_width, aper_height)
    plot_ycenter_vs_time(planet)
    plot_xcenter_vs_time(planet)
    plot_aperture_edges_with_angle(planet, img_id=42)
    plot_center_position_vs_scan_and_orbit(planet)
    plot_2D_stddev(planet.photometry_df,
                   aper_widths,
                   aper_heights,
                   signal_max=230)
