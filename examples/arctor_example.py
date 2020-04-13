import joblib
import numpy as np
import os

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from arctor import Arctor, create_raw_lc_stddev
from arctor import debug_message, warning_message, info_message


from arctor.plotting import (plot_aperture_edges_with_angle,
                             plot_lightcurve,
                             plot_lightcurve_orbit,
                             plot_xcenter_vs_time,
                             plot_ycenter_vs_time,
                             plot_xcenter_vs_flux,
                             plot_ycenter_vs_flux,
                             plot_center_position_vs_scan_and_orbit,
                             plot_2D_stddev)

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument('--planet_name', type=str, default=None)
    parser.add_argument('--file_type', type=str, default='flt.fits')
    parser.add_argument('--save_now', action='store_true')
    parser.add_argument('--fit_model_flag', action='store_true')
    parser.add_argument('--plot_verbose', action='store_true')

    clargs = parser.parse_args()

    planet_name = clargs.planet_name
    if clargs.planet_name is None:
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

    joblib_filename = (f'{planet_name}'
                       '_savedict_1683_1-200x1-200_x5.joblib.save')

    # csv_filename = f'{working_dir}/{csv_filename}'
    joblib_filename = f'{save_dir}/{joblib_filename}'

    assert(os.path.exists(joblib_filename)),\
        f'File not found:{joblib_filename}'

    planet = Arctor(
        planet_name=planet_name,
        data_dir=data_dir,
        working_dir=working_dir,
        file_type=file_type)

    # This forces the fits files to be opened in time order
    #   Very useful the *first* time we run the example
    # planet.rename_fits_files_by_time()

    if os.path.exists(joblib_filename):
        info_message('Loading Data from Save File')
        planet.load_dict(joblib_filename)
    else:
        info_message('Loading New Data Object')
        planet.load_data()  # sort_by_time=sort_by_time)

    if not hasattr(planet, 'gaussian_centers'):
        planet.clean_cosmic_rays()
        planet.calibration_trace_location(oversample=100)
        planet.identify_trace_direction()
        planet.simple_phots()
        planet.center_all_traces(
            stddev=2, notit_verbose=False, idx_buffer=10)
        # planet.fit_trace_slopes(stddev=2, notit_verbose=False)
        planet.compute_trace_slopes(
            stddev=2, notit_verbose=False, x_offset=100)
        planet.compute_sky_background(
            subpixels=32, positions=None, inner_width=75, outer_width=150,
            inner_height=225, outer_height=350, thetas=None,
            notit_verbose=False, done_it=False)
        planet.compute_columnwise_sky_background(inner_height=150, edge=10)

    # Set up the list of aperture widths and heights to search
    min_aper_width = 1
    max_aper_width = 200
    min_aper_height = 1
    max_aper_height = 200

    aper_widths = np.arange(min_aper_width, max_aper_width + 2, 5)
    aper_heights = np.arange(min_aper_height, max_aper_height + 2, 5)

    if 'yes' in input('Are you sure? (yes/no) ').lower()[:3]:
        planet.do_multi_phot(aper_widths, aper_heights)
    else:
        print('[INFO] Skipping `do_multi_phot` until you are ready')

    aper_width = 51
    aper_height = 51
    ax = plot_lightcurve(planet, aper_width, aper_height,
                         t0_base=planet.times.min(), ax=ax,
                         include_orbits=True,
                         highlight_outliers=True)

    plot_2D_stddev(planet, signal_max=235)  # , aper_widths, aper_heights
    plot_center_position_vs_scan_and_orbit(planet)

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

    if 'y' in input('Are you sure? ').lower()[:1]:
        planet.do_multi_phot(fine_aper_widths, fine_aper_heights)
    else:
        print('[INFO] Skipping `do_multi_phot` until you are ready')

    if clargs.plot_verbose:
        planet.plot_errorbars()

    if clargs.fit_model_flag:
        res = planet.do_fit(times, fluxes, errors,
                            init_params=[], static_params={})

    # csv_filename = f'{clargs.planet_name}_photometry.csv'
    # joblib_filename =
    # f'{planet_name}_savedict_not_finished_with_images.joblib.save'
    # f'{planet_name}_savedict_no_phots_with_images.joblib.save'

    if clargs.save_now:
        # joblib_filename = f'{planet_name}_savedict_NNN_NNNxNNN.joblib.save'
        joblib_filename = f'{planet_name}_savedict_1683_1-200x1-200_x5.joblib.save'

        # csv_filename = f'{working_dir}/{csv_filename}'
        joblib_filename = f'{save_dir}/{joblib_filename}'

        # planet.save_text_file(csv_filename)
        planet.save_dict(joblib_filename)

    aper_width = 136  # FINDME: Check these
    aper_height = 111  # FINDME: Check these
    ax = plot_lightcurve(planet, aper_width, aper_height,
                         t0_base=planet.times.min(), ax=ax,
                         include_orbits=True,
                         highlight_outliers=True)    plot_ycenter_vs_flux(planet, aper_width, aper_height)

    plot_xcenter_vs_flux(planet, aper_width, aper_height, ax=ax,
                         include_orbits=True,
                         highlight_outliers=True)

    plot_ycenter_vs_flux(planet, aper_width, aper_height, ax=ax,
                         include_orbits=True,
                         highlight_outliers=True)
    plot_ycenter_vs_time(planet, aper_width, aper_height, ax=ax,
                         include_orbits=True,
                         highlight_outliers=True)
    plot_xcenter_vs_time(planet, aper_width, aper_height, ax=ax,
                         include_orbits=True,
                         highlight_outliers=True)

    plot_aperture_edges_with_angle(planet, img_id=42)
    plot_center_position_vs_scan_and_orbit(planet)
    plot_2D_stddev(planet.photometry_df,
                   aper_widths,
                   aper_heights,
                   signal_max=230)
ax1, ax2 = None, None
ax1 = plot_trace_over_time(
    planet, ax=ax1, adjust_xloc=False, metric=np.std, delta_y=10)
ax2 = plot_trace_over_time(
    planet, ax=ax2, adjust_xloc=True, metric=np.std, delta_y=10)
