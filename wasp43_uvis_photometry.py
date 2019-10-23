import joblib
import numpy as np
import os

from HSTUVISTimeSeries import HSTUVISTimeSeries
from HSTUVISTimeSeries import debug_message, warning_message, info_message


def instantiate_wasp43(planet_name, data_dir, working_dir, file_type):
    wasp43 = HSTUVISTimeSeries(
        planet_name=planet_name,
        data_dir=data_dir,
        working_dir=working_dir,
        file_type=file_type)

    joblib_filename = f'{planet_name}_savedict.joblib.save'
    joblib_filename = f'{working_dir}/{joblib_filename}'
    if os.path.exists(joblib_filename):
        info_message('Loading Data from Save File')
        wasp43.load_data(joblib_filename)
    else:
        info_message('Loading New Data Object')
        wasp43.load_data()

    return wasp43


def create_raw_lc_stddev(wasp43):
    ppm = 1e6
    phot_vals = wasp43.photometry_df
    lc_std_rev = phot_vals.iloc[wasp43.idx_rev].std(axis=0)
    lc_std_fwd = phot_vals.iloc[wasp43.idx_fwd].std(axis=0)

    lc_med_rev = np.median(phot_vals.iloc[wasp43.idx_rev], axis=0)
    lc_med_fwd = np.median(phot_vals.iloc[wasp43.idx_rev], axis=0)

    lc_std = np.mean([lc_std_rev, lc_std_fwd], axis=0)
    lc_med = np.mean([lc_med_rev, lc_med_fwd], axis=0)

    return lc_std / lc_med * ppm

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument('--planet_name', type=str, default=None)
    parser.add_argument('--file_type', type=str, default='flt.fits')
    parser.add_argument('--save_now', action='store_true')
    parser.add_argument('--fit_model_flag', action='store_true')
    parser.add_argument('--plot_verbose', action='store_true')

    clargs = parser.parse_args()

    clargs.plot_verbose = True
    clargs.save_now = True
    planet_name = 'WASP43' \
        if clargs.planet_name is None else clargs.planet_name
    file_type = clargs.file_type

    HOME = os.environ['HOME']
    base_dir = os.path.join(HOME, 'Research', 'Planets', 'WASP43')
    data_dir = os.path.join(base_dir, 'data', 'UVIS', 'MAST_2019-07-03T0738')
    data_dir = os.path.join(data_dir, 'HST', 'FLTs')
    working_dir = os.path.join(base_dir, 'github_analysis', 'savefiles')

    wasp43 = instantiate_wasp43(planet_name, data_dir, working_dir, file_type)

    # if not hasattr(wasp43, 'gaussian_centers'):
    wasp43.clean_cosmic_rays()
    wasp43.calibration_trace_location()
    wasp43.identify_trace_direction()
    wasp43.simple_phots()
    wasp43.center_all_traces()
    wasp43.fit_trace_slopes()
    wasp43.compute_sky_background(subpixels=32)
    wasp43.compute_columnwise_sky_background()

    info_message('Loading in Best So Far Save File')
    bestsofar = 'savefiles/WASP43_savedict_backup_221019.joblib.save'
    newest = 'savefiles/WASP43_savedict_218ppm.joblib.save'

    WASP43_savedict_backup_221019 = joblib.load(bestsofar)
    WASP43_savedict_backup_231019 = joblib.load(newest)

    # Set up the list of aperture widths and heights to search
    min_aper_width = 1
    max_aper_width = 100
    min_aper_height = 1
    max_aper_height = 100

    aper_widths = np.arange(min_aper_width, max_aper_width + 2, 5)
    aper_heights = np.arange(min_aper_height, max_aper_height + 2, 5)

    wasp43.do_multi_phot(aper_widths, aper_heights)

    snr_lightcurves = create_raw_lc_stddev(wasp43)
    min_snr_colname = wasp43.photometry_df.columns[snr_lightcurves.argmin()]
    min_snr_col = wasp43.normed_photometry_df[min_snr_colname]
    temp = min_snr_colname.split('_')[-1].split('x')
    min_snr_aper_width, min_snr_aper_height = np.int32(temp)

    fine_buffer = 10
    fine_aper_widths = np.arange(min_snr_aper_width - fine_buffer,
                                 min_snr_aper_width + fine_buffer)
    fine_aper_heights = np.arange(min_snr_aper_height - fine_buffer,
                                  min_snr_aper_height + fine_buffer)

    wasp43.do_multi_phot(fine_aper_widths, fine_aper_heights)

    from plotting import plot_2D_stddev, plot_center_position_vs_scan_and_orbit
    plot_2D_stddev(wasp43, aper_widths, aper_heights, signal_max=235)
    plot_center_position_vs_scan_and_orbit(wasp43)

    if clargs.plot_verbose:
        wasp43.plot_errorbars()

    if clargs.fit_model_flag:
        res = wasp43.do_fit(times, fluxes, errors,
                            init_params=[], static_params={})

    if clargs.save_now:
        # csv_filename = f'{clargs.planet_name}_photometry.csv'
        joblib_filename = f'{planet_name}_savedict_NNN_NNNxNNN.joblib.save'

        # csv_filename = f'{working_dir}/{csv_filename}'
        joblib_filename = f'{working_dir}/{joblib_filename}'

        # wasp43.save_text_file(csv_filename)
        wasp43.save_dict(joblib_filename)

    from plotting import plot_aperture_edges_with_angle, plot_lightcurve
    from plotting import plot_xcenter_vs_time, plot_ycenter_vs_time
    from plotting import plot_xcenter_vs_flux, plot_ycenter_vs_flux
    from plotting import plot_center_position_vs_scan_and_orbit, plot_2D_stddev

    plot_lightcurve(wasp43, aper_width, aper_height)
    plot_ycenter_vs_flux(wasp43, aper_width, aper_height)
    plot_xcenter_vs_flux(wasp43, aper_width, aper_height)
    plot_ycenter_vs_time(wasp43)
    plot_xcenter_vs_time(wasp43)
    plot_aperture_edges_with_angle(wasp43, img_id=42)
    plot_center_position_vs_scan_and_orbit(wasp43)
    plot_2D_stddev(wasp43.photometry_df,
                   aper_widths,
                   aper_heights,
                   signal_max=230)
