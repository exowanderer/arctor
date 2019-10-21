import numpy as np
import os

from HSTUVISTimeSeries import HSTUVISTimeSeries


def instantiate_wasp43(planet_name, data_dir, working_dir, file_type):
    wasp43 = HSTUVISTimeSeries(
        planet_name=planet_name,
        data_dir=data_dir,
        working_dir=working_dir,
        file_type=file_type)

    joblib_filename = f'{planet_name}_savedict.joblib.save'
    joblib_filename = f'{working_dir}/{joblib_filename}'
    wasp43.load_data(joblib_filename)  # joblib_filename

    return wasp43

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
    wasp43.center_all_traces()
    wasp43.fit_trace_slopes()

    wasp43.identify_trace_direction()
    wasp43.clean_cosmic_rays()
    wasp43.compute_sky_background()
    wasp43.compute_columnwise_sky_background()

    # wasp43.do_phot()  # Default Settings

    min_aper_width = 1
    max_aper_width = 2
    min_aper_height = 1
    max_aper_height = 2

    aper_widths = np.arange(min_aper_width, max_aper_width + 1)  # + 2, 5)
    aper_heights = np.arange(min_aper_height, max_aper_height + 1)  # + 2, 5)
    wasp43.do_multi_phot(aper_widths, aper_heights, position=None, theta=None)

    if clargs.plot_verbose:
        wasp43.plot_errorbars()

    if clargs.fit_model_flag:
        res = wasp43.do_fit(times, fluxes, errors,
                            init_params=[], static_params={})

    if clargs.save_now:
        # csv_filename = f'{clargs.planet_name}_photometry.csv'
        joblib_filename = f'{planet_name}_savedict.joblib.save'

        # csv_filename = f'{working_dir}/{csv_filename}'
        joblib_filename = f'{working_dir}/{joblib_filename}'

        # wasp43.save_text_file(csv_filename)
        wasp43.save_dict(joblib_filename)
