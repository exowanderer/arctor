from matplotlib import use as mpl_use

try:
    mpl_use('Qt5Agg')
except Exception as err:
    print(f'[WARNING] {err}')

from matplotlib import pyplot as plt

import joblib
import numpy as np
import os
import pandas as pd
import pygtc

from arctor import Arctor
from arctor.utils import fit_2D_time_vs_other
from arctor.utils import extract_map_only_data, create_results_df
from arctor.utils import setup_and_plot_GTC, info_message
# from arctor.plotting import plot_32_subplots_for_each_feature
# from arctor.plotting import plot_best_aic_light_curve
from arctor import plotting

from exomast_api import exoMAST_API
from tqdm import tqdm

if __name__ == '__main__':

    HOME = os.environ['HOME']

    plot_verbose = False
    save_now = False
    planet_name = 'WASP43'
    file_type = 'flt.fits'

    core_dir = os.path.join('/', 'Volumes', 'WhenImSixtyFourGB')

    if not os.path.exists(core_dir):
        core_dir = os.path.join(HOME, 'Research', 'Planets')
        assert(os.path.exists(core_dir)),\
            'Did not find either Laptop or Server Directory Structure'
        print('Found Server Directory Structure:')
    else:
        print('Found Laptop Directory Structure:')

    print(core_dir)

    save_dir = os.path.join(core_dir, 'savefiles')
    base_dir = os.path.join(core_dir, 'WASP43')

    data_dir = os.path.join(base_dir, 'data', 'UVIS', 'MAST_2019-07-03T0738')
    data_dir = os.path.join(data_dir, 'HST', 'FLTs')

    working_dir = os.path.join(base_dir, 'github_analysis')
    notebook_dir = os.path.join(working_dir, 'notebooks')

    planet = Arctor(planet_name, data_dir, save_dir, file_type)
    joblib_filename = 'WASP43_savedict_206ppm_100x100_finescale.joblib.save'
    joblib_filename = os.path.join(save_dir, joblib_filename)

    info_message('Loading Joblib Savefile to Populate `planet`')
    planet.load_dict(joblib_filename)

    med_phot_df = np.median(planet.photometry_df, axis=0)
    planet.normed_photometry_df = planet.photometry_df / med_phot_df
    planet.normed_uncertainty_df = np.sqrt(planet.photometry_df) / med_phot_df

    times = planet.times

    wasp43 = exoMAST_API('WASP43b')
    t0_wasp43 = wasp43.transit_time  # 55528.3684  # exo.mast.stsci.edu
    period_wasp43 = wasp43.orbital_period
    n_epochs = np.int(
        np.round(((np.median(times) - t0_wasp43) / period_wasp43) - 0.5))
    t0_guess = t0_wasp43 + (n_epochs + 0.5) * period_wasp43

    data_dir = '/Volumes/WhenImSixtyFourGB/WASP43/github_analysis/notebooks'
    maps_only_filename = 'results_decor_span_MAPs_all400_SDNR_only.joblib.save'
    maps_only_filename = os.path.join(data_dir, maps_only_filename)

    idx_fwd = planet.idx_fwd
    idx_rev = planet.idx_rev

    decor_span_MAPs, keys_list, aper_widths, aper_heights, idx_split,\
        use_xcenters, use_ycenters, use_trace_angles, use_trace_lengths,\
        fine_grain_mcmcs_s, map_solns, res_std_ppm, phots_std_ppm,\
        res_diff_ppm, sdnr_apers, chisq_apers, aic_apers, bic_apers = \
        extract_map_only_data(
            planet, idx_fwd, idx_rev,
            maps_only_filename=maps_only_filename)

    # filename = 'decor_span_MAPs_only_aper_columns_list.joblib.save'
    # filename = os.path.join(data_dir, filename)

    decor_results_df = create_results_df(aper_widths,
                                         aper_heights,
                                         res_std_ppm,
                                         sdnr_apers,
                                         chisq_apers,
                                         aic_apers,
                                         bic_apers,
                                         idx_split,
                                         use_xcenters,
                                         use_ycenters,
                                         use_trace_angles,
                                         use_trace_lengths)

    # mcmc_13x45_filename = 'results_decor_span_MCMCs_25_bestest_SDNR_'\
    #     'aperture_sum_13x45.joblib.save'

    # mcmc_13x45_dir = os.path.join(notebook_dir, working_dir,
    #                               'all400_results_decor_MCMCs_SDNR')

    # mcmc_13x45_filename = os.path.join(mcmc_13x45_dir, mcmc_13x45_filename)
    # mcmc_13x45 = joblib.load(mcmc_13x45_filename)

    # mcmc_samples_fname = mcmc_13x45_filename.replace('.joblib.save',
    # '_mcmc_samples_df.csv')
    mcmc_samples_fname = (
        'results_decor_span_MCMCs_25_bestest_SDNR_aperture_sum_'
        '13x45_samples_df.csv')

    mcmc_samples_fname = os.path.join(notebook_dir, mcmc_samples_fname)
    # mcmc_samples.to_csv(mcmc_samples_fname, index=False)

    mcmc_samples_df = pd.read_csv(mcmc_samples_fname)
    varnames = mcmc_samples_df.columns
    # varnames = [key for key in map_soln.keys()
    #             if '__' not in key and 'light' not in key
    #             and 'line' not in key and 'le_edepth_0' not in key]

    # best = [False, True, True, True, False, True]

    # for k, thingy in enumerate(mcmc_13x45['aperture_sum_13x45']):
    #     isit = True
    #     for m, thingies in enumerate(thingy[:6]):
    #         isit = isit and (thingies == best[m])
    #         if isit:
    #             idx_best = k

    # idx_mcmc = 7
    # mcmc_samples = pm.trace_to_dataframe(
    #     mcmc_13x45['aperture_sum_13x45'][idx_best][idx_mcmc],
    #     varnames=varnames
    # )

    plotName = ''
    smoothingKernel = 1
    customLabelFont = {'rotation': 45, 'size': 20,
                       'xlabelpad': 0, 'ylabelpad': 0}

    pygtc.plotGTC(mcmc_samples_df,
                  plotName=plotName,
                  smoothingKernel=smoothingKernel,
                  labelRotation=[True] * 2,
                  # plotDensity=True,
                  customLabelFont=customLabelFont,
                  nContourLevels=3,
                  figureSize='APJ_page'
                  )

    fig, ax = plt.subplots()
    aper_width_bic_best = 13
    aper_height_bic_best = 45
    ax = plotting.plot_raw_light_curve(planet,
                                       aper_width_bic_best,
                                       aper_height_bic_best,
                                       t0_base=t0_guess,
                                       ax=ax)

    ax = plotting.plot_best_aic_light_curve(
        planet, map_solns,
        decor_results_df, mcmc_samples_df,
        aic_apers,  keys_list,
        aic_thresh=2, t0_base=t0_guess,
        plot_many=False, plot_raw=True,
        ax=ax)

    plotting.plot_32_subplots_for_each_feature(
        aper_widths, aper_heights,
        res_std_ppm, sdnr_apers, chisq_apers, aic_apers, bic_apers,
        idx_split, use_xcenters, use_ycenters, use_trace_angles,
        use_trace_lengths, one_fig=True, focus='aic')

    aper_width = 13
    aper_height = 45

    ax = plotting.plot_aperture_background_vs_time(
        planet, ax=ax, t0_base=t0_guess, size=100, include_orbits=False)
    ax = plotting.plot_columwise_background_vs_time(
        planet, ax=ax, t0_base=t0_guess, size=100, include_orbits=False)
    ax = plotting.plot_trace_angle_vs_time(
        planet, ax=ax, t0_base=t0_guess, size=100, include_orbits=False)
    ax = plotting.plot_trace_length_vs_time(
        planet, ax=ax, t0_base=t0_guess, size=100, include_orbits=False)
    ax = plotting.plot_xcenter_vs_time(
        planet, ax=ax, t0_base=t0_guess, size=100, include_orbits=False)
    ax = plotting.plot_ycenter_vs_time(
        planet, ax=ax, t0_base=t0_guess, size=100, include_orbits=False)

    ax = plotting.plot_center_position_vs_scan_and_orbit(
        planet, ax=ax, t0_base=0, size=100, include_orbits=True)

    ax = plotting.plot_ycenter_vs_flux(
        planet, aper_width, aper_height,
        t0_base=0, ax=ax, size=100, include_orbits=False)

    ax = plotting.plot_xcenter_vs_flux(
        planet, aper_width, aper_height,
        t0_base=0, ax=ax, size=100, include_orbits=False)
