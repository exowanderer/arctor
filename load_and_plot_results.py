from tqdm import tqdm
from exomast_api import exoMAST_API
from arctor.utils import setup_and_plot_GTC, info_message, warning_message
from arctor.utils import extract_map_only_data, create_results_df
from arctor import Arctor
import pandas as pd
import os
import numpy as np
import joblib
cannot_plot = False

try:
    from matplotlib import use as mpl_use
    mpl_use('Qt5Agg')
except Exception as err:
    cannot_plot = True
    warning_message(f'{err}')

try:
    from matplotlib import pyplot as plt
    import pygtc
    from arctor import plotting
except Exception as err:
    cannot_plot = True
    warning_message(f'{err}' +
                    '\n          You probably are running on a remote server')


if __name__ == '__main__':

    HOME = os.environ['HOME']

    ppm = 1e6

    want_plot = False
    save_plot_now = False
    planet_name = 'PlanetName'
    file_type = 'flt.fits'

    # Identify where the stored MAP solns and MCMC posteriors are
    aper_width_bic_best = 13
    aper_height_bic_best = 45

    if os.path.exists(os.path.join('/', 'Volumes', 'WhenImSixtyFourGB')):
        core_dir = os.path.join('/', 'Volumes', 'WhenImSixtyFourGB')
    elif os.path.exists(
            os.path.join('/', 'media', 'jonathan', 'WhenImSixtyFourGB')):
        core_dir = os.path.join('/', 'media', 'jonathan', 'WhenImSixtyFourGB')
    else:
        core_dir = 'None'
        warning_message('No path exists to `core_dir`; setting to "None"')

    if not os.path.exists(core_dir):
        core_dir = os.path.join(HOME, 'Research', 'Planets')
        assert(os.path.exists(core_dir)),\
            'Did not find either Laptop or Server Directory Structure'

    info_message(f'Setting `core_dir` to {core_dir}')

    base_dir = os.path.join(core_dir, planet_name)
    save_dir = os.path.join(base_dir, 'savefiles')

    data_dir = os.path.join(base_dir, 'path', 'to', 'flt', 'files')
    data_dir = os.path.join(data_dir, 'HST', 'FLTs')

    working_dir = os.path.join(base_dir, 'path', 'to', 'analysis')
    plot_dir = os.path.join(working_dir, 'path', 'to', 'figures')
    notebook_dir = os.path.join(working_dir, 'notebooks')

    planet = Arctor(planet_name, data_dir, save_dir, file_type)
    joblib_filename = f'{planet_name}_savedict_XXXppm_100x100_finescale.joblib.save'
    joblib_filename = os.path.join(save_dir, joblib_filename)

    info_message('Loading Joblib Savefile to Populate `planet`')
    planet.load_dict(joblib_filename)

    med_phot_df = np.median(planet.photometry_df, axis=0)
    planet.normed_photometry_df = planet.photometry_df / med_phot_df
    planet.normed_uncertainty_df = np.sqrt(planet.photometry_df) / med_phot_df

    times = planet.times
    idx_fwd = planet.idx_fwd
    idx_rev = planet.idx_rev

    planet_info = exoMAST_API(f'{planet_name}b')
    t0_planet = planet_info.transit_time  # 55528.3684  # exo.mast.stsci.edu
    period_planet = planet_info.orbital_period
    n_epochs = np.int(
        np.round(((np.median(times) - t0_planet) / period_planet) - 0.5))
    t0_guess = t0_planet + (n_epochs + 0.5) * period_planet

    data_dir = os.path.join(core_dir, 'path', 'to', 'notebooks', 'savefiles')
    # maps_only_filename = 'results_decor_span_MAPs_all400_SDNR_only.joblib.save'
    maps_only_filename = f'results_decor_span_MAPs_only_SDNR_aperture_sum_{best_width}x{best_height}.joblib.save'
    maps_only_filename = os.path.join(data_dir, maps_only_filename)

    try:
        decor_span_MAPs, keys_list, aper_widths, aper_heights, idx_split,\
            use_xcenters, use_ycenters, use_trace_angles, use_trace_lengths,\
            fine_grain_mcmcs_s, map_solns, res_std_ppm, phots_std_ppm,\
            res_diff_ppm, sdnr_apers, chisq_apers, aic_apers, bic_apers = \
            extract_map_only_data(planet, idx_fwd, idx_rev,
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
    except Exception as err:
        warning_message(f'{err}')

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
    best_mcmc_params = mcmc_samples_df.median()
    best_mcmc_params['slope_time'] = best_mcmc_params['slope']
    # best = [False, True, True, True, False, True]
    toggle_idx_split = False
    toggle_xcenters = True
    toggle_ycenters = True
    toggle_trace_angles = False
    toggle_trace_lengths = True

    map_soln_key = 'aper_column:aperture_sum_'
    map_soln_key = f'{map_soln_key}{aper_width_bic_best}'
    map_soln_key = f'{map_soln_key}x{aper_height_bic_best}'
    map_soln_key = f'{map_soln_key}-idx_split:{toggle_idx_split}'
    map_soln_key = f'{map_soln_key}-_use_xcenters:{toggle_xcenters}'
    map_soln_key = f'{map_soln_key}-_use_ycenters:{toggle_ycenters}'
    map_soln_key = f'{map_soln_key}-_use_trace_angles:{toggle_trace_angles}'
    map_soln_key = f'{map_soln_key}-_use_trace_lengths:{toggle_trace_lengths}'

    try:
        map_soln = map_solns[map_soln_key]
    except Exception as err:
        warning_message(f'{err}')

    varnames = mcmc_samples_df.columns
    # varnames = [key for key in map_soln.keys()
    #             if '__' not in key and 'light' not in key
    #             and 'line' not in key and 'le_edepth_0' not in key]

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
    customLabelFont = {'rotation': 45, 'size': 20}  # ,
    # 'xlabelpad': 0, 'ylabelpad': 0}
    customTickFont = {'size': 20}

    ppm_columns = ['edepth', 'slope', 'slope_xcenter', 'slope_ycenter',
                   'slope_trace_length'],
    param_names = ['Mean Offset', 'Eclipse Depth', 'Time Slope',
                   'X-Center Slope', 'Y-Center Slope',
                   'Trace Length Slope']

    plot_samples_df = mcmc_samples_df.copy()
    for colname in ppm_columns:
        plot_samples_df[colname] = plot_samples_df[colname] * ppm

    plot_samples_df['mean'] = (plot_samples_df['mean'] - 1) * ppm

    # Check if you can plot
    if not cannot_plot and want_plot:

        try:
            ax.clear()
        except:
            fig, ax = plt.subplots()

        # Values from L.C. Mayorga predictions
        eclipse_depths = {'fsed>0.1': [45.908286 / ppm, '--'],
                          'fsed=0.1': [96.379104 / ppm, ':']}
        aper_column = 'aperture_sum_13x45'

        ax = plotting.plot_set_of_models(planet, best_mcmc_params,
                                         eclipse_depths, planet_info,
                                         aper_column=aper_column,
                                         n_pts_th=1000, t0_base=t0_guess,
                                         include_null=False, plot_raw=False,
                                         ax=ax)

        fig = plt.gcf()
        plot_name_ = 'plasma_flux_vs_time_new_mcmc_and_'
        plot_name_ = plot_name_ + 'LCMayorga_predictions.pdf'
        if save_plot_now:
            fig.savefig(os.path.join(plot_dir, plot_name_))

        pygtc.plotGTC(plot_samples_df,
                      plotName=plotName,
                      smoothingKernel=smoothingKernel,
                      paramNames=param_names,
                      labelRotation=[True] * 2,
                      colorsOrder=['cmap'],
                      cmap=plt.cm.plasma,
                      # plotDensity=True,
                      customLabelFont=customLabelFont,
                      nContourLevels=3,
                      figureSize='MNRAS_page',
                      customTickFont=customTickFont
                      )

        plt.subplots_adjust(
            top=0.995,
            bottom=0.1,
            left=0.45,
            right=0.995,
            hspace=0.01,
            wspace=0.01
        )
        fig = plt.gcf()
        plot_name_ = 'plasma_corner_plot_MAP_best_fit_13x45.pdf'
        if save_plot_now:
            fig.savefig(os.path.join(plot_dir, plot_name_))

        # ax = plotting.plot_best_aic_light_curve(
        #     planet, map_solns, decor_results_df,
        #     aic_apers,  keys_list,
        #     aic_thresh=2, t0_base=t0_guess,
        #     plot_many=False, plot_raw=True,
        #     ax=ax)

        # ax = plotting.plot_best_aic_light_curve(
        #     planet, map_solns, decor_results_df,
        #     aic_apers,  keys_list,
        #     aic_thresh=2, t0_base=t0_guess,
        #     plot_many=False, plot_raw=True,
        #     ax=ax)

        ax = plotting.plot_raw_light_curve(planet,
                                           aper_width_bic_best,
                                           aper_height_bic_best,
                                           t0_base=t0_guess,
                                           ax=ax)
        plot_name_ = 'plasma_flux_vs_time_raw.pdf'
        plt.tight_layout()
        if save_plot_now:
            fig.savefig(os.path.join(plot_dir, plot_name_))

        axs = None  # for starters and re-starters
        for focus in ['AIC', 'BIC', 'SDNR', 'CHISQ']:
            axs = plotting.plot_32_subplots_for_each_feature(
                aper_widths, aper_heights,
                res_std_ppm, sdnr_apers, chisq_apers, aic_apers, bic_apers,
                idx_split, use_xcenters, use_ycenters, use_trace_angles,
                use_trace_lengths, one_fig=True, focus=focus.lower(), axs=axs)

            plot_name_ = 'New_Plot_32_subplots_for_each_feature_'
            plot_name_ = plot_name_ + f'{focus}_sorted.pdf'
            fig = plt.gcf()
            plt.tight_layout()
            if save_plot_now:
                fig.savefig(os.path.join(plot_dir, plot_name_))

        ax = plotting.plot_aperture_background_vs_time(
            planet, ax=ax, t0_base=t0_guess, size=200, include_orbits=False)
        plot_name_ = 'plasma_sky_background_aperture_median_vs_time.pdf'
        plt.tight_layout()
        if save_plot_now:
            fig.savefig(os.path.join(plot_dir, plot_name_))

        ax = plotting.plot_columwise_background_vs_time(
            planet, ax=ax, t0_base=t0_guess, size=200, include_orbits=False)
        plot_name_ = 'plasma_sky_background_columnwise_median_vs_time.pdf'
        plt.tight_layout()
        if save_plot_now:
            fig.savefig(os.path.join(plot_dir, plot_name_))

        ax = plotting.plot_trace_angle_vs_time(
            planet, ax=ax, t0_base=t0_guess, size=200, include_orbits=False)
        plot_name_ = 'plasma_trace_angles_vs_time.pdf'
        plt.tight_layout()
        if save_plot_now:
            fig.savefig(os.path.join(plot_dir, plot_name_))

        ax = plotting.plot_trace_length_vs_time(
            planet, ax=ax, t0_base=t0_guess, size=200, include_orbits=False)
        plot_name_ = 'plasma_trace_lengths_vs_time.pdf'
        plt.tight_layout()
        if save_plot_now:
            fig.savefig(os.path.join(plot_dir, plot_name_))

        ax = plotting.plot_xcenter_vs_time(
            planet, ax=ax, t0_base=t0_guess, size=200, include_orbits=False)
        plot_name_ = 'plasma_x-center_position_vs_time.pdf'
        plt.tight_layout()
        if save_plot_now:
            fig.savefig(os.path.join(plot_dir, plot_name_))

        ax = plotting.plot_ycenter_vs_time(
            planet, ax=ax, t0_base=t0_guess, size=200, include_orbits=False)
        plot_name_ = 'plasma_y-center_position_vs_time.pdf'
        plt.tight_layout()
        if save_plot_now:
            fig.savefig(os.path.join(plot_dir, plot_name_))

        ax = plotting.plot_center_position_vs_scan_and_orbit(
            planet, ax=ax, t0_base=0, size=200,
            include_orbits='only_the_first')
        plot_name_ = 'plasma_y-center_vs_x-center_position.pdf'
        plt.tight_layout()
        if save_plot_now:
            fig.savefig(os.path.join(plot_dir, plot_name_))

        ax = plotting.plot_xcenter_position_vs_trace_length(
            planet, ax=ax, t0_base=0, size=200, include_orbits=False)
        plot_name_ = 'plasma_x-center_position_vs_trace_lengths.pdf'
        plt.tight_layout()
        if save_plot_now:
            fig.savefig(os.path.join(plot_dir, plot_name_))

        ax = plotting.plot_ycenter_vs_flux(
            planet, aper_width_bic_best, aper_height_bic_best,
            t0_base=0, ax=ax, size=200, include_orbits=False)
        plot_name_ = 'plasma_y-center_position_vs_flux.pdf'
        plt.tight_layout()
        if save_plot_now:
            fig.savefig(os.path.join(plot_dir, plot_name_))

        ax = plotting.plot_xcenter_vs_flux(
            planet, aper_width_bic_best, aper_height_bic_best,
            t0_base=0, ax=ax, size=200, include_orbits=False)
        plot_name_ = 'plasma_x-center_position_vs_flux.pdf'
        plt.tight_layout()
        if save_plot_now:
            fig.savefig(os.path.join(plot_dir, plot_name_))

        ax = plotting.plot_trace_lengths_vs_flux(
            planet, aper_width_bic_best, aper_height_bic_best,
            t0_base=0, ax=ax, size=200, include_orbits=False)
        plot_name_ = 'plasma_trace_lengths_vs_flux.pdf'
        plt.tight_layout()
        if save_plot_now:
            fig.savefig(os.path.join(plot_dir, plot_name_))

        fine_min_aic_colname = f'aperture_sum_{aper_width_bic_best}'
        fine_min_aic_colname = f'{fine_min_aic_colname}x{aper_height_bic_best}'

        times = planet.times
        flux = planet.normed_photometry_df[fine_min_aic_colname]

        trace_angles = planet.trace_angles
        ax = plotting.plot_2D_fit_time_vs_other(
            times, flux, trace_angles, idx_fwd, idx_rev,
            xytext=(15, 15), n_sig=5,
            varname='Trace Angles', n_spaces=[10, 10],
            convert_to_ppm=True, fontsize=40,
            leg_fontsize=30, xlim=None, fig=None,
            ax=ax)
        plot_name_ = 'Plasma_Flux_2D_correlation_plot_with_model'
        plot_name_ = plot_name_ + '_trace-angles_long.pdf'
        plt.tight_layout()
        if save_plot_now:
            fig.savefig(os.path.join(plot_dir, plot_name_))

        trace_lengths = planet.trace_lengths
        ax = plotting.plot_2D_fit_time_vs_other(
            times, flux, trace_lengths, idx_fwd, idx_rev,
            xytext=(15, 15), n_sig=5,
            varname='Trace Lengths', n_spaces=[10, 10],
            convert_to_ppm=False, fontsize=40,
            leg_fontsize=30, units='pixels',
            xlim=None, fig=None, ax=ax)
        plot_name_ = 'Plasma_Flux_2D_correlation_plot_with_model'
        plot_name_ = plot_name_ + '_trace-lengths_long.pdf'
        plt.tight_layout()
        if save_plot_now:
            fig.savefig(os.path.join(plot_dir, plot_name_))

        xcenters = planet.trace_xcenters
        ax = plotting.plot_2D_fit_time_vs_other(
            times, flux, xcenters, idx_fwd, idx_rev,
            xytext=(15, 15), n_sig=5,
            varname='X-Centers', n_spaces=[10, 10],
            convert_to_ppm=False, fontsize=40,
            leg_fontsize=30, units='pixels',
            xlim=None, fig=None, ax=ax)
        plot_name_ = 'Plasma_Flux_2D_correlation_plot_with_model'
        plot_name_ = plot_name_ + '_xcenters_long.pdf'
        plt.tight_layout()
        if save_plot_now:
            fig.savefig(os.path.join(plot_dir, plot_name_))

        xticks = [-0.150, -0.100, -0.050, 0.0, 0.050]
        ycenters = planet.trace_ycenters
        ax = plotting.plot_2D_fit_time_vs_other(
            times, flux, ycenters, idx_fwd, idx_rev,
            xytext=(15, 15), n_sig=5,
            varname='Y-Centers', n_spaces=[10, 10],
            convert_to_ppm=False, lw=5, fontsize=40,
            leg_fontsize=30, units='pixels', xticks=xticks,
            xlim=None, fig=None, ax=ax)
        plot_name_ = 'Plasma_Flux_2D_correlation_plot_with_model'
        plot_name_ = plot_name_ + '_ycenters_long.pdf'
        plt.tight_layout()
        if save_plot_now:
            fig.savefig(os.path.join(plot_dir, plot_name_))

        kernels = ('gaussian', 'tophat', 'epanechnikov',
                   'exponential', 'linear', 'cosine')

        # fig, ax = plt.subplots()

        ax = plotting.plot_kde_with_BCR_annotation(mcmc_samples_df,
                                                   kernel='gaussian',
                                                   ax=ax, fontsize=30)
        plot_name_ = 'Plasma_Fancy_Histogram_and_KDE_Eclipse_Depth_MCMC_for_'\
            'best_MAP_fit_13x45_Square_take2.pdf'
        plt.tight_layout()
        plt.subplots_adjust(
            top=0.995,
            bottom=0.1,
            left=0.45,
            right=0.97,
            hspace=0.01,
            wspace=0.01,
        )

        if save_plot_now:
            fig.savefig(os.path.join(plot_dir, plot_name_))

        axs = plotting.plot_aperture_edges_with_angle(
            planet, img_id=42, fontsize=40, axs=axs)
        plot_name_ = f'{planet_name}_UVIS_aperture_zoom_before_and_after_tilt.pdf'
        plt.tight_layout()
        if save_plot_now:
            fig.savefig(os.path.join(plot_dir, plot_name_))

        from photutils import RectangularAperture
        pos_median = (np.median(planet.trace_xcenters),
                      np.median(planet.trace_ycenters))
        theta_median = np.median(planet.trace_angles)

        inner_width = 75
        outer_width = 150
        inner_height = 225
        outer_height = 350

        inner_width = np.median(planet.trace_lengths) + inner_width
        outer_width = np.median(planet.trace_lengths) + outer_width
        aperture_width = np.median(planet.trace_lengths) + aper_width_bic_best
        aperture_height = aper_height_bic_best

        aperture = RectangularAperture(
            pos_median, aperture_width, aperture_height, theta_median)

        inner_annular = RectangularAperture(
            pos_median, inner_width, inner_height, theta_median)
        outer_annular = RectangularAperture(
            pos_median, outer_width, outer_height, theta_median)

        ax = plotting.plot_apertures(image=planet.image_stack[42],
                                     aperture=aperture,
                                     inner_annular=inner_annular,
                                     outer_annular=outer_annular,
                                     lw=5, ax=ax)
        plot_name_ = f'{planet_name}_UVIS_aperture_photometry_'
        plot_name_ = plot_name_ + 'and_median_background.pdf'
        if save_plot_now:
            fig.savefig(os.path.join(plot_dir, plot_name_))
