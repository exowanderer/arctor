import joblib
import numpy as np
import os
import pandas as pd

from matplotlib import pyplot as plt


def plot_map_model(times, phots, uncs, model, t0_guess):
    fig, ax = plt.subplots()

    ax.errorbar(times - t0_guess, phots, uncs,
                fmt='o', ms=10, label='WASP43b UVIS')
    ax.plot(times - t0_guess, model, 'k--', lw=3, label='MAP Model')

    ax.legend(loc=0)

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
    plt.legend(loc=0)


def create_sub_sect(n_options, idx_split, use_xcenters, use_ycenters,
                    use_trace_angles, use_trace_lengths,
                    idx_split_, use_xcenters_, use_ycenters_,
                    use_trace_angles_, use_trace_lengths_):
    sub_sect = np.ones(n_options).astype(bool)
    _idx_split = idx_split == idx_split_
    _use_xcenters = use_xcenters == use_xcenters_
    _use_ycenters = use_ycenters == use_ycenters_
    _use_trace_angles = use_trace_angles == use_trace_angles_
    _use_tracelengths = use_trace_lengths == use_trace_lengths_

    sub_sect = np.bitwise_and(sub_sect, _idx_split)
    sub_sect = np.bitwise_and(sub_sect, _use_xcenters)
    sub_sect = np.bitwise_and(sub_sect, _use_ycenters)
    sub_sect = np.bitwise_and(sub_sect, _use_trace_angles)
    sub_sect = np.bitwise_and(sub_sect, _use_tracelengths)

    return sub_sect


def compute_chisq_aic(planet, aper_column, map_soln, idx_fwd, idx_rev,
                      use_idx_fwd_, use_xcenters_, use_ycenters_,
                      use_trace_angles_, use_trace_lengths_):

    phots = planet.normed_photometry_df[aper_column].values
    uncs = planet.normed_uncertainty_df[aper_column].values

    n_pts = len(phots)

    # 2 == eclipse depth + mean
    n_params = (2 + use_idx_fwd_ + use_xcenters_ + use_ycenters_ +
                use_trace_angles_ + use_trace_lengths_)

    if 'mean_fwd' not in map_soln.keys():
        map_model = map_soln['light_curves'].flatten() + map_soln['line_model']
    else:
        map_model = np.zeros_like(planet.times)
        map_model[idx_fwd] = map_soln['light_curves_fwd'].flatten() + \
            map_soln['line_model_fwd']
        map_model[idx_rev] = map_soln['light_curves_rev'].flatten() + \
            map_soln['line_model_rev']

        # if we split Fwd/Rev, then there are now 2 means
        n_params = n_params + 1

    correction = 2 * n_params * (n_params + 1) / (n_pts - n_params - 1)

    chisq_ = np.sum((map_model - phots)**2 / uncs**2)
    aic_ = chisq_ + 2 * n_params + correction
    bic_ = chisq_ + n_params * np.log10(n_pts)

    return chisq_, aic_, bic_


def extract_map_only_data(planet, idx_fwd, idx_rev,
                          maps_only_filename=None,
                          aper_columns_filename=None,
                          data_dir='notebooks'):
    if maps_only_filename is None:
        maps_only_filename = 'results_decor_span_MAPs_only_SDNR.joblib.save'
        maps_only_filename = os.path.join(data_dir, maps_only_filename)

    if aper_columns_filename is None:
        aper_columns_filename = 'decor_span_MAPs_only_aper_columns_list'
        aper_columns_filename = f'{aper_columns_filename}.joblib.save'
        aper_columns_filename = os.path.join(data_dir, aper_columns_filename)

    decor_span_MAPs_only_list = joblib.load(maps_only_filename)
    decor_aper_columns_list = joblib.load(aper_columns_filename)

    idx_split = np.zeros(len(decor_span_MAPs_only_list), dtype=bool)
    use_xcenters = np.zeros(len(decor_span_MAPs_only_list), dtype=bool)
    use_ycenters = np.zeros(len(decor_span_MAPs_only_list), dtype=bool)
    use_trace_angles = np.zeros(len(decor_span_MAPs_only_list), dtype=bool)
    use_trace_lengths = np.zeros(len(decor_span_MAPs_only_list), dtype=bool)

    chisq_apers = np.zeros(len(decor_span_MAPs_only_list))
    aic_apers = np.zeros(len(decor_span_MAPs_only_list))
    bic_apers = np.zeros(len(decor_span_MAPs_only_list))
    res_std_ppm = np.zeros(len(decor_span_MAPs_only_list))
    phots_std_ppm = np.zeros(len(decor_span_MAPs_only_list))
    res_diff_ppm = np.zeros(len(decor_span_MAPs_only_list))

    n_pts = len(planet.normed_photometry_df)

    zipper = zip(decor_aper_columns_list, decor_span_MAPs_only_list)
    map_solns = {}
    fine_grain_mcmcs_s = {}
    for k, (aper_column, map_results) in enumerate(zipper):
        idx_split[k] = map_results[0]
        use_xcenters[k] = map_results[2]
        use_ycenters[k] = map_results[3]
        use_trace_angles[k] = map_results[4]
        use_trace_lengths[k] = map_results[5]
        fine_grain_mcmcs_ = map_results[6]
        map_soln_ = map_results[7]
        res_std_ppm[k] = map_results[8]
        phots_std_ppm[k] = map_results[9]
        res_diff_ppm[k] = map_results[10]

        key = (f'idx_split:{idx_split[k]}'
               f'_use_xcenters:{use_xcenters[k]}'
               f'_use_ycenters:{use_ycenters[k]}'
               f'_use_trace_angles:{use_trace_angles[k]}'
               f'_use_trace_lengths:{use_trace_lengths[k]}')

        fine_grain_mcmcs_s[key] = fine_grain_mcmcs_
        map_solns[key] = map_soln_

        chisq_, aic_, bic_ = compute_chisq_aic(planet, aper_column, map_soln_,
                                               idx_fwd, idx_rev,
                                               idx_split[k],
                                               use_xcenters[k],
                                               use_ycenters[k],
                                               use_trace_angles[k],
                                               use_trace_lengths[k])

        chisq_apers[k] = chisq_
        aic_apers[k] = aic_
        bic_apers[k] = bic_

    return (idx_split, use_xcenters, use_ycenters, use_trace_angles,
            use_trace_lengths, fine_grain_mcmcs_s, map_solns, res_std_ppm,
            phots_std_ppm, res_diff_ppm, chisq_apers, aic_apers, bic_apers)


def plot_aper_width_grid():
    rand0 = np.random.normal(0, 0.1, 3200)

    plt.scatter(res_std_ppm[res_diff_ppm > 0],
                res_diff_ppm[res_diff_ppm > 0],
                c=res_diff_ppm[res_diff_ppm > 0])

    plt.scatter((decor_aper_widths_only + 0.25 * use_xcenters)[use_xcenters],
                (decor_aper_heights_only + 0.25 * use_ycenters)[~use_ycenters],
                c=res_std_ppm, alpha=0.25, label='x:True y:False', marker='o')

    plt.scatter((decor_aper_widths_only + 0.25 * use_xcenters)[use_xcenters],
                (decor_aper_heights_only + 0.25 * use_ycenters)[use_ycenters],
                c=res_std_ppm, alpha=0.25, label='x:True y:True', marker='s')

    plt.scatter((decor_aper_widths_only + 0.25 * use_xcenters)[~use_xcenters],
                (decor_aper_heights_only + 0.25 * use_ycenters)[use_ycenters],
                c=res_std_ppm, alpha=0.25, label='x:False y:True', marker='*')

    plt.scatter((decor_aper_widths_only + 0.25 * use_xcenters)[~use_xcenters],
                (decor_aper_heights_only + 0.25 * use_ycenters)[~use_ycenters],
                c=res_std_ppm[~use_xcenters], alpha=0.25,
                label='x:False y:False', marker='^')

    plt.legend(loc=0)


def plot_aper_grid_per_feature(ax, n_options, idx_split, use_xcenters,
                               use_ycenters, use_trace_angles,
                               use_trace_lengths, res_std_ppm,
                               chisq_apers, aic_apers, bic_apers,
                               decor_aper_widths_only, decor_aper_heights_only,
                               idx_split_, use_xcenters_, use_ycenters_,
                               use_trace_angles_, use_trace_lengths_,
                               one_fig=False, fig=None, hspace=0.5):

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

    aper_widths_ = decor_aper_widths_only[sub_sect]
    aper_heights_ = decor_aper_heights_only[sub_sect]
    out = ax.scatter(aper_widths_, aper_heights_,
                     c=aic_apers[sub_sect],
                     marker='s', s=200)

    min_aic_sub = aic_apers[sub_sect].min()
    argmin_aic_sub = aic_apers[sub_sect].argmin()
    manual_argmin = np.where(aic_apers[sub_sect] == min_aic_sub)[0][0]
    assert(manual_argmin == argmin_aic_sub), \
        f'{manual_argmin}, {argmin_aic_sub}'

    best_ppm = res_std_ppm[argmin_aic_sub]
    width_best = aper_widths_[argmin_aic_sub]
    height_best = aper_heights_[argmin_aic_sub]

    txt = f'{best_ppm:0.1f} ppm\n[{width_best}x{height_best}]'
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
                fontsize=8,
                color='black',
                weight='bold')

    title = f'AIC: {np.int(np.round(min_aic_sub)):0.0f}'
    ax.set_title(title)

    if one_fig:
        plt.subplots_adjust(hspace=hspace)

    if not one_fig:
        if fig is None:
            fig = plt.gcf()

        left, bottom, width, height = ax.get_position().bounds
        cbaxes = fig.add_axes([left + width, bottom, 0.025, height])
        cb = plt.colorbar(out, cax=cbaxes)


def organize_results_ppm_chisq_aic(n_options, idx_split, use_xcenters,
                                   use_ycenters, use_trace_angles,
                                   use_trace_lengths, res_std_ppm,
                                   chisq_apers, aic_apers, bic_apers,
                                   decor_aper_widths_only,
                                   decor_aper_heights_only,
                                   idx_split_, use_xcenters_, use_ycenters_,
                                   use_trace_angles_, use_trace_lengths_):

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

    aper_widths_sub = decor_aper_widths_only[sub_sect]
    aper_heights_sub = decor_aper_heights_only[sub_sect]

    argbest_ppm = res_std_ppm[sub_sect].argmin()

    best_ppm_sub = res_std_ppm[sub_sect][argbest_ppm]
    best_chisq_sub = chisq_apers[sub_sect][argbest_ppm]
    best_aic_sub = aic_apers[sub_sect][argbest_ppm]
    width_best = aper_widths_sub[argbest_ppm]
    height_best = aper_heights_sub[argbest_ppm]

    std_res_sub_min = res_std_ppm[sub_sect].min()
    chisq_sub_min = chisq_apers[sub_sect].min()
    aic_sub_min = aic_apers[sub_sect].min()

    entry = {f'idx_split': idx_split_,
              f'xcenters': use_xcenters_,
              f'ycenters': use_ycenters_,
              f'trace_angles': use_trace_angles_,
              f'trace_lengths': use_trace_lengths_,
              f'width_best': width_best,
              f'height_best': height_best,
              f'best_ppm_sub': best_ppm_sub,
              f'best_chisq_sub': best_chisq_sub,
              f'best_aic_sub': best_aic_sub,
              f'std_res_sub_min': std_res_sub_min,
              f'chisq_sub_min': chisq_sub_min,
              f'aic_sub_min': aic_sub_min}

    return entry


def create_results_df(decor_aper_widths_only, decor_aper_heights_only,
                      res_std_ppm, chisq_apers, aic_apers, bic_apers,
                      idx_split, use_xcenters, use_ycenters,
                      use_trace_angles, use_trace_lengths):

    n_options = len(decor_aper_widths_only)
    results_dict = {}
    for idx_split_ in [True, False]:
        for use_xcenters_ in [True, False]:
            for use_ycenters_ in [True, False]:
                for use_trace_angles_ in [True, False]:
                    for use_trace_lengths_ in [True, False]:
                        entry = organize_results_ppm_chisq_aic(
                            n_options, idx_split, use_xcenters, use_ycenters,
                            use_trace_angles, use_trace_lengths, res_std_ppm,
                            chisq_apers, aic_apers, bic_apers,
                            decor_aper_widths_only, decor_aper_heights_only,
                            idx_split_, use_xcenters_, use_ycenters_,
                            use_trace_angles_, use_trace_lengths_)

                        for key, val in entry.items():
                            if key not in results_dict.keys():
                                results_dict[key] = []

                            results_dict[key].append(val)

    return pd.DataFrame(results_dict)


def plot_32_subplots_for_each_feature(decor_aper_widths_only,
                                      decor_aper_heights_only,
                                      res_std_ppm, chisq_apers,
                                      aic_apers, bic_apers,
                                      idx_split, use_xcenters,
                                      use_ycenters, use_trace_angles,
                                      use_trace_lengths, one_fig=False):

    if one_fig:
        fig, axs = plt.subplots(nrows=4, ncols=8)

    n_options = len(decor_aper_widths_only)
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
                                                   chisq_apers,
                                                   aic_apers,
                                                   bic_apers,
                                                   decor_aper_widths_only,
                                                   decor_aper_heights_only,
                                                   idx_split_,
                                                   use_xcenters_,
                                                   use_ycenters_,
                                                   use_trace_angles_,
                                                   use_trace_lengths_,
                                                   one_fig=one_fig)

if __name__ == '__main__':
    import os
    HOME = os.environ['HOME']
    os.chdir(f'{HOME}/Research/Planets/WASP43/github_analysis/notebooks')

    import joblib
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    from exomast_api import exoMAST_API

    from arctor import Arctor
    from arctor.utils import setup_and_plot_GTC
    from arctor.utils import fit_2D_time_vs_other
    from anaylze_map_only_sdnr import *

    plot_verbose = False
    save_now = False
    planet_name = 'WASP43'
    file_type = 'flt.fits'

    HOME = os.environ['HOME']
    base_dir = os.path.join(HOME, 'Research', 'Planets', 'WASP43')
    data_dir = os.path.join(base_dir, 'data', 'UVIS', 'MAST_2019-07-03T0738')
    data_dir = os.path.join(data_dir, 'HST', 'FLTs')

    saving_dir = os.path.join(base_dir, 'github_analysis', 'savefiles')
    working_dir = os.path.join(base_dir, 'github_analysis')
    os.chdir(working_dir)

    planet = Arctor(planet_name, data_dir, saving_dir, file_type)
    joblib_filename = 'WASP43_savedict_206ppm_100x100_finescale.joblib.save'
    joblib_filename = f'{HOME}/Research/Planets/savefiles/{joblib_filename}'
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

    maps_only_filename = '../results_decor_span_MAPs_only_SDNR.joblib.save'
    aper_columns_filename = os.path.join(
        'notebooks', 'decor_span_MAPs_only_aper_columns_list')

    idx_fwd = planet.idx_fwd
    idx_rev = planet.idx_rev

    idx_split, use_xcenters, use_ycenters, use_trace_angles, \
        use_trace_lengths, fine_grain_mcmcs_s, map_solns, res_std_ppm, \
        phots_std_ppm, res_diff_ppm, chisq_apers, aic_apers, bic_apers \
        = extract_map_only_data(planet, idx_fwd, idx_rev)

    filename = 'decor_span_MAPs_only_aper_columns_list.joblib.save'
    filename = os.path.join('notebooks', filename)
    decor_aper_columns_list = joblib.load(filename)

    decor_aper_widths_heights_list = []
    for aper_column in decor_aper_columns_list:
        entry = np.int32(aper_column.split('_')[-1].split('x'))
        decor_aper_widths_heights_list.append(entry)

    decor_aper_widths_only, decor_aper_heights_only = np.transpose(
        decor_aper_widths_heights_list)

    decor_results_df = create_results_df(decor_aper_widths_only,
                                         decor_aper_heights_only,
                                         res_std_ppm,
                                         chisq_apers,
                                         aic_apers,
                                         bic_apers,
                                         ~idx_split,
                                         ~use_xcenters,
                                         ~use_ycenters,
                                         ~use_trace_angles,
                                         ~use_trace_lengths)

    plot_32_subplots_for_each_feature(decor_aper_widths_only,
                                      decor_aper_heights_only,
                                      res_std_ppm,
                                      chisq_apers,
                                      aic_apers,
                                      bic_apers,
                                      ~idx_split,
                                      ~use_xcenters,
                                      ~use_ycenters,
                                      ~use_trace_angles,
                                      ~use_trace_lengths,
                                      one_fig=True)
