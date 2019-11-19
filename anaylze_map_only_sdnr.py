import joblib
import numpy as np
import os
import pandas as pd

from matplotlib import pyplot as plt
from tqdm import tqdm


def debug_message(message, end='\n'):
    print(f'[DEBUG] {message}', end=end)


def warning_message(message, end='\n'):
    print(f'[WARNING] {message}', end=end)


def info_message(message, end='\n'):
    print(f'[INFO] {message}', end=end)


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

    return np.where(sub_sect)[0]


def compute_chisq_aic(planet, aper_column, map_soln, idx_fwd, idx_rev,
                      use_idx_fwd_, use_xcenters_, use_ycenters_,
                      use_trace_angles_, use_trace_lengths_):
    ppm = 1e6

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

    sdnr_ = np.std(map_model - phots) * ppm
    chisq_ = np.sum((map_model - phots)**2 / uncs**2)
    aic_ = chisq_ + 2 * n_params + correction
    bic_ = chisq_ + n_params * np.log10(n_pts)

    return chisq_, aic_, bic_, sdnr_


def extract_map_only_data(planet, idx_fwd, idx_rev,
                          maps_only_filename=None,
                          data_dir='notebooks'):

    if maps_only_filename is None:
        maps_only_filename = 'results_decor_span_MAPs_all400_SDNR_only.joblib.save'
        maps_only_filename = os.path.join(data_dir, maps_only_filename)

    info_message('Loading Decorrelation Results for MAPS only Results')
    decor_span_MAPs = joblib.load(maps_only_filename)
    decor_aper_columns_list = list(decor_span_MAPs.keys())

    n_apers = len(decor_span_MAPs)

    aper_widths = []
    aper_heights = []
    idx_split = []
    use_xcenters = []
    use_ycenters = []
    use_trace_angles = []
    use_trace_lengths = []

    sdnr_apers = []
    chisq_apers = []
    aic_apers = []
    bic_apers = []
    res_std_ppm = []
    phots_std_ppm = []
    res_diff_ppm = []
    keys_list = []

    n_pts = len(planet.normed_photometry_df)

    map_solns = {}
    fine_grain_mcmcs_s = {}
    generator = enumerate(decor_span_MAPs.items())
    for m, (aper_column, map_results) in tqdm(generator, total=n_apers):
        if aper_column in ['xcenter', 'ycenter']:
            continue

        n_results_ = len(map_results)
        for map_result in map_results:

            aper_width_, aper_height_ = np.int32(
                aper_column.split('_')[-1].split('x'))

            aper_widths.append(aper_width_)
            aper_heights.append(aper_height_)

            idx_split.append(map_result[0])
            use_xcenters.append(map_result[2])
            use_ycenters.append(map_result[3])
            use_trace_angles.append(map_result[4])
            use_trace_lengths.append(map_result[5])

            fine_grain_mcmcs_ = map_result[6]
            map_soln_ = map_result[7]

            res_std_ppm.append(map_result[8])
            phots_std_ppm.append(map_result[9])
            res_diff_ppm.append(map_result[10])

            key = (f'aper_column:{aper_column}-'
                   f'idx_split:{idx_split[-1]}-'
                   f'_use_xcenters:{use_xcenters[-1]}-'
                   f'_use_ycenters:{use_ycenters[-1]}-'
                   f'_use_trace_angles:{use_trace_angles[-1]}-'
                   f'_use_trace_lengths:{use_trace_lengths[-1]}')

            keys_list.append(key)
            fine_grain_mcmcs_s[key] = fine_grain_mcmcs_
            map_solns[key] = map_soln_

            chisq_, aic_, bic_, sdnr_ = compute_chisq_aic(
                planet,
                aper_column,
                map_soln_,
                idx_fwd,
                idx_rev,
                idx_split[-1],
                use_xcenters[-1],
                use_ycenters[-1],
                use_trace_angles[-1],
                use_trace_lengths[-1])

            sdnr_apers.append(sdnr_)
            chisq_apers.append(chisq_)
            aic_apers.append(aic_)
            bic_apers.append(bic_)

    aper_widths = np.array(aper_widths)
    aper_heights = np.array(aper_heights)
    idx_split = np.array(idx_split)
    use_xcenters = np.array(use_xcenters)
    use_ycenters = np.array(use_ycenters)
    use_trace_angles = np.array(use_trace_angles)
    use_trace_lengths = np.array(use_trace_lengths)

    sdnr_apers = np.array(sdnr_apers)
    chisq_apers = np.array(chisq_apers)
    aic_apers = np.array(aic_apers)
    bic_apers = np.array(bic_apers)
    res_std_ppm = np.array(res_std_ppm)
    phots_std_ppm = np.array(phots_std_ppm)
    res_diff_ppm = np.array(res_diff_ppm)
    keys_list = np.array(keys_list)

    return (decor_span_MAPs, keys_list, aper_widths, aper_heights,
            idx_split, use_xcenters, use_ycenters,
            use_trace_angles, use_trace_lengths,
            fine_grain_mcmcs_s, map_solns,
            res_std_ppm, phots_std_ppm, res_diff_ppm,
            sdnr_apers, chisq_apers, aic_apers, bic_apers)


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

    plt.legend(loc=0)


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


def organize_results_ppm_chisq_aic(n_options, idx_split, use_xcenters,
                                   use_ycenters, use_trace_angles,
                                   use_trace_lengths, res_std_ppm,
                                   sdnr_apers, chisq_apers,
                                   aic_apers, bic_apers,
                                   aper_widths,
                                   aper_heights,
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

    aper_widths_sub = aper_widths[sub_sect]
    aper_heights_sub = aper_heights[sub_sect]

    argbest_ppm = res_std_ppm[sub_sect].argmin()

    best_ppm_sub = res_std_ppm[sub_sect][argbest_ppm]
    best_sdnr_sub = sdnr_apers[sub_sect][argbest_ppm]
    best_chisq_sub = chisq_apers[sub_sect][argbest_ppm]
    best_aic_sub = aic_apers[sub_sect][argbest_ppm]
    width_best = aper_widths_sub[argbest_ppm]
    height_best = aper_heights_sub[argbest_ppm]

    sdnr_res_sub_min = sdnr_apers[sub_sect].min()
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
              f'best_sdnr_sub': best_sdnr_sub,
              f'best_chisq_sub': best_chisq_sub,
              f'best_aic_sub': best_aic_sub,
              f'std_res_sub_min': std_res_sub_min,
              f'sdnr_res_sub_min': sdnr_res_sub_min,
              f'chisq_sub_min': chisq_sub_min,
              f'aic_sub_min': aic_sub_min}

    return entry


def create_results_df(aper_widths, aper_heights,
                      res_std_ppm, sdnr_apers, chisq_apers,
                      aic_apers, bic_apers, idx_split,
                      use_xcenters, use_ycenters,
                      use_trace_angles, use_trace_lengths):

    n_options = len(aper_widths)
    results_dict = {}
    for idx_split_ in [True, False]:
        for use_xcenters_ in [True, False]:
            for use_ycenters_ in [True, False]:
                for use_trace_angles_ in [True, False]:
                    for use_trace_lengths_ in [True, False]:
                        entry = organize_results_ppm_chisq_aic(
                            n_options, idx_split, use_xcenters, use_ycenters,
                            use_trace_angles, use_trace_lengths, res_std_ppm,
                            sdnr_apers, chisq_apers, aic_apers, bic_apers,
                            aper_widths, aper_heights,
                            idx_split_, use_xcenters_, use_ycenters_,
                            use_trace_angles_, use_trace_lengths_)

                        for key, val in entry.items():
                            if key not in results_dict.keys():
                                results_dict[key] = []

                            results_dict[key].append(val)

    return pd.DataFrame(results_dict)


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


def get_map_results_models(times, map_soln, idx_fwd, idx_rev):
    if 'mean_fwd' not in map_soln.keys():
        map_model = map_soln['light_curves'].flatten()
        line_model = map_soln['line_model'].flatten()
    else:
        map_model = np.zeros_like(times)
        line_model = np.zeros_like(times)
        map_model[idx_fwd] = map_soln['light_curves_fwd'].flatten()
        line_model[idx_fwd] = map_soln['line_model_fwd'].flatten()

        map_model[idx_rev] = map_soln['light_curves_rev'].flatten()
        line_model[idx_rev] = map_soln['line_model_rev'].flatten()

    return map_model, line_model


def plot_best_aic_light_curve(planet, map_solns, decor_results_df,
                              aic_apers,  keys_list, aic_thresh=2,
                              t0_base=0, ax=None):
    ppm = 1e6

    idx_fwd = planet.idx_fwd
    idx_rev = planet.idx_rev
    times = planet.times - t0_base

    HOME = os.environ['HOME']
    save_dir = os.path.join(HOME, 'Research', 'Planets',
                            'WASP43', 'github_analysis',
                            'paper', 'figures')

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

    ax.errorbar(times, (phots - line_model) * ppm, uncs * ppm,
                fmt='o', color='k', ms=10, zorder=10)
    ax.plot(times[times.argsort()], map_model[times.argsort()] * ppm,
            color='C1', lw=3, zorder=5)

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

    ax.set_xlabel('Time [Days from Eclipse]', fontsize=20)
    ax.set_ylabel('Normalized Flux [ppm]', fontsize=20)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(15)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)

    plt.show()

    return ax


if __name__ == '__main__':
    import os

    import joblib
    import numpy as np

    import pandas as pd

    from exomast_api import exoMAST_API
    from matplotlib import use as mpl_use
    mpl_use('Qt5Agg')

    from matplotlib import pyplot as plt

    from arctor import Arctor
    from arctor.utils import setup_and_plot_GTC
    from arctor.utils import fit_2D_time_vs_other
    from anaylze_map_only_sdnr import *

    HOME = os.environ['HOME']
    # os.chdir(f'{HOME}/Research/Planets/WASP43/github_analysis/notebooks')

    plot_verbose = False
    save_now = False
    planet_name = 'WASP43'
    file_type = 'flt.fits'

    base_dir = os.path.join('/Volumes', 'WhenImSixtyFourGB', 'WASP43')
    data_dir = os.path.join(base_dir, 'data', 'UVIS', 'MAST_2019-07-03T0738')
    data_dir = os.path.join(data_dir, 'HST', 'FLTs')

    save_dir = os.path.join('/Volumes', 'WhenImSixtyFourGB', 'savefiles')
    working_dir = os.path.join(base_dir, 'github_analysis')

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
        res_diff_ppm, sdnr_apers, chisq_apers, aic_apers, bic_apers = extract_map_only_data(
            planet, idx_fwd, idx_rev, maps_only_filename=maps_only_filename)

    filename = 'decor_span_MAPs_only_aper_columns_list.joblib.save'
    filename = os.path.join(data_dir, filename)

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

    plot_32_subplots_for_each_feature(aper_widths,
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
                                      use_trace_lengths,
                                      one_fig=True,
                                      focus='aic')
    fig, ax = plt.subplots()
    ax = plot_best_aic_light_curve(planet, map_solns, decor_results_df,
                                   aic_apers, keys_list, aic_thresh=5, ax=ax)

    import os
    import joblib

    base_dir = os.path.join('/Volumes', 'WhenImSixtyFourGB', 'WASP43')
    working_dir = os.path.join(base_dir, 'github_analysis')
    mcmc_13x45_filename = 'results_decor_span_MCMCs_25_bestest_SDNR_'\
        'aperture_sum_13x45.joblib.save'

    mcmc_13x45_dir = 'notebooks/all400_results_decor_MCMCs_SDNR'
    mcmc_13x45_filename = os.path.join(mcmc_13x45_dir, mcmc_13x45_filename)
    mcmc_13x45 = joblib.load(mcmc_13x45_filename)

    varnames = [key for key in map_soln.keys()
                if '__' not in key and 'light' not in key
                and 'line' not in key and 'le_edepth_0' not in key]

    best = [False, True, True, True, False, True]

    for k, thingy in enumerate(mcmc_13x45['aperture_sum_13x45']):
        isit = True
        for m, thingies in enumerate(thingy[:6]):
            isit = isit and (thingies == best[m])
            if isit:
                idx_best = k

    idx_mcmc = 7
    samples = pm.trace_to_dataframe(
        mcmc_13x45['aperture_sum_13x45'][idx_best][idx_mcmc], varnames=varnames
    )

    samples_fname = mcmc_13x45_filename.replace('.joblib.save',
                                                '_samples_df.csv')

    # samples.to_csv(samples_fname, index=False)

    samples = pd.read_csv('notebooks/'
                          'results_decor_span_MCMCs_25_bestest_'
                          'SDNR_aperture_sum_13x45_samples_df.csv')

    pygtc.plotGTC(samples,
                  plotName=plotName,
                  smoothingKernel=smoothingKernel,
                  labelRotation=[True] * 2,
                  # plotDensity=True,
                  customLabelFont={'rotation': 45, 'size': 20},
                  nContourLevels=3,
                  figureSize='APJ_page'
                  )
