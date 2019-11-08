import joblib
import numpy as np
from matplotlib import pyplot as plt


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

# maps_only_filename = 'notebooks/results_decor_span_MAPs_only_SDNR.joblib.save'
# decor_span_MAPs_only_list = joblib.load(maps_only_filename)

decor_aper_columns_list = joblib.load(
    'notebooks/decor_span_MAPs_only_aper_columns_list.joblib.save')

chisq_apers = np.zeros(len(decor_span_MAPs_only_list), dtype=bool)
aic_apers = np.zeros(len(decor_span_MAPs_only_list), dtype=bool)
idx_split = np.zeros(len(decor_span_MAPs_only_list), dtype=bool)
use_xcenters = np.zeros(len(decor_span_MAPs_only_list), dtype=bool)
use_ycenters = np.zeros(len(decor_span_MAPs_only_list), dtype=bool)
use_trace_angles = np.zeros(len(decor_span_MAPs_only_list), dtype=bool)
use_trace_lengths = np.zeros(len(decor_span_MAPs_only_list), dtype=bool)

res_std_ppm = np.zeros(len(decor_span_MAPs_only_list))
phots_std_ppm = np.zeros(len(decor_span_MAPs_only_list))
res_diff_ppm = np.zeros(len(decor_span_MAPs_only_list))

n_pts = len(planet.normed_photometry_df)

zipper = zip(decor_aper_columns_list, decor_span_MAPs_only_list)
fine_grain_mcmcs_s = {}
map_solns = {}
for k, aper_colname, map_results in enumerate(zipper):
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

    key = (f'idx_split:{use_idx_fwd_}'
           f'_use_xcenters:{use_xcenters_}'
           f'_use_ycenters:{use_ycenters_}'
           f'_use_trace_angles:{use_trace_angles_}'
           f'_use_trace_lengths:{use_trace_lengths_}')

    fine_grain_mcmcs_s[key] = fine_grain_mcmcs_
    map_solns[key] = map_soln_

    phots = planet.normed_photometry_df[aper_column].values
    uncs = planet.normed_uncertainty_df[aper_column].values

    if 'mean_fwd' not in map_soln.keys():
        map_model = map_soln['light_curves'].flatten() + map_soln['line_model']
    else:
        map_model = np.zeros_like(times)
        map_model[idx_fwd] = map_soln['light_curves_fwd'].flatten() + \
            map_soln['line_model_fwd']
        map_model[idx_rev] = map_soln['light_curves_rev'].flatten() + \
            map_soln['line_model_rev']

    n_params = (1 + use_idx_fwd_ + use_xcenters_ + use_ycenters_ +
                use_trace_angles_ + use_trace_lengths_)

    correction = 2 * n_params * (n_params + 1) / (n_pts - n_params - 1)

    chisq_apers[k] = np.sum((map_model - phots)**2 / uncs**2)
    aic_apers[k] = 2 * n_params - chisq_apers[k] + correction


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
            c=res_std_ppm[~use_xcenters], alpha=0.25, label='x:False y:False', marker='^')

plt.legend(loc=0)


def create_sub_sect(n_options, idx_split, use_xcenters, use_ycenters,
                    use_trace_angles, use_trace_lengths):
    sub_sect = np.ones(n_options).astype(bool)
    _idx_split = idx_split == False
    _use_xcenters = use_xcenters == True
    _use_ycenters = use_ycenters == True
    _use_trace_angles = use_trace_angles == True
    _use_tracelengths = use_trace_lengths == True

    sub_sect = np.bitwise_and(sub_sect, _idx_split)
    sub_sect = np.bitwise_and(sub_sect, _use_xcenters)
    sub_sect = np.bitwise_and(sub_sect, _use_ycenters)
    sub_sect = np.bitwise_and(sub_sect, _use_trace_angles)
    sub_sect = np.bitwise_and(sub_sect, _use_tracelengths)

    return sub_sect


fig, axs = plt.subplots(nrows=4, ncols=8)
n_options = len(decor_aper_widths_only)
counter = 0
for idx_split_ in [True, False]:
    for use_xcenters_ in [True, False]:
        for use_ycenters_ in [True, False]:
            for use_trace_angles_ in [True, False]:
                for use_trace_lengths_ in [True, False]:
                    ax = axs.flatten()[counter]
                    counter = counter + 1

                    sub_sect = create_sub_sect(n_options,
                                               idx_split,
                                               use_xcenters,
                                               use_ycenters,
                                               use_trace_angles,
                                               use_trace_lengths)

                    best_res_std_ppm = res_std_ppm[sub_sect].min()

                    aper_widths_ = decor_aper_widths_only[sub_sect]
                    aper_heights_ = decor_aper_heights_only[sub_sect]
                    ax.scatter(aper_widths_, aper_heights_,
                               c=res_std_ppm[sub_sect],
                               marker='s', s=10000)

                    argbest_ppm = res_std_ppm[sub_sect].argmin()
                    best_ppm = res_std_ppm[argbest_ppm]
                    width_best = aper_widths_[argbest_ppm]
                    height_best = aper_heights_[argbest_ppm]

                    txt = f'{best_ppm:.0f} ppm [{width_best}x{height_best}]'
                    out = ax.plot(width_best, height_best, 'o',
                                  color='C1', ms=10)
                    ax.annotate(txt,
                                (width_best + 0.1, height_best + 0.1),
                                # xycoords='axes fraction',
                                xytext=(width_best + 0.1, height_best + 0.1),
                                # textcoords='offset points',
                                ha='left',
                                va='bottom',
                                fontsize=12,
                                color='C1',
                                weight='bold')

                    title = (f'idx_split: {idx_split_} '
                             f'_use_xcenters: {use_xcenters_} '
                             f'_use_ycenters: {use_ycenters_} '
                             f'_use_trace_angles: {use_trace_angles_} '
                             f'_use_trace_lengths :{use_trace_lengths_} '
                             f'width_best: {width_best} '
                             f'height_best: {height_best} '
                             f'best_ppm: {best_ppm} ')

                    best_res_std_ppm = res_std_ppm[sub_sect].min()
                    title = f'{title} : {best_res_std_ppm}'

                    print(title)

                    ax.set_title(title)
                    # plt.colorbar(out)
