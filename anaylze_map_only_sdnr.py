import joblib
import numpy as np
from matplotlib import pyplot as plt

# maps_only_filename = 'notebooks/results_decor_span_MAPs_only_SDNR.joblib.save'
# decor_span_MAPs_only_list = joblib.load(maps_only_filename)

decor_aper_columns_list = joblib.load(
    'notebooks/decor_span_MAPs_only_aper_columns_list.joblib.save')

idx_split = np.zeros(len(decor_span_MAPs_only_list), dtype=bool)
use_xcenters = np.zeros(len(decor_span_MAPs_only_list), dtype=bool)
use_ycenters = np.zeros(len(decor_span_MAPs_only_list), dtype=bool)
use_trace_angles = np.zeros(len(decor_span_MAPs_only_list), dtype=bool)
use_trace_lengths = np.zeros(len(decor_span_MAPs_only_list), dtype=bool)

res_std_ppm = np.zeros(len(decor_span_MAPs_only_list))
phots_std_ppm = np.zeros(len(decor_span_MAPs_only_list))
res_diff_ppm = np.zeros(len(decor_span_MAPs_only_list))

fine_grain_mcmcs_s = {}
map_solns = {}
for k, entry in enumerate(decor_span_MAPs_only_list):
    idx_split[k] = entry[0]
    use_xcenters[k] = entry[2]
    use_ycenters[k] = entry[3]
    use_trace_angles[k] = entry[4]
    use_trace_lengths[k] = entry[5]
    fine_grain_mcmcs_ = entry[6]
    map_soln_ = entry[7]
    res_std_ppm[k] = entry[8]
    phots_std_ppm[k] = entry[9]
    res_diff_ppm[k] = entry[10]

    key = (f'idx_split:{use_idx_fwd_}'
           f'_use_xcenters:{use_xcenters_}'
           f'_use_ycenters:{use_ycenters_}'
           f'_use_trace_angles:{use_trace_angles_}'
           f'_use_trace_lengths:{use_trace_lengths_}')

    fine_grain_mcmcs_s[key] = fine_grain_mcmcs_
    map_solns[key] = map_soln_


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

n_options = len(decor_aper_widths_only)
for idx_split_ in [True, False]:
    for use_xcenters_ in [True, False]:
        for use_ycenters_ in [True, False]:
            for use_trace_angles_ in [True, False]:
                for use_trace_lengths_ in [True, False]:
                    plt.figure()
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

                    aper_widths_ = decor_aper_widths_only[sub_sect]
                    aper_heights_ = decor_aper_heights_only[sub_sect]
                    plt.scatter(aper_widths_, aper_heights_,
                                c=res_std_ppm[sub_sect],
                                marker='s', s=10000)

                    argbest_ppm = res_std_ppm[sub_sect].argmin()
                    best_ppm = res_std_ppm[argbest_ppm]
                    width_best = aper_widths_[argbest_ppm]
                    height_best = aper_heights_[argbest_ppm]

                    txt = f'{best_ppm:.0f} ppm [{width_best}x{height_best}]'
                    plt.plot(width_best, height_best, 'o', color='C1', ms=10)
                    plt.annotate(txt,
                                 (width_best + 0.1, height_best + 0.1),
                                 # xycoords='axes fraction',
                                 xytext=(width_best + 0.1, height_best + 0.1),
                                 # textcoords='offset points',
                                 ha='left',
                                 va='bottom',
                                 fontsize=12,
                                 color='C1',
                                 weight='bold')

                    plt.colorbar()
