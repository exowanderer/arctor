df1 = tbl1.to_pandas()

for kimg in multi_aper_phots[1:]:
    df1 = pd.concat([df1, kimg.to_pandas()])

df_phottable = df1.reset_index().drop(['index', 'id'], axis=1)

meshgrid = np.meshgrid(aper_widths, aper_heights)

trace_width = wasp43.x_right - wasp43.x_left
area_grid = (meshgrid[0] + trace_width) * meshgrid[0]
sky_bg = np.array([skbg * area_grid.flatten() for skbg in wasp43.sky_bgs])

phot_vals = df_phottable.drop(['xcenter', 'ycenter'], axis=1).values - sky_bg
lc_std_rev = phot_vals[wasp43.idx_rev].std(axis=0)
lc_std_fwd = phot_vals[wasp43.idx_fwd].std(axis=0)

lc_med_rev = np.median(phot_vals[wasp43.idx_rev], axis=0)
lc_med_fwd = np.median(phot_vals[wasp43.idx_rev], axis=0)

lc_std = np.mean([lc_std_rev, lc_std_fwd], axis=0)
lc_med = np.mean([lc_med_rev, lc_med_fwd], axis=0)

signal = lc_std / lc_med * 1e6
good = signal < 1000  # ppm
sig_min, sig_max = np.percentile(signal[good], [0.1, 99.9])

max_widths = np.max(meshgrid[0].ravel()[good])
min_widths = np.min(meshgrid[0].ravel()[good])

max_height = np.max(meshgrid[1].ravel()[good])
min_height = np.min(meshgrid[1].ravel()[good])

idx_best = signal.argmin()
width_best = meshgrid[0].ravel()[idx_best]
height_best = meshgrid[1].ravel()[idx_best]
best_ppm = signal[idx_best]

plt.scatter(meshgrid[0].ravel()[good], meshgrid[1].ravel()[
            good], c=(lc_std / lc_med)[good] * 1e6, marker='s', s=1260)

cbar = plt.colorbar()
# cbar.ax.set_yticklabels()
cbar.set_label('Raw Light Curve Std-Dev [ppm]', rotation=270)
cbar.ax.get_yaxis().labelpad = 30

plt.plot(width_best, height_best, 'o', color='C1', ms=10)
plt.annotate(f'{best_ppm:.0f} ppm [{width_best}x{height_best}]',
             (width_best + 1, height_best + 1),
             # xycoords='axes fraction',
             xytext=(width_best + 1, height_best + 1),
             # textcoords='offset points',
             ha='left',
             va='bottom',
             fontsize=12,
             color='C1',
             weight='bold')

plt.xlabel('Aperture Width Outside Trace', fontsize=20)
plt.ylabel('Aperture Height Above Trace', fontsize=20)

plt.xlim(min_widths - 2, max_widths + 2)
plt.ylim(min_height - 5, max_height + 5)

plt.title(
    'Raw Lightcurve Normalized Std-Dev over Height x Width of Aperture', fontsize=20)
plt.tight_layout()
