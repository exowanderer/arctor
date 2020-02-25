# import seaborn as sns

from scipy.interpolate import CubicSpline
from sklearn.externals import joblib
from matplotlib import pyplot as plt
from numpy import array, transpose, arange, append, random
from numpy import loadtxt, sqrt, int32, linspace
from pandas import read_csv, DataFrame


def plot_geometric_albedos_vs_models(models, planet_bayes_test, planet_data,
                                     pub_data=[], deltaTs={}, idx_use=None,
                                     lw0=5, lw1=3, s=1, ylims=[0, 0.7],
                                     xlims=[1000, 2100], text_height=0.7,
                                     alpha=0.03, ms=10, rotation=90,
                                     fontcolor='black', seed=42, fontsize=20,
                                     colors=['blue', 'red'], wiggle=0.1,
                                     squidge=5, save_name=None, show_now=False,
                                     ax=None):
    random.seed(seed)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    for k, model in enumerate(models):
        model = model[['T_eq', 'Albedo']].iloc[idx_use]
        ax.plot(model['T_eq'].values, model['Albedo'].values,
                lw=lw0, color=colors[k])

    for k, val in pub_data.T.iteritems():

        if not val['plot']:
            continue

        pname = val['Planet_Name']

        color_ = val['color']
        alpha_ = val['alpha']
        uplim = val['upperlimit']

        x = val['Thom']  # T_hom
        xerr_l = val['Thom-']  # T_hom lower unc
        xerr_u = val['Thom+']  # T_hom upper unc

        y = val['Ag_hom']  # A_geo_hom
        yerr_l = val['Ag_hom-']  # A_geo_hom lower unc
        yerr_u = val['Ag_hom+']  # A_geo_hom upper unc

        y_mod = 0.01
        x_mod = -7

        if uplim:
            if pname == 'WASP-43 b':
                lw = 2 * lw1
            else:
                lw = lw1

            ax.errorbar(x=x, y=yerr_u, uplims=uplim,
                        xerr=[[xerr_l], [xerr_u]], yerr=yerr_u,
                        fmt='-', color=color_, ms=ms, lw=lw, zorder=50,
                        alpha=alpha_, capsize=0.5 * ms)

            ax.annotate(pname, [x + x_mod, yerr_u + y_mod], zorder=50,
                        fontsize=1.25 * fontsize, rotation=rotation, color=color_)
        else:
            ax.errorbar(x=x, y=y,
                        xerr=[[xerr_l], [xerr_u]], yerr=[[yerr_l], [yerr_u]],
                        fmt='o', color=color_, ms=ms, lw=lw1, zorder=50,
                        alpha=alpha_)

            ax.annotate(pname, [x + x_mod, y + yerr_u + y_mod], zorder=50,
                        fontsize=1.25 * fontsize, rotation=rotation, color=color_)

    for pname, val in planet_bayes_test.items():
        ax.scatter(val['Tp'], val['ColdTrap']['AlbedoRaw'],
                   s=s, alpha=alpha, color=colors[0])
        ax.scatter(val['Tp'], val['MgSiO3']['AlbedoRaw'],
                   s=s, alpha=alpha, color=colors[1])

    logg_s = array([val['logg'] for key, val in planet_data.items()])

    if len(planet_data.items()):
        logg_s = (logg_s - logg_s.min()) / (logg_s.max() - logg_s.min())

    for (pname, val), logg in zip(planet_data.items(), logg_s):
        albedo_coldtrap = random.normal(val['coldtrap'],
                                        wiggle * val['geo_alb_unc'])
        albedo_mgsio3 = random.normal(
            val['mgsio3'], wiggle * val['geo_alb_unc'])
        ax.errorbar(val['Tp'] - squidge, albedo_coldtrap, val['geo_alb_unc'],
                    50, fmt='o', ms=ms * (1 + logg), lw=lw1, color=colors[0],
                    zorder=100)
        ax.errorbar(val['Tp'] + squidge, albedo_mgsio3, val['geo_alb_unc'], 50,
                    fmt='o', ms=ms * (1 + logg), lw=lw1, color=colors[1],
                    zorder=100)

    zipper = zip(planet_data.items(), deltaTs.items())
    for (pname, val), (pname1, deltaT) in zipper:
        assert(pname == pname1)
        ax.annotate(pname, [val['Tp'] - deltaT, text_height],
                    fontsize=fontsize, rotation=rotation, color=fontcolor)

    zipper = zip(planet_data.items(), deltaTs.items())
    for (pname, val), (pname1, deltaT) in zipper:
        assert(pname == pname1)
        ax.annotate(pname, [val['Tp'] - deltaT, text_height],
                    fontsize=fontsize, rotation=rotation, color=fontcolor)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    ax.set_ylabel('Geometric Albedo', fontsize=fontsize)
    ax.set_xlabel('Equilibrium Temperature [k]', fontsize=fontsize)
    ax.set_ylim(ylims)
    ax.set_xlim(xlims)
    ax.grid(True)
    # mng = plt.get_current_fig_manager()
    # mng.window.showMaximized()
    # plt.tight_layout()

    if len(models):
        ax.plot([], [], color=colors[1], lw=lw0,
                label='MgSiO3 Predictions')
        ax.plot([], [], color=colors[0], lw=lw0,
                label='ColdTrap Predictions')

    pub_colors = pub_data['color'].values
    ax.errorbar([], [], [], color=pub_colors[0], lw=lw1, ms=ms * 3,
                label='Kepler (600nm)')

    df_view_ = pub_data[['Planet_Name', 'color', 'Instrument']]
    for _, pname_, color_, instr_ in df_view_.itertuples():
        if 'kepler' not in pname_.lower() and 'koi' not in pname_.lower()\
                and pname_ != 'HAT-P-7 b':
            label = f'{pname_} {instr_}'
            ax.errorbar([], [], [], color=color_, lw=lw1, ms=ms * 3,
                        label=label)

    leg_kwargs = {'loc': 2,
                  'fontsize': fontsize,
                  'ncol': 1,
                  # 'bbox_to_anchor': (1.025, 1.075)
                  # 'fancybox':  True,
                  # 'framealpha':  0.5,
                  # 'shadow':  True,
                  # 'borderpad':  1
                  }
    leg = ax.legend(**leg_kwargs)

    leg.set_zorder(200)
    if save_name is not None:
        print('[INFO] Saving image to {}'.format(save_name))
        plt.savefig(save_name)

    if show_now:
        plt.show()


def plot_Teq_vs_logg(planet_data, deltaLoggs=None, fontsize=20,
                     rotation=90, ylims=None, xlims=None,
                     text_height=3.9, fontcolor='k', size=100,
                     save_name=None, yticks=None, xticks=None,
                     ax=None):

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    if deltaLoggs is None:
        deltaLoggs = {p: 0 for p in planet_data.keys()}

    logg_s = [val['logg'] for key, val in planet_data.items()]
    T_eq_s = [val['Tp'] for key, val in planet_data.items()]
    ax.scatter(T_eq_s, logg_s, s=size)

    zipper = zip(planet_data.items(), deltaLoggs.items())
    for (pname, val), (pname1, deltaLogg) in zipper:
        assert(pname == pname1)
        ax.annotate(pname, [val['Tp'], val['logg'] + deltaLogg],
                    fontsize=fontsize, rotation=0, color=fontcolor)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    if yticks is not None:
        ax.set_yticks(yticks)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.set_ylabel('Gravity [log(g) cgs]', fontsize=fontsize)
    ax.set_xlabel('Equilibrium Temperature [K]', fontsize=fontsize)
    ax.set_ylim(ylims)
    ax.set_xlim(xlims)

    # mng = plt.get_current_fig_manager()
    # mng.window.showMaximized()
    # plt.tight_layout()

    if save_name is not None:
        plt.savefig(save_name)
    if show_now:
        plt.show()

plt.rcParams['text.usetex'] = False
plt.rcParams['axes.facecolor'] = plt.rcParams['figure.facecolor']
# plt.rcParams['axes.edgecolor'] = plt.rcParams['figure.facecolor']
plt.rcParams['axes.grid'] = True
# plt.rcParams['axes.grid.axis'] = True
plt.rcParams['xtick.major.size'] = 12.
plt.rcParams['xtick.major.width'] = 2.
plt.rcParams['ytick.major.width'] = 2.
plt.rcParams['ytick.major.size'] = 12.
plt.rcParams['xtick.minor.visible'] = False
plt.rcParams['ytick.minor.visible'] = False
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True

colorcycle = array(plt.rcParams['axes.prop_cycle'].by_key()['color'])

mgsio3_curve = read_csv('MgSiO3_albedo_vs_temperature.csv',
                        names=['T_eq', 'Albedo'])
coldtrap_curve = read_csv('ColdTrap_albedo_vs_temperature.csv',
                          names=['T_eq', 'Albedo'])

cspline_mgsio3 = CubicSpline(mgsio3_curve['T_eq'], mgsio3_curve['Albedo'])
cspline_coldtrap = CubicSpline(
    coldtrap_curve['T_eq'], coldtrap_curve['Albedo'])

coldtrap_curve_new = cspline_coldtrap(mgsio3_curve['T_eq'].values)

inputs = transpose([mgsio3_curve['T_eq'].values, coldtrap_curve_new])
coldtrap_curve = DataFrame(inputs, columns=list(mgsio3_curve.columns))

joblib_load_name = 'geometric_albedo_input_data_{}.joblib.save'

edepth_uncs = joblib.load(joblib_load_name.format('edepth_unc'))
exoplanet_info = joblib.load(joblib_load_name.format('exoplanet_info'))
planet_data = joblib.load(joblib_load_name.format('planet_data'))
planet_bayes_test = joblib.load(joblib_load_name.format('planet_bayes_test'))

planet_data['HD 209458b'] = {}
planet_data['HD 209458b']['Tp'] = 1476.81
planet_data['HD 209458b']['RpRs'] = 0.120033328704989
planet_data['HD 209458b']['ApRs'] = 8.506
planet_data['HD 209458b']['eDepth_unc'] = None
planet_data['HD 209458b']['logg'] = 2.9715
planet_data['HD 209458b']['Tp_unc'] = 12
planet_data['HD 209458b']['RpRs_unc'] = None
planet_data['HD 209458b']['ApRs_unc'] = None
planet_data['HD 209458b']['mgsio3'] = None
planet_data['HD 209458b']['coldtrap'] = None
planet_data['HD 209458b']['geo_alb_unc'] = None


black_dots = read_csv('esteves_table_of_geometric_albedos.csv', delimiter='\t')
black_dots['color'] = ['black'] * len(black_dots)
black_dots['alpha'] = [1.0] * len(black_dots)
black_dots['upperlimit'] = [False] * len(black_dots)

n = 8
color = plt.cm.plasma(linspace(0.1, 0.75, n))
color = int32(color[:, 0:-1] * 255)
hexcolor = ['#%02x%02x%02x' % (rgb[0], rgb[1], rgb[2]) for rgb in color]


hd189stis = read_csv('hd189733b_stis_eclipse_spectrum.csv')
hd189stis['uncAg'] = hd189stis[['Ag-', 'Ag+']].mean(axis=1)
hd189point = (hd189stis['Ag'] / hd189stis['uncAg']).sum()
hd189point = hd189point / (1 / hd189stis['uncAg']).sum()
hd189_unc = hd189stis['uncAg'].mean() / sqrt(len(hd189stis['uncAg']))

# hd189point = 0.15867990829405226

qatar2b = {'Planet Name': 'Qatar-2 b',
           'Ag,max': None, 'Ag,max-': None, 'Ag,max+': None,
           'Tmax': None, 'Tmax-': None, 'Tmax+': None,
           'Ag,hom': 0, 'Ag,hom-': 0,
           'Ag,hom+': 0.09,
           'Thom': 1344,
           'Thom-': 14,
           'Thom+': 14,
           'fref,max': None, 'fref,max-': None, 'fref,max+': None,
           'fref,hom': None, 'fref,hom-': None, 'fref,hom+': None,
           'plot': True,
           'color': hexcolor[-2],
           'alpha': 1.0,
           'upperlimit': True}

wasp104b = {'Planet Name': 'WASP-104 b',
            'Ag,max': None, 'Ag,max-': None, 'Ag,max+': None,
            'Tmax': None, 'Tmax-': None, 'Tmax+': None,
            'Ag,hom': 0.0211,
            'Ag,hom-': 0.0068,
            'Ag,hom+': 0.0068,
            # 'Thom':1422,
            'Thom': planet_data['WASP-104b']['Tp'],
            'Thom-': 20,
            'Thom+': 20,
            'fref,max': None,
            'fref,max-': None, 'fref,max+': None,
            'fref,hom': None, 'fref,hom-': None, 'fref,hom+': None,
            'plot': True,
            'color': hexcolor[-3],
            'alpha': 1.0,
            'upperlimit': False}

k2_141b = {'Planet Name': 'K2-141 b',
           'Ag,max': None, 'Ag,max-': None, 'Ag,max+': None,
           'Tmax': None, 'Tmax-': None, 'Tmax+': None,
           'Ag,hom': 0.205,
           'Ag,hom-': 0.059,
           'Ag,hom+': 0.077,
           'Thom': 1984,
           'Thom-': 108,
           'Thom+': 59,
           'fref,max': None,
           'fref,max-': None, 'fref,max+': None,
           'fref,hom': None, 'fref,hom-': None, 'fref,hom+': None,
           'plot': True,
           'color': hexcolor[-4],
           'alpha': 1.0,
           'upperlimit': False}

hd189b = {'Planet Name': 'HD 189733 b',
          'Ag,max': None, 'Ag,max-': None, 'Ag,max+': None,
          'Tmax': None, 'Tmax-': None, 'Tmax+': None,
          'Ag,hom': hd189point,
          'Ag,hom-': hd189_unc,
          'Ag,hom+': hd189_unc,
          'Thom': 1201,
          'Thom-': 12,
          'Thom+': 13,
          'fref,max': None,
          'fref,max-': None, 'fref,max+': None,
          'fref,hom': None, 'fref,hom-': None, 'fref,hom+': None,
          'plot': True,
          'color': hexcolor[-5],
          'alpha': 1.0,
          'upperlimit': False}

corot2b = {'Planet Name': 'Corot-2 b',
           'Ag,max': None, 'Ag,max-': None, 'Ag,max+': None,
           'Tmax': None, 'Tmax-': None, 'Tmax+': None,
           'Ag,hom': 0.09688,
           'Ag,hom-': 0.11,
           'Ag,hom+': 0.1938,
           'Thom': planet_data['Corot-2b']['Tp'] + 5,
           'Thom-': 18,
           'Thom+': 18,
           'fref,max': None,
           'fref,max-': None, 'fref,max+': None,
           'fref,hom': None, 'fref,hom-': None, 'fref,hom+': None,
           'plot': True,
           'color': hexcolor[-1],
           'alpha': 1.0,
           'upperlimit': True}

hd209b = {'Planet Name': 'HD 209458 b',
          'Ag,max': None, 'Ag,max-': None, 'Ag,max+': None,
          'Tmax': None, 'Tmax-': None, 'Tmax+': None,
          'Ag,hom': 0, 'Ag,hom-': 0,
          'Ag,hom+': 0.25,
          'Thom': planet_data['HD 209458b']['Tp'] + 0,
          'Thom-': 12,
          'Thom+': 12,
          'fref,max': None,
          'fref,max-': None, 'fref,max+': None,
          'fref,hom': None, 'fref,hom-': None, 'fref,hom+': None,
          'plot': True,
          'color': hexcolor[-7],
          'alpha': 1.0,
          'upperlimit': True}

wasp43b = {'Planet Name': 'WASP-43 b',
           'Ag,max': None, 'Ag,max-': None, 'Ag,max+': None,
           'Tmax': None, 'Tmax-': None, 'Tmax+': None,
           'Ag,hom': 0, 'Ag,hom-': 0,
           'Ag,hom+': 0.06365027580163687,
           'Thom': planet_data['WASP-43b']['Tp'] + 0,
           'Thom-': 9,
           'Thom+': 9,
           'fref,max': None,
           'fref,max-': None, 'fref,max+': None,
           'fref,hom': None, 'fref,hom-': None, 'fref,hom+': None,
           'plot': True,
           'color': hexcolor[-8],
           'alpha': 1.0,
           'upperlimit': True}

black_dots = black_dots.append(qatar2b, ignore_index=True)
black_dots = black_dots.append(wasp104b, ignore_index=True)
black_dots = black_dots.append(k2_141b, ignore_index=True)
black_dots = black_dots.append(hd189b, ignore_index=True)
black_dots = black_dots.append(corot2b, ignore_index=True)
black_dots = black_dots.append(hd209b, ignore_index=True)
black_dots = black_dots.append(wasp43b, ignore_index=True)

black_dots['Planet_Name'] = black_dots['Planet Name']
del black_dots['Planet Name']

for colname in black_dots.columns:
    if ',' in colname:
        black_dots[colname.replace(',', '_')] = black_dots[colname]
        del black_dots[colname]

idx_hd209 = int32(black_dots.query('Planet_Name == "HD 209458 b"').index)[0]
idx_qatar2 = int32(black_dots.query('Planet_Name == "Qatar-2 b"').index)[0]
idx_wasp104 = int32(black_dots.query('Planet_Name == "WASP-104 b"').index)[0]
idx_k2141 = int32(black_dots.query('Planet_Name == "K2-141 b"').index)[0]
idx_hd189 = int32(black_dots.query('Planet_Name == "HD 189733 b"').index)[0]
idx_corot2 = int32(black_dots.query('Planet_Name == "Corot-2 b"').index)[0]
idx_tres2 = int32(black_dots.query('Planet_Name == "TrES-2 b"').index)[0]
idx_wasp43 = int32(black_dots.query('Planet_Name == "WASP-43 b"').index)[0]
idx_koi13bb = int32(black_dots.query('Planet_Name == "KOI-13 bb"').index)[0]

black_dots.loc[idx_tres2, 'color'] = hexcolor[-6]
black_dots.loc[idx_koi13bb, 'Planet_Name'] = 'KOI-13 b'

black_dots['Instrument'] = ['(Kepler:600nm)'] * len(black_dots)

black_dots.loc[idx_hd209, 'Instrument'] = '(MOST:550nm)'
black_dots.loc[idx_qatar2, 'Instrument'] = '(K2:600nm)'
black_dots.loc[idx_wasp104, 'Instrument'] = '(K2:600nm)'
black_dots.loc[idx_k2141, 'Instrument'] = '(K2:600nm)'
black_dots.loc[idx_hd189, 'Instrument'] = '(STIS:500nm)'
black_dots.loc[idx_corot2, 'Instrument'] = '(CoRoT:700nm)'
black_dots.loc[idx_tres2, 'Instrument'] = '(Kepler:600nm)'
black_dots.loc[idx_wasp43, 'Instrument'] = '(UVIS:584nm; This Work)'

wiggle = 0.2
squidge = 1
ylims = [-0.01, 0.55]
xlims = [1100, 2050]
text_height = 0.799
colors = colorcycle[[0, 1]]
show_now = True
ms = 10

hd189733 = array([.4, 0, .45, .39, .32, .17, -.11, .02])
hd189733_u = array([.12, .12, .55, .27, .15, .12, .14, .14])

deltaLoggs = {'WASP-52b': 0.02,
              # 'Qatar-2b':5,
              'WASP-140b': 0.02,
              'WASP-43b': 0.02,
              'Qatar-1b': 0.02,
              'WASP-104b': 0.02,
              'Corot-2b': 0.02,
              'WASP-95b': -0.06,
              'WASP-77Ab': 0.02,
              'TrES-3b': 0.02}

ntargs = len(deltaLoggs)

print('[INFO] Commensing Logg vs T_eq Plot Without Point')
yticks = arange(2.9, 4.0, 0.1)
xticks = arange(1300, 1800, 100)

"""
save_name = 'gravity_vs_T_eq_for_{}_targets.png'.format(ntargs)
plot_Teq_vs_logg(planet_data, deltaLoggs = deltaLoggs, text_height = 3.8,
                    size=200, save_name = save_name, 
                    yticks=yticks, xticks=xticks)
"""
models = [coldtrap_curve, mgsio3_curve]

Temp1 = 1600
Temp2 = 1800
idx1 = abs(models[0]['T_eq'] - Temp1).idxmin()
idx2 = abs(models[0]['T_eq'] - Temp2).idxmin()

idx_use = arange(0, idx1, 50)
idx_use = append(idx_use, arange(idx1, idx2, 20))
idx_use = append(idx_use, arange(idx2, len(models[0]), 50))

""" Plot without the point cloud"""
save_name = 'geometric_albedo_with_errorbars_for_{}_targets_with{}_points.png'

try:
    ax.clear()
except:
    fig, ax = plt.subplots()

deltaTs = {'WASP-52b': 20,
           # 'Qatar-2b':5,
           'WASP-140b': -5,
           'WASP-43b': 20,
           'Qatar-1b': 0,
           'WASP-104b': 20,
           'Corot-2b': 0,
           'WASP-95b': 20,
           'WASP-77Ab': 0,
           'TrES-3b': 0}


print('[INFO] Commensing Plot Without Point')
plot_geometric_albedos_vs_models(models=[],
                                 planet_bayes_test={},
                                 planet_data={},
                                 pub_data=black_dots,
                                 deltaTs=deltaTs,
                                 idx_use=idx_use,
                                 ms=ms,
                                 wiggle=wiggle,
                                 squidge=wiggle,
                                 ylims=ylims,
                                 xlims=xlims,
                                 text_height=text_height,
                                 colors=hexcolor,
                                 show_now=show_now,
                                 ax=ax,
                                 save_name=None)
