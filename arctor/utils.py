from . import *
from .arctor import *
# from . import Arctor

import exoplanet as xo
import numpy as np
import os
import pygtc
import pymc3 as pm
import starry
import theano.tensor as tt

from statsmodels.robust.scale import mad

from astropy.io import fits
from astropy.stats import mad_std, sigma_clip
from astropy.time import Time
from astropy import units

from tqdm import tqdm


def debug_message(message, end='\n'):
    print(f'[DEBUG] {message}', end=end)


def warning_message(message, end='\n'):
    print(f'[WARNING] {message}', end=end)


def info_message(message, end='\n'):
    print(f'[INFO] {message}', end=end)


def create_raw_lc_stddev(planet):
    ppm = 1e6
    phot_vals = planet.photometry_df
    lc_std_rev = phot_vals.iloc[planet.idx_rev].std(axis=0)
    lc_std_fwd = phot_vals.iloc[planet.idx_fwd].std(axis=0)

    lc_med_rev = np.median(phot_vals.iloc[planet.idx_rev], axis=0)
    lc_med_fwd = np.median(phot_vals.iloc[planet.idx_rev], axis=0)

    lc_std = np.mean([lc_std_rev, lc_std_fwd], axis=0)
    lc_med = np.mean([lc_med_rev, lc_med_fwd], axis=0)

    return lc_std / lc_med * ppm


# def get_flux_idx_from_df(planet, aper_width, aper_height):
#     # There *must* be a faster way!
#     aperwidth_columns = [colname
#                          for colname in planet.photometry_df.columns
#                          if 'aper_width' in colname]

#     aperheight_columns = [colname
#                           for colname in planet.photometry_df.columns
#                           if 'aper_height' in colname]

#     trace_length = np.median(planet.trace_lengths) - 0.1

#     aperwidths_df = (planet.photometry_df[aperwidth_columns] - trace_length)
#     aperwidths_df = aperwidths_df.astype(int)

#     aperheight_df = planet.photometry_df[aperheight_columns].astype(int)
#     aperwidth_flag = aperwidths_df.values[0] == aper_width
#     aperheight_flag = aperheight_df.values[0] == aper_height

#     return np.where(aperwidth_flag * aperheight_flag)  # [0][0]


def print_flux_stddev(planet, aper_width, aper_height):
    # There *must* be a faster way!
    fluxes = planet.photometry_df[f'aperture_sum_{aper_width}x{aper_height}']
    fluxes = fluxes / np.median(fluxes)

    info_message(f'{aper_width}x{aper_height}: {np.std(fluxes)*1e6:0.0f} ppm')


def find_flux_stddev(planet, flux_std, aper_widths, aper_heights):
    # There *must* be a faster way!
    for aper_width in tqdm(aper_widths):
        for aper_height in tqdm(aper_heights):
            flux_key = f'aperture_sum_{aper_width}x{aper_height}'
            fluxes = planet.photometry_df[flux_key]
            fluxes = fluxes / np.median(fluxes)

            if np.std(fluxes) * 1e6 < flux_std:
                info_message(f'{aper_width}x{aper_height}: '
                             f'{np.std(fluxes)*1e6:0.0f} ppm')


def setup_and_plot_GTC(mcmc_fit, plotName='',
                       varnames=None,
                       smoothingKernel=1,
                       square_edepth=False):

    trace = mcmc_fit['trace']
    map_soln = mcmc_fit['map_soln']

    if trace is None:
        return

    if varnames is None:
        varnames = [key for key in map_soln.keys()
                    if '__' not in key and 'light' not in key
                    and 'line' not in key
                    and 'le_edepth_0' not in key]

    samples = pm.trace_to_dataframe(trace, varnames=varnames)

    varnames = [key for key in map_soln.keys()
                if '__' not in key and 'light' not in key
                and 'line' not in key]

    truths = [float(val) for key, val in map_soln.items() if key in varnames]
    pygtc.plotGTC(samples, plotName=plotName,  # truths=truths,
                  smoothingKernel=smoothingKernel,
                  labelRotation=[True] * 2,
                  customLabelFont={'rotation': 45},
                  nContourLevels=3, figureSize='MNRAS_page')


def run_pymc3_multi_dataset(times, data, dataerr, t0, u, period, b,
                            idx_fwd, idx_rev, random_state=42,
                            xcenters=None, tune=5000, draws=5000,
                            target_accept=0.9, do_mcmc=True,
                            use_log_edepth=False,
                            allow_negative_edepths=False):

    with pm.Model() as model:

        # The baseline flux
        mean = pm.Normal("mean", mu=1.0, sd=1.0, shape=2,
                         testval=np.array([0.999, 1.001]))

        if use_log_edepth:
            edepth = pm.Uniform("log_edepth", lower=-20, upper=-2)
            edepth = pm.Deterministic("edepth", pm.math.exp(logP))
            edepth = 10**(0.5 * edepth)
        else:
            if allow_negative_edepths:
                edepth = pm.Uniform("edepth", lower=-0.01, upper=0.01)
                edepth_sign = pm.math.sgn(edepth)
                if pm.math.lt(edepth_sign, 0):
                    edepth = -pm.math.sqrt(pm.math.abs_(edepth))
                else:
                    edepth = pm.math.sqrt(pm.math.abs_(edepth))
            else:
                edepth = pm.Uniform("edepth", lower=0, upper=0.01)
                edepth = pm.math.sqrt(edepth)

        # Set up a Keplerian orbit for the planets
        orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, b=b)

        # Compute the model light curve using starry
        light_curves = xo.LimbDarkLightCurve(u).get_light_curve(
            orbit=orbit, r=edepth, t=t)
        light_curve = pm.math.sum(light_curves, axis=-1) + mean

        # Here we track the value of the model light curve for plotting
        # purposes
        pm.Deterministic("light_curves", light_curves)

        # The likelihood function assuming known Gaussian uncertainty
        pm.Normal("obs", mu=light_curve, sd=dataerr, observed=data)

        # Fit for the maximum a posteriori parameters given the simuated
        # dataset
        map_soln = xo.optimize(start=model.test_point)

        np.random.seed(random_state)

        trace = pm.sample(
            tune=tune,
            draws=draws,
            start=map_soln,
            chains=mp.cpu_count(),
            step=xo.get_dense_nuts_step(target_accept=target_accept),
            cores=mp.cpu_count()
        )

    return trace, map_soln


def run_pymc3_fwd_rev(times, data, dataerr, t0, u, period, b, idx_fwd, idx_rev,
                      xcenters=None, tune=5000, draws=5000, target_accept=0.9,
                      do_mcmc=True, use_log_edepth=False,
                      allow_negative_edepths=False):

    times_bg = times - np.median(times)
    with pm.Model() as model:

        # The baseline flux
        mean_fwd = pm.Normal("mean_fwd", mu=1.0, sd=1.0)
        mean_rev = pm.Normal("mean_rev", mu=1.0, sd=1.0)

        assert(not (allow_negative_edepths and use_log_edepth)),\
            'Cannot have `allow_negative_edepths` with `use_log_edepth`'

        if use_log_edepth:
            log_edepth = pm.Uniform("log_edepth", lower=-20, upper=-2)
            edepth = pm.Deterministic("edepth", 10**(0.5 * log_edepth))
        else:
            if allow_negative_edepths:
                edepth = pm.Uniform("edepth", lower=-0.01, upper=0.01)
            else:
                edepth = pm.Uniform("edepth", lower=0, upper=0.01)
                edepth = pm.math.sqrt(edepth)

        slope_time = pm.Uniform("slope_time", lower=-0.1, upper=0.1)
        line_fwd = mean_fwd + slope_time * times_bg[idx_fwd]
        line_rev = mean_rev + slope_time * times_bg[idx_rev]

        if xcenters is not None:
            slope_xc = pm.Uniform("slope_xcenter", lower=-0.1, upper=0.1)
            line_fwd = line_fwd + slope_xc * xcenters[idx_fwd]
            line_rev = line_rev + slope_xc * xcenters[idx_rev]

        # Set up a Keplerian orbit for the planets
        orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, b=b)

        # # Compute the model light curve using starry
        star = xo.LimbDarkLightCurve(u)
        light_curves_fwd = star.get_light_curve(
            orbit=orbit, r=edepth, t=times[idx_fwd])
        light_curves_rev = star.get_light_curve(
            orbit=orbit, r=edepth, t=times[idx_rev])

        light_curve_fwd = pm.math.sum(light_curves_fwd, axis=-1)
        light_curve_rev = pm.math.sum(light_curves_rev, axis=-1)

        # # Here we track the value of the model light curve for plotting
        # # purposes
        pm.Deterministic("light_curves_fwd", light_curves_fwd)
        pm.Deterministic("light_curves_rev", light_curves_rev)

        # # The likelihood function assuming known Gaussian uncertainty
        pm.Normal("obs_fwd", mu=light_curve_fwd + line_fwd,
                  sd=dataerr[idx_fwd], observed=data[idx_fwd])
        pm.Normal("obs_rev", mu=light_curve_rev + line_rev,
                  sd=dataerr[idx_rev], observed=data[idx_rev])

        # Fit for the maximum a posteriori parameters
        #   given the simuated dataset
        map_soln = xo.optimize(start=model.test_point)
        if use_log_edepth:
            map_soln_edepth = 10**map_soln["log_edepth"]
        else:
            map_soln_edepth = map_soln["edepth"]

        info_message(f'map_soln_edepth:{map_soln_edepth*1e6}')

        np.random.seed(42)
        if do_mcmc:
            trace = pm.sample(
                tune=tune,
                draws=tune,
                start=map_soln,
                chains=mp.cpu_count(),
                step=xo.get_dense_nuts_step(target_accept=target_accept),
                cores=mp.cpu_count()
            )
        else:
            trace = None

    return trace, map_soln


def run_pymc3_direct(times, data, dataerr, t0, u, period, b, xcenters=None,
                     tune=5000, draws=5000, target_accept=0.9, do_mcmc=True,
                     use_log_edepth=False, allow_negative_edepths=False):

    times_bg = times - np.median(times)

    with pm.Model() as model:

        # The baseline flux
        mean = pm.Normal("mean", mu=1.0, sd=1.0)

        assert(not (allow_negative_edepths and use_log_edepth)),\
            'Cannot have `allow_negative_edepths` with `use_log_edepth`'

        if use_log_edepth:
            log_edepth = pm.Uniform("log_edepth", lower=-20, upper=-2)
            edepth = pm.Deterministic("edepth", 10**(0.5 * log_edepth))
        else:
            if allow_negative_edepths:
                edepth = pm.Uniform("edepth", lower=-0.01, upper=0.01)
            else:
                edepth = pm.Uniform("edepth", lower=0, upper=0.01)
                edepth = pm.math.sqrt(edepth)

        slope_time = pm.Uniform("slope_time", lower=-0.1, upper=0.1)
        line = mean + slope_time * times_bg

        if xcenters is not None:
            slope_xc = pm.Uniform("slope_xcenter", lower=-0.1, upper=0.1)
            line = line + slope_xc * xcenters

        # Set up a Keplerian orbit for the planets
        orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, b=b)

        # Compute the model light curve using starry
        light_curves = xo.LimbDarkLightCurve(
            u).get_light_curve(orbit=orbit, r=edepth, t=times)
        light_curve = pm.math.sum(light_curves, axis=-1)
        pm.Deterministic("light_curves", light_curves)

        # Combined model: light curve and background
        model_ = light_curve + line

        # The likelihood function assuming known Gaussian uncertainty
        pm.Normal("obs", mu=model_, sd=dataerr, observed=data)

        # Fit for the maximum a posteriori parameters given the simuated
        # dataset

        map_soln = xo.optimize(start=model.test_point)
        if use_log_edepth:
            map_soln_edepth = 10**map_soln["log_edepth"]
        else:
            map_soln_edepth = map_soln["edepth"]

        info_message(f'map_soln_edepth:{map_soln_edepth*1e6}')

        line_map_soln = (map_soln['mean'] +
                         map_soln['slope_time'] * times_bg.flatten())
        if xcenters is not None:
            line_map_soln = line_map_soln + \
                map_soln['slope_xcenter'] * xcenters

        np.random.seed(42)
        if do_mcmc:
            trace = pm.sample(
                tune=tune,
                draws=draws,
                start=map_soln,
                chains=mp.cpu_count(),
                step=xo.get_dense_nuts_step(target_accept=target_accept),
                cores=mp.cpu_count()
            )
        else:
            trace = None

    return trace, map_soln


def build_gp_pink_noise(times, data, dataerr,
                        log_Q=np.log(1.0 / np.sqrt(2))):

    log_w0 = pm.Normal("log_w0", mu=0.0, sigma=15.0,
                       testval=np.log(3.0))
    log_Sw4 = pm.Normal("log_variance_r", mu=0.0, sigma=15.0,
                        testval=np.log(np.var(data)))
    log_s2 = pm.Normal("log_variance_w", mu=0.0, sigma=15.0,
                       testval=np.log(np.var(data)))

    kernel = xo.gp.terms.SHOTerm(
        log_Sw4=log_Sw4, log_w0=log_w0, log_Q=log_Q)

    return xo.gp.GP(kernel, times, dataerr ** 2 + pm.math.exp(log_s2))


def run_pymc3_both(times, data, dataerr, t0, u, period, b,
                   xcenters=None, ycenters=None,
                   trace_angles=None, trace_lengths=None, log_Q=1 / np.sqrt(2),
                   idx_fwd=None, idx_rev=None, tune=5000, draws=5000,
                   target_accept=0.9, do_mcmc=True, use_log_edepth=False,
                   allow_negative_edepths=False, use_pink_gp=False):

    if idx_fwd is None or idx_rev is None:
        # Make use of idx_fwd and idx_rev trivial
        idx_fwd = np.ones_like(times, dtype=bool)

        strfwd = ''
        strrev = ''
    else:
        assert(len(idx_fwd) + len(idx_rev) == len(times)),\
            f"`idx_fwd` + `idx_rev` must include all idx from `times`"

        strfwd = '_fwd'
        strrev = '_rev'

    times_bg = times - np.median(times)
    with pm.Model() as model:

        # The baseline flux
        mean_fwd = pm.Normal(f"mean{strfwd}", mu=0.0, sd=1.0)

        if idx_rev is not None:
            mean_rev = pm.Normal(f"mean{strrev}", mu=0.0, sd=1.0)

        assert(not (allow_negative_edepths and use_log_edepth)),\
            'Cannot have `allow_negative_edepths` with `use_log_edepth`'

        if use_log_edepth:
            log_edepth = pm.Uniform("log_edepth", lower=-20, upper=-2)
            edepth = pm.Deterministic("edepth", 10**(0.5 * log_edepth))
        else:
            if allow_negative_edepths:
                edepth = pm.Uniform("edepth", lower=-0.01, upper=0.01)
            else:
                edepth = pm.Uniform("edepth", lower=0, upper=0.01)
                edepth = pm.math.sqrt(edepth)

        slope_time = pm.Uniform("slope_time", lower=-1, upper=1)
        line_fwd = mean_fwd + slope_time * times_bg[idx_fwd]

        if idx_rev is not None:
            line_rev = mean_rev + slope_time * times_bg[idx_rev]

        if xcenters is not None:
            slope_xc = pm.Uniform("slope_xcenter", lower=-1, upper=1)
            line_fwd = line_fwd + slope_xc * xcenters[idx_fwd]

            if idx_rev is not None:
                line_rev = line_rev + slope_xc * xcenters[idx_rev]

        if ycenters is not None:
            slope_yc = pm.Uniform("slope_ycenter", lower=-1, upper=1)
            line_fwd = line_fwd + slope_yc * ycenters[idx_fwd]

            if idx_rev is not None:
                line_rev = line_rev + slope_yc * ycenters[idx_rev]

        if trace_angles is not None:
            slope_ta = pm.Uniform("slope_trace_angle", lower=-1, upper=1)
            line_fwd = line_fwd + slope_ta * trace_angles[idx_fwd]

            if idx_rev is not None:
                line_rev = line_rev + slope_ta * trace_angles[idx_rev]

        if trace_lengths is not None:
            slope_tl = pm.Uniform("slope_trace_length", lower=-1, upper=1)
            line_fwd = line_fwd + slope_tl * trace_lengths[idx_fwd]

            if idx_rev is not None:
                line_rev = line_rev + slope_tl * trace_lengths[idx_rev]

        pm.Deterministic(f'line_model{strfwd}', line_fwd)
        if idx_rev is not None:
            pm.Deterministic(f'line_model{strrev}', line_rev)

        # Set up a Keplerian orbit for the planets
        orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, b=b)

        # # Compute the model light curve using starry
        star = xo.LimbDarkLightCurve(u)
        light_curves_fwd = star.get_light_curve(
            orbit=orbit, r=edepth, t=times[idx_fwd])

        if idx_rev is not None:
            light_curves_rev = star.get_light_curve(
                orbit=orbit, r=edepth, t=times[idx_rev])

        light_curve_fwd = pm.math.sum(light_curves_fwd, axis=-1)

        if idx_rev is not None:
            light_curve_rev = pm.math.sum(light_curves_rev, axis=-1)

        # # Here we track the value of the model light curve for plotting
        # # purposes
        pm.Deterministic(f"light_curves{strfwd}", light_curve_fwd)
        if idx_rev is not None:
            pm.Deterministic(f"light_curves{strrev}", light_curve_rev)

        # The likelihood function assuming known Gaussian uncertainty
        model_fwd = light_curve_fwd + line_fwd
        if idx_rev is not None:
            model_rev = light_curve_rev + line_rev

        if use_pink_gp:
            gp = build_gp_pink_noise(
                times, data, dataerr, log_Q=log_Q)
            # gp = build_gp_pink_noise(times, data, dataerr, log_Q=log_Q)
            # mu, _ = xo.eval_in_model(model.test_point)

            gp.marginal("gp", observed=data - model_fwd.flatten())
        else:
            pm.Normal(f"obs{strfwd}", mu=model_fwd,
                      sd=dataerr[idx_fwd], observed=data[idx_fwd])

            if idx_rev is not None:
                pm.Normal(f"obs{strrev}", mu=light_curve_rev + line_rev,
                          sd=dataerr[idx_rev], observed=data[idx_rev])

        # Fit for the maximum a posteriori parameters
        #   given the simuated dataset
        map_soln = xo.optimize(start=model.test_point)
        if use_log_edepth:
            map_soln_edepth = 10**map_soln["log_edepth"]
        else:
            map_soln_edepth = map_soln["edepth"]

        info_message(f'Map Soln Edepth:{map_soln_edepth*1e6}')

        np.random.seed(42)
        if do_mcmc:
            trace = pm.sample(
                tune=tune,
                draws=tune,
                start=map_soln,
                chains=mp.cpu_count(),
                step=xo.get_dense_nuts_step(target_accept=target_accept),
                cores=mp.cpu_count()
            )
        else:
            trace = None

    return trace, map_soln


def run_pymc3_w_gp(times, data, dataerr, t0, u, period, b,
                   xcenters=None, ycenters=None,
                   trace_angles=None, trace_lengths=None,
                   log_Q=1 / np.sqrt(2), tune=5000, draws=5000,
                   target_accept=0.9, do_mcmc=False, normalize=False,
                   use_pink_gp=False, verbose=False):

    times_bg = times - np.median(times)
    with pm.Model() as model:

        # The baseline flux
        mean = pm.Normal(f"mean", mu=0.0, sd=1.0)

        edepth = pm.Uniform("edepth", lower=0, upper=0.01)
        edepth = pm.math.sqrt(edepth)

        slope_time = pm.Uniform("slope_time", lower=-1, upper=1)
        line_model = mean + slope_time * times_bg

        if xcenters is not None:
            med_ = np.median(xcenters)
            std_ = np.std(xcenters)
            xcenters = (xcenters - med_) / std_ if normalize else xcenters

            slope_xc = pm.Uniform("slope_xcenter", lower=-1, upper=1)
            line_model = line_model + slope_xc * xcenters

        if ycenters is not None:
            med_ = np.median(ycenters)
            std_ = np.std(ycenters)
            ycenters = (ycenters - med_) / std_ if normalize else ycenters

            slope_yc = pm.Uniform("slope_ycenter", lower=-1, upper=1)
            line_model = line_model + slope_yc * ycenters

        if trace_angles is not None:
            med_ = np.median(trace_angles)
            std_ = np.std(trace_angles)
            if normalize:
                trace_angles = (trace_angles - med_) / std_

            slope_angles = pm.Uniform("slope_trace_angle", lower=-1, upper=1)
            line_model = line_model + slope_angles * trace_angles

        if trace_lengths is not None:
            med_ = np.median(trace_lengths)
            std_ = np.std(trace_lengths)
            if normalize:
                trace_lengths = (trace_lengths - med_) / std_

            slope_tl = pm.Uniform("slope_trace_length", lower=-1, upper=1)
            line_model = line_model + slope_tl * trace_lengths

        pm.Deterministic(f'line_model', line_model)

        # Set up a Keplerian orbit for the planets
        orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, b=b)

        # # Compute the model light curve using starry
        star = xo.LimbDarkLightCurve(u)
        light_curves = star.get_light_curve(
            orbit=orbit, r=edepth, t=times)

        light_curve = pm.math.sum(light_curves, axis=-1)

        # # Here we track the value of the model light curve for plotting
        # # purposes
        pm.Deterministic("light_curve", light_curve)

        # The likelihood function assuming known Gaussian uncertainty
        model_full = light_curve + line_model

        if use_pink_gp:
            gp = build_gp_pink_noise(
                times, data, dataerr, log_Q=log_Q)
            # gp = build_gp_pink_noise(times, data, dataerr, log_Q=log_Q)
            # mu, _ = xo.eval_in_model(model.test_point)

            gp.marginal("gp", observed=data - model_full.flatten())

            mu, _ = gp.predict(times, return_var=True, predict_mean=True)
            # pm.Deterministic("light_curve", light_curve)
            # help(pm.Deterministic)
            # print(type("light_curve2"))
            pm.Deterministic(name="gp_mu", var=mu)

        else:
            pm.Normal(f"obs", mu=model_full, sd=dataerr, observed=data)

        # with pm.Model() as model:
        # Fit for the maximum a posteriori parameters
        #   given the simuated dataset
        map_soln = xo.optimize(start=model.test_point)

        if verbose:
            ppm = 1e6
            info_message(f'Map Soln Edepth:{map_soln["edepth"]*ppm}')

        # with pm.Model() as model:
        np.random.seed(42)
        trace = None
        if do_mcmc:
            trace = pm.sample(
                tune=tune,
                draws=tune,
                start=map_soln,
                chains=mp.cpu_count(),
                step=xo.get_dense_nuts_step(target_accept=target_accept),
                cores=mp.cpu_count()
            )

    return trace, map_soln


def run_multiple_pymc3(times, fine_snr_flux, fine_snr_uncs, aper_sum_columns,
                       t0=0, u=[0], period=1.0, b=0.0, xcenters=None,
                       idx_fwd=None, idx_rev=None, tune=3000, draws=3000,
                       target_accept=0.9, do_mcmc=False, save_as_you_go=False,
                       allow_negative_edepths=False, use_rev_fwd_split=False,
                       use_log_edepth=False, injected_light_curve=1.0,
                       base_name=None, working_dir='./'):

    if use_rev_fwd_split and (idx_fwd is None or idx_rev is None):
        assert(False), (f'if `use_rev_fwd_split` is {use_rev_fwd_split}, '
                        'then you must provide `idx_fwd` and `idx_rev`. '
                        'One or both are current set to `None`')

    varnames = None
    filename = configure_save_name(
        base_name=base_name,
        do_mcmc=do_mcmc,
        use_xcenter=xcenters is not None,
        use_log_edepth=use_log_edepth,
        use_rev_fwd_split=use_rev_fwd_split,
        use_injection=hasattr(injected_light_curve, '__iter__'),
        allow_negative_edepths=allow_negative_edepths
    )

    filename = os.path.join(working_dir, filename)

    fine_grain_mcmcs = {}
    for colname in aper_sum_columns:
        start = time()
        info_message(f'Working on {colname} for MAP/Trace MCMCs')
        data = fine_snr_flux[colname] * injected_light_curve
        dataerr = fine_snr_uncs[colname]

        if use_rev_fwd_split:
            trace, map_soln = run_pymc3_fwd_rev(
                times, data, dataerr, t0, u, period, b,
                idx_fwd, idx_rev, xcenters,
                tune=tune, draws=draws,
                target_accept=target_accept,
                do_mcmc=do_mcmc,
                use_log_edepth=use_log_edepth,
                allow_negative_edepths=allow_negative_edepths
            )
        else:
            trace, map_soln = run_pymc3_direct(
                times, data, dataerr,
                t0, u, period, b, xcenters,
                tune=tune, draws=draws,
                target_accept=target_accept,
                do_mcmc=do_mcmc,
                use_log_edepth=use_log_edepth,
                allow_negative_edepths=allow_negative_edepths
            )

        fine_grain_mcmcs[colname] = {}
        fine_grain_mcmcs[colname]['trace'] = trace
        fine_grain_mcmcs[colname]['map_soln'] = map_soln
        if save_as_you_go and False:
            info_message(f'Saving MCMCs to {filename}')
            joblib.dump(fine_grain_mcmcs, filename)

        if varnames is None:
            varnames = [key for key in map_soln.keys()
                        if '__' not in key and 'light' not in key]

        if trace is not None:
            print(pm.summary(trace, var_names=varnames))

        stop = time() - start
        info_message(f'Completed {colname} for Trace MCMCs took {stop}')

    if save_as_you_go:
        info_message(f'Saving MCMCs to {filename}')
        joblib.dump(fine_grain_mcmcs, filename)

    return fine_grain_mcmcs, filename


def configure_save_name(base_name=None, working_dir='', do_mcmc=True,
                        use_xcenter=False, use_log_edepth=False,
                        use_rev_fwd_split=False, use_injection=False,
                        allow_negative_edepths=False, planet_name="planet_name"):

    if base_name is None:
        base_name = f'{planet_name}_fine_grain_photometry_20x20_208ppm'

    fname_split = {True: 'w_fwd_rev_split',
                   False: 'no_fwd_rev_split'}[use_rev_fwd_split]
    fname_mcmc = {True: 'MCMCs_w_MAPS', False: 'MAPS_only'}[do_mcmc]
    fname_xcenter = {True: 'fit_xcenter', False: 'no_xcenter'}[use_xcenter]
    fname_logedepth = {True: 'fit_log_edepth',
                       False: 'fit_linear_edepth'}[use_log_edepth]
    fname_injected = {True: '_injected_signal', False: ''}[use_injection]
    fname_neg_ecl = {True: '_allow_neg_edepth',
                     False: ''}[allow_negative_edepths]

    filename = f'{base_name}_{fname_mcmc}_{fname_split}_with_{fname_xcenter}_{fname_logedepth}{fname_injected}{fname_neg_ecl}.joblib.save'

    return filename


def compute_xo_lightcurve(planet_name, times, depth_ppm=1000, u=[0]):
    planet_params = exoMAST_API(planet_name)
    planet_params.orbital_period  # = 0.813475  # days # exo.mast.stsci.edu
    t0 = planet_params.transit_time
    edepth = np.sqrt(edepth_ppm / 1e6)  # convert 'depth' to 'radius'

    orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, b=b)
    return xo.LimbDarkLightCurve(u).get_light_curve(
        orbit=orbit, r=edepth, t=times).eval().flatten()


def instantiate_arctor(planet_name, data_dir, working_dir, file_type,
                       joblib_filename='', sort_by_time=False):
    assert(False), ("This needs to be places in the examples. "
                    "For some reason, `from .arctor import Arctor` "
                    "does not work as it is expected to work.")

    planet = Arctor(
        planet_name=planet_name,
        data_dir=data_dir,
        working_dir=working_dir,
        file_type=file_type)

    if os.path.exists(joblib_filename):
        info_message('Loading Data from Save File')
        planet.load_dict(joblib_filename)
    else:
        info_message('Loading New Data Object')
        planet.load_data(sort_by_time=sort_by_time)

    return planet


def instantiate_star_planet_system(  # Stellar parameters
        star_ydeg=0, star_udeg=2, star_L=1.0, star_inc=90.0,
        star_obl=0.0, star_m=1.0, star_r=1.0, star_prot=1.0,
        star_t0=0, star_theta0=0.0, star_A1=1.0, star_A2=0.0,
        star_length_unit=units.Rsun, star_mass_unit=units.Msun,
        # Planetary parameters
        planet_B1=1.0, planet_ydeg=1, planet_udeg=0, planet_L=1.0,
        planet_a=1.0, planet_phase_offset=0., planet_inc=90.0, planet_porb=1.0,
        planet_t0=0.0, planet_obl=0.0, planet_m=0.0, planet_r=0.1,
        planet_ecc=0.0, planet_w=90.0, planet_Omega=0.0, planet_theta0=0.0,
        planet_length_unit=units.Rjup, planet_mass_unit=units.Mjup,
        # Universal Parmaeters
        time_unit=units.day, angle_unit=units.degree):

    stellar_map = starry.Map(ydeg=star_ydeg,
                             udeg=star_udeg,
                             L=star_L,
                             inc=star_inc,
                             obl=star_obl)

    A = starry.Primary(
        stellar_map,
        m=star_m,
        r=star_r,
        prot=star_prot,
        t0=star_t0,
        theta0=star_theta0,
        length_unit=star_length_unit,
        mass_unit=star_mass_unit,
        time_unit=time_unit,
        angle_unit=angle_unit
    )

    A.map[1] = star_A1
    A.map[2] = star_A2

    planet_map = starry.Map(ydeg=planet_ydeg,
                            udeg=planet_udeg,
                            L=planet_L,
                            inc=planet_inc,
                            obl=planet_obl)

    b = starry.Secondary(
        planet_map,
        m=tt.as_tensor_variable(planet_m).astype("float64"),
        r=planet_r,
        a=planet_a,
        inc=planet_inc,
        t0=planet_t0,
        prot=planet_porb,  # synchronous rotation
        porb=planet_porb,
        ecc=planet_ecc,
        w=planet_w,
        Omega=planet_Omega,
        theta0=planet_theta0,
        length_unit=planet_length_unit,
        mass_unit=planet_mass_unit,
        time_unit=time_unit,
        angle_unit=angle_unit
    )

    if planet_ydeg > 0:
        b.map[1, 0] = planet_B1

    b.theta0 = 180.0 + planet_phase_offset

    return starry.System(A, b)


def previous_instantiate_arctor(planet_name, data_dir, working_dir, file_type,
                                save_name_base='savedict'):
    planet = Arctor(
        planet_name=planet_name,
        data_dir=data_dir,
        working_dir=working_dir,
        file_type=file_type)

    joblib_filename = f'{planet_name}_{save_name_base}.joblib.save'
    joblib_filename = f'{working_dir}/{joblib_filename}'
    if os.path.exists(joblib_filename):
        info_message('Loading Data from Save File')
        planet.load_data(joblib_filename)
    else:
        info_message('Loading New Data Object')
        planet.load_data()

    return planet


def create_raw_lc_stddev(planet, reject_outliers=True):
    ppm = 1e6
    phot_vals = planet.photometry_df
    n_columns = len(planet.photometry_df.columns)

    # lc_med_fwd = np.zeros_like(n_columns)
    # lc_med_rev = np.zeros_like(n_columns)

    # lc_std_fwd = np.zeros_like(n_columns)
    # lc_std_rev = np.zeros_like(n_columns)

    lc_med = np.zeros(n_columns)
    lc_std = np.zeros(n_columns)

    for k, colname in enumerate(phot_vals.columns):
        if reject_outliers:
            inliers_fwd, inliers_rev = compute_inliers(
                planet, aper_colname=colname, n_sig=2
            )
        else:
            inliers_fwd = np.arange(planet.idx_fwd.size)
            inliers_rev = np.arange(planet.idx_rev.size)

        phots_rev = phot_vals[colname].iloc[planet.idx_rev].iloc[inliers_rev]
        phots_fwd = phot_vals[colname].iloc[planet.idx_fwd].iloc[inliers_fwd]

        lc_std_rev = mad_std(phots_rev)
        lc_std_fwd = mad_std(phots_fwd)

        lc_med_rev = np.median(phots_fwd)
        lc_med_fwd = np.median(phots_fwd)

        lc_std[k] = np.mean([lc_std_rev, lc_std_fwd])
        lc_med[k] = np.mean([lc_med_rev, lc_med_fwd])

    return lc_std / lc_med * ppm


def center_one_trace(kcol, col, fitter, stddev, y_idx, inds, idx_buffer=10):
    model = Gaussian1D(amplitude=col.max(),
                       mean=y_idx, stddev=stddev)

    # idx_narrow = abs(inds - y_idx) < idx_buffer

    # results = fitter(model, inds[idx_narrow], col[idx_narrow])
    results = fitter(model, inds, col)

    return kcol, results, fitter


def fit_one_slopes(kimg, means, fitter, y_idx, slope_guess=2.0 / 466):
    model = Linear1D(slope=slope_guess, intercept=y_idx)

    inds = np.arange(len(means))
    inds = inds - np.median(inds)

    results = fitter(model, inds, means)

    return kimg, results, fitter


def cosmic_ray_flag_simple(image_, n_sig=5, window=7):
    cosmic_rays_ = np.zeros(image_.shape, dtype=bool)
    for k, row in enumerate(image_):
        row_Med = np.median(row)
        row_Std = np.std(row)
        cosmic_rays_[k] += abs(row - row_Med) > n_sig * row_Std
        image_[k][cosmic_rays_[k]] = row_Med

    return image_, cosmic_rays_


def cosmic_ray_flag_rolling(image_, n_sig=5, window=7):
    cosmic_rays_ = np.zeros(image_.shape, dtype=bool)
    for k, row in enumerate(image_):
        row_rMed = pd.Series(row).rolling(window).median()
        row_rStd = pd.Series(row).rolling(window).std()
        cosmic_rays_[k] += abs(row - row_rMed) > n_sig * row_rStd
        image_[k][cosmic_rays_[k]] = row_rMed[cosmic_rays_[k]]

    return image_, cosmic_rays_


def aper_table_2_df(aper_phots, aper_widths, aper_heights, n_images):
    info_message(f'Restructuring Aperture Photometry into DataFrames')
    if len(aper_phots) > 1:
        aper_df = aper_phots[0].to_pandas()
        for kimg in aper_phots[1:]:
            aper_df = pd.concat([aper_df, kimg.to_pandas()], ignore_index=True)
    else:
        aper_df = aper_phots.to_pandas()

    photometry_df_ = aper_df.reset_index().drop(['index', 'id'], axis=1)
    mesh_widths, mesh_heights = np.meshgrid(aper_widths, aper_heights)

    mesh_widths = mesh_widths.flatten()
    mesh_heights = mesh_heights.flatten()
    aperture_columns = [colname
                        for colname in photometry_df_.columns
                        if 'aperture_sum_' in colname]

    photometry_df = pd.DataFrame([])
    for colname in aperture_columns:
        aper_id = int(colname.replace('aperture_sum_', ''))
        aper_width_ = mesh_widths[aper_id].astype(int)
        aper_height_ = mesh_heights[aper_id].astype(int)
        newname = f'aperture_sum_{aper_width_}x{aper_height_}'

        photometry_df[newname] = photometry_df_[colname]

    photometry_df['xcenter'] = photometry_df_['xcenter']
    photometry_df['ycenter'] = photometry_df_['ycenter']

    return photometry_df


def make_mask_cosmic_rays_temporal_simple(val, kcol, krow, n_sig=5):
    val_Med = np.median(val)
    val_Std = np.std(val)
    mask = abs(val - val_Med) > n_sig * val_Std
    return kcol, krow, mask, val_Med


def check_if_column_exists(existing_photometry_df, new_photometry_df, colname):
    existing_columns = existing_photometry_df.columns

    exists = False
    similar = False
    if colname in existing_columns:
        existing_vec = existing_photometry_df[colname]
        new_vec = new_photometry_df[colname]

        exists = True
        similar = np.allclose(existing_vec, new_vec)

        if similar:
            return exists, similar, colname
        else:
            same_name = []
            for colname in existing_columns:
                if f'colname_{len(same_name)}' in existing_columns:
                    same_name.append(colname)

            return exists, similar, f'colname_{len(same_name)}'
    else:
        return exists, similar, colname


def run_all_12_options(times, flux, uncs,
                       list_of_aper_columns,
                       xcenters=None, ycenters=None,
                       trace_angles=None, trace_lengths=None,
                       t0=0, u=[0], period=1.0, b=0.0,
                       idx_fwd=None, idx_rev=None,
                       tune=3000, draws=3000, target_accept=0.9,
                       do_mcmc=False, save_as_you_go=False,
                       injected_light_curve=1.0, working_dir='./',
                       base_name=f'{planet_name}_fine_grain_photometry_208ppm'):

    decor_set = [xcenters, ycenters, trace_angles, trace_lengths]
    decor_options = [None, None, None, None,
                     None, None].extend([decor_set] * 6)

    neg_ecl_options = [True, True, True, True,
                       False, False, False, False,
                       False, False, False, False]

    use_split_options = [True, False, True, False,
                         True, False, True, False,
                         True, False, True, False]

    log_edepth_options = [False, False, False, False,
                          False, False, False, False,
                          True, True, True, True]

    pymc3_options = zip(decor_options, neg_ecl_options,
                        use_split_options, log_edepth_options)

    mcmc_fits = {}

    start0 = time()
    for decor_set_, allow_neg_, use_split_, use_log_edepth_ in pymc3_options:
        start1 = time()
        print(f'Fit xCenters: {xcenters_ is None}')
        print(f'Allow Negative Eclipse Depth: {allow_neg_}')
        print(f'Use Fwd/Rev Split: {use_split_}')
        print(f'Use Log Edepth: {use_log_edepth_}')

        if decor_set_ is not None:
            xcenters_, ycenters_, trace_angles_, trace_lengths_ = decor_set_
        else:
            xcenters_, ycenters_, trace_angles_, trace_lengths_ = [None] * 4

        idx_fwd_ = idx_fwd if use_split_ else None
        idx_rev_ = idx_rev if use_split_ else None

        fine_grain_mcmcs, filename = run_pymc3_both(
            times, flux, uncs, list_of_aper_columns,
            t0=t0, u=u, period=period, b=b,
            idx_fwd=idx_fwd_, idx_rev=idx_rev_,
            tune=tune, draws=draws, target_accept=target_accept,
            do_mcmc=do_mcmc, save_as_you_go=save_as_you_go,
            injected_light_curve=injected_light_curve,
            base_name=base_name, working_dir=working_dir,
            xcenters=xcenters_,
            ycenters=ycenters_,
            trace_angles=trace_angles_,
            trace_lengths=trace_lengths_,
            allow_negative_edepths=allow_neg_,
            use_rev_fwd_split=use_split_,
            use_log_edepth=use_log_edepth_
        )
        mcmc_fits[filename] = fine_grain_mcmcs
        print(f'[INFO] This MCMCs took {(time() - start1)/60:0.2f} minutes')

        del fine_grain_mcmcs, filename

    n_mcmcs = len(decor_options)
    full_time = (time() - start0) / 60
    print(f'[INFO] All {n_mcmcs} MCMCs took {full_time:0.2f} minutes')

    return mcmc_fits


def run_all_12_options_plain(times, fine_snr_flux, fine_snr_uncs,
                             near_best_apertures_NxN_small,
                             t0=0, u=[0], period=1.0, b=0.0,
                             idx_fwd=None, idx_rev=None,
                             tune=3000, draws=3000, target_accept=0.9,
                             do_mcmc=False, save_as_you_go=False,
                             injected_light_curve=1.0,
                             base_name=f'{planet_name}_fine_grain_photometry_208ppm'):

    base_name = f'{base_name}_near_best_{n_space}x{n_space}'

    start0 = time()

    # Linear Eclipse Depths with Negative Allowed
    start1 = time()
    print('Linear Eclipse depth fits - Default everything')
    fine_grain_mcmcs, filename = run_multiple_pymc3(
        times, fine_snr_flux, fine_snr_uncs, near_best_apertures_NxN_small,
        t0=t0_guess, u=u, period=period_planet, b=b_planet,
        idx_fwd=idx_fwd, idx_rev=idx_rev,
        tune=tune, draws=draws, target_accept=target_accept,
        do_mcmc=do_mcmc, save_as_you_go=save_as_you_go,
        injected_light_curve=injected_light_curve,
        base_name=base_name, working_dir=working_dir,
        xcenters=None,
        allow_negative_edepths=True,
        use_rev_fwd_split=False,
        use_log_edepth=False
    )

    fine_grain_mcmcs_no_xcenter_lin_edepth_no_split_w_negEcl = fine_grain_mcmcs
    filename_no_xcenter_lin_edepth_no_split_w_negEcl = filename
    print(f'[INFO] This MCMCs took {(time() - start1)/60:0.2f} minutes')

    del fine_grain_mcmcs, filename

    start1 = time()
    print('Linear Eclipse depth fits - Everything with splitting fwd rev')
    fine_grain_mcmcs, filename = run_multiple_pymc3(
        times, fine_snr_flux, fine_snr_uncs, near_best_apertures_NxN_small,
        t0=t0_guess, u=u, period=period_planet, b=b_planet,
        idx_fwd=idx_fwd, idx_rev=idx_rev,
        tune=tune, draws=draws, target_accept=target_accept,
        do_mcmc=do_mcmc, save_as_you_go=save_as_you_go,
        injected_light_curve=injected_light_curve,
        base_name=base_name, working_dir=working_dir,
        xcenters=None,  # SAME
        allow_negative_edepths=True,  # SAME
        use_rev_fwd_split=True,  # DIFFERENT
        use_log_edepth=False  # SAME
    )
    fine_grain_mcmcs_with_no_xcenter_lin_edepth_w_split_w_negEcl = fine_grain_mcmcs
    filename_with_no_xcenter_lin_edepth_w_split_w_negEcl = filename

    print(f'[INFO] This MCMCs took {(time() - start1)/60:0.2f} minutes')

    del fine_grain_mcmcs, filename

    start1 = time()
    print('Linear Eclipse depth fits - Everything with xcenter')
    fine_grain_mcmcs, filename = run_multiple_pymc3(
        times, fine_snr_flux, fine_snr_uncs, near_best_apertures_NxN_small,
        t0=t0_guess, u=u, period=period_planet, b=b_planet,
        idx_fwd=idx_fwd, idx_rev=idx_rev,
        tune=tune, draws=draws, target_accept=target_accept,
        do_mcmc=do_mcmc, save_as_you_go=save_as_you_go,
        injected_light_curve=injected_light_curve,
        base_name=base_name, working_dir=working_dir,
        xcenters=xcenters_mod,  # DIFFERENT
        allow_negative_edepths=True,  # SAME
        use_rev_fwd_split=False,  # SAME
        use_log_edepth=False,  # SAME
    )

    fine_grain_mcmcs_with_w_xcenter_lin_edepth_no_split_w_negEcl = fine_grain_mcmcs
    filename_with_w_xcenter_lin_edepth_no_split_w_negEcl = filename
    print(f'[INFO] This MCMCs took {(time() - start1)/60:0.2f} minutes')

    del fine_grain_mcmcs, filename

    start1 = time()
    print('Linear Eclipse depth fits - '
          'Everything with xcenter and splitting fwd rev')
    fine_grain_mcmcs, filename = run_multiple_pymc3(
        times, fine_snr_flux, fine_snr_uncs, near_best_apertures_NxN_small,
        t0=t0_guess, u=u, period=period_planet, b=b_planet,
        idx_fwd=idx_fwd, idx_rev=idx_rev,
        tune=tune, draws=draws, target_accept=target_accept,
        do_mcmc=do_mcmc, save_as_you_go=save_as_you_go,
        injected_light_curve=injected_light_curve,
        base_name=base_name, working_dir=working_dir,
        xcenters=xcenters_mod,  # SAME
        allow_negative_edepths=True,  # SAME
        use_rev_fwd_split=True,  # DIFFERENT
        use_log_edepth=False  # SAME
    )

    fine_grain_mcmcs_with_w_xcenter_lin_edepth_w_split_w_negEcl = fine_grain_mcmcs
    filename_with_w_xcenter_lin_edepth_w_split_w_negEcl = filename
    print(f'[INFO] This MCMCs took {(time() - start1)/60:0.2f} minutes')

    del fine_grain_mcmcs, filename

    start1 = time()
    # Linear Eclipse Depths without Negative Allowed
    print('Linear Eclipse depth fits - Default everything')
    fine_grain_mcmcs, filename = run_multiple_pymc3(
        times, fine_snr_flux, fine_snr_uncs, near_best_apertures_NxN_small,
        t0=t0_guess, u=u, period=period_planet, b=b_planet, idx_fwd=idx_fwd, idx_rev=idx_rev,
        tune=tune, draws=draws, target_accept=target_accept,
        injected_light_curve=injected_light_curve,
        base_name=base_name, working_dir=working_dir,
        do_mcmc=do_mcmc, save_as_you_go=save_as_you_go,
        xcenters=None,  # DIFFERENT
        allow_negative_edepths=False,  # DIFFERENT
        use_rev_fwd_split=False,  # DIFFERENT
        use_log_edepth=False  # SAME
    )

    fine_grain_mcmcs_with_no_xcenter_lin_edepth_no_split = fine_grain_mcmcs
    filename_with_no_xcenter_lin_edepth_no_split = filename
    print(f'[INFO] This MCMCs took {(time() - start1)/60:0.2f} minutes')

    del fine_grain_mcmcs, filename

    start1 = time()
    print('Linear Eclipse depth fits - Everything with splitting fwd rev')
    fine_grain_mcmcs, filename = run_multiple_pymc3(
        times, fine_snr_flux, fine_snr_uncs, near_best_apertures_NxN_small,
        t0=t0_guess, u=u, period=period_planet, b=b_planet, idx_fwd=idx_fwd, idx_rev=idx_rev,
        tune=tune, draws=draws, target_accept=target_accept,
        injected_light_curve=injected_light_curve,
        base_name=base_name, working_dir=working_dir,
        do_mcmc=do_mcmc, save_as_you_go=save_as_you_go,
        xcenters=None,  # SAME
        allow_negative_edepths=False,  # SAME
        use_rev_fwd_split=True,  # DIFFERENT
        use_log_edepth=False,  # SAME
    )

    fine_grain_mcmcs_with_no_xcenter_lin_edepth_w_split = fine_grain_mcmcs
    filename_with_no_xcenter_lin_edepth_w_split = filename

    print(f'[INFO] This MCMCs took {(time() - start1)/60:0.2f} minutes')

    del fine_grain_mcmcs, filename

    start1 = time()
    print('Linear Eclipse depth fits - Everything with xcenter')
    fine_grain_mcmcs, filename = run_multiple_pymc3(
        times, fine_snr_flux, fine_snr_uncs, near_best_apertures_NxN_small,
        t0=t0_guess, u=u, period=period_planet, b=b_planet,
        idx_fwd=idx_fwd, idx_rev=idx_rev,
        tune=tune, draws=draws, target_accept=target_accept,
        do_mcmc=do_mcmc, save_as_you_go=save_as_you_go,
        injected_light_curve=injected_light_curve,
        base_name=base_name, working_dir=working_dir,
        xcenters=xcenters_mod,  # DIFFERENT
        allow_negative_edepths=False,  # SAME
        use_rev_fwd_split=False,  # DIFFERENT
        use_log_edepth=False  # SAME
    )

    fine_grain_mcmcs_with_w_xcenter_lin_edepth_no_split = fine_grain_mcmcs
    filename_with_w_xcenter_lin_edepth_no_split = filename

    print(f'[INFO] This MCMCs took {(time() - start1)/60:0.2f} minutes')

    del fine_grain_mcmcs, filename

    start1 = time()
    print('Linear Eclipse depth fits - '
          'Everything with xcenter and splitting fwd rev')
    fine_grain_mcmcs, filename = run_multiple_pymc3(
        times, fine_snr_flux, fine_snr_uncs, near_best_apertures_NxN_small,
        t0=t0_guess, u=u, period=period_planet, b=b_planet,
        idx_fwd=idx_fwd, idx_rev=idx_rev,
        tune=tune, draws=draws, target_accept=target_accept,
        do_mcmc=do_mcmc, save_as_you_go=save_as_you_go,
        injected_light_curve=injected_light_curve,
        base_name=base_name, working_dir=working_dir,
        xcenters=xcenters_mod,  # SAME
        allow_negative_edepths=False,  # SAME
        use_rev_fwd_split=True,  # DIFFERENT
        use_log_edepth=False)  # SAME

    fine_grain_mcmcs_with_w_xcenter_lin_edepth_w_split = fine_grain_mcmcs
    filename_with_w_xcenter_lin_edepth_w_split = filename

    print(f'[INFO] This MCMCs took {(time() - start1)/60:0.2f} minutes')

    del fine_grain_mcmcs, filename

    # Logarithmic Eclipse Depths
    start1 = time()
    print('Log Eclipse depth fits - Default everything')
    fine_grain_mcmcs, filename = run_multiple_pymc3(
        times, fine_snr_flux, fine_snr_uncs, near_best_apertures_NxN_small,
        t0=t0_guess, u=u, period=period_planet, b=b_planet,
        idx_fwd=idx_fwd, idx_rev=idx_rev,
        tune=tune, draws=draws, target_accept=target_accept,
        do_mcmc=do_mcmc, save_as_you_go=save_as_you_go,
        injected_light_curve=injected_light_curve,
        base_name=base_name, working_dir=working_dir,
        xcenters=None,  # DIFFERENT
        allow_negative_edepths=False,  # SAME
        use_rev_fwd_split=False,  # DIFFERENT
        use_log_edepth=True  # DIFFERENT
    )

    fine_grain_mcmcs_with_no_xcenter_log_edepth_no_split = fine_grain_mcmcs
    filename_with_no_xcenter_log_edepth_no_split = filename

    print(f'[INFO] This MCMCs took {(time() - start1)/60:0.2f} minutes')

    del fine_grain_mcmcs, filename

    start1 = time()
    print('Log Eclipse depth fits - Everything with splitting fwd rev')
    fine_grain_mcmcs, filename = run_multiple_pymc3(
        times, fine_snr_flux, fine_snr_uncs, near_best_apertures_NxN_small,
        t0=t0_guess, u=u, period=period_planet, b=b_planet,
        idx_fwd=idx_fwd, idx_rev=idx_rev,
        tune=tune, draws=draws, target_accept=target_accept,
        do_mcmc=do_mcmc, save_as_you_go=save_as_you_go,
        injected_light_curve=injected_light_curve,
        base_name=base_name, working_dir=working_dir,
        xcenters=None,  # SAME
        allow_negative_edepths=False,  # SAME
        use_rev_fwd_split=True,  # DIFFERENT
        use_log_edepth=True  # SAME
    )

    fine_grain_mcmcs_with_no_xcenter_log_edepth_w_split = fine_grain_mcmcs
    filename_with_no_xcenter_log_edepth_w_split = filename

    print(f'[INFO] This MCMCs took {(time() - start1)/60:0.2f} minutes')

    start1 = time()
    print('Log Eclipse depth fits - Everything with xcenter')
    fine_grain_mcmcs, filename = run_multiple_pymc3(
        times, fine_snr_flux, fine_snr_uncs, near_best_apertures_NxN_small,
        t0=t0_guess, u=u, period=period_planet, b=b_planet,
        idx_fwd=idx_fwd, idx_rev=idx_rev,
        tune=tune, draws=draws, target_accept=target_accept,
        do_mcmc=do_mcmc, save_as_you_go=save_as_you_go,
        injected_light_curve=injected_light_curve,
        base_name=base_name, working_dir=working_dir,
        xcenters=xcenters_mod,  # DIFFERENT
        allow_negative_edepths=False,  # SAME
        use_rev_fwd_split=False,  # DIFFERENT
        use_log_edepth=True  # SAME
    )

    fine_grain_mcmcs_with_w_xcenter_log_edepth_no_split = fine_grain_mcmcs
    filename_with_w_xcenter_log_edepth_no_split = filename

    print(f'[INFO] This MCMCs took {(time() - start1)/60:0.2f} minutes')

    del fine_grain_mcmcs, filename

    start1 = time()
    print('Log Eclipse depth fits - Everything with xcenter and splitting fwd rev')
    fine_grain_mcmcs, filename = run_multiple_pymc3(
        times, fine_snr_flux, fine_snr_uncs, near_best_apertures_NxN_small,
        t0=t0_guess, u=u, period=period_planet, b=b_planet,
        idx_fwd=idx_fwd, idx_rev=idx_rev,
        tune=tune, draws=draws, target_accept=target_accept,
        do_mcmc=do_mcmc, save_as_you_go=save_as_you_go,
        injected_light_curve=injected_light_curve,
        base_name=base_name, working_dir=working_dir,
        xcenters=xcenters_mod,  # SAME
        allow_negative_edepths=False,  # SAME
        use_rev_fwd_split=True,  # DIFFERENT
        use_log_edepth=True  # SAME
    )

    fine_grain_mcmcs_with_w_xcenter_log_edepth_w_split = fine_grain_mcmcs
    filename_with_w_xcenter_log_edepth_w_split = filename

    print(f'[INFO] This MCMCs took {(time() - start1)/60:0.2f} minutes')

    print(f'[INFO] All 12 MCMCs took {(time() - start0)/60:0.2f} minutes')
    return [fine_grain_mcmcs_with_no_xcenter_lin_edepth_no_split_w_negEcl,
            fine_grain_mcmcs_with_no_xcenter_lin_edepth_w_split_w_negEcl,
            fine_grain_mcmcs_with_w_xcenter_lin_edepth_no_split_w_negEcl,
            fine_grain_mcmcs_with_w_xcenter_lin_edepth_w_split_w_negEcl,
            fine_grain_mcmcs_with_no_xcenter_lin_edepth_no_split,
            fine_grain_mcmcs_with_no_xcenter_lin_edepth_w_split,
            fine_grain_mcmcs_with_w_xcenter_lin_edepth_no_split,
            fine_grain_mcmcs_with_w_xcenter_lin_edepth_w_split,
            fine_grain_mcmcs_with_no_xcenter_log_edepth_no_split,
            fine_grain_mcmcs_with_no_xcenter_log_edepth_w_split,
            fine_grain_mcmcs_with_w_xcenter_log_edepth_no_split,
            fine_grain_mcmcs_with_w_xcenter_log_edepth_w_split
            ]


def rename_file(filename, data_dir='./', base_time=2400000.5,
                format='jd', scale='utc'):

    path_in = os.path.join(data_dir, filename)
    header = fits.getheader(path_in, ext=0)
    time_stamp = 0.5 * (header['EXPSTART'] + header['EXPEND'])
    time_obj = astropy.time.Time(val=time_stamp, val2=base_time,
                                 format=format, scale=scale)

    out_filename = f'{time_obj.isot}_{filename}'
    path_out = os.path.join(data_dir, out_filename)

    os.rename(path_in, path_out)


def compute_delta_sdnr(map_soln, phots, idx_fwd, idx_rev):
    ppm = 1e6
    phots_std_fwd = phots[idx_fwd].std()
    phots_std_rev = phots[idx_rev].std()
    phots_std = np.mean([phots_std_fwd, phots_std_rev])

    if 'mean_fwd' not in map_soln.keys():
        map_model = map_soln['light_curve'].flatten() + map_soln['line_model']
    else:
        map_model = np.zeros_like(times)
        map_model[idx_fwd] = map_soln['light_curve_fwd'].flatten() + \
            map_soln['line_model_fwd']
        map_model[idx_rev] = map_soln['light_curve_rev'].flatten() + \
            map_soln['line_model_rev']

    varnames = [key for key in map_soln.keys(
    ) if '__' not in key and 'light' not in key and 'line' not in key and 'le_edepth_0' not in key]

    res_fwd = np.std(map_model[idx_fwd] - phots[idx_fwd])
    res_rev = np.std(map_model[idx_rev] - phots[idx_rev])
    res_std = np.mean([res_fwd, res_rev])

    print(f'{str(varnames):<80}')
    print(f'{res_std*ppm:0.2f}, {phots_std*ppm:0.2f}, {(phots_std - res_std)*ppm:0.2f} ppm difference'),

    return res_std * ppm, phots_std * ppm, (phots_std - res_std) * ppm


def compute_chisq_aic(planet, aper_column, map_soln, idx_fwd, idx_rev,
                      use_idx_fwd_, use_xcenters_, use_ycenters_,
                      use_trace_angles_, use_trace_lengths_, use_pink_gp):
    ppm = 1e6

    phots = planet.normed_photometry_df[aper_column].values
    uncs = planet.normed_uncertainty_df[aper_column].values
    phots = phots - np.median(phots)

    n_pts = len(phots)

    # 2 == eclipse depth + mean
    n_params = (2 + use_idx_fwd_ + use_xcenters_ + use_ycenters_ +
                use_trace_angles_ + use_trace_lengths_ + 3 * use_pink_gp)

    if 'mean_fwd' not in map_soln.keys():
        map_model = map_soln['light_curve'].flatten() + map_soln['line_model']
    else:
        map_model = np.zeros_like(planet.times)
        map_model[idx_fwd] = map_soln['light_curve_fwd'].flatten() + \
            map_soln['line_model_fwd']
        map_model[idx_rev] = map_soln['light_curve_rev'].flatten() + \
            map_soln['line_model_rev']

        # if we split Fwd/Rev, then there are now 2 means
        n_params = n_params + 1

    correction = 2 * n_params * (n_params + 1) / (n_pts - n_params - 1)

    sdnr_ = np.std(map_model - phots) * ppm
    chisq_ = np.sum((map_model - phots)**2 / uncs**2)
    aic_ = chisq_ + 2 * n_params + correction
    bic_ = chisq_ + n_params * np.log10(n_pts)

    return chisq_, aic_, bic_, sdnr_


def compute_inliers(instance, aper_colname='aperture_sum_176x116', n_sig=2):

    phots_ = instance.normed_photometry_df[aper_colname]

    inliers_fwd = ~sigma_clip(phots_[instance.idx_fwd],
                              sigma=n_sig,
                              maxiters=1,
                              stdfunc=mad_std).mask

    inliers_rev = ~sigma_clip(phots_[instance.idx_rev],
                              sigma=n_sig,
                              maxiters=1,
                              stdfunc=mad_std).mask

    inliers_fwd = np.where(inliers_fwd)[0]
    inliers_rev = np.where(inliers_rev)[0]

    return inliers_fwd, inliers_rev


def compute_outliers(instance, aper_colname='aperture_sum_176x116', n_sig=2):
    phots_ = instance.normed_photometry_df[aper_colname]
    outliers_fwd = sigma_clip(phots_[instance.idx_fwd],
                              sigma=n_sig,
                              maxiters=1,
                              stdfunc=mad_std).mask
    outliers_rev = sigma_clip(phots_[instance.idx_rev],
                              sigma=n_sig,
                              maxiters=1,
                              stdfunc=mad_std).mask

    outliers_fwd = np.where(outliers_fwd)[0]
    outliers_rev = np.where(outliers_rev)[0]

    return outliers_fwd, outliers_rev


def extract_map_only_data(planet, idx_fwd, idx_rev,
                          maps_only_filename=None,
                          data_dir='../savefiles',
                          use_pink_gp=False):

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
                use_trace_lengths[-1],
                use_pink_gp=use_pink_gp)

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


def get_map_results_models(times, map_soln, idx_fwd, idx_rev):
    if 'mean_fwd' not in map_soln.keys():
        map_model = map_soln['light_curve'].flatten()
        line_model = map_soln['line_model'].flatten()
    else:
        map_model = np.zeros_like(times)
        line_model = np.zeros_like(times)
        map_model[idx_fwd] = map_soln['light_curve_fwd'].flatten()
        line_model[idx_fwd] = map_soln['line_model_fwd'].flatten()

        map_model[idx_rev] = map_soln['light_curve_rev'].flatten()
        line_model[idx_rev] = map_soln['line_model_rev'].flatten()

    return map_model, line_model


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
