from . import *
from .arctor import *

import astropy.units as units
import exoplanet as xo
import numpy as np
import os
import pygtc
import pymc3 as pm
import starry
import theano.tensor as tt

from statsmodels.robust.scale import mad


def debug_message(message, end='\n'):
    print(f'[DEBUG] {message}', end=end)


def warning_message(message, end='\n'):
    print(f'[WARNING] {message}', end=end)


def info_message(message, end='\n'):
    print(f'[INFO] {message}', end=end)


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

    # if square_edepth:
    #     info_message('Converting the `edepth` from `r` to real `edepth`')
    #     edepth_orig = samples['edepth'].copy()
    #     positive = edepth_orig > 0
    #     negative = edepth_orig < 0
    #     samples['edepth'][positive] = np.sqrt(edepth_orig[positive])
    #     samples['edepth'][negative] = -np.sqrt(edepth_orig[negative])
    #     samples['edepth_orig'] = edepth_orig

    #     med_edepth = np.median(samples['edepth'])
    #     std_edepth = np.std(samples['edepth'])
    #     sgn_edepth = np.sign(med_edepth)
    #     map_soln['edepth'] = sgn_edepth * np.sqrt(abs(med_edepth))

    #     med_edepth_orig = np.median(samples['edepth_orig'])
    #     std_edepth_orig = np.std(samples['edepth_orig'])
    #     sgn_edepth_orig = np.sign(med_edepth_orig)
    #     map_soln['edepth_orig'] = med_edepth_orig

    varnames = [key for key in map_soln.keys()
                if '__' not in key and 'light' not in key
                and 'line' not in key]

    truths = [float(val) for key, val in map_soln.items() if key in varnames]
    pygtc.plotGTC(samples, plotName=plotName,  # truths=truths,
                  smoothingKernel=smoothingKernel,
                  labelRotation=[True] * 2,
                  customLabelFont={'rotation': 45},
                  nContourLevels=3, figureSize='MNRAS_page')


def run_pymc3_multi_dataset(times, data, yerr, t0, u, period, b,
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

        # # In this line, we simulate the dataset that we will fit
        # y = xo.eval_in_model(light_curve)
        # y += yerr * np.random.randn(len(y))

        # The likelihood function assuming known Gaussian uncertainty
        pm.Normal("obs", mu=light_curve, sd=yerr, observed=data)

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


def run_pymc3_fwd_rev(times, data, yerr, t0, u, period, b, idx_fwd, idx_rev,
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
                # edepth_sign = pm.math.sgn(edepth)
                # if pm.math.lt(edepth_sign, 0):
                #     edepth = -pm.math.sqrt(pm.math.abs_(edepth))
                # else:
                #     edepth = pm.math.sqrt(pm.math.abs_(edepth))
            else:
                edepth = pm.Uniform("edepth", lower=0, upper=0.01)
                edepth = pm.math.sqrt(edepth)

        slope = pm.Uniform("slope", lower=-0.1, upper=0.1)
        line_fwd = mean_fwd + slope * times_bg[idx_fwd]
        line_rev = mean_rev + slope * times_bg[idx_rev]

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
                  sd=yerr[idx_fwd], observed=data[idx_fwd])
        pm.Normal("obs_rev", mu=light_curve_rev + line_rev,
                  sd=yerr[idx_rev], observed=data[idx_rev])

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


def run_pymc3_direct(times, data, yerr, t0, u, period, b, xcenters=None,
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
                # edepth_sign = pm.math.sgn(edepth)
                # if pm.math.lt(edepth_sign, 0):
                #     edepth = -pm.math.sqrt(pm.math.abs_(edepth))
                # else:
                #     edepth = pm.math.sqrt(pm.math.abs_(edepth))
            else:
                edepth = pm.Uniform("edepth", lower=0, upper=0.01)
                edepth = pm.math.sqrt(edepth)

        slope = pm.Uniform("slope", lower=-0.1, upper=0.1)
        line = mean + slope * times_bg

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
        pm.Normal("obs", mu=model_, sd=yerr, observed=data)

        # Fit for the maximum a posteriori parameters given the simuated
        # dataset

        map_soln = xo.optimize(start=model.test_point)
        if use_log_edepth:
            map_soln_edepth = 10**map_soln["log_edepth"]
        else:
            map_soln_edepth = map_soln["edepth"]

        info_message(f'map_soln_edepth:{map_soln_edepth*1e6}')

        line_map_soln = (map_soln['mean'] +
                         map_soln['slope'] * times_bg.flatten())
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


def run_pymc3_both(times, data, yerr, t0, u, period, b,
                   xcenters=None, ycenters=None,
                   trace_angles=None, trace_lengths=None,
                   idx_fwd=None, idx_rev=None, tune=5000, draws=5000,
                   target_accept=0.9, do_mcmc=True, use_log_edepth=False,
                   allow_negative_edepths=False):

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
        mean_fwd = pm.Normal(f"mean{strfwd}", mu=1.0, sd=1.0)

        if idx_rev is not None:
            mean_rev = pm.Normal(f"mean{strrev}", mu=1.0, sd=1.0)

        assert(not (allow_negative_edepths and use_log_edepth)),\
            'Cannot have `allow_negative_edepths` with `use_log_edepth`'

        if use_log_edepth:
            log_edepth = pm.Uniform("log_edepth", lower=-20, upper=-2)
            edepth = pm.Deterministic("edepth", 10**(0.5 * log_edepth))
        else:
            if allow_negative_edepths:
                edepth = pm.Uniform("edepth", lower=-0.01, upper=0.01)
                # pm.Deterministic('lt_edepth_0', pm.math.lt(edepth, 0.0))
                # if pm.math.lt(edepth, 0.0):
                #     edepth = -np.sqrt(abs(edepth))
                # else:
                #     edepth = np.sqrt(edepth)
            else:
                edepth = pm.Uniform("edepth", lower=0, upper=0.01)
                edepth = pm.math.sqrt(edepth)

        slope = pm.Uniform("slope", lower=-1, upper=1)
        line_fwd = mean_fwd + slope * times_bg[idx_fwd]

        if idx_rev is not None:
            line_rev = mean_rev + slope * times_bg[idx_rev]

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
        pm.Deterministic(f"light_curves{strfwd}", light_curves_fwd)
        if idx_rev is not None:
            pm.Deterministic(f"light_curves{strrev}", light_curves_rev)

        # # The likelihood function assuming known Gaussian uncertainty
        pm.Normal(f"obs{strfwd}", mu=light_curve_fwd + line_fwd,
                  sd=yerr[idx_fwd], observed=data[idx_fwd])

        if idx_rev is not None:
            pm.Normal(f"obs{strrev}", mu=light_curve_rev + line_rev,
                      sd=yerr[idx_rev], observed=data[idx_rev])

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
        yerr = fine_snr_uncs[colname]

        if use_rev_fwd_split:
            trace, map_soln = run_pymc3_fwd_rev(
                times, data, yerr, t0, u, period, b,
                idx_fwd, idx_rev, xcenters,
                tune=tune, draws=draws,
                target_accept=target_accept,
                do_mcmc=do_mcmc,
                use_log_edepth=use_log_edepth,
                allow_negative_edepths=allow_negative_edepths
            )
        else:
            trace, map_soln = run_pymc3_direct(
                times, data, yerr,
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
                        allow_negative_edepths=False):

    if base_name is None:
        base_name = 'WASP43_fine_grain_photometry_20x20_208ppm'

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


def instantiate_arctor(planet_name, data_dir, working_dir, file_type,
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
                       base_name='WASP43_fine_grain_photometry_208ppm'):

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
                             base_name='WASP43_fine_grain_photometry_208ppm'):

    base_name = f'{base_name}_near_best_{n_space}x{n_space}'

    start0 = time()

    # Linear Eclipse Depths with Negative Allowed
    start1 = time()
    print('Linear Eclipse depth fits - Default everything')
    fine_grain_mcmcs, filename = run_multiple_pymc3(
        times, fine_snr_flux, fine_snr_uncs, near_best_apertures_NxN_small,
        t0=t0_guess, u=u, period=period_wasp43, b=b_wasp43,
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
        t0=t0_guess, u=u, period=period_wasp43, b=b_wasp43,
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
        t0=t0_guess, u=u, period=period_wasp43, b=b_wasp43,
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
        t0=t0_guess, u=u, period=period_wasp43, b=b_wasp43,
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
        t0=t0_guess, u=u, period=period_wasp43, b=b_wasp43, idx_fwd=idx_fwd, idx_rev=idx_rev,
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
        t0=t0_guess, u=u, period=period_wasp43, b=b_wasp43, idx_fwd=idx_fwd, idx_rev=idx_rev,
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
        t0=t0_guess, u=u, period=period_wasp43, b=b_wasp43,
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
        t0=t0_guess, u=u, period=period_wasp43, b=b_wasp43,
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
        t0=t0_guess, u=u, period=period_wasp43, b=b_wasp43,
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
        t0=t0_guess, u=u, period=period_wasp43, b=b_wasp43,
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
        t0=t0_guess, u=u, period=period_wasp43, b=b_wasp43,
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
        t0=t0_guess, u=u, period=period_wasp43, b=b_wasp43,
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

from astropy.modeling.models import Planar2D, Linear1D
from astropy.modeling.fitting import LinearLSQFitter
from matplotlib import pyplot as plt


def fit_2D_time_vs_other(times, flux, other, idx_fwd, idx_rev,
                         xytext=(15, 15), n_sig=5, varname='Other',
                         n_spaces=[10, 10], convert_to_ppm=True, lw=3,
                         fig=None, ax=None):
    ppm = 1e6

    if fig is None and ax is None:
        fig, ax = plt.subplots()

    if ax is None:
        ax = fig.add_subplot(111)

    if ax is None:
        fig, ax = plt.subplots()

    inliers = np.sqrt((other - np.median(other))**2 + (flux - np.median(flux))
                      ** 2) < n_sig * np.sqrt(np.var(other) + np.var(flux))

    fitter_o = LinearLSQFitter()
    fitter_t = LinearLSQFitter()

    model_o = Linear1D(slope=1e-6, intercept=np.median(flux))
    model_t = Linear1D(slope=-1e-3, intercept=0)

    times_med = np.median(times[inliers])
    times_std = np.median(times[inliers])

    times_normed = (times - times_med) / times_std

    other_med = np.median(other[inliers])
    other_std = np.median(other[inliers])
    other_normed = (other - other_med) / other_std

    flux_med = np.median(flux[inliers])
    flux_std = np.median(flux[inliers])
    flux_normed = (flux - flux_med) / flux_std

    fit_t = fitter_t(model_t, times_normed[inliers], flux_normed[inliers])

    flux_corrected = flux_normed - fit_t(times_normed)
    fit_o = fitter_o(model_o, other_normed[inliers], flux_corrected[inliers])

    model_comb = Planar2D(slope_x=fit_o.slope,
                          slope_y=fit_t.slope,
                          intercept=fit_t.intercept)

    fit_comb = fitter_t(model_comb,
                        other_normed[inliers],
                        times_normed[inliers],
                        flux_normed[inliers])

    # annotation = (f'o_slope:{fit_o.slope.value:0.2e}\n'
    #               f't_slope:{fit_t.slope.value:0.2e}\n'
    #               f'c_slope_o:{fit_comb.slope_x.value:0.2e}\n'
    #               f'c_slope_t:{fit_comb.slope_y.value:0.2e}\n'
    #               f'o_intcpt:{fit_o.intercept.value:0.2e}\n'
    #               f't_intcpt:{fit_t.intercept.value:0.2e}\n'
    #               f'c_intcpt:{fit_comb.intercept.value:0.2e}'
    #               )

    n_sp0, n_sp1 = n_spaces
    annotation = (f'2D Slope {varname}: {fit_comb.slope_x.value:0.2e}\n'
                  f'2D Slope Time:{" "*n_sp0}{fit_comb.slope_y.value:0.2e}\n'
                  f'2D Intercept:{" "*(n_sp1)}'
                  f'{fit_comb.intercept.value * flux_std * ppm:0.2f} [ppm]'
                  )

    min_o = other_normed.min()
    max_o = other_normed.max()
    min_t = times_normed.min()
    max_t = times_normed.max()

    ax.plot(other_normed[idx_fwd] * other_std,
            flux_normed[idx_fwd] * flux_std * ppm,
            'o', label='Forward Scan')
    ax.plot(other_normed[idx_rev] * other_std,
            flux_normed[idx_rev] * flux_std * ppm,
            'o', label='Reverse Scan')

    other_normed_th = np.linspace(min_o * 0.9, max_o * 1.1, 100)
    times_normed_th = np.linspace(min_t * 0.9, max_t * 1.1, 100)

    best_model = fit_comb(other_normed_th, times_normed_th)
    ax.plot(other_normed_th * other_std, best_model * flux_std * ppm,
            lw=lw, zorder=0)

    ax.set_title(f'{varname} + Time 2D Fit to Flux')
    ax.annotate(annotation,
                (0, 0),
                xycoords="axes fraction",
                xytext=xytext,
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=12,
                )

    ax.set_xlim(min_o - 1e-4 * other_std, max_o + 1e-4 * other_std)
    ax.set_ylabel('Flux [ppm]')
    ax.set_xlabel(f'{varname} [Median Subtracted]')
    ax.legend(loc=0)

    return fig, ax
