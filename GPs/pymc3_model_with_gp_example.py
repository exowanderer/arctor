import multiprocessing as mp
import numpy as np
import exoplanet as xo
import pymc3 as pm

from matplotlib import pyplot as plt

from statsmodels.robust import scale as sc
from exomast_api import exoMAST_API

plt.ion()


def first_build_gp_pink_noise(times, data, dataerr,
                              log_Q=np.log(1.0 / np.sqrt(2))):

    log_S0 = pm.Normal("log_S0", mu=0.0, sigma=15.0,
                       testval=np.log(np.var(data)))
    log_w0 = pm.Normal("log_w0", mu=0.0, sigma=15.0,
                       testval=np.log(3.0))
    log_Sw4 = pm.Deterministic(
        "log_variance_r", log_S0 + 4 * log_w0)

    log_s2 = pm.Normal("log_variance_w", mu=0.0, sigma=15.0,
                       testval=np.log(np.var(data)))

    kernel = xo.gp.terms.SHOTerm(
        log_Sw4=log_Sw4, log_w0=log_w0, log_Q=log_Q)

    gp = xo.gp.GP(kernel, times, dataerr ** 2 + pm.math.exp(log_s2))

    return gp


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

    gp = xo.gp.GP(kernel, times, dataerr ** 2 + pm.math.exp(log_s2))

    return gp


def run_pymc3_with_gp(times, data, dataerr, orbit,
                      log_Q=np.log(1 / np.sqrt(2)),
                      tune=5000, draws=5000,
                      target_accept=0.9, u=[0]):

    # Create the PyMC3 model
    with pm.Model() as model:

        # The baseline flux
        mean = pm.Normal("mean", mu=0.0, sd=1.0)
        r = pm.Uniform("r", lower=0.0, upper=0.5, testval=0.15)

        # Compute the model light curve using starry
        light_curves = xo.LimbDarkLightCurve(u).get_light_curve(
            orbit=orbit, r=r, t=times
        )
        light_curve = pm.math.sum(light_curves, axis=-1) + mean

        # Here we track the value of the model light curve for plotting
        # purposes
        pm.Deterministic("light_curves", light_curves)

        # The likelihood function assuming known Gaussian uncertainty
        pm.Normal("obs", mu=light_curve, sd=dataerr, observed=data)

        gp = build_gp_pink_noise(times, data, dataerr, log_Q=log_Q)
        gp.marginal("gp", observed=data)

        # Fit for the maximum a posteriori parameters given the simuated
        # dataset
        map_soln = xo.optimize(start=model.test_point)

        # # MCMC the posterior distribution
        # with pm.Model() as model:
        np.random.seed(42)
        trace = pm.sample(
            tune=tune,
            draws=tune,
            start=map_soln,
            chains=mp.cpu_count(),
            step=xo.get_dense_nuts_step(target_accept=target_accept),
            cores=mp.cpu_count()
        )

        return trace, map_soln, model


def build_synthetic_model(min_phase=-0.1, max_phase=0.1, size=1000,
                          planet_name='HD 189733 b', u=[0.0]):

    # Planetary orbital parameters
    planet = exoMAST_API(planet_name)

    t0 = planet.transit_time
    deg2rad = np.pi / 180
    orbit = xo.orbits.KeplerianOrbit(
        t0=t0,
        period=planet.orbital_period,
        a=planet.a_Rs,
        # b=planet.impact_parameter,
        # incl=planet.inclination * deg2rad,  # None,  #
        duration=planet.transit_duration,  # None,  #
        ecc=planet.eccentricity,
        omega=planet.omega * deg2rad,
        m_planet=planet.Mp,
        r_star=None,  # planet.Rs,

        # Not in exoMAST or xo cannot use duplicates
        t_periastron=None,
        Omega=None,
        m_star=None,  # planet.Ms,
        rho_star=None,
        m_planet_units=None,
        rho_star_units=None,
        model=None,
        contact_points_kwargs=None,
    )

    star = xo.LimbDarkLightCurve(u)
    phase = np.linspace(min_phase, max_phase, size)
    times = phase * planet.orbital_period + t0
    model = star.get_light_curve(
        r=planet.Rp_Rs,
        orbit=orbit,
        t=times)

    model = model.eval().flatten()

    return times, model, orbit


def build_fake_data(sigma_ratio=0.1, size=1000):
    from exomast_api import exoMAST_API
    import colorednoise as cn

    ppm = 1e6

    std_gauss = 50 / ppm
    dataerr = np.random.uniform(-5, 5, size) / ppm + std_gauss
    times, synthetic_eclipse, _ = build_synthetic_model(size=size)

    pink_noise = cn.powerlaw_psd_gaussian(  # pink noise
        exponent=1, size=size)

    mad2std = 1.482601669
    med_noise = np.median(pink_noise)
    std_noise = sc.mad(pink_noise) * mad2std

    pink_noise = (pink_noise - med_noise) / std_noise * std_gauss * sigma_ratio
    data = np.random.normal(synthetic_eclipse, dataerr) + pink_noise

    return times, data, dataerr


def plot_results(map_soln, times, data, dataerr):

    plt.errorbar(times, data, dataerr, fmt='o', color='C0', ms=1)
    plt.plot(times, map_soln["light_curves"].flatten(),
             color='C1', lw=2, label="MAP Soln")

    plt.xlim(times.min(), times.max())
    plt.ylabel("relative flux")
    plt.xlabel("time [days]")
    plt.legend(fontsize=10)
    _ = plt.title("map model")


def plot_corner(trace, planet, varnames=None):
    # from pygtc import plotGTC
    import corner

    RpRs = planet.Rp_Rs
    mean = 0.0

    samples = pm.trace_to_dataframe(trace)

    varnames = [name for name in samples.columns if 'light_curves' not in name]
    # for name in varnames:
    #     if 'log' in name:
    #         samples[name.replace('log_', '')] = 10**samples[name]
    #         del samples[name]

    # varnames = [name for name in samples.columns if 'light_curves' not in name]
    # for name in varnames:
    #     if 'variance' in name:
    #         samples[name.replace('variance', 'sigma')] = 10**samples[name]
    #         del samples[name]

    # varnames = [name for name in samples.columns if 'light_curves' not in name]

    # truth = np.concatenate(
    #     xo.eval_in_model([mean, RpRs], model.test_point, model=model)
    # )
    _ = corner.corner(
        samples[varnames],
        bins=20,
        color='C0',
        smooth=True,
        smooth1d=True,
        labels=varnames,
        show_titles=True,
        truths=None,
        truth_color='C1',
        scale_hist=True,
    )


if __name__ == '__main__':
    n_pts = 1000
    times, synthetic_eclipse, orbit = build_synthetic_model(size=n_pts)
    times, data, dataerr = build_fake_data()

    plot_data = False
    if plot_data:
        plt.plot(times, synthetic_eclipse)
        plt.errorbar(times, data, dataerr, fmt='o')

    # # PyMC3 parameters
    log_Q = 1 / np.sqrt(2),
    tune = 100,
    draws = 100,
    target_accept = 0.9

    trace, map_soln, pm_model = run_pymc3_with_gp(times, data, dataerr, orbit,
                                                  log_Q=np.log(1 / np.sqrt(2)),
                                                  tune=5000, draws=5000,
                                                  target_accept=0.9)

    plot_results(map_soln, times, data, dataerr)
    plot_corner(trace, planet)
