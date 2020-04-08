m_planet = 0.3
m_star = 1.45

u = [0.0]
r = 0.1

r_star = 1.5
t0 = 0.0
period = 10.0
ecc = 0.0
omega = 0.5

orbit = KeplerianOrbit(
    # m_star=m_star,
    # r_star=r_star,
    t0=t0,
    period=period,
    # ecc=ecc,
    # omega=omega,
    # m_planet=m_planet,
)

lc = LimbDarkLightCurve(u)
t = np.linspace(-0.1, 0.1, 1000) * period
model1 = lc.get_light_curve(r=r, orbit=orbit, t=t).eval()
plt.plot(t, model1)
