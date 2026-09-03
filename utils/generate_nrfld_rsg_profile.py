#!/usr/bin/env python3
"""Generate a composite modified-Lane-Emden RSG envelope for NR-FLD.

The softened point-mass core and the gas envelope both contribute to the
fixed gravity.  A smoothly varying polytropic index makes only the outer
envelope convectively unstable.  The integrated pressure is TOTAL pressure;
LTE gas and radiation components are split afterwards.
"""

import argparse
import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

G = 6.67430e-8
KB = 1.380649e-16
# Atomic mass unit, consistent with R_gas used by Athena++'s ideal-gas FLD coupling.
MH = 1.66053906660e-24
AR = 7.5657e-15
C_LIGHT = 2.99792458e10
MSUN = 1.98847e33
RSUN = 6.957e10
LSUN = 3.828e33


def core_factor(u):
    if u < 0.5:
        return 32.0/3.0 - 192.0*u*u/5.0 + 32.0*u**3
    if u < 1.0:
        return (-1.0/(15.0*u**3) + 64.0/3.0 - 48.0*u
                + 192.0*u*u/5.0 - 32.0*u**3/3.0)
    return 1.0/u**3


def g_core(r, mass, hsoft):
    if r <= 0.0:
        return 0.0
    return G*mass*r*core_factor(r/hsoft)/hsoft**3


def n_poly(r, radius, n_inner, n_outer, transition, width):
    q = (r/radius-transition)/width
    w = 0.5*(1.0+math.tanh(q))
    return n_inner + w*(n_outer-n_inner)


def integrate(logpars, args, dense=False):
    rho0, p0 = 10.0**logpars[0], 10.0**logpars[1]
    r0 = args.radius*1.0e-8
    m0 = 4.0*math.pi*r0**3*rho0/3.0
    y0 = [math.log(rho0), math.log(p0), m0]

    def rhs(r, y):
        rho, press, menv = math.exp(y[0]), math.exp(y[1]), max(y[2], 0.0)
        grav = g_core(r, args.core_mass, args.hsoft) + G*menv/max(r*r, 1.0e-99)
        nloc = n_poly(r, args.radius, args.n_inner, args.n_outer,
                      args.transition, args.transition_width)
        gamma_p = 1.0 + 1.0/nloc
        dlnp = -rho*grav/press
        return [dlnp/gamma_p, dlnp, 4.0*math.pi*r*r*rho]

    r_eval = np.geomspace(r0, args.radius, args.points) if dense else None
    return solve_ivp(rhs, (r0, args.radius), y0, method="DOP853", t_eval=r_eval,
                     rtol=2.0e-10, atol=[2.0e-11, 2.0e-11, 1.0e20])


def temperature_from_total_pressure(rho, ptot, mu):
    rgas = KB/(mu*MH)
    out = np.empty_like(rho)
    for q, (d, p) in enumerate(zip(rho, ptot)):
        lo, hi = 0.0, max(p/(d*rgas), (3.0*p/AR)**0.25)
        for _ in range(100):
            mid = 0.5*(lo+hi)
            f = d*rgas*mid + AR*mid**4/3.0 - p
            if f > 0.0:
                hi = mid
            else:
                lo = mid
        out[q] = 0.5*(lo+hi)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="rsg10_modified_le.dat")
    parser.add_argument("--mass-msun", type=float, default=10.0)
    parser.add_argument("--radius-rsun", type=float, default=510.0)
    parser.add_argument("--core-fraction", type=float, default=0.10)
    parser.add_argument("--softening-fraction", type=float, default=0.05)
    parser.add_argument("--mu", type=float, default=0.62)
    parser.add_argument("--n-inner", type=float, default=3.0)
    parser.add_argument("--n-outer", type=float, default=1.45)
    parser.add_argument("--transition", type=float, default=0.70)
    parser.add_argument("--transition-width", type=float, default=0.025)
    parser.add_argument("--surface-temperature", type=float, default=3500.0)
    parser.add_argument("--luminosity-lsun", type=float, default=1.0e4)
    parser.add_argument("--heat-radius-fraction", type=float, default=0.20)
    parser.add_argument("--points", type=int, default=4096)
    args = parser.parse_args()
    args.mass = args.mass_msun*MSUN
    args.radius = args.radius_rsun*RSUN
    args.core_mass = args.core_fraction*args.mass
    args.env_mass = args.mass-args.core_mass
    args.hsoft = args.softening_fraction*args.radius

    def residual(x):
        sol = integrate(x, args)
        if not sol.success:
            return [10.0, 10.0]
        rho_surface = math.exp(sol.y[0, -1])
        p_surface = math.exp(sol.y[1, -1])
        t_surface = temperature_from_total_pressure(
            np.array([rho_surface]), np.array([p_surface]), args.mu)[0]
        return [math.log10(sol.y[2, -1]/args.env_mass),
                math.log10(t_surface/args.surface_temperature)]

    initial_guess = [-5.0, 8.5] if args.softening_fraction >= 0.10 else [-4.2, 9.74]
    fit = least_squares(residual, initial_guess, xtol=1e-11, ftol=1e-11,
                        gtol=1e-11, max_nfev=300, diff_step=1.0e-4)
    if not fit.success or np.max(np.abs(residual(fit.x))) > 2.0e-5:
        raise RuntimeError(f"profile shooting failed: {fit.message}; residual={residual(fit.x)}")
    sol = integrate(fit.x, args, dense=True)
    r = sol.t
    rho = np.exp(sol.y[0])
    ptot = np.exp(sol.y[1])
    menv = sol.y[2]
    temp = temperature_from_total_pressure(rho, ptot, args.mu)
    pgas = rho*KB*temp/(args.mu*MH)
    erad = AR*temp**4
    grav = np.array([g_core(rr, args.core_mass, args.hsoft) for rr in r])
    grav += G*menv/r**2
    nloc = np.array([n_poly(rr, args.radius, args.n_inner, args.n_outer,
                            args.transition, args.transition_width) for rr in r])
    # Invert the FLD closure for an opacity that transports the requested
    # luminosity without initially changing E_rad. Uniform volumetric heating
    # inside heat_radius implies L(r) proportional to r^3 there.
    luminosity = args.luminosity_lsun*LSUN
    heat_radius = args.heat_radius_fraction*args.radius
    lum_r = luminosity*np.minimum((r/heat_radius)**3, 1.0)
    flux = lum_r/(4.0*np.pi*r*r)
    flux_fraction = np.minimum(flux/(C_LIGHT*erad), 1.0-1.0e-12)
    fld_r = np.empty_like(r)
    for q, target in enumerate(flux_fraction):
        lo, hi = 0.0, max(1.0, 2.0/(1.0-target))
        for _ in range(100):
            mid = 0.5*(lo+hi)
            limiter = (2.0+mid)/(6.0+3.0*mid+mid*mid)
            if mid*limiter < target:
                lo = mid
            else:
                hi = mid
        fld_r[q] = 0.5*(lo+hi)
    # Differentiate the same hydrostatic/polytropic equations used by the
    # integrator instead of differencing E_rad on the logarithmic grid.  Near
    # the origin E_rad is constant to many significant digits while both the
    # desired flux and its true gradient are proportional to r; np.gradient
    # therefore amplifies roundoff and creates a wildly oscillatory opacity.
    gamma_poly = 1.0 + 1.0/nloc
    dptot_dr = -rho*grav
    drho_dr = rho*(dptot_dr/ptot)/gamma_poly
    rgas = KB/(args.mu*MH)
    dtemp_dr = (dptot_dr-rgas*temp*drho_dr) / (
        rgas*rho + 4.0*AR*temp**3/3.0
    )
    dedr = 4.0*AR*temp**3*dtemp_dr
    kappa_eq = np.abs(dedr)/np.maximum(fld_r*rho*erad, 1.0e-99)
    kappa_eq = np.clip(kappa_eq, 1.0e-6, 1.0e6)
    data = np.column_stack((r, rho, ptot, temp, pgas, erad, menv, grav, nloc,
                            kappa_eq))
    header = "\n".join([
        "Composite modified Lane-Emden RSG profile (cgs)",
        "Ptot=Pgas+Erad/3; gravity is positive inward and fixed in the pgen",
        "columns: r_cm rho_g_cm3 ptot_dyn_cm2 T_K pgas_dyn_cm2 erad_erg_cm3 menv_g grav_cm_s2 n_poly kappa_eq_cm2_g",
        f"Mstar_Msun={args.mass_msun:.16e} Rstar_Rsun={args.radius_rsun:.16e} mu={args.mu:.16e}",
        f"Mcore_g={args.core_mass:.16e} hsoft_cm={args.hsoft:.16e}",
        f"n_inner={args.n_inner:.16e} n_outer={args.n_outer:.16e} "
        f"transition={args.transition:.16e} transition_width={args.transition_width:.16e}",
        f"luminosity_Lsun={args.luminosity_lsun:.16e} "
        f"heat_radius_fraction={args.heat_radius_fraction:.16e}",
    ])
    np.savetxt(args.output, data, fmt="%.16e", header=header)
    beta = pgas/ptot
    print(f"wrote {args.output}")
    print(f"rho_c={rho[0]:.8e} rho_surface={rho[-1]:.8e} g cm^-3")
    print(f"Ptot_c={ptot[0]:.8e} T_c={temp[0]:.8e} K")
    print(f"Menv={menv[-1]/MSUN:.8e} Msun beta=[{beta.min():.6f},{beta.max():.6f}]")
    print(f"kappa_eq=[{kappa_eq.min():.6e},{kappa_eq.max():.6e}] cm2 g^-1")
    print(f"fit_residual={residual(fit.x)}")


if __name__ == "__main__":
    main()
