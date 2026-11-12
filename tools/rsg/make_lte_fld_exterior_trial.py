#!/usr/bin/env python3
"""Experimental LTE FLD/HSE exterior for an existing RSG interior profile.

The inner atmosphere integrates the same Levermore–Pomraning limiter as Athena.
When the flux approaches free streaming, a narrow explicit join replaces the
singular FLD ODE.  This is a short-test candidate, not a converged star model.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp

from make_gray_rsg_hse import (AR, CLIGHT, G, LSUN, MSUN, RSUN, AthenaEos,
                               Opacity, density_from_pt, diagnostics, gravity)
from make_fld_exterior_trial import fld_luminosity


def inverse_limiter_flux(q):
    """Return R where lambda(R)*R=q, 0<q<1."""
    return (3*q-2+np.sqrt((2-3*q)**2+24*q*(1-q)))/(2*(1-q))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--eos', type=Path, required=True)
    parser.add_argument('--opacity', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--radius-rsun', type=float, default=830.)
    parser.add_argument('--luminosity-lsun', type=float, default=91932.)
    parser.add_argument('--potential-mass-msun', type=float, default=12.)
    parser.add_argument('--r0-fraction', type=float, default=.2)
    parser.add_argument('--r1-fraction', type=float, default=1.2)
    parser.add_argument('--heating-radius-fraction', type=float, default=.2)
    parser.add_argument('--density-floor', type=float, default=1e-21)
    parser.add_argument('--join-q', type=float, default=.99)
    parser.add_argument('--join-end-rsun', type=float, default=850.)
    args = parser.parse_args()
    if not 0 < args.join_q < 1:
        parser.error('join-q must lie in (0,1)')
    base = np.loadtxt(args.base)
    eos, opacity = AthenaEos(args.eos), Opacity(args.opacity)
    radius = args.radius_rsun*RSUN
    luminosity = args.luminosity_lsun*LSUN
    gm = G*args.potential_mass_msun*MSUN
    r0, r1 = args.r0_fraction*radius, args.r1_fraction*radius
    interior = base[:, 0] <= radius*(1.+1e-12)
    original = base[interior]
    surface = original[-1]
    p0, t0 = surface[5], surface[4]
    q0 = luminosity/(4*np.pi*radius**2*CLIGHT*AR*t0**4)
    if not 0 < q0 < args.join_q:
        raise ValueError(f'surface flux ratio {q0} must be below join-q')

    def state(x, logp, q):
        r = x*RSUN
        flux = luminosity/(4*np.pi*r*r)
        temperature = (flux/(CLIGHT*AR*q))**.25
        pressure = np.exp(logp)
        floor_p, _ = eos.state_from_rho_temp(args.density_floor, temperature)
        density = (args.density_floor if pressure <= floor_p
                   else density_from_pt(eos, pressure, temperature))
        kappa = opacity.rosseland(density, temperature)
        return flux, temperature, pressure, density, kappa, floor_p

    def coupled_rhs(x, y):
        q = min(max(y[1], 1e-12), .999999)
        flux, _, pressure, density, kappa, _ = state(x, y[0], q)
        acceleration = gravity(x*RSUN, gm, r0, r1)-kappa*flux/CLIGHT
        return [-RSUN*density*acceleration/pressure,
                q*(-2/x+RSUN*kappa*density*inverse_limiter_flux(q))]

    def reach_join(x, y):
        return args.join_q-y[1]
    reach_join.terminal = True
    reach_join.direction = -1
    inner = solve_ivp(coupled_rhs,
                      (args.radius_rsun, args.join_end_rsun),
                      [np.log(p0), q0], events=reach_join,
                      dense_output=True, max_step=.2,
                      rtol=1e-8, atol=1e-10)
    if not inner.success or not inner.t_events[0].size:
        raise RuntimeError('LTE FLD exterior failed to reach the streaming '
                           f'join before {args.join_end_rsun} Rsun: '
                           f'{inner.message}; q_end={inner.y[1,-1]}')
    join_radius = float(inner.t_events[0][0])

    def joined_q(x):
        t = np.clip((x-join_radius)/(args.join_end_rsun-join_radius), 0., 1.)
        return args.join_q+(1.-args.join_q)*t*t*(3.-2.*t)

    def pressure_rhs(x, y):
        flux, _, pressure, density, kappa, floor_p = state(x, y[0], joined_q(x))
        return [-RSUN*density*(gravity(x*RSUN, gm, r0, r1)
                               -kappa*flux/CLIGHT)/max(pressure, floor_p)]

    outer_radii = base[~interior, 0]/RSUN
    outer = solve_ivp(pressure_rhs, (join_radius, float(outer_radii[-1])),
                      [float(inner.y_events[0][0, 0])], dense_output=True,
                      max_step=2., rtol=1e-8, atol=1e-10)
    if not outer.success:
        raise RuntimeError(f'outer HSE integration failed: {outer.message}')

    records = []
    for x in outer_radii:
        if x <= join_radius:
            logp, q = inner.sol(x)
        else:
            logp, q = outer.sol(x)[0], joined_q(x)
        flux, temperature, pressure, density, kappa, floor_p = state(x, logp, q)
        pressure = max(pressure, floor_p)
        records.append((x*RSUN, density, pressure, temperature, kappa, flux, q))
    records = np.asarray(records)
    r = np.r_[original[:, 0], records[:, 0]]
    rho = np.r_[original[:, 1], records[:, 1]]
    pressure = np.r_[original[:, 5], records[:, 2]]
    temperature = np.r_[original[:, 4], records[:, 3]]
    outer_r = np.r_[surface[0], records[:, 0]]
    outer_rho = np.r_[surface[1], records[:, 1]]
    outer_kappa = np.r_[surface[8], records[:, 4]]
    outer_mass = original[-1, 6]+cumulative_trapezoid(
        4*np.pi*outer_r**2*outer_rho, outer_r, initial=0.)[1:]
    mass = np.r_[original[:, 6], outer_mass]
    outer_tau = np.maximum(0., 2./3.-cumulative_trapezoid(
        outer_rho*outer_kappa, outer_r, initial=0.)[1:])
    tau = np.r_[original[:, 7], outer_tau]
    profile = (r, rho, pressure, temperature, mass, tau)
    egas, erad, kappa, flux, hse = diagnostics(args, eos, opacity, profile)
    output = np.column_stack((r, rho, egas, erad, temperature, pressure,
                              mass, tau, kappa, flux, hse))
    fld_lum = fld_luminosity(output, luminosity)
    sample = (r/RSUN >= args.radius_rsun) & (r/RSUN <= 1000.)
    unfloored = sample & (rho > 1.001*args.density_floor)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, output, fmt='%.16e',
               header='r_cm rho_g_cm3 egas_erg_cm3 erad_erg_cm3 T_K '
                      'Pgas_dyn_cm2 M_spherical_g tau kappa_R flux_rad '
                      'hse_relative\nExperimental LTE FLD/HSE exterior '
                      'with streaming join')
    summary = {'base': str(args.base), 'output': str(args.output),
               'surface_temperature_K': float(t0), 'q_surface': float(q0),
               'join_q': args.join_q, 'join_radius_Rsun': join_radius,
               'join_end_Rsun': args.join_end_rsun,
               'min_rho_g_cm3': float(np.min(rho)),
               'FLD_L_over_Lstar_830_1000_range': [
                   float(np.min(fld_lum[sample])),
                   float(np.max(fld_lum[sample]))],
               'max_abs_unfloored_outer_HSE_residual': float(np.max(
                   np.abs(hse[unfloored]))),
               'max_abs_outer_HSE_residual': float(np.max(np.abs(
                   hse[len(original):])))}
    path = args.output.with_suffix(args.output.suffix+'.json')
    path.write_text(json.dumps(summary, indent=2)+'\n')
    print(path)


if __name__ == '__main__':
    main()
