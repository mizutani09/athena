#!/usr/bin/env python3
"""Trial RSG profile with local diffusion + MLT flux closure.

This is a structural diagnostic, not a 3D-ready initial condition. It offers
either efficient MLT or Henyey eddy cooling. Both omit turbulent pressure
from HSE. The exterior is gray until separately rebuilt with LTE FLD.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.optimize import brentq

from make_gray_rsg_hse import (
    AR, CLIGHT, G, LSUN, MSUN, RSUN, Opacity, AthenaEos,
    density_from_pt, diagnostics, extend_atmosphere, gravity,
    heating_fraction, photosphere_state, state_derivatives,
    entropy_from_rho_t,
)
from make_hse_model import mapped_mass


def local_closure(eos, opacity, r, pressure, temperature, args):
    """Return gradient, radiative flux, and an MLT velocity scale."""
    density = density_from_pt(eos, pressure, temperature)
    entropy = entropy_from_rho_t(eos, density, temperature)
    _, _, _, p_rho, p_s, t_rho, t_s = state_derivatives(
        eos, density, entropy)
    nabla_ad = t_rho * pressure / p_rho
    dln_t_dln_s_at_p = t_s - t_rho * p_s / p_rho
    cp = entropy / dln_t_dln_s_at_p
    expansion = p_s / (p_rho * dln_t_dln_s_at_p)
    if min(cp, expansion, nabla_ad) <= 0:
        raise ValueError(f"invalid MLT thermodynamics at r={r / RSUN:g} Rsun")
    radius = args.radius_rsun * RSUN
    grav = gravity(r, G * args.potential_mass_msun * MSUN,
                   args.r0_fraction * radius, args.r1_fraction * radius)
    kappa = opacity.rosseland(density, temperature)
    total_flux = args.luminosity_lsun * LSUN * heating_fraction(
        r, args.heating_radius_fraction * radius) / (4 * np.pi * r*r)
    coefficient = 4 * CLIGHT * AR * temperature**4 / (3 * kappa * pressure)

    def fluxes(nabla):
        radiative = coefficient*nabla*grav / (
            1 + coefficient*nabla*kappa/CLIGHT)
        geff = grav-kappa*radiative/CLIGHT
        if geff <= 0:
            raise ValueError("radiative acceleration exceeds gravity")
        scale_height = pressure/(density*geff)
        excess = max(nabla-nabla_ad, 0.0)
        eddy_optical_depth = kappa*density*args.alpha*scale_height
        if args.mlt_option == "henyey":
            # Ludwig, Freytag & Steffen (1999), Appendix A, Henyey row:
            # Gamma=(nabla-nabla_e)/(nabla_e-nabla_ad),
            # v^2=f1*l^2*g*delta*(nabla-nabla_e)/H,
            # Fconv=f2*rho*cp*v*T*l*(nabla-nabla_e)/H.
            f1, f2 = 1/8, 1/2
            # MESA documents a correction to Ludwig et al. Table A1:
            # the Henyey f3 is 8*f4, not f4/8 as printed there.
            f4 = 4*np.pi**2/3
            f3 = 8*f4
            sigma = AR*CLIGHT/4
            speed_coefficient = np.sqrt(
                f1*grav*expansion*args.alpha**2*scale_height)
            gamma_coefficient = (density*cp*speed_coefficient
                                 *eddy_optical_depth/(f3*sigma*temperature**3)
                                 *(1+f4/eddy_optical_depth**2))
            inverse_gamma_coefficient = 1/gamma_coefficient
            contrast_root = (2*excess/(np.sqrt(
                inverse_gamma_coefficient**2+4*excess)
                +inverse_gamma_coefficient)) if excess else 0.0
            contrast = contrast_root**2
            speed = speed_coefficient*contrast_root
            gamma = gamma_coefficient*contrast_root
            convective = f2*density*cp*temperature*args.alpha*speed*contrast
        else:
            # Same optimistic coefficient as the earlier local audit.
            contrast = excess
            speed = np.sqrt(grav*expansion*args.alpha**2*scale_height
                            *contrast/args.velocity_coefficient)
            convective = (density*cp*temperature*np.sqrt(grav*expansion)
                          *args.alpha**2*np.sqrt(scale_height)
                          /np.sqrt(args.velocity_coefficient)*contrast**1.5)
            gamma = None
        return radiative, convective, speed, geff, gamma, eddy_optical_depth

    hi = max(1.0, nabla_ad + 1.0)
    while sum(fluxes(hi)[:2]) < total_flux:
        hi *= 2
        if hi > 100:
            raise ValueError(f"cannot close luminosity at r={r / RSUN:g} Rsun")
    nabla = brentq(lambda value: sum(fluxes(value)[:2])-total_flux,
                   0.0, hi, xtol=1e-10, rtol=1e-10)
    radiative, convective, speed, geff, gamma, eddy_tau = fluxes(nabla)
    return dict(nabla=nabla, nabla_ad=nabla_ad, density=density,
                radiative_flux=radiative, convective_flux=convective,
                total_flux=total_flux, turbulent_pressure=density*speed**2,
                gas_pressure=pressure, gravity=grav, geff=geff,
                eddy_efficiency_gamma=gamma, eddy_optical_depth=eddy_tau)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eos", type=Path, required=True)
    parser.add_argument("--opacity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--mlt-option", choices=("efficient", "henyey"),
                        default="efficient")
    parser.add_argument("--velocity-coefficient", type=float, default=8.0)
    parser.add_argument("--potential-mass-msun", type=float, default=12.0)
    parser.add_argument("--radius-rsun", type=float, default=830.0)
    parser.add_argument("--luminosity-lsun", type=float, default=91932.0)
    parser.add_argument("--effective-temperature", type=float, default=3227.8)
    parser.add_argument("--r0-fraction", type=float, default=0.2)
    parser.add_argument("--r1-fraction", type=float, default=1.2)
    parser.add_argument("--heating-radius-fraction", type=float, default=0.2)
    parser.add_argument("--box-half-rsun", type=float, default=1010.5)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--points", type=int, default=768)
    parser.add_argument("--atmosphere-points", type=int, default=384)
    parser.add_argument("--density-floor", type=float, default=1e-21)
    parser.add_argument("--photosphere-tau", type=float, default=2.0/3.0)
    parser.add_argument("--rtol", type=float, default=2e-6)
    args = parser.parse_args()
    if args.alpha <= 0 or args.velocity_coefficient <= 0:
        parser.error("alpha and velocity coefficient must be positive")
    eos, opacity = AthenaEos(args.eos), Opacity(args.opacity)
    photosphere = photosphere_state(args, eos, opacity)
    radius = args.radius_rsun*RSUN
    rmin = radius*1e-8

    def rhs(r, y):
        pressure, temperature = np.exp(y)
        closure = local_closure(eos, opacity, r, pressure, temperature, args)
        dlnp = -closure["density"]*closure["geff"]/pressure
        return [dlnp, closure["nabla"]*dlnp]

    sol = solve_ivp(rhs, (radius, rmin),
                    [np.log(photosphere[1]), np.log(photosphere[3])],
                    t_eval=np.linspace(radius, rmin, args.points),
                    method="DOP853", rtol=args.rtol, atol=(1e-9, 1e-9))
    if not sol.success:
        raise RuntimeError(sol.message)
    r = sol.t[::-1]
    pressure, temperature = np.exp(sol.y[:, ::-1])
    density = np.array([density_from_pt(eos, p, t)
                        for p, t in zip(pressure, temperature)])
    mass = cumulative_trapezoid(4*np.pi*r*r*density, r, initial=0)
    mass += 4*np.pi*r[0]**3*density[0]/3
    interior = (r, density, pressure, temperature, mass)
    profile = extend_atmosphere(args, eos, opacity, interior)
    egas, erad, kappa, flux, hse = diagnostics(args, eos, opacity, profile)
    columns = np.column_stack((profile[0], profile[1], egas, erad,
                               profile[3], profile[2], profile[4],
                               profile[5], kappa, flux, hse))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, columns, fmt="%.16e", header=(
        "r_cm rho_g_cm3 egas_erg_cm3 erad_erg_cm3 T_K Pgas_dyn_cm2 "
        "M_spherical_g tau kappa_R flux_rad hse_relative\n"
        f"{args.mlt_option} MLT local flux closure; no turbulent pressure in HSE"))
    samples = {}
    for nominal in (400, 500, 600, 700, 750, 800,
                    813, 814, 815, 816, 817, 818):
        i = int(np.argmin(abs(r/RSUN-nominal)))
        closed = local_closure(eos, opacity, r[i], pressure[i],
                               temperature[i], args)
        samples[str(nominal)] = {
            "radius_rsun": float(r[i]/RSUN),
            "nabla": float(closed["nabla"]),
            "radiative_luminosity_fraction": float(
                closed["radiative_flux"]/closed["total_flux"]),
            "turbulent_pressure_over_gas": float(
                closed["turbulent_pressure"]/closed["gas_pressure"]),
            "eddy_efficiency_gamma": (None if closed["eddy_efficiency_gamma"]
                                       is None else float(
                                           closed["eddy_efficiency_gamma"])),
            "eddy_optical_depth": float(closed["eddy_optical_depth"]),
            "closure_relative_error": float((
                closed["radiative_flux"]+closed["convective_flux"])
                /closed["total_flux"]-1),
        }
    measured = {
        "model": f"{args.mlt_option}-MLT flux-closed trial, no turbulent pressure in HSE",
        "mlt_option": args.mlt_option,
        "alpha": args.alpha, "velocity_coefficient": args.velocity_coefficient,
        "mlt_coefficients": ({"f1": 1/8, "f2": 1/2,
                              "f3_corrected": 32*np.pi**2/3,
                              "f4": 4*np.pi**2/3}
                             if args.mlt_option == "henyey"
                             else {"nu": args.velocity_coefficient}),
        "eos": str(args.eos), "opacity": str(args.opacity),
        "surface_temperature_K": args.effective_temperature,
        "mass_inside_radius_msun": float(mass[-1]/MSUN),
        "mapped_mass_msun": float(mapped_mass(profile[0], profile[1],
                                               args.grid,
                                               args.box_half_rsun*RSUN)/MSUN),
        "central_logrho_logT": [float(np.log10(density[0])),
                                 float(np.log10(temperature[0]))],
        "max_abs_resolved_interior_hse": float(np.max(np.abs(hse[
            (profile[0] > .01*radius) & (profile[0] < .99*radius)]))),
        "samples": samples,
    }
    args.output.with_suffix(args.output.suffix+".json").write_text(
        json.dumps(measured, indent=2)+"\n")
    print(json.dumps(measured, indent=2))


if __name__ == "__main__":
    main()
