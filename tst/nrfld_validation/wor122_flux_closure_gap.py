#!/usr/bin/env python3
"""Audit the missing convective-flux closure in the imported RSG profile.

The efficient-MLT estimate is deliberately optimistic: it ignores eddy
radiative leakage and the back reaction of turbulent pressure on HSE.
It is a local feasibility diagnostic, not a replacement 1D stellar model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/rsg"))
from make_gray_rsg_hse import AR, entropy_from_rho_t, state_derivatives  # noqa: E402
from make_hse_model import AthenaEos, gravity  # noqa: E402
from wor122_mlt_transport_scale import (  # noqa: E402
    CLIGHT, G, LSUN, MSUN, RSUN, local_scale,
)

RADII_RSUN = (400, 450, 500, 550, 600, 650, 675, 700, 725,
              750, 780, 800, 813, 817)
WINDOWS_RSUN = (3, 5, 10)
POTENTIAL_MASS_MSUN = 12.0
STELLAR_RADIUS_RSUN = 830.0
SOFTENING_INNER_FRACTION = 0.2
SOFTENING_OUTER_FRACTION = 1.2


def gradient(profile, center_rsun, half_width_rsun):
    radius = profile[:, 0] / RSUN
    mask = np.abs(radius - center_rsun) <= half_width_rsun
    if np.count_nonzero(mask) < 5:
        raise ValueError("too few profile samples for local gradient")
    return float(np.polyfit(np.log(profile[mask, 5]),
                            np.log(profile[mask, 4]), 1)[0])


def closure_at_radius(profile, eos, index, alpha, luminosity):
    row = profile[index]
    r, rho, _, _, temperature, pressure = row[:6]
    scale = local_scale(eos, row, alpha, luminosity)
    if scale is None:
        raise ValueError(f"invalid efficient-MLT state at r={r / RSUN} Rsun")
    entropy = entropy_from_rho_t(eos, rho, temperature)
    pcheck, _, _, p_rho, _, t_rho, _ = state_derivatives(
        eos, rho, entropy)
    adiabatic = float(pcheck * t_rho / p_rho)
    required_excess = scale["required_excess_nabla"]
    radiative_fraction = float(4 * np.pi * r*r * row[9] / luminosity)
    if not 0 < radiative_fraction < 1:
        raise ValueError(f"invalid radiative fraction at r={r / RSUN} Rsun")
    # Solve the local coupled diffusion/MLT flux equation at fixed (rho,T).
    # This is still not an integrated profile: pressure and opacity would
    # change when the temperature gradient and turbulent support change.
    kappa = row[8]
    grav = gravity(r, G*POTENTIAL_MASS_MSUN*MSUN,
                   SOFTENING_INNER_FRACTION*STELLAR_RADIUS_RSUN*RSUN,
                   SOFTENING_OUTER_FRACTION*STELLAR_RADIUS_RSUN*RSUN)
    total_flux = luminosity/(4*np.pi*r*r)
    coefficient = 4*CLIGHT*AR*temperature**4/(3*kappa*pressure)
    cp = scale["cp_cgs"]
    expansion = scale["isobaric_expansion_q"]

    def local_fluxes(nabla):
        rad = coefficient*nabla*grav/(1+coefficient*nabla*kappa/CLIGHT)
        geff = grav-kappa*rad/CLIGHT
        height = pressure/(rho*geff)
        excess = max(nabla-adiabatic, 0.0)
        conv = (rho*cp*temperature*np.sqrt(grav*expansion)*alpha**2
                *np.sqrt(height)/np.sqrt(8.0)*excess**1.5)
        return rad, conv, height

    hi = max(1.0, adiabatic+1.0)
    while sum(local_fluxes(hi)[:2]) < total_flux:
        hi *= 2
        if hi > 100:
            raise ValueError("cannot bracket the local flux closure")
    solved_nabla = brentq(
        lambda value: sum(local_fluxes(value)[:2])-total_flux,
        0.0, hi)
    solved_rad, solved_conv, solved_height = local_fluxes(solved_nabla)
    if abs((solved_rad+solved_conv)/total_flux-1) > 1e-8:
        raise ArithmeticError("local flux closure did not converge")
    solved_speed = np.sqrt(grav*expansion*max(solved_nabla-adiabatic, 0.0)
                           *alpha**2*solved_height/8.0)
    gradients = {}
    capacities = {}
    for half_width in WINDOWS_RSUN:
        label = f"{half_width}Rsun"
        nabla = gradient(profile, r / RSUN, half_width)
        excess = max(nabla - adiabatic, 0.0)
        gradients[label] = nabla
        capacities[label] = float((excess / required_excess) ** 1.5)
    return {
        "radius_rsun": float(r / RSUN),
        "radiative_luminosity_fraction": radiative_fraction,
        "required_convective_luminosity_fraction": 1-radiative_fraction,
        "nabla_ad_from_discrete_entropy": adiabatic,
        "required_excess_nabla_efficient_mlt": required_excess,
        "required_total_nabla_efficient_mlt": adiabatic+required_excess,
        "required_total_nabla_local_flux_closed": solved_nabla,
        "local_flux_closed_radiative_fraction": solved_rad/total_flux,
        "local_flux_closed_convective_fraction": solved_conv/total_flux,
        "local_flux_closed_velocity_km_s": solved_speed/1e5,
        "local_flux_closed_turbulent_pressure_over_gas":
            rho*solved_speed**2/pressure,
        "measured_nabla_by_window": gradients,
        "optimistic_convective_capacity_over_required_by_window": capacities,
        "required_velocity_km_s": scale["velocity_km_s"],
        "required_mach": scale["mach"],
        "implied_turbulent_pressure_over_gas":
            scale["turbulent_pressure_over_gas"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--eos", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--luminosity-lsun", type=float, default=91932.0)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    profile = np.loadtxt(args.profile)
    eos = AthenaEos(args.eos)
    luminosity = args.luminosity_lsun * LSUN
    rows = {}
    for nominal in RADII_RSUN:
        index = int(np.argmin(abs(profile[:, 0] / RSUN - nominal)))
        rows[str(nominal)] = closure_at_radius(
            profile, eos, index, args.alpha, luminosity)
    result = {
        "method": __doc__.strip(),
        "profile": str(args.profile),
        "eos": str(args.eos),
        "luminosity_lsun": args.luminosity_lsun,
        "mixing_length_alpha": args.alpha,
        "velocity_coefficient_nu": 8.0,
        "potential_mass_msun": POTENTIAL_MASS_MSUN,
        "stellar_radius_rsun": STELLAR_RADIUS_RSUN,
        "softening_inner_fraction": SOFTENING_INNER_FRACTION,
        "softening_outer_fraction": SOFTENING_OUTER_FRACTION,
        "gradient_windows_rsun": list(WINDOWS_RSUN),
        "points": rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n")

    radii = [row["radius_rsun"] for row in rows.values()]
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                   constrained_layout=True)
    ax0.plot(radii, [row["nabla_ad_from_discrete_entropy"]
                     for row in rows.values()], "k--", label="EOS adiabatic")
    ax0.plot(radii, [row["measured_nabla_by_window"]["5Rsun"]
                     for row in rows.values()], "o-", label="Imported profile")
    ax0.plot(radii, [row["required_total_nabla_local_flux_closed"]
                     for row in rows.values()], "s-", label="Local flux closure")
    ax0.set(ylabel="d ln T / d ln P", title="One-dimensional temperature gradient")
    ax0.legend()
    for width, style in (("3Rsun", ":"), ("5Rsun", "-"), ("10Rsun", "--")):
        ax1.plot(radii, [max(1e-5, row[
            "optimistic_convective_capacity_over_required_by_window"][width])
            for row in rows.values()], linestyle=style, marker="o", label=width)
    ax1.axhline(1, color="k", linewidth=.8)
    ax1.set(xlabel="Radius [Rsun]",
            ylabel="Optimistic convective flux / required flux",
            yscale="log", ylim=(1e-5, 20),
            title="Deep-envelope transport deficit; surface MLT is unreliable")
    ax1.legend(fontsize=8)
    for ax in (ax0, ax1):
        ax.grid(alpha=.25, which="both")
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=180)
    plt.close(fig)
    print(args.json)
    print(args.figure)


if __name__ == "__main__":
    main()
