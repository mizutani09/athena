#!/usr/bin/env python3
"""Optimistic local MLT transport/time scales for the imported RSG profile.

Uses Goldberg, Jiang & Bildsten (2022) Eq. (2)-(3) with nabla_e=nabla_ad.
It is a diagnostic, not a closure for the lossy photospheric layer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/rsg"))
from make_gray_rsg_hse import density_from_pt, entropy_from_rho_t, nabla_ad_from_rho_u  # noqa: E402
from make_hse_model import AthenaEos, gravity  # noqa: E402

RSUN = 6.957e10
LSUN = 3.828e33
MSUN = 1.98847e33
G = 6.67430e-8
CLIGHT = 2.99792458e10


def local_scale(eos, row, alpha, luminosity):
    r, rho, egas, _, temp, pgas = row[:6]
    eps = 0.002
    tplus, tminus = temp * (1 + eps), temp * (1 - eps)
    rhop = density_from_pt(eos, pgas, tplus)
    rhom = density_from_pt(eos, pgas, tminus)
    sp = entropy_from_rho_t(eos, rhop, tplus)
    sm = entropy_from_rho_t(eos, rhom, tminus)
    cp = (sp - sm) / (2 * eps)
    q = -(np.log(rhop) - np.log(rhom)) / (np.log(tplus) - np.log(tminus))
    grav = gravity(r, G * 12 * MSUN, .2 * 830 * RSUN, 1.2 * 830 * RSUN)
    geff = grav - row[8] * row[9] / CLIGHT
    h = pgas / (rho * geff)
    fconv = luminosity / (4 * np.pi * r * r) - row[9]
    if min(cp, q, geff, fconv) <= 0:
        return None
    nu = 8.0
    excess = (fconv * np.sqrt(nu) / (
        rho * cp * temp * np.sqrt(grav * q) * alpha**2 * np.sqrt(h))) ** (2 / 3)
    speed = np.sqrt(grav * q * excess * alpha**2 * h / nu)
    nabla_ad = nabla_ad_from_rho_u(eos, rho, egas / rho)
    asq = 10.0 ** eos.interp(
        eos.fields[2], np.log10(pgas / rho * eos.ratios[2]), np.log10(rho)) * pgas / rho
    return dict(required_excess_nabla=float(excess), nabla_ad=float(nabla_ad),
                velocity_km_s=float(speed / 1e5), mach=float(speed / np.sqrt(asq)),
                eddy_time_days=float(h / speed / 86400),
                mixing_length_time_days=float(alpha * h / speed / 86400),
                turbulent_pressure_over_gas=float(rho * speed**2 / pgas),
                gas_scale_height_rsun=float(h / RSUN), cp_cgs=float(cp),
                isobaric_expansion_q=float(q))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--eos", type=Path, required=True)
    p.add_argument("--alpha", type=float, default=3.0)
    p.add_argument("--luminosity-lsun", type=float, default=91932.)
    p.add_argument("--json", type=Path, required=True)
    p.add_argument("--figure", type=Path, required=True)
    args = p.parse_args()
    eos = AthenaEos(args.eos)
    profile = np.loadtxt(args.profile)
    luminosity = args.luminosity_lsun * LSUN
    results = {}
    for nominal in (500, 600, 700, 750, 780, 800, 810, 813, 817):
        i = int(np.argmin(abs(profile[:, 0] / RSUN - nominal)))
        row = profile[i]
        result = local_scale(eos, row, args.alpha, luminosity)
        results[str(nominal)] = {"radius_rsun": float(row[0] / RSUN), **result}
    payload = {"definition": __doc__.strip(), "alpha": args.alpha,
               "nu": 8, "profile": str(args.profile), "eos": str(args.eos),
               "points": results}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + "\n")
    radii = np.array([v["radius_rsun"] for v in results.values()])
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(radii, [v["eddy_time_days"] for v in results.values()], "o-",
            label="H / MLT velocity")
    ax.plot(radii, [v["mixing_length_time_days"] for v in results.values()], "s-",
            label="3 H / MLT velocity")
    ax.axhline(13.24, color="black", ls="--", label="Completed 256³ duration")
    ax.set(xlim=(480, 825), ylim=(1, 1000), yscale="log",
           xlabel="Radius [Rsun]", ylabel="Optimistic turnover time [days]")
    ax.grid(alpha=.25, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.figure, dpi=180)
    plt.close(fig)
    print(args.json)
    print(args.figure)


if __name__ == "__main__":
    main()
