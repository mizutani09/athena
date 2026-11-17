#!/usr/bin/env python3
"""Illustrative sonic two-stream enthalpy flux for the imported RSG profile.

This is a diagnostic scale, not a physical upper bound or an MLT solution.
At each radius, two equal-area streams share the profile gas pressure and
have temperatures T(1+delta) and T(1-delta). The hot stream rises at the
local adiabatic sound speed; the cold stream descends at the speed needed
for zero mean mass flux. The EOS supplies both densities and enthalpies.
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
from make_gray_rsg_hse import density_from_pt  # noqa: E402
from make_hse_model import AthenaEos, gravity  # noqa: E402

RSUN = 6.957e10
LSUN = 3.828e33
MSUN = 1.98847e33
G = 6.67430e-8
CLIGHT = 2.99792458e10


def capacity(eos, row, contrast):
    radius, density, _, _, temperature, pressure = row[:6]
    gamma1 = 10.0 ** eos.interp(
        eos.fields[2],
        np.log10(pressure / density * eos.ratios[2]),
        np.log10(density),
    )
    sound_speed = np.sqrt(gamma1 * pressure / density)
    hot_t = temperature * (1.0 + contrast)
    cold_t = temperature * (1.0 - contrast)
    hot_rho = density_from_pt(eos, pressure, hot_t)
    cold_rho = density_from_pt(eos, pressure, cold_t)
    _, hot_e = eos.state_from_rho_temp(hot_rho, hot_t)
    _, cold_e = eos.state_from_rho_temp(cold_rho, cold_t)
    hot_h = (hot_e + pressure) / hot_rho
    cold_h = (cold_e + pressure) / cold_rho
    luminosity = 4.0 * np.pi * radius**2 * 0.5 * hot_rho * sound_speed * (hot_h - cold_h)
    return luminosity, sound_speed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--eos", type=Path, required=True)
    parser.add_argument("--luminosity-lsun", type=float, default=91932.0)
    parser.add_argument("--potential-mass-msun", type=float, default=12.0)
    parser.add_argument("--stellar-radius-rsun", type=float, default=830.0)
    parser.add_argument("--r0-fraction", type=float, default=0.2)
    parser.add_argument("--r1-fraction", type=float, default=1.2)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--transport-figure", type=Path)
    args = parser.parse_args()
    eos = AthenaEos(args.eos)
    data = np.loadtxt(args.profile)
    r = data[:, 0] / RSUN
    selected = np.where((r >= 500.0) & (r <= 825.0))[0]
    selected = selected[::max(1, len(selected) // 180)]
    luminosity = args.luminosity_lsun * LSUN
    rr = r[selected]
    radiative = 4.0 * np.pi * data[selected, 0] ** 2 * data[selected, 9] / luminosity
    required = 1.0 - radiative
    grav = np.array([gravity(radius, G * args.potential_mass_msun * MSUN,
                             args.r0_fraction * args.stellar_radius_rsun * RSUN,
                             args.r1_fraction * args.stellar_radius_rsun * RSUN)
                     for radius in data[selected, 0]])
    gamma_full = (data[selected, 8] * luminosity
                  / (4.0 * np.pi * data[selected, 0] ** 2 * CLIGHT * grav))
    contrasts = (0.1, 0.3, 0.5)
    curves = {str(q): [] for q in contrasts}
    cs = []
    for i in selected:
        speed = float("nan")
        for q in contrasts:
            try:
                value, speed = capacity(eos, data[i], q)
                curves[str(q)].append(value / luminosity)
            except ValueError:
                curves[str(q)].append(float("nan"))
        cs.append(speed / 1e5)

    table = {}
    for radius in (700, 750, 780, 800, 810, 813, 817):
        j = int(np.argmin(abs(rr - radius)))
        table[str(radius)] = {
            "sample_radius_rsun": float(rr[j]),
            "rosseland_opacity_cm2_g": float(data[selected[j], 8]),
            "radiative_luminosity_Lstar": float(radiative[j]),
            "local_eddington_luminosity_Lstar": float(1.0 / gamma_full[j]),
            "full_luminosity_radiation_acceleration_over_gravity": float(gamma_full[j]),
            "required_convective_luminosity_Lstar": float(required[j]),
            "sound_speed_km_s": float(cs[j]),
            "sonic_two_stream_Lstar": {key: float(values[j]) for key, values in curves.items()},
        }
    result = {
        "definition": __doc__.strip(),
        "profile": str(args.profile),
        "eos": str(args.eos),
        "luminosity_lsun": args.luminosity_lsun,
        "potential_mass_msun": args.potential_mass_msun,
        "stellar_radius_rsun": args.stellar_radius_rsun,
        "r0_fraction": args.r0_fraction,
        "r1_fraction": args.r1_fraction,
        "table": table,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(rr, np.maximum(required, 1e-4), color="black", lw=2.1,
            label="Lstar minus imported radiative luminosity")
    for q in contrasts:
        ax.plot(rr, curves[str(q)], lw=1.6, label=f"sonic two-stream, T contrast ±{q:.0%}")
    ax.set(xlim=(650, 825), ylim=(0.01, 2.0), yscale="log",
           xlabel="Radius [Rsun]", ylabel="Luminosity / Lstar")
    ax.axvspan(813, 817, color="gray", alpha=0.12)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(args.figure, dpi=180)
    plt.close(fig)
    if args.transport_figure:
        args.transport_figure.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        ax.plot(rr, radiative, color="tab:blue", lw=2.0,
                label="Imported radiative luminosity")
        ax.plot(rr, 1.0 / gamma_full, color="tab:red", lw=1.8,
                label="Local Eddington luminosity")
        ax.axhline(1.0, color="black", ls="--", lw=1.0, label="Input luminosity")
        ax.set(xlim=(450, 825), ylim=(0, 1.3), xlabel="Radius [Rsun]",
               ylabel="Luminosity / Lstar")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.transport_figure, dpi=180)
        plt.close(fig)
        print(args.transport_figure)
    print(args.json)
    print(args.figure)


if __name__ == "__main__":
    main()
