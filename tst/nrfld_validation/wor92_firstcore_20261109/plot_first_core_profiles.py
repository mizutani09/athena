#!/usr/bin/env python3
"""Plot volume-weighted radial profiles from the first-core 3D outputs."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_profile(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {key: np.array([float(row[key]) if row[key] else np.nan
                           for row in rows])
            for key in rows[0] if key != "cells"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    cases = ((670, "#4477aa"), (780, "#cc6677"), (1010, "#228833"))
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.5), sharex=True,
                             constrained_layout=True)
    for cycle, color in cases:
        profile = read_profile(root / f"c{cycle}_profile.csv")
        radius = np.sqrt(profile["r_inner_au"] * profile["r_outer_au"])
        radius[0] = profile["r_outer_au"][0] / 2
        mask = (radius > 0.5) & (radius < 20)
        label = f"cycle {cycle}"
        axes[0].plot(radius[mask], profile["mean_rho_g_cm3"][mask],
                     label=label, color=color, lw=1.8)
        axes[1].plot(radius[mask], profile["mass_weighted_temp_K"][mask],
                     color=color, lw=1.8)
        axes[2].plot(radius[mask],
                     -profile["mass_weighted_vrad_km_s"][mask],
                     color=color, lw=1.8)
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"Density [g cm$^{-3}$]")
    axes[1].set_ylabel("Gas temperature [K]")
    axes[2].set_ylabel("Infall speed [km/s]")
    axes[2].set_xlabel("Radius [au]")
    axes[0].legend(frameon=False)
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlim(0.5, 20)
        axis.grid(alpha=0.25)
    fig.suptitle("WOR-92: first-core formation in the adopted AMR trajectory")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
