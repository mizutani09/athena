#!/usr/bin/env python3
"""Plot discrete first-stage RSG FLD fluxes against the imported 1-D profile."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RSUN = 6.957e10
LSUN = 3.828e33


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--audit", nargs=2, action="append", metavar=("LABEL", "JSON"),
                        required=True)
    parser.add_argument("--luminosity-lsun", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    profile = np.loadtxt(args.profile)
    r = profile[:, 0] / RSUN
    lrad = 4*np.pi*profile[:, 0]**2*profile[:, 9]/(args.luminosity_lsun*LSUN)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                             gridspec_kw={"height_ratios": [1.5, 1]})
    axes[0].plot(r, lrad, color="black", lw=1.5, label="1D imported FLD flux")
    palette = ["#e69f00", "#0072b2", "#009e73"]
    for (label, path), color in zip(args.audit, palette):
        data = json.loads(Path(path).read_text())
        bins = data["radial_bins"]
        midpoint = np.array([(row["rlo"]+row["rhi"])/2 for row in bins])
        luminosity = np.array([row["spherical_luminosity_estimate_over_lheat"]
                               for row in bins])
        rate = np.array([row["diffusion"]/data["lheat"] for row in bins])
        axes[0].plot(midpoint, luminosity, lw=1.3, label=f"{label} discrete faces",
                     color=color, alpha=0.9)
        axes[1].plot(midpoint, rate, lw=1.1, color=color, label=label)
    axes[0].axhline(1, ls=":", color="0.4", lw=0.8)
    axes[0].set_ylabel(r"Approx. $L_{\rm rad}/L_{\rm heat}$")
    axes[0].set_ylim(-0.05, 2.5)
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].grid(alpha=0.2)
    axes[1].axhline(0, color="0.4", lw=0.8)
    axes[1].set_ylabel(r"$\dot E_{\rm rad,diff}/L_{\rm heat}$ per 1 $R_\odot$ bin")
    axes[1].set_xlabel(r"Radius [$R_\odot$]")
    axes[1].set_ylim(-1.3, 1.3)
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="lower right", fontsize=9)
    axes[1].set_xlim(790, 850)
    fig.suptitle("Initial pre-implicit FLD balance on the Cartesian RSG grid")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(args.output)


if __name__ == "__main__":
    main()
