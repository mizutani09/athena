#!/usr/bin/env python3
"""Compare common first-core opacity means at rho=1e-10 g/cm3."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def table(path):
    with h5py.File(path) as source:
        density = source["axes/log10_density"][:]
        temperature = source["axes/log10_temperature"][:]
        means = {name: source["kappa/log10_" + name][:]
                 for name in ("planck", "rosseland")}
    return {name: RegularGridInterpolator((density, temperature), array,
                                          bounds_error=True)
            for name, array in means.items()}


def compare(semenov, absorption, optab, output):
    inputs = {"Semenov extinction": semenov,
              "Semenov absorption estimate": absorption,
              "Optab": optab}
    interpolators = {name: table(path) for name, path in inputs.items()}
    t = np.linspace(316.3, 431.3, 200)
    points = np.column_stack((np.full_like(t, -10.0), np.log10(t)))
    curve = {name: {mean: 10.0**interpolator[mean](points)
                    for mean in ("planck", "rosseland")}
             for name, interpolator in interpolators.items()}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    colors = {"Semenov extinction": "#bd6828",
              "Semenov absorption estimate": "#086b83",
              "Optab": "#4f627e"}
    for name in inputs:
        axes[0].plot(t, curve[name]["planck"], label=name, color=colors[name],
                     ls="--" if name == "Semenov absorption estimate" else "-")
        axes[1].plot(t, curve[name]["rosseland"], label=name,
                     color=colors[name], ls="--" if name == "Semenov absorption estimate" else "-")
    axes[0].set_title("Planck mean")
    axes[1].set_title("Rosseland mean")
    axes[1].set_yscale("log")
    axes[0].set_ylabel(r"Opacity [cm$^2$ g$^{-1}$]")
    axes[1].set_ylabel(r"Opacity [cm$^2$ g$^{-1}$]")
    for axis in axes:
        axis.set_xlabel("Gas temperature [K]")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(r"Common domain, $\rho=10^{-10}$ g cm$^{-3}$")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    at_431 = {name: {mean: float(np.interp(431.0, t, curve[name][mean]))
                     for mean in ("planck", "rosseland")}
              for name in inputs}
    summary = {
        "density_g_cm3": 1e-10, "temperature_range_K": [float(t[0]), float(t[-1])],
        "inputs": {name: {"path": str(path),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for name, path in inputs.items()},
        "at_431K_cm2_g": at_431,
        "trial_planck_over_optab_at_431K": (
            at_431["Semenov absorption estimate"]["planck"] /
            at_431["Optab"]["planck"]),
        "limitation": "Optab omits Semenov's low-temperature dust model; equal Planck means at one point do not imply equal physical opacity models",
        "plot": str(output),
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("semenov", type=Path)
    parser.add_argument("absorption_trial", type=Path)
    parser.add_argument("optab", type=Path)
    parser.add_argument("output_png", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.semenov, args.absorption_trial, args.optab,
                             args.output_png), indent=2))


if __name__ == "__main__":
    main()
