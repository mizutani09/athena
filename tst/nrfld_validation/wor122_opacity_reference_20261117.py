#!/usr/bin/env python3
"""Compare RSG Rosseland opacity tables at identical imported-profile states."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

RSUN = 6.957e10


def mesa_interpolator(path):
    with path.open() as handle:
        for _ in range(5):
            next(handle)
        logr = np.array([float(item) for item in next(handle).split()])
    table = np.loadtxt(path, skiprows=6)
    if table.shape[1] != len(logr) + 1:
        raise ValueError(f"Malformed MESA opacity table: {path}")
    return RegularGridInterpolator(
        (table[:, 0], logr), table[:, 1:], bounds_error=False,
        fill_value=np.nan)


def optab_interpolator(path):
    with h5py.File(path) as handle:
        logrho = handle["axes/log10_density"][:]
        logt = handle["axes/log10_temperature"][:]
        logk = handle["kappa/log10_rosseland"][:]
    return RegularGridInterpolator(
        (logrho, logt), logk, bounds_error=False, fill_value=np.nan)


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--wor55", type=Path, required=True)
    parser.add_argument("--t7", type=Path, required=True)
    parser.add_argument("--opal", type=Path, required=True)
    parser.add_argument("--lowt", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()

    profile = np.loadtxt(args.profile)
    radius = profile[:, 0] / RSUN
    logt = np.log10(profile[:, 4])
    logrho = np.log10(profile[:, 1])
    logr = logrho - 3.0 * logt + 18.0
    selected = (radius >= 100.0) & (radius <= 830.0)
    radius = radius[selected]
    logt, logrho, logr = logt[selected], logrho[selected], logr[selected]
    values = {
        "WOR-55 Optab": 10.0 ** optab_interpolator(args.wor55)(
            np.column_stack((logrho, logt))),
        "older t7": 10.0 ** optab_interpolator(args.t7)(
            np.column_stack((logrho, logt))),
        "MESA OPAL GS98": 10.0 ** mesa_interpolator(args.opal)(
            np.column_stack((logt, logr))),
        "MESA Ferguson GS98": 10.0 ** mesa_interpolator(args.lowt)(
            np.column_stack((logt, logr))),
    }
    samples = {}
    for target in (300, 500, 600, 700, 750, 780, 800, 810, 813, 817):
        i = np.argmin(abs(radius - target))
        samples[str(target)] = {
            "sample_radius_rsun": float(radius[i]),
            "rho_g_cm3": float(10.0 ** logrho[i]),
            "temperature_K": float(10.0 ** logt[i]),
            "log10_R": float(logr[i]),
            "opacity_cm2_g": {name: float(array[i]) for name, array in values.items()},
        }
    result = {
        "definition": "Opacity in cm2/g at identical imported-profile rho,T states; "
                      "MESA tables use logR=logrho-3logT+18 and are interpolated "
                      "bilinearly in log opacity. No extrapolation.",
        "inputs": {key: {"path": str(path), "sha256": checksum(path)}
                   for key, path in vars(args).items()
                   if key in ("profile", "wor55", "t7", "opal", "lowt")},
        "samples": samples,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for name, data in values.items():
        ax.plot(radius, data, label=name, lw=2 if name == "WOR-55 Optab" else 1.4)
    ax.set(xlim=(300, 825), ylim=(0.1, 100), yscale="log",
           xlabel="Radius in imported profile [Rsun]",
           ylabel="Rosseland opacity [cm2/g]")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.figure, dpi=180)
    plt.close(fig)
    print(args.json)
    print(args.figure)


if __name__ == "__main__":
    main()
