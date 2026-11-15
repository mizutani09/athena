#!/usr/bin/env python3
"""Plot RSG first-stage surface gas-force decomposition and order controls."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def series(path, field):
    shells = json.loads(path.read_text())["shells"]
    keys = [key for key in shells if int(key.split("-")[1]) <= 850]
    radius = np.array([sum(map(float, key.split("-"))) / 2 for key in keys])
    return radius, np.array([field(shells[key]) for key in keys])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    fig, (residual_ax, gas_ax) = plt.subplots(2, 1, figsize=(8.3, 6.6), sharex=True)
    controls = (
        ("force256_donor.json", "donor cell (xorder=1)", "#2563eb", "o"),
        ("force256_axis.json", "PLM (xorder=2)", "#059669", "s"),
        ("force256_ppm.json", "PPM (xorder=3)", "#f97316", "^"),
    )
    for name, label, color, marker in controls:
        r, y = series(args.data_dir / name,
                      lambda row: row["mass_weighted_mean_radial_acceleration_over_g"]["residual"])
        residual_ax.plot(r, 100*y, marker=marker, color=color, label=label,
                         linewidth=1.8)
    decomposition = args.data_dir / "gas256_axis.json"
    stages = (
        ("continuum_profile", "1D derivative", "#374151", ":"),
        ("sampled_profile_centered", "sampled 1D, centered", "#2563eb", "--"),
        ("mapped_hdf_centered", "mapped HDF, centered", "#7c3aed", "-."),
        ("actual_godunov", "actual PLM + HLLC", "#059669", "-"),
    )
    for name, label, color, style in stages:
        r, y = series(decomposition,
                      lambda row: row["gas_support_over_g"][name])
        gas_ax.plot(r, 100*y, color=color, linestyle=style, marker="s",
                    markersize=4, linewidth=1.6, label=label)
    for ax in (residual_ax, gas_ax):
        ax.axvspan(813, 817, color="#f59e0b", alpha=0.15)
        ax.grid(alpha=0.22)
        ax.set_xlim(750, 850)
    residual_ax.axhline(0, color="#111827", linewidth=0.9)
    gas_ax.axhline(100, color="#111827", linewidth=0.9)
    residual_ax.set_ylabel("net radial acceleration / g [%]")
    gas_ax.set_ylabel("gas radial support / g [%]")
    gas_ax.set_xlabel("radius [R☉]")
    residual_ax.legend(fontsize=8, ncol=2, frameon=False)
    gas_ax.legend(fontsize=7.5, ncol=2, frameon=False)
    fig.suptitle("RSG initial surface: mapping, gradients, and Godunov flux", fontsize=12)
    fig.text(0.5, 0.005,
             "One 256³ axis block; shaded region is the 3.35 R☉ radiative transition.",
             ha="center", fontsize=8, color="#4b5563")
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
