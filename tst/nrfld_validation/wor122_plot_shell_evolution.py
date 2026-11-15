#!/usr/bin/env python3
"""Plot RSG surface thermal redistribution and radial motion at code time 1."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SHELLS = ((750, 800), (800, 813), (813, 817),
          (817, 830), (830, 850), (850, 880))


def shell_values(data, case, quantity):
    metrics = data["cases"][case]["metrics"]
    result = []
    for lo, hi in SHELLS:
        row = metrics[f"{lo}-{hi}"]
        if quantity == "vr":
            result.append(row["mean_final_vr_kms"])
        else:
            ratio = row["fields"][quantity]["mass_weighted_geometric_mean_ratio"]
            result.append(100*(ratio-1))
    return np.asarray(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main_data = json.loads((args.data_dir / "shell_evolution256.json").read_text())
    nodiff_data = json.loads((args.data_dir / "nodiff_shell_evolution256.json").read_text())
    r = np.array([(lo+hi)/2 for lo, hi in SHELLS])
    fig, axes = plt.subplots(3, 1, figsize=(8.1, 7.1), sharex=True)
    cases = ((main_data, "plm_1", "PLM + diffusion", "#059669", "s"),
             (main_data, "ppm_1", "PPM + diffusion", "#f97316", "^"),
             (nodiff_data, "ppm_nodiff_1", "PPM, diffusion off", "#2563eb", "o"))
    for data, case, label, color, marker in cases:
        for ax, quantity in zip(axes, ("Tgas", "Er", "vr")):
            ax.plot(r, shell_values(data, case, quantity), color=color,
                    marker=marker, linewidth=1.7, label=label)
    labels = ("mean ΔTgas [%]", "mean ΔEr [%]", "mean radial velocity [km/s]")
    for ax, label in zip(axes, labels):
        ax.axhline(0, color="#111827", linewidth=0.8)
        ax.axvspan(813, 817, color="#f59e0b", alpha=0.15)
        ax.grid(alpha=0.22)
        ax.set_xlim(750, 880)
        ax.set_ylabel(label)
    axes[0].legend(fontsize=8, frameon=False, ncol=2)
    axes[-1].set_xlabel("radius [R☉]")
    fig.suptitle("RSG 256³ surface after 1 code time (0.255 day)", fontsize=12)
    fig.text(0.5, 0.005,
             "Full-sphere, initial-mass weighted shell means; Δ is relative to time 0.",
             ha="center", fontsize=8, color="#4b5563")
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
