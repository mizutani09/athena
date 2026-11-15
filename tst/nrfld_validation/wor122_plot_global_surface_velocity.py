#!/usr/bin/env python3
"""Plot global first-step and short-extension RSG surface velocity changes."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SHELLS = ((750, 800), (800, 813), (813, 817),
          (817, 830), (830, 850), (850, 880))


def values(data, case, metric):
    rows = data["cases"][case]["metrics"]
    return np.array([100*rows[f"{lo}-{hi}:all"][metric]
                     for lo, hi in SHELLS])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    first = json.loads((args.data_dir / "global_first_step256.json").read_text())
    extended = json.loads((args.data_dir / "global_t1_256.json").read_text())
    r = np.array([(lo+hi)/2 for lo, hi in SHELLS])
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.3), sharex=True)
    metric = "mass_weighted_rms_vector_acceleration_over_g"
    for case, label, color, marker in (
            ("donor", "donor (order 1)", "#2563eb", "o"),
            ("plm", "PLM (order 2)", "#059669", "s"),
            ("ppm", "PPM (order 3)", "#f97316", "^")):
        axes[0].plot(r, values(first, case, metric), marker=marker, color=color,
                     linewidth=1.8, label=label)
    for case, label, color, marker in (
            ("plm_t1", "PLM (order 2)", "#059669", "s"),
            ("ppm_t1", "PPM (order 3)", "#f97316", "^")):
        axes[1].plot(r, values(extended, case, metric), marker=marker,
                     color=color, linewidth=1.8, label=label)
    for ax in axes:
        ax.axvspan(813, 817, color="#f59e0b", alpha=0.15)
        ax.grid(alpha=0.22)
        ax.set_xlim(750, 880)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("RMS |Δv| / (Δt g) [%]")
        ax.legend(fontsize=8, frameon=False)
    axes[0].set_title("first step: Δt = 0.1 code units", fontsize=10)
    axes[1].set_title("short extension: Δt = 1 code unit", fontsize=10)
    axes[1].set_xlabel("radius [R☉]")
    fig.suptitle("RSG 256³ full-sphere early velocity response", fontsize=12)
    fig.text(0.5, 0.005, "Mass weighted within each spherical shell; initial density used as weight.",
             ha="center", fontsize=8, color="#4b5563")
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
