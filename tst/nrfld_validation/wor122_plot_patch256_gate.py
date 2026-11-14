#!/usr/bin/env python3
"""Plot the matched six-hour uniform and local-patch 256³ RSG histories."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SECONDS_PER_CODE_TIME = 21999.96568179141


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    fig, (lum_ax, mach_ax) = plt.subplots(2, 1, figsize=(7.5, 6.0), sharex=True)
    for filename, label, color, style in (
        ("uniform256.hst", "uniform 256³", "#059669", "-"),
        ("onepatch256.hst", "256³ + one surface patch", "#f97316", "--"),
    ):
        history = np.loadtxt(args.data_dir / filename, comments="#", ndmin=2)
        day = history[:, 0] * SECONDS_PER_CODE_TIME / 86400.0
        luminosity = history[:, 22:28].sum(axis=1) / history[:, 15]
        lum_ax.plot(day, luminosity, style, color=color, linewidth=2,
                    marker="o", markersize=3, label=label)
        mach_ax.plot(day, history[:, 16], style, color=color, linewidth=2,
                     marker="o", markersize=3)

    lum_ax.axhline(1.0, color="#111827", linestyle=":", linewidth=1,
                   label="input luminosity")
    lum_ax.set_ylabel(r"boundary $L_{\rm rad}/L_{\rm heat}$")
    mach_ax.set_ylabel("maximum Mach")
    mach_ax.set_xlabel("time [day]")
    lum_ax.set_ylim(0.75, 1.11)
    mach_ax.set_ylim(0, 0.0053)
    mach_ax.set_xlim(0, 0.26)
    lum_ax.legend(frameon=False, fontsize=9)
    for axis in (lum_ax, mach_ax):
        axis.grid(alpha=0.25)
    fig.suptitle("RSG 256³: six-hour surface-patch gate", fontsize=12)
    fig.text(0.5, 0.015,
             "Fine-cell width 3.95 R☉ in one local sector; global peak Mach can reflect a local hotspot.",
             ha="center", fontsize=8, color="#4b5563")
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
