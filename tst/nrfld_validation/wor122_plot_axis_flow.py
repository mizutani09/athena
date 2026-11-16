#!/usr/bin/env python3
"""Plot axis and diagonal radial velocities from the RSG cone audit."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).with_name("wor122_prep_thermal_20261115")


def main():
    snapshots = json.loads((ROOT / "axis_flow_t20_t28.json").read_text())["snapshots"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    panels = ((("0-166", "166-300", "300-500"), "interior"),
              (("700-830", "830-900"), "surface"))
    for ax, (shells, title) in zip(axes, panels):
        for time, color in (("t20", "#2c6fbb"), ("t28", "#b65336")):
            for region, style in (("axis15", "-o"), ("diagonal", "--s")):
                values = [snapshots[time]["shells"][shell][region]
                          ["mass_weighted_vr_kms"] for shell in shells]
                ax.plot(range(len(shells)), values, style, color=color,
                        label=f"{time}, {region}", linewidth=1.5,
                        markersize=4)
        ax.set_xticks(range(len(shells)), shells)
        ax.set_xlabel("radius shell [$R_\\odot$]")
        ax.set_title(title)
        ax.axhline(0.0, color="black", linewidth=0.6)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("mass-weighted radial velocity [km/s]")
    axes[1].legend(fontsize=8)
    fig.savefig(ROOT / "axis_flow_t20_t28.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
