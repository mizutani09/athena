#!/usr/bin/env python3
"""Plot the Henyey-MLT local mapping and short FLD resolution audit."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tst/nrfld_validation/wor122_mlt_patch_discrete_20261119.json"
OUT = ROOT / "doc/figures/wor122_mlt_patch_discrete_20261119.png"


def main():
    data = json.loads(DATA.read_text())
    rows = data["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.7), constrained_layout=True)
    colors = ("#277da1", "#8e6bbd", "#e07a37")
    for row, color in zip(rows, colors):
        snaps = row["snapshots"]
        axes[0].plot([s["time_code"] * data["time_unit_hours"] for s in snaps],
                     [s["Tgas_K"] / snaps[0]["Tgas_K"] for s in snaps],
                     "o-", color=color, label=f"Δx={row['dx_rsun']} R☉")
    axes[0].set(xlabel="Time [hours]", ylabel="813–817 R☉ mean T / initial T",
                ylim=(0.94, 1.005))
    axes[0].legend(fontsize=8)

    dx = [r["dx_rsun"] for r in rows]
    axes[1].plot(dx, [r["Egas_over_abs_diffusion_hours"] for r in rows],
                 "o-", color="#277da1")
    axes[1].set(xlabel="Cell width [R☉]",
                ylabel="Initial gas energy / |FLD divergence| [hours]")

    axes[2].plot(dx, [100*r["force_815_817_mean_abs_over_rhog"] for r in rows],
                 "o-", label="mean absolute", color="#4b7c5f")
    axes[2].plot(dx, [100*r["force_815_817_max_abs_over_rhog"] for r in rows],
                 "s--", label="maximum", color="#e07a37")
    axes[2].set(xlabel="Cell width [R☉]",
                ylabel="815–817 R☉ force residual / ρg [%]")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle("Static Henyey-MLT mapping: finer FLD grid still cools")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    print(OUT)


if __name__ == "__main__":
    main()
