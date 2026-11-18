#!/usr/bin/env python3
"""Plot the matched t=110->120 RSG thermal-support taper gate."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).with_name("wor122_prep_thermal_20261118")
OUT = Path(__file__).resolve().parents[2] / "doc/figures/wor122_prep256_taper_gate_20261118.png"
DAY_PER_CODE = 21999.96568179141 / 86400.0


def main():
    evolution = json.loads((ROOT / "prep256_taper_shell_evolution.json").read_text())["cases"]
    histories = {}
    for mode in ("hold", "taper"):
        directory = ROOT / f"prep256_{mode}_t120"
        histories[mode] = np.atleast_2d(np.loadtxt(
            directory / f"wor122_prep256_{mode}_t120.hst"))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    colors = {"hold": "tab:blue", "taper": "tab:orange"}
    for mode, arr in histories.items():
        ordinary = arr[:, 1] >= 0.08
        axes[0, 0].plot(arr[ordinary, 0] * DAY_PER_CODE,
                        arr[ordinary, 29] / arr[ordinary, 15],
                        marker="o", color=colors[mode], label=mode)
        axes[0, 1].plot(arr[:, 0] * DAY_PER_CODE,
                        arr[:, 22:28].sum(axis=1) / arr[:, 15],
                        marker="o", color=colors[mode], label=mode)
    axes[0, 0].set(xlabel="Time [days]", ylabel="Preparation source / Lheat",
                   title="Instantaneous source on ordinary timesteps")
    axes[0, 1].set(xlabel="Time [days]", ylabel="Boundary Lrad / Lheat",
                   title="Radiative response to taper")
    shells = ("750-800", "800-813", "813-817", "817-830", "830-850",
              "850-880", "880-900")
    x = np.arange(len(shells))
    for mode in ("hold", "taper"):
        metrics = evolution[mode]["metrics"]
        for ax, field in ((axes[1, 0], "Tgas"), (axes[1, 1], "Er")):
            values = [metrics[s]["fields"][field][
                "mass_weighted_geometric_mean_ratio"] for s in shells]
            ax.plot(x, values, marker="o", color=colors[mode], label=mode)
    for ax, field in ((axes[1, 0], "Gas temperature"),
                      (axes[1, 1], "Radiation energy")):
        ax.axhline(1, color="0.4", linewidth=0.7)
        ax.set_xticks(x, shells, rotation=35, ha="right")
        ax.set(xlabel="Radius shell [Rsun]", ylabel=f"{field} / t=110 value",
               title=f"{field} change by t=120")
    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    print(OUT)


if __name__ == "__main__":
    main()
