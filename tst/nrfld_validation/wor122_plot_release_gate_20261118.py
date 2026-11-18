#!/usr/bin/env python3
"""Plot the matched 256³ RSG thermal-support release experiment."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).with_name("wor122_prep_thermal_20261118")
OUT = Path(__file__).resolve().parents[2] / "doc/figures/wor122_prep256_release_gate_20261118.png"
DAY_PER_CODE = 21999.96568179141 / 86400.0


def history(mode):
    parent = "hold" if mode == "hold" else "taper"
    a = np.atleast_2d(np.loadtxt(ROOT / f"prep256_{parent}_t120"
                           / f"wor122_prep256_{parent}_t120.hst"))
    b = np.atleast_2d(np.loadtxt(ROOT / f"prep256_{mode}_t130"
                           / f"wor122_prep256_{mode}_t130.hst"))
    return np.vstack((a, b))


def main():
    shells = ("750-800", "800-813", "813-817", "817-830", "830-850",
              "850-880", "880-900")
    evolution = json.loads((ROOT / "prep256_release_shell_evolution.json").read_text())[
        "cases"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    colors = {"hold": "tab:blue", "off": "tab:orange"}
    for mode in ("hold", "off"):
        arr = history(mode)
        x = arr[:, 0] * DAY_PER_CODE
        ordinary = arr[:, 1] >= 0.08
        axes[0, 0].plot(x, arr[:, 22:28].sum(axis=1) / arr[:, 15],
                        color=colors[mode], marker="o", label=mode)
        axes[0, 1].plot(x[ordinary], arr[ordinary, 29] / arr[ordinary, 15],
                        color=colors[mode], marker="o", label=mode)
        axes[1, 0].plot(x, arr[:, 16], color=colors[mode], marker="o", label=mode)
        metrics = evolution[mode]["metrics"]
        ratios = [metrics[s]["fields"]["Tgas"][
            "mass_weighted_geometric_mean_ratio"] for s in shells]
        axes[1, 1].plot(np.arange(len(shells)), ratios, color=colors[mode],
                        marker="o", label=mode)
    axes[0, 0].set(xlabel="Time [days]", ylabel="Boundary Lrad / Lheat",
                   title="Radiative output after support release")
    axes[0, 1].set(xlabel="Time [days]", ylabel="Preparation source / Lheat",
                   title="Actual source on ordinary timesteps")
    axes[1, 0].set(xlabel="Time [days]", ylabel="Global maximum Mach",
                   title="Maximum remains in floor exterior")
    axes[1, 1].axhline(1, color="0.4", linewidth=0.7)
    axes[1, 1].set_xticks(np.arange(len(shells)), shells, rotation=35, ha="right")
    axes[1, 1].set(xlabel="Radius shell [Rsun]",
                   ylabel="Gas temperature / t=120 value",
                   title="Shell cooling over t=120–130")
    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    print(OUT)


if __name__ == "__main__":
    main()
