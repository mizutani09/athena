#!/usr/bin/env python3
"""Plot the XD RSG surface-source transfer gate from compact JSON."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.analysis.read_text())
    snaps = data["snapshots"]
    time = np.array([s["time_code"] for s in snaps])
    history = data["history"]
    ht = np.array([h["time_code"] for h in history])
    shell = [s["shells"]["813-817"] for s in snaps]

    fig, axs = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    for name in ("800-813", "813-817", "817-830"):
        axs[0].plot(time, [s["shells"][name]["Tgas_over_initial"] for s in snaps],
                    marker="o", label=f"{name} solar radii")
    axs[0].axhline(1, color="0.5", lw=0.8)
    axs[0].set_ylabel("Gas temperature / initial")

    for key, label in (("Frad_over_Fstar", "radiation"),
                       ("Fenth_over_Fstar", "raw enthalpy"),
                       ("Fenth_mass_corrected_over_Fstar", "mass-corrected enthalpy")):
        axs[1].plot(time, [s[key] for s in shell], marker="o", label=label)
    axs[1].axhline(1, color="0.5", lw=0.8)
    axs[1].set_ylabel("813–817 radiative/enthalpy flux / Fstar")

    axs[2].plot(ht, [np.nan if h["time_code"] == 0
                     else h["top_Frad_over_Fstar"] for h in history],
                label="top radiative flux / Fstar")
    axs[2].plot(ht, [h["surface_heat_power_code"] /
                     (h["surface_heat_power_code"]+h["bottom_heat_power_code"])
                     for h in history], label="surface source fraction")
    axs[2].set_ylabel("Flux or source fraction")

    axs[3].plot(ht, [100*h["mass_drift"] for h in history],
                color="tab:blue", label="mass drift")
    axs[3].set_ylabel("Mass drift [%]", color="tab:blue")
    mach_axis = axs[3].twinx()
    mach_axis.plot(time, [s["max_mach"] for s in shell], marker="o",
                   color="tab:orange", label="813–817 max Mach")
    mach_axis.set_ylabel("813–817 max Mach", color="tab:orange")
    axs[3].set_xlabel("Code time (1 unit = 0.2546 day)")

    for ax in axs:
        ax.axvspan(0, 80, color="tab:blue", alpha=0.055)
        ax.axvspan(80, 160, color="tab:orange", alpha=0.075)
        ax.axvline(160, color="0.6", lw=0.8)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")
    mach_axis.legend(fontsize=8, loc="upper right")
    fig.suptitle("RSG local surface-patch transfer, XD job 342523")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)


if __name__ == "__main__":
    main()
