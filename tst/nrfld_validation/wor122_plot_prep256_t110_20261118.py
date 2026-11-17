#!/usr/bin/env python3
"""Plot the compact t=52 -> 110 XD RSG preparation diagnostics."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).with_name("wor122_prep_thermal_20261118")
OUT = Path(__file__).resolve().parents[2] / "doc/figures/wor122_prep256_t110_20261118.png"
DAY_PER_CODE = 21999.96568179141 / 86400.0


def main():
    histories = []
    transport = {}
    for stage in (52, 80, 110):
        directory = ROOT / f"prep256_fgmres_r40_t{stage}"
        histories.append(np.atleast_2d(np.loadtxt(
            directory / f"wor122_prep256_fgmres_r40_t{stage}.hst")))
        if stage >= 80:
            transport[stage] = json.loads((directory / "convective_flux.json").read_text())[
                "cases"][f"prep256_fgmres_r40_t{stage}"]["final"]["shells"]
    transport[52] = json.loads((ROOT / "prep256_fgmres_r40_t52/convective_flux.json").read_text())[
        "cases"]["prep256_fgmres_r40_t52"]["final"]["shells"]
    arr = np.vstack(histories)
    arr = arr[np.argsort(arr[:, 0])]
    x = arr[:, 0] * DAY_PER_CODE
    ordinary = arr[:, 1] >= 0.08
    axis = json.loads((ROOT / "prep256_t110_axis_flow.json").read_text())["snapshots"]

    fig, panels = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    ax = panels[0, 0]
    ax.plot(x, arr[:, 16], color="tab:red")
    ax.axhline(0.5, color="0.5", linestyle=":")
    ax.set(xlabel="Time [days]", ylabel="Global maximum Mach",
           title="Maximum occurs in density-floor exterior")
    ax = panels[0, 1]
    ax.plot(x[ordinary], arr[ordinary, 29] / arr[ordinary, 15],
            label="Preparation source / Lheat")
    ax.plot(x[ordinary], arr[ordinary, 22:28].sum(axis=1) / arr[ordinary, 15],
            label="Box radiative luminosity / Lheat")
    ax.set(xlabel="Time [days]", ylabel="Luminosity ratio",
           title="Preparation remains energetically important")
    ax.legend(fontsize=8)

    ax = panels[1, 0]
    for stage in (52, 80, 110):
        rows = list(transport[stage].values())
        ax.plot([row["mean_radius_Rsun"] for row in rows],
                [row["convective_gas_enthalpy_Lstar"] for row in rows],
                marker="o", label=f"t={stage} ({stage*DAY_PER_CODE:.1f} d)")
    ax.axhline(0, color="0.4", linewidth=0.7)
    ax.set(xlabel="Shell radius [Rsun]", ylabel="Gas enthalpy covariance / Lstar",
           title="Convective transport remains small")
    ax.legend(fontsize=8)

    ax = panels[1, 1]
    shells = ("0-166", "166-300", "300-500", "500-700", "700-830", "830-900")
    xx = np.arange(len(shells))
    for stage in (52, 80, 110):
        rows = axis[f"t{stage}"]["shells"]
        ratio = [rows[s]["axis15"]["rms_vr_kms"] /
                 rows[s]["diagonal"]["rms_vr_kms"] for s in shells]
        ax.plot(xx, ratio, marker="o", label=f"t={stage}")
    ax.axhline(1, color="0.4", linewidth=0.7)
    ax.set_xticks(xx, shells, rotation=35, ha="right")
    ax.set(ylabel="Axis / diagonal radial RMS velocity",
           title="Anisotropy is confined to outer shells")
    ax.legend(fontsize=8)
    for ax in panels.flat:
        ax.grid(alpha=0.25)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    print(OUT)


if __name__ == "__main__":
    main()
