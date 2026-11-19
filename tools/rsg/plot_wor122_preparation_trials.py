#!/usr/bin/env python3
"""Plot local RSG preparation growth and one-day full-FLD release gates."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE = Path("tst/nrfld_validation/wor122_prep_shell_20261119")
OUT = Path("doc/figures/wor122_preparation_trials_20261119.png")
# The hydro time unit is length/sqrt(energy-density/density).
DAY_PER_CODE = 6.957e10 / np.sqrt(1e5 / 1e-8) / 86400


def case(filename, index=-1):
    return json.loads((BASE / filename).read_text())["snapshots"][index]


def shell(item):
    return item["shells"]["813-817"]


def main():
    first = json.loads((BASE / "coherent_nodiff_40r_transport.json").read_text())
    second = json.loads((BASE / "coherent_nodiff_ext80_160_transport.json").read_text())
    prepared = first["snapshots"] + second["snapshots"]
    # The second file starts at t≈100, so these times are unique.
    days = np.array([item["time_code"] * DAY_PER_CODE for item in prepared])
    flux = np.array([shell(item)["Fenth_mass_corrected_over_Fstar"]
                     for item in prepared])
    mach = np.array([shell(item)["max_mach"] for item in prepared])

    release_pairs = [
        (case("coherent_nodiff_40r_transport.json", 1),
         case("coherent_release_t20_transport.json", 0)),
        (case("coherent_nodiff_40r_transport.json", 3),
         case("coherent_release_t64_transport.json", 0)),
        (case("coherent_nodiff_ext80_160_transport.json", 0),
         case("coherent_release_t104_transport.json", 0)),
        (case("coherent_nodiff_ext80_160_transport.json", 3),
         case("coherent_release_t164_transport.json", 0)),
    ]
    release_days = np.array([before["time_code"] * DAY_PER_CODE
                             for before, _ in release_pairs])
    temp_retained = np.array([shell(after)["Tgas_K"] / shell(before)["Tgas_K"]
                              for before, after in release_pairs])
    flux_retained = np.array([
        shell(after)["Fenth_mass_corrected_over_Fstar"] /
        shell(before)["Fenth_mass_corrected_over_Fstar"]
        for before, after in release_pairs])
    release_mach = np.array([shell(after)["max_mach"]
                             for _, after in release_pairs])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), layout="constrained")
    axes[0].plot(days, flux, "o-", color="#147d64", label="prepared convection")
    axes[0].axhline(1, color="#aa4c4c", linestyle="--", linewidth=1,
                    label="stellar flux")
    axes[0].set_yscale("log")
    axes[0].set_ylim(1e-3, 2)
    axes[0].set_xlabel("Preparation duration [days]")
    axes[0].set_ylabel("Corrected convective flux / F★")
    axes[0].legend(fontsize=8, loc="upper left")

    axes[1].plot(release_days, temp_retained, "o-", color="#386db4",
                 label="mean T")
    axes[1].plot(release_days, flux_retained, "s-", color="#c27725",
                 label="convective flux")
    axes[1].axhline(1, color="0.4", linestyle="--", linewidth=1)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_xlabel("Preparation duration [days]")
    axes[1].set_ylabel("Fraction retained after 1 day of full FLD")
    axes[1].legend(fontsize=8)

    axes[2].plot(days, mach, "o-", color="#8068a8", label="preparation")
    axes[2].plot(release_days, release_mach, "s", color="#b44f60",
                 label="after 1 day full FLD")
    axes[2].axhline(1, color="0.4", linestyle="--", linewidth=1)
    axes[2].set_xlabel("Preparation duration [days]")
    axes[2].set_ylabel("Max Mach, 813–817 R☉")
    axes[2].legend(fontsize=8)

    fig.suptitle("Coarse local RSG patch (1.25 R☉ cells):"
                 " no-diffusion preparation and full-FLD release", fontsize=11)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    print(OUT)


if __name__ == "__main__":
    main()
