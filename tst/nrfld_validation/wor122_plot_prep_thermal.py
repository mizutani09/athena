#!/usr/bin/env python3
"""Plot compact XD RSG preparation diagnostics saved beside this script."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).with_name("wor122_prep_thermal_20261115")
DAY_PER_CODE = 21999.96568179141 / 86400.0


def history(*names):
    return np.vstack([np.atleast_2d(np.loadtxt(ROOT / name)) for name in names])


def main():
    cases = {
        "PLM control": history(
            "uniform128_day1/wor122_uniform128_day1.hst",
            "uniform128_baseline_day2/wor122_uniform128_baseline_day2.hst"),
        "gas, 790–850 R$_\\odot$": history(
            "prep128_t8/wor122_prep128_t8.hst"),
        "gas, 16 R$_\\odot$ bins": history(
            "prep128_bin16_t8/wor122_prep128_bin16_t8.hst"),
        "gas, inner 790–830 R$_\\odot$": history(
            "prep128_inner_t8/wor122_prep128_inner_t8.hst"),
        "gas, outer 790–900 R$_\\odot$": history(
            "prep128_outer_t8/wor122_prep128_outer_t8.hst"),
    }
    for endpoint in ("t20", "t36"):
        extra = ROOT / f"prep128_outer_{endpoint}/wor122_prep128_outer_{endpoint}.hst"
        if extra.exists():
            cases["gas, outer 790–900 R$_\\odot$"] = np.vstack(
                [cases["gas, outer 790–900 R$_\\odot$"], np.loadtxt(extra)])
    ppm = ROOT / "ppm128_t8/wor122_ppm128_t8.hst"
    if ppm.exists():
        cases["PPM control"] = np.loadtxt(ppm)
    plm_prep = ROOT / "prep128_outer_plm_t8/wor122_prep128_outer_plm_t8.hst"
    if plm_prep.exists():
        cases["PLM gas, outer 790–900 R$_\\odot$"] = np.loadtxt(plm_prep)
        plm_followup = (ROOT / "prep128_outer_plm_t24"
                        / "wor122_prep128_outer_plm_t24.hst")
        if plm_followup.exists():
            cases["PLM gas, outer 790–900 R$_\\odot$"] = np.vstack(
                [cases["PLM gas, outer 790–900 R$_\\odot$"],
                 np.loadtxt(plm_followup)])
    prep256 = ROOT / "prep256_outer_t3/wor122_prep256_outer_t3.hst"
    if prep256.exists():
        cases["256³ PPM gas, outer 790–900 R$_\\odot$"] = np.loadtxt(prep256)
        followup256 = ROOT / "prep256_outer_t20/wor122_prep256_outer_t20.hst"
        if followup256.exists():
            cases["256³ PPM gas, outer 790–900 R$_\\odot$"] = np.vstack(
                [cases["256³ PPM gas, outer 790–900 R$_\\odot$"],
                 np.loadtxt(followup256)])
    seed = (ROOT / "prep128_outer_plm_seed1e2_t8"
            / "wor122_prep128_outer_plm_seed1e2_t8.hst")
    if seed.exists():
        cases["PLM gas, seed 1e-2"] = np.loadtxt(seed)
    surface_seed = (ROOT / "prep128_outer_plm_surface_seed_t8"
                    / "wor122_prep128_outer_plm_surface_seed_t8.hst")
    if surface_seed.exists():
        cases["PLM gas, surface seed 1e-2"] = np.loadtxt(surface_seed)

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True,
                             constrained_layout=True)
    for label, arr in cases.items():
        order = np.argsort(arr[:, 0])
        arr = arr[order]
        x = arr[:, 0] * DAY_PER_CODE
        axes[0].plot(x, arr[:, 16], label=label)
        # Short output-alignment steps make instantaneous powers misleading.
        ordinary = (arr[:, 1] >= 0.08) & (arr[:, 0] > 0)
        if label not in ("PLM control", "PPM control"):
            axes[1].plot(x[ordinary], (arr[ordinary, 29] / arr[ordinary, 15]),
                         label=label)
        axes[2].plot(x[ordinary], arr[ordinary, 22:28].sum(axis=1)
                     / arr[ordinary, 15],
                     label=label)
    axes[0].set_ylabel("maximum Mach number")
    axes[1].set_ylabel("preparation power / $L_{heat}$")
    axes[2].set_ylabel("box radiative luminosity / $L_{heat}$")
    axes[2].set_xlabel("time (days)")
    axes[0].legend(fontsize=8, loc="upper left")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.savefig(ROOT / "history_comparison.png", dpi=160)
    plt.close(fig)

    json_cases = {
        "PLM control": ("prep128_t8/shell_comparison.json", "baseline128"),
        "gas, 790–850": ("prep128_t8/shell_comparison.json", "prep128"),
        "gas, 16 R$_\\odot$ bins":
            ("prep128_bin16_t8/shell_comparison.json", "prep128"),
        "gas, inner 790–830":
            ("prep128_inner_t8/shell_comparison.json", "prep128"),
        "gas, outer 790–900":
            ("prep128_outer_t8/shell_comparison.json", "prep128"),
    }
    matched = ROOT / "ppm128_t8/matched_comparison.json"
    if matched.exists():
        json_cases["PPM control"] = ("ppm128_t8/matched_comparison.json",
                                     "ppm128")
    plm_matched = ROOT / "prep128_outer_plm_t8/matched_comparison.json"
    if plm_matched.exists():
        json_cases["PLM gas, outer 790–900"] = (
            "prep128_outer_plm_t8/matched_comparison.json", "prep_plm")
    seed_shells = ROOT / "prep128_outer_plm_seed1e2_t8/shell_evolution.json"
    if seed_shells.exists():
        json_cases["PLM gas, seed 1e-2"] = (
            "prep128_outer_plm_seed1e2_t8/shell_evolution.json", "seed1e2")
    surface_shells = ROOT / "prep128_outer_plm_surface_seed_t8/shell_evolution.json"
    if surface_shells.exists():
        json_cases["PLM gas, surface seed 1e-2"] = (
            "prep128_outer_plm_surface_seed_t8/shell_evolution.json", "surface_seed")
    shells = ("800-813", "813-817", "817-830", "830-850", "850-880")
    x = np.arange(len(shells))
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                             constrained_layout=True)
    for label, (filename, name) in json_cases.items():
        metrics = json.loads((ROOT / filename).read_text())["cases"][name]["metrics"]
        t = [metrics[s]["fields"]["Tgas"]["mass_weighted_geometric_mean_ratio"]
             for s in shells]
        vmax = [metrics[s]["max_final_speed_kms"] for s in shells]
        axes[0].plot(x, t, marker="o", label=label)
        axes[1].plot(x, vmax, marker="o", label=label)
    axes[0].axhline(1, color="0.3", linewidth=0.8)
    axes[0].set_ylabel("Tgas / initial Tgas (mass weighted)")
    axes[1].set_ylabel("maximum speed (km/s)")
    axes[1].set_xticks(x, shells)
    axes[1].set_xlabel("radius shell (R$_\\odot$)")
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.savefig(ROOT / "shell_comparison.png", dpi=160)


if __name__ == "__main__":
    main()
