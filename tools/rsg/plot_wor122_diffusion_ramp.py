#!/usr/bin/env python3
"""Plot the compact WOR-122 preparation-release diagnostics."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tst/nrfld_validation/wor122_diffusion_ramp_20261119.json"
OUT = ROOT / "doc/figures/wor122_diffusion_ramp_20261119.png"


def values(rows, name):
    return [row[name] for row in rows]


def main():
    data = json.loads(DATA.read_text())
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    cases = (("first_ramp", "41-day preparation", "#277da1"),
             ("second_ramp", "61-day preparation", "#e07a37"))
    for key, label, color in cases:
        rows = data[key]
        t = values(rows, "offset_code")
        axes[0, 0].plot(t, values(rows, "T_transition_over_start"),
                        "o-", color=color, label=label)
        axes[0, 1].plot(t, values(rows, "Fconv_transition_over_Fstar"),
                        "o-", color=color, label=label)
        axes[1, 0].plot(t[1:], values(rows[1:], "top_Frad_over_Fstar"),
                        "o-", color=color, label=f"top radiation: {label}")
        axes[1, 0].plot(t, values(rows, "diffusion_factor"),
                        "--", color=color, alpha=0.7,
                        label=f"bottom heating: {label}")
        axes[1, 1].plot(t, values(rows, "Mach_830_850_max"),
                        "o-", color=color, label=label)

    axes[0, 0].set_ylabel("813–817 R☉ temperature / release value")
    axes[0, 0].axhline(1, color="0.5", lw=0.8)
    axes[0, 1].set_ylabel("Corrected convective flux / F★")
    axes[1, 0].set_ylabel("Top radiation or bottom input / F★")
    axes[1, 1].set_ylabel("Maximum Mach, 830–850 R☉")
    axes[1, 1].axhline(1, color="0.4", lw=0.8, ls=":")
    for ax in axes.flat:
        ax.set_xlabel("Time since FLD restoration began [code units]")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("RSG local patch: gradual FLD and bottom-heating restoration")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    print(OUT)


if __name__ == "__main__":
    main()
