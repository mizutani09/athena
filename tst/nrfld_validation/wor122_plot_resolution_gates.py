#!/usr/bin/env python3
"""Plot matched short RSG histories from WOR-122."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SECONDS_PER_CODE_TIME = 21999.96568179141


def read(*paths):
    pieces = [np.loadtxt(path, comments="#", ndmin=2) for path in paths]
    result = pieces[0]
    for piece in pieces[1:]:
        result = np.vstack((result, piece[piece[:, 0] > result[-1, 0] + 1.0e-12]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    d = args.data_dir
    cases = [
        (("uniform64.hst",), "64³ uniform", "#6b7280", "-"),
        (("uniform128.hst", "uniform128_day2.hst"), "128³ uniform", "#2563eb", "-"),
        (("onepatch128_t1.hst", "onepatch128_day1.hst"),
         "128³ one-patch", "#f97316", "--"),
        (("uniform256.hst", "uniform256_day2.hst", "uniform256_day2_tail.hst"),
         "256³ uniform", "#059669", "-"),
        (("uniform128_strongdrag.hst",), "128³ strong core drag", "#a855f7", ":"),
    ]
    fig, (ax_l, ax_m) = plt.subplots(2, 1, figsize=(8.2, 6.4), sharex=True,
                                     gridspec_kw={"height_ratios": [1.15, 1]})
    for filenames, label, color, style in cases:
        h = read(*(d / filename for filename in filenames))
        day = h[:, 0] * SECONDS_PER_CODE_TIME / 86400.0
        valid = (day > 0.0) & (h[:, 15] > 0.0)
        luminosity = np.sum(h[:, 22:28], axis=1) / np.maximum(h[:, 15], 1.0e-300)
        ax_l.plot(day[valid], luminosity[valid], style, color=color, lw=1.9,
                  label=label)
        ax_m.plot(day, h[:, 16], style, color=color, lw=1.9)
    ax_l.axhline(1.0, color="#111827", ls=":", lw=1.2, label="input luminosity")
    ax_l.set_ylabel(r"boundary $L_{\rm rad}/L_{\rm heat}$")
    ax_m.set_ylabel("maximum Mach")
    ax_m.set_xlabel("time [day]")
    ax_l.legend(ncol=2, fontsize=8.5, frameon=False)
    ax_l.set_ylim(bottom=0.0)
    ax_m.set_ylim(bottom=0.0)
    ax_m.set_xlim(0.0, 2.0)
    for ax in (ax_l, ax_m):
        ax.grid(alpha=0.22)
    fig.suptitle("RSG two-day gates: second-order LTE/FLD", fontsize=12)
    fig.text(0.5, 0.012,
             "The global Mach maximum can lie in dilute exterior gas; inspect radial location separately.",
             ha="center", fontsize=8, color="#4b5563")
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
