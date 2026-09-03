#!/usr/bin/env python3
"""Plot the one-dimensional HSE diagnostic written by nrfld_conv_star."""

import argparse

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", help="stellar_hse_profile.dat written by the pgen")
    parser.add_argument("--output", default="stellar_hse_profile.png")
    parser.add_argument("--rmin", type=float, default=1.0e-3,
                        help="minimum r/Rstar shown (default: 1e-3)")
    parser.add_argument("--font-size", type=float, default=16.0,
                        help="base font size in points (default: 16)")
    args = parser.parse_args()

    plt.rcParams.update({
        "font.size": args.font_size,
        "axes.labelsize": args.font_size,
        "axes.titlesize": args.font_size,
        "xtick.labelsize": args.font_size - 2,
        "ytick.labelsize": args.font_size - 2,
        "legend.fontsize": args.font_size - 2,
    })

    q = np.loadtxt(args.profile)
    r, rho, tgas, trad = q[:, 1], q[:, 2], q[:, 3], q[:, 4]
    rho_g, minus_dpt, minus_dpfld = q[:, 9], q[:, 10], q[:, 11]
    rel_total, rel_fld = q[:, 12], q[:, 13]
    use = (r >= args.rmin) & (r <= 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    ax = axes[0, 0]
    ax.loglog(r[use], rho[use])
    ax.set_ylabel(r"$\rho$ [g cm$^{-3}$]")

    ax = axes[0, 1]
    ax.loglog(r[use], tgas[use], label=r"$T_{\rm gas}$")
    ax.loglog(r[use], trad[use], "--", label=r"$T_{\rm rad}$")
    ax.set_ylabel("temperature [K]")
    ax.legend()

    ax = axes[1, 0]
    ax.loglog(r[use], np.abs(rho_g[use]), label=r"$\rho g$")
    ax.loglog(r[use], np.abs(minus_dpt[use]), "--",
              label=r"$-d(P_g+E_r/3)/dr$")
    ax.loglog(r[use], np.abs(minus_dpfld[use]), ":",
              label=r"$-[dP_g/dr+\lambda dE_r/dr]$")
    ax.set_ylabel(r"force density [dyn cm$^{-3}$]")
    ax.legend()

    ax = axes[1, 1]
    # Use a logarithmic magnitude while retaining sign information through
    # line style: solid is positive and dotted is negative.
    rr = r[use]
    total = rel_total[use]
    fld = rel_fld[use]
    floor = 1.0e-16
    total_pos = np.where(total >= 0.0, np.maximum(total, floor), np.nan)
    total_neg = np.where(total < 0.0, np.maximum(-total, floor), np.nan)
    fld_pos = np.where(fld >= 0.0, np.maximum(fld, floor), np.nan)
    fld_neg = np.where(fld < 0.0, np.maximum(-fld, floor), np.nan)
    ax.plot(rr, fld_pos, "-", color="C1", linewidth=1.5,
            label="FLD positive", zorder=2)
    ax.plot(rr, fld_neg, ":", color="C1", linewidth=1.8,
            label="FLD negative", zorder=2)
    ax.plot(rr, total_pos, "-", color="C0", linewidth=1.0,
            marker="o", markersize=3, markevery=150,
            label="total positive", zorder=3)
    ax.plot(rr, total_neg, ":", color="C0", linewidth=1.2,
            marker="o", markersize=3, markevery=150,
            label="total negative", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("absolute symmetric relative residual")
    ax.legend()

    for ax in axes.flat:
        ax.set_xlabel(r"$r/R_\star$")
        ax.grid(alpha=0.3, which="both")
        ax.set_xlim(args.rmin, 1.0)
    fig.savefig(args.output, dpi=180)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
