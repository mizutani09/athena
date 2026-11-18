#!/usr/bin/env python3
"""Summarize the local RSG surface-patch pilot from Athena++ HDF/HST outputs."""

import argparse
import glob
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


SHELLS = ((800, 813), (813, 817), (817, 830), (830, 850), (850, 860))
FSTAR_CGS = 8.389e9
TIME_UNIT_S = 6.957e10 / 3.1622776601683795e6


def snapshot(path):
    with h5py.File(path) as h5:
        z = np.asarray(h5["x3v"][:]).reshape(-1) + 800.0
        u = np.asarray(h5["user_out_var"][:])[:, :, :, 0, :]
        # A single x2 slice was written.  Average the resolved x1 direction.
        u = u.mean(axis=-1).reshape(u.shape[0], -1)
    order = np.argsort(z)
    return z[order], u[:, order]


def analyze(directory):
    directory = Path(directory)
    files = sorted(directory.glob("*.uov_xz_slice.*.athdf"))
    if len(files) < 2:
        raise ValueError(f"need initial and final xz snapshots: {directory}")
    z0, initial = snapshot(files[0])
    z1, final = snapshot(files[-1])
    if not np.allclose(z0, z1):
        raise ValueError("initial/final z coordinates differ")
    hst = np.loadtxt(glob.glob(str(directory / "*.hst"))[0])
    shells = {}
    for low, high in SHELLS:
        mask = (z0 >= low) & (z0 < high)
        shells[f"{low}-{high}"] = {
            "cells_in_z": int(mask.sum()),
            "Tgas_ratio": float(np.exp(np.log(final[2, mask] / initial[2, mask]).mean())),
            "Erad_ratio": float(np.exp(np.log(final[1, mask] / initial[1, mask]).mean())),
            "initial_Frad_over_Fstar": float(initial[10, mask].mean() / FSTAR_CGS),
            "final_Frad_over_Fstar": float(final[10, mask].mean() / FSTAR_CGS),
            "initial_Fenth_over_Fstar": float(initial[11, mask].mean() / FSTAR_CGS),
            "final_Fenth_over_Fstar": float(final[11, mask].mean() / FSTAR_CGS),
        }
    summary = {
        "directory": str(directory),
        "time_code": float(hst[-1, 0]),
        "time_days": float(hst[-1, 0] * TIME_UNIT_S / 86400),
        "energy_initial": float(hst[0, 16]),
        "energy_final": float(hst[-1, 16]),
        "top_radiative_flux_over_Fstar": float(hst[-1, 17] / FSTAR_CGS),
        "bottom_heat_power_code": float(hst[-1, 24]) if hst.shape[1] > 24 else 0.0,
        "surface_heat_power_code": float(hst[-1, 25]) if hst.shape[1] > 25 else 0.0,
        "shells": shells,
    }
    return summary, z0, initial, final


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--noheat", type=Path, required=True)
    ap.add_argument("--heat", type=Path, required=True)
    ap.add_argument("--seeded", type=Path)
    ap.add_argument("--support", type=Path)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--figure", type=Path, required=True)
    args = ap.parse_args()

    cases = {"noheat": analyze(args.noheat), "heat": analyze(args.heat)}
    if args.seeded:
        cases["seeded_heat"] = analyze(args.seeded)
    if args.support:
        cases["surface_support"] = analyze(args.support)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({k: v[0] for k, v in cases.items()}, indent=2) + "\n")

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    z, u0 = cases["noheat"][1:3]
    axes[1].plot(z, u0[10] / FSTAR_CGS, color="0.5", label="initial radiation")
    for name, (_, z, initial, final) in cases.items():
        axes[0].plot(z, final[2] / initial[2], label=name)
        line, = axes[1].plot(z, final[10] / FSTAR_CGS,
                             label=f"{name} radiation")
        if name in ("seeded_heat", "surface_support"):
            axes[1].plot(z, final[11] / FSTAR_CGS, ":",
                         color=line.get_color(), label=f"{name} enthalpy")
    axes[0].set_ylabel("Gas temperature / initial")
    axes[1].set_ylabel("Vertical flux / stellar flux")
    axes[1].set_xlabel("Radius [solar radii]")
    for ax in axes:
        ax.axvspan(813, 817, color="tab:orange", alpha=0.12)
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=160)


if __name__ == "__main__":
    main()
