#!/usr/bin/env python3
"""Summarize paired c930→940 opacity replays from archived XD evidence."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIELDS = {"rho_max_g_cm3": 15, "Tgas_max_K": 19, "Trad_max_K": 20,
          "gas_mass_g": 22, "Erad_total_erg": 33, "Jeans_min_cells": 34}


def read_case(path):
    history = np.loadtxt(path / "output/cfl030_flux_precision16_local.hst")
    # Athena writes the final row again at normal termination.
    if np.all(history[-1] == history[-2]):
        history = history[:-1]
    log = (path / "launcher.log").read_text()
    fgmres = []
    newton = []
    for line in log.splitlines():
        if line.startswith("[LINEAR_FGMRES] "):
            fgmres.append(dict(part.split("=", 1) for part in line.split()[1:]
                               if "=" in part))
        if line.startswith("[NR] status="):
            newton.append(dict(part.split("=", 1) for part in line.split()[1:]
                               if "=" in part))
    active = [row for row in newton if float(row["dt"]) > 0]
    if len(history) != 10 or len(active) != 10 or len(fgmres) != 30:
        raise ValueError("incomplete 10-cycle replay: {} {} {}".format(
            len(history), len(active), len(fgmres)))
    if not all(row["status"] == "converged" for row in active):
        raise ValueError("Newton did not converge")
    if "out_of_domain=0 clamped=0 negative_results=0 zero_results=0 nonfinite_results=0" not in log:
        raise ValueError("opacity diagnostic failure")
    return history, {
        "job": path.name, "cycles": 10, "newton_converged": len(active),
        "newton_max_final_max": max(float(row["final_max"]) for row in active),
        "fgmres_solves": len(fgmres),
        "fgmres_iterations_total": sum(int(row["iterations"]) for row in fgmres),
        "fgmres_max_true_relative_residual": max(float(row["relative_residual"])
                                                  for row in fgmres),
        "final": {name: float(history[-1, col]) for name, col in FIELDS.items()},
        "mass_change_c931_to_c940_relative": float(
            history[-1, FIELDS["gas_mass_g"]] / history[0, FIELDS["gas_mass_g"]] - 1),
        "status": (path / "status.txt").read_text().strip(),
    }


def analyze(reference, absorption, output):
    a, a_summary = read_case(reference)
    b, b_summary = read_case(absorption)
    if not np.allclose(a[:, 0], b[:, 0], atol=0, rtol=1e-10):
        raise ValueError("mismatched trajectory times")
    differences = {name: (b_summary["final"][name] / a_summary["final"][name] - 1)
                   for name in FIELDS}
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    cycles = np.arange(931, 941)
    col = FIELDS["rho_max_g_cm3"]
    axes[0].plot(cycles, a[:, col] * 1e11, "o-", ms=3,
                 label="Published Planck extinction table")
    axes[0].plot(cycles, b[:, col] * 1e11, "s--", ms=3,
                 label="Diagnostic Planck absorption estimate")
    axes[0].set_ylabel(r"$\rho_{\max}$ [$10^{-11}$ g cm$^{-3}$]")
    for name, label in (("rho_max_g_cm3", "Density"),
                        ("Tgas_max_K", "Gas temperature")):
        col = FIELDS[name]
        axes[1].plot(cycles, (b[:, col] / a[:, col] - 1) * 1e9,
                     "o-", ms=3, label=label)
    axes[1].set_ylabel("Trial − reference [parts per billion]")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    axes[1].set_xlabel("Cycle")
    fig.suptitle("Same c930 restart: Planck opacity definition sensitivity")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    summary = {"reference": a_summary, "absorption_trial": b_summary,
               "trial_over_reference_final_relative": differences,
               "limitation": "10 late cycles, same restart and binary; does not establish entire collapse-path sensitivity",
               "plot": str(output)}
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_job_dir", type=Path)
    parser.add_argument("absorption_job_dir", type=Path)
    parser.add_argument("output_png", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.reference_job_dir, args.absorption_job_dir,
                             args.output_png), indent=2))


if __name__ == "__main__":
    main()
