#!/usr/bin/env python3
"""Summarize the c930→1010 WOR-118 replay from its archived evidence."""

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(initial_hst, final_hst, launcher_log, profile_json, axis_json,
              baseline_profile_json):
    initial = np.loadtxt(initial_hst, ndmin=2)[-1]
    history = np.loadtxt(final_hst, ndmin=2)
    if np.array_equal(history[-1], history[-2]):
        history = history[:-1]
    log = launcher_log.read_text(errors="replace")
    newton = [dict(re.findall(r"(\w+)=([^ ]+)", line))
              for line in log.splitlines() if line.startswith("[NR] status=")]
    newton = [row for row in newton if float(row["dt"]) > 0]
    linear = [dict(re.findall(r"(\w+)=([^ ]+)", line))
              for line in log.splitlines() if line.startswith("[LINEAR_FGMRES] ")]
    profile = json.loads(profile_json.read_text())
    axis = json.loads(axis_json.read_text())
    baseline = json.loads(baseline_profile_json.read_text())
    if len(history) != 80 or len(newton) != 80 or profile["cycle"] != 1010:
        raise ValueError("c930→1010 replay is incomplete")
    if not all(row["status"] == "converged" for row in newton):
        raise ValueError("Newton failed")
    if "FLD_OPACITY_DIAGNOSTICS out_of_domain=0 clamped=0 negative_results=0 zero_results=0 nonfinite_results=0" not in log:
        raise ValueError("missing clean opacity diagnostic")
    return {
        "job": 338342, "start_cycle": 930, "end_cycle": 1010,
        "slurm_state": "COMPLETED 0:0", "elapsed_seconds": 294,
        "nodes": 5, "mpi_ranks": 320, "max_step_rss_kib": 20268088,
        "source_base_commit": "533ac4dca59b4901ba9a5c3d4e9435a25ca659fc",
        "remote_binary_sha256": "908d5def56faf5e0993526f1c709b2593eec814a2a26ffd83d4a0133b6ac0810",
        "history_sha256": sha256(final_hst), "launcher_log_sha256": sha256(launcher_log),
        "hdf5_sha256": "11452e7c798291604a9379113b155975bc72abbc20d0ee297d2e02a0383e0a01",
        "newton_converged": len(newton),
        "newton_max_final_max": max(float(row["final_max"]) for row in newton),
        "fgmres_solves": len(linear),
        "fgmres_iterations_total": sum(int(row["iterations"]) for row in linear),
        "fgmres_max_true_relative_residual": max(float(row["relative_residual"])
                                                  for row in linear),
        "history_initial_gas_mass_g": float(initial[22]),
        "history_final_gas_mass_g": float(history[-1, 22]),
        "history_relative_gas_mass_change": float(history[-1, 22]/initial[22]-1),
        "history_mass_range_over_initial": float(np.ptp(np.r_[initial[22], history[:, 22]])/initial[22]),
        "final_density_max_g_cm3": float(history[-1, 15]),
        "final_temperature_max_K": float(history[-1, 19]),
        "final_min_jeans_cells": float(history[-1, 34]),
        "eos_cells_outside_table": profile["eos_cells_outside_table"],
        "meshblocks": profile["meshblocks"],
        "shock_inner_au": axis["steepest_positive_x_density_drop_3_to_15_au"]["inner_au"],
        "shock_outer_au": axis["steepest_positive_x_density_drop_3_to_15_au"]["outer_au"],
        "hdf_gas_mass_relative_to_original": profile["gas_mass_g"] / baseline["gas_mass_g"] - 1,
        "hdf_density_max_relative_to_original": profile["rho_max_g_cm3"] / baseline["rho_max_g_cm3"] - 1,
        "hdf_temperature_max_relative_to_original": profile["temperature_max_K"] / baseline["temperature_max_K"] - 1,
        "notes": "HDF5 cell values are float32; history gas mass is the primary high-precision conservation measure.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("initial_hst", "final_hst", "launcher_log", "profile_json",
                 "axis_json", "baseline_profile_json", "output_json"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = summarize(args.initial_hst, args.final_hst, args.launcher_log,
                       args.profile_json, args.axis_json, args.baseline_profile_json)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
