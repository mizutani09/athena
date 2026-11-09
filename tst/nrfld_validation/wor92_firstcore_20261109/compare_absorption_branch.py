#!/usr/bin/env python3
"""Compare the completed c590→1010 absorption sensitivity branch."""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
EVIDENCE = Path("/home/kosuke/simulation/test/rad_nr/wor92_xd_20261107/firstcore_integration_evidence")


def load(name):
    return json.loads((ROOT / name).read_text())


def digest(path):
    h = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hdf_delta(cycle, evidence):
    reference = evidence / "profiles" / f"c{cycle}.athdf"
    trial = evidence / "profiles" / f"c{cycle}_absorption_trial.athdf"
    with h5py.File(reference) as left, h5py.File(trial) as right:
        if int(left.attrs["NumCycles"]) != cycle or int(right.attrs["NumCycles"]) != cycle:
            raise ValueError("wrong cycle in HDF")
        if not np.array_equal(left["Levels"], right["Levels"]) or not np.array_equal(
                left["LogicalLocations"], right["LogicalLocations"]):
            raise ValueError("AMR topology differs")
        mask = np.asarray(left["prim"][0], dtype=np.float64) > 1e4  # rho > 1e-14 g/cm^3
        data = {}
        for index, name in ((0, "rho"), (1, "pressure"), (8, "radiation_energy")):
            x = np.asarray(left["prim"][index], dtype=np.float64)[mask]
            y = np.asarray(right["prim"][index], dtype=np.float64)[mask]
            relative = np.abs(y - x) / np.maximum(np.abs(x), 1e-300)
            data[name] = {"max_absolute_relative": float(np.max(relative)),
                          "p99_absolute_relative": float(np.percentile(relative, 99))}
        return {"cycle": cycle, "meshblocks": int(left.attrs["NumMeshBlocks"]),
                "dense_cells": int(np.count_nonzero(mask)),
                "reference_hdf_sha256": digest(reference),
                "trial_hdf_sha256": digest(trial), "fields": data}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=ROOT / "absorption_branch_comparison.json")
    parser.add_argument("--output-png", type=Path, default=ROOT / "absorption_branch_comparison.png")
    parser.add_argument("--evidence-dir", type=Path, default=EVIDENCE)
    args = parser.parse_args()
    original = load("c1010_profile.json")
    trial = load("c1010_absorption_profile.json")
    original_axis = load("c1010_axis.json")
    trial_axis = load("c1010_absorption_axis.json")
    jobs = {str(job): load(f"job_{job}_summary.json") for job in
            (338156, 338171, 338230, 338233)}
    if not all(jobs[str(job)]["newton_status_counts"] ==
               {"converged": jobs[str(job)]["newton_status_counts"]["converged"]}
               and not jobs[str(job)]["opacity_nonzero_diagnostic_segments"]
               for job in (338230, 338233)):
        raise ValueError("trial solver or opacity diagnostics failed")
    fields = ("rho_max_g_cm3", "temperature_max_K", "gas_mass_g")
    final = {name: {"reference": original[name], "trial": trial[name],
                    "trial_over_reference_minus_one": trial[name] / original[name] - 1}
             for name in fields}
    if original_axis["steepest_positive_x_density_drop_3_to_15_au"]["inner_au"] != \
            trial_axis["steepest_positive_x_density_drop_3_to_15_au"]["inner_au"]:
        raise ValueError("shock location differs")
    final["positive_x_shock"] = {
        "reference": original_axis["steepest_positive_x_density_drop_3_to_15_au"],
        "trial": trial_axis["steepest_positive_x_density_drop_3_to_15_au"]}
    data = {"start_cycle": 590, "end_cycle": 1010,
            "table_trial_sha256": "347baa6e7b0d4303eca1abf762b2fde5042137e42739ef65961eeb68b88dad05",
            "final": final,
            "hdf_comparison": {str(cycle): hdf_delta(cycle, args.evidence_dir)
                               for cycle in (930, 1010)},
            "jobs": jobs,
            "trial_logs_sha256": {str(job): digest(args.evidence_dir / f"job_{job}_absorption_logs.tar.gz")
                                   for job in (338230, 338233)},
            "note": "Same c590 restart and binary. Only Planck table changed; Rosseland table identical. The trial table is validated here only through first-core temperatures."}
    args.output_json.write_text(json.dumps(data, indent=2) + "\n")

    def axis(name):
        with (ROOT / name).open() as stream:
            rows = list(csv.DictReader(stream))
        return {key: np.array([float(row[key]) for row in rows])
                for key in ("x_au", "rho_g_cm3", "temperature_K")}
    a = axis("c1010_axis.csv")
    b = axis("c1010_absorption_axis.csv")
    if not np.array_equal(a["x_au"], b["x_au"]):
        raise ValueError("axis grid differs")
    mask = (a["x_au"] >= 0) & (a["x_au"] <= 15)
    x = a["x_au"][mask]
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True, layout="constrained")
    axs[0].semilogy(x, a["rho_g_cm3"][mask], label="published Semenov mean")
    axs[0].semilogy(x, b["rho_g_cm3"][mask], "--", label="absorption trial")
    axs[0].set_ylabel(r"Density [g cm$^{-3}$]")
    axs[0].legend()
    axs[1].plot(x, a["temperature_K"][mask])
    axs[1].plot(x, b["temperature_K"][mask], "--")
    axs[1].set_ylabel("Gas temperature [K]")
    axs[2].plot(x, 1e6*(b["rho_g_cm3"][mask]/a["rho_g_cm3"][mask]-1), label="density")
    axs[2].plot(x, 1e6*(b["temperature_K"][mask]/a["temperature_K"][mask]-1),
                label="temperature")
    axs[2].axhline(0, color="0.5", lw=0.8)
    axs[2].set_ylabel("Trial / original − 1 [ppm]")
    axs[2].set_xlabel("Positive x [au]")
    axs[2].legend()
    fig.suptitle("WOR-92 first-core absorption sensitivity, cycle 1010")
    fig.savefig(args.output_png, dpi=155)
    print(args.output_json)
    print(args.output_png)


if __name__ == "__main__":
    main()
