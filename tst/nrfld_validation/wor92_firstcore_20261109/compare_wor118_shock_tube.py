#!/usr/bin/env python3
"""Compare the WOR-118 face fix with the same Rad-EOS shock-tube reference."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import shock_tube_validation as shock  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def variation(values):
    scale = np.ptp(values)
    return float(np.sum(np.abs(np.diff(values))) / scale) if scale else 0.0


def extrema(values):
    slope = np.diff(values)
    threshold = max(np.ptp(values) * 1e-3, np.finfo(float).eps)
    signs = np.sign(slope[np.abs(slope) > threshold])
    return int(np.sum(signs[1:] != signs[:-1]))


def compare(baseline, repaired, output_json, output_png):
    source = HERE.parents[2] / "inputs/radiation/athinput.nrfld_shock_tube"
    physical = shock.shock_settings(source)
    reference = shock.profile(baseline / "reference/cases/r4096", -1,
                              "reference", physical)
    fields = ("rho", "velocity", "temperature", "radiation_energy",
              "total_pressure")
    provenance = {}
    for label, root in (("baseline", baseline), ("repaired", repaired)):
        run = json.loads((root / "summary.json").read_text())
        if run["status"] != "passed":
            raise ValueError(f"{label} validation status is {run['status']}")
        build = root / "fld/build/build_source"
        cases = {}
        for n in (64, 128):
            case = root / f"fld/cases/n{n}"
            metadata = json.loads((case / "manifest.json").read_text())
            log_path = case / "run.log"
            log = log_path.read_text(errors="replace")
            linear = [(int(it), float(res)) for it, res in re.findall(
                r"^\[LINEAR_FGMRES\] iterations=(\d+) relative_residual=([\deE+.-]+)",
                log, re.M)]
            if len(linear) != metadata["solver"]["fgmres_calls"]:
                raise ValueError(f"{label} N={n} FGMRES log count mismatch")
            elapsed = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([^\n]+)",
                                (case / "resources.txt").read_text())
            metadata.update({
                "iteration_history_log": str(log_path),
                "iteration_history_log_sha256": sha256(log_path),
                "fgmres_iterations_total": sum(it for it, _ in linear),
                "fgmres_max_true_relative_residual": max(res for _, res in linear),
                "wall_clock_time": elapsed.group(1) if elapsed else None,
            })
            cases[f"n{n}"] = metadata
        provenance[label] = {
            "run_summary": str(root / "summary.json"),
            "binary_sha256": sha256(build / "bin/athena"),
            "face_states_source_sha256": sha256(
                build / "src/fld/calculate_radiation_face_states.cpp"),
            "time_integrator_source_sha256": sha256(
                build / "src/task_list/time_integrator.cpp"),
            "cases": cases,
        }
    results = {}
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.2), sharex="col")
    for row, n in enumerate((64, 128)):
        base = shock.profile(baseline / f"fld/cases/n{n}", -1, "fld", physical)
        trial = shock.profile(repaired / f"fld/cases/n{n}", -1, "fld", physical)
        if not np.array_equal(base["x"], trial["x"]):
            raise ValueError(f"N={n} grids differ")
        if abs(base["time"] - trial["time"]) > 3e-7:
            raise ValueError(f"N={n} final times differ")
        ref = shock.coarsen(reference, n)
        mask = (base["x"] >= 0.60) & (base["x"] <= 0.88)
        field_results = {}
        for field in fields:
            b, t, r = (np.asarray(a[field]) for a in (base, trial, ref))
            base_metric = shock.metrics(b, r)
            trial_metric = shock.metrics(t, r)
            scale = np.ptp(r)
            field_results[field] = {
                "baseline_reference_relative_l1": base_metric["relative_l1"],
                "repaired_reference_relative_l1": trial_metric["relative_l1"],
                "reference_l1_change_fraction": (trial_metric["relative_l1"] /
                                                  base_metric["relative_l1"] - 1)
                if base_metric["relative_l1"] else None,
                "paired_max_difference_over_reference_range":
                    float(np.max(np.abs(t-b))/scale) if scale else None,
                "baseline_total_variation_over_range": variation(b),
                "repaired_total_variation_over_range": variation(t),
                "baseline_significant_extrema": extrema(b),
                "repaired_significant_extrema": extrema(t),
                "front_baseline_variation_over_range": variation(b[mask]),
                "front_repaired_variation_over_range": variation(t[mask]),
                "front_baseline_significant_extrema": extrema(b[mask]),
                "front_repaired_significant_extrema": extrema(t[mask]),
            }
        results[f"n{n}"] = {
            "baseline_cycles": base["cycles"], "repaired_cycles": trial["cycles"],
            "final_time": trial["time"], "fields": field_results,
        }
        for col, field in enumerate(("rho", "temperature", "radiation_energy")):
            ax = axes[row, col]
            ax.plot(reference["x"], reference[field], color="#8a9ba5", lw=1.0,
                    label="Rad-EOS N4096")
            ax.plot(base["x"], base[field], "o-", color="#336b90", ms=2.5,
                    label="before")
            ax.plot(trial["x"], trial[field], "x--", color="#c55929", ms=3,
                    label="WOR-118")
            ax.set_xlim(float(base["x"][mask][0]), float(base["x"][mask][-1]))
            if col == 0:
                ax.set_ylabel(f"N={n}\nDensity [g cm$^{{-3}}$]")
            if col == 1:
                ax.set_ylabel("Gas temperature [K]")
            if col == 2:
                ax.set_ylabel("Radiation energy [erg cm$^{-3}$]")
            if row == 1:
                ax.set_xlabel("x")
            ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Shock-tube regression near the compression front, t=0.1")
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)
    output_json.write_text(json.dumps({
        "baseline_run": str(baseline), "repaired_run": str(repaired),
        "reference": "independent 1D Rad-EOS HLLC N4096",
        "significant_extrema_threshold": "0.1% of each field range",
        "provenance": provenance, "resolutions": results,
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("repaired", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("output_png", type=Path)
    args = parser.parse_args()
    compare(args.baseline, args.repaired, args.output_json, args.output_png)


if __name__ == "__main__":
    main()
