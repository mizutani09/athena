#!/usr/bin/env python3
"""Diagnostic N64 shock-tube ablation of radiation pressure in Riemann fluxes."""

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
from compare_wor118_shock_tube import variation, extrema  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze(on_root, off_case, output_json, output_png):
    physical = shock.shock_settings(HERE.parents[2] / "inputs/radiation/athinput.nrfld_shock_tube")
    on_case = on_root / "fld/cases/n64"
    on = [shock.profile(on_case, index, "fld", physical) for index in range(21)]
    off = [shock.profile(off_case, index, "fld", physical) for index in range(21)]
    reference = [shock.coarsen(shock.profile(on_root / "reference/cases/r4096",
                                               index, "reference", physical), 64)
                 for index in range(21)]
    if not all(abs(a["time"] - b["time"]) <= 3e-4 for a, b in zip(on, off)):
        raise ValueError("ON/OFF output schedules differ")
    if abs(off[-1]["time"] - 0.1) > 2e-7:
        raise ValueError("OFF control did not reach t=0.1")
    if not np.array_equal(on[-1]["x"], off[-1]["x"]):
        raise ValueError("ON/OFF grids differ")
    log_path = off_case / "run.log"
    log = log_path.read_text(errors="replace")
    newton = re.findall(r"^\[NR\] status=(\w+)", log, re.M)
    linear = [(int(a), float(b)) for a, b in re.findall(
        r"^\[LINEAR_FGMRES\] iterations=(\d+) relative_residual=([\deE+.-]+)",
        log, re.M)]
    if newton.count("converged") != off[-1]["cycles"] or (
            set(newton) - {"converged", "dt_zero_initialization"}) or (
            "Terminating on time limit" not in log):
        raise ValueError("OFF control did not converge cleanly")
    if "radiation_pressure_in_riemann = false" not in (off_case / "input.used").read_text():
        raise ValueError("OFF control input lacks ablation setting")
    fields = ("rho", "velocity", "temperature", "radiation_energy", "total_pressure")
    result = {}
    for field in fields:
        final_on, final_off, final_ref = (series[-1][field]
                                        for series in (on, off, reference))
        mask = (on[-1]["x"] >= 0.60) & (on[-1]["x"] <= 0.88)
        result[field] = {
            "on_reference_relative_l1": shock.metrics(final_on, final_ref)["relative_l1"],
            "off_reference_relative_l1_diagnostic_only":
                shock.metrics(final_off, final_ref)["relative_l1"],
            "final_max_on_off_difference_over_reference_range":
                float(np.max(np.abs(final_on-final_off))/np.ptp(final_ref)),
            "final_front_on_variation": variation(final_on[mask]),
            "final_front_off_variation": variation(final_off[mask]),
            "final_front_on_extrema": extrema(final_on[mask]),
            "final_front_off_extrema": extrema(final_off[mask]),
            "all_snapshots_max_global_extra_extrema_off_minus_on": max(
                extrema(b[field]) - extrema(a[field]) for a, b in zip(on, off)),
            "all_snapshots_max_global_variation_difference_off_minus_on": max(
                variation(b[field]) - variation(a[field]) for a, b in zip(on, off)),
        }
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.3))
    for row, index in enumerate((6, 20)):
        for col, field in enumerate(("rho", "temperature", "radiation_energy")):
            axis = axes[row, col]
            axis.plot(on[index]["x"], on[index][field], "o-", ms=2.6,
                      color="#287090", label="Riemann radiation pressure ON")
            axis.plot(off[index]["x"], off[index][field], "x--", ms=3.0,
                      color="#bc5e2f", label="OFF diagnostic")
            axis.set_xlim(.3 if row == 0 else .6, .85 if row == 0 else .88)
            if col == 0:
                axis.set_ylabel(f"t≈{on[index]['time']:.3f}\nDensity [g cm$^{{-3}}$]")
            elif col == 1:
                axis.set_ylabel("Gas temperature [K]")
            else:
                axis.set_ylabel("Radiation energy [erg cm$^{-3}$]")
            axis.grid(alpha=.25)
            if row == 1:
                axis.set_xlabel("x")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("N64 shock-tube diagnostic: Riemann radiation pressure ON/OFF")
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)
    resources = (off_case / "resources.txt").read_text()
    elapsed = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([^\n]+)",
                        resources)
    rss = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", resources)
    output_json.write_text(json.dumps({
        "scope": "Diagnostic ablation of only Riemann radiation pressure; radiation force remains enabled. This changes the model and does not identify all historical causes of shock oscillations.",
        "on_case": str(on_case), "off_case": str(off_case),
        "source_base_commit": "533ac4dca59b4901ba9a5c3d4e9435a25ca659fc",
        "binary_sha256": sha256(on_root / "fld/build/build_source/bin/athena"),
        "off_input_sha256": sha256(off_case / "input.used"),
        "off_log_sha256": sha256(log_path),
        "off_cycles": off[-1]["cycles"], "off_newton_converged": newton.count("converged"),
        "off_fgmres_solves": len(linear),
        "off_fgmres_iterations_total": sum(item[0] for item in linear),
        "off_fgmres_max_true_relative_residual": max(item[1] for item in linear),
        "off_wall_clock_time": elapsed.group(1) if elapsed else None,
        "off_max_rss_kib": int(rss.group(1)) if rss else None,
        "fields": result,
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("on_root", "off_case", "output_json", "output_png"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    analyze(args.on_root, args.off_case, args.output_json, args.output_png)


if __name__ == "__main__":
    main()
