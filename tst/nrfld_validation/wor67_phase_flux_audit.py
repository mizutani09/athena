#!/usr/bin/env python3
"""Diagnose Mach-5 peak phase and physical-flux spread from saved runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nrfld_validation.io import read_history
from plots.plot_radiative_shock import (jump_center, load_analytic_profile,
                                        load_numerical_profile)

HERE = Path(__file__).resolve().parent
REFERENCE = HERE/"reference/radiative_shock/mach5_semianalytic.csv"


def midpoint_crossing(x: np.ndarray, rho: np.ndarray, target: float) -> tuple[float, float]:
    jump = int(np.argmax(np.abs(np.diff(rho))))
    candidates = np.flatnonzero((rho[:-1]-target)*(rho[1:]-target) <= 0)
    if not len(candidates):
        raise ValueError("density profile does not cross reference midpoint")
    index = int(candidates[np.argmin(np.abs(candidates-jump))])
    crossing = x[index] + (target-rho[index])/(rho[index+1]-rho[index])*(x[index+1]-x[index])
    dx = x[jump+1]-x[jump]
    return float(crossing), float((crossing-0.5*(x[jump]+x[jump+1]))/dx)


def case_data(label: str, directory: Path, target: float) -> tuple[dict, dict]:
    history = read_history(directory/"output")
    late = (history["time"] >= .03) & (history["time"] <= .04)
    if not np.any(late):
        raise ValueError(f"{label}: no late history")
    snapshots = []
    profiles = sorted((directory/"output").glob("*uov_x*.athdf"))
    if len(profiles) < 2:
        raise ValueError(f"{label}: no saved profile series")
    for path in profiles:
        profile = load_numerical_profile(path)
        x = np.asarray(profile["x"])
        rho = np.asarray(profile["rho"])
        tgas = np.asarray(profile["tgas"])
        jump = jump_center(x, rho)
        crossing, phase_cells = midpoint_crossing(x, rho, target)
        near = np.abs(x-jump) <= 40
        snapshots.append({"time_s": float(profile["time"]),
                          "peak_K": float(np.max(tgas[near])),
                          "jump_center_cm": jump,
                          "density_midpoint_cm": crossing,
                          "midpoint_offset_cells": phase_cells})
    late_snapshots = [row for row in snapshots if .03 <= row["time_s"] <= .04]
    phase = np.array([row["midpoint_offset_cells"] for row in late_snapshots])
    peaks = np.array([row["peak_K"] for row in late_snapshots])
    correlation = (float(np.corrcoef(phase, peaks)[0, 1])
                   if len(phase) >= 3 and np.std(phase) > 0 and np.std(peaks) > 0
                   else None)
    row = {"label": label, "directory": str(directory.resolve()),
           "late_peak_mean_K": float(np.mean(history["Tgas_max"][late])),
           "late_peak_std_K": float(np.std(history["Tgas_max"][late])),
           "late_peak_range_K": [float(np.min(history["Tgas_max"][late])),
                                 float(np.max(history["Tgas_max"][late]))],
           "late_res_etot_median": float(np.median(history["res_etot"][late])),
           "late_res_etot_p95": float(np.percentile(history["res_etot"][late], 95)),
           "late_res_etot_max": float(np.max(history["res_etot"][late])),
           "late_res_etot_last": float(history["res_etot"][-1]),
           "late_peak_res_etot_correlation": float(np.corrcoef(
               history["Tgas_max"][late], history["res_etot"][late])[0, 1]),
           "late_history_shock_positions_cm": sorted(set(float(v) for v in history["x_shock"][late])),
           "saved_late_snapshot_count": len(late_snapshots),
           "saved_snapshot_peak_phase_correlation": correlation,
           "snapshots": snapshots}
    return row, {"history": history, "final_profile": load_numerical_profile(profiles[-1])}


def final_difference(reference: dict, other: dict) -> dict:
    xr = np.asarray(reference["x"])
    tr = np.asarray(reference["tgas"])
    xr = xr-jump_center(xr, np.asarray(reference["rho"]))
    xo = np.asarray(other["x"])
    to = np.asarray(other["tgas"])
    xo = xo-jump_center(xo, np.asarray(other["rho"]))
    diff = to-np.interp(xo, xr, tr)
    windows = {}
    for lo, hi in ((0, 8), (8, 25), (25, 60)):
        mask = (np.abs(xo) >= lo) & (np.abs(xo) < hi)
        windows[f"abs_x_{lo}_to_{hi}_cm"] = {
            "cells": int(np.count_nonzero(mask)),
            "mean_abs_Tgas_diff_K": float(np.mean(np.abs(diff[mask]))),
            "max_abs_Tgas_diff_K": float(np.max(np.abs(diff[mask])))}
    return {"x": xo, "difference": diff, "windows": windows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", required=True, help="LABEL=CASE_DIR")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analytic = load_analytic_profile(REFERENCE)
    target = float((analytic["rho"][0]+analytic["rho"][-1])/2)
    rows, curves = [], {}
    for spec in args.case:
        label, sep, path = spec.partition("=")
        if not sep or not re.fullmatch(r"[A-Za-z0-9_-]+", label) or label in curves:
            parser.error(f"invalid case: {spec}")
        row, data = case_data(label, Path(path), target)
        rows.append(row)
        curves[label] = data
    if "N2048" not in curves:
        parser.error("N2048 uniform baseline is required")
    reference = curves["N2048"]["final_profile"]
    differences = {}
    for row in rows:
        label = row["label"]
        if label == "N2048":
            continue
        difference = final_difference(reference, curves[label]["final_profile"])
        row["final_Tgas_difference_from_uniform"] = difference["windows"]
        differences[label] = difference
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/"snapshot_phase.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("layout", "time_s", "peak_K", "jump_center_cm",
                         "density_midpoint_cm", "midpoint_offset_cells"))
        for row in rows:
            for snapshot in row["snapshots"]:
                writer.writerow((row["label"], *(snapshot[key] for key in (
                    "time_s", "peak_K", "jump_center_cm", "density_midpoint_cm",
                    "midpoint_offset_cells"))))
    summary = {"reference": str(REFERENCE.resolve()), "density_midpoint_target": target,
               "cases": rows,
               "interpretation": "Density midpoint is linearly interpolated from saved profiles. Correlation uses only late saved snapshots and does not prove causation. res_etot is within-block physical-flux spread, not a discrete conservation residual."}
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    colors = {"N2048": "tab:blue", "AMR1_N2048": "tab:orange",
              "SMR3_N2048": "tab:green"}
    for row in rows:
        label = row["label"]
        h = curves[label]["history"]
        mask = (h["time"] >= .03) & (h["time"] <= .04)
        take = np.flatnonzero(mask)[::max(1, int(np.count_nonzero(mask)/1200))]
        axes[0, 0].plot(h["time"][take], h["Tgas_max"][take],
                        color=colors.get(label), label=label)
        axes[1, 0].plot(h["time"][take], h["res_etot"][take],
                        color=colors.get(label), label=label)
        late_snaps = [s for s in row["snapshots"] if .03 <= s["time_s"] <= .04]
        axes[0, 1].plot([s["midpoint_offset_cells"] for s in late_snaps],
                        [s["peak_K"] for s in late_snaps], "o-",
                        color=colors.get(label), label=label)
    for label, data in differences.items():
        mask = np.abs(data["x"]) <= 60
        axes[1, 1].plot(data["x"][mask], data["difference"][mask], "o-",
                        ms=3, color=colors.get(label), label=f"{label} − uniform")
    axes[0, 0].set(title="Late Tgas peak history", xlabel="t [s]", ylabel="Tgas max [K]")
    axes[0, 1].set(title="Saved snapshots: peak versus density-midpoint offset",
                   xlabel="midpoint offset from jump center [shock cells]", ylabel="Tgas peak [K]")
    axes[1, 0].set(title="Within-block physical energy-flux spread",
                   xlabel="t [s]", ylabel="res_etot (diagnostic)")
    axes[1, 1].set(title="Final Tgas difference from uniform",
                   xlabel="x from density jump [cm]", ylabel="ΔTgas [K]")
    for ax in axes.flat:
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    fig.savefig(args.output/"phase_flux_diagnostics.png", dpi=160)
    plt.close(fig)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
