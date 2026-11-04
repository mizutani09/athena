#!/usr/bin/env python3
"""Compare every-step sampled solar energy ledgers across CFL numbers."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from solar_convection_report import (input_values, make_campaign_index, parse_log,
                                     physical_time_unit, read_history,
                                     total_bottom_flux, total_top_flux)


def evaluate(label: str, run: Path, header: Path, start: float, end: float,
             reference_flux: float) -> tuple[dict, np.ndarray, np.ndarray]:
    h = read_history(next(run.glob("*.hst")), header)
    if not (h["time"][0] < start < end <= h["time"][-1]):
        raise ValueError(f"{run}: common interval is outside recorded history")
    inside = (h["time"] > start) & (h["time"] < end)
    t = np.r_[start, h["time"][inside], end]
    sample = {key: np.interp(t, h["time"], values)
              for key, values in h.items() if key != "time"}
    units = input_values(run/"athinput.generated")
    velocity_unit = np.sqrt(units["egas_unit"]/units["rho_unit"])
    seconds_per_code = physical_time_unit(units, velocity_unit)
    area = ((units["x1max"]-units["x1min"])
            *(units["x2max"]-units["x2min"])*units["leng_unit"]**2)
    mass_flux_unit = units["rho_unit"]*velocity_unit
    supply = (total_bottom_flux(sample) - total_top_flux(sample)
              + mass_flux_unit*units["grav_acc"]*units["leng_unit"]
              *(sample["BottomMdotNet"]*units["x3min"]
                - sample["TopMdotOut"]*units["x3max"]))
    boundary_integral = np.r_[0.0, np.cumsum(0.5*(supply[1:]+supply[:-1])
                                              *np.diff(t)*seconds_per_code)]*area
    stored_change = sample["EtotGrav"]-sample["EtotGrav"][0]
    normalization = area*reference_flux*(end-start)*seconds_per_code
    mismatch = (stored_change-boundary_integral)/normalization
    log = parse_log(run/"run.log")
    if not log["completed"] or log["fatal_count"]:
        raise ValueError(f"{run}: run is incomplete or fatal")
    manifest = json.loads((run/"manifest.json").read_text())
    cfl = next(float(x.split("=", 1)[1]) for x in manifest["command"]
               if x.startswith("time/cfl_number="))
    row = {
        "label": label, "run_dir": str(run), "cfl_number": cfl,
        "common_start_code": start, "common_end_code": end,
        "history_sample_count": len(t),
        "median_gap_code": float(np.median(np.diff(t))),
        "residual_over_solar": float(mismatch[-1]),
        "final_top_flux_over_solar": float(sample["FtotTop"][-1]/reference_flux),
        "newton_converged_count": log["nr_status_counts"].get("converged", 0),
        "slow_mg_warning_count": log["slow_mg_warning_count"],
        "fatal_count": log["fatal_count"],
    }
    return row, t, mismatch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs=2, action="append", required=True,
                        metavar=("LABEL", "RUN_DIR"))
    parser.add_argument("--header-from", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=float, default=850.2)
    parser.add_argument("--end", type=float, default=855.0)
    parser.add_argument("--reference-flux", type=float, default=6.3e10)
    args = parser.parse_args()
    if len(args.case) < 2:
        parser.error("Need at least two CFL cases")
    runs = [(label, Path(path)) for label, path in args.case]
    manifests = [json.loads((run/"manifest.json").read_text()) for _, run in runs]
    for (label, _), manifest in zip(runs[1:], manifests[1:]):
        for key in ("binary_sha256", "base_input_sha256", "data_sha256",
                    "restart", "problem_overrides"):
            if manifest.get(key) != manifests[0].get(key):
                raise ValueError(f"{label}: unmatched {key}")
    results = [evaluate(label, run, args.header_from, args.start, args.end,
                        args.reference_flux) for label, run in runs]
    rows = [item[0] for item in results]
    scan_starts = np.linspace(args.start, min(args.start+1.8, args.end-1.0), 19)
    scan_results = []
    for row, (_, run) in zip(rows, runs):
        residuals = np.array([
            evaluate(row["label"], run, args.header_from, float(start),
                     args.end, args.reference_flux)[0]["residual_over_solar"]
            for start in scan_starts])
        row["start_scan_rms_residual_over_solar"] = float(np.sqrt(np.mean(
            residuals**2)))
        row["start_scan_max_abs_residual_over_solar"] = float(np.max(
            np.abs(residuals)))
        scan_results.append(residuals)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"summary.json").write_text(json.dumps(rows, indent=2)+"\n")
    plt.rcParams.update({"font.size": 13})
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for row, t, mismatch in results:
        ax.plot(t, mismatch, label=f"{row['label']} (CFL {row['cfl_number']:g})")
    ax.axhline(0, color="black", lw=1)
    ax.set(xlabel="time [code]", ylabel="cumulative residual / solar interval",
           title="Every-step sampled energy ledger versus CFL")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(args.output_dir/"ledger_timestep.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for row, residuals in zip(rows, scan_results):
        ax.plot(scan_starts, residuals,
                label=f"{row['label']} (CFL {row['cfl_number']:g})")
    ax.axhline(0, color="black", lw=1)
    ax.set(xlabel="common start time [code]", ylabel="residual / solar interval",
           title=f"Endpoint {args.end:g}: sampled-ledger phase sensitivity")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(args.output_dir/"ledger_start_sensitivity.png", dpi=160)
    plt.close(fig)
    cells = []
    for row in rows:
        link = os.path.relpath(Path(row["run_dir"])/"index.html", args.output_dir)
        cells.append("<tr>"+"".join(f"<td>{value}</td>" for value in (
            f"<a href='{html.escape(link)}'>{html.escape(row['label'])}</a>",
            f"{row['cfl_number']:g}", str(row["history_sample_count"]),
            f"{row['median_gap_code']:.5f}",
            f"{row['residual_over_solar']:+.6f}",
            f"{row['start_scan_rms_residual_over_solar']:.6f}",
            f"{row['start_scan_max_abs_residual_over_solar']:.6f}",
            f"{row['final_top_flux_over_solar']:.4f}",
            str(row["newton_converged_count"])))+"</tr>")
    (args.output_dir/"index.html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<title>Solar ledger time-step comparison</title><style>"
        "body{font:19px/1.5 system-ui;max-width:1250px;margin:2rem auto;"
        "padding:0 1rem;color:#172b3a}table{border-collapse:collapse}"
        "th,td{padding:.5rem .75rem;border-bottom:1px solid #ddd;text-align:left}"
        "img{max-width:100%}</style><h1>Solar energy ledger: CFL comparison</h1>"
        f"<p>Common interval {args.start:g}–{args.end:g} code time from the same "
        "checkpoint. Every accepted step is recorded. Endpoint values are "
        "linearly interpolated to common times. The start-scan uses 19 "
        f"overlapping intervals from {args.start:g} to "
        f"{min(args.start+1.8, args.end-1.0):g}, all ending at "
        f"{args.end:g}. This sampled integral is still not the exact "
        "RK-stage flux budget.</p>"
        "<table><tr><th>Case</th><th>CFL</th><th>Samples</th>"
        "<th>Median gap [code]</th><th>Residual / solar</th>"
        "<th>Start-scan RMS / solar</th><th>Start-scan max |residual| / solar</th>"
        "<th>Final top / solar</th><th>Newton converged</th></tr>"
        +"".join(cells)+"</table>"
        "<img src='ledger_timestep.png' alt='CFL comparison'>"
        "<img src='ledger_start_sensitivity.png' alt='start-time sensitivity'>"
        "<p><a href='summary.json'>Numerical results</a></p></html>"
    )
    make_campaign_index(args.output_dir.parent)
    print(args.output_dir/"index.html")


if __name__ == "__main__":
    main()
