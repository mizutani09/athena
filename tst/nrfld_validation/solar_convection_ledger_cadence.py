#!/usr/bin/env python3
"""Compare sampled solar-box energy ledgers at different HST cadences."""

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

from solar_convection_report import (input_values, parse_log, physical_time_unit,
                                     read_history)


def analyze(label: str, run_dir: Path, header: Path, flux_ref: float,
            common_start: float, common_end: float):
    history = read_history(next(run_dir.glob("*.hst")), header)
    selected = ((history["time"] >= common_start-1e-8)
                & (history["time"] <= common_end+1e-8))
    history = {key: values[selected] for key, values in history.items()}
    units = input_values(run_dir / "athinput.generated")
    velocity_unit = np.sqrt(units["egas_unit"]/units["rho_unit"])
    seconds_per_code = physical_time_unit(units, velocity_unit)
    time_code = history["time"]
    time = (time_code-time_code[0])*seconds_per_code
    area = ((units["x1max"]-units["x1min"])
            *(units["x2max"]-units["x2min"])*units["leng_unit"]**2)
    mass_flux_unit = units["rho_unit"]*velocity_unit
    surface_flux = (history["FhydBottom"] + history["FradBottom"]
                    - history["FtotTop"]
                    + mass_flux_unit*units["grav_acc"]*units["leng_unit"]
                    *(history["BottomMdotNet"]*units["x3min"]
                      - history["TopMdotOut"]*units["x3max"]))
    observed = history["EtotGrav"]-history["EtotGrav"][0]
    step = np.diff(time)
    left = np.r_[0.0, np.cumsum(surface_flux[:-1]*step)]*area
    right = np.r_[0.0, np.cumsum(surface_flux[1:]*step)]*area
    trapezoid = 0.5*(left+right)
    normalization = area*flux_ref*time[-1]
    log = parse_log(run_dir/"run.log")
    row = {
        "label": label, "run_dir": str(run_dir),
        "start_time_code": float(time_code[0]),
        "end_time_code": float(time_code[-1]),
        "history_sample_count": len(time_code),
        "median_history_gap_code": float(np.median(np.diff(time_code))),
        "maximum_history_gap_code": float(np.max(np.diff(time_code))),
        "newton_converged_count": log["nr_status_counts"].get("converged", 0),
        "slow_mg_warning_count": log["slow_mg_warning_count"],
        "fatal_count": log["fatal_count"],
        "final_top_flux_over_solar": float(history["FtotTop"][-1]/flux_ref),
        "residual_left_over_solar": float((observed[-1]-left[-1])/normalization),
        "residual_trapezoid_over_solar": float(
            (observed[-1]-trapezoid[-1])/normalization),
        "residual_right_over_solar": float((observed[-1]-right[-1])/normalization),
    }
    return row, time_code, (observed-trapezoid)/normalization


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs=2, action="append", required=True,
                        metavar=("LABEL", "RUN_DIR"))
    parser.add_argument("--header-from", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-flux", type=float, default=6.3e10)
    args = parser.parse_args()
    if len(args.case) < 2:
        parser.error("Need at least two cadence cases")
    raw_times = [read_history(next(Path(path).glob("*.hst")),
                              args.header_from)["time"] for _, path in args.case]
    common_start = max(float(time[0]) for time in raw_times)
    common_end = min(float(time[-1]) for time in raw_times)
    if not common_start < common_end:
        parser.error("The history files have no common time interval")
    analyzed = [analyze(label, Path(path), args.header_from,
                        args.reference_flux, common_start, common_end)
                for label, path in args.case]
    starts = [row["start_time_code"] for row, _, _ in analyzed]
    ends = [row["end_time_code"] for row, _, _ in analyzed]
    if max(starts)-min(starts) > 1e-8 or max(ends)-min(ends) > 1e-8:
        parser.error("The cases lack shared sampled endpoints")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [item[0] for item in analyzed]
    (args.output_dir/"summary.json").write_text(json.dumps(rows, indent=2)+"\n")
    plt.rcParams.update({"font.size": 13})
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for row, time, mismatch in analyzed:
        ax.plot(time, mismatch, label=row["label"])
    ax.axhline(0, color="black", lw=1)
    ax.set(xlabel="time [code]", ylabel="cumulative ledger mismatch / solar interval",
           title="Stored energy minus sampled boundary flux integral")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(args.output_dir/"ledger_cadence.png", dpi=160)
    plt.close(fig)
    cells = []
    for row in rows:
        link = os.path.relpath(Path(row["run_dir"])/"index.html", args.output_dir)
        values = [
            f"<a href='{html.escape(link)}'>{html.escape(row['label'])}</a>",
            str(row["history_sample_count"]),
            f"{row['median_history_gap_code']:.4f}",
            f"{row['maximum_history_gap_code']:.4f}",
            f"{row['newton_converged_count']}",
            f"{row['residual_left_over_solar']:+.6f}",
            f"{row['residual_trapezoid_over_solar']:+.6f}",
            f"{row['residual_right_over_solar']:+.6f}",
        ]
        cells.append("<tr>"+"".join(f"<td>{v}</td>" for v in values)+"</tr>")
    (args.output_dir/"index.html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<title>Solar energy-ledger cadence</title><style>"
        "body{font:19px/1.5 system-ui;max-width:1250px;margin:2rem auto;"
        "padding:0 1rem;color:#172b3a}table{border-collapse:collapse}"
        "th,td{padding:.5rem .75rem;border-bottom:1px solid #ddd;text-align:left}"
        "img{max-width:100%}</style><h1>Solar sampled energy ledger</h1>"
        "<p>Both cases restart from the same checkpoint. Residuals are "
        "normalized by the solar-flux energy over the common interval. "
        "Left, trapezoid, and right refer to time integration of stored "
        "history samples. Even every-step history is not an exact RK-stage "
        "flux integral or a discrete conservation proof.</p>"
        "<table><tr><th>Case</th><th>Samples</th><th>Median gap [code]</th>"
        "<th>Max gap [code]</th>"
        "<th>Newton converged</th><th>Left / solar</th>"
        "<th>Trapezoid / solar</th><th>Right / solar</th></tr>"
        +"".join(cells)+"</table>"
        "<img src='ledger_cadence.png' alt='ledger cadence'>"
        "<p><a href='summary.json'>Numerical results</a></p></html>"
    )
    print(args.output_dir/"index.html")


if __name__ == "__main__":
    main()
