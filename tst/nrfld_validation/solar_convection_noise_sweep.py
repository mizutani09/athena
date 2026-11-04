#!/usr/bin/env python3
"""Compare how initial temperature perturbations seed solar convection."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from solar_convection_report import parse_log, read_history


def mean_interval(h: dict[str, np.ndarray], name: str, start: float,
                  end: float) -> float:
    selected = (h["time"] >= start) & (h["time"] <= end)
    if not np.any(selected):
        raise ValueError(f"Missing history samples for t={start}–{end}")
    return float(np.mean(h[name][selected]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs=2, action="append", required=True,
                        metavar=("LABEL", "RUN_DIR"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-flux", type=float, default=6.3e10)
    args = parser.parse_args()
    if len(args.case) < 2:
        parser.error("Need at least two --case entries")
    items = []
    for label, raw_dir in args.case:
        run_dir = Path(raw_dir)
        history = read_history(next(run_dir.glob("*.hst")))
        log = parse_log(run_dir / "run.log")
        full = json.loads((run_dir / "surface3d.json").read_text())["records"]
        if not full:
            raise ValueError(f"{run_dir}: no full-3D surface records")
        text = (run_dir / "run.log").read_text(errors="replace")
        match = re.search(r"\btemp_noise_amp=([0-9.eE+-]+)", text)
        if match is None:
            raise ValueError(f"{run_dir}: missing temperature-noise amplitude")
        row = {
            "label": label,
            "run_dir": str(run_dir),
            "temp_noise_amp": float(match.group(1)),
            "top_flux_t5_10_over_solar": mean_interval(
                history, "FtotTop", 5, 10)/args.reference_flux,
            "top_flux_t30_40_over_solar": mean_interval(
                history, "FtotTop", 30, 40)/args.reference_flux,
            "last_surface_time_code": float(full[-1]["time_code"]),
            "last_surface_tau23_temperature_K": float(full[-1]["T_tau23_mean_K"]),
            "last_surface_tau23_vertical_rms_km_s": float(
                full[-1]["vz_tau23_rms_km_s"]),
            "completed": log["completed"],
            "converged_nr": log["nr_status_counts"].get("converged", 0),
            "slow_mg_warning_count": log["slow_mg_warning_count"],
            "fatal_count": log["fatal_count"],
            "solver_pass": bool(
                log["completed"] and log["fatal_count"] == 0
                and log["slow_mg_warning_count"] == 0
                and log["nr_status_counts"].get("converged", 0) > 0
                and set(log["nr_status_counts"]).issubset(
                    {"converged", "dt_zero_initialization"})),
        }
        items.append((label, run_dir, history, full, row))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "description": "Short startup perturbation sweep; neither steady nor solar-calibrated",
        "reference_flux_cgs": args.reference_flux,
        "cases": [item[-1] for item in items],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    plt.rcParams.update({"font.size": 13})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for label, _, h, full, _ in items:
        t = np.array([entry["time_code"] for entry in full])
        axes[0, 0].plot(h["time"], h["FtotTop"]/args.reference_flux, label=label)
        axes[0, 1].plot(t, [entry["T_tau23_mean_K"] for entry in full],
                        marker="o", label=label)
        axes[1, 0].plot(t, [entry["vz_tau23_rms_km_s"] for entry in full],
                        marker="o", label=label)
        axes[1, 1].plot(t, [entry["T_tau23_std_K"] for entry in full],
                        marker="o", label=label)
    axes[0, 0].axhline(1, color="black", ls="--", lw=1)
    axes[0, 0].set(ylabel=r"$F_{top}/F_\odot$", title="Upper total flux")
    axes[0, 1].set(ylabel="mean T [K]", title="τ=2/3 surface temperature")
    axes[1, 0].set(ylabel="vertical RMS [km/s]", title="τ=2/3 vertical velocity")
    axes[1, 1].set(ylabel="horizontal T std [K]", title="τ=2/3 temperature contrast")
    for axis in axes.flat:
        axis.set_xlabel("time [code]")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.savefig(args.output_dir / "noise_sweep.png", dpi=160)
    plt.close(fig)
    headers = ("Case", "Initial T perturbation", "Ftop t5–10 / solar",
               "Ftop t30–40 / solar", "Tτ=2/3 at t40 [K]",
               "Vz RMS at t40 [km/s]", "Solver")
    rows = []
    for label, run_dir, _, _, row in items:
        link = os.path.relpath(run_dir / "index.html", args.output_dir)
        solver = "PASS" if row["solver_pass"] else "CHECK"
        rows.append("<tr>" + "".join(f"<td>{value}</td>" for value in (
            f"<a href='{html.escape(link)}'>{html.escape(label)}</a>",
            f"{100*row['temp_noise_amp']:.3g}%",
            f"{row['top_flux_t5_10_over_solar']:.3f}",
            f"{row['top_flux_t30_40_over_solar']:.3f}",
            f"{row['last_surface_tau23_temperature_K']:.0f}",
            f"{row['last_surface_tau23_vertical_rms_km_s']:.3f}",
            solver,
        )) + "</tr>")
    (args.output_dir / "index.html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<title>Solar startup perturbation sweep</title><style>"
        "body{font:19px/1.5 system-ui;max-width:1300px;margin:2rem auto;"
        "padding:0 1rem;color:#172b3a}table{border-collapse:collapse}"
        "th,td{padding:.5rem .75rem;border-bottom:1px solid #ddd;text-align:left}"
        "img{max-width:100%}</style><h1>Solar startup perturbation sweep</h1>"
        "<p><strong>All cases are short transients.</strong> Zero initial "
        "perturbation does not imply zero vertical motion because the box "
        "can support coherent acoustic oscillations. Compare radiative "
        "flux and horizontal temperature contrast as well as velocity.</p>"
        "<table><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in headers)
        + "</tr>" + "".join(rows) + "</table>"
        + "<img src='noise_sweep.png' alt='noise sweep'>"
        + "<p><a href='summary.json'>Numerical results</a></p></html>"
    )
    print(args.output_dir / "index.html")


if __name__ == "__main__":
    main()
