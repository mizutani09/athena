#!/usr/bin/env python3
"""Summarize a non-magnetic local solar-convection run from its history/log.

The stationarity flag is a screening rule, not evidence of agreement with the
Sun.  Two late windows and their duration in vertical crossing times are kept
in the JSON report so that a short run cannot accidentally pass as relaxed.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HEADER = re.compile(r"\[(\d+)\]=([^\s]+)")
KEY_VALUE = re.compile(r"^\s*([A-Za-z_][A-Za-z_0-9]*)\s*=\s*([^#\s]+)", re.M)


def read_history(path: Path, header_from: Path | None = None) -> dict[str, np.ndarray]:
    # Athena++ restart output in a new directory can omit the history header.
    source = header_from or path
    header = next(line for line in source.read_text().splitlines()
                  if line.startswith("# ["))
    names = [name for _, name in HEADER.findall(header)]
    values = np.loadtxt(path, comments="#", ndmin=2)
    if values.shape[1] != len(names):
        raise ValueError(f"{path}: {values.shape[1]} columns, {len(names)} names")
    # A continued run may repeat the initial history row. Keep the last value.
    order = np.argsort(values[:, 0], kind="stable")
    values = values[order]
    keep = np.r_[np.diff(values[:, 0]) > 0, True]
    values = values[keep]
    return dict(zip(names, values.T))


def input_values(path: Path) -> dict[str, float]:
    text = path.read_text()
    return {name: float(value) for name, value in KEY_VALUE.findall(text)
            if name in {"rho_unit", "egas_unit", "leng_unit", "time_unit",
                        "x3min", "x3max"}}


def physical_time_unit(units: dict[str, float], velocity_unit: float) -> float:
    # The pgen treats a negative time_unit as a request to derive it.
    specified = units.get("time_unit", -1.0)
    return specified if specified > 0 else units["leng_unit"] / velocity_unit


def parse_log(path: Path) -> dict[str, object]:
    text = path.read_text(errors="replace")
    statuses = re.findall(r"\[NR\] status=([^\s]+)", text)
    krylov = [int(n) for n in re.findall(r"\[LINEAR_FGMRES\] iterations=(\d+)", text)]
    final = re.findall(r"^time=([^\s]+) cycle=(\d+)$", text, re.M)
    wall = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([^\n]+)", text)
    return {
        "final_cycle": int(final[-1][1]) if final else None,
        "final_time": float(final[-1][0]) if final else None,
        "completed": bool(re.search(r"^Terminating on (cycle|time) limit", text, re.M)),
        "fatal_count": text.count("FATAL ERROR"),
        "slow_mg_warning_count": text.count("Slow multigrid convergence"),
        "divergent_vcycle_count": text.count("Rejecting divergent V-cycle"),
        "nr_status_counts": {name: statuses.count(name) for name in sorted(set(statuses))},
        "fgmres_calls": len(krylov),
        "fgmres_iterations_total": sum(krylov),
        "fgmres_iterations_max": max(krylov, default=0),
        "wall_time": wall.group(1).strip() if wall else None,
    }


def window_metrics(h: dict[str, np.ndarray], indices: np.ndarray,
                   flux_ref: float, velocity_unit: float, time_unit: float,
                   height_cm: float) -> dict[str, float]:
    t = h["time"][indices]
    vz = np.sqrt(np.maximum(h["VzRms2Mean"][indices], 0.0)) * velocity_unit
    speed = float(np.mean(vz))
    crossing = height_cm / speed if speed > 0 else float("inf")
    def mean(name: str) -> float:
        return float(np.mean(h[name][indices]))
    return {
        "start_time": float(t[0]), "end_time": float(t[-1]),
        "duration_seconds": float((t[-1] - t[0]) * time_unit),
        "vertical_crossing_times": float((t[-1] - t[0]) * time_unit / crossing),
        "vz_rms_km_s": speed / 1.0e5,
        "top_total_flux_over_solar": mean("FtotTop") / flux_ref,
        "top_radiation_flux_over_solar": mean("FradTop") / flux_ref,
        "bottom_hydro_flux_over_solar": mean("FhydBottom") / flux_ref,
        "bottom_radiation_flux_over_solar": mean("FradBottom") / flux_ref,
        "max_abs_mass_drift": float(np.max(np.abs(h["MassDrift"][indices]))),
        "pbnd_scale_mean": mean("PBNDScale"),
    }


def summarize(h: dict[str, np.ndarray], log: dict[str, object], units: dict[str, float],
              flux_ref: float) -> dict[str, object]:
    required = {"time", "VzRms2Mean", "FtotTop", "FradTop", "FhydBottom",
                "FradBottom", "MassDrift", "PBNDScale"}
    missing = required - h.keys()
    if missing:
        raise ValueError(f"Missing history columns: {sorted(missing)}")
    n = len(h["time"])
    quarter = n // 4
    velocity_unit = np.sqrt(units["egas_unit"] / units["rho_unit"])
    time_unit = physical_time_unit(units, velocity_unit)
    height_cm = (units["x3max"] - units["x3min"]) * units["leng_unit"]
    report: dict[str, object] = {
        "samples": n, "time_unit_seconds": time_unit,
        "vertical_box_height_cm": height_cm, "reference_flux_cgs": flux_ref,
        "log": log,
        "stationarity_rule": "Two late quarters, each >=1 vertical crossing time; "
        "top flux within 10% solar and changes <=5%; vertical RMS changes <=10%; "
        "mass drift <=1%; all reported NR solves converged; no fatal error.",
        "stationarity_candidate": False,
    }
    if quarter < 5:
        report["reason"] = "Fewer than five history samples per late window"
        return report
    first = window_metrics(h, np.arange(n - 2 * quarter, n - quarter), flux_ref,
                           velocity_unit, time_unit, height_cm)
    second = window_metrics(h, np.arange(n - quarter, n), flux_ref,
                            velocity_unit, time_unit, height_cm)
    report["early_late_window"] = first
    report["late_window"] = second
    flux_change = abs(second["top_total_flux_over_solar"]
                      - first["top_total_flux_over_solar"])
    speed_change = abs(second["vz_rms_km_s"] - first["vz_rms_km_s"])
    speed_change /= max(second["vz_rms_km_s"], 1e-30)
    report["top_flux_change_over_solar"] = flux_change
    report["vz_rms_relative_change"] = speed_change
    statuses = log["nr_status_counts"]
    report["stationarity_candidate"] = bool(
        min(first["vertical_crossing_times"], second["vertical_crossing_times"]) >= 1
        and abs(second["top_total_flux_over_solar"] - 1) <= 0.1
        and flux_change <= 0.05 and speed_change <= 0.1
        and second["max_abs_mass_drift"] <= 0.01
        and log["fatal_count"] == 0 and log["completed"]
        and bool(statuses)
        and set(statuses).issubset({"converged", "dt_zero_initialization"})
    )
    report["reason"] = ("Screening rule passed; compare profiles and resolution next"
                        if report["stationarity_candidate"]
                        else "Screening rule not met; inspect windows and continue/adjust")
    return report


def make_plot(h: dict[str, np.ndarray], units: dict[str, float], flux_ref: float,
              output: Path) -> None:
    velocity_unit = np.sqrt(units["egas_unit"] / units["rho_unit"])
    time_unit = physical_time_unit(units, velocity_unit)
    time_min = h["time"] * time_unit / 60.0
    plt.rcParams.update({"font.size": 13})
    fig, ax = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax[0, 0].plot(time_min, h["FtotTop"] / flux_ref, label="top total")
    ax[0, 0].plot(time_min, (h["FhydBottom"] + h["FradBottom"]) / flux_ref,
                  label="bottom total")
    ax[0, 0].axhline(1, color="black", lw=1, ls="--", label="solar flux")
    ax[0, 0].set(ylabel=r"$F/F_\odot$", title="Boundary energy flux")
    ax[0, 0].legend()
    ax[0, 1].plot(time_min,
                  np.sqrt(np.maximum(h["VzRms2Mean"], 0)) * velocity_unit / 1e5)
    ax[0, 1].set(ylabel="km/s", title="Volume vertical velocity RMS")
    ax[1, 0].plot(time_min, 100 * h["MassDrift"])
    ax[1, 0].axhline(0, color="black", lw=1)
    ax[1, 0].set(ylabel="%", title="Mass drift")
    ax[1, 1].plot(time_min, h["PBNDScale"])
    ax[1, 1].set(ylabel="scale", title="Bottom pressure control")
    for row in ax:
        for a in row:
            a.set_xlabel("physical time [min]")
            a.grid(alpha=0.25)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--history-header-from", type=Path,
                        help="Initial .hst for a restart run whose .hst lacks a header")
    parser.add_argument("--reference-flux", type=float, default=6.3e10)
    args = parser.parse_args()
    hst = next(args.run_dir.glob("*.hst"))
    h = read_history(hst, args.history_header_from)
    log = parse_log(args.run_dir / "run.log")
    units = input_values(args.input)
    report = summarize(h, log, units, args.reference_flux)
    (args.run_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    make_plot(h, units, args.reference_flux, args.run_dir / "history.png")
    candidate = report["stationarity_candidate"]
    status = "Stationarity screening passed" if candidate else "Stationarity not established"
    summary = html.escape(json.dumps(report, indent=2, ensure_ascii=False))
    gallery = "".join(
        f"<figure><img src='{html.escape(path.relative_to(args.run_dir).as_posix())}' "
        f"alt='{html.escape(path.stem)}'><figcaption>{html.escape(path.stem)}</figcaption></figure>"
        for path in sorted((args.run_dir / "imgs").glob("*.png"))
    )
    movies = "".join(
        f"<li><a href='{html.escape(path.relative_to(args.run_dir).as_posix())}'>"
        f"{html.escape(path.name)}</a></li>"
        for path in sorted((args.run_dir / "movies").glob("*.mp4"))
    )
    (args.run_dir / "index.html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<title>Solar convection run</title><style>body{font:19px/1.5 system-ui;"
        "max-width:1100px;margin:2rem auto;padding:0 1rem;color:#172b3a}"
        "h1{font-size:2rem}.status{font-weight:700;color:"
        + ("#117347" if candidate else "#a14b08") + "}img{max-width:100%}"
        "pre{font-size:15px;white-space:pre-wrap;background:#f3f6f8;padding:1rem}"
        "</style><h1>Non-magnetic solar convection</h1><p class='status'>"
        + status + "</p><p>Screening only. Full solar comparison requires vertical "
        "profiles, optical-depth diagnostics, and resolution tests.</p>"
        "<p><a href='history.png'>History plot</a> · <a href='summary.json'>JSON</a> · "
        "<a href='run.log'>run.log</a></p><img src='history.png' alt='history'>"
        f"<h2>Existing plots</h2>{gallery}"
        + (f"<h2>Movies</h2><ul>{movies}</ul>" if movies else "")
        + f"<h2>Metrics</h2><pre>{summary}</pre></html>"
    )
    print(f"{args.run_dir / 'index.html'}: {status}")


if __name__ == "__main__":
    main()
