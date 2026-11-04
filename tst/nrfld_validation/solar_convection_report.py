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
    header = next((line for line in path.read_text().splitlines()
                   if line.startswith("# [")), None)
    if header is None and header_from is not None:
        header = next((line for line in header_from.read_text().splitlines()
                       if line.startswith("# [")), None)
    if header is None:
        raise ValueError(f"History header missing in {path}; pass --history-header-from")
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


def combine_histories(paths: list[Path], header_from: Path | None = None
                      ) -> dict[str, np.ndarray]:
    """Join restart segments by time, keeping the last row at each timestamp."""
    segments = [read_history(path, header_from) for path in paths]
    names = set(segments[0])
    if any(set(segment) != names for segment in segments[1:]):
        raise ValueError("History column sets differ between restart segments")
    combined = {name: np.concatenate([segment[name] for segment in segments])
                for name in segments[0]}
    order = np.argsort(combined["time"], kind="stable")
    times = combined["time"][order]
    keep = np.r_[np.diff(times) > 0, True]
    return {name: values[order][keep] for name, values in combined.items()}


def input_values(path: Path) -> dict[str, float]:
    text = path.read_text()
    return {name: float(value) for name, value in KEY_VALUE.findall(text)
            if name in {"rho_unit", "egas_unit", "leng_unit", "time_unit",
                        "x1min", "x1max", "x2min", "x2max",
                        "x3min", "x3max", "grav_acc"}}


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
    up = mean("BottomMdotUp")
    down = -mean("BottomMdotDown")
    net = mean("BottomMdotNet")
    return {
        "start_time": float(t[0]), "end_time": float(t[-1]),
        "duration_seconds": float((t[-1] - t[0]) * time_unit),
        "vertical_crossing_times": float((t[-1] - t[0]) * time_unit / crossing),
        "vz_rms_km_s": speed / 1.0e5,
        "top_total_flux_over_solar": mean("FtotTop") / flux_ref,
        "top_radiation_flux_over_solar": mean("FradTop") / flux_ref,
        "bottom_hydro_flux_over_solar": mean("FhydBottom") / flux_ref,
        "bottom_radiation_flux_over_solar": mean("FradBottom") / flux_ref,
        "bottom_upflow_code": up,
        "bottom_downflow_magnitude_code": down,
        "bottom_net_mass_flux_code": net,
        "bottom_net_mean_fraction": abs(net) / max(up + down, 1.0e-30),
        "bottom_net_rms_fraction": (float(np.sqrt(np.mean(
            h["BottomMdotNet"][indices]**2))) / max(up + down, 1.0e-30)),
        "max_abs_mass_drift": float(np.max(np.abs(h["MassDrift"][indices]))),
        "pbnd_scale_mean": mean("PBNDScale"),
    }


def sampled_energy_ledger(h: dict[str, np.ndarray], indices: np.ndarray,
                          units: dict[str, float], flux_ref: float
                          ) -> dict[str, float] | None:
    """Integrate sampled face fluxes and compare stored gas+rad+gravity energy.

    The HST face fluxes are sampled after accepted steps, not the exact
    time-integrated stage fluxes. This is a diagnostic, not a conservation
    acceptance criterion.
    """
    needed = {"EtotGrav", "FhydBottom", "FradBottom", "FtotTop",
              "BottomMdotNet", "TopMdotOut"}
    geometry = {"x1min", "x1max", "x2min", "x2max", "x3min", "x3max",
                "grav_acc", "rho_unit", "egas_unit", "leng_unit"}
    if not needed.issubset(h) or not geometry.issubset(units) or len(indices) < 2:
        return None
    velocity_unit = np.sqrt(units["egas_unit"] / units["rho_unit"])
    time_unit = physical_time_unit(units, velocity_unit)
    t = h["time"][indices] * time_unit
    area = ((units["x1max"] - units["x1min"])
            * (units["x2max"] - units["x2min"])
            * units["leng_unit"]**2)
    regular_flux = (h["FhydBottom"][indices]
                    + h["FradBottom"][indices] - h["FtotTop"][indices])
    mass_flux_unit = units["rho_unit"] * velocity_unit
    potential_flux = (mass_flux_unit * units["grav_acc"] * units["leng_unit"]
                      * (h["BottomMdotNet"][indices] * units["x3min"]
                         - h["TopMdotOut"][indices] * units["x3max"]))
    regular_integral = float(np.trapezoid(regular_flux, t) * area)
    potential_integral = float(np.trapezoid(potential_flux, t) * area)
    stored_change = float(h["EtotGrav"][indices][-1]
                          - h["EtotGrav"][indices][0])
    residual = stored_change - regular_integral - potential_integral
    return {
        "start_time_code": float(h["time"][indices][0]),
        "end_time_code": float(h["time"][indices][-1]),
        "stored_energy_change_erg": stored_change,
        "sampled_hydro_radiation_boundary_erg": regular_integral,
        "sampled_potential_boundary_erg": potential_integral,
        "sampled_ledger_residual_erg": residual,
        "residual_equivalent_solar_flux": float(residual / (area * (t[-1]-t[0])
                                                         * flux_ref)),
    }


def summarize(h: dict[str, np.ndarray], log: dict[str, object], units: dict[str, float],
              flux_ref: float) -> dict[str, object]:
    required = {"time", "VzRms2Mean", "FtotTop", "FradTop", "FhydBottom",
                "FradBottom", "MassDrift", "PBNDScale", "BottomMdotUp",
                "BottomMdotDown", "BottomMdotNet"}
    missing = required - h.keys()
    if missing:
        raise ValueError(f"Missing history columns: {sorted(missing)}")
    n = len(h["time"])
    quarter = n // 4
    velocity_unit = np.sqrt(units["egas_unit"] / units["rho_unit"])
    time_unit = physical_time_unit(units, velocity_unit)
    height_cm = (units["x3max"] - units["x3min"]) * units["leng_unit"]
    report: dict[str, object] = {
        "samples": n, "last_history_time_code": float(h["time"][-1]),
        "time_unit_seconds": time_unit,
        "vertical_box_height_cm": height_cm, "reference_flux_cgs": flux_ref,
        "log": log,
        "stationarity_rule": "Two late quarters, each >=1 vertical crossing time; "
        "top flux within 10% solar and changes <=5%; vertical RMS changes <=10%; "
        "mass drift <=1%; all reported NR solves converged; no fatal error.",
        "stationarity_candidate": False,
    }
    ledger = sampled_energy_ledger(h, np.arange(n), units, flux_ref)
    if ledger:
        report["sampled_energy_ledger_full"] = ledger
    if quarter < 5:
        report["reason"] = "Fewer than five history samples per late window"
        return report
    first = window_metrics(h, np.arange(n - 2 * quarter, n - quarter), flux_ref,
                           velocity_unit, time_unit, height_cm)
    second = window_metrics(h, np.arange(n - quarter, n), flux_ref,
                            velocity_unit, time_unit, height_cm)
    report["early_late_window"] = first
    report["late_window"] = second
    ledger = sampled_energy_ledger(h, np.arange(n - quarter, n), units, flux_ref)
    if ledger:
        report["sampled_energy_ledger_late"] = ledger
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

    fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                           constrained_layout=True)
    ax[0].plot(time_min, h["BottomMdotUp"], label="upflow")
    ax[0].plot(time_min, -h["BottomMdotDown"], label="downflow magnitude")
    ax[0].set(ylabel="mass flux [code]", title="Bottom boundary mass transport")
    ax[0].legend()
    ax[1].plot(time_min, h["BottomMdotNet"], label="net")
    ax[1].axhline(0, color="black", lw=1)
    ax[1].set(xlabel="physical time [min]", ylabel="net mass flux [code]")
    for axis in ax:
        axis.grid(alpha=0.25)
    image_dir = output.parent / "imgs"
    image_dir.mkdir(exist_ok=True)
    fig.savefig(image_dir / "simple_convection_bottom_mass_flux.png", dpi=150)
    plt.close(fig)


def make_campaign_index(root: Path) -> None:
    rows = []
    for case in sorted(root.iterdir()):
        summary = case / "summary.json"
        if not case.is_dir():
            continue
        if not summary.is_file():
            if (case / "index.html").is_file():
                cells = [
                    f"<a href='{html.escape(case.name)}/index.html'>"
                    f"{html.escape(case.name)}</a>",
                    "sub-campaign", "see linked report", "—", "—", "—",
                ]
                rows.append("<tr>" + "".join(f"<td>{x}</td>" for x in cells)
                            + "</tr>")
            continue
        item = json.loads(summary.read_text())
        log = item["log"]
        statuses = log["nr_status_counts"]
        solver_ok = (log["completed"] and log["fatal_count"] == 0
                     and bool(statuses)
                     and set(statuses).issubset({"converged", "dt_zero_initialization"}))
        intentionally_stopped = bool(item.get("intentional_stop_reason"))
        late = item.get("late_window", {})
        ratio = late.get("top_total_flux_over_solar")
        cells = [
            f"<a href='{html.escape(case.name)}/index.html'>{html.escape(case.name)}</a>",
            ("<strong>PASS</strong>" if solver_ok else
             "intentionally stopped" if intentionally_stopped else "incomplete / failed"),
            "candidate" if item["stationarity_candidate"] else "not established",
            str(log["final_cycle"] if log["final_cycle"] is not None else "—"),
            f"{ratio:.3f}" if ratio is not None else "—",
            str(log["slow_mg_warning_count"]),
        ]
        rows.append("<tr>" + "".join(f"<td>{x}</td>" for x in cells) + "</tr>")
    headers = ("Case", "Solver completion", "Stationarity", "Final cycle",
               "Late top flux / solar", "Slow MG warnings")
    table = "<tr>" + "".join(f"<th>{x}</th>" for x in headers) + "</tr>"
    (root / "index.html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<title>Solar convection campaign</title>"
        "<style>body{font:19px/1.5 system-ui;max-width:1200px;margin:2rem auto;"
        "padding:0 1rem;color:#172b3a}h1{font-size:2rem}table{border-collapse:collapse}"
        "th,td{padding:.5rem .75rem;border-bottom:1px solid #ddd;text-align:left}"
        "strong{color:#117347}</style><h1>Non-magnetic solar convection</h1>"
        "<p><strong>Solver completion</strong> and statistical stationarity are "
        "separate checks. A short solver pass does not establish solar agreement.</p>"
        f"<table>{table}{''.join(rows)}</table></html>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--history-header-from", type=Path,
                        help="Initial .hst for a restart run whose .hst lacks a header")
    parser.add_argument("--history-prepend", type=Path, action="append", default=[],
                        help="Earlier .hst segment; repeat in chronological order")
    parser.add_argument("--reference-flux", type=float, default=6.3e10)
    args = parser.parse_args()
    hst = next(args.run_dir.glob("*.hst"))
    h = combine_histories([*args.history_prepend, hst], args.history_header_from)
    log = parse_log(args.run_dir / "run.log")
    units = input_values(args.input)
    report = summarize(h, log, units, args.reference_flux)
    manifest_path = args.run_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("intentional_stop_reason"):
            report["intentional_stop_reason"] = manifest["intentional_stop_reason"]
    tau_path = args.run_dir / "tau1_slice.json"
    if tau_path.is_file():
        tau_records = json.loads(tau_path.read_text())["records"]
        if tau_records:
            report["tau1_slice_last"] = tau_records[-1]
    surface_path = args.run_dir / "surface3d.json"
    if surface_path.is_file():
        surface_records = json.loads(surface_path.read_text())["records"]
        if surface_records:
            report["surface3d_last"] = surface_records[-1]
    late_profiles_path = args.run_dir / "profiles3d_late_mean.json"
    if late_profiles_path.is_file():
        late_profiles = json.loads(late_profiles_path.read_text())
        report["profiles3d_late_window"] = {
            key: late_profiles[key]
            for key in ("window_start_code", "window_end_code", "snapshot_count")
        }
    (args.run_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    make_plot(h, units, args.reference_flux, args.run_dir / "history.png")
    candidate = report["stationarity_candidate"]
    status = ("Run intentionally stopped; stationarity not established"
              if report.get("intentional_stop_reason") else
              "Stationarity screening passed" if candidate else
              "Stationarity not established")
    summary = html.escape(json.dumps(report, indent=2, ensure_ascii=False))
    late = report.get("late_window", {})
    log = report["log"]
    quick = [
        ("Completed", str(log["completed"])),
        ("Final cycle", str(log["final_cycle"])),
        ("Last history time", f"{report['last_history_time_code']:.3f} code"),
        ("Wall time", str(log["wall_time"] or "see manifest")),
        ("Slow MG warnings", str(log["slow_mg_warning_count"])),
        ("NR statuses", ", ".join(f"{k}: {v}" for k, v in log["nr_status_counts"].items())),
        ("Late top flux / solar", f"{late['top_total_flux_over_solar']:.3f}" if late else "—"),
        ("Late vertical RMS", f"{late['vz_rms_km_s']:.3f} km/s" if late else "—"),
        ("Late window crossings", f"{late['vertical_crossing_times']:.3f}" if late else "—"),
        ("Late bottom net-flow RMS / gross flow",
         f"{late['bottom_net_rms_fraction']:.3f}" if late else "—"),
    ]
    if report.get("intentional_stop_reason"):
        quick.insert(1, ("Stop reason", report["intentional_stop_reason"]))
    tau_last = report.get("tau1_slice_last")
    if tau_last and tau_last["T_tau1_mean_K"] is not None:
        quick.extend([
            ("Last slice T at tau=1", f"{tau_last['T_tau1_mean_K']:.0f} K"),
            ("Last slice T at tau=2/3", f"{tau_last['T_tau23_mean_K']:.0f} K"),
            ("Last slice tau=1 height", f"{tau_last['z_tau1_mean_code']:.3f} code"),
            ("Last slice tau=1 coverage", f"{100 * tau_last['tau1_coverage']:.0f}%"),
        ])
    surface_last = report.get("surface3d_last")
    if surface_last:
        quick.extend([
            ("Last full-3D T at tau=2/3",
             f"{surface_last['T_tau23_mean_K']:.0f} K"),
            ("Last full-3D tau=2/3 coverage",
             f"{100 * surface_last['coverage']:.0f}%"),
        ])
    late_profiles = report.get("profiles3d_late_window")
    if late_profiles:
        quick.append(("Full-3D late profile window",
                      f"{late_profiles['window_start_code']:.2f}–"
                      f"{late_profiles['window_end_code']:.2f} code, "
                      f"{late_profiles['snapshot_count']} snapshots"))
    ledger = report.get("sampled_energy_ledger_late")
    if ledger:
        quick.append(("Late sampled energy-ledger residual",
                      f"{ledger['residual_equivalent_solar_flux']:+.3f} F_sun"))
    table = "".join(f"<tr><th>{html.escape(key)}</th><td>{html.escape(value)}</td></tr>"
                    for key, value in quick)
    gallery = "".join(
        f"<figure><img src='{html.escape(path.relative_to(args.run_dir).as_posix())}' "
        f"alt='{html.escape(path.stem)}'><figcaption>{html.escape(path.stem)}</figcaption></figure>"
        for path in sorted((args.run_dir / "imgs").glob("*.png"))
    )
    movies = "".join(
        f"<figure><video controls preload='metadata' style='max-width:100%' "
        f"src='{html.escape(path.relative_to(args.run_dir).as_posix())}'></video>"
        f"<figcaption>{html.escape(path.name)}</figcaption></figure>"
        for path in sorted((args.run_dir / "movies").glob("*.mp4"))
    )
    flux_note = (
        "<p><strong>Flux profile:</strong> the 3D curves use cell-center "
        "reconstruction, including an approximate FLD gradient. Read exact "
        "top and bottom boundary fluxes from the history metrics above.</p>"
        if (args.run_dir / "flux3d.json").is_file() else "")
    energy_note = (
        "<p><strong>Energy ledger:</strong> the reported residual integrates "
        "sampled face fluxes and potential-energy transport. It is a diagnostic "
        "for follow-up, not an exact discrete conservation test.</p>"
        if ledger else "")
    (args.run_dir / "index.html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<title>Solar convection run</title><style>body{font:19px/1.5 system-ui;"
        "max-width:1100px;margin:2rem auto;padding:0 1rem;color:#172b3a}"
        "h1{font-size:2rem}.status{font-weight:700;color:"
        + ("#117347" if candidate else "#a14b08") + "}img{max-width:100%}"
        "table{border-collapse:collapse}th,td{border-bottom:1px solid #ddd;"
        "padding:.4rem 1rem;text-align:left}th{font-weight:650}"
        "pre{font-size:15px;white-space:pre-wrap;background:#f3f6f8;padding:1rem}"
        "</style><h1>Non-magnetic solar convection</h1><p class='status'>"
        + status + "</p><p>Screening only. Full solar comparison requires vertical "
        "profiles, optical-depth diagnostics, and resolution tests.</p>"
        + energy_note +
        "<p><a href='history.png'>History plot</a> · <a href='summary.json'>JSON</a> · "
        "<a href='run.log'>run.log</a></p>"
        f"<table>{table}</table><img src='history.png' alt='history'>"
        f"<h2>Existing plots</h2>{flux_note}{gallery}"
        + (f"<h2>Movies</h2>{movies}" if movies else "")
        + f"<h2>Metrics</h2><pre>{summary}</pre></html>"
    )
    make_campaign_index(args.run_dir.parent)
    print(f"{args.run_dir / 'index.html'}: {status}")


if __name__ == "__main__":
    main()
