#!/usr/bin/env python3
"""Compare old and FLD-consistent solar initial profiles and their startups."""

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

from solar_convection_initial_flux_audit import load_profile, sampled_interior_residuals
from solar_convection_report import input_values, parse_log, read_history


def initial_tau_temperature(profile: dict[str, np.ndarray], tau: float) -> float:
    order = np.argsort(profile["tau"])
    return float(np.interp(tau, profile["tau"][order],
                           profile["temperature"][order]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-profile", type=Path, required=True)
    parser.add_argument("--new-profile", type=Path, required=True)
    parser.add_argument("--old-run", type=Path, required=True)
    parser.add_argument("--new-run", type=Path, required=True)
    parser.add_argument("--nx3", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-flux", type=float, default=6.3e10)
    args = parser.parse_args()
    if args.nx3 < 4:
        parser.error("--nx3 must be at least 4")
    units = input_values(args.old_run / "athinput.generated")
    items = {}
    for label, profile_path, run_dir in (
        ("original", args.old_profile, args.old_run),
        ("FLD closure", args.new_profile, args.new_run),
    ):
        profile = load_profile(profile_path)
        history = read_history(next(run_dir.glob("*.hst")))
        log = parse_log(run_dir / "run.log")
        log_text = (run_dir / "run.log").read_text(errors="replace")
        boundary = {}
        for key in ("PBND", "s_in"):
            match = re.search(rf"\b{key}=([0-9.eE+-]+)", log_text)
            if match is None:
                raise ValueError(f"{run_dir}: missing initial {key} in log")
            boundary[key] = float(match.group(1))
        interior = sampled_interior_residuals(
            profile, units, args.nx3, args.reference_flux)
        surface_path = run_dir / "surface3d.json"
        surfaces = (json.loads(surface_path.read_text()).get("records", [])
                    if surface_path.is_file() else [])
        late = (history["time"] >= 5) & (history["time"] <= 10)
        if not np.any(late):
            raise ValueError(f"{run_dir}: no t=5–10 history samples")
        last_ten = (history["time"] >= history["time"][-1]-10)
        long_startup = history["time"][-1] > 20
        items[label] = {
            "profile": profile,
            "history": history,
            "interior": interior,
            "run_dir": run_dir,
            "profile_path": profile_path,
            "results": {
                "initial_tau23_temperature_K": initial_tau_temperature(profile, 2/3),
                "initial_top_flux_over_solar": float(history["FtotTop"][0]
                                                      / args.reference_flux),
                "top_flux_t5_10_mean_over_solar": float(np.mean(
                    history["FtotTop"][late])/args.reference_flux),
                "top_flux_t10_over_solar": float(history["FtotTop"][-1]
                                                  / args.reference_flux),
                "top_flux_last10_mean_over_solar": (
                    float(np.mean(history["FtotTop"][last_ten])/args.reference_flux)
                    if long_startup else None),
                "uppermost_interior_fld_flux_over_solar": float(
                    interior["discrete_flux"][-1]),
                "uppermost_interior_model_flux_over_solar": float(
                    interior["model_flux"][-1]),
                "completed": log["completed"],
                "nr_status_counts": log["nr_status_counts"],
                "fgmres_iterations_max": log["fgmres_iterations_max"],
                "slow_mg_warning_count": log["slow_mg_warning_count"],
                "fatal_count": log["fatal_count"],
                "run_dir": str(run_dir),
                "profile_path": str(profile_path),
                "initial_PBND_code": boundary["PBND"],
                "initial_s_in_cgs": boundary["s_in"],
                "surface3d_first_tau23_temperature_K": (
                    float(surfaces[0]["T_tau23_mean_K"]) if surfaces else None),
                "surface3d_last_tau23_temperature_K": (
                    float(surfaces[-1]["T_tau23_mean_K"]) if surfaces else None),
            },
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    old = items["original"]["results"]
    new = items["FLD closure"]["results"]
    result = {
        "description": "Matched short startup; transient comparison only",
        "nx3": args.nx3,
        "reference_flux_cgs": args.reference_flux,
        "cases": {label: item["results"] for label, item in items.items()},
        "change_top_flux_t5_10_over_solar": (
            new["top_flux_t5_10_mean_over_solar"]
            - old["top_flux_t5_10_mean_over_solar"]),
        "change_top_flux_last10_over_solar": (
            new["top_flux_last10_mean_over_solar"]
            - old["top_flux_last10_mean_over_solar"]
            if new["top_flux_last10_mean_over_solar"] is not None
            and old["top_flux_last10_mean_over_solar"] is not None else None),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    plt.rcParams.update({"font.size": 13})
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    for label, item in items.items():
        profile = item["profile"]
        h = item["history"]
        interior = item["interior"]
        axes[0].plot(h["time"], h["FtotTop"]/args.reference_flux, label=label)
        axes[1].plot(interior["z"], interior["discrete_flux"], label=label)
        axes[1].plot(interior["z"], interior["model_flux"], ls="--",
                     label=f"{label} continuous")
        axes[2].plot(profile["z"], profile["temperature"], label=label)
    axes[0].axhline(1, color="black", ls=":", lw=1)
    axes[0].set(xlabel="time [code]", ylabel=r"$F_{top}/F_\odot$",
                title="Matched startup top flux")
    axes[1].set(xlabel="height [code]", ylabel=r"$F_{rad}/F_\odot$",
                title="Initial upper radiative flux", xlim=(1.5, 3.1))
    axes[2].set(xlabel="height [code]", ylabel="T [K]",
                title="Initial upper temperature", xlim=(1.5, 3.1))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.savefig(args.output_dir / "comparison.png", dpi=160)
    plt.close(fig)
    headers = ("Case", "T at initial τ=2/3 [K]", "Initial s_in [cgs]",
               "Initial PBND [code]", "Initial top flux / solar",
               "Upper interior FLD / solar", "t=5–10 top flux / solar",
               "Last 10 top flux / solar", "Solver")
    rows = []
    for label, item in items.items():
        row = item["results"]
        link = os.path.relpath(item["run_dir"] / "index.html", args.output_dir)
        solver = ("complete" if row["completed"] else "incomplete")
        solver += f", NR={row['nr_status_counts'].get('converged', 0)}"
        rows.append("<tr>" + "".join(f"<td>{value}</td>" for value in (
            f"<a href='{html.escape(link)}'>{html.escape(label)}</a>",
            f"{row['initial_tau23_temperature_K']:.0f}",
            f"{row['initial_s_in_cgs']:.7e}",
            f"{row['initial_PBND_code']:.3f}",
            f"{row['initial_top_flux_over_solar']:.3f}",
            f"{row['uppermost_interior_fld_flux_over_solar']:.3f}",
            f"{row['top_flux_t5_10_mean_over_solar']:.3f}",
            (f"{row['top_flux_last10_mean_over_solar']:.3f}"
             if row["top_flux_last10_mean_over_solar"] is not None else "—"),
            html.escape(solver),
        )) + "</tr>")
    surface_note = ""
    if (old["surface3d_last_tau23_temperature_K"] is not None
            and new["surface3d_last_tau23_temperature_K"] is not None):
        surface_note = (
            "<p>Full-3D τ=2/3 surface at the last snapshot: "
            f"{old['surface3d_last_tau23_temperature_K']:.0f} K original; "
            f"{new['surface3d_last_tau23_temperature_K']:.0f} K FLD closure. "
            "These are transient surface temperatures.</p>"
        )
    (args.output_dir / "index.html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<title>Solar startup profile comparison</title><style>"
        "body{font:19px/1.5 system-ui;max-width:1400px;margin:2rem auto;"
        "padding:0 1rem;color:#172b3a}table{border-collapse:collapse}"
        "th,td{padding:.5rem .75rem;border-bottom:1px solid #ddd;text-align:left}"
        "img{max-width:100%}</style><h1>Solar initial-profile startup comparison</h1>"
        "<p><strong>Short transient experiment.</strong> The FLD-consistent "
        "profile inverts the same limiter used by the NR operator when "
        "constructing its radiative gradient. The interior-face estimate "
        "samples the 1D profile at cell centers and omits the exact C++ "
        "table interpolation and ghost cells. The lower boundary derives "
        "its initial pressure and inflow entropy from each profile; their "
        "changes are shown in the table and can affect longer runs.</p>"
        f"<p>Change in t=5–10 top flux: "
        f"<strong>{result['change_top_flux_t5_10_over_solar']:+.3f} "
        "solar</strong>. This does not establish long-time stationarity.</p>"
        "<table><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in headers)
        + "</tr>" + "".join(rows) + "</table>"
        + surface_note
        + "<img src='comparison.png' alt='profile and startup comparison'>"
        + "<p><a href='summary.json'>Numerical results</a></p></html>"
    )
    print(args.output_dir / "index.html")


if __name__ == "__main__":
    main()
