#!/usr/bin/env python3
"""Measure the omitted Godunov radiation-advection boundary flux in solar HST."""

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

from solar_convection_report import (input_values, make_campaign_index,
                                     make_energy_budget_plot, parse_log,
                                     physical_time_unit, read_history,
                                     sampled_energy_ledger, total_bottom_flux,
                                     total_top_flux)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--header-from", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=float, default=850.2)
    parser.add_argument("--end", type=float, default=855.0)
    parser.add_argument("--reference-flux", type=float, default=6.3e10)
    args = parser.parse_args()
    h = read_history(next(args.run.glob("*.hst")), args.header_from)
    if not {"FadvRadBottom", "FadvRadTop"}.issubset(h):
        raise ValueError("Run history lacks the two advective radiation fluxes")
    log = parse_log(args.run/"run.log")
    if not log["completed"] or log["fatal_count"]:
        raise ValueError("Diagnostic run has not completed cleanly")
    selected = (h["time"] >= args.start) & (h["time"] <= args.end)
    h = {key: values[selected] for key, values in h.items()}
    if len(h["time"]) < 5:
        raise ValueError("Selected interval has fewer than five HST samples")
    units = input_values(args.run/"athinput.generated")
    old_h = {key: values for key, values in h.items()
             if key not in {"FadvRadBottom", "FadvRadTop"}}
    indices = np.arange(len(h["time"]))
    without = sampled_energy_ledger(old_h, indices, units, args.reference_flux)
    with_adv = sampled_energy_ledger(h, indices, units, args.reference_flux)
    if without is None or with_adv is None:
        raise ValueError("History lacks an energy-ledger field")
    velocity_unit = np.sqrt(units["egas_unit"]/units["rho_unit"])
    seconds_per_code = physical_time_unit(units, velocity_unit)
    area = ((units["x1max"]-units["x1min"])
            *(units["x2max"]-units["x2min"])*units["leng_unit"]**2)
    supply = (total_bottom_flux(h)-total_top_flux(h)
              + units["rho_unit"]*velocity_unit*units["grav_acc"]
              *units["leng_unit"]
              *(h["BottomMdotNet"]*units["x3min"]
                - h["TopMdotOut"]*units["x3max"]))
    stored_rate = np.diff(h["EtotGrav"])/(area*np.diff(h["time"])
                                         *seconds_per_code)
    def step_rms(sampled_supply: np.ndarray) -> float:
        return float(np.sqrt(np.mean(((stored_rate-sampled_supply)
                                      /args.reference_flux)**2)))
    result = {
        "description": "Same-run sampled ledger with and without Godunov radiation advection",
        "run_dir": str(args.run),
        "start_time_code": float(h["time"][0]),
        "end_time_code": float(h["time"][-1]),
        "sample_count": len(h["time"]),
        "without_advective_radiation": without,
        "with_advective_radiation": with_adv,
        "net_advective_radiation_supply_over_solar": (
            without["residual_equivalent_solar_flux"]
            - with_adv["residual_equivalent_solar_flux"]),
        "mean_bottom_advective_radiation_flux_over_solar": float(np.mean(
            h["FadvRadBottom"])/args.reference_flux),
        "mean_top_advective_radiation_flux_over_solar": float(np.mean(
            h["FadvRadTop"])/args.reference_flux),
        "max_abs_bottom_advective_radiation_flux_over_solar": float(np.max(
            np.abs(h["FadvRadBottom"]))/args.reference_flux),
        "max_abs_top_advective_radiation_flux_over_solar": float(np.max(
            np.abs(h["FadvRadTop"]))/args.reference_flux),
        "stepwise_left_rms_residual_over_solar": step_rms(supply[:-1]),
        "stepwise_trapezoid_rms_residual_over_solar": step_rms(
            0.5*(supply[:-1]+supply[1:])),
        "stepwise_right_rms_residual_over_solar": step_rms(supply[1:]),
        "newton_converged_count": log["nr_status_counts"].get("converged", 0),
        "slow_mg_warning_count": log["slow_mg_warning_count"],
        "fatal_count": log["fatal_count"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    plt.rcParams.update({"font.size": 13})
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    time = h["time"]
    ax.plot(time, h["FadvRadBottom"]/args.reference_flux,
            label="bottom radiation advection")
    ax.plot(time, h["FadvRadTop"]/args.reference_flux,
            label="top radiation advection")
    ax.plot(time, (h["FadvRadBottom"]-h["FadvRadTop"])
            /args.reference_flux, label="net boundary supply")
    ax.axhline(0, color="black", lw=1)
    ax.set(xlabel="time [code]", ylabel=r"$F/F_\odot$",
           title="Godunov radiation advection at the open boundaries")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(args.output_dir/"advective_radiation_flux.png", dpi=160)
    plt.close(fig)
    make_energy_budget_plot(old_h, units, args.reference_flux,
                            args.output_dir/"energy_without_advection.png",
                            "Sampled ledger omitting radiation advection")
    make_energy_budget_plot(h, units, args.reference_flux,
                            args.output_dir/"energy_with_advection.png",
                            "Sampled ledger including radiation advection")
    link = html.escape(os.path.relpath(args.run/"index.html", args.output_dir))
    (args.output_dir/"index.html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<title>Solar radiation-advection flux audit</title><style>"
        "body{font:19px/1.5 system-ui;max-width:1250px;margin:2rem auto;"
        "padding:0 1rem;color:#172b3a}table{border-collapse:collapse}"
        "th,td{padding:.5rem .75rem;border-bottom:1px solid #ddd;text-align:left}"
        "img{max-width:100%}</style><h1>Solar radiation-advection audit</h1>"
        f"<p>Same completed trajectory, t={result['start_time_code']:.3f}–"
        f"{result['end_time_code']:.3f}, {result['sample_count']} HST samples. "
        f"<a href='{link}'>Case report</a> · "
        "<a href='summary.json'>JSON</a>.</p>"
        "<p><strong>Sampled boundary ledger:</strong> this compares the same "
        "stored energy against flux sums with and without the Godunov "
        "radiation-advection boundary term. Accepted-step HST samples are "
        "still not exact RK-stage integrals. The per-step stored-energy rate "
        f"minus left/trapezoid/right sampled supply has RMS "
        f"{result['stepwise_left_rms_residual_over_solar']:.4f}/"
        f"{result['stepwise_trapezoid_rms_residual_over_solar']:.4f}/"
        f"{result['stepwise_right_rms_residual_over_solar']:.4f} solar flux.</p>"
        "<table><tr><th>Boundary terms</th><th>Residual / solar</th></tr>"
        f"<tr><td>Without radiation advection</td><td>"
        f"{without['residual_equivalent_solar_flux']:+.6f}</td></tr>"
        f"<tr><td>With radiation advection</td><td>"
        f"{with_adv['residual_equivalent_solar_flux']:+.6f}</td></tr>"
        "</table><img src='advective_radiation_flux.png' alt='advective flux'>"
        "<img src='energy_without_advection.png' alt='ledger without advection'>"
        "<img src='energy_with_advection.png' alt='ledger with advection'>"
        "</html>"
    )
    make_campaign_index(args.output_dir.parent)
    print(args.output_dir/"index.html")


if __name__ == "__main__":
    main()
