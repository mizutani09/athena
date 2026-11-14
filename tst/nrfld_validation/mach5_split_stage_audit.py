#!/usr/bin/env python3
"""Report the explicit-to-implicit temperature change in a Mach-5 shock run."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from html import escape
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mach5_trad_peak_audit import spatial_wiggle  # noqa: E402
from plots.plot_radiative_shock import jump_center, load_analytic_profile  # noqa: E402

PHASES = ("before_step", "before_transport", "after_transport",
          "before_source", "after_source", "after_stage", "before_implicit",
          "after_implicit", "after_step")
FIELDS = ("x", "rho", "Tgas_K", "Tgas_primitive_K", "Trad_K", "Er", "egas",
          "S_couple", "S_diff_implicit", "implicit_rate_residual")


def read_stages(directory: Path):
    groups = defaultdict(list)
    paths = sorted(directory.glob("fld_stages.r*.b*.csv"))
    if not paths:
        raise ValueError(f"no FLD stage CSV files in {directory}")
    for path in paths:
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                key = (int(row["cycle"]), row["phase"], int(row["stage"]))
                if row["phase"] in PHASES:
                    groups[key].append(row)
    result = {}
    for key, rows in groups.items():
        rows.sort(key=lambda row: float(row["x"]))
        x = np.array([float(row["x"]) for row in rows])
        if len(x) != len(np.unique(x)):
            raise ValueError(f"duplicate sampled x cells for {key}")
        item = {name: np.array([float(row[name]) for row in rows])
                for name in FIELDS}
        item["time_s"] = float(rows[0]["time"])
        item["dt_s"] = float(rows[0]["dt"])
        result[key] = item
    return result, paths


def metrics(cycle: int, before: dict, after: dict, shock: float):
    if not np.array_equal(before["x"], after["x"]):
        raise ValueError(f"sample positions differ at cycle {cycle}")
    x = before["x"] - shock
    near = (x >= -100) & (x <= 100)
    wide = (x >= -300) & (x <= 300)
    drop_g = before["Tgas_K"] - after["Tgas_K"]
    drop_r = before["Trad_K"] - after["Trad_K"]
    total_rate = ((after["egas"]-before["egas"])
                  + (after["Er"]-before["Er"]))/before["dt_s"]
    energy_mismatch = total_rate-after["S_diff_implicit"]
    row = {
        "cycle": cycle, "time_start_s": before["time_s"],
        "time_end_s": before["time_s"] + before["dt_s"],
        "dt_s": before["dt_s"], "shock_x_cm": shock,
        "Tgas_peak_explicit_K": float(before["Tgas_K"][near].max()),
        "Tgas_peak_implicit_K": float(after["Tgas_K"][near].max()),
        "Trad_peak_explicit_K": float(before["Trad_K"][near].max()),
        "Trad_peak_implicit_K": float(after["Trad_K"][near].max()),
        "Trad_wiggle_explicit_K": spatial_wiggle(x, before["Trad_K"])["height_K"],
        "Trad_wiggle_implicit_K": spatial_wiggle(x, after["Trad_K"])["height_K"],
        "max_same_cell_Tgas_drop_K": float(drop_g[wide].max()),
        "max_same_cell_Trad_drop_K": float(drop_r[wide].max()),
        "max_abs_implicit_rate_residual": float(np.nanmax(np.abs(
            after["implicit_rate_residual"][wide]))),
        "max_abs_implicit_energy_budget_rate_error": float(np.nanmax(np.abs(
            energy_mismatch[wide]))),
        "relative_implicit_energy_budget_rate_error": float(np.nanmax(np.abs(
            energy_mismatch[wide]))/max(np.nanmax(np.abs(
                after["S_diff_implicit"][wide])), 1e-300)),
        "max_conserved_vs_primitive_Tgas_diff_K": float(max(
            np.max(np.abs(before["Tgas_K"]-before["Tgas_primitive_K"])),
            np.max(np.abs(after["Tgas_K"]-after["Tgas_primitive_K"])))),
    }
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=HERE/
                        "reference/radiative_shock/mach5_semianalytic.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stages, sources = read_stages(args.stage_dir)
    cycles = sorted({key[0] for key in stages})
    analytic = load_analytic_profile(args.reference)
    reference_shock = jump_center(analytic["x"], analytic["rho"])
    analytic_gas_peak = float(np.max(analytic["tgas"]))
    downstream_gas = float(analytic["tgas"][-1])
    above_half = analytic["tgas"] > 0.5*(analytic_gas_peak+downstream_gas)
    analytic_spike_width = float(np.ptp(analytic["x"][above_half]))
    results = []
    for cycle in cycles:
        pre = stages[cycle, "before_implicit", 0]
        post = stages[cycle, "after_implicit", 0]
        shock = jump_center(post["x"], post["rho"])
        results.append(metrics(cycle, pre, post, shock))
        for phase, stage in (("after_stage", 2), ("after_step", 0)):
            other = stages[cycle, phase, stage]
            target = pre if phase == "after_stage" else post
            for name in ("rho", "Tgas_K", "Trad_K", "Er"):
                if not np.array_equal(other[name], target[name]):
                    raise ValueError(f"{phase} and neighboring phase differ: {cycle} {name}")
    final = results[-1]
    cycle = final["cycle"]
    pre = stages[cycle, "before_implicit", 0]
    post = stages[cycle, "after_implicit", 0]
    shock = final["shock_x_cm"]
    x = pre["x"]-shock
    xa = analytic["x"]-reference_shock
    near = (x >= -100) & (x <= 100)
    hot_index = int(np.flatnonzero(near)[np.argmax(pre["Trad_K"][near])])
    def phase_temperature(phase: str, stage: int):
        return float(stages[cycle, phase, stage]["Trad_K"][hot_index])
    hotspot = {
        "x_from_shock_cm": float(x[hot_index]),
        "first_stage_transport_Trad_change_K":
            phase_temperature("after_transport", 1)
            - phase_temperature("before_transport", 1),
        "first_stage_explicit_source_Trad_change_K":
            phase_temperature("after_source", 1)-phase_temperature("before_source", 1),
        "second_stage_transport_Trad_change_K":
            phase_temperature("after_transport", 2)
            - phase_temperature("before_transport", 2),
        "second_stage_explicit_source_Trad_change_K":
            phase_temperature("after_source", 2)-phase_temperature("before_source", 2),
        "implicit_Trad_change_K": float(post["Trad_K"][hot_index]
                                        - pre["Trad_K"][hot_index]),
    }
    args.output.mkdir(parents=True, exist_ok=True)

    with (args.output/"stage_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    with (args.output/"last_step_profile.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x_relative_cm", "Tgas_explicit_K", "Tgas_implicit_K",
                         "Trad_explicit_K", "Trad_implicit_K", "S_couple_implicit",
                         "S_diff_implicit"])
        writer.writerows(zip(x, pre["Tgas_K"], post["Tgas_K"],
                             pre["Trad_K"], post["Trad_K"],
                             post["S_couple"], post["S_diff_implicit"]))

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    for ax, name, lo, hi in ((axes[0], "Trad_K", -100, 300),
                              (axes[1], "Tgas_K", -100, 150)):
        ref = "trad" if name == "Trad_K" else "tgas"
        ax.plot(xa, analytic[ref], "k--", lw=1.6, label="semi-analytic")
        ax.plot(x, pre[name], "o-", ms=4, label="after explicit RK2")
        ax.plot(x, post[name], "o-", ms=4, label="after implicit FLD")
        ax.set_xlim(lo, hi)
        values = [analytic[ref][(xa >= lo) & (xa <= hi)],
                  pre[name][(x >= lo) & (x <= hi)],
                  post[name][(x >= lo) & (x <= hi)]]
        vmin = min(float(np.min(value)) for value in values if len(value))
        vmax = max(float(np.max(value)) for value in values if len(value))
        margin = max(0.03, 0.06*(vmax-vmin))
        ax.set_ylim(vmin-margin, vmax+margin)
        ax.set_ylabel(("T_rad" if ref == "trad" else "T_gas") + " [K]")
        ax.grid(alpha=.3)
        ax.legend()
    axes[1].set_xlabel("x - density-jump center [cm]")
    fig.savefig(args.output/"last_step_profiles.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                             constrained_layout=True)
    t = np.array([row["time_end_s"] for row in results])
    for ax, key, ylabel in ((axes[0], "Tgas_peak", "T_gas peak near shock [K]"),
                            (axes[1], "Trad_peak", "T_rad peak near shock [K]")):
        ax.plot(t, [row[key+"_explicit_K"] for row in results], "o-",
                ms=3, label="after explicit RK2")
        ax.plot(t, [row[key+"_implicit_K"] for row in results], "o-",
                ms=3, label="after implicit FLD")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.3)
        ax.legend()
    axes[1].set_xlabel("end of step t [s]")
    fig.savefig(args.output/"stage_peak_history.png", dpi=180)
    plt.close(fig)

    report = {"stage_directory": str(args.stage_dir.resolve()),
              "stage_files": len(sources), "reference": str(args.reference.resolve()),
              "sampled_cycles": [cycles[0], cycles[-1]],
              "analytic_Tgas_peak_K": analytic_gas_peak,
              "analytic_Tgas_spike_half_height_width_cm": analytic_spike_width,
              "sampling": "central transverse cell, all x cells; conserved gas energy",
              "definition": "explicit=after RK2 before implicit; implicit=after Newton FLD",
              "final": final, "last_step_radiation_hotspot_updates": hotspot,
              "full_dt_median": float(np.median(
                  [row["dt_s"] for row in results[:-1]])),
              "max_after_implicit_wiggle_K": max(
                  row["Trad_wiggle_implicit_K"] for row in results),
              "max_before_implicit_wiggle_K": max(
                  row["Trad_wiggle_explicit_K"] for row in results)}
    (args.output/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    lines = ["<!doctype html><html><head><meta charset='utf-8'>",
             "<title>Mach 5 split step temperatures</title>",
             "<style>body{font:20px/1.5 system-ui;max-width:1200px;margin:32px auto;padding:0 20px}img{max-width:100%}table{border-collapse:collapse}th,td{padding:7px 12px;border:1px solid #bbb;text-align:right}th:first-child{text-align:left}</style></head><body>",
             "<h1>Mach 5: explicit and implicit temperatures</h1>",
             "<p><strong>Output phase:</strong> normal HDF5 is written after the implicit FLD solve. The stage CSV samples the same central transverse line before and after that solve.</p>",
             "<table><tr><th>Final step</th><th>After explicit RK2</th><th>After implicit FLD</th></tr>",
             f"<tr><th>T_gas peak [K]</th><td>{final['Tgas_peak_explicit_K']:.4f}</td><td>{final['Tgas_peak_implicit_K']:.4f}</td></tr>",
             f"<tr><th>T_rad peak [K]</th><td>{final['Trad_peak_explicit_K']:.4f}</td><td>{final['Trad_peak_implicit_K']:.4f}</td></tr>",
             f"<tr><th>T_rad spatial wiggle [K]</th><td>{final['Trad_wiggle_explicit_K']:.4f}</td><td>{final['Trad_wiggle_implicit_K']:.4f}</td></tr></table>",
             f"<p>At the explicit radiation-temperature hotspot ({hotspot['x_from_shock_cm']:.1f} cm from the jump), first-stage transport changes T_rad by <strong>{hotspot['first_stage_transport_Trad_change_K']:+.3f} K</strong>, its explicit source by {hotspot['first_stage_explicit_source_Trad_change_K']:+.3f} K, and the final implicit solve by <strong>{hotspot['implicit_Trad_change_K']:+.3f} K</strong>. RK2 stage averaging means these increments cannot be summed directly.</p>",
             f"<p>The semi-analytic gas spike peaks at <strong>{analytic_gas_peak:.1f} K</strong> and has an approximately <strong>{analytic_spike_width:.2f} cm</strong> half-height width.</p>",
             "<p><img src='last_step_profiles.png' alt='Last step profiles'></p>",
             "<p><img src='stage_peak_history.png' alt='Peak history'></p>",
             "<p><a href='stage_metrics.csv'>Step metrics</a> · <a href='last_step_profile.csv'>Last-step profile</a> · <a href='summary.json'>JSON summary</a></p>",
             f"<p>Sampled cycles {cycles[0]}–{cycles[-1]}; {escape(str(args.stage_dir.resolve()))}</p></body></html>"]
    (args.output/"index.html").write_text("\n".join(lines)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
