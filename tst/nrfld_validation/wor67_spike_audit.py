#!/usr/bin/env python3
"""Compare Mach-5 gas spikes with the point and finite-volume reference."""
from __future__ import annotations

import argparse
import csv
from html import escape
import json
from pathlib import Path
import re
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from diagnose_mach5_convergence import measure  # noqa: E402
from nrfld_validation.io import read_history  # noqa: E402
from plots.plot_radiative_shock import jump_center, load_analytic_profile  # noqa: E402
from radiative_shock_validation import stationarity  # noqa: E402
from shock_tube_validation import parse_solver_log, runtime_resources  # noqa: E402


def reference_state(path: Path) -> dict[str, np.ndarray | float]:
    data = np.genfromtxt(path, names=True, delimiter=",", encoding="utf-8")
    x = np.asarray(data["x_cm"], dtype=float)
    rho = np.asarray(data["rho_g_cm3"], dtype=float)
    vel = np.asarray(data["u_cm_s"], dtype=float)
    press = np.asarray(data["p_dyn_cm2"], dtype=float)
    tgas = np.asarray(data["Tgas_K"], dtype=float)
    if not np.all(np.diff(x) > 0):
        raise ValueError("analytic coordinate must be strictly increasing")
    gas_constant = np.median(press/(rho*tgas))
    if not np.allclose(press/(rho*tgas), gas_constant, rtol=1e-7):
        raise ValueError("analytic equation of state is inconsistent")
    return {"x": x, "rho": rho, "momentum": rho*vel,
            "gas_total_energy": press/(5.0/3.0-1.0)+0.5*rho*vel*vel,
            "tgas": tgas, "gas_constant": float(gas_constant)}


def cell_average(x: np.ndarray, y: np.ndarray, left: float, right: float) -> float:
    if left < x[0] or right > x[-1]:
        raise ValueError("reference does not cover a numerical cell")
    knots = x[(x > left) & (x < right)]
    grid = np.r_[left, knots, right]
    return float(np.trapezoid(np.interp(grid, x, y), grid)/(right-left))


def projected_temperature(reference: dict, centers: np.ndarray,
                          widths: np.ndarray) -> np.ndarray:
    """Average reference conserved gas state, then recover ideal-gas T."""
    x = reference["x"]
    result = np.empty(len(centers))
    for i, (center, width) in enumerate(zip(centers, widths)):
        lo, hi = center-width/2, center+width/2
        rho = cell_average(x, reference["rho"], lo, hi)
        mom = cell_average(x, reference["momentum"], lo, hi)
        energy = cell_average(x, reference["gas_total_energy"], lo, hi)
        press = (5.0/3.0-1.0)*(energy-0.5*mom*mom/rho)
        result[i] = press/(rho*reference["gas_constant"])
    return result


def input_value(path: Path, key: str) -> str:
    match = re.search(rf"^\s*{re.escape(key)}\s*=\s*([^\s#]+)",
                      path.read_text(), re.MULTILINE)
    if not match:
        raise ValueError(f"missing {key} in {path}")
    return match[1]


def stage_metrics(directory: Path, shock_raw: float) -> dict | None:
    files = sorted((directory/"output").glob("fld_stages.r*.b*.csv"))
    if not files:
        return None
    with files[0].open(newline="") as handle:
        sampled_cycles = sorted({int(row["cycle"]) for row in csv.DictReader(handle)})
    selected = sampled_cycles[-2:]
    records: dict[int, dict[str, dict[float, dict]]] = {
        cycle: {"before_implicit": {}, "after_implicit": {}} for cycle in selected}
    for path in files:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                cycle = int(row["cycle"])
                if cycle not in records or row["phase"] not in records[cycle]:
                    continue
                x = float(row["x"])
                if abs(x-shock_raw) <= 100.0:
                    if x in records[cycle][row["phase"]]:
                        raise ValueError(f"duplicate stage sample at {x}: {path}")
                    records[cycle][row["phase"]][x] = row

    def summarize(cycle: int) -> dict:
        pre = records[cycle]["before_implicit"]
        post = records[cycle]["after_implicit"]
        if not pre or set(pre) != set(post):
            raise ValueError(f"missing or mismatched stage samples at cycle {cycle}")
        points = sorted(pre)
        def peak(rows: dict, field: str) -> float:
            return max(float(rows[x][field]) for x in points)
        dt = float(pre[points[0]]["dt"])
        hottest_x = max(points, key=lambda x: float(pre[x]["Tgas_K"]))
        hottest_gas_change = (float(post[hottest_x]["egas"])
                              - float(pre[hottest_x]["egas"]))
        hottest_radiation_change = (float(post[hottest_x]["Er"])
                                    - float(pre[hottest_x]["Er"]))
        budget = [((float(post[x]["egas"])-float(pre[x]["egas"]))
                  + (float(post[x]["Er"])-float(pre[x]["Er"])))/dt
                  - float(post[x]["S_diff_implicit"]) for x in points]
        diffusion = max(abs(float(post[x]["S_diff_implicit"])) for x in points)
        transverse_spread = max(
            (float(post[x]["Er_transverse_max"])
             - float(post[x]["Er_transverse_min"]))
            / max(abs(float(post[x]["Er"])), 1e-300) for x in points)
        return {"cycle": cycle, "dt_s": dt, "sampled_x_cells": len(points),
                "Tgas_peak_after_explicit_K": peak(pre, "Tgas_K"),
                "Tgas_peak_after_implicit_K": peak(post, "Tgas_K"),
                "Trad_peak_after_explicit_K": peak(pre, "Trad_K"),
                "Trad_peak_after_implicit_K": peak(post, "Trad_K"),
                "largest_same_cell_Tgas_drop_K": max(float(pre[x]["Tgas_K"])
                    - float(post[x]["Tgas_K"]) for x in points),
                "hottest_pre_implicit_cell": {
                    "x_cm": hottest_x,
                    "Tgas_after_explicit_K": float(pre[hottest_x]["Tgas_K"]),
                    "Tgas_after_implicit_K": float(post[hottest_x]["Tgas_K"]),
                    "gas_energy_density_change": hottest_gas_change,
                    "radiation_energy_density_change": hottest_radiation_change,
                    "total_energy_density_change": (hottest_gas_change
                                                     + hottest_radiation_change),
                    "dt_times_implicit_diffusion_source":
                        dt*float(post[hottest_x]["S_diff_implicit"])},
                "implicit_energy_budget_relative_error": max(map(abs, budget))
                    / max(diffusion, 1e-300),
                "max_relative_transverse_Er_spread": transverse_spread}

    result = summarize(selected[-1])
    result["previous_step"] = summarize(selected[-2]) if len(selected) > 1 else None
    result["stage_csv_files"] = len(files)
    return result


def analyse_case(label: str, directory: Path, analytic: dict, reference: dict,
                 output: Path) -> tuple[dict, dict]:
    input_file = directory/"input.used"
    if not np.isclose(float(input_value(input_file, "gamma")), 5.0/3.0,
                      rtol=1e-10):
        raise ValueError(f"{label}: FV reference requires gamma=5/3")
    files = sorted((directory/"output").glob("*uov_x*.athdf"))
    if len(files) < 2:
        raise ValueError(f"need at least two profiles in {directory}")
    final, data = measure(files[-1], analytic, 60.0, (-3000.0, 1000.0))
    previous, old_data = measure(files[-2], analytic, 60.0, (-3000.0, 1000.0))
    if abs(final["time_s"]-0.04) > 1e-8:
        raise ValueError(f"{label} did not reach t=0.04: {final['time_s']}")
    x = data["x_from_shock_cm"]
    width = data["cell_width_cm"]
    near = np.abs(x) <= 60.0
    # The coarsest comparison has no cell center in the first 40 cm.
    spike = np.abs(x) <= max(40.0, float(np.max(width[near])))
    if not np.any(near) or not np.any(spike):
        raise ValueError(f"{label}: no shock-zone samples")
    center = jump_center(analytic["x"], analytic["rho"])
    fv = projected_temperature(reference, x[near]+center, width[near])
    point_peak = float(np.max(analytic["tgas"][np.abs(analytic["x"]-center) <= 60]))
    downstream = float(analytic["tgas"][-1])
    half_height = 0.5*(point_peak+downstream)
    spike_width = float(np.ptp(analytic["x"][analytic["tgas"] >= half_height]))
    numerical_peak = float(np.max(data["tgas"][near]))
    fv_peak = float(np.max(fv))
    phase_peaks = [float(np.max(projected_temperature(
        reference, x[near]+center+shift, width[near])))
        for shift in np.linspace(-final["shock_dx_cm"]/2,
                                 final["shock_dx_cm"]/2, 41)]
    near_x = x[near]
    numerical_peak_x = float(near_x[np.argmax(data["tgas"][near])])
    fv_peak_x = float(near_x[np.argmax(fv)])
    excess = point_peak-downstream
    fv_full = np.full_like(x, np.nan)
    fv_full[near] = fv
    spike_fv = spike & np.isfinite(fv_full)
    spike_l1 = float(np.sum(abs(data["tgas"][spike_fv]-fv_full[spike_fv])
                            * width[spike_fv])
                     / np.sum(np.maximum(fv_full[spike_fv]-downstream, 0)
                              * width[spike_fv]))
    with h5py.File(files[-1]) as hdf:
        levels = sorted(set(int(i) for i in hdf["Levels"][:]))
        meshblocks = len(hdf["Levels"])
        cycles = int(hdf.attrs["NumCycles"])
    layout = input_value(input_file, "refinement")
    solver = parse_solver_log(directory/"run.log", cycles)
    if layout != "none" and solver["nr_status_count"] > cycles:
        solver = parse_solver_log(directory/"run.log", solver["nr_status_count"])
        solver["global_cycles"] = cycles
    history = read_history(directory/"output")
    late_steps = (history["time"] >= .02) & (history["time"] < .0395)
    median_late_dt = float(np.median(history["dt"][late_steps])) if np.any(
        late_steps) else None
    late_peak_window = (history["time"] >= .03) & (history["time"] <= .04)
    if not np.any(late_peak_window):
        raise ValueError(f"{label}: missing late peak history")
    late_peak_min = float(np.min(history["Tgas_max"][late_peak_window]))
    late_peak_max = float(np.max(history["Tgas_max"][late_peak_window]))
    absorption_time = 1.0/(2.99792458e10*float(input_value(input_file, "kappa_P")))
    previous_history_index = int(np.argmin(abs(history["time"]-previous["time_s"])))
    if abs(history["time"][previous_history_index]-previous["time_s"]) > 1e-5:
        raise ValueError(f"{label}: HST does not cover the previous HDF snapshot")
    if not solver["good"] or solver["relaxed_accepted"]:
        raise ValueError(f"{label}: solver did not converge strictly: {solver}")
    rss = runtime_resources(directory/"resources.txt")
    rank_memory_file = directory/"rank_memory_snapshot.json"
    rank_memory = (json.loads(rank_memory_file.read_text())
                   if rank_memory_file.is_file() else None)
    elapsed = None
    resource_file = directory/"resources.txt"
    if resource_file.is_file():
        match = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([^\s]+)",
                          resource_file.read_text())
        if match:
            parts = [float(v) for v in match[1].split(":")]
            elapsed = sum(v*60**i for i, v in enumerate(reversed(parts)))
    change = stationarity(data, old_data)
    row = {"label": label, "directory": str(directory.resolve()),
           "input": str(input_file.resolve()),
           "final_profile": str(files[-1].resolve()),
           "time_s": final["time_s"], "previous_time_s": previous["time_s"],
           "nx1": int(input_value(input_file, "nx1")),
           "cfl": float(input_value(input_file, "cfl_number")),
           "layout": layout,
           "shock_dx_cm": final["shock_dx_cm"],
           "shock_raw_cm": final["shock_raw_cm"],
           "shock_position_error_over_domain": abs(final["shock_raw_cm"]-center)/6000,
           "levels": levels, "meshblocks_final": meshblocks,
           "profile_cells_within_20cm": int(np.count_nonzero(abs(x) <= 20.0)),
           "cycles": cycles,
           "median_late_history_dt_s": median_late_dt,
           "absorption_time_s": absorption_time,
           "late_dt_over_absorption_time": (median_late_dt/absorption_time
                                            if median_late_dt else None),
           "gas_peak_point_reference_K": point_peak,
           "gas_peak_fv_projected_reference_K": fv_peak,
           "gas_peak_fv_projection_phase_range_K": [min(phase_peaks),
                                                    max(phase_peaks)],
           "gas_peak_numerical_K": numerical_peak,
           "gas_peak_numerical_x_from_shock_cm": numerical_peak_x,
           "gas_peak_fv_reference_x_from_shock_cm": fv_peak_x,
           "gas_peak_relative_error": abs(numerical_peak-point_peak)/point_peak,
           "spike_amplitude_recovery": (numerical_peak-downstream)/excess,
           "fv_projected_peak_relative_error": abs(numerical_peak-fv_peak)/fv_peak,
           "fv_spike_excess_L1": spike_l1,
           "analytic_spike_half_height_width_cm": spike_width,
           "cells_in_analytic_half_height_width": spike_width/final["shock_dx_cm"],
           "aligned_L1_percent": {k: final["agreement"][k]["global_relative_L1_percent"]
                                  for k in ("rho", "tgas", "trad")},
           "aligned_change_since_previous": change,
           "max_aligned_change": max(change.values()),
           "gas_peak_change_since_previous_output_K": float(
               history["Tgas_max"][-1]-history["Tgas_max"][previous_history_index]),
           "late_history_peak_range_K": [late_peak_min, late_peak_max],
           "late_history_peak_span_K": late_peak_max-late_peak_min,
           "history_last_flux_spread": {k: float(history[k][-1])
                                        for k in ("res_mdot", "res_mom", "res_etot")},
           "history_initial_final_all_energy": [float(history["all-E"][0]),
                                                float(history["all-E"][-1])],
           "solver": solver, "launcher_max_rss_kib": rss,
           "sampled_sum_rank_vmhwm_kib": (rank_memory["sum_rank_vmhwm_kib"]
                                            if rank_memory else None),
           "wall_seconds": elapsed,
           "stage": stage_metrics(directory, final["shock_raw_cm"])}
    row["accuracy_pass"] = (all(v <= 2.0 for v in row["aligned_L1_percent"].values())
                            and row["shock_position_error_over_domain"] <= .05
                            and row["max_aligned_change"] <= .01
                            and row["gas_peak_relative_error"] <= .10
                            and row["spike_amplitude_recovery"] >= .50)
    with (output/f"{label}_near_shock.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("x_from_density_jump_cm", "cell_width_cm", "Tgas_numerical_K",
                         "Tgas_reference_point_K", "Tgas_reference_FV_projection_K"))
        writer.writerows(zip(x[near], width[near], data["tgas"][near],
                             data["tgas_reference"][near], fv))
    curves = {"x": x[near], "tgas": data["tgas"][near], "point":
              data["tgas_reference"][near], "fv": fv,
              "history_time": history["time"], "history_peak": history["Tgas_max"]}
    return row, curves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=HERE/
                        "reference/radiative_shock/mach5_semianalytic.csv")
    parser.add_argument("--case", action="append", required=True, help="LABEL=CASE_DIR")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    analytic = load_analytic_profile(args.reference)
    reference = reference_state(args.reference)
    cases, curves = [], {}
    for spec in args.case:
        label, sep, path = spec.partition("=")
        if not sep or not re.fullmatch(r"[A-Za-z0-9_-]+", label) or label in curves:
            parser.error(f"invalid case: {spec}")
        row, curve = analyse_case(label, Path(path), analytic, reference, args.output)
        cases.append(row)
        curves[label] = curve
        print(label, row["gas_peak_numerical_K"], row["spike_amplitude_recovery"],
              "PASS" if row["accuracy_pass"] else "accuracy_not_met", flush=True)
    cases.sort(key=lambda row: (-row["shock_dx_cm"], -row["cfl"]))
    center = jump_center(analytic["x"], analytic["rho"])
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True,
                             sharey=True)
    mask = np.abs(analytic["x"]-center) <= 40
    for ax in axes:
        ax.plot(analytic["x"][mask]-center, analytic["tgas"][mask], "k-",
                lw=2, label="semi-analytic point profile")
    projected_widths = set()
    colors = plt.cm.viridis(np.linspace(.05, .9, len(cases)))
    for row, color in zip(cases, colors):
        data = curves[row["label"]]
        axes[0].plot(data["x"], data["tgas"], "o-", ms=3, lw=1,
                     color=color, label=row["label"])
        width_key = round(row["shock_dx_cm"], 8)
        if row["layout"] == "none" and width_key not in projected_widths:
            projected_widths.add(width_key)
            axes[1].plot(data["x"], data["fv"], "o-", ms=3, lw=1,
                         color=color, label=f"dx={row['shock_dx_cm']:.2f} cm")
    for ax, title in zip(axes, ("Numerical profiles", "Finite-volume reference")):
        ax.set(xlim=(-40, 40), xlabel="x - density-jump center [cm]",
               title=title)
        ax.grid(alpha=.3)
        ax.legend(fontsize=9, ncol=2)
    axes[0].set_ylabel("T_gas [K]")
    fig.savefig(args.output/"gas_spike_profiles.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    label_index_by_dx: dict[float, int] = {}
    for row in cases:
        dx = row["shock_dx_cm"]
        key = round(dx, 8)
        stagger = label_index_by_dx.get(key, 0)
        label_index_by_dx[key] = stagger+1
        label_offset = (5, (8, -22, 38)[stagger % 3])
        label_align = "left"
        if abs(dx - 6000/1024) < 1e-6:
            label_offset = (-8, (12, -24, 42)[stagger % 3])
            label_align = "right"
        if row["label"] == "N2048":
            label_offset = (-7, -16)
            label_align = "right"
        axes[0].plot(dx, row["gas_peak_numerical_K"], "o", color="tab:blue")
        axes[0].annotate(row["label"], (dx, row["gas_peak_numerical_K"]),
                         xytext=label_offset, textcoords="offset points",
                         ha=label_align, fontsize=8)
        axes[0].plot(dx, row["gas_peak_fv_projected_reference_K"], "s",
                     color="tab:orange")
        lo, hi = row["gas_peak_fv_projection_phase_range_K"]
        axes[0].vlines(dx, lo, hi, color="tab:orange", lw=2, alpha=.65)
        axes[1].plot(dx, 100*row["spike_amplitude_recovery"], "o",
                     color="tab:blue")
        axes[1].annotate(row["label"], (dx, 100*row["spike_amplitude_recovery"]),
                         xytext=label_offset, textcoords="offset points",
                         ha=label_align, fontsize=8)
    axes[0].axhline(cases[0]["gas_peak_point_reference_K"], color="black",
                    ls="--", label="semi-analytic point peak")
    axes[0].set(ylabel="T_gas peak [K]", xlabel="shock cell width [cm]")
    axes[1].axhline(50, color="black", ls="--", label="required recovery")
    axes[1].set(ylabel="spike amplitude recovery [%]", xlabel="shock cell width [cm]")
    for axis in axes:
        axis.set_xscale("log")
        axis.invert_xaxis()
        axis.grid(alpha=.3)
        axis.legend(fontsize=9)
    fig.savefig(args.output/"convergence.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for row in cases:
        data = curves[row["label"]]
        mask = data["history_time"] >= .03
        index = np.flatnonzero(mask)[::max(1, int(np.count_nonzero(mask)/500))]
        ax.plot(data["history_time"][index], data["history_peak"][index],
                label=row["label"])
    point_peak = cases[0]["gas_peak_point_reference_K"]
    downstream = float(analytic["tgas"][-1])
    ax.axhline(.9*point_peak, color="tab:red", ls="--",
               label="10% point-peak error limit")
    ax.axhline(.5*(point_peak+downstream), color="tab:orange", ls=":",
               label="50% spike-recovery limit")
    ax.set(xlabel="time [s]", ylabel="post-implicit T_gas max [K]",
           xlim=(.03, .04))
    ax.grid(alpha=.3)
    ax.legend(ncol=2, fontsize=9)
    fig.savefig(args.output/"late_peak_history.png", dpi=180)
    plt.close(fig)
    by_label = {row["label"]: row for row in cases}
    comparisons = []
    for kind, before_label, after_label in (
        ("time step at N512", "N512", "N512_dt_half"),
        ("spatial, matched dt near 2.2 us", "N512_dt_half", "N1024"),
        ("time step at N1024", "N1024", "N1024_dt_half"),
        ("spatial, matched dt near 1.1 us", "N1024_dt_half", "N2048")):
        if before_label not in by_label or after_label not in by_label:
            continue
        before, after = by_label[before_label], by_label[after_label]
        comparisons.append({
            "kind": kind, "before": before_label, "after": after_label,
            "peak_change_K": (after["gas_peak_numerical_K"]
                              - before["gas_peak_numerical_K"]),
            "recovery_change_percentage_points": 100*(
                after["spike_amplitude_recovery"]
                - before["spike_amplitude_recovery"]),
            "dt_ratio_after_over_before": (after["median_late_history_dt_s"]
                                           / before["median_late_history_dt_s"]),
            "dx_ratio_after_over_before": (after["shock_dx_cm"]
                                           / before["shock_dx_cm"])})
    summary = {"reference": str(args.reference.resolve()), "cases": cases,
               "comparisons": comparisons,
               "definitions": {"point_peak": "maximum analytic Tgas within 60 cm of density jump",
                               "FV_projection": "average analytic rho, rho*u and total gas energy over each aligned numerical cell, then recover ideal-gas T; diagnostic comparison, not a strict convergence ceiling",
                               "FV_phase_range": "minimum and maximum projected reference peak when density-jump registration is perturbed by plus/minus half one shock cell; sensitivity band only, never optimized for the accuracy gate",
                               "spike_recovery": "(numerical peak - analytic downstream T)/(analytic point peak - analytic downstream T)",
                               "aligned_L1": "density-jump registration only, over -3000..1000 cm",
                               "history_flux_spread": "within-block relative spread of physical mass/momentum/energy flux; open-domain diagnostic, not exact discrete conservation"}}
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    finest_uniform = min((row for row in cases if row["layout"] == "none"),
                         key=lambda row: (row["shock_dx_cm"], -row["cfl"]))
    html = ["<!doctype html><html lang='en'><head><meta charset='utf-8'>",
            "<title>WOR-67 Mach 5 gas spike</title>",
            "<style>body{font:20px/1.5 system-ui;max-width:1300px;margin:32px auto;padding:0 20px}h1,h2{line-height:1.2}table{border-collapse:collapse;width:100%}th,td{border:1px solid #bbb;padding:8px;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#eef3fa}img{max-width:100%}.scroll{overflow-x:auto}strong{color:#17537d}.pass{color:#14743f;font-weight:700}.fail{color:#a13b2b;font-weight:700}</style></head><body>",
            "<h1>WOR-67: Mach 5 gas-temperature spike</h1>",
            "<p><strong>Finest completed uniform case ({label}, dx={dx:.3f} cm):</strong> {peak:.2f} K peak, {error:.2%} point-peak error, {recovery:.2%} amplitude recovery; <span class='{class_name}'>{status}</span>.</p>".format(
                label=escape(finest_uniform["label"]),
                dx=finest_uniform["shock_dx_cm"],
                peak=finest_uniform["gas_peak_numerical_K"],
                error=finest_uniform["gas_peak_relative_error"],
                recovery=finest_uniform["spike_amplitude_recovery"],
                class_name="pass" if finest_uniform["accuracy_pass"] else "fail",
                status="PASS" if finest_uniform["accuracy_pass"] else "accuracy_not_met"),
            "<p><strong>Reference point peak:</strong> {:.2f} K. The finite-volume projection averages the analytic conserved gas state over each numerical cell before converting to temperature; it is a diagnostic comparison, not a strict accuracy ceiling.</p>".format(cases[0]["gas_peak_point_reference_K"]),
            "<p>The orange vertical bars vary the density-jump registration by ±half a shock cell to show sampling sensitivity. The reported accuracy gate uses the original density-jump registration; it does not select the best shift.</p>",
            "<p><strong>Physical gate:</strong> aligned density, gas and radiation L1 ≤2%; raw shock-position error ≤5% of domain width; aligned change since previous output ≤1%; point-peak error ≤10%; spike-amplitude recovery ≥50%. All cases also require strict Newton/FGMRES convergence.</p>",
            "<p><img src='gas_spike_profiles.png' alt='Gas spike profiles'></p>",
            "<p><img src='convergence.png' alt='Peak convergence'></p>",
            "<p><img src='late_peak_history.png' alt='Late gas-peak history'></p>",
            "<div class='scroll'><table><tr><th>Case</th><th>dx [cm]</th><th>CFL</th><th>Late dt [µs]</th><th>Numerical peak [K]</th><th>Peak span, t=.03–.04 [K]</th><th>FV reference peak [K]</th><th>Peak x from jump [cm]</th><th>Point peak error</th><th>Recovery</th><th>Aligned Tgas L1</th><th>Stage explicit → implicit [K]</th><th>Gate</th></tr>"]
    for row in cases:
        stage = row["stage"]
        st = ("{:.2f} → {:.2f}".format(stage["Tgas_peak_after_explicit_K"],
                                        stage["Tgas_peak_after_implicit_K"])
              if stage else "—")
        html.append("<tr><td><a href='{}_near_shock.csv'>{}</a></td><td>{:.3f}</td><td>{:.3f}</td><td>{:.3f}</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td><td>{:+.2f}</td><td>{:.2%}</td><td>{:.2%}</td><td>{:.3f}%</td><td>{}</td><td>{}</td></tr>".format(
            escape(row["label"]), escape(row["label"]), row["shock_dx_cm"],
            row["cfl"], 1e6*row["median_late_history_dt_s"],
            row["gas_peak_numerical_K"],
            row["late_history_peak_span_K"],
            row["gas_peak_fv_projected_reference_K"],
            row["gas_peak_numerical_x_from_shock_cm"],
            row["gas_peak_relative_error"], row["spike_amplitude_recovery"],
            row["aligned_L1_percent"]["tgas"], st,
            "<span class='pass'>PASS</span>" if row["accuracy_pass"]
            else "<span class='fail'>accuracy_not_met</span>"))
    html.extend(["</table></div>"])
    if comparisons:
        html.extend(["<h2>Time-step and spatial comparisons</h2>",
                     "<p>Each row changes one intended factor. The measured late dt ratio and shock-cell width ratio show how closely the paired control matches.</p>",
                     "<div class='scroll'><table><tr><th>Comparison</th><th>Before → after</th><th>Peak change [K]</th><th>Recovery change [percentage points]</th><th>dt ratio</th><th>dx ratio</th></tr>"])
        for item in comparisons:
            html.append("<tr><td>{}</td><td>{} → {}</td><td>{:+.2f}</td><td>{:+.2f}</td><td>{:.3f}</td><td>{:.3f}</td></tr>".format(
                escape(item["kind"]), escape(item["before"]),
                escape(item["after"]), item["peak_change_K"],
                item["recovery_change_percentage_points"],
                item["dt_ratio_after_over_before"],
                item["dx_ratio_after_over_before"]))
        html.append("</table></div>")
    html.extend([
                 "<h2>Other accuracy and runtime checks</h2>",
                 "<div class='scroll'><table><tr><th>Case</th><th>Density L1</th><th>Gas T L1</th><th>Radiation T L1</th><th>Raw shock error / domain</th><th>Previous-output change</th><th>Strict solver</th><th>Wall time [min]</th><th>Sampled sum of rank high-water RSS [MiB]</th></tr>"])
    for row in cases:
        html.append("<tr><td>{}</td><td>{:.3f}%</td><td>{:.3f}%</td><td>{:.3f}%</td><td>{:.2%}</td><td>{:.3%}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            escape(row["label"]), row["aligned_L1_percent"]["rho"],
            row["aligned_L1_percent"]["tgas"],
            row["aligned_L1_percent"]["trad"],
            row["shock_position_error_over_domain"],
            row["max_aligned_change"],
            "PASS" if row["solver"]["good"] else "FAIL",
            f"{row['wall_seconds']/60:.1f}" if row["wall_seconds"] is not None else "—",
            f"{row['sampled_sum_rank_vmhwm_kib']/1024:.1f}"
            if row["sampled_sum_rank_vmhwm_kib"] is not None else "—"))
    html.extend(["</table></div>", "<h2>Stage timing</h2>"])
    for row in cases:
        stage = row["stage"]
        if not stage:
            continue
        previous = stage["previous_step"]
        html.append("<p><strong>{}</strong>: final shortened step dt={:.3g} s, "
                    "T_gas peak {:.2f} → {:.2f} K (explicit → implicit).{}</p>".format(
                        escape(row["label"]), stage["dt_s"],
                        stage["Tgas_peak_after_explicit_K"],
                        stage["Tgas_peak_after_implicit_K"],
                        (" Previous full step dt={:.3g} s, peak {:.2f} → {:.2f} K."
                         .format(previous["dt_s"],
                                 previous["Tgas_peak_after_explicit_K"],
                                 previous["Tgas_peak_after_implicit_K"])
                         if previous else "")))
        if previous:
            cell = previous["hottest_pre_implicit_cell"]
            html.append("<p>At the previous full step's hottest cell in {}, "
                        "Δ(E_gas+E_rad)={:.6g} and dt·S_diff={:.6g} "
                        "(code energy-density units). The reported implicit "
                        "energy-budget residual is {:.2g} relative to the "
                        "largest diffusion source.</p>".format(
                            escape(row["label"]),
                            cell["total_energy_density_change"],
                            cell["dt_times_implicit_diffusion_source"],
                            previous["implicit_energy_budget_relative_error"]))
    html.extend([
                 "<p>Full aligned L1, raw shock position, peak change since the previous output, solver convergence, energy-flux spread, runtime and exact paths are in <a href='summary.json'>summary.json</a>. The sampled rank high-water RSS is a sum of per-process values, so shared pages may be counted more than once; the separate /usr/bin/time RSS is for the MPI launcher. The shock domain has fixed inflow/outflow states, so total energy is not expected to remain constant.</p>",
                 "</body></html>"])
    (args.output/"index.html").write_text("\n".join(html)+"\n")


if __name__ == "__main__":
    main()
