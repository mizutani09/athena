#!/usr/bin/env python3
"""Separate a Mach-5 T_rad spatial wiggle from the reference residual bump."""
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
from diagnose_mach5_convergence import reference_at  # noqa: E402
from plots.plot_radiative_shock import (decode_names, jump_center, load_analytic_profile,
                                        merge_profile, variable_index)  # noqa: E402

RADIATION_C = 2.99792458e10


def input_number(path: Path, key: str) -> float:
    match = re.search(rf"^\s*{re.escape(key)}\s*=\s*([0-9.eE+-]+)",
                      path.read_text(), re.MULTILINE)
    if not match:
        raise ValueError(f"{key} missing in {path}")
    return float(match.group(1))


def load(path: Path):
    with h5py.File(path) as data:
        names = decode_names(data.attrs["VariableNames"])
        values = np.asarray(data["user_out_var"], dtype=float)
        rho = values[variable_index(names, ("rho",))]
        trad = values[variable_index(names, ("T_rad", "Trad"))]
        tgas = values[variable_index(names, ("T_gas", "Tgas"))]
        limiter = values[variable_index(names, ("lambda",))]
        xyz = np.asarray(data["x1v"], dtype=float)
        face = np.asarray(data["x1f"], dtype=float)
        x = np.broadcast_to(xyz[:, None, None, :], rho.shape).reshape(-1)
        width = np.broadcast_to(np.diff(face, axis=-1)[:, None, None, :],
                                rho.shape).reshape(-1)
        x, arrays = merge_profile(x, [rho.ravel(), trad.ravel(), tgas.ravel(),
                                      limiter.ravel(), width])
        return float(data.attrs["Time"]), x, arrays


def spatial_wiggle(x: np.ndarray, trad: np.ndarray) -> dict:
    """Largest fall after a post-shock local maximum, before the next rise."""
    mask = (x >= 0.0) & (x <= 300.0)
    xx, tt = x[mask], trad[mask]
    if len(xx) < 3:
        raise ValueError("need at least three post-shock cells within 300 cm")
    best = {"height_K": 0.0, "peak_x_cm": None, "trough_x_cm": None}
    for i in range(1, len(tt)-1):
        if not (tt[i] > tt[i-1] and tt[i] > tt[i+1]):
            continue
        j = i+1
        while j+1 < len(tt) and tt[j+1] < tt[j]:
            j += 1
        height = float(tt[i]-tt[j])
        if height > best["height_K"]:
            best = {"height_K": height, "peak_x_cm": float(xx[i]),
                    "trough_x_cm": float(xx[j])}
    best["largest_adjacent_drop_K"] = float(max(0.0, np.max(tt[:-1]-tt[1:])))
    return best


def analyse(label: str, profile: Path, input_file: Path, analytic: dict):
    time, raw_x, (rho, trad, tgas, limiter, width) = load(profile)
    reference_center = jump_center(analytic["x"], analytic["rho"])
    numerical_center = jump_center(raw_x, rho)
    x = raw_x-numerical_center
    reference_trad = reference_at(x+reference_center, analytic, "trad", reference_center)
    residual = trad-reference_trad
    near = (x >= -60.0) & (x <= 60.0)
    shock_zone = (x >= -100.0) & (x <= 100.0)
    peak = np.flatnonzero(near)[np.argmax(residual[near])]
    chi_r = input_number(input_file, "chi_R")
    kappa_p = input_number(input_file, "kappa_P")
    dx = float(np.median(width[shock_zone]))
    wiggle = spatial_wiggle(x, trad)
    row = {
        "label": label, "profile": str(profile.resolve()),
        "input": str(input_file.resolve()), "time_s": time,
        "shock_raw_x_cm": float(numerical_center),
        "shock_cell_width_cm": dx,
        "limiter_near_shock_min_max": [float(np.min(limiter[shock_zone])),
                                       float(np.max(limiter[shock_zone]))],
        "chi_R_cm_inv": chi_r, "kappa_P_cm_inv": kappa_p,
        "tau_R_per_shock_cell": chi_r*dx,
        "tau_P_per_shock_cell": kappa_p*dx,
        "absorption_time_s": 1.0/(RADIATION_C*kappa_p),
        "cell_diffusion_time_s": 3.0*chi_r*dx*dx/RADIATION_C,
        "gas_temperature_peak_near_shock_K": float(np.max(tgas[shock_zone])),
        "gas_minus_radiation_peak_near_shock_K": float(np.max(
            (tgas-trad)[shock_zone])),
        "residual_bump_near_shock_K": float(max(0.0, residual[peak])),
        "residual_bump_x_cm": float(x[peak]),
        "spatial_wiggle": wiggle,
        "reference_postshock_monotone": bool(np.min(np.diff(
            reference_trad[(x >= 0.0) & (x <= 300.0)])) >= -1e-8),
    }
    curves = {"x_from_shock_cm": x, "T_rad_K": trad,
              "T_gas_K": tgas, "T_rad_reference_K": reference_trad,
              "T_rad_residual_K": residual, "lambda": limiter,
              "cell_width_cm": width}
    return row, curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=HERE/
                        "reference/radiative_shock/mach5_semianalytic.csv")
    parser.add_argument("--case", action="append", required=True,
                        help="LABEL=PROFILE.athdf,INPUT.used")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analytic = load_analytic_profile(args.reference)
    args.output.mkdir(parents=True, exist_ok=True)
    results, curves = [], {}
    for item in args.case:
        label, sep, files = item.partition("=")
        if not sep or not re.fullmatch(r"[A-Za-z0-9_-]+", label):
            parser.error(f"invalid --case {item}")
        profile, sep, input_file = files.partition(",")
        if not sep or label in curves:
            parser.error(f"invalid or duplicate --case {item}")
        row, data = analyse(label, Path(profile), Path(input_file), analytic)
        results.append(row)
        curves[label] = data
        with (args.output/f"{label}_profile.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(data)
            writer.writerows(zip(*data.values()))
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False,
                             constrained_layout=True)
    axes[0].plot(analytic["x"]-jump_center(analytic["x"], analytic["rho"]),
                 analytic["trad"], "k--", label="semi-analytic")
    for row in results:
        label = row["label"]
        data = curves[label]
        axes[0].plot(data["x_from_shock_cm"], data["T_rad_K"], ".-", ms=3,
                     label=label)
        axes[1].plot(data["x_from_shock_cm"], data["T_rad_residual_K"],
                     ".-", ms=3, label=label)
    axes[0].set_ylabel("T_rad [K]")
    axes[0].set_xlabel("x - numerical shock [cm]")
    axes[1].set_ylabel("T_rad - semi-analytic [K]")
    axes[1].set_xlabel("x - numerical shock [cm]")
    axes[1].axhline(0, color="k", lw=0.7)
    axes[0].set_xlim(0, 300)
    local_values = [np.asarray(analytic["trad"])[
        (analytic["x"]-jump_center(analytic["x"], analytic["rho"]) >= 0)
        & (analytic["x"]-jump_center(analytic["x"], analytic["rho"]) <= 300)]]
    for curve in curves.values():
        local_values.append(curve["T_rad_K"][(curve["x_from_shock_cm"] >= 0)
                                              & (curve["x_from_shock_cm"] <= 300)])
    low = min(float(np.min(v)) for v in local_values if len(v))
    high = max(float(np.max(v)) for v in local_values if len(v))
    axes[0].set_ylim(low-0.08, high+0.08)
    axes[1].set_xlim(-300, 300)
    for ax in axes:
        ax.axvline(0, color="grey", ls=":")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(args.output/"trad_peak.png", dpi=180)
    plt.close(fig)
    report = {"reference": str(args.reference.resolve()),
              "definitions": {
                  "spatial_wiggle": "post-shock local T_rad maximum minus following minimum before next rise, x=0..300 cm",
                  "residual_bump": "maximum positive T_rad minus semi-analytic reference, x=-60..60 cm",
                  "shock_alignment": "each numerical density jump centered for reference comparison; local T_rad differences do not depend on this shift",
                  "tau_R": "chi_R times median shock-zone cell width",
              }, "cases": results}
    (args.output/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    lines = ["<!doctype html><html><head><meta charset='utf-8'><title>Mach 5 T_rad peak</title>",
             "<style>body{font:20px/1.5 system-ui;max-width:1250px;margin:30px auto;padding:0 20px}table{border-collapse:collapse}td,th{padding:8px 13px;border:1px solid #bbb;text-align:right}th:first-child,td:first-child{text-align:left}img{max-width:100%}</style></head><body>",
             "<h1>Mach 5 T_rad peak audit</h1><p><strong>Spatial wiggle</strong> and <strong>reference residual bump</strong> are different measurements. The semi-analytic profile is a comparison reference; the numerical shock positions are aligned only for plotting.</p>",
             "<p><img src='trad_peak.png' alt='Mach 5 radiation temperature and residual'></p>",
             "<table><tr><th>Case</th><th>dx [cm]</th><th>tau_R/cell</th><th>Spatial wiggle [K]</th><th>Residual bump [K]</th></tr>"]
    for row in results:
        lines.append("<tr><td><a href='{}.csv'>{}</a></td><td>{:.3f}</td><td>{:.2f}</td><td>{:.6g}</td><td>{:.6g}</td></tr>".format(
            escape(row["label"]+"_profile"), escape(row["label"]),
            row["shock_cell_width_cm"], row["tau_R_per_shock_cell"],
            row["spatial_wiggle"]["height_K"], row["residual_bump_near_shock_K"]))
    lines += ["</table><p><a href='summary.json'>Full metrics and source paths</a></p></body></html>"]
    (args.output/"index.html").write_text("\n".join(lines)+"\n")
    for row in results:
        print(row["label"], "wiggle", row["spatial_wiggle"]["height_K"],
              "residual bump", row["residual_bump_near_shock_K"])


if __name__ == "__main__":
    main()
