#!/usr/bin/env python3
"""Audit how a segmented solar run transports prescribed inflow entropy.

Entropy is evaluated from the same EOS table at the first *interior* cell
using paired full-volume dumps. Upflow/downflow means are weighted by the
absolute cell-centered vertical mass flux. Exact boundary energy fluxes come
from the run's HST/report; this diagnostic does not reconstruct face fluxes.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE / "scripts"))
from build_solar_initial_profile import AthenaEosTable  # noqa: E402
from plot_simple_convection_surface3d import read_volume  # noqa: E402
from solar_convection_report import input_values  # noqa: E402


def snapshot(prim_path: Path, uov_path: Path, eos: AthenaEosTable,
             rho_unit: float, velocity_unit: float,
             entropy_input: float) -> dict | None:
    prim, _, time = read_volume(prim_path, "prim")
    uov, _, other_time = read_volume(uov_path, "user_out_var")
    if not np.isclose(time, other_time, rtol=0, atol=1e-6):
        raise ValueError(f"Unpaired full-volume times: {prim_path}, {uov_path}")
    rho = prim[0, 0] * rho_unit
    vz = prim[4, 0] * velocity_unit
    internal = uov[0, 0]
    if np.any(rho <= 0) or np.any(internal <= 0):
        raise ValueError(f"Nonpositive EOS state: {prim_path}")
    logu = np.log10(internal / rho)
    logrho = np.log10(rho)
    entropy = 10.0 ** eos.interp[4](np.stack((logu, logrho), axis=-1))
    mass_flux = rho * vz
    up = mass_flux > 0
    down = mass_flux < 0
    if not np.any(up) or not np.any(down):
        if time == 0.0:  # The unperturbed initial profile has zero velocity.
            return None
        raise ValueError(f"Both flow directions needed at first interior cell: {prim_path}")
    up_mass = mass_flux[up]
    down_mass = -mass_flux[down]
    s_up = float(np.dot(entropy[up], up_mass) / np.sum(up_mass))
    s_down = float(np.dot(entropy[down], down_mass) / np.sum(down_mass))
    return {"time_code": float(time), "input_entropy_cgs": entropy_input,
            "upflow_entropy_cgs": s_up, "downflow_entropy_cgs": s_down,
            "entropy_contrast_cgs": s_up - s_down,
            "area_mean_entropy_cgs": float(np.mean(entropy)),
            "area_mean_temperature_K": float(np.mean(uov[2, 0])),
            "upward_mass_flux_cgs": float(np.mean(np.where(up, mass_flux, 0))),
            "downward_mass_flux_cgs": float(np.mean(np.where(down, mass_flux, 0)))}


def mean_window(rows: list[dict], bounds: list[float]) -> dict:
    subset = [row for row in rows if bounds[0] <= row["time_code"] <= bounds[1]]
    if len(subset) < 2:
        raise ValueError(f"Fewer than two full snapshots in {bounds}")
    keys = [key for key in subset[0] if key != "time_code"]
    return {"start_code": bounds[0], "end_code": bounds[1],
            "snapshots": len(subset),
            **{key: float(np.mean([row[key] for row in subset])) for key in keys}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback-state", type=Path, required=True)
    parser.add_argument("--eos-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--early-window", type=float, nargs=2)
    parser.add_argument("--late-window", type=float, nargs=2)
    args = parser.parse_args()
    state_path = args.feedback_state.resolve()
    state = json.loads(state_path.read_text())
    runs = [(state_path.parent / row["case"], row) for row in state["completed"]]
    if not runs:
        parser.error("No completed segments")
    output = (args.output_dir or state_path.parent / "entropy_transport").resolve()
    output.mkdir(parents=True, exist_ok=True)
    eos = AthenaEosTable(args.eos_table.resolve())
    if eos.nvar < 5:
        parser.error("The EOS table has no entropy field")
    units = input_values(runs[0][0] / "athinput.generated")
    velocity_unit = np.sqrt(units["egas_unit"] / units["rho_unit"])
    records = []
    for run, state_row in runs:
        uov = {path.name.split(".uov_full.")[-1]: path
               for path in run.glob("*.uov_full.*.athdf")}
        for prim in sorted(run.glob("*.prim_full.*.athdf")):
            suffix = prim.name.split(".prim_full.")[-1]
            if suffix not in uov:
                raise ValueError(f"Missing paired user-output volume: {prim}")
            item = snapshot(prim, uov[suffix], eos,
                            units["rho_unit"], velocity_unit,
                            state_row["entropy_cgs"])
            if item is not None:
                records.append(item)
    records.sort(key=lambda row: row["time_code"])
    if len(records) < 3 or any(b["time_code"] <= a["time_code"]
                               for a, b in zip(records, records[1:])):
        raise ValueError("Need three or more strictly time-ordered full snapshots")
    final = records[-1]["time_code"]
    early = list(args.early_window or (0.2 * final, 0.3 * final))
    late = list(args.late_window or (0.9 * final, final))
    if not (early[0] < early[1] <= late[0] < late[1]):
        parser.error("Expected nonoverlapping, increasing early and late windows")
    first, last = mean_window(records, early), mean_window(records, late)
    difference = {key: last[key] - first[key] for key in first
                  if key in last and key not in {"start_code", "end_code", "snapshots"}}
    report = {"method": "first interior cell; EOS-table entropy; mass-flux-weighted up/down means",
              "feedback_state": str(state_path),
              "eos_table": str(args.eos_table.resolve()),
              "early_window": first, "late_window": last,
              "late_minus_early": difference, "records": records}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    # Keep the startup transient in JSON, but let the figure resolve the
    # much smaller slow-controller changes after the first window approaches.
    plotted = [row for row in records if row["time_code"] >= max(0.0, early[0] - 500)]
    time = [row["time_code"] for row in plotted]
    base = state["config"]["initial_entropy_cgs"]
    plt.rcParams.update({"font.size": 16})
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True,
                             constrained_layout=True)
    for key, label in (("input_entropy_cgs", "prescribed inflow"),
                       ("upflow_entropy_cgs", "bottom upflow"),
                       ("downflow_entropy_cgs", "bottom downflow")):
        axes[0].plot(time, [(row[key] - base) / 1e6 for row in plotted],
                     label=label)
    axes[0].set_ylabel("Entropy relative to initial\n[10⁶ erg g⁻¹ K⁻¹]")
    axes[0].legend()
    axes[1].plot(time, [row["entropy_contrast_cgs"] / 1e6
                        for row in plotted], color="tab:purple")
    axes[1].set_ylabel("Upflow − downflow entropy\n[10⁶ erg g⁻¹ K⁻¹]")
    axes[2].plot(time, [row["area_mean_temperature_K"] for row in plotted],
                 color="tab:red")
    axes[2].set(ylabel="Bottom-cell mean T [K]", xlabel="Code time")
    for axis in axes:
        axis.grid(alpha=0.3)
        for bounds in (early, late):
            axis.axvspan(*bounds, color="gray", alpha=0.08)
    fig.savefig(output / "entropy_transport.png", dpi=150)
    plt.close(fig)

    rows = "".join(
        f"<tr><td>{html.escape(label)}</td><td>{first[key]:.6g}</td>"
        f"<td>{last[key]:.6g}</td><td>{difference[key]:+.6g}</td></tr>"
        for label, key in (("Input entropy", "input_entropy_cgs"),
                           ("Upflow entropy", "upflow_entropy_cgs"),
                           ("Downflow entropy", "downflow_entropy_cgs"),
                           ("Up−down contrast", "entropy_contrast_cgs"),
                           ("Bottom-cell mean T [K]", "area_mean_temperature_K")))
    page = ("<!doctype html><html lang='en'><meta charset='utf-8'>"
            "<title>Solar bottom entropy transport</title>"
            "<style>body{font:21px/1.5 system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem}"
            "img{max-width:100%}table{border-collapse:collapse}"
            "th,td{padding:.5rem .8rem;border-bottom:1px solid #ccc;text-align:right}"
            "th:first-child,td:first-child{text-align:left}</style>"
            "<h1>Bottom entropy transport</h1>"
            "<p><strong>Cell-centered diagnostic:</strong> entropy uses the same EOS "
            "table in the first interior cell and vertical-mass-flux weighting. "
            "It is not the exact boundary face energy flux.</p>"
            f"<p>Early window {early[0]:g}–{early[1]:g}; late window "
            f"{late[0]:g}–{late[1]:g} code time.</p>"
            "<table><tr><th>Quantity</th><th>Early</th><th>Late</th><th>Change</th></tr>"
            f"{rows}</table><p><a href='summary.json'>Data and method (JSON)</a></p>"
            "<img src='entropy_transport.png' alt='Entropy transport'>"
            "</html>")
    (output / "index.html").write_text(page + "\n")
    print(output / "index.html")


if __name__ == "__main__":
    main()
