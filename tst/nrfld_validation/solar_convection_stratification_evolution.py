#!/usr/bin/env python3
"""Plot transient horizontal-mean temperature changes in solar-box startups."""

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

from solar_convection_report import input_values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs=2, action="append", required=True,
                        metavar=("LABEL", "RUN_DIR"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.case) < 2:
        parser.error("Need at least two --case entries")
    cases = []
    for label, raw_dir in args.case:
        run_dir = Path(raw_dir)
        records = json.loads((run_dir / "profiles3d.json").read_text())["records"]
        if len(records) < 2:
            raise ValueError(f"{run_dir}: need at least two full-3D profiles")
        t = np.array([item["time_code"] for item in records])
        z = np.asarray(records[0]["z_code"])
        if any(not np.allclose(item["z_code"], z) for item in records[1:]):
            raise ValueError(f"{run_dir}: inconsistent vertical grids")
        temperature = np.asarray([item["T_mean_K"] for item in records])
        if temperature.shape != (len(t), len(z)) or not np.isfinite(temperature).all():
            raise ValueError(f"{run_dir}: invalid temperature profiles")
        delta = temperature-temperature[0]
        units = input_values(run_dir / "athinput.generated")
        upper = z >= units["x3max"]-1.0e8/units["leng_unit"]
        row = {
            "label": label, "run_dir": str(run_dir),
            "time_code": t.tolist(), "z_code": z.tolist(),
            "temperature_change_K": delta.tolist(),
            "final_upper_1Mm_mean_temperature_change_K": float(
                np.mean(delta[-1][upper])),
            "final_coolest_layer_temperature_change_K": float(np.min(delta[-1])),
            "final_coolest_layer_z_code": float(z[np.argmin(delta[-1])]),
        }
        cases.append((label, run_dir, t, z, temperature, delta, row))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "description": "Transient 3D horizontal-mean temperature change from each case's own t=0 state",
        "cases": [item[-1] for item in cases],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    limit = max(float(np.max(np.abs(item[5]))) for item in cases)
    limit = max(limit, 1.0)
    plt.rcParams.update({"font.size": 13})
    cols = min(2, len(cases))
    rows = (len(cases)+cols-1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(7*cols, 4.8*rows),
                             constrained_layout=True, squeeze=False)
    for axis, (label, _, t, z, _, delta, _) in zip(axes.flat, cases):
        mesh = axis.pcolormesh(z, t, delta, shading="auto", cmap="coolwarm",
                               vmin=-limit, vmax=limit)
        axis.set(title=label, xlabel="height [code]", ylabel="time [code]")
    for axis in list(axes.flat)[len(cases):]:
        axis.set_visible(False)
    fig.colorbar(mesh, ax=[axis for axis in axes.flat if axis.get_visible()],
                 label="horizontal mean T − initial [K]", shrink=0.9)
    fig.savefig(args.output_dir / "thermal_evolution.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for label, _, _, z, _, delta, _ in cases:
        ax.plot(z, delta[-1], label=label)
    ax.axhline(0, color="black", lw=1)
    ax.set(xlabel="height [code]", ylabel="final horizontal mean ΔT [K]",
           title="Temperature change at the final full-3D snapshot")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(args.output_dir / "thermal_final_profiles.png", dpi=160)
    plt.close(fig)
    rows_html = []
    for label, run_dir, _, _, _, _, row in cases:
        link = os.path.relpath(run_dir / "index.html", args.output_dir)
        rows_html.append("<tr>" + "".join(f"<td>{value}</td>" for value in (
            f"<a href='{html.escape(link)}'>{html.escape(label)}</a>",
            f"{row['final_upper_1Mm_mean_temperature_change_K']:+.0f}",
            f"{row['final_coolest_layer_temperature_change_K']:+.0f}",
            f"{row['final_coolest_layer_z_code']:.2f}",
        )) + "</tr>")
    (args.output_dir / "index.html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<title>Solar startup stratification</title><style>"
        "body{font:19px/1.5 system-ui;max-width:1300px;margin:2rem auto;"
        "padding:0 1rem;color:#172b3a}table{border-collapse:collapse}"
        "th,td{padding:.5rem .75rem;border-bottom:1px solid #ddd;text-align:left}"
        "img{max-width:100%}</style><h1>Solar startup thermal stratification</h1>"
        "<p><strong>Transient horizontal means.</strong> Each panel subtracts "
        "that case's own initial full-3D profile. This shows where the "
        "temperature changes; it is not an energy-conservation metric or "
        "evidence of a steady solar atmosphere.</p>"
        "<table><tr><th>Case</th><th>Final upper 1 Mm mean ΔT [K]</th>"
        "<th>Coolest-layer ΔT [K]</th><th>Coolest-layer z [code]</th></tr>"
        + "".join(rows_html) + "</table>"
        + "<img src='thermal_evolution.png' alt='thermal evolution'>"
        + "<img src='thermal_final_profiles.png' alt='final profiles'>"
        + "<p><a href='summary.json'>Numerical profiles</a></p></html>"
    )
    print(args.output_dir / "index.html")


if __name__ == "__main__":
    main()
