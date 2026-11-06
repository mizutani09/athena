#!/usr/bin/env python3
"""Test whether a segmented solar run trends toward solar surface flux."""

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

from solar_convection_report import (combine_histories, input_values,
                                     physical_time_unit)


F_SUN = 6.3e10


def mean(t: np.ndarray, y: np.ndarray) -> float:
    if len(t) < 2 or t[-1] <= t[0]:
        raise ValueError("Need at least two samples")
    return float(np.trapezoid(y, t) / (t[-1] - t[0]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback-state", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bin-width", type=float, default=500.0)
    args = parser.parse_args()
    if args.bin_width <= 0:
        parser.error("--bin-width must be positive")
    state_path = args.feedback_state.resolve()
    state = json.loads(state_path.read_text())
    root = state_path.parent
    runs = [root / item["case"] for item in state["completed"]]
    if not runs:
        parser.error("No completed run segments")
    histories = [next(run.glob("*.hst")) for run in runs]
    history = combine_histories(histories, histories[0])
    units = input_values(runs[0] / "athinput.generated")
    velocity_unit = np.sqrt(units["egas_unit"] / units["rho_unit"])
    time_unit = physical_time_unit(units, velocity_unit)
    area = ((units["x1max"] - units["x1min"])
            * (units["x2max"] - units["x2min"])
            * units["leng_unit"]**2)
    end = history["time"][-1]
    if abs(end/args.bin_width - round(end/args.bin_width)) > 1e-8:
        parser.error("Last history time must be an integer multiple of bin width")
    top = history["FtotTop"] + history.get("FadvRadTop", 0)
    bottom = (history["FhydBottom"] + history["FradBottom"]
              + history.get("FadvRadBottom", 0))
    bins = []
    for lo in np.arange(0.0, end, args.bin_width):
        hi = lo + args.bin_width
        selected = (history["time"] >= lo) & (history["time"] <= hi)
        t = history["time"][selected]
        stored = history["EtotGrav"][selected]
        bins.append({"start_code": float(lo), "end_code": float(hi),
                     "top_over_solar": mean(t, top[selected])/F_SUN,
                     "bottom_over_solar": mean(t, bottom[selected])/F_SUN,
                     "stored_rate_over_solar": float(
                         (stored[-1] - stored[0])
                         / (area*F_SUN*(t[-1]-t[0])*time_unit))})
    if len(bins) < 3:
        parser.error("Need at least three complete time bins")
    after_startup = bins[1:]
    middle = np.array([(row["start_code"] + row["end_code"])/2
                       for row in after_startup])
    flux = np.array([row["top_over_solar"] for row in after_startup])
    slope = float(np.polyfit(middle, flux, 1)[0])
    startup_flux = json.loads((runs[0] / "flux3d.json").read_text())["records"][0]
    if abs(startup_flux["time_code"]) > 1e-6:
        raise ValueError("First flux3d record is not t=0")
    z = np.asarray(startup_flux["z_code"])
    radiative = np.asarray(startup_flux["radiative"])
    mid_index = int(np.argmin(np.abs(z)))
    output = (args.output_dir or root / "wait_diagnostic").resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "method": "HST trapezoidal nonoverlapping flux means and endpoint energy rates; t=0 interior radiation from cell-centered FLD reconstruction",
        "feedback_state": str(state_path), "bin_width_code": args.bin_width,
        "initial_top_over_solar": float(top[0]/F_SUN),
        "initial_bottom_over_solar": float(bottom[0]/F_SUN),
        "initial_interior_rad_z0_over_solar": float(radiative[mid_index]/F_SUN),
        "initial_interior_rad_upper_over_solar": float(radiative[-2]/F_SUN),
        "initial_interior_z0_code": float(z[mid_index]),
        "post_startup_top_range_over_solar": [float(flux.min()), float(flux.max())],
        "post_startup_top_linear_slope_per_1000_code": slope*1000,
        "warning": "The descriptive slope is not a forecast of the thermal equilibrium.",
        "bins": bins,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    plt.rcParams.update({"font.size": 16})
    fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True,
                             constrained_layout=True)
    x = np.array([(row["start_code"] + row["end_code"])/2 for row in bins])
    axes[0].plot(x, [row["top_over_solar"] for row in bins], "o-",
                 label="top total")
    axes[0].plot(x, [row["bottom_over_solar"] for row in bins], "o-",
                 label="bottom total")
    axes[0].axhline(1, color="black", ls="--", label="solar")
    axes[0].set_ylabel("Flux / solar")
    axes[0].legend()
    axes[1].plot(x[1:], [row["top_over_solar"] for row in after_startup],
                 "o-", label="top total")
    axes[1].plot(x[1:], [row["bottom_over_solar"] for row in after_startup],
                 "o-", label="bottom total")
    axes[1].set_ylabel("After-startup flux / solar")
    axes[1].legend()
    axes[2].plot(x, [row["stored_rate_over_solar"] for row in bins], "o-")
    axes[2].axhline(0, color="black", lw=1)
    axes[2].set(ylabel="Stored-energy rate / solar", xlabel="Code time")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.savefig(output / "wait_diagnostic.png", dpi=150)
    plt.close(fig)

    rows = "".join(
        f"<tr><td>{item['start_code']:.0f}–{item['end_code']:.0f}</td>"
        f"<td>{item['top_over_solar']:.5f}</td>"
        f"<td>{item['bottom_over_solar']:.5f}</td>"
        f"<td>{item['stored_rate_over_solar']:+.5f}</td></tr>"
        for item in bins)
    source_link = html.escape(os.path.relpath(root / "index.html", output))
    page = ("<!doctype html><html lang='en'><meta charset='utf-8'>"
            "<title>Solar flux waiting diagnostic</title>"
            "<style>body{font:21px/1.5 system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem}"
            "img{max-width:100%}table{border-collapse:collapse}"
            "th,td{padding:.5rem .8rem;border-bottom:1px solid #ccc;text-align:right}"
            "th:first-child,td:first-child{text-align:left}strong{color:#18446f}</style>"
            "<h1>Would waiting alone raise the top flux?</h1>"
            f"<p>At t=0 the top flux was {top[0]/F_SUN:.4f} solar, but the "
            f"bottom flux was only {bottom[0]/F_SUN:.6f} solar. "
            "The initial state has no convective motion. Its reconstructed "
            f"interior radiative flux near z=0 was {radiative[mid_index]/F_SUN:.5g} "
            "solar; the initial solar-like top flux is therefore not a "
            "box-wide steady luminosity.</p>"
            f"<p><strong>After startup, the {args.bin_width:g}-code means of the top flux "
            f"range from {flux.min():.5f} to {flux.max():.5f} solar.</strong> "
            "There is no demonstrated trend to 1 solar. The slow entropy "
            "feedback changes the boundary condition over time, so these "
            "bins cannot determine the eventual fixed-entropy equilibrium.</p>"
            f"<p><a href='{source_link}'>Original campaign</a> · "
            "<a href='summary.json'>JSON data</a></p>"
            "<table><tr><th>Code time</th><th>Top / solar</th>"
            f"<th>Bottom / solar</th><th>Stored rate / solar</th></tr>{rows}</table>"
            "<img src='wait_diagnostic.png' alt='Flux and storage by time bin'>"
            "</html>")
    (output / "index.html").write_text(page + "\n")
    print(output / "index.html")


if __name__ == "__main__":
    main()
