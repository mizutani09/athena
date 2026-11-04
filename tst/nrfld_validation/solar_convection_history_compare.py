#!/usr/bin/env python3
"""Compare matched time windows of two solar-convection history files."""

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

from solar_convection_report import input_values, read_history


FLUX_REF = 6.3e10


def series(case: Path, header: Path | None) -> dict[str, np.ndarray]:
    paths = list(case.glob("*.hst"))
    if len(paths) != 1:
        raise ValueError(f"Expected one history file in {case}, found {len(paths)}")
    return read_history(paths[0], header)


def sampled(h: dict[str, np.ndarray], time: np.ndarray,
            velocity_unit: float) -> dict[str, np.ndarray]:
    interp = lambda name: np.interp(time, h["time"], h[name])
    return {
        "top_flux": interp("FtotTop") / FLUX_REF,
        "bottom_flux": interp("FhydBottom") / FLUX_REF,
        "bottom_net": interp("BottomMdotNet"),
        "mass_drift": 100.0 * interp("MassDrift"),
        "vz_rms": velocity_unit * np.sqrt(np.maximum(interp("VzRms2Mean"), 0.0)) / 1e5,
        "pbnd_scale": interp("PBNDScale"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--header-from", type=Path)
    parser.add_argument("--start", required=True, type=float)
    parser.add_argument("--end", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-label", default="Reference")
    parser.add_argument("--candidate-label", default="Candidate")
    parser.add_argument("--title", default="Solar convection: matched history comparison")
    args = parser.parse_args()
    if not args.start < args.end:
        parser.error("--start must be less than --end")
    cases = {args.reference_label: args.reference.resolve(),
             args.candidate_label: args.candidate.resolve()}
    if len(cases) != 2:
        parser.error("The labels must differ")
    histories = {label: series(path, args.header_from) for label, path in cases.items()}
    for label, h in histories.items():
        if h["time"][0] > args.start or h["time"][-1] < args.end:
            parser.error(f"{label} history does not cover {args.start}–{args.end}")
    # Evaluate both runs on the union of their sample times. Endpoints are
    # inserted explicitly so the averages cover exactly the requested window.
    grid = np.unique(np.r_[args.start, args.end, *(
        h["time"][(h["time"] > args.start) & (h["time"] < args.end)]
        for h in histories.values())])
    if len(grid) < 4:
        parser.error("Too few history samples in the matched window")
    samples = {}
    for label, h in histories.items():
        units = input_values(cases[label] / "athinput.generated")
        vunit = np.sqrt(units["egas_unit"] / units["rho_unit"])
        samples[label] = sampled(h, grid, vunit)
    metrics = {}
    duration = args.end - args.start
    for label, values in samples.items():
        metrics[label] = {
            "mean_top_flux_over_solar": float(np.trapezoid(values["top_flux"], grid) / duration),
            "mean_bottom_hydro_flux_over_solar": float(np.trapezoid(values["bottom_flux"], grid) / duration),
            "bottom_net_mass_flux_rms_code": float(np.sqrt(
                np.trapezoid(values["bottom_net"]**2, grid) / duration)),
            "maximum_absolute_mass_drift_percent": float(np.max(np.abs(values["mass_drift"]))),
            "mean_vertical_velocity_rms_km_s": float(np.trapezoid(values["vz_rms"], grid) / duration),
            "mean_pbnd_scale": float(np.trapezoid(values["pbnd_scale"], grid) / duration),
        }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps({
        "start_code": args.start, "end_code": args.end,
        "cases": {label: str(path) for label, path in cases.items()},
        "metrics": metrics,
    }, indent=2) + "\n")

    panels = [
        ("top_flux", "Top flux / solar"),
        ("bottom_flux", "Bottom hydro flux / solar"),
        ("bottom_net", "Bottom net mass flux [code]"),
        ("mass_drift", "Mass drift [%]"),
        ("vz_rms", "Vertical velocity RMS [km/s]"),
        ("pbnd_scale", "PBND scale"),
    ]
    plt.rcParams.update({"font.size": 13})
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True,
                             constrained_layout=True)
    for ax, (key, ylabel) in zip(axes.flat, panels):
        for label, values in samples.items():
            ax.plot(grid, values[key], label=label, linewidth=1.6)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("Time [code]")
    axes[0, 0].legend()
    fig.suptitle(args.title)
    fig.savefig(output / "comparison.png", dpi=170)
    plt.close(fig)

    labels = list(cases)
    names = {
        "mean_top_flux_over_solar": "Mean top flux / solar",
        "mean_bottom_hydro_flux_over_solar": "Mean bottom hydro flux / solar",
        "bottom_net_mass_flux_rms_code": "RMS bottom net mass flux [code]",
        "maximum_absolute_mass_drift_percent": "Maximum |mass drift| [%]",
        "mean_vertical_velocity_rms_km_s": "Mean vertical velocity RMS [km/s]",
        "mean_pbnd_scale": "Mean PBND scale",
    }
    rows = "".join(
        f"<tr><th>{html.escape(title)}</th>"
        + "".join(f"<td>{metrics[label][key]:.5g}</td>" for label in labels)
        + "</tr>" for key, title in names.items())
    links = " · ".join(
        f"<a href='{html.escape(os.path.relpath(path / 'index.html', output))}'>"
        f"{html.escape(label)} report</a>" for label, path in cases.items()
        if (path / "index.html").is_file())
    (output / "index.html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        f"<title>{html.escape(args.title)}</title>"
        "<style>body{font:19px/1.5 system-ui;max-width:1200px;margin:2rem auto;"
        "padding:0 1rem;color:#172b3a}h1{font-size:2rem}img{max-width:100%}"
        "table{border-collapse:collapse}th,td{padding:.5rem .8rem;"
        "border-bottom:1px solid #ddd;text-align:left}</style>"
        f"<h1>{html.escape(args.title)}</h1>"
        f"<p>Matched interval: <strong>{args.start:g}–{args.end:g} code time</strong>. "
        "Fluxes and velocities are time averages. This comparison alone does "
        "not establish stationarity or solar agreement.</p>"
        f"<p>{links} · <a href='comparison.json'>JSON</a></p>"
        "<table><tr><th>Metric</th>"
        + "".join(f"<th>{html.escape(label)}</th>" for label in labels)
        + f"</tr>{rows}</table><figure><img src='comparison.png' alt='comparison'></figure>"
        "</html>")
    print(output / "index.html")


if __name__ == "__main__":
    main()
