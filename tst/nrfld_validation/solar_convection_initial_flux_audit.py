#!/usr/bin/env python3
"""Show initial solar-profile flux assumptions, thermal times, and startup output."""

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

from solar_convection_report import read_history


def load_profile(path: Path) -> dict[str, np.ndarray]:
    with path.open() as stream:
        stream.readline()
        columns = stream.readline().lstrip("# ").split()
    values = np.loadtxt(path)
    return {name: values[:, index] for index, name in enumerate(columns)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--startup-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-flux", type=float, default=6.3e10)
    args = parser.parse_args()
    profile = load_profile(args.profile)
    with args.profile.open() as source:
        match = re.search(r"flux_target=([0-9.eE+-]+)", source.readline())
    if match is None:
        raise ValueError(f"{args.profile}: missing model flux target")
    model_flux = float(match.group(1))
    history = read_history(next(args.startup_dir.glob("*.hst")))
    z_cm = profile["z_cm"]
    z_code = profile["z"]
    if not np.all(np.diff(z_cm) > 0):
        raise ValueError("Initial profile height must increase monotonically")
    energy_density = profile["rho"]*profile["specific_u"]
    segment = 0.5*(energy_density[1:]+energy_density[:-1])*np.diff(z_cm)
    energy_above = np.r_[np.cumsum(segment[::-1])[::-1], 0.0]
    thermal_hours = energy_above/(args.reference_flux*3600.0)
    first = int(np.argmin(abs(history["time"])))
    last = int(np.argmin(abs(history["time"]-10.0)))
    late = (history["time"] >= 5.0) & (history["time"] <= 10.0)
    samples = {}
    for z in (-3.0, 0.0, 2.0, 3.0):
        samples[f"z_code_{z:+.1f}"] = {
            "radiative_flux_over_solar": float(np.interp(
                z, z_code, profile["flux_rad"])/args.reference_flux),
            "model_convective_flux_over_solar": float(np.interp(
                z, z_code, profile["convective_flux_fraction"])
                * model_flux/args.reference_flux),
        }
    top_cm = float(z_cm[-1])
    result = {
        "description": "Initial model flux split and energy-over-flux times; "
                       "discrete FLD equilibrium is not yet audited",
        "profile": str(args.profile), "startup_dir": str(args.startup_dir),
        "profile_flux_target_cgs": model_flux,
        "profile_samples": samples,
        "whole_box_thermal_hours": float(thermal_hours[0]),
        "upper_1Mm_thermal_hours": float(np.interp(
            top_cm-1e8, z_cm, thermal_hours)),
        "upper_0p5Mm_thermal_hours": float(np.interp(
            top_cm-0.5e8, z_cm, thermal_hours)),
        "initial_vertical_rms_code": float(np.sqrt(max(
            history["VzRms2Mean"][first], 0.0))),
        "initial_top_flux_over_solar": float(
            history["FtotTop"][first]/args.reference_flux),
        "top_flux_near_t10_over_solar": float(
            history["FtotTop"][last]/args.reference_flux),
        "top_flux_t5_10_mean_over_solar": float(np.mean(
            history["FtotTop"][late])/args.reference_flux),
        "startup_history_end_code": float(history["time"][-1]),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    plt.rcParams.update({"font.size": 13})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes[0, 0].plot(z_code, profile["flux_rad"]/args.reference_flux,
                    label="radiation")
    axes[0, 0].plot(z_code, profile["convective_flux_fraction"]
                    * model_flux/args.reference_flux,
                    label="model convection")
    axes[0, 0].axhline(1, color="black", ls="--", lw=1)
    axes[0, 0].set(title="Initial 1D model flux split", xlabel="height [code]",
                   ylabel=r"fraction of $F_\odot$")
    axes[0, 0].legend()
    axes[0, 1].semilogy(z_code, np.maximum(thermal_hours, 1e-6))
    axes[0, 1].set(title="Internal energy above height / solar flux",
                   xlabel="height [code]", ylabel="thermal time [hours]")
    t = history["time"]
    axes[1, 0].plot(t, history["FtotTop"]/args.reference_flux)
    axes[1, 0].axhline(1, color="black", ls="--", lw=1)
    axes[1, 0].set(title="Startup top flux", xlabel="time [code]",
                   ylabel=r"$F_{top}/F_\odot$")
    axes[1, 1].plot(t, history["FhydBottom"]/args.reference_flux,
                    label="bottom hydro")
    axes[1, 1].plot(t, history["FradBottom"]/args.reference_flux,
                    label="bottom radiation")
    axes[1, 1].set(title="Startup lower-boundary flux",
                   xlabel="time [code]", ylabel=r"$F/F_\odot$")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    fig.savefig(args.output_dir / "initial_flux_audit.png", dpi=160)
    plt.close(fig)
    table = "".join(
        f"<tr><th>{html.escape(label)}</th><td>{value:.5g}</td></tr>"
        for label, value in (
            ("Whole-box energy / solar flux [days]",
             result["whole_box_thermal_hours"]/24),
            ("Upper 1 Mm energy / solar flux [hours]",
             result["upper_1Mm_thermal_hours"]),
            ("Upper 0.5 Mm energy / solar flux [hours]",
             result["upper_0p5Mm_thermal_hours"]),
            ("Initial vertical RMS [code]", result["initial_vertical_rms_code"]),
            ("Initial top flux / solar", result["initial_top_flux_over_solar"]),
            ("t=5–10 mean top flux / solar",
             result["top_flux_t5_10_mean_over_solar"]),
        ))
    samples_table = "".join(
        f"<tr><td>{html.escape(key)}</td>"
        f"<td>{item['radiative_flux_over_solar']:.5g}</td>"
        f"<td>{item['model_convective_flux_over_solar']:.5g}</td></tr>"
        for key, item in samples.items())
    startup_link = (f"<a href='../{html.escape(args.startup_dir.name)}/index.html'>"
                    "Startup case report</a>")
    (args.output_dir / "index.html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<title>Solar convection initial flux audit</title><style>"
        "body{font:19px/1.5 system-ui;max-width:1200px;margin:2rem auto;"
        "padding:0 1rem;color:#172b3a}table{border-collapse:collapse}"
        "th,td{padding:.5rem .75rem;border-bottom:1px solid #ddd;text-align:left}"
        "img{max-width:100%}</style><h1>Initial solar-box flux and thermal time</h1>"
        "<p><strong>The 1D model assumes deep convective transport.</strong> "
        "The 3D calculation starts from zero velocity. Missing initial "
        "convective transport is a plausible cause of the early flux drop; "
        "the discrete FLD and boundary residuals still need an audit.</p>"
        f"<p>{startup_link} · <a href='summary.json'>JSON</a></p>"
        f"<table>{table}</table>"
        "<h2>Initial 1D model flux by height</h2><table><tr>"
        "<th>Height [code]</th><th>Radiation / solar</th>"
        f"<th>Model convection / solar</th></tr>{samples_table}</table>"
        "<img src='initial_flux_audit.png' alt='initial flux audit'></html>"
    )
    print(args.output_dir / "index.html")


if __name__ == "__main__":
    main()
