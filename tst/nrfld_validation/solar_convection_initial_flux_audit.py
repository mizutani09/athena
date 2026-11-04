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

from solar_convection_report import input_values, parse_log, read_history, sampled_energy_ledger


def load_profile(path: Path) -> dict[str, np.ndarray]:
    with path.open() as stream:
        stream.readline()
        columns = stream.readline().lstrip("# ").split()
    values = np.loadtxt(path)
    return {name: values[:, index] for index, name in enumerate(columns)}


def sampled_interior_residuals(profile: dict[str, np.ndarray], units: dict[str, float],
                               nx3: int, reference_flux: float) -> dict[str, np.ndarray]:
    """Estimate 1D face FLD flux and centered HSE error on the actual z grid.

    This mirrors the NR face opacity and limiter using the imported profile
    sampled at cell centers. It excludes ghost faces, temperature noise, and
    C++ EOS/opacity interpolation, so it is a diagnostic, not an exact solver
    residual.
    """
    z = np.linspace(units["x3min"], units["x3max"], nx3, endpoint=False)
    z += 0.5*(units["x3max"]-units["x3min"])/nx3
    dz = (units["x3max"]-units["x3min"])*units["leng_unit"]/nx3
    center = {key: np.interp(z, profile["z"], profile[key])
              for key in ("rho", "pressure", "erad", "kappa_R")}
    sigma = center["rho"]*center["kappa_R"]
    arithmetic = 0.5*(sigma[1:]+sigma[:-1])
    harmonic = 2*sigma[1:]*sigma[:-1]/(sigma[1:]+sigma[:-1])
    sigma_face = np.minimum(arithmetic, np.maximum(harmonic, 4/(3*dz)))
    e_face = 0.5*(center["erad"][1:]+center["erad"][:-1])
    grad_e = np.diff(center["erad"])/dz
    reduced_gradient = np.abs(grad_e)/(sigma_face*e_face)
    limiter = (2+reduced_gradient)/(6+3*reduced_gradient+reduced_gradient**2)
    discrete_flux = -2.99792458e10*limiter*grad_e/sigma_face
    face_z = 0.5*(z[1:]+z[:-1])
    model_flux = np.interp(face_z, profile["z"], profile["flux_rad"])
    # The input's problem/grav_acc is in physical cm/s^2; pgen converts it
    # to code units. input_values() returns that original physical value.
    gravity = units["grav_acc"]
    rho_face = 0.5*(center["rho"][1:]+center["rho"][:-1])
    hse = np.abs(np.diff(center["pressure"])/dz+rho_face*gravity)/(rho_face*gravity)
    return {"z": face_z, "discrete_flux": discrete_flux/reference_flux,
            "model_flux": model_flux/reference_flux, "hse_fraction": hse}


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
    units = input_values(args.startup_dir / "athinput.generated")
    log = parse_log(args.startup_dir / "run.log")
    mesh = re.search(r"^<mesh>\s*$(.*?)(?=^<|\Z)",
                     (args.startup_dir / "athinput.generated").read_text(),
                     re.M | re.S)
    if mesh is None or (nmatch := re.search(r"^\s*nx3\s*=\s*(\d+)",
                                         mesh.group(1), re.M)) is None:
        raise ValueError("Cannot read mesh/nx3 from generated input")
    nx3 = int(nmatch.group(1))
    manifest = json.loads((args.startup_dir / "manifest.json").read_text())
    for item in manifest.get("command", []):
        if item.startswith("mesh/nx3="):
            nx3 = int(item.split("=", 1)[1])
    interior = sampled_interior_residuals(profile, units, nx3, args.reference_flux)
    grid_estimates = {}
    for grid_nx3 in (nx3, 2*nx3, 4*nx3):
        sampled = sampled_interior_residuals(
            profile, units, grid_nx3, args.reference_flux)
        grid_estimates[str(grid_nx3)] = {
            "uppermost_face_z_code": float(sampled["z"][-1]),
            "uppermost_face_fld_flux_over_solar": float(sampled["discrete_flux"][-1]),
            "uppermost_face_model_flux_over_solar": float(sampled["model_flux"][-1]),
            "max_centered_hse_fraction": float(np.max(sampled["hse_fraction"])),
        }
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
    ledger = sampled_energy_ledger(
        history, np.arange(len(history["time"])), units, args.reference_flux)
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
        "solver": log,
        "startup_sampled_energy_ledger": ledger,
        "interior_discrete_profile_estimate": {
            "method": "profile interpolated to cell centers; NR face-opacity and limiter; "
                      "no ghost face, perturbation, or exact EOS/opacity inversion",
            "nx3": nx3,
            "uppermost_face_z_code": float(interior["z"][-1]),
            "uppermost_face_fld_flux_over_solar": float(interior["discrete_flux"][-1]),
            "uppermost_face_model_flux_over_solar": float(interior["model_flux"][-1]),
            "max_centered_hse_fraction": float(np.max(interior["hse_fraction"])),
            "uppermost_centered_hse_fraction": float(interior["hse_fraction"][-1]),
        },
        "sampled_grid_resolution_estimates": grid_estimates,
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
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    axes[0].semilogy(interior["z"], interior["model_flux"], label="1D model")
    axes[0].semilogy(interior["z"], interior["discrete_flux"], label="sampled FLD faces")
    axes[0].set(title="Interior radiative flux", xlabel="height [code]",
                ylabel=r"$F_{rad}/F_\odot$")
    axes[0].legend()
    axes[1].plot(interior["z"], interior["discrete_flux"]-interior["model_flux"])
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set(title="Grid minus 1D model", xlabel="height [code]",
                ylabel=r"$\Delta F_{rad}/F_\odot$")
    axes[2].semilogy(interior["z"], interior["hse_fraction"])
    axes[2].set(title="Centered hydrostatic estimate", xlabel="height [code]",
                ylabel=r"$|dP/dz+\rho g|/(\rho g)$")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.savefig(args.output_dir / "sampled_interior_residuals.png", dpi=160)
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
            ("t=0–10 stored-energy rate / solar",
             ledger["stored_energy_rate_over_solar"] if ledger else float("nan")),
            ("t=0–10 sampled ledger residual / solar",
             ledger["residual_equivalent_solar_flux"] if ledger else float("nan")),
            ("Uppermost interior FLD face / solar",
             result["interior_discrete_profile_estimate"]["uppermost_face_fld_flux_over_solar"]),
            ("Same face in 1D model / solar",
             result["interior_discrete_profile_estimate"]["uppermost_face_model_flux_over_solar"]),
            ("Maximum centered HSE fraction",
             result["interior_discrete_profile_estimate"]["max_centered_hse_fraction"]),
        ))
    samples_table = "".join(
        f"<tr><td>{html.escape(key)}</td>"
        f"<td>{item['radiative_flux_over_solar']:.5g}</td>"
        f"<td>{item['model_convective_flux_over_solar']:.5g}</td></tr>"
        for key, item in samples.items())
    startup_link = (f"<a href='../{html.escape(args.startup_dir.name)}/index.html'>"
                    "Startup case report</a>")
    grid_table = "".join(
        "<tr>" + "".join(f"<td>{item}</td>" for item in (
            grid_nx3,
            f"{row['uppermost_face_z_code']:.3f}",
            f"{row['uppermost_face_fld_flux_over_solar']:.4f}",
            f"{row['uppermost_face_model_flux_over_solar']:.4f}",
            f"{row['max_centered_hse_fraction']:.4f}",
        )) + "</tr>"
        for grid_nx3, row in grid_estimates.items())
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
        "the discrete FLD and boundary residuals still need an audit. "
        "The energy ledger below uses sampled output fluxes, so its residual "
        "is diagnostic rather than a conservation acceptance test. The "
        "interior face estimate samples the imported profile on the run grid; "
        "it excludes ghost faces, temperature noise, and exact C++ table "
        "interpolation.</p>"
        f"<p>{startup_link} · <a href='summary.json'>JSON</a></p>"
        f"<table>{table}</table>"
        "<h2>Initial 1D model flux by height</h2><table><tr>"
        "<th>Height [code]</th><th>Radiation / solar</th>"
        f"<th>Model convection / solar</th></tr>{samples_table}</table>"
        "<img src='initial_flux_audit.png' alt='initial flux audit'>"
        "<h2>Sampled interior grid residuals</h2>"
        "<p>These higher-resolution rows are resamplings of the initial "
        "1D profile, not additional 3D runs.</p>"
        "<table><tr><th>Vertical cells</th><th>Top interior face z</th>"
        "<th>Sampled FLD flux / solar</th><th>Model flux / solar</th>"
        f"<th>Max centered HSE fraction</th></tr>{grid_table}</table>"
        "<img src='sampled_interior_residuals.png' alt='interior residuals'>"
        "</html>"
    )
    print(args.output_dir / "index.html")


if __name__ == "__main__":
    main()
