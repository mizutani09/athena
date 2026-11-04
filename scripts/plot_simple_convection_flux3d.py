#!/usr/bin/env python3
"""Estimate interior vertical energy-flux profiles from full 3D FLD dumps.

Hydro terms use cell-centered primitive fields. The radiative term reconstructs
the Levermore-Pomraning FLD closure from cell-centered gradients; it is an
interior diagnostic, not the exact finite-volume boundary flux. Use the HST
FhydBottom/FradTop/FtotTop columns for the boundary energy ledger.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_simple_convection_surface3d import read_volume
from plot_simple_convection_tau import opacity_interpolator, units_from_input


SOLAR_FLUX = 6.3e10
LIGHT_SPEED = 2.99792458e10
FIELDS = ("radiative", "convective_enthalpy", "mean_flow_enthalpy",
          "kinetic", "gravitational", "total")


def periodic_gradient(data: np.ndarray, spacing: float, axis: int) -> np.ndarray:
    return (np.roll(data, -1, axis=axis) - np.roll(data, 1, axis=axis)) / (2*spacing)


def flux_profiles(prim: np.ndarray, uov: np.ndarray,
                  coords: tuple[np.ndarray, ...], units: dict[str, float],
                  opacity, grav_cgs: float) -> dict[str, list[float]]:
    x, y, z = coords
    if len(z) < 3:
        raise ValueError("Need at least three vertical cells")
    rho = prim[0] * units["rho_unit"]
    pressure = prim[1] * units["egas_unit"]
    velocity_unit = units["leng_unit"] / units["time_unit"]
    vx, vy, vz = (prim[index] * velocity_unit for index in (2, 3, 4))
    internal = uov[0]
    erad = uov[1]
    temp = uov[2]
    if (np.any(rho <= 0) or np.any(internal <= 0) or
            np.any(erad <= 0) or np.any(temp <= 0)):
        raise ValueError("Nonpositive thermodynamic state in full volume")
    sigma = 10**opacity(np.stack((np.log10(rho), np.log10(temp)), axis=-1)) * rho
    if not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("Invalid Rosseland opacity in full volume")
    dx, dy, dz = (float(np.mean(np.diff(coord))) * units["leng_unit"]
                  for coord in coords)
    grad_x = periodic_gradient(erad, dx, axis=2)
    grad_y = periodic_gradient(erad, dy, axis=1)
    grad_z = np.gradient(erad, dz, axis=0, edge_order=2)
    gradient = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    ratio = gradient / (sigma * erad)
    limiter = np.empty_like(ratio)
    small = ratio <= 1
    limiter[small] = (2 + ratio[small]) / (6 + 3*ratio[small] + ratio[small]**2)
    inv = 1 / ratio[~small]
    limiter[~small] = (inv + 2*inv**2) / (1 + 3*inv + 6*inv**2)
    flux_rad = -LIGHT_SPEED * limiter * grad_z / sigma
    enthalpy = internal + pressure
    mass_flux = np.mean(rho*vz, axis=(1, 2))
    mass_mean = np.mean(rho, axis=(1, 2))
    enthalpy_mean = np.mean(enthalpy, axis=(1, 2))
    mean_flow_enthalpy = mass_flux * enthalpy_mean / mass_mean
    advective_enthalpy = np.mean(enthalpy*vz, axis=(1, 2))
    convective_enthalpy = advective_enthalpy - mean_flow_enthalpy
    kinetic = np.mean(0.5*rho*(vx**2 + vy**2 + vz**2)*vz, axis=(1, 2))
    gravity = mass_flux * grav_cgs * z * units["leng_unit"]
    radiation = np.mean(flux_rad, axis=(1, 2))
    total = radiation + advective_enthalpy + kinetic + gravity
    return {"z_code": z.tolist(),
            "radiative": radiation.tolist(),
            "convective_enthalpy": convective_enthalpy.tolist(),
            "mean_flow_enthalpy": mean_flow_enthalpy.tolist(),
            "kinetic": kinetic.tolist(),
            "gravitational": gravity.tolist(),
            "total": total.tolist(),
            "mass_flux_cgs": mass_flux.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--opacity-table", type=Path, required=True)
    parser.add_argument("--late-fraction", type=float, default=0.25)
    args = parser.parse_args()
    if not (0 < args.late_fraction <= 1):
        parser.error("--late-fraction must be in (0, 1]")
    match = re.search(r"^\s*grav_acc\s*=\s*([^#\s]+)", args.input.read_text(), re.M)
    if not match:
        parser.error("Input has no problem/grav_acc")
    grav_cgs = float(match.group(1))
    units = units_from_input(args.input)
    opacity = opacity_interpolator(args.opacity_table)
    prim_files = sorted(args.run_dir.glob("*.prim_full.*.athdf"))
    uov_files = {path.name.split(".uov_full.")[-1]: path
                 for path in args.run_dir.glob("*.uov_full.*.athdf")}
    records = []
    for prim_path in prim_files:
        suffix = prim_path.name.split(".prim_full.")[-1]
        if suffix not in uov_files:
            continue
        prim, coords, time = read_volume(prim_path, "prim")
        uov, other_coords, other_time = read_volume(uov_files[suffix], "user_out_var")
        if not np.isclose(time, other_time) or any(
                not np.array_equal(a, b) for a, b in zip(coords, other_coords)):
            raise ValueError("Paired full-volume time or grid mismatch")
        item = flux_profiles(prim, uov, coords, units, opacity, grav_cgs)
        item["time_code"] = time
        records.append(item)
    if not records:
        raise ValueError("No paired full-volume dumps")
    times = np.asarray([item["time_code"] for item in records])
    if not np.all(np.diff(times) > 0):
        raise ValueError("Full-volume times must increase")
    start = times[-1] - args.late_fraction * (times[-1] - times[0])
    late = [item for item in records if item["time_code"] >= start]
    mean = None
    if len(late) >= 2:
        mean = {key: np.mean([item[key] for item in late], axis=0).tolist()
                for key in FIELDS}
        mean.update({"z_code": late[0]["z_code"],
                     "window_start_code": late[0]["time_code"],
                     "window_end_code": late[-1]["time_code"],
                     "snapshot_count": len(late)})
    output = {"method": "cell-centered interior reconstruction; boundary fluxes from HST",
              "flux_units": "erg cm^-2 s^-1", "reference_solar_flux_cgs": SOLAR_FLUX,
              "records": records, "late_mean": mean}
    (args.run_dir / "flux3d.json").write_text(json.dumps(output, indent=2) + "\n")

    selected = mean or records[-1]
    z = np.asarray(selected["z_code"])
    surface_file = args.run_dir / "surface3d.json"
    photosphere_z = None
    if surface_file.is_file():
        surface_records = json.loads(surface_file.read_text())["records"]
        if surface_records:
            photosphere_z = float(surface_records[-1]["z_tau23_mean_code"])
    plt.rcParams.update({"font.size": 14})
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True,
                             constrained_layout=True)
    for key, label in (("radiative", "FLD radiation"),
                       ("convective_enthalpy", "convective enthalpy"),
                       ("mean_flow_enthalpy", "mean-flow enthalpy"),
                       ("kinetic", "kinetic")):
        axes[0].plot(np.asarray(selected[key]) / SOLAR_FLUX, z, label=label)
    for key, label in (("total", "hydro + radiation + gravity"),
                       ("gravitational", "gravitational")):
        axes[1].plot(np.asarray(selected[key]) / SOLAR_FLUX, z, label=label)
    for axis in axes:
        axis.axvline(1, color="black", lw=1, ls="--", label="solar flux")
        if photosphere_z is not None:
            axis.axhline(photosphere_z, color="0.4", lw=1, ls=":",
                         label="last mean tau=2/3 height")
        axis.set(xlabel="vertical energy flux / solar", ylabel="z [code]")
        axis.grid(alpha=0.25)
        axis.legend()
    label = (f"late mean t={mean['window_start_code']:.3g}–"
             f"{mean['window_end_code']:.3g} ({mean['snapshot_count']} dumps)"
             if mean else f"last dump t={records[-1]['time_code']:.3g}")
    fig.suptitle(f"3D interior vertical flux profiles: {label}\n"
                 "Cell-center estimate; HST gives exact boundary fluxes")
    images = args.run_dir / "imgs"
    images.mkdir(exist_ok=True)
    fig.savefig(images / "simple_convection_flux3d.png", dpi=180)
    plt.close(fig)
    print(f"Wrote {args.run_dir / 'flux3d.json'} and flux-profile plot")


if __name__ == "__main__":
    main()
