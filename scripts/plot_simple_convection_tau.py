#!/usr/bin/env python3
"""Track the Rosseland tau=1 surface in one xz slice of a convection run.

This is a slice diagnostic, not a horizontal mean over the full 3D box.  The
opacity lookup matches the tabulated rho-T logarithmic interpolation used by
the problem generator.  Optical depth above the top boundary is neglected;
the initial profile places only ~3.6e-5 there for the current pilot input.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from plot_simple_convection_movies import pairs, read_plane


def units_from_input(path: Path) -> dict[str, float]:
    values = dict(re.findall(r"^\s*(rho_unit|egas_unit|leng_unit|time_unit)\s*=\s*"
                             r"([^#\s]+)", path.read_text(), re.M))
    units = {name: float(value) for name, value in values.items()}
    for name in ("rho_unit", "egas_unit", "leng_unit"):
        if name not in units or units[name] <= 0:
            raise ValueError(f"Missing positive {name} in {path}")
    units["time_unit"] = (units.get("time_unit", -1.0)
                          if units.get("time_unit", -1.0) > 0
                          else units["leng_unit"]
                          / np.sqrt(units["egas_unit"] / units["rho_unit"]))
    return units


def opacity_interpolator(path: Path) -> RegularGridInterpolator:
    with h5py.File(path, "r") as table:
        log_rho = table["axes/log10_density"][...]
        log_temp = table["axes/log10_temperature"][...]
        log_kappa = table["kappa/log10_rosseland"][...]
    return RegularGridInterpolator((log_rho, log_temp), log_kappa,
                                   bounds_error=False, fill_value=None)


def frame_tau1(prim_path: Path, uov_path: Path,
               opacity: RegularGridInterpolator, units: dict[str, float]
               ) -> dict[str, float]:
    with h5py.File(prim_path, "r") as source:
        time = float(source.attrs["Time"])
    prim, _, z = read_plane(prim_path, "prim", "xz")
    uov, _, _ = read_plane(uov_path, "user_out_var", "xz")
    rho = prim[0] * units["rho_unit"]
    temp = uov[2]  # gas temperature in K from this problem generator
    if not (np.isfinite(rho).all() and np.isfinite(temp).all()
            and (rho > 0).all() and (temp > 0).all()):
        raise ValueError(f"Invalid density or gas temperature in {prim_path}")
    query = np.stack((np.log10(rho), np.log10(temp)), axis=-1)
    sigma = 10.0 ** opacity(query) * rho * units["leng_unit"]
    dz = float(z[1] - z[0])
    if not np.allclose(np.diff(z), dz):
        raise ValueError("Expected a uniform vertical slice grid")
    tau = np.cumsum((sigma * dz)[::-1], axis=0)[::-1] - 0.5 * sigma * dz
    surface_z = np.full(rho.shape[1], np.nan)
    surface_temp = np.full(rho.shape[1], np.nan)
    for j in range(rho.shape[1]):
        if tau[-1, j] <= 1.0 <= tau[0, j]:
            surface_z[j] = np.interp(1.0, tau[::-1, j], z[::-1])
            surface_temp[j] = np.interp(surface_z[j], z, temp[:, j])
    coverage = float(np.mean(np.isfinite(surface_z)))
    return {
        "time_code": time,
        "time_minutes": time * units["time_unit"] / 60.0,
        "coverage": coverage,
        "tau_top_mean": float(np.mean(tau[-1])),
        "z_tau1_mean_code": float(np.nanmean(surface_z)) if coverage else None,
        "z_tau1_std_code": float(np.nanstd(surface_z)) if coverage else None,
        "T_tau1_mean_K": float(np.nanmean(surface_temp)) if coverage else None,
        "T_tau1_std_K": float(np.nanstd(surface_temp)) if coverage else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--opacity-table", type=Path, required=True)
    parser.add_argument("--solar-teff", type=float, default=5772.0)
    args = parser.parse_args()
    units = units_from_input(args.input)
    opacity = opacity_interpolator(args.opacity_table)
    frames = pairs(args.run_dir, "xz")
    if not frames:
        raise ValueError(f"No paired xz slices in {args.run_dir}")
    records = [frame_tau1(prim, uov, opacity, units) for prim, uov in frames]
    (args.run_dir / "tau1_slice.json").write_text(
        json.dumps({"scope": "one xz slice", "records": records}, indent=2) + "\n")

    t = np.array([record["time_minutes"] for record in records])
    z = np.array([record["z_tau1_mean_code"] for record in records], dtype=float)
    zstd = np.array([record["z_tau1_std_code"] for record in records], dtype=float)
    temp = np.array([record["T_tau1_mean_K"] for record in records], dtype=float)
    tstd = np.array([record["T_tau1_std_K"] for record in records], dtype=float)
    coverage = np.array([record["coverage"] for record in records])
    zscale = units["leng_unit"] / 1.0e5
    plt.rcParams.update({"font.size": 13})
    fig, ax = plt.subplots(3, 1, figsize=(11, 10), sharex=True,
                           constrained_layout=True)
    ax[0].plot(t, z * zscale)
    ax[0].fill_between(t, (z-zstd)*zscale, (z+zstd)*zscale, alpha=0.2)
    ax[0].set(ylabel="height above box bottom [km]",
              title="Rosseland optical-depth-one surface in one xz slice")
    ax[1].plot(t, temp)
    ax[1].fill_between(t, temp-tstd, temp+tstd, alpha=0.2)
    ax[1].axhline(args.solar_teff, color="black", lw=1, ls="--",
                  label=f"solar Teff = {args.solar_teff:g} K")
    ax[1].set(ylabel="gas temperature at tau=1 [K]")
    ax[1].legend()
    ax[2].plot(t, coverage)
    ax[2].set(xlabel="physical time [min]", ylabel="slice coverage", ylim=(-0.02, 1.02))
    for axis in ax:
        axis.grid(alpha=0.25)
    image_dir = args.run_dir / "imgs"
    image_dir.mkdir(exist_ok=True)
    fig.savefig(image_dir / "simple_convection_tau1_slice.png", dpi=180)
    plt.close(fig)
    print(f"Wrote {args.run_dir / 'tau1_slice.json'} and tau=1 slice plot")


if __name__ == "__main__":
    main()
