#!/usr/bin/env python3
"""Measure optical-depth-aligned surface statistics from full 3D HDF5 dumps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_simple_convection_tau import opacity_interpolator, units_from_input


def read_volume(path: Path, dataset: str) -> tuple[np.ndarray, tuple[np.ndarray, ...], float]:
    with h5py.File(path, "r") as source:
        raw = source[dataset][...]
        locations = source["LogicalLocations"][...]
        levels = source["Levels"][...] if "Levels" in source else np.zeros(len(locations))
        if np.any(levels != 0):
            raise ValueError("Full-volume diagnostic currently requires a uniform grid")
        size = np.asarray(source.attrs["RootGridSize"], dtype=int)
        bounds = [np.asarray(source.attrs[f"RootGridX{axis}"])
                  for axis in (1, 2, 3)]
        time = float(source.attrs["Time"])
    bx, by, bz = raw.shape[-1], raw.shape[-2], raw.shape[-3]
    if tuple(size % (bx, by, bz)) != (0, 0, 0):
        raise ValueError("MeshBlock sizes do not divide the root grid")
    data = np.full((raw.shape[0], size[2], size[1], size[0]), np.nan)
    for block, (ix, iy, iz) in enumerate(locations):
        data[:, iz*bz:(iz+1)*bz, iy*by:(iy+1)*by,
             ix*bx:(ix+1)*bx] = raw[:, block]
    if not np.isfinite(data).all():
        raise ValueError(f"Incomplete or nonfinite full volume in {path}")
    coords = tuple(limits[0] + (np.arange(n) + 0.5) *
                   (limits[1] - limits[0]) / n
                   for limits, n in zip(bounds, size))
    return data, coords, time


def surface_at(tau: np.ndarray, z: np.ndarray, field: np.ndarray,
               target: float) -> tuple[np.ndarray, np.ndarray]:
    above = np.count_nonzero(tau >= target, axis=0)
    valid = (above > 0) & (above < len(z))
    lower = np.clip(above - 1, 0, len(z) - 2)[None]
    t0 = np.take_along_axis(tau, lower, axis=0)[0]
    t1 = np.take_along_axis(tau, lower + 1, axis=0)[0]
    f0 = np.take_along_axis(field, lower, axis=0)[0]
    f1 = np.take_along_axis(field, lower + 1, axis=0)[0]
    fraction = (target - t0) / np.where(t1 != t0, t1 - t0, np.nan)
    z_surface = z[lower[0]] + fraction * (z[lower[0] + 1] - z[lower[0]])
    f_surface = f0 + fraction * (f1 - f0)
    return np.where(valid, z_surface, np.nan), np.where(valid, f_surface, np.nan)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--opacity-table", type=Path, required=True)
    parser.add_argument("--reference-profile", type=Path,
                        help="Imported solar 1D profile used only as a visual reference")
    parser.add_argument("--solar-teff", type=float, default=5772.0)
    args = parser.parse_args()
    units = units_from_input(args.input)
    opacity = opacity_interpolator(args.opacity_table)
    prim_files = sorted(args.run_dir.glob("*.prim_full.*.athdf"))
    uov_files = {path.name.split(".uov_full.")[-1]: path
                 for path in args.run_dir.glob("*.uov_full.*.athdf")}
    pairs = [(path, uov_files[path.name.split(".prim_full.")[-1]])
             for path in prim_files
             if path.name.split(".prim_full.")[-1] in uov_files]
    if not pairs:
        raise ValueError(f"No paired 3D prim/uov volumes in {args.run_dir}")

    records = []
    vertical_profiles = []
    optical_profiles = []
    tau_targets = np.unique(np.r_[np.logspace(-2, 5, 24),
                                  2.0/3.0, 1.0, 10.0, 100.0])
    last_map = None
    for prim_path, uov_path in pairs:
        prim, (x, y, z), time = read_volume(prim_path, "prim")
        uov, _, other_time = read_volume(uov_path, "user_out_var")
        if not np.isclose(time, other_time):
            raise ValueError("Paired full-volume times differ")
        rho = prim[0] * units["rho_unit"]
        temp = uov[2]
        if not ((rho > 0).all() and (temp > 0).all()):
            raise ValueError(f"Non-positive density or temperature in {prim_path}")
        query = np.stack((np.log10(rho), np.log10(temp)), axis=-1)
        sigma = 10.0 ** opacity(query) * rho * units["leng_unit"]
        dz = float(z[1] - z[0])
        if not np.allclose(np.diff(z), dz):
            raise ValueError("Expected uniform z spacing")
        tau = np.cumsum((sigma * dz)[::-1], axis=0)[::-1] - 0.5 * sigma * dz
        z23, temp23 = surface_at(tau, z, temp, 2.0 / 3.0)
        _, vz23 = surface_at(tau, z, prim[4], 2.0 / 3.0)
        coverage = float(np.mean(np.isfinite(z23)))
        if coverage == 0:
            raise ValueError(f"tau=2/3 lies outside the 3D volume in {prim_path}")
        velocity_unit = np.sqrt(units["egas_unit"] / units["rho_unit"])
        records.append({
            "time_code": time,
            "time_minutes": time * units["time_unit"] / 60.0,
            "coverage": coverage,
            "z_tau23_mean_code": float(np.nanmean(z23)),
            "z_tau23_std_code": float(np.nanstd(z23)),
            "T_tau23_mean_K": float(np.nanmean(temp23)),
            "T_tau23_std_K": float(np.nanstd(temp23)),
            "vz_tau23_rms_km_s": float(np.sqrt(np.nanmean(vz23**2))
                                         * velocity_unit / 1.0e5),
        })
        vertical_profiles.append({
            "time_code": time,
            "z_code": z.tolist(),
            "rho_mean_cgs": np.mean(rho, axis=(1, 2)).tolist(),
            "T_mean_K": np.mean(temp, axis=(1, 2)).tolist(),
            "T_std_K": np.std(temp, axis=(1, 2)).tolist(),
            "vz_rms_km_s": (np.sqrt(np.mean(prim[4]**2, axis=(1, 2)))
                              * velocity_unit / 1.0e5).tolist(),
            "qrad_mean_cgs": np.mean(uov[4], axis=(1, 2)).tolist(),
        })
        optical_temperature = []
        optical_density = []
        optical_coverage = []
        for target in tau_targets:
            z_at_tau, T_at_tau = surface_at(tau, z, temp, target)
            _, rho_at_tau = surface_at(tau, z, rho, target)
            coverage_at_tau = float(np.mean(np.isfinite(z_at_tau)))
            optical_coverage.append(coverage_at_tau)
            optical_temperature.append(float(np.nanmean(T_at_tau))
                                       if coverage_at_tau >= 0.95 else None)
            optical_density.append(float(np.nanmean(rho_at_tau))
                                   if coverage_at_tau >= 0.95 else None)
        optical_profiles.append({
            "time_code": time,
            "temperature_K": optical_temperature,
            "density_cgs": optical_density,
            "coverage": optical_coverage,
        })
        last_map = (x, y, temp23, vz23 * velocity_unit / 1.0e5, time)

    (args.run_dir / "surface3d.json").write_text(json.dumps(
        {"scope": "full uniform 3D domain", "records": records}, indent=2) + "\n")
    (args.run_dir / "profiles3d.json").write_text(json.dumps(
        {"scope": "horizontal means over full uniform 3D domain",
         "records": vertical_profiles}, indent=2) + "\n")
    profile_times = np.asarray([item["time_code"] for item in optical_profiles])
    late_start = profile_times[-1] - 0.25*(profile_times[-1] - profile_times[0])
    selected_optical = [item for item in optical_profiles
                        if item["time_code"] >= late_start]
    if len(selected_optical) < 2 and len(optical_profiles) >= 2:
        selected_optical = optical_profiles[-2:]
    late_optical = {
        "description": "Full 3D horizontal means on Rosseland optical-depth surfaces, "
                       "then temporal means; not a stationarity claim",
        "tau_targets": tau_targets.tolist(),
        "window_start_code": selected_optical[0]["time_code"],
        "window_end_code": selected_optical[-1]["time_code"],
        "snapshot_count": len(selected_optical),
        "records": optical_profiles,
    }
    for field in ("temperature_K", "density_cgs"):
        means, stds, counts = [], [], []
        for index in range(len(tau_targets)):
            values = [item[field][index] for item in selected_optical
                      if item[field][index] is not None]
            counts.append(len(values))
            means.append(float(np.mean(values)) if len(values) >= 2 else None)
            stds.append(float(np.std(values)) if len(values) >= 2 else None)
        late_optical[field] = {"temporal_mean": means,
                               "temporal_std": stds,
                               "snapshot_count": counts}
    reference = None
    if args.reference_profile is not None:
        raw = np.loadtxt(args.reference_profile, comments="#", ndmin=2)
        if raw.shape[1] < 5:
            raise ValueError("Expected solar profile with tau in column 3")
        raw = raw[np.isfinite(raw[:, 2]) & (raw[:, 2] > 0)]
        if len(raw) < 2:
            raise ValueError("Solar profile has too few positive tau values")
        order = np.argsort(raw[:, 2])
        reference = {
            "temperature_K": np.interp(np.log10(tau_targets),
                                         np.log10(raw[order, 2]), raw[order, 1]),
            "density_cgs": np.interp(np.log10(tau_targets),
                                      np.log10(raw[order, 2]), raw[order, 4]),
        }
        late_optical["reference_profile"] = {
            "description": "Imported solar 1D starting profile; not an independent observation",
            "path": str(args.reference_profile.resolve()),
            "temperature_K": reference["temperature_K"].tolist(),
            "density_cgs": reference["density_cgs"].tolist(),
        }
    (args.run_dir / "optical_profiles3d_late_mean.json").write_text(
        json.dumps(late_optical, indent=2) + "\n")
    images = args.run_dir / "imgs"
    images.mkdir(exist_ok=True)
    t = np.array([record["time_minutes"] for record in records])
    temp_mean = np.array([record["T_tau23_mean_K"] for record in records])
    temp_std = np.array([record["T_tau23_std_K"] for record in records])
    vz_rms = np.array([record["vz_tau23_rms_km_s"] for record in records])
    plt.rcParams.update({"font.size": 13})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, field, ylabel in ((axes[0], "temperature_K", "gas temperature [K]"),
                              (axes[1], "density_cgs", "density [g cm$^{-3}$]")):
        values = np.asarray([np.nan if x is None else x
                             for x in late_optical[field]["temporal_mean"]])
        scatter = np.asarray([np.nan if x is None else x
                              for x in late_optical[field]["temporal_std"]])
        valid = np.isfinite(values)
        ax.plot(tau_targets[valid], values[valid], label="late 3D mean")
        lower = values[valid] - scatter[valid]
        if field == "density_cgs":
            lower = np.maximum(lower, np.finfo(float).tiny)
        ax.fill_between(tau_targets[valid], lower,
                        values[valid] + scatter[valid], alpha=0.2,
                        label="snapshot std")
        if reference is not None:
            ax.plot(tau_targets, reference[field], color="black", ls="--",
                    label="initial solar 1D")
        ax.set(xscale="log", xlabel="Rosseland optical depth", ylabel=ylabel)
        if field == "density_cgs":
            ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Full 3D optical-depth profiles, "
                 f"t={late_optical['window_start_code']:.3g}–"
                 f"{late_optical['window_end_code']:.3g} code "
                 "(transient until stationarity passes)")
    fig.savefig(images / "simple_convection_optical_profiles3d_late_mean.png",
                dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                           constrained_layout=True)
    ax[0].plot(t, temp_mean)
    ax[0].fill_between(t, temp_mean-temp_std, temp_mean+temp_std, alpha=0.2)
    ax[0].axhline(args.solar_teff, color="black", lw=1, ls="--",
                  label=f"solar Teff = {args.solar_teff:g} K")
    ax[0].set(ylabel="T at Rosseland tau=2/3 [K]", title="Full 3D photospheric surface")
    ax[0].legend()
    ax[1].plot(t, vz_rms)
    ax[1].set(xlabel="physical time [min]", ylabel="vertical RMS speed [km/s]")
    for axis in ax:
        axis.grid(alpha=0.25)
    fig.savefig(images / "simple_convection_surface3d_history.png", dpi=180)
    plt.close(fig)

    x, y, temp23, vz23, time = last_map
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    im = ax[0].pcolormesh(x, y, temp23, shading="auto", cmap="inferno")
    fig.colorbar(im, ax=ax[0], label="K")
    ax[0].set(xlabel="x [code]", ylabel="y [code]", title="T at tau=2/3")
    scale = max(float(np.nanpercentile(np.abs(vz23), 99)), 1.0e-12)
    im = ax[1].pcolormesh(x, y, vz23, shading="auto", cmap="RdBu_r",
                          vmin=-scale, vmax=scale)
    fig.colorbar(im, ax=ax[1], label="km/s")
    ax[1].set(xlabel="x [code]", ylabel="y [code]", title="vertical velocity at tau=2/3")
    fig.suptitle(f"Full 3D tau=2/3 surface at t={time:.3g} code")
    fig.savefig(images / "simple_convection_surface3d_last.png", dpi=180)
    plt.close(fig)

    first, last = vertical_profiles[0], vertical_profiles[-1]
    fig, ax = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for label, profile in (("initial", first), ("latest", last)):
        z = np.asarray(profile["z_code"])
        ax[0, 0].plot(profile["T_mean_K"], z, label=label)
        ax[0, 1].plot(profile["rho_mean_cgs"], z, label=label)
        ax[1, 0].plot(profile["vz_rms_km_s"], z, label=label)
        ax[1, 1].plot(profile["qrad_mean_cgs"], z, label=label)
    ax[0, 0].set(xlabel="mean gas temperature [K]", ylabel="z [code]")
    ax[0, 1].set(xlabel="mean density [g cm$^{-3}$]", ylabel="z [code]",
                 xscale="log")
    ax[1, 0].set(xlabel="vertical RMS speed [km/s]", ylabel="z [code]")
    ax[1, 1].set(xlabel="mean radiative cooling [erg cm$^{-3}$ s$^{-1}$]",
                 ylabel="z [code]")
    for axis in ax.flat:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Full 3D horizontal mean: initial and latest snapshots")
    fig.savefig(images / "simple_convection_profiles3d.png", dpi=180)
    plt.close(fig)
    print(f"Wrote {args.run_dir / 'surface3d.json'}, profiles3d.json and plots")


if __name__ == "__main__":
    main()
