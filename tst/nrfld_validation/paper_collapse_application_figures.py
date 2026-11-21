#!/usr/bin/env python3
"""Create publication-oriented static figures for the collapse application."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

from paper_collapse_movie import TIME_UNIT_YR, raster_plane


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def field_index(hdf, name):
    names = [value.decode() for value in hdf.attrs["VariableNames"]]
    return names.index(name)


def extract_peak_track(records):
    rows = []
    for frame, record in enumerate(records):
        path = Path(record["path"])
        with h5py.File(path, "r") as hdf:
            hydro = hdf["hydro"]
            rho = np.asarray(hydro[field_index(hdf, "rho_cgs")], dtype=np.float64)
            temperature = np.asarray(
                hydro[field_index(hdf, "Tgas_K")], dtype=np.float64)
            peak = np.unravel_index(np.argmax(rho), rho.shape)
            peak_xyz = (float(hdf["x1v"][peak[0], peak[3]]),
                        float(hdf["x2v"][peak[0], peak[2]]),
                        float(hdf["x3v"][peak[0], peak[1]]))
            row = {
                "frame": frame,
                "cycle": int(hdf.attrs["NumCycles"]),
                "time_code": float(hdf.attrs["Time"]),
                "time_kyr": float(hdf.attrs["Time"])*TIME_UNIT_YR/1.0e3,
                "rho_peak_g_cm3": float(rho[peak]),
                "temperature_at_density_peak_K": float(temperature[peak]),
                "temperature_max_K": float(np.max(temperature)),
                "peak_x_au": peak_xyz[0], "peak_y_au": peak_xyz[1],
                "peak_z_au": peak_xyz[2],
                "amr_level_index_max": int(np.asarray(hdf["Levels"][:]).max()),
                "meshblocks": int(hdf.attrs["NumMeshBlocks"]),
                "source": str(path),
            }
        if row["cycle"] != record["cycle"]:
            raise RuntimeError("cycle mismatch for {}".format(path))
        rows.append(row)
        print("track {}/{} cycle {}".format(frame+1, len(records), row["cycle"]),
              flush=True)
    return rows


def write_track_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_temperature_density(rows, output):
    rho = np.asarray([row["rho_peak_g_cm3"] for row in rows])
    temperature = np.asarray(
        [row["temperature_at_density_peak_K"] for row in rows])
    time = np.asarray([row["time_kyr"] for row in rows])
    cycles = np.asarray([row["cycle"] for row in rows])

    figure, axis = plt.subplots(figsize=(6.6, 4.8), constrained_layout=True)
    axis.plot(rho, temperature, color="0.55", linewidth=1.3, zorder=1)
    points = axis.scatter(rho, temperature, c=time, cmap="viridis", s=24,
                          edgecolor="none", zorder=2)
    axis.scatter(rho[-1], temperature[-1], marker="*", s=160, color="#d62728",
                 edgecolor="black", linewidth=0.6, zorder=3)
    transition = int(np.where(cycles == 1800)[0][0])
    axis.scatter(rho[transition], temperature[transition], marker="D", s=42,
                 color="#ffbf00", edgecolor="black", linewidth=0.5, zorder=3)
    axis.axvline(1.0e-10, color="#d62728", linestyle="--", linewidth=1.0,
                 alpha=0.75, label=r"analysis endpoint: $\rho_{max}=10^{-10}$")
    axis.axhline(14.0, color="#1f77b4", linestyle=":", linewidth=1.0,
                 alpha=0.75, label="initial 14 K")
    axis.annotate("final\n{:.0f} K".format(temperature[-1]),
                  (rho[-1], temperature[-1]), xytext=(-54, -2),
                  textcoords="offset points", ha="right", va="center",
                  fontsize=9)
    axis.annotate("AMR+1 branch", (rho[transition], temperature[transition]),
                  xytext=(-14, 18), textcoords="offset points", ha="right",
                  fontsize=8)
    axis.set(xscale="log", yscale="log",
             xlim=(5.0e-18, 1.3e-10), ylim=(10.0, 450.0),
             xlabel=r"density maximum $\rho_{max}$ [g cm$^{-3}$]",
             ylabel=r"$T_{gas}$ at the density maximum [K]")
    axis.grid(True, which="both", alpha=0.20)
    axis.legend(loc="upper left", frameon=False, fontsize=8)
    colorbar = figure.colorbar(points, ax=axis, pad=0.02)
    colorbar.set_label("simulation time [kyr]")
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(output.with_suffix("."+suffix), dpi=240)
    plt.close(figure)


def load_final_slices(path, extent, pixels):
    with h5py.File(path, "r") as hdf:
        rho = np.asarray(hdf["hydro"][field_index(hdf, "rho_cgs")],
                         dtype=np.float64)
        temperature = np.asarray(
            hdf["hydro"][field_index(hdf, "Tgas_K")], dtype=np.float64)
        peak = np.unravel_index(np.argmax(rho), rho.shape)
        peak_xyz = np.asarray((hdf["x1v"][peak[0], peak[3]],
                               hdf["x2v"][peak[0], peak[2]],
                               hdf["x3v"][peak[0], peak[1]]), dtype=np.float64)
        faces = tuple(np.asarray(hdf[name][:], dtype=np.float64)
                      for name in ("x1f", "x2f", "x3f"))
        metadata = {
            "cycle": int(hdf.attrs["NumCycles"]),
            "time_kyr": float(hdf.attrs["Time"])*TIME_UNIT_YR/1.0e3,
            "rho_peak_g_cm3": float(rho[peak]),
            "temperature_at_density_peak_K": float(temperature[peak]),
            "peak_xyz_au": peak_xyz.tolist(),
        }
    fields = {"rho": rho, "temperature": temperature}
    grid, xy = raster_plane(fields, *faces, peak_xyz, extent, pixels, "xy")
    _, xz = raster_plane(fields, *faces, peak_xyz, extent, pixels, "xz")
    for plane in (xy, xz):
        for name, values in plane.items():
            if np.count_nonzero(np.isfinite(values)) != values.size:
                raise RuntimeError("unfilled {} slice pixels".format(name))
    return grid, xy, xz, metadata


def plot_spatial(path, extent, pixels, output):
    grid, xy, xz, metadata = load_final_slices(path, extent, pixels)
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 7.7), constrained_layout=True)
    rho_norm = colors.LogNorm(1.0e-16, 1.2e-10)
    temp_norm = colors.LogNorm(14.0, 360.0)
    panels = ((axes[0, 0], xy["rho"], rho_norm, "magma", "xy density", "y"),
              (axes[0, 1], xy["temperature"], temp_norm, "inferno",
               "xy gas temperature", "y"),
              (axes[1, 0], xz["rho"], rho_norm, "magma", "xz density", "z"),
              (axes[1, 1], xz["temperature"], temp_norm, "inferno",
               "xz gas temperature", "z"))
    density_image = None
    temperature_image = None
    for axis, values, norm, cmap, title, vertical in panels:
        image = axis.imshow(values, origin="lower", interpolation="nearest",
                            extent=(grid[0], grid[-1], grid[0], grid[-1]),
                            norm=norm, cmap=cmap, rasterized=True)
        axis.set(title=title, xlabel="x [au]", ylabel=vertical+" [au]",
                 aspect="equal")
        if "temperature" in title:
            density = xy["rho"] if title.startswith("xy") else xz["rho"]
            axis.contour(grid, grid, density, levels=(1.0e-14, 1.0e-12, 1.0e-10),
                         colors=("#69d2e7", "#e0e4cc", "white"),
                         linewidths=(0.6, 0.8, 1.0), alpha=0.85)
            temperature_image = image
        else:
            density_image = image
    rho_bar = figure.colorbar(density_image, ax=axes[:, 0], shrink=0.88, pad=0.02)
    rho_bar.set_label(r"density $\rho$ [g cm$^{-3}$]")
    temp_bar = figure.colorbar(temperature_image, ax=axes[:, 1], shrink=0.88,
                               pad=0.02)
    temp_bar.set_label(r"gas temperature $T_{gas}$ [K]")
    figure.suptitle(
        "cycle {cycle}, t={time_kyr:.3f} kyr, centered on density maximum; "
        "+/-{extent:g} au".format(extent=extent, **metadata), fontsize=11)
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(output.with_suffix("."+suffix), dpi=240)
    plt.close(figure)
    return metadata


def write_index(output, metadata):
    page = """<!doctype html><meta charset='utf-8'><title>Collapse paper application figures</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:2rem auto;line-height:1.5}}img{{width:100%;height:auto}}.note{{background:#fff4dd;padding:.8rem 1rem}}code{{background:#eee;padding:.1rem .25rem}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #bbb;padding:.4rem .55rem;text-align:left}}th{{background:#eee}}</style>
<h1>Protostellar-collapse paper application package</h1>
<p>Final AMR+1 state: cycle {cycle}, t={time_kyr:.3f} kyr,
&rho;<sub>max</sub>={rho_peak_g_cm3:.5e} g cm<sup>-3</sup>,
T at &rho;<sub>max</sub>={temperature_at_density_peak_K:.2f} K.</p>
<div class='note'><strong>Scoped numerical claim.</strong> Conservative AMR-FLD with a tabulated material EOS, variable opacity, rotation, and ideal MHD reaches the first-core density regime stably.  This endpoint is called a <em>first-core density candidate</em>; it is not presented as proof of a time-stationary hydrostatic object.</div>
<h2>Numerical gate</h2>
<table><tr><th>diagnostic</th><th>result</th></tr>
<tr><td>volume-weighted inner sonic transition</td><td>6.7384 au</td></tr>
<tr><td>enclosed mass at transition</td><td>0.0204525 M<sub>sun</sub></td></tr>
<tr><td>minimum Jeans resolution at endpoint</td><td>31.98 cells</td></tr>
<tr><td>AMR+1 sensitivity</td><td>&rho;<sub>max</sub> -0.76%; T -0.68%; M(&lt;5 au) -1.43%; radial &rho; L1 1.01%</td></tr>
<tr><td>implicit-solver audit</td><td>all Newton/FGMRES solves converged; no opacity-table excursions</td></tr></table>
<h2>Figure 1 candidate: final central spatial structure</h2>
<p><a href='final_spatial_distribution_20au.svg'><img src='final_spatial_distribution_20au.png' alt='Final density and temperature slices at 20 au'></a></p>
<p><a href='final_spatial_distribution_20au.pdf'>PDF</a> · <a href='final_spatial_distribution_20au.svg'>SVG</a></p>
<h2>Supplementary wider view: +/-50 au</h2>
<p><a href='final_spatial_distribution_50au.svg'><img src='final_spatial_distribution_50au.png' alt='Final density and temperature slices at 50 au'></a></p>
<p><a href='final_spatial_distribution_50au.pdf'>PDF</a> · <a href='final_spatial_distribution_50au.svg'>SVG</a></p>
<h2>Figure 2 candidate: central thermodynamic evolution</h2>
<p><a href='central_temperature_density.svg'><img src='central_temperature_density.png' alt='Temperature density evolution'></a></p>
<p><a href='central_temperature_density.pdf'>PDF</a> · <a href='central_temperature_density.svg'>SVG</a> · <a href='central_temperature_density.csv'>CSV</a></p>
<div class='note'><strong>Interpretation.</strong> The trajectory uses the gas temperature in the cell containing the density maximum, not the temperature maximum.  The spatial slices are centered on the final density maximum and are piecewise-constant displays of native AMR leaf cells.  The line from the lower-AMR trajectory to the AMR+1 continuation connects consecutive real snapshots and is not an interpolation in physical time.</div>
<h2>Comparison envelope</h2>
<table><tr><th>reference</th><th>comparison used here</th></tr>
<tr><td><a href='https://arxiv.org/html/2510.12620v4'>Mayer et al. (2025)</a></td><td>6--7 au first-core scale and 300--400 K central phase-space temperature near &rho;&gt;10<sup>-10</sup> g cm<sup>-3</sup>; the present 6.74 au and 336 K are consistent in scale.  The magnetic setup is deliberately weaker and uses ideal MHD, so detailed morphology is not claimed to be identical.</td></tr>
<tr><td><a href='https://doi.org/10.1051/0004-6361/201016213'>Commer&ccedil;on et al. (2011)</a></td><td>typical first-core radius about 8 au and mass about 0.02 M<sub>sun</sub>; the present values are 6.74 au and 0.02045 M<sub>sun</sub>.</td></tr>
<tr><td><a href='https://arxiv.org/abs/1206.3567'>Tomida et al.</a></td><td>context for the realistic-EOS radiation-MHD collapse sequence; not a one-to-one initial-condition comparison.</td></tr>
<tr><td><a href='https://arxiv.org/abs/1205.5143'>Vaytet et al.</a></td><td>supports grey transport as a controlled first-core-regime approximation below the high-temperature second-collapse phase.</td></tr></table>
<h2>Cost summary</h2>
<table><tr><th>scope</th><th>cost</th></tr>
<tr><td>final AMR+1 segment, 5 M-large-a nodes / 320 MPI ranks</td><td>1:04:33 wall; 5.38 node-hours; 610 cycles; 1.48&times;10<sup>9</sup> leaf-cell&times;cycle updates</td></tr>
<tr><td>actual initial-to-endpoint lineage</td><td>about 16.5 node-hours, about 1.05&times;10<sup>3</sup> rank-hours, 4.54&times;10<sup>9</sup> leaf-cell&times;cycle updates</td></tr>
<tr><td>production evidence including the matched-resolution comparison</td><td>about 20.4 node-hours, about 1.31&times;10<sup>3</sup> rank-hours</td></tr></table>
<h2>Existing quantitative and supplementary products</h2>
<ul>
<li><a href='../resolution_comparison/index.html'>matched-density AMR resolution comparison</a></li>
<li><a href='../additional_core_analysis/index.html'>core structure, support, beta, and anisotropy</a></li>
<li><a href='../movies/index.html'>2-D time evolution and final magnetic views</a></li>
<li><a href='../three_d/index.html'>final interactive 3-D views</a></li>
<li><a href='../three_d_evolution_v2/index.html'>3-D time evolution</a></li>
</ul>
<p><a href='figure_manifest.json'>figure provenance</a></p>
""".format(**metadata)
    (output/"index.html").write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("final_hdf", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pixels", type=int, default=1024)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads(args.manifest.read_text())
    rows = extract_peak_track(source_manifest["records"])
    write_track_csv(args.output/"central_temperature_density.csv", rows)
    plot_temperature_density(rows, args.output/"central_temperature_density")
    final20 = plot_spatial(args.final_hdf, 20.0, args.pixels,
                           args.output/"final_spatial_distribution_20au")
    final50 = plot_spatial(args.final_hdf, 50.0, args.pixels,
                           args.output/"final_spatial_distribution_50au")
    if final20 != final50:
        raise RuntimeError("final-slice metadata changed between scales")
    provenance = {
        "status": "completed", "frames": len(rows), "pixels": args.pixels,
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": sha256(args.manifest),
        "final_hdf": str(args.final_hdf),
        "final_hdf_sha256": sha256(args.final_hdf),
        "first_track_row": rows[0], "final_track_row": rows[-1],
        "final_snapshot": final20,
        "spatial_extents_au": [20.0, 50.0],
        "limitations": [
            "The thermodynamic track follows the cell containing rho_max, not Tmax.",
            "Snapshots are not uniformly spaced in physical time.",
            "Spatial panels resample native AMR leaf cells piecewise-constantly.",
            "The endpoint is a first-core density candidate, not a time-stationary proof."
        ]}
    (args.output/"figure_manifest.json").write_text(
        json.dumps(provenance, indent=2)+"\n")
    write_index(args.output, final20)


if __name__ == "__main__":
    main()
