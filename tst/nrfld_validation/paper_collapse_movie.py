#!/usr/bin/env python3
"""Render AMR collapse slices and magnetic-field-line movies from Athena HDF5."""

import argparse
import json
import subprocess
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np


AU = 1.495978707e13
YEAR = 3.15576e7
VELOCITY_UNIT = np.sqrt(1.0e-9/1.0e-18)
TIME_UNIT_YR = AU/VELOCITY_UNIT/YEAR


def selected_plane_index(faces, coordinate):
    """Return the cell containing coordinate, using a deterministic half-open interval."""
    if coordinate < faces[0] or coordinate >= faces[-1]:
        return None
    index = int(np.searchsorted(faces, coordinate, side="right")-1)
    return min(max(index, 0), len(faces)-2)


def raster_plane(fields, face1, face2, face3, center, extent, pixels, plane):
    """Piecewise-constant raster of leaf cells intersecting an xy or xz plane."""
    if plane == "xy":
        ufaces, vfaces, nfaces = face1, face2, face3
        ucenter, vcenter, normal = center[0], center[1], center[2]
    elif plane == "xz":
        ufaces, vfaces, nfaces = face1, face3, face2
        ucenter, vcenter, normal = center[0], center[2], center[1]
    else:
        raise ValueError(plane)

    relative = np.linspace(-extent+extent/pixels, extent-extent/pixels, pixels)
    ugrid = relative+ucenter
    vgrid = relative+vcenter
    outputs = {name: np.full((pixels, pixels), np.nan, dtype=np.float64)
               for name in fields}
    for block in range(nfaces.shape[0]):
        normal_index = selected_plane_index(nfaces[block], normal)
        if normal_index is None:
            continue
        use_u = np.where((ugrid >= ufaces[block, 0])
                         & (ugrid < ufaces[block, -1]))[0]
        use_v = np.where((vgrid >= vfaces[block, 0])
                         & (vgrid < vfaces[block, -1]))[0]
        if not len(use_u) or not len(use_v):
            continue
        iu = np.searchsorted(ufaces[block], ugrid[use_u], side="right")-1
        iv = np.searchsorted(vfaces[block], vgrid[use_v], side="right")-1
        iu = np.clip(iu, 0, len(ufaces[block])-2)
        iv = np.clip(iv, 0, len(vfaces[block])-2)
        for name, values in fields.items():
            if plane == "xy":
                source = values[block, normal_index]
            else:
                source = values[block, :, normal_index, :]
            outputs[name][np.ix_(use_v, use_u)] = source[np.ix_(iv, iu)]
    return relative, outputs


def read_snapshot(path, overview_extent, core_extent, pixels):
    with h5py.File(path, "r") as hdf:
        names = [value.decode() for value in hdf.attrs["VariableNames"]]
        hydro = hdf["hydro"]

        def field(name):
            return np.asarray(hydro[names.index(name)], dtype=np.float64)

        rho = field("rho_cgs")
        peak = np.unravel_index(np.argmax(rho), rho.shape)
        peak_center = np.array([hdf["x1v"][peak[0], peak[3]],
                                hdf["x2v"][peak[0], peak[2]],
                                hdf["x3v"][peak[0], peak[1]]], dtype=np.float64)
        center = np.zeros(3, dtype=np.float64)
        temperature = field("Tgas_K")
        bmag = field("B_G")
        bx, by, bz = (field("Bcc1"), field("Bcc2"), field("Bcc3"))
        bcode = np.sqrt(bx*bx+by*by+bz*bz)
        conversion = np.divide(bmag, bcode, out=np.zeros_like(bmag), where=bcode > 0.0)
        bx *= conversion
        by *= conversion
        bz *= conversion
        x1f = np.asarray(hdf["x1f"][:], dtype=np.float64)
        x2f = np.asarray(hdf["x2f"][:], dtype=np.float64)
        x3f = np.asarray(hdf["x3f"][:], dtype=np.float64)
        cycle = int(hdf.attrs["NumCycles"])
        time_code = float(hdf.attrs["Time"])
        level = int(np.asarray(hdf["Levels"][:]).max())
        blocks = int(hdf.attrs["NumMeshBlocks"])

    relative_global, global_xy = raster_plane(
        {"rho": rho}, x1f, x2f, x3f, center, overview_extent, pixels, "xy")
    core_fields = {"rho": rho, "temperature": temperature,
                   "bx": bx, "by": by, "bz": bz, "bmag": bmag}
    relative_core, core_xy = raster_plane(
        core_fields, x1f, x2f, x3f, center, core_extent, pixels, "xy")
    _, core_xz = raster_plane(
        core_fields, x1f, x2f, x3f, center, core_extent, pixels, "xz")
    return {
        "path": str(path), "cycle": cycle, "time_code": time_code,
        "time_kyr": time_code*TIME_UNIT_YR/1.0e3, "level": level,
        "blocks": blocks, "slice_center": center.tolist(),
        "density_peak_center": peak_center.tolist(),
        "rho_peak": float(rho[peak]), "temperature_peak": float(temperature[peak]),
        "global_grid": relative_global, "global_xy": global_xy,
        "core_grid": relative_core, "core_xy": core_xy, "core_xz": core_xz,
    }


def annotation(snapshot):
    return ("cycle {:d}   t={:.3f} kyr   "
            r"$\rho_{{max}}$={:.3e} g cm$^{{-3}}$   AMR index {:d}").format(
                snapshot["cycle"], snapshot["time_kyr"], snapshot["rho_peak"],
                snapshot["level"])


def streamlines(axis, grid, bx, by, bmag):
    stride = max(1, len(grid)//128)
    x = grid[::stride]
    u = bx[::stride, ::stride]
    v = by[::stride, ::stride]
    magnitude = bmag[::stride, ::stride]
    good = np.isfinite(u) & np.isfinite(v) & np.isfinite(magnitude) & (magnitude > 0.0)
    scale = np.hypot(u, v)
    u = np.divide(u, scale, out=np.zeros_like(u), where=good & (scale > 0.0))
    v = np.divide(v, scale, out=np.zeros_like(v), where=good & (scale > 0.0))
    color = magnitude.copy()
    color[~good] = np.nan
    return axis.streamplot(x, x, u, v, color=color, cmap="cool",
                           norm=colors.LogNorm(1.0e-5, 1.0), density=1.15,
                           linewidth=0.65, arrowsize=0.55, minlength=0.15)


def render_overview(snapshot, output):
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.25), constrained_layout=True)
    density_norm = colors.LogNorm(1.0e-18, 2.0e-10)
    core_norm = colors.LogNorm(1.0e-17, 2.0e-10)
    temp_norm = colors.LogNorm(10.0, 400.0)
    panels = ((snapshot["global_xy"]["rho"], density_norm, "magma", "Density: global"),
              (snapshot["core_xy"]["rho"], core_norm, "magma", "Density: central 100 au"),
              (snapshot["core_xy"]["temperature"], temp_norm, "inferno", "Gas temperature"))
    extents = (snapshot["global_grid"], snapshot["core_grid"], snapshot["core_grid"])
    labels = (r"$\rho$ [g cm$^{-3}$]", r"$\rho$ [g cm$^{-3}$]", "$T_{gas}$ [K]")
    for axis, (values, norm, cmap, title), grid, label in zip(axes, panels, extents, labels):
        image = axis.imshow(values, origin="lower",
                            extent=(grid[0], grid[-1], grid[0], grid[-1]),
                            norm=norm, cmap=cmap, interpolation="nearest")
        axis.set(title=title, xlabel="x [au]", ylabel="y [au]",
                 aspect="equal")
        figure.colorbar(image, ax=axis, label=label, shrink=0.86)
    figure.suptitle(annotation(snapshot), fontsize=11)
    figure.savefig(output, dpi=140)
    plt.close(figure)


def render_magnetic(snapshot, output, static_svg=None):
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))
    figure.subplots_adjust(left=0.07, right=0.87, bottom=0.27, top=0.82, wspace=0.22)
    norm = colors.LogNorm(1.0e-17, 2.0e-10)
    grid = snapshot["core_grid"]
    planes = ((snapshot["core_xy"], "xy midplane", "y [au]", "bx", "by"),
              (snapshot["core_xz"], "xz plane", "z [au]", "bx", "bz"))
    stream = None
    for axis, (plane, title, ylabel, bu, bv) in zip(axes, planes):
        image = axis.imshow(plane["rho"], origin="lower",
            extent=(grid[0], grid[-1], grid[0], grid[-1]), norm=norm,
            cmap="magma", interpolation="nearest")
        stream = streamlines(axis, grid, plane[bu], plane[bv], plane["bmag"])
        axis.set(title=title, xlabel="x [au]", ylabel=ylabel, aspect="equal")
    density_axis = figure.add_axes((0.19, 0.105, 0.56, 0.045))
    density_bar = figure.colorbar(image, cax=density_axis, orientation="horizontal")
    density_bar.set_label(r"$\rho$ [g cm$^{-3}$]")
    magnetic_axis = figure.add_axes((0.90, 0.25, 0.018, 0.50))
    magnetic_bar = figure.colorbar(stream.lines, cax=magnetic_axis)
    magnetic_bar.set_label(r"$|B|$ [G]")
    figure.suptitle(annotation(snapshot), fontsize=11, y=0.96)
    figure.savefig(output, dpi=150)
    if static_svg is not None:
        figure.savefig(static_svg)
    plt.close(figure)


def encode_movie(frames, output, fps):
    command = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
               "-i", str(frames/"%05d.png"),
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
               "-pix_fmt", "yuv420p",
               "-crf", "20", "-movflags", "+faststart", str(output)]
    subprocess.run(command, check=True)


def write_html(output, manifest):
    page = """<!doctype html><meta charset='utf-8'><title>Collapse movies and magnetic field lines</title>
<style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;line-height:1.5}video,img{width:100%;height:auto}.note{background:#fff4dd;padding:.8rem 1rem}code{background:#eee;padding:.1rem .25rem}</style>
<h1>Rotating, magnetized protostellar-collapse movies</h1>
<h2>Density and temperature evolution</h2>
<video controls preload='metadata'><source src='collapse_density_temperature.mp4' type='video/mp4'></video>
<h2>Central density with magnetic field lines</h2>
<video controls preload='metadata'><source src='collapse_core_magnetic.mp4' type='video/mp4'></video>
<h2>Final magnetic field-line view</h2>
<a href='final_magnetic_field_lines.svg'><img src='final_magnetic_field_lines.png' alt='Final magnetic field lines'></a>
<h2>Final envelope-scale magnetic field-line view</h2>
<a href='final_magnetic_field_lines_envelope.svg'><img src='final_magnetic_field_lines_envelope.png' alt='Envelope-scale magnetic field lines'></a>
<div class='note'><strong>Visualization scope.</strong> Each panel is centered on the fixed initial core origin; the separately recorded density-peak position remains within the manifest. AMR leaf-cell values are sampled piecewise-constantly onto a uniform display raster. Streamlines use the sampled cell-centred magnetic-field direction and are qualitative; quantitative values remain those in the original HDF files. The lower-AMR trajectory is followed by the AMR+1 trajectory near first-core density.</div>
<p><a href='movie_manifest.json'>frame manifest and provenance</a> · <a href='../three_d/index.html'>3-D density and magnetic field</a> · <a href='../resolution_comparison/index.html'>resolution analysis</a> · <a href='../additional_core_analysis/index.html'>core diagnostics</a></p>
"""
    (output/"index.html").write_text(page)
    (output/"movie_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--input-list", type=Path,
                         help="Text file containing one local HDF path per line")
    sources.add_argument("--input-dir", type=Path, action="append",
                         help="Directory of HDF files; may be supplied more than once")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overview-extent", type=float, default=3500.0)
    parser.add_argument("--core-extent", type=float, default=50.0)
    parser.add_argument("--field-envelope-extent", type=float, default=300.0)
    parser.add_argument("--pixels", type=int, default=512)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    if args.input_list is not None:
        paths = [Path(line.strip()) for line in args.input_list.read_text().splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
    else:
        candidates = []
        for directory in args.input_dir:
            candidates.extend(directory.glob("*.athdf"))
        by_cycle = {}
        for path in sorted(candidates):
            try:
                with h5py.File(path, "r") as hdf:
                    cycle = int(hdf.attrs["NumCycles"])
            except OSError as error:
                print("skip incomplete HDF {}: {}".format(path, error), flush=True)
                continue
            by_cycle[cycle] = path
        paths = [by_cycle[key] for key in sorted(by_cycle)]
    if not paths:
        raise SystemExit("empty input list")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("missing inputs: "+", ".join(missing[:5]))
    overview_frames = args.output/"frames_overview"
    magnetic_frames = args.output/"frames_magnetic"
    overview_frames.mkdir(parents=True, exist_ok=True)
    magnetic_frames.mkdir(parents=True, exist_ok=True)
    records = []
    previous_cycle = -1
    for frame, path in enumerate(paths):
        snapshot = read_snapshot(path, args.overview_extent, args.core_extent, args.pixels)
        if snapshot["cycle"] <= previous_cycle:
            raise RuntimeError("input cycles must increase: {} after {}".format(
                snapshot["cycle"], previous_cycle))
        previous_cycle = snapshot["cycle"]
        render_overview(snapshot, overview_frames/("{:05d}.png".format(frame)))
        render_magnetic(snapshot, magnetic_frames/("{:05d}.png".format(frame)),
                        args.output/"final_magnetic_field_lines.svg" if frame == len(paths)-1 else None)
        if frame == len(paths)-1:
            render_magnetic(snapshot, args.output/"final_magnetic_field_lines.png")
        records.append({key: snapshot[key] for key in
                        ("path", "cycle", "time_code", "time_kyr", "level", "blocks",
                         "slice_center", "density_peak_center", "rho_peak",
                         "temperature_peak")})
        print("frame {}/{} cycle={} rho_max={:.4e}".format(
            frame+1, len(paths), snapshot["cycle"], snapshot["rho_peak"]), flush=True)
    encode_movie(overview_frames, args.output/"collapse_density_temperature.mp4", args.fps)
    encode_movie(magnetic_frames, args.output/"collapse_core_magnetic.mp4", args.fps)
    envelope = read_snapshot(paths[-1], args.overview_extent,
                             args.field_envelope_extent, args.pixels)
    render_magnetic(envelope, args.output/"final_magnetic_field_lines_envelope.png",
                    args.output/"final_magnetic_field_lines_envelope.svg")
    manifest = {"status": "completed", "frames": len(records), "fps": args.fps,
                "overview_extent_au": args.overview_extent,
                "core_extent_au": args.core_extent, "pixels": args.pixels,
                "field_envelope_extent_au": args.field_envelope_extent,
                "time_unit_year": TIME_UNIT_YR, "records": records}
    write_html(args.output, manifest)


if __name__ == "__main__":
    main()
