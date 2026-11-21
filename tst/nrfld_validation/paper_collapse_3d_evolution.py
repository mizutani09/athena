#!/usr/bin/env python3
"""Render fixed-camera 3-D density and magnetic-field time evolution."""

import argparse
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pyvista as pv

from paper_collapse_3d import field, raster_cube


def load_source(path):
    """Load the AMR leaf-cell density and cell-centred magnetic field once."""
    with h5py.File(path, "r") as hdf:
        names = [value.decode() for value in hdf.attrs["VariableNames"]]
        hydro = hdf["hydro"]
        rho = field(hydro, names, "rho_cgs")
        bmag = field(hydro, names, "B_G")
        bx, by, bz = (field(hydro, names, name)
                      for name in ("Bcc1", "Bcc2", "Bcc3"))
        bcode = np.sqrt(bx*bx+by*by+bz*bz)
        conversion = np.divide(bmag, bcode, out=np.zeros_like(bmag),
                               where=bcode > 0.0)
        bx *= conversion
        by *= conversion
        bz *= conversion
        faces = tuple(np.asarray(hdf[name][:], dtype=np.float32)
                      for name in ("x1f", "x2f", "x3f"))
        metadata = {
            "cycle": int(hdf.attrs["NumCycles"]),
            "time_code": float(hdf.attrs["Time"]),
            "blocks": int(hdf.attrs["NumMeshBlocks"]),
            "level": int(np.asarray(hdf["Levels"][:]).max()),
            "rho_max": float(np.max(rho)),
            "B_max_G": float(np.max(bmag)),
        }
    return {"rho": rho, "bx": bx, "by": by, "bz": bz, "bmag": bmag}, faces, metadata


def make_grid(coordinates, spacing, data):
    samples = len(coordinates)
    grid = pv.ImageData(dimensions=(samples, samples, samples),
                        spacing=(spacing, spacing, spacing),
                        origin=(coordinates[0], coordinates[0], coordinates[0]))
    grid.point_data["logrho"] = np.log10(
        np.maximum(data["rho"], 1.0e-99)).ravel(order="F")
    grid.point_data["logB"] = np.log10(
        np.maximum(data["bmag"], 1.0e-99)).ravel(order="F")
    vectors = np.column_stack([data["bx"].ravel(order="F"),
                               data["by"].ravel(order="F"),
                               data["bz"].ravel(order="F")])
    magnitude = np.linalg.norm(vectors, axis=1)
    direction = np.divide(vectors, magnitude[:, None], out=np.zeros_like(vectors),
                          where=magnitude[:, None] > 0.0)
    grid.point_data["Bdir"] = direction
    grid.set_active_vectors("Bdir")
    return grid


def seed_points(extent):
    points = []
    for ring, radius in enumerate((0.10, 0.25, 0.50, 0.80)):
        count = 6 if ring < 2 else 8
        offset = (ring % 2)*np.pi/count
        for angle in np.linspace(0.0, 2.0*np.pi, count, endpoint=False)+offset:
            points.append((extent*radius*np.cos(angle),
                           extent*radius*np.sin(angle), 0.0))
    return pv.PolyData(np.asarray(points))


def field_lines(grid, extent):
    return grid.streamlines_from_source(
        seed_points(extent), vectors="Bdir", integration_direction="both",
        integrator_type=45, initial_step_length=0.45, step_unit="cl",
        min_step_length=0.08, max_step_length=1.0, max_steps=4000,
        terminal_speed=1.0e-8, max_error=2.0e-5, max_length=4.0*extent,
        compute_vorticity=False, interpolator_type="point")


def render(grid, extent, levels, label, frame_text, output):
    plotter = pv.Plotter(off_screen=True, window_size=(960, 720))
    existing = []
    for level in levels:
        surface = grid.contour([level], scalars="logrho", method="contour")
        if surface.n_cells:
            existing.append((level, surface))
    for index, (level, surface) in enumerate(existing):
        opacity = float(np.interp(level, (levels[0], levels[-1]), (0.16, 0.55)))
        plotter.add_mesh(
            surface, scalars="logrho", cmap="coolwarm", clim=(levels[0], levels[-1]),
            opacity=opacity, smooth_shading=True,
            show_scalar_bar=index == len(existing)-1,
            scalar_bar_args={"title": "log10 rho [g cm-3]", "vertical": False,
                             "position_x": 0.07, "position_y": 0.035,
                             "width": 0.39, "height": 0.07, "color": "white",
                             "title_font_size": 12, "label_font_size": 9,
                             "n_labels": 5})
    stream = field_lines(grid, extent)
    if stream.n_cells:
        tube = stream.tube(radius=0.0032*extent, n_sides=7)
        plotter.add_mesh(
            tube, scalars="logB", cmap="cool", clim=(-5.7, 0.3),
            smooth_shading=True,
            scalar_bar_args={"title": "log10 |B| [G]", "vertical": True,
                             "position_x": 0.89, "position_y": 0.13,
                             "width": 0.07, "height": 0.40, "color": "white",
                             "title_font_size": 12, "label_font_size": 9,
                             "n_labels": 5})
    plotter.add_mesh(
        pv.Box(bounds=(-extent, extent, -extent, extent, -extent, extent)),
        style="wireframe", color="gray", opacity=0.20, line_width=1)
    plotter.add_text("{}: +/-{:g} au".format(label, extent), position="upper_left",
                     font_size=11, color="white")
    plotter.add_text(frame_text, position="upper_right", font_size=9, color="white")
    if not existing:
        plotter.add_text("density below first displayed contour", position="lower_left",
                         font_size=9, color="lightgray")
    plotter.set_background("#05060a")
    plotter.camera_position = [(1.55*extent, -1.75*extent, 1.25*extent),
                               (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    plotter.camera.zoom(1.12)
    plotter.show(screenshot=str(output), auto_close=True)
    return int(stream.n_cells), int(stream.n_points), len(existing)


def encode_frames(output, fps):
    common = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
              "-movflags", "+faststart"]
    for label in ("global", "core"):
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
             "-i", str(output/label/"%04d.png"), *common,
             str(output/(label+"_3d_evolution.mp4"))], check=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
         "-i", str(output/"global"/"%04d.png"), "-framerate", str(fps),
         "-i", str(output/"core"/"%04d.png"), "-filter_complex",
         "hstack=inputs=2", *common, str(output/"global_core_3d_evolution.mp4")],
        check=True)


def write_index(output, summary):
    page = """<!doctype html><meta charset='utf-8'><title>3-D collapse time evolution</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:2rem auto;line-height:1.5;background:#f7f7f7}}video,img{{width:100%;height:auto}}.note{{background:#fff4dd;padding:.8rem 1rem}}</style>
<h1>3-D protostellar-collapse time evolution</h1>
<p>{frames} real HDF snapshots at {fps} frames/s ({duration:.1f} s).  Frame labels give the actual simulation time, cycle, and native peak density.</p>
<h2>Global and core views</h2>
<video controls preload='metadata'><source src='global_core_3d_evolution.mp4' type='video/mp4'></video>
<p><a href='global_3d_evolution.mp4'>global-only video</a> · <a href='core_3d_evolution.mp4'>core-only video</a> · <a href='evolution_manifest.json'>frame manifest</a></p>
<div class='note'><strong>Timing and scope.</strong> Frames are saved simulation snapshots shown at equal playback intervals; they are not uniformly spaced in physical time. Density surfaces and magnetic trajectories use uniformly resampled cell-centred data. Use native HDF/history diagnostics for magnetic flux, divB, reconnection, and quantitative topology.</div>
<p><a href='../three_d/index.html'>final-state interactive 3-D views</a> · <a href='../movies/index.html'>2-D time evolution</a> · <a href='../additional_core_analysis/index.html'>core diagnostics</a></p>
""".format(frames=summary["frames"], fps=summary["fps"],
           duration=summary["frames"]/summary["fps"])
    (output/"index.html").write_text(page)


def select_records(records, maximum):
    if not maximum or maximum >= len(records):
        return records
    indices = np.linspace(0, len(records)-1, maximum, dtype=int)
    return [records[index] for index in indices]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    pv.OFF_SCREEN = True
    source_manifest = json.loads(args.manifest.read_text())
    records = select_records(source_manifest["records"], args.max_frames)
    args.output.mkdir(parents=True, exist_ok=True)
    for label in ("global", "core"):
        (args.output/label).mkdir(parents=True, exist_ok=True)
    rendered = []
    for frame_index, record in enumerate(records):
        fields, faces, metadata = load_source(record["path"])
        if metadata["cycle"] != record["cycle"]:
            raise RuntimeError("cycle mismatch for {}".format(record["path"]))
        frame_text = ("t={:.5g} kyr  cycle={}\nlog10 rho_max={:.3f}"
                      .format(record["time_kyr"], metadata["cycle"],
                              np.log10(metadata["rho_max"])))
        outcome = {"frame": frame_index, "path": record["path"],
                   "time_kyr": record["time_kyr"], **metadata}
        for label, extent, levels in (
                ("global", 2500.0, (-18.8, -17.8, -16.0, -13.5, -10.2)),
                ("core", 50.0, (-16.0, -14.5, -13.0, -11.5, -10.1))):
            coordinates, spacing, data = raster_cube(fields, faces, extent,
                                                      args.samples)
            grid = make_grid(coordinates, spacing, data)
            lines, points, surfaces = render(
                grid, extent, levels, label.title(), frame_text,
                args.output/label/("{:04d}.png".format(frame_index)))
            outcome[label] = {"spacing_au": spacing, "streamline_cells": lines,
                              "streamline_points": points,
                              "density_surfaces": surfaces,
                              "sampled_rho_max": float(np.max(data["rho"])),
                              "sampled_B_max_G": float(np.max(data["bmag"]))}
        rendered.append(outcome)
        print("rendered {}/{} cycle {}".format(
            frame_index+1, len(records), metadata["cycle"]), flush=True)
    encode_frames(args.output, args.fps)
    summary = {
        "status": "completed", "frames": len(rendered), "fps": args.fps,
        "samples_per_axis": args.samples, "global_extent_au": 2500.0,
        "core_extent_au": 50.0, "records": rendered,
        "limitations": [
            "Snapshots are equally spaced in playback, not physical time.",
            "AMR leaf-cell values are sampled piecewise-constantly onto uniform display lattices.",
            "Magnetic trajectories integrate normalized sampled cell-centred B, not face-centred CT fluxes."
        ]}
    (args.output/"evolution_manifest.json").write_text(
        json.dumps(summary, indent=2)+"\n")
    write_index(args.output, summary)


if __name__ == "__main__":
    main()
