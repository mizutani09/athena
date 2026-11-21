#!/usr/bin/env python3
"""Create 3-D density and magnetic-field visualizations from an Athena HDF."""

import argparse
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pyvista as pv


def field(hydro, names, name):
    return np.asarray(hydro[names.index(name)], dtype=np.float32)


def raster_cube(fields, faces, extent, samples):
    """Sample AMR leaf cells piecewise-constantly onto an xyz point lattice."""
    spacing = 2.0*extent/samples
    coordinates = np.linspace(-extent+0.5*spacing, extent-0.5*spacing, samples)
    outputs = {name: np.full((samples, samples, samples), np.nan, dtype=np.float32)
               for name in fields}
    x1f, x2f, x3f = faces
    for block in range(x1f.shape[0]):
        use_x = np.where((coordinates >= x1f[block, 0])
                         & (coordinates < x1f[block, -1]))[0]
        use_y = np.where((coordinates >= x2f[block, 0])
                         & (coordinates < x2f[block, -1]))[0]
        use_z = np.where((coordinates >= x3f[block, 0])
                         & (coordinates < x3f[block, -1]))[0]
        if not len(use_x) or not len(use_y) or not len(use_z):
            continue
        ix = np.clip(np.searchsorted(x1f[block], coordinates[use_x], side="right")-1,
                     0, x1f.shape[1]-2)
        iy = np.clip(np.searchsorted(x2f[block], coordinates[use_y], side="right")-1,
                     0, x2f.shape[1]-2)
        iz = np.clip(np.searchsorted(x3f[block], coordinates[use_z], side="right")-1,
                     0, x3f.shape[1]-2)
        for name, values in fields.items():
            # Athena data are (block,z,y,x); VTK point arrays are kept as (x,y,z).
            sample = values[block][np.ix_(iz, iy, ix)].transpose(2, 1, 0)
            outputs[name][np.ix_(use_x, use_y, use_z)] = sample
    missing = {name: int(np.count_nonzero(~np.isfinite(values)))
               for name, values in outputs.items()}
    if any(missing.values()):
        raise RuntimeError("unfilled display voxels: {}".format(missing))
    return coordinates, spacing, outputs


def read_and_sample(path, extent, samples):
    with h5py.File(path, "r") as hdf:
        names = [value.decode() for value in hdf.attrs["VariableNames"]]
        hydro = hdf["hydro"]
        rho = field(hydro, names, "rho_cgs")
        temperature = field(hydro, names, "Tgas_K")
        bmag = field(hydro, names, "B_G")
        bx, by, bz = (field(hydro, names, name)
                      for name in ("Bcc1", "Bcc2", "Bcc3"))
        bcode = np.sqrt(bx*bx+by*by+bz*bz)
        conversion = np.divide(bmag, bcode, out=np.zeros_like(bmag), where=bcode > 0.0)
        bx *= conversion
        by *= conversion
        bz *= conversion
        faces = tuple(np.asarray(hdf[name][:], dtype=np.float32)
                      for name in ("x1f", "x2f", "x3f"))
        cycle = int(hdf.attrs["NumCycles"])
        time_code = float(hdf.attrs["Time"])
        source_blocks = int(hdf.attrs["NumMeshBlocks"])
        source_level = int(np.asarray(hdf["Levels"][:]).max())
        source_rho_max = float(np.max(rho))
        source_b_max = float(np.max(bmag))
    coordinates, spacing, data = raster_cube(
        {"rho": rho, "temperature": temperature, "bx": bx, "by": by,
         "bz": bz, "bmag": bmag}, faces, extent, samples)
    return coordinates, spacing, data, {
        "cycle": cycle, "time_code": time_code, "source_blocks": source_blocks,
        "source_level_index_max": source_level, "source_rho_max_g_cm3": source_rho_max,
        "source_B_max_G": source_b_max,
    }


def make_grid(coordinates, spacing, data):
    samples = len(coordinates)
    grid = pv.ImageData(dimensions=(samples, samples, samples),
                        spacing=(spacing, spacing, spacing),
                        origin=(coordinates[0], coordinates[0], coordinates[0]))
    logrho = np.log10(np.maximum(data["rho"], 1.0e-99))
    logb = np.log10(np.maximum(data["bmag"], 1.0e-99))
    grid.point_data["logrho"] = logrho.ravel(order="F")
    grid.point_data["temperature_K"] = data["temperature"].ravel(order="F")
    grid.point_data["logB"] = logb.ravel(order="F")
    vectors = np.column_stack([data["bx"].ravel(order="F"),
                               data["by"].ravel(order="F"),
                               data["bz"].ravel(order="F")])
    magnitude = np.linalg.norm(vectors, axis=1)
    direction = np.divide(vectors, magnitude[:, None], out=np.zeros_like(vectors),
                          where=magnitude[:, None] > 0.0)
    grid.point_data["B_G"] = vectors
    grid.point_data["Bdir"] = direction
    grid.set_active_vectors("Bdir")
    return grid


def seeds(extent):
    points = []
    radii = np.asarray((0.06, 0.12, 0.22, 0.40, 0.65, 0.86))*extent
    for ring, radius in enumerate(radii):
        count = 8 if ring < 4 else 12
        offset = (ring % 2)*np.pi/count
        for angle in np.linspace(0.0, 2.0*np.pi, count, endpoint=False)+offset:
            points.append((radius*np.cos(angle), radius*np.sin(angle), 0.0))
    points.extend(((0.0, 0.0, 0.08*extent), (0.0, 0.0, -0.08*extent)))
    return pv.PolyData(np.asarray(points))


def field_lines(grid, extent):
    stream = grid.streamlines_from_source(
        seeds(extent), vectors="Bdir", integration_direction="both",
        integrator_type=45, initial_step_length=0.35, step_unit="cl",
        min_step_length=0.05, max_step_length=1.0, max_steps=6000,
        terminal_speed=1.0e-8, max_error=1.0e-5, max_length=4.0*extent,
        compute_vorticity=False, interpolator_type="point")
    if stream.n_cells == 0:
        raise RuntimeError("VTK produced no magnetic streamlines")
    return stream


def density_surfaces(grid, levels):
    surfaces = []
    for level in levels:
        surface = grid.contour([level], scalars="logrho", method="contour")
        if surface.n_cells:
            surfaces.append((level, surface))
    if not surfaces:
        raise RuntimeError("density levels do not intersect the sampled cube")
    return surfaces


def scalar_bar(title, vertical, position):
    arguments = {"title": title, "color": "white", "title_font_size": 14,
                 "label_font_size": 11, "n_labels": 5}
    if vertical:
        arguments.update({"vertical": True, "position_x": position[0],
                          "position_y": position[1], "width": 0.07, "height": 0.42})
    else:
        arguments.update({"vertical": False, "position_x": position[0],
                          "position_y": position[1], "width": 0.38, "height": 0.07})
    return arguments


def add_isosurface_scene(plotter, surfaces, stream, extent, title):
    levels = [level for level, _ in surfaces]
    lo, hi = min(levels), max(levels)
    for index, (level, surface) in enumerate(surfaces):
        opacity = np.interp(level, (lo, hi), (0.08, 0.46))
        plotter.add_mesh(surface, scalars="logrho", cmap="magma", clim=(lo, hi),
                         opacity=float(opacity), smooth_shading=True,
                         show_scalar_bar=index == len(surfaces)-1,
                         scalar_bar_args=scalar_bar("log10 rho [g cm-3]", False,
                                                    (0.08, 0.04)))
    tube = stream.tube(radius=0.0035*extent, n_sides=8)
    plotter.add_mesh(tube, scalars="logB", cmap="cool", clim=(-5.0, 0.3),
                     smooth_shading=True,
                     scalar_bar_args=scalar_bar("log10 |B| [G]", True, (0.90, 0.12)))
    plotter.add_mesh(pv.Box(bounds=(-extent, extent, -extent, extent, -extent, extent)),
                     style="wireframe", color="gray", opacity=0.22, line_width=1)
    plotter.add_text(title, position="upper_edge", font_size=12, color="white")
    plotter.set_background("#05060a")
    plotter.camera_position = [(1.55*extent, -1.75*extent, 1.25*extent),
                               (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    plotter.camera.zoom(1.15)
    return tube


def render_isosurfaces(grid, surfaces, stream, extent, title, output, orbit_frames, fps):
    plotter = pv.Plotter(off_screen=True, window_size=(1280, 960))
    tube = add_isosurface_scene(plotter, surfaces, stream, extent, title)
    plotter.show(screenshot=str(output/"density_fieldlines_3d.png"), auto_close=False)
    plotter.export_html(str(output/"density_fieldlines_3d_interactive.html"))
    frames = output/"orbit_frames"
    frames.mkdir(parents=True, exist_ok=True)
    plotter.render()
    for frame in range(orbit_frames):
        if frame:
            plotter.camera.azimuth += 360.0/orbit_frames
        plotter.render()
        plotter.screenshot(str(frames/("{:04d}.png".format(frame))))
    plotter.close()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", str(frames/"%04d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
                    str(output/"density_fieldlines_3d_orbit.mp4")], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate",
                    str(fps/4.0), "-i", str(frames/"%04d.png"), "-r", str(fps),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                    "-movflags", "+faststart",
                    str(output/"density_fieldlines_3d_orbit_slow.mp4")], check=True)
    return tube


def render_volume(grid, stream, extent, title, output):
    plotter = pv.Plotter(off_screen=True, window_size=(1280, 960))
    transfer = np.interp(np.linspace(-17.0, -9.8, 256),
                         (-17.0, -14.2, -13.5, -12.5, -11.5, -10.5, -9.8),
                         (0.0, 0.0, 0.002, 0.015, 0.08, 0.42, 0.85))
    # A full 256-element PyVista transfer function is interpreted as uint8
    # opacity directly; convert [0,1] here rather than silently rounding it
    # to a transparent array.
    transfer = np.rint(255.0*transfer).astype(np.uint8)
    plotter.add_volume(grid, scalars="logrho", cmap="magma", clim=(-17.0, -9.8),
                       opacity=transfer, shade=True, blending="composite",
                       scalar_bar_args=scalar_bar("log10 rho [g cm-3]", False,
                                                  (0.08, 0.04)))
    tube = stream.tube(radius=0.003*extent, n_sides=8)
    plotter.add_mesh(tube, scalars="logB", cmap="cool", clim=(-5.0, 0.3),
                     smooth_shading=True,
                     scalar_bar_args=scalar_bar("log10 |B| [G]", True, (0.90, 0.12)))
    plotter.add_text(title, position="upper_edge", font_size=12, color="white")
    plotter.set_background("#05060a")
    plotter.camera_position = [(1.55*extent, -1.75*extent, 1.25*extent),
                               (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    plotter.camera.zoom(1.15)
    plotter.show(screenshot=str(output/"density_volume_fieldlines_3d.png"), auto_close=True)


def build_case(path, output, extent, samples, levels, orbit_frames, fps):
    output.mkdir(parents=True, exist_ok=True)
    coordinates, spacing, data, metadata = read_and_sample(path, extent, samples)
    grid = make_grid(coordinates, spacing, data)
    stream = field_lines(grid, extent)
    surfaces = density_surfaces(grid, levels)
    title = ("cycle {cycle}: density and magnetic field, +/-{extent:g} au"
             .format(cycle=metadata["cycle"], extent=extent))
    tube = render_isosurfaces(grid, surfaces, stream, extent, title, output,
                              orbit_frames, fps)
    render_volume(grid, stream, extent, title, output)
    grid.save(output/"resampled_density_B.vti", binary=True)
    stream.save(output/"magnetic_streamlines.vtp", binary=True)
    pv.MultiBlock([surface for _, surface in surfaces]).save(
        output/"density_isosurfaces.vtm", binary=True)
    lengths = stream.compute_cell_sizes(length=True)["Length"]
    metadata.update({
        "extent_au": extent, "samples_per_axis": samples, "spacing_au": spacing,
        "density_levels_log10_g_cm3": list(levels),
        "sampled_rho_max_g_cm3": float(np.max(data["rho"])),
        "sampled_B_max_G": float(np.max(data["bmag"])),
        "streamline_cells": int(stream.n_cells),
        "streamline_points": int(stream.n_points),
        "streamline_total_length_au": float(np.sum(lengths)),
        "streamline_max_length_au": float(np.max(lengths)) if len(lengths) else 0.0,
        "tube_cells": int(tube.n_cells),
        "orbit_frames": orbit_frames, "orbit_fps": fps,
        "limitations": [
            "AMR leaf-cell values are sampled piecewise-constantly onto a uniform display grid.",
            "Streamlines integrate the normalized, sampled cell-centred B field, not native face-centred CT fluxes.",
            "The visualization does not measure magnetic flux, divB, reconnection, or unique field-line topology."
        ]})
    (output/"summary.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return metadata


def write_index(output, results):
    sections = []
    for label, result in results.items():
        sections.append("""<h2>{title}: &plusmn;{extent:g} au</h2>
<p>{samples}<sup>3</sup> display lattice; spacing {spacing:.5g} au.
Sampled maxima: &rho;={rho:.5g} g cm<sup>-3</sup>, |B|={bmag:.5g} G.</p>
<video controls preload='metadata'><source src='{label}/density_fieldlines_3d_orbit.mp4' type='video/mp4'></video>
<p><a href='{label}/density_fieldlines_3d_orbit_slow.mp4'>slower 24 s orbit</a></p>
<h3>Translucent density isosurfaces + magnetic trajectories</h3>
<p><a href='{label}/density_fieldlines_3d_interactive.html'><img src='{label}/density_fieldlines_3d.png' alt='{label} 3D density and field lines'></a></p>
<h3>Density volume rendering + magnetic trajectories</h3>
<p><img src='{label}/density_volume_fieldlines_3d.png' alt='{label} density volume rendering'></p>
<p><a href='{label}/density_fieldlines_3d_interactive.html'>interactive 3-D HTML</a> · <a href='{label}/resampled_density_B.vti'>VTI volume</a> · <a href='{label}/magnetic_streamlines.vtp'>VTP trajectories</a> · <a href='{label}/density_isosurfaces.vtm'>VTM isosurfaces</a> · <a href='{label}/summary.json'>summary</a></p>""".format(
            label=label, title=label.title(), extent=result["extent_au"],
            samples=result["samples_per_axis"], spacing=result["spacing_au"],
            rho=result["sampled_rho_max_g_cm3"],
            bmag=result["sampled_B_max_G"]))
    page = """<!doctype html><meta charset='utf-8'><title>3D collapse density and magnetic field</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:2rem auto;line-height:1.5;background:#f7f7f7}}video,img{{width:100%;height:auto}}.note{{background:#fff4dd;padding:.8rem 1rem}}</style>
<h1>3D protostellar-collapse density and magnetic field</h1>
<p>Final AMR+1 state, cycle 2400.  Click an isosurface image to rotate and
zoom the self-contained interactive scene.</p>
{}
<div class='note'><strong>Visualization scope.</strong> Density surfaces and volumes are AMR leaf-cell values resampled onto a uniform display lattice. The envelope lattice is intentionally too coarse to recover the central maximum; use the core view for the inner object. Magnetic lines are Runge-Kutta integrations of the normalized sampled cell-centred B field. Use the native HDF/history audits—not these lines—for magnetic flux, divB, or topology claims.</div>
<p><a href='../three_d_evolution_v2/index.html'>3-D time evolution</a> · <a href='../movies/index.html'>2D movies</a> · <a href='../additional_core_analysis/index.html'>core diagnostics</a> · <a href='../resolution_comparison/index.html'>resolution comparison</a></p>
""".format("\n".join(sections))
    (output/"index.html").write_text(page)
    (output/"summary.json").write_text(json.dumps(results, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--core-samples", type=int, default=128)
    parser.add_argument("--envelope-samples", type=int, default=128)
    parser.add_argument("--orbit-frames", type=int, default=72)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    pv.OFF_SCREEN = True
    results = {}
    results["core"] = build_case(
        args.snapshot, args.output/"core", 50.0, args.core_samples,
        (-14.5, -13.5, -12.5, -11.5, -10.5), args.orbit_frames, args.fps)
    results["envelope"] = build_case(
        args.snapshot, args.output/"envelope", 300.0, args.envelope_samples,
        (-15.5, -14.8, -14.1, -13.4, -12.7), args.orbit_frames, args.fps)
    write_index(args.output, results)


if __name__ == "__main__":
    main()
