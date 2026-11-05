#!/usr/bin/env python3
"""Make time-ordered solar-convection movies across checkpoint continuations.

The source run order comes from a continuation report. A later run supersedes
frames in an interrupted parent run from its first accepted history sample.
Only a bounded, approximately uniform subset of the surviving slices is read.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from plot_simple_convection_movies import pairs, read_plane  # noqa: E402
from plot_simple_convection_tau import opacity_interpolator  # noqa: E402
from solar_convection_report import input_values, physical_time_unit  # noqa: E402


@dataclass(frozen=True)
class Slice:
    time: float
    prim: Path
    uov: Path
    source: Path


def first_history_time(run: Path) -> float:
    paths = list(run.glob("*.hst"))
    if len(paths) != 1:
        raise ValueError(f"Expected one history in {run}, found {len(paths)}")
    with paths[0].open() as stream:
        for line in stream:
            if line.strip() and not line.lstrip().startswith("#"):
                return float(line.split()[0])
    raise ValueError(f"History has no samples: {paths[0]}")


def collect(runs: list[Path], plane: str,
            end_time: float | None = None) -> list[Slice]:
    starts = [first_history_time(run) for run in runs]
    if any(b <= a for a, b in zip(starts, starts[1:])):
        raise ValueError("Source runs are not in increasing restart-time order")
    records = []
    for index, run in enumerate(runs):
        cutoff = starts[index + 1] if index + 1 < len(runs) else np.inf
        for prim, uov in pairs(run, plane):
            try:
                with h5py.File(prim) as handle:
                    frame_time = float(handle.attrs["Time"])
            except OSError:
                if end_time is not None and time.time() - prim.stat().st_mtime < 120:
                    continue  # A live run may be writing its newest frame.
                raise
            if frame_time < cutoff - 1e-7 and (end_time is None or
                                                 frame_time <= end_time + 1e-7):
                records.append(Slice(frame_time, prim, uov, run))
    records.sort(key=lambda item: item.time)
    if len(records) < 3 or any(b.time <= a.time for a, b in zip(records, records[1:])):
        raise ValueError(f"Missing or overlapping {plane} time-ordered slices")
    return records


def choose(records: list[Slice], count: int) -> list[Slice]:
    if len(records) <= count:
        return records
    times = np.asarray([item.time for item in records])
    targets = np.linspace(times[0], times[-1], count)
    upper = np.clip(np.searchsorted(times, targets), 0, len(times) - 1)
    lower = np.maximum(upper - 1, 0)
    chosen = np.where(abs(times[lower] - targets) <= abs(times[upper] - targets),
                      lower, upper)
    return [records[int(index)] for index in np.unique(chosen)]


def load(record: Slice, plane: str, velocity_unit: float,
         length_unit: float, rho_unit: float,
         opacity) -> dict[str, object]:
    prim, x, y = read_plane(record.prim, "prim", plane)
    uov, _, _ = read_plane(record.uov, "user_out_var", plane)
    if not (np.isfinite(prim).all() and np.isfinite(uov).all()):
        raise ValueError(f"Nonfinite slice: {record.prim}")
    temperature = uov[2]
    if np.any(temperature <= 0):
        raise ValueError(f"Nonpositive temperature: {record.uov}")
    if plane == "xz":
        mean = np.mean(temperature, axis=1, keepdims=True)
        vertical_velocity = prim[4] - np.mean(prim[4], axis=1, keepdims=True)
    else:
        mean = np.mean(temperature)
        vertical_velocity = prim[4]
    frame = {
        "time": record.time,
        "x": x * length_unit / 1e8,
        "y": y * length_unit / 1e8,
        "temperature_percent": 100.0 * (temperature - mean) / mean,
        "vertical_velocity_km_s": vertical_velocity * velocity_unit / 1e5,
    }
    if plane == "xy":
        frame["plane_mean_vertical_velocity_km_s"] = float(
            np.mean(frame["vertical_velocity_km_s"]))
        frame["vertical_velocity_fluct_km_s"] = (
            frame["vertical_velocity_km_s"]
            - frame["plane_mean_vertical_velocity_km_s"])
    if plane == "xz" and opacity is not None:
        rho = prim[0] * rho_unit
        query = np.stack((np.log10(rho), np.log10(temperature)), axis=-1)
        sigma = 10.0**opacity(query) * rho * length_unit
        dz = float(y[1] - y[0])
        tau = np.cumsum((sigma*dz)[::-1], axis=0)[::-1] - 0.5*sigma*dz
        height = np.full(len(x), np.nan)
        for j in range(len(x)):
            if tau[-1, j] <= 1.0 <= tau[0, j]:
                height[j] = np.interp(1.0, tau[::-1, j], y[::-1])
        frame["tau1_height_mm"] = height * length_unit / 1e8
    return frame


def render(frames: list[dict[str, object]], plane: str, destination: Path,
           fps: int, time_unit: float,
           velocity_field: str = "vertical_velocity_km_s") -> dict[str, float]:
    plt.rcParams.update({"font.size": 23})
    fields = ("temperature_percent", velocity_field)
    scales = {field: max(float(np.percentile(np.concatenate([
        np.abs(frame[field]).ravel() for frame in frames]), 99.0)), 1e-12)
        for field in fields}
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.3), constrained_layout=True)
    images = []
    velocity_title = ("$v_z$ - plane mean [km/s]"
                      if velocity_field == "vertical_velocity_fluct_km_s" else
                      "$v_z$ - height mean [km/s]" if plane == "xz"
                      else "$v_z$ [km/s]")
    lines = []
    for ax, field, title in zip(axes, fields,
                                ("Gas $\\Delta T/\\langle T\\rangle$ [%]",
                                 velocity_title)):
        image = ax.pcolormesh(frames[0]["x"], frames[0]["y"],
                              frames[0][field], shading="auto", cmap="RdBu_r",
                              vmin=-scales[field], vmax=scales[field])
        ax.set(xlabel="x [Mm]", ylabel=("z [Mm]" if plane == "xz" else "y [Mm]"),
               title=title)
        ax.set_aspect("equal")
        fig.colorbar(image, ax=ax, shrink=0.83)
        images.append(image)
        if "tau1_height_mm" in frames[0]:
            line, = ax.plot(frames[0]["x"], frames[0]["tau1_height_mm"],
                            "k--", lw=2, label=r"$\tau_R=1$")
            ax.legend(loc="lower left", fontsize=19)
            lines.append(line)
    title = fig.suptitle("")

    def update(index: int):
        frame = frames[index]
        for image, field in zip(images, fields):
            image.set_array(frame[field].ravel())
        for line in lines:
            line.set_ydata(frame["tau1_height_mm"])
        time = frame["time"]
        mean_note = (f" | plane mean vz={frame['plane_mean_vertical_velocity_km_s']:+.2f} km/s"
                     if plane == "xy" else "")
        title.set_text(f"{plane.upper()} convection | t={time:.1f} code "
                       f"({time*time_unit/3600:.2f} h){mean_note}")
        return (*images, *lines, title)

    animation = FuncAnimation(fig, update, frames=len(frames),
                              interval=1000/fps, blit=False)
    animation.save(destination, writer=FFMpegWriter(
        fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"],
        bitrate=2400))
    update(0)
    fig.savefig(destination.with_suffix(".png"), dpi=140)
    plt.close(fig)
    return scales


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-summary", type=Path, required=True,
                        help="Combined report summary.json with ordered source_runs")
    parser.add_argument("--extra-run", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Campaign root; writes movies/ beneath it")
    parser.add_argument("--max-frames", type=int, default=220)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--end-time", type=float,
                        help="Use completed frames through this code time")
    parser.add_argument("--opacity-table", type=Path,
                        help="Rosseland table for the xz tau=1 contour")
    args = parser.parse_args()
    if args.max_frames < 3 or args.fps < 1:
        parser.error("--max-frames must be >=3 and --fps must be >=1")
    if args.end_time is not None and not np.isfinite(args.end_time):
        parser.error("--end-time must be finite")
    summary = json.loads(args.source_summary.read_text())
    runs = [Path(item).resolve() for item in summary["source_runs"]]
    runs += [item.resolve() for item in args.extra_run]
    for run in runs:
        if not (run / "athinput.generated").is_file():
            raise ValueError(f"Missing generated input: {run}")
    units = input_values(runs[0] / "athinput.generated")
    velocity_unit = np.sqrt(units["egas_unit"] / units["rho_unit"])
    length_unit = units["leng_unit"]
    time_unit = physical_time_unit(units, velocity_unit)
    opacity = (opacity_interpolator(args.opacity_table.resolve())
               if args.opacity_table else None)
    movie_dir = args.output_root.resolve() / "movies"
    movie_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"source_summary": str(args.source_summary.resolve()),
                "source_runs": [str(run) for run in runs],
                "max_frames": args.max_frames, "fps": args.fps,
                "end_time_code": args.end_time,
                "time_unit_seconds": time_unit,
                "tau1_opacity_table": (str(args.opacity_table.resolve())
                                       if args.opacity_table else None),
                "movies": {}}
    manifest["source_provenance"] = {}
    for run in runs:
        source = json.loads((run / "manifest.json").read_text())
        manifest["source_provenance"][str(run)] = {
            "git_commit": source.get("git_commit"),
            "binary_sha256": source.get("binary_sha256"),
            "data_sha256": source.get("data_sha256"),
        }
    for plane in ("xz", "xy"):
        records = collect(runs, plane, end_time=args.end_time)
        sampled = choose(records, args.max_frames)
        frames = [load(record, plane, velocity_unit, length_unit,
                       units["rho_unit"], opacity)
                  for record in sampled]
        destination = movie_dir / f"solar_convection_{plane}.mp4"
        scales = render(frames, plane, destination, args.fps, time_unit)
        manifest["movies"][plane] = {
            "file": destination.name, "poster": destination.with_suffix(".png").name,
            "slice_count": len(records), "rendered_frames": len(sampled),
            "first_time_code": sampled[0].time,
            "last_time_code": sampled[-1].time,
            "color_scales": scales,
            "frame_source_runs": sorted({str(item.source) for item in sampled}),
        }
        print(f"{destination}: {len(sampled)} frames, "
              f"t={sampled[0].time:.1f}–{sampled[-1].time:.1f}")
        if plane == "xy":
            destination = movie_dir / "solar_convection_xy_fluct.mp4"
            scales = render(frames, plane, destination, args.fps, time_unit,
                            velocity_field="vertical_velocity_fluct_km_s")
            manifest["movies"]["xy_fluct"] = {
                **manifest["movies"][plane],
                "file": destination.name,
                "poster": destination.with_suffix(".png").name,
                "color_scales": scales,
                "velocity_definition": "plane mean removed at every frame",
            }
            print(f"{destination}: {len(sampled)} frames, plane mean removed")
    (movie_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
