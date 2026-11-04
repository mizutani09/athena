#!/usr/bin/env python3
"""Time-average already assembled full-3D convection profiles over a late window.

The averaging window is descriptive: it does not imply statistical stationarity.
Use the history report's crossing-time and flux checks before interpreting it
as a relaxed solar atmosphere.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIELDS = (
    ("T_mean_K", "mean gas temperature [K]", False),
    ("rho_mean_cgs", "mean density [g cm$^{-3}$]", True),
    ("vz_rms_km_s", "vertical RMS speed [km/s]", False),
    ("qrad_mean_cgs", "mean radiative cooling [erg cm$^{-3}$ s$^{-1}$]", False),
)


def summarize(run_dir: Path, fraction: float) -> dict[str, object] | None:
    source = json.loads((run_dir / "profiles3d.json").read_text())
    records = source["records"]
    if len(records) < 3:
        return None
    times = np.asarray([entry["time_code"] for entry in records], dtype=float)
    if not np.all(np.diff(times) > 0):
        raise ValueError("Full-3D profile times must increase strictly")
    start = times[-1] - fraction * (times[-1] - times[0])
    selected = [entry for entry in records if entry["time_code"] >= start]
    if len(selected) < 2:
        return None
    z = np.asarray(selected[0]["z_code"], dtype=float)
    if any(not np.array_equal(z, entry["z_code"]) for entry in selected[1:]):
        raise ValueError("Full-3D profile grids differ across snapshots")
    summary: dict[str, object] = {
        "description": "Horizontal mean at each snapshot, then temporal mean; no stationarity claim",
        "window_start_code": float(selected[0]["time_code"]),
        "window_end_code": float(selected[-1]["time_code"]),
        "snapshot_count": len(selected),
        "time_code": [float(entry["time_code"]) for entry in selected],
        "z_code": z.tolist(),
    }
    for key, _, _ in FIELDS:
        series = np.asarray([entry[key] for entry in selected], dtype=float)
        if series.shape != (len(selected), len(z)) or not np.isfinite(series).all():
            raise ValueError(f"Invalid {key} profile series")
        summary[key] = {
            "temporal_mean": np.mean(series, axis=0).tolist(),
            "temporal_std": np.std(series, axis=0).tolist(),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--late-fraction", type=float, default=0.25,
                        help="Fraction of recorded time span used for the late mean")
    parser.add_argument("--reference-profile", type=Path,
                        help="Imported 1D solar profile for a visual stratification reference")
    args = parser.parse_args()
    if not (0 < args.late_fraction <= 1):
        parser.error("--late-fraction must be in (0, 1]")
    summary = summarize(args.run_dir, args.late_fraction)
    if summary is None:
        print("Too few late full-3D snapshots for a time average")
        return
    z = np.asarray(summary["z_code"])
    reference = None
    if args.reference_profile is not None:
        raw = np.loadtxt(args.reference_profile, comments="#", ndmin=2)
        if raw.shape[1] < 5 or not np.all(np.diff(raw[:, 3]) > 0):
            raise ValueError("Expected a monotonic 1D solar profile with z in column 4")
        reference = {
            "temperature_K": np.interp(z, raw[:, 3], raw[:, 1]),
            "density_cgs": np.interp(z, raw[:, 3], raw[:, 4]),
        }
        summary["reference_profile"] = {
            "description": "Imported 1D initial solar profile; visual reference, not an independent observation",
            "path": str(args.reference_profile.resolve()),
            "temperature_K": reference["temperature_K"].tolist(),
            "density_cgs": reference["density_cgs"].tolist(),
        }
    output = args.run_dir / "profiles3d_late_mean.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    plt.rcParams.update({"font.size": 13})
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for axis, (key, label, logarithmic) in zip(axes.flat, FIELDS):
        mean = np.asarray(summary[key]["temporal_mean"])
        std = np.asarray(summary[key]["temporal_std"])
        axis.plot(mean, z, color="tab:blue", label="late mean")
        axis.fill_betweenx(z, mean - std, mean + std, color="tab:blue",
                           alpha=0.2, label="snapshot std")
        if logarithmic:
            axis.set_xscale("log")
        axis.set(xlabel=label, ylabel="z [code]")
        axis.grid(alpha=0.25)
        axis.legend()
    if reference is not None:
        axes[0, 0].plot(reference["temperature_K"], z, color="black", ls="--",
                        label="initial solar 1D")
        axes[0, 1].plot(reference["density_cgs"], z, color="black", ls="--",
                        label="initial solar 1D")
        axes[0, 0].legend()
        axes[0, 1].legend()
    fig.suptitle("Full 3D late profiles: t={:.3g}–{:.3g} code ({} snapshots)\n"
                 "Transient until history stationarity checks pass".format(
                     summary["window_start_code"], summary["window_end_code"],
                     summary["snapshot_count"]))
    image = args.run_dir / "imgs" / "simple_convection_profiles3d_late_mean.png"
    image.parent.mkdir(exist_ok=True)
    fig.savefig(image, dpi=180)
    plt.close(fig)
    print(f"Wrote {output} and {image}")


if __name__ == "__main__":
    main()
