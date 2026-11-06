#!/usr/bin/env python3
"""Transfer developed 3D convection onto a chosen 1D solar-box mean state.

The output is a little-endian, exact-grid FLD3D001 primitive seed for
simple_convection_lhllc_fld. A seed is a controlled t=0 experiment; it is
not a thermally relaxed solar-luminosity solution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from plot_simple_convection_surface3d import read_volume  # noqa: E402
from plot_simple_convection_tau import units_from_input  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bounds(path: Path) -> tuple[float, ...]:
    with h5py.File(path) as source:
        return tuple(float(value) for axis in (1, 2, 3)
                     for value in source.attrs[f"RootGridX{axis}"][:2])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--developed-prim", type=Path, required=True)
    parser.add_argument("--developed-uov", type=Path, required=True)
    parser.add_argument("--target-prim", type=Path, required=True)
    parser.add_argument("--target-uov", type=Path, required=True)
    parser.add_argument("--target-input", type=Path, required=True,
                        help="Athena input that sets the target code units")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--velocity-scale", type=float, default=1.0)
    parser.add_argument("--upper-taper-start", type=float, default=2.4,
                        help="Fade velocity smoothly to zero at the upper boundary")
    parser.add_argument("--lower-taper-end", type=float, default=-2.4,
                        help="Grow velocity smoothly from zero at the lower boundary")
    args = parser.parse_args()
    paths = {key: getattr(args, key).resolve() for key in
             ("developed_prim", "developed_uov", "target_prim", "target_uov")}
    if not 0 < args.velocity_scale <= 2:
        parser.error("velocity-scale must lie in (0, 2]")
    target_input = args.target_input.resolve()
    if not target_input.is_file():
        parser.error(f"Target Athena input missing: {target_input}")
    egas_unit = units_from_input(target_input)["egas_unit"]
    developed, xyz, developed_time = read_volume(paths["developed_prim"], "prim")
    developed_uov, xyz_uov, uov_time = read_volume(paths["developed_uov"],
                                                    "user_out_var")
    target, target_xyz, target_time = read_volume(paths["target_prim"], "prim")
    target_uov, target_uov_xyz, target_uov_time = read_volume(
        paths["target_uov"], "user_out_var")
    if (not all(np.array_equal(a, b) for other in
                (xyz_uov, target_xyz, target_uov_xyz)
                for a, b in zip(xyz, other)) or
            developed.shape != target.shape or
            developed_time != uov_time or target_time != target_uov_time):
        parser.error("Paired dumps must have identical regular grids and times")
    bounds_all = [bounds(path) for path in paths.values()]
    if any(not np.allclose(bounds_all[0], item, rtol=0, atol=1e-9)
           for item in bounds_all[1:]):
        parser.error("Source and target physical box bounds differ")
    if target_time != 0:
        parser.error("Target dump must be the t=0 1D profile")
    mean = lambda a: a.mean(axis=(1, 2), keepdims=True)
    rho = mean(target[0]) * developed[0] / mean(developed[0])
    pressure = mean(target[1]) * developed[1] / mean(developed[1])
    # Athena's UOV Erad is cgs, while the optional primitive seed is in
    # code units like the native radiation state.
    erad = (mean(target_uov[1]) * developed_uov[1] /
            mean(developed_uov[1]) / egas_unit)
    velocities = developed[2:5].astype(np.float64)
    # Remove each layer's net mass flow. This preserves convective velocity
    # and thermodynamic correlations without starting a box-wide wind.
    velocities -= np.mean(developed[0][None] * velocities,
                          axis=(2, 3), keepdims=True) / mean(developed[0])[None]
    bottom, top = bounds_all[0][-2:]
    if not bottom < args.lower_taper_end < args.upper_taper_start < top:
        parser.error("Taper limits must lie in order inside the box")
    z = xyz[2]
    taper = np.ones_like(z)
    lower = z < args.lower_taper_end
    taper[lower] = 0.5 * (1 - np.cos(np.pi *
                         (z[lower] - bottom) /
                         (args.lower_taper_end - bottom)))
    region = z > args.upper_taper_start
    taper[region] = 0.5 * (1 + np.cos(np.pi *
                         (z[region] - args.upper_taper_start) /
                         (top - args.upper_taper_start)))
    velocities *= args.velocity_scale * taper[None, :, None, None]
    data = np.concatenate((rho[None], pressure[None], velocities,
                           erad[None]), axis=0).astype("<f8")
    if not np.isfinite(data).all() or np.any(data[[0, 1, 5]] <= 0):
        raise ValueError("Seed contains nonfinite or nonpositive thermodynamics")
    nz, ny, nx = rho.shape
    output = args.output.resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite existing seed: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        stream.write(struct.pack("<8sIII6d", b"FLD3D001", nx, ny, nz,
                                 *bounds_all[0]))
        stream.write(data.tobytes(order="C"))
    # The primitive means are set by the target profile at every height;
    # fluctuations and velocity correlations come from the developed dump.
    summary = {
        "format": "FLD3D001, little-endian, 6 float64 fields in C order [z,y,x]",
        "fields": ["rho", "pressure", "vx", "vy", "vz", "erad"],
        "dims_xyz": [nx, ny, nz], "bounds_xyz": bounds_all[0],
        "developed_time_code": developed_time,
        "target_time_code": target_time,
        "velocity_scale": args.velocity_scale,
        "egas_unit_cgs": egas_unit,
        "upper_taper_start_code": args.upper_taper_start,
        "lower_taper_end_code": args.lower_taper_end,
        "inputs": {key: {"path": str(path), "sha256": sha256(path)}
                   for key, path in paths.items()},
        "target_input": {"path": str(target_input),
                         "sha256": sha256(target_input)},
        "output": str(output), "output_sha256": sha256(output),
        "max_abs_vz_code": float(np.max(np.abs(data[4]))),
        "max_horizontal_mass_flux_code": float(np.max(np.abs(
            np.mean(data[0] * data[4], axis=(1, 2))))),
        "rho_mean_max_relative_error": float(np.max(np.abs(
            mean(data[0]) / mean(target[0]) - 1))),
        "pressure_mean_max_relative_error": float(np.max(np.abs(
            mean(data[1]) / mean(target[1]) - 1))),
        "erad_mean_max_relative_error": float(np.max(np.abs(
            mean(data[5])*egas_unit / mean(target_uov[1]) - 1))),
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
