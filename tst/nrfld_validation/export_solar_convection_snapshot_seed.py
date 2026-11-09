#!/usr/bin/env python3
"""Export an exact-grid solar convection snapshot as an FLD3D001 seed.

This starts a new calculation at code time zero.  It preserves the six
primitive fields, but not the former run's restart-only boundary feedback.
Use the same seed for all branches of a controlled MHD comparison.
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


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prim", type=Path, required=True)
    parser.add_argument("--uov", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prim_path, uov_path, input_path, output = (
        path.resolve() for path in (args.prim, args.uov, args.input, args.output)
    )
    if output.exists():
        parser.error(f"Refusing to overwrite {output}")
    prim, coordinates, time = read_volume(prim_path, "prim")
    uov, other_coordinates, other_time = read_volume(uov_path, "user_out_var")
    if prim.shape[0] < 5 or uov.shape[0] < 2 or time != other_time or any(
        not np.array_equal(a, b) for a, b in zip(coordinates, other_coordinates)
    ):
        parser.error("Primitive and radiation volumes are incompatible")
    with h5py.File(prim_path) as source:
        bounds = tuple(float(v) for axis in (1, 2, 3)
                       for v in source.attrs[f"RootGridX{axis}"][:2])
    egas_unit = units_from_input(input_path)["egas_unit"]
    data = np.concatenate((prim[:5], uov[1:2] / egas_unit), axis=0).astype("<f8")
    if not np.isfinite(data).all() or np.any(data[[0, 1, 5]] <= 0):
        parser.error("Snapshot has nonfinite or nonpositive thermodynamics")
    nz, ny, nx = data.shape[1:]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        stream.write(struct.pack("<8sIII6d", b"FLD3D001", nx, ny, nz, *bounds))
        stream.write(data.tobytes(order="C"))
    manifest = {
        "format": "FLD3D001", "time_source_code": time,
        "fields": ["rho", "pressure", "vx", "vy", "vz", "erad"],
        "shape_zyx": [nz, ny, nx], "bounds_xyz": bounds,
        "egas_unit_cgs": egas_unit,
        "source": {str(path): digest(path)
                   for path in (prim_path, uov_path, input_path)},
        "output_sha256": digest(output),
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
