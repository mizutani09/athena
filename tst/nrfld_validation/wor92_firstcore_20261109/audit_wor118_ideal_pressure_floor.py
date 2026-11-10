#!/usr/bin/env python3
"""Locate pressure-floor cells in a leaf-block ideal-EOS HDF5 snapshot."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


AU_CM = 1.495978707e13
MSUN_G = 1.98847e33


def audit(path, pressure_floor=1e-10):
    with h5py.File(path) as h:
        names = [x.decode() for x in h.attrs["VariableNames"]]
        prim = h["prim"]
        pressure = np.asarray(prim[names.index("press")], dtype=np.float64)
        density = np.asarray(prim[names.index("rho")], dtype=np.float64)
        tracer = np.asarray(prim[names.index("r0")], dtype=np.float64)
        positions = [np.asarray(h[f"x{d}v"], dtype=np.float64) for d in (1, 2, 3)]
        faces = [np.asarray(h[f"x{d}f"], dtype=np.float64) for d in (1, 2, 3)]
        levels = np.asarray(h["Levels"])
        cycle = int(h.attrs["NumCycles"])
    block, k, j, i = np.where(pressure <= pressure_floor*(1.0 + 1e-4))
    result = {"source": str(path), "cycle": cycle,
              "pressure_floor_code": pressure_floor,
              "pressure_floor_cell_count": len(block)}
    if len(block) == 0:
        return result
    x = positions[0][block, i]
    y = positions[1][block, j]
    z = positions[2][block, k]
    radius = np.sqrt(x*x + y*y + z*z)
    volume = ((faces[0][block, i+1] - faces[0][block, i])
              * (faces[1][block, j+1] - faces[1][block, j])
              * (faces[2][block, k+1] - faces[2][block, k]))
    rho = density[block, k, j, i]*1e-18
    mass = float(np.sum(rho*volume*AU_CM**3))
    result.update({
        "r_min_au": float(radius.min()), "r_max_au": float(radius.max()),
        "density_min_g_cm3": float(rho.min()),
        "density_max_g_cm3": float(rho.max()),
        "mass_g": mass, "mass_msun": mass/MSUN_G,
        "ambient_tracer_max_fraction": float(np.max(
            tracer[block, k, j, i]/density[block, k, j, i])),
        "levels": {str(int(level)): int(np.sum(levels[block] == level))
                   for level in np.unique(levels[block])},
        "coordinates_au": [[float(a), float(b), float(c)]
                           for a, b, c in zip(x, y, z)],
    })
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("hdf5", type=Path)
    p.add_argument("output_json", type=Path)
    args = p.parse_args()
    result = audit(args.hdf5)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
