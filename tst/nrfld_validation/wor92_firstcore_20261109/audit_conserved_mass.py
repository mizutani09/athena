#!/usr/bin/env python3
"""Integrate conserved density on matched AMR leaf meshes in two HDF5 files."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


RHO_UNIT = 1e-18
LENGTH_UNIT = 1.495978707e13


def snapshot(path):
    with h5py.File(path) as source:
        names = [name.decode() for name in source.attrs["VariableNames"]]
        rho = np.asarray(source["cons"][names.index("dens")], dtype=np.float64)
        x = [np.asarray(source[f"x{i}f"], dtype=np.float64) for i in (1, 2, 3)]
        volume = (np.diff(x[2], axis=1)[:, :, None, None]
                  * np.diff(x[1], axis=1)[:, None, :, None]
                  * np.diff(x[0], axis=1)[:, None, None, :])
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "cycle": int(source.attrs["NumCycles"]),
                "blocks": int(len(source["Levels"])),
                "minimum_code_density": float(rho.min()),
                "gas_mass_g": float(np.sum(rho*volume)*RHO_UNIT*LENGTH_UNIT**3),
                "levels": source["Levels"][:],
                "logical_locations": source["LogicalLocations"][:]}


def audit(before_path, after_path, output):
    before, after = snapshot(before_path), snapshot(after_path)
    mesh_equal = (np.array_equal(before.pop("levels"), after.pop("levels"))
                  and np.array_equal(before.pop("logical_locations"),
                                     after.pop("logical_locations")))
    result = {"before": before, "after": after,
              "same_amr_topology": bool(mesh_equal),
              "relative_conserved_mass_change": after["gas_mass_g"]/before["gas_mass_g"]-1,
              "density_floor_code": 1e-3,
              "method": "cons/dens and face coordinates cast to float64 before cell-volume integration; HDF5 cell values are float32"}
    if before["cycle"] >= after["cycle"] or not mesh_equal:
        raise ValueError("unexpected snapshots or AMR topology")
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before_hdf", type=Path)
    parser.add_argument("after_hdf", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.before_hdf, args.after_hdf,
                           args.output_json), indent=2))


if __name__ == "__main__":
    main()
