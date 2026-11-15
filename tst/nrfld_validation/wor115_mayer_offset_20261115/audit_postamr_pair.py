#!/usr/bin/env python3
"""Quantify the one-step effect of resetting newly refined background cells."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("uncorrected", type=Path)
    parser.add_argument("corrected", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with h5py.File(args.uncorrected) as old, h5py.File(args.corrected) as new:
        assert old.attrs["NumCycles"] == new.attrs["NumCycles"]
        assert np.array_equal(old["LogicalLocations"][:], new["LogicalLocations"][:])
        names = [item.decode() for item in old.attrs["VariableNames"]]
        assert names == [item.decode() for item in new.attrs["VariableNames"]]
        old_p = np.asarray(old["prim"][names.index("press")], dtype=np.float64)
        new_p = np.asarray(new["prim"][names.index("press")], dtype=np.float64)
        old_rho = np.asarray(old["prim"][names.index("rho")], dtype=np.float64)
        new_rho = np.asarray(new["prim"][names.index("rho")], dtype=np.float64)
        tracer = np.asarray(old["prim"][names.index("r0")], dtype=np.float64)
        x1, x2, x3 = [np.asarray(old[f"x{axis}v"]) for axis in (1, 2, 3)]
        radius = np.sqrt(x1[:, None, None, :] ** 2
                         + x2[:, None, :, None] ** 2
                         + x3[:, :, None, None] ** 2)
        volume = (np.diff(old["x1f"][:], axis=1)[:, None, None, :]
                  * np.diff(old["x2f"][:], axis=1)[:, None, :, None]
                  * np.diff(old["x3f"][:], axis=1)[:, :, None, None])
        changed = np.abs(new_p / old_p - 1.0) > 1e-4
        delta = np.sum((new_p - old_p) * volume)
        result = {
            "cycle": int(old.attrs["NumCycles"]),
            "uncorrected_sha256": digest(args.uncorrected),
            "corrected_sha256": digest(args.corrected),
            "cells": int(changed.size),
            "pressure_changed_over_1e-4_cells": int(np.count_nonzero(changed)),
            "changed_with_background_tracer_ge_half": int(
                np.count_nonzero(changed & (tracer >= 0.5))),
            "changed_inside_1000au": int(np.count_nonzero(changed & (radius < 1000))),
            "changed_radius_min_au": float(np.min(radius[changed])),
            "changed_radius_max_au": float(np.max(radius[changed])),
            "max_density_relative_difference": float(
                np.max(np.abs(new_rho / old_rho - 1.0))),
            "pressure_ratio_min": float(np.min(new_p / old_p)),
            "pressure_ratio_max": float(np.max(new_p / old_p)),
            "internal_energy_change_over_total": float(delta / np.sum(new_p * volume)),
            "internal_energy_change_over_r_lt_3000au": float(
                delta / np.sum(new_p[radius < 3000] * volume[radius < 3000])),
        }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
