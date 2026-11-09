#!/usr/bin/env python3
"""Measure gas/radiation temperature separation in a first-core HDF5 snapshot."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from analyze_core_profile import eos_temperature


RAD_CONST = 7.5657e-15  # erg cm^-3 K^-4, same as input physics constants


def audit(snapshot, eos_table, output):
    with h5py.File(snapshot) as source:
        names = [name.decode() for name in source.attrs["VariableNames"]]
        values = source["prim"]
        rho = np.asarray(values[names.index("rho")], dtype=np.float64) * 1e-18
        pressure = np.asarray(values[names.index("press")], dtype=np.float64) * 1e-9
        erad = np.asarray(values[names.index("Er")], dtype=np.float64) * 1e-9
        x = [np.asarray(source[f"x{i}v"], dtype=np.float64) for i in (1, 2, 3)]
        cycle = int(source.attrs["NumCycles"])
    gas, outside = eos_temperature(eos_table, rho.ravel(), pressure.ravel())
    rad = (erad.ravel() / RAD_CONST)**0.25
    shape = rho.shape
    radius = np.sqrt(x[0][:, None, None, :]**2
                     + x[1][:, None, :, None]**2
                     + x[2][:, :, None, None]**2)
    regions = {"core_r_below_5au_and_rho_above_1e-14":
               (radius < 5) & (rho > 1e-14),
               "shock_5_to_12au_and_rho_above_1e-14":
               (radius >= 5) & (radius < 12) & (rho > 1e-14)}
    result = {"snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
              "eos_table_sha256": hashlib.sha256(eos_table.read_bytes()).hexdigest(),
              "cycle": cycle, "eos_cells_outside_table": outside,
              "regions": {}}
    for name, mask in regions.items():
        delta = np.abs(gas[mask.ravel()] - rad[mask.ravel()])
        temp = gas[mask.ravel()]
        if not len(delta):
            raise ValueError("empty region " + name)
        result["regions"][name] = {
            "cells": int(len(delta)), "median_abs_Tgas_minus_Trad_K": float(np.median(delta)),
            "p99_abs_Tgas_minus_Trad_K": float(np.percentile(delta, 99)),
            "max_abs_Tgas_minus_Trad_K": float(np.max(delta)),
            "p99_relative_difference": float(np.percentile(delta/temp, 99)),
        }
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("eos_table", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.snapshot, args.eos_table, args.output_json), indent=2))


if __name__ == "__main__":
    main()
