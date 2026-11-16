#!/usr/bin/env python3
"""Track radial flow in axis and diagonal cones of RSG snapshots."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from check_rsg_initial_force_balance import RSUN_CGS, parameters


SHELLS = ((0, 166), (166, 300), (300, 500), (500, 700),
          (700, 830), (830, 900))
REGIONS = ("all", "axis15", "diagonal")


def measure(path, length_to_rsun, velocity_to_kms):
    stats = {(lo, hi, region): np.zeros(6, dtype=float)
             for lo, hi in SHELLS for region in REGIONS}
    with h5py.File(path) as file:
        names = [value.decode() for value in file.attrs["VariableNames"]]
        data = file["prim"]
        for block in range(data.shape[1]):
            x = np.asarray(file["x1v"][block])[None, None, :]
            y = np.asarray(file["x2v"][block])[None, :, None]
            z = np.asarray(file["x3v"][block])[:, None, None]
            r = np.sqrt(x*x + y*y + z*z)
            rho = np.asarray(data[names.index("rho"), block], dtype=float)
            vx, vy, vz = (np.asarray(data[names.index(f"vel{axis}"), block],
                                     dtype=float)*velocity_to_kms
                          for axis in (1, 2, 3))
            vr = (vx*x + vy*y + vz*z)/r
            speed2 = vx*vx + vy*vy + vz*vz
            max_cos = np.maximum(np.maximum(abs(x), abs(y)), abs(z))/r
            masks = {"all": np.ones(r.shape, dtype=bool),
                     "axis15": max_cos > np.cos(np.deg2rad(15.0)),
                     "diagonal": max_cos < 0.70}
            rr = r*length_to_rsun
            for lo, hi in SHELLS:
                shell = (rr >= lo) & (rr < hi)
                for region, cone in masks.items():
                    selected = shell & cone
                    if not np.any(selected):
                        continue
                    w = rho[selected]
                    v = vr[selected]
                    entry = stats[(lo, hi, region)]
                    entry[0] += selected.sum()
                    entry[1] += w.sum()
                    entry[2] += np.dot(w, v)
                    entry[3] += np.dot(w, v*v)
                    entry[4] += np.dot(w, speed2[selected])
                    entry[5] = max(entry[5], np.sqrt(speed2[selected].max()))
        result = {"time_code": float(file.attrs["Time"]), "shells": {}}
    for (lo, hi, region), entry in stats.items():
        if entry[1] <= 0:
            continue
        result["shells"].setdefault(f"{lo}-{hi}", {})[region] = {
            "cells": int(entry[0]),
            "mass_weighted_vr_kms": float(entry[2]/entry[1]),
            "rms_vr_kms": float(np.sqrt(entry[3]/entry[1])),
            "rms_speed_kms": float(np.sqrt(entry[4]/entry[1])),
            "max_speed_kms": float(entry[5]),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--snapshot", nargs=2, action="append", required=True,
                        metavar=("NAME", "PRIM_ATHDF"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = parameters(args.input)
    length_to_rsun = float(config["leng_unit"])/RSUN_CGS
    velocity_to_kms = np.sqrt(float(config["egas_unit"])
                              /float(config["rho_unit"]))/1e5
    results = {name: measure(Path(path), length_to_rsun, velocity_to_kms)
               for name, path in args.snapshot}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"snapshots": results}, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
