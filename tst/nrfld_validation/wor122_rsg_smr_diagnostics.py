#!/usr/bin/env python3
"""Summarize short RSG static-refinement runs with cell-volume weights."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def snapshot(path):
    with h5py.File(path) as f:
        names = [name.decode() for name in f.attrs["VariableNames"]]
        data = f["prim"][:].astype(np.float64)
        fields = {name: data[i] for i, name in enumerate(names)}
        dx = [np.diff(f[f"x{axis}f"][:].astype(np.float64))
              for axis in (1, 2, 3)]
        volume = dx[2][:, :, None, None] * dx[1][:, None, :, None] * dx[0][:, None, None, :]
        radius = np.sqrt(f["x3v"][:].astype(np.float64)[:, :, None, None] ** 2
                         + f["x2v"][:].astype(np.float64)[:, None, :, None] ** 2
                         + f["x1v"][:].astype(np.float64)[:, None, None, :] ** 2)
        rho = fields["rho"]
        er = fields["Er"]
        speed = np.sqrt(sum(fields[f"vel{axis}"] ** 2 for axis in (1, 2, 3)))
        shell = (radius >= 800.0) & (radius <= 860.0)
        selected = volume[shell]
        return {
            "time": float(f.attrs["Time"]),
            "blocks": int(f.attrs["NumMeshBlocks"]),
            "level_counts": {str(level): int(np.sum(f["Levels"][:] == level))
                             for level in np.unique(f["Levels"][:])},
            "mass": float(np.sum(rho * volume)),
            "radiation_energy": float(np.sum(er * volume)),
            "min_rho": float(np.min(rho)),
            "min_er": float(np.min(er)),
            "max_speed": float(np.max(speed)),
            "shell_mean_rho": float(np.sum(rho[shell] * selected) / np.sum(selected)),
            "shell_mean_er": float(np.sum(er[shell] * selected) / np.sum(selected)),
            "shell_max_speed": float(np.max(speed[shell])),
            "shell_volume": float(np.sum(selected)),
            "min_dx": [float(np.min(axis_dx)) for axis_dx in dx],
        }


def history(path):
    lines = path.read_text().splitlines()
    header = next(line for line in lines if line.startswith("# [1]="))
    labels = [part.split("=", 1)[1].strip().split()[0] for part in header.split("[")[1:]]
    rows = np.loadtxt(path, comments="#", ndmin=2)
    return {"columns": labels, "rows": rows.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    prim = sorted(args.directory.glob("*.prim.*.athdf"))
    hst = next(args.directory.glob("*.hst"))
    result = {"snapshots": [snapshot(path) for path in prim], "history": history(hst)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
