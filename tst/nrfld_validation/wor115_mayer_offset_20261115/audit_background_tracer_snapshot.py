#!/usr/bin/env python3
"""Measure where tracer-marked ambient material is in an AMR leaf snapshot."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with h5py.File(args.snapshot) as h:
        names = [name.decode() for name in h.attrs["VariableNames"]]
        rho = np.asarray(h["prim"][names.index("rho")], dtype=np.float64)
        tracer = np.asarray(h["prim"][names.index("r0")], dtype=np.float64)
        x1 = np.asarray(h["x1v"])
        x2 = np.asarray(h["x2v"])
        x3 = np.asarray(h["x3v"])
        radius = np.sqrt(x1[:, None, None, :] ** 2
                         + x2[:, None, :, None] ** 2
                         + x3[:, :, None, None] ** 2)
        radius = np.broadcast_to(radius, rho.shape)
        volume = (np.diff(h["x1f"][:], axis=1)[:, None, None, :]
                  * np.diff(h["x2f"][:], axis=1)[:, None, :, None]
                  * np.diff(h["x3f"][:], axis=1)[:, :, None, None])
        volume = np.broadcast_to(volume, rho.shape)
        result = {"source": str(args.snapshot),
                  "cycle": int(h.attrs["NumCycles"]),
                  "bands_au": {}}
        for outer in (7, 30, 100, 300, 1000, 3000):
            inside = radius < outer
            weighted_mass = rho[inside] * volume[inside]
            result["bands_au"][str(outer)] = {
                "cells": int(np.count_nonzero(inside)),
                "max_tracer": float(np.max(tracer[inside])),
                "cells_tracer_ge_half": int(np.count_nonzero(tracer[inside] >= .5)),
                "mass_weighted_tracer": float(
                    np.sum(weighted_mass * tracer[inside]) / np.sum(weighted_mass)),
            }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
