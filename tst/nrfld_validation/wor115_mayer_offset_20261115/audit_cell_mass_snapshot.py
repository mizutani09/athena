#!/usr/bin/env python3
"""Measure AMR cell masses against Mayer's AREPO target at first core."""

import argparse
import json

import h5py
import numpy as np


AU_CM = 1.495978707e13
MSUN_G = 1.98847e33
TARGET_MSUN = 3.33e-7


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot")
    parser.add_argument("output_json")
    args = parser.parse_args()
    with h5py.File(args.snapshot) as h5:
        names = [name.decode() for name in h5.attrs["VariableNames"]]
        rho = h5["prim"][names.index("rho")].astype(np.float64) * 1e-18
        x = h5["x1v"][:].astype(np.float64)
        y = h5["x2v"][:].astype(np.float64)
        z = h5["x3v"][:].astype(np.float64)
    volume = ((x[:, 1] - x[:, 0]) * (y[:, 1] - y[:, 0])
              * (z[:, 1] - z[:, 0]) * AU_CM**3)[:, None, None, None]
    mass = rho * volume / MSUN_G
    radius = np.sqrt(x[:, None, None, :] ** 2 + y[:, None, :, None] ** 2
                     + z[:, :, None, None] ** 2)
    radius = np.broadcast_to(radius, mass.shape)
    result = {"snapshot": args.snapshot,
              "mayer_target_cell_mass_msun": TARGET_MSUN,
              "mayer_split_threshold_msun": 2 * TARGET_MSUN,
              "regions": {}}
    for label, mask in (
        ("r_below_7au", radius < 7),
        ("r_below_7au_rho_above_1e-11", (radius < 7) & (rho >= 1e-11)),
        ("r_below_7au_rho_above_5e-11", (radius < 7) & (rho >= 5e-11)),
    ):
        values = mass[mask]
        result["regions"][label] = {
            "cells": int(values.size),
            "mass_msun_p10_p50_p90": [float(v) for v in np.percentile(values, [10, 50, 90])],
            "fraction_above_mayer_split_threshold": float(np.mean(values > 2 * TARGET_MSUN)),
        }
    i = np.unravel_index(np.argmax(rho), rho.shape)
    result["densest_cell"] = {
        "rho_g_cm3": float(rho[i]), "mass_msun": float(mass[i]),
        "mass_over_mayer_target": float(mass[i] / TARGET_MSUN),
    }
    with open(args.output_json, "w") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
