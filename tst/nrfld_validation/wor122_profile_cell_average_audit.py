#!/usr/bin/env python3
"""Audit point-sampled versus cubature-averaged RSG initial profile fields."""

import argparse
import json
from pathlib import Path

import numpy as np

from check_rsg_initial_force_balance import RSUN_CGS, force_rows

SHELLS = ((800, 813), (813, 817), (817, 830), (830, 850), (850, 880))
FIELDS = {"rho": 1, "egas": 2, "erad": 3}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--dx-rsun", required=True, type=float)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    profile = np.loadtxt(args.profile)
    radius = profile[:, 0]/RSUN_CGS
    log_profiles = {name: np.log(profile[:, column]) for name, column in FIELDS.items()}
    xyz = force_rows(args.log)[:, :3]
    r = np.linalg.norm(xyz, axis=1)
    point = {name: np.exp(np.interp(r, radius, values))
             for name, values in log_profiles.items()}
    average = {name: np.zeros(len(xyz)) for name in FIELDS}
    nodes, weights = np.polynomial.legendre.leggauss(args.order)
    nodes *= args.dx_rsun/2
    weights /= 2
    for i, x in enumerate(nodes):
        for j, y in enumerate(nodes):
            for k, z in enumerate(nodes):
                rr = np.linalg.norm(xyz+np.array([x, y, z]), axis=1)
                w = weights[i]*weights[j]*weights[k]
                for name, values in log_profiles.items():
                    average[name] += w*np.exp(np.interp(rr, radius, values))
    result = {"dx_rsun": args.dx_rsun, "quadrature_order_per_axis": args.order,
              "cells": len(xyz), "shells": {}}
    for lo, hi in SHELLS:
        selected = (r >= lo) & (r < hi)
        if not np.any(selected):
            continue
        result["shells"][f"{lo}-{hi}"] = {
            "cells": int(np.count_nonzero(selected)),
            "cell_average_over_point": {
                name: {
                    "mean": float(np.mean((average[name]/point[name])[selected])),
                    "minimum": float(np.min((average[name]/point[name])[selected])),
                    "maximum": float(np.max((average[name]/point[name])[selected])),
                }
                for name in FIELDS},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(args.output)


if __name__ == "__main__":
    main()
