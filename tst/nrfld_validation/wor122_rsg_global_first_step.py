#!/usr/bin/env python3
"""Measure full-sphere first-step RSG acceleration from primitive HDF files."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from check_rsg_initial_force_balance import G_CGS, MSUN_CGS, RSUN_CGS, parameters

SHELLS = ((0, 166), (166, 500), (500, 750), (750, 800),
          (800, 813), (813, 817), (817, 830), (830, 850), (850, 880))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--case", nargs=3, action="append", metavar=("NAME", "INITIAL", "FINAL"),
                        required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    p = parameters(args.input)
    length = float(p["leng_unit"])
    gm = (G_CGS * float(p["potential_mass_msun"]) * MSUN_CGS
          * float(p["rho_unit"]) / (length * float(p["egas_unit"])))
    radius = float(p["radius_rsun"]) * RSUN_CGS / length
    core = radius * float(p["core_radius_fraction"])
    outer = radius * float(p["outer_flattening_fraction"])
    result = {"cases": {}}
    for name, initial, final in args.case:
        sums = {}
        for lo, hi in SHELLS:
            for sector in ("all", "axis", "diagonal"):
                sums[f"{lo}-{hi}:{sector}"] = np.zeros(5, dtype=float)
        with h5py.File(initial) as f0, h5py.File(final) as f1:
            if f0["prim"].shape != f1["prim"].shape:
                raise ValueError(f"Grid changed in {name}")
            dt = float(f1.attrs["Time"] - f0.attrs["Time"])
            if dt <= 0:
                raise ValueError(f"Invalid first-step dt in {name}")
            rho_all = np.asarray(f0["prim"][0])
            v0_all = np.asarray(f0["prim"][2:5])
            v1_all = np.asarray(f1["prim"][2:5])
            nblock = rho_all.shape[0]
            for b in range(nblock):
                xyz = np.stack(np.meshgrid(
                    f0["x1v"][b], f0["x2v"][b], f0["x3v"][b], indexing="ij"), axis=-1)
                xyz = xyz.transpose(2, 1, 0, 3).reshape(-1, 3)
                r = np.linalg.norm(xyz, axis=1)
                radial = xyz/r[:, None]
                s = (r/outer)**8
                a = core**4 + r**4/np.sqrt(1+s)
                gmag = gm*r**3/((1+s)**1.5*a**1.25)
                rho = rho_all[b].reshape(-1).astype(float)
                dv = (v1_all[:, b].reshape(3, -1).T.astype(float)
                      - v0_all[:, b].reshape(3, -1).T.astype(float))
                ar = np.einsum("ij,ij->i", dv, radial)/(dt*gmag)
                amag2 = np.einsum("ij,ij->i", dv, dv)/(dt*gmag)**2
                maxcos = np.abs(radial).max(axis=1)
                rr = r*length/RSUN_CGS
                for lo, hi in SHELLS:
                    layer = (rr >= lo) & (rr < hi)
                    if not np.any(layer):
                        continue
                    for sector, angular in (("all", np.ones_like(layer)),
                                            ("axis", maxcos > 0.9),
                                            ("diagonal", maxcos < 0.7)):
                        selected = layer & angular
                        if not np.any(selected):
                            continue
                        w = rho[selected]
                        sums[f"{lo}-{hi}:{sector}"] += np.array([
                            selected.sum(), w.sum(), np.dot(w, ar[selected]),
                            np.dot(w, ar[selected]**2), np.dot(w, amag2[selected])])
        metrics = {}
        for key, (cells, mass, radial_sum, radial2_sum, total2_sum) in sums.items():
            if cells == 0:
                continue
            metrics[key] = {
                "cells": int(cells),
                "mass_weighted_mean_radial_acceleration_over_g": float(radial_sum/mass),
                "mass_weighted_rms_radial_acceleration_over_g": float(np.sqrt(radial2_sum/mass)),
                "mass_weighted_rms_vector_acceleration_over_g": float(np.sqrt(total2_sum/mass)),
            }
        result["cases"][name] = {"dt_code": dt, "blocks": nblock, "metrics": metrics}
        print(name, nblock, "blocks analyzed", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(args.output)


if __name__ == "__main__":
    main()
