#!/usr/bin/env python3
"""Measure full-sphere early RSG thermodynamic changes from saved HDF states."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from check_rsg_initial_force_balance import RSUN_CGS, parameters

SHELLS = ((0, 166), (500, 750), (750, 800), (800, 813),
          (813, 817), (817, 830), (830, 850), (850, 880),
          (880, 900), (900, 950), (950, 1010), (1010, 1200),
          (1200, 1750))
FIELDS = ("rho", "press", "Er", "Tgas", "Trad")


def read_fields(prim_path, uov_path):
    with h5py.File(prim_path) as f, h5py.File(uov_path) as u:
        if f["prim"].shape[1:] != u["user_out_var"].shape[1:]:
            raise ValueError("Primitive and user-output grids differ")
        arrays = {
            "rho": np.asarray(f["prim"][0]),
            "press": np.asarray(f["prim"][1]),
            "vx": np.asarray(f["prim"][2]),
            "vy": np.asarray(f["prim"][3]),
            "vz": np.asarray(f["prim"][4]),
            "Er": np.asarray(f["prim"][5]),
            "Tgas": np.asarray(u["user_out_var"][0]),
            "Trad": np.asarray(u["user_out_var"][1]),
            "vr": np.asarray(u["user_out_var"][5]),
        }
        coords = tuple(np.asarray(f[f"x{i}v"]) for i in (1, 2, 3))
        return arrays, coords, float(f.attrs["Time"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--case", nargs=5, action="append", required=True,
                        metavar=("NAME", "INITIAL_PRIM", "INITIAL_UOV", "FINAL_PRIM", "FINAL_UOV"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    p = parameters(args.input)
    code_velocity_kms = np.sqrt(float(p["egas_unit"])/float(p["rho_unit"]))/1e5
    length = float(p["leng_unit"])
    output = {"cases": {}}
    for name, p0, u0, p1, u1 in args.case:
        initial, coords0, t0 = read_fields(p0, u0)
        final, coords1, t1 = read_fields(p1, u1)
        if any(not np.array_equal(a, b) for a, b in zip(coords0, coords1)):
            raise ValueError(f"Coordinates differ for {name}")
        if t1 <= t0:
            raise ValueError(f"Final time is not later for {name}")
        stats = {f"{lo}-{hi}": np.zeros(5+2*len(FIELDS), dtype=float)
                 for lo, hi in SHELLS}
        maxima = {f"{lo}-{hi}": {"speed": 0.0, "vr": 0.0,
                                   "xyz_rsun": None}
                  for lo, hi in SHELLS}
        for block in range(initial["rho"].shape[0]):
            x, y, z = (axis[block] for axis in coords0)
            r = np.sqrt(x[None,None,:]**2 + y[None,:,None]**2 + z[:,None,None]**2)
            rr = r*length/RSUN_CGS
            rho0 = initial["rho"][block].astype(float)
            vr = final["vr"][block].astype(float)*code_velocity_kms
            speed = np.sqrt(sum(final[f"v{axis}"][block].astype(float)**2
                                for axis in "xyz"))*code_velocity_kms
            log_ratios = {}
            for field in FIELDS:
                before = initial[field][block].astype(float)
                after = final[field][block].astype(float)
                if np.any(before <= 0) or np.any(after <= 0):
                    raise ValueError(f"Nonpositive {field} in {name}, block {block}")
                log_ratios[field] = np.log(after/before)
            for lo, hi in SHELLS:
                selected = (rr >= lo) & (rr < hi)
                if not np.any(selected):
                    continue
                w = rho0[selected]
                v = vr[selected]
                entry = stats[f"{lo}-{hi}"]
                entry[0] += selected.sum()
                entry[1] += w.sum()
                entry[2] += np.sum(w*v)
                entry[3] += np.dot(w, v*v)
                entry[4] += np.dot(w, speed[selected]**2)
                vmax_index = int(np.argmax(speed[selected]))
                vmax = float(speed[selected][vmax_index])
                if vmax > maxima[f"{lo}-{hi}"]["speed"]:
                    k, j, i = np.argwhere(selected)[vmax_index]
                    maxima[f"{lo}-{hi}"] = {
                        "speed": vmax,
                        "vr": float(v[vmax_index]),
                        "xyz_rsun": [float(x[i]*length/RSUN_CGS),
                                     float(y[j]*length/RSUN_CGS),
                                     float(z[k]*length/RSUN_CGS)]}
                for j, field in enumerate(FIELDS):
                    q = log_ratios[field][selected]
                    entry[5+2*j] += np.dot(w, q)
                    entry[6+2*j] += np.dot(w, q*q)
        metrics = {}
        for shell, entry in stats.items():
            count, mass, vr_sum = entry[:3]
            if count == 0:
                continue
            metrics[shell] = {
                "cells": int(count),
                "mean_final_vr_kms": float(vr_sum/mass),
                "rms_final_vr_kms": float(np.sqrt(entry[3]/mass)),
                "rms_final_speed_kms": float(np.sqrt(entry[4]/mass)),
                "max_final_speed_kms": maxima[shell]["speed"],
                "vr_at_max_speed_kms": maxima[shell]["vr"],
                "max_final_speed_xyz_rsun": maxima[shell]["xyz_rsun"],
                "fields": {},
            }
            for j, field in enumerate(FIELDS):
                mean_log = entry[5+2*j]/mass
                rms_log = np.sqrt(entry[6+2*j]/mass)
                metrics[shell]["fields"][field] = {
                    "mass_weighted_geometric_mean_ratio": float(np.exp(mean_log)),
                    "mass_weighted_rms_log_ratio": float(rms_log),
                }
        output["cases"][name] = {"time0_code": t0, "time1_code": t1,
                                 "metrics": metrics}
        print(name, "analyzed", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+"\n")
    print(args.output)


if __name__ == "__main__":
    main()
