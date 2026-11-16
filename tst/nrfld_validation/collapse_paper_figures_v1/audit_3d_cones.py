#!/usr/bin/env python3
"""Compare 3D Cartesian-axis and body-diagonal cones in a collapse snapshot."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "tst/nrfld_validation/wor92_firstcore_20261109"))
from analyze_core_profile import eos_temperature  # noqa: E402

RHO_UNIT = 1e-18
PRESS_UNIT = 1e-9
A_RAD = 7.5657e-15
SHELLS = ((1, 5), (5, 7), (7, 9), (10, 15), (15, 20), (20, 30))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def analyze(hdf, eos, half_angle_deg):
    with h5py.File(hdf) as h:
        names = [v.decode() for v in h.attrs["VariableNames"]]
        names = names[:int(h.attrs["NumVariables"][0])]
        rho = np.asarray(h["prim"][names.index("rho")], dtype=float) * RHO_UNIT
        pressure = np.asarray(h["prim"][names.index("press")], dtype=float) * PRESS_UNIT
        er = np.asarray(h["prim"][names.index("Er")], dtype=float) * PRESS_UNIT
        x = np.asarray(h["x1v"])[:, None, None, :]
        y = np.asarray(h["x2v"])[:, None, :, None]
        z = np.asarray(h["x3v"])[:, :, None, None]
        xf = np.asarray(h["x1f"])
        yf = np.asarray(h["x2f"])
        zf = np.asarray(h["x3f"])
        volume = np.broadcast_to(
            np.diff(xf, axis=1)[:, None, None, :]
            * np.diff(yf, axis=1)[:, None, :, None]
            * np.diff(zf, axis=1)[:, :, None, None], rho.shape)
        level = np.broadcast_to(np.asarray(h["Levels"])[:, None, None, None], rho.shape)
        cycle, time = int(h.attrs["NumCycles"]), float(h.attrs["Time"])
    radius = np.sqrt(x*x + y*y + z*z)
    cos_half = np.cos(np.deg2rad(half_angle_deg))
    axis = np.maximum(np.maximum(abs(x), abs(y)), abs(z)) / radius >= cos_half
    diagonal = (abs(x) + abs(y) + abs(z)) / (np.sqrt(3) * radius) >= cos_half
    output = {"cycle": cycle, "time_code": time, "half_angle_deg": half_angle_deg,
              "hdf_sha256": sha256(hdf), "eos_sha256": sha256(eos), "shells": []}
    for lo, hi in SHELLS:
        row = {"inner_au": lo, "outer_au": hi}
        for name, cone in (("axis", axis), ("body_diagonal", diagonal)):
            selected = (radius >= lo) & (radius < hi) & cone
            weights = volume[selected]
            if len(weights) == 0:
                row = None
                break
            tgas, outside = eos_temperature(eos, rho[selected], pressure[selected])
            if outside:
                raise ValueError("EOS domain exceeded")
            row[name] = {
                "cells": int(len(weights)),
                "mean_radius_au": float(np.average(np.broadcast_to(radius, rho.shape)[selected], weights=weights)),
                "mean_rho_g_cm3": float(np.average(rho[selected], weights=weights)),
                "mean_Tgas_K": float(np.average(tgas, weights=weights)),
                "mean_Trad_K": float(np.average((er[selected] / A_RAD)**0.25, weights=weights)),
                "level_min": int(level[selected].min()),
                "level_max": int(level[selected].max()),
                "mean_level": float(np.average(level[selected], weights=weights)),
            }
        if row is None:
            continue
        a, d = row["axis"], row["body_diagonal"]
        for key in ("rho_g_cm3", "Tgas_K", "Trad_K"):
            av, dv = a["mean_" + key], d["mean_" + key]
            row[key + "_axis_minus_diag_fraction"] = (av - dv) / ((av + dv) / 2)
        output["shells"].append(row)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf", type=Path, required=True)
    parser.add_argument("--eos", type=Path, required=True)
    parser.add_argument("--expected-hdf-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--half-angle", type=float, default=15)
    args = parser.parse_args()
    if sha256(args.hdf) != args.expected_hdf_sha256:
        raise ValueError("HDF SHA mismatch")
    result = analyze(args.hdf, args.eos, args.half_angle)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for shell in result["shells"]:
        print(f"c{result['cycle']} {shell['inner_au']:2}-{shell['outer_au']:2} au "
              f"dT={shell['Tgas_K_axis_minus_diag_fraction']:+.2%} "
              f"drho={shell['rho_g_cm3_axis_minus_diag_fraction']:+.2%} "
              f"levels={shell['axis']['level_min']}-{shell['axis']['level_max']}/"
              f"{shell['body_diagonal']['level_min']}-{shell['body_diagonal']['level_max']}")


if __name__ == "__main__":
    main()
