#!/usr/bin/env python3
"""Measure temperature/density retention of an imposed RSG buoyancy seed."""

import argparse
import glob
import json
from pathlib import Path

import h5py
import numpy as np


def snapshot(run_dir, index, zmin=813.0, zmax=817.0):
    path = glob.glob(str(run_dir / f"*.prim_full.{index:05d}.athdf"))
    uov_path = glob.glob(str(run_dir / f"*.uov_full.{index:05d}.athdf"))
    if len(path) != 1 or len(uov_path) != 1:
        raise ValueError(f"expected matching full-domain snapshot {index}")
    with h5py.File(path[0]) as hp, h5py.File(uov_path[0]) as hu:
        if not np.array_equal(hp["LogicalLocations"][:],
                              hu["LogicalLocations"][:]):
            raise ValueError("primitive and user-output block orders differ")
        if abs(float(hp.attrs["Time"]) - float(hu.attrs["Time"])) > 1e-10:
            raise ValueError("primitive and user-output times differ")
        lx = float(np.diff(hp.attrs["RootGridX1"][:2])[0])
        ly = float(np.diff(hp.attrs["RootGridX2"][:2])[0])
        x0 = float(hp.attrs["RootGridX1"][0])
        y0 = float(hp.attrs["RootGridX2"][0])
        x = hp["x1v"][:]
        y = hp["x2v"][:]
        z = hp["x3v"][:]
        # Older patch inputs use a height above 800 Rsun; newer ones use
        # the physical radius directly. Keep both conventions comparable.
        radius_offset = (800.0 if float(hp.attrs["RootGridX3"][0]) < 100.0
                         else 0.0)
        rho = hp["prim"][0]
        press = hp["prim"][1]
        temp = hu["user_out_var"][2]
        sound = hu["user_out_var"][6]
        vz = hp["prim"][4]
        time = float(hp.attrs["Time"])

    levels = {}
    for block in range(z.shape[0]):
        pattern = (np.sin(2*np.pi*(x[block]-x0)/lx)[None, :]
                   * np.sin(2*np.pi*(y[block]-y0)/ly)[:, None])
        for k, zcell in enumerate(z[block]):
            height = float(zcell) + radius_offset
            if zmin <= height < zmax:
                item = levels.setdefault(height, {"pattern": [], "rho": [],
                                                  "press": [], "temp": [],
                                                  "sound": [], "vz": []})
                item["pattern"].append(pattern.reshape(-1))
                for key, values in (("rho", rho), ("press", press), ("temp", temp),
                                    ("sound", sound), ("vz", vz)):
                    item[key].append(values[block, k].reshape(-1))

    profiles = []
    for height, item in sorted(levels.items()):
        pattern = np.concatenate(item["pattern"]).astype(np.float64)
        denom = np.mean(pattern*pattern)
        row = {"z_rsun": height}
        for key in ("rho", "press", "temp", "sound", "vz"):
            values = np.concatenate(item[key]).astype(np.float64)
            mean = np.mean(values)
            row[f"mean_{key}"] = float(mean)
            row[f"rms_{key}"] = float(np.std(values))
            row[f"mode_{key}"] = float(np.mean((values-mean)*pattern)/denom)
            if key in ("rho", "press", "temp"):
                row[f"rms_{key}_fraction"] = float(np.std(values)/mean)
                row[f"mode_{key}_fraction"] = float(row[f"mode_{key}"]/mean)
        gamma1 = (row["mean_sound"]**2 * row["mean_rho"] * 1e-8
                  / (row["mean_press"] * 1e5))
        row["gamma1"] = gamma1
        row["mode_adiabatic_residual"] = (
            row["mode_press_fraction"] - gamma1*row["mode_rho_fraction"])
        profiles.append(row)
    shell = {key: float(np.mean([row[key] for row in profiles]))
             for key in profiles[0] if key != "z_rsun"}
    return {"time_code": time, "shell": shell, "height_profiles": profiles}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--snapshots", default="0,1,2")
    args = parser.parse_args()
    indices = [int(i) for i in args.snapshots.split(",")]
    result = {}
    for run_dir in args.run_dirs:
        cases = [snapshot(run_dir, index) for index in indices]
        init = cases[0]["shell"]
        for case in cases:
            shell = case["shell"]
            for key in ("mode_temp_fraction", "mode_rho_fraction",
                        "mode_press_fraction", "mode_adiabatic_residual"):
                shell[f"{key}_over_initial"] = shell[key]/init[key]
        result[run_dir.name] = cases
        for case in cases:
            shell = case["shell"]
            print(f"{run_dir.name}: t={case['time_code']:.1f} "
                  f"T-mode={shell['mode_temp_fraction_over_initial']:.4f} "
                  f"rho-mode={shell['mode_rho_fraction_over_initial']:.4f} "
                  f"T-rms={shell['rms_temp_fraction']:.5f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
