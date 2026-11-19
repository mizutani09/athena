#!/usr/bin/env python3
"""Measure horizontally averaged transport in a local RSG surface patch."""

import argparse
import glob
import json
from pathlib import Path

import h5py
import numpy as np


FSTAR_CGS = 8.389e9
RHO_UNIT = 1e-8
EGAS_UNIT = 1e5
VEL_UNIT = np.sqrt(EGAS_UNIT / RHO_UNIT)
SHELLS = ((800, 813), (813, 817), (817, 830), (830, 850), (850, 860))


def history(run_dir):
    files = glob.glob(str(run_dir / "*.hst"))
    if len(files) != 1:
        raise ValueError(f"expected one history file in {run_dir}")
    h = np.loadtxt(files[0], ndmin=2)
    return [{"time_code": float(row[0]),
             "top_Frad_over_Fstar": float(row[17] / FSTAR_CGS),
             "top_Ftotal_over_Fstar": float(row[18] / FSTAR_CGS),
             "mass_drift": float(row[21]),
             "bottom_heat_power_code": float(row[24]),
             "surface_heat_power_code": float(row[25]),
             "total_energy_history_units": float(row[16])} for row in h]


def load_snapshot(run_dir, index):
    prefix = str(run_dir / f"*.{{kind}}.{index:05d}.athdf")
    paths = {}
    for kind in ("prim_full", "uov_full"):
        matches = glob.glob(prefix.format(kind=kind))
        if len(matches) != 1:
            raise ValueError(f"expected one {kind} snapshot {index}")
        paths[kind] = matches[0]
    with h5py.File(paths["prim_full"]) as hp, h5py.File(paths["uov_full"]) as hu:
        if abs(float(hp.attrs["Time"]) - float(hu.attrs["Time"])) > 1e-10:
            raise ValueError("primitive and user-output times differ")
        if not np.array_equal(hp["LogicalLocations"][:], hu["LogicalLocations"][:]):
            raise ValueError("primitive and user-output block orders differ")
        z = hp["x3v"][:].reshape(-1) + 800.0
        # Athena++ HDF arrays are (variable, block, z, y, x), while x3v
        # is (block, z).  Reshape only after checking the metadata.
        prim_hdf = hp["prim"][:]
        uov_hdf = hu["user_out_var"][:]
        nblock = int(hp.attrs["NumMeshBlocks"])
        if prim_hdf.shape[:2] != (6, nblock) or uov_hdf.shape[:2] != (14, nblock):
            raise ValueError("unexpected HDF variable/block axis order")
        prim = prim_hdf.reshape(6, z.size, -1)
        uov = uov_hdf.reshape(14, z.size, -1)
        cell_volume = float(np.diff(hp["x1f"][0, :2])[0]
                            * np.diff(hp["x2f"][0, :2])[0]
                            * np.diff(hp["x3f"][0, :2])[0])
        time = float(hp.attrs["Time"])

    unique_z, inverse = np.unique(z, return_inverse=True)
    profiles = {key: [] for key in (
        "z_rsun", "Tgas_K", "Frad_over_Fstar", "Fenth_over_Fstar",
        "Fkin_over_Fstar", "Ftotal_over_Fstar",
        "Fenth_mass_corrected_over_Fstar", "mean_vz_kms",
        "rms_vz_kms", "rms_horizontal_vz_kms", "rms_horizontal_T_fraction",
        "max_mach", "mean_rho_cgs")}
    for level, height in enumerate(unique_z):
        where = inverse == level
        rho = prim[0, where].reshape(-1) * RHO_UNIT
        press = prim[1, where].reshape(-1) * EGAS_UNIT
        vz = prim[4, where].reshape(-1) * VEL_UNIT
        egas = uov[0, where].reshape(-1)
        h = egas + press
        favre_vz = np.mean(rho * vz) / np.mean(rho)
        fenth = np.mean(uov[11, where])
        profiles["z_rsun"].append(float(height))
        profiles["Tgas_K"].append(float(np.mean(uov[2, where])))
        profiles["Frad_over_Fstar"].append(float(np.mean(uov[10, where]) / FSTAR_CGS))
        profiles["Fenth_over_Fstar"].append(float(fenth / FSTAR_CGS))
        profiles["Fkin_over_Fstar"].append(float(np.mean(uov[12, where]) / FSTAR_CGS))
        profiles["Ftotal_over_Fstar"].append(float(np.mean(uov[13, where]) / FSTAR_CGS))
        profiles["Fenth_mass_corrected_over_Fstar"].append(
            float((fenth - np.mean(h) * favre_vz) / FSTAR_CGS))
        profiles["mean_vz_kms"].append(float(np.mean(vz) / 1e5))
        profiles["rms_vz_kms"].append(float(np.sqrt(np.mean(vz * vz)) / 1e5))
        profiles["rms_horizontal_vz_kms"].append(float(np.std(vz) / 1e5))
        temperature = uov[2, where].reshape(-1)
        profiles["rms_horizontal_T_fraction"].append(
            float(np.std(temperature) / np.mean(temperature)))
        profiles["max_mach"].append(float(np.max(uov[7, where])))
        profiles["mean_rho_cgs"].append(float(np.mean(rho)))

    shells = {}
    z_array = np.asarray(profiles["z_rsun"])
    for low, high in SHELLS:
        mask = (z_array >= low) & (z_array < high)
        shells[f"{low}-{high}"] = {
            key: float(np.mean(np.asarray(value)[mask]))
            for key, value in profiles.items() if key != "z_rsun" and key != "max_mach"}
        shells[f"{low}-{high}"]["max_mach"] = float(
            np.max(np.asarray(profiles["max_mach"])[mask]))
    mass_code = float(np.sum(prim[0], dtype=np.float64) * cell_volume)
    return {"time_code": time, "mass_code": mass_code,
            "shells": shells, "height_profiles": profiles}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshots", default="0,1,2,3,4,5",
                        help="comma-separated HDF snapshot indices")
    args = parser.parse_args()
    cases = [load_snapshot(args.run_dir, int(i))
             for i in args.snapshots.split(",")]
    mass0 = cases[0]["mass_code"]
    for case in cases:
        case["mass_drift_from_hdf"] = case["mass_code"] / mass0 - 1.0
        for name, shell in case["shells"].items():
            shell["Tgas_over_initial"] = (
                shell["Tgas_K"] / cases[0]["shells"][name]["Tgas_K"])
    result = {"run_dir": str(args.run_dir), "history": history(args.run_dir),
              "snapshots": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for case in cases:
        shell = case["shells"]["813-817"]
        print(f"t={case['time_code']:5.1f} "
              f"T/T0={shell['Tgas_over_initial']:.4f} "
              f"Frad={shell['Frad_over_Fstar']:.4f} "
              f"Fenth={shell['Fenth_over_Fstar']:.4f} "
              f"Fconv={shell['Fenth_mass_corrected_over_Fstar']:.4f} "
              f"Machmax={shell['max_mach']:.3f} "
              f"mass_drift={case['mass_drift_from_hdf']:.4f}")


if __name__ == "__main__":
    main()
