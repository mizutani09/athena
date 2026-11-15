#!/usr/bin/env python3
"""Decompose the initial RSG gas-pressure force on one uniform-grid block.

Read the full initial pressure field to use true neighboring cells across
MeshBlock boundaries.  Compare the continuum 1-D derivative, a Cartesian
centered difference of sampled 1-D pressure, a centered difference of mapped
3-D HDF pressure, and the actual Godunov gas flux divergence from the log.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from check_rsg_initial_force_balance import (G_CGS, MSUN_CGS, RSUN_CGS,
                                             force_rows, parameters)

SHELLS = ((750, 800), (800, 813), (813, 817), (817, 830),
          (830, 850), (850, 880))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--initial", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--gid", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    settings = parameters(args.input)
    length = float(settings["leng_unit"])
    energy = float(settings["egas_unit"])
    rho_unit = float(settings["rho_unit"])
    gm = G_CGS * float(settings["potential_mass_msun"]) * MSUN_CGS * rho_unit / (length * energy)
    core = float(settings["radius_rsun"]) * RSUN_CGS / length * float(settings["core_radius_fraction"])
    outer = float(settings["radius_rsun"]) * RSUN_CGS / length * float(settings["outer_flattening_fraction"])
    profile = np.loadtxt(args.profile)
    profile_r = profile[:, 0] / length
    profile_p = profile[:, 5] / energy
    profile_dpdr = np.gradient(profile_p, profile_r, edge_order=2)

    rows = force_rows(args.log)
    xyz = rows[:, :3]
    with h5py.File(args.initial) as hdf:
        locations = np.asarray(hdf["LogicalLocations"], dtype=int)
        if np.any(np.asarray(hdf["Levels"]) != 0):
            raise ValueError("Uniform level-zero grid required")
        shape = hdf["prim"].shape[2:]
        if len(set(shape)) != 1:
            raise ValueError("Cubic blocks required")
        nb = shape[0]
        n = (locations.max(axis=0) + 1) * nb
        pressure = np.empty((n[2], n[1], n[0]), dtype=np.float64)
        block_pressure = np.asarray(hdf["prim"][1], dtype=np.float64)
        for block, (bx, by, bz) in enumerate(locations):
            pressure[bz*nb:(bz+1)*nb,
                     by*nb:(by+1)*nb,
                     bx*nb:(bx+1)*nb] = block_pressure[block]
        del block_pressure
        rho = np.asarray(hdf["prim"][0, args.gid], dtype=np.float64).reshape(-1)
        mapped_p = np.asarray(hdf["prim"][1, args.gid], dtype=np.float64).reshape(-1)
        centers = [np.asarray(hdf[f"x{axis}v"][args.gid], dtype=np.float64)
                   for axis in (1, 2, 3)]
        expected = np.stack(np.meshgrid(*centers, indexing="ij"), axis=-1)
        expected = expected.transpose(2, 1, 0, 3).reshape(-1, 3)
        spacing = np.array([hdf[f"x{axis}f"][args.gid, 1]
                            - hdf[f"x{axis}f"][args.gid, 0]
                            for axis in (1, 2, 3)], dtype=float)
        origin = np.array([hdf[f"x{axis}f"][:, 0].min()
                           for axis in (1, 2, 3)], dtype=float)
        mass = rho * np.prod(spacing)
    if rows.shape != (rho.size, 9) or not np.allclose(xyz, expected, atol=1e-5):
        raise ValueError("Force rows do not match selected HDF block")
    ijk = np.rint((xyz - origin) / spacing - 0.5).astype(int)
    interior = np.all((ijk >= 1) & (ijk < n - 1), axis=1)
    excluded_boundary_cells = int((~interior).sum())
    rows, xyz, rho, mapped_p, mass, ijk = (value[interior] for value in
                                          (rows, xyz, rho, mapped_p, mass, ijk))
    ix, iy, iz = ijk.T
    sampled_force = np.zeros_like(xyz)
    mapped_force = np.empty_like(xyz)
    for axis in range(3):
        offset = np.zeros(3)
        offset[axis] = spacing[axis]
        p_plus = np.interp(np.linalg.norm(xyz + offset, axis=1), profile_r, profile_p)
        p_minus = np.interp(np.linalg.norm(xyz - offset, axis=1), profile_r, profile_p)
        sampled_force[:, axis] = -(p_plus - p_minus) / (2*spacing[axis])
    mapped_force[:, 0] = -(pressure[iz, iy, ix+1] - pressure[iz, iy, ix-1]) / (2*spacing[0])
    mapped_force[:, 1] = -(pressure[iz, iy+1, ix] - pressure[iz, iy-1, ix]) / (2*spacing[1])
    mapped_force[:, 2] = -(pressure[iz+1, iy, ix] - pressure[iz-1, iy, ix]) / (2*spacing[2])
    r = np.linalg.norm(xyz, axis=1)
    radial = xyz / r[:, None]
    s = (r / outer) ** 8
    a = core ** 4 + r ** 4 / np.sqrt(1+s)
    gmag = gm * r ** 3 / ((1+s) ** 1.5 * a ** 1.25)
    norms = rho * gmag
    actual_force = rows[:, 6:9]
    continuum = -np.interp(r, profile_r, profile_dpdr) / norms
    cases = {
        "continuum_profile": continuum,
        "sampled_profile_centered": np.einsum("ij,ij->i", sampled_force, radial) / norms,
        "mapped_hdf_centered": np.einsum("ij,ij->i", mapped_force, radial) / norms,
        "actual_godunov": np.einsum("ij,ij->i", actual_force, radial) / norms,
    }
    reference_p = np.interp(r, profile_r, profile_p)
    pressure_deviation = (mapped_p - reference_p) / reference_p
    result = {"gid": args.gid, "mesh_n": n.tolist(),
              "excluded_domain_boundary_cells": excluded_boundary_cells,
              "dx_rsun": (spacing*length/RSUN_CGS).tolist(),
              "shells": {}}
    for lo, hi in SHELLS:
        selected = (r*length/RSUN_CGS >= lo) & (r*length/RSUN_CGS < hi)
        if not np.any(selected):
            continue
        w = mass[selected]
        means = {name: float(np.average(value[selected], weights=w))
                 for name, value in cases.items()}
        differences = {
            "sampling_minus_continuum": cases["sampled_profile_centered"] - continuum,
            "mapped_minus_sampling": cases["mapped_hdf_centered"] - cases["sampled_profile_centered"],
            "godunov_minus_mapped": cases["actual_godunov"] - cases["mapped_hdf_centered"],
        }
        result["shells"][f"{lo}-{hi}"] = {
            "cells": int(np.count_nonzero(selected)),
            "gas_support_over_g": means,
            "difference_mean_over_g": {
                name: float(np.average(value[selected], weights=w))
                for name, value in differences.items()},
            "difference_rms_over_g": {
                name: float(np.sqrt(np.average(value[selected]**2, weights=w)))
                for name, value in differences.items()},
            "mapped_pressure_minus_profile_relative_mean": float(
                np.average(pressure_deviation[selected], weights=w)),
            "mapped_pressure_minus_profile_relative_rms": float(
                np.sqrt(np.average(pressure_deviation[selected]**2, weights=w))),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
