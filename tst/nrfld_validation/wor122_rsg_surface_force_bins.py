#!/usr/bin/env python3
"""Compare first-stage 3-D RSG surface forces with the mapped 1-D HSE profile."""

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
    parser.add_argument("--final", type=Path,
                        help="optional first-step primitive HDF for a velocity check")
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--gid", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    settings = parameters(args.input)
    length = float(settings["leng_unit"])
    velocity2 = float(settings["egas_unit"]) / float(settings["rho_unit"])
    gm = G_CGS * float(settings["potential_mass_msun"]) * MSUN_CGS / (length * velocity2)
    radius = float(settings["radius_rsun"]) * RSUN_CGS / length
    core = radius * float(settings["core_radius_fraction"])
    outer = radius * float(settings["outer_flattening_fraction"])

    rows = force_rows(args.log)
    xyz = rows[:, :3]
    with h5py.File(args.initial) as hdf:
        rho = np.asarray(hdf["prim"][0, args.gid], dtype=float).reshape(-1)
        x, y, z = (np.asarray(hdf[f"x{axis}v"][args.gid], dtype=float)
                   for axis in (1, 2, 3))
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        hdf_xyz = np.stack((xx, yy, zz), axis=-1).transpose(2, 1, 0, 3).reshape(-1, 3)
        dx, dy, dz = (np.diff(hdf[f"x{axis}f"][args.gid])
                      for axis in (1, 2, 3))
        volume = (dz[:, None, None] * dy[None, :, None]
                  * dx[None, None, :]).reshape(-1)
        initial_velocity = np.asarray(hdf["prim"][2:5, args.gid], dtype=float).reshape(3, -1).T
        initial_time = float(hdf.attrs["Time"])
    if rows.shape != (rho.size, 9) or not np.allclose(xyz, hdf_xyz, atol=1e-5):
        raise ValueError("Force rows do not match the chosen initial HDF block")

    r = np.linalg.norm(xyz, axis=1)
    radial = xyz / r[:, None]
    s = (r / outer) ** 8
    a = core ** 4 + r ** 4 / np.sqrt(1 + s)
    gravity_magnitude = gm * r ** 3 / ((1 + s) ** 1.5 * a ** 1.25)
    observed_acceleration_over_g = None
    observed_radial_acceleration_over_g = None
    if args.final is not None:
        with h5py.File(args.final) as hdf:
            final_velocity = np.asarray(hdf["prim"][2:5, args.gid], dtype=float).reshape(3, -1).T
            dt = float(hdf.attrs["Time"]) - initial_time
        if dt <= 0.0:
            raise ValueError("Final snapshot must follow the initial snapshot")
        observed_radial_acceleration_over_g = (
            np.einsum("ij,ij->i", final_velocity - initial_velocity, radial)
            / (dt * gravity_magnitude))
        observed_acceleration_over_g = ((final_velocity - initial_velocity)
                                        / (dt * gravity_magnitude[:, None]))
    forces = {
        "gas": rows[:, 6:9],
        "radiation": rows[:, 3:6],
        "gravity": -rho[:, None] * gravity_magnitude[:, None] * radial,
    }
    residual = sum(forces.values())
    residual_acceleration_over_g = residual / (rho * gravity_magnitude)[:, None]
    residual_radial_acceleration_over_g = np.einsum(
        "ij,ij->i", residual_acceleration_over_g, radial)
    residual_tangential_acceleration_over_g = np.sqrt(np.maximum(
        0.0, np.sum(residual_acceleration_over_g ** 2, axis=1)
        - residual_radial_acceleration_over_g ** 2))
    support = sum(np.linalg.norm(force, axis=1) for force in forces.values())
    profile = np.loadtxt(args.profile)
    continuum_hse = np.interp(r * length, profile[:, 0], profile[:, 10])
    profile_r = profile[:, 0]
    profile_core, profile_outer = core * length, outer * length
    profile_s = (profile_r / profile_outer) ** 8
    profile_a = profile_core ** 4 + profile_r ** 4 / np.sqrt(1 + profile_s)
    profile_gravity = (G_CGS * float(settings["potential_mass_msun"]) * MSUN_CGS
                       * profile_r ** 3 / ((1 + profile_s) ** 1.5 * profile_a ** 1.25))
    continuum_gas = np.interp(
        r * length, profile_r,
        -np.gradient(profile[:, 5], profile_r, edge_order=2)
        / (profile[:, 1] * profile_gravity))
    continuum_rad = np.interp(
        r * length, profile_r,
        profile[:, 8] * profile[:, 9] / (2.99792458e10 * profile_gravity))
    mass = rho * volume

    result = {"gid": args.gid, "time_code": 0.0, "shells": {}}
    for lo, hi in SHELLS:
        selected = (r >= lo) & (r < hi)
        if not selected.any():
            continue
        m = mass[selected]
        total_acceleration = np.linalg.norm(residual[selected], axis=1) / rho[selected]
        relative_acceleration = total_acceleration / gravity_magnitude[selected]
        radial_terms = {}
        for name, force in (*forces.items(), ("residual", residual)):
            acceleration_over_g = (np.einsum("ij,ij->i", force[selected], radial[selected])
                                   / (rho[selected] * gravity_magnitude[selected]))
            radial_terms[name] = float(np.average(acceleration_over_g, weights=m))
        continuum_terms = {
            "gas": float(np.average(continuum_gas[selected], weights=m)),
            "radiation": float(np.average(continuum_rad[selected], weights=m)),
        }
        continuum_terms["residual"] = (continuum_terms["gas"]
                                        + continuum_terms["radiation"] - 1.0)
        peak = np.flatnonzero(selected)[np.argmax(relative_acceleration)]
        result["shells"][f"{lo}-{hi}"] = {
            "cells": int(selected.sum()),
            "mass_fraction_of_selected_block": float(m.sum() / mass.sum()),
            "residual_over_force_l1": float(
                np.linalg.norm(residual[selected], axis=1).sum() / support[selected].sum()),
            "mass_weighted_rms_abs_acceleration_over_g": float(
                np.sqrt(np.average(relative_acceleration ** 2, weights=m))),
            "mass_weighted_rms_radial_acceleration_over_g": float(
                np.sqrt(np.average(
                    residual_radial_acceleration_over_g[selected] ** 2, weights=m))),
            "mass_weighted_rms_tangential_acceleration_over_g": float(
                np.sqrt(np.average(
                    residual_tangential_acceleration_over_g[selected] ** 2, weights=m))),
            "mass_weighted_mean_radial_acceleration_over_g": radial_terms,
            "continuum_mean_radial_acceleration_over_g": continuum_terms,
            "profile_mass_weighted_mean_abs_hse_relative": float(
                np.average(np.abs(continuum_hse[selected]), weights=m)),
            "peak_abs_acceleration_over_g": float(
                np.linalg.norm(residual[peak]) / (rho[peak] * gravity_magnitude[peak])),
            "peak_xyz_rsun": xyz[peak].tolist(),
            "peak_r_rsun": float(r[peak]),
        }
        if observed_radial_acceleration_over_g is not None:
            observed = observed_radial_acceleration_over_g[selected]
            observed_vector = observed_acceleration_over_g[selected]
            observed_tangential = np.sqrt(np.maximum(
                0.0, np.sum(observed_vector ** 2, axis=1) - observed ** 2))
            result["shells"][f"{lo}-{hi}"]["observed_first_step_radial_acceleration_over_g"] = {
                "mass_weighted_mean": float(np.average(observed, weights=m)),
                "mass_weighted_rms": float(np.sqrt(np.average(observed ** 2, weights=m))),
                "mass_weighted_rms_tangential": float(
                    np.sqrt(np.average(observed_tangential ** 2, weights=m))),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
