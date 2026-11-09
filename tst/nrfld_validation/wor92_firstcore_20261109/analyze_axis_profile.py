#!/usr/bin/env python3
"""Measure the two central x-axis profiles and their steepest density jump."""

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

from analyze_core_profile import eos_temperature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hdf5", type=Path)
    parser.add_argument("eos_table", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--rho-unit", type=float, default=1e-18)
    parser.add_argument("--pressure-unit", type=float, default=1e-9)
    parser.add_argument("--length-unit", type=float, default=1.495978707e13)
    args = parser.parse_args()

    with h5py.File(args.hdf5) as h:
        names = [x.decode() for x in h.attrs["VariableNames"]]
        prim_names = names[:int(h.attrs["NumVariables"][0])]
        data = {name: np.asarray(h["prim"][prim_names.index(name)], dtype=np.float64)
                for name in ("rho", "press", "vel1")}
        x = [np.asarray(h["x%sv" % i], dtype=np.float64) for i in (1, 2, 3)]
        xf = [np.asarray(h["x%sf" % i], dtype=np.float64) for i in (1, 2, 3)]
        cycle = int(h.attrs["NumCycles"])
        time = float(h.attrs["Time"])

    shape = data["rho"].shape
    xx = np.broadcast_to(x[0][:, None, None, :], shape)
    yy = np.broadcast_to(x[1][:, None, :, None], shape)
    zz = np.broadcast_to(x[2][:, :, None, None], shape)
    dy = np.broadcast_to(np.diff(xf[1], axis=1)[:, None, :, None], shape)
    dz = np.broadcast_to(np.diff(xf[2], axis=1)[:, :, None, None], shape)
    selection = ((np.abs(xx) <= 30.0) & (np.abs(yy) <= 0.5001 * dy)
                 & (np.abs(zz) <= 0.5001 * dz))
    positions = xx[selection]
    rho = data["rho"][selection] * args.rho_unit
    pressure = data["press"][selection] * args.pressure_unit
    velocity = data["vel1"][selection] * np.sqrt(
        args.pressure_unit / args.rho_unit) / 1e5
    temperature, outside = eos_temperature(args.eos_table, rho, pressure)
    volume = np.broadcast_to(
        np.diff(xf[2], axis=1)[:, :, None, None]
        * np.diff(xf[1], axis=1)[:, None, :, None]
        * np.diff(xf[0], axis=1)[:, None, None, :], shape)[selection]
    grid, index = np.unique(positions, return_inverse=True)
    weights = np.bincount(index, weights=volume)
    mean_rho = np.bincount(index, weights=rho * volume) / weights
    mean_temp = np.bincount(index, weights=temperature * rho * volume) / np.bincount(
        index, weights=rho * volume)
    mean_vel = np.bincount(index, weights=velocity * rho * volume) / np.bincount(
        index, weights=rho * volume)
    counts = np.bincount(index)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("x_au", "rho_g_cm3", "temperature_K", "vx_km_s", "cells"))
        writer.writerows(zip(grid, mean_rho, mean_temp, mean_vel, counts))

    positive = np.flatnonzero((grid >= 3.0) & (grid <= 15.0))
    jumps = []
    for inner, outer in zip(positive[:-1], positive[1:]):
        jumps.append((mean_rho[inner] / mean_rho[outer], inner, outer))
    ratio, inner, outer = max(jumps) if jumps else (None, None, None)
    mirrored_rho = []
    mirrored_temp = []
    negative_lookup = {round(float(-grid[i]), 6): i
                       for i in range(len(grid)) if grid[i] < 0}
    for i in positive:
        other = negative_lookup.get(round(float(grid[i]), 6))
        if other is not None:
            mirrored_rho.append(abs(mean_rho[i] - mean_rho[other])
                                / (0.5 * (mean_rho[i] + mean_rho[other])))
            mirrored_temp.append(abs(mean_temp[i] - mean_temp[other])
                                 / (0.5 * (mean_temp[i] + mean_temp[other])))
    result = {
        "source": str(args.hdf5), "cycle": cycle, "time_code": time,
        "axis_cells": len(positions), "eos_cells_outside_table": outside,
        "rho_max_on_axis_g_cm3": float(max(mean_rho)),
        "temperature_max_on_axis_K": float(max(mean_temp)),
        "positive_negative_x_max_relative_rho_3_to_15_au": (
            float(max(mirrored_rho)) if mirrored_rho else None),
        "positive_negative_x_max_relative_temperature_3_to_15_au": (
            float(max(mirrored_temp)) if mirrored_temp else None),
        "steepest_positive_x_density_drop_3_to_15_au": (
            {"inner_au": float(grid[inner]), "outer_au": float(grid[outer]),
             "inner_over_outer_density": float(ratio),
             "inner_infall_km_s": float(-mean_vel[inner]),
             "outer_infall_km_s": float(-mean_vel[outer])}
            if inner is not None else None),
        "profile_csv": str(args.output_csv),
    }
    args.output_csv.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
