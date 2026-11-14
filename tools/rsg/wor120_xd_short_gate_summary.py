#!/usr/bin/env python3
"""Summarize short XD RSG runs from Athena++ history and primitive outputs."""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TIME_DAY = 6.957e10 / np.sqrt(1.0e5 / 1.0e-8) / 86400.0
SHELLS = ((0, 166), (750, 800), (800, 830), (830, 900))


def snapshot(primitive, user_output):
    with h5py.File(primitive) as f, h5py.File(user_output) as u:
        time = float(f.attrs["Time"])
        if abs(time - float(u.attrs["Time"])) > 1.0e-5:
            raise ValueError(f"Output times do not match: {primitive}, {user_output}")
        names = [name.decode() for name in f.attrs["VariableNames"]]
        rho = np.asarray(f["prim"][names.index("rho")], dtype=float)
        pressure = np.asarray(f["prim"][names.index("press")], dtype=float)
        velocity = [np.asarray(f["prim"][names.index(f"vel{i}")], dtype=float)
                    for i in (1, 2, 3)]
        x = np.asarray(f["x1v"])[:, None, None, :]
        y = np.asarray(f["x2v"])[:, None, :, None]
        z = np.asarray(f["x3v"])[:, :, None, None]
        radius = np.sqrt(x*x + y*y + z*z)
        radial_velocity = sum(v*coord for v, coord in zip(velocity, (x, y, z)))
        radial_velocity /= np.maximum(radius, 1.0e-30)
        temperature = np.asarray(u["user_out_var"][0], dtype=float)
        speed = np.sqrt(sum(v*v for v in velocity))
        sound_proxy = np.sqrt((5.0/3.0)*pressure/rho)
        signal_proxy = speed + sound_proxy
        limiting_cell = np.unravel_index(np.nanargmax(signal_proxy), signal_proxy.shape)
        shells = {}
        for low, high in SHELLS:
            mask = (radius >= low) & (radius < high)
            weights = rho[mask]
            shells[f"{low}-{high}"] = {
                "cells": int(mask.sum()),
                "temperature_K": float(np.average(temperature[mask], weights=weights)),
                "radial_velocity_code": float(np.average(radial_velocity[mask], weights=weights)),
            }
        return {
            "time_code": time, "time_day": time*TIME_DAY, "shells": shells,
            "largest_signal_proxy_cell": {
                "radius_Rsun": float(radius[limiting_cell]),
                "density_code": float(rho[limiting_cell]),
                "pressure_code": float(pressure[limiting_cell]),
                "temperature_K": float(temperature[limiting_cell]),
                "speed_code": float(speed[limiting_cell]),
                "sqrt_gammaP_over_rho_code": float(sound_proxy[limiting_cell]),
                "speed_plus_sqrt_gammaP_over_rho_code": float(signal_proxy[limiting_cell]),
            },
        }


def summarize(directory):
    histories = list(directory.glob("*.hst"))
    if len(histories) != 1:
        raise ValueError(f"Expected one history in {directory}, found {histories}")
    history = np.atleast_2d(np.loadtxt(histories[0]))
    if history.shape[1] < 28:
        raise ValueError(f"Missing six-face luminosities in {histories[0]}")
    light = history[:, 22:28].sum(axis=1) / history[:, 15]
    times = history[:, 0] * TIME_DAY
    primitives = sorted(directory.glob("*.prim.*.athdf"))
    user_outputs = sorted(directory.glob("*.uov.*.athdf"))
    if len(primitives) < 2 or len(user_outputs) < 2:
        raise ValueError(f"Need distinct initial and final outputs in {directory}")
    states = []
    for p in (primitives[0], primitives[-1]):
        with h5py.File(p) as f:
            time = float(f.attrs["Time"])
        matches = []
        for u in user_outputs:
            with h5py.File(u) as f:
                matches.append((abs(float(f.attrs["Time"]) - time), u))
        delta, u = min(matches)
        if delta > 1.0e-5:
            raise ValueError(f"No matching user output for {p}")
        states.append(snapshot(p, u))
    surface = "800-830"
    cooling = 1.0 - (states[-1]["shells"][surface]["temperature_K"] /
                     states[0]["shells"][surface]["temperature_K"])
    return {
        "directory": str(directory),
        "history": {
            "time_day": times.tolist(),
            "boundary_luminosity_over_Lstar": light.tolist(),
            "Mach_max": history[:, 16].tolist(),
            "relative_mass_change": float(history[-1, 2]/history[0, 2] - 1.0),
        },
        "snapshots": states,
        "surface_cooling_fraction": float(cooling),
    }


def plot(cases, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for name, case in cases.items():
        history = case["history"]
        states = case["snapshots"]
        axes[0].plot(history["time_day"], history["boundary_luminosity_over_Lstar"],
                     marker=".", label=name)
        axes[1].plot(history["time_day"], history["Mach_max"], marker=".", label=name)
        axes[2].plot([state["time_day"] for state in states],
                     [state["shells"]["800-830"]["temperature_K"] for state in states],
                     marker="o", label=name)
    axes[0].axhline(1, color="black", linewidth=0.7, linestyle="--")
    for ax, title, ylabel in zip(
            axes,
            ("Box radiation loss", "Maximum Mach", "Surface gas cooling"),
            ("Lbox / Lstar", "Mach max", "Mass weighted T at 800–830 Rsun [K]")):
        ax.set(title=title, xlabel="Time [days]", ylabel=ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", required=True, help="label:run-directory")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    cases = {}
    for item in args.case:
        name, directory = item.split(":", 1)
        cases[name] = summarize(Path(directory))
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({"time_unit_day": TIME_DAY, "cases": cases}, indent=2) + "\n")
    plot(cases, args.figure)
    print(args.json)
    print(args.figure)


if __name__ == "__main__":
    main()
