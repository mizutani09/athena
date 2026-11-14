#!/usr/bin/env python3
"""Estimate gas enthalpy transport in saved RSG spherical shells.

The shell-volume average is a diagnostic of unresolved surface transport, not
an exact finite-volume flux through a spherical face.
"""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
RHO_UNIT = 1.0e-8
EGAS_UNIT = 1.0e5
RSUN = 6.957e10
TIME_DAY = RSUN / np.sqrt(EGAS_UNIT/RHO_UNIT) / 86400.0
SHELLS = ((166, 300), (500, 700), (700, 750), (750, 800), (800, 830))


def load_eos(path):
    """Read the Athena EOS pressure-coordinate field needed for enthalpy."""
    def tokens():
        for line in path.read_text().splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                yield from line.split()
    values = iter(tokens())
    nvar, nq, nrho = (int(next(values)) for _ in range(3))
    qlim = [float(next(values)) for _ in range(2)]
    rholim = [float(next(values)) for _ in range(2)]
    ratios = [float(next(values)) for _ in range(nvar)]
    fields = np.fromiter((float(next(values)) for _ in range(nvar*nq*nrho)),
                         dtype=float, count=nvar*nq*nrho).reshape(nvar, nq, nrho)
    return np.linspace(*qlim, nq), np.linspace(*rholim, nrho), ratios[1], fields[1]


def interpolate_e_over_p(eos, logq, logrho):
    q_axis, rho_axis, _, field = eos
    uq = (logq-q_axis[0])/(q_axis[-1]-q_axis[0])*(q_axis.size-1)
    ur = (logrho-rho_axis[0])/(rho_axis[-1]-rho_axis[0])*(rho_axis.size-1)
    valid = (uq >= 0) & (uq <= q_axis.size-1) & (ur >= 0) & (ur <= rho_axis.size-1)
    if not valid.all():
        raise ValueError(f"{(~valid).sum()} EOS states outside the table")
    iq = np.minimum(uq.astype(int), q_axis.size-2)
    ir = np.minimum(ur.astype(int), rho_axis.size-2)
    fq, fr = uq-iq, ur-ir
    return ((1-fq)*(1-fr)*field[iq, ir] + fq*(1-fr)*field[iq+1, ir]
            + (1-fq)*fr*field[iq, ir+1] + fq*fr*field[iq+1, ir+1])


def matching_outputs(directory):
    primitives = sorted(directory.glob("*.prim.*.athdf"))
    user_outputs = sorted(directory.glob("*.uov.*.athdf"))
    if len(primitives) < 2 or len(user_outputs) < 2:
        raise ValueError(f"Need distinct initial and final states in {directory}")
    result = []
    user_times = []
    for path in user_outputs:
        with h5py.File(path) as handle:
            user_times.append(float(handle.attrs["Time"]))
    for path in (primitives[0], primitives[-1]):
        with h5py.File(path) as handle:
            time = float(handle.attrs["Time"])
        index = int(np.argmin(np.abs(np.asarray(user_times)-time)))
        if abs(user_times[index]-time) > 1.0e-5:
            raise ValueError(f"No matching temperature output for {path}")
        result.append((path, user_outputs[index], time))
    return result


def shell_transport(primitive, user_output, eos, luminosity):
    with h5py.File(primitive) as f, h5py.File(user_output) as u:
        names = [value.decode() for value in f.attrs["VariableNames"]]
        rho = np.asarray(f["prim"][names.index("rho")], dtype=float)
        pressure = np.asarray(f["prim"][names.index("press")], dtype=float)
        velocity = [np.asarray(f["prim"][names.index(f"vel{i}")], dtype=float)
                    for i in (1, 2, 3)]
        temperature = np.asarray(u["user_out_var"][0], dtype=float)
        x = np.asarray(f["x1v"])[:, None, None, :]
        y = np.asarray(f["x2v"])[:, None, :, None]
        z = np.asarray(f["x3v"])[:, :, None, None]
        radius = np.sqrt(x*x+y*y+z*z)
        radial_velocity = (velocity[0]*x+velocity[1]*y+velocity[2]*z)
        radial_velocity /= np.maximum(radius, 1.0e-30)
        shells = {}
        for low, high in SHELLS:
            selected = (radius >= low) & (radius < high)
            density = rho[selected]
            gas_pressure = pressure[selected]
            log_rho = np.log10(density*RHO_UNIT)
            log_q = np.log10(gas_pressure*EGAS_UNIT*eos[2]
                             /(density*RHO_UNIT))
            table_values = interpolate_e_over_p(eos, log_q, log_rho)
            egas = gas_pressure*np.power(10.0, table_values)
            enthalpy = (egas+gas_pressure)/density
            mean_h = np.average(enthalpy, weights=density)
            shell_radius = radius[selected]
            shell_vr = radial_velocity[selected]
            area = 4.0*np.pi*shell_radius*shell_radius
            gas_mass_flux = area*density*shell_vr
            convective = float(np.mean(gas_mass_flux*(enthalpy-mean_h)))
            advective = float(np.mean(gas_mass_flux*mean_h))
            shells[f"{low}-{high}"] = {
                "cells": int(selected.sum()),
                "mean_radius_Rsun": float(np.mean(shell_radius)),
                "mass_weighted_temperature_K": float(np.average(
                    temperature[selected], weights=density)),
                "mass_weighted_radial_velocity_code": float(np.average(
                    shell_vr, weights=density)),
                "convective_gas_enthalpy_Lstar": convective/luminosity,
                "mean_advective_gas_enthalpy_Lstar": advective/luminosity,
                "total_gas_enthalpy_Lstar": (convective+advective)/luminosity,
            }
        return shells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eos", type=Path)
    parser.add_argument("--case", action="append",
                        help="label:run-directory")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--plot-from-json", type=Path,
                        help="regenerate the figure without reading large Athena outputs")
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    if args.plot_from_json:
        results = json.loads(args.plot_from_json.read_text())["cases"]
    else:
        if not args.eos or not args.case or not args.json:
            parser.error("--eos, --case, and --json are required for a new audit")
        eos = load_eos(args.eos)
        results = {}
        for item in args.case:
            name, path = item.split(":", 1)
            directory = Path(path)
            histories = list(directory.glob("*.hst"))
            if len(histories) != 1:
                raise ValueError(f"Expected one history in {directory}")
            history = np.atleast_2d(np.loadtxt(histories[0]))
            luminosity = float(history[0, 15])
            states = {}
            for label, (primitive, user_output, time) in zip(
                    ("initial", "final"), matching_outputs(directory)):
                states[label] = {"time_code": time, "time_day": time*TIME_DAY,
                                 "shells": shell_transport(
                                     primitive, user_output, eos, luminosity)}
            results[name] = states
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"method": __doc__.strip(),
                                         "cases": results}, indent=2)+"\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for name, states in results.items():
        for phase, style in (("initial", "--"), ("final", "-")):
            shell_data = list(states[phase]["shells"].values())
            radii = [row["mean_radius_Rsun"] for row in shell_data]
            label = f"{name}, {states[phase]['time_day']:.2f} d"
            for ax, field in zip(
                    axes, ("convective_gas_enthalpy_Lstar",
                           "mean_advective_gas_enthalpy_Lstar")):
                ax.plot(radii, [row[field] for row in shell_data],
                        linestyle=style, marker="o", label=label)
    for ax, title in zip(axes, ("Gas enthalpy covariance", "Mean gas advection")):
        ax.axhline(0.0, color="black", linewidth=0.7)
        ax.set(title=title, xlabel="Radius [Rsun]", ylabel="L / Lstar")
        ax.set_yscale("symlog", linthresh=1.0e-4)
        ax.set_ylim(-1.2, 1.2)
        ax.grid(alpha=0.25)
    axes[0].axhline(1.0, color="gray", linestyle=":", linewidth=1.0,
                    label="Lstar target")
    for ax in axes:
        ax.legend(fontsize=7)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=170)
    plt.close(fig)
    print(args.json or args.plot_from_json)
    print(args.figure)


if __name__ == "__main__":
    main()
