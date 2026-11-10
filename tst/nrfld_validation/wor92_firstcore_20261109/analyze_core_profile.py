#!/usr/bin/env python3
"""Extract a volume-correct radial profile from an Athena++ leaf-block HDF5 file.

The output is a diagnostic profile; a density threshold alone is not a
first-core or accretion-shock detection. Units default to the WOR-92 input.
"""

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


def eos_temperature(table_path, density, pressure):
    """Match the table EOS bilinear P→e→T lookup used by Athena++."""
    with table_path.open() as stream:
        rows = [line for line in stream if line.strip() and not line.startswith("#")]
    nvar, nx2, nr = map(int, rows[0].split())
    x2min, x2max = map(float, rows[1].split())
    rmin, rmax = map(float, rows[2].split())
    ratios = np.fromstring(rows[3], sep=" ")
    fields = np.asarray([np.fromstring(row, sep=" ") for row in rows[4:]])
    fields = fields.reshape(nvar, nx2, nr)

    def interpolate(field, x2, logrho):
        u = (x2 - x2min) * (nx2 - 1) / (x2max - x2min)
        v = (logrho - rmin) * (nr - 1) / (rmax - rmin)
        i = np.clip(np.floor(u).astype(np.int64), 0, nx2 - 2)
        j = np.clip(np.floor(v).astype(np.int64), 0, nr - 2)
        a, b = u - i, v - j
        data = fields[field]
        return ((1-a) * (1-b) * data[i, j]
                + (1-a) * b * data[i, j+1]
                + a * (1-b) * data[i+1, j]
                + a * b * data[i+1, j+1])

    logrho = np.log10(density)
    logp = np.log10(pressure)
    pcoord = logp + np.log10(ratios[1]) - logrho
    loge = logp + interpolate(1, pcoord, logrho)
    ecoord = loge + np.log10(ratios[3]) - logrho
    domain = ((logrho >= rmin) & (logrho <= rmax)
              & (pcoord >= x2min) & (pcoord <= x2max)
              & (ecoord >= x2min) & (ecoord <= x2max))
    return 10.0**interpolate(3, ecoord, logrho), int((~domain).sum())


def ideal_temperature(density, pressure, mean_molecular_weight):
    gas_constant = 8.3144621e7  # erg K^-1 mol^-1; match protostellar_core_fld.cpp
    return pressure * mean_molecular_weight / (density * gas_constant)


def analyze(source, output, rho_unit, pressure_unit, length_unit, eos_table,
            ideal_mu=None):
    with h5py.File(source) as h:
        names = [x.decode() for x in h.attrs["VariableNames"]]
        prim_names = names[: int(h.attrs["NumVariables"][0])]
        var = {name: np.asarray(h["prim"][prim_names.index(name)], dtype=np.float64)
               for name in ("rho", "press", "vel1", "vel2", "vel3", "Er")}
        x = [np.asarray(h["x%sv" % i], dtype=np.float64) for i in (1, 2, 3)]
        xf = [np.asarray(h["x%sf" % i], dtype=np.float64) for i in (1, 2, 3)]
        cycle, time, max_level = (int(h.attrs["NumCycles"]),
                                  float(h.attrs["Time"]), int(h.attrs["MaxLevel"]))

    shape = var["rho"].shape
    z = np.broadcast_to(x[2][:, :, None, None], shape)
    y = np.broadcast_to(x[1][:, None, :, None], shape)
    xx = np.broadcast_to(x[0][:, None, None, :], shape)
    radius = np.sqrt(xx * xx + y * y + z * z).ravel()
    volume = np.broadcast_to(
        np.diff(xf[2], axis=1)[:, :, None, None]
        * np.diff(xf[1], axis=1)[:, None, :, None]
        * np.diff(xf[0], axis=1)[:, None, None, :], shape).ravel()
    density = var["rho"].ravel() * rho_unit
    pressure = var["press"].ravel() * pressure_unit
    erad = var["Er"].ravel() * pressure_unit
    if eos_table is not None:
        temperature, eos_outside = eos_temperature(eos_table, density, pressure)
    elif ideal_mu is not None:
        temperature = ideal_temperature(density, pressure, ideal_mu)
        eos_outside = None
    else:
        temperature, eos_outside = None, None
    velocity_unit = np.sqrt(pressure_unit / rho_unit)
    vr = ((var["vel1"].ravel() * xx.ravel()
           + var["vel2"].ravel() * y.ravel()
           + var["vel3"].ravel() * z.ravel())
          / np.maximum(radius, 1e-100) * velocity_unit / 1e5)

    # Include a finite central bin, then logarithmic shells out to 10,000 AU.
    edges = np.r_[0.0, np.geomspace(0.5, 10000.0, 61)]
    shell = np.searchsorted(edges, radius, side="right") - 1
    valid = (shell >= 0) & (shell < len(edges) - 1)
    shell = shell[valid]
    dv = volume[valid]
    mass = density[valid] * dv * length_unit**3
    n = len(edges) - 1
    count = np.bincount(shell, minlength=n)
    shell_volume = np.bincount(shell, weights=dv, minlength=n)
    shell_mass = np.bincount(shell, weights=mass, minlength=n)
    shell_press = np.bincount(shell, weights=pressure[valid] * dv, minlength=n)
    shell_erad = np.bincount(shell, weights=erad[valid] * dv, minlength=n)
    shell_vr = np.bincount(shell, weights=vr[valid] * mass, minlength=n)
    shell_temperature = (np.bincount(shell, weights=temperature[valid] * mass,
                                     minlength=n) if temperature is not None else None)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("r_inner_au", "r_outer_au", "cells", "mean_rho_g_cm3",
                         "mean_pressure_dyn_cm2", "mean_erad_erg_cm3",
                         "mass_weighted_vrad_km_s", "mass_weighted_temp_K",
                         "shell_mass_g", "enclosed_mass_g"))
        enclosed = 0.0
        for i in range(n):
            enclosed += shell_mass[i]
            writer.writerow((edges[i], edges[i+1], count[i],
                             shell_mass[i] / (shell_volume[i] * length_unit**3)
                             if shell_volume[i] else "",
                             shell_press[i] / shell_volume[i] if shell_volume[i] else "",
                             shell_erad[i] / shell_volume[i] if shell_volume[i] else "",
                             shell_vr[i] / shell_mass[i] if shell_mass[i] else "",
                             shell_temperature[i] / shell_mass[i]
                             if shell_temperature is not None and shell_mass[i] else "",
                             shell_mass[i], enclosed))

    imax = int(np.argmax(density))
    octant = ((xx.ravel() >= 0).astype(np.int8)
              + 2 * (y.ravel() >= 0).astype(np.int8)
              + 4 * (z.ravel() >= 0).astype(np.int8))
    octant_spread = {}
    for low, high in ((0, 5), (5, 10), (10, 20)):
        in_shell = (radius >= low) & (radius < high)
        v = np.bincount(octant[in_shell], weights=volume[in_shell], minlength=8)
        rv = np.bincount(octant[in_shell],
                         weights=density[in_shell] * volume[in_shell], minlength=8)
        means = np.divide(rv, v, out=np.full(8, np.nan), where=v > 0)
        octant_spread["%s_to_%s_au" % (low, high)] = (
            float((np.nanmax(means) - np.nanmin(means)) / np.nanmean(means)))
    result = {
        "source": str(source), "cycle": cycle, "time_code": time,
        "time_years": time * length_unit / velocity_unit / (365.25 * 86400),
        "max_level": max_level, "meshblocks": shape[0],
        "rho_max_g_cm3": float(density[imax]),
        "rho_max_radius_au": float(radius[imax]),
        "rho_max_velocity_radial_km_s": float(vr[imax]),
        "temperature_at_rho_max_K": (float(temperature[imax])
                                     if temperature is not None else None),
        "temperature_max_K": (float(np.max(temperature))
                              if temperature is not None else None),
        "eos_cells_outside_table": eos_outside,
        "gas_mass_g": float(np.sum(density * volume) * length_unit**3),
        "octant_density_relative_spread": octant_spread,
        "profile_csv": str(output),
    }
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hdf5", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--rho-unit", type=float, default=1e-18)
    parser.add_argument("--pressure-unit", type=float, default=1e-9)
    parser.add_argument("--length-unit", type=float, default=1.495978707e13)
    parser.add_argument("--eos-table", type=Path)
    parser.add_argument("--ideal-mu", type=float)
    args = parser.parse_args()
    if args.eos_table is not None and args.ideal_mu is not None:
        parser.error("Choose one temperature model: --eos-table or --ideal-mu")
    print(json.dumps(analyze(args.hdf5, args.output_csv, args.rho_unit,
                             args.pressure_unit, args.length_unit,
                             args.eos_table, args.ideal_mu), indent=2))


if __name__ == "__main__":
    main()
