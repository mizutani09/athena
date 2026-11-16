#!/usr/bin/env python3
"""Check cellwise total-energy closure in a table-EOS MHD HDF snapshot."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def load_eos(table):
    with table.open() as stream:
        rows = [line for line in stream if line.strip() and not line.startswith("#")]
    nvar, nx2, nrho = map(int, rows[0].split())
    x2min, x2max = map(float, rows[1].split())
    rhomin, rhomax = map(float, rows[2].split())
    ratios = np.fromstring(rows[3], sep=" ")
    data = np.asarray([np.fromstring(row, sep=" ") for row in rows[4:]])
    return (data.reshape(nvar, nx2, nrho), ratios, x2min, x2max,
            rhomin, rhomax, nx2, nrho)


def egas_from_rho_p(table, density, pressure):
    fields, ratios, x2min, x2max, rhomin, rhomax, nx2, nrho = load_eos(table)
    logrho = np.log10(density)
    logp = np.log10(pressure)
    x2 = logp + np.log10(ratios[1]) - logrho
    u = (x2 - x2min) * (nx2 - 1) / (x2max - x2min)
    v = (logrho - rhomin) * (nrho - 1) / (rhomax - rhomin)
    i = np.clip(np.floor(u).astype(np.int64), 0, nx2 - 2)
    j = np.clip(np.floor(v).astype(np.int64), 0, nrho - 2)
    a, b = u - i, v - j
    f = fields[1]
    value = ((1-a)*(1-b)*f[i,j] + (1-a)*b*f[i,j+1]
             + a*(1-b)*f[i+1,j] + a*b*f[i+1,j+1])
    domain = ((logrho >= rhomin) & (logrho <= rhomax)
              & (x2 >= x2min) & (x2 <= x2max))
    return pressure * 10.0**value, int((~domain).sum())


def audit(hdf, eos):
    with h5py.File(hdf) as h:
        names = [v.decode() for v in h.attrs["VariableNames"]]
        a = h["hydro"][:].astype(np.float64)
        cycle = int(h.attrs["NumCycles"])
    v = {name: a[names.index(name)] for name in
         ("dens", "rho", "Etot", "press", "mom1", "mom2", "mom3",
          "Bcc1", "Bcc2", "Bcc3")}
    kinetic = (v["mom1"]**2 + v["mom2"]**2 + v["mom3"]**2) / (2*v["dens"])
    magnetic = (v["Bcc1"]**2 + v["Bcc2"]**2 + v["Bcc3"]**2) / 2
    gas_cons = v["Etot"] - kinetic - magnetic
    gas_eos, outside = egas_from_rho_p(eos, v["rho"]*1e-18, v["press"]*1e-9)
    gas_eos /= 1e-9
    relative = np.abs(gas_cons - gas_eos) / gas_eos
    return {
        "cycle": cycle, "cells": int(relative.size), "eos_outside_cells": outside,
        "negative_gas_energy_cells": int(np.sum(gas_cons <= 0)),
        "magnetic_energy_code_max": float(magnetic.max()),
        "magnetic_over_gas_energy_max": float(np.max(magnetic/gas_cons)),
        "closure_median_relative": float(np.median(relative)),
        "closure_p99_relative": float(np.percentile(relative, 99)),
        "closure_max_relative": float(relative.max()),
        "closure_without_magnetic_median_relative": float(np.median(
            np.abs(v["Etot"]-kinetic-gas_eos)/gas_eos)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hdf", type=Path)
    parser.add_argument("eos", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.hdf, args.eos), indent=2))


if __name__ == "__main__":
    main()
