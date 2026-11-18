#!/usr/bin/env python3
"""Check that an accepted Athena++ HDF state lies inside its opacity table."""

import argparse
import json
import sys

import h5py
import numpy as np


def audit(snapshot, table):
    with h5py.File(snapshot, "r") as state, h5py.File(table, "r") as opacity:
        names = [name.decode() if isinstance(name, bytes) else str(name)
                 for name in state.attrs["VariableNames"]]
        rho = np.asarray(state["hydro"][names.index("rho_cgs")], dtype=np.float64)
        temp = np.asarray(state["hydro"][names.index("Tgas_K")], dtype=np.float64)
        log_rho_axis = np.asarray(opacity["axes/log10_density"], dtype=np.float64)
        log_temp_axis = np.asarray(opacity["axes/log10_temperature"], dtype=np.float64)

        finite_positive = (np.isfinite(rho) & np.isfinite(temp)
                           & (rho > 0.0) & (temp > 0.0))
        inside = np.zeros(rho.shape, dtype=bool)
        inside[finite_positive] = (
            (np.log10(rho[finite_positive]) >= log_rho_axis.min())
            & (np.log10(rho[finite_positive]) <= log_rho_axis.max())
            & (np.log10(temp[finite_positive]) >= log_temp_axis.min())
            & (np.log10(temp[finite_positive]) <= log_temp_axis.max()))
        result = {
            "cycle": int(state.attrs["NumCycles"]),
            "cells": int(rho.size),
            "nonfinite_or_nonpositive_cells": int(np.count_nonzero(~finite_positive)),
            "outside_opacity_domain_cells": int(np.count_nonzero(~inside)),
            "rho_min_g_cm3": float(np.nanmin(rho)),
            "rho_max_g_cm3": float(np.nanmax(rho)),
            "temperature_min_K": float(np.nanmin(temp)),
            "temperature_max_K": float(np.nanmax(temp)),
            "table_rho_min_g_cm3": float(10.0**log_rho_axis.min()),
            "table_rho_max_g_cm3": float(10.0**log_rho_axis.max()),
            "table_temperature_min_K": float(10.0**log_temp_axis.min()),
            "table_temperature_max_K": float(10.0**log_temp_axis.max()),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot")
    parser.add_argument("opacity_table")
    args = parser.parse_args()
    result = audit(args.snapshot, args.opacity_table)
    print(json.dumps(result, sort_keys=True))
    if (result["nonfinite_or_nonpositive_cells"] != 0
            or result["outside_opacity_domain_cells"] != 0):
        sys.exit(1)


if __name__ == "__main__":
    main()
