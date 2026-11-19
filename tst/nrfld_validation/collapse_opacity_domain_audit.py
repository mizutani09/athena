#!/usr/bin/env python3
"""Check that an accepted Athena++ HDF state lies inside its opacity table."""

import argparse
import array
import json
import math
import os
import re
import subprocess
import sys
import tempfile


def audit_h5py(snapshot, table):
    import h5py
    import numpy as np

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


def h5dump_text(arguments):
    return subprocess.check_output(
        ["h5dump"] + arguments, universal_newlines=True)


def h5dump_header(path, dataset):
    text = h5dump_text(["-H", "-d", dataset, path])
    dimensions = re.search(
        r"DATASPACE\s+SIMPLE\s*\{\s*\(\s*([^)]*)\)", text)
    if dimensions is None:
        raise RuntimeError("cannot parse dimensions for {}".format(dataset))
    dims = tuple(int(value.strip())
                 for value in dimensions.group(1).split(","))
    if "H5T_IEEE_F32" in text:
        typecode = "f"
    elif "H5T_IEEE_F64" in text:
        typecode = "d"
    else:
        raise RuntimeError("unsupported datatype for {}".format(dataset))
    return dims, typecode


def h5dump_values(path, dataset, start=None, count=None):
    dims, typecode = h5dump_header(path, dataset)
    handle, raw_path = tempfile.mkstemp(prefix="opacity-audit-", suffix=".bin")
    os.close(handle)
    try:
        command = ["h5dump", "-d", dataset]
        if start is not None:
            command += ["-s", ",".join(str(value) for value in start)]
        if count is not None:
            command += ["-c", ",".join(str(value) for value in count)]
        command += ["-o", raw_path, "-b", "LE", path]
        with open(os.devnull, "w") as sink:
            subprocess.check_call(command, stdout=sink)
        values = array.array(typecode)
        with open(raw_path, "rb") as stream:
            values.fromfile(stream, os.path.getsize(raw_path) // values.itemsize)
        if sys.byteorder != "little":
            values.byteswap()
    finally:
        os.unlink(raw_path)
    expected = 1
    for dimension in (count if count is not None else dims):
        expected *= dimension
    if len(values) != expected:
        raise RuntimeError("unexpected data length for {}: {} != {}".format(
            dataset, len(values), expected))
    return dims, values


def audit_h5dump(snapshot, table):
    variable_text = h5dump_text(["-a", "VariableNames", snapshot])
    variable_data = re.search(r"DATA\s*\{(.*?)\}", variable_text, re.DOTALL)
    if variable_data is None:
        raise RuntimeError("cannot parse VariableNames")
    names = re.findall(r'"([^"\n]+)"', variable_data.group(1))
    for required in ("rho_cgs", "Tgas_K"):
        if required not in names:
            raise RuntimeError("{} is absent from VariableNames".format(required))
    dims, _ = h5dump_header(snapshot, "/hydro")
    if len(dims) != 5 or dims[0] != len(names):
        raise RuntimeError("unexpected /hydro dimensions")
    count = (1,) + dims[1:]
    _, rho = h5dump_values(
        snapshot, "/hydro", (names.index("rho_cgs"), 0, 0, 0, 0), count)
    _, temp = h5dump_values(
        snapshot, "/hydro", (names.index("Tgas_K"), 0, 0, 0, 0), count)
    _, log_rho_axis = h5dump_values(table, "/axes/log10_density")
    _, log_temp_axis = h5dump_values(table, "/axes/log10_temperature")
    rho_lower, rho_upper = 10.0**min(log_rho_axis), 10.0**max(log_rho_axis)
    temp_lower, temp_upper = 10.0**min(log_temp_axis), 10.0**max(log_temp_axis)
    invalid = 0
    outside = 0
    for density, temperature in zip(rho, temp):
        valid = (math.isfinite(density) and math.isfinite(temperature)
                 and density > 0.0 and temperature > 0.0)
        if not valid:
            invalid += 1
            outside += 1
        elif not (rho_lower <= density <= rho_upper
                  and temp_lower <= temperature <= temp_upper):
            outside += 1
    cycle_text = h5dump_text(["-a", "NumCycles", snapshot])
    cycle_match = re.search(r"\(\d+\):\s*(-?\d+)", cycle_text)
    if cycle_match is None:
        raise RuntimeError("cannot parse NumCycles")
    return {
        "cycle": int(cycle_match.group(1)),
        "cells": len(rho),
        "nonfinite_or_nonpositive_cells": invalid,
        "outside_opacity_domain_cells": outside,
        "rho_min_g_cm3": min(rho),
        "rho_max_g_cm3": max(rho),
        "temperature_min_K": min(temp),
        "temperature_max_K": max(temp),
        "table_rho_min_g_cm3": rho_lower,
        "table_rho_max_g_cm3": rho_upper,
        "table_temperature_min_K": temp_lower,
        "table_temperature_max_K": temp_upper,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot")
    parser.add_argument("opacity_table")
    parser.add_argument("--backend", choices=("auto", "h5py", "h5dump"),
                        default="auto")
    args = parser.parse_args()
    if args.backend == "h5dump":
        result = audit_h5dump(args.snapshot, args.opacity_table)
    elif args.backend == "h5py":
        result = audit_h5py(args.snapshot, args.opacity_table)
    else:
        try:
            result = audit_h5py(args.snapshot, args.opacity_table)
        except ImportError:
            result = audit_h5dump(args.snapshot, args.opacity_table)
    print(json.dumps(result, sort_keys=True))
    if (result["nonfinite_or_nonpositive_cells"] != 0
            or result["outside_opacity_domain_cells"] != 0):
        sys.exit(1)


if __name__ == "__main__":
    main()
