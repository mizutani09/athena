#!/usr/bin/env python3
"""Compare paired NR-FLD cache ON/OFF runs, including MG operator fingerprints."""

import argparse
import json
from pathlib import Path
import re

import h5py
import numpy as np


def rows(path, prefix):
    return [line for line in path.read_text(errors="replace").splitlines()
            if line.startswith(prefix)]


def fields(case):
    return {path.name.split(".", 1)[1]: path
            for path in (case / "output").glob("*.athdf")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="directory containing cache_true/false")
    parser.add_argument("--require-drift", action="store_true")
    parser.add_argument("--require-topology-change", action="store_true")
    parser.add_argument("--require-schur-change", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    paths = [root / f"cache_{value}" for value in ("true", "false")]
    logs = [path / "run.log" for path in paths]
    prefix = "[NRFLD coefficient fingerprint]"
    fingerprints = [rows(log, prefix) for log in logs]
    diagnostics = [rows(log, "[NRFLD coefficients]") for log in logs]
    octets = [rows(log, "[NRFLD octet fingerprint]") for log in logs]
    newton = [rows(log, "[NR] status=") for log in logs]
    krylov = [rows(log, "[LINEAR_FGMRES]") for log in logs]
    errors = []
    for log in logs:
        text = log.read_text(errors="replace")
        if "FATAL ERROR" in text or "Terminating on cycle limit" not in text:
            errors.append(f"run did not finish cleanly: {log}")
    for name, pair in (("fingerprints", fingerprints), ("coefficient diagnostics", diagnostics),
                       ("Newton summaries", newton), ("FGMRES summaries", krylov)):
        if not pair[0] or pair[0] != pair[1]:
            errors.append(f"{name} differ or are missing")
    if octets[0] != octets[1]:
        errors.append("AMR octet fingerprints differ")
    if len(fingerprints[0]) < 2:
        errors.append("fewer than two linear-solve fingerprints")
    row_errors = [float(value) for line in diagnostics[0]
                  for value in re.findall(r"row_error_max=([^\s]+)", line)]
    if not row_errors or not np.all(np.isfinite(row_errors)) or max(row_errors) > 1e-12:
        errors.append("A applied to the constant vector disagrees with coefficient row sums")
    if args.require_drift:
        drift = [float(x) for line in diagnostics[0]
                 for x in re.findall(r"drift_fraction_max=([^\s]+)", line)]
        if not drift or max(drift) <= 0.0:
            errors.append("implicit mixed drift was not exercised")
    if args.require_schur_change:
        finest = [line for line in diagnostics[0]
                  if "kind=meshblock" in line and "solve=" in line]
        values = []
        for line in finest:
            match = re.search(r"DCCS=\[([^,]+),([^]]+)\]", line)
            if match:
                values.append(tuple(map(float, match.groups())))
        if len(values) < 2 or len(set(values)) < 2:
            errors.append("DCCS did not change across Newton solves")
    left, right = fields(paths[0]), fields(paths[1])
    if not left or left.keys() != right.keys():
        errors.append("HDF snapshot names/counts differ")
    topology = []
    compared = 0
    for name in sorted(left.keys() & right.keys()):
        with h5py.File(left[name]) as a, h5py.File(right[name]) as b:
            if a.keys() != b.keys():
                errors.append(f"dataset names differ: {name}")
                continue
            for key in a.keys():
                if not np.array_equal(a[key][:], b[key][:]):
                    errors.append(f"dataset differs: {name}/{key}")
            if "Levels" in a:
                topology.append({"cycle": int(a.attrs["NumCycles"]),
                                 "levels": sorted(map(int, set(a["Levels"][:])))})
            compared += 1
    if args.require_topology_change and len({tuple(x["levels"]) for x in topology}) < 2:
        errors.append("AMR topology did not change")
    if args.require_topology_change and not octets[0]:
        errors.append("AMR octet operator was not fingerprinted")
    octet_row_errors = [float(value) for line in octets[0]
                        for value in re.findall(r"row_error_max=([^\s]+)", line)]
    if octet_row_errors and (not np.all(np.isfinite(octet_row_errors))
                             or max(octet_row_errors) > 1e-12):
        errors.append("AMR octet constant-vector row sum disagrees")
    result = {"status": "passed" if not errors else "failed",
              "fingerprint_rows": len(fingerprints[0]),
              "octet_fingerprint_rows": len(octets[0]),
              "max_constant_vector_row_error": max(row_errors, default=None),
              "max_octet_row_error": max(octet_row_errors, default=None),
              "newton_rows": len(newton[0]), "fgmres_rows": len(krylov[0]),
              "hdf_snapshots": compared, "topology": topology, "errors": errors}
    (root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
