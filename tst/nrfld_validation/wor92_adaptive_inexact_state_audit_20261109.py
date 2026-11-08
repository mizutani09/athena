#!/usr/bin/env python3
"""Compare double-precision physical state in two WOR-92 XD restart files.

This layout is specific to the recorded MHD, one-scalar, NGHOST=2, 8^3
MeshBlock binary.  It checks the expected bytes before interpreting arrays.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "wor92_xd_20261108"))
from read_athena_restart_metadata_py36 import read_metadata  # noqa: E402


NBLOCK = 960
NCELL = 12**3
BLOCK_BYTES = 196992
ID_BYTES = 40
FIELDS = {"rho": 0, "momentum_x1": NCELL, "momentum_x2": 2*NCELL,
          "momentum_x3": 3*NCELL, "hydro_total_energy": 4*NCELL,
          "radiation_energy": 5*NCELL + 3*12*12*13}
ID_DTYPE = np.dtype([("lx1", "<i8"), ("lx2", "<i8"), ("lx3", "<i8"),
                     ("level", "<i4"), ("pad", "<i4"), ("cost", "<f8")])


def load(path):
    meta = read_metadata(path)
    if meta["meshblocks"] != NBLOCK or meta["meshblock_data_bytes"] != BLOCK_BYTES:
        raise ValueError("unexpected restart layout: {}".format(meta))
    data_offset = meta["file_size"] - NBLOCK*BLOCK_BYTES
    ids = np.memmap(path, dtype=ID_DTYPE, mode="r",
                    offset=data_offset - NBLOCK*ID_BYTES, shape=(NBLOCK,))
    data = np.memmap(path, dtype="<f8", mode="r", offset=data_offset,
                     shape=(NBLOCK, BLOCK_BYTES//8))
    return meta, ids, data


def active(data, offset):
    return data[:, offset:offset+NCELL].reshape(NBLOCK, 12, 12, 12)[
        :, 2:10, 2:10, 2:10]


def metrics(left, right, volume):
    difference = right-left
    denominator = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1e-300)
    integral_left = np.sum(left*volume[:, None, None, None], dtype=np.longdouble)
    integral_right = np.sum(right*volume[:, None, None, None], dtype=np.longdouble)
    return {"max_relative": float(np.max(np.abs(difference)/denominator)),
            "rms_relative": float(np.sqrt(np.mean(difference*difference)) /
                max(np.sqrt(np.mean(left*left)), 1e-300)),
            "integral_relative": float((integral_right-integral_left) /
                max(abs(integral_left), 1e-300)),
            "changed_cells": int(np.count_nonzero(difference))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--matched-time", action="store_true",
                        help="Allow different cycle counts at the same final time")
    args = parser.parse_args()
    lm, li, ld = load(args.left)
    rm, ri, rd = load(args.right)
    if (lm["cycle"] != rm["cycle"] and not args.matched_time) or abs(
            lm["time"]-rm["time"]) > (
            1e-8*max(abs(lm["time"]), abs(rm["time"]), 1.0)):
        raise ValueError("restart cycle/time differ beyond comparison window")
    if not np.array_equal(li[["lx1", "lx2", "lx3", "level"]],
                          ri[["lx1", "lx2", "lx3", "level"]]):
        raise ValueError("MeshBlock topology differs")
    root_dx = 12000.0/32.0
    volume = (root_dx/np.power(2.0, li["level"]-lm["root_level"]))**3
    result = {"left": str(args.left), "right": str(args.right),
              "left_cycle": lm["cycle"], "right_cycle": rm["cycle"],
              "left_time": lm["time"],
              "right_time": rm["time"],
              "fields": {name: metrics(active(ld, offset), active(rd, offset), volume)
                         for name, offset in FIELDS.items()}}
    energy_left = active(ld, FIELDS["hydro_total_energy"]) + active(
        ld, FIELDS["radiation_energy"])
    energy_right = active(rd, FIELDS["hydro_total_energy"]) + active(
        rd, FIELDS["radiation_energy"])
    result["combined_energy"] = metrics(energy_left, energy_right, volume)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
