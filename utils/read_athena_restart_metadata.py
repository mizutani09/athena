#!/usr/bin/env python3
"""Read the portable-to-the-build metadata prefix of an Athena++ restart.

The restart body is intentionally not loaded.  This helper is for selecting an
absolute ``time/nlim`` from an existing restart without guessing its cycle.
Athena++ restart files are build-ABI files; the reader therefore defaults to
the native little-endian, double-precision layout used by the NR-FLD runs.
"""
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import struct


class RegionSizeDouble(ctypes.Structure):
    _fields_ = [
        ("x1min", ctypes.c_double), ("x2min", ctypes.c_double),
        ("x3min", ctypes.c_double), ("x1max", ctypes.c_double),
        ("x2max", ctypes.c_double), ("x3max", ctypes.c_double),
        ("x1rat", ctypes.c_double), ("x2rat", ctypes.c_double),
        ("x3rat", ctypes.c_double), ("nx1", ctypes.c_int),
        ("nx2", ctypes.c_int), ("nx3", ctypes.c_int),
    ]


def read_metadata(path: Path) -> dict:
    with path.open("rb") as stream:
        prefix = stream.read(65536)
    marker = b"<par_end>"
    position = prefix.find(marker)
    if position < 0:
        raise ValueError("<par_end> was not found in the first 64 KiB")
    newline = prefix.find(b"\n", position + len(marker))
    if newline < 0:
        raise ValueError("unterminated <par_end> marker")
    offset = newline + 1

    # restart.cpp writes two ints, one native RegionSize POD, two Reals, one
    # int and one IOWrapperSizeT.  The target NR-FLD builds use double Real and
    # a 64-bit size type.  ctypes preserves the native tail padding of the POD.
    required = offset + 2*4 + ctypes.sizeof(RegionSizeDouble) + 2*8 + 4 + 8
    if len(prefix) < required:
        raise ValueError("restart metadata prefix is truncated")
    nbtotal, root_level = struct.unpack_from("=ii", prefix, offset)
    offset += 8
    region = RegionSizeDouble.from_buffer_copy(
        prefix[offset:offset + ctypes.sizeof(RegionSizeDouble)])
    offset += ctypes.sizeof(RegionSizeDouble)
    time, dt = struct.unpack_from("=dd", prefix, offset)
    offset += 16
    cycle = struct.unpack_from("=i", prefix, offset)[0]
    offset += 4
    data_size = struct.unpack_from("=Q", prefix, offset)[0]
    return {
        "path": str(path.resolve()),
        "file_size": path.stat().st_size,
        "parameter_bytes": newline + 1,
        "meshblocks": nbtotal,
        "root_level": root_level,
        "root_grid": [region.nx1, region.nx2, region.nx3],
        "time": time,
        "dt": dt,
        "cycle": cycle,
        "meshblock_data_bytes": data_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("restart", type=Path)
    parser.add_argument("--field", choices=["cycle", "time", "dt", "meshblocks"],
                        help="Print one value for shell command substitution")
    args = parser.parse_args()
    metadata = read_metadata(args.restart)
    if args.field:
        print(metadata[args.field])
    else:
        print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
