#!/usr/bin/env python3
"""Attribute c930 hydro face imbalance to ambient-tagged versus core faces."""

import argparse
from collections import defaultdict
from decimal import Decimal
import json
from pathlib import Path

import h5py
import numpy as np

from audit_high_density_faces import AXES, fields, groups


def audit(log_dir, hdf):
    with h5py.File(hdf) as source:
        if int(source.attrs["NumCycles"]) != 930:
            raise ValueError("expected c930 HDF")
        tracer = np.asarray(source["prim"][7], dtype=np.float64)
        positions = [np.asarray(source[f"x{i}f"], dtype=np.float64)
                     for i in (1, 2, 3)]
    blocks = defaultdict(list)
    for path in sorted(log_dir.glob("rank-*.log")):
        for line in path.open():
            if line.startswith("[WOR92_HYDRO_MASS_FLUX] "):
                raw = fields(line)
                item = {key: (int(value) if key in ("rank", "cycle", "gid", "level")
                              else Decimal(value)) for key, value in raw.items()}
                blocks[item["wght"]].append(item)
    if len(blocks) != 2 or any(len(items) != 3536 for items in blocks.values()):
        raise ValueError("incomplete 2-stage ledger")
    out = {}
    for weight, items in sorted(blocks.items()):
        totals = defaultdict(lambda: {"count": 0, "net": Decimal(0),
                                  "abs_net": Decimal(0), "scale": Decimal(0)})
        for axis, names in enumerate(AXES, 1):
            lo, hi, u0, u1, v0, v1, minus, plus = names
            plane_faces = defaultdict(list)
            for block in items:
                gid = block["gid"]
                if not np.isclose(float(block[lo]), positions[axis-1][gid, 0]) or not np.isclose(
                        float(block[hi]), positions[axis-1][gid, -1]):
                    raise ValueError(f"gid {gid} does not match HDF face")
                rect = ((block[u0], block[u1]), (block[v0], block[v1]))
                for side, plane, delta in (("minus", block[lo], block[minus]),
                                           ("plus", block[hi], block[plus])):
                    face_index = 0 if side == "minus" else -1
                    adjacent = np.take(tracer[gid], face_index, axis=3-axis)
                    plane_faces[plane].append({"side": side, "rect": rect,
                                               "level": block["level"],
                                               "delta": delta,
                                               "ambient_fraction": float(np.mean(adjacent >= 0.5))})
            for faces in plane_faces.values():
                for component in groups(faces):
                    left = [f for f in component if f["side"] == "plus"]
                    right = [f for f in component if f["side"] == "minus"]
                    kind = "boundary" if not left or not right else (
                        "coarse_fine" if {f["level"] for f in left} !=
                        {f["level"] for f in right} else "same")
                    values = [f["ambient_fraction"] for f in component]
                    region = ("ambient" if all(value == 1 for value in values) else
                              "core" if all(value == 0 for value in values) else "mixed")
                    left_sum = sum((f["delta"] for f in left), Decimal(0))
                    right_sum = sum((f["delta"] for f in right), Decimal(0))
                    net = left_sum + right_sum
                    entry = totals[f"x{axis}:{kind}:{region}"]
                    entry["count"] += 1
                    entry["net"] += net
                    entry["abs_net"] += abs(net)
                    entry["scale"] += abs(left_sum) + abs(right_sum)
        out[str(weight)] = {key: {"count": val["count"],
                                  "net_code": float(val["net"]),
                                  "abs_net_code": float(val["abs_net"]),
                                  "weighted_relative_mismatch": float(val["abs_net"] / val["scale"])
                                  if val["scale"] else None}
                            for key, val in sorted(totals.items())}
    return {"log_dir": str(log_dir), "hdf": str(hdf), "stages": out,
            "region_definition": "all adjacent face cells have r0>=0.5: ambient; all below: core; else mixed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rank_log_dir", type=Path)
    parser.add_argument("c930_hdf", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    result = audit(args.rank_log_dir, args.c930_hdf)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output_json)


if __name__ == "__main__":
    main()
