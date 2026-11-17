#!/usr/bin/env python3
"""Compare equal-cycle parent and aligned central-crop Athena++ HDF snapshots."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


FIELDS = ("rho_cgs", "Tgas_K", "Erad_cgs", "Trad_K", "vel1", "vel2", "vel3",
          "Bcc1", "Bcc2", "Bcc3")
BINS = ((0, 10), (10, 50), (50, 150), (150, 300), (300, 650), (650, 1300))


def load(path):
    with h5py.File(path) as h:
        variables = [x.decode() for x in h.attrs["VariableNames"]]
        indices = [variables.index(x) for x in FIELDS]
        return {
            "cycle": int(h.attrs["NumCycles"]),
            "time": float(h.attrs["Time"]),
            "blocks": int(h.attrs["NumMeshBlocks"]),
            "root_x1": h.attrs["RootGridX1"][:].astype(float),
            "root_cells": h.attrs["RootGridSize"][:].astype(int),
            "block_cells": h.attrs["MeshBlockSize"][:].astype(int),
            "level": h["Levels"][:].astype(int),
            "loc": h["LogicalLocations"][:].astype(int),
            "xv": [h[f"x{axis}v"][:].astype(float) for axis in (1, 2, 3)],
            "xf": [h[f"x{axis}f"][:].astype(float) for axis in (1, 2, 3)],
            "u": h["hydro"][:][indices].astype(float),
        }


def compare(parent, crop):
    p, c = load(parent), load(crop)
    if p["cycle"] != c["cycle"]:
        raise ValueError(f"cycles differ: {p['cycle']} vs {c['cycle']}")
    if abs(p["time"] - c["time"]) > 2e-4:
        raise ValueError(f"times differ: {p['time']} vs {c['time']}")
    # Both root grids are centered, and the crop has two root blocks per axis.
    parent_width = p["root_x1"][1] - p["root_x1"][0]
    crop_width = c["root_x1"][1] - c["root_x1"][0]
    ratio = parent_width/crop_width
    parent_root_blocks = p["root_cells"][0]//p["block_cells"][0]
    crop_root_blocks = c["root_cells"][0]//c["block_cells"][0]
    if crop_root_blocks != 2 or parent_root_blocks != 4 or not np.isclose(
            ratio, round(ratio)):
        raise ValueError("unsupported parent/crop root-grid relation")
    ratio = round(ratio)
    if ratio & (ratio-1):
        raise ValueError("parent/crop size ratio must be a power of two")
    level_shift = ratio.bit_length()-2
    start = parent_root_blocks*(1 << level_shift)//2-1
    parent_index = {(int(level), *map(int, loc)): i
                    for i, (level, loc) in enumerate(zip(p["level"], p["loc"]))}
    pairs = []
    for ci, (level, loc) in enumerate(zip(c["level"], c["loc"])):
        factor = 1 << int(level)
        key = (int(level)+level_shift,
               *(int(q)+start*factor for q in loc))
        if key in parent_index:
            pairs.append((parent_index[key], ci))
    if not pairs:
        raise ValueError("no blocks overlap exactly")
    pi, ci = (np.array(x, dtype=int) for x in zip(*pairs))
    if len({tuple(p["loc"][j]) for j in pi}) != len(pi):
        raise ValueError("duplicate parent blocks in match")
    x1, x2, x3 = (v[ci] for v in c["xv"])
    radius = np.sqrt(x1[:, None, None, :]**2
                     + x2[:, None, :, None]**2
                     + x3[:, :, None, None]**2)
    result = {"parent": str(parent), "crop": str(crop),
              "cycle": p["cycle"], "parent_time": p["time"],
              "crop_time": c["time"], "parent_blocks": p["blocks"],
              "crop_blocks": c["blocks"], "crop_half_width_au": crop_width/2,
              "matched_blocks": len(pairs),
              "matched_cells": len(pairs)*512, "bins": {}}
    dx1, dx2, dx3 = (np.diff(v[ci], axis=1) for v in c["xf"])
    volume = dx1[:, None, None, :] * dx2[:, None, :, None] * dx3[:, :, None, None]
    result["matched_volume_fraction"] = float(
        np.sum(volume) / crop_width**3)
    result["volume_integrals"] = {}
    for field in ("rho_cgs", "Erad_cgs"):
        j = FIELDS.index(field)
        a = p["u"][j, pi]
        b = c["u"][j, ci]
        total = np.sum(a*volume)
        result["volume_integrals"][field] = {
            "relative_signed_difference": float(np.sum((b-a)*volume)/total),
            "relative_l1_difference": float(np.sum(abs(b-a)*volume)/total),
        }
    result["spherical_integrals"] = {}
    for limit in (10, 50, 150, 300):
        mask = radius < limit
        if not mask.any():
            continue
        region = {"matched_cells": int(mask.sum())}
        for field in ("rho_cgs", "Erad_cgs"):
            j = FIELDS.index(field)
            a = p["u"][j, pi][mask]
            b = c["u"][j, ci][mask]
            v = volume[mask]
            total = np.sum(a*v)
            region[field] = {
                "relative_signed_difference": float(np.sum((b-a)*v)/total),
                "relative_l1_difference": float(np.sum(abs(b-a)*v)/total),
            }
        result["spherical_integrals"][f"r_lt_{limit}au"] = region
    for low, high in BINS:
        mask = (radius >= low) & (radius < high)
        if not mask.any():
            continue
        group = {"cells": int(mask.sum())}
        for j, field in enumerate(FIELDS):
            a = p["u"][j, pi][mask]
            b = c["u"][j, ci][mask]
            d = b-a
            group[field] = {
                "relative_l2": float(np.linalg.norm(d)/max(np.linalg.norm(a), 1e-300)),
                "max_absolute": float(np.max(np.abs(d))),
                "median_absolute": float(np.median(np.abs(d))),
            }
        result["bins"][f"{low}-{high}au"] = group
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("crop", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.parent, args.crop)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
