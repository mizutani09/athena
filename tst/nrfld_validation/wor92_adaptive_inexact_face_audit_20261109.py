#!/usr/bin/env python3
"""Audit complete rank-isolated XD WOR-92 energy and 3D interface ledgers."""

import argparse
from collections import defaultdict
from decimal import Decimal
import json
from pathlib import Path


def fields(line):
    return dict(part.split("=", 1) for part in line.split() if "=" in part)


def overlap(a, b):
    return min(a[1], b[1]) > max(a[0], b[0])


def area(face):
    return (face["rect"][0][1] - face["rect"][0][0]) * (
        face["rect"][1][1] - face["rect"][1][0])


def find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def groups(faces):
    parent = list(range(len(faces)))
    for i, first in enumerate(faces):
        for j in range(i + 1, len(faces)):
            second = faces[j]
            if first["side"] == second["side"]:
                continue
            if overlap(first["rect"][0], second["rect"][0]) and overlap(
                    first["rect"][1], second["rect"][1]):
                parent[find(parent, i)] = find(parent, j)
    out = defaultdict(list)
    for i, face in enumerate(faces):
        out[find(parent, i)].append(face)
    return out.values()


AXES = (
    ("x1min", "x1max", "x2min", "x2max", "x3min", "x3max",
     "xminus_delta", "xplus_delta"),
    ("x2min", "x2max", "x1min", "x1max", "x3min", "x3max",
     "yminus_delta", "yplus_delta"),
    ("x3min", "x3max", "x1min", "x1max", "x2min", "x2max",
     "zminus_delta", "zplus_delta"),
)


def audit(job_dir):
    paths = sorted(job_dir.glob("rank-*.log"))
    if len(paths) != 64:
        raise ValueError("expected 64 direct rank logs, found {}".format(len(paths)))
    energy, blocks = {}, []
    for path in paths:
        for line in path.read_text().splitlines():
            if line.startswith("[NR_ENERGY_LEDGER]"):
                d = fields(line)
                rank = int(d["rank"])
                if rank in energy:
                    raise ValueError("duplicate energy rank {}".format(rank))
                energy[rank] = d
            elif line.startswith("[NR_INTERFACE_LEDGER]"):
                d = fields(line)
                blocks.append({key: int(value) if key in ("rank", "cycle", "stage",
                    "gid", "level") else Decimal(value) for key, value in d.items()})
    if set(energy) != set(range(64)) or len(blocks) != 960:
        raise ValueError("incomplete ranks or blocks: {} and {}".format(
            len(energy), len(blocks)))
    if len(set(b["gid"] for b in blocks)) != 960:
        raise ValueError("duplicate MeshBlock gid")
    stages = {(b["cycle"], b["stage"]) for b in blocks}
    if len(stages) != 1:
        raise ValueError("expected one implicit stage, found {}".format(stages))

    by_kind = defaultdict(lambda: {"count": 0, "numerator": Decimal(0),
                               "denominator": Decimal(0), "max": Decimal(0),
                               "net": Decimal(0)})
    internal_sum = Decimal(0)
    for axis, names in enumerate(AXES, 1):
        lo, hi, u0, u1, v0, v1, minus, plus = names
        planes = defaultdict(list)
        for block in blocks:
            rect = ((block[u0], block[u1]), (block[v0], block[v1]))
            planes[block[lo]].append(dict(side="minus", rect=rect,
                delta=block[minus], level=block["level"]))
            planes[block[hi]].append(dict(side="plus", rect=rect,
                delta=block[plus], level=block["level"]))
        for faces in planes.values():
            for component in groups(faces):
                left = [f for f in component if f["side"] == "plus"]
                right = [f for f in component if f["side"] == "minus"]
                internal = bool(left and right)
                if internal and sum((area(f) for f in left), Decimal(0)) != sum(
                        (area(f) for f in right), Decimal(0)):
                    raise ValueError("unmatched internal face area")
                value_left = sum((f["delta"] for f in left), Decimal(0))
                value_right = sum((f["delta"] for f in right), Decimal(0))
                balance = value_left + value_right
                scale = abs(value_left) + abs(value_right)
                if internal:
                    internal_sum += balance
                kind = "boundary" if not internal else (
                    "coarse_fine" if set(f["level"] for f in left) != set(
                        f["level"] for f in right) else "same")
                item = by_kind[(axis, kind)]
                item["count"] += 1
                item["numerator"] += abs(balance)
                item["denominator"] += scale
                item["max"] = max(item["max"], abs(balance)/scale if scale else 0)
                item["net"] += balance

    sums = {key: sum((Decimal(d[key]) for d in energy.values()), Decimal(0))
            for key in ("gas_delta", "radiation_delta", "diffusion_delta",
                        "marshak_delta", "interior_delta", "balance")}
    if abs(internal_sum - sums["interior_delta"]) > Decimal("1e-5"):
        raise ValueError("face/energy ledger sum mismatch: {} vs {}".format(
            internal_sum, sums["interior_delta"]))
    absolute_scale = sum((abs(Decimal(d["balance"])) /
        Decimal(d["relative_balance"]) for d in energy.values()
        if Decimal(d["relative_balance"]) > 0), Decimal(0))
    summary = {"job": job_dir.name, "cycle_stage": sorted(stages)[0],
        "ranks": len(energy), "blocks": len(blocks),
        "level_counts": {str(level): sum(b["level"] == level for b in blocks)
            for level in sorted(set(b["level"] for b in blocks))},
        "global_signed_balance_fraction": float(abs(sums["balance"]) / absolute_scale),
        "rank_abs_balance_fraction": float(sum(
            (abs(Decimal(d["balance"])) for d in energy.values()), Decimal(0)) /
            absolute_scale),
        "rank_max_relative_balance": float(max(
            Decimal(d["relative_balance"]) for d in energy.values())),
        "rank_max_absolute_relative_balance": float(max(
            Decimal(d["absolute_relative_balance"]) for d in energy.values())),
        "face_interior_sum_minus_ledger": float(internal_sum - sums["interior_delta"]),
        "faces": {"x{}:{}".format(axis, kind): dict(
            count=value["count"], max_relative_mismatch=float(value["max"]),
            weighted_relative_mismatch=float(value["numerator"] /
                value["denominator"]) if value["denominator"] else None,
            net_balance=float(value["net"])) for (axis, kind), value in
            sorted(by_kind.items())}}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps([audit(path) for path in args.job_dirs], indent=2))


if __name__ == "__main__":
    main()
