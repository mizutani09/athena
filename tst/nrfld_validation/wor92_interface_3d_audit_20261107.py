#!/usr/bin/env python3
"""Pair complete 3D FLD block-face ledgers by actual face overlap."""
from __future__ import annotations

from collections import defaultdict
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import tarfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/kosuke/simulation/test/rad_nr/wor92_operator_audit_20261107")
RUN = ROOT / "rank_ledger_direct"
OUT = Path(__file__).with_name("wor92_interface_3d_results_20261107")
FIELDS = ("rank", "cycle", "stage", "dt", "gid", "level", "x1min", "x1max",
          "x2min", "x2max", "x3min", "x3max", "xminus_delta",
          "xplus_delta", "yminus_delta", "yplus_delta", "zminus_delta",
          "zplus_delta")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse(line):
    return dict(part.split("=", 1) for part in line.split() if "=" in part)


blocks = defaultdict(list)
energy = defaultdict(dict)
rank_files = sorted(RUN.glob("ledger.rank*.log"))
if len(rank_files) != 2:
    raise SystemExit(f"expected two rank logs, found {rank_files}")
for path in rank_files:
    for raw in path.read_text().splitlines():
        if raw.startswith("[NR_INTERFACE_LEDGER]"):
            d = parse(raw)
            if set(FIELDS) - d.keys():
                raise ValueError(f"{path}: incomplete interface line: {raw}")
            key = int(d["cycle"]), int(d["stage"])
            blocks[key].append({k: int(v) if k in ("rank", "cycle", "stage", "gid", "level")
                                else Decimal(v) for k, v in d.items()})
        elif raw.startswith("[NR_ENERGY_LEDGER]"):
            d = parse(raw)
            key = int(d["cycle"]), int(d["stage"])
            rank = int(d["rank"])
            if rank in energy[key]:
                raise ValueError(f"duplicate rank {rank} energy ledger at {key}")
            energy[key][rank] = d
if set(blocks) != set(energy):
    raise ValueError("interface and energy stages differ")


def overlap(a, b):
    return min(a[1], b[1]) > max(a[0], b[0])


def area(rect):
    return (rect[0][1]-rect[0][0])*(rect[1][1]-rect[1][0])


def parent(par, i):
    while par[i] != i:
        par[i] = par[par[i]]
        i = par[i]
    return i


def pair_faces(faces):
    n = len(faces)
    par = list(range(n))
    for i in range(n):
        for j in range(i+1, n):
            if faces[i]["side"] == faces[j]["side"]:
                continue
            if overlap(faces[i]["rect"][0], faces[j]["rect"][0]) and \
               overlap(faces[i]["rect"][1], faces[j]["rect"][1]):
                par[parent(par, i)] = parent(par, j)
    groups = defaultdict(list)
    for i, face in enumerate(faces):
        groups[parent(par, i)].append(face)
    return groups.values()


FACE_NAMES = (("x1min", "x1max", "x2min", "x2max", "x3min", "x3max",
               "xminus_delta", "xplus_delta"),
              ("x2min", "x2max", "x1min", "x1max", "x3min", "x3max",
               "yminus_delta", "yplus_delta"),
              ("x3min", "x3max", "x1min", "x1max", "x2min", "x2max",
               "zminus_delta", "zplus_delta"))
rows = []
stages = []
for (cycle, stage), members in sorted(blocks.items()):
    dt = members[0]["dt"]
    if len(members) != 120:
        raise ValueError(f"expected 120 complete leaf records at cycle {cycle}, got {len(members)}")
    gids = [b["gid"] for b in members]
    if len(set(gids)) != len(gids):
        raise ValueError(f"duplicate gids at cycle {cycle}")
    if set(energy[(cycle, stage)]) != {0, 1}:
        raise ValueError(f"missing energy rank at cycle {cycle}")
    e = energy[(cycle, stage)]
    stages.append(dict(cycle=cycle, stage=stage, block_count=len(members),
        levels={str(lev):sum(b["level"] == lev for b in members)
                for lev in sorted(set(b["level"] for b in members))},
        interior_delta=float(sum(Decimal(d["interior_delta"]) for d in e.values())),
        balance=float(sum(Decimal(d["balance"]) for d in e.values()))))
    for axis, names in enumerate(FACE_NAMES, start=1):
        lo, hi, u0, u1, v0, v1, minus, plus = names
        planes = defaultdict(list)
        for b in members:
            rect = ((b[u0], b[u1]), (b[v0], b[v1]))
            planes[b[lo]].append(dict(side="minus", rect=rect,
                                      delta=b[minus], level=b["level"], gid=b["gid"]))
            planes[b[hi]].append(dict(side="plus", rect=rect,
                                      delta=b[plus], level=b["level"], gid=b["gid"]))
        for coord, faces in sorted(planes.items()):
            for component in pair_faces(faces):
                left = [f for f in component if f["side"] == "plus"]
                right = [f for f in component if f["side"] == "minus"]
                internal = bool(left and right)
                left_area = sum((area(f["rect"]) for f in left), Decimal(0))
                right_area = sum((area(f["rect"]) for f in right), Decimal(0))
                if internal and left_area != right_area:
                    raise ValueError(f"unmatched face area at cycle {cycle} axis {axis} {coord}")
                left_delta = sum((f["delta"] for f in left), Decimal(0))
                right_delta = sum((f["delta"] for f in right), Decimal(0))
                scale = abs(left_delta) + abs(right_delta)
                balance = left_delta + right_delta
                rows.append(dict(cycle=cycle, stage=stage, axis=axis, coord=float(coord),
                    internal=internal, coarse_fine=internal and
                    set(f["level"] for f in left) != set(f["level"] for f in right),
                    left_count=len(left), right_count=len(right),
                    left_levels=sorted(set(f["level"] for f in left)),
                    right_levels=sorted(set(f["level"] for f in right)),
                    area=float(left_area or right_area),
                    left_delta=float(left_delta), right_delta=float(right_delta),
                    balance=float(balance), relative_mismatch=float(abs(balance)/scale)
                    if scale else 0.0))

for stage in stages:
    internal_sum = sum(r["balance"] for r in rows
                       if r["cycle"] == stage["cycle"] and r["internal"])
    stage["matched_internal_face_delta"] = internal_sum
    if abs(internal_sum-stage["interior_delta"]) > 1e-3:
        raise ValueError(f"face/energy ledger mismatch at cycle {stage['cycle']}")

summary = []
for axis in (1, 2, 3):
    for kind in ("same", "coarse_fine", "boundary"):
        selected = [r for r in rows if r["axis"] == axis and
                    ((kind == "boundary" and not r["internal"]) or
                     (kind == "coarse_fine" and r["coarse_fine"]) or
                     (kind == "same" and r["internal"] and not r["coarse_fine"]))]
        scale = sum(abs(r["left_delta"]) + abs(r["right_delta"]) for r in selected)
        summary.append(dict(axis=axis, kind=kind, count=len(selected),
            max_relative_mismatch=max((r["relative_mismatch"] for r in selected), default=None),
            weighted_relative_mismatch=sum(abs(r["balance"]) for r in selected)/scale
                if scale else None,
            net_balance=sum(r["balance"] for r in selected)))

OUT.mkdir(exist_ok=True)
(OUT / "summary.json").write_text(json.dumps(dict(
    source=[dict(path=str(p), sha256=sha(p)) for p in rank_files],
    base_commit="143b3403324db869b9690a2b33569252fdb5cb80",
    source_patch_sha256=sha(ROOT / "source_rank_ledger_trial.patch"),
    binary_sha256="ca8ef1ca6f78f0a75f854369d0f251d47085a2598d083288b4d754d94e281447",
    input_sha256=sha(Path("/home/kosuke/simulation/test/rad_nr/wor92_wor105_retest_20261107/input_mixed_amr_restart40_mg2.used")),
    stages=stages, summary=summary, interface_rows=rows), indent=2) + "\n")
with (OUT / "axis_summary.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(summary)
cf = [s for s in summary if s["kind"] == "coarse_fine"]
fig, ax = plt.subplots(figsize=(6.5, 3.7))
ax.bar([f"x{s['axis']}" for s in cf], [s["max_relative_mismatch"] for s in cf],
       color="#3a8193")
ax.set_ylabel("Maximum coarse/fine relative face mismatch")
ax.set_yscale("log")
ax.set_ylim(1e-8, 3e-6)
fig.tight_layout()
fig.savefig(OUT / "coarse_fine_mismatch_3d.png", dpi=160)
with tarfile.open(OUT / "rank_ledger_evidence.tar.gz", "w:gz") as archive:
    inputs = rank_files + [RUN / "run.log", ROOT / "source_rank_ledger_trial.patch",
        Path("/home/kosuke/simulation/test/rad_nr/wor92_wor105_retest_20261107/input_mixed_amr_restart40_mg2.used")]
    for path in inputs:
        archive.add(path, arcname=path.name)
for item in summary:
    print(item)
