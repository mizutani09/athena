#!/usr/bin/env python3
"""Pair complete 3D AMR face ledgers for weighted-state trial and baseline."""

import csv
from collections import defaultdict
from decimal import Decimal
import hashlib
import json
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
ROOT = Path("/home/kosuke/simulation/test/rad_nr/wor92_wor105_retest_20261107")
OUT = HERE / "wor92_weighted_state_ledger_results_20261107"
OUT.mkdir(exist_ok=True)
CASES = {
    "baseline": "fixed_high_baseline_rankledger_2rank",
    "weighted_state": "fixed_high_weighted_state_rankledger_2rank",
}
FIELDS = {"rank", "cycle", "stage", "gid", "level", "dt", "x1min", "x1max",
          "x2min", "x2max", "x3min", "x3max", "xminus_delta", "xplus_delta",
          "yminus_delta", "yplus_delta", "zminus_delta", "zplus_delta"}
FACE_NAMES = (("x1min", "x1max", "x2min", "x2max", "x3min", "x3max", "xminus_delta", "xplus_delta"),
              ("x2min", "x2max", "x1min", "x1max", "x3min", "x3max", "yminus_delta", "yplus_delta"),
              ("x3min", "x3max", "x1min", "x1max", "x2min", "x2max", "zminus_delta", "zplus_delta"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse(line):
    return dict(part.split("=", 1) for part in line.split() if "=" in part)


def overlap(a, b):
    return min(a[1], b[1]) > max(a[0], b[0])


def pair_faces(faces):
    parent = list(range(len(faces)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, left in enumerate(faces):
        for j in range(i + 1, len(faces)):
            right = faces[j]
            if left["side"] != right["side"] and overlap(left["rect"][0], right["rect"][0]) and overlap(left["rect"][1], right["rect"][1]):
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i, face in enumerate(faces):
        groups[find(i)].append(face)
    return groups.values()


stages, groups_out, paths = [], [], []
for case, dirname in CASES.items():
    directory = ROOT / dirname
    rank_files = sorted(directory.glob("ledger.rank*.log"))
    if len(rank_files) != 2:
        raise ValueError(f"{case}: expected two rank-local ledgers")
    blocks, energy = defaultdict(list), defaultdict(dict)
    for path in rank_files:
        paths.append(path)
        for line in path.read_text().splitlines():
            if line.startswith("[NR_INTERFACE_LEDGER]"):
                d = parse(line)
                if FIELDS - d.keys():
                    raise ValueError(f"incomplete interface ledger in {path}: {line}")
                key = int(d["cycle"]), int(d["stage"])
                blocks[key].append({k: int(v) if k in ("rank", "cycle", "stage", "gid", "level") else Decimal(v)
                                    for k, v in d.items()})
            elif line.startswith("[NR_ENERGY_LEDGER]"):
                d = parse(line)
                key = int(d["cycle"]), int(d["stage"])
                rank = int(d["rank"])
                if rank in energy[key]:
                    raise ValueError(f"duplicate energy ledger for {case} {key} rank {rank}")
                energy[key][rank] = d
    if set(blocks) != set(energy) or len(blocks) != 3:
        raise ValueError(f"incomplete stages in {case}")
    for (cycle, stage), members in sorted(blocks.items()):
        if len(members) != 120 or len({b["gid"] for b in members}) != 120 or set(energy[(cycle, stage)]) != {0, 1}:
            raise ValueError(f"incomplete 120-block/two-rank stage in {case} cycle {cycle}")
        e = energy[(cycle, stage)]
        interior = sum((Decimal(d["interior_delta"]) for d in e.values()), Decimal(0))
        balance = sum((Decimal(d["balance"]) for d in e.values()), Decimal(0))
        stage_groups = []
        for axis, names in enumerate(FACE_NAMES, 1):
            lo, hi, u0, u1, v0, v1, minus, plus = names
            planes = defaultdict(list)
            for b in members:
                rect = ((b[u0], b[u1]), (b[v0], b[v1]))
                planes[b[lo]].append({"side": "minus", "rect": rect, "delta": b[minus], "level": b["level"]})
                planes[b[hi]].append({"side": "plus", "rect": rect, "delta": b[plus], "level": b["level"]})
            for coord, faces in planes.items():
                for component in pair_faces(faces):
                    left = [f for f in component if f["side"] == "plus"]
                    right = [f for f in component if f["side"] == "minus"]
                    if not left or not right:
                        continue
                    left_area = sum(((f["rect"][0][1] - f["rect"][0][0]) *
                                     (f["rect"][1][1] - f["rect"][1][0]) for f in left), Decimal(0))
                    right_area = sum(((f["rect"][0][1] - f["rect"][0][0]) *
                                      (f["rect"][1][1] - f["rect"][1][0]) for f in right), Decimal(0))
                    if left_area != right_area:
                        raise ValueError(f"unmatched face area in {case} cycle {cycle} axis {axis}")
                    ld = sum((f["delta"] for f in left), Decimal(0))
                    rd = sum((f["delta"] for f in right), Decimal(0))
                    scale = abs(ld) + abs(rd)
                    row = {"case": case, "cycle": cycle, "stage": stage, "axis": axis,
                           "coord": float(coord), "coarse_fine": {f["level"] for f in left} != {f["level"] for f in right},
                           "left_delta": float(ld), "right_delta": float(rd),
                           "balance": float(ld + rd), "relative_mismatch": float(abs(ld + rd) / scale) if scale else 0.0}
                    stage_groups.append(row)
                    groups_out.append(row)
        all_internal = sum((Decimal(str(r["balance"])) for r in stage_groups), Decimal(0))
        # Face rows are serialized as floats; allow their final serialization error.
        if abs(all_internal - interior) > Decimal("1e-3"):
            raise ValueError(f"face/energy mismatch in {case} cycle {cycle}: {all_internal} vs {interior}")
        cf = [r for r in stage_groups if r["coarse_fine"]]
        scale = sum(abs(r["left_delta"]) + abs(r["right_delta"]) for r in cf)
        stages.append({"case": case, "cycle": cycle, "stage": stage,
                       "block_count": len(members), "coarse_fine_groups": len(cf),
                       "max_coarse_fine_relative_mismatch": max(r["relative_mismatch"] for r in cf),
                       "weighted_coarse_fine_relative_mismatch": sum(abs(r["balance"]) for r in cf) / scale,
                       "coarse_fine_net_balance": sum(r["balance"] for r in cf),
                       "interior_delta": float(interior), "energy_balance": float(balance)})

for name, rows in (("stages.csv", stages), ("face_groups.csv", groups_out)):
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

input_path = ROOT / "input_table_fixed_rosseland_high_cfl030_strict_3cycles.used"
builds = Path("/home/kosuke/simulation/test/rad_nr/wor92_operator_audit_20261107")
summary = {"base_git_hash": "ebf7634a8e877edfd58724de1e3810638a76ca7e",
           "input_sha256": sha(input_path),
           "binary_sha256": {"baseline": sha(builds / "build_table_coupled_jv/bin/athena"),
                             "weighted_state": sha(builds / "build_table_weighted_state/bin/athena")},
           "rank_ledger_sha256": {str(p): sha(p) for p in paths}, "stages": stages}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
with tarfile.open(OUT / "evidence.tar.gz", "w:gz") as archive:
    archive.add(input_path, arcname=f"inputs/{input_path.name}")
    for case, dirname in CASES.items():
        directory = ROOT / dirname
        for name in ("run.log", "status.txt", "ledger.rank0.log", "ledger.rank1.log"):
            archive.add(directory / name, arcname=f"runs/{case}/{name}")
    patch = HERE / "wor92_transfer_consistency_results_20261107/weighted_state.patch"
    archive.add(patch, arcname="weighted_state.patch")

fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.4, 3.8))
for case, label, color in (("baseline", "baseline", "#d95f02"),
                           ("weighted_state", "weighted state", "#1b9e77")):
    rows = [r for r in stages if r["case"] == case]
    ax.plot([r["cycle"] for r in rows], [r["weighted_coarse_fine_relative_mismatch"] for r in rows],
            marker="o", color=color, label=label)
    bx.plot([r["cycle"] for r in rows], [abs(r["interior_delta"]) for r in rows],
            marker="o", color=color, label=label)
ax.set_yscale("log")
bx.set_yscale("log")
ax.set_ylabel("Weighted coarse-fine face mismatch")
bx.set_ylabel("Absolute interior diffusion sum")
for axes in (ax, bx):
    axes.set_xlabel("Cycle")
    axes.set_xticks([0, 1, 2])
    axes.grid(alpha=0.25)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "ledger_comparison.png", dpi=160)
plt.close(fig)
print(json.dumps(stages, indent=2))
