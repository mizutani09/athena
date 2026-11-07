"""Audit the ten-cycle precision-floor table run's implicit energy ledger."""

import csv
import hashlib
import json
import re
import tarfile
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
ROOT = Path("/home/kosuke/simulation/test/rad_nr/wor92_wor105_retest_20261107")
RUN = ROOT / "table_floor32_energy_10cycles_2rank"
PRIOR = ROOT / "table_local_jacobian_floor32_cfl030_10cycles_2rank/run.log"
INPUT = ROOT / "input_table_local_jacobian_floor32_cfl030_10cycles.used"
BINARY = Path("/home/kosuke/simulation/test/rad_nr/wor92_operator_audit_20261107/build_table_precision/bin/athena")
OUT = HERE / "wor92_floor32_energy_audit_results_20261107"
FACES = (
    ("x1min", "x1max", "x2min", "x2max", "x3min", "x3max", "xminus_delta", "xplus_delta"),
    ("x2min", "x2max", "x1min", "x1max", "x3min", "x3max", "yminus_delta", "yplus_delta"),
    ("x3min", "x3max", "x1min", "x1max", "x2min", "x2max", "zminus_delta", "zplus_delta"),
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fields(line):
    return dict(re.findall(r"([A-Za-z_][A-Za-z_0-9]*)=([^\s]+)", line))


def trace(path):
    lines = [s for s in path.read_text(errors="replace").splitlines()
             if s.startswith(("[LINEAR_FGMRES]", "[NR] status=converged", "cycle="))]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest(), lines


def overlap(a, b):
    return min(a[1], b[1]) > max(a[0], b[0])


def groups(faces):
    parent = list(range(len(faces)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, left in enumerate(faces):
        for j in range(i+1, len(faces)):
            right = faces[j]
            if left["side"] != right["side"] and overlap(left["rect"][0], right["rect"][0]) and overlap(left["rect"][1], right["rect"][1]):
                parent[find(i)] = find(j)
    result = defaultdict(list)
    for i, face in enumerate(faces):
        result[find(i)].append(face)
    return result.values()


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    old_hash, old_trace = trace(PRIOR)
    new_hash, new_trace = trace(RUN / "run.log")
    assert old_hash == new_hash and len(new_trace) == 43
    rank_files = sorted(RUN.glob("ledger.rank*.log"))
    assert len(rank_files) == 2
    blocks, energy = defaultdict(list), defaultdict(dict)
    for path in rank_files:
        for line in path.read_text().splitlines():
            if line.startswith("[NR_INTERFACE_LEDGER]"):
                d = fields(line)
                key = int(d["cycle"]), int(d["stage"])
                blocks[key].append({k: int(v) if k in ("rank", "cycle", "stage", "gid", "level") else Decimal(v)
                                    for k,v in d.items()})
            elif line.startswith("[NR_ENERGY_LEDGER]"):
                d = fields(line)
                key = int(d["cycle"]), int(d["stage"])
                rank = int(d["rank"])
                assert rank not in energy[key]
                energy[key][rank] = d
    assert set(blocks) == set(energy) and len(blocks) == 10
    stage_rows, face_rows = [], []
    for (cycle, stage), members in sorted(blocks.items()):
        assert stage == 0 and len(members) == 120
        assert len({b["gid"] for b in members}) == 120
        assert set(energy[(cycle, stage)]) == {0,1}
        stage_faces = []
        for axis, names in enumerate(FACES, 1):
            lo, hi, u0, u1, v0, v1, minus, plus = names
            planes = defaultdict(list)
            for b in members:
                rect = ((b[u0], b[u1]), (b[v0], b[v1]))
                planes[b[lo]].append({"side": "minus", "rect": rect,
                                      "delta": b[minus], "level": b["level"]})
                planes[b[hi]].append({"side": "plus", "rect": rect,
                                      "delta": b[plus], "level": b["level"]})
            for coord, faces in planes.items():
                for component in groups(faces):
                    left = [f for f in component if f["side"] == "plus"]
                    right = [f for f in component if f["side"] == "minus"]
                    if not left or not right:
                        continue
                    def area(f):
                        return ((f["rect"][0][1]-f["rect"][0][0])
                                *(f["rect"][1][1]-f["rect"][1][0]))
                    assert sum(map(area,left),Decimal(0)) == sum(map(area,right),Decimal(0))
                    ld = sum((f["delta"] for f in left),Decimal(0))
                    rd = sum((f["delta"] for f in right),Decimal(0))
                    scale = abs(ld)+abs(rd)
                    row = {"cycle": cycle, "stage": stage, "axis": axis,
                           "coord": float(coord),
                           "coarse_fine": {f["level"] for f in left} != {f["level"] for f in right},
                           "left_delta": float(ld), "right_delta": float(rd),
                           "balance": float(ld+rd),
                           "relative_mismatch": float(abs(ld+rd)/scale) if scale else 0.0}
                    stage_faces.append(row)
                    face_rows.append(row)
        ledgers = list(energy[(cycle,stage)].values())
        interior = sum((Decimal(d["interior_delta"]) for d in ledgers),Decimal(0))
        summed_faces = sum((Decimal(str(r["balance"])) for r in stage_faces),Decimal(0))
        assert abs(interior-summed_faces) < Decimal("1e-3")
        cf = [r for r in stage_faces if r["coarse_fine"]]
        assert len(cf) == 24
        denominator = sum(abs(r["left_delta"])+abs(r["right_delta"]) for r in cf)
        stage_rows.append({
            "cycle": cycle, "stage": stage, "blocks": len(members),
            "coarse_fine_groups": len(cf),
            "weighted_cf_mismatch": sum(abs(r["balance"]) for r in cf)/denominator,
            "max_cf_mismatch": max(r["relative_mismatch"] for r in cf),
            "cf_net_balance": sum(r["balance"] for r in cf),
            "interior_diffusion_delta": float(interior),
            "energy_balance": float(sum((Decimal(d["balance"]) for d in ledgers),Decimal(0))),
            "max_rank_relative_energy_balance": max(float(d["relative_balance"]) for d in ledgers),
            "max_rank_absolute_relative_energy_balance": max(float(d["absolute_relative_balance"]) for d in ledgers),
        })
    write_csv(OUT / "stages.csv",stage_rows)
    write_csv(OUT / "faces.csv",face_rows)
    summary = {
        "binary_sha256": sha(BINARY), "input_sha256": sha(INPUT),
        "run_log_sha256": sha(RUN / "run.log"),
        "rank_log_sha256": {p.name:sha(p) for p in rank_files},
        "numerical_trace_sha256": new_hash,
        "same_numerical_trace_as_prior": old_hash == new_hash,
        "cycles": len(stage_rows),
        "max_weighted_cf_mismatch": max(r["weighted_cf_mismatch"] for r in stage_rows),
        "max_relative_energy_balance": max(r["max_rank_relative_energy_balance"] for r in stage_rows),
        "max_absolute_relative_energy_balance": max(r["max_rank_absolute_relative_energy_balance"] for r in stage_rows),
        "stages": stage_rows,
    }
    (OUT / "summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    with tarfile.open(OUT / "evidence.tar.gz","w:gz") as archive:
        archive.add(INPUT,arcname="input.used")
        archive.add(RUN / "run.log",arcname="run.log")
        for path in rank_files:
            archive.add(path,arcname=path.name)
    fig,(ax,bx) = plt.subplots(1,2,figsize=(9,3.5))
    ax.semilogy([r["cycle"] for r in stage_rows],
                [r["weighted_cf_mismatch"] for r in stage_rows],"o-")
    ax.set_ylabel("weighted coarse-fine face mismatch")
    bx.semilogy([r["cycle"] for r in stage_rows],
                [r["max_rank_relative_energy_balance"] for r in stage_rows],"o-",label="signed")
    bx.semilogy([r["cycle"] for r in stage_rows],
                [r["max_rank_absolute_relative_energy_balance"] for r in stage_rows],"s-",label="absolute")
    bx.legend()
    bx.set_ylabel("relative implicit energy-balance error")
    for axes in (ax,bx):
        axes.set_xlabel("cycle")
        axes.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT / "floor32_energy.png",dpi=170)
    plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k != "stages"},indent=2))


if __name__ == "__main__":
    main()
