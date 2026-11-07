#!/usr/bin/env python3
"""Audit per-rank energy and matched AMR-face fluxes after positive ghost limiting."""

import csv
import hashlib
import json
import tarfile
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wor92_floor32_energy_audit_20261107 import FACES, fields, groups


HERE = Path(__file__).resolve().parent
OUT = HERE / "wor92_positive_ghost_ledger_results_20261107"
ROOT = Path("/home/kosuke/simulation/test/rad_nr/wor92_wor105_retest_20261107")
RUN = ROOT / "table_domain_mean_limiter_ledger_isolated_restart74_to100_2rank"
PRIOR = ROOT / "table_domain_mean_limiter_restart74_to100_2rank"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    rank_logs = sorted(RUN.glob("rank.*.log"))
    assert len(rank_logs) == 2
    blocks, energy = defaultdict(list), defaultdict(dict)
    for path in rank_logs:
        for line in path.read_text(errors="replace").splitlines():
            if line.startswith("[NR_INTERFACE_LEDGER]"):
                d = fields(line)
                key = int(d["cycle"]), int(d["stage"])
                blocks[key].append({k: int(v) if k in ("rank", "cycle", "stage", "gid", "level")
                                    else Decimal(v) for k, v in d.items()})
            if line.startswith("[NR_ENERGY_LEDGER]"):
                d = fields(line)
                key = int(d["cycle"]), int(d["stage"])
                rank = int(d["rank"])
                assert rank not in energy[key]
                energy[key][rank] = d
    assert set(blocks) == set(energy)
    assert len(blocks) == 26
    assert {cycle for cycle, _ in blocks} == set(range(74, 100))
    stages, faces_out = [], []
    for (cycle, stage), members in sorted(blocks.items()):
        assert stage == 0 and len(members) == 120
        assert len({m["gid"] for m in members}) == 120
        assert set(energy[(cycle, stage)]) == {0, 1}
        matched = []
        for axis, names in enumerate(FACES, 1):
            lo, hi, u0, u1, v0, v1, minus, plus = names
            planes = defaultdict(list)
            for block in members:
                rect = ((block[u0], block[u1]), (block[v0], block[v1]))
                planes[block[lo]].append({"side": "minus", "rect": rect,
                                          "delta": block[minus], "level": block["level"]})
                planes[block[hi]].append({"side": "plus", "rect": rect,
                                          "delta": block[plus], "level": block["level"]})
            for coord, entries in planes.items():
                for component in groups(entries):
                    left = [f for f in component if f["side"] == "plus"]
                    right = [f for f in component if f["side"] == "minus"]
                    if not left or not right:
                        continue
                    area = lambda f: ((f["rect"][0][1] - f["rect"][0][0])
                                      * (f["rect"][1][1] - f["rect"][1][0]))
                    assert sum(map(area, left), Decimal(0)) == sum(map(area, right), Decimal(0))
                    ld = sum((f["delta"] for f in left), Decimal(0))
                    rd = sum((f["delta"] for f in right), Decimal(0))
                    scale = abs(ld) + abs(rd)
                    row = {"cycle": cycle, "stage": stage, "axis": axis, "coord": float(coord),
                           "coarse_fine": {f["level"] for f in left} != {f["level"] for f in right},
                           "left_delta": float(ld), "right_delta": float(rd),
                           "balance": float(ld + rd),
                           "relative_mismatch": float(abs(ld + rd) / scale) if scale else 0.0}
                    matched.append(row)
                    faces_out.append(row)
        ledgers = list(energy[(cycle, stage)].values())
        interior = sum((Decimal(d["interior_delta"]) for d in ledgers), Decimal(0))
        summed_faces = sum((Decimal(str(r["balance"])) for r in matched), Decimal(0))
        assert abs(interior - summed_faces) < Decimal("1e-3")
        cf = [r for r in matched if r["coarse_fine"]]
        assert len(cf) == 24
        denominator = sum(abs(r["left_delta"]) + abs(r["right_delta"]) for r in cf)
        stages.append({"cycle": cycle, "stage": stage, "blocks": len(members),
                       "coarse_fine_groups": len(cf),
                       "weighted_cf_mismatch": sum(abs(r["balance"]) for r in cf) / denominator,
                       "max_cf_mismatch": max(r["relative_mismatch"] for r in cf),
                       "cf_net_balance": sum(r["balance"] for r in cf),
                       "interior_diffusion_delta": float(interior),
                       "energy_balance": float(sum((Decimal(d["balance"]) for d in ledgers), Decimal(0))),
                       "max_rank_relative_energy_balance": max(float(d["relative_balance"]) for d in ledgers),
                       "max_rank_absolute_relative_energy_balance": max(float(d["absolute_relative_balance"]) for d in ledgers)})
    write_csv(OUT / "stages.csv", stages)
    write_csv(OUT / "faces.csv", faces_out)
    def numerical_lines(path):
        return [s for s in path.read_text(errors="replace").splitlines()
                if s.startswith(("cycle=", "[NR] status=", "[LINEAR_FGMRES] iterations="))]
    assert numerical_lines(RUN / "rank.0.log") == numerical_lines(PRIOR / "run.log")
    assert (RUN / "cfl030_flux_precision16_local.hst").read_bytes() == (PRIOR / "cfl030_flux_precision16_local.hst").read_bytes()
    summary = {"cycles": 26, "cycle_range": [74, 99], "blocks_per_stage": 120,
               "coarse_fine_groups_per_stage": 24,
               "weighted_cf_mismatch_max": max(s["weighted_cf_mismatch"] for s in stages),
               "weighted_cf_mismatch_cycle": max(stages, key=lambda s: s["weighted_cf_mismatch"])["cycle"],
               "max_rank_relative_energy_balance": max(s["max_rank_relative_energy_balance"] for s in stages),
               "max_rank_absolute_relative_energy_balance": max(s["max_rank_absolute_relative_energy_balance"] for s in stages),
               "numerical_trace_identical_to_no_ledger": True,
               "history_byte_identical_to_no_ledger": True,
               "rank_log_sha256": {p.name: sha(p) for p in rank_logs},
               "launcher_log_sha256": sha(RUN / "launcher.log")}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    cycles = [r["cycle"] for r in stages]
    axes[0].semilogy(cycles, [r["weighted_cf_mismatch"] for r in stages], marker="o", ms=3)
    axes[0].set_ylabel("weighted coarse-fine face mismatch")
    axes[1].semilogy(cycles, [r["max_rank_relative_energy_balance"] for r in stages], marker="o", ms=3)
    axes[1].set_ylabel("maximum rank signed energy balance / scale")
    for ax in axes:
        ax.axvspan(87, 88, color="#e5a965", alpha=0.25)
        ax.set_xlabel("cycle")
    fig.tight_layout()
    fig.savefig(OUT / "ledger.png", dpi=180)
    plt.close(fig)
    with tarfile.open(OUT / "evidence.tar.gz", "w:gz") as archive:
        for path in rank_logs + [RUN / "launcher.log", RUN / "cfl030_flux_precision16_local.hst"]:
            archive.add(path, arcname="logs/" + path.name)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
