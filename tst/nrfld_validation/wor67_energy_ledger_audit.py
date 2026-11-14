#!/usr/bin/env python3
"""Aggregate the opt-in NRFLD implicit energy ledger across MPI ranks.

The summed non-Marshak diffusion term is an interior-face plus fixed-domain-
boundary budget. It is a coarse/fine interface check only when the physical
boundary diffusion flux is negligible, and cannot rule out cancelling defects
at separate interfaces.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal
import json
from pathlib import Path
import re


FIELDS = ("gas_delta", "radiation_delta", "diffusion_delta", "marshak_delta",
          "interior_delta", "balance")
ENERGY_LINE = re.compile(r"\[NR_ENERGY_LEDGER\]\s+(.*)")
INTERFACE_LINE = re.compile(r"\[NR_INTERFACE_LEDGER\]\s+(.*)")


def tokens(line: str, pattern: re.Pattern) -> dict[str, str] | None:
    match = pattern.search(line)
    if match is None:
        return None
    return dict(token.split("=", 1) for token in match[1].split()
                if "=" in token)


def audit(label: str, run_source: Path, ranks: int) -> dict:
    groups = defaultdict(dict)
    faces = defaultdict(list)
    paths = sorted(run_source.glob("rank.*.log")) if run_source.is_dir() else [run_source]
    if not paths:
        raise ValueError(f"{run_source}: no rank logs")
    for path in paths:
        for line in path.read_text(errors="replace").splitlines():
            record = tokens(line, ENERGY_LINE)
            if record is not None:
                missing = (set(FIELDS) | {"rank", "cycle", "stage", "dt",
                                          "relative_balance",
                                          "absolute_relative_balance"}) - record.keys()
                if missing:
                    raise ValueError(f"{path}: missing energy fields {sorted(missing)}")
                key = int(record["cycle"]), int(record["stage"]), Decimal(record["dt"])
                rank = int(record["rank"])
                if rank in groups[key]:
                    raise ValueError(f"{path}: duplicate rank {rank} at {key}")
                groups[key][rank] = {field: Decimal(record[field]) for field in FIELDS}
                groups[key][rank]["relative_balance"] = Decimal(record["relative_balance"])
                groups[key][rank]["absolute_relative_balance"] = Decimal(
                    record["absolute_relative_balance"])
            record = tokens(line, INTERFACE_LINE)
            if record is not None:
                keys = {"rank", "cycle", "stage", "dt", "gid", "level",
                        "x1min", "x1max", "x2min", "x2max", "x3min", "x3max",
                        "xminus_delta", "xplus_delta"}
                if keys - record.keys():
                    raise ValueError(f"{path}: missing interface fields {sorted(keys-record.keys())}")
                key = int(record["cycle"]), int(record["stage"]), Decimal(record["dt"])
                faces[key].append({k: (int(record[k]) if k in ("rank", "gid", "level")
                                      else Decimal(record[k])) for k in keys if k != "cycle"
                                   and k != "stage" and k != "dt"})
    if not groups:
        raise ValueError(f"{run_source}: no energy ledger records")
    stages = []
    for (cycle, stage, dt), records in sorted(groups.items()):
        if set(records) != set(range(ranks)):
            raise ValueError(f"{run_source}: incomplete cycle {cycle} stage {stage}: "
                             f"{sorted(records)} versus {ranks} ranks")
        totals = {field: sum((row[field] for row in records.values()), Decimal(0))
                  for field in FIELDS}
        abs_diffusion = sum((abs(row["diffusion_delta"])
                             for row in records.values()), Decimal(0))
        totals.update(cycle=cycle, stage=stage, dt=dt,
                      sum_abs_rank_diffusion=abs_diffusion,
                      interior_fraction_of_rank_diffusion=(
                          abs(totals["interior_delta"])/abs_diffusion
                          if abs_diffusion else Decimal(0)),
                      max_rank_relative_balance=max(
                          row["relative_balance"] for row in records.values()),
                      max_rank_absolute_relative_balance=max(
                          row["absolute_relative_balance"] for row in records.values()))
        stages.append({key: (float(value) if isinstance(value, Decimal) else value)
                       for key, value in totals.items()})
    interface_rows = []
    for (cycle, stage, dt), blocks in sorted(faces.items()):
        grouped = defaultdict(lambda: {"left": Decimal(0), "right": Decimal(0),
                                       "left_levels": set(), "right_levels": set(),
                                       "left_blocks": 0, "right_blocks": 0})
        for block in blocks:
            left = grouped[block["x1min"]]
            left["left"] += block["xminus_delta"]
            left["left_levels"].add(block["level"])
            left["left_blocks"] += 1
            right = grouped[block["x1max"]]
            right["right"] += block["xplus_delta"]
            right["right_levels"].add(block["level"])
            right["right_blocks"] += 1
        for x, row in sorted(grouped.items()):
            internal = row["left_blocks"] > 0 and row["right_blocks"] > 0
            coarse_fine = internal and row["left_levels"] != row["right_levels"]
            balance = row["left"] + row["right"]
            scale = abs(row["left"]) + abs(row["right"])
            interface_rows.append({"cycle": cycle, "stage": stage, "dt": float(dt),
                                   "x_code": float(x), "internal": internal,
                                   "coarse_fine": coarse_fine,
                                   "left_levels": sorted(row["left_levels"]),
                                   "right_levels": sorted(row["right_levels"]),
                                   "left_blocks": row["left_blocks"],
                                   "right_blocks": row["right_blocks"],
                                   "left_delta": float(row["left"]),
                                   "right_delta": float(row["right"]),
                                   "balance": float(balance),
                                   "relative_mismatch": float(abs(balance)/scale)
                                   if scale else 0.0})
    coarse_fine_rows = [r for r in interface_rows if r["coarse_fine"]]
    same_level_rows = [r for r in interface_rows if r["internal"]
                       and not r["coarse_fine"]]
    boundary_rows = [r for r in interface_rows if not r["internal"]]
    def weighted_mismatch(rows: list[dict]) -> float | None:
        scale = sum(abs(r["left_delta"]) + abs(r["right_delta"]) for r in rows)
        return sum(abs(r["balance"]) for r in rows)/scale if scale else None
    return {"label": label, "run_source": str(run_source.resolve()), "ranks": ranks,
            "stage_count": len(stages), "cycle_count": len(set(s["cycle"] for s in stages)),
            "max_interior_fraction": max(s["interior_fraction_of_rank_diffusion"]
                                         for s in stages),
            "max_rank_relative_balance": max(s["max_rank_relative_balance"]
                                             for s in stages),
            "sum_interior_delta": sum(s["interior_delta"] for s in stages),
            "sum_diffusion_delta": sum(s["diffusion_delta"] for s in stages),
            "stages": stages, "interface_rows": interface_rows,
            "coarse_fine_count": len(coarse_fine_rows),
            "coarse_fine_weighted_mismatch": weighted_mismatch(coarse_fine_rows),
            "same_level_weighted_mismatch": weighted_mismatch(same_level_rows),
            "physical_boundary_delta": sum(r["balance"] for r in boundary_rows),
            "max_abs_coarse_fine_balance": max((abs(r["balance"])
                                                 for r in coarse_fine_rows), default=None),
            "max_coarse_fine_mismatch": max((r["relative_mismatch"]
                                              for r in coarse_fine_rows), default=None),
            "sum_coarse_fine_balance": sum(r["balance"] for r in coarse_fine_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", required=True,
                        help="LABEL:RANKS:RUN_LOG_OR_RANK_LOG_DIRECTORY")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = []
    for spec in args.case:
        try:
            label, rank_text, path = spec.split(":", 2)
            ranks = int(rank_text)
        except ValueError as exc:
            parser.error(f"invalid --case {spec!r}: {exc}")
        if not label or ranks < 1:
            parser.error(f"invalid --case {spec!r}")
        cases.append(audit(label, Path(path), ranks))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"summary.json").write_text(json.dumps({"cases": cases}, indent=2)+"\n")
    for case in cases:
        print(case["label"], "cycles", case["cycle_count"],
              "max_interior_fraction", f"{case['max_interior_fraction']:.6e}",
              "max_rank_relative_balance", f"{case['max_rank_relative_balance']:.6e}",
              "coarse_fine_count", case["coarse_fine_count"],
              "max_coarse_fine_mismatch", case["max_coarse_fine_mismatch"])


if __name__ == "__main__":
    main()
