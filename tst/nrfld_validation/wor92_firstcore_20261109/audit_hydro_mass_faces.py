#!/usr/bin/env python3
"""Locate explicit-hydro mass flux mismatch by AMR face type and RK stage."""

import argparse
from collections import defaultdict
from decimal import Decimal
import json
from pathlib import Path

from audit_high_density_faces import AXES, area, fields, groups


FLOAT_KEYS = ("wght", "x1min", "x1max", "x2min", "x2max", "x3min", "x3max",
              "xminus_delta", "xplus_delta", "yminus_delta", "yplus_delta",
              "zminus_delta", "zplus_delta", "delta_code")


def audit(log_dir):
    paths = sorted(log_dir.glob("rank-*.log"))
    if len(paths) != 64:
        raise ValueError("expected 64 rank logs")
    by_weight = defaultdict(list)
    for path in paths:
        for line in path.open():
            if not line.startswith("[WOR92_HYDRO_MASS_FLUX] "):
                continue
            raw = fields(line)
            item = {key: Decimal(value) if key in FLOAT_KEYS else int(value)
                    for key, value in raw.items()}
            if item["cycle"] != 930:
                raise ValueError("unexpected cycle")
            by_weight[item["wght"]].append(item)
    if len(by_weight) != 2:
        raise ValueError("expected two RK stages")
    stages = []
    for weight, blocks in sorted(by_weight.items()):
        if len(blocks) != 3536 or len({block["gid"] for block in blocks}) != 3536:
            raise ValueError("incomplete block set")
        by_kind = defaultdict(lambda: {"count": 0, "net": Decimal(0),
                        "abs_mismatch": Decimal(0), "scale": Decimal(0),
                        "max_relative": Decimal(0), "net_by_level": defaultdict(Decimal)})
        for axis, names in enumerate(AXES, 1):
            lo, hi, u0, u1, v0, v1, minus, plus = names
            planes = defaultdict(list)
            for block in blocks:
                rectangle = ((block[u0], block[u1]), (block[v0], block[v1]))
                planes[block[lo]].append({"side": "minus", "rect": rectangle,
                                         "level": block["level"],
                                         "delta": block[minus]})
                planes[block[hi]].append({"side": "plus", "rect": rectangle,
                                         "level": block["level"],
                                         "delta": block[plus]})
            for plane_faces in planes.values():
                for component in groups(plane_faces):
                    left = [face for face in component if face["side"] == "plus"]
                    right = [face for face in component if face["side"] == "minus"]
                    internal = bool(left and right)
                    if internal and sum((area(face) for face in left), Decimal(0)) != sum(
                            (area(face) for face in right), Decimal(0)):
                        raise ValueError("unmatched face area")
                    left_sum = sum((face["delta"] for face in left), Decimal(0))
                    right_sum = sum((face["delta"] for face in right), Decimal(0))
                    mismatch = left_sum + right_sum
                    scale = abs(left_sum) + abs(right_sum)
                    kind = "boundary" if not internal else (
                        "coarse_fine" if {face["level"] for face in left}
                        != {face["level"] for face in right} else "same")
                    data = by_kind[f"x{axis}:{kind}"]
                    data["count"] += 1
                    data["net"] += mismatch
                    data["abs_mismatch"] += abs(mismatch)
                    data["scale"] += scale
                    data["max_relative"] = max(data["max_relative"],
                                               abs(mismatch)/scale if scale else 0)
                    if kind == "same":
                        data["net_by_level"][left[0]["level"]] += mismatch
        face_total = sum((data["net"] for data in by_kind.values()), Decimal(0))
        block_total = sum((block["delta_code"] for block in blocks), Decimal(0))
        stages.append({"weight": float(weight), "blocks": len(blocks),
                       "block_net_delta_code": float(block_total),
                       "face_net_delta_code": float(face_total),
                       "face_minus_block_delta_code": float(face_total-block_total),
                       "faces": {name: {"count": data["count"],
                           "net_delta_code": float(data["net"]),
                           "weighted_relative_mismatch": float(data["abs_mismatch"] / data["scale"])
                           if data["scale"] else None,
                           "max_relative_mismatch": float(data["max_relative"]),
                           "net_by_level_code": {str(level): float(value)
                               for level, value in sorted(data["net_by_level"].items())}}
                           for name, data in sorted(by_kind.items())}})
    return {"job_dir": str(log_dir), "rank_logs": len(paths), "stages": stages}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rank_log_dir", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    result = audit(args.rank_log_dir)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
