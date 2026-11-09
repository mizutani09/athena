#!/usr/bin/env python3
"""Sum isolated per-block hydro flux-divergence mass ledgers by RK stage."""

import argparse
from collections import defaultdict
from decimal import Decimal
import json
from pathlib import Path


MASS_UNIT = Decimal("1e-18") * Decimal("1.495978707e13")**3


def fields(line):
    return dict(item.split("=", 1) for item in line.split()[1:] if "=" in item)


def audit(log_dir, output, before_history=None, after_history=None):
    paths = sorted(log_dir.glob("rank-*.log"))
    if len(paths) != 64:
        raise ValueError("incomplete rank logs")
    by_weight = defaultdict(list)
    for path in paths:
        for line in path.open():
            if line.startswith("[WOR92_HYDRO_MASS_FLUX] "):
                item = fields(line)
                if int(item["cycle"]) != 930:
                    raise ValueError("unexpected cycle")
                by_weight[item["wght"]].append((int(item["gid"]),
                                                Decimal(item["delta_code"])))
    if len(by_weight) != 2:
        raise ValueError("expected two RK2 hydro stages, got {}".format(
            len(by_weight)))
    result = {"job_dir": str(log_dir), "rank_logs": len(paths),
              "mass_unit_g": float(MASS_UNIT), "stages": []}
    for weight, entries in sorted(by_weight.items(), key=lambda item: Decimal(item[0])):
        if len(entries) != 3536 or len(set(gid for gid, _ in entries)) != 3536:
            raise ValueError("missing or duplicate MeshBlocks at weight " + weight)
        code_sum = sum((value for _, value in entries), Decimal(0))
        absolute_sum = sum((abs(value) for _, value in entries), Decimal(0))
        result["stages"].append({"weight": float(weight),
                                 "blocks": len(entries),
                                 "net_delta_code": float(code_sum),
                                 "net_delta_g": float(code_sum*MASS_UNIT),
                                 "sum_abs_delta_code": float(absolute_sum),
                                 "relative_net_to_sum_abs": float(
                                     abs(code_sum)/absolute_sum)})
    if before_history is not None and after_history is not None:
        def mass(path):
            rows = [row.split() for row in path.read_text().splitlines() if row.strip()]
            return Decimal(rows[-1][22])
        observed = mass(after_history) - mass(before_history)
        predicted = sum((Decimal(str(stage["net_delta_g"]))
                         for stage in result["stages"]), Decimal(0))
        result["history_mass_change_g"] = float(observed)
        result["predicted_flux_mass_change_g"] = float(predicted)
        result["predicted_over_observed"] = float(predicted/observed)
        result["history_paths"] = [str(before_history), str(after_history)]
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rank_log_dir", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--before-history", type=Path)
    parser.add_argument("--after-history", type=Path)
    args = parser.parse_args()
    if (args.before_history is None) != (args.after_history is None):
        parser.error("both history paths are required together")
    print(json.dumps(audit(args.rank_log_dir, args.output_json,
                           args.before_history, args.after_history), indent=2))


if __name__ == "__main__":
    main()
