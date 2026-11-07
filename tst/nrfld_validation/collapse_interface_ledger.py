#!/usr/bin/env python3
"""Compare opposing coarse/fine integrated FLD face terms in an Athena log.

When present, report the opt-in cancellation scale alongside each face pair.
This scale is a diagnostic heuristic, not a conservation tolerance.
"""

import argparse
import json
import re
from pathlib import Path


def read_faces(path):
    rows = []
    required = {"cycle", "stage", "gid", "level", "x1min", "x1max",
                "x2min", "x2max", "x3min", "x3max", "xminus_delta",
                "xplus_delta", "yminus_delta", "yplus_delta",
                "zminus_delta", "zplus_delta"}
    for line in path.read_text().splitlines():
        if line.startswith("[NR_INTERFACE_LEDGER]"):
            row = {key: float(value) for key, value in re.findall(
                r"([A-Za-z0-9_]+)=([+-]?[0-9.eE+-]+)", line)}
            missing = required - row.keys()
            if missing:
                raise ValueError(f"Truncated MPI ledger line in {path}: "
                                 f"missing {sorted(missing)}")
            rows.append(row)
    return rows


def compare(rows):
    pairs = []
    for coarse in rows:
        for axis, letter in enumerate("xyz", 1):
            transverse = [other for other in range(1, 4) if other != axis]
            for side in ("min", "max"):
                opposite = "max" if side == "min" else "min"
                position = coarse[f"x{axis}{side}"]
                fine = [row for row in rows
                        if row["level"] == coarse["level"] + 1
                        and row["cycle"] == coarse["cycle"]
                        and row["stage"] == coarse["stage"]
                        and row[f"x{axis}{opposite}"] == position
                        and all(row[f"x{other}min"] >= coarse[f"x{other}min"]
                                and row[f"x{other}max"] <= coarse[f"x{other}max"]
                                for other in transverse)]
                if not fine:
                    continue
                coarse_key = f"{letter}{'minus' if side == 'min' else 'plus'}_delta"
                fine_key = f"{letter}{'plus' if side == 'min' else 'minus'}_delta"
                coarse_flux = coarse[coarse_key]
                fine_flux = sum(row[fine_key] for row in fine)
                mismatch = coarse_flux + fine_flux
                scale = abs(coarse_flux) + abs(fine_flux)
                pair = {"cycle": int(coarse["cycle"]),
                        "stage": int(coarse["stage"]),
                        "coarse_gid": int(coarse["gid"]),
                        "axis": axis, "side": side,
                        "fine_count": len(fine),
                        "coarse_flux": coarse_flux,
                        "fine_flux": fine_flux,
                        "mismatch": mismatch,
                        "relative_mismatch": abs(mismatch) / scale if scale else 0.0}
                coarse_scale_key = coarse_key.replace("_delta", "_roundoff_scale")
                fine_scale_key = fine_key.replace("_delta", "_roundoff_scale")
                if coarse_scale_key in coarse and all(
                        fine_scale_key in row for row in fine):
                    roundoff_scale = coarse[coarse_scale_key] + sum(
                        row[fine_scale_key] for row in fine)
                    pair["roundoff_scale"] = roundoff_scale
                    pair["mismatch_over_roundoff_scale"] = (
                        abs(mismatch) / roundoff_scale if roundoff_scale else None)
                pairs.append(pair)
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, nargs="+")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--require-conservation", action="store_true")
    args = parser.parse_args()
    pairs = compare([row for path in args.log for row in read_faces(path)])
    maximum = max((pair["relative_mismatch"] for pair in pairs), default=None)
    result = {"logs": [str(path) for path in args.log], "pairs": pairs,
              "max_relative_mismatch": maximum,
              "pass": maximum is not None and maximum <= args.tolerance}
    print(json.dumps(result, indent=2))
    if args.require_conservation and not result["pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
