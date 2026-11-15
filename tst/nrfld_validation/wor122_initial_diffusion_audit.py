#!/usr/bin/env python3
"""Aggregate the first implicit-stage discrete FLD audit printed by NRFLD."""

import argparse
import json
import math
import re
from pathlib import Path


TAG = "[NRFLD_INITIAL_DIFFUSION]"
FIELDS = ("cells", "erad", "diffusion", "diffusion_abs", "radial_flux",
          "coupling", "boundary_loss")
SHELLS = ((750, 800), (800, 813), (813, 817), (817, 830),
          (830, 850), (850, 880), (880, 900))


def parse(paths):
    rows = {}
    for path in paths:
        for line in path.open():
            if TAG not in line:
                continue
            line = line[line.index(TAG) + len(TAG):]
            fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
            if any(key not in fields for key in (*FIELDS, "gid", "bin", "rlo", "rhi")):
                raise ValueError(f"Incomplete audit row in {path}: {line}")
            key = (int(fields["gid"]), int(fields["bin"]))
            if key in rows:
                raise ValueError(f"Duplicate audit row {key}")
            rows[key] = {name: (int(fields[name]) if name == "cells"
                                else float(fields[name])) for name in FIELDS}
            rows[key]["rlo"] = float(fields["rlo"])
            rows[key]["rhi"] = float(fields["rhi"])
    if not rows:
        raise ValueError("No diffusion audit rows found")
    return rows


def summarize(rows, lo, hi, dx, lheat):
    matches = [row for row in rows.values()
               if row["rlo"] >= lo - 1e-8 and row["rhi"] <= hi + 1e-8]
    if not matches:
        return None
    total = {name: sum(row[name] for row in matches) for name in FIELDS}
    n = total["cells"]
    volume = n * dx**3
    rmid = 0.5 * (lo + hi)
    lrad = 4.0 * math.pi * rmid**2 * total["radial_flux"] / volume
    return {"cells": n, "erad": total["erad"],
            "diffusion": total["diffusion"],
            "diffusion_over_erad_per_code_time": total["diffusion"] / total["erad"],
            "abs_diffusion_over_erad_per_code_time":
                total["diffusion_abs"] / total["erad"],
            "mean_radial_flux": total["radial_flux"] / volume,
            "spherical_luminosity_estimate_over_lheat": lrad / lheat,
            "initial_coupling": total["coupling"],
            "boundary_loss": total["boundary_loss"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, nargs="+", required=True)
    parser.add_argument("--dx", type=float, required=True,
                        help="uniform Cartesian cell width in code units")
    parser.add_argument("--lheat", type=float, required=True,
                        help="stellar heating luminosity in code units")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = parse(args.log)
    shells = {f"{lo}-{hi}": summarize(rows, lo, hi, args.dx, args.lheat)
              for lo, hi in SHELLS}
    edges = sorted({(row["rlo"], row["rhi"]) for row in rows.values()})
    radial_bins = [dict(rlo=lo, rhi=hi,
                        **summarize(rows, lo, hi, args.dx, args.lheat))
                   for lo, hi in edges]
    rmin = min(row["rlo"] for row in rows.values())
    rmax = max(row["rhi"] for row in rows.values())
    full = summarize(rows, rmin, rmax, args.dx, args.lheat)
    result = {"logs": [str(p) for p in args.log], "blocks": len({gid for gid, _ in rows}),
              "rows": len(rows), "rmin": rmin, "rmax": rmax,
              "dx": args.dx, "lheat": args.lheat, "shells": shells,
              "radial_bins": radial_bins,
              "covered_domain": full}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)
    for label, shell in shells.items():
        if shell:
            print(label, shell["cells"],
                  f"diff/Er={shell['diffusion_over_erad_per_code_time']:+.4g}",
                  f"L/Lheat={shell['spherical_luminosity_estimate_over_lheat']:.4g}")


if __name__ == "__main__":
    main()
