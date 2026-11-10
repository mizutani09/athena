#!/usr/bin/env python3
"""Stitch accepted ideal-EOS collapse history, excluding the failed c1095 state."""

import argparse
import csv
from pathlib import Path


TIME_TO_KYR = 1.495978707e13 / (1e9 ** 0.5) / 31557600.0 / 1000.0
FAILED_SEGMENT_START_TIME = 1943.8447769882148


def rows(path, limit=None):
    with path.open() as stream:
        for line in stream:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split()
            if len(fields) != 39:
                raise ValueError(f"Unexpected history width in {path}: {len(fields)}")
            time = float(fields[0])
            if limit is not None and time > limit + 1e-10:
                continue
            yield (time, float(fields[15]), float(fields[19]),
                   float(fields[22]), float(fields[29]), float(fields[34]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("preflight_hst", type=Path)
    p.add_argument("original_long_hst", type=Path)
    p.add_argument("replay_hst", type=Path)
    p.add_argument("continuation_hst", type=Path)
    p.add_argument("output_csv", type=Path)
    args = p.parse_args()
    parts = ((args.preflight_hst, None, "338717"),
             (args.original_long_hst, FAILED_SEGMENT_START_TIME, "338718"),
             (args.replay_hst, None, "338740"),
             (args.continuation_hst, None, "continuation"))
    merged = {}
    for path, limit, job in parts:
        for time, rho, temp, mass, pressure, jeans in rows(path, limit):
            merged[round(time, 10)] = (job, time, rho, temp, mass, pressure, jeans)
    ordered = [merged[k] for k in sorted(merged)]
    mass0 = ordered[0][4]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("job", "time_code", "time_kyr", "rho_max_g_cm3",
                         "Tgas_max_K", "gas_mass_g", "mass_rel_from_c0",
                         "pressure_min_dyn_cm2", "jeans_min_cells"))
        for job, time, rho, temp, mass, pressure, jeans in ordered:
            writer.writerow((job, f"{time:.15e}", f"{time * TIME_TO_KYR:.12e}",
                             f"{rho:.15e}", f"{temp:.15e}", f"{mass:.15e}",
                             f"{(mass-mass0)/mass0:.15e}", f"{pressure:.15e}",
                             f"{jeans:.15e}"))
    print(f"rows={len(ordered)} start={ordered[0][1]:.15g} "
          f"end={ordered[-1][1]:.15g} max_rho={ordered[-1][2]:.15g} "
          f"mass_rel={(ordered[-1][4]-mass0)/mass0:.3e} "
          f"min_jeans={min(row[6] for row in ordered):.3e}")


if __name__ == "__main__":
    main()
