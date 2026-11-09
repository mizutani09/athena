#!/usr/bin/env python3
"""Summarize one segmented WOR-92 first-core continuation job."""

import argparse
import csv
import json
from pathlib import Path
import re


def summarize(case):
    with (case / "progress.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    newton = []
    linear = []
    opacity_missing = []
    opacity_bad = []
    nonmonotone_converged = []
    for i, row in enumerate(rows):
        path = case / ("segment_%03d.log" % i)
        text = path.read_text(errors="replace")
        for iteration, residual in re.findall(
                r"^\[LINEAR_FGMRES\] iterations=(\d+) relative_residual=([\d.eE+-]+)",
                text, re.M):
            linear.append((int(iteration), float(residual)))
        for status, maximum in re.findall(
                r"^\[NR\] status=(\w+) .*?final_max=([\d.eE+-]+)",
                text, re.M):
            newton.append((status, float(maximum)))
        for line in text.splitlines():
            if line.startswith("[NR_TRIAL]") and "accepted=1 converged_trial=1" in line:
                match = re.search(r"l2_factor=([\d.eE+-]+)", line)
                if match and float(match.group(1)) > 1.0:
                    nonmonotone_converged.append((i, float(match.group(1))))
        diagnostics = [line for line in text.splitlines()
                       if line.startswith("FLD_OPACITY_DIAGNOSTICS ")]
        if not diagnostics:
            opacity_missing.append(i)
        elif any(line != "FLD_OPACITY_DIAGNOSTICS out_of_domain=0 clamped=0 "
                      "negative_results=0 zero_results=0 nonfinite_results=0"
                 for line in diagnostics):
            opacity_bad.append(i)
    statuses = {}
    for status, _ in newton:
        statuses[status] = statuses.get(status, 0) + 1
    mass_first = float(rows[0]["gas_mass"]) if rows else None
    mass_last = float(rows[-1]["gas_mass"]) if rows else None
    return {
        "case": str(case), "segments": len(rows),
        "first_cycle": int(rows[0]["start_cycle"]) if rows else None,
        "last_cycle": int(rows[-1]["end_cycle"]) if rows else None,
        "last_reason": rows[-1]["reason"] if rows else None,
        "rho_max_g_cm3": float(rows[-1]["rho_max"]) if rows else None,
        "tgas_max_K": float(rows[-1]["tgas_max"]) if rows else None,
        "min_jeans_cells": min(float(row["jeans_min"]) for row in rows) if rows else None,
        "gas_mass_relative_change_history_precision": (
            (mass_last - mass_first) / mass_first if mass_first else None),
        "newton_status_counts": statuses,
        "newton_converged_max_final": max((value for status, value in newton
                                           if status == "converged"), default=None),
        "fgmres_solves": len(linear),
        "fgmres_iterations_total": sum(item[0] for item in linear),
        "fgmres_max_true_relative_residual": max((item[1] for item in linear),
                                                  default=None),
        "accepted_converged_trial_with_l2_increase": nonmonotone_converged,
        "opacity_missing_diagnostic_segments": opacity_missing,
        "opacity_nonzero_diagnostic_segments": opacity_bad,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.case)
    content = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(content)
    print(content, end="")


if __name__ == "__main__":
    main()
