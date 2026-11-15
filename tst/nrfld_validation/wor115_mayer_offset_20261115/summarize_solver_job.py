#!/usr/bin/env python3
"""Summarize all Newton and FGMRES solves in a guarded XD job."""

import argparse
import json
import re
import subprocess
from pathlib import Path


def field(line, name):
    match = re.search(r"(?:^|\s)" + re.escape(name) + r"=([^\s]+)", line)
    if match is None:
        raise ValueError(f"Missing {name}: {line[:160]}")
    return match.group(1)


def kib(value):
    if not value:
        return 0
    match = re.fullmatch(r"([0-9.]+)([KMGTP]?)", value)
    if match is None:
        raise ValueError(f"Unknown MaxRSS unit: {value}")
    scale = {"": 1 / 1024, "K": 1, "M": 1024, "G": 1024**2,
             "T": 1024**3, "P": 1024**4}[match.group(2)]
    return round(float(match.group(1)) * scale)


def summarize_segment(path):
    result = {"log": path.name, "newton_total": 0, "newton_converged": 0,
              "newton_dt_zero_initialization": 0,
              "newton_iterations_max": 0, "fgmres_solves": 0,
              "fgmres_iterations_total": 0, "fgmres_iterations_max": 0,
              "fgmres_true_relative_max": 0.0, "fatal_lines": 0}
    with path.open(errors="replace") as stream:
        for line in stream:
            if line.startswith("[NR] status="):
                result["newton_total"] += 1
                status = field(line, "status")
                if status == "converged":
                    result["newton_converged"] += 1
                elif status == "dt_zero_initialization":
                    result["newton_dt_zero_initialization"] += 1
                result["newton_iterations_max"] = max(
                    result["newton_iterations_max"], int(field(line, "iterations")))
            elif line.startswith("[LINEAR_FGMRES] "):
                iterations = int(field(line, "iterations"))
                result["fgmres_solves"] += 1
                result["fgmres_iterations_total"] += iterations
                result["fgmres_iterations_max"] = max(
                    result["fgmres_iterations_max"], iterations)
                result["fgmres_true_relative_max"] = max(
                    result["fgmres_true_relative_max"],
                    float(field(line, "relative_residual")))
            if "FATAL ERROR" in line:
                result["fatal_lines"] += 1
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_dir", type=Path)
    parser.add_argument("job_id", type=int)
    args = parser.parse_args()
    logs = sorted(args.job_dir.glob("segment_*.log"))
    if not logs:
        logs = list(args.job_dir.glob("launcher.log"))
    if not logs:
        raise ValueError(f"No logs in {args.job_dir}")
    segments = [summarize_segment(path) for path in logs]
    fields = ("newton_total", "newton_converged", "newton_dt_zero_initialization",
              "fgmres_solves",
              "fgmres_iterations_total", "fatal_lines")
    result = {"job_id": args.job_id, "segments": segments}
    result.update({key: sum(row[key] for row in segments) for key in fields})
    result["newton_active_total"] = (result["newton_total"]
                                     - result["newton_dt_zero_initialization"])
    for key in ("newton_iterations_max", "fgmres_iterations_max",
                "fgmres_true_relative_max"):
        result[key] = max(row[key] for row in segments)
    accounting = subprocess.check_output(
        ["sacct", "-j", str(args.job_id), "--format=JobID,State,ElapsedRaw,MaxRSS,ExitCode",
         "-n", "-P"], universal_newlines=True)
    entries = [line.split("|") for line in accounting.splitlines() if line.strip()]
    parent = next(row for row in entries if row[0] == str(args.job_id))
    result.update(slurm_state=parent[1], elapsed_seconds=int(parent[2]),
                  exit_code=parent[4],
                  maxrss_kib=max(kib(row[3]) for row in entries))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
