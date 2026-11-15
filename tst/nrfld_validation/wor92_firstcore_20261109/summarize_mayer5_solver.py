#!/usr/bin/env python3
"""Summarize one guarded XD collapse job's solver logs and Slurm accounting."""

import argparse
import json
import re
import subprocess
from pathlib import Path


def field(line, name):
    match = re.search(r"(?:^|\s)" + re.escape(name) + r"=([^\s]+)", line)
    if match is None:
        raise ValueError(f"Missing {name} in {line[:160]}")
    return match.group(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_dir", type=Path)
    parser.add_argument("job_id", type=int)
    args = parser.parse_args()
    logs = sorted(args.job_dir.glob("segment_*.log"))
    if not logs and (args.job_dir / "launcher.log").exists():
        logs = [args.job_dir / "launcher.log"]
    if not logs:
        raise ValueError("No segment logs found")
    result = {"job_id": args.job_id, "segment_logs": len(logs),
              "newton_total": 0, "newton_converged": 0,
              "newton_nonconverged": 0, "newton_iterations_max": 0,
              "fgmres_solves": 0, "fgmres_iterations": 0,
              "fgmres_true_relative_max": 0.0, "fatal_lines": 0}
    for path in logs:
        with path.open(errors="replace") as stream:
            for line in stream:
                if line.startswith("[NR] status="):
                    result["newton_total"] += 1
                    if field(line, "status") == "converged":
                        result["newton_converged"] += 1
                    else:
                        result["newton_nonconverged"] += 1
                    result["newton_iterations_max"] = max(
                        result["newton_iterations_max"], int(field(line, "iterations")))
                elif line.startswith("[LINEAR_FGMRES] "):
                    result["fgmres_solves"] += 1
                    result["fgmres_iterations"] += int(field(line, "iterations"))
                    result["fgmres_true_relative_max"] = max(
                        result["fgmres_true_relative_max"],
                        float(field(line, "relative_residual")))
                if "FATAL ERROR" in line:
                    result["fatal_lines"] += 1
    accounting = subprocess.check_output(
        ["sacct", "-j", str(args.job_id), "--format", "JobID,State,ElapsedRaw,MaxRSS",
         "-n", "-P"], universal_newlines=True)
    entries = [line.split("|") for line in accounting.splitlines() if line.strip()]
    parent = next((x for x in entries if x[0] == str(args.job_id)), None)
    if parent is None:
        raise ValueError(f"No Slurm accounting for job {args.job_id}")
    result["slurm_state"] = parent[1]
    result["elapsed_seconds"] = int(parent[2])
    result["maxrss_kib"] = max(
        (int(x[3][:-1]) for x in entries if x[3].endswith("K")), default=0)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
