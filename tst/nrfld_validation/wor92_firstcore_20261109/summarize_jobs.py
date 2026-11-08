#!/usr/bin/env python3
"""Summarize the archived XD pilot logs and Slurm accounting."""

import argparse
import csv
import re
from pathlib import Path


JOBS = (337947, 337948, 337949, 337950, 337956, 337957, 337958,
        337959, 337960, 337961, 337962, 337963, 337965, 337968,
        337972, 337973, 337974, 337976)


def accounting(path):
    with path.open() as stream:
        return {row["JobID"]: row for row in csv.DictReader(stream, delimiter="|")}


def summarize(root, account, job):
    case = root / "debug" / ("job_%d" % job)
    log = (case / "launcher.log").read_text(errors="replace")
    solves = re.findall(r"^\[LINEAR_FGMRES\] iterations=(\d+) "
                        r"relative_residual=([\d.eE+-]+)", log, re.M)
    newton = re.findall(r"^\[NR\] status=converged iterations=(\d+) .*?"
                        r"final_max=([\d.eE+-]+)", log, re.M)
    cycles = re.findall(r"^cycle=(\d+) time=([\d.eE+-]+)", log, re.M)
    histories = list((case / "output").glob("*.hst"))
    mass_drift = ""
    if histories:
        rows = [line.split() for line in histories[0].read_text().splitlines()
                if line and not line.startswith("#")]
        if len(rows) > 1 and len(rows[0]) >= 23:
            start, end = float(rows[0][22]), float(rows[-1][22])
            if start:
                mass_drift = "%.9e" % ((end-start)/start)
    step = account.get("%d.0" % job, {})
    batch = account.get(str(job), {})
    return {
        "job": job, "state": batch.get("State", ""),
        "exit_code": batch.get("ExitCode", ""),
        "elapsed": batch.get("Elapsed", ""),
        "max_rss_step": step.get("MaxRSS", ""),
        "first_cycle_line": cycles[0][0] if cycles else "",
        "last_cycle_line": cycles[-1][0] if cycles else "",
        "newton_converged": len(newton),
        "newton_max_final": "%.9e" % max((float(x[1]) for x in newton), default=0),
        "fgmres_solves": len(solves),
        "fgmres_iterations_total": sum(int(x[0]) for x in solves),
        "fgmres_true_max": "%.9e" % max((float(x[1]) for x in solves), default=0),
        "mass_relative_change": mass_drift,
        "opacity_clean": "out_of_domain=0 clamped=0 negative_results=0"
                         in log,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("accounting", type=Path)
    args = parser.parse_args()
    account = accounting(args.accounting)
    rows = [summarize(args.evidence, account, job) for job in JOBS]
    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=rows[0].keys(),
                            lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
