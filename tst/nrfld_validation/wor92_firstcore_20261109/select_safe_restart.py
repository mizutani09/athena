#!/usr/bin/env python3
"""Select the last validated restart before a Jeans-resolution stop."""

import argparse
import csv
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous_case", type=Path)
    parser.add_argument("--min-jeans", type=float, default=16.0)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from read_athena_restart_metadata_py36 import read_metadata

    with (args.previous_case / "progress.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("previous job has no completed segment")
    if rows[-1]["reason"].startswith("FAIL_") or rows[-1]["reason"] == "STOP_OPACITY_DOMAIN":
        raise RuntimeError("previous job stopped for a solver or opacity error")
    safe = [row for row in rows if row["reason"] == "CONTINUE"
            and int(row["exit_code"]) == 0
            and float(row["jeans_min"]) >= args.min_jeans]
    if not safe:
        raise RuntimeError("no validated checkpoint with adequate Jeans resolution")
    cycle = int(safe[-1]["end_cycle"])
    matches = []
    for path in (args.previous_case / "output").glob("*.rst"):
        try:
            if read_metadata(path)["cycle"] == cycle:
                matches.append(path)
        except (OSError, ValueError):
            pass
    if not matches:
        raise RuntimeError("no restart file for validated cycle %s" % cycle)
    selected = max(matches, key=lambda path: path.stat().st_mtime)
    print("SELECTED_CYCLE=%s JEANS_MIN=%s" % (cycle, safe[-1]["jeans_min"]),
          file=sys.stderr)
    print(selected.resolve())


if __name__ == "__main__":
    main()
