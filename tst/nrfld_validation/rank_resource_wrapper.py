#!/usr/bin/env python3
"""Run one MPI-rank payload and record that rank's child resource usage."""
from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import subprocess
import sys


def mpi_rank() -> int:
    for name in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK"):
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    raise RuntimeError("MPI rank environment variable is unavailable")


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit("usage: rank_resource_wrapper.py OUTPUT_DIR COMMAND [ARGS ...]")
    output = Path(sys.argv[1]).resolve()
    rank = mpi_rank()
    # Slurm PMI2 supplies each rank with an inherited control descriptor.  The
    # Python default (close_fds=True) discards it before an MPI payload can call
    # MPI_Init, producing "PMI_Init returned 14" on XD2000.  Preserve inherited
    # descriptors just as a shell/perf/time launcher would; this wrapper is the
    # direct child of srun and does not open unrelated descriptors first.
    result = subprocess.run(sys.argv[2:], check=False, close_fds=False)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    record = {
        "rank": rank,
        "returncode": result.returncode,
        "max_rss_kib": int(usage.ru_maxrss),
        "user_seconds": usage.ru_utime,
        "system_seconds": usage.ru_stime,
        "minor_page_faults": usage.ru_minflt,
        "major_page_faults": usage.ru_majflt,
        "voluntary_context_switches": usage.ru_nvcsw,
        "involuntary_context_switches": usage.ru_nivcsw,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / f"resource_rank{rank}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
