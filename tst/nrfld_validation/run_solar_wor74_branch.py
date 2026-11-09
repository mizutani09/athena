#!/usr/bin/env python3
"""Run one isolated solar-convection restart branch with resource guards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resources(root: Path) -> tuple[float, float, float]:
    # /proc/meminfo is in KiB. Divide by 1024 for MiB, not GiB.
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, _, raw = line.partition(":")
        if name in {"MemAvailable", "SwapTotal", "SwapFree"}:
            values[name] = int(raw.split()[0]) / 1024
    available_mib = values["MemAvailable"]
    swap_used_mib = values["SwapTotal"] - values["SwapFree"]
    disk_free_gib = shutil.disk_usage(root).free / 1024**3
    return available_mib, swap_used_mib, disk_free_gib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--restart", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--s-in", type=float, required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--tlim", type=float, required=True)
    parser.add_argument("--nproc", type=int, default=16)
    parser.add_argument("--affinity", default="0-15")
    parser.add_argument("--history-dt", type=float, default=0.1)
    parser.add_argument("--full-dt", type=float, default=100)
    parser.add_argument("--max-wall-seconds", type=float, default=7200)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.nproc < 1 or args.s_in <= 0 or args.tlim <= 0 or args.alpha <= 0:
        parser.error("Invalid ranks, entropy, final time, or upper coefficient")
    args.root.mkdir(parents=True, exist_ok=True)
    data_link = args.root / "data"
    if data_link.is_symlink():
        if data_link.resolve() != args.data_dir.resolve():
            parser.error("Existing data link points elsewhere")
    elif not data_link.exists():
        data_link.symlink_to(args.data_dir.resolve(), target_is_directory=True)
    else:
        parser.error("Expected data symlink in the output root")
    case = args.root / args.case
    if case.exists() and any(case.iterdir()):
        parser.error(f"Case directory already contains output: {case}")
    command = ["taskset", "-c", args.affinity, "mpirun",
               "--use-hwthread-cpus", "--bind-to", "hwthread",
               "--map-by", "hwthread", "-np", str(args.nproc),
               str(args.binary.resolve()), "-r", str(args.restart.resolve()),
               "-i", str(args.input.resolve()), "-d", str(case.resolve()),
               "mesh/nx1=64", "mesh/nx2=64", "mesh/nx3=32",
               "time/cfl_number=0.3", f"time/tlim={args.tlim:.15g}",
               "time/nlim=10000000", "time/ncycle_out=1000",
               f"output1/dt={args.history_dt:.15g}",
               f"output6/dt={args.full_dt:.15g}",
               f"output7/dt={args.full_dt:.15g}",
               f"output8/dt={args.full_dt:.15g}",
               "problem/mass_balance_relax_time=2.0",
               "problem/mass_balance_rate_gain=40",
               "problem/s_in_from_profile=false",
               f"problem/s_in={args.s_in:.15e}",
               f"problem/rad_top_alpha={args.alpha:.15g}"]
    manifest = {
        "case": args.case, "command": command, "s_in": args.s_in,
        "rad_top_alpha": args.alpha,
        "binary_sha256": sha256(args.binary),
        "restart_sha256": sha256(args.restart),
        "input_sha256": sha256(args.input),
        "data_dir": str(args.data_dir.resolve()),
        "resource_guards": {"min_available_mib": 3072,
                            "max_swap_used_mib": 4096,
                            "min_disk_free_gib": 50,
                            "max_wall_seconds": args.max_wall_seconds},
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    case.mkdir(exist_ok=True)
    (case / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    start = time.time()
    samples = []
    guard_reason = None
    with (case / "run.log").open("w") as output:
        process = subprocess.Popen(command, cwd=args.root, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True,
                                   env={**os.environ, "OMPI_ALLOW_RUN_AS_ROOT": "1",
                                        "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1"})
        while process.poll() is None:
            available, swap, disk = resources(args.root)
            samples.append((time.time(), available, swap, disk))
            if available < 3072:
                guard_reason = f"Available memory {available:.0f} MiB < 3072 MiB"
            elif swap > 4096:
                guard_reason = f"Swap use {swap:.0f} MiB > 4096 MiB"
            elif disk < 50:
                guard_reason = f"Free disk {disk:.1f} GiB < 50 GiB"
            elif time.time() - start > args.max_wall_seconds:
                guard_reason = f"Wall time exceeded {args.max_wall_seconds:.0f} s"
            if guard_reason:
                os.killpg(process.pid, signal.SIGTERM)
                break
            time.sleep(5)
        exit_code = process.wait()
    with (case / "memory_monitor.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("epoch", "available_mib", "swap_used_mib", "disk_free_gib"))
        writer.writerows(samples)
    manifest.update({"exit_code": exit_code, "wall_seconds": time.time()-start,
                     "guard_reason": guard_reason,
                     "min_available_mib": min(row[1] for row in samples),
                     "max_swap_used_mib": max(row[2] for row in samples)})
    (case / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in
                      ("case", "exit_code", "wall_seconds", "guard_reason",
                       "min_available_mib", "max_swap_used_mib")}))
    if exit_code != 0 or guard_reason:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
