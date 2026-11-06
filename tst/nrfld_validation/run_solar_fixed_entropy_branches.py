#!/usr/bin/env python3
"""Run three matched fixed-entropy solar continuations from one checkpoint."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from run_solar_entropy_feedback import digest, memory_status


CASES = (("fixed", 0.0, "0-3"),
         ("plus1e6", 1.0e6, "4-7"),
         ("plus3e6", 3.0e6, "8-11"))
DATA_NAMES = ("AthenaNataEosTable_solar_deep.tab",
              "opacity_table_rhoT_3p0_6p3.h5", "solar_initial_profile.dat")


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def save(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--restart", type=Path, required=True)
    parser.add_argument("--header-from", type=Path, required=True)
    parser.add_argument("--base-input", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-s-in", type=float, required=True)
    parser.add_argument("--end-time", type=float, default=9400.0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.prepare_only and args.resume:
        parser.error("Use either --prepare-only or --resume")
    repo, root = args.repo.resolve(), args.output_root.resolve()
    binary, restart = args.binary.resolve(), args.restart.resolve()
    header, base_input = args.header_from.resolve(), args.base_input.resolve()
    data_dir = args.data_dir.resolve()
    if args.end_time <= 9100 or args.base_s_in <= 0:
        parser.error("Require end time after t=9100 and positive entropy")
    for path in (binary, restart, header, base_input,
                 *(data_dir / name for name in DATA_NAMES)):
        if not path.is_file():
            parser.error(f"Missing input: {path}")
    source_manifest = json.loads((restart.parent / "manifest.json").read_text())
    if source_manifest["binary_sha256"] != digest(binary):
        parser.error("Binary differs from source checkpoint")
    if source_manifest["base_input_sha256"] != digest(base_input):
        parser.error("Campaign input differs from source checkpoint")
    source_entropy = [float(item.split("=", 1)[1])
                      for item in source_manifest["problem_overrides"]
                      if item.startswith("s_in=")]
    if len(source_entropy) != 1 or abs(source_entropy[0]-args.base_s_in) > 1e-3:
        parser.error("Baseline entropy differs from source checkpoint")
    if digest(restart) != json.loads((restart.parent.parent / "state.json").read_text()
                                     )["completed"][-1]["restart_sha256"]:
        parser.error("Source checkpoint digest differs from controller state")
    expected_data = source_manifest["data_sha256"]
    if {name: digest(data_dir / name) for name in DATA_NAMES} != expected_data:
        parser.error("EOS, opacity, or profile differs from source checkpoint")

    config = {"source_restart": str(restart), "source_restart_sha256": digest(restart),
              "binary": str(binary), "binary_sha256": digest(binary),
              "base_input": str(base_input), "base_input_sha256": digest(base_input),
              "data_sha256": expected_data, "header_from": str(header),
              "base_s_in_cgs": args.base_s_in, "start_time_code": 9100.0,
              "end_time_code": args.end_time, "ranks_per_case": 4,
              "cases": [{"name": name, "offset_cgs": offset, "cpus": cpus}
                        for name, offset, cpus in CASES]}
    state_path = root / "state.json"
    if args.resume:
        if not state_path.is_file():
            parser.error("--resume requires an existing state.json")
        state = json.loads(state_path.read_text())
        if state["config"] != config:
            parser.error("Prepared plan changed")
        if state["status"] == "completed":
            print("All branches already completed")
            return
        if state["status"] == "running":
            parser.error("A branch campaign may already be running")
    else:
        if root.exists() and any(root.iterdir()):
            parser.error(f"Refusing nonempty output root: {root}")
        root.mkdir(parents=True, exist_ok=True)
        state = {"status": "prepared", "created_at": now(), "config": config,
                 "cases": {}, "stop_reason": None}
        save(state_path, state)
    (root / "data").mkdir(exist_ok=True)
    for name in DATA_NAMES:
        link = root / "data" / name
        target = (data_dir / name).resolve()
        if link.is_symlink() and link.resolve() == target:
            continue
        if link.exists() or link.is_symlink():
            parser.error(f"Unexpected data link: {link}")
        link.symlink_to(target)
    if args.prepare_only:
        print(state_path)
        return

    stop = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    commands = []
    for name, offset, cpus in CASES:
        entropy = args.base_s_in + offset
        command = ["taskset", "-c", cpus, sys.executable,
                   str(repo / "tst/nrfld_validation/run_solar_convection.py"),
                   "--repo", str(repo), "--binary", str(binary),
                   "--base-input", str(base_input), "--data-dir", str(data_dir),
                   "--output-root", str(root), "--case", name,
                   "--solver", "fgmres", "--cfl", "0.3",
                   "--nx1", "64", "--nx2", "64", "--nx3", "32",
                   "--ranks", "4", "--mpi-bind-to", "hwthread",
                   "--mpi-map-by", "hwthread", "--tlim", f"{args.end_time:.9f}",
                   "--nlim", "10000000", "--checkpoint-dt", "100",
                   "--history-dt", "0.1", "--history-precision", "16",
                   "--restart", str(restart), "--history-header-from", str(header),
                   "--problem-set", "mass_balance_relax_time=2.0",
                   "--problem-set", "mass_balance_rate_gain=40",
                   "--problem-set", "s_in_from_profile=false",
                   "--problem-set", f"s_in={entropy:.15e}"]
        commands.append((name, command))
    state["status"] = "running"
    state["stop_reason"] = None
    state["started_at"] = now()
    save(state_path, state)
    monitor = root / "memory_monitor.csv"
    if not monitor.exists():
        monitor.write_text("utc,mem_available_mib,swap_used_mib,rank_rss_mib,"
                           "observed_ranks,disk_free_gib\n")
    processes = {}
    logs = {}
    try:
        for name, command in commands:
            case = root / name
            if case.is_dir() and (case / "manifest.json").is_file():
                manifest = json.loads((case / "manifest.json").read_text())
                if (manifest.get("exit_code") == 0 and
                        (case / "summary.json").is_file() and
                        len(list(case.glob("*.final.rst"))) == 1):
                    state["cases"][name] = {"status": "completed",
                                            "wall_seconds": manifest["wall_seconds"]}
                    print(f"{now()} reuse completed {name}", flush=True)
                    continue
                raise RuntimeError(f"Incomplete existing case needs manual recovery: {case}")
            log = (root / f"{name}_launcher.log").open("w")
            logs[name] = log
            processes[name] = subprocess.Popen(
                command, cwd=repo, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            state["cases"][name] = {"status": "running", "pid": processes[name].pid,
                                    "entropy_cgs": args.base_s_in +
                                    next(offset for n, offset, _ in CASES if n == name)}
            print(f"{now()} start {name} pid={processes[name].pid}", flush=True)
        save(state_path, state)
        while any(process.poll() is None for process in processes.values()):
            samples = [memory_status(root / name) for name in processes]
            available = min(item["mem_available_mib"] for item in samples)
            swap = max(item["swap_used_mib"] for item in samples)
            rss = sum(item["rank_rss_mib"] for item in samples)
            ranks = sum(item["observed_ranks"] for item in samples)
            disk = shutil.disk_usage(root).free / 1024**3
            with monitor.open("a") as stream:
                stream.write(f"{now()},{available:.1f},{swap:.1f},"
                             f"{rss:.1f},{ranks},{disk:.1f}\n")
            reason = ("stop signal" if stop else
                      "available memory below 3 GiB" if available < 3072 else
                      "swap use above 4 GiB" if swap > 4096 else
                      "disk free below 50 GiB" if disk < 50 else None)
            if reason:
                raise RuntimeError(reason)
            for name, process in processes.items():
                if process.poll() not in (None, 0):
                    raise RuntimeError(f"{name} failed with exit {process.returncode}")
            time.sleep(5)
        for name, process in processes.items():
            if process.returncode != 0:
                raise RuntimeError(f"{name} failed with exit {process.returncode}")
            case = root / name
            report = json.loads((case / "summary.json").read_text())
            manifest = json.loads((case / "manifest.json").read_text())
            log = report["log"]
            if (not log["completed"] or log["fatal_count"] or
                    log["slow_mg_warning_count"] or
                    set(log["nr_status_counts"]) != {"converged"}):
                raise RuntimeError(f"{name} failed solver checks")
            state["cases"][name] = {"status": "completed",
                                    "wall_seconds": manifest["wall_seconds"],
                                    "newton_status_counts": log["nr_status_counts"]}
        state["status"] = "completed"
        state["completed_at"] = now()
        save(state_path, state)
        print(f"{now()} all fixed-entropy branches completed", flush=True)
    except Exception as error:
        for process in processes.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes.values():
            if process.poll() is None:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        state["status"] = "stopped"
        state["stop_reason"] = str(error)
        save(state_path, state)
        print(f"{now()} STOPPED: {error}", file=sys.stderr, flush=True)
        raise
    finally:
        for stream in logs.values():
            stream.close()


if __name__ == "__main__":
    main()
