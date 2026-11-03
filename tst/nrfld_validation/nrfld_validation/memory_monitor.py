"""Sample a validation command's process tree and cgroup OOM counters."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time


def _status(pid: int):
    try:
        text = Path(f"/proc/{pid}/status").read_text()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    def value(key, default=0):
        match = re.search(rf"^{key}:\s*(\d+)", text, re.M)
        return int(match[1]) if match else default
    name = re.search(r"^Name:\s*(\S+)", text, re.M)
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except (FileNotFoundError, PermissionError):
        command = ""
    rank = None
    if name and name[1] == "athena":
        try:
            environment = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
            for prefix in (b"OMPI_COMM_WORLD_RANK=", b"PMI_RANK=", b"PMIX_RANK="):
                found = next((item[len(prefix):] for item in environment if item.startswith(prefix)), None)
                if found is not None:
                    rank = int(found)
                    break
        except (FileNotFoundError, PermissionError, ValueError):
            pass
    return {"pid": pid, "ppid": value("PPid"), "name": name[1] if name else "unknown",
            "rss_kib": value("VmRSS"), "hwm_kib": value("VmHWM"),
            "mpi_rank": rank, "command": command[:240]}


def _tree(root_pid: int):
    records = {}
    for item in Path("/proc").iterdir():
        if item.name.isdigit():
            record = _status(int(item.name))
            if record is not None:
                records[record["pid"]] = record
    descendants = {root_pid}
    changed = True
    while changed:
        before = len(descendants)
        descendants.update(pid for pid, item in records.items() if item["ppid"] in descendants)
        changed = len(descendants) != before
    return [records[pid] for pid in descendants if pid in records]


def _cgroup_events():
    path = _cgroup_path() / "memory.events"
    try:
        return {key: int(value) for key, value in
                (line.split() for line in path.read_text().splitlines())}
    except (FileNotFoundError, PermissionError, ValueError):
        return {}


def _cgroup_path() -> Path:
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        hierarchy, _, location = line.partition("::")
        if hierarchy == "0":
            return Path("/sys/fs/cgroup") / location.lstrip("/")
    return Path("/sys/fs/cgroup")


def _machine_memory():
    result = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("MemTotal", "SwapTotal"):
            result[key + "_kib"] = int(value.strip().split()[0])
    for name in ("memory.max", "memory.swap.max"):
        path = _cgroup_path() / name
        if path.is_file():
            result[name] = path.read_text().strip()
    return result


def run_monitored(command: list[str], cwd: Path, log: Path, timeout: float,
                  resource_json: Path, sample_seconds: float = .5) -> dict:
    """Run command and record per-MPI-rank peaks without starting another job."""
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    initial_events = _cgroup_events()
    peak_sum = 0
    peak_athena_sum = 0
    max_rank = {}
    max_process = {}
    samples = 0
    timeout_hit = False
    with log.open("w") as handle:
        process = subprocess.Popen(command, cwd=cwd, stdout=handle,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        while True:
            live = _tree(process.pid)
            if live:
                samples += 1
                peak_sum = max(peak_sum, sum(item["rss_kib"] for item in live))
                peak_athena_sum = max(peak_athena_sum, sum(
                    item["rss_kib"] for item in live if item["name"] == "athena"))
                for item in live:
                    key = str(item["pid"])
                    if item["hwm_kib"] > max_process.get(key, {}).get("hwm_kib", 0):
                        max_process[key] = item
                    if item["name"] == "athena":
                        rank = str(item["mpi_rank"]) if item["mpi_rank"] is not None else f"pid:{item['pid']}"
                        if item["hwm_kib"] > max_rank.get(rank, {}).get("hwm_kib", 0):
                            max_rank[rank] = item
            returncode = process.poll()
            if returncode is not None:
                break
            if time.monotonic()-started > timeout:
                timeout_hit = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                returncode = process.returncode
                break
            time.sleep(sample_seconds)
    final_events = _cgroup_events()
    event_delta = {key: final_events.get(key, 0)-initial_events.get(key, 0)
                   for key in set(initial_events) | set(final_events)}
    oom_killed = event_delta.get("oom_kill", 0) > 0
    report = {"command": command, "started_utc": started_utc,
              "elapsed_seconds": time.monotonic()-started, "returncode": returncode,
              "timeout": timeout_hit, "samples": samples,
              "peak_process_tree_rss_kib": peak_sum,
              "peak_athena_rss_sum_kib": peak_athena_sum,
              "mpi_ranks": max_rank, "max_processes": max_process,
              "cgroup_memory_event_delta": event_delta,
              "machine_memory": _machine_memory(),
              "oom_classification": ("cgroup_oom_kill" if oom_killed else
                                     "possible_sigkill" if returncode in (137, -signal.SIGKILL)
                                     else "none")}
    resource_json.parent.mkdir(parents=True, exist_ok=True)
    resource_json.write_text(json.dumps(report, indent=2) + "\n")
    return report
