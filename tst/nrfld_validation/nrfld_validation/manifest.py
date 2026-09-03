"""Reproducibility metadata for NRFLD validation runs."""

from __future__ import annotations

import json
import os
import platform
import shlex
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def command_string(command: Sequence[str]) -> str:
    return shlex.join(str(item) for item in command)


def git_metadata(repo: Path) -> dict[str, Any]:
    def capture(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=repo, check=True, text=True, capture_output=True
        )
        return result.stdout.strip()

    try:
        status = capture("status", "--porcelain")
        return {
            "commit": capture("rev-parse", "HEAD"),
            "branch": capture("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status),
            "status": status.splitlines(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None, "status": []}


def base_manifest(repo: Path, case: str, max_cores: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "case": case,
        "status": "created",
        "started_at": utc_now(),
        "finished_at": None,
        "repository": str(repo),
        "git": git_metadata(repo),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "resources": {"max_cores": max_cores},
        "commands": {},
        "input": {},
        "metrics": {},
        "artifacts": {},
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)

