"""Reproducibility metadata for NRFLD validation runs."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
import platform
import shlex
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any, Sequence


PROVENANCE_ENVIRONMENT_KEYS = (
    "ATHENA_FLD_PRESSURE_IN_FLUX",
    "CC",
    "CXX",
    "MPICXX",
    "HDF5_ROOT",
    "HDF5_DIR",
    "OMP_NUM_THREADS",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "HDF5_USE_FILE_LOCKING",
    "OMPI_MCA_btl",
    "OMPI_MCA_pml",
    "MPICH_ASYNC_PROGRESS",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def command_string(command: Sequence[str]) -> str:
    return shlex.join(str(item) for item in command)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str:
    """Hash a source snapshot without depending on its absolute location."""
    digest = sha256()
    for entry in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = entry.relative_to(path).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(entry)))
    return digest.hexdigest()


def environment_metadata() -> dict[str, str]:
    return {
        key: os.environ[key]
        for key in PROVENANCE_ENVIRONMENT_KEYS
        if key in os.environ
    }


def tool_version(command: str | None, timeout: float = 10.0) -> dict[str, Any]:
    if not command:
        return {"command": None, "path": None, "version": None}
    executable = shutil.which(command)
    result: dict[str, Any] = {
        "command": command,
        "path": executable,
        "version": None,
    }
    if executable is None:
        return result
    try:
        completed = subprocess.run(
            [executable, "--version"],
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        output = (completed.stdout or completed.stderr).strip()
        result["version"] = output.splitlines()[0] if output else None
        result["returncode"] = completed.returncode
    except (OSError, subprocess.TimeoutExpired) as error:
        result["error"] = str(error)
    return result

def _input_path_candidates(raw_value: str, input_path: Path, repo: Path) -> list[Path]:
    value = raw_value.strip().strip("\"'")
    if not value or value.lower() in {"true", "false", "none", "auto", "ascii", "hdf5"}:
        return []
    candidates = [Path(value).expanduser()]
    if not Path(value).is_absolute():
        candidates.extend((input_path.parent / value, repo / value))
    return candidates


def input_file_references(input_path: Path, repo: Path) -> list[dict[str, Any]]:
    """Return hashes for likely external files, including missing references."""
    references: list[dict[str, Any]] = []
    with input_path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            content = raw_line.split("#", 1)[0].strip()
            if "=" not in content:
                continue
            key, raw_value = [part.strip() for part in content.split("=", 1)]
            normalized_key = key.lower()
            if (
                normalized_key == "file_type"
                or normalized_key.endswith(("_file_type", "_dataset", "_axis", "_type"))
                or normalized_key.startswith("const_opacity")
                or normalized_key.endswith("_coeff")
                or not (
                    normalized_key.endswith(("_file", "_path"))
                    or "profile" in normalized_key
                    or "opacity_table_file" in normalized_key
                    or "eos_file" in normalized_key
                )
            ):
                continue
            candidates = _input_path_candidates(raw_value, input_path, repo)
            if not candidates:
                continue
            resolved = next(
                (candidate.resolve() for candidate in candidates if candidate.is_file()), None
            )
            item: dict[str, Any] = {
                "key": key,
                "line": line_number,
                "configured": raw_value,
                "path": str(resolved or candidates[0].resolve()),
                "exists": resolved is not None,
                "sha256": sha256_file(resolved) if resolved is not None else None,
            }
            references.append(item)
    return references


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
    git = git_metadata(repo)
    return {
        "schema_version": 2,
        "case": case,
        "status": "created",
        "started_at": utc_now(),
        "finished_at": None,
        "repository": str(repo),
        "git": git,
        "provenance": {
            "analysis": {"repository": str(repo), "git": git},
            "calculation": None,
        },
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "environment": environment_metadata(),
        "toolchain": {
            "compiler": tool_version(os.environ.get("CXX", "g++")),
            "mpi_launcher": tool_version("mpirun"),
            "binary_configuration": None,
        },
        "resources": {"max_cores": max_cores},
        "commands": {},
        "input": {},
        "metrics": {},
        "artifacts": {},
        "termination": {
            "reason": "created",
            "reached_cycle": None,
            "reached_time": None,
        },
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
