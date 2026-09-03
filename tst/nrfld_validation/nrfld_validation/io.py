"""Common readers used by NRFLD validation and plotting scripts."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np


def find_repo_root(start: Path | None = None) -> Path:
    path = (start or Path(__file__)).resolve()
    for candidate in (path, *path.parents):
        if (candidate / "configure.py").is_file() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("could not locate the Athena++ repository root")


def find_files(directory: Path, suffix: str) -> list[Path]:
    files = sorted(directory.glob(f"*{suffix}"))
    if not files:
        raise FileNotFoundError(f"no *{suffix} files found in {directory}")
    return files


def latest_athdf(directory: Path, output_id: str | None = None) -> Path:
    files = find_files(directory, ".athdf")
    if output_id is not None:
        files = [path for path in files if f".{output_id}." in path.name]
    if not files:
        raise FileNotFoundError(f"no athdf output with id={output_id!r} in {directory}")
    return files[-1]


def read_athdf(path: Path, num_ghost: int = 0) -> dict:
    repo = find_repo_root()
    vis_python = str(repo / "vis" / "python")
    if vis_python not in sys.path:
        sys.path.insert(0, vis_python)
    import athena_read  # pylint: disable=import-outside-toplevel

    return athena_read.athdf(str(path), num_ghost=num_ghost)


def history_labels(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        lines = [handle.readline(), handle.readline()]
    if len(lines) < 2 or not lines[1]:
        raise ValueError(f"invalid Athena++ history header: {path}")
    return re.findall(r"\[\d+\]=([^\s]+)", lines[1])


def read_history(path_or_directory: Path) -> dict[str, np.ndarray]:
    path = path_or_directory
    if path.is_dir():
        path = find_files(path, ".hst")[0]
    labels = history_labels(path)
    values = np.loadtxt(path, comments="#", ndmin=2)
    if values.shape[1] != len(labels):
        raise ValueError(
            f"history column mismatch in {path}: {values.shape[1]} data columns, "
            f"{len(labels)} labels"
        )
    return {label: values[:, index] for index, label in enumerate(labels)}


def parse_key_value_file(path: Path) -> dict[str, float | str]:
    values: dict[str, float | str] = {}
    if not path.exists():
        return values
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith(("#", ">>>", "-")) or "=" not in line:
                continue
            key, value = [part.strip() for part in line.split("=", 1)]
            token = value.split()[0]
            try:
                values[key] = float(token)
            except ValueError:
                values[key] = token
    return values

