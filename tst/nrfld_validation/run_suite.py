#!/usr/bin/env python3
"""Run lightweight NRFLD tests and organize downloaded radiative shocks."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from nrfld_validation.io import find_repo_root, read_history  # noqa: E402
from nrfld_validation.manifest import (  # noqa: E402
    base_manifest,
    command_string,
    input_file_references,
    sha256_file,
    tool_version,
    tree_sha256,
    utc_now,
    write_manifest,
)
from plots.plot_couple import coupling_analytic, plot_coupling  # noqa: E402
from plots.plot_diff import plot_diffusion  # noqa: E402
from plots.plot_radiative_shock import plot_radiative_shock  # noqa: E402


DEFAULT_CASES = ("diffusion", "couple")
RADIATIVE_SHOCK_CASES = ("radiative_shock_mach2", "radiative_shock_mach5")
CASES = DEFAULT_CASES + RADIATIVE_SHOCK_CASES


class CommandExecutionError(RuntimeError):
    """A command failed or timed out, with a manifest-ready execution record."""

    def __init__(self, message: str, record: dict[str, Any]):
        super().__init__(message)
        self.record = record


def default_max_cores() -> int:
    return min(4, max(1, os.cpu_count() or 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=CASES, dest="cases")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path.home() / "simulation" / "test" / "rad_nr" / "runs",
    )
    parser.add_argument(
        "--max-cores",
        type=int,
        default=default_max_cores(),
        help="Upper bound for make jobs and MPI ranks (default: at most 4).",
    )
    parser.add_argument("--mpi-ranks", type=int, default=None)
    parser.add_argument("--diffusion-resolution", type=int, default=16)
    parser.add_argument("--diffusion-block-size", type=int, default=8)
    parser.add_argument("--couple-nlim", type=int, default=100)
    parser.add_argument(
        "--run-radiative-shock",
        action="store_true",
        help="Build and run selected radiative shocks; otherwise only plot downloaded output.",
    )
    parser.add_argument("--radiative-shock-mach2-dir", type=Path, default=None)
    parser.add_argument("--radiative-shock-mach5-dir", type=Path, default=None)
    parser.add_argument("--radiative-shock-mach2-analytic", type=Path, default=None)
    parser.add_argument("--radiative-shock-mach5-analytic", type=Path, default=None)
    parser.add_argument(
        "--radiative-shock-nlim",
        type=int,
        default=None,
        help="Optional cycle limit for an explicitly requested local shock run.",
    )
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument(
        "--binary",
        type=Path,
        default=None,
        help="Existing binary for --no-build; its compiled pgen is checked with -c.",
    )
    parser.add_argument("--build-timeout", type=float, default=300.0)
    parser.add_argument("--run-timeout", type=float, default=300.0)
    parser.add_argument("--config-timeout", type=float, default=30.0)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def checked(
    command: list[str], cwd: Path, log: Path, timeout: float, phase: str
) -> dict[str, Any]:
    started = utc_now()
    started_monotonic = time.monotonic()
    record: dict[str, Any] = {
        "phase": phase,
        "command": command_string(command),
        "cwd": str(cwd),
        "log": str(log),
        "timeout_seconds": timeout,
        "started_at": started,
        "finished_at": None,
        "elapsed_seconds": None,
        "status": "running",
        "returncode": None,
    }
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        try:
            process = subprocess.run(
                command,
                cwd=cwd,
                text=True,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            record.update(
                {
                    "status": "timeout",
                    "finished_at": utc_now(),
                    "elapsed_seconds": time.monotonic() - started_monotonic,
                }
            )
            raise CommandExecutionError(
                f"{phase} timed out after {timeout:g}s: {command_string(command)}",
                record,
            ) from error
        except OSError as error:
            record.update(
                {
                    "status": "failed_to_start",
                    "finished_at": utc_now(),
                    "elapsed_seconds": time.monotonic() - started_monotonic,
                }
            )
            raise CommandExecutionError(
                f"could not start {phase}: {error}", record
            ) from error
    record.update(
        {
            "status": "passed" if process.returncode == 0 else "failed",
            "returncode": process.returncode,
            "finished_at": utc_now(),
            "elapsed_seconds": time.monotonic() - started_monotonic,
        }
    )
    if process.returncode != 0:
        raise CommandExecutionError(
            f"{phase} failed with exit code {process.returncode}: "
            f"{command_string(command)}",
            record,
        )
    return record


def configure_command(repo: Path, problem: str) -> list[str]:
    return [
        sys.executable,
        str(repo / "configure.py"),
        f"--prob={problem}",
        "-nrmgfld",
        "-hdf5",
        "-mpi",
        "--cflag=-O3",
    ]


def prepare_build_source(repo: Path, build_source: Path) -> None:
    """Copy only the source needed by configure/make, including dirty files."""
    build_source.mkdir(parents=True, exist_ok=False)
    for filename in ("configure.py", "Makefile.in"):
        shutil.copy2(repo / filename, build_source / filename)
    # defs.hpp is a generated configuration file, not part of the source snapshot.
    shutil.copytree(
        repo / "src",
        build_source / "src",
        ignore=shutil.ignore_patterns("defs.hpp"),
    )


def build_case(
    repo: Path,
    case_dir: Path,
    problem: str,
    max_cores: int,
    timeout: float,
    command_records: list[dict[str, Any]],
) -> dict[str, Any]:
    build_source = case_dir / "build_source"
    prepare_build_source(repo, build_source)
    configure = configure_command(repo, problem)
    make = ["make", f"-j{max_cores}"]
    source_sha256 = tree_sha256(build_source)
    configure[1] = str(build_source / "configure.py")
    try:
        configure_record = checked(
            configure, build_source, case_dir / "configure.log", timeout, "configure"
        )
    except CommandExecutionError as error:
        command_records.append(error.record)
        raise
    command_records.append(configure_record)
    try:
        build_record = checked(make, build_source, case_dir / "build.log", timeout, "build")
    except CommandExecutionError as error:
        command_records.append(error.record)
        raise
    command_records.append(build_record)
    return {
        "configure": command_string(configure),
        "build": command_string(make),
        "build_source": build_source,
        "build_source_sha256": source_sha256,
    }


def case_parameters(case: str, args: argparse.Namespace) -> dict[str, Any]:
    if case == "diffusion":
        resolution = args.diffusion_resolution
        block = args.diffusion_block_size
        if resolution % block != 0:
            raise ValueError("diffusion resolution must be divisible by block size")
        blocks = (resolution // block) ** 3
        ranks = min(args.mpi_ranks or args.max_cores, args.max_cores, blocks)
        overrides = [
            f"mesh/nx1={resolution}",
            f"mesh/nx2={resolution}",
            f"mesh/nx3={resolution}",
            f"meshblock/nx1={block}",
            f"meshblock/nx2={block}",
            f"meshblock/nx3={block}",
            "output3/dcycle=92",
            "output4/dcycle=92",
            "output5/dcycle=92",
            "time/nlim=92",
            "time/ncycle_out=92",
            "job/problem_id=nrfld_diffusion_validation",
        ]
        return {
            "problem": "nrfld_diff",
            "input": "inputs/radiation/athinput.nrfld_diff",
            "ranks": max(1, ranks),
            "overrides": overrides,
        }
    if case in RADIATIVE_SHOCK_CASES:
        mach = "mach2" if case.endswith("mach2") else "mach5"
        overrides = [f"job/problem_id=nrfld_radiative_shock_{mach}_validation"]
        if args.radiative_shock_nlim is not None:
            overrides.extend(
                [
                    f"time/nlim={args.radiative_shock_nlim}",
                    f"time/ncycle_out={args.radiative_shock_nlim}",
                ]
            )
        return {
            "problem": "nrfld_radiative_shock",
            "input": f"inputs/radiation/athinput.nrfld_radiative_shock_{mach}",
            "ranks": min(args.mpi_ranks or args.max_cores, args.max_cores),
            "overrides": overrides,
        }
    ranks = 1  # The canonical coupling problem contains one MeshBlock.
    overrides = [
        f"time/nlim={args.couple_nlim}",
        f"time/ncycle_out={args.couple_nlim}",
        "job/problem_id=nrfld_couple_validation",
    ]
    return {
        "problem": "nrfld_couple",
        "input": "inputs/radiation/athinput.nrfld_couple",
        "ranks": ranks,
        "overrides": overrides,
    }


def binary_configuration(
    executable: Path,
    case_dir: Path,
    timeout: float,
    command_records: list[dict[str, Any]],
) -> dict[str, str]:
    command = [str(executable), "-c"]
    try:
        record = checked(command, case_dir, case_dir / "config.log", timeout, "binary_config")
    except CommandExecutionError as error:
        command_records.append(error.record)
        raise
    command_records.append(record)
    configuration: dict[str, str] = {}
    for line in (case_dir / "config.log").read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.strip().split(":", 1)
        configuration[key.strip().lower().replace(" ", "_")] = value.strip()
    return configuration


def validate_binary_configuration(
    configuration: dict[str, str], expected_problem: str, executable: Path
) -> None:
    expected = {
        "problem_generator": expected_problem,
        "coordinate_system": "cartesian",
        "equation_of_state": "adiabatic",
        "riemann_solver": "hllc_fld",
        "fld_with_newton-raphson": "ON",
        "mpi_parallelism": "ON",
        "hdf5_output": "ON",
    }
    mismatches = {
        key: {"expected": value, "actual": configuration.get(key)}
        for key, value in expected.items()
        if configuration.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"binary configuration mismatch for {executable}: {mismatches}"
        )


def effective_settings(input_path: Path, overrides: list[str]) -> dict[str, Any]:
    settings: dict[str, str] = {}
    section = ""
    for raw_line in input_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("<") and line.endswith(">"):
            section = line[1:-1].strip()
        elif "=" in line:
            key, value = [part.strip() for part in line.split("=", 1)]
            settings[f"{section}/{key}" if section else key] = value
    for override in overrides:
        key, value = override.split("=", 1)
        settings[key] = value
    return settings


def reached_cycle(output_dir: Path, log: Path | None = None) -> int | None:
    if log is not None and log.is_file():
        import re

        matches = re.findall(r"(?:^|\s)cycle=(\d+)", log.read_text(encoding="utf-8"))
        if matches:
            return int(matches[-1])
    try:
        history = read_history(output_dir)
    except (FileNotFoundError, OSError, ValueError):
        return None
    for key in ("ncycle", "cycle", "cycle_number"):
        if key in history and len(history[key]):
            return int(history[key][-1])
    return None


def reached_time(output_dir: Path, log: Path | None = None) -> float | None:
    if log is not None and log.is_file():
        import re

        matches = re.findall(
            r"(?:^|\s)time=([0-9.eE+-]+)", log.read_text(encoding="utf-8")
        )
        if matches:
            return float(matches[-1])
    try:
        history = read_history(output_dir)
    except (FileNotFoundError, OSError, ValueError):
        return None
    if "time" in history and len(history["time"]):
        return float(history["time"][-1])
    return None


def run_case(
    repo: Path,
    case_dir: Path,
    case: str,
    args: argparse.Namespace,
    executable: Path,
    timeout: float,
    command_records: list[dict[str, Any]],
) -> dict[str, Any]:
    parameters = case_parameters(case, args)
    output_dir = case_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = repo / parameters["input"]
    shutil.copy2(input_path, case_dir / "input.used")
    command = []
    if parameters["ranks"] > 1:
        command.extend(["mpirun", "-np", str(parameters["ranks"])])
    command.extend(
        [
            str(executable),
            "-i",
            str(case_dir / "input.used"),
            "-d",
            str(output_dir),
            *parameters["overrides"],
        ]
    )
    try:
        record = checked(command, case_dir, case_dir / "run.log", timeout, "run")
    except CommandExecutionError as error:
        command_records.append(error.record)
        raise
    command_records.append(record)
    return {
        "run": command_string(command),
        "ranks": parameters["ranks"],
        "input": str(case_dir / "input.used"),
        "overrides": parameters["overrides"],
        "output_dir": output_dir,
        "executable": executable,
    }


def analyse_case(case: str, output_dir: Path) -> dict[str, float | bool]:
    history = read_history(output_dir)
    if case == "diffusion":
        l1 = float(history["L1norm"][-1])
        relative = float(history["L1norm_rel"][-1])
        finite = bool(np_all_finite(history))
        return {
            "final_l1": l1,
            "final_relative_l1": relative,
            "relative_l1_limit": 1.0e-2,
            "finite": finite,
            "passed": finite and relative < 1.0e-2,
        }
    import numpy as np

    analytic_time, analytic_egas = coupling_analytic("low")
    valid = (history["Rtime"] > 0.0) & (history["e_gas"] > 0.0)
    analytic_log_egas = np.interp(
        np.log(history["Rtime"][valid]), np.log(analytic_time), np.log(analytic_egas)
    )
    log_error = np.abs(np.log(history["e_gas"][valid]) - analytic_log_egas)
    max_log_error = float(np.max(log_error)) if log_error.size else float("inf")
    finite = bool(np_all_finite(history) and np.isfinite(max_log_error))
    return {
        "final_egas": float(history["e_gas"][-1]),
        "final_erad": float(history["E_rad"][-1]),
        "max_log_egas_error": max_log_error,
        "max_log_egas_error_limit": 5.0e-3,
        "finite": finite,
        "passed": bool(finite and max_log_error < 5.0e-3),
    }


def shock_case_options(
    case: str, args: argparse.Namespace
) -> tuple[str, Path | None, Path | None]:
    if case == "radiative_shock_mach2":
        return "mach2", args.radiative_shock_mach2_dir, args.radiative_shock_mach2_analytic
    return "mach5", args.radiative_shock_mach5_dir, args.radiative_shock_mach5_analytic


def copy_download_metadata(source_dir: Path, case_dir: Path) -> dict[str, str]:
    """Archive small reproducibility files without duplicating heavy HDF5 output."""
    archived: dict[str, str] = {}
    candidates = [source_dir / "athena", source_dir / "problem_parameters.txt"]
    candidates.extend(sorted(source_dir.glob("athinput*")))
    for source in candidates:
        if not source.is_file():
            continue
        destination = case_dir / (
            "input.used" if source.name.startswith("athinput") else source.name
        )
        if destination.exists():
            continue
        shutil.copy2(source, destination)
        archived[source.name] = str(destination)
    return archived


def configured_compiler(build_source: Path | None, configuration: dict[str, str]) -> str | None:
    if build_source is not None:
        makefile = build_source / "Makefile"
        if makefile.is_file():
            for line in makefile.read_text(encoding="utf-8").splitlines():
                if line.startswith("CXX :="):
                    return line.split(":=", 1)[1].strip().split()[0]
    command = configuration.get("compilation_command", "")
    return command.split()[0] if command else None


def update_toolchain(
    manifest: dict[str, Any], build_source: Path | None, configuration: dict[str, str]
) -> None:
    compiler = configured_compiler(build_source, configuration)
    manifest["toolchain"] = {
        "compiler": tool_version(compiler),
        "mpi_launcher": tool_version("mpirun"),
        "binary_configuration": configuration,
    }


def downloaded_calculation_provenance(source_dir: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source": "downloaded_run",
        "run_directory": str(source_dir),
        "manifest": None,
        "git": None,
    }
    downloaded_binary = source_dir / "athena"
    if downloaded_binary.is_file():
        metadata["binary"] = {
            "path": str(downloaded_binary),
            "sha256": sha256_file(downloaded_binary),
        }
    source_manifest = source_dir / "manifest.json"
    if not source_manifest.is_file():
        return metadata
    metadata["manifest"] = {
        "path": str(source_manifest),
        "sha256": sha256_file(source_manifest),
    }
    try:
        downloaded = json.loads(source_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return metadata
    provenance = downloaded.get("provenance", {})
    metadata["git"] = (
        provenance.get("calculation", {}).get("git")
        if isinstance(provenance.get("calculation"), dict)
        else downloaded.get("git")
    )
    return metadata


def error_reason(error: Exception) -> str:
    if isinstance(error, CommandExecutionError):
        return str(error.record.get("status", "failed"))
    if isinstance(error, FileNotFoundError):
        return "reference_missing"
    return "failed"


def np_all_finite(history: dict[str, Any]) -> bool:
    import numpy as np

    return all(np.all(np.isfinite(values)) for values in history.values())


def main() -> int:
    args = parse_args()
    if args.max_cores < 1:
        raise ValueError("--max-cores must be positive")
    if args.mpi_ranks is not None and args.mpi_ranks < 1:
        raise ValueError("--mpi-ranks must be positive")
    if args.build_timeout <= 0 or args.run_timeout <= 0 or args.config_timeout <= 0:
        raise ValueError("command timeouts must be positive")
    cases = args.cases or list(DEFAULT_CASES)
    cases_requiring_build = [
        case for case in cases
        if case not in RADIATIVE_SHOCK_CASES or args.run_radiative_shock
    ]
    if args.no_build and len(cases_requiring_build) > 1:
        raise ValueError(
            "--no-build requires exactly one --case because cases use different pgens"
        )
    repo = find_repo_root(SCRIPT_DIR)
    if args.no_build and cases_requiring_build and args.binary is not None:
        args.binary = args.binary.expanduser().resolve()
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = args.output_root.expanduser().resolve() / run_name
    suite_dir.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {
        "run": run_name,
        "cases": {},
        "max_cores": args.max_cores,
        "started_at": utc_now(),
    }

    failed = False
    for case in cases:
        case_dir = suite_dir / case
        case_dir.mkdir()
        manifest_path = case_dir / "manifest.json"
        manifest = base_manifest(repo, case, args.max_cores)
        command_records: list[dict[str, Any]] = []
        manifest["commands"]["records"] = command_records
        write_manifest(manifest_path, manifest)
        try:
            if case in RADIATIVE_SHOCK_CASES and not args.run_radiative_shock:
                mach, source_dir, analytic = shock_case_options(case, args)
                if source_dir is None:
                    raise ValueError(
                        f"--radiative-shock-{mach}-dir is required unless "
                        "--run-radiative-shock is specified"
                    )
                source_dir = source_dir.expanduser().resolve()
                if not source_dir.is_dir():
                    raise FileNotFoundError(source_dir)
                archived = copy_download_metadata(source_dir, case_dir)
                manifest["input"] = {
                    "downloaded_run_dir": str(source_dir),
                    "metadata_files": [
                        {
                            "path": str(source_dir / name),
                            "sha256": sha256_file(source_dir / name),
                        }
                        for name in ("athena", "problem_parameters.txt")
                        if (source_dir / name).is_file()
                    ],
                }
                manifest["resources"]["mode"] = "plot-only"
                manifest["provenance"]["calculation"] = downloaded_calculation_provenance(
                    source_dir
                )
                manifest["artifacts"].update(archived)
                manifest["artifacts"]["metadata_sha256"] = {
                    name: sha256_file(source_dir / name)
                    for name in archived
                    if (source_dir / name).is_file()
                }
                figure = case_dir / "figures" / f"{case}.png"
                metrics = plot_radiative_shock(
                    source_dir,
                    figure,
                    mach,
                    analytic_path=analytic,
                )
                analytic_file = Path(str(metrics["analytic_file"]))
                if str(metrics["analytic_file"]) and analytic_file.is_file():
                    reference_dir = case_dir / "reference"
                    reference_dir.mkdir()
                    archived_analytic = reference_dir / analytic_file.name
                    shutil.copy2(analytic_file, archived_analytic)
                    manifest["artifacts"]["analytic_profile"] = str(archived_analytic)
                    manifest["artifacts"]["analytic_profile_sha256"] = sha256_file(
                        analytic_file
                    )
                manifest["metrics"] = metrics
                manifest["artifacts"]["plot"] = str(figure)
                manifest["status"] = "passed" if metrics["passed"] else "failed"
                manifest["termination"] = {
                    "reason": "completed" if metrics["passed"] else "validation_failed",
                    "reached_cycle": None,
                    "reached_time": None,
                }
                continue

            parameters = case_parameters(case, args)
            manifest["resources"]["mpi_ranks"] = parameters["ranks"]
            omp_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
            input_source = repo / parameters["input"]
            input_hash = sha256_file(input_source)
            manifest["input"] = {
                "source_path": str(input_source),
                "sha256": input_hash,
                "overrides": parameters["overrides"],
                "effective_settings": effective_settings(input_source, parameters["overrides"]),
                "external_files": input_file_references(input_source, repo),
            }
            manifest["resources"]["omp_threads"] = omp_threads
            manifest["resources"]["mode"] = "no-build" if args.no_build else "isolated-build"
            build_source: Path | None = None
            if not args.no_build:
                build_info = build_case(
                    repo,
                    case_dir,
                    parameters["problem"],
                    args.max_cores,
                    args.build_timeout,
                    command_records,
                )
                build_source = build_info["build_source"]
                manifest["commands"].update(
                    {
                        "configure": build_info["configure"],
                        "build": build_info["build"],
                    }
                )
                executable = build_source / "bin" / "athena"
                manifest["provenance"]["calculation"] = {
                    "source": "local_isolated_build",
                    "repository": str(repo),
                    "git": manifest["git"],
                    "build_source": str(build_source),
                    "source_sha256": build_info["build_source_sha256"],
                    "configured_files": {
                        name: {
                            "path": str(build_source / name),
                            "sha256": sha256_file(build_source / name),
                        }
                        for name in ("Makefile", "src/defs.hpp", "configure.log")
                    },
                }
            else:
                executable = args.binary or (repo / "bin" / "athena")
                executable = executable.expanduser().resolve()
                if not executable.is_file():
                    raise FileNotFoundError(
                        f"--no-build binary does not exist: {executable}"
                    )
                manifest["provenance"]["calculation"] = {
                    "source": "existing_binary",
                    "path": str(executable),
                    "sha256": sha256_file(executable),
                    "git": None,
                }
            configuration = binary_configuration(
                executable, case_dir, args.config_timeout, command_records
            )
            validate_binary_configuration(configuration, parameters["problem"], executable)
            if (
                configuration.get("openmp_parallelism") == "ON"
                and parameters["ranks"] * omp_threads > args.max_cores
            ):
                raise ValueError(
                    "MPI ranks multiplied by OMP_NUM_THREADS exceeds --max-cores: "
                    f"{parameters['ranks']} * {omp_threads} > {args.max_cores}"
                )
            update_toolchain(manifest, build_source, configuration)
            manifest["provenance"]["calculation"]["binary_configuration"] = configuration
            manifest["provenance"]["calculation"]["binary_sha256"] = sha256_file(executable)
            run_info = run_case(
                repo, case_dir, case, args, executable, args.run_timeout, command_records
            )
            manifest["commands"]["run"] = run_info["run"]
            manifest["artifacts"]["executable"] = {
                "path": str(run_info["executable"]),
                "sha256": sha256_file(run_info["executable"]),
            }
            manifest["artifacts"]["input.used"] = {
                "path": str(case_dir / "input.used"),
                "sha256": sha256_file(case_dir / "input.used"),
            }
            if case in RADIATIVE_SHOCK_CASES:
                mach, _, analytic = shock_case_options(case, args)
                figure = case_dir / "figures" / f"{case}.png"
                metrics = plot_radiative_shock(
                    case_dir,
                    figure,
                    mach,
                    analytic_path=analytic,
                    output_dir=run_info["output_dir"],
                )
            else:
                metrics = analyse_case(case, run_info["output_dir"])
            manifest["metrics"] = metrics
            manifest["input"]["used_path"] = str(case_dir / "input.used")
            manifest["input"]["used_sha256"] = sha256_file(case_dir / "input.used")
            manifest["input"]["external_files"] = input_file_references(
                case_dir / "input.used", repo
            )
            if not args.no_plots:
                figure = case_dir / "figures" / f"{case}.png"
                if case == "diffusion":
                    plot_diffusion(run_info["output_dir"], figure)
                elif case == "couple":
                    plot_coupling(run_info["output_dir"], figure)
                elif not figure.is_file():
                    mach, _, analytic = shock_case_options(case, args)
                    plot_radiative_shock(
                        case_dir, figure, mach, analytic_path=analytic,
                        output_dir=run_info["output_dir"],
                    )
                manifest["artifacts"]["plot"] = str(figure)
            manifest["status"] = "passed" if metrics["passed"] else "failed"
        except Exception as error:  # Keep metadata for failed external runs.
            failed = True
            manifest["status"] = "failed"
            manifest["error"] = {"message": str(error), "traceback": traceback.format_exc()}
            if isinstance(error, CommandExecutionError) and error.record not in command_records:
                command_records.append(error.record)
            manifest["termination"] = {
                "reason": error_reason(error),
                "reached_cycle": reached_cycle(case_dir / "output", case_dir / "run.log"),
                "reached_time": reached_time(case_dir / "output", case_dir / "run.log"),
            }
        finally:
            if manifest["status"] == "created":
                manifest["status"] = "failed"
                manifest["termination"] = {
                    "reason": "failed",
                    "reached_cycle": reached_cycle(case_dir / "output", case_dir / "run.log"),
                    "reached_time": reached_time(case_dir / "output", case_dir / "run.log"),
                }
            manifest["commands"]["records"] = command_records
            if manifest["termination"]["reason"] == "created":
                manifest["termination"] = {
                    "reason": (
                        "completed" if manifest["status"] == "passed" else "validation_failed"
                    ),
                    "reached_cycle": reached_cycle(case_dir / "output", case_dir / "run.log"),
                    "reached_time": reached_time(case_dir / "output", case_dir / "run.log"),
                }
            manifest["finished_at"] = utc_now()
            write_manifest(manifest_path, manifest)
            summary["cases"][case] = {
                "status": manifest["status"],
                "metrics": manifest.get("metrics", {}),
                "manifest": str(manifest_path),
            }
            if manifest["status"] != "passed":
                failed = True

    (suite_dir / "summary.json").write_text(
        json.dumps({**summary, "finished_at": utc_now()}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
