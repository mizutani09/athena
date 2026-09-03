#!/usr/bin/env python3
"""Run lightweight NRFLD tests and organize downloaded radiative shocks."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
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
    utc_now,
    write_manifest,
)
from plots.plot_couple import coupling_analytic, plot_coupling  # noqa: E402
from plots.plot_diff import plot_diffusion  # noqa: E402
from plots.plot_radiative_shock import plot_radiative_shock  # noqa: E402


DEFAULT_CASES = ("diffusion", "couple")
RADIATIVE_SHOCK_CASES = ("radiative_shock_mach2", "radiative_shock_mach5")
CASES = DEFAULT_CASES + RADIATIVE_SHOCK_CASES


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
        default=max(1, os.cpu_count() or 1),
        help="Upper bound for make jobs and MPI ranks.",
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
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def checked(command: list[str], cwd: Path, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)


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


def build_case(repo: Path, case_dir: Path, problem: str, max_cores: int) -> dict[str, str]:
    configure = configure_command(repo, problem)
    make = ["make", f"-j{max_cores}"]
    checked(configure, repo, case_dir / "configure.log")
    checked(make, repo, case_dir / "build.log")
    return {"configure": command_string(configure), "build": command_string(make)}


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


def run_case(repo: Path, case_dir: Path, case: str, args: argparse.Namespace) -> dict[str, Any]:
    parameters = case_parameters(case, args)
    output_dir = case_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = repo / parameters["input"]
    shutil.copy2(input_path, case_dir / "input.used")
    source_executable = repo / "bin" / "athena"
    executable = case_dir / "athena"
    shutil.copy2(source_executable, executable)
    command = []
    if parameters["ranks"] > 1:
        command.extend(["mpirun", "-np", str(parameters["ranks"])])
    command.extend(
        [str(executable), "-i", str(input_path), "-d", str(output_dir), *parameters["overrides"]]
    )
    checked(command, case_dir, case_dir / "run.log")
    return {
        "run": command_string(command),
        "ranks": parameters["ranks"],
        "input": str(input_path),
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
        destination = case_dir / ("input.used" if source.name.startswith("athinput") else source.name)
        if destination.exists():
            continue
        shutil.copy2(source, destination)
        archived[source.name] = str(destination)
    return archived


def np_all_finite(history: dict[str, Any]) -> bool:
    import numpy as np

    return all(np.all(np.isfinite(values)) for values in history.values())


def main() -> int:
    args = parse_args()
    if args.max_cores < 1:
        raise ValueError("--max-cores must be positive")
    if args.mpi_ranks is not None and args.mpi_ranks < 1:
        raise ValueError("--mpi-ranks must be positive")
    cases = args.cases or list(DEFAULT_CASES)
    cases_requiring_build = [
        case for case in cases
        if case not in RADIATIVE_SHOCK_CASES or args.run_radiative_shock
    ]
    if args.no_build and len(cases_requiring_build) > 1:
        raise ValueError("--no-build requires exactly one --case because cases use different pgens")
    repo = find_repo_root(SCRIPT_DIR)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = args.output_root.expanduser().resolve() / run_name
    suite_dir.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {"run": run_name, "cases": {}, "max_cores": args.max_cores}

    failed = False
    for case in cases:
        case_dir = suite_dir / case
        case_dir.mkdir()
        manifest_path = case_dir / "manifest.json"
        manifest = base_manifest(repo, case, args.max_cores)
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
                manifest["input"] = {"downloaded_run_dir": str(source_dir)}
                manifest["resources"]["mode"] = "plot-only"
                manifest["artifacts"].update(archived)
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
                manifest["metrics"] = metrics
                manifest["artifacts"]["plot"] = str(figure)
                manifest["status"] = "passed" if metrics["passed"] else "failed"
                continue

            parameters = case_parameters(case, args)
            manifest["resources"]["mpi_ranks"] = parameters["ranks"]
            manifest["input"] = {
                "path": str(repo / parameters["input"]),
                "overrides": parameters["overrides"],
            }
            if not args.no_build:
                manifest["commands"].update(
                    build_case(repo, case_dir, parameters["problem"], args.max_cores)
                )
            run_info = run_case(repo, case_dir, case, args)
            manifest["commands"]["run"] = run_info["run"]
            manifest["artifacts"]["executable"] = str(run_info["executable"])
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
        finally:
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
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
