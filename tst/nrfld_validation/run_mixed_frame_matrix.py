#!/usr/bin/env python3
"""Run the mixed-frame ON/OFF matrix, then build its HTML report.

All runs are sequential to bound WSL memory use. Each run stores its effective
input, full stdout and per-launcher resource report under the output root.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from advection_pulse_validation import overrides as pulse_overrides
from diffusion_validation import materialize_input
from mixed_frame_validation import PULSE_MATRIX, SHOCK_MATRIX
from nrfld_validation.io import find_repo_root
from radiative_shock_validation import input_overrides as shock_overrides
from run_suite import prepare_build_source


def case_matrix(preset: str):
    for mach in (2, 5):
        for n, cfl in SHOCK_MATRIX[preset]:
            for implicit in (True, False):
                yield (f"shock{mach}_n{n}_cfl{cfl:g}_{'implicit' if implicit else 'explicit'}",
                       "shock", mach, n, cfl, implicit)
    for n, cfl in PULSE_MATRIX[preset]:
        yield (f"pulse_n{n}_cfl{cfl:g}_static_implicit", "pulse", "static", n, cfl, True)
        for implicit in (True, False):
            yield (f"pulse_n{n}_cfl{cfl:g}_moving_{'implicit' if implicit else 'explicit'}",
                   "pulse", "moving", n, cfl, implicit)
    for implicit in (True, False):
        yield ("box_" + ("implicit" if implicit else "explicit"),
               "box", "closed_box", 8, .2, implicit)


def build(repo: Path, root: Path, kind: str, jobs: int, timeout: float) -> Path:
    target = root / f"{kind}_build_source"
    prepare_build_source(repo, target)
    problem = "nrfld_radiative_shock" if kind == "shock" else "nrfld_advect_radiation_pulse"
    commands = ([sys.executable, str(target / "configure.py"), f"--prob={problem}",
                 "--eos=adiabatic", "-nrmgfld", "-hdf5", "-mpi", "--cflag=-O3"],
                ["make", f"-j{jobs}"])
    for name, command in zip(("configure", "build"), commands):
        with (root / f"{kind}_{name}.log").open("w") as log:
            result = subprocess.run(command, cwd=target, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=timeout)
        if result.returncode:
            raise RuntimeError(f"{kind} {name} failed with {result.returncode}")
    return target / "bin/athena"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/d/output/rad"))
    parser.add_argument("--run-name", default="mixed_frame_compare_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--preset", choices=tuple(SHOCK_MATRIX), default="full")
    parser.add_argument("--mpi-ranks", type=int, default=4)
    parser.add_argument("--build-jobs", type=int, default=8)
    parser.add_argument("--build-timeout", type=float, default=900)
    parser.add_argument("--run-timeout", type=float, default=21600)
    parser.add_argument("--reuse-pulse-build", type=Path)
    parser.add_argument("--reuse-shock-build", type=Path)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if min(args.mpi_ranks, args.build_jobs, args.build_timeout, args.run_timeout) <= 0:
        parser.error("MPI ranks, build jobs and timeouts must be positive")
    if args.plan:
        for tag, *_ in case_matrix(args.preset):
            print(tag)
        print("jacobian_implicit")
        print("jacobian_explicit")
        return
    repo = find_repo_root(HERE)
    root = args.output_root.expanduser().resolve() / args.run_name
    root.mkdir(parents=True, exist_ok=False)
    binaries = {}
    for kind, reuse in (("shock", args.reuse_shock_build), ("pulse", args.reuse_pulse_build)):
        if reuse:
            source = reuse.expanduser().resolve()
            target = root / f"{kind}_build_source"
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("*.o"))
            binaries[kind] = target / "bin/athena"
        else:
            binaries[kind] = build(repo, root, kind, args.build_jobs, args.build_timeout)
    for tag, kind, subtype, n, cfl, implicit in case_matrix(args.preset):
        directory = root / tag
        directory.mkdir()
        (directory / "output").mkdir()
        if kind == "shock":
            template = repo / "inputs/radiation" / f"athinput.nrfld_radiative_shock_mach{subtype}"
            updates = shock_overrides(f"mach{subtype}", n, 4, None, "uniform")
            updates += [f"time/cfl_number={cfl:.17g}", f"job/problem_id={tag}"]
            executable = binaries["shock"]
        elif kind == "pulse":
            template = repo / "inputs/radiation/athinput.nrfld_advect_pulse"
            updates = pulse_overrides(subtype, n, cfl)
            updates += ["nrfld/fgmres_threshold=1e-8", f"job/problem_id={tag}"]
            executable = binaries["pulse"]
        else:
            template = repo / "inputs/radiation/athinput.nrfld_closed_box"
            updates = ["time/nlim=-1", "time/tlim=1", "time/cfl_number=0.2",
                       "problem/v0=1000000", f"job/problem_id={tag}"]
            executable = binaries["pulse"]
        updates += [f"fld/implicit_mixed_frame_drift={str(implicit).lower()}"]
        materialize_input(template, directory / "input.used", updates)
        command = ["/usr/bin/time", "-v", "-o", str(directory / "resources.txt"),
                   "mpirun", "-np", str(1 if kind == "box" else args.mpi_ranks), str(executable),
                   "-i", str(directory / "input.used"), "-d", str(directory / "output")]
        with (directory / "run.log").open("w") as log:
            result = subprocess.run(command, cwd=directory, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=args.run_timeout)
        print(tag, result.returncode, flush=True)
        text = (directory / "run.log").read_text(errors="replace")
        if result.returncode or "Terminating on time limit" not in text or "FATAL ERROR" in text:
            raise RuntimeError(f"{tag} failed; inspect {directory / 'run.log'}")
    for implicit in (True, False):
        tag = "jacobian_" + ("implicit" if implicit else "explicit")
        directory = root / tag
        directory.mkdir()
        (directory / "output").mkdir()
        updates = pulse_overrides("moving", 32, .03) + [
            "time/nlim=1", "time/ncycle_out=1", "output3/dt=-1", "output3/dcycle=1",
            "nrfld/fgmres_threshold=1e-8", "nrfld/jacobian_check=true",
            "nrfld/jacobian_check_epsilon=1e-4",
            "fld/mixed_frame_ledger_limit=2",
            f"fld/implicit_mixed_frame_drift={str(implicit).lower()}",
            f"job/problem_id={tag}"]
        materialize_input(repo / "inputs/radiation/athinput.nrfld_advect_pulse",
                          directory / "input.used", updates)
        command = ["/usr/bin/time", "-v", "-o", str(directory / "resources.txt"),
                   "mpirun", "-np", "1", str(binaries["pulse"]),
                   "-i", str(directory / "input.used"), "-d", str(directory / "output")]
        with (directory / "run.log").open("w") as log:
            result = subprocess.run(command, cwd=directory, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=args.run_timeout)
        text = (directory / "run.log").read_text(errors="replace")
        print(tag, result.returncode, flush=True)
        if result.returncode or "Terminating on cycle limit" not in text or "FATAL ERROR" in text:
            raise RuntimeError(f"{tag} failed; inspect {directory / 'run.log'}")
    subprocess.run([sys.executable, str(HERE / "mixed_frame_validation.py"),
                    "--postprocess", str(root), "--preset", args.preset], check=True)


if __name__ == "__main__":
    main()
