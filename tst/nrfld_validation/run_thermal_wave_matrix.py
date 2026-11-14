#!/usr/bin/env python3
"""Sequential 3-D thermal-wave uniform/AMR validation with an isolated build."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from diffusion_validation import materialize_input
from nrfld_validation.io import find_repo_root
from run_suite import prepare_build_source
from thermal_wave_validation import CASE_SETS

def cases_for(preset: str):
    return tuple((tag, int(tag.split("_")[0].removeprefix("uniform").removeprefix("amr")),
                  tag.startswith("amr")) for tag in CASE_SETS[preset])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/d/output/rad"))
    parser.add_argument("--run-name", default="thermal_wave_validation_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--preset", choices=tuple(CASE_SETS), default="full")
    parser.add_argument("--mpi-ranks", type=int, default=4)
    parser.add_argument("--build-jobs", type=int, default=8)
    parser.add_argument("--build-timeout", type=float, default=900)
    parser.add_argument("--run-timeout", type=float, default=21600)
    parser.add_argument("--reuse-build", type=Path)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if min(args.mpi_ranks, args.build_jobs, args.build_timeout, args.run_timeout) <= 0:
        parser.error("MPI ranks, build jobs and timeouts must be positive")
    if args.plan:
        for tag, _, _ in cases_for(args.preset):
            print(tag)
        return
    repo = find_repo_root(HERE)
    root = args.output_root.expanduser().resolve() / args.run_name
    root.mkdir(parents=True, exist_ok=False)
    source = root / "build_source"
    if args.reuse_build:
        shutil.copytree(args.reuse_build.expanduser().resolve(), source,
                        ignore=shutil.ignore_patterns("*.o"))
    else:
        prepare_build_source(repo, source)
        commands = ([sys.executable, str(source / "configure.py"),
                     "--prob=nrfld_thermal_wave", "--eos=general/thermal_wave",
                     "-nrmgfld", "-hdf5", "-mpi", "--cflag=-O3"],
                    ["make", f"-j{args.build_jobs}"])
        for name, cmd in zip(("configure", "build"), commands):
            with (root / f"{name}.log").open("w") as log:
                result = subprocess.run(cmd, cwd=source, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=args.build_timeout)
            if result.returncode:
                raise RuntimeError(f"{name} failed with {result.returncode}")
    executable = source / "bin/athena"
    template = repo / "inputs/radiation/athinput.nrfld_thermal_wave"
    for tag, n, amr in cases_for(args.preset):
        directory = root / tag
        directory.mkdir()
        (directory / "output").mkdir()
        updates = [f"mesh/nx1={n}", f"mesh/nx2={n}", f"mesh/nx3={n}",
                   "meshblock/nx1=8", "meshblock/nx2=8", "meshblock/nx3=8",
                   "mesh/refinement=" + ("adaptive" if amr else "none"),
                   "mesh/numlevel=" + ("2" if amr else "1"),
                   "time/nlim=200", "time/tlim=1e-7", "time/ncycle_out=100",
                   "output1/data_format=%.16e",
                   "output2/dcycle=20", "output3/dcycle=20", "output4/dcycle=20",
                   "output5/file_type=hdf5", "output5/variable=uov",
                   "output5/id=full3d", "output5/dt=1e-7", "output5/ghost_zones=false",
                   "problem/use_analytic_profile_init=true", "problem/analytic_init_lte=true",
                   "problem/init_time=1e-6", "problem/t_ambient=1",
                   "problem/dt0=1e-9", "problem/dt_growth=1",
                   "fld/only_rad=false", "nrfld/linear_solver=fgmres",
                   "nrfld/fgmres_threshold=1e-8", "nrfld/nr_threshold=1e-7",
                   "nrfld/threshold=1e-10", "nrfld/failure_policy=abort",
                   f"job/problem_id={tag}"]
        materialize_input(template, directory / "input.used", updates)
        command = ["/usr/bin/time", "-v", "-o", str(directory / "resources.txt"),
                   "mpirun", "-np", str(args.mpi_ranks), str(executable),
                   "-i", str(directory / "input.used"), "-d", str(directory / "output")]
        (directory / "command.txt").write_text(" ".join(command) + "\n")
        with (directory / "run.log").open("w") as log:
            result = subprocess.run(command, cwd=directory, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=args.run_timeout)
        print(tag, result.returncode, flush=True)
        text = (directory / "run.log").read_text(errors="replace")
        if result.returncode or "Terminating on time limit" not in text or "FATAL ERROR" in text:
            raise RuntimeError(f"{tag} failed; inspect {directory / 'run.log'}")
    subprocess.run([sys.executable, str(HERE / "thermal_wave_validation.py"),
                    "--postprocess", str(root), "--preset", args.preset], check=True)


if __name__ == "__main__":
    main()
