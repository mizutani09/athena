#!/usr/bin/env python3
"""Run a reproducible MG/FGMRES solar-convection case with external tables.

Build Athena++ separately with --prob=simple_convection_lhllc_fld,
--eos=general/eos_table, --flux=lhllc, -nrmgfld, -mpi, and -hdf5.
Run from the output root so that the input's data/ paths resolve correctly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


DATA_NAMES = ("AthenaNataEosTable_solar_wide.tab",
              "opacity_table_rhoT_3p0_6p3.h5", "solar_initial_profile.dat")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def solver_input(base: str, solver: str, xy_slice_z: float | None = None) -> str:
    if base.count("<nrfld>") != 1:
        raise ValueError("Expected exactly one <nrfld> block")
    block = (f"<nrfld>\nlinear_solver = {solver}\n"
             "coarse_operator = rediscretized\nfgmres_threshold = 1e-8")
    # Remove any old entries before inserting explicit controlled values.
    base = re.sub(r"^\s*(linear_solver|coarse_operator|fgmres_threshold)\s*=.*\n",
                  "", base, flags=re.M)
    result = base.replace("<nrfld>", block, 1)
    if xy_slice_z is not None:
        if len(re.findall(r"^x3_slice\s*=", result, flags=re.M)) != 2:
            raise ValueError("Expected two xy-slice x3_slice entries")
        result = re.sub(r"^x3_slice\s*=\s*[^#\n]+",
                        f"x3_slice = {xy_slice_z:.9g}", result, flags=re.M)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--base-input", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory with the three named EOS/opacity/profile files")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case", required=True, help="New run-directory name")
    parser.add_argument("--solver", choices=("mg", "fgmres"), required=True)
    parser.add_argument("--cfl", type=float, required=True)
    parser.add_argument("--nx", type=int, required=True,
                        help="Cubic grid size; physical box remains the input's 1.536 Mm cube")
    parser.add_argument("--ranks", type=int, required=True)
    parser.add_argument("--mpi-bind-to", choices=("hwthread", "none"),
                        default="hwthread",
                        help="Use none with an outer taskset CPU mask")
    parser.add_argument("--tlim", type=float, required=True)
    parser.add_argument("--nlim", type=int, default=1000000)
    parser.add_argument("--history-dt", type=float,
                        help="Override history cadence in code time units")
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--history-header-from", type=Path,
                        help="Header-bearing .hst for a restart whose parent history has no header")
    parser.add_argument("--history-prepend", type=Path, action="append", default=[],
                        help="Earlier history segment to include in the report; repeat as needed")
    parser.add_argument("--xy-slice-z", type=float,
                        help="Code-coordinate height of xy slices, e.g. 2.0 near initial tau=1")
    parser.add_argument("--no-slices", action="store_true",
                        help="Disable HDF5 slices and restart output for a short solver test")
    parser.add_argument("--plan", action="store_true", help="Print command without writing/running")
    args = parser.parse_args()
    repo = args.repo.resolve()
    binary = (args.binary or repo / "bin/athena").resolve()
    base_input = (args.base_input or repo / "inputs/radiation"
                  / "athinput.simple_convection_lhllc_fld_general_hpc_256").resolve()
    data_dir = args.data_dir.resolve()
    root = args.output_root.resolve()
    case = root / args.case
    if args.nx % 16 or args.nx < 16:
        parser.error("--nx must be a multiple of the input's 16-cell MeshBlock")
    if args.ranks < 1 or args.ranks > (args.nx // 16) ** 3:
        parser.error("--ranks must be between 1 and the MeshBlock count")
    if args.restart and not args.restart.is_file():
        parser.error(f"Restart not found: {args.restart}")
    if args.history_header_from and not args.history_header_from.is_file():
        parser.error(f"History header file not found: {args.history_header_from}")
    for earlier in args.history_prepend:
        if not earlier.is_file():
            parser.error(f"Earlier history file not found: {earlier}")
    for path in (binary, base_input, *(data_dir / name for name in DATA_NAMES)):
        if not path.is_file():
            parser.error(f"Required file not found: {path}")
    command = (["mpirun", "--use-hwthread-cpus", "--bind-to", args.mpi_bind_to,
                "--map-by", "hwthread", "-np", str(args.ranks)]
               if args.ranks > 1 else [])
    command += [str(binary)]
    if args.restart:
        command += ["-r", str(args.restart.resolve())]
    command += ["-i", str(case / "athinput.generated"), "-d", str(case),
                f"mesh/nx1={args.nx}", f"mesh/nx2={args.nx}",
                f"mesh/nx3={args.nx}", f"time/cfl_number={args.cfl}",
                f"time/tlim={args.tlim}", f"time/nlim={args.nlim}",
                "time/ncycle_out=100"]
    if args.history_dt is not None:
        if args.history_dt <= 0:
            parser.error("--history-dt must be positive")
        command += [f"output1/dt={args.history_dt}"]
    if args.no_slices:
        command += [f"output{n}/dt=-1" for n in range(2, 7)]
    if args.plan:
        print("cwd:", root, "\ncommand:", " ".join(command))
        return
    if case.exists() and any(case.iterdir()):
        parser.error(f"Refusing to reuse nonempty run directory: {case}")
    case.mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)
    for name in DATA_NAMES:
        link = root / "data" / name
        target = (data_dir / name).resolve()
        if link.is_symlink() and link.resolve() == target:
            continue
        if link.exists() or link.is_symlink():
            parser.error(f"Output data link points elsewhere: {link}")
        link.symlink_to(target)
    generated = case / "athinput.generated"
    generated.write_text(solver_input(base_input.read_text(), args.solver,
                                      args.xy_slice_z))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo,
                                     text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"],
                                         cwd=repo, text=True).strip())
    manifest = {
        "command": command, "cwd": str(root), "git_commit": commit,
        "git_dirty": dirty, "binary_sha256": digest(binary),
        "launch_cpu_affinity": sorted(os.sched_getaffinity(0)),
        "mpi_bind_to": args.mpi_bind_to,
        "base_input_sha256": digest(base_input),
        "generated_input_sha256": digest(generated),
        "data_sha256": {name: digest((data_dir / name).resolve()) for name in DATA_NAMES},
        "restart": str(args.restart.resolve()) if args.restart else None,
        "history_header_from": (str(args.history_header_from.resolve())
                                if args.history_header_from else None),
        "history_prepend": [str(path.resolve()) for path in args.history_prepend],
    }
    manifest_path = case / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    start = time.monotonic()
    with (case / "run.log").open("w") as log:
        result = subprocess.run(command, cwd=root, env=env, stdout=log,
                                stderr=subprocess.STDOUT, check=False)
    manifest["wall_seconds"] = time.monotonic() - start
    manifest["exit_code"] = result.returncode
    log_text = (case / "run.log").read_text(errors="replace")
    manifest["fatal_in_log"] = "FATAL ERROR" in log_text
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if result.returncode or manifest["fatal_in_log"]:
        raise SystemExit(f"Simulation failed; inspect {case / 'run.log'}")
    history = (args.history_header_from.resolve() if args.history_header_from
               else next(args.restart.parent.glob("*.hst"), None)
               if args.restart else None)
    if history and not any(line.startswith("# [")
                           for line in history.read_text().splitlines()):
        raise SystemExit("Restart history lacks column names; pass "
                         "--history-header-from with an original header-bearing .hst")
    if not args.no_slices:
        plot = repo / "scripts/plot_simple_convection.py"
        plot_command = [sys.executable, str(plot), str(case)]
        if history:
            plot_command += ["--history-header-from", str(history)]
        subprocess.run(plot_command, check=True)
        temperature_unit = re.search(r"^\s*T_unit\s*=\s*([^#\s]+)",
                                     generated.read_text(), re.M)
        profile_command = [sys.executable,
                           str(repo / "scripts/plot_simple_convection_profiles.py"),
                           str(case)]
        if temperature_unit:
            profile_command += ["--temperature-unit", temperature_unit.group(1)]
        subprocess.run(profile_command, check=True)
        tau_command = [sys.executable,
                       str(repo / "scripts/plot_simple_convection_tau.py"),
                       str(case), "--input", str(generated),
                       "--opacity-table",
                       str(root / "data" / "opacity_table_rhoT_3p0_6p3.h5")]
        subprocess.run(tau_command, check=True)
    report = Path(__file__).with_name("solar_convection_report.py")
    report_command = [sys.executable, str(report), str(case), "--input",
                      str(generated)]
    if history:
        report_command += ["--history-header-from", str(history)]
    for earlier in args.history_prepend:
        report_command += ["--history-prepend", str(earlier.resolve())]
    subprocess.run(report_command, check=True)


if __name__ == "__main__":
    main()
