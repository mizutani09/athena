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


DATA_KEYS = ("eos_file_name", "opacity_table_file", "initial_profile_file")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def solver_input(base: str, solver: str, xy_slice_z: float | None = None,
                 history_precision: int | None = None) -> str:
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
    if history_precision is not None:
        output1 = re.search(r"^<output1>[ \t]*$.*?(?=^<|\Z)", result,
                            re.M | re.S)
        if output1 is None:
            raise ValueError("Expected an <output1> history block")
        block = output1.group(0)
        setting = f"data_format = %22.{history_precision-1}e"
        if re.search(r"^[ \t]*data_format[ \t]*=", block, re.M):
            block = re.sub(r"^[ \t]*data_format[ \t]*=[^\n]*", setting,
                           block, count=1, flags=re.M)
        else:
            block = block.replace("<output1>", "<output1>\n" + setting, 1)
        result = result[:output1.start()] + block + result[output1.end():]
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
    parser.add_argument("--nx", type=int,
                        help="Cubic grid size; box extent remains set by the input")
    parser.add_argument("--nx1", type=int, help="Non-cubic x1 grid size")
    parser.add_argument("--nx2", type=int, help="Non-cubic x2 grid size")
    parser.add_argument("--nx3", type=int, help="Non-cubic x3 grid size")
    parser.add_argument("--ranks", type=int, required=True)
    parser.add_argument("--mpi-bind-to", choices=("hwthread", "core", "none"),
                        default="hwthread",
                        help="Use none with an outer taskset CPU mask")
    parser.add_argument("--mpi-map-by", choices=("hwthread", "core"),
                        default="hwthread",
                        help="Map ranks to logical CPUs or physical cores")
    parser.add_argument("--tlim", type=float, required=True)
    parser.add_argument("--nlim", type=int, default=1000000)
    parser.add_argument("--history-dt", type=float,
                        help="Override history cadence in code time units")
    parser.add_argument("--history-precision", type=int,
                        help="Significant digits in history floating-point columns (6–16)")
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--history-header-from", type=Path,
                        help="Header-bearing .hst for a restart whose parent history has no header")
    parser.add_argument("--history-prepend", type=Path, action="append", default=[],
                        help="Earlier history segment to include in the report; repeat as needed")
    parser.add_argument("--xy-slice-z", type=float,
                        help="Code-coordinate height of xy slices, e.g. 2.0 near initial tau=1")
    parser.add_argument("--no-slices", action="store_true",
                        help="Disable HDF5 slices and restart output for a short solver test")
    parser.add_argument("--movie", action="store_true",
                        help="Create an xz MP4 from the slice outputs")
    parser.add_argument("--problem-set", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="Override one existing <problem> key; repeat for controls")
    parser.add_argument("--plan", action="store_true", help="Print command without writing/running")
    args = parser.parse_args()
    repo = args.repo.resolve()
    binary = (args.binary or repo / "bin/athena").resolve()
    base_input = (args.base_input or repo / "inputs/radiation"
                  / "athinput.simple_convection_lhllc_fld_general_hpc_256").resolve()
    if not base_input.is_file():
        parser.error(f"Input file not found: {base_input}")
    base_text = base_input.read_text()
    problem_block = re.search(r"^<problem>\s*$(.*?)(?=^<|\Z)",
                              base_text, re.M | re.S)
    if not problem_block:
        parser.error(f"Missing <problem> block in {base_input}")
    problem_keys = set(re.findall(r"^\s*([A-Za-z_][A-Za-z_0-9]*)\s*=",
                                  problem_block.group(1), re.M))
    problem_overrides = []
    for override in args.problem_set:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*=[^\s=]+", override):
            parser.error(f"Invalid --problem-set value: {override!r}")
        key = override.split("=", 1)[0]
        if key not in problem_keys:
            parser.error(f"Unknown <problem> key in --problem-set: {key}")
        problem_overrides.append(f"problem/{override}")
    data_names = []
    for key in DATA_KEYS:
        match = re.search(rf"^\s*{key}\s*=\s*([^#\s]+)", base_text, re.M)
        if not match or Path(match.group(1)).parent != Path("data"):
            parser.error(f"Expected {key} = data/<filename> in {base_input}")
        data_names.append(Path(match.group(1)).name)
    data_dir = args.data_dir.resolve()
    root = args.output_root.resolve()
    case = root / args.case
    if args.nx is not None and any(x is not None for x in (args.nx1, args.nx2, args.nx3)):
        parser.error("Use either --nx or all of --nx1, --nx2, --nx3")
    dims = ((args.nx,) * 3 if args.nx is not None
            else (args.nx1, args.nx2, args.nx3))
    if any(x is None or x < 16 or x % 16 for x in dims):
        parser.error("Set three grid sizes, each a multiple of the 16-cell MeshBlock")
    spacing = []
    for axis, cells in enumerate(dims, start=1):
        endpoints = []
        for suffix in ("min", "max"):
            match = re.search(rf"^\s*x{axis}{suffix}\s*=\s*([^#\s]+)",
                              base_text, re.M)
            if not match:
                parser.error(f"Missing x{axis}{suffix} in {base_input}")
            endpoints.append(float(match.group(1)))
        spacing.append((endpoints[1] - endpoints[0]) / cells)
    if max(spacing) - min(spacing) > 1.0e-10 * max(spacing):
        parser.error("NR-FLD multigrid requires cubic cells; choose grid sizes "
                     "proportional to the three box lengths")
    if args.ranks < 1 or args.ranks > (dims[0] // 16) * (dims[1] // 16) * (dims[2] // 16):
        parser.error("--ranks must be between 1 and the MeshBlock count")
    if args.movie and args.no_slices:
        parser.error("--movie requires slice outputs")
    if args.restart and not args.restart.is_file():
        parser.error(f"Restart not found: {args.restart}")
    if args.history_header_from and not args.history_header_from.is_file():
        parser.error(f"History header file not found: {args.history_header_from}")
    for earlier in args.history_prepend:
        if not earlier.is_file():
            parser.error(f"Earlier history file not found: {earlier}")
    for path in (binary, base_input, *(data_dir / name for name in data_names)):
        if not path.is_file():
            parser.error(f"Required file not found: {path}")
    command = (["mpirun", "--use-hwthread-cpus", "--bind-to", args.mpi_bind_to,
                "--map-by", args.mpi_map_by, "-np", str(args.ranks)]
               if args.ranks > 1 else [])
    command += [str(binary)]
    if args.restart:
        command += ["-r", str(args.restart.resolve())]
    command += ["-i", str(case / "athinput.generated"), "-d", str(case),
                f"mesh/nx1={dims[0]}", f"mesh/nx2={dims[1]}",
                f"mesh/nx3={dims[2]}", f"time/cfl_number={args.cfl}",
                f"time/tlim={args.tlim}", f"time/nlim={args.nlim}",
                "time/ncycle_out=100"]
    if args.history_dt is not None:
        if args.history_dt <= 0:
            parser.error("--history-dt must be positive")
        command += [f"output1/dt={args.history_dt}"]
    if args.history_precision is not None:
        if not 6 <= args.history_precision <= 16:
            parser.error("--history-precision must be between 6 and 16")
    command += problem_overrides
    if args.no_slices:
        output_sections = re.findall(r"^<output(\d+)>", base_text, re.M)
        command += [f"output{n}/dt=-1" for n in output_sections if int(n) >= 2]
    if args.plan:
        print("cwd:", root, "\ncommand:", " ".join(command))
        return
    if case.exists() and any(case.iterdir()):
        parser.error(f"Refusing to reuse nonempty run directory: {case}")
    case.mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)
    for name in data_names:
        link = root / "data" / name
        target = (data_dir / name).resolve()
        if link.is_symlink() and link.resolve() == target:
            continue
        if link.exists() or link.is_symlink():
            parser.error(f"Output data link points elsewhere: {link}")
        link.symlink_to(target)
    generated = case / "athinput.generated"
    generated.write_text(solver_input(base_text, args.solver,
                                      args.xy_slice_z, args.history_precision))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo,
                                     text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"],
                                         cwd=repo, text=True).strip())
    manifest = {
        "command": command, "cwd": str(root), "git_commit": commit,
        "git_dirty": dirty, "binary_sha256": digest(binary),
        "launch_cpu_affinity": sorted(os.sched_getaffinity(0)),
        "mpi_bind_to": args.mpi_bind_to,
        "mpi_map_by": args.mpi_map_by,
        "base_input_sha256": digest(base_input),
        "generated_input_sha256": digest(generated),
        "data_sha256": {name: digest((data_dir / name).resolve()) for name in data_names},
        "restart": str(args.restart.resolve()) if args.restart else None,
        "history_header_from": (str(args.history_header_from.resolve())
                                if args.history_header_from else None),
        "history_prepend": [str(path.resolve()) for path in args.history_prepend],
        "problem_overrides": args.problem_set,
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
        if args.movie:
            plot_command += ["--movie"]
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
                       str(root / "data" / data_names[1])]
        subprocess.run(tau_command, check=True)
        if any(case.glob("*.prim_full.*.athdf")):
            surface_command = [sys.executable,
                               str(repo / "scripts/plot_simple_convection_surface3d.py"),
                               str(case), "--input", str(generated),
                               "--opacity-table", str(root / "data" / data_names[1])]
            surface_command += ["--reference-profile",
                                str(root / "data" / data_names[2])]
            subprocess.run(surface_command, check=True)
            flux_command = [sys.executable,
                            str(repo / "scripts/plot_simple_convection_flux3d.py"),
                            str(case), "--input", str(generated),
                            "--opacity-table", str(root / "data" / data_names[1])]
            subprocess.run(flux_command, check=True)
            late_profile_command = [sys.executable,
                                    str(repo / "scripts/plot_simple_convection_late_profiles3d.py"),
                                    str(case), "--reference-profile",
                                    str(root / "data" / data_names[2])]
            subprocess.run(late_profile_command, check=True)
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
