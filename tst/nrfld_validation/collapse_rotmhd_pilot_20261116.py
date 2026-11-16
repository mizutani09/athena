#!/usr/bin/env python3
"""One-cycle, matched-input table-EOS rotation/MHD gates on a local MPI host."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from nrfld_validation.io import read_history


ENV = {
    "ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE": "1",
    "ATHENA_NRFLD_CONSERVATIVE_SHARED_COEFF": "1",
    "ATHENA_NRFLD_EXPERIMENTAL_TANGENTIAL_RECONSTRUCTION": "1",
    "ATHENA_NRFLD_EXPERIMENTAL_VARIABLE_LIMITER": "1",
    "ATHENA_NRFLD_EXPERIMENTAL_COUPLED": "1",
    "ATHENA_NRFLD_EXPERIMENTAL_REGRID": "1",
    "ATHENA_NRFLD_CONSERVATIVE_DIFFERENCE_JV": "0",
    "ATHENA_WOR92_INEXACT_TRUE_CAP": "1e-7",
    "ATHENA_WOR92_INEXACT_STAGNATION_RATIO": "0.9",
    "ATHENA_WOR92_CENTERED_FACE_TRIAL": "1",
    "ATHENA_WOR92_POSITIVE_NR_GHOST": "2",
    "OMP_NUM_THREADS": "1",
}
CASES = {
    "base": (False, False),
    "rotation": (True, False),
    "magnetic": (False, True),
    "combined": (True, True),
}
ASSETS = (
    "data/eos/protostellar_tomida2015_material.tab",
    "data/opacity/protostellar_semenov2003_nrm_c_a.h5",
)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mpi-ranks", type=int, default=2)
    parser.add_argument("--cycles", type=int, default=1)
    args = parser.parse_args()
    if args.cycles < 1 or args.mpi_ranks < 1:
        parser.error("positive cycles and MPI ranks required")
    repo = Path(__file__).resolve().parents[2]
    source = repo / "inputs/radiation/athinput.protostellar_firstcore_tabular_v1"
    if not args.binary.is_file():
        parser.error("binary missing")
    case_dir = args.output_root.resolve() / args.case
    case_dir.mkdir(parents=True, exist_ok=False)
    for relative in ASSETS:
        target = case_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.assets_root.resolve() / relative, target)
    output = case_dir / "output"
    output.mkdir()
    rotation, magnetic = CASES[args.case]
    input_text = source.read_text()
    marker = "enable_magnetic_field = false"
    if input_text.count(marker) != 1:
        raise RuntimeError("unexpected base input")
    input_text = input_text.replace(
        marker, marker + "\nenable_tabular_mhd_pilot = false")
    local_input = case_dir / "input.in"
    local_input.write_text(input_text)
    overrides = [
        "job/problem_id=" + args.case,
        "time/nlim=" + str(args.cycles),
        "time/ncycle_out=" + str(args.cycles),
        "output2/dcycle=" + str(args.cycles),
        "output2/variable=cons,prim,uov",
        "output4/dcycle=" + str(args.cycles),
        "nrfld/diagnostic_verbosity=0",
        "nrfld/jacobian_check=false",
        "nrfld/jacobian_check_each_cycle=false",
        "problem/enable_rotation=" + str(rotation).lower(),
        "problem/enable_magnetic_field=" + str(magnetic).lower(),
        "problem/enable_tabular_mhd_pilot=" + str(magnetic).lower(),
    ]
    command = ["/usr/bin/time", "-v", "-o", "resources.txt", "mpirun",
               "-np", str(args.mpi_ranks), str(args.binary.resolve()),
               "-i", str(local_input), "-d", str(output)] + overrides
    (case_dir / "command.json").write_text(json.dumps({
        "command": command, "environment": ENV,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "source_status": subprocess.check_output(["git", "status", "--short", "--", "src"], cwd=repo, text=True).strip(),
        "base_input_sha256": sha(source), "input_sha256": sha(local_input),
        "binary_sha256": sha(args.binary),
        "asset_sha256": {relative: sha(case_dir / relative) for relative in ASSETS},
    }, indent=2) + "\n")
    start = time.monotonic()
    with (case_dir / "run.log").open("w") as stream:
        result = subprocess.run(command, cwd=case_dir, env={**os.environ, **ENV},
                                stdout=stream, stderr=subprocess.STDOUT, timeout=1800)
    elapsed = time.monotonic() - start
    log = (case_dir / "run.log").read_text(errors="replace")
    history_files = list(output.glob("*.hst"))
    history = read_history(history_files[0]) if len(history_files) == 1 else {}
    linear = [dict(re.findall(r"(\w+)=(\S+)", line))
              for line in log.splitlines() if line.startswith("[LINEAR_FGMRES] ")]
    newton = [line for line in log.splitlines() if line.startswith("[NR] status=")]
    active_newton = [line for line in newton if "status=dt_zero_initialization" not in line]
    resource_text = (case_dir / "resources.txt").read_text()
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", resource_text)
    last = {key: float(value[-1]) for key, value in history.items()
            if key in ("rho_max", "Tgas_max", "B_max", "divB_max", "Lz",
                       "gas_mass", "Emag_cgs", "jeans_min", "level_max")}
    summary = {
        "exit_code": result.returncode,
        "passed": result.returncode == 0 and "FATAL ERROR" not in log
                  and "Terminating on cycle limit" in log
                  and bool(active_newton)
                  and all("status=converged " in x for x in active_newton),
        "elapsed_seconds": elapsed, "launcher_maxrss_kib": int(rss[1]) if rss else None,
        "newton_calls": len(active_newton), "fgmres_calls": len(linear),
        "fgmres_max_iterations": max((int(x["iterations"]) for x in linear), default=None),
        "fgmres_max_true_relative_residual": max(
            (float(x["relative_residual"]) for x in linear), default=None),
        "history": last,
    }
    (case_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
