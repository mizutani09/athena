#!/usr/bin/env python3
"""One-cycle table-EOS/opacity smoke test for the collaborator first-core source."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np

from nrfld_validation.io import read_history
from run_suite import prepare_build_source


SOURCE_TREE = "c4ab200383c7317a2feccc50aa9578d5cf71478d"
INPUT_SHA = "aad81db83087d27351db59ef70b9b14a68e8a2dc3e6fc164a4cb73e8d413e7e6"
ASSETS = {
    "data/eos/protostellar_tomida2015_material.tab":
        "742a3456923d4c7056c3705accbcc793af3e0c79408a2f597a0cd9271b13ba43",
    "data/opacity/protostellar_semenov2003_nrm_c_a.h5":
        "6c42445452bca6aeca1253f31ab049b2ff107dea3601f946b4c61c93db022f94",
}
PRODUCTION_ENV = {
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
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def execute(command, cwd, logfile, timeout, env=None):
    started = time.monotonic()
    with logfile.open("w") as output:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=output,
                                stderr=subprocess.STDOUT, timeout=timeout)
    elapsed = time.monotonic() - started
    if result.returncode:
        raise RuntimeError("command failed ({}): {}".format(result.returncode, logfile))
    return elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, required=True,
                        help="Directory containing data/eos and data/opacity")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="New, empty result directory outside the source tree")
    parser.add_argument("--mpi-ranks", type=int, default=2)
    parser.add_argument("--build-jobs", type=int, default=2)
    args = parser.parse_args()
    if args.mpi_ranks < 1 or args.build_jobs < 1:
        parser.error("MPI ranks and build jobs must be positive")

    repo = Path(__file__).resolve().parents[2]
    source_tree = subprocess.check_output(["git", "rev-parse", "HEAD:src"],
                                          cwd=repo, text=True).strip()
    if source_tree != SOURCE_TREE:
        raise RuntimeError("solver source tree differs from frozen version: " + source_tree)
    source_status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", "configure.py", "Makefile.in"],
        cwd=repo, text=True).strip()
    if source_status:
        raise RuntimeError("solver source is dirty: " + source_status)
    input_file = repo / "inputs/radiation/athinput.protostellar_firstcore_tabular_v1"
    if sha256(input_file) != INPUT_SHA:
        raise RuntimeError("input SHA-256 mismatch")
    assets_root = args.assets_root.resolve()
    for relative, expected in ASSETS.items():
        path = assets_root / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError("missing or changed table: " + str(path))

    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    build = root / "build_source"
    prepare_build_source(repo, build)
    for relative in ASSETS:
        target = build / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(assets_root / relative, target)
    configure = ["python3", "configure.py", "--prob=protostellar_core_fld",
                 "--coord=cartesian", "--eos=general/eos_table", "-b",
                 "--grav=mg", "-nrmgfld", "--flux=hlld_fld",
                 "--nscalars=1", "-hdf5", "-mpi", "--cflag=-O3"]
    times = {}
    times["configure_seconds"] = execute(configure, build, root / "configure.log", 120)
    times["build_seconds"] = execute(["make", "-j{}".format(args.build_jobs)],
                                     build, root / "build.log", 1200)
    binary = build / "bin/athena"
    output = root / "output"
    output.mkdir()
    env = {**os.environ, **PRODUCTION_ENV, "OMP_NUM_THREADS": "1"}
    command = ["/usr/bin/time", "-v", "-o", str(root / "resources.txt"),
               "mpirun", "-np", str(args.mpi_ranks), str(binary),
               "-i", str(input_file), "-d", str(output),
               "time/nlim=1", "time/ncycle_out=1"]
    times["run_seconds"] = execute(command, build, root / "run.log", 600, env)
    log = (root / "run.log").read_text(errors="replace")
    if "FATAL ERROR" in log or "Terminating on cycle limit" not in log:
        raise RuntimeError("Athena reported a failure; see run.log")
    nr = [line for line in log.splitlines() if line.startswith("[NR] status=")]
    active_nr = [line for line in nr if "dt=0.000000000000000" not in line]
    if not active_nr or any("status=converged " not in line for line in active_nr):
        raise RuntimeError("Newton failed or was not recorded")
    linear = [line for line in log.splitlines()
              if line.startswith("[LINEAR_FGMRES] ")]
    if not linear:
        raise RuntimeError("FGMRES results missing")
    relative = []
    accepted = []
    iterations = []
    for line in linear:
        fields = dict(re.findall(r"(\w+)=(\S+)", line))
        value, target = float(fields["relative_residual"]), float(fields["target"])
        inexact = fields.get("inexact_accept") == "1" and value <= 1e-7
        if not np.isfinite(value) or not (value <= target or inexact):
            raise RuntimeError("linear solve failed: " + line)
        relative.append(value)
        accepted.append(inexact)
        iterations.append(int(fields["iterations"]))
    history_files = list(output.glob("*.hst"))
    if len(history_files) != 1:
        raise RuntimeError("expected one history file")
    hist = read_history(history_files[0])
    required = ("gas_mass", "rho_max", "Tgas_max", "jeans_min")
    if any(key not in hist or len(hist[key]) < 2
           or not np.isfinite(hist[key]).all() for key in required):
        raise RuntimeError("missing or nonfinite history diagnostics")
    mass_drift = float(hist["gas_mass"][-1] / hist["gas_mass"][0] - 1.0)
    passed = (abs(mass_drift) <= 1e-10 and hist["rho_max"][-1] > 0.0
              and hist["Tgas_max"][-1] > 0.0 and hist["jeans_min"][-1] > 0.0)
    resources = (root / "resources.txt").read_text()
    memory = re.search(r"Maximum resident set size \(kbytes\): (\d+)", resources)
    result = {
        "passed": bool(passed), "source_tree_git": source_tree,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                              cwd=repo, text=True).strip(),
        "input_sha256": INPUT_SHA, "table_sha256": ASSETS,
        "binary_sha256": sha256(binary), "mpi_ranks": args.mpi_ranks,
        "configure_command": configure, "run_command": command,
        "environment": PRODUCTION_ENV, "times": times,
        "launcher_maxrss_kib": int(memory[1]) if memory else None,
        "newton_calls": len(active_nr), "fgmres_calls": len(linear),
        "fgmres_max_iterations": max(iterations),
        "fgmres_max_true_relative_residual": max(relative),
        "fgmres_inexact_accepts": sum(accepted),
        "gas_mass_relative_change": mass_drift,
        "rho_max_g_cm3": float(hist["rho_max"][-1]),
        "Tgas_max_K": float(hist["Tgas_max"][-1]),
        "jeans_min_cells": float(hist["jeans_min"][-1]),
        "history_sha256": sha256(history_files[0]),
    }
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
