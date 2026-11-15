#!/usr/bin/env python3
"""One-cycle Jeans AMR regression on the exact WOR-92/118 table-run source."""

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

from run_suite import effective_settings, prepare_build_source


# Match the flags recorded by the full c0→first-core WOR-118 production run.
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
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command, directory, log, timeout):
    started = time.monotonic()
    with open(log, "w") as stream:
        process = subprocess.run(command, cwd=directory, stdout=stream,
                                 stderr=subprocess.STDOUT, timeout=timeout,
                                 env={**os.environ, **PRODUCTION_ENV,
                                      "OMP_NUM_THREADS": "1"},
                                 check=False)
    elapsed = time.monotonic() - started
    if process.returncode:
        raise RuntimeError("Command failed: {} (see {})".format(command, log))
    return elapsed


def write_input(source, target, n_jeans):
    overrides = [
        "job/problem_id=wor95_table_source_nj{}".format(n_jeans),
        "time/cfl_number=0.02", "time/nlim=1", "time/ncycle_out=1",
        "mesh/nx1=32", "mesh/nx2=32", "mesh/nx3=32", "mesh/numlevel=2",
        "meshblock/nx1=16", "meshblock/nx2=16", "meshblock/nx3=16",
        "problem/N_J={}".format(n_jeans), "problem/refinement_mode=jeans",
        "nrfld/linear_solver=fgmres", "nrfld/fgmres_threshold=2e-8",
        "nrfld/nr_threshold=2e-8", "nrfld/diagnostic_verbosity=1",
        "nrfld/nr_accept_l2_threshold=-1", "nrfld/nr_accept_max_threshold=-1",
        "nrfld/failure_policy=abort", "output1/data_format=%.16e",
        "output6/dcycle=1",
    ]
    settings = effective_settings(source, overrides)
    sections = {}
    for key, value in settings.items():
        section, name = key.split("/", 1)
        sections.setdefault(section, []).append("{} = {}".format(name, value))
    target.write_text("\n\n".join("<{}>\n{}".format(name, "\n".join(values))
                                   for name, values in sections.items()) + "\n")
    return settings


def analyze(case, n_jeans, elapsed):
    log = (case / "run.log").read_text()
    if "FATAL ERROR" in log:
        raise RuntimeError("Athena reported FATAL ERROR in {}".format(case))
    history_files = list((case / "output").glob("*.hst"))
    if len(history_files) != 1:
        raise RuntimeError("Expected exactly one HST in {}".format(case))
    data = np.loadtxt(history_files[0], comments="#")
    rows = np.atleast_2d(data)
    active = [line for line in log.splitlines()
              if line.startswith("[NR] status=") and "stage=0 " in line]
    fgmres = [line for line in log.splitlines()
              if line.startswith("[LINEAR_FGMRES] ")]
    if not active or not fgmres:
        raise RuntimeError("Missing Newton/FGMRES diagnostics in {}".format(case))
    if any("status=converged " not in line for line in active):
        raise RuntimeError("Nonconverged Newton solve in {}".format(case))
    blocks_match = re.search(r"Number of MeshBlocks = (\d+);\s*(\d+)\s+created", log)
    memory_match = re.search(r"Maximum resident set size \(kbytes\): (\d+)", log)
    dt_match = re.search(r"\bdt=(\S+)", active[0])
    rel = []
    iterations = []
    for line in fgmres:
        fields = dict(re.findall(r"(\w+)=(\S+)", line))
        value = float(fields["relative_residual"])
        target = float(fields["target"])
        if not np.isfinite(value) or value > target or fields["breakdown"] != "0":
            raise RuntimeError("FGMRES failure in {}: {}".format(case, line))
        rel.append(value)
        iterations.append(int(fields["iterations"]))
    maximum = float(np.max(rows[:, 36])) if rows.shape[1] > 36 else None
    result = {
        "N_J": n_jeans, "elapsed_seconds": elapsed,
        "maxrss_kib": int(memory_match[1]) if memory_match else None,
        "meshblocks": int(blocks_match[1]) if blocks_match else None,
        "created_meshblocks": int(blocks_match[2]) if blocks_match else None,
        "solve_dt_code": float(dt_match[1]) if dt_match else None,
        "history_rows": int(len(rows)), "initial_level": float(rows[0, 35]),
        "final_level": float(rows[-1, 35]),
        "initial_jeans_cells": float(rows[0, 34]),
        "final_jeans_cells": float(rows[-1, 34]),
        "rho_max_g_cm3": float(rows[-1, 15]),
        "pressure_min_dyn_cm2": float(np.min(rows[:, 29])),
        "mass_relative_change": float(rows[-1, 22] / rows[0, 22] - 1),
        "background_temperature_error_max": maximum,
        "newton_converged": len(active), "fgmres_solves": len(fgmres),
        "fgmres_max_iterations": max(iterations),
        "fgmres_max_true_relative_residual": max(rel),
        "history_sha256": sha256(history_files[0]),
    }
    if result["pressure_min_dyn_cm2"] <= 0 or abs(result["mass_relative_change"]) > 1e-12:
        raise RuntimeError("Physical gate failed in {}".format(case))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--build-cores", type=int, default=4)
    parser.add_argument("--binary", type=Path,
                        help="Reuse a previously built binary from the same source")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    configure = None
    build_elapsed = 0.0
    if args.binary:
        binary = args.binary.resolve()
    else:
        source = root / "build_source"
        prepare_build_source(repo, source)
        configure = ["python3", str(source / "configure.py"),
                     "--prob=protostellar_core_fld", "--coord=cartesian",
                     "--eos=adiabatic", "-b", "--grav=mg", "-nrmgfld",
                     "--flux=hlld_fld", "--nscalars=1", "-hdf5", "-mpi",
                     "--cflag=-O3"]
        build_elapsed = run(configure, source, root / "configure.log", 120)
        build_elapsed += run(["make", "-j{}".format(args.build_cores)], source,
                             root / "build.log", 900)
        binary = source / "bin/athena"
    input_source = repo / "inputs/radiation/athinput.protostellar_core_fld_mayer5"
    summary = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                cwd=repo, text=True).strip(),
        "source_status": subprocess.check_output(["git", "status", "--short"],
                                                cwd=repo, text=True).strip(),
        "configure_command": configure,
        "build_elapsed_seconds": build_elapsed,
        "runtime_environment": PRODUCTION_ENV,
        "binary_sha256": sha256(binary),
        "base_input_sha256": sha256(input_source),
        "cases": [],
    }
    for n_jeans in (4, 32):
        case = root / "nj{}".format(n_jeans)
        case.mkdir()
        (case / "output").mkdir()
        settings = write_input(input_source, case / "input.used", n_jeans)
        command = ["/usr/bin/time", "-v", "mpirun", "-np", "2",
                   str(binary), "-i", str(case / "input.used"),
                   "-d", str(case / "output")]
        elapsed = run(command, case, case / "run.log", 300)
        result = analyze(case, n_jeans, elapsed)
        result.update(input_sha256=sha256(case / "input.used"), settings=settings)
        summary["cases"].append(result)
        (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    low, high = summary["cases"]
    summary["passed"] = bool(low["final_level"] == 1 and high["final_level"] == 2
                             and high["initial_jeans_cells"] < 32
                             and low["initial_jeans_cells"] > 4)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"passed": summary["passed"], "cases": summary["cases"]}, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
