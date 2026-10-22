#!/usr/bin/env python3
"""Isolated, sequential HD/MHD/FLD/gravity benchmark (WOR-20).

Example: python /path/to/performance.py --cases hd hd_fld --ranks 1 2 4
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
import re
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import h5py
import numpy as np

from run_suite import prepare_build_source
from nrfld_validation.manifest import base_manifest, sha256_file, tree_sha256, write_manifest

CASES = ("hd", "hd_fld", "hd_gravity", "hd_fld_gravity",
         "mhd", "mhd_fld", "mhd_gravity", "mhd_fld_gravity")


def execute(command, cwd, log, timeout, records):
    """Kill the entire launcher/build process group on timeout or interruption."""
    record = {"command": shlex.join(map(str, command)), "cwd": str(cwd), "log": str(log)}
    records.append(record)
    start = time.monotonic()
    with log.open("w") as stream:
        process = subprocess.Popen(list(map(str, command)), cwd=cwd, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            result = process.wait(timeout=timeout)
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            record["status"] = "interrupted_or_timeout"
            raise
        finally:
            record["elapsed_seconds"] = time.monotonic() - start
            record["returncode"] = process.returncode
    record["status"] = "passed" if result == 0 else "failed"
    if result != 0:
        raise RuntimeError(f"Command failed; see {log}")
    return record["elapsed_seconds"]


def input_text(args, ranks, enabled):
    factors = {1: (1, 1, 1), 2: (2, 1, 1), 4: (2, 2, 1), 8: (2, 2, 2)}
    shape = factors[ranks] if args.mode == "weak" else (1, 1, 1)
    nlim = args.warmup + args.steps
    # Match Athena's sequential floating-point time accumulation. nlim*dt can
    # be one ulp larger, causing termination on nlim before the final output.
    end_time = 0.0
    for _ in range(nlim):
        end_time += args.dt
    sections = {
        "job": {"problem_id": "benchmark"},
        "time": {"cfl_number": 0.3, "nlim": nlim, "tlim": end_time,
                 "integrator": "vl2", "xorder": 2, "ncycle_out": nlim},
        "mesh": {}, "meshblock": {},
        "hydro": {"gamma": 5/3, "rho_unit": 1e-10, "egas_unit": 1e2,
                  "leng_unit": 1e13, "mu": 1.0},
        "problem": {"dt": args.dt},
        "performance": {"enabled": str(enabled).lower(), "warmup": args.warmup},
        "fld": {"is_couple": "true", "only_rad": "false", "cut_diff": "false",
                "include_radiation_force": "true", "const_opacity": 0.4,
                "stage_diagnostics": "false"},
        "nrfld": {"mgmode": "MGI", "fas": "false", "smoother": "jacobi-rb",
                  "prolongation": "trilinear", "npresmooth": 2, "npostsmooth": 2,
                  "omega": 1, "threshold": 1e-10, "nr_threshold": 1e-8,
                  "nr_max_iterations": 100, "failure_policy": "abort",
                  "diagnostic_verbosity": 0, "coarse_solver": "direct",
                  "coarse_direct_max_cells": 64, "coarse_diagnostics": "false"},
        "gravity": {"mgmode": "FMG", "threshold": 1e-10, "show_defect": "false"},
        "output1": {"file_type": "hst", "dcycle": nlim + 1},
        "output2": {"file_type": "hdf5", "variable": "prim", "id": "prim",
                    "dcycle": nlim + 1},
        "output3": {"file_type": "hdf5", "variable": "uov", "id": "rad",
                    "dcycle": nlim + 1},
    }
    for axis, factor in enumerate(shape, 1):
        sections["mesh"].update({f"nx{axis}": args.resolution * factor,
                                 f"x{axis}min": 0, f"x{axis}max": factor,
                                 f"ix{axis}_bc": "periodic", f"ox{axis}_bc": "periodic"})
        sections["meshblock"][f"nx{axis}"] = args.block
        for group in ("nrfld", "gravity"):
            for side in ("i", "o"):
                sections[group][f"{side}x{axis}_bc"] = "periodic"
    return "\n".join(f"<{section}>\n" + "\n".join(f"{k} = {v}" for k, v in values.items())
                     + "\n" for section, values in sections.items())


def final_fields(directory):
    fields = {}
    for kind in ("prim", "rad"):
        paths = sorted(directory.glob(f"benchmark.{kind}.*.athdf"))
        if len(paths) != 2:
            raise ValueError(f"Expected initial and final {kind} snapshots: {directory}")
        with h5py.File(paths[-1]) as h:
            fields[f"{kind}_time"] = np.array(h.attrs["Time"])
            names = [n.decode() if isinstance(n, bytes) else str(n)
                     for n in h.attrs["VariableNames"]]
            offset = 0
            for dataset, number in zip(h.attrs["DatasetNames"], h.attrs["NumVariables"]):
                dataset = dataset.decode() if isinstance(dataset, bytes) else str(dataset)
                for index in range(int(number)):
                    fields[names[offset + index]] = h[dataset][index]
                offset += int(number)
    return fields


def compare_fields(left, right):
    """Report bitwise equality separately from a strict double-roundoff check.

    Parallel MHD magnetic-field output can differ by one ulp. Do not
    confuse that with a numerical perturbation comparable to solver tolerance.
    """
    if left.keys() != right.keys():
        return {"final_arrays_identical": False, "within_roundoff": False}
    errors = {}
    identical = True
    for key in left:
        a, b = left[key], right[key]
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            return {"final_arrays_identical": False, "within_roundoff": False}
        identical = identical and np.array_equal(a, b)
        errors[key] = float(np.max(np.abs(a-b))) / max(1.0, float(np.max(np.abs(a))),
                                                      float(np.max(np.abs(b))))
    limit = float(32 * np.finfo(np.float64).eps)
    return {"final_arrays_identical": bool(identical), "within_roundoff": bool(max(errors.values()) <= limit),
            "max_scaled_difference": max(errors.values()), "per_field_scaled_difference": errors,
            "scaled_difference_limit": limit}


def analyse(directory, args, ranks, case):
    log = (directory / "run.log").read_text()
    if "FATAL" in log or "### WARNING" in log:
        raise ValueError("Athena reported a fatal error or solver warning")
    fields = final_fields(directory)
    if not all(np.isfinite(a).all() for a in fields.values()):
        raise ValueError("Nonfinite output")
    for name in ("rho", "press") + (("Er",) if "fld" in case else ()):
        if name not in fields or not (fields[name] > 0).all():
            raise ValueError(f"Nonpositive/missing {name}")
    expected_time = (args.warmup + args.steps) * args.dt
    if not np.isclose(fields["prim_time"], expected_time, rtol=1e-10, atol=1e-14):
        raise ValueError("Unexpected final time (CFL may have reduced requested dt)")
    raw = [json.loads((directory / f"performance_rank{rank}.json").read_text())
           for rank in range(ranks)]
    for rank, item in enumerate(raw):
        if item["rank"] != rank or len(item["steps"]) != args.steps:
            raise ValueError("Incomplete timing window")
        if [s["cycle"] for s in item["steps"]] != list(range(args.warmup, args.warmup+args.steps)):
            raise ValueError("Incorrect measured cycles")
        if not np.isfinite(item["wall_seconds"]) or item["wall_seconds"] <= 0:
            raise ValueError("Invalid wall time")
        if any(not np.isfinite(v) or v < 0 for v in item["seconds"].values()):
            raise ValueError("Invalid component time")
        for step in item["steps"]:
            if not np.isclose(step["dt"], args.dt, rtol=1e-10, atol=1e-14):
                raise ValueError("Timesteps do not match the controlled experiment")
            if "fld" in case and (step["newton_reason"] != "converged" or
                    not np.isfinite(step["newton_l2"]) or
                    not np.isfinite(step["newton_max"]) or
                    max(step["newton_l2"], step["newton_max"]) > 1e-8):
                raise ValueError("FLD did not meet strict convergence criteria")
    seconds = {key: {"min": min(r["seconds"][key] for r in raw),
                     "mean": float(np.mean([r["seconds"][key] for r in raw])),
                     "max": max(r["seconds"][key] for r in raw)}
               for key in raw[0]["seconds"]}
    wall = max(r["wall_seconds"] for r in raw)
    mean_wall = float(np.mean([r["wall_seconds"] for r in raw]))
    implicit = sum(v["mean"] for k, v in seconds.items() if k.startswith("fld_"))
    iterations = [s["newton_iterations"] for s in raw[0]["steps"]]
    if "fld" in case and sum(iterations) == 0:
        raise ValueError("Trivial FLD measurement: no Newton iterations")
    history = np.atleast_2d(np.loadtxt(directory / "benchmark.hst"))
    mass_drift = float((history[-1, 2] - history[0, 2]) / history[0, 2])
    if not np.isfinite(history).all() or abs(mass_drift) > 1e-5:
        raise ValueError("Nonfinite history or excessive mass drift")
    return {"wall_seconds": wall, "seconds_per_step": wall / args.steps,
            "zone_cycles_per_second": sum(s["global_cells"] for s in raw[0]["steps"]) / wall,
            "parts": seconds, "implicit_fld_fraction_rank_mean": implicit / mean_wall,
            "newton_iterations_mean": float(np.mean(iterations)) if "fld" in case else None,
            "newton_iterations_total": sum(iterations) if "fld" in case else None,
            "mass_relative_drift": mass_drift, "minimum_rho": float(fields["rho"].min()),
            "minimum_pressure": float(fields["press"].min()),
            "short_window": wall < 10.0,
            "rank_mean_unassigned_seconds": mean_wall - sum(v["mean"] for v in seconds.values())}


def plot_summary(root, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    good = [r for r in runs if r["status"] == "passed"]
    if not good:
        return
    groups = sorted({(r["case"], r["ranks"]) for r in good})
    fig, ax = plt.subplots(figsize=(max(8, len(groups)*0.7), 5))
    bottom = np.zeros(len(groups))
    for name in good[0]["metrics"]["parts"]:
        values = np.array([np.median([r["metrics"]["parts"][name]["mean"] /
                            r["measured_steps"] for r in good
                            if (r["case"], r["ranks"]) == g]) for g in groups])
        ax.bar(np.arange(len(groups)), values, bottom=bottom, label=name)
        bottom += values
    ax.set_xticks(np.arange(len(groups)), [f"{c}\np={p}" for c, p in groups], rotation=45, ha="right")
    ax.set_ylabel("Rank-mean exclusive seconds / step (median of repeats)")
    ax.set_title("FLD implicit is separate; radiation transport remains in explicit_integrator")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "timing_breakdown.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for case in sorted({r["case"] for r in good}):
        ranks = sorted({r["ranks"] for r in good if r["case"] == case})
        values = [[r["metrics"]["seconds_per_step"] for r in good
                   if r["case"] == case and r["ranks"] == p] for p in ranks]
        med = np.array([np.median(v) for v in values])
        ax.errorbar(ranks, med, yerr=[med-np.array([min(v) for v in values]),
                    np.array([max(v) for v in values])-med], marker="o", label=case)
    ax.set(xlabel="MPI ranks (see hardware.json for SMT topology)", ylabel="Max-rank seconds / step",
           yscale="log", title="Measured scaling: median and full repeat range")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "scaling.png", dpi=150)
    plt.close(fig)


def aggregate(root, runs, mode):
    good = [r for r in runs if r["status"] == "passed"]
    rows = []
    for case, ranks in sorted({(r["case"], r["ranks"]) for r in good}):
        group = [r for r in good if r["case"] == case and r["ranks"] == ranks]
        times = [r["metrics"]["seconds_per_step"] for r in group]
        rows.append({"case": case, "ranks": ranks, "repeats": len(group),
                     "seconds_per_step_median": float(np.median(times)),
                     "seconds_per_step_min": min(times), "seconds_per_step_max": max(times),
                     "implicit_fld_fraction_mean": float(np.mean([
                         r["metrics"]["implicit_fld_fraction_rank_mean"] for r in group])),
                     "all_windows_at_least_10s": all(not r["metrics"]["short_window"] for r in group)})
    for row in rows:
        base = min((r for r in rows if r["case"] == row["case"]), key=lambda r: r["ranks"])
        row["efficiency"] = base["seconds_per_step_median"] / row["seconds_per_step_median"]
        if mode == "strong":
            row["efficiency"] *= base["ranks"] / row["ranks"]
        row["baseline_ranks"] = base["ranks"]
    if rows:
        with (root / "summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", nargs="+", choices=CASES, default=["hd", "hd_fld"])
    p.add_argument("--ranks", nargs="+", type=int, choices=[1, 2, 4, 8], default=[1, 2, 4])
    p.add_argument("--resolution", type=int, default=32)
    p.add_argument("--block", type=int, default=16)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--dt", type=float, default=1e-4)
    p.add_argument("--mode", choices=["strong", "weak"], default="strong")
    p.add_argument("--build-jobs", type=int, default=4)
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--hdf5-path", type=Path)
    p.add_argument("--output-root", type=Path, default=Path.cwd())
    p.add_argument("--build-root", type=Path, help="Reuse builds from an earlier suite; records their original provenance")
    p.add_argument("--compare-timer", action="store_true", help="Repeat each timed run with timing disabled; compare final arrays to roundoff")
    p.add_argument("--bind-to", choices=["core", "hwthread", "none"], default="core")
    args = p.parse_args()
    if (args.block < 4 or args.block & (args.block-1) or args.resolution < args.block or
            args.resolution % args.block or args.steps < 1 or args.warmup < 0 or
            args.repeat < 1 or not 1 <= args.build_jobs <= 8 or args.dt <= 0 or
            not np.isfinite(args.dt) or args.timeout <= 0):
        p.error("Invalid grid, steps, repeat, dt, timeout or build-jobs (1..8)")
    for ranks in args.ranks:
        blocks = (args.resolution // args.block)**3 * (ranks if args.mode == "weak" else 1)
        if blocks < ranks or blocks % ranks:
            p.error("Require at least one MeshBlock/rank and even block distribution")
    repo = Path(__file__).resolve().parents[2]
    root = args.output_root.resolve() / datetime.now(ZoneInfo("Asia/Tokyo")).strftime("performance_run_%Y%m%d_%H%M%S_%f_JST")
    root.mkdir(parents=True)
    print(root, flush=True)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    manifest = base_manifest(repo, "performance", max(args.ranks + [args.build_jobs]))
    manifest["options"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    manifest["runs"] = []
    manifest["builds"] = {}
    (root / "source.diff").write_text(subprocess.check_output(["git", "diff", "HEAD"], cwd=repo, text=True))
    shutil.copy2(__file__, root / "performance.py.archive")
    # Freeze once: concurrent edits must not change the physics between builds.
    if not args.build_root:
        prepare_build_source(repo, root / "source_snapshot")
        manifest["source_snapshot_sha256"] = tree_sha256(root / "source_snapshot")
    hardware = {"lscpu": subprocess.check_output(["lscpu"], text=True),
                "affinity": sorted(os.sched_getaffinity(0)),
                "mpi": subprocess.check_output(["mpirun", "--version"], text=True)}
    write_manifest(root / "hardware.json", hardware)
    write_manifest(root / "manifest.json", manifest)
    failed = False
    try:
        for case in dict.fromkeys(args.cases):
            build = root / "builds" / case
            records = []
            if args.build_root:
                source = args.build_root.resolve() / "builds" / case
                info = json.loads((source / "build.json").read_text())
                binary = source / "build_source" / "bin" / "athena"
                if sha256_file(binary) != info["binary_sha256"]:
                    raise ValueError(f"Binary changed since original build: {binary}")
                info = {**info, "reused_from": str(source)}
            else:
                source = build / "build_source"
                prepare_build_source(root / "source_snapshot", source)
                info = {"commands": records, "source_sha256": tree_sha256(source),
                        "git": manifest["git"]}
                manifest["builds"][case] = info
                write_manifest(root / "manifest.json", manifest)
                command = [sys.executable, "configure.py", "--prob=nrfld_performance",
                           "-mpi", "-hdf5", "-h5double", "--cflag=-O3"]
                if args.hdf5_path:
                    command.append(f"--hdf5_path={args.hdf5_path.resolve()}")
                if "fld" in case:
                    command.append("-nrmgfld")
                if case.startswith("mhd"):
                    command.append("-b")
                if "gravity" in case:
                    command.append("--grav=mg")
                print(f"Building {case}", flush=True)
                execute(command, source, build / "configure.log", args.timeout, records)
                execute(["make", f"-j{args.build_jobs}"], source, build / "build.log", args.timeout, records)
                binary = source / "bin" / "athena"
                info["binary_sha256"] = sha256_file(binary)
                write_manifest(build / "build.json", info)
            manifest["builds"][case] = info
            for ranks in dict.fromkeys(args.ranks):
                for repeat in range(args.repeat):
                    directory = root / f"{case}_p{ranks}_r{repeat}"
                    directory.mkdir()
                    run = {"case": case, "ranks": ranks, "repeat": repeat, "path": str(directory),
                           "measured_steps": args.steps, "commands": [], "status": "running"}
                    manifest["runs"].append(run)
                    try:
                        (directory / "input.used").write_text(input_text(args, ranks, True))
                        run["input_sha256"] = sha256_file(directory / "input.used")
                        command = ["mpirun", "--bind-to", args.bind_to]
                        if args.bind_to == "hwthread":
                            command += ["--use-hwthread-cpus"]
                        command += ["--report-bindings", "-np", str(ranks), str(binary), "-i", "input.used"]
                        run["elapsed_seconds"] = execute(command, directory, directory / "run.log",
                                                          args.timeout, run["commands"])
                        run["metrics"] = analyse(directory, args, ranks, case)
                        if args.compare_timer:
                            control = directory / "timer_off"
                            control.mkdir()
                            (control / "input.used").write_text(input_text(args, ranks, False))
                            off_time = execute(command, control, control / "run.log", args.timeout, run["commands"])
                            if "FATAL" in (control / "run.log").read_text():
                                raise ValueError("Timer-off control failed")
                            left, right = final_fields(directory), final_fields(control)
                            comparison = compare_fields(left, right)
                            # Summary records are emitted even at diagnostic_verbosity=0.
                            pattern = r"\[NR\] status=(\w+) iterations=(\d+)"
                            same_iterations = re.findall(pattern, (directory / "run.log").read_text()) == re.findall(
                                pattern, (control / "run.log").read_text())
                            run["timer_control"] = {**comparison,
                                "newton_status_and_iterations_identical": same_iterations,
                                "end_to_end_ratio_on_over_off": run["elapsed_seconds"] / off_time,
                                "note": "Includes MPI startup and output; not an isolated timer overhead estimate"}
                            if not comparison["within_roundoff"] or not same_iterations:
                                raise ValueError("Timer changes fields beyond roundoff or Newton iterations")
                        run["status"] = "passed"
                        print(f"{case} p={ranks} repeat={repeat}: {run['metrics']['seconds_per_step']:.6g} s/step", flush=True)
                    except Exception as error:
                        run.update(status="failed", error=str(error))
                        failed = True
                        print(f"FAILED {directory}: {error}", flush=True)
                    finally:
                        write_manifest(root / "manifest.json", manifest)
        plot_summary(root, manifest["runs"])
    except BaseException as error:
        failed = True
        manifest["error"] = repr(error)
        raise
    finally:
        manifest["status"] = "failed" if failed else "passed"
        write_manifest(root / "manifest.json", manifest)
        rows = aggregate(root, manifest["runs"], args.mode)
        write_manifest(root / "summary.json", {"status": manifest["status"],
                       "aggregates": rows, "runs": manifest["runs"]})
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
