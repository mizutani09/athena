#!/usr/bin/env python3
"""Gaussian diffusion: fixed diffusion number, spatial and temporal convergence.

Uses the existing nrfld_diff pgen and the isolated-build/provenance helpers.
No evolution solver or problem generator is modified by this validation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from run_suite import (build_case, binary_configuration, checked, effective_settings,
                       validate_binary_configuration, CommandExecutionError)
from nrfld_validation.io import find_repo_root, latest_athdf, read_athdf
from nrfld_validation.manifest import base_manifest, sha256_file, utc_now, write_manifest


def gaussian(x, t):
    # Code units and point samples match src/pgen/nrfld_diff.cpp exactly.
    chi = 2.99792458e10 / (3 * 1e-7 * 0.4) * 4000 / (4e10)**2
    t0 = (4e10)**2 * 1e-7 * 0.4 / (4 * 2.99792458e10) / 4000 * 0.1
    return 1e5 / np.sqrt(4 * np.pi * chi * (t + t0)) * np.exp(
        -(x - 0.5)**2 / (4 * chi * (t + t0)))


def relative_l1(value, reference):
    # Integral-relative L1; does not overweight nearly empty Gaussian tails.
    return float(np.sum(np.abs(value - reference)) / np.sum(np.abs(reference)))


def orders(errors, ratio):
    return [math.log(a / b) / math.log(ratio) for a, b in zip(errors, errors[1:])]


def run_study(args):
    repo = find_repo_root()
    root = args.output_root.expanduser().resolve() / args.run_name
    root.mkdir(parents=True, exist_ok=False)
    manifest = base_manifest(repo, "diffusion_convergence", args.max_cores)
    manifest["analysis_script_sha256"] = sha256_file(Path(__file__))
    records = []
    manifest["commands"]["records"] = records
    manifest["samples"] = []
    manifest_path = root / "manifest.json"
    write_manifest(manifest_path, manifest)
    try:
        if args.binary:
            exe = args.binary.expanduser().resolve()
            manifest["provenance"]["calculation"] = {
                "source": "existing_binary", "git": None}
        else:
            build = build_case(repo, root, "nrfld_diff", args.max_cores,
                               args.build_timeout, records)
            exe = build["build_source"] / "bin/athena"
            manifest["provenance"]["calculation"] = {
                "source": "local_isolated_build", "git": manifest["git"],
                "source_sha256": build["build_source_sha256"]}
        config = binary_configuration(exe, root, 30, records)
        validate_binary_configuration(config, "nrfld_diff", exe)
        manifest["provenance"]["calculation"].update(
            binary=str(exe), binary_sha256=sha256_file(exe), configuration=config)
        input_path = repo / "inputs/radiation/athinput.nrfld_diff"
        shutil.copy2(input_path, root / "input.used")
        manifest["input"] = {"source": str(input_path), "sha256": sha256_file(input_path)}
        chi = 2.99792458e10 / (3 * 1e-7 * 0.4) * 4000 / (4e10)**2
        cs = math.sqrt(1.666666666667 * 100.0)
        cache = {}

        def sample(n, steps):
            if (n, steps) in cache:
                return cache[n, steps]
            case = root / f"n{n}_steps{steps}"
            case.mkdir()
            output = case / "output"
            output.mkdir()
            dt = args.final_time / steps
            cfl = dt * cs * n
            if not 0 < cfl <= 0.3:
                raise ValueError(f"Requested timestep gives invalid CFL {cfl}")
            # Thin, transverse-periodic 3-D box: cubic cells/MeshBlocks as required
            # by MG, but only 64*N active cells for a transversely uniform solution.
            overrides = [
                f"mesh/nx1={n}", "mesh/nx2=8", "mesh/nx3=8",
                "meshblock/nx1=8", "meshblock/nx2=8", "meshblock/nx3=8",
                f"mesh/x2max={8/n:.17g}", f"mesh/x3max={8/n:.17g}",
                f"time/cfl_number={cfl:.17g}", f"time/tlim={args.final_time:.17g}",
                "time/nlim=-1", "time/ncycle_out=1",
                "fld/fixed_flux_limiter=true", "nrfld/threshold=1e-12",
                "nrfld/nr_threshold=1e-11", "nrfld/diagnostic_verbosity=1",
                "nrfld/failure_policy=abort", "job/problem_id=diff_convergence",
                "output1/data_format=%.16e", "output1/dcycle=1",
                "output3/dcycle=0", f"output3/dt={args.final_time:.17g}",
                "output3/ghost_zones=false", f"output3/x2_slice={4/n:.17g}",
                f"output3/x3_slice={4/n:.17g}",
                "output4/dt=-1", "output4/dcycle=0",
                "output5/dt=-1", "output5/dcycle=0",
            ]
            if args.linear_divergence_patience is not None:
                overrides.extend([
                    "nrfld/linear_divergence_patience="
                    f"{args.linear_divergence_patience}",
                    "nrfld/linear_max_transient_growth="
                    f"{args.linear_max_transient_growth:.17g}",
                ])
            ranks = min(args.max_cores, n // 8)
            settings = effective_settings(input_path, overrides)
            sections = {}
            for key, value in settings.items():
                section, name = key.split("/", 1)
                sections.setdefault(section, []).append(f"{name} = {value}")
            # Athena's command-line parser cannot create missing parameters.
            # Materialize every setting in this case's input instead.
            case_input = case / "input.used"
            case_input.write_text("\n\n".join(
                f"<{section}>\n" + "\n".join(values)
                for section, values in sections.items()) + "\n")
            command = ["mpirun", "-np", str(ranks), str(exe), "-i",
                       str(case_input), "-d", str(output)]
            entry = {"resolution": n, "steps_target": steps, "cfl": cfl,
                     "dt_target": dt, "diffusion_number_target": chi * dt * n*n,
                     "ranks": ranks, "settings": settings,
                     "input_sha256": sha256_file(case_input),
                     "directory": str(case)}
            manifest["samples"].append(entry)
            write_manifest(manifest_path, manifest)
            print(f"Running N={n}, steps={steps}, CFL={cfl:.6g}", flush=True)
            record = checked(command, root, case / "run.log", args.run_timeout, "run")
            records.append(record)
            text = (case / "run.log").read_text()
            dt_values = [float(d) for d in re.findall(
                r"^cycle=\d+ time=[^ ]+ dt=([^\s]+)", text, re.M)]
            active = [line for line in text.splitlines()
                      if line.startswith("[NR] status=") and "stage=0 " in line]
            if not active or any("status=converged " not in line for line in active):
                raise RuntimeError(f"Nonconverged or missing Newton solves: {case}")
            if not dt_values or not np.isclose(dt_values[0], dt, rtol=1e-8):
                raise RuntimeError(f"Actual timestep differs from requested timestep: {case}")
            solve_dt = [float(re.search(r"\bdt=(\S+)", line)[1]) for line in active]
            # Roundoff can leave one negligible last step when tlim is reached.
            if (len(solve_dt) not in (steps, steps+1)
                    or not np.allclose(solve_dt[:steps], dt, rtol=1e-8, atol=1e-14)
                    or not np.isclose(sum(solve_dt), args.final_time, rtol=1e-10)
                    or (len(solve_dt) > steps and solve_dt[-1] > 1e-8*dt)):
                raise RuntimeError(f"Unexpected timestep sequence: {case}")
            for line in active:
                for norm in ("final_l2", "final_max"):
                    value = float(re.search(rf"\b{norm}=(\S+)", line)[1])
                    if not np.isfinite(value) or value > 1e-11:
                        raise RuntimeError(f"Newton tolerance not met: {case}")
            data = read_athdf(latest_athdf(output, "uov_x_slice"))
            x = np.asarray(data["x1v"])
            y = np.asarray(data["E_rad"], dtype=float).reshape(-1, n).mean(axis=0) / 1e7
            # Default Athena HDF5 stores Time as float32. Check the full-precision
            # termination log as well as the rounded output timestamp.
            end_times = re.findall(r"^time=([^ ]+) cycle=\d+", text, re.M)
            time = float(end_times[-1]) if end_times else float("nan")
            if (not np.isclose(time, args.final_time, rtol=1e-10, atol=1e-14)
                    or not np.isclose(float(data["Time"]), time, rtol=2e-7)):
                raise RuntimeError(f"Did not reach the common final time: {time}")
            if not np.all(np.isfinite(y)) or np.any(y <= 0):
                raise RuntimeError("Nonfinite/nonpositive radiation energy")
            ref = gaussian(x, time)
            entry.update(dt_measured=dt_values[0], dt_min=min(solve_dt), dt_max=max(solve_dt),
                         timestep_sum=sum(solve_dt), diffusion_number=chi*dt_values[0]*n*n,
                         final_time=time, newton_solves=len(active),
                         relative_l1=relative_l1(y, ref), x=x.tolist(), profile=y.tolist())
            cache[n, steps] = y, ref, entry
            write_manifest(manifest_path, manifest)
            return cache[n, steps]

        fixed_errors, spatial_errors = [], []
        resolutions = [16, 32, 64]
        for n in resolutions:
            steps = 40 * (n // 16)**2
            y, ref, _ = sample(n, steps)
            half, _, _ = sample(n, 2*steps)
            fixed_errors.append(relative_l1(y, ref))
            # Cancel leading backward-Euler error at EACH fixed grid, then test dx.
            spatial_errors.append(relative_l1(2*half - y, ref))
        temporal_profiles = [sample(32, steps)[0] for steps in (40, 80, 160, 320)]
        temporal_differences = [relative_l1(a, b) for a, b in
                                zip(temporal_profiles, temporal_profiles[1:])]
        fixed_orders_x = orders(fixed_errors, 2)
        fixed_orders_t = orders(fixed_errors, 4)
        spatial_orders = orders(spatial_errors, 2)
        temporal_orders = orders(temporal_differences, 2)
        metrics = {
            "resolutions": resolutions,
            "fixed_diffusion_number": {"relative_l1": fixed_errors,
                "order_in_dx": fixed_orders_x, "order_in_dt": fixed_orders_t,
                "interpretation": "coupled refinement; not an independent temporal-order proof"},
            "space_time_extrapolated": {"relative_l1": spatial_errors,
                "order_in_dx": spatial_orders},
            "time_fixed_grid": {"resolution": 32, "steps": [40,80,160,320],
                "successive_profile_differences": temporal_differences,
                "order_in_dt": temporal_orders},
        }
        finite = all(np.isfinite(v) for seq in (fixed_errors, spatial_errors,
                     temporal_differences, spatial_orders, temporal_orders) for v in seq)
        metrics["passed"] = bool(finite and all(1.7 < p < 2.3 for p in spatial_orders)
                                 and all(0.8 < p < 1.2 for p in temporal_orders)
                                 and all(1.7 < p < 2.3 for p in fixed_orders_x))
        manifest["metrics"] = metrics
        manifest["status"] = "passed" if metrics["passed"] else "failed"
        manifest["termination"] = {"reason": "completed" if metrics["passed"]
                                   else "validation_failed", "reached_time": args.final_time}
        (root / "summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
        if not args.no_plots:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            dx = 1 / np.array(resolutions, dtype=float)
            axes[0].loglog(dx, fixed_errors, "o-", label="fixed diffusion number")
            axes[0].loglog(dx, spatial_errors, "s-", label="time-extrapolated")
            axes[0].loglog(dx, spatial_errors[-1]*(dx/dx[-1])**2, "k--", label="order 2")
            axes[0].set(xlabel="dx", ylabel="relative L1 error")
            dt = args.final_time / np.array([40,80,160])
            axes[1].loglog(dt, temporal_differences, "o-", label="fixed N=32 differences")
            axes[1].loglog(dt, temporal_differences[-1]*dt/dt[-1], "k--", label="order 1")
            axes[1].set(xlabel="dt", ylabel="successive-solution relative L1")
            for ax in axes:
                ax.legend(fontsize=8)
                ax.grid(True, which="both", alpha=0.25)
            fig.tight_layout()
            fig.savefig(root / "convergence.png", dpi=160)
            plt.close(fig)
        print(json.dumps(metrics, indent=2))
    except Exception as error:
        if isinstance(error, CommandExecutionError):
            records.append(error.record)
        manifest.update(status="failed", error=str(error))
        manifest["termination"] = {"reason": "failed"}
        raise
    finally:
        manifest["finished_at"] = utc_now()
        write_manifest(manifest_path, manifest)
    return 0 if manifest["status"] == "passed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-cores", type=int, default=2)
    parser.add_argument("--binary", type=Path, help="Reuse a verified nrfld_diff MPI binary")
    parser.add_argument("--output-root", type=Path, default=Path("validation"))
    parser.add_argument("--run-name", default="diff_convergence_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--build-timeout", type=float, default=900)
    parser.add_argument("--run-timeout", type=float, default=300)
    parser.add_argument("--final-time", type=float, default=0.02)
    parser.add_argument("--linear-divergence-patience", type=int)
    parser.add_argument("--linear-max-transient-growth", type=float, default=2.0)
    parser.add_argument("--no-plots", action="store_true")
    arguments = parser.parse_args()
    if (arguments.max_cores < 1 or arguments.final_time <= 0
            or (arguments.linear_divergence_patience is not None
                and arguments.linear_divergence_patience < 1)
            or arguments.linear_max_transient_growth < 1):
        parser.error("max-cores and final-time must be positive")
    os.environ["OMP_NUM_THREADS"] = "1"
    raise SystemExit(run_study(arguments))
