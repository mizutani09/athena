#!/usr/bin/env python3
"""Extend a Marshak report with time-extrapolated spatial convergence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nrfld-validation")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import ticker  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from marshak_validation import (error_metrics, materialize_input, profile,
                                save_figure, write_report)  # noqa: E402
from nrfld_validation.io import find_repo_root, read_athdf  # noqa: E402
from nrfld_validation.manifest import (base_manifest, sha256_file, tree_sha256,
                                       utc_now, write_manifest)  # noqa: E402
from run_suite import (CommandExecutionError, binary_configuration, checked,
                       prepare_build_source)  # noqa: E402

BASE_DELTA_TAU = 3.0e-4
HALF_DELTA_TAU = 1.5e-4
FINAL_TAU = 0.3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--resolutions", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--max-cores", type=int, default=8)
    parser.add_argument("--build-timeout", type=float, default=900)
    parser.add_argument("--run-timeout", type=float, default=1800)
    args = parser.parse_args()
    if (args.max_cores < 1 or len(args.resolutions) < 3
            or any(n < 32 or n % 32 for n in args.resolutions)
            or len(set(args.resolutions)) != len(args.resolutions)):
        parser.error("use at least three distinct resolutions divisible by 32")
    args.resolutions.sort()
    return args


def input_overrides(n: int, steps: int, delta_tau: float, problem_id: str) -> list[str]:
    return [
        f"mesh/nx1={n}", "mesh/nx2=32", "mesh/nx3=32",
        "mesh/x2min=0", f"mesh/x2max={160/n:.17g}",
        "mesh/x3min=0", f"mesh/x3max={160/n:.17g}",
        "meshblock/nx1=32", "meshblock/nx2=32", "meshblock/nx3=32",
        f"time/nlim={steps}", "time/tlim=3.0", "time/ncycle_out=100",
        "output1/dcycle=20", "output3/dcycle=2000",
        f"output4/dcycle={steps}",
        f"output3/x2_slice={80/n:.17g}", f"output3/x3_slice={80/n:.17g}",
        f"output4/x2_slice={80/n:.17g}", f"output4/x3_slice={80/n:.17g}",
        f"problem/delta_tau={delta_tau:.17g}", "problem/Tgas0=1.0",
        "nrfld/linear_solver=fgmres", "nrfld/threshold=1e-11",
        "nrfld/nr_threshold=1e-10", "nrfld/diagnostic_verbosity=0",
        "nrfld/failure_policy=abort", f"job/problem_id={problem_id}",
    ]


def final_profile(output: Path, steps: int) -> dict:
    paths = sorted(output.glob("*.uov_x_final.*.athdf"))
    path = next((candidate for candidate in paths
                 if int(read_athdf(candidate)["NumCycles"]) == steps), None)
    if path is None:
        raise FileNotFoundError(f"No final cycle {steps} HDF5 in {output}")
    return profile(path, 0.1)


def base_profile(root: Path, n: int) -> dict[str, np.ndarray]:
    path = root/f"cases/n{n}/profile_final.csv"
    matrix = np.loadtxt(path, delimiter=",", skiprows=1)
    if matrix.shape != (n, 5):
        raise ValueError(f"Unexpected baseline CSV shape {matrix.shape}: {path}")
    return {"x": matrix[:, 0], "u": matrix[:, 1], "u_analytic": matrix[:, 2],
            "v": matrix[:, 3], "v_analytic": matrix[:, 4]}


def solver_checks(log: Path, steps: int, requested_dt: float) -> dict:
    content = log.read_text(encoding="utf-8", errors="replace")
    statuses = re.findall(r"\[NR\] status=([^\s]+)", content)
    active = [status for status in statuses if status != "dt_zero_initialization"]
    active_dt = [float(value) for value in re.findall(
        r"\[NR\] status=converged[^\n]*?stage=0 dt=([^\s]+)", content)]
    krylov = re.findall(r"\[LINEAR_FGMRES\].*?relative_residual=([^\s]+).*?target=([^\s]+)",
                        content)
    end = re.findall(r"^time=([^ ]+) cycle=(\d+)", content, re.M)
    final_time = float(end[-1][0]) if end else math.nan
    final_cycle = int(end[-1][1]) if end else -1
    return {"newton_steps": len(active), "fgmres_calls": len(krylov),
            "newton_ok": len(active) == steps and all(s == "converged" for s in active),
            "dt_min": min(active_dt) if active_dt else math.nan,
            "dt_max": max(active_dt) if active_dt else math.nan,
            "dt_ok": len(active_dt) == steps and all(math.isclose(value, requested_dt,
                                             rel_tol=1e-9, abs_tol=1e-13) for value in active_dt),
            "fgmres_ok": bool(krylov) and all(float(a) <= float(b)*1.000001
                                               for a, b in krylov),
            "final_time": final_time, "final_cycle": final_cycle,
            "time_ok": final_cycle == steps and math.isclose(final_time, 3.0,
                                                               rel_tol=1e-10, abs_tol=1e-12)}


def run_case(root: Path, repo: Path, source_input: Path, exe: Path,
             configuration: dict, source_hash: str, args: argparse.Namespace,
             n: int, steps: int, delta_tau: float, case: Path) -> tuple[dict, dict]:
    case.mkdir(parents=True, exist_ok=False)
    output = case/"output"
    output.mkdir()
    key = f"n{n}_dtau_{delta_tau:.1e}"
    manifest = base_manifest(repo, key, args.max_cores)
    manifest["provenance"]["calculation"] = {
        "source_sha256": source_hash, "binary_sha256": sha256_file(exe),
        "binary_configuration": configuration}
    settings = materialize_input(source_input, case/"input.used",
                                 input_overrides(n, steps, delta_tau, key))
    manifest["input"] = {"source_path": str(source_input),
                         "source_sha256": sha256_file(source_input),
                         "used_sha256": sha256_file(case/"input.used"),
                         "effective_settings": settings}
    ranks = min(args.max_cores, n//32)
    manifest["resources"] = {"mpi_ranks": ranks, "omp_threads": 1}
    command = ["/usr/bin/time", "-v", "-o", str(case/"resources.txt"),
               "mpirun", "-np", str(ranks), str(exe), "-i", str(case/"input.used"),
               "-d", str(output)]
    manifest["commands"]["run"] = command
    records = manifest["commands"].setdefault("records", [])
    write_manifest(case/"manifest.json", manifest)
    print(f"Running {key}: {steps} steps, MPI={ranks}", flush=True)
    try:
        record = checked(command, case, case/"run.log", args.run_timeout, "run")
        records.append(record)
        checks = solver_checks(case/"run.log", steps, delta_tau/0.1)
        data = final_profile(output, steps)
        if data["cycles"] != steps or not math.isclose(data["time"], 3.0, rel_tol=1e-6):
            raise RuntimeError(f"Wrong HDF5 final time/cycle: {data['time']}, {data['cycles']}")
        checks["finite_nonnegative"] = all(
            np.all(np.isfinite(data[field])) and np.all(data[field] >= 0)
            for field in ("u", "v"))
        rss_match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)",
                              (case/"resources.txt").read_text())
        rss = int(rss_match.group(1)) if rss_match else None
        manifest["resources"]["max_rss_kib"] = rss
        checks["wall_seconds"] = record["elapsed_seconds"]
        checks["max_rss_kib"] = rss
        checks["passed"] = bool(checks["newton_ok"] and checks["fgmres_ok"]
                                and checks["dt_ok"]
                                and checks["time_ok"] and checks["finite_nonnegative"])
        result = {"status": "passed" if checks["passed"] else "failed", "checks": checks}
        manifest["metrics"] = checks
        print(f"  {result['status']}: NR={checks['newton_steps']}/{steps}, "
              f"FGMRES={checks['fgmres_calls']}", flush=True)
    except Exception as error:
        if isinstance(error, CommandExecutionError):
            records.append(error.record)
        result = {"status": "failed", "error": str(error)}
        data = {}
        manifest["error"] = str(error)
        print(f"  failed: {error}", flush=True)
    manifest["status"] = result["status"]
    manifest["finished_at"] = utc_now()
    write_manifest(case/"manifest.json", manifest)
    return result, data


def analyze(root: Path, study: dict, fine_profiles: dict[int, dict]) -> None:
    rows = []
    resolutions = sorted(fine_profiles)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for axis, field in zip(axes, ("u", "v")):
        errors = {"base": [], "half_dt": [], "extrapolated": []}
        for i, n in enumerate(resolutions):
            base = base_profile(root, n)
            fine = fine_profiles[n]
            if not np.allclose(base["x"], fine["x"], rtol=2e-6, atol=1e-8):
                raise ValueError(f"N={n} baseline and half-dt grids differ")
            numeric = fine[field]
            reference = base[f"{field}_analytic"]
            extrapolated = 2*numeric - base[field]
            b = error_metrics(base[field], reference)["l1_relative"]
            h = error_metrics(numeric, reference)["l1_relative"]
            e = error_metrics(extrapolated, reference)["l1_relative"]
            for key, value in (("base", b), ("half_dt", h), ("extrapolated", e)):
                errors[key].append(value)
            order = (math.log(errors["extrapolated"][i-1]/e)
                     /math.log(n/resolutions[i-1])) if i else None
            rows.append({"resolution": n, "field": field,
                         "baseline_relative_l1": b, "half_relative_l1": h,
                         "extrapolated_relative_l1": e,
                         "order_from_previous": order})
            profile_path = root/f"spatial/cases/n{n}/profile_final.csv"
            if field == "u":
                with profile_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(("x_dimless", "u_baseline", "u_half_dt", "u_extrapolated",
                                     "u_analytic", "v_baseline", "v_half_dt", "v_extrapolated",
                                     "v_analytic"))
                    writer.writerows(zip(base["x"], base["u"], fine["u"],
                                         2*fine["u"]-base["u"], base["u_analytic"],
                                         base["v"], fine["v"],
                                         2*fine["v"]-base["v"], base["v_analytic"]))
        for key, label, style in (("base", "Δt", "o-"),
                                  ("half_dt", "Δt/2", "s-"),
                                  ("extrapolated", "2U(Δt/2)−U(Δt)", "D-")):
            axis.loglog(resolutions, errors[key], style, label=label)
        reference_line = errors["extrapolated"][0] * (
            np.array(resolutions)/resolutions[0])**-2
        axis.loglog(resolutions, reference_line, "k--", label="N⁻² reference")
        axis.set(title=f"{field}, τ=0.3", xlabel="N in x₁", ylabel="relative L1")
        axis.xaxis.set_major_locator(ticker.FixedLocator(resolutions))
        axis.xaxis.set_major_formatter(ticker.FixedFormatter([str(n) for n in resolutions]))
        axis.xaxis.set_minor_formatter(ticker.NullFormatter())
        axis.grid(True, which="both", alpha=.25)
        axis.legend(fontsize=9)
    save_figure(fig, root/"figures/spatial_convergence")
    with (root/"spatial/convergence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("resolution", "field",
                                                   "baseline_relative_l1", "half_relative_l1",
                                                   "extrapolated_relative_l1", "order_from_previous"))
        writer.writeheader()
        writer.writerows(rows)
    study["rows"] = rows
    study["orders"] = {field: [row["order_from_previous"] for row in rows
                                if row["field"] == field and row["order_from_previous"] is not None]
                       for field in ("u", "v")}
    study["order_window"] = [1.7, 2.3]
    study["order_passed"] = all(1.7 <= order <= 2.3
                                for sequence in study["orders"].values() for order in sequence)


def main() -> int:
    args = parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    repo = find_repo_root(HERE)
    root = args.report_root.expanduser().resolve()
    summary_path = root/"summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "passed" or summary.get("steps") != 1000:
        raise RuntimeError("Expected a passed 1000-step Marshak baseline report")
    for n in args.resolutions:
        if f"n{n}" not in summary["cases"] or summary["cases"][f"n{n}"]["status"] != "passed":
            raise RuntimeError(f"Missing passed baseline N={n}")
    spatial_dir = root/"spatial"
    spatial_dir.mkdir(exist_ok=False)
    study = {"status": "running", "started_at": utc_now(),
             "resolutions": args.resolutions, "base_delta_tau": BASE_DELTA_TAU,
             "half_delta_tau": HALF_DELTA_TAU, "final_tau": FINAL_TAU,
             "cases": {}, "rows": []}
    summary["spatial_study"] = study
    summary["status"] = "running"
    summary_path.write_text(json.dumps(summary, indent=2)+"\n")
    write_report(root, summary)
    manifest = base_manifest(repo, "marshak_spatial", args.max_cores)
    records = manifest["commands"].setdefault("records", [])
    manifest["analysis_script_sha256"] = sha256_file(Path(__file__))
    build_dir = spatial_dir/"build"
    build_dir.mkdir()
    try:
        source = build_dir/"build_source"
        prepare_build_source(repo, source)
        source_hash = tree_sha256(source/"src")
        cfg = ["python3", str(source/"configure.py"), "--prob=nrfld_marshak",
               "--eos=general/marshak", "-nrmgfld", "-hdf5", "-mpi", "--cflag=-O3"]
        records.append(checked(cfg, source, build_dir/"configure.log", args.build_timeout,
                               "configure"))
        records.append(checked(["make", f"-j{args.max_cores}"], source,
                               build_dir/"build.log", args.build_timeout, "build"))
        exe = source/"bin/athena"
        configuration = binary_configuration(exe, build_dir, 30, records)
        for key, expected in {"problem_generator": "nrfld_marshak",
                              "equation_of_state": "general/marshak",
                              "fld_with_newton-raphson": "ON", "mpi_parallelism": "ON",
                              "hdf5_output": "ON"}.items():
            if configuration.get(key) != expected:
                raise RuntimeError(f"Binary {key}={configuration.get(key)!r}; expected {expected!r}")
        manifest["provenance"]["calculation"] = {
            "source_sha256": source_hash, "binary_sha256": sha256_file(exe),
            "binary_configuration": configuration}
        source_input = repo/"inputs/radiation/athinput.nrfld_marshak"
        probe_n = args.resolutions[0]
        probe_result, probe_data = run_case(root, repo, source_input, exe, configuration,
                                            source_hash, args, probe_n, 1000, BASE_DELTA_TAU,
                                            spatial_dir/"baseline_probe")
        if probe_result["status"] != "passed":
            raise RuntimeError(f"Baseline-equivalence probe failed: {probe_result}")
        base = base_profile(root, probe_n)
        probe_difference = {field: error_metrics(probe_data[field], base[field])["l1_relative"]
                            for field in ("u", "v")}
        study["baseline_equivalence"] = {"resolution": probe_n,
                                          "relative_difference": probe_difference,
                                          "passed": all(value < 1e-7 for value in
                                                        probe_difference.values())}
        if not study["baseline_equivalence"]["passed"]:
            raise RuntimeError(f"New binary does not reproduce baseline: {probe_difference}")
        print(f"Baseline equivalence: {probe_difference}", flush=True)
        fine_profiles = {}
        for n in args.resolutions:
            case = spatial_dir/"cases"/f"n{n}"
            result, data = run_case(root, repo, source_input, exe, configuration,
                                    source_hash, args, n, 2000, HALF_DELTA_TAU, case)
            study["cases"][f"n{n}"] = result
            if result["status"] == "passed":
                fine_profiles[n] = data
            summary_path.write_text(json.dumps(summary, indent=2)+"\n")
            write_report(root, summary)
        if len(fine_profiles) != len(args.resolutions):
            raise RuntimeError("At least one half-dt calculation failed")
        analyze(root, study, fine_profiles)
        study["status"] = "passed" if study["order_passed"] else "order_not_confirmed"
    except Exception as error:
        if isinstance(error, CommandExecutionError):
            records.append(error.record)
        study["error"] = str(error)
        study["status"] = "failed"
        print(f"Spatial study failed: {error}", flush=True)
    study["finished_at"] = utc_now()
    summary["status"] = "passed" if study["status"] == "passed" else "failed"
    summary_path.write_text(json.dumps(summary, indent=2)+"\n")
    manifest["metrics"] = {"baseline_equivalence": study.get("baseline_equivalence"),
                           "orders": study.get("orders"), "order_passed": study.get("order_passed")}
    manifest["status"] = study["status"]
    manifest["finished_at"] = study["finished_at"]
    write_manifest(spatial_dir/"manifest.json", manifest)
    root_manifest_path = root/"manifest.json"
    root_manifest = json.loads(root_manifest_path.read_text())
    root_manifest["status"] = summary["status"]
    root_manifest["analysis_script_sha256"] = sha256_file(HERE/"marshak_validation.py")
    root_manifest["spatial_study"] = {"status": study["status"],
                                      "manifest": "spatial/manifest.json",
                                      "orders": study.get("orders")}
    write_manifest(root_manifest_path, root_manifest)
    write_report(root, summary)
    print(f"Report: {root}/index.html; spatial status={study['status']}", flush=True)
    return 0 if study["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
