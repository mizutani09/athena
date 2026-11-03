#!/usr/bin/env python3
"""Run the three Mach 5 diagnostic cases with one prebuilt executable.

The executable and analysis sources are copied into the result directory.
Simulations run sequentially; progress.json and case run logs show progress.
No simulation source or parent build file is modified.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

import h5py
import numpy as np


HERE = Path(__file__).resolve().parent
CYCLE = re.compile(r"cycle=(\d+) time=(\S+) dt=(\S+)")


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def make_input(base, nx, slab, cfl):
    # Preserve cubic cells by scaling the physically irrelevant periodic width.
    half_width = 3000. * slab / nx
    overrides = {
        "time": {"cfl_number": cfl, "nlim": -1, "ncycle_out": 100},
        "mesh": {"nx1": nx, "nx2": slab, "nx3": slab,
                 "x2min": -half_width, "x2max": half_width,
                 "x3min": -half_width, "x3max": half_width,
                 "refinement": "none", "numlevel": 1},
        "meshblock": {"nx1": slab, "nx2": slab, "nx3": slab},
        "problem": {"use_amr_refinement": "false"},
        "output1": {"dcycle": 1},
        "output2": {"dt": -1},
        "output3": {"dt": .002},
        "output4": {"dt": .01},
    }
    section = ""
    lines = []
    for line in base.splitlines():
        if line.startswith("<") and line.endswith(">"):
            section = line[1:-1]
        match = re.match(r"\s*(\w+)\s*=", line)
        if match and match[1] in overrides.get(section, {}):
            line = f"{match[1]:<24} = {overrides[section][match[1]]}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def scan_log(path):
    samples = []
    iterations = collections.Counter()
    max_residual = 0.
    failures = []
    statuses = collections.Counter()
    normal_end = False
    with path.open(errors="replace") as handle:
        for line in handle:
            match = CYCLE.search(line)
            if match:
                samples.append((int(match[1]), float(match[2]), float(match[3])))
            if "[NR] status=" in line:
                status = re.search(r"status=(\w+)", line)[1]
                statuses[status] += 1
                if status == "converged":
                    iterations[int(re.search(r"iterations=(\d+)", line)[1])] += 1
                    maximum = re.search(r" final_max=(\S+)", line)
                    if maximum:
                        max_residual = max(max_residual, float(maximum[1]))
                elif status not in ("dt_zero_initialization",):
                    failures.append(line.strip())
            if any(token in line for token in ("FATAL", "double free", "corrupted double", "Segmentation")):
                failures.append(line.strip())
            if "Terminating on time limit" in line:
                normal_end = True
    late_dt = [dt for _, t, dt in samples if .02 <= t < .0395]
    return {"last_cycle_record": samples[-1] if samples else None,
            "median_late_dt_s": float(np.median(late_dt)) if late_dt else None,
            "NR_iteration_histogram": dict(sorted(iterations.items())),
            "NR_status_counts": dict(statuses), "NR_max_final_residual": max_residual,
            "normal_time_limit_termination": normal_end, "failure_lines": failures[:20]}


def report(root, final, temporal):
    def optional(value):
        return "—" if value is None else f"{value:.4f}"

    cases = final["cases"]
    lines = ["# Mach 5 一様thin slab・最新commitの診断", "",
             f"commit: `{json.loads((root / 'build_provenance.json').read_text())['commit']}`", "",
             "solver・物理実装は変更していない。全3ケースは同じbinary・一様格子・周期thin slab。", "",
             "shockは密度ジャンプの中点で登録。Trad誤差はshock ±60 cm。bump高さはsemi-analyticへの正の残差、幅はその半値幅。", "",
             "| ケース | dx [cm] | 実dt中央値 [s] | Trad最大誤差 [%] | bump [K] | 位置 [cm] | 半値幅 [cm] | Tgas L1 [%] | density L1 [%] |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label in ("baseline", "resolution2", "dt_half"):
        c = cases[label]
        execution = json.loads((root / label / "execution.json").read_text())
        lines.append(f"| {label} | {c['shock_dx_cm']:.6g} | {execution['median_late_dt_s']:.6g} | "
                     f"{c['Trad_near_shock_max_abs_relative_percent']:.4f} | {c['bump_height_K']:.4f} | "
                     f"{optional(c['bump_peak_offset_from_shock_cm'])} | "
                     f"{optional(c['bump_fwhm']['width_cm'] if c['bump_fwhm'] else None)} | "
                     f"{c['agreement']['tgas']['global_relative_L1_percent']:.4f} | "
                     f"{c['agreement']['rho']['global_relative_L1_percent']:.4f} |")
    lines += ["", "## 直前出力の変動幅", "",
              "| ケース | 最終4出力のbump最小–最大 [K] | Trad最大誤差の最小–最大 [%] |",
              "|---|---:|---:|"]
    for label in ("baseline", "resolution2", "dt_half"):
        subset = [c for name, c in temporal["cases"].items() if name.startswith(label + "_")]
        heights = [c["bump_height_K"] for c in subset]
        errors = [c["Trad_near_shock_max_abs_relative_percent"] for c in subset]
        lines.append(f"| {label} | {min(heights):.4f}–{max(heights):.4f} | {min(errors):.4f}–{max(errors):.4f} |")
    baseline = cases["baseline"]["bump_height_K"]
    spatial_ratio = cases["resolution2"]["bump_height_K"] / baseline if baseline else None
    dt_ratio = cases["dt_half"]["bump_height_K"] / baseline if baseline else None
    lines += ["", f"baselineに対する最終bump高さ比：resolution2={spatial_ratio}、dt_half={dt_ratio}。", "",
              "依存性と論文採用の最終判定は、上記比と直前出力の変動幅・全体agreementを合わせて評価する。"
              "2解像度・2dtの結果だけで厳密な漸近収束を証明したとは書かない。", "",
              "[最終比較図](final_diagnostics/comparison.pdf) / [測定値](final_diagnostics/metrics.json) / "
              "[直前出力](stationarity_diagnostics/metrics.json)。"]
    (root / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--base-input", type=Path, required=True)
    parser.add_argument("--analytic", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--nx1", type=int, default=4096)
    parser.add_argument("--slab", type=int, default=4)
    parser.add_argument("--ranks", type=int, default=4)
    args = parser.parse_args()
    root = args.output_root.resolve()
    if args.slab < 4 or args.slab & (args.slab - 1) or args.nx1 % args.slab:
        raise ValueError("slab must be power of two >=4 and divide nx1")
    if not 1 <= args.ranks <= 4:
        raise ValueError("ranks must be 1..4")
    root.mkdir(parents=True, exist_ok=True)
    state = {"status": "initializing", "pid": os.getpid(), "cases": {}}
    write_json(root / "progress.json", state)
    try:
        binary = root / "athena"
        shutil.copy2(args.binary, binary)
        binary.chmod(0o755)
        shutil.copy2(args.analytic, root / "mach5_semianalytic.csv")
        frozen = root / "analysis_source"
        frozen.mkdir(exist_ok=False)
        for name in ("diagnose_mach5_convergence.py",):
            shutil.copy2(HERE / name, frozen / name)
        shutil.copytree(HERE / "plots", frozen / "plots", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(HERE / "nrfld_validation", frozen / "nrfld_validation", ignore=shutil.ignore_patterns("__pycache__"))
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env.pop("LD_LIBRARY_PATH", None)
        base = args.base_input.read_text()
        plan = {"binary_sha256": sha(binary), "analytic_sha256": sha(root / "mach5_semianalytic.csv"),
                "analysis_sha256": sha(frozen / "diagnose_mach5_convergence.py"),
                "nx1_baseline": args.nx1, "slab": args.slab, "ranks": args.ranks,
                "solver_physics_changes": False, "cases": {}}
        final_files = []
        recent_files = []
        specifications = (("baseline", args.nx1, .3),
                          ("dt_half", args.nx1, .15),
                          ("resolution2", args.nx1 * 2, .3))
        for label, nx, cfl in specifications:
            case = root / label
            case.mkdir(exist_ok=False)
            (case / "input.used").write_text(make_input(base, nx, args.slab, cfl))
            output = case / "output"
            output.mkdir()
            command = ["mpirun", "-np", str(args.ranks), str(binary), "-i",
                       str(case / "input.used"), "-d", str(output)]
            plan["cases"][label] = {"nx1": nx, "nx2": args.slab, "nx3": args.slab,
                                   "cfl": cfl, "dx_cm": 6000. / nx,
                                   "input_sha256": sha(case / "input.used"), "command": command}
            state["cases"][label] = {"status": "pending"}
        write_json(root / "plan.json", plan)
        # The cheaper timestep experiment precedes the more expensive fine grid.
        for label, nx, cfl in specifications:
            case = root / label
            output = case / "output"
            command = plan["cases"][label]["command"]
            state.update(status="running", active_case=label)
            state["cases"][label] = {"status": "running", "started_unix": time.time()}
            write_json(root / "progress.json", state)
            print(f"started {label}: nx={nx} slab={args.slab} CFL={cfl}", flush=True)
            start = time.monotonic()
            with (case / "run.log").open("w") as log:
                process = subprocess.Popen(command, cwd=case, env=env, stdout=log, stderr=subprocess.STDOUT)
                state["cases"][label]["mpirun_pid"] = process.pid
                while process.poll() is None:
                    with (case / "run.log").open("rb") as handle:
                        handle.seek(max(0, handle.seek(0, 2) - 100000))
                        tail = handle.read().decode(errors="replace")
                    matches = CYCLE.findall(tail)
                    if matches:
                        cycle, t, dt = matches[-1]
                        state["cases"][label].update(cycle=int(cycle), time_s=float(t), dt_s=float(dt))
                    state["cases"][label]["elapsed_s"] = time.monotonic() - start
                    write_json(root / "progress.json", state)
                    time.sleep(5)
            execution = scan_log(case / "run.log")
            execution.update(exit_code=process.returncode, elapsed_s=time.monotonic() - start)
            files = sorted(output.glob("*uov_x*.athdf"))
            if not files:
                raise RuntimeError(f"{label}: no HDF5 profile")
            with h5py.File(files[-1]) as data:
                execution["last_output_time_s"] = float(data.attrs["Time"])
                execution["uniform_grid"] = bool(np.all(data["Levels"][:] == 0))
            write_json(case / "execution.json", execution)
            if (process.returncode != 0 or execution["failure_lines"] or
                    not execution["normal_time_limit_termination"] or
                    not execution["uniform_grid"] or
                    abs(execution["last_output_time_s"] - .04) > 1e-10 or
                    execution["NR_max_final_residual"] > 1e-12):
                raise RuntimeError(f"{label}: execution did not pass verification; see execution.json")
            state["cases"][label].update(status="complete", elapsed_s=execution["elapsed_s"], time_s=.04)
            write_json(root / "progress.json", state)
            final_files.extend(["--profile", f"{label}={files[-1]}"])
            for i, file in enumerate(files[-4:]):
                recent_files.extend(["--profile", f"{label}_recent{i}={file}"])
            # Emit a per-case diagnostic as soon as the expensive calculation ends.
            command = [sys.executable, str(frozen / "diagnose_mach5_convergence.py"),
                       "--analytic", str(root / "mach5_semianalytic.csv"),
                       "--profile", f"{label}={files[-1]}", "--output-dir", str(case / "diagnostics")]
            with (case / "analysis.log").open("w") as log:
                subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
            print(f"completed {label}: {execution['elapsed_s']:.1f} s", flush=True)
        for name, files in (("final_diagnostics", final_files), ("stationarity_diagnostics", recent_files)):
            command = [sys.executable, str(frozen / "diagnose_mach5_convergence.py"),
                       "--analytic", str(root / "mach5_semianalytic.csv"), *files,
                       "--output-dir", str(root / name)]
            with (root / (name + ".log")).open("w") as log:
                subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
        final = json.loads((root / "final_diagnostics/metrics.json").read_text())
        temporal = json.loads((root / "stationarity_diagnostics/metrics.json").read_text())
        report(root, final, temporal)
        state.update(status="complete", active_case=None, report=str(root / "report.md"))
        write_json(root / "progress.json", state)
        print(f"complete: {root / 'report.md'}", flush=True)
    except BaseException:
        state.update(status="failed", error=traceback.format_exc())
        write_json(root / "progress.json", state)
        (root / "failure.txt").write_text(state["error"])
        raise


if __name__ == "__main__":
    main()
