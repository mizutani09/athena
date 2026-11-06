#!/usr/bin/env python3
"""WOR-59: independent time/space and control sweeps for the streaming front."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html import escape
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nrfld-validation")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["pdf.fonttype"] = 42

from diffusion_validation import materialize_input
from nrfld_validation.io import find_repo_root, read_athdf
from nrfld_validation.manifest import sha256_file, tree_sha256
from run_suite import prepare_build_source
from streaming_validation import threshold_position


HERE = Path(__file__).resolve().parent
C = 2.99792458e10
FINAL_TIME = 2e-9
RIGHT = 1e-10
REFERENCE_DT = (100 / 128) / (20 * C)


@dataclass(frozen=True)
class Case:
    name: str
    nx: int = 128
    divisor: float = 20
    chi: float = 1e-7
    xmax: float = 100
    front0: float = 10
    fixed_dt: float = -1
    strict: bool = False


CASES = (
    Case("n128_d2", divisor=2),
    Case("n128_d20"),
    Case("n128_d100", divisor=100),
    Case("n256_fixed_dt", nx=256, fixed_dt=REFERENCE_DT),
    Case("n512_fixed_dt", nx=512, fixed_dt=REFERENCE_DT),
    Case("n256_d20", nx=256),
    Case("n512_d20", nx=512),
    Case("n128_chi1e-4", chi=1e-4),
    Case("n128_aligned", front0=13 * 100 / 128),
    Case("n256_aligned_fixed", nx=256, front0=13 * 100 / 128,
         fixed_dt=REFERENCE_DT),
    Case("n512_aligned_fixed", nx=512, front0=13 * 100 / 128,
         fixed_dt=REFERENCE_DT),
    Case("n192_long_domain", nx=192, xmax=150),
    Case("n128_strict", strict=True),
)


def case_settings(case: Case) -> list[str]:
    dx = case.xmax / case.nx
    width = 8 * dx
    settings = [
        f"mesh/nx1={case.nx}", "mesh/nx2=8", "mesh/nx3=8",
        f"mesh/x1max={case.xmax:.17g}",
        f"mesh/x2min={-width/2:.17g}", f"mesh/x2max={width/2:.17g}",
        f"mesh/x3min={-width/2:.17g}", f"mesh/x3max={width/2:.17g}",
        "meshblock/nx1=8", "meshblock/nx2=8", "meshblock/nx3=8",
        f"problem/chi_R={case.chi:.17g}",
        f"problem/x_front0={case.front0:.17g}",
        f"problem/dt_light_divisor={case.divisor:.17g}",
        f"problem/fixed_dt={case.fixed_dt:.17g}",
        "nrfld/linear_solver=fgmres", "nrfld/threshold=1e-11",
        f"nrfld/nr_threshold={1e-10 if case.strict else 1e-8:.17g}",
        "nrfld/diagnostic_verbosity=0", "nrfld/fgmres_restart=40",
        "nrfld/fgmres_max_iterations=500", "nrfld/fgmres_mg_vcycles=2",
        f"nrfld/fgmres_threshold={1e-10 if case.strict else 1e-8:.17g}",
        "nrfld/failure_policy=abort", "time/ncycle_out=1000",
        "output2/ghost_zones=false", "output3/ghost_zones=false",
        "output3/dt=2e-9", f"job/problem_id={case.name}",
    ]
    return settings


def profile(path: Path):
    data = read_athdf(path)
    x = np.asarray(data["x1v"], dtype=float)
    n = len(x)
    values = {field: np.asarray(data[field], dtype=float).reshape(-1, n).mean(axis=0)
              for field in ("Erad", "Rlim", "lambda", "Frad_x")}
    if not np.all(np.diff(x) > 0):
        raise ValueError("nonascending cell centers")
    return x, values, float(data["Time"]), int(data["NumCycles"])


def analyze(case: Case, directory: Path) -> dict:
    paths = sorted((directory / "output").glob("*.uov_x_final.*.athdf"))
    if not paths:
        raise FileNotFoundError(f"final profile missing: {directory}")
    x, values, t, cycles = profile(paths[-1])
    energy = values["Erad"]
    if not math.isclose(t, FINAL_TIME, rel_tol=1e-6, abs_tol=1e-15):
        raise ValueError(f"wrong final time: {t}")
    if not np.all(np.isfinite(energy)) or np.min(energy) < 0:
        raise ValueError("invalid radiation field")
    dx = case.xmax / case.nx
    y = (energy - RIGHT) / (1 - RIGHT)
    x90, x50, x10 = (threshold_position(x, y, p)
                     for p in (.9, .5, .1))
    initial = np.where(x < case.front0, 1.0, RIGHT)
    initial_x50 = threshold_position(x, initial, .5)
    expected = case.front0 + C*t
    ideal = np.where(x < expected, 1.0, RIGHT)
    step_l1 = float(np.mean(np.abs(energy - ideal)) / (1 - RIGHT))
    ahead = x > expected
    ahead_fraction = float(np.sum(np.maximum(energy[ahead] - RIGHT, 0)) * dx
                           / ((1 - RIGHT) * max(C*t, dx)))
    near = (x >= x90) & (x <= x10)
    limiter_r = values["lambda"][near] * values["Rlim"][near]
    flux_ratio = np.abs(values["Frad_x"][near]) / np.maximum(C*energy[near], 1e-30)
    text = (directory / "run.log").read_text(errors="replace")
    nr = [dict(re.findall(r"(\w+)=([^\s]+)", line))
          for line in text.splitlines() if line.startswith("[NR] status=")]
    active = [row for row in nr if row.get("status") != "dt_zero_initialization"]
    fg = [dict(re.findall(r"(\w+)=([^\s]+)", line))
          for line in text.splitlines() if line.startswith("[LINEAR_FGMRES]")]
    solver_ok = (len(active) == cycles
                 and all(row.get("status") == "converged" for row in active)
                 and bool(fg)
                 and all(float(row["relative_residual"]) <=
                         float(row["target"]) * 1.000001 for row in fg))
    mass = float(np.sum(energy) * dx)
    ideal_mass = float(expected * (1 - RIGHT) + case.xmax * RIGHT)
    result = {
        "status": "passed" if solver_ok else "failed", "nx": case.nx,
        "dx_cm": dx, "chi_R_cm-1": case.chi, "front0_cm": case.front0,
        "xmax_cm": case.xmax, "dt_s": case.fixed_dt if case.fixed_dt > 0 else dx/(case.divisor*C),
        "cycles": cycles, "time_s": t,
        "initial_x50_bias_cm": initial_x50 - case.front0,
        "expected_front_cm": expected, "x90_cm": x90, "x50_cm": x50,
        "x10_cm": x10, "x50_error_cm": x50 - expected,
        "x50_error_cells": (x50 - expected)/dx,
        "x10_error_cm": x10 - expected,
        "width_10_90_cm": x10-x90, "width_10_90_cells": (x10-x90)/dx,
        "integrated_E_cm": mass, "ideal_integrated_E_cm": ideal_mass,
        "integrated_E_deficit_cm": mass-ideal_mass,
        "step_l1_normalized": step_l1,
        "ahead_energy_fraction": ahead_fraction,
        "median_lambda_R_front": float(np.median(limiter_r)),
        "median_abs_F_over_cE_front": float(np.median(flux_ratio)),
        "solver_ok": solver_ok, "newton_calls": len(active),
        "fgmres_calls": len(fg),
        "max_newton_residual": max((float(x["final_max"]) for x in active), default=None),
        "max_fgmres_ratio": max((float(x["relative_residual"])/float(x["target"])
                                 for x in fg), default=None),
        "profile": str(paths[-1].relative_to(directory)),
    }
    return result


def assess_convergence(cases: dict) -> dict:
    """Check trends, without declaring a finite-grid FLD step analytically exact."""
    required = ("n128_d2", "n128_d20", "n128_d100", "n256_d20", "n512_d20",
                "n192_long_domain", "n128_strict")
    if any(name not in cases or cases[name].get("status") != "passed"
           for name in required):
        return {"status": "incomplete", "checks": {}}
    lag = lambda name: abs(cases[name]["x50_error_cm"])
    width = lambda name: cases[name]["width_10_90_cm"]
    checks = {
        "temporal_50pct_lag_decreases":
            lag("n128_d2") > lag("n128_d20") > lag("n128_d100"),
        "temporal_width_decreases":
            width("n128_d2") > width("n128_d20") > width("n128_d100"),
        "joint_refinement_50pct_lag_decreases":
            lag("n128_d20") > lag("n256_d20") > lag("n512_d20"),
        "joint_refinement_10pct_leading_error_decreases":
            abs(cases["n128_d20"]["x10_error_cm"])
            > abs(cases["n256_d20"]["x10_error_cm"])
            > abs(cases["n512_d20"]["x10_error_cm"]),
        "joint_refinement_width_decreases":
            width("n128_d20") > width("n256_d20") > width("n512_d20"),
        "strict_tolerance_profile_position_unchanged":
            abs(cases["n128_d20"]["x50_cm"]
                - cases["n128_strict"]["x50_cm"]) < 1e-3,
        "far_boundary_position_unchanged":
            abs(cases["n128_d20"]["x50_cm"]
                - cases["n192_long_domain"]["x50_cm"]) < .02,
        "free_streaming_limiter_saturated":
            cases["n128_d20"]["median_lambda_R_front"] > .999,
    }
    aligned = ("n128_aligned", "n256_aligned_fixed", "n512_aligned_fixed")
    if all(name in cases and cases[name].get("status") == "passed" for name in aligned):
        checks["aligned_fixed_dt_50pct_lag_decreases"] = (
            lag(aligned[0]) > lag(aligned[1]) > lag(aligned[2]))
    return {"status": "passed" if all(checks.values()) else "failed",
            "checks": checks}


def render(root: Path, summary: dict):
    cases = summary["cases"]
    completed = {name: row for name, row in cases.items() if row.get("status") == "passed"}
    fig_dir = root / "figures"
    fig_dir.mkdir(exist_ok=True)
    if completed:
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        for name in ("n128_d2", "n128_d20", "n128_d100", "n256_d20", "n512_d20"):
            if name not in completed:
                continue
            path = root / "cases" / name / completed[name]["profile"]
            x, values, _, _ = profile(path)
            ax.plot(x, values["Erad"], label=name, lw=1.8)
        ax.axvline(10 + C*FINAL_TIME, color="black", ls="--", label="x0+ct")
        ax.set(xlim=(50, 80), ylim=(-.03, 1.05), xlabel="x [cm]", ylabel="Erad [erg cm⁻³]")
        ax.grid(alpha=.25); ax.legend(fontsize=10)
        fig.tight_layout()
        fig.savefig(fig_dir / "profiles.png", dpi=170)
        fig.savefig(fig_dir / "profiles.pdf")
        plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
        for name, row in completed.items():
            if name in ("n128_d2", "n128_d20", "n128_d100"):
                axes[0, 0].scatter(row["dt_s"], -row["x50_error_cm"], label=name)
                axes[1, 0].scatter(row["dt_s"], row["width_10_90_cm"], label=name)
            if name in ("n128_d20", "n256_fixed_dt", "n512_fixed_dt"):
                axes[0, 1].scatter(row["dx_cm"], -row["x50_error_cm"], label=name)
                axes[1, 1].scatter(row["dx_cm"], row["width_10_90_cm"], label=name)
            if name in ("n128_aligned", "n256_aligned_fixed", "n512_aligned_fixed"):
                axes[0, 1].scatter(row["dx_cm"], -row["x50_error_cm"],
                                   marker="x", s=65, label=name)
        for ax in axes.flat:
            ax.grid(alpha=.25)
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=7)
        axes[0, 0].set(xlabel="dt [s] at N=128", ylabel="50% lag [cm]")
        axes[0, 1].set(xlabel="dx [cm] at fixed dt", ylabel="50% lag [cm]")
        axes[1, 0].set(xlabel="dt [s] at N=128", ylabel="10–90% width [cm]")
        axes[1, 1].set(xlabel="dx [cm] at fixed dt", ylabel="10–90% width [cm]")
        for ax in axes.flat:
            ax.set_xscale("log")
        for ax in (axes[0, 1], axes[1, 1]):
            ax.set_xticks((0.2, 0.4, 0.8), labels=("0.2", "0.4", "0.8"))
        fig.tight_layout()
        fig.savefig(fig_dir / "convergence.png", dpi=170)
        fig.savefig(fig_dir / "convergence.pdf")
        plt.close(fig)
    rows = []
    for name, row in cases.items():
        if row.get("status") == "passed":
            cells = (f'<td>{row["nx"]}</td><td>{row["dt_s"]:.2e}</td>'
                     f'<td>{row["initial_x50_bias_cm"]:+.3f}</td>'
                     f'<td>{row["x50_error_cm"]:+.3f}</td>'
                     f'<td>{row["x10_error_cm"]:+.3f}</td>'
                     f'<td>{row["width_10_90_cm"]:.3f}</td>'
                     f'<td>{row["integrated_E_deficit_cm"]:+.3f}</td>')
        else:
            cells = f'<td colspan="7">{escape(row.get("error", "failed"))}</td>'
        rows.append(f'<tr><th><a href="cases/{escape(name)}/run.log">{escape(name)}</a></th>'
                    f'{cells}<td><a href="cases/{escape(name)}/input.used">input</a></td></tr>')
    images = ''.join(f'<figure><img src="figures/{name}.png" alt="{name}"></figure>'
                     for name in ("profiles", "convergence", "experimental_faces")
                     if (fig_dir / f"{name}.png").is_file())
    trial_rows = []
    for name, row in summary.get("experimental_trials", {}).items():
        trial_rows.append(
            f'<tr><th><a href="{escape(row["directory"])}/run.log">{escape(name)}</a></th>'
            f'<td>{row["x50_error_cm"]:+.3f}</td>'
            f'<td>{row["x10_error_cm"]:+.3f}</td>'
            f'<td>{row["width_10_90_cm"]:.3f}</td>'
            f'<td>{row["step_l1_normalized"]:.4f}</td>'
            f'<td><a href="{escape(row["directory"])}/source.patch">source patch</a> · '
            f'<a href="{escape(row["directory"])}/input.used">input</a></td></tr>')
    trial_section = ('' if not trial_rows else
        '<h2>Diagnostic variants (not adopted)</h2>'
        '<p>A stronger upwind face value reduces center lag but spreads the leading edge; '
        'raising the left ghost energy leaves the result nearly unchanged. '
        'Both use isolated binary builds. Source patches and exact inputs are in each experiment folder.</p>'
        '<table><thead><tr><th>Variant</th><th>50% error [cm]</th>'
        '<th>10% error [cm]</th><th>Width [cm]</th><th>Step L1</th><th>Files</th></tr></thead><tbody>'
        + ''.join(trial_rows) + '</tbody></table>')
    (root / "index.html").write_text(f'''<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>WOR-59 streaming front convergence</title><style>
:root{{font:21px/1.6 system-ui,sans-serif;color:#17212b}}body{{max-width:1400px;margin:1.5rem auto;padding:0 1rem}}
h1{{font-size:2rem}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.5rem;border-bottom:1px solid #cbd5e1;text-align:left}}
img{{max-width:100%}}a{{color:#075ca8}}@media(max-width:700px){{:root{{font-size:19px}}}}</style></head><body>
<h1>Streaming front: time and space convergence</h1>
<p><strong>Calculation status: {escape(summary['status'])}</strong>. This status checks solver
convergence and completion; front accuracy is reported separately. The reference is x₀+ct at 2 ns.
The curves are never translated. 10% is the leading edge; 50% is the center.
Integrated E deficit compares the whole x-domain energy with a perfect translated step;
the boundary is an imposed radiation energy, so this is a transport diagnostic.</p>
<p><strong>Convergence trend checks: {escape(summary.get('front_convergence', {}).get('status', 'incomplete'))}</strong>.
These check improvement on refinement and sensitivity to the boundary and solver tolerance;
they do not assert zero finite-grid front error. The individual checks are in summary.json.</p>
<p>The initial bias is the unshifted 50% position of the discretized initial step minus x₀.
Errors always use the stated x₀; no curve is translated. <a href="summary.json">Full metrics</a></p>
<table><thead><tr><th>Case</th><th>N</th><th>dt [s]</th><th>Initial bias [cm]</th>
<th>50% error [cm]</th><th>10% error [cm]</th><th>10–90% width [cm]</th>
<th>Integrated E deficit [cm]</th><th>Input</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
{trial_section}{images}</body></html>''')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/d/output/rad"))
    parser.add_argument("--run-name", default="wor59_streaming_convergence_"
                        + datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S_JST"))
    parser.add_argument("--reuse-build", type=Path, help="configured streaming build_source")
    parser.add_argument("--max-cores", type=int, default=4)
    parser.add_argument("--case", action="append", choices=[x.name for x in CASES])
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--postprocess", type=Path)
    args = parser.parse_args()
    if args.postprocess:
        root = args.postprocess.resolve()
        summary = json.loads((root / "summary.json").read_text())
        summary["front_convergence"] = assess_convergence(summary["cases"])
        (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        render(root, summary)
        return 0
    repo = find_repo_root(HERE)
    root = (args.output_root / args.run_name).resolve()
    root.mkdir(parents=True, exist_ok=False)
    executable = (args.reuse_build.resolve() / "bin/athena") if args.reuse_build else None
    if executable is None:
        source = root / "build_source"
        prepare_build_source(repo, source)
        for command, logfile in (([sys.executable, "configure.py", "--prob=nrfld_streaming_front",
                                  "--eos=adiabatic", "-nrmgfld", "-hdf5", "-mpi", "--cflag=-O3"],
                                 "configure.log"),
                                (["make", f"-j{args.max_cores}"], "build.log")):
            with (root / logfile).open("w") as output:
                subprocess.run(command, cwd=source, stdout=output,
                               stderr=subprocess.STDOUT, timeout=args.timeout, check=True)
        executable = source / "bin/athena"
    config = subprocess.check_output([str(executable), "-c"], text=True)
    if not re.search(r"Problem generator:\s+nrfld_streaming_front\b", config):
        raise ValueError("wrong build: expected nrfld_streaming_front")
    selected = [case for case in CASES if args.case is None or case.name in args.case]
    summary = {"status": "running", "binary_sha256": sha256_file(executable),
               "source_sha256": tree_sha256(executable.parent.parent / "src"),
               "analysis_script_sha256": sha256_file(Path(__file__)),
               "input_source_sha256": sha256_file(
                   repo / "inputs/radiation/athinput.nrfld_streaming_front"),
               "binary_config": config, "cases": {}}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for case in selected:
        directory = root / "cases" / case.name
        directory.mkdir(parents=True)
        (directory / "output").mkdir()
        materialize_input(repo / "inputs/radiation/athinput.nrfld_streaming_front",
                          directory / "input.used", case_settings(case))
        command = (["mpirun", "-np", str(args.max_cores)] if args.max_cores > 1 else [])
        command += [str(executable), "-i", "input.used", "-d", "output"]
        (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        start = time.monotonic()
        with (directory / "run.log").open("w") as output:
            completed = subprocess.run(command, cwd=directory, stdout=output,
                                       stderr=subprocess.STDOUT, timeout=args.timeout,
                                       env=dict(os.environ, OMP_NUM_THREADS="1"))
        text = (directory / "run.log").read_text(errors="replace")
        try:
            if completed.returncode or "FATAL ERROR" in text or "Terminating on time limit" not in text:
                raise RuntimeError(f"run failed, exit={completed.returncode}")
            row = analyze(case, directory)
        except Exception as error:
            row = {"status": "failed", "error": str(error)}
        row["wall_seconds"] = time.monotonic() - start
        summary["cases"][case.name] = row
        summary["status"] = "passed" if all(
            x.get("status") == "passed" for x in summary["cases"].values()) else "failed"
        summary["front_convergence"] = assess_convergence(summary["cases"])
        (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        render(root, summary)
        print(case.name, row.get("status"), row.get("x50_error_cm"),
              row.get("width_10_90_cm"), flush=True)
    print(root)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
