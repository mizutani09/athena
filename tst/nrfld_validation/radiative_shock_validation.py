#!/usr/bin/env python3
"""Run and report Mach 2/5 NR-FLD radiative shocks on a cubic-cell slab."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
from html import escape
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nrfld-validation")
import h5py  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from diagnose_mach5_convergence import measure  # noqa: E402
from diffusion_validation import materialize_input  # noqa: E402
from nrfld_validation.io import find_repo_root  # noqa: E402
from nrfld_validation.acceptance import (radiative_shock as assess_accuracy,
                                         acceptance_html)  # noqa: E402
from nrfld_validation.report_style import REPORT_TYPOGRAPHY  # noqa: E402
from nrfld_validation.manifest import (base_manifest, sha256_file, tree_sha256,
                                       utc_now, write_manifest)  # noqa: E402
from plots.plot_radiative_shock import (jump_center, load_analytic_profile,
                                        load_numerical_profile,
                                        plot_radiative_shock)  # noqa: E402
from run_suite import (CommandExecutionError, binary_configuration, checked,
                       effective_settings, prepare_build_source)  # noqa: E402
from shock_tube_validation import (assert_effective_input, parse_solver_log,
                                   runtime_resources, verify_source_snapshot)  # noqa: E402

CASES = ("mach2", "mach5")
T_FINAL = {"mach2": 0.05, "mach5": 0.04}
DOMAIN = {"mach2": (-1000., 500.), "mach5": (-3000., 1000.)}
WINDOW = {"mach2": 40., "mach5": 60.}
REFERENCE_NAME = {"mach2": "mach2_semianalytic_bvp.csv",
                  "mach5": "mach5_semianalytic.csv"}
SOURCE_DIR = HERE/"reference/radiative_shock"
ANALYSIS_FILES = (Path(__file__), HERE/"diagnose_mach5_convergence.py",
                  HERE/"plots/plot_radiative_shock.py",
                  HERE/"nrfld_validation/report_style.py",
                  HERE/"nrfld_validation/acceptance.py")


def analysis_hashes() -> dict[str, str]:
    return {str(path.relative_to(HERE)): sha256_file(path) for path in ANALYSIS_FILES}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/d/output/rad"))
    parser.add_argument("--run-name")
    parser.add_argument("--preset", choices=("local", "hpc"), default="local")
    parser.add_argument("--layout", choices=("uniform", "amr", "smr"), default="uniform",
                        help="AMR uses one adaptive level; SMR uses three static levels.")
    parser.add_argument("--case", action="append", choices=CASES)
    parser.add_argument("--nx1-mach2", type=int)
    parser.add_argument("--nx1-mach5", type=int)
    parser.add_argument("--slab", type=int, default=4)
    parser.add_argument("--mpi-ranks", type=int, default=4)
    parser.add_argument("--build-jobs", type=int, default=8)
    parser.add_argument("--run-timeout", type=float, default=86400.)
    parser.add_argument("--build-timeout", type=float, default=900.)
    parser.add_argument("--nlim", type=int, help="Short exploratory run; not a completed comparison.")
    parser.add_argument("--reuse-build", type=Path, help="Verified isolated build_source directory.")
    parser.add_argument("--import-case", action="append", default=[], metavar="MACH=DIR",
                        help="Import a matching completed case with input.used/output/run.log.")
    parser.add_argument("--postprocess", type=Path)
    parser.add_argument("--reassess", type=Path,
                        help="Apply current accuracy criteria to a completed saved report.")
    parser.add_argument("--attach-amr-run", type=Path,
                        help="With --postprocess, copy a completed one-level AMR run into this report.")
    args = parser.parse_args()
    if args.attach_amr_run and not args.postprocess:
        parser.error("--attach-amr-run requires --postprocess")
    if args.slab < 4 or args.slab & (args.slab-1):
        parser.error("--slab must be a power of two >=4; MeshBlocks must be cubic")
    if args.mpi_ranks < 1 or args.build_jobs < 1 or args.run_timeout <= 0:
        parser.error("ranks, jobs and timeouts must be positive")
    args.cases = tuple(args.case or CASES)
    if args.layout == "smr":
        defaults = {"mach2": 64 if args.preset == "local" else 256,
                    "mach5": 64 if args.preset == "local" else 1024}
    elif args.layout == "amr":
        defaults = {"mach2": 128 if args.preset == "local" else 512,
                    "mach5": 128 if args.preset == "local" else 2048}
    else:
        defaults = {"mach2": 256 if args.preset == "local" else 1024,
                    "mach5": 256 if args.preset == "local" else 4096}
    args.nx = {"mach2": args.nx1_mach2 or defaults["mach2"],
               "mach5": args.nx1_mach5 or defaults["mach5"]}
    for case in args.cases:
        if args.nx[case] % args.slab or args.nx[case] < args.slab:
            parser.error(f"{case} nx1 must be a positive multiple of slab")
    imported = {}
    for item in args.import_case:
        name, separator, location = item.partition("=")
        if not separator or name not in CASES or name in imported:
            parser.error(f"invalid or repeated --import-case: {item}")
        imported[name] = Path(location).expanduser().resolve()
    args.imported = imported
    return args


def input_overrides(case: str, n: int, slab: int, nlim: int | None,
                    layout: str) -> list[str]:
    width = (1500. if case == "mach2" else 6000.)/n
    half = 0.5*slab*width
    overrides = [f"mesh/nx1={n}", f"mesh/nx2={slab}", f"mesh/nx3={slab}",
                 f"mesh/x2min={-half:.17g}", f"mesh/x2max={half:.17g}",
                 f"mesh/x3min={-half:.17g}", f"mesh/x3max={half:.17g}",
                 f"meshblock/nx1={slab}", f"meshblock/nx2={slab}",
                 f"meshblock/nx3={slab}",
                 f"mesh/refinement={'static' if layout == 'smr' else 'adaptive' if layout == 'amr' else 'none'}",
                 f"mesh/numlevel={3 if layout == 'smr' else 2 if layout == 'amr' else 1}",
                 f"time/nlim={-1 if nlim is None else nlim}", "time/ncycle_out=100",
                 "nrfld/linear_solver=fgmres", "nrfld/nr_threshold=1e-8",
                 "nrfld/threshold=1e-10", "nrfld/failure_policy=abort",
                 f"problem/use_amr_refinement={'true' if layout == 'amr' else 'false'}",
                 "output2/ghost_zones=false", "output3/ghost_zones=false",
                 f"job/problem_id=radshock_{case}_{'amr' if layout == 'amr' else 'smr' if layout == 'smr' else 'local' if n == 256 else 'validation'}"]
    if layout == "smr":
        xlo, xhi = (-1000., 500.) if case == "mach2" else (-4000., 2000.)
        inner_lo, inner_hi = (-850., 450.) if case == "mach2" else (-400., 400.)
        for level, lo, hi in ((1, xlo, xhi), (2, inner_lo, inner_hi)):
            section = f"refinement{level}"
            overrides.extend([f"{section}/x1min={lo:.17g}",
                              f"{section}/x1max={hi:.17g}",
                              f"{section}/x2min={-half:.17g}",
                              f"{section}/x2max={half:.17g}",
                              f"{section}/x3min={-half:.17g}",
                              f"{section}/x3max={half:.17g}",
                              f"{section}/level={level}"])
    if nlim is not None:
        overrides.extend(["output2/dt=-1", "output3/dt=1e-12"])
    return overrides


def ensure_build(args, repo: Path, root: Path, manifest: dict) -> Path:
    build = root/"build"
    build.mkdir()
    source = build/"build_source"
    records = manifest["commands"].setdefault("records", [])
    if args.reuse_build:
        original = args.reuse_build.expanduser().resolve()
        shutil.copytree(original, source, ignore=shutil.ignore_patterns("*.o"))
        verify_source_snapshot(repo, source, adapted_reference=False)
        manifest["build_imported_from"] = str(original)
    else:
        prepare_build_source(repo, source)
        command = [sys.executable, str(source/"configure.py"),
                   "--prob=nrfld_radiative_shock", "--eos=adiabatic",
                   "-nrmgfld", "-hdf5", "-mpi", "--cflag=-O3"]
        records.append(checked(command, source, build/"configure.log",
                               args.build_timeout, "configure"))
        records.append(checked(["make", f"-j{args.build_jobs}"], source,
                               build/"build.log", args.build_timeout, "build"))
    executable = source/"bin/athena"
    config = binary_configuration(executable, build, 30., records)
    expected = {"problem_generator": "nrfld_radiative_shock",
                "equation_of_state": "adiabatic", "riemann_solver": "hllc_fld",
                "fld_with_newton-raphson": "ON", "mpi_parallelism": "ON",
                "hdf5_output": "ON"}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"binary {key}: expected {value}, got {config.get(key)}")
    manifest["build"] = {"source_sha256": tree_sha256(source/"src"),
                         "binary_sha256": sha256_file(executable),
                         "configuration": config}
    return executable


def files(case_dir: Path) -> list[Path]:
    found = sorted((case_dir/"output").glob("*.uov_x.*.athdf"))
    if len(found) < 2:
        raise ValueError(f"need initial and evolved uov_x HDF5 snapshots: {case_dir}")
    return found


def launcher_wall_seconds(path: Path) -> float | None:
    if not path.is_file():
        return None
    match = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([^\s]+)",
                      path.read_text())
    if not match:
        return None
    parts = [float(value) for value in match[1].split(":")]
    return sum(value*60**power for power, value in enumerate(reversed(parts)))


def verify_case(case_dir: Path, case: str, nlim: int | None) -> dict:
    paths = files(case_dir)
    with h5py.File(paths[-1]) as hdf:
        levels = sorted({int(level) for level in hdf["Levels"][:]})
        final_blocks = len(hdf["Levels"])
    refinement = effective_settings(case_dir/"input.used", []).get("mesh/refinement", "none")
    if refinement == "adaptive" and 1 not in levels:
        raise ValueError(f"adaptive refinement did not create a level-one block: {case_dir}")
    final = load_numerical_profile(paths[-1])
    initial = load_numerical_profile(paths[0])
    log = (case_dir/"run.log").read_text(errors="replace")
    if re.search(r"FATAL|Segmentation fault|double free|corrupted double", log, re.I):
        raise ValueError(f"fatal text in run.log: {case_dir}")
    expected_termination = ("Terminating on time limit" if nlim is None
                            else "Terminating on cycle limit")
    if expected_termination not in log:
        raise ValueError(f"did not reach expected termination: {case_dir}")
    if nlim is None and not math.isclose(final["time"], T_FINAL[case], rel_tol=1e-6):
        raise ValueError(f"{case} final output time {final['time']} != {T_FINAL[case]}")
    if nlim is not None and not 0 < final["time"] <= T_FINAL[case]:
        raise ValueError(f"invalid exploratory final time: {final['time']}")
    if initial["time"] != 0.0:
        raise ValueError(f"missing time-zero output: {case_dir}")
    for field in ("rho", "tgas", "trad"):
        if not np.all(np.isfinite(final[field])) or np.any(final[field] <= 0):
            raise ValueError(f"nonpositive/nonfinite {field} in {case_dir}")
    cycles = int(re.search(r"time=[^\s]+ cycle=(\d+)\s+tlim=", log)[1])
    solver = parse_solver_log(case_dir/"run.log", cycles)
    if refinement == "adaptive":
        # Refined levels perform additional solves beyond the global mesh cycles.
        # Keep the same residual/status checks, but count every actual NR solve.
        if solver["nr_status_count"] < cycles:
            raise ValueError(f"fewer AMR Newton solves than global cycles: {case_dir}")
        solver = parse_solver_log(case_dir/"run.log", solver["nr_status_count"])
        solver["global_cycles"] = cycles
    if not solver["good"]:
        raise ValueError(f"solver did not pass recorded limits: {case_dir}: {solver}")
    return {"time": final["time"], "cycles": cycles, "snapshots": len(paths),
            "mesh_levels_final": levels, "meshblocks_final": final_blocks,
            "solver": solver, "max_rss_kib_launcher": runtime_resources(case_dir/"resources.txt"),
            "wall_seconds": launcher_wall_seconds(case_dir/"resources.txt"),
            "final_output_sha256": sha256_file(paths[-1])}


def run_cases(args, repo: Path, root: Path, executable: Path, manifest: dict):
    for case in args.cases:
        destination = root/"cases"/case
        source = args.imported.get(case)
        template = repo/f"inputs/radiation/athinput.nrfld_radiative_shock_{case}"
        overrides = input_overrides(case, args.nx[case], args.slab, args.nlim,
                                    args.layout)
        if source:
            shutil.copytree(source, destination, ignore=shutil.ignore_patterns("*.xdmf"))
            assert_effective_input(destination/"input.used", template, overrides)
            manifest.setdefault("case_imported_from", {})[case] = str(source)
        else:
            destination.mkdir(parents=True)
            (destination/"output").mkdir()
            materialize_input(template, destination/"input.used", overrides)
            command = ["/usr/bin/time", "-v", "-o", str(destination/"resources.txt"),
                       "mpirun", "-np", str(args.mpi_ranks), str(executable),
                       "-i", str(destination/"input.used"),
                       "-d", str(destination/"output")]
            try:
                record = checked(command, destination, destination/"run.log",
                                 args.run_timeout, f"run_{case}")
            except CommandExecutionError as error:
                manifest["commands"].setdefault("records", []).append(error.record)
                raise
            manifest["commands"].setdefault("records", []).append(record)
        manifest.setdefault("case_inputs", {})[case] = {
            "input_sha256": sha256_file(destination/"input.used"),
            "effective_settings": effective_settings(destination/"input.used", []),
            "resolution": [args.nx[case], args.slab, args.slab],
            "layout": args.layout,
            "dx_cm": (1500. if case == "mach2" else 6000.)/args.nx[case],
            "ranks": args.mpi_ranks}


def stationarity(current: dict, previous: dict) -> dict:
    x = current["x_from_shock_cm"]
    xp = previous["x_from_shock_cm"]
    mask = (x >= xp[0]) & (x <= xp[-1])
    widths = current["cell_width_cm"][mask]
    result = {}
    for field in ("rho", "tgas", "trad"):
        older = np.interp(x[mask], xp, previous[field])
        newer = current[field][mask]
        result[field] = float(np.sum(np.abs(newer-older)*widths)
                              /np.sum(np.abs(newer)*widths))
    return result


def make_movie(root: Path, case: str, paths: list[Path], analytic: dict) -> Path:
    destination = root/"video"/f"{case}.mp4"
    destination.parent.mkdir(exist_ok=True)
    center = jump_center(analytic["x"], analytic["rho"])
    fig, (rho_ax, temp_ax) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    rho_ax.plot(analytic["x"], analytic["rho"], color="black", lw=1.2,
                label="semi-analytic")
    temp_ax.plot(analytic["x"], analytic["tgas"], color="black", ls="--",
                 lw=1.1, label="Tgas semi-analytic")
    temp_ax.plot(analytic["x"], analytic["trad"], color="grey", ls="-.",
                 lw=1.1, label="Trad semi-analytic")
    numeric_rho, = rho_ax.plot([], [], color="tab:blue", marker="o", ms=2,
                               lw=.8, label="NR-FLD rho")
    numeric_tgas, = temp_ax.plot([], [], color="tab:orange", marker="+", ms=3,
                                 lw=.8, label="NR-FLD Tgas")
    numeric_trad, = temp_ax.plot([], [], color="tab:green", marker="s", ms=2,
                                 lw=.8, label="NR-FLD Trad")
    rho_ax.set(xlim=DOMAIN[case], ylabel="density [g cm$^{-3}$]")
    rho_ax.set_ylim(.4e-12, 1.35e-12 if case == "mach2" else 2.1e-12)
    temp_ax.set(xlabel="shock-aligned x [cm]", ylabel="temperature [K]",
                ylim=(85, 240 if case == "mach2" else 1150))
    for axis in (rho_ax, temp_ax):
        axis.grid(alpha=.25)
        axis.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    with FFMpegWriter(fps=6, bitrate=1400).saving(fig, str(destination), dpi=120) as writer:
        for path in paths:
            profile = load_numerical_profile(path)
            x = profile["x"] + center - jump_center(profile["x"], profile["rho"])
            numeric_rho.set_data(x, profile["rho"])
            numeric_tgas.set_data(x, profile["tgas"])
            numeric_trad.set_data(x, profile["trad"])
            fig.suptitle(f"{case.upper()}   t = {profile['time']:.4g} s")
            writer.grab_frame()
    plt.close(fig)
    return destination


def analyze_case(root: Path, case: str) -> dict:
    case_dir = root/"cases"/case
    analytic_path = root/"reference"/REFERENCE_NAME[case]
    analytic = load_analytic_profile(analytic_path)
    paths = files(case_dir)
    first = load_numerical_profile(paths[0])
    far_state = {field: {side: float(abs(first[field][index]-analytic[field][index])
                                      /abs(analytic[field][index]))
                         for side, index in (("left", 0), ("right", -1))}
                 for field in ("rho", "tgas", "trad")}
    if max(value for sides in far_state.values() for value in sides.values()) > 5e-3:
        raise ValueError(f"{case} initial far states differ from semi-analytic reference: {far_state}")
    metrics, arrays = measure(paths[-1], analytic, WINDOW[case], DOMAIN[case])
    earlier, old_arrays = measure(paths[-2], analytic, WINDOW[case], DOMAIN[case])
    report = {"final": metrics, "previous": earlier,
              "aligned_change_since_previous": stationarity(arrays, old_arrays),
              "initial_far_state_relative_difference": far_state,
              "analytic_sha256": sha256_file(analytic_path)}
    numerical = load_numerical_profile(paths[-1])
    center = jump_center(analytic["x"], analytic["rho"])
    shifted_x = numerical["x"] + metrics["shock_alignment_shift_cm"]
    analytical_near = np.abs(analytic["x"]-center) <= WINDOW[case]
    numerical_near = np.abs(shifted_x-center) <= WINDOW[case]
    analytical_peak = float(np.max(analytic["tgas"][analytical_near]))
    numerical_peak = float(np.max(numerical["tgas"][numerical_near]))
    downstream = float(analytic["tgas"][-1])
    report["gas_temperature_peak"] = {
        "analytic_K": analytical_peak, "numerical_K": numerical_peak,
        "relative_peak_error": abs(numerical_peak-analytical_peak)/analytical_peak,
        "spike_amplitude_recovery": ((numerical_peak-downstream)/(analytical_peak-downstream)
                                     if analytical_peak > downstream else None),
        "cells_within_20cm": int(np.count_nonzero(np.abs(shifted_x-center) <= 20.))}
    with (case_dir/"comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(arrays)
        writer.writerows(zip(*arrays.values()))
    figures = root/"figures"
    figures.mkdir(exist_ok=True)
    for alignment in (True, False):
        name = "aligned" if alignment else "raw"
        plot_radiative_shock(case_dir, figures/f"{case}_{name}.png", case,
                             analytic_path=analytic_path, align_shock=alignment)
    history = []
    for path in paths:
        item, _ = measure(path, analytic, WINDOW[case], DOMAIN[case])
        history.append({"time_s": item["time_s"], "shift_cm": item["shock_alignment_shift_cm"],
                        **{f"{field}_relative_l1_percent":
                           item["agreement"][field]["global_relative_L1_percent"]
                           for field in ("rho", "tgas", "trad")}})
    with (case_dir/"error_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for field in ("rho", "tgas", "trad"):
        axes[0].plot([r["time_s"] for r in history],
                     [r[f"{field}_relative_l1_percent"] for r in history],
                     marker="o", ms=3, label=field)
    axes[1].plot([r["time_s"] for r in history], [r["shift_cm"] for r in history],
                 marker="o", ms=3)
    axes[0].set(xlabel="time [s]", ylabel="shock-aligned relative L1 [%]")
    axes[1].set(xlabel="time [s]", ylabel="numerical shift to analytic shock [cm]")
    for ax in axes:
        ax.grid(alpha=.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(figures/f"{case}_history.png", dpi=180)
    fig.savefig(figures/f"{case}_history.pdf")
    plt.close(fig)
    report["video"] = str(make_movie(root, case, paths, analytic).relative_to(root))
    return report


def write_html(root: Path, summary: dict):
    status = summary["status"]
    tone = "ok" if status == "passed" else "warning"
    cards = []
    for case, data in summary.get("cases", {}).items():
        if "analysis" not in data:
            cards.append(f'<section><h2>{escape(case)}</h2><p>Incomplete. '
                         f'<a href="cases/{case}/run.log">Run log</a></p></section>')
            continue
        m = data["analysis"]["final"]
        solver = data["execution"]["solver"]
        agreement = m["agreement"]
        change = data["analysis"]["aligned_change_since_previous"]
        peak = data["analysis"]["gas_temperature_peak"]
        far = data["analysis"]["initial_far_state_relative_difference"]
        far_max = max(value for sides in far.values() for value in sides.values())
        cards.append(f'''<section><h2>{escape(case.upper())}</h2>
<p><strong>t = {m['time_s']:.5g} s</strong> · {data['execution']['cycles']} cycles ·
{data['execution']['snapshots']} snapshots · {m['profile_points']} profile cells ·
numerical shock at {m['shock_raw_cm']:.4g} cm · alignment shift {m['shock_alignment_shift_cm']:.4g} cm</p>
<table><thead><tr><th>Field</th><th>Aligned relative L1</th><th>Previous snapshot change</th></tr></thead><tbody>
{''.join(f'<tr><th>{field}</th><td>{agreement[field]["global_relative_L1_percent"]:.3g}%</td><td>{100*change[field]:.3g}%</td></tr>' for field in ('rho','tgas','trad'))}
</tbody></table>
<p><strong>Gas-temperature peak:</strong> numerical {peak['numerical_K']:.2f} K / analytic
{peak['analytic_K']:.2f} K; peak error {100*peak['relative_peak_error']:.2f}%.
The ±20 cm shock region contains {peak['cells_within_20cm']} numerical cell centers.
Spike-amplitude recovery relative to the downstream state:
{100*peak['spike_amplitude_recovery']:.2f}%.</p>
<p>Initial left/right far states agree with the saved reference to within
{100*far_max:.4g}% across density and gas/radiation temperature.</p>
<p>Newton strict {solver['strict_converged']}/{solver['nr_status_count']} solves
across {data['execution']['cycles']} global cycles; relaxed {solver['relaxed_accepted']};
FGMRES calls {solver['fgmres_calls']}. Launcher maximum RSS {data['execution']['max_rss_kib_launcher']} KiB
(MPI rank memory is not summed). Wall time {data['execution'].get('wall_seconds')} s.</p>
<div class="plot"><h3>Shock aligned</h3><img src="figures/{case}_aligned.png" alt="{case} aligned profile"></div>
<div class="plot"><h3>Raw coordinates</h3><img src="figures/{case}_raw.png" alt="{case} raw profile"></div>
<div class="plot"><h3>Error and shock-position history</h3><img src="figures/{case}_history.png" alt="{case} error history"></div>
<div class="plot"><h3>Shock-aligned evolution</h3><video controls preload="metadata" width="920"><source
src="video/{case}.mp4" type="video/mp4"></video></div>
<p><a href="cases/{case}/comparison.csv">Comparison CSV</a> ·
<a href="cases/{case}/error_history.csv">History CSV</a> ·
<a href="cases/{case}/input.used">Effective input</a> ·
<a href="cases/{case}/run.log">Run log</a> ·
<a href="cases/{case}/resources.txt">Resources</a> ·
<a href="reference/{REFERENCE_NAME[case]}">Semi-analytic reference</a></p></section>''')
    amr_section = ""
    attached = summary.get("amr_attachment")
    if attached:
        rows = []
        plots = []
        for case in CASES:
            if case not in attached["cases"]:
                continue
            outcome = attached["cases"][case]
            execution = outcome["execution"]
            final = outcome["analysis"]["final"]
            peak = outcome["analysis"]["gas_temperature_peak"]
            baseline = summary["cases"][case]["analysis"]
            baseline_final = baseline["final"]
            def l1_pair(field: str) -> str:
                old = baseline_final["agreement"][field]["global_relative_L1_percent"]
                new = final["agreement"][field]["global_relative_L1_percent"]
                return f"{old:.3g}% → {new:.3g}%"
            rows.append(f'<tr><th>{case.upper()}</th><td>{execution["mesh_levels_final"]}</td>'
                        f'<td>{execution["meshblocks_final"]}</td>'
                        f'<td>{execution["solver"]["strict_converged"]}/{execution["solver"]["nr_status_count"]}</td>'
                        f'<td>{l1_pair("rho")}</td><td>{l1_pair("tgas")}</td>'
                        f'<td>{l1_pair("trad")}</td>'
                        f'<td>{100*baseline["gas_temperature_peak"]["relative_peak_error"]:.2f}%'
                        f' → {100*peak["relative_peak_error"]:.2f}%</td></tr>')
            plots.append(f'<h3>{case.upper()} AMR</h3>'
                         f'<img src="amr/figures/{case}_aligned.png" alt="{case} AMR comparison">'
                         f'<p><a href="amr/figures/{case}_raw.png">Raw coordinates</a> · '
                         f'<a href="amr/figures/{case}_history.png">Error history</a> · '
                         f'<a href="amr/cases/{case}/input.used">AMR input</a> · '
                         f'<a href="amr/cases/{case}/run.log">AMR log</a></p>')
        amr_section = (f'<section id="amr"><h2>One-level adaptive mesh refinement</h2>'
                       f'<p><strong>Status: {escape(attached["status"])}</strong>. The level-one cells '
                       'refine around the detected shock. The AMR run has the same final time and '
                       'physical reference as the uniform slab; its own report contains all CSV, '
                       'profiles, videos and provenance: <a href="amr/index.html">open AMR report</a>.</p>'
                       '<div class="scroll"><table><thead><tr><th>Case</th><th>Levels</th>'
                       '<th>Final blocks</th><th>Strict Newton</th><th>ρ L1 uniform → AMR</th>'
                       '<th>Tgas L1 uniform → AMR</th><th>Trad L1 uniform → AMR</th>'
                       '<th>Tgas peak error uniform → AMR</th></tr></thead><tbody>'
                       + ''.join(rows) + '</tbody></table></div>' + ''.join(plots) + '</section>')
    root.joinpath("index.html").write_text(f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Radiative shock validation</title>
<style>body{{font:18px/1.6 system-ui,sans-serif;max-width:1200px;margin:auto;padding:1.5rem;color:#1b2430}}
h1{{font-size:2rem}}h2{{border-bottom:2px solid #cbd5e1;padding-bottom:.2rem}}
section{{margin:2rem 0;padding:1rem 1.4rem;border:1px solid #cbd5e1;border-radius:.6rem}}
.warning{{color:#9a4f00;font-weight:700}}.ok{{color:#12653a;font-weight:700}}
table{{border-collapse:collapse;min-width:55%}}td,th{{padding:.45rem .8rem;border-bottom:1px solid #cbd5e1;text-align:right}}
th:first-child{{text-align:left}}img{{max-width:100%;height:auto}}.plot{{margin:1.4rem 0}}.scroll{{overflow-x:auto}}
a{{color:#095ca6}}{REPORT_TYPOGRAPHY}
:root{{font-size:21px}}
body{{font-size:1rem;max-width:1450px}}
table{{font-size:1rem}}
@media(max-width:700px){{:root{{font-size:19px}}}}
</style></head><body><h1>Mach 2/5 radiative shocks</h1>
<p class="{tone}">{escape(status.upper())}</p>
<p>The low-memory run uses a periodic slab with cubic cells and cubic MeshBlocks.
The reference is the saved semi-analytic steady solution. The raw plots retain physical x;
aligned plots and L1 place the midpoint of the largest adjacent density jump at the reference shock.
The reported shift is therefore a separate measured quantity. L1 uses cell-width weights over
the reference domain and samples each analytic branch on its own side of the discontinuity.
Previous-snapshot change uses cell-width weights after independently registering both shock positions.
The physical accuracy criteria below also require the gas-temperature spike;
coarse Mach 5 calculations can complete successfully while missing that feature.</p>
{acceptance_html(summary.get('acceptance'))}
{''.join(cards)}
{amr_section}
<section><h2>Reproduction</h2><p><a href="manifest.json">Manifest</a> ·
<a href="summary.json">Summary JSON</a> ·
<a href="build/build_source/bin/athena">Executable</a> ·
<a href="build/build_source/src/pgen/nrfld_radiative_shock.cpp">Problem generator</a> ·
<a href="reference/README.md">Reference provenance</a>. Inputs, source snapshot,
binary, reference CSV and generator scripts are stored in this directory.</p></section>
</body></html>''')


def main() -> int:
    args = arguments()
    os.environ["OMP_NUM_THREADS"] = "1"
    repo = find_repo_root(HERE)
    if args.reassess:
        root = args.reassess.expanduser().resolve()
        manifest = json.loads((root/"manifest.json").read_text())
        summary = json.loads((root/"summary.json").read_text())
        if summary.get("status") not in ("diagnostic_complete", "passed", "accuracy_not_met"):
            raise ValueError("reassess requires a completed strict-solver report")
        amr_report = root/"amr/summary.json"
        if amr_report.is_file() and "amr_attachment" in summary:
            amr_summary = json.loads(amr_report.read_text())
            summary["amr_attachment"]["status"] = amr_summary["status"]
            summary["amr_attachment"]["cases"] = amr_summary["cases"]
        summary["acceptance"] = assess_accuracy(summary)
        summary["status"] = "passed" if summary["acceptance"]["passed"] else "accuracy_not_met"
        manifest["status"] = summary["status"]
        manifest["acceptance"] = summary["acceptance"]
        manifest["analysis_script_sha256"] = sha256_file(Path(__file__))
        manifest["analysis_dependency_hashes"] = analysis_hashes()
        (root/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        write_manifest(root/"manifest.json", manifest)
        write_html(root, summary)
        print(root, flush=True)
        return 0
    if args.postprocess:
        root = args.postprocess.expanduser().resolve()
        manifest = json.loads((root/"manifest.json").read_text())
        summary = json.loads((root/"summary.json").read_text())
        if args.attach_amr_run:
            source = args.attach_amr_run.expanduser().resolve()
            amr_summary = json.loads((source/"summary.json").read_text())
            amr_manifest = json.loads((source/"manifest.json").read_text())
            if amr_summary.get("layout") != "amr" or amr_summary.get("status") not in (
                    "diagnostic_complete", "completed_with_solver_caveat",
                    "passed", "accuracy_not_met"):
                raise ValueError("attached run must be a completed AMR calculation")
            if set(amr_summary["selected_cases"]) != set(summary["selected_cases"]):
                raise ValueError("attached AMR run has different Mach cases")
            for case in summary["selected_cases"]:
                if (amr_manifest["analytic_references"][case]
                        != manifest["analytic_references"][case]):
                    raise ValueError(f"AMR reference hash mismatch: {case}")
                amr_time = amr_summary["cases"][case]["execution"]["time"]
                original_time = summary["cases"][case]["execution"]["time"]
                if not math.isclose(amr_time, original_time, rel_tol=1e-6):
                    raise ValueError(f"AMR final time mismatch: {case}")
            destination = root/"amr"
            if destination.exists():
                existing = manifest.get("amr_attachment", {})
                if existing.get("source") != str(source):
                    raise FileExistsError(destination)
            shutil.copytree(source, destination, ignore=shutil.ignore_patterns("*.o"),
                            dirs_exist_ok=True)
            summary["amr_attachment"] = {"run": amr_summary["run"],
                                         "status": amr_summary["status"],
                                         "cases": amr_summary["cases"]}
            manifest["amr_attachment"] = {"directory": "amr", "source": str(source),
                                          "manifest_sha256": sha256_file(source/"manifest.json")}
        for case in summary["selected_cases"]:
            if sha256_file(root/"reference"/REFERENCE_NAME[case]) != manifest["analytic_references"][case]:
                raise ValueError(f"saved analytic reference changed: {case}")
            summary["cases"].setdefault(case, {})
            summary["cases"][case]["execution"] = verify_case(
                root/"cases"/case, case, None)
            summary["cases"][case]["analysis"] = analyze_case(root, case)
        strict = all(summary["cases"][case]["execution"]["solver"]["relaxed_accepted"] == 0
                     for case in summary["selected_cases"])
        summary["acceptance"] = assess_accuracy(summary)
        summary["status"] = ("completed_with_solver_caveat" if not strict else
                             "passed" if summary["acceptance"]["passed"] else
                             "accuracy_not_met")
        if "error" in summary:
            summary["recovered_analysis_error"] = {
                "message": summary.pop("error"),
                "resolution": "Rechecked the saved AMR substep logs with the corrected Newton event count."}
        if "error" in manifest:
            manifest["recovered_analysis_error"] = {
                "message": manifest.pop("error"),
                "resolution": "Rechecked the saved AMR substep logs with the corrected Newton event count."}
        summary["finished_at"] = utc_now()
        manifest["status"] = summary["status"]
        manifest["analysis_script_sha256"] = sha256_file(Path(__file__))
        manifest["analysis_dependency_hashes"] = analysis_hashes()
        manifest["metrics"] = summary["cases"]
        manifest["acceptance"] = summary["acceptance"]
        write_manifest(root/"manifest.json", manifest)
        (root/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        write_html(root, summary)
        print(root, flush=True)
        return 0
    name = args.run_name or f"validation_radiative_shock_{datetime.now(timezone(timedelta(hours=9))):%Y%m%d_%H%M%S}_JST"
    root = args.output_root.expanduser().resolve()/name
    root.mkdir(parents=True, exist_ok=False)
    manifest = base_manifest(repo, "radiative_shock_validation", args.mpi_ranks)
    summary = {"run": name, "status": "running", "started_at": utc_now(),
               "preset": args.preset, "selected_cases": args.cases,
               "resolution": args.nx, "slab": args.slab, "layout": args.layout,
               "cases": {}}
    (root/"reference").mkdir()
    for filename in (*REFERENCE_NAME.values(), "README.md", "radiative_shock_mach2_bvp.py",
                     "radiative_shock_mach5_bvp.py", "radiative_shock_semianalytic.py"):
        shutil.copy2(SOURCE_DIR/filename, root/"reference"/filename)
    manifest["analytic_references"] = {case: sha256_file(root/"reference"/REFERENCE_NAME[case])
                                       for case in args.cases}
    write_manifest(root/"manifest.json", manifest)
    (root/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    write_html(root, summary)
    try:
        executable = ensure_build(args, repo, root, manifest)
        write_manifest(root/"manifest.json", manifest)
        run_cases(args, repo, root, executable, manifest)
        for case in args.cases:
            location = root/"cases"/case
            execution = verify_case(location, case, args.nlim)
            summary["cases"][case] = {"execution": execution}
            if args.nlim is None:
                summary["cases"][case]["analysis"] = analyze_case(root, case)
            (root/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
            write_html(root, summary)
        if args.nlim is None:
            summary["acceptance"] = assess_accuracy(summary)
        summary["status"] = ("exploratory_cycle_limit" if args.nlim is not None else
                             "completed_with_solver_caveat" if any(
                                 summary["cases"][case]["execution"]["solver"]["relaxed_accepted"]
                                 for case in args.cases) else
                             "passed" if summary["acceptance"]["passed"] else
                             "accuracy_not_met")
        summary["finished_at"] = utc_now()
        manifest["status"] = summary["status"]
        manifest["analysis_script_sha256"] = sha256_file(Path(__file__))
        manifest["analysis_dependency_hashes"] = analysis_hashes()
        manifest["metrics"] = summary["cases"]
        if args.nlim is None:
            manifest["acceptance"] = summary["acceptance"]
        write_manifest(root/"manifest.json", manifest)
        (root/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        write_html(root, summary)
        print(root, flush=True)
        return 0
    except Exception as error:
        summary.update(status="failed", error=str(error), finished_at=utc_now())
        manifest.update(status="failed", error=str(error))
        (root/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        write_manifest(root/"manifest.json", manifest)
        write_html(root, summary)
        print(f"Radiative shock validation failed: {error}\n{root}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
