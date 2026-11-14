#!/usr/bin/env python3
"""Run and report the Zhang et al. advecting radiation pulse comparison."""

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
from diffusion_validation import materialize_input  # noqa: E402
from nrfld_validation.io import find_repo_root  # noqa: E402
from nrfld_validation.acceptance import (advection_pulse as assess_accuracy,
                                         acceptance_html)  # noqa: E402
from nrfld_validation.manifest import (base_manifest, sha256_file, tree_sha256,
                                       utc_now, write_manifest)  # noqa: E402
from run_suite import (binary_configuration, checked, effective_settings,
                       prepare_build_source)  # noqa: E402
from shock_tube_validation import parse_solver_log, runtime_resources, verify_source_snapshot  # noqa: E402

TEMPLATE = Path("inputs/radiation/athinput.nrfld_advect_pulse")
FIELDS = ("rho", "vel1", "Tgas", "Trad", "Erad", "Pgas", "Prad", "Ptot")
POSITIVE = ("rho", "Tgas", "Trad", "Erad", "Pgas", "Prad", "Ptot")
A_RAD = 7.5657e-15
FINAL_TIME_CODE = 906.75278
BACKGROUND_V = 1.0e6
SOURCE_PLOT = HERE/"reference/advect_pulse_plot_user.py"


def args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/d/output/rad"))
    parser.add_argument("--run-name")
    parser.add_argument("--postprocess", type=Path,
                        help="Analyze complete existing static/moving cases without rerunning them.")
    parser.add_argument("--reassess", type=Path,
                        help="Apply current accuracy criteria to a completed saved report.")
    parser.add_argument("--reuse-build", type=Path,
                        help="Copy a matching isolated build_source; required for postprocessing a manual run.")
    parser.add_argument("--nx1", type=int, default=512)
    parser.add_argument("--cfl", type=float, default=0.03)
    parser.add_argument("--mpi-ranks", type=int, default=4)
    parser.add_argument("--build-jobs", type=int, default=8)
    parser.add_argument("--run-timeout", type=float, default=21600)
    parser.add_argument("--build-timeout", type=float, default=900)
    args = parser.parse_args()
    if args.nx1 < 32 or args.nx1 % 4 or min(args.cfl, args.run_timeout,
                                               args.build_timeout) <= 0:
        parser.error("nx1 must be a multiple of four >=32; cfl/timeouts must be positive")
    if args.mpi_ranks < 1 or args.build_jobs < 1:
        parser.error("MPI ranks and build jobs must be positive")
    return args


def overrides(case: str, n: int, cfl: float) -> list[str]:
    if case not in ("static", "moving"):
        raise ValueError(case)
    dx = 1024.0/n
    transverse_half = 2*dx
    return [f"mesh/nx1={n}", f"mesh/x2min={-transverse_half:.17g}",
            f"mesh/x2max={transverse_half:.17g}",
            f"mesh/x3min={-transverse_half:.17g}",
            f"mesh/x3max={transverse_half:.17g}",
            f"time/cfl_number={cfl:.17g}", "time/nlim=200000",
            "time/ncycle_out=1000", "output2/ghost_zones=false",
            "output3/ghost_zones=false", "output3/dt=45.337639",
            "nrfld/linear_solver=fgmres", "nrfld/failure_policy=abort",
            "nrfld/nr_threshold=1e-8", "nrfld/threshold=1e-10",
            "nrfld/nr_backtrack_max=12", "nrfld/nr_min_step_scale=0.0001",
            f"problem/v0={0 if case == 'static' else BACKGROUND_V:g}",
            f"job/problem_id=advect_pulse_{case}"]


def ensure_build(root: Path, repo: Path, args, manifest: dict) -> Path:
    build = root/"build"
    build.mkdir(exist_ok=True)
    source = build/"build_source"
    records = manifest["commands"].setdefault("records", [])
    if not source.exists():
        if args.reuse_build:
            original = args.reuse_build.expanduser().resolve()
            shutil.copytree(original, source, ignore=shutil.ignore_patterns("*.o"))
            verify_source_snapshot(repo, source, adapted_reference=False)
            manifest["build_imported_from"] = str(original)
        else:
            if args.postprocess:
                raise ValueError("--postprocess needs --reuse-build for a run without build/")
            prepare_build_source(repo, source)
            command = [sys.executable, str(source/"configure.py"),
                       "--prob=nrfld_advect_radiation_pulse", "--eos=adiabatic",
                       "-nrmgfld", "-hdf5", "-mpi", "--cflag=-O3"]
            records.append(checked(command, source, build/"configure.log",
                                   args.build_timeout, "configure"))
            records.append(checked(["make", f"-j{args.build_jobs}"], source,
                                   build/"build.log", args.build_timeout, "build"))
    executable = source/"bin/athena"
    configuration = binary_configuration(executable, build, 30., records)
    expected = {"problem_generator": "nrfld_advect_radiation_pulse",
                "equation_of_state": "adiabatic", "fld_with_newton-raphson": "ON",
                "mpi_parallelism": "ON", "hdf5_output": "ON"}
    for key, value in expected.items():
        if configuration.get(key) != value:
            raise ValueError(f"unexpected binary setting {key}: {configuration.get(key)}")
    manifest["build"] = {"source_sha256": tree_sha256(source/"src"),
                         "binary_sha256": sha256_file(executable),
                         "configuration": configuration}
    return executable


def run_cases(root: Path, repo: Path, args, executable: Path, manifest: dict):
    for case in ("static", "moving"):
        directory = root/"cases"/case
        directory.mkdir(parents=True)
        (directory/"output").mkdir()
        materialize_input(repo/TEMPLATE, directory/"input.used",
                          overrides(case, args.nx1, args.cfl))
        command = ["/usr/bin/time", "-v", "-o", str(directory/"resources.txt"),
                   "mpirun", "-np", str(args.mpi_ranks), str(executable),
                   "-i", str(directory/"input.used"), "-d", str(directory/"output")]
        record = checked(command, directory, directory/"run.log",
                         args.run_timeout, f"run_{case}")
        manifest["commands"].setdefault("records", []).append(record)


def outputs(directory: Path) -> list[Path]:
    found = sorted((directory/"output").glob("*.uov_x_final.*.athdf"))
    if len(found) < 2:
        raise ValueError(f"need initial and final user-output HDF5: {directory}")
    return found


def profile(path: Path) -> dict:
    with h5py.File(path) as data:
        names = [x.decode() if isinstance(x, bytes) else str(x)
                 for x in data.attrs["VariableNames"]]
        raw = np.asarray(data["user_out_var"])
        x = np.asarray(data["x1v"], dtype=float).ravel()
        order = np.argsort(x)
        result = {"x": x[order], "time_code": float(data.attrs["Time"]),
                  "cycles": int(data.attrs["NumCycles"])}
        for field in FIELDS:
            values = raw[names.index(field)].reshape(-1)
            if len(values) != len(x):
                raise ValueError(f"expected x-sliced HDF5 for {field}: {path}")
            result[field] = np.asarray(values[order], dtype=float)
    if np.any(np.diff(result["x"]) <= 0):
        raise ValueError(f"non-increasing x grid: {path}")
    for field in FIELDS:
        if not np.all(np.isfinite(result[field])):
            raise ValueError(f"non-finite {field}: {path}")
    for field in POSITIVE:
        if np.any(result[field] <= 0):
            raise ValueError(f"non-positive {field}: {path}")
    return result


def verify_case(directory: Path, case: str, repo: Path) -> dict:
    paths = outputs(directory)
    initial, final = profile(paths[0]), profile(paths[-1])
    settings = effective_settings(directory/"input.used", [])
    expected = effective_settings(repo/TEMPLATE, overrides(case, len(final["x"]),
                                                    float(settings["time/cfl_number"])))
    for key in ("mesh/nx1", "problem/v0", "nrfld/linear_solver",
                "nrfld/failure_policy", "nrfld/nr_backtrack_max"):
        if settings.get(key) != expected.get(key):
            raise ValueError(f"unexpected {key}: {settings.get(key)}")
    if not math.isclose(final["time_code"], FINAL_TIME_CODE, rel_tol=1e-6):
        raise ValueError(f"did not reach final time: {final['time_code']}")
    if initial["time_code"] != 0:
        raise ValueError("initial snapshot is not at t=0")
    log = (directory/"run.log").read_text(errors="replace")
    if "Terminating on time limit" not in log or re.search(r"FATAL|Segmentation fault", log, re.I):
        raise ValueError(f"run did not terminate cleanly: {directory}")
    cycles = final["cycles"]
    solver = parse_solver_log(directory/"run.log", cycles)
    if not solver["good"] or solver["relaxed_accepted"]:
        raise ValueError(f"solver failed strict convergence: {directory}: {solver}")
    step_sizes = [float(value) for value in re.findall(
        r"^\[NR\] status=converged[^\n]+? dt=([^\s]+)", log, re.M)]
    if len(step_sizes) != cycles:
        raise ValueError(f"missing timestep log entries: {directory}")
    dx = float(np.median(np.diff(final["x"])))
    if not np.allclose(np.diff(final["x"]), dx, atol=1e-7):
        raise ValueError("non-uniform pulse grid")
    return {"cycles": cycles, "snapshots": len(paths), "time_code": final["time_code"],
            "nx1": len(final["x"]), "dx_cm": dx, "solver": solver,
            "max_step_code": max(step_sizes),
            "max_rss_kib_launcher": runtime_resources(directory/"resources.txt"),
            "final_output_sha256": sha256_file(paths[-1]),
            "input_sha256": sha256_file(directory/"input.used"),
            "effective_settings": settings}


def physical_time_unit(settings: dict) -> float:
    return float(settings["hydro/leng_unit"])/math.sqrt(
        float(settings["hydro/egas_unit"])/float(settings["hydro/rho_unit"]))


def align_moving(reference: dict, moving: dict, shift_cm: float) -> tuple[dict, str]:
    x = reference["x"]
    if not np.allclose(x, moving["x"], rtol=0, atol=1e-6):
        raise ValueError("moving and static x grids differ")
    dx = float(np.median(np.diff(x)))
    cells = shift_cm/dx
    aligned = {"x": x, "time_code": moving["time_code"], "cycles": moving["cycles"]}
    if abs(cells-round(cells)) < 1e-4:
        steps = int(round(cells))
        for field in FIELDS:
            aligned[field] = np.roll(moving[field], -steps)
        return aligned, f"exact periodic index shift of {steps} cells"
    length = dx*len(x)
    # y_moving(x + shift); extend periodically before interpolating.
    query = x + shift_cm
    for field in FIELDS:
        aligned[field] = np.interp(query, np.r_[x-length, x, x+length],
                                   np.tile(moving[field], 3))
    return aligned, "periodic linear interpolation"


def comparison(reference: dict, aligned: dict, v0: float) -> tuple[dict, dict]:
    metrics, arrays = {}, {}
    for field in FIELDS:
        moving = aligned[field] - (v0 if field == "vel1" else 0)
        static = reference[field]
        difference = moving-static
        arrays[field] = difference
        scale = float(np.sum(np.abs(static)))
        metrics[field] = {"relative_L1": float(np.sum(np.abs(difference))/scale)
                          if scale else None,
                          "max_abs": float(np.max(np.abs(difference))),
                          "max_pointwise_relative": float(np.max(np.abs(difference)
                              /np.maximum(np.abs(static), 1e-300)))
                          if field != "vel1" else None}
    for field, baseline in (("rho", 1.2), ("Tgas", 1e7),
                            ("Erad", A_RAD*1e28)):
        contrast = float(np.sum(np.abs(reference[field]-baseline)))
        metrics[field]["pulse_contrast_L1"] = (
            float(np.sum(np.abs(arrays[field]))/contrast) if contrast else None)
    metrics["vel1"]["max_abs_km_s"] = metrics["vel1"]["max_abs"]*1e-5
    return metrics, arrays


def save_csv(path: Path, columns: dict):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))


def plot_profile(root: Path, ref: dict, moved: dict):
    x = ref["x"]
    plt.rcParams.update({"font.size": 20, "axes.labelsize": 20,
                         "xtick.labelsize": 15, "ytick.labelsize": 15,
                         "legend.fontsize": 15})
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fields = (("rho", 1, r"$\rho$ [g cm$^{-3}$]"),
              ("vel1", 1e-5, r"$\delta v_x$ [km s$^{-1}$]"),
              ("Tgas", 1e-7, r"$T_{\rm gas}$ [$10^7$ K]"))
    for axis, (field, factor, label) in zip(axes, fields):
        axis.plot(x, ref[field]*factor, color="tab:blue", lw=1.8, label="static")
        axis.plot(x, moved[field]*factor, color="tab:orange", lw=1.8,
                  ls="--", label="moving, shifted back")
        axis.set_ylabel(label)
        axis.grid(alpha=.25)
    axes[0].legend()
    axes[-1].set_xlabel(r"$x$ [cm]")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(root/"figures"/f"pulse_profiles.{suffix}", dpi=300)
    plt.close(fig)


def plot_radiation(root: Path, ref: dict, moved: dict, errors: dict):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for axis, field, label in ((axes[0, 0], "Trad", r"$T_{\rm rad}$ [K]"),
                               (axes[0, 1], "Erad", r"$E_{\rm rad}$ [erg cm$^{-3}$]")):
        axis.plot(ref["x"], ref[field], label="static", lw=1.8)
        axis.plot(ref["x"], moved[field], label="moving, shifted", ls="--", lw=1.8)
        axis.set_ylabel(label)
        axis.grid(alpha=.25)
        axis.legend()
    for axis, fields in ((axes[1, 0], ("rho", "Tgas")),
                         (axes[1, 1], ("Trad", "Erad"))):
        for field in fields:
            axis.plot(ref["x"], 100*errors[field]/ref[field], label=field)
        axis.set(xlabel="x [cm]", ylabel="relative difference [%]")
        axis.grid(alpha=.25)
        axis.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(root/"figures"/f"pulse_radiation_errors.{suffix}", dpi=300)
    plt.close(fig)


def make_movie(root: Path, pairs: list[tuple[dict, dict, dict]]):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    lines = {}
    for axis, field, label in ((axes[0], "Tgas", "$T_{gas}$ [K]"),
                               (axes[1], "Erad", "$E_{rad}$ [erg cm$^{-3}$]")):
        lines[field] = (axis.plot([], [], lw=1.5, label="static")[0],
                        axis.plot([], [], lw=1.5, ls="--", label="moving, shifted")[0])
        axis.set(xlim=(-512, 512), ylabel=label)
        axis.grid(alpha=.25)
        axis.legend()
    axes[-1].set_xlabel("x [cm]")
    fig.tight_layout()
    with FFMpegWriter(fps=6, bitrate=1400).saving(fig,
            str(root/"video"/"pulse_evolution.mp4"), dpi=120) as writer:
        for static, moving, aligned in pairs:
            for field, (left, right) in lines.items():
                left.set_data(static["x"], static[field])
                right.set_data(static["x"], aligned[field])
                axes[0 if field == "Tgas" else 1].relim()
                axes[0 if field == "Tgas" else 1].autoscale_view(scalex=False)
            fig.suptitle(f"t = {static['time_code']:.4g} code units")
            writer.grab_frame()
    plt.close(fig)


def analyze(root: Path, summary: dict) -> dict:
    static_paths = outputs(root/"cases/static")
    moving_paths = outputs(root/"cases/moving")
    if len(static_paths) != len(moving_paths):
        raise ValueError("static and moving have different snapshot counts")
    unit = physical_time_unit(summary["cases"]["static"]["effective_settings"])
    max_output_offset = max(summary["cases"][case]["max_step_code"]
                            for case in ("static", "moving"))
    pairs = []
    history = []
    for static_path, moving_path in zip(static_paths, moving_paths):
        static, moving = profile(static_path), profile(moving_path)
        # Athena writes a scheduled output after each case crosses the output
        # time. Their hydro timesteps differ, so intermediate snapshots need
        # not have identical Time attributes. The final tlim is exact in both.
        time_offset_code = moving["time_code"]-static["time_code"]
        # Each scheduled output is written at the first step crossing its
        # target time. The two runs can differ by up to one of their steps.
        if abs(time_offset_code) > max_output_offset + 1e-3:
            raise ValueError(f"unmatched snapshot times: {static_path}, {moving_path}")
        shift = BACKGROUND_V*moving["time_code"]*unit
        aligned, method = align_moving(static, moving, shift)
        aligned["vel1"] -= BACKGROUND_V
        metrics, errors = comparison(static, aligned, 0.)
        pairs.append((static, moving, aligned))
        history.append({"time_code": static["time_code"],
                        "time_s": static["time_code"]*unit,
                        "time_offset_code": time_offset_code,
                        "shift_cm": shift,
                        "rho_relative_L1": metrics["rho"]["relative_L1"],
                        "Tgas_relative_L1": metrics["Tgas"]["relative_L1"],
                        "Trad_relative_L1": metrics["Trad"]["relative_L1"],
                        "Erad_relative_L1": metrics["Erad"]["relative_L1"],
                        "velocity_max_km_s": metrics["vel1"]["max_abs_km_s"]})
    static, moving, aligned = pairs[-1]
    if not math.isclose(static["time_code"], moving["time_code"],
                        rel_tol=1e-7, abs_tol=1e-5):
        raise ValueError("final pulse snapshots do not share the target time")
    metrics, errors = comparison(static, aligned, 0.)
    source_plot_hash = sha256_file(SOURCE_PLOT) if SOURCE_PLOT.is_file() else None
    (root/"reference").mkdir(exist_ok=True)
    if SOURCE_PLOT.is_file():
        shutil.copy2(SOURCE_PLOT, root/"reference"/SOURCE_PLOT.name)
    (root/"figures").mkdir(exist_ok=True)
    (root/"video").mkdir(exist_ok=True)
    plot_profile(root, static, aligned)
    plot_radiation(root, static, aligned, errors)
    save_csv(root/"final_profiles.csv", {"x_cm": static["x"],
        **{f"static_{field}": static[field] for field in FIELDS},
        **{f"moving_shifted_{field}": aligned[field] for field in FIELDS},
        **{f"difference_{field}": errors[field] for field in FIELDS}})
    save_csv(root/"error_history.csv", {key: np.array([row[key] for row in history])
                                         for key in history[0]})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for field in ("rho", "Tgas", "Trad", "Erad"):
        axes[0].plot([row["time_s"] for row in history],
                     [100*row[f"{field}_relative_L1"] for row in history],
                     marker="o", ms=3, label=field)
    axes[1].plot([row["time_s"] for row in history],
                 [row["velocity_max_km_s"] for row in history], marker="o", ms=3)
    axes[0].set(xlabel="physical time [s]", ylabel="shifted relative L1 [%]")
    axes[1].set(xlabel="physical time [s]", ylabel="max velocity difference [km/s]")
    for axis in axes: axis.grid(alpha=.25)
    axes[0].legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(root/"figures"/f"pulse_error_history.{suffix}", dpi=300)
    plt.close(fig)
    make_movie(root, pairs)
    return {"time_unit_s": unit, "final_time_s": static["time_code"]*unit,
            "shift_cm": history[-1]["shift_cm"], "shift_cells": history[-1]["shift_cm"]
                /summary["cases"]["static"]["dx_cm"],
            "max_intermediate_time_offset_code": max(
                abs(row["time_offset_code"]) for row in history[:-1]),
            "final_time_offset_code": history[-1]["time_offset_code"],
            "alignment_method": method, "metrics": metrics,
            "snapshots": len(history), "source_plot_sha256": source_plot_hash,
            "comparison_basis": "moving field after periodic shift -v0*t; velocity minus v0"}


def write_html(root: Path, summary: dict):
    status = summary["status"]
    tone = ("ok" if status == "passed" else "failed" if status == "failed"
            else "warning" if status == "accuracy_not_met" else "pending")
    analysis = summary.get("analysis")
    details = "<p>Computation in progress; open the case logs below.</p>"
    if analysis:
        m = analysis["metrics"]
        rows = ''.join(f'<tr><th>{field}</th><td>{100*m[field]["relative_L1"]:.4g}%</td>'
                       f'<td>{100*m[field]["max_pointwise_relative"]:.4g}%</td></tr>'
                       for field in ("rho", "Tgas", "Trad", "Erad"))
        details = f'''<p><strong>Static versus moving, after the prescribed periodic shift:</strong>
        t = {analysis['final_time_s']:.8g} s; v₀t = {analysis['shift_cm']:.6g} cm
        ({analysis['shift_cells']:.6g} cells). {escape(analysis['alignment_method'])}.
        Moving velocity is shown after subtracting v₀ = 10⁶ cm/s.</p>
        <p>The final comparison uses the common target time. Intermediate
        snapshots are written after each case crosses a scheduled output time;
        their maximum time offset is
        {analysis['max_intermediate_time_offset_code']:.4g} code units
        ({analysis['max_intermediate_time_offset_code']*analysis['time_unit_s']:.4g} s).
        This offset is included in the time-history diagnostics.</p>
        <table><thead><tr><th>Field</th><th>Relative L1</th><th>Maximum pointwise relative error</th>
        </tr></thead><tbody>{rows}</tbody></table>
        <p>Maximum velocity perturbation difference:
        <strong>{m['vel1']['max_abs_km_s']:.4g} km/s</strong>. Pulse contrast L1:
        density {100*m['rho']['pulse_contrast_L1']:.4g}%,
        Tgas {100*m['Tgas']['pulse_contrast_L1']:.4g}%,
        Erad {100*m['Erad']['pulse_contrast_L1']:.4g}%.
        These contrast metrics divide by the static profile's final
        background-subtracted pulse magnitude.</p>
        <p>Both cases reached the final time with strict Newton convergence and FGMRES
        residuals below their configured targets. The supplied script's optional
        4096-cell static reference is not included in this local comparison.</p>
        <h2>Profiles and errors</h2>
        <figure><img src="figures/pulse_profiles.png" alt="Density velocity and gas temperature profiles">
        <figcaption>Three-panel presentation following the supplied Fig. 11 script.</figcaption></figure>
        <figure><img src="figures/pulse_radiation_errors.png" alt="Radiation and relative errors"></figure>
        <figure><img src="figures/pulse_error_history.png" alt="Error history"></figure>
        <video controls preload="metadata" width="1100"><source src="video/pulse_evolution.mp4" type="video/mp4"></video>
        <p><a href="final_profiles.csv">Final profile CSV</a> ·
        <a href="error_history.csv">Error history CSV</a> ·
        <a href="figures/pulse_profiles.pdf">Profile PDF</a> ·
        <a href="figures/pulse_radiation_errors.pdf">Error PDF</a></p>'''
    cards = []
    for case in ("static", "moving"):
        item = summary.get("cases", {}).get(case, {})
        if item:
            solver = item["solver"]
            cards.append(f'<p><strong>{case}:</strong> {item["cycles"]} cycles, '
                         f'Newton strict {solver["strict_converged"]}/{solver["cycles"]}, '
                         f'FGMRES {solver["fgmres_calls"]} calls; launcher RSS '
                         f'{item["max_rss_kib_launcher"]} KiB (MPI ranks not summed).</p>')
    root.joinpath("index.html").write_text(f'''<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Radiation pulse validation</title><style>
:root{{font-family:system-ui,sans-serif;font-size:21px;line-height:1.6}}
body{{max-width:1450px;margin:1.5rem auto;padding:0 1rem;color:#17212b}}
h1{{font-size:2.1rem}}h2{{font-size:1.55rem}}figure{{margin:1.5rem 0}}
img{{max-width:100%;height:auto}}figcaption{{font-weight:600}}
table{{border-collapse:collapse;font-size:1rem}}th,td{{padding:.5rem .8rem;
border-bottom:1px solid #cbd5e1;text-align:right}}th:first-child{{text-align:left}}
a{{color:#075ca8}}.status{{font-weight:700}}.ok{{color:#08764b}}
.failed{{color:#b42318}}.pending{{color:#526170}}
.warning{{color:#9a4f00}}
@media(max-width:700px){{:root{{font-size:19px}}}}
</style></head><body><h1>Advection of radiation pulse</h1>
<p class="status {tone}">{escape(status.upper())}</p>
{acceptance_html(summary.get('acceptance'))}
<p>Both runs use the same 1D-like periodic slab and the current FGMRES NR-FLD solver.
The moving result is translated by v₀t with periodic wrapping; no fitted phase shift
or amplitude rescaling enters the error.</p>{details}
<h2>Execution and reproducibility</h2>{''.join(cards)}
<p><a href="summary.json">Summary JSON</a> · <a href="manifest.json">Manifest</a> ·
<a href="build/build_source/bin/athena">Binary</a> ·
<a href="cases/static/input.used">Static input</a> ·
<a href="cases/moving/input.used">Moving input</a> ·
<a href="cases/static/run.log">Static log</a> ·
<a href="cases/moving/run.log">Moving log</a> ·
<a href="cases/static/resources.txt">Static resources</a> ·
<a href="cases/moving/resources.txt">Moving resources</a></p>
<p><a href="reference/advect_pulse_plot_user.py">User-supplied plotting reference</a></p>
</body></html>''')


def main() -> int:
    args = args_parser()
    os.environ["OMP_NUM_THREADS"] = "1"
    repo = find_repo_root(HERE)
    if args.reassess:
        root = args.reassess.expanduser().resolve()
        summary_path, manifest_path = root/"summary.json", root/"manifest.json"
        summary = json.loads(summary_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        if summary.get("status") not in ("diagnostic_complete", "passed", "accuracy_not_met"):
            raise ValueError("reassess requires a completed strict-solver report")
        summary["acceptance"] = assess_accuracy(summary)
        summary["status"] = "passed" if summary["acceptance"]["passed"] else "accuracy_not_met"
        manifest["status"] = summary["status"]
        manifest["acceptance"] = summary["acceptance"]
        manifest["analysis_script_sha256"] = sha256_file(Path(__file__))
        manifest["acceptance_policy_sha256"] = sha256_file(HERE/"nrfld_validation/acceptance.py")
        summary_path.write_text(json.dumps(summary, indent=2)+"\n")
        write_manifest(manifest_path, manifest)
        write_html(root, summary)
        print(root, flush=True)
        return 0
    root = (args.postprocess.expanduser().resolve() if args.postprocess else
            args.output_root.expanduser().resolve() /
            (args.run_name or f"validation_advection_pulse_{datetime.now(timezone(timedelta(hours=9))):%Y%m%d_%H%M%S}_JST"))
    if not args.postprocess:
        root.mkdir(parents=True, exist_ok=False)
    manifest_file, summary_file = root/"manifest.json", root/"summary.json"
    manifest = (json.loads(manifest_file.read_text()) if manifest_file.is_file()
                else base_manifest(repo, "advection_pulse_validation", args.mpi_ranks))
    summary = (json.loads(summary_file.read_text()) if summary_file.is_file()
               else {"status": "running", "started_at": utc_now(), "cases": {}})
    summary["run"] = root.name
    summary["requested_nx1"] = args.nx1
    summary["requested_cfl"] = args.cfl
    write_html(root, summary)
    try:
        executable = ensure_build(root, repo, args, manifest)
        if not args.postprocess:
            run_cases(root, repo, args, executable, manifest)
        for case in ("static", "moving"):
            summary["cases"][case] = verify_case(root/"cases"/case, case, repo)
        summary["analysis"] = analyze(root, summary)
        summary["acceptance"] = assess_accuracy(summary)
        summary["status"] = "passed" if summary["acceptance"]["passed"] else "accuracy_not_met"
        summary["finished_at"] = utc_now()
        manifest["status"] = summary["status"]
        manifest["metrics"] = summary["analysis"]
        manifest["case_results"] = summary["cases"]
        manifest["acceptance"] = summary["acceptance"]
        manifest["analysis_script_sha256"] = sha256_file(Path(__file__))
        manifest["acceptance_policy_sha256"] = sha256_file(HERE/"nrfld_validation/acceptance.py")
        manifest["user_plot_source_sha256"] = summary["analysis"]["source_plot_sha256"]
        write_manifest(manifest_file, manifest)
        summary_file.write_text(json.dumps(summary, indent=2)+"\n")
        write_html(root, summary)
        print(root, flush=True)
        return 0
    except Exception as error:
        summary.update(status="failed", error=str(error), finished_at=utc_now())
        manifest.update(status="failed", error=str(error))
        write_manifest(manifest_file, manifest)
        summary_file.write_text(json.dumps(summary, indent=2)+"\n")
        write_html(root, summary)
        print(f"Pulse validation failed: {error}; {root}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
