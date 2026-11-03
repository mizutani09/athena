#!/usr/bin/env python3
"""Run and quantify the Zhang et al. 2D static gas/radiation equilibrium."""

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
from advection_pulse_validation import ensure_build, physical_time_unit, save_csv  # noqa: E402
from nrfld_validation.acceptance import (static_equilibrium as assess_accuracy,
                                         acceptance_html)  # noqa: E402
from diffusion_validation import materialize_input  # noqa: E402
from nrfld_validation.io import find_repo_root  # noqa: E402
from nrfld_validation.manifest import base_manifest, sha256_file, utc_now, write_manifest  # noqa: E402
from run_suite import checked, effective_settings  # noqa: E402
from shock_tube_validation import parse_solver_log, runtime_resources  # noqa: E402

TEMPLATE = Path("inputs/radiation/athinput.nrfld_static_equilibrium")
FINAL_TIME_CODE = 1888.84724652487
FIELDS = ("rho", "vel1", "vel2", "vel3", "vel_mag", "Pgas", "Prad", "Ptot",
          "Tgas", "Trad", "Erad")
POSITIVE = ("rho", "Pgas", "Prad", "Ptot", "Tgas", "Trad", "Erad")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/d/output/rad"))
    parser.add_argument("--run-name")
    parser.add_argument("--postprocess", type=Path)
    parser.add_argument("--reassess", type=Path,
                        help="Apply current accuracy criteria to a completed saved report.")
    parser.add_argument("--reuse-build", type=Path)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--cfl", type=float, default=.03)
    parser.add_argument("--mpi-ranks", type=int, default=4)
    parser.add_argument("--build-jobs", type=int, default=8)
    parser.add_argument("--run-timeout", type=float, default=21600)
    parser.add_argument("--build-timeout", type=float, default=900)
    args = parser.parse_args()
    if args.nx < 16 or args.nx % 4 or args.cfl <= 0:
        parser.error("nx must be a multiple of four >=16; cfl must be positive")
    if min(args.mpi_ranks, args.build_jobs, args.run_timeout, args.build_timeout) <= 0:
        parser.error("MPI ranks, build jobs and timeouts must be positive")
    return args


def overrides(n: int, cfl: float) -> list[str]:
    half_z = 2*1024/n
    return [f"mesh/nx1={n}", f"mesh/nx2={n}",
            f"mesh/x3min={-half_z:.17g}", f"mesh/x3max={half_z:.17g}",
            f"time/cfl_number={cfl:.17g}", "time/nlim=200000",
            "time/ncycle_out=1000", "output2/ghost_zones=false",
            "output3/ghost_zones=false", "nrfld/linear_solver=fgmres",
            "nrfld/failure_policy=abort", "nrfld/nr_threshold=1e-8",
            "nrfld/threshold=1e-10", "nrfld/nr_backtrack_max=12",
            "nrfld/nr_min_step_scale=0.0001",
            "job/problem_id=static_equilibrium"]


def outputs(root: Path) -> list[Path]:
    paths = sorted((root/"case/output").glob("*.uov_final.*.athdf"))
    if len(paths) < 2:
        raise ValueError("static equilibrium needs initial and evolved HDF5 snapshots")
    return paths


def load(path: Path) -> dict:
    with h5py.File(path) as hdf:
        names = [x.decode() if isinstance(x, bytes) else str(x)
                 for x in hdf.attrs["VariableNames"]]
        raw = hdf["user_out_var"][:]
        result = {"time_code": float(hdf.attrs["Time"]),
                  "cycles": int(hdf.attrs["NumCycles"]),
                  "locations": hdf["LogicalLocations"][:],
                  "block_size": tuple(map(int, hdf.attrs["MeshBlockSize"])),
                  "grid_size": tuple(map(int, hdf.attrs["RootGridSize"])),
                  "x1v": hdf["x1v"][:], "x2v": hdf["x2v"][:]}
        for field in FIELDS:
            result[field] = np.asarray(raw[names.index(field)], dtype=float)
    for field in FIELDS:
        if not np.all(np.isfinite(result[field])):
            raise ValueError(f"non-finite {field}: {path}")
    for field in POSITIVE:
        if np.any(result[field] <= 0):
            raise ValueError(f"non-positive {field}: {path}")
    return result


def check_pair(initial: dict, current: dict):
    if initial["grid_size"] != current["grid_size"] or initial["block_size"] != current["block_size"]:
        raise ValueError("grid changed during static equilibrium run")
    if not np.array_equal(initial["locations"], current["locations"]):
        raise ValueError("MeshBlock ordering changed")


def metrics(initial: dict, current: dict, gamma: float) -> dict:
    check_pair(initial, current)
    result = {}
    for field in ("rho", "Tgas", "Trad", "Erad", "Pgas", "Prad", "Ptot"):
        old, new = initial[field], current[field]
        difference = new-old
        result[field] = {
            "relative_L1_drift": float(np.sum(np.abs(difference))/np.sum(np.abs(old))),
            "max_pointwise_relative_drift": float(np.max(np.abs(difference)/np.abs(old))),
            "max_abs_drift": float(np.max(np.abs(difference)))}
    result["velocity_max_cm_s"] = float(np.max(current["vel_mag"]))
    result["velocity_rms_cm_s"] = float(np.sqrt(np.mean(current["vel_mag"]**2)))
    ptot = current["Ptot"]
    result["total_pressure_spatial_range_over_mean"] = float(
        (ptot.max()-ptot.min())/np.mean(ptot))
    result["mass_relative_drift"] = float(
        abs(np.sum(current["rho"])-np.sum(initial["rho"]))
        /np.sum(initial["rho"]))
    def energy(state):
        return state["Pgas"]/(gamma-1)+state["Erad"]+0.5*state["rho"]*state["vel_mag"]**2
    e0, ef = energy(initial), energy(current)
    result["total_energy_relative_drift"] = float(abs(ef.sum()-e0.sum())/e0.sum())
    return result


def map2d(state: dict, field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = state["grid_size"]
    bx, by, bz = state["block_size"]
    if nz != bz or nx % bx or ny % by:
        raise ValueError("unexpected 2D slab geometry")
    values = np.empty((ny, nx), dtype=float)
    x = np.empty(nx, dtype=float)
    y = np.empty(ny, dtype=float)
    for block, loc in enumerate(state["locations"]):
        i0, j0 = int(loc[0])*bx, int(loc[1])*by
        values[j0:j0+by, i0:i0+bx] = state[field][block].mean(axis=0)
        x[i0:i0+bx] = state["x1v"][block]
        y[j0:j0+by] = state["x2v"][block]
    return x, y, values


def plot_maps(root: Path, initial: dict, final: dict):
    x, y, t0 = map2d(initial, "Tgas")
    _, _, tf = map2d(final, "Tgas")
    _, _, speed = map2d(final, "vel_mag")
    extent = [x[0]-(x[1]-x[0])/2, x[-1]+(x[1]-x[0])/2,
              y[0]-(y[1]-y[0])/2, y[-1]+(y[1]-y[0])/2]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for axis, array, title, cmap in ((axes[0, 0], t0, "Initial Tgas [K]", "magma"),
                                     (axes[0, 1], tf, "Final Tgas [K]", "magma"),
                                     (axes[1, 0], (tf-t0)/t0, "Relative Tgas drift", "coolwarm"),
                                     (axes[1, 1], speed, "Final |v| [cm/s]", "viridis")):
        color_limits = {}
        if title == "Relative Tgas drift":
            scale = max(float(np.max(np.abs(array))), 1e-7)
            color_limits = {"vmin": -scale, "vmax": scale}
        im = axis.imshow(array, origin="lower", extent=extent,
                         cmap=cmap, **color_limits)
        axis.set(xlabel="x [cm]", ylabel="y [cm]", title=title)
        fig.colorbar(im, ax=axis)
    for suffix in ("png", "pdf"):
        fig.savefig(root/"figures"/f"equilibrium_maps.{suffix}", dpi=180)
    plt.close(fig)


def plot_radial(root: Path, initial: dict, final: dict):
    x, y, _ = map2d(initial, "Tgas")
    radius = np.hypot(*np.meshgrid(x, y)).ravel()
    bins = np.linspace(0, radius.max(), 40)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, field in zip(axes, ("Tgas", "Trad")):
        for state, label, style in ((initial, "initial", "-"), (final, "final", "--")):
            _, _, values = map2d(state, field)
            index = np.digitize(radius, bins)
            radial = np.array([values.ravel()[index == i].mean() if np.any(index == i)
                               else np.nan for i in range(1, len(bins))])
            axis.plot(.5*(bins[:-1]+bins[1:]), radial, ls=style, label=label)
        axis.set(xlabel="radius [cm]", ylabel=f"{field} [K]")
        axis.grid(alpha=.25)
        axis.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(root/"figures"/f"equilibrium_radial.{suffix}", dpi=180)
    plt.close(fig)


def movie(root: Path, initial: dict, paths: list[Path], max_drift: float,
          max_speed: float):
    x, y, _ = map2d(initial, "Tgas")
    extent = [x.min(), x.max(), y.min(), y.max()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    drift_image = axes[0].imshow(np.zeros((len(y), len(x))), origin="lower",
                                  extent=extent, cmap="coolwarm",
                                  vmin=-max(max_drift, 1e-8), vmax=max(max_drift, 1e-8))
    velocity_image = axes[1].imshow(np.zeros((len(y), len(x))), origin="lower",
                                     extent=extent, cmap="viridis", vmin=0,
                                     vmax=max(max_speed, 1e-8))
    axes[0].set_title("(Tgas - initial) / initial")
    axes[1].set_title("|v| [cm/s]")
    for axis, image in ((axes[0], drift_image), (axes[1], velocity_image)):
        axis.set(xlabel="x [cm]", ylabel="y [cm]")
        fig.colorbar(image, ax=axis)
    with FFMpegWriter(fps=8, bitrate=1400).saving(fig,
            str(root/"video"/"equilibrium_evolution.mp4"), dpi=110) as writer:
        initial_map = map2d(initial, "Tgas")[2]
        for path in paths:
            state = load(path)
            drift = (map2d(state, "Tgas")[2]-initial_map)/initial_map
            velocity = map2d(state, "vel_mag")[2]
            drift_image.set_data(drift)
            velocity_image.set_data(velocity)
            fig.suptitle(f"t = {state['time_code']:.4g} code units")
            writer.grab_frame()
    plt.close(fig)


def analyze(root: Path, settings: dict) -> dict:
    plt.rcParams.update({"font.size": 16, "axes.labelsize": 17,
                         "xtick.labelsize": 13, "ytick.labelsize": 13,
                         "legend.fontsize": 14})
    paths = outputs(root)
    first, last = load(paths[0]), load(paths[-1])
    if first["time_code"] != 0 or not math.isclose(last["time_code"], FINAL_TIME_CODE,
                                                    rel_tol=1e-6):
        raise ValueError("static equilibrium snapshots do not span the target time")
    log = (root/"case/run.log").read_text(errors="replace")
    if "Terminating on time limit" not in log or re.search(r"FATAL|Segmentation fault", log, re.I):
        raise ValueError("static equilibrium did not terminate cleanly")
    solver = parse_solver_log(root/"case/run.log", last["cycles"])
    active_lines = [line for line in log.splitlines()
                    if line.startswith("[NR] status=")
                    and "status=dt_zero_initialization" not in line]
    zero_iteration_steps = sum("iterations=0 " in line for line in active_lines)
    solver["zero_iteration_steps"] = zero_iteration_steps
    solver["fgmres_not_needed"] = (solver["fgmres_calls"] == 0
                                   and zero_iteration_steps == last["cycles"])
    strict = (solver["nr_status_count"] == last["cycles"]
              and solver["strict_converged"] == last["cycles"]
              and solver["relaxed_accepted"] == 0)
    if not strict or not (solver["good"] or solver["fgmres_not_needed"]):
        raise ValueError(f"static equilibrium solver failed strict convergence: {solver}")
    solver["good"] = True
    gamma = float(settings["hydro/gamma"])
    rows = []
    for path in paths:
        state = load(path)
        row = metrics(first, state, gamma)
        rows.append({"time_code": state["time_code"],
                     "time_s": state["time_code"]*physical_time_unit(settings),
                     "Tgas_max_relative": row["Tgas"]["max_pointwise_relative_drift"],
                     "Trad_max_relative": row["Trad"]["max_pointwise_relative_drift"],
                     "rho_max_relative": row["rho"]["max_pointwise_relative_drift"],
                     "velocity_max_cm_s": row["velocity_max_cm_s"],
                     "Ptot_spatial_relative": row["total_pressure_spatial_range_over_mean"]})
    final = metrics(first, last, gamma)
    (root/"figures").mkdir(exist_ok=True)
    (root/"video").mkdir(exist_ok=True)
    save_csv(root/"drift_history.csv", {key: np.array([row[key] for row in rows])
                                         for key in rows[0]})
    plot_maps(root, first, last)
    plot_radial(root, first, last)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for field in ("Tgas", "Trad", "rho"):
        axes[0].plot([r["time_s"] for r in rows],
                     [100*r[f"{field}_max_relative"] for r in rows], label=field)
    axes[0].set_ylim(0, max(1e-5, 1.1*max(
        100*r[f"{field}_max_relative"]
        for r in rows for field in ("Tgas", "Trad", "rho"))))
    axes[1].plot([r["time_s"] for r in rows],
                 [r["velocity_max_cm_s"] for r in rows], label="max |v|")
    axes[0].set(xlabel="time [s]", ylabel="max pointwise drift [%]")
    axes[1].set(xlabel="time [s]", ylabel="max |v| [cm/s]")
    for axis in axes: axis.grid(alpha=.25)
    axes[0].legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(root/"figures"/f"equilibrium_history.{suffix}", dpi=180)
    plt.close(fig)
    movie(root, first, paths,
          max(row["Tgas_max_relative"] for row in rows),
          max(row["velocity_max_cm_s"] for row in rows))
    x, y, init_temp = map2d(first, "Tgas")
    _, _, final_temp = map2d(last, "Tgas")
    save_csv(root/"temperature_map.csv", {
        "x_cm": np.tile(x, len(y)), "y_cm": np.repeat(y, len(x)),
        "Tgas_initial_K": init_temp.ravel(), "Tgas_final_K": final_temp.ravel(),
        "Tgas_delta_K": (final_temp-init_temp).ravel()})
    return {"final": final, "snapshots": len(paths), "cycles": last["cycles"],
            "final_time_s": last["time_code"]*physical_time_unit(settings),
            "solver": solver, "max_rss_kib_launcher": runtime_resources(root/"case/resources.txt"),
            "initial_output_sha256": sha256_file(paths[0]),
            "final_output_sha256": sha256_file(paths[-1])}


def write_html(root: Path, summary: dict):
    analysis = summary.get("analysis")
    tone = ("ok" if summary["status"] == "passed" else
            "failed" if summary["status"] == "failed" else
            "warning" if summary["status"] == "accuracy_not_met" else "pending")
    settings = summary.get("effective_settings", {})
    grid = "×".join(settings.get(f"mesh/nx{axis}", "?") for axis in (1, 2, 3))
    detail = "<p>Calculation in progress. The run log is linked below.</p>"
    if analysis:
        m = analysis["final"]
        rows = ''.join(f'<tr><th>{field}</th><td>{100*m[field]["relative_L1_drift"]:.4g}%</td>'
                       f'<td>{100*m[field]["max_pointwise_relative_drift"]:.4g}%</td></tr>'
                       for field in ("rho", "Tgas", "Trad", "Erad", "Ptot"))
        detail = f'''<p>Final physical time <strong>{analysis['final_time_s']:.8g} s</strong>;
        {analysis['cycles']} cycles; {analysis['snapshots']} snapshots.
        Every Newton step converged strictly.
        {('FGMRES was not called because every Newton residual was already below '
          'threshold.' if analysis['solver']['fgmres_not_needed'] else
          'Every FGMRES residual met its target.')}</p>
        <table><thead><tr><th>Quantity</th><th>Relative L1 drift</th><th>Maximum pointwise drift</th>
        </tr></thead><tbody>{rows}</tbody></table>
        <p><strong>Maximum velocity:</strong> {m['velocity_max_cm_s']:.5g} cm/s.
        <strong>Total-pressure spatial range / mean:</strong>
        {100*m['total_pressure_spatial_range_over_mean']:.5g}%.
        <strong>Mass drift:</strong> {100*m['mass_relative_drift']:.5g}%.
        <strong>Total energy drift:</strong> {100*m['total_energy_relative_drift']:.5g}%.
        The initial HDF5 snapshot is the reference at every cell; there is no spatial shift.
        Scalar HDF5 outputs are single precision, so relative changes below roughly
        10⁻⁷ may round to zero.</p>
        <figure><img src="figures/equilibrium_maps.png" alt="Initial final and drift maps"></figure>
        <figure><img src="figures/equilibrium_radial.png" alt="Radial temperature profiles"></figure>
        <figure><img src="figures/equilibrium_history.png" alt="Drift history"></figure>
        <video controls preload="metadata" width="1100"><source
        src="video/equilibrium_evolution.mp4" type="video/mp4"></video>
        <p><a href="drift_history.csv">History CSV</a> ·
        <a href="temperature_map.csv">Temperature map CSV</a> ·
        <a href="figures/equilibrium_maps.pdf">Map PDF</a> ·
        <a href="figures/equilibrium_radial.pdf">Radial PDF</a></p>'''
    root.joinpath("index.html").write_text(f'''<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Static equilibrium validation</title><style>
:root{{font-family:system-ui,sans-serif;font-size:21px;line-height:1.6}}
body{{max-width:1450px;margin:1.5rem auto;padding:0 1rem;color:#17212b}}
h1{{font-size:2.1rem}}h2{{font-size:1.55rem}}figure{{margin:1.5rem 0}}
img{{max-width:100%;height:auto}}table{{border-collapse:collapse;font-size:1rem}}
th,td{{padding:.5rem .8rem;border-bottom:1px solid #cbd5e1;text-align:right}}
th:first-child{{text-align:left}}a{{color:#075ca8}}
.status{{font-weight:700}}.ok{{color:#08764b}}.failed{{color:#b42318}}
.pending{{color:#526170}}.warning{{color:#9a4f00}}
@media(max-width:700px){{:root{{font-size:19px}}}}
</style></head><body><h1>Static gas/radiation equilibrium</h1>
<p class="status {tone}">{escape(summary['status'].upper())}</p>
{acceptance_html(summary.get('acceptance'))}
<p>The radial initial state has constant total pressure and Tgas = Trad.
We compare every saved field with its initial cell value to test whether the
state remains in equilibrium under the current FGMRES NR-FLD evolution.
This local slab has {escape(grid)} cells; the published 2D test used 512×512,
so its quoted velocity should be treated as a separate resolution reference.</p>
{detail}<h2>Reproduction</h2><p>
<a href="summary.json">Summary JSON</a> · <a href="manifest.json">Manifest</a> ·
<a href="build/build_source/bin/athena">Binary</a> ·
<a href="case/input.used">Input</a> · <a href="case/run.log">Run log</a> ·
<a href="case/resources.txt">Resources</a></p></body></html>''')


def main() -> int:
    args = parse_args()
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
            (args.run_name or f"validation_static_equilibrium_{datetime.now(timezone(timedelta(hours=9))):%Y%m%d_%H%M%S}_JST"))
    if not args.postprocess:
        root.mkdir(parents=True, exist_ok=False)
    manifest_path, summary_path = root/"manifest.json", root/"summary.json"
    manifest = (json.loads(manifest_path.read_text()) if manifest_path.exists()
                else base_manifest(repo, "static_equilibrium_validation", args.mpi_ranks))
    summary = (json.loads(summary_path.read_text()) if summary_path.exists()
               else {"status": "running", "started_at": utc_now()})
    summary["run"] = root.name
    write_html(root, summary)
    try:
        executable = ensure_build(root, repo, args, manifest)
        directory = root/"case"
        if not args.postprocess:
            directory.mkdir()
            (directory/"output").mkdir()
            materialize_input(repo/TEMPLATE, directory/"input.used", overrides(args.nx, args.cfl))
            command = ["/usr/bin/time", "-v", "-o", str(directory/"resources.txt"),
                       "mpirun", "-np", str(args.mpi_ranks), str(executable),
                       "-i", str(directory/"input.used"), "-d", str(directory/"output")]
            record = checked(command, directory, directory/"run.log",
                             args.run_timeout, "run_static_equilibrium")
            manifest["commands"].setdefault("records", []).append(record)
        settings = effective_settings(directory/"input.used", [])
        summary["effective_settings"] = settings
        summary["analysis"] = analyze(root, settings)
        summary["acceptance"] = assess_accuracy(summary)
        summary["status"] = "passed" if summary["acceptance"]["passed"] else "accuracy_not_met"
        summary["finished_at"] = utc_now()
        manifest["status"] = summary["status"]
        manifest["metrics"] = summary["analysis"]
        manifest["acceptance"] = summary["acceptance"]
        manifest["input_sha256"] = sha256_file(directory/"input.used")
        manifest["analysis_script_sha256"] = sha256_file(Path(__file__))
        manifest["acceptance_policy_sha256"] = sha256_file(HERE/"nrfld_validation/acceptance.py")
        write_manifest(manifest_path, manifest)
        summary_path.write_text(json.dumps(summary, indent=2)+"\n")
        write_html(root, summary)
        print(root, flush=True)
        return 0
    except Exception as error:
        summary.update(status="failed", error=str(error), finished_at=utc_now())
        manifest.update(status="failed", error=str(error))
        write_manifest(manifest_path, manifest)
        summary_path.write_text(json.dumps(summary, indent=2)+"\n")
        write_html(root, summary)
        print(f"Static equilibrium validation failed: {error}; {root}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
