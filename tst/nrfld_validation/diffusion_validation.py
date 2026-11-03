#!/usr/bin/env python3
"""Run the NR-FLD Gaussian diffusion matrix and publish plots plus an MP4 report."""

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
import tempfile
from urllib.parse import quote

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nrfld-validation")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402
import numpy as np  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from diffusion_convergence import gaussian, relative_l1, orders  # noqa: E402
from nrfld_validation.io import (find_repo_root, latest_athdf, read_athdf,
                                 read_history)  # noqa: E402
from nrfld_validation.report_style import REPORT_TYPOGRAPHY  # noqa: E402
from nrfld_validation.manifest import (base_manifest, sha256_file, utc_now,
                                       write_manifest)  # noqa: E402
from run_suite import (CommandExecutionError, binary_configuration, build_case,
                       checked, effective_settings,
                       validate_binary_configuration)  # noqa: E402

DEFAULT_RESOLUTIONS = (128, 256, 512)
CFL_FACTORS = (1.0, 0.5, 0.25)
CS = math.sqrt((5.0/3.0)*100.0)
CHI = 2.99792458e10/(3.0*1.0e-7*0.4)*4000.0/(4.0e10)**2
ENERGY_UNIT = 1.0e7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/d/output/rad"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-cores", type=int, default=8)
    parser.add_argument("--final-time", type=float, default=0.0535561)
    parser.add_argument("--resolutions", nargs=3, type=int,
                        default=list(DEFAULT_RESOLUTIONS),
                        metavar=("N", "2N", "4N"))
    parser.add_argument("--base-steps", type=int, default=296,
                        help="Steps for the coarsest grid at CFL multiplier 1 (default CFL≈0.299).")
    parser.add_argument("--build-timeout", type=float, default=900)
    parser.add_argument("--run-timeout", type=float, default=900)
    args = parser.parse_args()
    if (args.max_cores < 1 or args.base_steps < 4 or args.final_time <= 0
            or args.build_timeout <= 0 or args.run_timeout <= 0):
        parser.error("cores, steps, final time and timeouts must be positive")
    if (args.resolutions[0] < 8 or args.resolutions[0] % 8
            or args.resolutions != [args.resolutions[0]*factor for factor in (1, 2, 4)]):
        parser.error("resolutions must be N, 2N, 4N with N divisible by 8")
    base_cfl = args.final_time/args.base_steps*CS*args.resolutions[0]
    if base_cfl > 0.3:
        parser.error(f"base CFL {base_cfl:.4g} exceeds 0.3; increase --base-steps")
    return args


def tag(factor: float) -> str:
    return {1.0: "cfl_1", 0.5: "cfl_1_2", 0.25: "cfl_1_4"}[factor]


def steps_for(n: int, factor: float, base_steps: int, first_resolution: int) -> int:
    return base_steps*(n//first_resolution)*int(round(1/factor))


def materialize_input(source: Path, target: Path, overrides: list[str]) -> dict:
    """Put all effective values in the input, including keys absent from the template."""
    settings = effective_settings(source, overrides)
    sections: dict[str, list[str]] = {}
    for key, value in settings.items():
        section, name = key.split("/", 1)
        sections.setdefault(section, []).append(f"{name} = {value}")
    target.write_text("\n\n".join(
        f"<{section}>\n" + "\n".join(values)
        for section, values in sections.items()) + "\n", encoding="utf-8")
    return settings


def profile_from_athdf(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    data = read_athdf(path)
    x = np.asarray(data["x1v"], dtype=float)
    n = x.size
    energy = np.asarray(data["E_rad"], dtype=float).reshape(-1, n).mean(axis=0)/ENERGY_UNIT
    return x, energy, float(data["Time"])


def _log_rows(log: Path, prefix: str) -> list[dict[str, str]]:
    return [dict(re.findall(r"(\w+)=([^\s]+)", line))
            for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.startswith(prefix)]


def analyse_case(output: Path, log: Path, n: int, steps: int,
                 final_time: float, requested_dt: float) -> tuple[dict, dict]:
    text = log.read_text(encoding="utf-8", errors="replace")
    active = [row for row in _log_rows(log, "[NR] status=")
              if row.get("stage") == "0"]
    nr_ok = (len(active) in (steps, steps+1)
             and all(row.get("status") == "converged"
                     and float(row["final_l2"]) <= 1.0e-11
                     and float(row["final_max"]) <= 1.0e-11 for row in active))
    actual_dt = np.array([float(row["dt"]) for row in active])
    dt_ok = (actual_dt.size >= steps
             and np.allclose(actual_dt[:steps], requested_dt, rtol=1.0e-8, atol=1.0e-14)
             and np.isclose(np.sum(actual_dt), final_time, rtol=1.0e-10, atol=1.0e-14)
             and (actual_dt.size == steps or actual_dt[-1] < 1.0e-8*requested_dt))
    krylov = _log_rows(log, "[LINEAR_FGMRES]")
    krylov_ok = bool(krylov) and all(
        np.isfinite(float(row["relative_residual"]))
        and float(row["relative_residual"]) <= float(row["target"])
        for row in krylov)
    end_times = re.findall(r"^time=([^ ]+) cycle=\d+", text, re.M)
    time = float(end_times[-1]) if end_times else float("nan")
    x, value, hdf_time = profile_from_athdf(latest_athdf(output, "uov_x_slice"))
    time_ok = (np.isclose(time, final_time, rtol=1.0e-10, atol=1.0e-14)
               and np.isclose(hdf_time, time, rtol=2.0e-7))
    reference = gaussian(x, time)
    finite = bool(np.all(np.isfinite(value)) and np.all(value > 0))
    absolute_l1 = float(np.mean(np.abs(value-reference)))
    relative = relative_l1(value, reference)
    max_relative = float(np.max(np.abs(value-reference)/reference))
    history = read_history(output)
    history_l1 = float(history["L1norm"][-1])
    history_l1_relative = float(history["L1norm_rel"][-1])
    metrics = {
        "resolution": n, "steps_target": steps, "steps_measured": len(active),
        "requested_dt": requested_dt, "measured_dt_min": float(np.min(actual_dt)),
        "measured_dt_max": float(np.max(actual_dt)),
        "actual_cfl": float(np.max(actual_dt)*CS*n),
        "diffusion_number": float(CHI*requested_dt*n*n),
        "final_time": time, "hdf_final_time": hdf_time,
        "absolute_l1_code_units": absolute_l1,
        "relative_l1": relative, "max_pointwise_relative_error": max_relative,
        "history_l1_code_units": history_l1,
        "history_mean_pointwise_relative_l1": history_l1_relative,
        "newton_converged_steps": len(active), "fgmres_calls": len(krylov),
        "newton_ok": nr_ok, "fgmres_ok": krylov_ok,
        "dt_ok": bool(dt_ok), "time_ok": bool(time_ok), "finite_positive": finite,
        "passed": bool(nr_ok and krylov_ok and dt_ok and time_ok and finite),
    }
    return metrics, {"x": x, "value": value, "reference": reference,
                     "history": history}


def write_profile(path: Path, profile: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("x_code", "energy_code", "analytic_code", "absolute_error_code"))
        for x, value, reference in zip(profile["x"], profile["value"], profile["reference"]):
            writer.writerow((f"{x:.17g}", f"{value:.17g}", f"{reference:.17g}",
                             f"{abs(value-reference):.17g}"))


def report(root: Path, summary: dict) -> None:
    cases = summary.get("cases", {})
    rows = []
    for key, entry in cases.items():
        metrics = entry.get("metrics", {})
        status = entry.get("status", "running")
        badge = ("pass" if status == "passed" else "fail" if status == "failed" else "pending")
        def value(name: str) -> str:
            item = metrics.get(name)
            return f"{item:.4g}" if isinstance(item, (float, int)) else "—"
        prefix = f"cases/{key}/"
        history_files = sorted((root/prefix/"output").glob("*.hst"))
        history_link = (f' · <a href="{quote(prefix + "output/" + history_files[0].name)}">history</a>'
                        if history_files else "")
        rows.append(
            f'<tr><th scope="row">{escape(key)}</th><td><span class="badge {badge}">{escape(status.upper())}</span></td>'
            f'<td>{value("actual_cfl")}</td><td>{value("absolute_l1_code_units")}</td>'
            f'<td><strong>{value("relative_l1")}</strong></td>'
            f'<td>{value("history_mean_pointwise_relative_l1")}</td>'
            f'<td>{value("max_pointwise_relative_error")}</td>'
            f'<td><a href="{quote(prefix)}profile.csv">profile</a> · '
            f'<a href="{quote(prefix)}input.used">input</a>{history_link} · '
            f'<a href="{quote(prefix)}manifest.json">manifest</a> · '
            f'<a href="{quote(prefix)}run.log">log</a></td></tr>')
    metrics = summary.get("metrics", {})
    passed = sum(entry.get("status") == "passed" for entry in cases.values())
    status = summary.get("status", "running")
    status_css = "pass" if status == "passed" else "fail" if status == "failed" else "pending"
    images = {"convergence": "Spatial and temporal convergence",
              "profiles": "Final spatial profiles: numerical versus analytic",
              "profile_errors": "Final spatial error distributions",
              "time_traces": "Radiation energy at edge, middle and center",
              "snapshots": "Representative evolution snapshots",
              "l1_history": "L1 error over time"}
    figures = "".join(
        f'<figure><img src="figures/{name}.png" alt="{escape(caption)}" loading="lazy">'
        f'<figcaption>{escape(caption)} · <a href="figures/{name}.pdf">PDF</a></figcaption></figure>'
        for name, caption in images.items() if (root/"figures"/f"{name}.png").is_file())
    middle_resolution = summary.get("resolutions", DEFAULT_RESOLUTIONS)[1]
    video = (f'''<figure id="video"><video controls preload="metadata" playsinline
        poster="figures/snapshots.png" width="100%"><source src="video/diffusion_evolution.mp4" type="video/mp4">
        <a href="video/diffusion_evolution.mp4">Download MP4</a></video>
        <figcaption>Evolution of the N={middle_resolution}, CFL multiplier 1/2 profile ·
        <a href="video/diffusion_evolution.mp4">MP4</a></figcaption></figure>'''
        if (root/"video/diffusion_evolution.mp4").is_file() else "")
    spatial = metrics.get("spatial_extrapolated_order", [])
    diagonal = metrics.get("coupled_refinement_order", [])
    history_order = metrics.get("history_coupled_refinement_order", [])
    temporal = metrics.get("temporal_difference_order")
    def order_text(value) -> str:
        return ", ".join(f"{x:.3f}" for x in value) if isinstance(value, list) else (
            f"{value:.3f}" if isinstance(value, (int, float)) else "—")
    fixed_rows = []
    for n, item in metrics.get("fixed_n", {}).items():
        values = item["relative_l1"]
        slopes = item["raw_l1_order_in_dt"]
        fixed_rows.append(
            f'<tr><th scope="row">{escape(n)}</th>'
            + "".join(f"<td>{value:.5g}</td>" for value in values)
            + "".join(f"<td>{slope:.3f}</td>" for slope in slopes)
            + "</tr>")
    root.joinpath("index.html").write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>NR-FLD diffusion validation — {escape(summary['run'])}</title>
<style>
:root {{font:18px/1.6 system-ui,sans-serif;color:#17212b;background:#f7f9fb}}
body {{max-width:1450px;margin:2rem auto;padding:0 1.2rem}} h1,h2 {{line-height:1.25}}
h1 {{font-size:2rem}} h2 {{font-size:1.5rem}} section {{background:#fff;border:1px solid #dce4eb;border-radius:10px;padding:1.3rem;margin:1.4rem 0}}
.overview {{background:#eaf3fb;border-left:5px solid #1769aa;padding:.8rem 1rem;font-size:1.1rem}}
.badge {{display:inline-block;border-radius:5px;padding:.1rem .55rem;font-weight:750}}
.pass {{color:#0b683e;background:#e4f5eb}} .fail {{color:#ad261f;background:#fdebea}}
.pending {{color:#41556b;background:#e8edf2}} .scroll {{overflow-x:auto}}
table {{border-collapse:collapse;width:100%;font-size:.96rem}} th,td {{padding:.55rem .65rem;border-bottom:1px solid #dce4eb;text-align:left;white-space:nowrap}}
thead th {{background:#eef3f7}} tr:nth-child(even) {{background:#f8fafc}}
figure {{margin:1.3rem 0}} img,video {{max-width:100%;height:auto;display:block}} figcaption {{font-weight:600;margin:.4rem 0}}
a {{color:#075ca8}} @media(max-width:700px) {{:root {{font-size:17px}} body {{padding:0 .75rem}} section {{padding:.8rem}}}}
{REPORT_TYPOGRAPHY}</style></head><body>
<h1>NR-FLD diffusion validation <span class="badge {status_css}">{escape(status.upper())}</span></h1>
<p class="overview"><strong>{passed}/9 cases passed</strong> · N={', '.join(map(str, summary.get('resolutions', DEFAULT_RESOLUTIONS)))} × CFL multiplier 1, 1/2, 1/4 · FGMRES</p>
<p><a href="#convergence">Convergence and L1 table</a> · <a href="#plots">Profiles and time traces</a> · <a href="#video">Video</a></p>
<p>Gaussian diffusion with a fixed flux limiter (lambda=1/3), no gas-radiation coupling, and a common final time of {summary.get('final_time', 0.02):.7g} code units. Base CFL is approximately {summary.get('base_cfl', 0):.4g}; CFL is also halved as N doubles along the diagonal so dt scales as dx².</p>
<section id="convergence"><h2>Convergence</h2><p><strong>Coupled dx/dt refinement order:</strong> {order_text(diagonal)} · <strong>Original HST relative-L1 order:</strong> {order_text(history_order)} · <strong>Time-extrapolated spatial order:</strong> {order_text(spatial)} · <strong>Fixed-grid temporal difference order:</strong> {order_text(temporal)}.</p>
<p>Acceptance: all nine calculations converge at the requested dt and final time; both spatial-order estimates lie between 1.7 and 2.3, and the independent temporal-order estimate lies between 0.8 and 1.2.</p>
<p>Fixed-N relative L1 values and their measured rates as dt is halved:</p>
<div class="scroll"><table><thead><tr><th>N</th><th>CFL ×1</th><th>CFL ×1/2</th><th>CFL ×1/4</th><th>Rate 1→1/2</th><th>Rate 1/2→1/4</th></tr></thead><tbody>{''.join(fixed_rows)}</tbody></table></div>
<p>These raw rates include a fixed spatial error, so they need not equal one. The independent temporal rate above compares successive solution differences at N={middle_resolution}. Absolute L1 is mean |E_num−E_analytic| in code units; relative L1 is sum |E_num−E_analytic| / sum |E_analytic|. The HST relative L1 is the original code's mean pointwise relative error. Spatial order uses 2E(dt/2)−E(dt).</p>
<div class="scroll"><table><thead><tr><th>Case</th><th>Status</th><th>Actual CFL</th><th>Absolute L1</th><th>Relative L1</th><th>HST relative L1</th><th>Max pointwise relative error</th><th>Files</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<section id="plots"><h2>Spatial distribution and time evolution</h2>{figures}{video}</section>
<p><a href="summary.json">Summary JSON</a> · <a href="manifest.json">Run manifest</a> · <a href="build/build.log">Build log</a> · <a href="build/config.log">Binary configuration</a></p>
</body></html>\n""", encoding="utf-8")


def _save_figure(fig, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination.with_suffix(".png"), dpi=180)
    fig.savefig(destination.with_suffix(".pdf"))
    plt.close(fig)


def plot_results(root: Path, profiles: dict, summary: dict) -> None:
    final_time = summary["final_time"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    metrics = summary["metrics"]
    resolutions = summary["resolutions"]
    middle_resolution = resolutions[1]
    n_values = np.array(resolutions)
    diagonal = np.array(metrics["coupled_refinement_l1"])
    extrapolated = np.array(metrics["spatial_extrapolated_l1"])
    axes[0].loglog(n_values, diagonal, "o-", label="CFL halved with N doubled")
    axes[0].loglog(n_values, extrapolated, "s-", label="time-extrapolated")
    axes[0].loglog(n_values, extrapolated[0]*(n_values/n_values[0])**-2,
                   "k--", label="order 2")
    axes[0].set(xlabel="N", ylabel="relative L1", title="Space-time refinement")
    for n in resolutions:
        x = [summary["cases"][f"n{n}/{tag(f)}"]["metrics"]["actual_cfl"]
             for f in CFL_FACTORS]
        y = [summary["cases"][f"n{n}/{tag(f)}"]["metrics"]["relative_l1"]
             for f in CFL_FACTORS]
        axes[1].loglog(x, y, "o-", label=f"N={n}")
    axes[1].invert_xaxis()
    axes[1].set(xlabel="actual CFL", ylabel="relative L1", title="Fixed-grid L1")
    differences = metrics["temporal_profile_differences"]
    dts = np.array([final_time/steps_for(middle_resolution, f, summary["base_steps"],
                                            resolutions[0])
                    for f in CFL_FACTORS[:2]])
    axes[2].loglog(dts, differences, "o-", label=f"N={middle_resolution} differences")
    axes[2].loglog(dts, differences[-1]*dts/dts[-1], "k--", label="order 1")
    axes[2].set(xlabel="dt [code units]", ylabel="relative L1 difference",
                title="Temporal order")
    axes[2].set_xticks(dts, labels=[f"{value:.2e}" for value in dts])
    axes[2].xaxis.set_minor_formatter(NullFormatter())
    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=9)
    _save_figure(fig, root/"figures/convergence")

    selected = list(zip(resolutions, CFL_FACTORS))
    smooth_x = np.linspace(0, 1, 600)
    smooth_reference = gaussian(smooth_x, final_time)*ENERGY_UNIT
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5), sharey=True)
    for axis, (n, factor) in zip(axes, selected):
        profile = profiles[(n, factor)]
        axis.plot(smooth_x, smooth_reference, "k-", label="analytic")
        axis.scatter(profile["x"], profile["value"]*ENERGY_UNIT,
                     s=17, label="FGMRES")
        axis.set(title=f"N={n}, CFL ×{factor:g}", xlabel="x / L")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel(r"$E_{\rm rad}$ [erg cm$^{-3}$]")
    axes[0].legend()
    _save_figure(fig, root/"figures/profiles")

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5), sharey=True)
    for axis, (n, factor) in zip(axes, selected):
        profile = profiles[(n, factor)]
        normalized = np.abs(profile["value"]-profile["reference"])/max(profile["reference"])
        axis.semilogy(profile["x"], np.maximum(normalized, 1.0e-16), "o-", ms=3)
        axis.set(title=f"N={n}, CFL ×{factor:g}", xlabel="x / L")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel(r"$|E_{\rm num}-E_{\rm an}|/\max E_{\rm an}$")
    _save_figure(fig, root/"figures/profile_errors")

    representative_output = root/f"cases/n{middle_resolution}/cfl_1_2/output"
    movie_files = sorted(representative_output.glob("*.uov_x_slice.*.athdf"))
    if len(movie_files) < 5:
        raise RuntimeError(f"Expected at least five evolution frames, got {len(movie_files)}")
    frames = [profile_from_athdf(path) for path in movie_files]
    trace_indices = tuple(middle_resolution//denom for denom in (8, 4, 2))
    traces_x = [frames[0][0][index] for index in trace_indices]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()
    smooth_t = np.linspace(0, final_time, 300)
    for axis, index, label, x in zip(axes[:3], trace_indices,
                                     ("edge", "middle", "center"), traces_x):
        times = [frame[2] for frame in frames]
        values = [frame[1][index]*ENERGY_UNIT for frame in frames]
        axis.plot(smooth_t*4000, gaussian(x, smooth_t)*ENERGY_UNIT,
                  "k-", label="analytic")
        axis.scatter(np.array(times)*4000, values, s=18, label="FGMRES")
        axis.set(title=f"{label}: x/L={x:.3f}", xlabel="time since start [s]",
                 ylabel=r"$E_{\rm rad}$ [erg cm$^{-3}$]")
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].legend()
    final_x, final_value, _ = frames[-1]
    axes[3].plot(smooth_x, smooth_reference, "k-", label="analytic")
    axes[3].scatter(final_x, final_value*ENERGY_UNIT, s=18, label="FGMRES")
    axes[3].set(title="final spatial profile", xlabel="x / L",
                ylabel=r"$E_{\rm rad}$ [erg cm$^{-3}$]")
    axes[3].grid(True, alpha=0.25)
    _save_figure(fig, root/"figures/time_traces")

    sample_indices = np.linspace(0, len(frames)-1, 4, dtype=int)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for axis, index in zip(axes.ravel(), sample_indices):
        x, value, time = frames[index]
        axis.plot(smooth_x, gaussian(smooth_x, time)*ENERGY_UNIT,
                  "k-", label="analytic")
        axis.scatter(x, value*ENERGY_UNIT, s=15, label="FGMRES")
        axis.set(title=f"t={time*4000:.1f} s", xlabel="x / L",
                 ylabel=r"$E_{\rm rad}$ [erg cm$^{-3}$]")
        axis.grid(True, alpha=0.25)
    axes[0, 0].legend()
    _save_figure(fig, root/"figures/snapshots")

    fig, axis = plt.subplots(figsize=(9, 5.5))
    for n, factor in selected:
        history = profiles[(n, factor)]["history"]
        axis.semilogy(history["time"], np.maximum(history["L1norm"], 1.0e-16),
                      label=f"N={n}, CFL ×{factor:g}")
    axis.set(xlabel="time [code units]", ylabel="absolute L1 [code units]",
             title="Error evolution")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    _save_figure(fig, root/"figures/l1_history")
    return frames


def make_video(root: Path, frames: list, records: list) -> dict:
    video_dir = root/"video"
    video_dir.mkdir(exist_ok=True)
    all_values = np.concatenate([frame[1] for frame in frames])*ENERGY_UNIT
    ymax = float(np.max(all_values)*1.12)
    smooth_x = np.linspace(0, 1, 600)
    with tempfile.TemporaryDirectory(prefix="diff_frames_") as temp_dir:
        directory = Path(temp_dir)
        for index, (x, value, time) in enumerate(frames):
            fig, axis = plt.subplots(figsize=(9, 5.3))
            axis.plot(smooth_x, gaussian(smooth_x, time)*ENERGY_UNIT,
                      "k-", lw=1.8, label="analytic")
            axis.scatter(x, value*ENERGY_UNIT, s=20, color="#2878b5", label="FGMRES")
            axis.set(xlim=(0, 1), ylim=(0, ymax), xlabel="x / L",
                     ylabel=r"$E_{\rm rad}$ [erg cm$^{-3}$]",
                     title=f"Gaussian diffusion: t={time*4000:.1f} s")
            axis.grid(True, alpha=0.25)
            axis.legend()
            fig.tight_layout()
            fig.savefig(directory/f"frame_{index:03d}.png", dpi=160)
            plt.close(fig)
        target = video_dir/"diffusion_evolution.mp4"
        command = ["ffmpeg", "-y", "-loglevel", "warning", "-framerate", "5",
                   "-i", str(directory/"frame_%03d.png"), "-vf",
                   "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
                   "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                   str(target)]
        records.append(checked(command, root, video_dir/"encode.log", 300, "video_encode"))
    if not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError("MP4 encoding produced no video")
    return {"path": str(target), "sha256": sha256_file(target),
            "bytes": target.stat().st_size, "frames": len(frames), "codec": "H.264"}


def run_case(root: Path, repo: Path, source_input: Path, exe: Path,
             build: dict, config: dict, args: argparse.Namespace,
             n: int, factor: float) -> tuple[dict, dict]:
    steps = steps_for(n, factor, args.base_steps, args.resolutions[0])
    dt = args.final_time/steps
    cfl = dt*CS*n
    key = f"n{n}/{tag(factor)}"
    case = root/"cases"/key
    case.mkdir(parents=True, exist_ok=False)
    output = case/"output"
    output.mkdir()
    manifest = base_manifest(repo, key, args.max_cores)
    manifest["provenance"]["analysis"]["files"] = {
        path.relative_to(repo).as_posix(): sha256_file(path)
        for path in (Path(__file__), SCRIPT_DIR/"diffusion_convergence.py")}
    manifest["provenance"]["calculation"] = {
        "source": "shared_isolated_build", "source_sha256": build["build_source_sha256"],
        "binary_sha256": sha256_file(exe), "binary_configuration": config}
    snapshot_dt = (args.final_time/20 if (n, factor) == (args.resolutions[1], 0.5)
                   else args.final_time)
    overrides = [
        f"mesh/nx1={n}", "mesh/nx2=8", "mesh/nx3=8",
        "meshblock/nx1=8", "meshblock/nx2=8", "meshblock/nx3=8",
        f"mesh/x2max={8/n:.17g}", f"mesh/x3max={8/n:.17g}",
        f"time/cfl_number={cfl:.17g}", f"time/tlim={args.final_time:.17g}",
        "time/nlim=-1", "time/ncycle_out=1", "fld/fixed_flux_limiter=true",
        "nrfld/linear_solver=fgmres", "nrfld/threshold=1e-12",
        "nrfld/nr_threshold=1e-11", "nrfld/diagnostic_verbosity=0",
        "nrfld/failure_policy=abort", f"job/problem_id=diff_n{n}_{tag(factor)}",
        "output1/data_format=%.16e", "output1/dcycle=1",
        "output3/dcycle=0", f"output3/dt={snapshot_dt:.17g}",
        "output3/ghost_zones=false", f"output3/x2_slice={4/n:.17g}",
        f"output3/x3_slice={4/n:.17g}",
        "output4/dt=-1", "output4/dcycle=0",
        "output5/dt=-1", "output5/dcycle=0",
    ]
    input_path = case/"input.used"
    settings = materialize_input(source_input, input_path, overrides)
    manifest["input"] = {"source_path": str(source_input),
                         "source_sha256": sha256_file(source_input),
                         "used_path": str(input_path),
                         "used_sha256": sha256_file(input_path),
                         "effective_settings": settings}
    ranks = min(args.max_cores, n//8)
    resources = case/"resources.txt"
    manifest["resources"] = {"mpi_ranks": ranks, "omp_threads": 1,
                             "max_cores": args.max_cores}
    command = ["/usr/bin/time", "-v", "-o", str(resources),
               "mpirun", "-np", str(ranks), str(exe), "-i", str(input_path),
               "-d", str(output)]
    manifest["commands"] = {"run": command, "records": []}
    write_manifest(case/"manifest.json", manifest)
    print(f"Running {key}: N={n}, CFL={cfl:.6g}, steps={steps}, MPI={ranks}", flush=True)
    try:
        record = checked(command, case, case/"run.log", args.run_timeout, "run")
        manifest["commands"]["records"].append(record)
        metrics, profile = analyse_case(output, case/"run.log", n, steps,
                                        args.final_time, dt)
        metrics["cfl_multiplier"] = factor
        if not np.isclose(metrics["actual_cfl"], cfl, rtol=1.0e-8):
            metrics["passed"] = False
        match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)",
                          resources.read_text(encoding="utf-8"))
        manifest["resources"]["max_rss_kib"] = int(match.group(1)) if match else None
        write_profile(case/"profile.csv", profile)
        manifest["artifacts"] = {"profile": str(case/"profile.csv"),
                                 "output": str(output)}
        manifest["metrics"] = metrics
        manifest["status"] = "passed" if metrics["passed"] else "failed"
        print(f"  relative L1={metrics['relative_l1']:.5g}, "
              f"status={manifest['status']}", flush=True)
        return {"status": manifest["status"], "metrics": metrics}, profile
    except Exception as error:
        if isinstance(error, CommandExecutionError):
            manifest["commands"]["records"].append(error.record)
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        print(f"  failed: {error}", flush=True)
        return {"status": "failed", "error": str(error), "metrics": {}}, {}
    finally:
        manifest["finished_at"] = utc_now()
        write_manifest(case/"manifest.json", manifest)


def compute_orders(summary: dict, profiles: dict) -> dict:
    cases = summary["cases"]
    def error(n: int, factor: float) -> float:
        return cases[f"n{n}/{tag(factor)}"]["metrics"]["relative_l1"]
    def history_error(n: int, factor: float) -> float:
        return cases[f"n{n}/{tag(factor)}"]["metrics"]["history_mean_pointwise_relative_l1"]
    resolutions = summary["resolutions"]
    middle_resolution = resolutions[1]
    diagonal_keys = list(zip(resolutions, CFL_FACTORS))
    diagonal = [error(n, factor) for n, factor in diagonal_keys]
    history_diagonal = [history_error(n, factor) for n, factor in diagonal_keys]
    spatial_extrapolated = [relative_l1(
        2*profiles[(n, 0.5)]["value"]-profiles[(n, 1.0)]["value"],
        profiles[(n, 1.0)]["reference"]) for n in resolutions]
    temporal_differences = [relative_l1(profiles[(middle_resolution, a)]["value"],
                                        profiles[(middle_resolution, b)]["value"])
                            for a, b in ((1.0, 0.5), (0.5, 0.25))]
    fixed_n = {}
    for n in resolutions:
        errors = [error(n, factor) for factor in CFL_FACTORS]
        fixed_n[str(n)] = {"relative_l1": errors,
                           "raw_l1_order_in_dt": orders(errors, 2)}
    spatial_orders = orders(spatial_extrapolated, 2)
    diagonal_orders = orders(diagonal, 2)
    temporal_order = math.log(temporal_differences[0]/temporal_differences[1], 2)
    metrics = {
        "fixed_n": fixed_n,
        "coupled_refinement_cases": [f"n{n}/{tag(f)}" for n, f in diagonal_keys],
        "coupled_refinement_l1": diagonal,
        "coupled_refinement_order": diagonal_orders,
        "history_coupled_refinement_l1": history_diagonal,
        "history_coupled_refinement_order": orders(history_diagonal, 2),
        "spatial_extrapolated_l1": spatial_extrapolated,
        "spatial_extrapolated_order": spatial_orders,
        "temporal_profile_differences": temporal_differences,
        "temporal_difference_order": temporal_order,
    }
    finite = all(np.isfinite(x) for sequence in
                 (diagonal, spatial_extrapolated, temporal_differences,
                  spatial_orders, diagonal_orders, [temporal_order]) for x in sequence)
    metrics["passed"] = bool(finite
                             and all(1.7 < p < 2.3 for p in diagonal_orders)
                             and all(1.7 < p < 2.3 for p in spatial_orders)
                             and 0.8 < temporal_order < 1.2)
    return metrics


def main() -> int:
    args = parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    repo = find_repo_root(SCRIPT_DIR)
    now_jst = datetime.now(timezone(timedelta(hours=9)))
    run_name = args.run_name or f"validation_diffusion_{now_jst:%Y%m%d_%H%M%S}_JST"
    root = args.output_root.expanduser().resolve()/run_name
    root.mkdir(parents=True, exist_ok=False)
    summary = {"run": run_name, "status": "running", "started_at": utc_now(),
               "final_time": args.final_time, "base_steps": args.base_steps,
               "base_cfl": args.final_time/args.base_steps*CS*args.resolutions[0],
               "resolutions": args.resolutions,
               "cases": {}, "metrics": {}}
    report(root, summary)
    manifest = base_manifest(repo, "diffusion_validation", args.max_cores)
    records = []
    manifest["commands"]["records"] = records
    manifest["analysis_script_sha256"] = sha256_file(Path(__file__))
    build_dir = root/"build"
    build_dir.mkdir()
    try:
        build = build_case(repo, build_dir, "nrfld_diff", args.max_cores,
                           args.build_timeout, records)
        exe = build["build_source"]/"bin/athena"
        config = binary_configuration(exe, build_dir, 30, records)
        validate_binary_configuration(config, "nrfld_diff", exe)
        manifest["provenance"]["calculation"] = {
            "source": "shared_isolated_build", "source_sha256": build["build_source_sha256"],
            "binary_sha256": sha256_file(exe), "binary_configuration": config}
    except Exception as error:
        if isinstance(error, CommandExecutionError):
            records.append(error.record)
        summary.update(status="build_failed", error=str(error), finished_at=utc_now())
        (root/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        report(root, summary)
        manifest.update(status="failed", error=str(error), finished_at=utc_now())
        write_manifest(root/"manifest.json", manifest)
        raise
    source_input = repo/"inputs/radiation/athinput.nrfld_diff"
    profiles = {}
    for n in args.resolutions:
        for factor in CFL_FACTORS:
            key = f"n{n}/{tag(factor)}"
            result, profile = run_case(root, repo, source_input, exe, build,
                                       config, args, n, factor)
            summary["cases"][key] = result
            if profile:
                profiles[(n, factor)] = profile
            (root/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
            report(root, summary)
    all_calculations_passed = (len(profiles) == len(args.resolutions)*len(CFL_FACTORS)
                               and all(item["status"] == "passed"
                                       for item in summary["cases"].values()))
    if all_calculations_passed:
        try:
            summary["metrics"] = compute_orders(summary, profiles)
            frames = plot_results(root, profiles, summary)
            summary["video"] = make_video(root, frames, records)
        except Exception as error:
            summary["error"] = str(error)
            print(f"Analysis/plot/video failed: {error}", flush=True)
    summary["status"] = ("passed" if all_calculations_passed
                         and summary.get("metrics", {}).get("passed")
                         and "video" in summary else "failed")
    summary["finished_at"] = utc_now()
    manifest["metrics"] = summary["metrics"]
    manifest["status"] = summary["status"]
    manifest["finished_at"] = summary["finished_at"]
    write_manifest(root/"manifest.json", manifest)
    (root/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    report(root, summary)
    print(f"HTML report: {root/'index.html'}", flush=True)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
