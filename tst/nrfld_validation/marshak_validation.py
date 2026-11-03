#!/usr/bin/env python3
"""Run the Marshak wave test and compare u,v with the Robin-boundary solution."""

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
import tempfile

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nrfld-validation")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import ticker  # noqa: E402
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from diffusion_validation import materialize_input  # noqa: E402
from nrfld_validation.io import find_repo_root, read_athdf, read_history  # noqa: E402
from nrfld_validation.report_style import REPORT_TYPOGRAPHY  # noqa: E402
from nrfld_validation.manifest import (base_manifest, sha256_file, tree_sha256,
                                       utc_now, write_manifest)  # noqa: E402
from run_suite import (CommandExecutionError, binary_configuration, checked,
                       prepare_build_source)  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/d/output/rad"))
    parser.add_argument("--run-name")
    parser.add_argument("--resolutions", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--max-cores", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--run-timeout", type=float, default=1800)
    parser.add_argument("--build-timeout", type=float, default=900)
    parser.add_argument("--reuse-build", type=Path,
                        help="Copy a previously validated Marshak build directory into this run.")
    parser.add_argument("--resume-from", type=Path,
                        help="Copy a previous run and calculate only missing resolutions.")
    args = parser.parse_args()
    if (not args.resolutions or any(n < 32 or n % 32 for n in args.resolutions)
            or args.max_cores < 1 or args.steps < 34):
        parser.error("resolutions must be multiples of 32; cores >=1 and steps >=34")
    if len(set(args.resolutions)) != len(args.resolutions):
        parser.error("resolutions must be distinct")
    if args.resume_from and args.reuse_build:
        parser.error("--resume-from already includes a build; omit --reuse-build")
    args.resolutions.sort()
    return args


def analytic(x: np.ndarray, tau: float, beta: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Inverse Laplace solution of the semi-infinite, fixed-limiter Marshak system.

    x=sqrt(3)*kappa*z, tau=beta*c*kappa*t,
    U=exp(-k*x)/[s*(1+2*k/sqrt(3))], V=U/(s+1),
    k=sqrt(beta*s+1-1/(s+1)).
    """
    mp.mp.dps = 32
    u = np.empty(len(x))
    v = np.empty(len(x))
    for i, position in enumerate(x):
        def uhat(s):
            k = mp.sqrt(beta*s + 1 - 1/(s+1))
            return mp.exp(-k*float(position))/(s*(1+2*k/mp.sqrt(3)))
        u[i] = float(mp.re(mp.invertlaplace(uhat, tau, method="dehoog")))
        v[i] = float(mp.re(mp.invertlaplace(lambda s: uhat(s)/(s+1), tau,
                                           method="dehoog")))
    return u, v


def profile(path: Path, tau_factor: float) -> dict:
    data = read_athdf(path)
    x = np.asarray(data["x_dimless"], dtype=float).reshape(-1, len(data["x1v"])).mean(axis=0)
    u = np.asarray(data["u"], dtype=float).reshape(-1, len(x)).mean(axis=0)
    v = np.asarray(data["v"], dtype=float).reshape(-1, len(x)).mean(axis=0)
    return {"x": x, "u": u, "v": v, "time": float(data["Time"]),
            "tau": float(data["Time"])*tau_factor, "cycles": int(data["NumCycles"])}


def error_metrics(numerical: np.ndarray, reference: np.ndarray) -> dict:
    delta = np.abs(numerical-reference)
    return {"l1_mean": float(np.mean(delta)),
            "l1_relative": float(np.sum(delta)/np.sum(np.abs(reference))),
            "max_absolute": float(np.max(delta)),
            "max_over_peak": float(np.max(delta)/np.max(np.abs(reference)))}


def save_figure(fig, path: Path):
    path.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def write_report(root: Path, summary: dict):
    cases = summary["cases"]
    rows = []
    for key, case in sorted(cases.items(), key=lambda pair: int(pair[0][1:])):
        case_status = case.get("status", "running")
        case_css = "pass" if case_status == "passed" else "fail" if case_status == "failed" else "pending"
        for stage, item in case.get("stages", {}).items():
            for field in ("u", "v"):
                metric = item[field]
                rows.append(f'<tr><th>{escape(key)}</th><td><span class="badge {case_css}">{escape(case_status.upper())}</span></td>'
                            f'<td>{escape(stage)}</td><td>{field}</td>'
                            f'<td><strong>{metric["l1_relative"]:.4g}</strong></td>'
                            f'<td>{metric["l1_mean"]:.4g}</td>'
                            f'<td>{metric["max_absolute"]:.4g}</td>'
                            f'<td>{metric["max_over_peak"]:.4g}</td></tr>')
    links = ""
    for key, case in sorted(cases.items(), key=lambda pair: int(pair[0][1:])):
        files = [(f"profile_{stage}.csv", f"{stage} profile CSV")
                 for stage in case.get("stages", {})]
        files += [("input.used", "input"), ("run.log", "log"),
                  ("resources.txt", "resources"), ("problem_parameters.txt", "parameters"),
                  ("manifest.json", "manifest")]
        files += [(f"output/{path.name}", "history") for path in
                  sorted((root/"cases"/key/"output").glob("*.hst"))]
        files += [(f"output/{path.name}", f"{stage} HDF5")
                  for stage, pattern in (("early", "*.uov_x_early.*.athdf"),
                                         ("final", "*.uov_x_final.*.athdf"))
                  for path in sorted((root/"cases"/key/"output").glob(pattern))[-1:]]
        available = [(path, label) for path, label in files if (root/"cases"/key/path).is_file()]
        links += (f'<li><strong>{escape(key)}</strong>: '
                  + " · ".join(f'<a href="cases/{key}/{escape(path)}">{escape(label)}</a>'
                             for path, label in available) + '</li>')
    plots = "".join(f'<figure><img src="figures/{name}.png" alt="{escape(label)}" loading="lazy">'
                    f'<figcaption>{escape(label)} · <a href="figures/{name}.pdf">PDF</a></figcaption></figure>'
                    for name, label in (("comparison", "u and v against the analytic solution on shared log axes"),
                                        ("convergence", "Relative L1 versus resolution and measured slopes"),
                                        ("profiles", "Numerical and analytic spatial profiles"),
                                        ("profiles_log", "Log-scale wave profiles and penetration tails"),
                                        ("errors", "Spatial error distributions"),
                                        ("history", "Evolution of wave maxima"))
                    if (root/"figures"/f"{name}.png").is_file())
    video = ('<figure><video controls preload="metadata" playsinline width="100%">'
             '<source src="video/marshak_wave.mp4" type="video/mp4"></video>'
             f'<figcaption>N={max(summary["resolutions"])} wave evolution · '
             '<a href="video/marshak_wave.mp4">MP4</a></figcaption></figure>'
             if (root/"video/marshak_wave.mp4").is_file() else "")
    status = summary["status"]
    css = "pass" if status == "passed" else "pending" if status == "running" else "fail"
    runtime_rows = "".join(
        f'<tr><th>{escape(key)}</th><td>{case.get("wall_seconds", float("nan")):.1f}</td>'
        f'<td>{float(case.get("max_rss_kib") or float("nan"))/1024:.1f}</td>'
        f'<td>{case.get("nr_status_count", "—")}</td><td>{case.get("fgmres_calls", "—")}</td></tr>'
        for key, case in sorted(cases.items(), key=lambda pair: int(pair[0][1:]))
        if "wall_seconds" in case)
    convergence_rows = "".join(
        f'<tr><th>{row["resolution"]}</th><td>{row["stage"]}</td><td>{row["field"]}</td>'
        f'<td>{row["relative_l1"]:.5g}</td>'
        f'<td>{row["order_from_previous"]:.3f}</td></tr>'
        if row["order_from_previous"] is not None else
        f'<tr><th>{row["resolution"]}</th><td>{row["stage"]}</td><td>{row["field"]}</td>'
        f'<td>{row["relative_l1"]:.5g}</td><td>—</td></tr>'
        for row in summary.get("convergence", {}).get("rows", []))
    spatial = summary.get("spatial_study")
    spatial_section = ""
    if spatial:
        spatial_status = spatial.get("status", "running")
        spatial_css = ("pass" if spatial_status == "passed" else
                       "pending" if spatial_status == "running" else "fail")
        spatial_rows = "".join(
            f'<tr><th>{row["resolution"]}</th><td>{row["field"]}</td>'
            f'<td>{row["baseline_relative_l1"]:.5g}</td>'
            f'<td>{row["half_relative_l1"]:.5g}</td>'
            f'<td><strong>{row["extrapolated_relative_l1"]:.5g}</strong></td>'
            f'<td>{row["order_from_previous"]:.3f}</td></tr>'
            if row["order_from_previous"] is not None else
            f'<tr><th>{row["resolution"]}</th><td>{row["field"]}</td>'
            f'<td>{row["baseline_relative_l1"]:.5g}</td>'
            f'<td>{row["half_relative_l1"]:.5g}</td>'
            f'<td><strong>{row["extrapolated_relative_l1"]:.5g}</strong></td>'
            f'<td>—</td></tr>'
            for row in spatial.get("rows", []))
        spatial_figure = ('''<figure><img src="figures/spatial_convergence.png"
            alt="Time-extrapolated spatial L1 convergence" loading="lazy">
            <figcaption>Spatial convergence after first-order time-error cancellation ·
            <a href="figures/spatial_convergence.pdf">PDF</a></figcaption></figure>'''
            if (root/"figures/spatial_convergence.png").is_file() else "")
        spatial_links_list = []
        for n in spatial.get("resolutions", []):
            case = root/f"spatial/cases/n{n}"
            if not (case/"manifest.json").is_file():
                continue
            items = [("input.used", "input"), ("run.log", "log"),
                     ("profile_final.csv", "profile CSV"),
                     ("manifest.json", "manifest")]
            items += [(f"output/{path.name}", "final HDF5") for path in
                      sorted((case/"output").glob("*.uov_x_final.*.athdf"))[-1:]]
            spatial_links_list.append(f'<strong>N={n}</strong>: ' + " · ".join(
                f'<a href="spatial/cases/n{n}/{path}">{label}</a>' for path, label in items
                if (case/path).is_file()))
        spatial_links = "<br>".join(spatial_links_list)
        spatial_section = f'''<section id="spatial"><h2>Spatial order with time error removed
            <span class="badge {spatial_css}">{escape(spatial_status.upper())}</span></h2>
            <p>At τ=0.3, each N is run with Δτ=3×10<sup>−4</sup> and 1.5×10<sup>−4</sup>.
            The analysis field <strong>2 U(Δt/2) − U(Δt)</strong> cancels the leading
            first-order time error. The evolved fields are unchanged; only the comparison
            field is extrapolated. Relative L1 is measured at native cell centers with no shift.
            For second-order spatial discretization, the remaining leading terms are
            O(Δx²) and O(Δt²).</p>
            {spatial_figure}<div class="scroll"><table><thead><tr><th>N</th><th>Field</th>
            <th>Base L1</th><th>Half-Δt L1</th><th>Extrapolated L1</th>
            <th>Order from previous N</th></tr></thead><tbody>{spatial_rows}</tbody></table></div>
            <p>{'<a href="spatial/convergence.csv">Convergence CSV</a> · ' if (root/'spatial/convergence.csv').is_file() else ''}
            {'<a href="spatial/manifest.json">Study manifest</a> · ' if (root/'spatial/manifest.json').is_file() else ''}{spatial_links}</p></section>'''
    root.joinpath("index.html").write_text(f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Marshak wave validation</title><style>
:root{{font:18px/1.6 system-ui,sans-serif;color:#17212b;background:#f7f9fb}}
body{{max-width:1300px;margin:2rem auto;padding:0 1.2rem}}h1,h2{{line-height:1.25}}
section{{background:white;border:1px solid #dce4eb;border-radius:10px;padding:1.3rem;margin:1.4rem 0}}
.overview{{background:#eaf3fb;border-left:5px solid #1769aa;padding:.8rem 1rem;font-size:1.1rem}}
.badge{{display:inline-block;border-radius:5px;padding:.1rem .55rem;font-weight:750}}
.pass{{color:#0b683e;background:#e4f5eb}}.fail{{color:#ad261f;background:#fdebea}}
.pending{{color:#41556b;background:#e8edf2}}.scroll{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-size:.96rem}}th,td{{padding:.55rem .65rem;border-bottom:1px solid #dce4eb;text-align:left;white-space:nowrap}}
thead th{{background:#eef3f7}}figure{{margin:1.3rem 0}}img,video{{max-width:100%;height:auto;display:block}}
figcaption{{font-weight:600;margin:.4rem 0}}a{{color:#075ca8}}
{REPORT_TYPOGRAPHY}</style></head><body><h1>Marshak wave validation <span class="badge {css}">{escape(status.upper())}</span></h1>
<p class="overview"><strong>{sum(c.get('passed',False) for c in cases.values())}/{len(summary['resolutions'])} cases passed</strong> · N={', '.join(map(str,summary['resolutions']))} · FGMRES · Robin boundary</p>
{f'<p>Earlier passing cases were copied from <code>{escape(summary["resumed_from"])}</code> and their HDF5 profiles, CSV values and saved L1 metrics were rechecked before reuse.</p>' if summary.get('resumed_from') else ''}
<p>The fixed gas problem uses the constant flux limiter λ=1/3. The analytic solution is the inverse Laplace transform of U=exp(−kx)/[s(1+2k/√3)], V=U/(s+1), k=√(βs+1−1/(s+1)); β=0.1. Coordinates are x=√3κz, τ=βcκt; u=cE<sub>rad</sub>/(4F<sub>inc</sub>) and v=cβe<sub>gas</sub>/(4F<sub>inc</sub>). The numerical gas starts at 1 K, corresponding to v≈5.67×10<sup>−12</sup>, to avoid an extremely cold initial state where the current Newton iteration fails; the analytic initial energy is zero.</p>
<p><strong>No spatial alignment is applied.</strong> Numerical and analytic values are compared at the native <code>x_dimless</code> cell centers. {('A diagnostic linearized least-squares fit at final N=' + str(max(summary['resolutions'])) + ' estimates shifts of ' + ', '.join(f'{field}: {value:.4g} cell widths' for field, value in summary['alignment']['diagnostic_shift_cells'].items()) + '; these estimates are not used in any error metric.') if summary.get('alignment') else ''}</p>
<section><h2>Accuracy against analytic solution</h2><p>Early output at cycle 34 (τ≈0.0102); final output at cycle {summary['steps']} (τ≈{summary['steps']*0.0003:.4g}). <strong>L1 mean</strong> is mean absolute error, <strong>relative L1</strong> is sum absolute error divided by sum absolute analytic values, and <strong>max/peak</strong> divides the largest absolute error by the analytic peak. All statistics include the full x domain.</p>
<div class="scroll"><table><thead><tr><th>N</th><th>Status</th><th>Output</th><th>Field</th><th>Relative L1</th><th>L1 mean</th><th>Max absolute</th><th>Max/peak</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>Pass requires the requested cycle, finite nonnegative fields, converged Newton/FGMRES solves, and a relative L1 below 0.2 for both fields at both times. The analytic solution assumes a semi-infinite domain; the simulation has a cold boundary at z=5 cm.</p></section>
<section><h2>L1 convergence with resolution</h2><p>The plot and table compare the same physical times across all resolutions. The adjacent-point rate is log(error at coarse N / error at fine N) / log(N<sub>fine</sub> / N<sub>coarse</sub>). <strong>The time step is fixed at Δt=0.003</strong>, so these rates include time-discretization error and are not by themselves a pure spatial-order test.</p>{'<p><strong>All four L1 series decrease as N increases.</strong></p>' if summary.get('convergence', {}).get('monotonic') else ''}<div class="scroll"><table><thead><tr><th>N</th><th>Output</th><th>Field</th><th>Relative L1</th><th>Rate from previous N</th></tr></thead><tbody>{convergence_rows}</tbody></table></div>{'<p><a href="convergence.csv">Convergence CSV</a></p>' if (root/'convergence.csv').is_file() else ''}</section>
<section><h2>Compute resources</h2><div class="scroll"><table><thead><tr><th>N</th><th>Wall time [s]</th><th>Reported max RSS [MiB]</th><th>Newton records</th><th>FGMRES solves</th></tr></thead><tbody>{runtime_rows}</tbody></table></div><p>RSS is the maximum reported by GNU time for the MPI launcher command, not a sum across ranks. The runner uses one OpenMP thread per rank and at most {summary.get('max_cores', 4)} MPI ranks.</p></section>
{spatial_section}
<section><h2>Profiles and evolution</h2>{plots}{video}</section>
<section><h2>Files and provenance</h2><ul>{links}</ul><p><a href="summary.json">Summary JSON</a> · <a href="manifest.json">Run manifest</a> · <a href="build/build.log">Build log</a> · <a href="build/config.log">Binary configuration</a></p></section></body></html>\n''', encoding="utf-8")


def make_plots(root: Path, profiles: dict, histories: dict, summary: dict):
    finest = profiles[max(profiles)]["final"]
    alignment = {}
    for field in ("u", "v"):
        derivative = np.gradient(finest[f"{field}_analytic"], finest["x"])
        shift = float(np.dot(finest[field]-finest[f"{field}_analytic"], derivative)
                      / np.dot(derivative, derivative))
        alignment[field] = shift/float(np.mean(np.diff(finest["x"])))
    summary["alignment"] = {"applied_shift": 0.0,
                            "diagnostic_shift_cells": alignment,
                            "definition": "linearized unweighted least-squares shift; diagnostic only"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for col, stage in enumerate(("early", "final")):
        for row, field in enumerate(("u", "v")):
            ax = axes[row, col]
            for n, p in profiles.items():
                q = p[stage]
                ax.plot(q["x"], q[field], "o", ms=3, label=f"N={n}")
                if n == max(profiles):
                    ax.plot(q["x"], q[f"{field}_analytic"], "k-", lw=1.8, label="analytic")
            ax.set(xlabel="x = √3κz", ylabel=field, title=f"{field}, τ≈{profiles[max(profiles)][stage]['tau']:.4g}")
            ax.grid(alpha=.25)
            ax.legend()
    save_figure(fig, root/"figures/profiles")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    n = max(profiles)
    for ax, field in zip(axes, ("u", "v")):
        for stage, color in (("early", "#2878b5"), ("final", "#d65c27")):
            q = profiles[n][stage]
            ax.loglog(q["x"], np.maximum(q[f"{field}_analytic"], 1e-16),
                      color=color, lw=2, label=f"analytic {stage}")
            ax.loglog(q["x"], np.maximum(q[field], 1e-16), "o", color=color,
                      ms=3, label=f"N={n} {stage}")
        ax.set(xlabel="x = √3κz", ylabel=field, ylim=(1e-12,1),
               title=f"{field}: wave penetration")
        ax.grid(alpha=.25, which="both")
        ax.legend(fontsize=9)
    save_figure(fig, root/"figures/profiles_log")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    n = max(profiles)
    major_ticks = np.array([0.2, 0.5, 1.0, 2.0, 5.0])
    minor_ticks = np.concatenate([decade*np.arange(1, 10)
                                  for decade in (0.1, 1.0, 10.0)])
    minor_ticks = np.array([tick for tick in minor_ticks
                            if not np.any(np.isclose(tick, major_ticks, rtol=1e-10))])
    for axis, stage in zip(axes, ("early", "final")):
        q = profiles[n][stage]
        for field, color, linestyle in (("u", "#2878b5", "--"),
                                        ("v", "#d65c27", "-.")):
            axis.loglog(q["x"], np.maximum(q[field], 1e-16), color=color,
                        lw=1.8, label=f"{field} numerical, N={n}")
            axis.loglog(q["x"], np.maximum(q[f"{field}_analytic"], 1e-16),
                        color="#555555", lw=1.8, linestyle=linestyle,
                        label=f"{field} analytic")
        axis.set(xlim=(q["x"][0], q["x"][-1]), xlabel="x = √3κz",
                 ylabel="u, v", title=f"τ={q['tau']:.4g}")
        axis.xaxis.set_major_locator(ticker.FixedLocator(major_ticks))
        axis.xaxis.set_major_formatter(ticker.FixedFormatter(
            ["0.2", "0.5", "1", "2", "5"]))
        axis.xaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
        axis.xaxis.set_minor_formatter(ticker.NullFormatter())
        axis.tick_params(axis="x", which="major", length=8, width=1.2)
        axis.tick_params(axis="x", which="minor", length=4)
        axis.grid(True, which="major", color="lightgray", linewidth=.6)
        axis.legend(fontsize=9)
    save_figure(fig, root/"figures/comparison")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for col, stage in enumerate(("early", "final")):
        for row, field in enumerate(("u", "v")):
            ax = axes[row, col]
            for n, p in profiles.items():
                q = p[stage]
                err = np.abs(q[field]-q[f"{field}_analytic"])
                ax.semilogy(q["x"], np.maximum(err,1e-16), label=f"N={n}")
            ax.set(xlabel="x = √3κz", ylabel=f"|{field}−analytic|", title=f"{field}, {stage}")
            ax.grid(alpha=.25, which="both")
            ax.legend()
    save_figure(fig, root/"figures/errors")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))
    for n, hist in histories.items():
        for ax, key in zip(axes, ("Er_max", "egas_max")):
            if key in hist:
                ax.plot(hist["tau"], hist[key], label=f"N={n}")
                ax.set(xlabel="τ", ylabel=key)
                ax.grid(alpha=.25)
                ax.legend()
    save_figure(fig, root/"figures/history")
    convergence_rows = []
    resolutions = sorted(profiles)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1))
    for axis, stage in zip(axes, ("early", "final")):
        for field, color in (("u", "#2878b5"), ("v", "#d65c27")):
            errors = [error_metrics(profiles[n][stage][field],
                                    profiles[n][stage][f"{field}_analytic"])["l1_relative"]
                      for n in resolutions]
            for i, (n, error) in enumerate(zip(resolutions, errors)):
                order = (math.log(errors[i-1]/error)/
                         math.log(n/resolutions[i-1])) if i else None
                convergence_rows.append({"resolution": n, "stage": stage,
                                         "field": field, "relative_l1": error,
                                         "order_from_previous": order})
            axis.loglog(resolutions, errors, "o-", color=color, label=field)
            if len(resolutions) > 1:
                ref = errors[0]*(np.array(resolutions)/resolutions[0])**-2
                axis.loglog(resolutions, ref, "k--", alpha=.55,
                            label="N⁻² reference" if field == "u" else None)
        axis.set(xlabel="N in x₁", ylabel="relative L1", title=f"τ≈{profiles[resolutions[-1]][stage]['tau']:.4g}")
        axis.set_xticks(resolutions, labels=[str(n) for n in resolutions])
        axis.xaxis.set_minor_formatter(ticker.NullFormatter())
        axis.grid(True, which="both", alpha=.25)
        axis.legend()
    save_figure(fig, root/"figures/convergence")
    with (root/"convergence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("resolution", "stage", "field",
                                                   "relative_l1", "order_from_previous"))
        writer.writeheader()
        writer.writerows(convergence_rows)
    summary["convergence"] = {"rows": convergence_rows,
                              "definition": "sum(abs(numerical-analytic))/sum(abs(analytic))",
                              "time_step": 0.003,
                              "monotonic": all(
                                  all(errors[i+1] < errors[i] for i in range(len(errors)-1))
                                  for stage in ("early", "final") for field in ("u", "v")
                                  for errors in [[row["relative_l1"] for row in convergence_rows
                                                  if row["stage"] == stage and row["field"] == field]])}


def make_video(root: Path, frames: list[dict], records: list):
    with tempfile.TemporaryDirectory(prefix="marshak_frames_") as tmp:
        directory = Path(tmp)
        ymax = max(float(max(f["u"].max(), f["v"].max())) for f in frames)*1.1
        for i, frame in enumerate(frames):
            fig, ax = plt.subplots(figsize=(9, 5.2))
            ax.plot(frame["x"], frame["u"], label="u: radiation")
            ax.plot(frame["x"], frame["v"], label="v: gas")
            ax.set(xlim=(0, float(frame["x"][-1])), ylim=(0,ymax),
                   xlabel="x = √3κz", ylabel="normalized energy", title=f"Marshak wave: τ={frame['tau']:.4g}")
            ax.grid(alpha=.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(directory/f"frame_{i:03d}.png", dpi=150)
            plt.close(fig)
        target = root/"video/marshak_wave.mp4"
        target.parent.mkdir(exist_ok=True)
        command = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", "5", "-i",
                   str(directory/"frame_%03d.png"), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                   "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                   "-movflags", "+faststart", str(target)]
        records.append(checked(command, root, target.parent/"encode.log", 300, "video_encode"))
    return {"path": str(target), "frames": len(frames), "sha256": sha256_file(target)}


def load_existing_case(case: Path, result: dict, steps: int) -> tuple[dict, dict]:
    """Recheck the saved profiles and load them without rerunning Athena++."""
    if result.get("status") != "passed":
        raise RuntimeError(f"Cannot reuse a failed case: {case}")
    params = (case/"problem_parameters.txt").read_text()
    match = re.search(r"^tau_factor\s*=\s*([\d.eE+-]+)", params, re.M)
    if match is None:
        raise ValueError(f"No tau_factor in {case}/problem_parameters.txt")
    tau_factor = float(match.group(1))
    stages = {}
    for stage, pattern, cycle in (("early", "*.uov_x_early.*.athdf", 34),
                                  ("final", "*.uov_x_final.*.athdf", steps)):
        paths = sorted((case/"output").glob(pattern))
        path = next((candidate for candidate in paths
                     if int(read_athdf(candidate)["NumCycles"]) == cycle), None)
        if path is None:
            raise FileNotFoundError(f"Missing cycle {cycle} {stage} HDF5 in {case}")
        data = profile(path, tau_factor)
        matrix = np.loadtxt(case/f"profile_{stage}.csv", delimiter=",", skiprows=1)
        if matrix.shape != (len(data["x"]), 5) or not np.allclose(matrix[:, 0], data["x"]):
            raise ValueError(f"Saved profile does not match HDF5: {case}, {stage}")
        for field, numeric_col, analytic_col in (("u", 1, 2), ("v", 3, 4)):
            if not np.allclose(matrix[:, numeric_col], data[field]):
                raise ValueError(f"Saved {field} does not match HDF5: {case}, {stage}")
            data[f"{field}_analytic"] = matrix[:, analytic_col]
            measured = error_metrics(data[field], data[f"{field}_analytic"])
            if not np.isclose(measured["l1_relative"],
                              result["stages"][stage][field]["l1_relative"], rtol=1e-7):
                raise ValueError(f"Saved error does not match CSV: {case}, {stage}, {field}")
        stages[stage] = data
    return stages, read_history(case/"output")


def main() -> int:
    args = parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    repo = find_repo_root(HERE)
    name = args.run_name or f"validation_marshak_{datetime.now(timezone(timedelta(hours=9))):%Y%m%d_%H%M%S}_JST"
    root = args.output_root.expanduser().resolve()/name
    root.mkdir(parents=True, exist_ok=False)
    if args.resume_from:
        prior = args.resume_from.expanduser().resolve()
        shutil.copytree(prior, root, dirs_exist_ok=True)
        summary = json.loads((root/"summary.json").read_text())
        if summary["steps"] != args.steps:
            raise ValueError("--steps must match the copied run when using --resume-from")
        summary.update(run=name, status="running", started_at=utc_now(),
                       max_cores=args.max_cores,
                       resolutions=args.resolutions, resumed_from=str(prior))
        summary.pop("video", None)
        summary.pop("convergence", None)
        summary["cases"] = {key: value for key, value in summary["cases"].items()
                            if int(key[1:]) in args.resolutions}
        manifest = json.loads((root/"manifest.json").read_text())
        manifest["resumed_from"] = str(prior)
        manifest.update(started_at=utc_now(), status="running", finished_at=None)
        manifest["resources"]["max_cores"] = args.max_cores
    else:
        summary = {"run": name, "status": "running", "started_at": utc_now(),
                   "resolutions": args.resolutions, "steps": args.steps,
                   "max_cores": args.max_cores, "cases": {}}
        manifest = base_manifest(repo, "marshak_validation", args.max_cores)
    write_report(root, summary)
    records = manifest["commands"].setdefault("records", [])
    manifest["analysis_script_sha256"] = sha256_file(Path(__file__))
    build_dir = root/"build"
    build_dir.mkdir(exist_ok=True)
    try:
        source = build_dir/"build_source"
        if args.resume_from:
            pass
        elif args.reuse_build:
            shutil.copytree(args.reuse_build.resolve(), build_dir, dirs_exist_ok=True)
            manifest["build_reused_from"] = str(args.reuse_build.resolve())
        else:
            prepare_build_source(repo, source)
            cfg = ["python3", str(source/"configure.py"), "--prob=nrfld_marshak",
                   "--eos=general/marshak", "-nrmgfld", "-hdf5", "-mpi", "--cflag=-O3"]
            records.append(checked(cfg, source, build_dir/"configure.log", args.build_timeout, "configure"))
            records.append(checked(["make", f"-j{args.max_cores}"], source,
                                   build_dir/"build.log", args.build_timeout, "build"))
        source_hash = tree_sha256(source/"src")
        exe = source/"bin/athena"
        config = binary_configuration(exe, build_dir, 30, records)
        for key, expected in {"problem_generator":"nrfld_marshak",
                              "equation_of_state":"general/marshak",
                              "fld_with_newton-raphson":"ON", "mpi_parallelism":"ON",
                              "hdf5_output":"ON"}.items():
            if config.get(key) != expected:
                raise RuntimeError(f"binary {key}={config.get(key)!r}; expected {expected!r}")
        manifest["provenance"]["calculation"] = {"source_sha256": source_hash,
                                                  "binary_sha256": sha256_file(exe),
                                                  "binary_configuration": config}
        source_input = repo/"inputs/radiation/athinput.nrfld_marshak"
        profiles = {}
        histories = {}
        frames = []
        for n in args.resolutions:
            key = f"n{n}"
            case = root/"cases"/key
            if key in summary["cases"]:
                stages, history = load_existing_case(case, summary["cases"][key], args.steps)
                profiles[n] = stages
                histories[n] = history
                case_manifest = json.loads((case/"manifest.json").read_text())
                case_manifest["imported_from"] = str(args.resume_from.resolve()/"cases"/key)
                write_manifest(case/"manifest.json", case_manifest)
                print(f"Reused Marshak N={n} from {args.resume_from}", flush=True)
                continue
            case.mkdir(parents=True)
            output = case/"output"
            output.mkdir()
            case_manifest = base_manifest(repo, key, args.max_cores)
            case_manifest["provenance"]["calculation"] = manifest["provenance"]["calculation"]
            overrides = [f"mesh/nx1={n}", "mesh/nx2=32", "mesh/nx3=32",
                         "mesh/x2min=0", f"mesh/x2max={160/n:.17g}",
                         "mesh/x3min=0", f"mesh/x3max={160/n:.17g}",
                         "meshblock/nx1=32", "meshblock/nx2=32", "meshblock/nx3=32",
                         f"time/nlim={args.steps}", f"time/tlim={args.steps*.003:.17g}",
                         "time/ncycle_out=100", "output1/dcycle=10", "output3/dcycle=34",
                         f"output4/dcycle={args.steps}",
                         f"output3/x2_slice={80/n:.17g}", f"output3/x3_slice={80/n:.17g}",
                         f"output4/x2_slice={80/n:.17g}", f"output4/x3_slice={80/n:.17g}",
                         "nrfld/linear_solver=fgmres", "nrfld/threshold=1e-11",
                         "nrfld/nr_threshold=1e-10", "nrfld/diagnostic_verbosity=0",
                         "nrfld/failure_policy=abort", "problem/Tgas0=1.0",
                         f"job/problem_id=marshak_n{n}"]
            settings = materialize_input(source_input, case/"input.used", overrides)
            case_manifest["input"] = {"source_path":str(source_input),
                                      "source_sha256":sha256_file(source_input),
                                      "used_sha256":sha256_file(case/"input.used"),
                                      "effective_settings":settings}
            ranks = min(args.max_cores,n//32)
            command = ["/usr/bin/time", "-v", "-o", str(case/"resources.txt"),
                       "mpirun", "-np", str(ranks), str(exe), "-i", str(case/"input.used"),
                       "-d", str(output)]
            case_manifest["resources"] = {"mpi_ranks":ranks,"omp_threads":1}
            case_manifest["commands"]["run"] = command
            case_records = case_manifest["commands"].setdefault("records", [])
            write_manifest(case/"manifest.json", case_manifest)
            print(f"Running Marshak N={n}, {args.steps} steps, MPI={ranks}", flush=True)
            try:
                record = checked(command, case, case/"run.log", args.run_timeout, "run")
                case_records.append(record)
                rss_match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)",
                                      (case/"resources.txt").read_text())
                case_manifest["resources"]["max_rss_kib"] = (
                    int(rss_match.group(1)) if rss_match else None)
                params = (case/"problem_parameters.txt").read_text()
                tau_factor = float(re.search(r"^tau_factor\s*=\s*([\d.eE+-]+)", params, re.M).group(1))
                early = sorted(output.glob("*.uov_x_early.*.athdf"))
                final = sorted(output.glob("*.uov_x_final.*.athdf"))
                early_path = next(p for p in early if int(read_athdf(p)["NumCycles"]) == 34)
                final_path = next(p for p in final if int(read_athdf(p)["NumCycles"]) == args.steps)
                stages = {"early":profile(early_path,tau_factor),
                          "final":profile(final_path,tau_factor)}
                case_result = {"status":"running", "stages":{}}
                for stage, p in stages.items():
                    au,av = analytic(p["x"],p["tau"])
                    p["u_analytic"],p["v_analytic"] = au,av
                    case_result["stages"][stage] = {field:error_metrics(p[field],p[f"{field}_analytic"])
                                                     for field in ("u","v")}
                    with (case/f"profile_{stage}.csv").open("w",newline="") as handle:
                        writer=csv.writer(handle)
                        writer.writerow(("x_dimless","u","u_analytic","v","v_analytic"))
                        writer.writerows(zip(p["x"],p["u"],au,p["v"],av))
                log = (case/"run.log").read_text(errors="replace")
                nr = re.findall(r"\[NR\] status=([^\s]+)",log)
                krylov = re.findall(r"\[LINEAR_FGMRES\].*?relative_residual=([^\s]+).*?target=([^\s]+)",log)
                active_nr = [status for status in nr if status != "dt_zero_initialization"]
                solver_ok = (len(active_nr)==args.steps and all(x=="converged" for x in active_nr)
                             and bool(krylov)
                             and all(float(a)<=float(b)*1.000001 for a,b in krylov))
                finite = all(np.all(np.isfinite(p[field])) and np.all(p[field]>=0)
                             for p in stages.values() for field in ("u","v"))
                errors_ok = all(case_result["stages"][stage][field]["l1_relative"]<.2
                                for stage in stages for field in ("u","v"))
                case_result.update(passed=bool(solver_ok and finite and errors_ok),
                                   solver_ok=solver_ok,finite_nonnegative=bool(finite),
                                   nr_status_count=len(nr),fgmres_calls=len(krylov),
                                   wall_seconds=record["elapsed_seconds"],
                                   max_rss_kib=case_manifest["resources"]["max_rss_kib"],
                                   status="passed" if solver_ok and finite and errors_ok else "failed")
                case_manifest["metrics"] = case_result
                profiles[n] = stages
                histories[n] = read_history(output)
                if n == max(args.resolutions):
                    frames = [profile(p,tau_factor) for p in early]
                    if frames[-1]["cycles"] != args.steps:
                        frames.append(stages["final"])
                print(f"  {case_result['status']}: early u L1={case_result['stages']['early']['u']['l1_relative']:.4g}, final u L1={case_result['stages']['final']['u']['l1_relative']:.4g}",flush=True)
            except Exception as error:
                if isinstance(error,CommandExecutionError):
                    case_records.append(error.record)
                case_result = {"status":"failed","passed":False,"error":str(error),"stages":{}}
                case_manifest["error"] = str(error)
                print(f"  failed: {error}",flush=True)
            case_manifest["status"] = case_result["status"]
            case_manifest["finished_at"] = utc_now()
            write_manifest(case/"manifest.json",case_manifest)
            summary["cases"][key] = case_result
            (root/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
            write_report(root,summary)
        if len(profiles)==len(args.resolutions):
            make_plots(root,profiles,histories,summary)
            if len(frames)>=5:
                summary["video"] = make_video(root,frames,records)
        summary["status"] = ("passed" if all(c.get("passed") for c in summary["cases"].values())
                             and "video" in summary else "failed")
    except Exception as error:
        if isinstance(error,CommandExecutionError):
            records.append(error.record)
        summary.update(status="failed",error=str(error))
        print(f"Marshak validation failed: {error}",flush=True)
    summary["finished_at"] = utc_now()
    manifest["metrics"] = {"cases": {key: value.get("status")
                                     for key, value in summary["cases"].items()},
                           "convergence": summary.get("convergence", {})}
    manifest.update(status=summary["status"],finished_at=summary["finished_at"])
    write_manifest(root/"manifest.json",manifest)
    (root/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    write_report(root,summary)
    print(root,flush=True)
    return 0 if summary["status"]=="passed" else 1


if __name__=="__main__":
    raise SystemExit(main())
