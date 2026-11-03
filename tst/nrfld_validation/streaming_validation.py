#!/usr/bin/env python3
"""Re-run and quantify the optically thin NR-FLD streaming-front benchmark."""

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
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from diffusion_validation import materialize_input  # noqa: E402
from nrfld_validation.io import find_repo_root, read_athdf  # noqa: E402
from nrfld_validation.report_style import REPORT_TYPOGRAPHY  # noqa: E402
from nrfld_validation.manifest import (base_manifest, sha256_file, tree_sha256,
                                       utc_now, write_manifest)  # noqa: E402
from run_suite import (CommandExecutionError, binary_configuration, checked,
                       prepare_build_source)  # noqa: E402

LIGHT_SPEED = 2.99792458e10
FINAL_TIME = 2.0e-9
LEFT, RIGHT, FRONT0 = 1.0, 1.0e-10, 10.0
CHIS = (1.0e-4, 1.0e-7)
DIVISORS = (2, 20)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/d/output/rad"))
    parser.add_argument("--run-name")
    parser.add_argument("--max-cores", type=int, default=4)
    parser.add_argument("--run-timeout", type=float, default=7200)
    parser.add_argument("--build-timeout", type=float, default=900)
    parser.add_argument("--reuse-build", type=Path,
                        help="Copy an existing streaming build into the new run.")
    parser.add_argument("--postprocess", type=Path,
                        help="Regenerate metrics and figures from an existing run; do not calculate.")
    args = parser.parse_args()
    if args.max_cores < 1 or args.run_timeout <= 0 or args.build_timeout <= 0:
        parser.error("cores and timeouts must be positive")
    return args


def case_key(chi: float, divisor: int) -> str:
    return f"chi{chi:.0e}_dt{divisor}".replace("e-0", "e-")


def load_profile(path: Path) -> dict:
    data = read_athdf(path)
    x = np.asarray(data["x"], dtype=float).reshape(-1, len(data["x1v"])).mean(axis=0)
    energy = np.asarray(data["Erad"], dtype=float).reshape(-1, len(x)).mean(axis=0)
    if len(x) != 128 or not np.all(np.diff(x) > 0):
        raise ValueError(f"unexpected x grid in {path}: {len(x)} cells")
    if not np.allclose(x, data["x1v"], rtol=0, atol=1e-5):
        raise ValueError(f"user-output x disagrees with cell centers: {path}")
    if not np.all(np.isfinite(energy)) or np.any(energy < 0):
        raise ValueError(f"nonfinite or negative radiation energy: {path}")
    return {"x": x, "energy": energy, "time": float(data["Time"]),
            "cycles": int(data["NumCycles"]), "path": path}


def threshold_position(x: np.ndarray, y: np.ndarray, level: float) -> float:
    """Rightmost downward crossing, interpolated between native cell centers."""
    candidates = np.flatnonzero((y[:-1] >= level) & (y[1:] < level))
    if len(candidates) != 1:
        raise ValueError(f"expected one crossing at level {level:g}; found {len(candidates)}")
    i = int(candidates[0])
    return float(x[i] + (level-y[i])*(x[i+1]-x[i])/(y[i+1]-y[i]))


def front_metrics(profile: dict) -> dict:
    x, energy, t = profile["x"], profile["energy"], profile["time"]
    normalized = (energy-RIGHT)/(LEFT-RIGHT)
    x90 = threshold_position(x, normalized, .9)
    x50 = threshold_position(x, normalized, .5)
    x10 = threshold_position(x, normalized, .1)
    expected = FRONT0+LIGHT_SPEED*t
    dx = float(np.median(np.diff(x)))
    ideal = np.where(x < expected, LEFT, RIGHT)
    ahead = x > expected
    leakage = (float(np.trapezoid(np.maximum(energy[ahead]-RIGHT, 0), x[ahead]))
               if np.count_nonzero(ahead) > 1 else 0.0)
    return {"time_s": t, "cycles": profile["cycles"],
            "expected_x_cm": expected, "x90_cm": x90, "x50_cm": x50,
            "x10_cm": x10, "x50_minus_expected_cm": x50-expected,
            "x10_minus_expected_cm": x10-expected,
            "x50_error_cells": (x50-expected)/dx, "width_10_90_cm": x10-x90,
            "width_10_90_cells": (x10-x90)/dx,
            "step_l1_normalized": float(np.mean(np.abs(energy-ideal))/(LEFT-RIGHT)),
            "ahead_energy_fraction": leakage/((LEFT-RIGHT)*max(expected-FRONT0, dx)),
            "energy_min": float(energy.min()), "energy_max": float(energy.max())}


def save_profile_csv(path: Path, profile: dict):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("x_cm", "Erad_erg_cm-3"))
        writer.writerows(zip(profile["x"], profile["energy"]))


def analyze_case(case: Path, chi: float, divisor: int) -> tuple[dict, list[dict], dict]:
    output = case/"output"
    early_paths = sorted(output.glob("*.uov_x_early.*.athdf"))
    final_paths = sorted(output.glob("*.uov_x_final.*.athdf"))
    if len(early_paths) < 10 or not final_paths:
        raise FileNotFoundError(f"missing early/final HDF5 snapshots in {output}")
    profiles = [load_profile(path) for path in early_paths]
    final = load_profile(final_paths[-1])
    if not math.isclose(final["time"], FINAL_TIME, rel_tol=1e-8, abs_tol=1e-15):
        raise ValueError(f"final time is {final['time']}, expected {FINAL_TIME}")
    if not math.isclose(profiles[-1]["time"], FINAL_TIME, rel_tol=1e-8, abs_tol=1e-15):
        raise ValueError("early-output series does not include the final time")
    if not np.allclose(profiles[-1]["energy"], final["energy"], rtol=1e-6, atol=1e-8):
        raise ValueError("early and final HDF5 disagree at the final time")
    trace = [front_metrics(p) for p in profiles]
    with (case/"front_trace.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace[0]))
        writer.writeheader()
        writer.writerows(trace)
    save_profile_csv(case/"profile_final.csv", final)
    fit = np.array([(row["time_s"], row["x50_cm"]) for row in trace
                    if FINAL_TIME*.25 <= row["time_s"] <= FINAL_TIME], dtype=float)
    if len(fit) < 5:
        raise ValueError("too few snapshots to fit the front speed")
    speed = float(np.polyfit(fit[:, 0]/FINAL_TIME, fit[:, 1], 1)[0]/FINAL_TIME)
    log = (case/"run.log").read_text(errors="replace")
    nr = re.findall(r"\[NR\] status=([^\s]+)", log)
    active_nr = [status for status in nr if status != "dt_zero_initialization"]
    krylov = re.findall(r"\[LINEAR_FGMRES\].*?relative_residual=([^\s]+).*?target=([^\s]+)", log)
    solver_ok = (len(active_nr) == final["cycles"]
                 and all(status == "converged" for status in active_nr)
                 and bool(krylov)
                 and all(float(value) <= float(target)*1.000001 for value, target in krylov))
    dx = float(np.median(np.diff(final["x"])))
    requested_dt = dx/(divisor*LIGHT_SPEED)
    measured_dt = np.array([float(s) for s in re.findall(
        r"\[NR\] status=converged[^\n]*? stage=0 dt=([^\s]+)", log)])
    dt_ok = (len(measured_dt) == final["cycles"]
             and np.allclose(measured_dt[:-1], requested_dt, rtol=1e-7, atol=1e-16)
             and measured_dt[-1] > 0 and measured_dt[-1] <= requested_dt*(1+1e-7)
             and math.isclose(float(measured_dt.sum()), FINAL_TIME, rel_tol=1e-8,
                              abs_tol=1e-15))
    last = trace[-1]
    result = {"status": "passed" if solver_ok and dt_ok else "failed",
              "chi_R_cm-1": chi, "dt_light_divisor": divisor,
              "final": last, "front_speed_over_c": speed/LIGHT_SPEED,
              "front_speed_fit_time_range_s": [FINAL_TIME*.25, FINAL_TIME],
              "nr_steps": len(active_nr), "fgmres_calls": len(krylov),
              "solver_ok": solver_ok, "dt_ok": bool(dt_ok),
              "requested_dt_s": requested_dt,
              "measured_dt_min_s": float(measured_dt.min()) if len(measured_dt) else None,
              "measured_dt_max_s": float(measured_dt.max()) if len(measured_dt) else None}
    return result, profiles, final


def save_figure(fig, path: Path):
    path.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def make_plots(root: Path, profiles: dict, finals: dict, results: dict):
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0})
    for ax, chi in zip(axes, CHIS):
        for divisor, color in ((2, "#2166ac"), (20, "#d95f02")):
            p = finals[case_key(chi, divisor)]
            ax.plot(p["x"], np.maximum(p["energy"], 1e-12), color=color,
                    lw=2, label=rf"$\Delta t=\Delta x/({divisor}c)$")
        ax.axvline(FRONT0+LIGHT_SPEED*FINAL_TIME, color="black", ls="--", lw=1.6,
                   label=r"$x_0+ct$")
        ax.set_yscale("log")
        ax.set_ylim(1e-10, 2)
        ax.set_xlim(0, 100)
        ax.text(.03, .43, rf"$\chi_R={chi:.0e}\ \mathrm{{cm}}^{{-1}}$",
                transform=ax.transAxes, fontsize=15)
        ax.grid(which="both", alpha=.25)
        ax.legend(fontsize=11, loc="lower left")
    axes[-1].set_xlabel("x [cm]")
    fig.supylabel(r"$E_{\rm rad}$ [erg cm$^{-3}$]")
    save_figure(fig, root/"figures/two_panel")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for row, chi in enumerate(CHIS):
        ax, bx = axes[row]
        for divisor, color in ((2, "#2166ac"), (20, "#d95f02")):
            p = finals[case_key(chi, divisor)]
            ax.plot(p["x"], p["energy"], lw=2, color=color,
                    label=rf"$\Delta x/({divisor}c)$")
            b = results[case_key(chi, divisor)]["final"]
            bx.plot(p["x"], np.maximum(p["energy"], 1e-12), lw=2, color=color)
            bx.axvline(b["x50_cm"], ls=":", lw=1.2, color=color)
        for axis in (ax, bx):
            axis.axvline(FRONT0+LIGHT_SPEED*FINAL_TIME, color="black", ls="--")
            axis.set_xlim(40, 90)
            axis.grid(alpha=.25, which="both")
        ax.set(ylabel=rf"$\chi_R={chi:.0e}$: $E_{{\rm rad}}$", ylim=(-.03, 1.05))
        ax.legend(loc="lower left")
        bx.set_yscale("log")
        bx.set_ylim(1e-7, 2)
    axes[0, 0].set_title("Linear front profile")
    axes[0, 1].set_title("Log profile and 50% crossings")
    axes[-1, 0].set_xlabel("x [cm]")
    axes[-1, 1].set_xlabel("x [cm]")
    save_figure(fig, root/"figures/front_zoom")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    times = np.linspace(0, FINAL_TIME, 100)
    axes[0].plot(times*1e9, FRONT0+LIGHT_SPEED*times, "k--", label=r"$x_0+ct$")
    for chi, style in ((CHIS[0], "-"), (CHIS[1], "--")):
        for divisor, color in ((2, "#2166ac"), (20, "#d95f02")):
            key = case_key(chi, divisor)
            trace = [front_metrics(p) for p in profiles[key]]
            t = np.array([r["time_s"] for r in trace])*1e9
            label = rf"$\chi_R={chi:.0e}$, $\Delta x/({divisor}c)$"
            axes[0].plot(t, [r["x50_cm"] for r in trace], color=color,
                         ls=style, marker="o", ms=3, label=label)
            axes[1].plot(t, [r["width_10_90_cm"] for r in trace], color=color,
                         ls=style, marker="o", ms=3, label=label)
    axes[0].set(xlabel="t [ns]", ylabel="50% crossing [cm]", title="Front position")
    axes[1].set(xlabel="t [ns]", ylabel="10–90% width [cm]", title="Numerical broadening")
    for ax in axes:
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    save_figure(fig, root/"figures/front_trace")


def make_video(root: Path, profiles: dict, records: list):
    target = root/"video/streaming_front.mp4"
    target.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="streaming_frames_") as temp:
        directory = Path(temp)
        for frame in range(min(len(profiles[case_key(chi, divisor)])
                               for chi in CHIS for divisor in DIVISORS)):
            fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                     gridspec_kw={"hspace": 0})
            for ax, chi in zip(axes, CHIS):
                for divisor, color in ((2, "#2166ac"), (20, "#d95f02")):
                    p = profiles[case_key(chi, divisor)][frame]
                    ax.plot(p["x"], np.maximum(p["energy"], 1e-12), color=color,
                            lw=2, label=rf"$\Delta x/({divisor}c)$")
                ax.axvline(FRONT0+LIGHT_SPEED*p["time"], color="black", ls="--",
                           label=r"$x_0+ct$")
                ax.set(yscale="log", ylim=(1e-10, 2), xlim=(0, 100))
                ax.text(.03, .43, rf"$\chi_R={chi:.0e}\ \mathrm{{cm}}^{{-1}}$",
                        transform=ax.transAxes)
                ax.grid(alpha=.2, which="both")
                ax.legend(loc="lower left", fontsize=9)
            axes[0].set_title(f"Streaming front, t={p['time']*1e9:.2f} ns")
            axes[-1].set_xlabel("x [cm]")
            fig.supylabel(r"$E_{\rm rad}$ [erg cm$^{-3}$]")
            fig.tight_layout()
            fig.savefig(directory/f"frame_{frame:03d}.png", dpi=130)
            plt.close(fig)
        command = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", "5",
                   "-i", str(directory/"frame_%03d.png"),
                   "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                   "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                   "-movflags", "+faststart", str(target)]
        records.append(checked(command, root, target.parent/"encode.log", 300, "video_encode"))
    return {"path": str(target.relative_to(root)), "sha256": sha256_file(target),
            "frames": frame+1}


def write_report(root: Path, summary: dict):
    rows = []
    links = []
    for chi in CHIS:
        for divisor in DIVISORS:
            key = case_key(chi, divisor)
            result = summary["cases"].get(key, {})
            status = result.get("status", "pending")
            final = result.get("final")
            if final:
                rows.append(f'<tr><th>{chi:.0e}</th><th>Δx/({divisor}c)</th>'
                            f'<td class="{status}"><strong>{status.upper()}</strong></td>'
                            f'<td>{final["x50_cm"]:.3f}</td>'
                            f'<td><strong>{final["x50_minus_expected_cm"]:+.3f}</strong></td>'
                            f'<td>{final["x50_error_cells"]:+.2f}</td>'
                            f'<td>{final.get("x10_minus_expected_cm", final["x10_cm"]-final["expected_x_cm"]):+.3f}</td>'
                            f'<td>{final["width_10_90_cm"]:.3f}</td>'
                            f'<td>{result["front_speed_over_c"]:.3f}</td>'
                            f'<td>{final["ahead_energy_fraction"]:.3g}</td>'
                            f'<td>{final["step_l1_normalized"]:.3g}</td></tr>')
            else:
                rows.append(f'<tr><th>{chi:.0e}</th><th>Δx/({divisor}c)</th>'
                            f'<td class="{status}"><strong>{status.upper()}</strong></td>'
                            '<td colspan="8">' + escape(result.get("error", "Awaiting calculation")) + '</td></tr>')
            case = root/"cases"/key
            available = [(name, label) for name, label in
                         (("input.used", "input"), ("run.log", "log"),
                          ("front_trace.csv", "front trace CSV"),
                          ("profile_final.csv", "final profile CSV"),
                          ("manifest.json", "manifest")) if (case/name).is_file()]
            available += [(f"output/{p.name}", "final HDF5") for p in
                          sorted((case/"output").glob("*.uov_x_final.*.athdf"))[-1:]]
            links.append(f'<li><strong>{escape(key)}</strong>: ' + ' · '.join(
                f'<a href="cases/{key}/{escape(name)}">{escape(label)}</a>'
                for name, label in available) + '</li>')
    figures = ''.join(f'<figure><img src="figures/{name}.png" alt="{escape(label)}" loading="lazy">'
                      f'<figcaption>{escape(label)} · <a href="figures/{name}.pdf">PDF</a></figcaption></figure>'
                      for name, label in (("two_panel", "Final profiles: two-panel comparison"),
                                          ("front_zoom", "Front shape: linear and log views"),
                                          ("front_trace", "Front position and width over time"))
                      if (root/"figures"/f"{name}.png").is_file())
    movie = ('<figure><video controls preload="metadata" playsinline width="100%">'
             '<source src="video/streaming_front.mp4" type="video/mp4"></video>'
             '<figcaption>Time evolution · <a href="video/streaming_front.mp4">MP4</a></figcaption></figure>'
             if (root/"video/streaming_front.mp4").is_file() else "")
    status = summary["status"]
    root.joinpath("index.html").write_text(f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Streaming-front validation</title><style>
:root{{font:18px/1.6 system-ui,sans-serif;color:#17212b;background:#f7f9fb}}
body{{max-width:1250px;margin:2rem auto;padding:0 1.2rem}}h1,h2{{line-height:1.25}}
section{{background:white;border:1px solid #dce4eb;border-radius:10px;padding:1.3rem;margin:1.4rem 0}}
.overview{{background:#eaf3fb;border-left:5px solid #1769aa;padding:.8rem 1rem;font-size:1.1rem}}
.passed{{color:#0b683e;font-weight:700}}.failed{{color:#ad261f;font-weight:700}}.pending{{color:#41556b}}
.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:.96rem}}
th,td{{padding:.55rem .65rem;border-bottom:1px solid #dce4eb;text-align:left;white-space:nowrap}}
thead th{{background:#eef3f7}}figure{{margin:1.3rem 0}}img,video{{max-width:100%;height:auto;display:block}}
figcaption{{font-weight:600;margin:.4rem 0}}a{{color:#075ca8}}
{REPORT_TYPOGRAPHY}</style></head><body><h1>Streaming-front validation <span class="{escape(status)}">{escape(status.upper())}</span></h1>
<p class="overview"><strong>{sum(c.get('status') == 'passed' for c in summary['cases'].values())}/4 calculations passed</strong> · FGMRES (restart 40, max 500, 2 MG V-cycles) · t=2 ns · N<sub>x</sub>=128</p>
<p>The kinematic reference is <strong>x<sub>front</sub>=x<sub>0</sub>+ct={FRONT0+LIGHT_SPEED*FINAL_TIME:.3f} cm</strong>. It gives a front position, not an analytic FLD profile. No numerical curve is shifted or registered. For each output, the front is the interpolated crossing of <code>(E−E<sub>right</sub>)/(E<sub>left</sub>−E<sub>right</sub>)</code> at 90%, 50%, and 10%; all must have exactly one crossing.</p>
<section><h2>Front accuracy at 2 ns</h2>
<p><strong>Position error</strong> is x<sub>50</sub>−(x<sub>0</sub>+ct), so negative values mean a lag. The 10% crossing tracks the leading edge. <strong>Width</strong> is x<sub>10</sub>−x<sub>90</sub>. Speed is a linear fit to x<sub>50</sub> over 0.5–2 ns, divided by c. The ahead fraction integrates energy above the cold background beyond x<sub>0</sub>+ct and divides by (E<sub>left</sub>−E<sub>right</sub>)ct. Step L1 is the mean absolute discrepancy from an ideal sharp step, normalized by E<sub>left</sub>−E<sub>right</sub>; it measures spreading as well as lag and is not a formal exact-solution error.</p>
<div class="scroll"><table><thead><tr><th>χ<sub>R</sub> [cm⁻¹]</th><th>Δt</th><th>Status</th><th>x<sub>50</sub> [cm]</th><th>50% error [cm]</th><th>50% error [cells]</th><th>10% error [cm]</th><th>10–90% width [cm]</th><th>Speed / c</th><th>Ahead fraction</th><th>Step L1</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>PASS means the simulation reached 2 ns with converged Newton and FGMRES solves, correct time steps, finite nonnegative output, and defined front crossings. The measured accuracy values are reported separately; the discontinuous step and FLD closure do not provide a formal profile-error tolerance.</p></section>
<section><h2>Figures and movie</h2>{figures}{movie}</section>
<section><h2>Reproduction</h2><p><a href="summary.json">Summary JSON</a> · <a href="manifest.json">Run manifest</a>. Each case contains its effective input, raw output, log, front trace, final profile, and manifest.</p><ul>{''.join(links)}</ul></section>
</body></html>''', encoding="utf-8")


def postprocess(root: Path, summary: dict, manifest: dict) -> int:
    manifest["analysis_script_sha256"] = sha256_file(Path(__file__))
    profiles, finals = {}, {}
    for chi in CHIS:
        for divisor in DIVISORS:
            key = case_key(chi, divisor)
            case = root/"cases"/key
            if not (case/"run.log").is_file():
                continue
            try:
                result, profiles[key], finals[key] = analyze_case(case, chi, divisor)
                previous = summary["cases"].get(key, {})
                result.update(wall_seconds=previous.get("wall_seconds"),
                              max_rss_kib=previous.get("max_rss_kib"))
                summary["cases"][key] = result
                case_manifest = json.loads((case/"manifest.json").read_text())
                case_manifest["metrics"] = result
                case_manifest["status"] = result["status"]
                write_manifest(case/"manifest.json", case_manifest)
                print(f"{key}: {result['status']}, x50−(x0+ct)="
                      f"{result['final']['x50_minus_expected_cm']:+.3f} cm, "
                      f"width={result['final']['width_10_90_cm']:.3f} cm", flush=True)
            except Exception as error:
                summary["cases"][key] = {"status": "failed", "error": str(error)}
                print(f"{key}: analysis failed: {error}", flush=True)
            (root/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
            write_report(root, summary)
    if len(profiles) == 4:
        make_plots(root, profiles, finals, summary["cases"])
        if shutil.which("ffmpeg"):
            summary["video"] = make_video(root, profiles, manifest["commands"].setdefault("records", []))
    summary["status"] = ("passed" if len(profiles) == 4 and "video" in summary
                         and all(c.get("status") == "passed" for c in summary["cases"].values())
                         else "failed")
    summary["finished_at"] = utc_now()
    manifest["status"] = summary["status"]
    manifest["finished_at"] = summary["finished_at"]
    manifest["metrics"] = {k: v.get("status") for k, v in summary["cases"].items()}
    write_manifest(root/"manifest.json", manifest)
    (root/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    write_report(root, summary)
    print(root, flush=True)
    return 0 if summary["status"] == "passed" else 1


def main() -> int:
    args = parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    repo = find_repo_root(HERE)
    if args.postprocess:
        root = args.postprocess.resolve()
        return postprocess(root, json.loads((root/"summary.json").read_text()),
                           json.loads((root/"manifest.json").read_text()))
    name = args.run_name or f"validation_streaming_{datetime.now(timezone(timedelta(hours=9))):%Y%m%d_%H%M%S}_JST"
    root = args.output_root.expanduser().resolve()/name
    root.mkdir(parents=True, exist_ok=False)
    summary = {"run": name, "status": "running", "started_at": utc_now(),
               "cases": {}, "reference_front_cm": FRONT0+LIGHT_SPEED*FINAL_TIME}
    manifest = base_manifest(repo, "streaming_validation", args.max_cores)
    manifest["analysis_script_sha256"] = sha256_file(Path(__file__))
    write_report(root, summary)
    records = manifest["commands"].setdefault("records", [])
    try:
        build = root/"build"
        build.mkdir()
        source = build/"build_source"
        if args.reuse_build:
            shutil.copytree(args.reuse_build.expanduser().resolve(), build, dirs_exist_ok=True)
            manifest["build_reused_from"] = str(args.reuse_build.expanduser().resolve())
        else:
            prepare_build_source(repo, source)
            cfg = ["python3", str(source/"configure.py"), "--prob=nrfld_streaming_front",
                   "--eos=adiabatic", "-nrmgfld", "-hdf5", "-mpi", "--cflag=-O3"]
            records.append(checked(cfg, source, build/"configure.log", args.build_timeout, "configure"))
            records.append(checked(["make", f"-j{args.max_cores}"], source,
                                   build/"build.log", args.build_timeout, "build"))
        exe = source/"bin/athena"
        config = binary_configuration(exe, build, 30, records)
        for key, expected in {"problem_generator": "nrfld_streaming_front",
                              "equation_of_state": "adiabatic",
                              "fld_with_newton-raphson": "ON", "mpi_parallelism": "ON",
                              "hdf5_output": "ON"}.items():
            if config.get(key) != expected:
                raise ValueError(f"binary {key}={config.get(key)!r}; expected {expected!r}")
        provenance = {"source_sha256": tree_sha256(source/"src"),
                      "binary_sha256": sha256_file(exe), "binary_configuration": config}
        manifest["provenance"]["calculation"] = provenance
        template = repo/"inputs/radiation/athinput.nrfld_streaming_front"
        for chi in CHIS:
            for divisor in DIVISORS:
                key = case_key(chi, divisor)
                case = root/"cases"/key
                case.mkdir(parents=True)
                output = case/"output"
                output.mkdir()
                case_manifest = base_manifest(repo, key, args.max_cores)
                case_manifest["provenance"]["calculation"] = provenance
                settings = materialize_input(template, case/"input.used", [
                    f"problem/chi_R={chi:.17g}", f"problem/dt_light_divisor={divisor}",
                    "nrfld/linear_solver=fgmres", "nrfld/threshold=1e-11",
                    "nrfld/nr_threshold=1e-8", "nrfld/diagnostic_verbosity=0",
                    "nrfld/fgmres_restart=40", "nrfld/fgmres_max_iterations=500",
                    "nrfld/fgmres_mg_vcycles=2",
                    "nrfld/failure_policy=abort", "output2/ghost_zones=false",
                    "output3/ghost_zones=false", f"job/problem_id=streaming_{key}"])
                case_manifest["input"] = {"source_path": str(template),
                                          "source_sha256": sha256_file(template),
                                          "used_sha256": sha256_file(case/"input.used"),
                                          "effective_settings": settings}
                ranks = min(args.max_cores, 4)
                case_manifest["resources"] = {"mpi_ranks": ranks, "omp_threads": 1}
                command = ["/usr/bin/time", "-v", "-o", str(case/"resources.txt"),
                           "mpirun", "-np", str(ranks), str(exe),
                           "-i", str(case/"input.used"), "-d", str(output)]
                case_manifest["commands"]["run"] = command
                write_manifest(case/"manifest.json", case_manifest)
                print(f"Running {key}, MPI={ranks}", flush=True)
                try:
                    record = checked(command, case, case/"run.log", args.run_timeout, "run")
                    case_manifest["commands"]["records"] = [record]
                    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)",
                                      (case/"resources.txt").read_text())
                    summary["cases"][key] = {"status": "running",
                                             "wall_seconds": record["elapsed_seconds"],
                                             "max_rss_kib": int(match.group(1)) if match else None}
                except Exception as error:
                    if isinstance(error, CommandExecutionError):
                        case_manifest["commands"]["records"] = [error.record]
                    case_manifest["status"] = "failed"
                    case_manifest["error"] = str(error)
                    summary["cases"][key] = {"status": "failed", "error": str(error)}
                    print(f"{key} failed: {error}", flush=True)
                write_manifest(case/"manifest.json", case_manifest)
                (root/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
                write_report(root, summary)
        return postprocess(root, summary, manifest)
    except Exception as error:
        if isinstance(error, CommandExecutionError):
            records.append(error.record)
        summary.update(status="failed", error=str(error), finished_at=utc_now())
        manifest.update(status="failed", finished_at=summary["finished_at"], error=str(error))
        (root/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        write_manifest(root/"manifest.json", manifest)
        write_report(root, summary)
        print(f"Streaming validation failed: {error}", flush=True)
        print(root, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
