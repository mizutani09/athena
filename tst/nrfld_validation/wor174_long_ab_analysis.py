#!/usr/bin/env python3
"""Audit the clean-source WOR-174 mature-state false/true solar A/B run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
from pathlib import Path

import numpy as np


FSUN = 6.3e10
AREA = (6.144e8) ** 2
TIME_UNIT = 22.360679774997898
GRAVITY_FACTOR = ((1.0e-7 * 5.0e7 / TIME_UNIT)
                  * (2.74e4 * 3.072 * 5.0e7) / FSUN)
WINDOWS = (100, 500, 1000)

COL = {
    "time": 0, "dt": 1, "mass": 2, "all_energy": 16,
    "vz_rms": 17, "frad_top": 19, "vz_rms2_mean": 21,
    "top_mdot_out": 23, "mass_drift": 24, "pbnd_scale": 25,
    "bottom_mdot_up": 27, "bottom_mdot_down": 28,
    "etot_grav": 31, "fhyd_bottom": 32, "frad_bottom": 33,
    "fhyd_top": 34, "ftot_top": 35,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def only_file(directory: Path, pattern: str) -> Path:
    found = sorted(directory.glob(pattern))
    if len(found) != 1:
        raise ValueError(f"Expected one {pattern} in {directory}, got {found}")
    return found[0]


def key_values(line: str) -> dict[str, str]:
    return dict(re.findall(r"([A-Za-z0-9_]+)=([^ ]+)", line))


def parse_solver_log(path: Path) -> dict:
    nr = []
    linear = []
    opacity = None
    text = path.read_text(errors="replace")
    for line in text.splitlines():
        if line.startswith("[NR] status="):
            nr.append(key_values(line))
        elif line.startswith("[LINEAR_FGMRES] "):
            linear.append(key_values(line))
        elif line.startswith("FLD_OPACITY_DIAGNOSTICS "):
            opacity = key_values(line)
    nr_statuses = [row.get("status") for row in nr]
    fgmres_iterations = np.asarray(
        [int(row["iterations"]) for row in linear], dtype=int)
    final_l2 = np.asarray(
        [float(row["final_l2"]) for row in nr], dtype=float)
    return {
        "newton_solves": len(nr),
        "newton_converged": nr_statuses.count("converged"),
        "newton_iterations_total": sum(int(row["iterations"]) for row in nr),
        "newton_final_l2_max": float(final_l2.max()) if len(final_l2) else None,
        "fgmres_solves": len(linear),
        "fgmres_iterations_total": int(fgmres_iterations.sum()),
        "fgmres_iterations_mean": (float(fgmres_iterations.mean())
                                    if len(fgmres_iterations) else None),
        "fgmres_iterations_max": (int(fgmres_iterations.max())
                                   if len(fgmres_iterations) else None),
        "fgmres_breakdowns": sum(int(row.get("breakdown", "0")) for row in linear),
        "fgmres_inexact_accepts": sum(
            int(row.get("inexact_accept", "0")) for row in linear),
        "opacity_diagnostics": opacity,
        "fatal_error": "FATAL ERROR" in text,
        "terminated_on_time_limit": "Terminating on time limit" in text,
    }


def audit_window(history: np.ndarray, width: int) -> dict | None:
    end = float(history[-1, COL["time"]])
    rows = history[history[:, COL["time"]] >= end - width]
    span = float(rows[-1, COL["time"]] - rows[0, COL["time"]])
    if len(rows) < 10 or span < 0.95 * width:
        return None
    top_rad = float(np.mean(rows[:, COL["frad_top"]] / FSUN))
    top_gas = float(np.mean(rows[:, COL["fhyd_top"]] / FSUN))
    top_grav = float(np.mean(rows[:, COL["top_mdot_out"]] * GRAVITY_FACTOR))
    bottom_gas = float(np.mean(rows[:, COL["fhyd_bottom"]] / FSUN))
    bottom_rad = float(np.mean(rows[:, COL["frad_bottom"]] / FSUN))
    bottom_grav = float(np.mean(
        -(rows[:, COL["bottom_mdot_up"]] + rows[:, COL["bottom_mdot_down"]])
        * GRAVITY_FACTOR))
    storage = float(np.polyfit(rows[:, COL["time"]],
                               rows[:, COL["etot_grav"]], 1)[0]
                    / (AREA * FSUN * TIME_UNIT))
    bottom = bottom_gas + bottom_rad + bottom_grav
    top = top_rad + top_gas + top_grav
    return {
        "start_time": float(rows[0, COL["time"]]),
        "end_time": end,
        "span": span,
        "samples": int(len(rows)),
        "top_radiation": top_rad,
        "top_gas": top_gas,
        "top_gravity": top_grav,
        "top_total": top,
        "bottom_gas": bottom_gas,
        "bottom_radiation": bottom_rad,
        "bottom_gravity": bottom_grav,
        "bottom_total": bottom,
        "storage": storage,
        "closure_error": bottom - top - storage,
        "mass_drift_final": float(rows[-1, COL["mass_drift"]]),
        "pbnd_scale_mean": float(np.mean(rows[:, COL["pbnd_scale"]])),
        "bottom_mdot_up_mean": float(np.mean(rows[:, COL["bottom_mdot_up"]])),
        "bottom_mdot_down_mean": float(np.mean(rows[:, COL["bottom_mdot_down"]])),
        "top_mdot_out_mean": float(np.mean(rows[:, COL["top_mdot_out"]])),
        "dt_median": float(np.median(rows[:, COL["dt"]])),
        "vz_rms_mean": float(np.mean(rows[:, COL["vz_rms"]])),
        "vz_rms_time_mean": float(np.mean(
            np.sqrt(np.maximum(rows[:, COL["vz_rms2_mean"]], 0.0)))),
    }


def audit_mode(directory: Path) -> tuple[dict, np.ndarray]:
    result = (directory / "result.txt").read_text()
    if "validated=yes\n" not in result:
        raise ValueError(f"No validated final restart in {directory}")
    history_path = only_file(directory, "*.hst")
    history = np.loadtxt(history_path, comments="#", ndmin=2)
    if history.shape[1] != 36:
        raise ValueError(f"Expected clean-source 36-column history, got {history.shape}")
    if not np.all(np.isfinite(history)):
        raise ValueError(f"Non-finite history in {history_path}")
    log_path = directory / "run.log"
    solver = parse_solver_log(log_path)
    if (solver["fatal_error"] or not solver["terminated_on_time_limit"]
            or solver["newton_solves"] != solver["newton_converged"]
            or solver["fgmres_breakdowns"]):
        raise ValueError(f"Solver validation failed in {directory}: {solver}")
    return ({
        "directory": str(directory.resolve()),
        "history_file": history_path.name,
        "history_sha256": sha256(history_path),
        "history_rows": int(history.shape[0]),
        "start_time": float(history[0, COL["time"]]),
        "end_time": float(history[-1, COL["time"]]),
        "solver": solver,
        "windows": {str(width): audit_window(history, width) for width in WINDOWS},
    }, history)


def relative_change(enabled: float, baseline: float) -> float:
    return (enabled - baseline) / baseline if baseline != 0.0 else math.nan


def compare_modes(false: dict, true: dict) -> dict:
    windows = {}
    for width in WINDOWS:
        key = str(width)
        left, right = false["windows"][key], true["windows"][key]
        if left is None or right is None:
            windows[key] = None
            continue
        windows[key] = {
            "top_radiation_relative_change": relative_change(
                right["top_radiation"], left["top_radiation"]),
            "top_total_relative_change": relative_change(
                right["top_total"], left["top_total"]),
            "bottom_total_relative_change": relative_change(
                right["bottom_total"], left["bottom_total"]),
            "storage_difference": right["storage"] - left["storage"],
            "closure_error_difference": (right["closure_error"]
                                         - left["closure_error"]),
            "mass_drift_difference": (right["mass_drift_final"]
                                      - left["mass_drift_final"]),
            "dt_median_relative_change": relative_change(
                right["dt_median"], left["dt_median"]),
            "vz_rms_time_mean_relative_change": relative_change(
                right["vz_rms_time_mean"], left["vz_rms_time_mean"]),
        }
    fs = false["solver"]
    ts = true["solver"]
    return {
        "windows": windows,
        "newton_iterations_relative_change": relative_change(
            ts["newton_iterations_total"], fs["newton_iterations_total"]),
        "fgmres_iterations_relative_change": relative_change(
            ts["fgmres_iterations_total"], fs["fgmres_iterations_total"]),
    }


def plot_histories(histories: dict[str, np.ndarray], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for mode, data in histories.items():
        t = data[:, COL["time"]]
        axes[0, 0].plot(t, data[:, COL["frad_top"]] / FSUN, label=mode)
        axes[0, 1].plot(t, data[:, COL["ftot_top"]] / FSUN, label=mode)
        axes[1, 0].plot(t, data[:, COL["dt"]], label=mode)
        axes[1, 1].plot(t, data[:, COL["mass_drift"]], label=mode)
    axes[0, 0].set_ylabel("top radiative flux / Fsun")
    axes[0, 1].set_ylabel("recorded top total flux / Fsun")
    axes[1, 0].set_ylabel("dt (code)")
    axes[1, 1].set_ylabel("mass drift")
    axes[1, 0].set_xlabel("time (code)")
    axes[1, 1].set_xlabel("time (code)")
    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("WOR-174 clean-source mature same-restart A/B")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def write_csv(summary: dict, output: Path) -> None:
    fields = ["mode", "window", "samples", "top_radiation", "top_total",
              "bottom_total", "storage", "closure_error", "mass_drift_final",
              "dt_median", "vz_rms_time_mean"]
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for mode in ("false", "true"):
            for width in WINDOWS:
                row = summary["modes"][mode]["windows"][str(width)]
                if row is not None:
                    writer.writerow({"mode": mode, "window": width,
                                     **{key: row[key] for key in fields[2:]}})


def write_html(summary: dict, output: Path) -> None:
    rows = []
    for width in WINDOWS:
        key = str(width)
        for mode in ("false", "true"):
            row = summary["modes"][mode]["windows"][key]
            if row is None:
                continue
            rows.append("<tr>" + "".join((
                f"<td>{mode}</td><td>{width}</td><td>{row['samples']}</td>",
                f"<td>{row['top_radiation']:.6f}</td>",
                f"<td>{row['top_total']:.6f}</td>",
                f"<td>{row['bottom_total']:.6f}</td>",
                f"<td>{row['storage']:+.6f}</td>",
                f"<td>{row['closure_error']:+.3e}</td>",
                f"<td>{row['dt_median']:.6g}</td>",
            )) + "</tr>")
    comp_rows = []
    for width in WINDOWS:
        value = summary["comparison"]["windows"][str(width)]
        if value is not None:
            comp_rows.append(
                f"<tr><td>{width}</td><td>{100*value['top_radiation_relative_change']:+.4f}%</td>"
                f"<td>{100*value['top_total_relative_change']:+.4f}%</td>"
                f"<td>{100*value['dt_median_relative_change']:+.4f}%</td>"
                f"<td>{100*value['vz_rms_time_mean_relative_change']:+.4f}%</td></tr>")
    solver_rows = []
    for mode in ("false", "true"):
        s = summary["modes"][mode]["solver"]
        solver_rows.append(
            f"<tr><td>{mode}</td><td>{s['newton_solves']}</td>"
            f"<td>{s['newton_iterations_total']}</td><td>{s['newton_final_l2_max']:.3e}</td>"
            f"<td>{s['fgmres_solves']}</td><td>{s['fgmres_iterations_total']}</td>"
            f"<td>{s['fgmres_iterations_mean']:.3f}</td><td>{s['fgmres_iterations_max']}</td></tr>")
    output.write_text(f"""<!doctype html><html lang='en'><meta charset='utf-8'>
<title>WOR-174 mature clean-source A/B</title><style>
body{{font:17px/1.5 system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#172b3a}}
table{{border-collapse:collapse;font-size:14px}}th,td{{padding:.4rem .55rem;border-bottom:1px solid #ddd;text-align:right}}
th:first-child,td:first-child{{text-align:left}}img{{max-width:100%}}code{{background:#f3f6f8;padding:.1rem .3rem}}
</style><h1>WOR-174: mature-state clean-source same-restart A/B</h1>
<p><b>Validation:</b> {html.escape(summary['validation'])}. The option remains default-off.</p>
<p>The clean 36-column source does not record the two historical advective-radiation boundary columns;
the table therefore compares like-for-like clean-source runs and does not silently substitute columns.</p>
<p><a href='summary.json'>JSON</a> · <a href='windows.csv'>CSV</a> · <a href='histories.png'>PNG</a></p>
<img src='histories.png' alt='A/B histories'>
<h2>Energy and state windows</h2><table><tr><th>mode</th><th>window</th><th>samples</th>
<th>top rad/Fsun</th><th>top total/Fsun</th><th>bottom total/Fsun</th><th>storage/Fsun</th>
<th>closure</th><th>median dt</th></tr>{''.join(rows)}</table>
<h2>Enabled relative to disabled</h2><table><tr><th>window</th><th>top rad</th>
<th>top total</th><th>median dt</th><th>Vz RMS</th></tr>{''.join(comp_rows)}</table>
<h2>Nonlinear and linear solves</h2><table><tr><th>mode</th><th>Newton solves</th>
<th>Newton iters</th><th>max final L2</th><th>FGMRES solves</th><th>FGMRES iters</th>
<th>FGMRES mean</th><th>FGMRES max</th></tr>{''.join(solver_rows)}</table>
<h2>Provenance</h2><pre>{html.escape(json.dumps(summary['provenance'], indent=2))}</pre></html>""")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Directory containing false/ and true/")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--job-id", default="unknown")
    args = parser.parse_args()
    run = args.run.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    modes = {}
    histories = {}
    for mode in ("false", "true"):
        modes[mode], histories[mode] = audit_mode(run / mode)
    if abs(modes["false"]["end_time"] - modes["true"]["end_time"]) > 1e-10:
        raise ValueError("A/B end times differ")
    summary = {
        "validation": "PASS",
        "job_id": args.job_id,
        "modes": modes,
        "comparison": compare_modes(modes["false"], modes["true"]),
        "provenance": {
            "run": str(run),
            "input_sha256": (sha256(run / "input.used")
                              if (run / "input.used").is_file() else None),
            "inputs_sha256_text": ((run / "inputs.sha256").read_text()
                                   if (run / "inputs.sha256").is_file() else None),
            "analyzer": str(Path(__file__).resolve()),
            "history_layout": "clean-source 36-column simple_convection history",
            "constants": {"fsun": FSUN, "area": AREA, "time_unit": TIME_UNIT,
                          "gravity_factor": GRAVITY_FACTOR},
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(summary, output / "windows.csv")
    plot_histories(histories, output / "histories.png")
    write_html(summary, output / "index.html")
    print(json.dumps(summary["comparison"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
