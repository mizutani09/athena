#!/usr/bin/env python3
"""Run simple and fixed-step NR-FLD coupling tests with one HTML report."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
from html import escape
import json
import os
from pathlib import Path
import re
import sys
import traceback
from urllib.parse import quote

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nrfld-validation")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from nrfld_validation.io import find_repo_root, read_history  # noqa: E402
from nrfld_validation.report_style import REPORT_TYPOGRAPHY  # noqa: E402
from nrfld_validation.manifest import (base_manifest, input_file_references,
                                       sha256_file, utc_now, write_manifest)  # noqa: E402
from plots.plot_couple import (coupling_analytic_at, coupling_equilibrium,
                               plot_coupling)  # noqa: E402
from run_suite import (CommandExecutionError, analyse_case, binary_configuration,
                       build_case, checked, effective_settings,
                       validate_binary_configuration)  # noqa: E402

DT_VALUES = (1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4)
INITIAL_GAS = {"low": 1.0e2, "high": 1.0e10}
SIMPLE_SETTINGS = {"low": (1.0e-11, 1.05, 100),
                   "high": (1.0e-17, 1.01, 1800)}
RELATIVE_TOLERANCE = 5.0e-3
TIME_UNIT_S = 1.0e-7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/d/output/rad"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-cores", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dt-values", type=float, nargs="+", default=DT_VALUES)
    parser.add_argument("--linear-solver", choices=("fgmres", "mg"), default="fgmres")
    parser.add_argument("--build-timeout", type=float, default=900)
    parser.add_argument("--run-timeout", type=float, default=300)
    args = parser.parse_args()
    if (args.max_cores < 1 or args.steps < 4 or args.build_timeout <= 0
            or args.run_timeout <= 0 or any(dt <= 0 for dt in args.dt_values)
            or len(set(args.dt_values)) != len(args.dt_values)):
        parser.error("positive distinct dt values, >=4 steps, and positive limits required")
    return args


def dt_tag(dt: float) -> str:
    mantissa, exponent = f"{dt:.15e}".split("e")
    return f"dt_{mantissa.rstrip('0').rstrip('.')}e{int(exponent):+d}"


def materialize_input(source: Path, target: Path, changes: dict[str, str]) -> None:
    """Write all effective overrides into a readable Athena input file."""
    grouped: dict[str, dict[str, str]] = {}
    for full_key, value in changes.items():
        block, key = full_key.split("/", 1)
        grouped.setdefault(block, {})[key] = value
    seen: set[str] = set()
    section = None
    lines: list[str] = []
    for original in source.read_text(encoding="utf-8").splitlines():
        block_match = re.fullmatch(r"<([^>]+)>", original.strip())
        if block_match:
            if section in grouped:
                for key, value in grouped[section].items():
                    if f"{section}/{key}" not in seen:
                        lines.append(f"{key} = {value}")
                        seen.add(f"{section}/{key}")
            section = block_match.group(1)
            lines.append(original)
            continue
        key_match = re.match(r"^\s*([A-Za-z_][\w]*)\s*=", original)
        if section in grouped and key_match and key_match.group(1) in grouped[section]:
            key = key_match.group(1)
            lines.append(f"{key} = {grouped[section][key]}")
            seen.add(f"{section}/{key}")
        else:
            lines.append(original)
    if section in grouped:
        for key, value in grouped[section].items():
            if f"{section}/{key}" not in seen:
                lines.append(f"{key} = {value}")
                seen.add(f"{section}/{key}")
    for block, values in grouped.items():
        missing = [(key, value) for key, value in values.items()
                   if f"{block}/{key}" not in seen]
        if missing:
            lines.append(f"<{block}>")
            lines.extend(f"{key} = {value}" for key, value in missing)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def coupling_timescales(initial: float) -> tuple[float, float]:
    a_rad, r_gas, c_light = 7.5657e-15, 8.31451e7, 2.99792458e10
    rho, mu, gamma, erad, sigma = 1.0e-7, 0.6, 5.0/3.0, 1.0e12, 4.0e-8
    temp0 = (gamma - 1.0)*mu*initial/(rho*r_gas)
    source0 = c_light*sigma*abs(erad - a_rad*temp0**4)
    initial_source_time = initial/source0
    equilibrium = coupling_equilibrium()
    equilibrium_relaxation_time = equilibrium/(4.0*c_light*sigma*erad)
    return initial_source_time, equilibrium_relaxation_time


def unique_history(output_dir: Path) -> dict[str, np.ndarray]:
    history = read_history(output_dir)
    _, indices = np.unique(history["Rtime"], return_index=True)
    indices.sort()
    return {name: values[indices] for name, values in history.items()}


def analyse(output_dir: Path, log_path: Path, state: str, requested_dt: float,
            steps: int, solver: str) -> tuple[dict, dict[str, np.ndarray]]:
    history = unique_history(output_dir)
    times = history["Rtime"]
    gas = history["e_gas"]
    radiation = history["E_rad"]
    if len(times) != steps + 1 or times[0] != 0.0 or np.any(np.diff(times) <= 0):
        raise ValueError(f"expected initial row and {steps} distinct evolved samples")
    if not all(np.all(np.isfinite(values)) for values in history.values()):
        raise ValueError("nonfinite history data")
    if np.any(gas <= 0) or np.any(radiation <= 0):
        raise ValueError("nonpositive gas or radiation energy")
    measured_dt = np.diff(times)
    max_dt_relative_error = float(np.max(np.abs(measured_dt/requested_dt - 1.0)))
    reference = coupling_analytic_at(times[1:], state)
    analytic_error = np.abs(gas[1:]/reference - 1.0)
    equilibrium = coupling_equilibrium()
    equilibrium_error = np.abs(gas[1:]/equilibrium - 1.0)
    initial_source_time, relaxation_time = coupling_timescales(INITIAL_GAS[state])
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    nr_rows = [dict(re.findall(r"(\w+)=([^\s]+)", line))
               for line in log_text.splitlines() if line.startswith("[NR] status=")]
    converged = [row for row in nr_rows if row.get("status") == "converged"]
    nr_ok = (len(converged) == steps and len(nr_rows) == steps + 1
             and nr_rows[0].get("status") == "dt_zero_initialization"
             and all(float(row["final_max"]) <= 1.0e-10
                     and float(row["final_l2"]) <= 1.0e-10 for row in converged))
    krylov_rows = [dict(re.findall(r"(\w+)=([^\s]+)", line))
                   for line in log_text.splitlines() if line.startswith("[LINEAR_FGMRES]")]
    krylov_ok = (bool(krylov_rows) and all(
        float(row["relative_residual"]) <= float(row["target"])
        for row in krylov_rows)) if solver == "fgmres" else not krylov_rows
    first_sustained_step = next((index + 1 for index in range(steps)
                                 if np.all(analytic_error[index:] <= RELATIVE_TOLERANCE)), None)
    tail_max = float(np.max(analytic_error[-3:]))
    radiation_drift = float(np.max(np.abs(radiation/1.0e12 - 1.0)))
    metrics = {
        "requested_dt_s": requested_dt,
        "measured_first_dt_s": float(measured_dt[0]),
        "max_dt_relative_error": max_dt_relative_error,
        "dt_over_initial_source_time": float(measured_dt[0]/initial_source_time),
        "dt_over_equilibrium_relaxation_time": float(measured_dt[0]/relaxation_time),
        "initial_source_time_s": initial_source_time,
        "equilibrium_relaxation_time_s": relaxation_time,
        "first_step_relative_analytic_error": float(analytic_error[0]),
        "max_post_first_relative_analytic_error": float(np.max(analytic_error[1:])),
        "tail_3_max_relative_analytic_error": tail_max,
        "first_sustained_step_within_tolerance": first_sustained_step,
        "final_relative_equilibrium_error": float(equilibrium_error[-1]),
        "max_relative_radiation_drift": radiation_drift,
        "nr_converged_steps": len(converged),
        "fgmres_calls": len(krylov_rows),
        "nr_all_converged": nr_ok,
        "fgmres_all_converged": krylov_ok,
        "passed": bool(max_dt_relative_error < 1.0e-4 and radiation_drift < 1.0e-10
                       and tail_max < RELATIVE_TOLERANCE and nr_ok and krylov_ok),
    }
    return metrics, history


def run_simple_cases(root: Path, repo: Path, binary: Path, build: dict,
                     configuration: dict, args: argparse.Namespace,
                     summary: dict) -> None:
    """Run the resolved, growing-dt couple tests before the stiff sweep."""
    source_input = repo / "inputs/radiation/athinput.nrfld_couple"
    for state, (dt0, growth, steps) in SIMPLE_SETTINGS.items():
        case_dir = root / "simple" / state
        case_dir.mkdir(parents=True)
        manifest = base_manifest(repo, f"simple/{state}", args.max_cores)
        manifest["provenance"]["analysis"]["files"] = {
            path.relative_to(repo).as_posix(): sha256_file(path)
            for path in (Path(__file__), SCRIPT_DIR / "plots/plot_couple.py",
                         SCRIPT_DIR / "run_suite.py")}
        manifest["provenance"]["calculation"] = {
            "source": "shared_isolated_build", "build_source": str(build["build_source"]),
            "source_sha256": build["build_source_sha256"],
            "binary_sha256": sha256_file(binary), "binary_configuration": configuration}
        changes = {
            "job/problem_id": f"nrfld_couple_simple_{state}",
            "time/nlim": str(steps), "time/ncycle_out": str(steps),
            "problem/eg_0": f"{INITIAL_GAS[state]:.17g}",
            "problem/dt0": f"{dt0:.17g}",
            "problem/dt_growth": f"{growth:.17g}",
            "nrfld/linear_solver": args.linear_solver,
        }
        input_path = case_dir / "input.used"
        materialize_input(source_input, input_path, changes)
        manifest["input"] = {
            "source_path": str(source_input), "source_sha256": sha256_file(source_input),
            "used_path": str(input_path), "used_sha256": sha256_file(input_path),
            "effective_settings": effective_settings(input_path, []),
            "external_files": input_file_references(input_path, repo)}
        manifest["resources"] = {
            "mpi_ranks": 1, "omp_threads": int(os.environ.get("OMP_NUM_THREADS", "1")),
            "max_cores": args.max_cores}
        output_dir = case_dir / "output"
        output_dir.mkdir()
        resources = case_dir / "resources.txt"
        command = ["/usr/bin/time", "-v", "-o", str(resources),
                   str(binary), "-i", str(input_path), "-d", str(output_dir)]
        records: list[dict] = []
        manifest["commands"] = {"run": command, "records": records}
        write_manifest(case_dir / "manifest.json", manifest)
        print(f"Running simple/{state}: dt0={dt0:.0e} s, growth={growth:g}", flush=True)
        try:
            records.append(checked(command, case_dir, case_dir / "run.log",
                                   args.run_timeout, "run"))
            metrics = analyse_case(f"couple_{state}", output_dir,
                                   case_dir / "run.log", steps, args.linear_solver)
            history = unique_history(output_dir)
            times = history["Rtime"]
            measured_dt = np.diff(times)
            _, relaxation_time = coupling_timescales(INITIAL_GAS[state])
            metrics.update({
                "dt0_s": dt0, "dt_growth": growth, "steps": steps,
                "measured_first_dt_s": float(measured_dt[0]),
                "measured_max_dt_s": float(np.max(measured_dt)),
                "max_dt_over_equilibrium_relaxation_time":
                    float(np.max(measured_dt)/relaxation_time),
                "equilibrium_relaxation_time_s": relaxation_time,
            })
            metrics["passed"] = bool(metrics["passed"]
                                     and len(times) == steps + 1
                                     and np.all(measured_dt > 0)
                                     and np.isclose(measured_dt[0], dt0,
                                                    rtol=2.0e-10, atol=0)
                                     and metrics["max_dt_over_equilibrium_relaxation_time"] < 1)
            figure = case_dir / "figures" / f"couple_{state}.png"
            plot_coupling(output_dir, figure, state)
            match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)",
                              resources.read_text(encoding="utf-8"))
            manifest["resources"]["max_rss_kib"] = int(match.group(1)) if match else None
            manifest["artifacts"] = {"plot": str(figure), "output": str(output_dir)}
            manifest["metrics"] = metrics
            manifest["status"] = "passed" if metrics["passed"] else "failed"
            print(f"  max log error={metrics['max_log_egas_error']:.4g}, "
                  f"max dt/t_couple={metrics['max_dt_over_equilibrium_relaxation_time']:.4g}, "
                  f"status={manifest['status']}", flush=True)
        except Exception as error:
            if isinstance(error, CommandExecutionError):
                records.append(error.record)
            manifest["status"] = "failed"
            manifest["error"] = {"message": str(error), "traceback": traceback.format_exc()}
            print(f"  failed: {error}", flush=True)
        finally:
            manifest["finished_at"] = utc_now()
            write_manifest(case_dir / "manifest.json", manifest)
            summary["simple"][state] = {"status": manifest["status"],
                                         "metrics": manifest.get("metrics", {}),
                                         "manifest": str(case_dir / "manifest.json")}
            (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            report(root, summary)


def plot_results(root: Path, runs: dict[tuple[str, float], dict[str, np.ndarray]],
                 summary: dict) -> list[str]:
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    available_dts = sorted({dt for _, dt in runs})
    colors = {dt: color for dt, color in zip(
        available_dts, plt.cm.viridis(np.linspace(0.12, 0.88, len(available_dts))))}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    for axis, state in zip(axes, INITIAL_GAS):
        curve_time = np.geomspace(1.0e-9, 1.0e-3, 3000)
        axis.loglog(curve_time, coupling_analytic_at(curve_time, state),
                    "k-", lw=1.7, label="analytic")
        for dt in available_dts:
            history = runs.get((state, dt))
            if history is None:
                continue
            axis.loglog(history["Rtime"][1:], history["e_gas"][1:],
                        "-o", ms=4, lw=1, color=colors[dt],
                        label=rf"$\Delta t={dt:.0e}\,$s")
        axis.set_title(f"{state}: $e_{{gas}}(0)={INITIAL_GAS[state]:.0e}$")
        axis.set_xlabel(r"$t\;[\mathrm{s}]$")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=9)
    axes[0].set_ylabel(r"$e_{\rm gas}\;[\mathrm{erg\,cm^{-3}}]$")
    fig.tight_layout()
    fig.savefig(figures / "couple_dt_sweep.png", dpi=180)
    fig.savefig(figures / "couple_dt_sweep.pdf")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 5.2))
    for state in INITIAL_GAS:
        dt_values = [dt for dt in available_dts if (state, dt) in runs]
        errors = [summary["cases"][f"{state}/{dt_tag(dt)}"]["metrics"]
                  ["first_step_relative_analytic_error"] for dt in dt_values]
        if dt_values:
            axis.loglog(dt_values, errors, "-o", lw=1.5, label=state)
    axis.set_xlabel(r"first-step $\Delta t\;[\mathrm{s}]$")
    axis.set_ylabel(r"$|e_{\rm num}-e_{\rm exact}|/e_{\rm exact}$")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "couple_first_step_error.png", dpi=180)
    fig.savefig(figures / "couple_first_step_error.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    for axis, state in zip(axes, INITIAL_GAS):
        for dt in available_dts:
            history = runs.get((state, dt))
            if history is None:
                continue
            times = history["Rtime"][1:]
            error = np.abs(history["e_gas"][1:]
                           / coupling_analytic_at(times, state) - 1.0)
            axis.loglog(times, np.maximum(error, 1.0e-16), "-o", ms=4, lw=1,
                        color=colors[dt], label=rf"$\Delta t={dt:.0e}\,$s")
            axis.plot(times[0], max(error[0], 1.0e-16), marker="o", ms=7,
                      mfc="white", mec=colors[dt])
        axis.axhline(RELATIVE_TOLERANCE, color="black", ls="--", lw=1,
                     label="0.5% criterion")
        axis.set_title(state)
        axis.set_xlabel(r"$t\;[\mathrm{s}]$")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("relative error against analytic solution")
    fig.tight_layout()
    fig.savefig(figures / "couple_error_by_step.png", dpi=180)
    fig.savefig(figures / "couple_error_by_step.pdf")
    plt.close(fig)
    return ["couple_dt_sweep", "couple_first_step_error", "couple_error_by_step"]


def report(root: Path, summary: dict) -> None:
    def number(metrics: dict, name: str, fmt: str = ".3g") -> str:
        value = metrics.get(name)
        return format(value, fmt) if isinstance(value, (int, float)) else "—"

    def status_badge(status: str) -> str:
        label = {"passed": "PASS", "failed": "FAIL"}.get(status, status)
        css_class = "pass" if status == "passed" else "fail" if status == "failed" else "pending"
        return f'<span class="badge {css_class}">{escape(label)}</span>'

    simple_rows = []
    simple_figures = []
    for state, item in summary.get("simple", {}).items():
        metrics = item.get("metrics", {})
        prefix = f"simple/{state}/"
        simple_rows.append(
            f'<tr><th scope="row">{escape(state)}</th><td>{status_badge(item["status"])}</td>'
            f'<td>{number(metrics, "measured_first_dt_s")}</td>'
            f'<td>{number(metrics, "measured_max_dt_s")}</td>'
            f'<td><strong>{number(metrics, "max_dt_over_equilibrium_relaxation_time")}</strong></td>'
            f'<td><strong>{number(metrics, "max_log_egas_error")}</strong></td>'
            f'<td>{number(metrics, "nr_converged_steps", "g")}/{number(metrics, "nr_expected_steps", "g")}</td>'
            f'<td><a href="{prefix}manifest.json">manifest</a> · '
            f'<a href="{prefix}run.log">log</a> · '
            f'<a href="{prefix}input.used">input</a></td></tr>'
        )
        figure = root / prefix / "figures" / f"couple_{state}.png"
        if figure.is_file():
            simple_figures.append(
                f'<figure><img src="{prefix}figures/couple_{state}.png" '
                f'alt="{escape(state)}: numerical and analytic solutions" loading="lazy">'
                f'<figcaption>{escape(state)}: gas energy and analytic reference, with relative error</figcaption></figure>'
            )
    sweep_rows = []
    for key, item in summary.get("cases", {}).items():
        metrics = item.get("metrics", {})
        prefix = "cases/" + key + "/"
        first_error = metrics.get("first_step_relative_analytic_error")
        first_class = "attention" if isinstance(first_error, (int, float)) and first_error >= RELATIVE_TOLERANCE else ""
        sweep_rows.append(
            f'<tr><th scope="row">{escape(key)}</th><td>{status_badge(item["status"])}</td>'
            f'<td>{number(metrics, "measured_first_dt_s")}</td>'
            f'<td>{number(metrics, "dt_over_equilibrium_relaxation_time")}</td>'
            f'<td class="{first_class}">{number(metrics, "first_step_relative_analytic_error")}</td>'
            f'<td>{number(metrics, "max_post_first_relative_analytic_error")}</td>'
            f'<td><strong>{number(metrics, "tail_3_max_relative_analytic_error")}</strong></td>'
            f'<td>{escape(str(metrics.get("first_sustained_step_within_tolerance") or "—"))}</td>'
            f'<td><a href="{quote(prefix)}manifest.json">manifest</a> · '
            f'<a href="{quote(prefix)}run.log">log</a> · '
            f'<a href="{quote(prefix)}input.used">input</a></td></tr>'
        )
    captions = {"couple_dt_sweep": "Gas energy evolution and analytic solution",
                "couple_first_step_error": "Relative error after the first step",
                "couple_error_by_step": "Relative error at each step"}
    sweep_figures = "".join(
        f'<figure><img src="figures/{name}.png" alt="{escape(captions.get(name, name))}" loading="lazy">'
        f'<figcaption>{escape(captions.get(name, name))}</figcaption></figure>'
        for name in summary.get("figures", [])
    )
    status = summary.get("status", "running")
    simple_passed = sum(item["status"] == "passed" for item in summary.get("simple", {}).values())
    sweep_passed = sum(item["status"] == "passed" for item in summary.get("cases", {}).values())
    tcouple = coupling_timescales(INITIAL_GAS["low"])[1]
    root.joinpath("index.html").write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>NR-FLD coupling validation — {escape(summary['run'])}</title>
<style>
:root {{font-family:system-ui,sans-serif;font-size:18px;line-height:1.65;color:#17212b;background:#f7f9fb}}
body {{max-width:1450px;margin:2rem auto;padding:0 1.2rem}}
h1,h2,h3 {{line-height:1.3;margin-top:0}} h1 {{font-size:2.1rem}} h2 {{font-size:1.55rem}} h3 {{font-size:1.2rem}}
section {{background:#fff;border:1px solid #dce4eb;border-radius:10px;padding:1.3rem;margin:1.5rem 0}}
.overview {{font-size:1.1rem;background:#eaf3fb;border-left:5px solid #1769aa;padding:.8rem 1rem;margin:1rem 0}}
.badge {{display:inline-block;border-radius:5px;padding:.1rem .55rem;font-weight:750}}
.pass {{color:#0b683e;background:#e4f5eb}} .fail {{color:#ad261f;background:#fdebea}}
.pending {{color:#41556b;background:#e8edf2}} .attention {{color:#9a5100;font-weight:700}}
.nowrap {{white-space:nowrap}}
.scroll {{overflow-x:auto}} table {{border-collapse:collapse;width:100%;font-size:.96rem}}
th,td {{border-bottom:1px solid #dce4eb;padding:.55rem .65rem;text-align:left;vertical-align:top;white-space:nowrap}}
thead th {{background:#eef3f7;font-weight:700}} tbody th {{font-weight:700}} tr:nth-child(even) {{background:#f8fafc}}
figure {{margin:1.2rem 0;padding:.6rem;background:#fff}} img {{max-width:100%;height:auto;display:block}}
figcaption {{font-size:1rem;font-weight:600;margin:.45rem 0}} a {{color:#075ca8}} code {{overflow-wrap:anywhere}}
@media(max-width:700px) {{:root {{font-size:17px}} body {{padding:0 .75rem}} section {{padding:.8rem}}}}
{REPORT_TYPOGRAPHY}</style></head><body>
<h1>NR-FLD coupling validation {status_badge(status)}</h1>
<p class="overview"><strong>Simple test: {simple_passed}/2 PASS</strong> &nbsp; <strong>Fixed-dt sweep: {sweep_passed}/{2*len(summary.get('dt_values', DT_VALUES))} PASS</strong><br>
Gas energy is compared with the analytic ODE for a fixed radiation bath. The equilibrium linear relaxation time is <strong class="nowrap">t_couple = {tcouple:.3g} s</strong>.</p>
<p>The simple test comes first, followed by the detailed sweep. Both check <strong>{escape(summary.get('linear_solver', 'fgmres'))}</strong> and Newton convergence and constancy of the radiation field. Inputs, logs and manifests have relative links.</p>
<section id="simple"><h2>1. Simple test: dt &lt; t_couple</h2>
<p>This repeats the earlier low/high test: start with a small timestep and grow it to follow the transient. Acceptance requires <strong>max(dt)/t_couple &lt; 1</strong>, maximum |ln(e_num/e_ref)| &lt; 0.005, and convergence at every step.</p>
<div class="scroll"><table><thead><tr><th>Case</th><th>Status</th><th>First dt [s]</th><th>Max dt [s]</th><th>Max dt/t_couple</th><th>Max log error</th><th>Newton steps</th><th>Files</th></tr></thead><tbody>{''.join(simple_rows)}</tbody></table></div>
{''.join(simple_figures)}</section>
<section id="sweep"><h2>2. Detailed test: fixed-dt sweep</h2>
<p>Low/high each use dt = 1e-7, 1e-6, 1e-5 and 1e-4 s for 10 steps. <strong>First-step error</strong> is shown separately from later relaxation. Acceptance requires relative analytic error &lt; 0.5% in the last three steps, the requested dt, convergence of every Newton/linear solve, and a constant radiation field.</p>
<div class="scroll"><table><thead><tr><th>Case</th><th>Status</th><th>First dt [s]</th><th>dt/t_couple</th><th>First error</th><th>Max later error</th><th>Last-3 max error</th><th>First sustained step &lt; 0.5%</th><th>Files</th></tr></thead><tbody>{''.join(sweep_rows)}</tbody></table></div>
{sweep_figures}</section>
<p><a href="summary.json">All results (JSON)</a> · <a href="first_step_errors.csv">Sweep errors (CSV)</a> · <a href="build/configure.log">Configure log</a> · <a href="build/build.log">Build log</a></p>
</body></html>\n""", encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo = find_repo_root(SCRIPT_DIR)
    now_jst = datetime.now(timezone(timedelta(hours=9)))
    run_name = args.run_name or f"validation_couple_{now_jst:%Y%m%d_%H%M%S}_JST"
    root = args.output_root.expanduser().resolve() / run_name
    root.mkdir(parents=True, exist_ok=False)
    summary: dict = {"run": run_name, "status": "running", "started_at": utc_now(),
                     "repository": str(repo), "simple": {}, "cases": {},
                     "dt_values": args.dt_values, "linear_solver": args.linear_solver,
                     "figures": [],
                     "relative_tolerance": RELATIVE_TOLERANCE}
    report(root, summary)
    build_dir = root / "build"
    build_dir.mkdir()
    build_records: list[dict] = []
    try:
        build = build_case(repo, build_dir, "nrfld_couple", args.max_cores,
                           args.build_timeout, build_records)
        binary = build["build_source"] / "bin" / "athena"
        configuration = binary_configuration(binary, build_dir, 30, build_records)
        validate_binary_configuration(configuration, "nrfld_couple", binary)
    except Exception as error:
        summary.update(status="build_failed", error=str(error), finished_at=utc_now())
        (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        report(root, summary)
        raise
    summary["build"] = {"source_sha256": build["build_source_sha256"],
                        "binary_sha256": sha256_file(binary),
                        "configuration": configuration, "records": build_records}
    run_simple_cases(root, repo, binary, build, configuration, args, summary)
    runs: dict[tuple[str, float], dict[str, np.ndarray]] = {}
    source_input = repo / "inputs/radiation/athinput.nrfld_couple"
    for state, initial in INITIAL_GAS.items():
        for dt in args.dt_values:
            tag = dt_tag(dt)
            case_dir = root / "cases" / state / tag
            case_dir.mkdir(parents=True)
            case_key = f"{state}/{tag}"
            manifest = base_manifest(repo, case_key, args.max_cores)
            manifest["provenance"]["analysis"]["files"] = {
                path.relative_to(repo).as_posix(): sha256_file(path)
                for path in (Path(__file__), SCRIPT_DIR / "plots/plot_couple.py")}
            manifest["provenance"]["calculation"] = {
                "source": "shared_isolated_build", "build_source": str(build["build_source"]),
                "source_sha256": build["build_source_sha256"],
                "binary_sha256": sha256_file(binary), "binary_configuration": configuration}
            changes = {
                "job/problem_id": f"nrfld_couple_{state}_{tag}",
                "time/nlim": str(args.steps), "time/ncycle_out": str(args.steps),
                "time/cfl_number": "1.0", "hydro/time_unit": f"{TIME_UNIT_S:.17g}",
                "problem/eg_0": f"{initial:.17g}", "problem/dt0": f"{dt:.17g}",
                "problem/dt_growth": "1.0", "nrfld/nr_backtrack_max": "20",
                "nrfld/nr_min_step_scale": "1e-6",
                "nrfld/linear_solver": args.linear_solver,
            }
            for axis in ("x1", "x2", "x3"):
                changes[f"mesh/{axis}min"] = "-5000"
                changes[f"mesh/{axis}max"] = "5000"
            input_path = case_dir / "input.used"
            materialize_input(source_input, input_path, changes)
            manifest["input"] = {
                "source_path": str(source_input), "source_sha256": sha256_file(source_input),
                "used_path": str(input_path), "used_sha256": sha256_file(input_path),
                "effective_settings": effective_settings(input_path, []),
                "external_files": input_file_references(input_path, repo)}
            manifest["resources"] = {"mpi_ranks": 1, "omp_threads": int(os.environ.get("OMP_NUM_THREADS", "1")),
                                     "max_cores": args.max_cores}
            output_dir = case_dir / "output"
            output_dir.mkdir()
            resource_path = case_dir / "resources.txt"
            command = ["/usr/bin/time", "-v", "-o", str(resource_path),
                       str(binary), "-i", str(input_path), "-d", str(output_dir)]
            records: list[dict] = []
            manifest["commands"] = {"run": command, "records": records}
            write_manifest(case_dir / "manifest.json", manifest)
            print(f"Running {case_key}: dt={dt:.0e} s", flush=True)
            try:
                record = checked(command, case_dir, case_dir / "run.log", args.run_timeout, "run")
                records.append(record)
                metrics, history = analyse(output_dir, case_dir / "run.log", state, dt,
                                           args.steps, args.linear_solver)
                runs[(state, dt)] = history
                match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)",
                                  resource_path.read_text(encoding="utf-8"))
                manifest["resources"]["max_rss_kib"] = int(match.group(1)) if match else None
                manifest["metrics"] = metrics
                manifest["status"] = "passed" if metrics["passed"] else "failed"
                print(f"  first error={metrics['first_step_relative_analytic_error']:.4g}, "
                      f"tail={metrics['tail_3_max_relative_analytic_error']:.4g}, "
                      f"status={manifest['status']}", flush=True)
            except Exception as error:
                if isinstance(error, CommandExecutionError):
                    records.append(error.record)
                manifest["status"] = "failed"
                manifest["error"] = {"message": str(error), "traceback": traceback.format_exc()}
                print(f"  failed: {error}", flush=True)
            finally:
                manifest["finished_at"] = utc_now()
                write_manifest(case_dir / "manifest.json", manifest)
                summary["cases"][case_key] = {"status": manifest["status"],
                                               "metrics": manifest.get("metrics", {}),
                                               "manifest": str(case_dir / "manifest.json")}
                (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                report(root, summary)
    summary["figures"] = plot_results(root, runs, summary) if runs else []
    rows = [{"case": key, **value["metrics"]} for key, value in summary["cases"].items()
            if value["metrics"]]
    if rows:
        with (root / "first_step_errors.csv").open("w", newline="", encoding="utf-8") as stream:
            keys = ["case", "requested_dt_s", "measured_first_dt_s",
                    "first_step_relative_analytic_error", "tail_3_max_relative_analytic_error",
                    "first_sustained_step_within_tolerance"]
            writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    all_results = [*summary["simple"].values(), *summary["cases"].values()]
    expected = len(SIMPLE_SETTINGS) + len(INITIAL_GAS)*len(args.dt_values)
    summary["status"] = ("passed" if len(all_results) == expected
                         and all(item["status"] == "passed" for item in all_results)
                         else "failed")
    summary["finished_at"] = utc_now()
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report(root, summary)
    print(f"HTML report: {root / 'index.html'}", flush=True)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
