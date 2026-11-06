#!/usr/bin/env python3
"""Screen stationarity across consecutive solar runs or entropy feedback segments."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

import numpy as np

from solar_convection_report import (input_values, make_campaign_index,
                                     make_energy_budget_plot, make_plot, parse_log,
                                     read_history, summarize)


def combined_log(logs: list[dict], manifests: list[dict]) -> dict:
    result = dict(logs[-1])
    result["completed"] = (logs[-1]["completed"] and all(
        log["completed"] or bool(manifest.get("interruption_reason"))
        for log, manifest in zip(logs, manifests)))
    for key in ("fatal_count", "slow_mg_warning_count", "divergent_vcycle_count",
                "fgmres_calls", "fgmres_iterations_total"):
        result[key] = sum(log[key] for log in logs)
    result["fgmres_iterations_max"] = max(log["fgmres_iterations_max"]
                                           for log in logs)
    statuses = {}
    for log in logs:
        for name, count in log["nr_status_counts"].items():
            statuses[name] = statuses.get(name, 0)+count
    result["nr_status_counts"] = statuses
    result["wall_time"] = None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append",
                        help="Consecutive run directory; documented interruptions "
                             "may be followed by their own checkpoint restart")
    parser.add_argument("--feedback-state", type=Path,
                        help="state.json from a segmented t=0 entropy-feedback run; "
                             "discovers completed cases relative to its own directory")
    parser.add_argument("--allow-s-in-variation", action="store_true",
                        help="Allow only the explicit problem/s_in override to vary "
                             "across manually listed run segments")
    parser.add_argument("--header-from", type=Path,
                        help="Header-bearing HST; defaults to the first run's HST")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-flux", type=float, default=6.3e10)
    args = parser.parse_args()
    if bool(args.run) == bool(args.feedback_state):
        parser.error("Choose either repeated --run or --feedback-state")
    if args.feedback_state:
        feedback_state = json.loads(args.feedback_state.read_text())
        args.run = [args.feedback_state.parent / row["case"]
                    for row in feedback_state["completed"]]
        args.allow_s_in_variation = True
    if len(args.run) < 2:
        parser.error("Need at least two consecutive run segments")
    args.run = [run.resolve() for run in args.run]
    if args.header_from is None:
        args.header_from = next(args.run[0].glob("*.hst"), None)
    if args.header_from is None or not args.header_from.is_file():
        parser.error("No header-bearing first HST; pass --header-from")
    args.header_from = args.header_from.resolve()
    manifests = [json.loads((run/"manifest.json").read_text())
                 for run in args.run]
    fixed = ("binary_sha256", "base_input_sha256",
             "generated_input_sha256", "data_sha256")
    numerical_prefixes = ("mesh/nx1=", "mesh/nx2=", "mesh/nx3=",
                          "time/cfl_number=")
    def numerical_overrides(manifest: dict) -> list[str]:
        return sorted(arg for arg in manifest["command"]
                      if arg.startswith(numerical_prefixes))
    def controlled_overrides(manifest: dict) -> list[str]:
        overrides = manifest.get("problem_overrides", [])
        if args.allow_s_in_variation:
            if not any(arg.startswith("s_in=") for arg in overrides):
                raise ValueError("Entropy feedback requires an explicit s_in override")
            overrides = [arg for arg in overrides if not arg.startswith("s_in=")]
        return sorted(overrides)
    for run, manifest in zip(args.run[1:], manifests[1:]):
        for key in fixed:
            if manifest.get(key) != manifests[0].get(key):
                raise ValueError(f"{run}: {key} differs from the first segment")
        if numerical_overrides(manifest) != numerical_overrides(manifests[0]):
            raise ValueError(f"{run}: grid or CFL differs from the first segment")
        if controlled_overrides(manifest) != controlled_overrides(manifests[0]):
            raise ValueError(f"{run}: non-entropy problem overrides differ")
    for previous, current, manifest in zip(args.run[:-1], args.run[1:],
                                           manifests[1:]):
        original_checkpoint = Path(manifest.get("restart") or "")
        checkpoint = previous / original_checkpoint.name
        if (original_checkpoint.parent.name != previous.name
                or not checkpoint.name.endswith(".rst")
                or not checkpoint.is_file()):
            raise ValueError(f"{current}: restart is not a checkpoint in the "
                             "previous segment")
        if ("interruption_reason" not in manifests[args.run.index(previous)]
                and not checkpoint.name.endswith(".final.rst")):
            raise ValueError(f"{current}: a completed parent must restart "
                             "from its final checkpoint")
    logs = [parse_log(run/"run.log") for run in args.run]
    for index, (run, log, manifest) in enumerate(zip(args.run, logs, manifests)):
        statuses = set(log["nr_status_counts"])
        if (log["fatal_count"] or not statuses
                or not statuses.issubset({"converged", "dt_zero_initialization"})):
            raise ValueError(f"{run}: fatal or nonconverged Newton solve")
        if not log["completed"]:
            if index == len(args.run)-1 or not manifest.get("interruption_reason"):
                raise ValueError(f"{run}: incomplete without a documented "
                                 "interruption and completed continuation")
            # A terminated runner records 143. If its launcher is killed too,
            # it cannot record an exit code; require the exact documented
            # checkpoint and successor instead of inventing an exit status.
            interrupted_runner = manifest.get("exit_code") == 143
            killed_launcher = (
                "exit_code" not in manifest
                and manifest.get("last_checkpoint") == manifests[index+1]["restart"]
                and Path(manifest.get("resume_case", "")).name
                == args.run[index+1].name)
            if not (interrupted_runner or killed_launcher):
                raise ValueError(f"{run}: incomplete without a verifiable "
                                 "checkpoint continuation")
    histories = [next(run.glob("*.hst")) for run in args.run]
    segments = [read_history(path, args.header_from) for path in histories]
    names = set(segments[0])
    if any(set(segment) != names for segment in segments[1:]):
        raise ValueError("History column sets differ between segments")
    h = segments[0]
    joins = []
    for current, manifest, previous_run in zip(segments[1:], manifests[1:],
                                               args.run[:-1]):
        first = current["time"][0]
        if first <= h["time"][0]:
            raise ValueError("A continuation starts before the prior history")
        keep = h["time"] < first
        dropped = int(np.count_nonzero(~keep))
        cadence = max(np.median(np.diff(h["time"])),
                      np.median(np.diff(current["time"])))
        gap = float(first - h["time"][keep][-1])
        # Restart HST writers may emit their first row after two accepted
        # steps; allow that sampling delay but still reject a true gap.
        if gap > 3.1*cadence:
            raise ValueError("Gap between segment histories exceeds three samples")
        joins.append({"restart": str(previous_run / Path(manifest["restart"]).name),
                      "overlap_dropped_samples": dropped,
                      "history_gap_code": gap})
        h = {name: np.r_[values[keep], current[name]]
             for name, values in h.items()}
    units = input_values(args.run[0]/"athinput.generated")
    report = summarize(h, combined_log(logs, manifests), units,
                       args.reference_flux)
    report["stationarity_window_scope"] = (
        "Combined consecutive entropy-feedback segments with identical binary, "
        "base input, and tables; only explicit inflow entropy may change; "
        "overlapping HST rows removed"
        if args.allow_s_in_variation else
        "Combined consecutive segments with identical binary, base input, "
        "tables, and problem overrides; overlapping superseded HST rows removed")
    report["entropy_feedback_controlled"] = args.allow_s_in_variation
    report["source_runs"] = [str(run) for run in args.run]
    report["source_runs_relative"] = [os.path.relpath(run, args.output_dir.resolve())
                                      for run in args.run]
    report["history_joins"] = joins
    report["interrupted_segments"] = [str(run) for run, manifest in
                                       zip(args.run, manifests)
                                       if manifest.get("interruption_reason")]
    report["segment_log_summaries"] = logs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    make_plot(h, units, args.reference_flux, args.output_dir/"history.png")
    make_energy_budget_plot(h, units, args.reference_flux,
                            args.output_dir/"energy_budget.png",
                            "Combined-segment energy budget")
    checks = report.get("stationarity_checks", {})
    checks_html = "".join(
        f"<tr><td>{html.escape(key.replace('_', ' '))}</td>"
        f"<td class='{('pass' if passed else 'fail')}'>"
        f"{('PASS' if passed else 'FAIL')}</td></tr>"
        for key, passed in checks.items())
    windows_html = ""
    for label, key in (("Earlier late window", "early_late_window"),
                       ("Latest window", "late_window")):
        row = report[key]
        windows_html += (
            f"<tr><td>{label}</td>"
            f"<td>{row['start_time']:.1f}–{row['end_time']:.1f}</td>"
            f"<td>{row['vertical_crossing_times']:.2f}</td>"
            f"<td>{row['top_total_flux_over_solar']:.3f}</td>"
            f"<td>{row['bottom_hydro_flux_over_solar']:.3f}</td>"
            f"<td>{row['stored_energy_rate_over_solar']:+.3f}</td>"
            f"<td>{row['stored_energy_trend_over_solar']:+.3f}</td>"
            f"<td>{100*row['max_abs_mass_drift']:.3f}</td></tr>")
    links = " · ".join(
        f"<a href='{html.escape(os.path.relpath(run/'index.html', args.output_dir))}'>"
        f"segment {i+1}</a>" for i, run in enumerate(args.run))
    verdict = "PASS" if report["stationarity_candidate"] else "NOT ESTABLISHED"
    interruption_note = (
        "<p><strong>Interrupted segment:</strong> one execution session stopped "
        "and was resumed from its numbered checkpoint. "
        "The joined plot removes overlapping HST rows from that restart. "
        "Solver convergence is aggregated over both attempts.</p>"
        if report["interrupted_segments"] else "")
    movie_section = ""
    movie_manifest = args.output_dir / "movies" / "manifest.json"
    if movie_manifest.is_file():
        movie_data = json.loads(movie_manifest.read_text())
        cards = []
        for key, title in (("xz", "Vertical slice with τ=1 contour"),
                           ("xy", "Horizontal slice: raw vertical velocity"),
                           ("xy_fluct", "Horizontal slice: plane mean removed")):
            item = movie_data.get("movies", {}).get(key)
            if not item or not (args.output_dir / "movies" / item["file"]).is_file():
                continue
            poster = args.output_dir / "movies" / item["poster"]
            poster_attr = (f" poster='movies/{html.escape(poster.name)}'"
                           if poster.is_file() else "")
            cards.append(
                f"<figure><video controls preload='metadata'{poster_attr}>"
                f"<source src='movies/{html.escape(item['file'])}' type='video/mp4'>"
                "Your browser does not support MP4 video.</video>"
                f"<figcaption><strong>{html.escape(title)}</strong> · "
                f"{item['rendered_frames']} frames · "
                f"t={item['first_time_code']:.1f}–"
                f"{item['last_time_code']:.1f}</figcaption></figure>")
        if cards:
            movie_section = "<h2>Evolution movies</h2>" + "".join(cards)
    mode_section = ""
    if (args.output_dir / "surface_mode" / "index.html").is_file():
        mode_section = (
            "<p><a href='surface_mode/index.html'>Coherent vertical-mode "
            "history and fixed-period amplitudes</a></p>")
    if (args.output_dir / "flux_oscillation" / "index.html").is_file():
        mode_section += (
            "<p><a href='flux_oscillation/index.html'>Lower-boundary flux "
            "oscillation across early and late windows</a></p>")
    flux_note = ""
    if report["failed_stationarity_checks"] == ["both_top_fluxes_within_10pct_solar"]:
        flux_note = (
            "<p><strong>All listed stability and solver checks pass except "
            "solar-flux agreement.</strong> The upper flux is stable in the "
            "late windows but remains below the solar target.</p>")
    (args.output_dir/"index.html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<title>Solar convection continuation screen</title><style>"
        "body{font:22px/1.5 system-ui;max-width:1250px;margin:2rem auto;"
        "padding:0 1rem;color:#172b3a}table{border-collapse:collapse}"
        "th,td{padding:.5rem .75rem;border-bottom:1px solid #ddd;text-align:left}"
        "img{max-width:100%}video{width:100%;max-width:1050px}"
        "figure{margin:2rem 0}figcaption{margin-top:.5rem}"
        ".pass{font-weight:bold;color:#117347}"
        ".fail{font-weight:bold;color:#a12929}</style>"
        "<h1>Solar convection: combined continuation</h1>"
        f"<p><strong>Stationarity screen: {verdict}.</strong> "
        +("The joined runs have identical binary and physics inputs except "
          "for the controlled inflow entropy. " if args.allow_s_in_variation else
          "The joined runs have identical binary and physics inputs. ")+
        "A screen pass would still require optical-profile and resolution "
        "comparison with the Sun.</p>"
        +interruption_note+flux_note+
        f"<p>{links} · <a href='summary.json'>JSON</a></p>"
        "<h2>Late windows</h2><table><tr><th>Window</th><th>Code time</th>"
        "<th>Crossings</th><th>Top / solar</th><th>Bottom hydro / solar</th>"
        "<th>Stored endpoint rate / solar</th><th>Stored trend / solar</th>"
        "<th>Max |mass drift| [%]</th></tr>"
        +windows_html+"</table>"
        "<h2>Acceptance checks</h2><table><tr><th>Check</th><th>Result</th></tr>"
        +checks_html+"</table>"+mode_section+movie_section+
        "<img src='history.png' alt='joined history'>"
        "<img src='energy_budget.png' alt='joined energy budget'></html>"
    )
    make_campaign_index(args.output_dir.parent)
    print(args.output_dir/"index.html")


if __name__ == "__main__":
    main()
