#!/usr/bin/env python3
"""Run and resume a single, slow solar inflow-entropy feedback trajectory.

The controller operates between restart segments. It is a bounded pilot of
luminosity feedback, not evidence that the photospheric flux is calibrated.
An earlier fixed-entropy trajectory is available under WOR-70, but its early
mass-controller settings differ and it is not an exactly matched control.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_solar_entropy_feedback import F_SUN, digest, memory_status, top_radiative_flux, weighted_mean
from solar_convection_report import (combine_histories, input_values,
                                     read_history, sampled_energy_ledger,
                                     total_bottom_flux)


DATA_NAMES = ("AthenaNataEosTable_solar_deep.tab",
              "opacity_table_rhoT_3p0_6p3.h5", "solar_initial_profile.dat")
OUTPUT_CADENCE = {2: 5.0, 3: 5.0, 4: 10.0, 5: 10.0,
                  6: 100.0, 7: 50.0, 8: 50.0}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def campaign_input(base: str, cadence: dict[int, float]) -> str:
    result = base
    for number, interval in cadence.items():
        pattern = rf"(^<output{number}>\s*$)(.*?)(?=^<|\Z)"
        matches = list(re.finditer(pattern, result, re.M | re.S))
        if len(matches) != 1:
            raise ValueError(f"Expected one <output{number}> section")
        match = matches[0]
        block = match.group(0)
        block, count = re.subn(r"^\s*dt\s*=\s*[^#\n]+",
                               f"dt = {interval:g}", block, count=1, flags=re.M)
        if count != 1:
            raise ValueError(f"Missing dt in <output{number}>")
        result = result[:match.start()] + block + result[match.end():]
    return result


def rolling_fluxes(root: Path, completed: list[dict], header: Path,
                   end_time: float, window: float, units: dict) -> dict:
    """Integrate the last ``window`` code times across restart boundaries."""
    cutoff = end_time - window
    paths = [root / row["case"] / row["history_name"]
             for row in completed if row["end_time_code"] >= cutoff - 1.0e-8]
    if not paths:
        raise ValueError("No histories cover the requested feedback window")
    history = combine_histories(paths, header)
    time_values = history["time"]
    if time_values[0] > cutoff + 1.0e-6 or time_values[-1] < end_time - 1.0e-6:
        raise ValueError("Histories do not cover the complete feedback window")
    selected = (time_values > cutoff) & (time_values <= end_time)
    t = np.r_[cutoff, time_values[selected]]
    if len(t) < 2:
        raise ValueError("Too few samples in feedback window")

    def mean(values: np.ndarray) -> float:
        y = np.r_[np.interp(cutoff, time_values, values), values[selected]]
        return weighted_mean(t, y) / F_SUN

    ledger_indices = np.flatnonzero((time_values >= cutoff) &
                                    (time_values <= end_time))
    ledger = sampled_energy_ledger(history, ledger_indices, units, F_SUN)
    return {"start_time_code": cutoff, "end_time_code": end_time,
            "top_over_solar": mean(top_radiative_flux(history)),
            "bottom_total_over_solar": mean(total_bottom_flux(history)),
            "bottom_hydro_over_solar": mean(history["FhydBottom"]),
            "stored_energy_trend_over_solar":
                ledger["stored_energy_trend_over_solar"] if ledger else None}


def render(root: Path, state: dict) -> None:
    completed = state["completed"]
    config = state["config"]
    last_time = completed[-1]["end_time_code"] if completed else 0.0
    last_flux = completed[-1]["top_radiative_flux_over_solar"] if completed else None
    mean_wall = (sum(row["total_wall_seconds"] for row in completed[-5:])
                 / len(completed[-5:])) if completed else None
    remaining_h = ((config["target_time_code"] - last_time)
                   / config["segment_code"] * mean_wall / 3600.0
                   if mean_wall is not None else None)
    if completed:
        plt.rcParams.update({"font.size": 14})
        figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                   constrained_layout=True)
        times = [row["end_time_code"] for row in completed]
        axes[0].plot(times, [row["top_radiative_flux_over_solar"]
                             for row in completed], "o-", label="top radiation")
        axes[0].plot(times, [row["bottom_hydro_flux_over_solar"]
                             for row in completed], "o-", label="bottom hydro")
        axes[0].axhline(1.0, color="black", ls="--", lw=1, label="solar")
        axes[0].set_ylabel("Flux / solar")
        axes[0].legend()
        axes[1].plot(times, [row["next_entropy_cgs"] - config["initial_entropy_cgs"]
                             for row in completed], "o-")
        axes[1].set_ylabel("Inflow entropy change\n[erg g$^{-1}$ K$^{-1}$]")
        axes[1].set_xlabel("Code time")
        for axis in axes:
            axis.grid(alpha=0.3)
        figure.savefig(root / "feedback_progress.png", dpi=150)
        plt.close(figure)
    table_rows = []
    for row in completed:
        name = html.escape(row["case"])
        table_rows.append(
            f"<tr><td><a href='{name}/index.html'>{name}</a></td>"
            f"<td>{row['end_time_code']:.0f}</td>"
            f"<td>{row['top_radiative_flux_over_solar']:.5f}</td>"
            f"<td>{row.get('rolling_top_flux_over_solar', row['top_radiative_flux_over_solar']):.5f}</td>"
            f"<td>{row['bottom_hydro_flux_over_solar']:.5f}</td>"
            f"<td>{row['entropy_cgs']:.7e}</td>"
            f"<td>{row['wall_seconds']/60:.1f}</td></tr>")
    lines = ["<!doctype html><html lang='en'><meta charset='utf-8'>",
             "<title>Solar entropy feedback from t=0</title>",
             "<style>body{font:20px/1.55 system-ui;max-width:1250px;margin:2rem auto;padding:0 1rem}"
             "h1{font-size:2rem}h2{font-size:1.45rem}strong{color:#18446f}"
             "img{max-width:100%}table{border-collapse:collapse;width:100%}"
             "th,td{padding:.4rem .6rem;border-bottom:1px solid #bbb;text-align:right}"
             "th:first-child,td:first-child{text-align:left}</style>",
             "<h1>Solar inflow-entropy feedback from t=0</h1>",
             f"<p><strong>Status: {html.escape(state['status'])}.</strong> "
             f"Reached t={last_time:.0f} / {config['target_time_code']:.0f} "
             f"with {len(completed)} completed restart segments. "
             f"Last top radiative flux: {last_flux:.5f} F_sun. " if last_flux is not None else
             f"<p><strong>Status: {html.escape(state['status'])}.</strong> "
             "No segment has completed yet.",
             f"Projected compute time remaining from last five segments: "
             f"{remaining_h:.1f} h.</p>" if remaining_h is not None else "</p>",
             "<p>The slow controller is a transient experiment. Top-flux agreement "
             "and thermal stationarity require long-window checks.</p>",
             "<p><a href='state.json'>State and controls</a> · "
             "<a href='memory_monitor.csv'>Memory monitor</a> · "
             "<a href='driver.log'>Driver log</a></p>"]
    if completed:
        lines.append("<img src='feedback_progress.png' alt='Flux and entropy history'>")
    if "control_switch" in state:
        switch = state["control_switch"]
        lines.append(f"<p><strong>Controller switched at t={switch['at_time_code']:.0f}:</strong> "
                     f"{config['flux_average_code']:.0f}-code-time rolling top flux, "
                     f"{config['segment_code']:.0f}-code-time updates, "
                     f"cumulative entropy limit ±{config['max_total_entropy_change_cgs']:.0f}. "
                     f"Source: {html.escape(state['forked_from'])}</p>")
    if (root / "combined" / "index.html").is_file():
        lines.append("<p><strong>Full-campaign analysis:</strong> "
                     "<a href='combined/index.html'>joined flux checks, plots, "
                     "and movies</a></p>")
    if (root / "entropy_transport" / "index.html").is_file():
        lines.append("<p><strong>Bottom entropy transport:</strong> "
                     "<a href='entropy_transport/index.html'>prescribed, upflow, "
                     "and downflow entropy</a></p>")
    if (root / "wait_diagnostic" / "index.html").is_file():
        lines.append("<p><strong>Flux trend audit:</strong> "
                     "<a href='wait_diagnostic/index.html'>startup mismatch "
                     "and 500-code-time means</a></p>")
    lines += ["<h2>Restart segments</h2>",
              "<table><thead><tr><th>Case</th><th>End t</th><th>Top / solar</th>"
              "<th>Rolling top / solar</th>"
              "<th>Bottom / solar</th><th>Input entropy</th><th>Wall min</th>"
              "</tr></thead><tbody>", *table_rows, "</tbody></table></html>"]
    (root / "index.html").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument("--binary", type=Path,
                        help="Frozen Athena executable for every restart segment")
    parser.add_argument("--base-input", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--initial-s-in", type=float, required=True)
    parser.add_argument("--target-time", type=float, default=5000.0)
    parser.add_argument("--segment-code", type=float, default=100.0)
    parser.add_argument("--flux-average-code", type=float, default=100.0)
    parser.add_argument("--tau-kh-code", type=float, default=2.0e4)
    parser.add_argument("--response-fsun-per-1e6", type=float, default=0.322326259)
    parser.add_argument("--max-entropy-change", type=float, default=5.0e4)
    parser.add_argument("--max-total-entropy-change", type=float, default=1.0e6,
                        help="Absolute entropy offset allowed from the initial profile")
    parser.add_argument("--ranks", type=int, default=16)
    parser.add_argument("--min-available-gib", type=float, default=3.0)
    parser.add_argument("--max-swap-used-gib", type=float, default=4.0)
    parser.add_argument("--min-disk-free-gib", type=float, default=50.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fork-from", type=Path,
                        help="Fork a completed campaign without copying its output data")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Prepare a fork and its report without starting Athena")
    parser.add_argument("--allow-binary-change", action="store_true",
                        help="Record an intentional diagnostic-only binary change on fork")
    args = parser.parse_args()
    for name in ("initial_s_in", "target_time", "segment_code", "flux_average_code",
                 "tau_kh_code", "response_fsun_per_1e6", "max_entropy_change",
                 "max_total_entropy_change",
                 "min_available_gib", "max_swap_used_gib", "min_disk_free_gib"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.ranks < 1 or args.ranks > 32:
        parser.error("The 64x64x32 grid has 32 MeshBlocks; ranks must be 1–32")
    if args.resume and args.fork_from:
        parser.error("Use either --resume or --fork-from")
    if args.prepare_only and not args.fork_from:
        parser.error("--prepare-only requires --fork-from")
    if abs(args.target_time / args.segment_code
           - round(args.target_time / args.segment_code)) > 1.0e-9:
        parser.error("Target time must be a multiple of segment length")

    repo = args.repo.resolve()
    base_input = args.base_input.resolve()
    data_dir = args.data_dir.resolve()
    reference_path = args.reference_manifest.resolve()
    binary = args.binary.resolve() if args.binary else repo / "bin/athena"
    root = args.output_root.resolve()
    for path in (base_input, reference_path, binary, *(data_dir / n for n in DATA_NAMES)):
        if not path.is_file():
            parser.error(f"Missing input: {path}")
    reference = json.loads(reference_path.read_text())
    if digest(binary) != reference["binary_sha256"] and not args.allow_binary_change:
        parser.error("Binary SHA256 differs from the validated 96 km baseline")
    if digest(base_input) != reference["base_input_sha256"]:
        parser.error("Base input SHA256 differs from the validated 96 km baseline")
    hashes = {name: digest(data_dir / name) for name in DATA_NAMES}
    if hashes != reference["data_sha256"]:
        parser.error("EOS, opacity, or profile SHA256 differs from baseline")

    config = {"initial_entropy_cgs": args.initial_s_in,
              "target_time_code": args.target_time,
              "segment_code": args.segment_code,
              "flux_average_code": args.flux_average_code,
              "tau_kh_code": args.tau_kh_code,
              "response_fsun_per_1e6": args.response_fsun_per_1e6,
              "max_entropy_change_cgs": args.max_entropy_change,
              "max_total_entropy_change_cgs": args.max_total_entropy_change,
              "ranks": args.ranks, "binary_sha256": digest(binary),
              "base_input_sha256": digest(base_input), "data_sha256": hashes,
              "reference_manifest": str(reference_path),
              "output_cadence_code": {str(k): v for k, v in OUTPUT_CADENCE.items()}}
    state_path = root / "state.json"
    custom_input = root / "athinput.campaign"
    if args.fork_from:
        source_root = args.fork_from.resolve()
        source_state_path = source_root / "state.json"
        if root.exists() and any(root.iterdir()):
            parser.error(f"Refusing nonempty output root: {root}")
        if not source_state_path.is_file():
            parser.error(f"Missing source state: {source_state_path}")
        state = json.loads(source_state_path.read_text())
        if state["status"] != "completed" or not state["completed"]:
            parser.error("Source campaign must be complete")
        old = state["config"]
        for key, value in config.items():
            if key in {"target_time_code", "flux_average_code", "tau_kh_code",
                       "max_total_entropy_change_cgs"}:
                continue
            if key == "binary_sha256" and args.allow_binary_change:
                continue
            if value != old.get(key):
                parser.error(f"Source configuration differs: {key}")
        last = state["completed"][-1]
        if args.target_time <= last["end_time_code"]:
            parser.error("Fork target must exceed the source end time")
        if (source_root / "athinput.campaign").read_text() != campaign_input(
                base_input.read_text(), OUTPUT_CADENCE):
            parser.error("Source campaign input has changed")
        last_restart = source_root / last["case"] / last["restart_name"]
        if digest(last_restart) != last["restart_sha256"]:
            parser.error("Source final restart has changed")
        root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / "athinput.campaign", custom_input)
        for row in state["completed"]:
            source_case = source_root / row["case"]
            if not source_case.is_dir():
                parser.error(f"Source segment missing: {source_case}")
            (root / row["case"]).symlink_to(source_case, target_is_directory=True)
        state["config"] = config
        state["status"] = "running"
        state["created_at"] = now()
        state["forked_from"] = str(source_root)
        state["binary_path"] = str(binary)
        if old["binary_sha256"] != config["binary_sha256"]:
            state["binary_change"] = {
                "source_sha256": old["binary_sha256"],
                "new_sha256": config["binary_sha256"],
                "reason": "Directional lower-face flux diagnostics added; matched "
                          "restart showed identical time and total face fluxes"}
        state["stop_reason"] = None
        state["min_available_memory_mib"] = None
        state["peak_swap_used_mib"] = 0.0
        state["peak_rank_rss_mib"] = 0.0
        header = root / state["completed"][0]["case"] / state["completed"][0]["history_name"]
        filtered = rolling_fluxes(root, state["completed"], header,
                                  last["end_time_code"], args.flux_average_code,
                                  input_values(custom_input))
        proposed = (args.segment_code / args.tau_kh_code) * (1.0 - filtered["top_over_solar"])
        proposed *= 1.0e6 / args.response_fsun_per_1e6
        next_entropy = float(np.clip(last["entropy_cgs"] + np.clip(
            proposed, -args.max_entropy_change, args.max_entropy_change),
            args.initial_s_in - args.max_total_entropy_change,
            args.initial_s_in + args.max_total_entropy_change))
        state["next_entropy_cgs"] = next_entropy
        state["control_switch"] = {
            "at_time_code": last["end_time_code"], "rolling_flux": filtered,
            "previous_next_entropy_cgs": last["next_entropy_cgs"],
            "next_entropy_cgs": next_entropy}
    elif args.resume:
        if not state_path.is_file() or not custom_input.is_file():
            parser.error("--resume requires existing state and campaign input")
        state = json.loads(state_path.read_text())
        if state.get("binary_path") not in (None, str(binary)):
            parser.error("Resume binary path differs from the recorded executable")
        state["binary_path"] = str(binary)
        old = state["config"]
        for key, value in config.items():
            if key == "target_time_code":
                if value < old[key]:
                    parser.error("Cannot reduce an existing target time")
            elif value != old.get(key, 1.0e6 if key == "max_total_entropy_change_cgs" else None):
                parser.error(f"Resume config differs: {key}")
        if custom_input.read_text() != campaign_input(base_input.read_text(), OUTPUT_CADENCE):
            parser.error("Campaign input has changed")
        state["config"]["target_time_code"] = args.target_time
        state["status"] = "running"
        state["stop_reason"] = None
        state["resumed_at"] = now()
    else:
        if root.exists() and any(root.iterdir()):
            parser.error(f"Refusing nonempty output root: {root}")
        root.mkdir(parents=True, exist_ok=True)
        custom_input.write_text(campaign_input(base_input.read_text(), OUTPUT_CADENCE))
        state = {"status": "running", "created_at": now(), "config": config,
                 "binary_path": str(binary),
                 "completed": [], "next_entropy_cgs": args.initial_s_in,
                 "min_available_memory_mib": None, "peak_swap_used_mib": 0.0,
                 "peak_rank_rss_mib": 0.0, "stop_reason": None}
    save_json(state_path, state)
    render(root, state)
    if args.prepare_only:
        state["status"] = "prepared"
        save_json(state_path, state)
        render(root, state)
        print(f"{now()} prepared {root}: next entropy={state['next_entropy_cgs']:.9e}",
              flush=True)
        return
    monitor_path = root / "memory_monitor.csv"
    if not monitor_path.exists():
        monitor_path.write_text("utc,case,mem_available_mib,swap_used_mib,"
                                "rank_rss_mib,observed_ranks,disk_free_gib\n")
    stop_requested = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    reference_header: Path | None = None
    if state["completed"]:
        reference_header = root / state["completed"][0]["case"] / state["completed"][0]["history_name"]
        if not reference_header.is_file():
            raise RuntimeError("Original header-bearing HST is missing")
    total_segments = round(args.target_time / args.segment_code)
    try:
        for index in range(len(state["completed"]), total_segments):
            if stop_requested:
                raise RuntimeError("Stop signal received between segments")
            end_time = (index + 1) * args.segment_code
            stem = f"feedback_{index+1:03d}_t{end_time:g}"
            name = stem
            attempt = 1
            while (root / name).exists():
                attempt += 1
                name = f"{stem}_retry{attempt:02d}"
            case = root / name
            prior = state["completed"][-1] if state["completed"] else None
            restart = (root / prior["case"] / prior["restart_name"]) if prior else None
            if restart and (not restart.is_file() or digest(restart) != prior["restart_sha256"]):
                raise RuntimeError("Completed-segment restart is missing or changed")
            entropy = state["next_entropy_cgs"]
            command = ["taskset", "-c", "0-15", sys.executable,
                       str(repo / "tst/nrfld_validation/run_solar_convection.py"),
                       "--repo", str(repo), "--binary", str(binary),
                       "--base-input", str(custom_input),
                       "--data-dir", str(data_dir), "--output-root", str(root),
                       "--case", name, "--solver", "fgmres", "--cfl", "0.3",
                       "--nx1", "64", "--nx2", "64", "--nx3", "32",
                       "--ranks", str(args.ranks), "--mpi-bind-to", "hwthread",
                       "--mpi-map-by", "hwthread", "--tlim", f"{end_time:.9f}",
                       "--nlim", "10000000", "--checkpoint-dt", f"{args.segment_code:.9f}",
                       "--history-dt", "0.1", "--history-precision", "16",
                       "--problem-set", "mass_balance_relax_time=2.0",
                       "--problem-set", "mass_balance_rate_gain=40",
                       "--problem-set", "s_in_from_profile=false",
                       "--problem-set", f"s_in={entropy:.15e}"]
            if restart:
                command += ["--restart", str(restart),
                            "--history-header-from", str(reference_header)]
            launcher_log = root / f"{name}_launcher.log"
            print(f"{now()} start {name} s_in={entropy:.9e}", flush=True)
            state["active_case"] = name
            save_json(state_path, state)
            segment_started = time.monotonic()
            with launcher_log.open("w") as launcher:
                process = subprocess.Popen(command, cwd=repo, stdout=launcher,
                                           stderr=subprocess.STDOUT,
                                           start_new_session=True)
                while process.poll() is None:
                    sample = memory_status(case)
                    disk_free = shutil.disk_usage(root).free / 1024**3
                    state["peak_rank_rss_mib"] = max(
                        state["peak_rank_rss_mib"], sample["rank_rss_mib"])
                    previous_min = state["min_available_memory_mib"]
                    state["min_available_memory_mib"] = min(
                        previous_min if previous_min is not None else float("inf"),
                        sample["mem_available_mib"])
                    state["peak_swap_used_mib"] = max(
                        state["peak_swap_used_mib"], sample["swap_used_mib"])
                    with monitor_path.open("a") as monitor:
                        monitor.write(f"{now()},{name},{sample['mem_available_mib']:.1f},"
                                      f"{sample['swap_used_mib']:.1f},"
                                      f"{sample['rank_rss_mib']:.1f},"
                                      f"{sample['observed_ranks']},{disk_free:.1f}\n")
                    reason = ("stop signal" if stop_requested else
                              "available memory below guard" if
                              sample["mem_available_mib"] < args.min_available_gib*1024 else
                              "swap use above guard" if
                              sample["swap_used_mib"] > args.max_swap_used_gib*1024 else
                              "disk free below guard" if
                              disk_free < args.min_disk_free_gib else None)
                    if reason:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        raise RuntimeError(f"Stopped {name}: {reason}")
                    time.sleep(5.0)
                if process.returncode != 0:
                    raise RuntimeError(f"{name} failed with exit {process.returncode}; "
                                       f"inspect {launcher_log} and {case / 'run.log'}")
            restart_files = list(case.glob("*.final.rst"))
            history_files = list(case.glob("*.hst"))
            if len(restart_files) != 1 or len(history_files) != 1:
                raise RuntimeError(f"{name} lacks a unique final restart or history")
            if reference_header is None:
                reference_header = history_files[0]
            history = read_history(history_files[0], reference_header)
            if abs(history["time"][-1] - end_time) > 1.0e-8:
                raise RuntimeError(f"{name} ended at unexpected time {history['time'][-1]}")
            report = json.loads((case / "summary.json").read_text())
            log = report["log"]
            counts = log["nr_status_counts"]
            if (not log["completed"] or log["fatal_count"] or
                    set(counts).difference({"converged", "dt_zero_initialization"})):
                raise RuntimeError(f"{name} did not pass solver-status checks: {counts}")
            selected = history["time"] >= max(
                history["time"][0], end_time-args.segment_code)
            t = history["time"][selected]
            top = weighted_mean(t, top_radiative_flux(history)[selected]) / F_SUN
            bottom = weighted_mean(t, history["FhydBottom"][selected]) / F_SUN
            completed_for_window = [*state["completed"], {
                "case": name, "end_time_code": end_time,
                "history_name": history_files[0].name}]
            filtered = rolling_fluxes(root, completed_for_window,
                                      reference_header, end_time,
                                      args.flux_average_code,
                                      input_values(custom_input))
            proposed = (args.segment_code/args.tau_kh_code) * (1.0-filtered["top_over_solar"])
            proposed *= 1.0e6/args.response_fsun_per_1e6
            next_entropy = float(np.clip(entropy + np.clip(
                proposed, -args.max_entropy_change, args.max_entropy_change),
                args.initial_s_in - args.max_total_entropy_change,
                args.initial_s_in + args.max_total_entropy_change))
            delta = next_entropy - entropy
            manifest = json.loads((case / "manifest.json").read_text())
            state["completed"].append({
                "case": name, "end_time_code": end_time,
                "entropy_cgs": entropy, "next_entropy_cgs": entropy+delta,
                "proposed_delta_s_cgs": proposed, "applied_delta_s_cgs": delta,
                "top_radiative_flux_over_solar": top,
                "rolling_top_flux_over_solar": filtered["top_over_solar"],
                "rolling_bottom_total_flux_over_solar": filtered["bottom_total_over_solar"],
                "rolling_stored_energy_trend_over_solar": filtered["stored_energy_trend_over_solar"],
                "bottom_hydro_flux_over_solar": bottom,
                "wall_seconds": manifest["wall_seconds"],
                "total_wall_seconds": time.monotonic() - segment_started,
                "newton_status_counts": counts,
                "slow_mg_warning_count": log["slow_mg_warning_count"],
                "max_mass_drift": report["late_window"]["max_abs_mass_drift"],
                "restart_name": restart_files[0].name,
                "restart_sha256": digest(restart_files[0]),
                "history_name": history_files[0].name})
            state["next_entropy_cgs"] = entropy+delta
            state.pop("active_case", None)
            state["status"] = "completed" if index+1 == total_segments else "running"
            state["last_updated"] = now()
            save_json(state_path, state)
            render(root, state)
            print(f"{now()} complete {name}: top={top:.6f} F_sun, "
                  f"bottom={bottom:.6f} F_sun, delta_s={delta:.3f}, "
                  f"wall={manifest['wall_seconds']/60:.2f} min", flush=True)
    except Exception as error:
        state["status"] = "stopped"
        state["stop_reason"] = str(error)
        state["last_updated"] = now()
        save_json(state_path, state)
        render(root, state)
        print(f"{now()} STOPPED: {error}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
