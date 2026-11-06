#!/usr/bin/env python3
"""Run a bounded, restart-based solar inflow-entropy feedback pilot.

This is an outer-loop approximation to a slow luminosity controller. It uses
the measured bottom-flux response to express the top-flux error in entropy
units, then relaxes that correction over the box thermal time. A matched
fixed-entropy continuation provides the control. It does not claim thermal
equilibrium or a calibrated solar solution.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from solar_convection_report import combine_histories, read_history


F_SUN = 6.3e10


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def memory_status(case: Path) -> dict[str, float]:
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in {"MemAvailable", "SwapTotal", "SwapFree"}:
            info[key] = int(value.split()[0]) / 1024.0
    rss = 0.0
    ranks = 0
    case_bytes = str(case).encode()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if (proc / "comm").read_text().strip() not in {"athena", "athena.frozen"}:
                continue
            if case_bytes not in (proc / "cmdline").read_bytes():
                continue
            ranks += 1
            status = (proc / "status").read_text()
            match = re.search(r"^VmRSS:\s*(\d+) kB", status, re.M)
            if match:
                rss += int(match.group(1)) / 1024.0
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return {"mem_available_mib": info["MemAvailable"],
            "swap_used_mib": info["SwapTotal"] - info["SwapFree"],
            "rank_rss_mib": rss, "observed_ranks": ranks}


def weighted_mean(t: np.ndarray, y: np.ndarray) -> float:
    if len(t) < 2 or t[-1] <= t[0]:
        raise ValueError("Need at least two distinct history samples")
    return float(np.trapezoid(y, t) / (t[-1] - t[0]))


def top_radiative_flux(history: dict[str, np.ndarray]) -> np.ndarray:
    return history["FradTop"] + history.get("FadvRadTop", 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument("--restart", type=Path, required=True)
    parser.add_argument("--header-from", type=Path, required=True)
    parser.add_argument("--base-input", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--initial-s-in", type=float, required=True)
    parser.add_argument("--start-time", type=float, default=5000.0)
    parser.add_argument("--segment-code", type=float, default=25.0)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--flux-average-code", type=float, default=20.0)
    parser.add_argument("--tau-kh-code", type=float, default=2.0e4)
    parser.add_argument("--response-fsun-per-1e6", type=float, default=0.322326259)
    parser.add_argument("--max-entropy-change", type=float, default=5.0e4,
                        help="Maximum absolute entropy update per segment [erg g^-1 K^-1]")
    parser.add_argument("--ranks", type=int, default=16)
    parser.add_argument("--max-wall-minutes", type=float, default=27.0)
    parser.add_argument("--min-available-gib", type=float, default=3.0)
    parser.add_argument("--max-swap-used-gib", type=float, default=4.0)
    args = parser.parse_args()

    repo = args.repo.resolve()
    restart = args.restart.resolve()
    header = args.header_from.resolve()
    base_input = args.base_input.resolve()
    data_dir = args.data_dir.resolve()
    root = args.output_root.resolve()
    binary = repo / "bin/athena"
    if root.exists() and any(root.iterdir()):
        parser.error(f"Refusing nonempty output root: {root}")
    if args.segments < 2 or args.segment_code <= 0 or args.ranks < 1:
        parser.error("Need at least two positive-length segments and positive ranks")
    if any(value <= 0 for value in (args.initial_s_in, args.flux_average_code,
                                    args.tau_kh_code, args.response_fsun_per_1e6,
                                    args.max_entropy_change, args.max_wall_minutes,
                                    args.min_available_gib, args.max_swap_used_gib)):
        parser.error("Controller and resource parameters must be positive")
    source_manifest = restart.parent / "manifest.json"
    for path in (restart, header, base_input, binary, source_manifest):
        if not path.is_file():
            parser.error(f"Missing input: {path}")
    source = json.loads(source_manifest.read_text())
    if source.get("binary_sha256") != digest(binary):
        parser.error("Binary differs from source restart")
    if source.get("base_input_sha256") != digest(base_input):
        parser.error("Base input differs from source restart")
    for name, expected in source.get("data_sha256", {}).items():
        path = data_dir / name
        if not path.is_file() or digest(path) != expected:
            parser.error(f"External input differs from source restart: {path}")

    root.mkdir(parents=True, exist_ok=True)
    monitor_path = root / "memory_monitor.csv"
    monitor = monitor_path.open("w")
    monitor.write("wall_s,case,mem_available_mib,swap_used_mib,rank_rss_mib,observed_ranks\n")
    monitor.flush()
    started = time.monotonic()
    peak_rss = 0.0
    lowest_available = float("inf")
    peak_swap = 0.0
    runs = []
    actions = []

    def run_case(name: str, from_restart: Path, end_time: float,
                 entropy: float) -> tuple[Path, dict[str, np.ndarray]]:
        nonlocal peak_rss, lowest_available, peak_swap
        case = root / name
        command = ["taskset", "-c", "0-15", sys.executable,
                   str(repo / "tst/nrfld_validation/run_solar_convection.py"),
                   "--repo", str(repo), "--binary", str(binary),
                   "--base-input", str(base_input), "--data-dir", str(data_dir),
                   "--output-root", str(root), "--case", name,
                   "--solver", "fgmres", "--cfl", "0.3",
                   "--nx1", "64", "--nx2", "64", "--nx3", "32",
                   "--ranks", str(args.ranks), "--mpi-bind-to", "hwthread",
                   "--mpi-map-by", "hwthread", "--tlim", f"{end_time:.9f}",
                   "--checkpoint-dt", f"{args.segment_code:.9f}",
                   "--history-dt", "0.1", "--history-precision", "16",
                   "--restart", str(from_restart),
                   "--history-header-from", str(header),
                   "--problem-set", "mass_balance_relax_time=2.0",
                   "--problem-set", "mass_balance_rate_gain=40",
                   "--problem-set", "s_in_from_profile=false",
                   "--problem-set", f"s_in={entropy:.15e}",
                   "--history-and-restart-only"]
        launcher_log = root / f"{name}_launcher.log"
        with launcher_log.open("w") as log:
            process = subprocess.Popen(command, cwd=repo, stdout=log,
                                       stderr=subprocess.STDOUT,
                                       start_new_session=True)
            while process.poll() is None:
                status = memory_status(case)
                wall = time.monotonic() - started
                peak_rss = max(peak_rss, status["rank_rss_mib"])
                lowest_available = min(lowest_available,
                                       status["mem_available_mib"])
                peak_swap = max(peak_swap, status["swap_used_mib"])
                monitor.write(f"{wall:.3f},{name},{status['mem_available_mib']:.1f},"
                              f"{status['swap_used_mib']:.1f},"
                              f"{status['rank_rss_mib']:.1f},"
                              f"{status['observed_ranks']}\n")
                monitor.flush()
                reason = None
                if wall > args.max_wall_minutes * 60:
                    reason = "wall-time limit"
                elif status["mem_available_mib"] < args.min_available_gib*1024:
                    reason = "low available memory"
                elif status["swap_used_mib"] > args.max_swap_used_gib*1024:
                    reason = "excess swap use"
                if reason:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                    raise RuntimeError(f"Stopped {name}: {reason}; see {monitor_path}")
                time.sleep(2.0)
            if process.returncode != 0:
                raise RuntimeError(f"{name} failed with exit {process.returncode}; "
                                   f"see {launcher_log} and {case / 'run.log'}")
        final_restart = next(case.glob("*.final.rst"), None)
        if final_restart is None:
            raise RuntimeError(f"{name} produced no final restart")
        history = read_history(next(case.glob("*.hst")), header)
        manifest = json.loads((case / "manifest.json").read_text())
        runs.append({"case": name, "entropy_cgs": entropy,
                     "wall_seconds": manifest["wall_seconds"],
                     "final_restart": str(final_restart),
                     "final_time_code": float(history["time"][-1])})
        return final_restart, history

    try:
        end = args.start_time + args.segments*args.segment_code
        fixed_restart = restart
        fixed_histories = []
        for index in range(args.segments):
            segment_end = args.start_time + (index+1)*args.segment_code
            name = f"fixed_{index+1:02d}_t{segment_end:g}"
            fixed_restart, _ = run_case(
                name, fixed_restart, segment_end, args.initial_s_in)
            fixed_histories.append(next((root / name).glob("*.hst")))
        fixed_history = combine_histories(fixed_histories, header)
        branch_restart = restart
        entropy = args.initial_s_in
        branch_histories = []
        for index in range(args.segments):
            segment_end = args.start_time + (index+1)*args.segment_code
            name = f"feedback_{index+1:02d}_t{segment_end:g}"
            branch_restart, history = run_case(
                name, branch_restart, segment_end, entropy)
            branch_histories.append(next((root / name).glob("*.hst")))
            selected = history["time"] >= max(
                history["time"][0], history["time"][-1]-args.flux_average_code)
            t = history["time"][selected]
            mean_top = weighted_mean(t, top_radiative_flux(history)[selected])
            duration = args.segment_code
            # The response is measured in F_sun per 1e6 entropy units. The
            # dt/tau_KH factor gives the slow MURaM-like luminosity relaxation.
            proposed = (duration/args.tau_kh_code)*(1.0-mean_top/F_SUN)
            proposed *= 1.0e6/args.response_fsun_per_1e6
            applied = float(np.clip(proposed, -args.max_entropy_change,
                                    args.max_entropy_change))
            actions.append({"case": name, "time_code": segment_end,
                            "s_in_before_cgs": entropy,
                            "mean_top_radiative_flux_over_solar": mean_top/F_SUN,
                            "duration_code": duration,
                            "proposed_delta_s_cgs": proposed,
                            "applied_delta_s_cgs": applied if index+1 < args.segments else 0.0,
                            "s_in_next_cgs": entropy + applied if index+1 < args.segments else None})
            if index+1 < args.segments:
                entropy += applied
        branch = combine_histories(branch_histories, header)
        last_start = end-args.segment_code
        def last_mean(history: dict[str, np.ndarray], values: np.ndarray) -> float:
            chosen = history["time"] >= last_start
            return weighted_mean(history["time"][chosen], values[chosen])
        summary = {
            "status": "completed_feedback_pilot_not_thermal_calibration",
            "source_restart": str(restart), "start_time_code": args.start_time,
            "end_time_code": end, "segment_code": args.segment_code,
            "tau_kh_code": args.tau_kh_code,
            "response_fsun_per_1e6": args.response_fsun_per_1e6,
            "reference_flux_cgs": F_SUN, "runs": runs, "feedback_actions": actions,
            "fixed_late_top_flux_over_solar": last_mean(
                fixed_history, top_radiative_flux(fixed_history))/F_SUN,
            "feedback_late_top_flux_over_solar": last_mean(
                branch, top_radiative_flux(branch))/F_SUN,
            "fixed_late_bottom_hydro_flux_over_solar": last_mean(
                fixed_history, fixed_history["FhydBottom"])/F_SUN,
            "feedback_late_bottom_hydro_flux_over_solar": last_mean(
                branch, branch["FhydBottom"])/F_SUN,
            "min_available_memory_mib": lowest_available,
            "peak_swap_used_mib": peak_swap,
            "peak_rank_rss_mib": peak_rss,
            "memory_monitor_csv": str(monitor_path),
            "elapsed_wall_minutes": (time.monotonic()-started)/60.0,
        }
        (root / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        plt.rcParams.update({"font.size": 13})
        fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True,
                                 constrained_layout=True)
        for history, name in ((fixed_history, "fixed"), (branch, "feedback")):
            x = history["time"]-args.start_time
            axes[0].plot(x, top_radiative_flux(history)/F_SUN, label=name)
            axes[1].plot(x, history["FhydBottom"]/F_SUN, label=name, alpha=0.8)
        axes[0].axhline(1, color="black", lw=1, ls="--")
        axes[0].set_ylabel("Top radiation / solar")
        axes[1].set_ylabel("Bottom hydro / solar")
        step_x = [0.0] + [entry["time_code"]-args.start_time
                          for entry in actions[:-1]] + [end-args.start_time]
        step_y = [args.initial_s_in] + [entry["s_in_next_cgs"]
                                        for entry in actions[:-1]]
        axes[2].step(step_x, step_y+[step_y[-1]], where="post")
        axes[2].set_ylabel("Commanded inflow entropy\n[erg g$^{-1}$ K$^{-1}$]")
        axes[2].set_xlabel("Time since restart [code]")
        for ax in axes:
            ax.grid(alpha=0.3)
        axes[0].legend()
        axes[1].legend()
        fig.savefig(root / "feedback_history.png", dpi=150)
        plt.close(fig)
        lines = ["<!doctype html><html><meta charset='utf-8'><title>Solar entropy feedback pilot</title>",
                 "<style>body{font:20px/1.55 system-ui;max-width:1150px;margin:2rem auto;padding:0 1rem}"
                 "h1{font-size:2rem}strong{color:#18446f}img{max-width:100%}li{margin:.5rem 0}</style>",
                 "<h1>Solar inflow-entropy feedback pilot</h1>",
                 "<p><strong>Completed pilot, not a solar calibration.</strong> "
                 "The entropy controller uses the top radiative-flux error and a slow "
                 "thermal-time update. The fixed run is the matched control.</p>",
                 f"<p>Final-window top flux: fixed {summary['fixed_late_top_flux_over_solar']:.6f}, "
                 f"feedback {summary['feedback_late_top_flux_over_solar']:.6f} F_sun. "
                 f"Peak rank RSS {peak_rss:.0f} MiB; minimum available WSL memory "
                 f"{lowest_available:.0f} MiB.</p>",
                 "<img src='feedback_history.png' alt='Fluxes and commanded entropy'>",
                 "<h2>Runs</h2><ul>"]
        for run in runs:
            case = html.escape(run["case"])
            lines.append(f"<li><a href='{case}/index.html'>{case}</a>: "
                         f"{run['wall_seconds']:.1f} s; s_in={run['entropy_cgs']:.6e}</li>")
        lines += ["</ul><p><a href='summary.json'>Detailed metrics</a> · "
                  "<a href='memory_monitor.csv'>Memory log</a></p></html>"]
        (root / "index.html").write_text("\n".join(lines)+"\n")
        print(root / "index.html", flush=True)
    finally:
        monitor.close()


if __name__ == "__main__":
    main()
