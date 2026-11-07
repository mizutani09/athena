#!/usr/bin/env python3
"""Summarize bounded WOR-92 Rosseland and Arnoldi restart controls."""

import csv
import difflib
import hashlib
import json
import re
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
ROOT = Path("/home/kosuke/simulation/test/rad_nr/wor92_wor105_retest_20261107")
BUILDS = Path("/home/kosuke/simulation/test/rad_nr/wor92_operator_audit_20261107")
OUT = HERE / "wor92_restart_floor_results_20261107"
OUT.mkdir(exist_ok=True)
CASES = {
    "fixed_0p0400": ("fixed_r_threshold_0p0400_2rank", "input_fixed_r_threshold_0p0400_2rank.used"),
    "fixed_0p0450": ("fixed_r_threshold_0p0450_2rank", "input_fixed_r_threshold_0p0450_2rank.used"),
    "fixed_0p0450_repeat": ("fixed_r_threshold_0p0450_repeat_2rank", "input_fixed_r_threshold_0p0450_2rank.used"),
    "fixed_0p0475": ("fixed_r_threshold_0p0475_2rank", "input_fixed_r_threshold_0p0475_2rank.used"),
    "table_trace": ("table_restart_trace_strict_2rank", "input_table_component_freeze_cfl030_strict_3cycles.used"),
    "fixed_0p0450_trace": ("fixed_r_0p0450_restart_trace_2rank", "input_fixed_r_threshold_0p0450_2rank.used"),
    "fixed_0p0400_trace": ("fixed_r_0p0400_restart_trace_2rank", "input_fixed_r_threshold_0p0400_2rank.used"),
    "table_deep": ("table_deep_arnoldi_strict_2rank", "input_table_component_freeze_cfl030_strict_3cycles.used"),
    "fixed_0p0450_deep": ("fixed_r_0p0450_deep_arnoldi_2rank", "input_fixed_r_threshold_0p0450_2rank.used"),
    "table_deep_binary_control": ("table_deep_binary_control_2rank", "input_table_component_freeze_cfl030_strict_3cycles.used"),
}
FG = re.compile(r"\[LINEAR_FGMRES\] iterations=(\d+) relative_residual=([^ ]+)")
ETA = re.compile(r"\[LINEAR_FGMRES_BACKWARD\] local_stencil_eta_max=([^ ]+)")
RESTART = re.compile(r"\[WOR92 FGMRES restart\] cycle=(\d+) total_iterations=(\d+) used=(\d+) arnoldi_relative=([^ ]+) true_relative=([^ ]+) true_to_arnoldi=([^\s]+)")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


runs, traces, inputs = [], [], set()
with tarfile.open(OUT / "evidence.tar.gz", "w:gz") as archive:
    for case, (dirname, inputname) in CASES.items():
        directory = ROOT / dirname
        inp = ROOT / inputname
        inputs.add(inp)
        for filename in ("run.log", "status.txt"):
            archive.add(directory / filename, arcname=f"runs/{case}/{filename}")
        log = (directory / "run.log").read_text(errors="replace")
        fgmres = [(int(i), float(r)) for i, r in FG.findall(log)]
        eta = [float(x) for x in ETA.findall(log)]
        runs.append({"case": case, "exit_code": int((directory / "status.txt").read_text().strip()),
                     "iterations": ";".join(str(i) for i, _ in fgmres),
                     "true_residuals": ";".join(f"{r:.16e}" for _, r in fgmres),
                     "local_stencil_eta": ";".join(f"{x:.16e}" for x in eta),
                     "input_sha256": sha(inp), "log_sha256": sha(directory / "run.log")})
        solve = 0
        prev_iteration = None
        for cycle, iteration, used, arnoldi, true, ratio in RESTART.findall(log):
            iteration = int(iteration)
            if prev_iteration is not None and iteration < prev_iteration:
                solve += 1
            traces.append({"case": case, "solve_index": solve, "cycle": int(cycle),
                           "iteration": iteration, "arnoldi_relative": float(arnoldi),
                           "true_relative": float(true), "true_to_arnoldi": float(ratio),
                           "inner_iterations": int(used)})
            prev_iteration = iteration
    for inp in sorted(inputs):
        archive.add(inp, arcname=f"inputs/{inp.name}")
    base = (BUILDS / "build_table_precision/src/linear_solver/linearMG/linearMG.cpp").read_text().splitlines(True)
    deep_text = (BUILDS / "build_table_restart_trace/src/linear_solver/linearMG/linearMG.cpp").read_text()
    old_break = """      if (relative_residual <= fgmres_threshold_\n          || hnext <= std::numeric_limits<Real>::epsilon()*bnorm) break;"""
    new_break = """      const Real arnoldi_target = (std::getenv("ATHENA_WOR92_FGMRES_DEEP_ARNOLDI")\n                                   != nullptr ? 0.1 : 1.0)*fgmres_threshold_;\n      if (relative_residual <= arnoldi_target\n          || hnext <= std::numeric_limits<Real>::epsilon()*bnorm) break;"""
    assert deep_text.count(new_break) == 1
    trace_text = deep_text.replace(new_break, old_break)
    traced = trace_text.splitlines(True)
    deep = deep_text.splitlines(True)
    trace_patch = "".join(difflib.unified_diff(base, traced, fromfile="base/linearMG.cpp", tofile="trace/linearMG.cpp"))
    (OUT / "trace_only.patch").write_text(trace_patch)
    archive.add(OUT / "trace_only.patch", arcname="trace_only.patch")
    deep_patch = "".join(difflib.unified_diff(base, deep, fromfile="base/linearMG.cpp", tofile="deep_arnoldi/linearMG.cpp"))
    (OUT / "diagnostic_and_trial.patch").write_text(deep_patch)
    archive.add(OUT / "diagnostic_and_trial.patch", arcname="diagnostic_and_trial.patch")

for filename, rows in (("runs.csv", runs), ("restarts.csv", traces)):
    with (OUT / filename).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

lookup = {r["case"]: r for r in runs}
summary = {
    "base_git_hash": "ebf7634a8e877edfd58724de1e3810638a76ca7e",
    "baseline_binary_sha256": sha(BUILDS / "build_table_precision/bin/athena"),
    "trace_binary_sha256": sha(BUILDS / "build_table_restart_trace/bin/athena_trace"),
    "deep_arnoldi_binary_sha256": sha(BUILDS / "build_table_restart_trace/bin/athena"),
    "fixed_0p0450_repeat_equal": (lookup["fixed_0p0450"]["iterations"] == lookup["fixed_0p0450_repeat"]["iterations"] and lookup["fixed_0p0450"]["true_residuals"] == lookup["fixed_0p0450_repeat"]["true_residuals"]),
    "deep_binary_control_equal": (lookup["table_trace"]["iterations"] == lookup["table_deep_binary_control"]["iterations"] and lookup["table_trace"]["true_residuals"] == lookup["table_deep_binary_control"]["true_residuals"]),
    "runs": runs,
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

fig, ax = plt.subplots(figsize=(8.1, 4.6))
for case, label, color in (("table_trace", "table, baseline", "#d95f02"),
                           ("table_deep", "table, deep Arnoldi", "#7570b3"),
                           ("fixed_0p0450_trace", "fixed R=0.045, baseline", "#1b9e77"),
                           ("fixed_0p0450_deep", "fixed R=0.045, deep Arnoldi", "#e7298a"),
                           ("fixed_0p0400_trace", "fixed R=0.040, passes", "#66a61e")):
    rows = [r for r in traces if r["case"] == case and r["solve_index"] == 1]
    if rows:
        ax.plot([r["iteration"] for r in rows], [r["true_relative"] for r in rows],
                label=label, color=color, linewidth=1.4)
        if case == "table_trace":
            ax.plot([r["iteration"] for r in rows], [r["arnoldi_relative"] for r in rows],
                    label="table Arnoldi estimate", color=color, linestyle="--", linewidth=1)
ax.axhline(2e-8, color="black", linestyle=":", label="strict target")
ax.set_yscale("log")
ax.set_xlabel("FGMRES iterations in second solve")
ax.set_ylabel("Relative residual at restart")
ax.grid(alpha=0.25)
ax.legend(fontsize=7, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "restart_residuals.png", dpi=160)
plt.close(fig)
print(json.dumps({"trace_rows": len(traces), "reproducible": summary["fixed_0p0450_repeat_equal"],
                  "control_equal": summary["deep_binary_control_equal"]}))
