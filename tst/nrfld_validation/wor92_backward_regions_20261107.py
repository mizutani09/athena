#!/usr/bin/env python3
"""Summarize final FGMRES backward-error proxy by AMR row region."""

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
OUT = HERE / "wor92_backward_regions_results_20261107"
OUT.mkdir(exist_ok=True)
CASES = {
    "table_amr": ("table_backward_regions_2rank", "input_table_component_freeze_cfl030_strict_3cycles.used", "table_coupled_global_jv_v3_tangent_on_strict_2rank"),
    "uniform_level1": ("uniform_l1_backward_regions_2rank", "input_uniform_l1_restart40_mg2_strict.used", "table_uniform_l1_restart40_mg2_strict"),
    "fixed_high_amr": ("fixed_high_backward_regions_2rank", "input_table_fixed_rosseland_high_cfl030_strict_3cycles.used", "table_coupled_global_jv_v3_fixed_rosseland_high_2rank"),
}
PAT = re.compile(r"\[WOR92 backward region\] cycle=(\d+) solve_iterations=(\d+) region=([^ ]+) cells=(\d+) eta_max=([^ ]+) residual_l2=([^ ]+) denominator_l2=([^\s]+)")
FG = re.compile(r"\[LINEAR_FGMRES\] iterations=(\d+) relative_residual=([^ ]+)")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


rows, runs, inputs = [], [], set()
with tarfile.open(OUT / "evidence.tar.gz", "w:gz") as archive:
    for case, (dirname, inputname, comparator) in CASES.items():
        directory = ROOT / dirname
        inp = ROOT / inputname
        inputs.add(inp)
        for filename in ("run.log", "status.txt"):
            archive.add(directory / filename, arcname=f"runs/{case}/{filename}")
        content = (directory / "run.log").read_text(errors="replace")
        fg = FG.findall(content)
        original = FG.findall((ROOT / comparator / "run.log").read_text(errors="replace"))
        runrows = PAT.findall(content)
        if len(runrows) != 3 * len(fg) - (len(fg) if case == "uniform_level1" else 0):
            raise ValueError(f"unexpected region count in {case}")
        solve = -1
        prior = None
        for cycle, iterations, region, cells, eta, residual, denominator in runrows:
            marker = (cycle, iterations)
            if marker != prior:
                solve += 1
                prior = marker
            rows.append({"case": case, "solve_index": solve, "cycle": int(cycle),
                         "iterations": int(iterations), "region": region, "cells": int(cells),
                         "eta_max": float(eta), "residual_l2": float(residual),
                         "denominator_l2": float(denominator)})
        runs.append({"case": case, "exit_code": int((directory / "status.txt").read_text().strip()),
                     "fgmres_iterations": ";".join(i for i, _ in fg),
                     "fgmres_true_residuals": ";".join(r for _, r in fg),
                     "comparator_fgmres_equal": fg == original,
                     "input_sha256": sha(inp), "log_sha256": sha(directory / "run.log")})
    for inp in sorted(inputs):
        archive.add(inp, arcname=f"inputs/{inp.name}")
    base = (BUILDS / "build_table_precision/src/linear_solver/linearMG/linearMG.cpp").read_text().splitlines(True)
    trial = (BUILDS / "build_table_backward_regions/src/linear_solver/linearMG/linearMG.cpp").read_text().splitlines(True)
    patch = OUT / "diagnostic.patch"
    patch.write_text("".join(difflib.unified_diff(base, trial,
                     fromfile="base/linearMG.cpp", tofile="diagnostic/linearMG.cpp")))
    archive.add(patch, arcname="diagnostic.patch")

for filename, values in (("regions.csv", rows), ("runs.csv", runs)):
    with (OUT / filename).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)
summary = {"base_git_hash": "ebf7634a8e877edfd58724de1e3810638a76ca7e",
           "diagnostic_binary_sha256": sha(BUILDS / "build_table_backward_regions/bin/athena"),
           "runs": runs,
           "interpretation": "Interior and same-level local rows have direct ghost copies; coarse-fine local eta does not expand interpolation weights and is not assembled-global componentwise backward error"}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

fig, ax = plt.subplots(figsize=(7.4, 4.1))
labels = [("table_amr", "table AMR"), ("fixed_high_amr", "fixed R AMR"),
          ("uniform_level1", "uniform L1")]
regions = [("interior", "interior", "o"),
           ("same_level_face", "same-level face", "s"),
           ("coarse_fine_face", "coarse-fine face", "^")]
colors = {"table_amr": "#d95f02", "fixed_high_amr": "#1b9e77", "uniform_level1": "#7570b3"}
for ci, (case, title) in enumerate(labels):
    for ri, (region, rtitle, marker) in enumerate(regions):
        row = next((r for r in rows if r["case"] == case and r["solve_index"] == 1 and r["region"] == region), None)
        if row:
            ax.scatter(ci + (ri - 1) * 0.16, row["eta_max"], marker=marker,
                       color=colors[case], s=70, label=rtitle if ci == 0 else None)
ax.axhline(2.220446049250313e-16, color="black", linestyle="--", linewidth=1, label="double epsilon")
ax.set_yscale("log")
ax.set_xticks(range(3), [title for _, title in labels])
ax.set_ylabel("Max local-stencil componentwise backward error")
ax.grid(axis="y", alpha=0.25)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "backward_regions.png", dpi=160)
plt.close(fig)
print(json.dumps({"regions": len(rows), "runs": runs}, indent=2))
