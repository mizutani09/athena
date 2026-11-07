#!/usr/bin/env python3
"""Archive the isolated coupled Schur Jv checks for WOR-92."""

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
OUT = HERE / "wor92_coupled_global_jv_results_20261107"
OUT.mkdir(exist_ok=True)

CASES = {
    "table_tangent_on": ("table_coupled_global_jv_v3_tangent_on_strict_2rank", "input_table_component_freeze_cfl030_strict_3cycles.used"),
    "table_tangent_off": ("table_coupled_global_jv_v3_tangent_off_strict_2rank", "input_table_component_freeze_cfl030_strict_3cycles.used"),
    "fixed_rosseland_high": ("table_coupled_global_jv_v3_fixed_rosseland_high_2rank", "input_table_fixed_rosseland_high_cfl030_strict_3cycles.used"),
    "uniform_level1": ("table_coupled_global_jv_v3_uniform_l1_2rank", "input_uniform_l1_restart40_mg2_strict.used"),
    "tangent_off_diagnostic_off": ("table_coupled_global_jv_v3_tangent_off_diag_off_2rank", "input_table_component_freeze_cfl030_strict_3cycles.used"),
}
JVPAT = re.compile(r"\[WOR92 coupled global Jv\] cycle=(\d+) epsilon=([^ ]+) region=([^ ]+) cells=(\d+) l2_rel=([^ ]+) tangent_l2_rel=([^ ]+) ghost_l2_rel=([^ ]+) max_abs=([^\s]+)")
FGPAT = re.compile(r"\[LINEAR_FGMRES\] iterations=(\d+) relative_residual=([^ ]+)")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


jv = []
runs = []
inputs = set()
with tarfile.open(OUT / "evidence.tar.gz", "w:gz") as archive:
    for case, (directory, input_name) in CASES.items():
        log = ROOT / directory / "run.log"
        status = ROOT / directory / "status.txt"
        inp = ROOT / input_name
        inputs.add(inp)
        archive.add(log, arcname=f"runs/{case}/run.log")
        archive.add(status, arcname=f"runs/{case}/status.txt")
        content = log.read_text(errors="replace")
        fg = [(int(i), float(r)) for i, r in FGPAT.findall(content)]
        rows = JVPAT.findall(content)
        for cyc, eps, region, cells, err, tangent, ghost, maxabs in rows:
            jv.append({"case": case, "cycle": int(cyc), "epsilon": float(eps), "region": region,
                       "cells": int(cells), "l2_rel": float(err), "tangent_l2_rel": float(tangent),
                       "ghost_l2_rel": float(ghost), "max_abs": float(maxabs)})
        runs.append({"case": case, "exit_code": int(status.read_text().strip()),
                     "fgmres_solves": len(fg), "fgmres_iterations": ";".join(str(i) for i, _ in fg),
                     "fgmres_residuals": ";".join(f"{r:.16e}" for _, r in fg),
                     "input_sha256": sha(inp), "log_sha256": sha(log)})
    for inp in sorted(inputs):
        archive.add(inp, arcname=f"inputs/{inp.name}")
    patches = []
    for name in ("NRFLD.hpp", "NRFLD_jacobian_diagnostic.cpp"):
        before = BUILDS / "build_table_precision" / "src/nr_multigrid" / name
        after = BUILDS / "build_table_coupled_jv" / "src/nr_multigrid" / name
        patches.extend(difflib.unified_diff(before.read_text().splitlines(True), after.read_text().splitlines(True),
                     fromfile=f"base/{name}", tofile=f"diagnostic/{name}"))
    patch = OUT / "diagnostic.patch"
    patch.write_text("".join(patches))
    archive.add(patch, arcname="diagnostic.patch")

for filename, records in (("jv.csv", jv), ("runs.csv", runs)):
    with (OUT / filename).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

on = (ROOT / CASES["table_tangent_off"][0] / "run.log").read_text(errors="replace")
off = (ROOT / CASES["tangent_off_diagnostic_off"][0] / "run.log").read_text(errors="replace")
exact_fgmres = FGPAT.findall(on) == FGPAT.findall(off)
summary = {
    "base_git_hash": "ebf7634a8e877edfd58724de1e3810638a76ca7e",
    "diagnostic_binary_sha256": sha(BUILDS / "build_table_coupled_jv/bin/athena"),
    "diagnostic_isolation": "copied build tree; no production source change committed",
    "tangent_off_diagnostic_on_off_fgmres_equal": exact_fgmres,
    "runs": runs,
    "limitations": ["frozen face coefficients", "fixed gas state and local elimination ratio", "periodic physical boundaries", "diagnostic direction, not the actual FGMRES Krylov vector"],
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

selected = [r for r in jv if r["cycle"] == 0 and abs(r["epsilon"] - 1e-5) < 1e-12 and r["region"] in ("interior", "same_level_face", "coarse_fine_face")]
fig, ax = plt.subplots(figsize=(8.0, 4.4))
labels = ["table tangent on", "table tangent off", "fixed Rosseland", "uniform L1"]
cases = ["table_tangent_on", "table_tangent_off", "fixed_rosseland_high", "uniform_level1"]
regions = [("interior", "Interior"), ("same_level_face", "Same-level face"), ("coarse_fine_face", "Coarse-fine face")]
for ci, case in enumerate(cases):
    for ri, (region, title) in enumerate(regions):
        row = next((r for r in selected if r["case"] == case and r["region"] == region), None)
        if row:
            ax.scatter(ci + (ri - 1) * 0.19, row["l2_rel"], s=65, marker=["o", "s", "^"][ri],
                       color=["#386cb0", "#7fc97f", "#f0027f"][ri], label=title if ci == 0 else None)
ax.set_yscale("log")
ax.set_ylabel("Relative L2 error of frozen Schur Jv")
ax.set_xticks(range(len(cases)), labels, rotation=13)
ax.grid(axis="y", alpha=0.25)
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "jv_regions.png", dpi=160)
plt.close(fig)
print(json.dumps({"rows": len(jv), "runs": runs, "on_off_equal": exact_fgmres}, indent=2))
