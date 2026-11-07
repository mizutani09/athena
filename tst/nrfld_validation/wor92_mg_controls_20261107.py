#!/usr/bin/env python3
"""Summarize bounded WOR-92 AMR multigrid parameter controls."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import tarfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/kosuke/simulation/test/rad_nr/wor92_wor105_retest_20261107")
OUT = Path(__file__).with_name("wor92_mg_controls_results_20261107")
OUT.mkdir(exist_ok=True)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


CASES = [
    ("baseline", "component_coefficient_diagnostics_unfrozen_2rank",
     "input_table_component_coefficient_diagnostics_cfl030_strict_3cycles.used", 300),
    ("constant_prolongation", "table_constant_prolongation_cfl030_strict_2rank",
     "input_table_constant_prolongation_cfl030_strict_3cycles.used", 120),
    ("coarse_scale_0.5", "table_coarse_scale_half_cfl030_strict_2rank",
     "input_table_coarse_scale_half_cfl030_strict_3cycles.used", 120),
    ("sweeps_4_4", "table_sweeps4_cfl030_strict_2rank",
     "input_table_sweeps4_cfl030_strict_3cycles.used", 300),
]
rows = []
archive_files = []
for label, directory, input_name, timeout in CASES:
    run = ROOT / directory
    s = (run / "run.log").read_text(errors="replace").replace("\x00", "")
    linear = re.findall(r"^\[LINEAR_FGMRES\][^\n]*", s, re.M)
    mg = re.findall(r"^\[NRFLD MG transfer\] vcycle=1 [^\n]*", s, re.M)
    def metric(line: str, name: str):
        m = re.search(r"\b" + name + r"=([-+\d.eE]+)", line)
        return float(m.group(1)) if m else None
    cycles = [int(x) for x in re.findall(r"^cycle=(\d+) time=", s, re.M)]
    rows.append(dict(label=label, timeout_seconds=timeout,
                     exit_code=int((run / "status.txt").read_text()),
                     completed_cycles=max(cycles), linear_solves=len(linear),
                     first_iterations=int(metric(linear[0], "iterations")) if linear else None,
                     first_true_residual=metric(linear[0], "relative_residual") if linear else None,
                     second_true_residual=metric(linear[1], "relative_residual") if len(linear) > 1 else None,
                     first_vcycle_correction_factor=metric(mg[0], "correction_factor") if mg else None,
                     first_vcycle_full_factor=metric(mg[0], "full_factor") if mg else None,
                     input_sha256=sha(ROOT / input_name),
                     log_sha256=sha(run / "run.log")))
    archive_files += [ROOT / input_name, run / "run.log", run / "status.txt"]

with (OUT / "runs.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
(OUT / "summary.json").write_text(json.dumps(dict(base_commit="ebf7634a8e877edfd58724de1e3810638a76ca7e",
                                                   cases=rows, skipped="sweeps_8_8 after sweeps_4_4 timeout"),
                                             indent=2) + "\n")
fig, ax = plt.subplots(figsize=(6, 3.5))
for n, row in enumerate(rows[:3]):
    ax.bar(n-.15, row["first_true_residual"], width=.3, label="first FGMRES" if n == 0 else None)
    if row["second_true_residual"] is not None:
        ax.bar(n+.15, row["second_true_residual"], width=.3,
               label="second FGMRES" if n == 0 else None)
ax.axhline(2e-8, color="black", linestyle="--", linewidth=1, label="strict target")
ax.set_xticks(range(3), ["baseline", "constant\nprolongation", "coarse scale\n0.5"])
ax.set_ylabel("True relative residual")
ax.set_yscale("log")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "mg_controls.png", dpi=160)
with tarfile.open(OUT / "evidence.tar.gz", "w:gz") as archive:
    for path in archive_files:
        archive.add(path, arcname=path.relative_to(ROOT))
print(json.dumps(rows, indent=2))
