#!/usr/bin/env python3
"""Archive isolated coarse-fine transfer consistency controls for WOR-92."""

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
OUT = HERE / "wor92_transfer_consistency_results_20261107"
OUT.mkdir(exist_ok=True)
CASES = {
    "table_baseline": ("table_coupled_global_jv_v3_tangent_on_strict_2rank", "input_table_component_freeze_cfl030_strict_3cycles.used"),
    "fixed_baseline": ("table_coupled_global_jv_v3_fixed_rosseland_high_2rank", "input_table_fixed_rosseland_high_cfl030_strict_3cycles.used"),
    "table_unweighted_delta": ("table_unweighted_delta_jv_2rank", "input_table_component_freeze_cfl030_strict_3cycles.used"),
    "fixed_unweighted_delta": ("fixed_high_unweighted_delta_jv_2rank", "input_table_fixed_rosseland_high_cfl030_strict_3cycles.used"),
    "table_weighted_state": ("table_weighted_state_jv_2rank", "input_table_component_freeze_cfl030_strict_3cycles.used"),
    "fixed_weighted_state": ("fixed_high_weighted_state_jv_2rank", "input_table_fixed_rosseland_high_cfl030_strict_3cycles.used"),
    "unweighted_binary_control": ("table_unweighted_binary_control_2rank", "input_table_component_freeze_cfl030_strict_3cycles.used"),
    "weighted_binary_control": ("table_weighted_state_binary_control_2rank", "input_table_component_freeze_cfl030_strict_3cycles.used"),
}
JVPAT = re.compile(r"\[WOR92 coupled global Jv\] cycle=0 epsilon=([^ ]+) region=coarse_fine_face cells=(\d+) l2_rel=([^ ]+) tangent_l2_rel=([^ ]+) ghost_l2_rel=([^ ]+) max_abs=([^\s]+)")
FGPAT = re.compile(r"\[LINEAR_FGMRES\] iterations=(\d+) relative_residual=([^ ]+)")
NRPAT = re.compile(r"\[NR\] status=([^ ]+) iterations=(\d+)")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


runs, inputs = [], set()
with tarfile.open(OUT / "evidence.tar.gz", "w:gz") as archive:
    for case, (dirname, inputname) in CASES.items():
        directory = ROOT / dirname
        inp = ROOT / inputname
        inputs.add(inp)
        for filename in ("run.log", "status.txt"):
            archive.add(directory / filename, arcname=f"runs/{case}/{filename}")
        log = (directory / "run.log").read_text(errors="replace")
        jv = next((x for x in JVPAT.findall(log) if abs(float(x[0]) - 1e-5) < 1e-12), None)
        fg = FGPAT.findall(log)
        nr = NRPAT.findall(log)
        runs.append({"case": case, "exit_code": int((directory / "status.txt").read_text().strip()),
                     "coarse_fine_cells": int(jv[1]) if jv else None,
                     "coarse_fine_jv_l2_rel": float(jv[2]) if jv else None,
                     "ghost_tangent_l2_rel": float(jv[4]) if jv else None,
                     "fgmres_iterations": ";".join(i for i, _ in fg),
                     "fgmres_true_residuals": ";".join(r for _, r in fg),
                     "newton_statuses": ";".join(status for status, _ in nr if status != "dt_zero_initialization"),
                     "input_sha256": sha(inp), "log_sha256": sha(directory / "run.log")})
    for inp in sorted(inputs):
        archive.add(inp, arcname=f"inputs/{inp.name}")
    base = (BUILDS / "build_table_coupled_jv/src/bvals/cc/bvals_cc.cpp").read_text().splitlines(True)
    for name, build in (("unweighted_delta", "build_table_unweighted_delta"),
                        ("weighted_state", "build_table_weighted_state")):
        trial = (BUILDS / build / "src/bvals/cc/bvals_cc.cpp").read_text().splitlines(True)
        patch = OUT / f"{name}.patch"
        patch.write_text("".join(difflib.unified_diff(base, trial,
                         fromfile="base/bvals_cc.cpp", tofile=f"{name}/bvals_cc.cpp")))
        archive.add(patch, arcname=f"patches/{patch.name}")

with (OUT / "runs.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(runs[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(runs)
lookup = {r["case"]: r for r in runs}
summary = {
    "base_git_hash": "ebf7634a8e877edfd58724de1e3810638a76ca7e",
    "baseline_diagnostic_binary_sha256": sha(BUILDS / "build_table_coupled_jv/bin/athena"),
    "unweighted_delta_binary_sha256": sha(BUILDS / "build_table_unweighted_delta/bin/athena"),
    "weighted_state_binary_sha256": sha(BUILDS / "build_table_weighted_state/bin/athena"),
    "unweighted_flag_off_equal": all(lookup["table_baseline"][k] == lookup["unweighted_binary_control"][k] for k in ("fgmres_iterations", "fgmres_true_residuals")),
    "weighted_flag_off_equal": all(lookup["table_baseline"][k] == lookup["weighted_binary_control"][k] for k in ("fgmres_iterations", "fgmres_true_residuals")),
    "runs": runs,
    "limitations": ["three-cycle fixed-R control only", "no long collapse or conservation regression for the trial transfer", "table-R still fails strict FGMRES"],
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

fig, ax = plt.subplots(figsize=(7.3, 4.2))
variants = ["baseline", "unweighted_delta", "weighted_state"]
for offset, family, label, color in ((-0.12, "table", "table Rosseland", "#d95f02"),
                                     (0.12, "fixed", "fixed Rosseland 0.0652", "#1b9e77")):
    ax.scatter([i + offset for i in range(3)],
               [lookup[f"{family}_{v}"]["coarse_fine_jv_l2_rel"] for v in variants],
               label=label, color=color, s=70)
ax.set_yscale("log")
ax.set_xticks(range(3), ["baseline", "unweighted correction", "weighted state"])
ax.set_ylabel("Coarse-fine Schur Jv relative L2 error")
ax.grid(axis="y", alpha=0.25)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "transfer_jv.png", dpi=160)
plt.close(fig)
print(json.dumps({"runs": len(runs), "unweighted_off_equal": summary["unweighted_flag_off_equal"],
                  "weighted_off_equal": summary["weighted_flag_off_equal"]}))
