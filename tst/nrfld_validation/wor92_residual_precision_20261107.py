#!/usr/bin/env python3
"""Archive independent long-double residual evaluation for WOR-92."""

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
OUT = HERE / "wor92_residual_precision_results_20261107"
OUT.mkdir(exist_ok=True)
CASES = {
    "table": ("table_ld_residual_2rank", "input_table_component_freeze_cfl030_strict_3cycles.used"),
    "fixed_0p0450": ("fixed_r_0p0450_ld_residual_2rank", "input_fixed_r_threshold_0p0450_2rank.used"),
    "fixed_0p0400": ("fixed_r_0p0400_ld_residual_2rank", "input_fixed_r_threshold_0p0400_2rank.used"),
    "uniform_level1": ("uniform_l1_ld_residual_2rank", "input_uniform_l1_restart40_mg2_strict.used"),
    "table_diagnostic_off": ("table_ld_binary_control_2rank", "input_table_component_freeze_cfl030_strict_3cycles.used"),
}
PAT = re.compile(r"\[WOR92 long-double residual\] cycle=(\d+) iterations=(\d+) reported_relative=([^ ]+) long_double_relative=([^ ]+) cast_difference_relative=([^ ]+) local_eta_max=([^\s]+)")
FG = re.compile(r"\[LINEAR_FGMRES\] iterations=(\d+) relative_residual=([^ ]+)")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


rows, runs, inputs = [], [], set()
with tarfile.open(OUT / "evidence.tar.gz", "w:gz") as archive:
    for case, (dirname, inputname) in CASES.items():
        directory = ROOT / dirname
        inp = ROOT / inputname
        inputs.add(inp)
        for filename in ("run.log", "status.txt"):
            archive.add(directory / filename, arcname=f"runs/{case}/{filename}")
        log = (directory / "run.log").read_text(errors="replace")
        fgmres = FG.findall(log)
        runs.append({"case": case, "exit_code": int((directory / "status.txt").read_text().strip()),
                     "iterations": ";".join(i for i, _ in fgmres),
                     "true_residuals": ";".join(r for _, r in fgmres),
                     "input_sha256": sha(inp), "log_sha256": sha(directory / "run.log")})
        for solve, (cycle, iteration, reported, ld, cast, eta) in enumerate(PAT.findall(log)):
            rows.append({"case": case, "solve_index": solve, "cycle": int(cycle),
                         "iterations": int(iteration), "reported_relative": float(reported),
                         "long_double_relative": float(ld),
                         "absolute_relative_difference": abs(float(ld) - float(reported)),
                         "cast_difference_relative": float(cast), "local_eta_max": float(eta)})
    for inp in sorted(inputs):
        archive.add(inp, arcname=f"inputs/{inp.name}")
    base = (BUILDS / "build_table_precision/src/linear_solver/linearMG/linearMG.cpp").read_text().splitlines(True)
    trial = (BUILDS / "build_table_residual_precision/src/linear_solver/linearMG/linearMG.cpp").read_text().splitlines(True)
    (OUT / "diagnostic.patch").write_text("".join(difflib.unified_diff(base, trial,
                  fromfile="base/linearMG.cpp", tofile="diagnostic/linearMG.cpp")))
    archive.add(OUT / "diagnostic.patch", arcname="diagnostic.patch")

for filename, values in (("residuals.csv", rows), ("runs.csv", runs)):
    with (OUT / filename).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)

run_lookup = {x["case"]: x for x in runs}
summary = {
    "base_git_hash": "ebf7634a8e877edfd58724de1e3810638a76ca7e",
    "baseline_binary_sha256": sha(BUILDS / "build_table_precision/bin/athena"),
    "diagnostic_binary_sha256": sha(BUILDS / "build_table_residual_precision/bin/athena"),
    "diagnostic_on_off_fgmres_equal": (run_lookup["table"]["iterations"] == run_lookup["table_diagnostic_off"]["iterations"] and run_lookup["table"]["true_residuals"] == run_lookup["table_diagnostic_off"]["true_residuals"]),
    "runs": runs,
    "limitations": ["same synchronized correction ghosts as FGMRES", "frozen assembled coefficients", "local stencil eta does not expand AMR interpolation weights in the mixed-level cases; the uniform periodic level-1 case has direct same-level ghost copies"],
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.5, 3.8))
for case, label, color in (("table", "table", "#d95f02"),
                           ("fixed_0p0450", "fixed 0.045", "#7570b3"),
                           ("fixed_0p0400", "fixed 0.040", "#1b9e77"),
                           ("uniform_level1", "uniform L1", "#e7298a")):
    row = next(x for x in rows if x["case"] == case and x["solve_index"] == 1)
    ax.scatter(row["reported_relative"], row["long_double_relative"], color=color, label=label, s=60)
    bx.scatter(label, row["absolute_relative_difference"], color=color, s=60)
lo, hi = 1.5e-8, 1.0e-7
ax.plot([lo, hi], [lo, hi], color="black", linestyle=":", linewidth=1)
ax.axvline(2e-8, color="gray", linestyle="--", linewidth=1)
ax.axhline(2e-8, color="gray", linestyle="--", linewidth=1)
ax.set_xlabel("Reported true residual")
ax.set_ylabel("Long-double recomputation")
ax.legend(fontsize=8)
bx.set_yscale("log")
bx.set_ylabel("Absolute difference in relative residual")
bx.tick_params(axis="x", rotation=15)
fig.tight_layout()
fig.savefig(OUT / "residual_precision.png", dpi=160)
plt.close(fig)
print(json.dumps({"rows": len(rows), "on_off_equal": summary["diagnostic_on_off_fgmres_equal"]}))
