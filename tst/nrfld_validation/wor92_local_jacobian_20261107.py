#!/usr/bin/env python3
"""Summarize the trial-only local gas/radiation Jacobian probe for WOR-92."""
from __future__ import annotations

import csv
import difflib
import hashlib
import json
from pathlib import Path
import re
import tarfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/kosuke/simulation/test/rad_nr/wor92_wor105_retest_20261107")
AUDIT = Path("/home/kosuke/simulation/test/rad_nr/wor92_operator_audit_20261107")
OUT = Path(__file__).with_name("wor92_local_jacobian_results_20261107")
OUT.mkdir(exist_ok=True)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


CASES = [
    ("strict_failure", "table_local_jacobian_cfl030_strict_2rank",
     "input_table_local_jacobian_cfl030_strict_3cycles.used"),
    ("floor32_10cycles", "table_local_jacobian_floor32_cfl030_10cycles_2rank",
     "input_table_local_jacobian_floor32_cfl030_10cycles.used"),
]
rows, runs, files = [], [], []
for label, directory, input_name in CASES:
    run = ROOT / directory
    s = (run / "run.log").read_text(errors="replace").replace("\x00", "")
    for line in re.findall(r"^\[NRFLD Jacobian\][^\n]*", s, re.M):
        p = dict(re.findall(r"(\w+)=([^\s]+)", line))
        rows.append(dict(label=label, cycle=int(p["cycle"]), gid=int(p["gid"]),
                         epsilon=float(p["epsilon"]), cells=int(p["cells"]),
                         gas_l2_rel=float(p["gas_l2_rel"]),
                         gas_max_rel=float(p["gas_max_rel"]),
                         radiation_l2_rel=float(p["radiation_l2_rel"]),
                         radiation_max_rel=float(p["radiation_max_rel"])))
    linear = re.findall(r"^\[LINEAR_FGMRES\].*", s, re.M)
    runs.append(dict(label=label, exit_code=int((run / "status.txt").read_text()),
                     jacobian_records=sum(x["label"] == label for x in rows),
                     linear_solves=len(linear),
                     newton_converged=len(re.findall(r"^\[NR\] status=converged", s, re.M)),
                     linear_lines_sha256=hashlib.sha256("\n".join(linear).encode()).hexdigest(),
                     input_sha256=sha(ROOT / input_name),
                     log_sha256=sha(run / "run.log")))
    files += [ROOT / input_name, run / "run.log", run / "status.txt"]
for name, data in (("jacobian.csv", rows), ("runs.csv", runs)):
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)

before = AUDIT / "build_table_precision/src/nr_multigrid/NRFLD.cpp"
after = AUDIT / "build_table_jacobian_diag/src/nr_multigrid/NRFLD.cpp"
patch = "".join(difflib.unified_diff(before.read_text().splitlines(keepends=True),
                                     after.read_text().splitlines(keepends=True),
                                     fromfile="src/nr_multigrid/NRFLD.cpp",
                                     tofile="src/nr_multigrid/NRFLD.cpp (diagnostic hook)"))
(OUT / "local_jacobian_diagnostic.patch").write_text(patch)
baseline = ROOT / "table_mixed_amr_cfl030_floor32_10cycles/run.log"
baseline_lines = re.findall(r"^\[LINEAR_FGMRES\].*", baseline.read_text(errors="replace"), re.M)
baseline_hash = hashlib.sha256("\n".join(baseline_lines).encode()).hexdigest()
payload = dict(base_commit="ebf7634a8e877edfd58724de1e3810638a76ca7e",
               diagnostic_binary_sha256=sha(AUDIT / "build_table_jacobian_diag/bin/athena"),
               diagnostic_patch_sha256=sha(OUT / "local_jacobian_diagnostic.patch"),
               baseline_floor32_linear_lines_sha256=baseline_hash,
               runs=runs,
               max_gas_l2_rel_by_epsilon={str(h): max(x["gas_l2_rel"] for x in rows if x["epsilon"] == h)
                                          for h in sorted({x["epsilon"] for x in rows})},
               max_radiation_l2_rel_by_epsilon={str(h): max(x["radiation_l2_rel"] for x in rows if x["epsilon"] == h)
                                                for h in sorted({x["epsilon"] for x in rows})})
(OUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
fig, ax = plt.subplots(figsize=(6, 3.5))
for field, marker in (("gas_l2_rel", "o"), ("radiation_l2_rel", "s")):
    for label, alpha in (("strict_failure", 1), ("floor32_10cycles", .5)):
        subset = [x for x in rows if x["label"] == label]
        ax.scatter([x["epsilon"] for x in subset], [x[field] for x in subset],
                   marker=marker, alpha=alpha, s=24, label=f"{field}, {label}")
ax.set(xlabel="Relative perturbation h", ylabel="FD versus local Jv L2 relative error",
       xscale="log", yscale="log")
ax.legend(fontsize=6)
fig.tight_layout()
fig.savefig(OUT / "local_jacobian.png", dpi=160)
with tarfile.open(OUT / "evidence.tar.gz", "w:gz") as archive:
    for path in files + [OUT / "local_jacobian_diagnostic.patch", baseline]:
        archive.add(path, arcname=path.relative_to(ROOT) if path.is_relative_to(ROOT)
                    else Path("diagnostic") / path.name)
print(json.dumps(payload, indent=2))
