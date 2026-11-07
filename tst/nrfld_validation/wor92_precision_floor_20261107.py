#!/usr/bin/env python3
"""Summarize bounded ten-cycle WOR-92 FGMRES floor sensitivity runs."""
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
OUT = Path(__file__).with_name("wor92_precision_floor_results_20261107")
OUT.mkdir(exist_ok=True)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


CASES = [
    (16, "table_mixed_amr_cfl030_floor16_10cycles",
     "input_mixed_amr_cfl030_restart40_mg2_floor16_10cycles.used"),
    (32, "table_mixed_amr_cfl030_floor32_10cycles",
     "input_mixed_amr_cfl030_restart40_mg2_floor32_10cycles.used"),
]
runs, solves, files = [], [], []
for factor, directory, input_name in CASES:
    run = ROOT / directory
    s = (run / "run.log").read_text(errors="replace").replace("\x00", "")
    linear = re.findall(r"^\[LINEAR_FGMRES\][^\n]*", s, re.M)
    backward = re.findall(r"^\[LINEAR_FGMRES_BACKWARD\][^\n]*", s, re.M)
    assert len(linear) == len(backward)
    cycles = [int(x) for x in re.findall(r"^cycle=(\d+) time=", s, re.M)]
    dts = [float(x) for x in re.findall(r"^cycle=\d+ time=[-+\d.eE]+ dt=([-+\d.eE]+)", s, re.M)]
    for i, (line, eta_line) in enumerate(zip(linear, backward)):
        def number(key: str, text: str, cast=float):
            m = re.search(r"\b" + key + r"=([-+\d.eE]+)", text)
            return cast(m.group(1))
        solves.append(dict(factor=factor, solve=i,
                           iterations=number("iterations", line, int),
                           true_relative_residual=number("relative_residual", line),
                           local_stencil_eta_max=number("local_stencil_eta_max", eta_line),
                           precision_floor_accept=number("precision_floor_accept", line, int)))
    mine = [x for x in solves if x["factor"] == factor]
    runs.append(dict(factor=factor, exit_code=int((run / "status.txt").read_text()),
                     requested_cycles=10, last_cycle=max(cycles),
                     linear_solves=len(linear), newton_converged=len(re.findall(r"^\[NR\] status=converged", s, re.M)),
                     first_dt=dts[0], last_reported_dt=dts[-1],
                     max_true_relative_residual=max(x["true_relative_residual"] for x in mine),
                     floor_acceptances=sum(x["precision_floor_accept"] for x in mine),
                     input_sha256=sha(ROOT / input_name),
                     log_sha256=sha(run / "run.log")))
    files += [ROOT / input_name, run / "run.log", run / "status.txt"]

for name, data in (("runs.csv", runs), ("solves.csv", solves)):
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)
(OUT / "summary.json").write_text(json.dumps(dict(base_commit="ebf7634a8e877edfd58724de1e3810638a76ca7e",
                                                   runs=runs, solves=solves), indent=2) + "\n")
fig, ax = plt.subplots(figsize=(8, 3.5))
for factor, color in ((16, "tab:blue"), (32, "tab:orange")):
    subset = [x for x in solves if x["factor"] == factor]
    ax.plot([x["solve"] for x in subset],
            [x["true_relative_residual"] for x in subset], "o-", label=f"floor factor {factor}",
            color=color, markersize=3)
    accepted = [x for x in subset if x["precision_floor_accept"]]
    ax.scatter([x["solve"] for x in accepted],
               [x["true_relative_residual"] for x in accepted], color=color,
               edgecolor="black", s=65, zorder=3)
ax.axhline(2e-8, color="black", linestyle="--", linewidth=1, label="strict target")
ax.set(xlabel="FGMRES solve index", ylabel="True relative residual", yscale="log")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "floor_sensitivity.png", dpi=160)
with tarfile.open(OUT / "evidence.tar.gz", "w:gz") as archive:
    for path in files:
        archive.add(path, arcname=path.relative_to(ROOT))
print(json.dumps(runs, indent=2))
