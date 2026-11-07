#!/usr/bin/env python3
"""Summarize uniform-grid one-epsilon precision-floor trial."""

import csv
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
OUT = HERE / "wor92_uniform_floor1_results_20261107"
OUT.mkdir(exist_ok=True)
CASES = {
    "strict": ("table_uniform_l1_restart40_mg2_strict", "input_uniform_l1_restart40_mg2_strict.used"),
    "floor_1epsilon": ("uniform_l1_floor1_3cycles_2rank", "input_uniform_l1_floor1_3cycles.used"),
    "floor_1epsilon_10cycles": ("uniform_l1_floor1_10cycles_2rank", "input_uniform_l1_floor1_10cycles.used"),
}
FG = re.compile(r"\[LINEAR_FGMRES\] iterations=(\d+) relative_residual=([^ ]+)[^\n]*precision_floor_accept=(\d+)")
ETA = re.compile(r"\[LINEAR_FGMRES_BACKWARD\] local_stencil_eta_max=([^ ]+)")
NR = re.compile(r"\[NR\] status=converged iterations=(\d+)[^\n]*cycle=(\d+) time=([^ ]+) stage=(\d+) dt=([^\s]+)")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


solves, runs, inputs = [], [], set()
with tarfile.open(OUT / "evidence.tar.gz", "w:gz") as archive:
    for case, (dirname, inputname) in CASES.items():
        directory = ROOT / dirname
        inp = ROOT / inputname
        inputs.add(inp)
        for filename in ("run.log", "status.txt"):
            archive.add(directory / filename, arcname=f"runs/{case}/{filename}")
        log = (directory / "run.log").read_text(errors="replace")
        fg = FG.findall(log)
        eta = ETA.findall(log)
        assert len(fg) == len(eta)
        for index, ((iterations, residual, accepted), error) in enumerate(zip(fg, eta)):
            solves.append({"case": case, "solve_index": index, "iterations": int(iterations),
                           "true_relative_residual": float(residual),
                           "local_componentwise_eta": float(error),
                           "precision_floor_accept": bool(int(accepted))})
        nr = [{"cycle": int(cycle), "time": float(time), "dt": float(dt),
               "newton_iterations": int(iterations)}
              for iterations, cycle, time, _, dt in NR.findall(log)]
        runs.append({"case": case, "exit_code": int((directory / "status.txt").read_text().strip()),
                     "linear_solves": len(fg), "floor_acceptances": sum(int(x[2]) for x in fg),
                     "newton_converged_steps": len(nr), "newton_steps": nr,
                     "input_sha256": sha(inp), "log_sha256": sha(directory / "run.log")})
    for inp in sorted(inputs):
        archive.add(inp, arcname=f"inputs/{inp.name}")

with (OUT / "solves.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(solves[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(solves)
(OUT / "summary.json").write_text(json.dumps({
    "base_git_hash": "ebf7634a8e877edfd58724de1e3810638a76ca7e",
    "binary_sha256": sha(Path("/home/kosuke/simulation/test/rad_nr/wor92_operator_audit_20261107/build_table_precision/bin/athena")),
    "same_first_two_fgmres": [(x["iterations"], x["true_relative_residual"]) for x in solves if x["case"] == "strict"] == [(x["iterations"], x["true_relative_residual"]) for x in solves if x["case"] == "floor_1epsilon"][:2],
    "ten_cycle_first_six_equal": [(x["iterations"], x["true_relative_residual"], x["local_componentwise_eta"]) for x in solves if x["case"] == "floor_1epsilon"] == [(x["iterations"], x["true_relative_residual"], x["local_componentwise_eta"]) for x in solves if x["case"] == "floor_1epsilon_10cycles"][:6],
    "runs": runs,
    "limitation": "Exploratory acceptance policy; normwise target is not met for ten solves and no full protostellar collapse is validated"
}, indent=2) + "\n")

rows = [x for x in solves if x["case"] == "floor_1epsilon_10cycles"]
fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.5, 3.8))
color = ["#d95f02" if x["precision_floor_accept"] else "#1b9e77" for x in rows]
ax.scatter([x["solve_index"] + 1 for x in rows], [x["true_relative_residual"] for x in rows], c=color)
bx.scatter([x["solve_index"] + 1 for x in rows], [x["local_componentwise_eta"] for x in rows], c=color)
ax.axhline(2e-8, color="black", linestyle="--", linewidth=1, label="strict target")
bx.axhline(2.220446049250313e-16, color="black", linestyle="--", linewidth=1, label="one epsilon")
for axes in (ax, bx):
    axes.set_yscale("log")
    axes.set_xlabel("Linear solve")
    axes.grid(alpha=0.2)
ax.set_ylabel("True relative residual")
bx.set_ylabel("Componentwise backward error")
ax.legend(fontsize=8)
bx.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "uniform_floor1.png", dpi=160)
plt.close(fig)
print(json.dumps({"solves": len(solves), "runs": runs}, indent=2))
