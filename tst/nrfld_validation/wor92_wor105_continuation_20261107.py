#!/usr/bin/env python3
"""Summarize the local WOR-92 continuation runs without hiding failed solves."""
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
AUDIT = Path("/home/kosuke/simulation/test/rad_nr/wor92_operator_audit_20261107")
OUT = Path(__file__).with_name("wor92_wor105_continuation_results_20261107")

# Label, directory, EOS, CFL, restart, V-cycles, floor factor, tangent, ranks, input.
CASES = [
    ("ideal_r10_mg1_on", "ideal_mixed_amr_tangent", "ideal", .3, 10, 1, 16, 1, 2, "input_ideal_mixed_amr.used"),
    ("ideal_r10_mg1_off", "ideal_mixed_amr_no_tangent", "ideal", .3, 10, 1, 16, 0, 2, "input_ideal_mixed_amr.used"),
    ("ideal_r40_mg1_on", "ideal_mixed_amr_tangent_restart40_mg1", "ideal", .3, 40, 1, 16, 1, 2, "input_ideal_mixed_amr_restart40_mg1.used"),
    ("ideal_r40_mg2_on", "ideal_mixed_amr_tangent_restart40_mg2", "ideal", .3, 40, 2, 16, 1, 2, "input_ideal_mixed_amr_restart40_mg2.used"),
    ("table_r40_mg1_on", "table_mixed_amr_tangent_restart40", "table", .3, 40, 1, 16, 1, 2, "input_mixed_amr_restart40.used"),
    ("table_r40_mg2_on", "table_mixed_amr_tangent_restart40_mg2", "table", .3, 40, 2, 16, 1, 2, "input_mixed_amr_restart40_mg2.used"),
    ("table_r40_mg2_strict", "table_mixed_amr_tangent_restart40_mg2_strict", "table", .3, 40, 2, 0, 1, 2, "input_mixed_amr_restart40_mg2_strict.used"),
    ("table_r80_mg2_strict", "table_mixed_amr_tangent_restart80_mg2_strict", "table", .3, 80, 2, 0, 1, 2, "input_mixed_amr_restart80_mg2_strict.used"),
    ("table_r40_mg3_strict", "table_mixed_amr_tangent_restart40_mg3_strict", "table", .3, 40, 3, 0, 1, 2, "input_mixed_amr_restart40_mg3_strict.used"),
    ("table_unbatched_strict", "table_mixed_amr_tangent_restart40_mg2_strict_unbatched", "table", .3, 40, 2, 0, 1, 2, "input_mixed_amr_restart40_mg2_strict_unbatched.used"),
    ("table_no_mixed_strict", "table_mixed_amr_tangent_restart40_mg2_strict_mixed_off", "table", .3, 40, 2, 0, 1, 2, "input_mixed_amr_restart40_mg2_strict_mixed_off.used"),
    ("table_original_matvec_strict", "table_mixed_amr_tangent_restart40_mg2_strict_original_matvec", "table", .3, 40, 2, 0, 1, 2, "input_mixed_amr_restart40_mg2_strict_original_matvec.used"),
    ("table_cfl01_strict", "table_mixed_amr_cfl01_restart40_mg2_strict", "table", .1, 40, 2, 0, 1, 2, "input_mixed_amr_cfl01_restart40_mg2_strict.used"),
    ("table_cfl005_strict", "table_mixed_amr_cfl005_restart40_mg2_strict", "table", .05, 40, 2, 0, 1, 2, "input_mixed_amr_cfl005_restart40_mg2_strict.used"),
    ("table_cfl005_4rank_strict", "table_mixed_amr_cfl005_restart40_mg2_strict_4rank", "table", .05, 40, 2, 0, 1, 4, "input_mixed_amr_cfl005_restart40_mg2_strict.used"),
    ("table_uniform_l1_strict", "table_uniform_l1_restart40_mg2_strict", "table", .3, 40, 2, 0, 1, 2, "input_uniform_l1_restart40_mg2_strict.used"),
    ("table_local_gas_off_strict", "table_mixed_amr_restart40_mg2_strict_local_off", "table", .3, 40, 2, 0, 1, 2, "input_mixed_amr_restart40_mg2_strict_local_off.used"),
    ("table_linearity_each_solve", "table_mixed_amr_restart40_mg2_strict_linearity", "table", .3, 40, 2, 0, 1, 2, "input_mixed_amr_restart40_mg2_strict_linearity.used"),
    ("table_max600_strict", "table_mixed_amr_restart40_mg2_strict_max600", "table", .3, 40, 2, 0, 1, 2, "input_mixed_amr_restart40_mg2_strict_max600.used"),
    ("table_resmin_prolongation", "table_mixed_amr_restart40_mg2_strict_resmin", "table", .3, 40, 2, 0, 1, 2, "input_mixed_amr_restart40_mg2_strict_resmin.used"),
]


def sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def field(line: str, name: str, cast=float):
    m = re.search(r"\b" + re.escape(name) + r"=([-+\d.eE]+)", line)
    return cast(m.group(1)) if m else None


rows = []
for label, dirname, eos, cfl, restart, mg, floor_factor, tangent, ranks, inp in CASES:
    base = ROOT / dirname
    log = base / "run.log"
    if not log.exists():
        continue
    s = log.read_text(errors="replace").replace("\x00", "")
    fgmres = re.findall(r"\[LINEAR_FGMRES\][^\n]*", s)
    res = [field(line, "relative_residual") for line in fgmres]
    floor = [field(line, "precision_floor_accept", int) for line in fgmres]
    iters = [field(line, "iterations", int) for line in fgmres]
    fatal_match = re.search(r"FGMRES did not reach its requested relative residual: "
                            r"iterations=(\d+), relative_residual=([-+\d.eE]+)", s)
    fatal_residual = float(fatal_match.group(2)) if fatal_match else None
    # MPI stdout can drop the final rank-0 summary during an abort. Preserve
    # the fatal value as a lower-precision fallback in that one case.
    if fatal_match and (not res or abs(res[-1]-fatal_residual)
                        > 1e-4*max(abs(fatal_residual), 1e-30)):
        res.append(fatal_residual)
        iters.append(int(fatal_match.group(1)))
        floor.append(0)
    newton = re.findall(r"\[NR\] status=converged[^\n]*", s)
    linearity = re.findall(r"\[LINEAR_FGMRES_LINEARITY\][^\n]*", s)
    cycles = [int(x) for x in re.findall(r"^cycle=(\d+) time=", s, re.M)]
    dt = [float(x) for x in re.findall(r"^cycle=\d+ time=[-+\d.eE]+ dt=([-+\d.eE]+)", s, re.M)]
    status = base / "status.txt"
    record = dict(label=label, directory=str(base), eos=eos, cfl=cfl,
                  restart=restart, mg_vcycles=mg, floor_factor=floor_factor,
                  tangent=bool(tangent), ranks=ranks, input=str(ROOT / inp),
                  input_sha256=sha(ROOT / inp), log_sha256=sha(log),
                  exit_code=int(status.read_text()) if status.exists() else None,
                  cycle_reached=max(cycles) if cycles else None,
                  first_dt=dt[0] if dt else None, fgmres_solves=len(res),
                  fatal_relative_residual=fatal_residual,
                  max_true_residual=max(res) if res else None,
                  final_true_residual=res[-1] if res else None,
                  failed_linear_solves=sum(x is not None and x > 2e-8 for x in res),
                  floor_acceptances=sum(x for x in floor if x is not None),
                  max_fgmres_iterations=max(iters) if iters else None,
                  strict_newton_solves=len(newton),
                  operator_additivity_errors=[field(x, "additivity_rel") for x in linearity],
                  operator_homogeneity_errors=[field(x, "homogeneity_rel") for x in linearity],
                  max_newton_final_l2=max((field(x, "final_l2") for x in newton), default=None))
    rows.append(record)

# The WOR-25 audit branch includes additional Galerkin diagnostics.
for label, name in (("audit_tangent_on", "tangent_on"), ("audit_tangent_off", "tangent_off")):
    s = (AUDIT / name / "run.log").read_text(errors="replace")
    amr = re.findall(r"\[NRFLD AMR Galerkin\][^\n]*", s)
    local = re.findall(r"\[NRFLD Galerkin\][^\n]*", s)
    rows.append(dict(label=label, directory=str(AUDIT / name), eos="ideal",
                     cfl=.3, restart=10, mg_vcycles=1, floor_factor=16,
                     tangent=name.endswith("on"), ranks=2,
                     input=str(AUDIT / "input_ideal_onecycle.used"),
                     input_sha256=sha(AUDIT / "input_ideal_onecycle.used"),
                     log_sha256=sha(AUDIT / name / "run.log"),
                     exit_code=int((AUDIT / name / "status.txt").read_text()),
                     amr_galerkin_mismatch=[field(x, "mismatch_relative_l2") for x in amr],
                     local_galerkin_mismatch=[field(x, "mismatch_relative_l2") for x in local],
                     amr_galerkin_samples=len(amr), local_galerkin_samples=len(local)))

OUT.mkdir(exist_ok=True)
(OUT / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
csv_fields = ["label", "eos", "cfl", "restart", "mg_vcycles", "floor_factor",
              "tangent", "ranks", "exit_code", "cycle_reached", "first_dt",
              "fgmres_solves", "failed_linear_solves", "floor_acceptances",
              "max_true_residual", "final_true_residual", "strict_newton_solves",
              "max_newton_final_l2", "fatal_relative_residual",
              "input_sha256", "log_sha256"]
with (OUT / "runs.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows({k: row.get(k) for k in csv_fields} for row in rows)

plot_rows = [r for r in rows if r.get("max_true_residual") is not None]
fig, ax = plt.subplots(figsize=(11, 7))
colors = ["#2d8c72" if r["exit_code"] == 0 and r["floor_acceptances"] == 0
          else "#e4a028" if r["exit_code"] == 0 else "#bc4b51" for r in plot_rows]
ax.barh([r["label"] for r in plot_rows],
        [max(r["max_true_residual"], 1e-12) for r in plot_rows], color=colors)
ax.axvline(2e-8, color="black", linestyle="--", label="FGMRES target 2e-8")
ax.set_xscale("log")
ax.invert_yaxis()
ax.set_xlabel("Maximum reported true relative residual (all solves)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "true_residual_matrix.png", dpi=160)

precise_cases = []
for name in ("precise_residual_on", "precise_residual_off", "residual_level_diag"):
    path = AUDIT / name / "run.log"
    s = path.read_text(errors="replace")
    precise = re.findall(r"\[LINEAR_FGMRES_PRECISE_RESIDUAL\][^\n]*", s)
    standard = re.findall(r"\[LINEAR_FGMRES\][^\n]*", s)
    levels = re.findall(r"\[LINEAR_FGMRES_RESIDUAL_LEVEL\][^\n]*", s)
    precise_cases.append(dict(case=name, log_sha256=sha(path),
        standard_true_residuals=[field(x, "relative_residual") for x in standard],
        precise_true_residuals=[field(x, "relative") for x in precise],
        rounded_ax_error_relative=[field(x, "rounded_ax_error_relative") for x in precise],
        level_diagnostics=[dict(level=field(x, "level", int),
            residual_relative_global=field(x, "residual_relative_global"),
            rhs_fraction=field(x, "rhs_fraction"),
            edge_fraction_of_residual=field(x, "edge_fraction_of_residual"),
            interior_fraction_of_residual=field(x, "interior_fraction_of_residual"))
            for x in levels]))
(OUT / "precision_diagnostics.json").write_text(
    json.dumps(dict(base_commit="143b3403324db869b9690a2b33569252fdb5cb80",
                    source_patch_sha256=sha(AUDIT / "source_precise_residual_trial.patch"),
                    level_source_patch_sha256=sha(AUDIT / "source_level_breakdown_trial.patch"),
                    initial_binary_sha256="591a2f79652ad57ba6c3a71b7137e4a104cf3e9921a4318723e1dd8fa6036452",
                    level_binary_sha256="d6adaf5d9e570ccdcfe7a8746ab81bf12d60716830bc24c93fe0d6913d8edc9f",
                    cases=precise_cases), indent=2) + "\n")

# Preserve the compact raw evidence with the pushed report. Large simulation
# output remains in the isolated local artifact directory.
with tarfile.open(OUT / "logs_inputs.tar.gz", "w:gz") as archive:
    paths = {ROOT / row[9] for row in CASES}
    paths.add(ROOT / "source_integration.patch")
    paths.add(AUDIT / "input_ideal_onecycle.used")
    paths.add(AUDIT / "source_trial.patch")
    paths.add(AUDIT / "source_precise_residual_trial.patch")
    paths.add(AUDIT / "source_level_breakdown_trial.patch")
    for row in rows:
        directory = Path(row["directory"])
        paths.add(directory / "run.log")
        paths.add(directory / "status.txt")
    paths.add(ROOT / "table_mixed_amr_tangent_restart40_mg2_ledger" / "run.log")
    paths.add(ROOT / "table_mixed_amr_tangent_restart40_mg2_ledger" /
              "ledger_audit" / "summary.json")
    for name in ("precise_residual_on", "precise_residual_off", "residual_level_diag"):
        paths.add(AUDIT / name / "run.log")
        paths.add(AUDIT / name / "status.txt")
    for path in sorted(paths):
        if path.exists():
            prefix = "audit" if AUDIT in path.parents else "trial"
            root = AUDIT if prefix == "audit" else ROOT
            archive.add(path, arcname=str(Path(prefix) / path.relative_to(root)))
print(OUT)
