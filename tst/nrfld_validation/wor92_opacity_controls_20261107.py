#!/usr/bin/env python3
"""Summarize WOR-92 EOS/opacity controls at CFL 0.05 and 0.3."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import tarfile

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/kosuke/simulation/test/rad_nr/wor92_wor105_retest_20261107")
BUNDLE = Path("/home/kosuke/simulation/test/rad_nr/wor92_20261107/bundle")
OUT = Path(__file__).with_name("wor92_opacity_controls_results_20261107")
AUDIT = Path("/home/kosuke/simulation/test/rad_nr/wor92_operator_audit_20261107")

# label, directory, input, EOS, opacity selection, CFL, requested cycles
CASES = [
    ("table_semenov_005", "table_mixed_amr_cfl005_restart40_mg2_strict_10cycles", "input_mixed_amr_cfl005_restart40_mg2_strict_10cycles.used", "table", "Semenov", .05, 10),
    ("ideal_constant1_005", "ideal_cfl005_restart40_mg2_strict_10cycles", "input_ideal_cfl005_restart40_mg2_strict_10cycles.used", "ideal", "P=1 R=1", .05, 10),
    ("table_constant1_005", "table_constant_opacity_cfl005_strict_10cycles", "input_table_constant_opacity_cfl005_strict_10cycles.used", "table", "P=1 R=1", .05, 10),
    ("table_matched_005", "table_matched_constant_opacity_cfl005_strict_10cycles", "input_table_matched_constant_opacity_cfl005_strict_10cycles.used", "table", "P=.2152 R=.06478", .05, 10),
    ("table_low_R_005", "table_low_rosseland_cfl005_strict_10cycles", "input_table_low_rosseland_cfl005_strict_10cycles.used", "table", "P=1 R=.06478", .05, 10),
    ("table_low_P_005", "table_low_planck_cfl005_strict_10cycles", "input_table_low_planck_cfl005_strict_10cycles.used", "table", "P=.2152 R=1", .05, 10),
    ("table_semenov_030", "table_mixed_amr_tangent_restart40_mg2_strict", "input_mixed_amr_restart40_mg2_strict.used", "table", "Semenov", .3, 3),
    ("table_constant1_030", "table_constant1_cfl030_strict_3cycles", "input_table_constant1_cfl030_strict_3cycles.used", "table", "P=1 R=1", .3, 3),
    ("table_low_R_030", "table_low_rosseland_cfl030_strict_3cycles", "input_table_low_rosseland_cfl030_strict_3cycles.used", "table", "P=1 R=.06478", .3, 3),
    ("table_matched_030", "table_matched_constant_opacity_cfl030_strict_3cycles", "input_table_matched_constant_opacity_cfl030_strict_3cycles.used", "table", "P=.2152 R=.06478", .3, 3),
    ("table_exact_030", "table_exact_constant_opacity_cfl030_strict_3cycles", "input_table_exact_constant_opacity_cfl030_strict_3cycles.used", "table", "P=.21524823808128721 R=.064784772977394392", .3, 3),
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def value(line: str, key: str, cast=float):
    m = re.search(r"\b" + key + r"=([-+\d.eE]+)", line)
    return cast(m.group(1)) if m else None


opacity_file = BUNDLE / "data/opacity/protostellar_semenov2003_nrm_c_a.h5"
eos_file = BUNDLE / "data/eos/protostellar_tomida2015_material.tab"
rho0, t0 = 5.251585416e-18, 14.0
reference = {}
with h5py.File(opacity_file) as h:
    rho_axis = h["axes/log10_density"][:]
    temp_axis = h["axes/log10_temperature"][:]
    lr, lt = np.log10(rho0), np.log10(t0)
    i = int(np.clip(np.searchsorted(rho_axis, lr)-1, 0, len(rho_axis)-2))
    j = int(np.clip(np.searchsorted(temp_axis, lt)-1, 0, len(temp_axis)-2))
    fr = (lr-rho_axis[i])/(rho_axis[i+1]-rho_axis[i])
    ft = (lt-temp_axis[j])/(temp_axis[j+1]-temp_axis[j])
    for label, dataset in (("kappa_P", "log10_planck"),
                           ("kappa_R", "log10_rosseland")):
        a = h["kappa/" + dataset]
        logk = ((1-fr)*(1-ft)*a[i,j] + fr*(1-ft)*a[i+1,j]
                + (1-fr)*ft*a[i,j+1] + fr*ft*a[i+1,j+1])
        reference[label] = float(10**logk)

rows = []
for label, dirname, inp, eos, opacity, cfl, limit in CASES:
    case = ROOT / dirname
    s = (case / "run.log").read_text(errors="replace").replace("\x00", "")
    linear = re.findall(r"\[LINEAR_FGMRES\][^\n]*", s)
    residuals = [value(x, "relative_residual") for x in linear]
    newton = re.findall(r"\[NR\] status=converged[^\n]*", s)
    cycles = [int(x) for x in re.findall(r"^cycle=(\d+) time=", s, re.M)]
    dts = [float(x) for x in re.findall(r"^cycle=\d+ time=[-+\d.eE]+ dt=([-+\d.eE]+)", s, re.M)]
    row = dict(label=label, directory=str(case), input=str(ROOT / inp),
        input_sha256=sha(ROOT / inp), log_sha256=sha(case / "run.log"),
        exit_code=int((case / "status.txt").read_text()),
        eos=eos, opacity=opacity, cfl=cfl, cycle_limit=limit,
        cycle_reached=max(cycles), first_dt=dts[0],
        linear_solves=len(linear), newton_converged=len(newton),
        floor_acceptances=sum(value(x, "precision_floor_accept", int) for x in linear),
        max_true_residual=max(residuals), last_true_residual=residuals[-1],
        failed_iteration_count=value(linear[-1], "iterations", int)
            if (case / "status.txt").read_text().strip() != "0" else None,
        final_time=float(re.findall(r"^cycle=\d+ time=([-+\d.eE]+) dt=", s, re.M)[-1]))
    rows.append(row)

OUT.mkdir(exist_ok=True)
payload = dict(base_commit="ebf7634a8e877edfd58724de1e3810638a76ca7e",
    trial_patch_sha256=sha(ROOT / "source_integration.patch"),
    table_binary_sha256="cf1e1520ca23334a08615439d8c616b6e88551801ac64101cd6c454f63d3d27a",
    ideal_binary_sha256="ff9887c40d80c1a2a64a1a77bb104992298b917dd28921ca84c37f1db89877c0",
    eos_sha256=sha(eos_file), opacity_sha256=sha(opacity_file),
    reference_state=dict(rho_cgs=rho0, T_K=t0, **reference), runs=rows)
(OUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
fields = ["label", "eos", "opacity", "cfl", "cycle_limit", "cycle_reached",
          "exit_code", "linear_solves", "newton_converged", "floor_acceptances",
          "first_dt", "final_time", "max_true_residual", "last_true_residual",
          "failed_iteration_count", "input_sha256", "log_sha256"]
with (OUT / "runs.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    w.writerows({k:r[k] for k in fields} for r in rows)

fig, axes = plt.subplots(1, 2, figsize=(12, 5.7), sharex=True)
for ax, cfl in zip(axes, (.05, .3)):
    subset = [r for r in rows if r["cfl"] == cfl]
    colors = ["#2d8c72" if r["exit_code"] == 0 else "#bc4b51" for r in subset]
    ax.barh([r["label"] for r in subset],
            [r["max_true_residual"] for r in subset], color=colors)
    ax.axvline(2e-8, color="black", linestyle="--")
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_title(f"CFL={cfl:g}")
    ax.set_xlabel("Maximum true relative residual")
fig.tight_layout()
fig.savefig(OUT / "opacity_controls.png", dpi=160)

with tarfile.open(OUT / "logs_inputs_tables.tar.gz", "w:gz") as archive:
    for _, dirname, inp, *_ in CASES:
        for path in (ROOT / inp, ROOT / dirname / "run.log", ROOT / dirname / "status.txt"):
            archive.add(path, arcname=str(path.relative_to(ROOT)))
    for path in (ROOT / "source_integration.patch", eos_file, opacity_file):
        archive.add(path, arcname="tables/" + path.name if path != ROOT / "source_integration.patch"
                    else path.name)

diag_dir = AUDIT / "opacity_range_cycle45"
by_cycle = {4: [], 5: []}
rank_logs = sorted(diag_dir.glob("range.rank*.log"))
if len(rank_logs) != 2:
    raise ValueError("expected two direct rank opacity logs")
for path in rank_logs:
    for line in path.read_text().splitlines():
        if not line.startswith("[WOR92_OPACITY_RANGE]"):
            raise ValueError(f"unexpected opacity line in {path}")
        tokens = dict(t.split("=", 1) for t in line.split() if "=" in t)
        cycle = int(tokens["cycle"])
        if cycle not in by_cycle:
            raise ValueError(f"unexpected cycle {cycle}")
        by_cycle[cycle].append(tokens)
ranges = []
for cycle, blocks in sorted(by_cycle.items()):
    if len(blocks) != 120 or len({int(x["gid"]) for x in blocks}) != 120:
        raise ValueError(f"incomplete opacity records for cycle {cycle}")
    ranges.append(dict(cycle=cycle, block_count=len(blocks),
        kp_min=min(float(x["kp_min"]) for x in blocks),
        kp_max=max(float(x["kp_max"]) for x in blocks),
        kr_min=min(float(x["kr_min"]) for x in blocks),
        kr_max=max(float(x["kr_max"]) for x in blocks),
        T_min=min(float(x["T_min"]) for x in blocks),
        T_max=max(float(x["T_max"]) for x in blocks),
        rho_min=min(float(x["rho_min"]) for x in blocks),
        rho_max=max(float(x["rho_max"]) for x in blocks)))
diag_payload = dict(base_commit="143b3403324db869b9690a2b33569252fdb5cb80",
    source_patch_sha256=sha(AUDIT / "source_opacity_range_trial.patch"),
    binary_sha256="5b726ae497c67d97f4615cb459b1877b629a4d0ed125df43bb829eefe63a352f",
    run_log_sha256=sha(diag_dir / "run.log"),
    rank_log_sha256={p.name:sha(p) for p in rank_logs}, ranges=ranges)
(OUT / "opacity_range_diagnostics.json").write_text(json.dumps(diag_payload, indent=2)+"\n")
with tarfile.open(OUT / "opacity_range_evidence.tar.gz", "w:gz") as archive:
    for path in rank_logs + [diag_dir / "run.log", diag_dir / "status.txt",
                            AUDIT / "source_opacity_range_trial.patch",
                            ROOT / "input_mixed_amr_cfl005_restart40_mg2_strict_10cycles.used"]:
        archive.add(path, arcname=path.name)
for r in rows:
    print(r["label"], r["exit_code"], r["cycle_reached"], r["max_true_residual"])
