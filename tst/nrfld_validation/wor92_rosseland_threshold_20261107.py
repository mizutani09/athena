#!/usr/bin/env python3
"""Summarize the local WOR-92 cycle-0 Rosseland and coefficient controls."""
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
OUT = Path(__file__).with_name("wor92_rosseland_threshold_results_20261107")
OUT.mkdir(exist_ok=True)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pairs(line: str) -> dict:
    return dict(re.findall(r"([A-Za-z_][A-Za-z_0-9]*)=([^\s]+)", line))


def csv_write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


CASES = [
    ("table_R", "component_coefficient_diagnostics_unfrozen_2rank",
     "input_table_component_coefficient_diagnostics_cfl030_strict_3cycles.used", None),
    ("fixed_R_0.0321456353", "component_fixed_rosseland_half_2rank",
     "input_table_fixed_rosseland_half_cfl030_strict_3cycles.used", .032145635298572645),
    ("fixed_R_0.05", "component_fixed_rosseland_mid_2rank",
     "input_table_fixed_rosseland_mid_cfl030_strict_3cycles.used", .05),
    ("fixed_R_0.0644", "component_fixed_rosseland_low_2rank",
     "input_table_fixed_rosseland_low_cfl030_strict_3cycles.used", .0644),
    ("fixed_R_0.064784773", "component_coefficient_diagnostics_fixed_rosseland_2rank",
     "input_table_component_coefficient_diagnostics_cfl030_strict_3cycles.used", .064784772977394392),
    ("fixed_R_0.0652", "component_fixed_rosseland_high_2rank",
     "input_table_fixed_rosseland_high_cfl030_strict_3cycles.used", .0652),
]
runs = []
coefficients = []
for label, directory, input_name, fixed_r in CASES:
    run = ROOT / directory
    lines = (run / "run.log").read_text(errors="replace").replace("\x00", "").splitlines()
    linear = [pairs(s) for s in lines if s.startswith("[LINEAR_FGMRES]")]
    cycles = [int(x) for x in re.findall(r"^cycle=(\d+) time=", "\n".join(lines), re.M)]
    runs.append(dict(label=label, fixed_rosseland=fixed_r, ranks=2, cfl=.3,
                     exit_code=int((run / "status.txt").read_text()),
                     completed_cycles=max(cycles), linear_solves=len(linear),
                     first_iterations=int(linear[0]["iterations"]),
                     first_true_residual=float(linear[0]["relative_residual"]),
                     second_iterations=int(linear[1]["iterations"]),
                     second_true_residual=float(linear[1]["relative_residual"]),
                     max_true_residual=max(float(x["relative_residual"]) for x in linear),
                     input_sha256=sha(ROOT / input_name),
                     log_sha256=sha(run / "run.log")))
    for s in lines:
        if not s.startswith("[NRFLD coefficients]"):
            continue
        p = pairs(s)
        if int(p["solve"]) not in (0, 1):
            continue
        coefficients.append(dict(label=label, solve=int(p["solve"]), kind=p["kind"],
                                 level=int(p["level"]), diffusion_diag=p["diffusion_diag"],
                                 face_abs=p["face_abs"], reaction_fraction=p["reaction_fraction"],
                                 dominance_min=float(p["dominance_min"]),
                                 row_error_max=float(p["row_error_max"]),
                                 nonpositive_diag=int(p["nonpositive_diag"]),
                                 positive_or_nonfinite_offdiag=int(p["positive_or_nonfinite_offdiag"])))
csv_write(OUT / "runs.csv", runs)
csv_write(OUT / "coefficients.csv", coefficients)

opacity = []
opacity_logs = []
for label in ("unfrozen", "fixed_rosseland"):
    run = ROOT / f"component_initial_opacity_allcalls_{label}_2rank"
    for log in sorted(run.glob("opacity.rank*.log")):
        opacity_logs.append(log)
        for s in log.read_text().splitlines():
            p = pairs(s)
            opacity.append(dict(label=label, call=int(p["call"]),
                                rank=int(p["rank"]), gid=int(p["gid"]),
                                level=int(p["level"]),
                                kappa_p_min=float(p["kp_min"]),
                                kappa_p_max=float(p["kp_max"]),
                                kappa_r_min=float(p["kr_min"]),
                                kappa_r_max=float(p["kr_max"]),
                                temperature_min=float(p["T_min"]),
                                temperature_max=float(p["T_max"]),
                                rho_min=float(p["rho_min"]),
                                rho_max=float(p["rho_max"])))
csv_write(OUT / "opacity_calls.csv", opacity)
calls = []
for label in ("unfrozen", "fixed_rosseland"):
    for call in sorted({x["call"] for x in opacity if x["label"] == label}):
        subset = [x for x in opacity if x["label"] == label and x["call"] == call]
        calls.append(dict(label=label, call=call, blocks=len(subset),
                          kappa_r_min=min(x["kappa_r_min"] for x in subset),
                          kappa_r_max=max(x["kappa_r_max"] for x in subset),
                          temperature_min=min(x["temperature_min"] for x in subset),
                          temperature_max=max(x["temperature_max"] for x in subset),
                          blocks_below_13K=sum(x["temperature_min"] < 13 for x in subset)))
csv_write(OUT / "opacity_call_summary.csv", calls)

source_before = AUDIT / "build_table_precision/src/pgen/protostellar_core_fld.cpp"
source_after = AUDIT / "build_table_initial_opacity/src/pgen/protostellar_core_fld.cpp"
patch = "".join(difflib.unified_diff(source_before.read_text().splitlines(keepends=True),
                                     source_after.read_text().splitlines(keepends=True),
                                     fromfile="src/pgen/protostellar_core_fld.cpp",
                                     tofile="src/pgen/protostellar_core_fld.cpp (diagnostic trial)"))
(OUT / "initial_opacity_diagnostic.patch").write_text(patch)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
for label, color in (("unfrozen", "tab:blue"), ("fixed_rosseland", "tab:orange")):
    subset = [x for x in calls if x["label"] == label]
    axes[0].fill_between([x["call"] for x in subset],
                         [x["kappa_r_min"] for x in subset],
                         [x["kappa_r_max"] for x in subset], alpha=.25, color=color)
    axes[0].plot([x["call"] for x in subset],
                 [x["kappa_r_min"] for x in subset], "o-", color=color, label=label)
axes[0].set(xlabel="Opacity callback within cycle 0", ylabel=r"$\kappa_R$ (cm$^2$ g$^{-1}$)")
axes[0].legend(fontsize=8)
fixed = [x for x in runs if x["fixed_rosseland"] is not None]
axes[1].plot([x["fixed_rosseland"] for x in fixed],
             [x["second_true_residual"] for x in fixed], "o-")
axes[1].axhline(2e-8, color="black", linestyle="--", linewidth=1, label="strict target")
axes[1].set(xlabel=r"Fixed $\kappa_R$ (cm$^2$ g$^{-1}$)",
            ylabel="Second FGMRES true relative residual", yscale="log")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "rosseland_threshold.png", dpi=160)

payload = dict(base_commit="ebf7634a8e877edfd58724de1e3810638a76ca7e",
               trial_binary_sha256=sha(AUDIT / "build_table_precision/bin/athena"),
               initial_opacity_binary_sha256=sha(AUDIT / "build_table_initial_opacity/bin/athena"),
               initial_opacity_patch_sha256=sha(OUT / "initial_opacity_diagnostic.patch"),
               initial_opacity_input_sha256=sha(ROOT / "input_table_component_initial_opacity_cfl030_strict_1cycle.used"),
               runs=runs, opacity_calls=calls,
               initial_opacity_logs=[dict(path=str(x), sha256=sha(x)) for x in opacity_logs])
(OUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")

files = [OUT / "initial_opacity_diagnostic.patch"]
for _, directory, input_name, _ in CASES:
    run = ROOT / directory
    files += [run / "run.log", run / "status.txt", ROOT / input_name]
for label in ("unfrozen", "fixed_rosseland"):
    run = ROOT / f"component_initial_opacity_allcalls_{label}_2rank"
    files += [run / "run.log", run / "status.txt"]
files += opacity_logs + [ROOT / "input_table_component_initial_opacity_cfl030_strict_1cycle.used"]
with tarfile.open(OUT / "evidence.tar.gz", "w:gz") as archive:
    for path in dict.fromkeys(files):
        archive.add(path, arcname=path.relative_to(ROOT) if path.is_relative_to(ROOT)
                    else Path("diagnostic") / path.name)
print(json.dumps(dict(runs=runs, call4=[x for x in calls if x["call"] == 4]), indent=2))
