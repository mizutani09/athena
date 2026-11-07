#!/usr/bin/env python3
"""Summarize isolated exponential background-reservoir pilot controls."""

import csv
import gzip
import hashlib
import json
import re
import subprocess
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = Path("/home/kosuke/simulation/test/rad_nr/wor92_wor105_retest_20261107")
TRIAL = Path("/home/kosuke/simulation/athena_wor92_background_relax_trial")
OUT = HERE / "wor92_background_relax_results_20261107"
CASES = [
    ("short", 0, 0.3, "table_relax_tau0_short_cfl03_v2_2rank"),
    ("short", 8, 0.3, "table_relax_tau8_short_cfl03_v2_2rank"),
    ("short", 8, 0.15, "table_relax_tau8_short_cfl015_v2_2rank"),
    ("short", 8, 0.075, "table_relax_tau8_short_cfl0075_v2_2rank"),
    ("short", 32, 0.3, "table_relax_tau32_short_cfl03_v2_2rank"),
    ("short", 32, 0.075, "table_relax_tau32_short_cfl0075_v2_2rank"),
    ("long", 8, 0.3, "table_relax_tau8_long_cfl03_2rank"),
    ("long", 8, 0.15, "table_relax_tau8_long_cfl015_2rank"),
    ("long", 8, 0.075, "table_relax_tau8_long_cfl0075_2rank"),
]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    patch_path = OUT / "relaxation_trial.patch.gz"
    if patch_path.exists():
        with gzip.open(patch_path, "rb") as stream:
            patch = stream.read()
    else:
        patch = subprocess.check_output(["git", "diff", "--", "src/pgen/protostellar_core_fld.cpp"], cwd=TRIAL)
    assert b"background_relaxation_time_code" in patch
    if not patch_path.exists():
        with gzip.open(patch_path, "wb") as stream:
            stream.write(patch)
    source_input = ROOT / "input_table_background_relax_trial.used"
    (OUT / "input_table_background_relax_trial.used").write_bytes(source_input.read_bytes())
    rows = []
    with tarfile.open(OUT / "evidence.tar.gz", "w:gz") as archive:
        for interval, tau, cfl, folder in CASES:
            run = ROOT / folder
            log = run / "run.log"
            hst = run / "cfl030_flux_precision16_local.hst"
            assert log.is_file() and hst.is_file(), run
            s = log.read_text(errors="replace")
            end = re.findall(r"^time=(\S+) cycle=(\d+)$", s, re.M)
            target = 2275.4309789960685 if interval == "short" else 2436.7392236433016
            assert end and abs(float(end[-1][0]) - target) < 1e-8
            stages = [x for x in re.findall(r"^\[NR\] status=(\w+) iterations=(\d+).*?final_max=(\S+).*?cycle=(\d+) time=(\S+) stage=(\d+) dt=(\S+)", s, re.M) if x[5] == "0"]
            assert stages and all(x[0] == "converged" for x in stages)
            fg = re.findall(r"^\[LINEAR_FGMRES\] iterations=(\d+) relative_residual=(\S+)", s, re.M)
            assert fg
            assert max(float(x[1]) for x in fg) < 2e-8
            assert "[WOR92_NR_GHOST_CLIP]" not in s
            assert "created, 0 destroyed" in s
            diag = re.findall(r"^FLD_OPACITY_DIAGNOSTICS out_of_domain=(\d+) clamped=(\d+) negative_results=(\d+) zero_results=(\d+) nonfinite_results=(\d+)", s, re.M)
            assert len(diag) == 1 and all(int(x) == 0 for x in diag[0])
            accepted = re.findall(r"^\[WOR92_OPACITY_ACCEPTED\] cycle=(\d+) Tmin_K=(\S+)", s, re.M)
            assert len(accepted) == len(stages)
            a = np.atleast_2d(np.loadtxt(hst))[-1]
            rows.append(dict(interval=interval, tau_code=tau, cfl=cfl,
                             end_cycle=int(end[-1][1]), end_time=float(end[-1][0]),
                             newton_converged=len(stages), max_newton_iterations=max(int(x[1]) for x in stages),
                             max_final_newton_residual=max(float(x[2]) for x in stages),
                             fgmres_solves=len(fg), fgmres_true_residual_max=max(float(x[1]) for x in fg),
                             accepted_Tmin_min_K=min(float(x[1]) for x in accepted),
                             rho_max_cgs=float(a[15]), Tgas_max_K=float(a[19]),
                             gas_internal_erg=float(a[30]), radiation_total_erg=float(a[33]),
                             background_gas_T_relative_error=float(a[36]),
                             background_rad_T_relative_error=float(a[37]),
                             log_sha256=sha(log), history_sha256=sha(hst),
                             restart_sha256=sha(run / "cfl030_flux_precision16_local.final.rst")))
            for path in (log, hst):
                archive.add(path, arcname=f"{interval}/tau{tau}/cfl{cfl}/{path.name}")
    write_csv(OUT / "cases.csv", rows)
    by = {(r["interval"], r["tau_code"], r["cfl"]): r for r in rows}
    original = ROOT / "table_sink_on_short_cfl03_2rank/cfl030_flux_precision16_local.hst"
    assert (ROOT / CASES[0][3] / "cfl030_flux_precision16_local.hst").read_bytes() == original.read_bytes()
    comparisons = []
    for interval, tau in (("short", 8), ("short", 32), ("long", 8)):
        a, b = by[(interval, tau, 0.3)], by[(interval, tau, 0.075)]
        comparisons.append(dict(interval=interval, tau_code=tau,
                                radiation_relative_difference=(b["radiation_total_erg"]-a["radiation_total_erg"])/a["radiation_total_erg"],
                                gas_internal_relative_difference=(b["gas_internal_erg"]-a["gas_internal_erg"])/a["gas_internal_erg"],
                                Tgas_max_relative_difference=(b["Tgas_max_K"]-a["Tgas_max_K"])/a["Tgas_max_K"],
                                background_rad_error_max=max(a["background_rad_T_relative_error"],b["background_rad_T_relative_error"])))
    write_csv(OUT / "comparisons.csv", comparisons)
    summary = dict(source_base="7fd53c00bdba49095ff5ad13659b2b4c09e1451c",
                   source_patch_sha256=sha(OUT / "relaxation_trial.patch.gz"),
                   binary_sha256=sha(Path("/home/kosuke/simulation/test/rad_nr/wor92_operator_audit_20261107/build_background_relax_trial/bin/athena")),
                   input_sha256=sha(source_input),
                   checkpoint_sha256=sha(ROOT / "table_domain_mean_limiter_restart74_to100_2rank/cfl030_flux_precision16_local.final.rst"),
                   tau_zero_history_byte_equal=True, cases=rows, comparisons=comparisons,
                   limitation="The relaxation times are exploratory and physically uncalibrated; no production source change is adopted.")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for interval, tau, label in (("short", 8, "short τ=8"), ("short", 32, "short τ=32"),
                                 ("long", 8, "long τ=8")):
        points = sorted((r for r in rows if r["interval"] == interval and r["tau_code"] == tau),
                        key=lambda r: r["cfl"])
        axes[0].plot([r["cfl"] for r in points], [r["radiation_total_erg"]/1e42 for r in points], marker="o", label=label)
        axes[1].plot([r["cfl"] for r in points], [r["background_rad_T_relative_error"] for r in points], marker="o", label=label)
    axes[0].set_ylabel("radiation energy (10$^{42}$ erg)")
    axes[1].set_ylabel("background radiation T relative error")
    for ax in axes:
        ax.set_xlabel("CFL")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "relaxation_controls.png", dpi=160)
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
