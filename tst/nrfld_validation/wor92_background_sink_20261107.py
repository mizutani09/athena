#!/usr/bin/env python3
"""Audit short matched-time background heat-sink controls for WOR-92."""

import csv
import hashlib
import json
import re
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = Path("/home/kosuke/simulation/test/rad_nr/wor92_wor105_retest_20261107")
OUT = HERE / "wor92_background_sink_results_20261107"
CASES = {
    "gas_and_radiation_before": ("sink_on_short", ["03", "015", "0075"]),
    "sink_off": ("sink_off_short", ["03", "015", "0075"]),
    "gas_only_before": ("sink_gas_only_short", ["03", "0075"]),
    "gas_and_radiation_after": ("sink_after_short", ["03", "0075"]),
}
CFL = {"03": 0.3, "015": 0.15, "0075": 0.075}


def digest(path):
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
    rows = []
    with tarfile.open(OUT / "evidence.tar.gz", "w:gz") as archive:
        for treatment, (prefix, cfls) in CASES.items():
            for cfl in cfls:
                run = ROOT / f"table_{prefix}_cfl{cfl}_2rank"
                log, hst = run / "run.log", run / "cfl030_flux_precision16_local.hst"
                assert log.is_file() and hst.is_file(), run
                s = log.read_text(errors="replace")
                stages = [x for x in re.findall(r"^\[NR\] status=(\w+) iterations=(\d+).*?cycle=(\d+) time=(\S+) stage=(\d+) dt=(\S+)", s, re.M) if x[4] == "0"]
                assert stages and all(x[0] == "converged" for x in stages)
                end = re.findall(r"^time=(\S+) cycle=(\d+)$", s, re.M)
                assert end and abs(float(end[-1][0]) - 2275.4309789960685) < 1e-8
                fg = re.findall(r"^\[LINEAR_FGMRES\] iterations=(\d+) relative_residual=(\S+)", s, re.M)
                assert fg
                op = re.findall(r"^FLD_OPACITY_DIAGNOSTICS out_of_domain=(\d+) clamped=(\d+) negative_results=(\d+) zero_results=(\d+) nonfinite_results=(\d+)", s, re.M)
                assert len(op) == 1 and all(int(x) == 0 for x in op[0])
                a = np.atleast_2d(np.loadtxt(hst))[-1]
                rows.append(dict(treatment=treatment, cfl=CFL[cfl], steps=len(stages),
                                 final_time=float(end[-1][0]), dt_min=min(float(x[5]) for x in stages),
                                 dt_max=max(float(x[5]) for x in stages),
                                 fgmres_solves=len(fg), fgmres_true_residual_max=max(float(x[1]) for x in fg),
                                 rho_max_cgs=float(a[15]), Tgas_max_K=float(a[19]),
                                 gas_internal_erg=float(a[30]), radiation_total_erg=float(a[33]),
                                 background_gas_T_relative_error=float(a[36]),
                                 background_rad_T_relative_error=float(a[37]),
                                 log_sha256=digest(log), history_sha256=digest(hst),
                                 restart_sha256=digest(run / "cfl030_flux_precision16_local.final.rst")))
                for path in (log, hst):
                    archive.add(path, arcname=f"{treatment}/cfl{CFL[cfl]}/{path.name}")
        for kind, prefix in (("mixed_explicit", "table_first_step_mixed_ledger"),
                             ("implicit", "table_first_step_implicit_ledger")):
            for cfl in ("03", "0075"):
                run = ROOT / f"{prefix}_cfl{cfl}_2rank"
                for path in (run / "run.log", run / "cfl030_flux_precision16_local.hst"):
                    archive.add(path, arcname=f"{kind}/cfl{CFL[cfl]}/{path.name}")
    write_csv(OUT / "cases.csv", rows)
    by = {(r["treatment"], r["cfl"]): r for r in rows}
    deltas = []
    for treatment in CASES:
        x, y = by[(treatment, 0.3)], by[(treatment, 0.075)]
        deltas.append(dict(treatment=treatment,
                           radiation_relative_difference=(y["radiation_total_erg"]-x["radiation_total_erg"])/x["radiation_total_erg"],
                           gas_internal_relative_difference=(y["gas_internal_erg"]-x["gas_internal_erg"])/x["gas_internal_erg"],
                           Tgas_max_relative_difference=(y["Tgas_max_K"]-x["Tgas_max_K"])/x["Tgas_max_K"]))
    write_csv(OUT / "deltas.csv", deltas)
    mixed = {}
    implicit = {}
    for cfl in ("03", "0075"):
        s = (ROOT / f"table_first_step_mixed_ledger_cfl{cfl}_2rank/run.log").read_text()
        floor = [float(x) for x in re.findall(r"radiation_floor_volume=(\S+)", s)]
        pair = [float(x) for x in re.findall(r"pair_balance_volume=(\S+)", s)]
        assert len(floor) == 120 and all(x == 0 for x in floor)
        mixed[str(CFL[cfl])] = dict(blocks=len(floor), total_radiation_floor_volume=sum(floor),
                                    total_explicit_pair_balance_volume=sum(pair))
        s = (ROOT / f"table_first_step_implicit_ledger_cfl{cfl}_2rank/run.log").read_text()
        relative = [float(x) for x in re.findall(r"^\[NR_ENERGY_LEDGER\][^\n]*? relative_balance=(\S+)", s, re.M)]
        assert len(relative) == 2
        implicit[str(CFL[cfl])] = dict(max_rank_relative_balance=max(relative))
    summary = dict(source_branch="codex/wor92-domain-guard-trial-20261107",
                   source_commits=["6da81d20", "ff632195", "14633c31"],
                   binary_sha256=digest(Path("/home/kosuke/simulation/test/rad_nr/wor92_operator_audit_20261107/build_table_domain_guard_trial/bin/athena")),
                   input_sha256=digest(ROOT / "input_table_domain_guard_cfl030_restart.used"),
                   checkpoint_sha256=digest(ROOT / "table_domain_mean_limiter_restart74_to100_2rank/cfl030_flux_precision16_local.final.rst"),
                   matched_end_time=2275.4309789960685, cases=rows, deltas=deltas,
                   mixed_explicit_ledger=mixed, implicit_ledger=implicit,
                   interpretation="The per-step background gas/radiation reset is a dominant contributor to the short-interval CFL sensitivity, especially radiation; turning it off is a model change, not an adopted production fix.")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    for treatment in CASES:
        points = sorted((r for r in rows if r["treatment"] == treatment), key=lambda r: r["cfl"])
        ax.plot([r["cfl"] for r in points], [r["radiation_total_erg"] / 1e42 for r in points],
                marker="o", label=treatment.replace("_", " "))
    ax.set_xlabel("CFL")
    ax.set_ylabel("radiation energy at matched time (10$^{42}$ erg)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "sink_controls.png", dpi=160)
    print(json.dumps(deltas, indent=2))


if __name__ == "__main__":
    main()
