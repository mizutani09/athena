#!/usr/bin/env python3
"""Summarize bounded clean-source CFL=0.3 restart continuations."""

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
OUT = HERE / "wor92_long_continuation_results_20261107"
CASES = [(100, 120), (120, 160), (160, 200)]


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
    cases = []
    cycles = []
    history = []
    with tarfile.open(OUT / "evidence.tar.gz", "w:gz") as archive:
        for start, stop in CASES:
            label = f"{start}_to_{stop}"
            run = ROOT / f"table_clean_guard_ghost_restart{start}_to{stop}_2rank"
            log = run / "run.log"
            hst = run / "cfl030_flux_precision16_local.hst"
            rst = run / "cfl030_flux_precision16_local.final.rst"
            assert all(p.is_file() for p in (log, hst, rst))
            source = log.read_text(errors="replace")
            stage = {}
            for match in re.finditer(r"^\[NR\] status=(\w+) iterations=(\d+).*?final_l2=(\S+) final_max=(\S+).*?cycle=(\d+) time=(\S+) stage=(\d+) dt=(\S+)", source, re.M):
                status, iters, l2, maxerr, cycle, time, stage_no, dt = match.groups()
                if int(stage_no) == 0:
                    stage[int(cycle)] = dict(status=status, newton_iterations=int(iters),
                                             final_l2=float(l2), final_max=float(maxerr),
                                             time=float(time), dt=float(dt))
            opacity = re.findall(r"^\[WOR92_OPACITY_ACCEPTED\] cycle=(\d+) Tmin_K=(\S+)", source, re.M)
            assert len(stage) == stop - start, (label, len(stage))
            assert len(opacity) == stop - start, (label, len(opacity))
            assert all(row["status"] == "converged" for row in stage.values())
            assert all(float(t) >= 5.0 for _, t in opacity)
            assert "Terminating on cycle limit" in source
            assert re.search(rf"^time=\S+ cycle={stop}$", source, re.M)
            diag = re.findall(r"^FLD_OPACITY_DIAGNOSTICS out_of_domain=(\d+) clamped=(\d+) negative_results=(\d+) zero_results=(\d+) nonfinite_results=(\d+)", source, re.M)
            assert len(diag) == 1, (label, diag)
            assert all(int(x) == 0 for x in diag[-1]), (label, diag[-1])
            assert "[WOR92_NR_GHOST_CLIP]" not in source
            assert "FATAL ERROR" not in source
            assert "created, 0 destroyed" in source
            fg = re.findall(r"^\[LINEAR_FGMRES\] iterations=(\d+) relative_residual=(\S+).*?inexact_accept=(\d+)", source, re.M)
            assert fg
            local_gas = re.findall(r"^\[NR_LOCAL_GAS\] exact_cells=(\d+) residual_over_gas_scale_max=(\S+) mixed_active=(\d+)", source, re.M)
            assert local_gas and all(x[2] == "1" for x in local_gas)
            temps = dict((int(c), float(t)) for c, t in opacity)
            for cycle in sorted(stage):
                cycles.append(dict(case=label, cycle=cycle, Tmin_K=temps[cycle], **stage[cycle]))
            arr = np.atleast_2d(np.loadtxt(hst))
            for row in arr:
                history.append(dict(case=label, time=float(row[0]), dt=float(row[1]),
                                    rho_max_cgs=float(row[15]), Tgas_max_K=float(row[19])))
            cases.append(dict(case=label, start_cycle=start, stop_cycle=stop,
                              newton_converged=len(stage), newton_iterations_max=max(r["newton_iterations"] for r in stage.values()),
                              final_max_max=max(r["final_max"] for r in stage.values()),
                              dt_min=min(r["dt"] for r in stage.values()),
                              dt_max=max(r["dt"] for r in stage.values()),
                              accepted_Tmin_min_K=min(temps.values()),
                              fgmres_solves=len(fg), fgmres_iterations_max=max(int(x[0]) for x in fg),
                              fgmres_true_residual_max=max(float(x[1]) for x in fg),
                              fgmres_above_strict_2e8_count=sum(float(x[1]) > 2e-8 for x in fg),
                              fgmres_inexact_gate_enabled_count=sum(int(x[2]) for x in fg),
                              local_gas_residual_max=max(float(x[1]) for x in local_gas),
                              opacity_clamped=int(diag[0][1]), ghost_limiter_calls=0,
                              history_last_time=float(arr[-1, 0]), history_last_rho_max_cgs=float(arr[-1, 15]),
                              history_last_Tgas_max_K=float(arr[-1, 19]),
                              log_sha256=digest(log), history_sha256=digest(hst), restart_sha256=digest(rst)))
            for path in (log, hst):
                archive.add(path, arcname=f"{label}/{path.name}")
    write_csv(OUT / "cases.csv", cases)
    write_csv(OUT / "cycles.csv", cycles)
    write_csv(OUT / "history.csv", history)
    binary = Path("/home/kosuke/simulation/test/rad_nr/wor92_operator_audit_20261107/build_table_domain_guard_trial/bin/athena")
    input_file = ROOT / "input_table_domain_guard_cfl030_restart.used"
    first_restart = ROOT / "table_domain_mean_limiter_restart74_to100_2rank/cfl030_flux_precision16_local.final.rst"
    summary = dict(source_branch="codex/wor92-domain-guard-trial-20261107",
                   source_commits=["6da81d20", "ff632195", "14633c31"],
                   binary_sha256=digest(binary), input_sha256=digest(input_file),
                   first_restart_sha256=digest(first_restart),
                   total_newton_converged=sum(c["newton_converged"] for c in cases),
                   overall_final_max_max=max(c["final_max_max"] for c in cases),
                   cases=cases,
                   limitations=["First-core density is not reached.",
                                "No AMR regrid occurs in these continuations; the accepted-state opacity guard still runs before regrid.",
                                "CFL and mesh convergence at matched physical time are not established."])
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, ax = plt.subplots(2, 1, figsize=(8.2, 6.4), sharex=False)
    ax[0].plot([r["time"] for r in history], [r["rho_max_cgs"] for r in history], marker=".", ms=2)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("code time")
    ax[0].set_ylabel("maximum density (g cm$^{-3}$)")
    ax[1].plot([r["cycle"] for r in cycles], [r["final_max"] for r in cycles], marker=".", ms=2)
    ax[1].axhline(1e-5, color="crimson", linestyle="--", label="Newton maximum threshold")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("cycle")
    ax[1].set_ylabel("final Newton maximum residual")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "continuation.png", dpi=160)
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2))
    for case in cases:
        print(case["case"], "rho_max", case["history_last_rho_max_cgs"], "Newton", case["newton_converged"])


if __name__ == "__main__":
    main()
