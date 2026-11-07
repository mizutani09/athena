#!/usr/bin/env python3
"""Compare short matched-time background controls down to CFL=0.0375."""

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
OUT = HERE / "wor92_finer_cfl_results_20261107"
RUNS = {
    "reset_on": [(0.3, "table_sink_on_short_cfl03_2rank"),
                 (0.075, "table_sink_on_short_cfl0075_2rank"),
                 (0.0375, "table_sink_on_short_cfl00375_2rank")],
    "reset_off": [(0.3, "table_sink_off_short_cfl03_2rank"),
                  (0.075, "table_sink_off_short_cfl0075_2rank"),
                  (0.0375, "table_sink_off_short_cfl00375_2rank")],
    "relax_tau8": [(0.3, "table_relax_tau8_short_cfl03_v2_2rank"),
                   (0.075, "table_relax_tau8_short_cfl0075_v2_2rank"),
                   (0.0375, "table_tau8_short_cfl00375_2rank")],
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    with tarfile.open(OUT / "evidence.tar.gz", "w:gz") as archive:
        for treatment, cases in RUNS.items():
            for cfl, folder in cases:
                run = ROOT / folder
                log = run / "run.log"
                hst = run / "cfl030_flux_precision16_local.hst"
                s = log.read_text(errors="replace")
                end = re.findall(r"^time=(\S+) cycle=(\d+)$", s, re.M)
                assert end and abs(float(end[-1][0]) - 2275.4309789960685) < 1e-8
                stages = [x for x in re.findall(r"^\[NR\] status=(\w+) iterations=(\d+).*?cycle=(\d+) time=(\S+) stage=(\d+) dt=(\S+)", s, re.M) if x[4] == "0"]
                assert stages and all(x[0] == "converged" for x in stages)
                fg = re.findall(r"^\[LINEAR_FGMRES\] iterations=(\d+) relative_residual=(\S+)", s, re.M)
                assert fg and max(float(x[1]) for x in fg) < 2e-8
                diag = re.findall(r"^FLD_OPACITY_DIAGNOSTICS out_of_domain=(\d+) clamped=(\d+) negative_results=(\d+) zero_results=(\d+) nonfinite_results=(\d+)", s, re.M)
                assert len(diag) == 1 and all(int(x) == 0 for x in diag[0])
                assert "created, 0 destroyed" in s
                a = np.atleast_2d(np.loadtxt(hst))[-1]
                rows.append(dict(treatment=treatment, cfl=cfl, steps=len(stages),
                                 end_time=float(end[-1][0]), Tgas_max_K=float(a[19]),
                                 Eint_gas_erg=float(a[30]), Erad_erg=float(a[33]),
                                 background_rad_T_relative_error=float(a[37]),
                                 fgmres_solves=len(fg), fgmres_max_true_residual=max(float(x[1]) for x in fg),
                                 log_sha256=sha(log), history_sha256=sha(hst),
                                 restart_sha256=sha(run / "cfl030_flux_precision16_local.final.rst")))
                for path in (log, hst):
                    archive.add(path, arcname=f"{treatment}/cfl{cfl}/{path.name}")
    with (OUT / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(end_time=2275.4309789960685, cases=rows,
                   binary_standard_sha256=sha(Path("/home/kosuke/simulation/test/rad_nr/wor92_operator_audit_20261107/build_table_domain_guard_trial/bin/athena")),
                   binary_relax_trial_sha256=sha(Path("/home/kosuke/simulation/test/rad_nr/wor92_operator_audit_20261107/build_background_relax_trial/bin/athena")),
                   input_standard_sha256=sha(ROOT / "input_table_domain_guard_cfl030_restart.used"),
                   input_relax_trial_sha256=sha(ROOT / "input_table_background_relax_trial.used"),
                   checkpoint_sha256=sha(ROOT / "table_domain_mean_limiter_restart74_to100_2rank/cfl030_flux_precision16_local.final.rst"))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for treatment in RUNS:
        group = sorted((r for r in rows if r["treatment"] == treatment), key=lambda r: r["cfl"])
        axes[0].plot([r["cfl"] for r in group], [r["Tgas_max_K"] for r in group], marker="o", label=treatment)
        axes[1].plot([r["cfl"] for r in group], [r["Erad_erg"]/1e42 for r in group], marker="o", label=treatment)
    axes[0].set_ylabel("maximum gas temperature (K)")
    axes[1].set_ylabel("total radiation energy (10$^{42}$ erg)")
    for ax in axes:
        ax.set_xlabel("CFL")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "finer_cfl.png", dpi=160)
    print(json.dumps({t: [r for r in rows if r["treatment"] == t] for t in RUNS}, indent=2))


if __name__ == "__main__":
    main()
