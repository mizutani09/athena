#!/usr/bin/env python3
"""Summarize the first-step background reset energy removal."""

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
OUT = HERE / "wor92_sink_removal_ledger_results_20261107"
CASES = [("03", 0.3), ("015", 0.15), ("0075", 0.075)]


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
        for tag, cfl in CASES:
            run = ROOT / f"table_sink_removal_ledger_cfl{tag}_2rank"
            log = run / "run.log"
            hst = run / "cfl030_flux_precision16_local.hst"
            s = log.read_text(errors="replace")
            entries = re.findall(r"^\[WOR92_BACKGROUND_SINK_LEDGER\] rank=(\d+) gid=(\d+) cycle=(\d+) dt=(\S+) cells=(\d+) gas_delta_erg=(\S+) radiation_delta_erg=(\S+)", s, re.M)
            assert len(entries) == 120 and len(set(int(x[1]) for x in entries)) == 120
            assert set(int(x[2]) for x in entries) == {100}
            assert "Terminating on cycle limit" in s
            assert "FATAL ERROR" not in s
            baseline = ROOT / f"table_sink_on_short_cfl{tag}_2rank/cfl030_flux_precision16_local.hst"
            baseline_row = np.atleast_2d(np.loadtxt(baseline))[0]
            ledger_row = np.atleast_2d(np.loadtxt(hst))[-1]
            # The history dt column is the *next* step and differs when tlim
            # truncates the baseline's continuation. The physical state agrees.
            assert baseline_row[0] == ledger_row[0]
            assert np.array_equal(baseline_row[2:], ledger_row[2:])
            dt = float(entries[0][3])
            assert all(abs(float(x[3]) - dt) < 1e-12 for x in entries)
            rows.append(dict(cfl=cfl, first_dt=dt, reset_blocks=len(entries),
                             reset_cells=sum(int(x[4]) for x in entries),
                             gas_delta_erg=sum(float(x[5]) for x in entries),
                             radiation_delta_erg=sum(float(x[6]) for x in entries),
                             radiation_removal_per_dt_erg=sum(float(x[6]) for x in entries)/dt,
                             numerical_history_matches_baseline=True,
                             log_sha256=sha(log), history_sha256=sha(hst),
                             restart_sha256=sha(run / "cfl030_flux_precision16_local.final.rst")))
            for path in (log, hst):
                archive.add(path, arcname=f"cfl{cfl}/{path.name}")
    with (OUT / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(branch="codex/wor92-sink-ledger-trial-20261107",
                   base_commit="873a1b4c3bda584b52e1cae2b41e333ddb2f793a",
                   binary_sha256=sha(Path("/home/kosuke/simulation/test/rad_nr/wor92_operator_audit_20261107/build_sink_ledger_trial/bin/athena")),
                   input_sha256=sha(ROOT / "input_table_domain_guard_cfl030_restart.used"),
                   checkpoint_sha256=sha(ROOT / "table_domain_mean_limiter_restart74_to100_2rank/cfl030_flux_precision16_local.final.rst"),
                   cases=rows)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["first_dt"] for r in rows],
            [-r["radiation_delta_erg"]/1e41 for r in rows], marker="o")
    ax.set_xlabel("first dt (code units)")
    ax.set_ylabel("radiation removed in one reset (10$^{41}$ erg)")
    fig.tight_layout()
    fig.savefig(OUT / "removal_vs_dt.png", dpi=160)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
