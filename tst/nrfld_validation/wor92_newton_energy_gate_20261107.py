#!/usr/bin/env python3
"""Compare Newton maximum gates from an identical cycle-92 restart state."""

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
OUT = HERE / "wor92_newton_energy_gate_results_20261107"
ROOT = Path("/home/kosuke/simulation/test/rad_nr/wor92_wor105_retest_20261107")
CASES = {
    "from92_max1e5": "table_mean_limiter_from92_nrmax1e5_2rank",
    "from92_max3e6": "table_mean_limiter_from92_nrmax3e6_2rank",
    "from74_max3e6": "table_domain_mean_limiter_nrmax3e6_restart74_to95_2rank",
    "from0_max3e6": "table_clean_nrmax3e6_from0_3cycles_2rank",
    "from0_verbose2": "table_clean_nrmax3e6_cycle0_verbose2_2rank",
    "from0_max_merit_tradeoff": "table_max_merit_nrmax3e6_cycle0_2rank",
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fields(line):
    return dict(re.findall(r"([A-Za-z_][A-Za-z_0-9]*)=([^\s]+)", line))


def main():
    OUT.mkdir(exist_ok=True)
    rows, overview = [], {}
    for label, dirname in CASES.items():
        log = ROOT / dirname / "run.log"
        lines = log.read_text(errors="replace").splitlines()
        nr = [fields(line) for line in lines if line.startswith("[NR] status=")]
        ledger = [fields(line) for line in lines if line.startswith("[NR_ENERGY_LEDGER]")]
        by_cycle = {}
        for d in nr:
            if "cycle" not in d or "dt" not in d or float(d["dt"]) <= 0:
                continue
            by_cycle.setdefault(int(d["cycle"]), {})["nr"] = d
        for d in ledger:
            cycle = int(d["cycle"])
            by_cycle.setdefault(cycle, {}).setdefault("ledger", []).append(d)
        for cycle, data in sorted(by_cycle.items()):
            d = data.get("nr", {})
            e = data.get("ledger", [])
            rows.append({"case": label, "cycle": cycle,
                         "nr_status": d.get("status", ""),
                         "nr_iterations": int(d["iterations"]) if d else None,
                         "nr_final_l2": float(d["final_l2"]) if d else None,
                         "nr_final_max": float(d["final_max"]) if d else None,
                         "max_rank_signed_energy_balance": max((float(x["relative_balance"]) for x in e), default=None),
                         "max_rank_absolute_energy_balance": max((float(x["absolute_relative_balance"]) for x in e), default=None)})
        overview[label] = {"dirname": dirname, "log_sha256": sha(log),
                           "cycles": sorted(by_cycle),
                           "converged": sum(d.get("status") == "converged" and float(d.get("dt", "0")) > 0 for d in nr),
                           "backtracking_exhausted": sum(d.get("status") == "backtracking_exhausted" for d in nr)}
    at92 = {r["case"]: r for r in rows if r["cycle"] == 92}
    assert at92["from92_max1e5"]["nr_iterations"] == 3
    assert at92["from92_max3e6"]["nr_iterations"] == 4
    assert at92["from92_max1e5"]["max_rank_signed_energy_balance"] > 2e-6
    assert at92["from92_max3e6"]["max_rank_signed_energy_balance"] < 1e-10
    assert overview["from0_max3e6"]["backtracking_exhausted"] == 1
    assert overview["from0_verbose2"]["backtracking_exhausted"] == 1
    assert any(r["nr_status"] == "stagnation" for r in rows if r["case"] == "from0_max_merit_tradeoff")

    with (OUT / "cycles.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    checkpoint = ROOT / "table_mean_limiter_checkpoint92_baseline_2rank/cfl030_flux_precision16_local.final.rst"
    summary = {"checkpoint_cycle": 92, "checkpoint_sha256": sha(checkpoint),
               "binary_sha256": sha(Path("/home/kosuke/simulation/test/rad_nr/wor92_operator_audit_20261107/build_table_domain_guard_trial/bin/athena")),
               "same_checkpoint_cycle92": {
                   name: {k: at92[name][k] for k in ("nr_iterations", "nr_final_l2", "nr_final_max", "max_rank_signed_energy_balance", "max_rank_absolute_energy_balance")}
                   for name in ("from92_max1e5", "from92_max3e6")},
               "cases": overview,
               "max_merit_trial_binary_sha256": "04ed8c9ff4ec35df5cc5fab841154aa6bbea9638673fdc330e641148ad8fbdc9",
               "interpretation": "One additional Newton iteration from the same restart state reduces the implicit energy-balance fraction at cycle 92 by five orders of magnitude. A globally tighter maximum criterion fails at the initial cycle, so adaptive control or an accuracy budget is needed."}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for label, color in (("from92_max1e5", "#3577a5"), ("from92_max3e6", "#d59037")):
        selection = [r for r in rows if r["case"] == label and r["max_rank_signed_energy_balance"] is not None]
        axes[0].semilogy([r["cycle"] for r in selection], [r["max_rank_signed_energy_balance"] for r in selection],
                         marker="o", label=label, color=color)
        axes[1].semilogy([r["cycle"] for r in selection], [r["nr_final_max"] for r in selection],
                         marker="o", label=label, color=color)
    axes[0].set_ylabel("max rank signed energy balance / scale")
    axes[1].set_ylabel("Newton final maximum residual")
    for ax in axes:
        ax.set_xlabel("cycle")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "energy_gate.png", dpi=180)
    plt.close(fig)
    with tarfile.open(OUT / "evidence.tar.gz", "w:gz") as archive:
        for label, dirname in CASES.items():
            archive.add(ROOT / dirname / "run.log", arcname=f"logs/{label}.log")
        archive.add(ROOT / "table_mean_limiter_checkpoint92_baseline_2rank/run.log",
                    arcname="logs/checkpoint92_production.log")
        archive.add(OUT / "max_merit_trial.patch.gz", arcname="patches/max_merit_trial.patch.gz")
    print(json.dumps(summary["same_checkpoint_cycle92"], indent=2))


if __name__ == "__main__":
    main()
