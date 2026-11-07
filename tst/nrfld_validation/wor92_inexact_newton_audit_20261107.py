"""Audit the local-root/inexact-Newton controls and their failure limits."""

import csv
import difflib
import hashlib
import json
import math
import re
import tarfile
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path("/home/kosuke/simulation/test/rad_nr/wor92_wor105_retest_20261107")
BUILD = Path("/home/kosuke/simulation/test/rad_nr/wor92_operator_audit_20261107")
OUT = HERE / "wor92_inexact_newton_results_20261107"
CASES = {
    "floor03_10": ("table_floor32_cfl030_matched_time_2rank", "input_table_floor32_cfl030_matched_time.used", "build_table_precision"),
    "cap03_10": ("table_inexact_cfl030_matched_time_2rank", "input_table_inexact_cfl030_matched_time.used", "build_table_inexact_trial"),
    "floor03_30": ("table_floor32_energy_30cycles_2rank", "input_table_local_jacobian_floor32_cfl030_30cycles.used", "build_table_precision"),
    "cap03_30": ("table_inexact_cfl030_30cycles_2rank", "input_table_inexact_cfl030_30cycles.used", "build_table_inexact_trial"),
    "floor015": ("table_floor32_cfl015_matched_time_2rank", "input_table_floor32_cfl015_matched_time.used", "build_table_precision"),
    "cap015": ("table_inexact_cfl015_matched_time_2rank", "input_table_inexact_cfl015_matched_time.used", "build_table_inexact_trial"),
    "nr1e6": ("table_floor32_nr1e6_10cycles_2rank", "input_table_local_jacobian_floor32_nr1e6_cfl030_10cycles.used", "build_table_precision"),
    "restart_error": ("table_floor32_restart30_to80_2rank", "input_table_local_jacobian_floor32_cfl030_30cycles.used", "build_table_precision"),
    "clamp": ("table_floor32_clamp_count_from2000_2rank", "input_table_local_jacobian_floor32_cfl030_30cycles.used", "build_table_opacity_count_trial"),
    "clamp_snapshots": ("table_floor32_clamp_snapshots_from2000_2rank", "input_table_local_jacobian_floor32_cfl030_30cycles.used", "build_table_opacity_count_trial"),
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def fields(line):
    return dict(re.findall(r"([A-Za-z_][A-Za-z_0-9]*)=([^\s]+)", line))

def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

def main():
    OUT.mkdir(exist_ok=True)
    evidence, runs, solves, traces = {}, [], [], {}
    for name, (directory, input_name, build) in CASES.items():
        run = ROOT / directory
        log = run / "run.log"
        inp = ROOT / input_name
        binary = BUILD / build / "bin/athena"
        assert all(p.exists() for p in (log, inp, binary)), name
        lines = log.read_text(errors="replace").splitlines()
        fgmres = [fields(line) for line in lines if "[LINEAR_FGMRES] iterations=" in line]
        nr = [fields(line) for line in lines if "[NR] status=converged" in line and "stage=0" in line]
        cycles = [fields(line) for line in lines if line.startswith("cycle=")]
        traces[name] = {
            "linear": [(int(d["iterations"]), d["relative_residual"]) for d in fgmres],
            "newton": [line for line in lines if line.startswith("[NR] status=converged") and "stage=0" in line],
            "cycles": [line for line in lines if line.startswith("cycle=")],
        }
        for index, d in enumerate(fgmres):
            solves.append({"case": name, "solve": index+1,
                           "iterations": int(d["iterations"]),
                           "true_relative_residual": float(d["relative_residual"]),
                           "strict_pass": float(d["relative_residual"]) <= 2e-8,
                           "floor_accept": int(d["precision_floor_accept"]),
                           "cap_eligible": int(d.get("inexact_accept", "0"))})
        runs.append({"case": name, "cycles_logged": max((int(d["cycle"]) for d in cycles), default=-1),
                     "newton_convergences": len(nr), "fgmres_solves": len(fgmres),
                     "strict_failures": sum(float(d["relative_residual"]) > 2e-8 for d in fgmres),
                     "floor_accepts": sum(int(d["precision_floor_accept"]) for d in fgmres),
                     "newton_final_max": max((float(d["final_max"]) for d in nr), default=None),
                     "log_sha256": sha(log), "input_sha256": sha(inp),
                     "binary_sha256": sha(binary)})
        evidence[f"{name}/run.log"] = log
        evidence[f"{name}/input.used"] = inp
        hst = run / "cfl030_flux_precision16_local.hst"
        if hst.exists(): evidence[f"{name}/history.hst"] = hst
    assert len(traces["floor03_10"]["newton"]) == 10
    assert len(traces["cap03_10"]["newton"]) == 10
    assert len(traces["floor03_30"]["newton"]) == 30
    assert len(traces["cap03_30"]["newton"]) == 30
    assert len(traces["restart_error"]["newton"]) == 48
    assert len(traces["clamp"]["newton"]) >= 6
    equality = {}
    for count in (10, 30):
        a, b = traces[f"floor03_{count}"], traces[f"cap03_{count}"]
        equality[str(count)] = {key: a[key] == b[key] for key in a}
        assert all(equality[str(count)].values()), equality[str(count)]
    history_30_equal = (ROOT / CASES["floor03_30"][0] / "cfl030_flux_precision16_local.hst").read_bytes() == \
        (ROOT / CASES["cap03_30"][0] / "cfl030_flux_precision16_local.hst").read_bytes()
    assert history_30_equal
    hdf5_equal = {}
    for kind, dataset in (("prim", "prim"), ("fld", "user_out_var")):
        path = f"cfl030_flux_precision16_local.{kind}.00001.athdf"
        a = ROOT / CASES["floor03_10"][0] / path
        b = ROOT / CASES["cap03_10"][0] / path
        assert a.exists() and b.exists()
        with h5py.File(a) as left, h5py.File(b) as right:
            hdf5_equal[kind] = all(np.array_equal(left[key][...], right[key][...])
                                   for key in (dataset, "Levels", "LogicalLocations"))
        evidence[f"floor03_10/{path}"] = a
        evidence[f"cap03_10/{path}"] = b
    assert all(hdf5_equal.values())
    tight = (ROOT / CASES["nr1e6"][0] / "run.log").read_text(errors="replace")
    tight_failure = fields(next(line for line in tight.splitlines() if "[NR] status=backtracking_exhausted" in line))
    assert tight_failure["cycle"] == "0"
    restart_text = (ROOT / CASES["restart_error"][0] / "run.log").read_text(errors="replace")
    match = re.search(r"Planck opacity lookup is outside the table domain: log10\(T\)=([0-9.]+).*?valid \[([0-9.]+),", restart_text)
    assert match
    clamp_text = (ROOT / CASES["clamp"][0] / "run.log").read_text(errors="replace")
    count = fields(next(line for line in clamp_text.splitlines() if "FLD_OPACITY_DIAGNOSTICS" in line))
    snapshot_dir = ROOT / CASES["clamp_snapshots"][0]
    snapshot_rows = []
    for path in sorted(snapshot_dir.glob("*.fld.*.athdf")):
        with h5py.File(path) as f:
            temperature = f["user_out_var"][2,...]
            snapshot_rows.append({"cycle": int(f.attrs["NumCycles"]),
                                  "time_code": float(f.attrs["Time"]),
                                  "Tgas_min_kelvin": float(np.min(temperature)),
                                  "Tgas_max_kelvin": float(np.max(temperature)),
                                  "cells_below_5K": int(np.count_nonzero(temperature < 5.0)),
                                  "sha256": sha(path)})
    assert len(snapshot_rows) == 7 and all(r["cells_below_5K"] == 0 for r in snapshot_rows)
    write_csv(OUT / "accepted_temperature.csv", snapshot_rows)
    snapshot_text = (snapshot_dir / "run.log").read_text(errors="replace")
    snapshot_count = fields(next(line for line in snapshot_text.splitlines()
                                 if "FLD_OPACITY_DIAGNOSTICS" in line))
    assert snapshot_count == count
    source = BUILD / "build_table_inexact_trial/src/linear_solver/linearMG/linearMG.cpp"
    base = BUILD / "build_table_precision/src/linear_solver/linearMG/linearMG.cpp"
    patch = "".join(difflib.unified_diff(base.read_text().splitlines(True), source.read_text().splitlines(True),
                                         fromfile="baseline/linearMG.cpp", tofile="trial/linearMG.cpp"))
    (OUT / "inexact_gate.patch").write_text(patch)
    pgen_base = BUILD / "build_table_precision/src/pgen/protostellar_core_fld.cpp"
    pgen_trial = BUILD / "build_table_opacity_count_trial/src/pgen/protostellar_core_fld.cpp"
    pgen_patch = "".join(difflib.unified_diff(pgen_base.read_text().splitlines(True),
            pgen_trial.read_text().splitlines(True), fromfile="baseline/protostellar_core_fld.cpp",
            tofile="diagnostic/protostellar_core_fld.cpp"))
    (OUT / "opacity_count.patch").write_text(pgen_patch)
    write_csv(OUT / "runs.csv", runs)
    write_csv(OUT / "solves.csv", solves)
    ledger = json.loads((HERE / "wor92_floor32_restart30_results_20261107/summary.json").read_text())
    summary = {"source_git_base": "44611e8ece4c7da695733a39a0892646703d7607",
               "matched_trace": equality, "matched_30cycle_history_bitwise": history_30_equal,
               "matched_final_hdf5_bitwise": hdf5_equal,
               "tight_newton_failure": {k: tight_failure[k] for k in ("cycle", "iterations", "final_l2", "final_max", "final_gas_l2")},
               "restart_opacity_T_kelvin": 10**float(match.group(1)),
               "restart_opacity_table_Tmin_kelvin": 10**float(match.group(2)),
               "clamp_diagnostics": count,
               "accepted_state_temperature_min_kelvin": min(r["Tgas_min_kelvin"] for r in snapshot_rows),
               "accepted_state_cells_below_5K": sum(r["cells_below_5K"] for r in snapshot_rows),
               "restart_energy_max_relative_balance": ledger["max_relative_energy_balance"],
               "restart_cf_max_weighted_mismatch": ledger["max_weighted_cf_mismatch"],
               "runs": runs}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    with tarfile.open(OUT / "evidence.tar.gz", "w:gz") as archive:
        for name, path in evidence.items(): archive.add(path, arcname=name)
        archive.add(OUT / "inexact_gate.patch", arcname="inexact_gate.patch")
        archive.add(OUT / "opacity_count.patch", arcname="opacity_count.patch")
    fig, ax = plt.subplots(figsize=(9, 4.3))
    for name in ("cap03_30", "cap015"):
        values = [r["true_relative_residual"] for r in solves if r["case"] == name]
        ax.semilogy(range(1, len(values)+1), values, "o-", ms=2.6, label=name)
    ax.axhline(2e-8, color="black", linestyle="--", label="strict target")
    ax.axhline(1e-7, color="red", linestyle=":", label="trial true-residual cap")
    ax.set_xlabel("FGMRES solve index")
    ax.set_ylabel("True relative residual")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT / "inexact_residuals.png", dpi=160)
    print(json.dumps({k:v for k,v in summary.items() if k != "runs"}, indent=2))

if __name__ == "__main__":
    main()
