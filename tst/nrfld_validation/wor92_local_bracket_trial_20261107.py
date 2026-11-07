"""Summarize a bounded mixed-frame local-root bracket expansion trial."""

import csv
import difflib
import hashlib
import importlib.util
import json
import re
import tarfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SIM = Path("/home/kosuke/simulation")
HERE = Path(__file__).resolve().parent
ROOT = SIM / "test/rad_nr/wor92_operator_audit_20261107"
BASE = ROOT / "build_table_precision"
TRIAL = ROOT / "build_table_local_bracket_trial"
CELL = SIM / "test/rad_nr/wor92_local_general_cell_20261107"
RUN = SIM / "test/rad_nr/wor92_wor105_retest_20261107"
EOS = SIM / "test/rad_nr/wor92_20261107/bundle/data/eos/protostellar_tomida2015_material.tab"
OUT = HERE / "wor92_local_bracket_trial_results_20261107"
LOGS = {
    "table_base": RUN / "table_local_jacobian_cfl030_strict_2rank/run.log",
    "table_trial": RUN / "table_local_bracket_table_true_strict_3cycles_2rank.log",
    "fixed_base": RUN / "table_coupled_global_jv_v3_fixed_rosseland_high_2rank/run.log",
    "fixed_trial": RUN / "table_local_bracket_fixed_strict_3cycles_2rank.log",
    "stress_base": CELL / "mixed_stress2.log",
    "stress_trial": CELL / "bracket_stress2.log",
    "ordinary_base": CELL / "mixed_valid.log",
    "ordinary_trial": CELL / "bracket_mixed.log",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric_lines(path):
    return [line for line in path.read_text(errors="replace").splitlines()
            if re.match(r"^\d+ ", line)]


def root_reference():
    spec = importlib.util.spec_from_file_location(
        "eos_reference", HERE.parents[1] / "tools/diagnose_rsg_lte_exchange.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    table = module.TableTemperature(EOS, 1e-18, 1e-9, 1)
    _, _, old, old_rad, new_rad, coupling = map(
        float, (CELL / "cases_mixed_stress.txt").read_text().split())
    def residual(gas):
        temp = table.temperature_and_slope(5, gas)[0]
        return gas-old + coupling*(7.5657e-20*temp**4-new_rad) + 1e8*(new_rad-old_rad)
    lower, upper = old*0.5, old*1.5
    assert residual(lower) < 0 < residual(upper)
    for _ in range(200):
        mid = 0.5*(lower+upper)
        if mid == lower or mid == upper:
            break
        if residual(mid) < 0:
            lower = mid
        else:
            upper = mid
    return old, 0.5*(lower+upper), residual


def csv_write(path, rows, columns):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    old, reference, residual = root_reference()
    stress = []
    for state in ("base", "trial"):
        line = numeric_lines(LOGS["stress_" + state])
        assert len(line) == 1
        _, gas, _, _, reported_residual = map(float, line[0].split())
        stress.append({"state": state, "old_gas": old,
                       "reference_gas": reference, "production_gas": gas,
                       "relative_root_error": abs(gas-reference)/old,
                       "normalized_residual": abs(residual(gas))/old,
                       "reported_raw_residual": reported_residual})
    ordinary_same = (numeric_lines(LOGS["ordinary_base"])
                     == numeric_lines(LOGS["ordinary_trial"]))
    assert ordinary_same and len(numeric_lines(LOGS["ordinary_base"])) == 11
    runs = []
    for name in ("table_base", "table_trial", "fixed_base", "fixed_trial"):
        data = LOGS[name].read_text(errors="replace")
        linear = [(int(a), float(b)) for a, b in re.findall(
            r"\[LINEAR_FGMRES\] iterations=(\d+) relative_residual=([0-9.e+-]+)",
            data)]
        cycles = [int(c) for c in re.findall(r"^cycle=(\d+)", data, re.M)]
        runs.append({"case": name, "completed_cycles": max(cycles),
                     "linear_solves": len(linear),
                     "first_iterations": linear[0][0],
                     "first_true_residual": linear[0][1],
                     "second_iterations": linear[1][0],
                     "second_true_residual": linear[1][1],
                     "fatal_fgmres": "FGMRES did not reach" in data,
                     "log_sha256": digest(LOGS[name])})
    assert runs[0]["second_true_residual"] == runs[1]["second_true_residual"]
    assert runs[2]["second_true_residual"] == runs[3]["second_true_residual"]
    csv_write(OUT / "stress.csv", stress, list(stress[0]))
    csv_write(OUT / "runs.csv", runs, list(runs[0]))
    patch = "".join(difflib.unified_diff(
        (BASE / "src/nr_multigrid/NRFLD.cpp").read_text().splitlines(True),
        (TRIAL / "src/nr_multigrid/NRFLD.cpp").read_text().splitlines(True),
        fromfile="base/src/nr_multigrid/NRFLD.cpp",
        tofile="trial/src/nr_multigrid/NRFLD.cpp"))
    (OUT / "bracket.patch").write_text(patch)
    summary = {
        "base_binary_sha256": digest(BASE / "bin/athena"),
        "trial_binary_sha256": digest(TRIAL / "bin/athena"),
        "patch_sha256": digest(OUT / "bracket.patch"),
        "ordinary_11_case_numeric_results_identical": ordinary_same,
        "stress": stress, "runs": runs,
        "clean_branch_integration_observation":
            "The committed protostellar_core_fld pgen requires hydro/gamma "
            "and initializes an ideal gas; table-EOS pgen trial is a separate "
            "uncommitted dependency. No source change was committed.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].bar(["original", "expanded"],
                [r["normalized_residual"] for r in stress],
                color=["tab:orange", "tab:blue"])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("synthetic mixed-frame gas residual / old gas")
    for group, color in (("table", "tab:orange"), ("fixed", "tab:blue")):
        values = [r["second_true_residual"] for r in runs
                  if r["case"].startswith(group)]
        axes[1].plot([0, 1], values, "o-", color=color, label=group)
    axes[1].axhline(2e-8, color="black", linestyle="--", linewidth=1)
    axes[1].set_xticks([0, 1], ["original", "expanded"])
    axes[1].set_ylabel("second FGMRES true relative residual")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "bracket_trial.png", dpi=170)
    plt.close(fig)
    with tarfile.open(OUT / "evidence.tar.gz", "w:gz") as archive:
        for name, path in LOGS.items():
            archive.add(path, arcname="logs/" + name + ".log")
        for name in ("cases_mixed_valid.txt", "cases_mixed_stress.txt",
                     "local_probe.cpp", "local_probe_stress2.cpp",
                     "clean_valid.log"):
            archive.add(CELL / name, arcname="cell/" + name)
        for name in ("input_table_component_freeze_cfl030_strict_3cycles.used",
                     "input_table_fixed_rosseland_high_cfl030_strict_3cycles.used"):
            archive.add(RUN / name, arcname="inputs/" + name)
        archive.add(TRIAL / "build_local_bracket.log",
                    arcname="build/build_local_bracket.log")
    print(json.dumps({"stress": stress,
                      "ordinary_identical": ordinary_same,
                      "runs": [r["case"] for r in runs]}, indent=2))


if __name__ == "__main__":
    main()
