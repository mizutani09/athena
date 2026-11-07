"""Summarize the isolated general-EOS local-gas-solve controls for WOR-92."""

import csv
import hashlib
import json
import re
import tarfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/home/kosuke/simulation")
RUNS = ROOT / "test/rad_nr/wor92_wor105_retest_20261107"
BUILD = ROOT / "test/rad_nr/wor92_operator_audit_20261107/build_table_precision"
IDEAL_BUILD = ROOT / "test/rad_nr/wor92_20261107/build_ideal"
HERE = Path(__file__).resolve().parent
OUT = HERE / "wor92_local_general_eos_results_20261107"

CASES = {
    "table_on_floor32": (
        RUNS / "table_local_jacobian_floor32_cfl030_10cycles_2rank/run.log",
        RUNS / "input_table_local_jacobian_floor32_cfl030_10cycles.used",
        False,
    ),
    "table_off_floor32": (
        RUNS / "table_local_general_off_floor32_cfl030_10cycles_2rank.log",
        RUNS / "input_table_local_general_off_floor32_cfl030_10cycles.used",
        False,
    ),
    "fixed_r_on_strict": (
        RUNS / "table_coupled_global_jv_v3_fixed_rosseland_high_2rank/run.log",
        RUNS / "input_table_fixed_rosseland_high_cfl030_strict_3cycles.used",
        True,
    ),
    "fixed_r_off_strict": (
        RUNS / "table_local_general_off_fixed_rosseland_high_env_cfl030_strict_3cycles_2rank.log",
        RUNS / "input_table_local_general_off_fixed_rosseland_high_cfl030_strict_3cycles.used",
        True,
    ),
    "table_on_ledger": (
        RUNS / "table_local_general_on_floor32_cfl030_1cycle_ledger_2rank.log",
        RUNS / "input_table_local_general_on_floor32_cfl030_1cycle_ledger.used",
        False,
    ),
}
IDEAL = {
    state: (
        RUNS / f"ideal_local_flag_{state}_cfl030_1cycle_2rank.log",
        RUNS / f"input_ideal_local_flag_{state}_cfl030_1cycle.used",
    ) for state in ("on", "off")
}


def fields(line):
    return dict(re.findall(r"([A-Za-z_][A-Za-z_0-9]*)=([^\s]+)", line))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse(name, path, inp, fixed_r):
    data = path.read_bytes().decode("utf-8", errors="replace")
    result = {
        "case": name,
        "log_sha256": sha(path),
        "input_sha256": sha(inp),
        "fixed_rosseland_env": fixed_r,
        "completed_cycles": 0,
        "terminated_on_cycle_limit": "Terminating on cycle limit" in data,
        "fatal_fgmres": "FGMRES did not reach its requested relative residual" in data,
        "linear": [], "trials": [], "local": [], "newton": [], "ledger": [],
    }
    cycle = -1
    for line in data.splitlines():
        if line.startswith("cycle="):
            m = re.match(r"cycle=(\d+)", line)
            if m:
                cycle = int(m.group(1))
                result["completed_cycles"] = max(result["completed_cycles"], cycle)
        elif "[LINEAR_FGMRES]" in line:
            # MPI may interleave its abort stack trace with the final line.
            result["linear"].append({
                "cycle": cycle,
                **fields(line.split("[LINEAR_FGMRES]", 1)[1])})
        elif line.startswith("[NR_TRIAL]"):
            result["trials"].append({"cycle": cycle, **fields(line)})
        elif line.startswith("[NR_LOCAL_GAS]"):
            result["local"].append({"cycle": cycle, **fields(line)})
        elif line.startswith("[NR] status=converged"):
            result["newton"].append(fields(line))
        elif line.startswith("[NR_ENERGY_LEDGER]"):
            result["ledger"].append(fields(line))
    result["linear_solves"] = len(result["linear"])
    result["linear_above_target"] = sum(
        float(row["relative_residual"]) > float(row["target"])
        for row in result["linear"])
    result["linear_floor_accepts"] = sum(
        int(row["precision_floor_accept"]) for row in result["linear"])
    result["first_trial_gas_l2"] = (
        float(result["trials"][0]["gas_l2"]) if result["trials"] else None)
    result["first_trial_l2"] = (
        float(result["trials"][0]["l2"]) if result["trials"] else None)
    result["max_logged_local_residual_over_gas_scale"] = max(
        (float(row["residual_over_gas_scale_max"])
         for row in result["local"]), default=None)
    return result


def csv_write(path, rows, columns):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: row.get(k) for k in columns} for row in rows)


def make_figure(cases):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex="row")
    for column, prefix in enumerate(("table", "fixed_r")):
        for state, color in (("on", "tab:blue"), ("off", "tab:orange")):
            key = prefix + "_" + state + ("_floor32" if prefix == "table" else "_strict")
            case = cases[key]
            trials = [r for r in case["trials"] if r["cycle"] == 0]
            axes[0, column].semilogy(
                [int(r["iteration"]) + 1 for r in trials],
                [float(r["gas_l2"]) for r in trials],
                "o-", color=color, label="local solve " + state)
            linear = [r for r in case["linear"] if r["cycle"] == 0]
            axes[1, column].semilogy(
                range(1, len(linear) + 1),
                [float(r["relative_residual"]) for r in linear],
                "o-", color=color, label="local solve " + state)
        axes[0, column].set_title(
            "Semenov table R, floor 32" if prefix == "table"
            else "fixed R=0.0652, strict")
        axes[1, column].axhline(2e-8, color="black", linestyle="--", linewidth=1)
        axes[0, column].set_xlabel("Newton correction in cycle 0")
        axes[1, column].set_xlabel("FGMRES solve in cycle 0")
        axes[0, column].grid(alpha=.2)
        axes[1, column].grid(alpha=.2)
        axes[0, column].legend()
    axes[0, 0].set_ylabel("gas residual L2")
    axes[1, 0].set_ylabel("true relative FGMRES residual")
    fig.tight_layout()
    fig.savefig(OUT / "local_general_eos_comparison.png", dpi=170)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = {name: parse(name, *args) for name, args in CASES.items()}
    summary = {
        "binary_sha256": sha(BUILD / "bin/athena"),
        "cases": cases,
    }
    ideal_rows = []
    ideal_trace_hashes = {}
    for state, (log, inp) in IDEAL.items():
        parsed = parse("ideal_" + state, log, inp, False)
        lines = [line for line in log.read_text(errors="replace").splitlines()
                 if line.startswith(("[LINEAR_FGMRES]", "[NR_TRIAL]",
                                     "[NR] status=converged", "cycle="))]
        trace = "\n".join(lines).encode()
        ideal_trace_hashes[state] = hashlib.sha256(trace).hexdigest()
        ideal_rows.append({"state": state,
                           "binary_sha256": sha(IDEAL_BUILD / "bin/athena"),
                           "input_sha256": sha(inp), "log_sha256": sha(log),
                           "numerical_trace_sha256": ideal_trace_hashes[state],
                           "completed_cycles": parsed["completed_cycles"],
                           "linear_solves": parsed["linear_solves"],
                           "newton_solves": len(parsed["newton"])})
    summary["ideal_flag_control"] = {
        "same_numerical_trace": ideal_trace_hashes["on"] == ideal_trace_hashes["off"],
        "build_defines": {"GENERAL_EOS": 0, "EOS_TABLE_ENABLED": 0},
        "runs": ideal_rows,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    csv_write(OUT / "runs.csv", [
        {k: v for k, v in case.items() if k not in
         {"linear", "trials", "local", "newton", "ledger"}}
        for case in cases.values()],
        ["case", "completed_cycles", "terminated_on_cycle_limit",
         "fatal_fgmres", "linear_solves", "linear_above_target",
         "linear_floor_accepts", "first_trial_l2", "first_trial_gas_l2",
         "max_logged_local_residual_over_gas_scale", "fixed_rosseland_env",
         "input_sha256", "log_sha256"])
    csv_write(OUT / "cycle0_trials.csv", [
        {"case": name, **row} for name, case in cases.items()
        for row in case["trials"] if row["cycle"] == 0],
        ["case", "cycle", "iteration", "trial", "accepted", "step_scale",
         "l2", "max", "gas_l2", "gas_max", "total_energy_l2",
         "total_energy_max"])
    csv_write(OUT / "linear.csv", [
        {"case": name, **row} for name, case in cases.items()
        for row in case["linear"]],
        ["case", "cycle", "iterations", "relative_residual", "target",
         "precision_floor_accept", "breakdown", "restart", "mg_vcycles"])
    csv_write(OUT / "ledger.csv", [
        {"case": name, **row} for name, case in cases.items()
        for row in case["ledger"]],
        ["case", "rank", "cycle", "stage", "dt", "gas_delta",
         "radiation_delta", "diffusion_delta", "marshak_delta", "balance",
         "relative_balance", "absolute_relative_balance"])
    csv_write(OUT / "ideal_flag_control.csv", ideal_rows,
              ["state", "binary_sha256", "input_sha256", "log_sha256",
               "numerical_trace_sha256", "completed_cycles",
               "linear_solves", "newton_solves"])
    make_figure(cases)
    with tarfile.open(OUT / "evidence.tar.gz", "w:gz") as archive:
        for name, (log, inp, _) in CASES.items():
            archive.add(log, arcname="logs/" + name + ".log")
            archive.add(inp, arcname="inputs/" + name + ".used")
        for state, (log, inp) in IDEAL.items():
            archive.add(log, arcname="logs/ideal_" + state + ".log")
            archive.add(inp, arcname="inputs/ideal_" + state + ".used")
        archive.add(BUILD / "src", arcname="build/src")
        archive.add(BUILD / "Makefile", arcname="build/Makefile")
        archive.add(BUILD / "configure.log", arcname="build/configure.log")
        archive.add(IDEAL_BUILD / "src/defs.hpp", arcname="ideal_build/defs.hpp")
        archive.add(IDEAL_BUILD / "src/nr_multigrid/NRFLD.cpp",
                    arcname="ideal_build/NRFLD.cpp")
        archive.add(IDEAL_BUILD / "configure.log",
                    arcname="ideal_build/configure.log")
    print(json.dumps({name: {
        "cycles": c["completed_cycles"],
        "linear": c["linear_solves"],
        "fatal": c["fatal_fgmres"],
        "first_gas_l2": c["first_trial_gas_l2"]}
        for name, c in cases.items()}, indent=2))


if __name__ == "__main__":
    main()
