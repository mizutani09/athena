#!/usr/bin/env python3
"""Summarize isolated WOR-92 FGMRES correction-rounding controls."""

import csv
import difflib
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
OUT = HERE / "wor92_roundoff_mechanism_results_20261107"
RUNS = Path("/home/kosuke/simulation/test/rad_nr/wor92_wor105_retest_20261107")
BUILDS = Path("/home/kosuke/simulation/test/rad_nr/wor92_operator_audit_20261107")
CASES = {
    "table_baseline": ("table_local_jacobian_original_binary_strict_2rank", "build_table_precision"),
    "table_trace": ("table_gap_roundoff_strict_2rank", "build_table_step_trace_trial"),
    "fixed_rosseland_trace": ("fixed_high_gap_decompose_strict_2rank", "build_table_step_trace_trial"),
    "uniform_trace": ("uniform_gap_decompose_strict_2rank", "build_table_step_trace_trial"),
    "table_kahan": ("table_kahan_strict_2rank", "build_table_kahan_trial"),
    "table_long_double_update": ("table_long_double_update_strict_2rank", "build_table_long_double_update_trial"),
}
INPUTS = [
    "input_table_component_freeze_cfl030_strict_3cycles.used",
    "input_table_fixed_rosseland_high_cfl030_strict_3cycles.used",
    "input_uniform_l1_restart40_mg2_strict.used",
]
FIELDS = ["iterations", "relative_residual", "target", "restart", "mg_vcycles", "precision_floor_accept", "breakdown"]
STEP_FIELDS = ["iteration", "projected", "predicted", "true", "operator_gap", "update_rep_gap",
               "operator_eval_gap", "gap_interior", "gap_same_face", "gap_cf_face", "xnorm",
               "requested_step", "realized_step", "step_roundoff", "step_cosine", "stale_cells",
               "requested_cells"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fields(line):
    return dict(re.findall(r"\b([A-Za-z_]+)=([^\s]+)", line))


def number(value):
    if value is None:
        return None
    return float(value) if any(c in value for c in ".eE") else int(value)


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    details = {}
    for label, (directory, build) in CASES.items():
        log = RUNS / directory / "run.log"
        content = log.read_text(errors="replace")
        solves = []
        final_steps = []
        last_step = None
        for line in content.splitlines():
            if "[LINEAR_FGMRES_STEP]" in line:
                last_step = fields(line)
            if "[LINEAR_FGMRES] iterations=" in line:
                solves.append(fields(line))
                final_steps.append(last_step)
                last_step = None
        assert len(solves) >= 2, (label, len(solves))
        assert all(key in solves[1] for key in FIELDS), label
        row = {"case": label, "run_dir": directory,
               "log_sha256": sha256(log),
               "binary_sha256": sha256(BUILDS / build / "bin/athena")}
        for index in (0, 1):
            for key in FIELDS:
                row[f"solve{index+1}_{key}"] = number(solves[index][key])
        if final_steps[1]:
            for key in STEP_FIELDS:
                row[key] = number(final_steps[1].get(key))
        rows.append(row)
        details[label] = {"log": str(log), "build": str(BUILDS / build),
                          "solves": solves[:2], "second_solve_final_step": final_steps[1]}
    by = {row["case"]: row for row in rows}
    assert by["table_baseline"]["solve2_relative_residual"] == by["table_trace"]["solve2_relative_residual"]
    assert by["table_trace"]["requested_cells"] == 61440
    assert by["uniform_trace"]["stale_cells"] == by["uniform_trace"]["requested_cells"] == 262144

    patches = {}
    original = (BUILDS / "build_table_precision/src/linear_solver/linearMG/linearMG.cpp").read_text().splitlines(keepends=True)
    for name, build in (("step_trace", "build_table_step_trace_trial"),
                        ("kahan", "build_table_kahan_trial"),
                        ("long_double_update", "build_table_long_double_update_trial")):
        modified = (BUILDS / build / "src/linear_solver/linearMG/linearMG.cpp").read_text().splitlines(keepends=True)
        patch = "".join(difflib.unified_diff(original, modified, fromfile="a/src/linear_solver/linearMG/linearMG.cpp",
                                             tofile="b/src/linear_solver/linearMG/linearMG.cpp"))
        assert patch
        patches[f"{name}.patch"] = patch.encode()

    with (OUT / "solves.csv").open("w", newline="") as stream:
        keys = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "base_git_sha": "34ebf35869833aedb35cedf8306adc38ccbd0e0a",
        "condition": "Tomida table EoS, Semenov opacity, CFL=0.3, 32^3 root, small AMR, 2 MPI ranks, actual dt=31.114819909, FGMRES target=2e-8",
        "cases": details,
        "controls": {name: {key: value for key, value in row.items() if key != "run_dir"} for name, row in by.items()},
        "interpretation": "The projected correction residual matches the independently predicted residual. Most mixed-AMR true-residual gap follows finite-precision representation of the updated iterate; the uniform correction is entirely below double ulp. Kahan and long-double one-cast update do not meet the strict target.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    labels = ["baseline", "Kahan", "long double", "fixed opacity", "uniform"]
    selected = [by[key] for key in ("table_baseline", "table_kahan", "table_long_double_update",
                                    "fixed_rosseland_trace", "uniform_trace")]
    axes[0].bar(labels, [r["solve2_relative_residual"] * 1e8 for r in selected], color="#3577a5")
    axes[0].axhline(2.0, color="crimson", linestyle="--", label="target")
    axes[0].set_ylabel("second true residual (×10⁻⁸)")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend()
    keys = ["projected", "predicted", "true", "operator_gap", "update_rep_gap", "operator_eval_gap"]
    colors = ["#8b9dc3", "#8b9dc3", "#3577a5", "#d59037", "#c5604c", "#8073ac"]
    axes[1].bar(["projected", "predicted", "true", "gap", "update", "eval"],
                [by["table_trace"][key] * 1e8 for key in keys], color=colors)
    axes[1].axhline(2.0, color="crimson", linestyle="--")
    axes[1].set_ylabel("mixed-AMR second solve (×10⁻⁸)")
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(OUT / "roundoff_mechanism.png", dpi=180)
    plt.close(fig)

    with tarfile.open(OUT / "evidence.tar.gz", "w:gz") as archive:
        for label, (directory, _) in CASES.items():
            archive.add(RUNS / directory / "run.log", arcname=f"logs/{label}.log")
        for name in INPUTS:
            archive.add(RUNS / name, arcname=f"inputs/{name}")
        for name, payload in patches.items():
            info = tarfile.TarInfo(f"patches/{name}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    print(json.dumps({"rows": len(rows), "result_dir": str(OUT),
                      "table_second_residuals": {key: by[key]["solve2_relative_residual"] for key in by}}, indent=2))


if __name__ == "__main__":
    main()
