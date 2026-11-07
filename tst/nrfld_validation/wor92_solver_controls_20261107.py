"""Archive and compare bounded local CFL=0.3 solver controls."""

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
ROOT = Path("/home/kosuke/simulation/test/rad_nr/wor92_wor105_retest_20261107")
BUILD = Path("/home/kosuke/simulation/test/rad_nr/wor92_operator_audit_20261107")
OUT = HERE / "wor92_solver_controls_results_20261107"
CASES = {
    "Hessenberg long double": ("hessenberg_ld_strict_3cycles", "component_freeze_cfl030_strict_3cycles", "build_table_hessenberg_ld_trial"),
    "MG V-cycles 4": ("mg4_cfl030_strict_3cycles", "mg4_cfl030_strict_3cycles", "build_table_precision"),
    "MG V-cycles 5": ("mg5_cfl030_strict_3cycles", "mg5_cfl030_strict_3cycles", "build_table_precision"),
    "sweeps 4+4": ("sweeps4_cfl030_strict_3cycles", "sweeps4_cfl030_strict_3cycles", "build_table_precision"),
    "omega 0.7": ("omega070_cfl030_strict_3cycles", "omega070_cfl030_strict_3cycles", "build_table_precision"),
    "omega 0.8": ("omega080_cfl030_strict_3cycles", "omega080_cfl030_strict_3cycles", "build_table_precision"),
    "omega 0.9": ("omega090_cfl030_strict_3cycles", "omega090_cfl030_strict_3cycles", "build_table_precision"),
    "no basis reuse": ("no_basis_reuse_cfl030_strict_3cycles", "no_basis_reuse_cfl030_strict_3cycles", "build_table_precision"),
    "noncompact basis": ("noncompact_basis_cfl030_strict_3cycles", "noncompact_basis_cfl030_strict_3cycles", "build_table_precision"),
    "AMR BE upper": ("be_upper_mixed", "component_freeze_cfl030_strict_3cycles", "build_table_be_upper_trial"),
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def fields(line):
    return dict(re.findall(r"([A-Za-z_][A-Za-z_0-9]*)=([^\s]+)", line))

def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    evidence = {}
    for name, (stem, input_stem, build) in CASES.items():
        log = ROOT / f"table_{stem}_2rank.log"
        inp = ROOT / f"input_table_{input_stem}.used"
        binary = BUILD / build / "bin/athena"
        assert log.exists() and inp.exists() and binary.exists(), name
        solves = [fields(line) for line in log.read_text(errors="replace").splitlines()
                  if "[LINEAR_FGMRES] iterations=" in line]
        upper = [fields(line) for line in log.read_text(errors="replace").splitlines()
                 if "[LINEAR_FGMRES_BE_UPPER]" in line]
        assert len(solves) == (0 if name == "sweeps 4+4" else 2), (name, len(solves))
        for index, solve in enumerate(solves):
            rows.append({"case": name, "solve": index + 1,
                         "iterations": solve["iterations"],
                         "relative_residual": solve["relative_residual"],
                         "target": solve["target"],
                         "strict_pass": float(solve["relative_residual"]) <= float(solve["target"]),
                         "be_upper": upper[index]["max"] if upper else "",
                         "be_upper_above_eps": upper[index]["above_eps"] if upper else "",
                         "log_sha256": sha(log), "input_sha256": sha(inp),
                         "binary_sha256": sha(binary)})
        evidence[log.name] = log
        evidence[inp.name] = inp
    with (OUT / "solves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with tarfile.open(OUT / "evidence.tar.gz", "w:gz") as tar:
        for name, path in evidence.items():
            tar.add(path, arcname=name)
        tar.add(BUILD / "build_table_hessenberg_ld_trial/src/linear_solver/linearMG/linearMG.cpp",
                arcname="hessenberg_ld_trial_linearMG.cpp")
        tar.add(BUILD / "build_table_be_upper_trial/src/linear_solver/linearMG/linearMG.cpp",
                arcname="be_upper_trial_linearMG.cpp")
    summary = {"cases": len(CASES), "solves": len(rows),
               "strict_failed_second_solve": sum(r["solve"] == 2 and not r["strict_pass"] for r in rows),
               "rows": rows}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    second = [r for r in rows if r["solve"] == 2]
    fig, ax = plt.subplots(figsize=(9, 4.7))
    ax.barh([r["case"] for r in second],
            [float(r["relative_residual"]) / 2e-8 for r in second])
    ax.axvline(1, color="red", linestyle="--", label="strict target")
    ax.set_xlabel("Second true residual / requested target")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "second_solve.png", dpi=160)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))

if __name__ == "__main__":
    main()
