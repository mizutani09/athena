"""Check a clean checkout plus archived candidate patch against the trial run."""

import hashlib
import json
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path("/home/kosuke/simulation/test/rad_nr/wor92_wor105_retest_20261107")
BUILD = Path("/home/kosuke/simulation/test/rad_nr/wor92_operator_audit_20261107/build_table_clean_localroot_trial")
OUT = HERE / "wor92_clean_localroot_results_20261107"
PATCH = OUT / "candidate.patch"
INPUT = ROOT / "input_table_inexact_cfl030_30cycles.used"
CASES = {
    "clean3": ROOT / "table_clean_localroot_inexact_3cycles_2rank",
    "clean30": ROOT / "table_clean_localroot_inexact_30cycles_2rank",
    "prior30": ROOT / "table_inexact_cfl030_30cycles_2rank",
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def trace(path):
    lines = path.read_text(errors="replace").splitlines()
    return {
        "linear": [line for line in lines if line.startswith("[LINEAR_FGMRES]")],
        "newton": [line for line in lines if line.startswith("[NR] status=converged") and "stage=0" in line],
        "cycles": [line for line in lines if line.startswith("cycle=")],
    }

def main():
    OUT.mkdir(exist_ok=True)
    assert PATCH.exists() and INPUT.exists() and (BUILD / "bin/athena").exists()
    t = {name:trace(directory / "run.log") for name,directory in CASES.items()}
    assert len(t["clean3"]["newton"]) == 3
    assert len(t["clean30"]["newton"]) == 30
    comparison = {}
    for label, count in (("clean3",3),("clean30",30)):
        comparison[label] = {key:t[label][key] == t["prior30"][key][:len(t[label][key])]
                             for key in t[label]}
        assert all(comparison[label].values()), comparison[label]
    history = "cfl030_flux_precision16_local.hst"
    history_equal = (CASES["clean30"]/history).read_bytes() == (CASES["prior30"]/history).read_bytes()
    assert history_equal
    summary = {
        "base_commit": "9c584cc107a4303476f3d36dc180b57d0c1dfc27",
        "candidate_patch_sha256": sha(PATCH),
        "binary_sha256": sha(BUILD/"bin/athena"),
        "input_sha256": sha(INPUT),
        "comparison": comparison,
        "history_30_bitwise_equal": history_equal,
        "clean_build_has_rank_face_and_energy_ledgers": False,
        "run_log_sha256": {name:sha(directory/"run.log") for name,directory in CASES.items()},
        "build_log_sha256": sha(BUILD/"build_inexact.log"),
        "configure_log_sha256": sha(BUILD/"configure.log"),
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    with tarfile.open(OUT/"evidence.tar.gz","w:gz") as archive:
        archive.add(PATCH,arcname="candidate.patch")
        archive.add(INPUT,arcname="input.used")
        archive.add(BUILD/"configure.log",arcname="configure.log")
        archive.add(BUILD/"build.log",arcname="build_floor.log")
        archive.add(BUILD/"build_inexact.log",arcname="build_inexact.log")
        for name in ("clean3","clean30"):
            archive.add(CASES[name]/"run.log",arcname=f"{name}/run.log")
        archive.add(CASES["clean30"]/history,arcname=f"clean30/{history}")
    print(json.dumps(summary,indent=2))

if __name__ == "__main__":
    main()
