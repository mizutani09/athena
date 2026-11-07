#!/usr/bin/env python3
"""Archive and compare bounded WOR-92 opacity and ghost-boundary trials."""

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
OUT = HERE / "wor92_domain_guard_followup_results_20261107"
ROOT = Path("/home/kosuke/simulation/test/rad_nr/wor92_wor105_retest_20261107")
CASES = {
    "clamp_guard_to82": "table_domain_guard_restart74_to82_2rank",
    "clamp_no_guard_to82": "table_domain_clamp_no_guard_restart74_to82_2rank",
    "clamp_guard_to100_failure": "table_domain_guard_restart74_to100_2rank",
    "ghost_location_failure": "table_domain_ghost_diag_restart74_to88_2rank",
    "ghost_floor_to100": "table_domain_ghost_clip_restart74_to100_2rank",
    "mean_preserving_to100": "table_domain_mean_limiter_restart74_to100_2rank",
    "mean_preserving_clean_to90": "table_domain_mean_limiter_clean_restart74_to90_2rank",
    "cfl015_guard_to110": "table_domain_guard_cfl015_restart74_to110_2rank",
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    all_temperatures = []
    for label, dirname in CASES.items():
        path = ROOT / dirname
        log = path / "run.log"
        lines = log.read_text(errors="replace").splitlines()
        cycles = re.findall(r"^cycle=(\d+) time=(\S+) dt=(\S+)", "\n".join(lines), re.M)
        accepted = re.findall(r"\[WOR92_OPACITY_ACCEPTED\] cycle=(\d+) Tmin_K=(\S+)", "\n".join(lines))
        diag = re.findall(r"FLD_OPACITY_DIAGNOSTICS out_of_domain=(\d+) clamped=(\d+) negative_results=(\d+) zero_results=(\d+) nonfinite_results=(\d+)", "\n".join(lines))
        clips = re.findall(r"\[WOR92_NR_GHOST_CLIP\][^\n]+", "\n".join(lines))
        counts = [int(re.search(r"count=(\d+)", line).group(1)) for line in clips]
        errors = re.findall(r"Er at \(k,j,i\)=\([^\n]+", "\n".join(lines))
        row = {"case": label, "run_dir": dirname, "log_sha256": sha(log),
               "first_cycle": int(cycles[0][0]), "last_cycle_line": int(cycles[-1][0]),
               "last_time": float(cycles[-1][1]), "last_dt": float(cycles[-1][2]),
               "newton_converged": sum("[NR] status=converged" in line for line in lines),
               "fgmres_solves": sum("[LINEAR_FGMRES] iterations=" in line for line in lines),
               "accepted_checks": len(accepted),
               "accepted_tmin_min_K": min((float(t) for _, t in accepted), default=None),
               "ghost_limiter_calls": len(clips), "ghost_values_modified": sum(counts),
               "opacity_out_of_domain": int(diag[-1][0]) if diag else None,
               "opacity_clamped": int(diag[-1][1]) if diag else None,
               "radiation_ghost_error": errors[-1] if errors else "",
               "stopped_on_cycle_limit": any("Terminating on cycle limit" in line for line in lines)}
        hst = next(path.glob("*.hst"), None)
        if hst:
            row["history_sha256"] = sha(hst)
            data = np.atleast_2d(np.loadtxt(hst))
            row["history_rows"] = len(data)
            row["last_rho_max_cgs"] = float(data[-1, 15])
            row["last_Tgas_max_K"] = float(data[-1, 19])
        rows.append(row)
        for cycle, temp in accepted:
            all_temperatures.append({"case": label, "cycle": int(cycle), "Tmin_K": float(temp)})
    by = {row["case"]: row for row in rows}
    assert by["clamp_guard_to82"]["newton_converged"] == 9
    assert by["clamp_guard_to82"]["opacity_clamped"] == 560
    assert by["clamp_guard_to82"]["accepted_tmin_min_K"] > 5.0
    assert by["ghost_floor_to100"]["newton_converged"] == 27
    assert by["mean_preserving_to100"]["newton_converged"] == 27
    assert by["cfl015_guard_to110"]["last_time"] > 2156.43
    for name, other in (("clamp_guard_to82", "clamp_no_guard_to82"),
                        ("ghost_floor_to100", "mean_preserving_to100")):
        assert by[name]["fgmres_solves"] > 0 and by[other]["fgmres_solves"] > 0

    with (OUT / "runs.csv").open("w", newline="") as stream:
        keys = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "accepted_temperatures.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case", "cycle", "Tmin_K"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_temperatures)

    prefix = (ROOT / CASES["clamp_guard_to100_failure"] / "cfl030_flux_precision16_local.hst").read_bytes().splitlines()
    floor = (ROOT / CASES["ghost_floor_to100"] / "cfl030_flux_precision16_local.hst").read_bytes().splitlines()
    mean = (ROOT / CASES["mean_preserving_to100"] / "cfl030_flux_precision16_local.hst").read_bytes().splitlines()
    control_patterns = [r"^cycle=", r"^\[NR\] status=", r"\[LINEAR_FGMRES\] iterations="]
    control_equal = {}
    for pattern in control_patterns:
        a = [line for line in (ROOT / CASES["clamp_guard_to82"] / "run.log").read_text(errors="replace").splitlines() if re.search(pattern, line)]
        b = [line for line in (ROOT / CASES["clamp_no_guard_to82"] / "run.log").read_text(errors="replace").splitlines() if re.search(pattern, line)]
        control_equal[pattern] = a == b
    assert all(control_equal.values())
    assert (ROOT / CASES["clamp_guard_to82"] / "cfl030_flux_precision16_local.hst").read_bytes() == (ROOT / CASES["clamp_no_guard_to82"] / "cfl030_flux_precision16_local.hst").read_bytes()
    assert prefix == floor[:len(prefix)] == mean[:len(prefix)]
    clean_path = ROOT / CASES["mean_preserving_clean_to90"]
    prior_path = ROOT / "table_domain_mean_limiter_restart74_to90_2rank"
    def numerical_lines(path):
        return [line for line in path.read_text(errors="replace").splitlines()
                if re.search(r"^cycle=|^\[NR\] status=|\[LINEAR_FGMRES\] iterations=", line)]
    assert numerical_lines(clean_path / "run.log") == numerical_lines(prior_path / "run.log")
    assert (clean_path / "cfl030_flux_precision16_local.hst").read_bytes() == (prior_path / "cfl030_flux_precision16_local.hst").read_bytes()

    trial_binary = Path("/home/kosuke/simulation/test/rad_nr/wor92_operator_audit_20261107/build_table_domain_guard_trial/bin/athena")
    checkpoint = ROOT / "table_floor32_restart30_to80_2rank/cfl030_flux_precision16_local.00002.rst"
    input_file = ROOT / "input_table_domain_guard_cfl030_restart.used"
    summary = {"source_commits": ["6da81d20", "ff632195", "14633c31"],
               "original_trial_binary_sha256": "e7854e44e35a3ed85f9ab1963a801c1b09a6736db7356c6ff53dd2c0519217da",
               "clean_trial_binary_sha256": sha(trial_binary),
               "checkpoint_sha256": sha(checkpoint), "input_sha256": sha(input_file),
               "guard_on_off_trace_equal": control_equal,
               "guard_on_off_history_byte_equal": True,
               "through_cycle86_history_byte_equal": True,
               "clean_ghost_replay_trace_and_history_equal": True,
               "cases": by,
               "limitations": ["The mean-preserving positive NR ghost method is committed as opt-in after a separate clean replay; the simple floor and failure-location instrumentation remain archived trials. Full first-core collapse is outstanding.",
                               "The accepted-state opacity check runs after implicit solve but before AMR regrid.",
                               "CFL=0.15 from cycle 74 follows a different trajectory; it does not establish CFL convergence."]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    for label, name, color in (("CFL 0.3 ghost floor", "ghost_floor_to100", "#3577a5"),
                               ("CFL 0.3 mean limiter", "mean_preserving_to100", "#d59037"),
                               ("CFL 0.15 no limiter", "cfl015_guard_to110", "#65956a")):
        hist = np.atleast_2d(np.loadtxt(next((ROOT / CASES[name]).glob("*.hst"))))
        axes[0].plot(hist[:, 0], hist[:, 15], label=label, color=color)
        temps = [r for r in all_temperatures if r["case"] == name]
        cycles = np.array([r["cycle"] for r in temps])
        tempvals = np.array([r["Tmin_K"] for r in temps])
        axes[1].plot(cycles, tempvals, label=label, color=color)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("code time")
    axes[0].set_ylabel("maximum density (g cm⁻³)")
    axes[1].axhline(5.0, color="crimson", linestyle="--", label="table lower bound")
    axes[1].set_xlabel("cycle")
    axes[1].set_ylabel("minimum accepted active T (K)")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "continuation.png", dpi=180)
    plt.close(fig)

    with tarfile.open(OUT / "evidence.tar.gz", "w:gz") as archive:
        for label, dirname in CASES.items():
            path = ROOT / dirname
            archive.add(path / "run.log", arcname=f"logs/{label}.log")
            hst = next(path.glob("*.hst"), None)
            if hst:
                archive.add(hst, arcname=f"history/{label}.hst")
        archive.add(input_file, arcname="inputs/input_table_domain_guard_cfl030_restart.used")
        archive.add(OUT / "ghost_trial.patch.gz", arcname="patches/ghost_trial.patch.gz")
    print(json.dumps({"cases": len(rows), "result_dir": str(OUT), "binary_sha256": sha(trial_binary)}, indent=2))


if __name__ == "__main__":
    main()
