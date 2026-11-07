#!/usr/bin/env python3
"""Compare same-checkpoint local CFL controls at one physical time."""

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
import h5py


HERE = Path(__file__).resolve().parent
ROOT = Path("/home/kosuke/simulation/test/rad_nr/wor92_wor105_retest_20261107")
OUT = HERE / "wor92_matched_time_results_20261107"
RUNS = {
    "0.3": "table_clean_guard_ghost_restart100_to120_2rank",
    "0.15": "table_clean_guard_ghost_cfl015_restart100_to_time2436_2rank",
    "0.075": "table_clean_guard_ghost_cfl0075_restart100_to_time2436_2rank",
}
FINAL_DUMP_03 = ROOT / "table_clean_guard_ghost_cfl030_final_state_dump_2rank"
INITIAL_DUMP = ROOT / "table_cycle100_initial_state_dump_2rank"
STRICT_CONTROL = ROOT / "table_clean_strict_no_ghost_restart100_to120_2rank"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_out(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    cases = []
    trajectories = []
    with tarfile.open(OUT / "evidence.tar.gz", "w:gz") as archive:
        for cfl, folder in RUNS.items():
            run = ROOT / folder
            log = run / "run.log"
            hst = run / "cfl030_flux_precision16_local.hst"
            rst = run / "cfl030_flux_precision16_local.final.rst"
            assert all(p.is_file() for p in (log, hst, rst))
            s = log.read_text(errors="replace")
            end = re.findall(r"^time=(\S+) cycle=(\d+)$", s, re.M)
            assert end, folder
            time, end_cycle = float(end[-1][0]), int(end[-1][1])
            target = 2436.7392236433016
            assert abs(time - target) < 1e-8, (cfl, time)
            stages = re.findall(r"^\[NR\] status=(\w+) iterations=(\d+).*?final_max=(\S+).*?cycle=(\d+) time=(\S+) stage=(\d+) dt=(\S+)", s, re.M)
            stages = [x for x in stages if x[5] == "0"]
            assert len(stages) == end_cycle - 100 and all(x[0] == "converged" for x in stages)
            op = re.findall(r"^FLD_OPACITY_DIAGNOSTICS out_of_domain=(\d+) clamped=(\d+) negative_results=(\d+) zero_results=(\d+) nonfinite_results=(\d+)", s, re.M)
            assert len(op) == 1 and all(int(x) == 0 for x in op[0])
            accepted = re.findall(r"^\[WOR92_OPACITY_ACCEPTED\] cycle=(\d+) Tmin_K=(\S+)", s, re.M)
            assert len(accepted) == len(stages) and min(float(x[1]) for x in accepted) > 5
            fg = re.findall(r"^\[LINEAR_FGMRES\] iterations=(\d+) relative_residual=(\S+)", s, re.M)
            assert fg
            assert max(float(x[1]) for x in fg) < 2e-8
            assert "[WOR92_NR_GHOST_CLIP]" not in s
            assert "created, 0 destroyed" in s
            h = np.atleast_2d(np.loadtxt(hst))
            assert abs(float(h[-1, 0]) - target) < 0.01
            for row in h:
                trajectories.append(dict(cfl=float(cfl), time=float(row[0]),
                                         rho_max_cgs=float(row[15]), Tgas_max_K=float(row[19]),
                                         Trad_max_K=float(row[20]), Eint_gas_cgs=float(row[30]),
                                         Erad_tot_cgs=float(row[33])))
            cases.append(dict(cfl=float(cfl), end_cycle=end_cycle, end_time=time,
                              newton_converged=len(stages), max_newton_iterations=max(int(x[1]) for x in stages),
                              max_final_newton_residual=max(float(x[2]) for x in stages),
                              dt_min=min(float(x[6]) for x in stages), dt_max=max(float(x[6]) for x in stages),
                              fgmres_solves=len(fg), fgmres_true_residual_max=max(float(x[1]) for x in fg),
                              accepted_Tmin_min_K=min(float(x[1]) for x in accepted),
                              rho_max_cgs=float(h[-1, 15]), Tgas_max_K=float(h[-1, 19]),
                              Trad_max_K=float(h[-1, 20]), gas_mass_g=float(h[-1, 22]),
                              Lz_cgs=float(h[-1, 24]), Eint_gas_cgs=float(h[-1, 30]),
                              Erad_tot_cgs=float(h[-1, 33]),
                              log_sha256=sha(log), history_sha256=sha(hst),
                              restart_sha256=sha(rst)))
            for path in (log, hst):
                archive.add(path, arcname=f"cfl{cfl}/{path.name}")
            for path in run.glob("*.prim.*.athdf"):
                archive.add(path, arcname=f"cfl{cfl}/{path.name}")
        dump_path = next(FINAL_DUMP_03.glob("*.prim.*.athdf"))
        archive.add(dump_path, arcname=f"cfl0.3/{dump_path.name}")
        archive.add(FINAL_DUMP_03 / "run.log", arcname="cfl0.3/final_state_dump.log")
        for path in (INITIAL_DUMP / "run.log", INITIAL_DUMP / "cfl030_flux_precision16_local.hst",
                     next(INITIAL_DUMP.glob("*.prim.*.athdf"))):
            archive.add(path, arcname=f"initial/{path.name}")
        for path in (STRICT_CONTROL / "run.log", STRICT_CONTROL / "cfl030_flux_precision16_local.hst"):
            archive.add(path, arcname=f"cfl0.3_strict/{path.name}")
    csv_out(OUT / "cases.csv", cases)
    csv_out(OUT / "trajectories.csv", trajectories)
    fields = {}
    for cfl, folder in RUNS.items():
        run = FINAL_DUMP_03 if cfl == "0.3" else ROOT / folder
        hdf5_path = next(run.glob("*.prim.*.athdf"))
        with h5py.File(hdf5_path) as stream:
            loc = np.column_stack((stream["Levels"][:], stream["LogicalLocations"][:]))
            order = np.lexsort(tuple(loc[:, i] for i in range(3, -1, -1)))
            fields[cfl] = (loc[order], stream["prim"][:][:, order].astype(np.float64))
            assert abs(float(stream.attrs["Time"]) - 2436.7392236433016) < 1e-3
        assert np.array_equal(fields[cfl][0], fields["0.3"][0])
    field_comparison = []
    for field, index in (("rho", 0), ("press", 1), ("Er", 8)):
        for reference, candidate in (("0.3", "0.15"), ("0.3", "0.075"), ("0.15", "0.075")):
            x = fields[reference][1][index]
            y = fields[candidate][1][index]
            delta = y - x
            field_comparison.append(dict(field=field, reference_cfl=reference, candidate_cfl=candidate,
                                         relative_l1=float(np.sum(np.abs(delta)) / np.sum(np.abs(x))),
                                         relative_l2=float(np.linalg.norm(delta.ravel()) / np.linalg.norm(x.ravel())),
                                         relative_max=float(np.max(np.abs(delta)) / np.max(np.abs(x)))))
    csv_out(OUT / "fields.csv", field_comparison)
    initial_history = np.atleast_2d(np.loadtxt(INITIAL_DUMP / "cfl030_flux_precision16_local.hst"))
    source_history = np.atleast_2d(np.loadtxt(ROOT / "table_domain_mean_limiter_restart74_to100_2rank/cfl030_flux_precision16_local.hst"))
    assert np.array_equal(initial_history[-1], source_history[-1])
    normal = ROOT / RUNS["0.3"]
    normal_log = (normal / "run.log").read_text(errors="replace").splitlines()
    strict_log = (STRICT_CONTROL / "run.log").read_text(errors="replace").splitlines()
    equal_traces = {}
    for label, pattern in (("cycle", r"^cycle="), ("newton", r"^\[NR\] status="),
                           ("fgmres", r"^\[LINEAR_FGMRES\] iterations="),
                           ("local_gas", r"^\[NR_LOCAL_GAS\]")):
        def selected(lines):
            chosen = [line for line in lines if re.search(pattern, line)]
            if label == "fgmres":
                chosen = [re.sub(r"inexact_accept=\d+ inexact_cap=\S+", "", line) for line in chosen]
            return chosen
        equal_traces[label] = selected(normal_log) == selected(strict_log)
    assert all(equal_traces.values())
    assert (normal / "cfl030_flux_precision16_local.hst").read_bytes() == (STRICT_CONTROL / "cfl030_flux_precision16_local.hst").read_bytes()
    assert set(re.findall(r"inexact_accept=(\d+)", "\n".join(strict_log))) == {"0"}
    by = {c["cfl"]: c for c in cases}
    keys = ["rho_max_cgs", "Tgas_max_K", "Trad_max_K", "gas_mass_g", "Lz_cgs", "Eint_gas_cgs", "Erad_tot_cgs"]
    comparison = {str(cfl): {
        f"{key}_relative_to_cfl_0p3": (by[cfl][key] - by[0.3][key]) / by[0.3][key]
        for key in keys
    } for cfl in (0.15, 0.075)}
    summary = dict(start_cycle=100, matched_end_time=2436.7392236433016,
                   source_branch="codex/wor92-domain-guard-trial-20261107",
                   binary_sha256=sha(Path("/home/kosuke/simulation/test/rad_nr/wor92_operator_audit_20261107/build_table_domain_guard_trial/bin/athena")),
                   input_sha256=sha(ROOT / "input_table_domain_guard_cfl030_restart.used"),
                   start_restart_sha256=sha(ROOT / "table_domain_mean_limiter_restart74_to100_2rank/cfl030_flux_precision16_local.final.rst"),
                   initial_state_dump_sha256=sha(next(INITIAL_DUMP.glob("*.prim.*.athdf"))),
                   cfl_0p3_final_state_dump_sha256=sha(next(FINAL_DUMP_03.glob("*.prim.*.athdf"))),
                   cases=cases, comparison=comparison, field_comparison=field_comparison,
                   strict_no_ghost_control=dict(trace_equal=equal_traces, history_byte_equal=True,
                                                log_sha256=sha(STRICT_CONTROL / "run.log")),
                   limitations=["One matched-time interval from a shared checkpoint does not establish asymptotic temporal order.",
                                "Maximum temperature is a nonsmooth statistic and can move between cells.",
                                "All comparisons use the same 120-block mesh without regrid in this interval."])
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for cfl, color in ((0.3, "#2b6a9c"), (0.15, "#d68e2f"), (0.075, "#67945e")):
        rows = [r for r in trajectories if r["cfl"] == cfl]
        for ax, key in ((axes[0], "rho_max_cgs"), (axes[1], "Tgas_max_K")):
            ax.plot([r["time"] for r in rows], [r[key] for r in rows], label=f"CFL={cfl}", color=color)
    axes[0].set_ylabel("maximum density (g cm$^{-3}$)")
    axes[1].set_ylabel("maximum gas temperature (K)")
    for ax in axes:
        ax.set_xlabel("code time")
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "matched_time.png", dpi=160)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
