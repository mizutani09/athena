#!/usr/bin/env python3
"""Audit the central eight cells and Jeans resolution in WOR-92 local controls."""

import csv
import hashlib
import json
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
OUT = HERE / "wor92_core_symmetry_results_20261107"
RUNS = {
    "cfl030_2rank": "table_symmetry_trace74to82_2rank",
    "cfl030_1rank": "table_symmetry_trace74to82_1rank",
    "cfl015_2rank": "table_symmetry_cfl015_t2082_2rank",
    "cfl030_sinkoff": "table_symmetry_sinkoff_cfl030_t2082_2rank",
    "cfl030_fixed_opacity": "table_symmetry_fixed_opacity_guardoff_cfl030_t2082_2rank",
    "cfl030_level3_late": "table_symmetry_level3_cfl030_t2082_2rank",
    "cfl030_level3_early": "table_symmetry_level3_from36_t2082_2rank",
    "cfl030_level4_from63": "table_symmetry_level4_from63_one_step_2rank",
    "cfl030_level4_to70": "table_symmetry_level4_from64_to70_cycleio_dt0_2rank",
}
DUMP_DIRS = {
    "cfl030_level3_early": "table_symmetry_level3_cycle63_state_dump_corrected_2rank",
    "cfl030_level4_from63": "table_symmetry_level4_cycle64_state_dump_2rank",
}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def central_eight(path):
    with h5py.File(path) as data:
        xx, yy, zz = (data[name][:] for name in ("x1v", "x2v", "x3v"))
        fields = data["prim"][:]
        min_x, min_y, min_z = (np.min(np.abs(axis)) for axis in (xx, yy, zz))
        cells = []
        for block in range(len(xx)):
            for i in np.where(np.abs(xx[block]) == min_x)[0]:
                for j in np.where(np.abs(yy[block]) == min_y)[0]:
                    for k in np.where(np.abs(zz[block]) == min_z)[0]:
                        cells.append(dict(x_AU=float(xx[block, i]), y_AU=float(yy[block, j]),
                                          z_AU=float(zz[block, k]),
                                          rho_cgs=float(fields[0, block, k, j, i]) * 1e-18,
                                          press_cgs=float(fields[1, block, k, j, i]) * 1e-9,
                                          Er_cgs=float(fields[8, block, k, j, i]) * 1e-9,
                                          phi_code=float(fields[5, block, k, j, i])))
        assert len(cells) == 8, (path, len(cells))
        return cells, dict(cycle=int(data.attrs["NumCycles"]), time=float(data.attrs["Time"]),
                           blocks=int(data.attrs["NumMeshBlocks"]),
                           max_level=int(data.attrs["MaxLevel"]),
                           rho_max_cgs=float(np.max(fields[0])) * 1e-18,
                           pressure_min_code=float(np.min(fields[1])),
                           pressure_floor_cells=int(np.count_nonzero(fields[1] < 1e-9)))


def main():
    OUT.mkdir(exist_ok=True)
    centers, trajectories, cases, dump_diagnostics = [], [], [], []
    with tarfile.open(OUT / "evidence.tar.gz", "w:gz") as archive:
        for label, dirname in RUNS.items():
            folder = ROOT / dirname
            log = folder / "run.log"
            hst = folder / "cfl030_flux_precision16_local.hst"
            dumps = sorted((ROOT / DUMP_DIRS.get(label, dirname)).glob("*.prim.*.athdf"))
            assert log.exists() and hst.exists() and dumps, dirname
            history = np.atleast_2d(np.loadtxt(hst))
            lines = log.read_text(errors="replace")
            assert history.shape[1] >= 36
            residuals = [float(v) for v in re.findall(
                r"^\[LINEAR_FGMRES\] iterations=\d+ relative_residual=(\S+)", lines, re.M)]
            fgmres_iterations = [int(v) for v in re.findall(
                r"^\[LINEAR_FGMRES\] iterations=(\d+)", lines, re.M)]
            backward = [float(v) for v in re.findall(
                r"^\[LINEAR_FGMRES_BACKWARD\] local_stencil_eta_max=(\S+)", lines, re.M)]
            local_gas = [float(v) for v in re.findall(
                r"^\[NR_LOCAL_GAS\] exact_cells=\d+ residual_over_gas_scale_max=(\S+)", lines, re.M)]
            accepted_temp = [float(v) for v in re.findall(
                r"^\[WOR92_OPACITY_ACCEPTED\] cycle=\d+ Tmin_K=(\S+)", lines, re.M)]
            jacobian = [tuple(map(float, match)) for match in re.findall(
                r"^\[NRFLD Jacobian\].*?epsilon=(\S+).*?gas_l2_rel=(\S+).*?radiation_l2_rel=(\S+)",
                lines, re.M)]
            newton = re.findall(r"^\[NR\] status=(\w+) iterations=(\d+)", lines, re.M)
            opacity = re.findall(r"^FLD_OPACITY_DIAGNOSTICS out_of_domain=(\d+) clamped=(\d+)", lines, re.M)
            for row in history:
                trajectories.append(dict(case=label, time=float(row[0]), dt=float(row[1]),
                                         rho_max_cgs=float(row[15]), jeans_min=float(row[34]),
                                         logical_level_max=float(row[35])))
            for dump in dumps:
                eight, meta = central_eight(dump)
                dump_diagnostics.append(dict(case=label, **meta, hdf5_sha256=sha(dump)))
                for cell in eight:
                    centers.append(dict(case=label, **meta, **cell))
            final_dump = dumps[-1]
            eight, meta = central_eight(final_dump)
            density = [cell["rho_cgs"] for cell in eight]
            cases.append(dict(case=label, **meta, center_rho_ratio=max(density) / min(density),
                              jeans_min_end=float(history[-1, 34]),
                              jeans_target=32, newton_stages=len(newton),
                              newton_nonconverged=sum(status != "converged" for status, _ in newton),
                              fgmres_max_true_residual=max(residuals) if residuals else None,
                              fgmres_max_iterations=max(fgmres_iterations) if fgmres_iterations else None,
                              fgmres_backward_max=max(backward) if backward else None,
                              fgmres_backward_last=backward[-1] if backward else None,
                              local_gas_residual_max=max(local_gas) if local_gas else None,
                              jacobian_gas_l2_eps1e4_max=max((v[1] for v in jacobian
                                  if abs(v[0] / 1e-4 - 1) < 1e-12), default=None),
                              jacobian_gas_l2_eps1e6_max=max((v[1] for v in jacobian
                                  if abs(v[0] / 1e-6 - 1) < 1e-12), default=None),
                              jacobian_rad_l2_max=max((v[2] for v in jacobian), default=None),
                              accepted_Tmin_K=min(accepted_temp) if accepted_temp else None,
                              precision_floor_accepts=sum(int(v) for v in re.findall(
                                  r"\bprecision_floor_accept=(\d+)", lines)),
                              positive_ghost_events=lines.count("[WOR92_NR_GHOST_CLIP]"),
                              opacity_trial_clamps=int(opacity[-1][1]) if opacity else None,
                              log_sha256=sha(log), history_sha256=sha(hst),
                              final_hdf5_sha256=sha(final_dump)))
            archive.add(log, arcname=f"{label}/run.log")
            archive.add(hst, arcname=f"{label}/history.hst")
    write_csv(OUT / "cases.csv", cases)
    write_csv(OUT / "history.csv", trajectories)
    write_csv(OUT / "central_cells.csv", centers)
    write_csv(OUT / "dump_diagnostics.csv", dump_diagnostics)
    grouped = {}
    for row in centers:
        grouped.setdefault((row["case"], row["cycle"]), []).append(row)
    ratio_rows = []
    for (label, cycle), cells in grouped.items():
        rho = [x["rho_cgs"] for x in cells]
        ratio_rows.append(dict(case=label, cycle=cycle, time=cells[0]["time"],
                               center_rho_ratio=max(rho) / min(rho),
                               center_rho_max_cgs=max(rho), center_rho_min_cgs=min(rho)))
    ratio_rows.sort(key=lambda x: (x["case"], x["cycle"]))
    write_csv(OUT / "central_ratios.csv", ratio_rows)
    summary = dict(source_branch="codex/wor92-domain-guard-trial-20261107",
                   source_commit="873a1b4c3bda584b52e1cae2b41e333ddb2f793a", binary_sha256=sha(Path(
                       "/home/kosuke/simulation/test/rad_nr/wor92_operator_audit_20261107/"
                       "build_table_domain_guard_trial/bin/athena")),
                   input_sha256=sha(ROOT / "input_table_domain_guard_cfl030_restart.used"),
                   cycle74_checkpoint_sha256=sha(ROOT / "table_floor32_restart30_to80_2rank/"
                                                    "cfl030_flux_precision16_local.00002.rst"),
                   cycle36_checkpoint_sha256=sha(ROOT / "table_early_restart_state_dump_2rank/"
                                                    "cfl030_flux_precision16_local.final.rst"),
                   cases=cases)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))
    for label in RUNS:
        rows = [x for x in ratio_rows if x["case"] == label]
        if label not in {"cfl030_level3_early", "cfl030_level4_from63", "cfl030_level4_to70"}:
            ax1.plot([x["time"] for x in rows], [x["center_rho_ratio"] for x in rows],
                     marker="o", label=label)
        h = [x for x in trajectories if x["case"] == label]
        if label in {"cfl030_2rank", "cfl015_2rank", "cfl030_level3_late",
                     "cfl030_level3_early", "cfl030_level4_to70"}:
            ax2.plot([x["time"] for x in h], [x["jeans_min"] for x in h], label=label)
    ax1.set_yscale("log")
    ax1.set_xlim(1990, 2110)
    ax1.set_ylim(0.95, 7)
    ax1.set_ylabel("Central octant max/min density")
    ax1.set_xlabel("Code time near asymmetry onset")
    ax1.grid(True)
    ax1.legend(fontsize=7, ncol=2)
    ax2.axhline(32, color="black", ls="--", label="Jeans target 32")
    ax2.axhline(4, color="gray", ls=":", label="4 cells")
    ax2.set_yscale("log")
    ax2.set_xlim(1000, 2110)
    ax2.set_ylim(0.7, 60)
    ax2.set_ylabel("Minimum Jeans cells")
    ax2.set_xlabel("Code time")
    ax2.grid(True)
    ax2.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "core_symmetry.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
