#!/usr/bin/env python3
"""FGMRES AMR regrid, restart, and MPI-rank invariance (WOR-65)."""
from __future__ import annotations

import argparse
from html import escape
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nrfld-validation")
import h5py
import matplotlib.pyplot as plt
import numpy as np

from nrfld_validation.io import find_repo_root, read_athdf
from nrfld_validation.manifest import base_manifest, sha256_file, utc_now, write_manifest
from run_suite import prepare_build_source, binary_configuration, validate_binary_configuration

HERE = Path(__file__).resolve().parent
INPUT = Path("inputs/radiation/athinput.nrfld_fgmres_amr_restart")
TAGS = ("direct", "split", "restart1", "mpi2", "restart2")


def args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/d/output/rad"))
    parser.add_argument("--run-name", default="fgmres_amr_restart_validation")
    parser.add_argument("--reuse-build", type=Path)
    parser.add_argument("--max-cores", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=600)
    parsed = parser.parse_args()
    if parsed.max_cores < 2:
        parser.error("MPI rank-change test requires --max-cores >= 2")
    return parsed


def solver_metrics(log: str):
    nr = [dict(re.findall(r"(\w+)=([^\s]+)", line)) for line in log.splitlines()
          if line.startswith("[NR] status=")]
    fgmres = [dict(re.findall(r"(\w+)=([^\s]+)", line)) for line in log.splitlines()
              if line.startswith("[LINEAR_FGMRES]")]
    physical = [row for row in nr if row.get("status") != "dt_zero_initialization"]
    return {"newton_calls": len(physical), "fgmres_calls": len(fgmres),
            "zero_dt_initialization_calls": sum(
                row.get("status") == "dt_zero_initialization" for row in nr),
            "newton_converged": bool(physical) and all(
                row.get("status") == "converged" and
                max(float(row["final_l2"]), float(row["final_max"])) <= 1.00001e-10
                for row in physical),
            # SolveFGMRES recomputes r=b-Ax before printing this value.
            "max_true_fgmres_relative_residual": max(
                (float(row["relative_residual"]) for row in fgmres), default=None),
            "fgmres_converged": bool(fgmres) and all(
                float(row["relative_residual"]) <= float(row["target"]) * 1.000001
                for row in fgmres),
            "fgmres_restart_exercised": any(int(row["iterations"]) > 5 for row in fgmres),
            "fgmres_iterations": sum(int(row["iterations"]) for row in fgmres)}


def output_sequence(case: Path):
    sequence = []
    for path in sorted((case / "output").glob("*.fields.*.athdf")):
        with h5py.File(path) as hdf:
            levels, counts = np.unique(hdf["Levels"][:], return_counts=True)
            sequence.append({"cycle": int(hdf.attrs["NumCycles"]),
                             "time": float(hdf.attrs["Time"]),
                             "blocks": int(hdf.attrs["NumMeshBlocks"]),
                             "levels": {int(k): int(v) for k, v in zip(levels, counts)},
                             "file": path.name})
    return sequence


def report_html(root: Path, summary: dict):
    rows = []
    for tag in TAGS:
        case = summary["cases"].get(tag, {})
        rows.append(f'<tr><th><a href="{tag}/run.log">{escape(tag)}</a></th>'
                    f'<td>{case.get("ranks", "—")}</td><td>{case.get("cycles", "—")}</td>'
                    f'<td>{case.get("zero_dt_initialization_calls", "—")}</td>'
                    f'<td>{case.get("fgmres_iterations", "—")}</td>'
                    f'<td>{case.get("max_true_fgmres_relative_residual", "—")}</td></tr>')
    comparisons = ''.join(f'<tr><th>{escape(tag)}</th><td>{v["relative_L1"]:.3e}</td>'
                          f'<td>{v["relative_max"]:.3e}</td></tr>'
                          for tag, v in summary.get("comparison", {}).items())
    figures = ''.join(f'<figure><img src="figures/{name}.png" alt="{name}"></figure>'
                      for name in ("regrid", "final_profile", "difference")
                      if (root / "figures" / f"{name}.png").is_file())
    (root / "index.html").write_text(f'''<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AMR FGMRES restart validation</title><style>
:root{{font:21px/1.6 system-ui,sans-serif;color:#17212b}}body{{max-width:1300px;margin:1.5rem auto;padding:0 1rem}}
h1{{font-size:2rem}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.5rem;border-bottom:1px solid #cbd5e1;text-align:left}}
img{{max-width:100%}}a{{color:#075ca8}}@media(max-width:700px){{:root{{font-size:19px}}}}</style></head><body>
<h1>AMR FGMRES: regrid, restart, MPI</h1>
<p><strong>Status: {escape(summary['status'])}</strong>. A controlled pulse starts on a one-level AMR mesh;
the refinement rule removes the fine blocks after cycle 2. Direct, restarted, and two-rank calculations
use the same input and FGMRES restart length 5. The checkpoint is written after cycle 2.</p>
<table><thead><tr><th>Calculation</th><th>MPI ranks</th><th>Final cycle</th>
<th>Zero-dt initialization solves</th><th>FGMRES iterations</th>
<th>Maximum recomputed relative residual</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Final radiation-field comparison to direct</h2><table><thead><tr><th>Calculation</th><th>Relative L1</th>
<th>Relative max</th></tr></thead><tbody>{comparisons}</tbody></table>
<p>Fields use double-precision HDF5 output. Relative L1 is Σ|E−Eref|/Σ|Eref| on the finest common
Cartesian grid; max is max|E−Eref|/max|Eref|. No alignment or renormalization is applied.</p>
{figures}<p><a href="summary.json">Summary JSON</a> · <a href="manifest.json">Manifest</a></p>
</body></html>''')


def main():
    options = args()
    repo = find_repo_root(HERE)
    root = (options.output_root / options.run_name).resolve()
    root.mkdir(parents=True, exist_ok=False)
    manifest = base_manifest(repo, "WOR-65_amr_fgmres_restart", options.max_cores)
    manifest["analysis_script_sha256"] = sha256_file(Path(__file__))
    manifest["input_sha256"] = sha256_file(repo / INPUT)
    manifest["started_at"] = utc_now()
    summary = {"status": "running", "cases": {}, "comparison": {}}
    try:
        if options.reuse_build:
            build = options.reuse_build.resolve()
        else:
            build = root / "build_source"
            prepare_build_source(repo, build)
            with (root / "configure.log").open("w") as log:
                subprocess.run([sys.executable, "configure.py", "--prob=nrfld_diff_amr_interface",
                                "-nrmgfld", "-hdf5", "-h5double", "-mpi", "--cflag=-O3"],
                               cwd=build, stdout=log, stderr=subprocess.STDOUT,
                               timeout=options.timeout, check=True)
            with (root / "build.log").open("w") as log:
                subprocess.run(["make", f"-j{options.max_cores}"], cwd=build,
                               stdout=log, stderr=subprocess.STDOUT,
                               timeout=options.timeout, check=True)
        exe = build / "bin/athena"
        config = binary_configuration(exe, root, 30, [])
        validate_binary_configuration(config, "nrfld_diff_amr_interface", exe)
        if config.get("hdf5_precision") != "double":
            raise ValueError(f"double HDF5 precision required: {config}")
        manifest["build_source"] = str(build)
        manifest["binary_sha256"] = sha256_file(exe)

        def run(tag: str, ranks: int, restart: Path | None = None, nlim: int = 6):
            case = root / tag
            case.mkdir()
            (case / "output").mkdir()
            shutil.copy2(repo / INPUT, case / "input.used")
            command = (["mpirun", "-np", str(ranks)] if ranks > 1 else [])
            command += [str(exe)]
            if restart:
                command += ["-r", str(restart)]
            command += ["-i", "input.used", "-d", "output", f"time/nlim={nlim}",
                        f"job/problem_id={tag}"]
            env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            with (case / "run.log").open("w") as log:
                result = subprocess.run(["/usr/bin/time", "-v", "-o", "resources.txt",
                                         *command], cwd=case, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=options.timeout)
            manifest.setdefault("runs", []).append({"tag": tag, "command": command,
                                                       "returncode": result.returncode})
            if result.returncode:
                raise RuntimeError(f"{tag} failed: see {case/'run.log'}")
            sequence = output_sequence(case)
            if not sequence:
                raise ValueError(f"{tag} produced no HDF5 fields")
            metrics = solver_metrics((case / "run.log").read_text(errors="replace"))
            metrics.update(ranks=ranks, cycles=sequence[-1]["cycle"],
                           time=sequence[-1]["time"], topology=sequence)
            summary["cases"][tag] = metrics
            return case

        direct = run("direct", 1)
        split = run("split", 1, nlim=2)
        restarts = sorted((split / "output").glob("*.rst"))
        if not restarts:
            raise FileNotFoundError("cycle-2 checkpoint missing")
        checkpoint = restarts[-1]
        summary["checkpoint"] = str(checkpoint.relative_to(root))
        run("restart1", 1, restart=checkpoint)
        run("mpi2", 2)
        run("restart2", 2, restart=checkpoint)

        direct_seq = summary["cases"]["direct"]["topology"]
        topology_change = (any(1 in step["levels"] for step in direct_seq[:3])
                           and all(1 not in step["levels"] for step in direct_seq[-2:]))
        reference = read_athdf(sorted((direct / "output").glob("*.fields.*.athdf"))[-1])
        ref_er = np.asarray(reference["Er"], dtype=float)
        x = np.asarray(reference["x1v"], dtype=float)
        for tag in ("restart1", "mpi2", "restart2"):
            case = root / tag
            data = read_athdf(sorted((case / "output").glob("*.fields.*.athdf"))[-1])
            er = np.asarray(data["Er"], dtype=float)
            if er.shape != ref_er.shape or not np.allclose(data["x1v"], x, atol=0, rtol=0):
                raise ValueError(f"{tag} output grid differs")
            delta = np.abs(er-ref_er)
            summary["comparison"][tag] = {
                "relative_L1": float(np.sum(delta)/np.sum(np.abs(ref_er))),
                "relative_max": float(np.max(delta)/np.max(np.abs(ref_er)))}
        summary["topology_changed"] = topology_change
        summary["status"] = "passed" if (
            topology_change and summary["cases"]["direct"]["fgmres_restart_exercised"]
            and all(summary["cases"][tag]["zero_dt_initialization_calls"]
                    == (0 if tag.startswith("restart") else 1) for tag in TAGS)
            and all(summary["cases"][tag]["newton_converged"]
                                    and summary["cases"][tag]["fgmres_converged"]
                                    and summary["cases"][tag]["cycles"] == (2 if tag == "split" else 6)
                                    for tag in TAGS)
            and all(value["relative_max"] < 1e-11
                    for value in summary["comparison"].values())) else "failed"

        (root / "figures").mkdir()
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
        axes[0].step([v["cycle"] for v in direct_seq], [v["blocks"] for v in direct_seq], where="post")
        axes[0].set_ylabel("Active MeshBlocks")
        axes[1].plot(x, ref_er.mean(axis=(0, 1)), label="direct, 1 rank")
        for tag in ("restart1", "mpi2", "restart2"):
            data = read_athdf(sorted((root / tag / "output").glob("*.fields.*.athdf"))[-1])
            axes[1].plot(x, np.asarray(data["Er"]).mean(axis=(0, 1)), "--", label=tag)
        axes[1].set_xlabel("x1")
        axes[1].set_ylabel("Radiation energy")
        for ax in axes:
            ax.grid(True, alpha=.3)
            if ax.get_legend_handles_labels()[0]:
                ax.legend()
        fig.savefig(root / "figures/regrid.png", dpi=160)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        ax.plot(x, ref_er.mean(axis=(0, 1)))
        ax.set(xlabel="x1", ylabel="Radiation energy", title="Final common-grid profile")
        ax.grid(True, alpha=.3)
        fig.savefig(root / "figures/final_profile.png", dpi=160)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        for tag, value in summary["comparison"].items():
            ax.bar(tag, max(value["relative_max"], 1e-18))
        ax.set_yscale("log")
        ax.set_ylabel("Relative maximum difference (zero plotted at 1e-18)")
        fig.savefig(root / "figures/difference.png", dpi=160)
        plt.close(fig)
    except Exception as error:
        summary.update(status="failed", error=str(error))
    manifest.update(status=summary["status"], finished_at=utc_now(), metrics=summary)
    write_manifest(root / "manifest.json", manifest)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report_html(root, summary)
    print(root)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
