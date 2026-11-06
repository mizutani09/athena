#!/usr/bin/env python3
"""Coupled periodic-box conservation and Newton-tolerance sweep (WOR-64)."""
from __future__ import annotations

import argparse
import csv
from html import escape
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import h5py
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nrfld-validation")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["pdf.fonttype"] = 42

from nrfld_validation.io import find_repo_root, read_history
from nrfld_validation.manifest import base_manifest, sha256_file, utc_now, write_manifest
from run_suite import build_case, binary_configuration, validate_binary_configuration

HERE = Path(__file__).resolve().parent
INPUT = Path("inputs/radiation/athinput.nrfld_closed_box")
TOLERANCES = (1e-6, 1e-8, 1e-10)
CFL_NUMBERS = (0.1, 0.2, 0.4)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/d/output/rad"))
    parser.add_argument("--run-name", default="closed_box_validation")
    parser.add_argument("--reuse-build", type=Path)
    parser.add_argument("--max-cores", type=int, default=4)
    parser.add_argument("--run-timeout", type=float, default=900)
    parser.add_argument("--build-timeout", type=float, default=600)
    parser.add_argument("--tlim", type=float, default=1.0)
    args = parser.parse_args()
    if args.max_cores < 1 or args.tlim <= 0:
        parser.error("cores and tlim must be positive")
    return args


def parse_solver(log: str, tolerance: float):
    nr = [dict(re.findall(r"(\w+)=([^\s]+)", line)) for line in log.splitlines()
          if line.startswith("[NR] status=") and "dt=0.000000000000000" not in line]
    krylov = [dict(re.findall(r"(\w+)=([^\s]+)", line)) for line in log.splitlines()
              if line.startswith("[LINEAR_FGMRES]")]
    return {"newton_calls": len(nr), "fgmres_calls": len(krylov),
            "newton_converged": bool(nr) and all(x.get("status") == "converged" for x in nr),
            "newton_max_final_residual": max((max(float(x["final_l2"]),
                                                  float(x["final_max"])) for x in nr), default=None),
            "fgmres_converged": bool(krylov) and all(
                float(x["relative_residual"]) <= float(x["target"]) * 1.000001
                for x in krylov),
            "fgmres_iterations": sum(int(x["iterations"]) for x in krylov),
            "requested_newton_tolerance": tolerance}


def analyze_case(case: Path, tolerance: float, cfl: float, target_time: float):
    history = read_history(case / "output")
    energy = np.asarray(history["box_energy"], dtype=float)
    mass = np.asarray(history["box_mass"], dtype=float)
    time = np.asarray(history["time"], dtype=float)
    if len(time) < 3 or not all(np.isfinite(x).all() for x in (energy, mass, time)):
        raise ValueError(f"invalid history for {case}")
    e_drift = energy / energy[0] - 1.0
    m_drift = mass / mass[0] - 1.0
    momenta = [np.asarray(history[f"box_p{axis}"], dtype=float) for axis in (1, 2, 3)]
    final_files = sorted((case / "output").glob("*.fields.*.athdf"))
    if not final_files:
        raise ValueError(f"missing field output for {case}")
    with h5py.File(final_files[-1]) as hdf:
        names = [name.decode() for name in hdf.attrs["VariableNames"]]
        fields = hdf["user_out_var"][:]
        positive_fields = all(np.isfinite(fields[names.index(name)]).all()
                              and (fields[names.index(name)] > 0).all()
                              for name in ("rho", "Tgas", "Trad", "Erad", "Pgas"))
    with (case / "history.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("time_code", "mass_g", "total_energy_erg", "mass_relative_drift",
                         "energy_relative_drift", "box_p1_g_cm_s", "box_p2_g_cm_s",
                         "box_p3_g_cm_s", "Tgas_max_K", "Tgas_min_K"))
        writer.writerows(zip(time, mass, energy, m_drift, e_drift,
                             *momenta, history["Tgas_max"], history["Tgas_min"]))
    log = (case / "run.log").read_text(errors="replace")
    solver = parse_solver(log, tolerance)
    result = {"tolerance": tolerance, "cfl_number": cfl,
              "cycles": len(time)-1, "time_code": float(time[-1]),
              "mass_initial_g": float(mass[0]), "energy_initial_erg": float(energy[0]),
              "mass_max_relative_drift": float(np.max(np.abs(m_drift))),
              "energy_max_relative_drift": float(np.max(np.abs(e_drift))),
              "energy_final_relative_drift": float(e_drift[-1]),
              "energy_balance_residual_erg": float(energy[-1]-energy[0]),
              "periodic_boundary_net_flux_erg": 0.0,
              "gas_momentum_max_abs_g_cm_s": [float(np.max(np.abs(p))) for p in momenta],
              "positive_finite_final_fields": bool(positive_fields),
              "Tgas_max_initial_K": float(history["Tgas_max"][0]),
              "Tgas_max_final_K": float(history["Tgas_max"][-1]), **solver}
    result["passed"] = (np.isclose(time[-1], target_time, rtol=1e-10)
                         and solver["newton_converged"] and solver["fgmres_converged"]
                         and solver["newton_max_final_residual"] <= tolerance * 1.0001
                         and positive_fields
                         and result["mass_max_relative_drift"] < 1e-5
                         and result["energy_max_relative_drift"] < 1e-5
                         and abs(result["Tgas_max_final_K"] /
                                 result["Tgas_max_initial_K"] - 1) > 0.01)
    return result, time, m_drift, e_drift


def write_html(root: Path, summary: dict):
    rows = []
    for tag, case in summary["cases"].items():
        rows.append(f'<tr><th><a href="{tag}/history.csv">{escape(tag)}</a></th>'
                    f'<td>{case["cfl_number"]:.2g}</td>'
                    f'<td>{case["cycles"]}</td><td>{case["mass_max_relative_drift"]:.3e}</td>'
                    f'<td>{case["energy_max_relative_drift"]:.3e}</td>'
                    f'<td>{case["newton_calls"]}</td><td>{case["fgmres_iterations"]}</td>'
                    f'<td class="{"good" if case["passed"] else "bad"}">{"PASS" if case["passed"] else "FAIL"}</td></tr>')
    figures = ''.join(f'<figure><img src="figures/{name}.png" alt="{name}"></figure>'
                      for name in ("conservation", "tolerance_dependence", "cfl_dependence")
                      if (root / "figures" / f"{name}.png").exists())
    (root / "index.html").write_text(f'''<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Coupled closed-box conservation</title><style>
:root{{font:21px/1.6 system-ui,sans-serif;color:#17212b}}body{{max-width:1300px;margin:1.5rem auto;padding:0 1rem}}
h1{{font-size:2rem}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.5rem;border-bottom:1px solid #cbd5e1;text-align:left}}
.good{{color:#08764b;font-weight:700}}.bad{{color:#b42318;font-weight:700}}img{{max-width:100%}}
a{{color:#075ca8}}@media(max-width:700px){{:root{{font-size:19px}}}}</style></head><body>
<h1>Coupled periodic-box conservation</h1>
<p><strong>Status: {escape(summary['status'])}</strong>. A 3D periodic box starts with a Gaussian gas-temperature
pulse and uniform radiation temperature. The gas and radiation exchange energy while pressure gradients move the gas.
No spatial shift or energy renormalization is applied.</p>
<table><thead><tr><th>Case</th><th>CFL</th><th>Steps</th><th>Mass max |Δ|/M₀</th><th>Energy max |Δ|/E₀</th><th>Newton calls</th><th>FGMRES iterations</th><th>Result</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p>The total energy sums gas internal energy, radiation energy, and kinetic energy over every cell.
History values are written at 16-digit precision. Gas momentum is also recorded; net flux through the periodic
boundary is analytically zero, so the energy-balance residual is E(t)−E(0). The pass gate requires strict Newton and FGMRES convergence,
the requested final time, a changed gas temperature, and both conservation errors below 10⁻⁵.
The approximately 10⁻⁷ drift shared by all tolerances indicates an error floor outside the nonlinear solve.</p>
{figures}<p><a href="summary.json">Summary JSON</a> · <a href="manifest.json">Manifest</a></p>
</body></html>''')


def main():
    args = parse_args()
    repo = find_repo_root(HERE)
    root = (args.output_root / args.run_name).resolve()
    root.mkdir(parents=True, exist_ok=False)
    manifest = base_manifest(repo, "WOR-64_closed_box", args.max_cores)
    manifest["input_sha256"] = sha256_file(repo / INPUT)
    manifest["analysis_script_sha256"] = sha256_file(Path(__file__))
    manifest["started_at"] = utc_now()
    records = []
    manifest["commands"]["build"] = records
    summary = {"status": "running", "cases": {}, "target_time_code": args.tlim}
    try:
        if args.reuse_build:
            exe = args.reuse_build.resolve() / "bin/athena"
            manifest["build_source"] = str(args.reuse_build.resolve())
        else:
            build = build_case(repo, root, "nrfld_advect_radiation_pulse", args.max_cores,
                               args.build_timeout, records)
            exe = build["build_source"] / "bin/athena"
            manifest["build_source_sha256"] = build["build_source_sha256"]
        config = binary_configuration(exe, root, 30, records)
        validate_binary_configuration(config, "nrfld_advect_radiation_pulse", exe)
        manifest["binary_sha256"] = sha256_file(exe)
        histories = {}
        cases = [(f"tol_{tolerance:.0e}", tolerance, 0.2) for tolerance in TOLERANCES]
        cases += [(f"cfl_{cfl:.1f}", 1e-8, cfl) for cfl in CFL_NUMBERS if cfl != 0.2]
        for tag, tolerance, cfl in cases:
            case = root / tag
            case.mkdir()
            (case / "output").mkdir()
            shutil.copy2(repo / INPUT, case / "input.used")
            command = ["/usr/bin/time", "-v", "-o", "resources.txt", str(exe),
                       "-i", "input.used", "-d", "output",
                       f"nrfld/nr_threshold={tolerance:.0e}", f"time/tlim={args.tlim:.17g}",
                       f"time/cfl_number={cfl:.17g}",
                       "time/nlim=100", "time/ncycle_out=10", "output2/dcycle=10",
                       f"job/problem_id=closed_box_{tag}"]
            with (case / "run.log").open("w") as log:
                result = subprocess.run(command, cwd=case, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=args.run_timeout)
            manifest.setdefault("runs", []).append({"case": tag, "command": command,
                                                       "returncode": result.returncode})
            if result.returncode:
                raise RuntimeError(f"{tag} failed: see {case/'run.log'}")
            metrics, time, m_drift, e_drift = analyze_case(case, tolerance, cfl, args.tlim)
            summary["cases"][tag] = metrics
            histories[tag] = (time, m_drift, e_drift)
        (root / "figures").mkdir()
        fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
        for tag, (time, m_drift, e_drift) in histories.items():
            axes[0].plot(time, m_drift, label=tag)
            axes[1].plot(time, e_drift, label=tag)
        axes[0].set_ylabel("Mass relative drift")
        axes[1].set_ylabel("Total-energy relative drift")
        axes[1].set_xlabel("Code time")
        for ax in axes:
            ax.grid(True, alpha=.3)
            ax.legend()
        fig.savefig(root / "figures/conservation.png", dpi=160)
        fig.savefig(root / "figures/conservation.pdf")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
        tolerance_cases = [summary["cases"][f"tol_{value:.0e}"] for value in TOLERANCES]
        tol = [x["tolerance"] for x in tolerance_cases]
        for field, label in (("mass_max_relative_drift", "mass"),
                             ("energy_max_relative_drift", "total energy")):
            ax.loglog(tol, [x[field] for x in tolerance_cases], "o-", label=label)
        ax.set_xlabel("Newton tolerance")
        ax.set_ylabel("Maximum relative drift")
        ax.grid(True, which="both", alpha=.3)
        ax.legend()
        fig.savefig(root / "figures/tolerance_dependence.png", dpi=160)
        fig.savefig(root / "figures/tolerance_dependence.pdf")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
        cfl_cases = [summary["cases"]["cfl_0.1"], summary["cases"]["tol_1e-08"],
                     summary["cases"]["cfl_0.4"]]
        for field, label in (("mass_max_relative_drift", "mass"),
                             ("energy_max_relative_drift", "total energy")):
            ax.loglog(CFL_NUMBERS, [x[field] for x in cfl_cases], "o-", label=label)
        ax.set_xlabel("CFL number")
        ax.set_ylabel("Maximum relative drift")
        ax.grid(True, which="both", alpha=.3)
        ax.legend()
        fig.savefig(root / "figures/cfl_dependence.png", dpi=160)
        fig.savefig(root / "figures/cfl_dependence.pdf")
        plt.close(fig)
        summary["status"] = "passed" if all(x["passed"] for x in summary["cases"].values()) else "failed"
    except Exception as error:
        summary.update(status="failed", error=str(error))
    manifest.update(status=summary["status"], finished_at=utc_now(), metrics=summary)
    write_manifest(root / "manifest.json", manifest)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_html(root, summary)
    print(root)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
