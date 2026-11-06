#!/usr/bin/env python3
"""Audit periodic radiation-only energy balance across an AMR interface.

Build Athena with ``--prob=nrfld_diff_amr_interface -nrmgfld`` first. The
existing interface ledger must be enabled for the AMR face-flux comparison.
This script reports the baseline mismatch; ``--require-conservation`` turns
it into a regression gate for a future conservative operator.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess


def fields(line: str) -> dict[str, str]:
    return dict(re.findall(r"(\w+)=([^\s]+)", line))


def run_case(repo: Path, binary: Path, root: Path, layout: str,
             contrast: float, dt: float, linear_solver: str = "mg",
             variable_limiter: bool = False,
             transverse_amplitude: float = 0.0) -> dict:
    case = root / f"{layout}_contrast{contrast:g}"
    case.mkdir()
    source = (repo / "inputs/radiation/athinput.nrfld_diff_amr_interface").read_text()
    source = source.replace("file_type = hst", "file_type = hst\ndata_format = %.16e")
    source = source.replace("diagnostic_verbosity = 1", "diagnostic_verbosity = 0")
    if variable_limiter:
        source = source.replace("fixed_flux_limiter        = true",
                                "fixed_flux_limiter        = false")
    if transverse_amplitude:
        source = source.replace("density_low    = 1.0",
                                f"transverse_amplitude = {transverse_amplitude:.17g}\n"
                                "density_low    = 1.0")
    if linear_solver == "fgmres":
        source = source.replace("fas             = false",
                                "fas             = false\nlinear_solver = fgmres\n"
                                "fgmres_threshold = 1e-10\n"
                                "fgmres_linearity_check = true")
    input_path = case / "input.used"
    input_path.write_text(source)
    command = [str(binary), "-i", str(input_path), "-d", str(case),
               f"time/tlim={dt:.17g}", "time/nlim=1",
               f"problem/density_high={contrast:.17g}"]
    if layout.startswith("uniform"):
        command.append("mesh/refinement=none")
        if layout == "uniform_fine":
            command += ["mesh/nx1=64", "mesh/nx2=16", "mesh/nx3=16"]
    elif layout == "amr_shifted":
        command += ["refinement1/x1min=0.0", "refinement1/x1max=0.5"]
    env = os.environ.copy()
    env["ATHENA_NRFLD_ENERGY_LEDGER"] = "1"
    env["ATHENA_NRFLD_INTERFACE_LEDGER"] = "1"
    proc = subprocess.run(command, cwd=repo, env=env, capture_output=True,
                          text=True, timeout=60, check=False)
    log = proc.stdout + proc.stderr
    (case / "run.log").write_text(log)
    history = list(case.glob("*.hst"))
    if proc.returncode or len(history) != 1:
        raise RuntimeError(f"{case}: Athena exit={proc.returncode}, history={history}")
    rows = [list(map(float, line.split())) for line in history[0].read_text().splitlines()
            if line and not line.startswith("#")]
    if len(rows) < 2 or rows[-1][0] <= rows[0][0]:
        raise RuntimeError(f"{case}: expected initial and final history rows")
    nr = [data for line in log.splitlines() if line.startswith("[NR] status=")
          if float((data := fields(line))["dt"]) > 0.0]
    ledger = [fields(line) for line in log.splitlines()
              if line.startswith("[NR_ENERGY_LEDGER]")]
    if len(nr) != 1 or nr[0].get("status") != "converged" or len(ledger) != 1:
        raise RuntimeError(f"{case}: Newton solve or energy ledger missing")
    er0, er1 = rows[0][-1], rows[1][-1]
    result = {"case": case.name, "layout": layout, "contrast": contrast,
              "Er0": er0, "Er1": er1, "delta_Er": er1-er0,
              "relative_change": (er1-er0)/er0,
              "nr_final_l2": float(nr[0]["final_l2"]),
              "operator_delta": float(ledger[0]["diffusion_delta"]),
              "ledger_balance": float(ledger[0]["balance"])}
    faces = [fields(line) for line in log.splitlines()
             if line.startswith("[NR_INTERFACE_LEDGER]")]
    for x in (.25, .75):
        left = sum(float(face["xplus_delta"]) for face in faces
                   if abs(float(face["x1max"])-x) < 1e-12)
        right = sum(float(face["xminus_delta"]) for face in faces
                    if abs(float(face["x1min"])-x) < 1e-12)
        if layout == "amr_aligned":
            result[f"face_{x:g}"] = {"left": left, "right": right,
                                      "sum": left+right}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dt", type=float, default=2e-4)
    parser.add_argument("--require-conservation", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--linear-solver", choices=("mg", "fgmres"), default="mg")
    parser.add_argument("--variable-limiter", action="store_true")
    parser.add_argument("--transverse-amplitude", type=float, default=0.0)
    parser.add_argument("--contrasts", type=float, nargs="+", default=(1.0, 1.15))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    binary = args.binary.resolve()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    summary = [run_case(repo, binary, root, layout, contrast, args.dt,
                        args.linear_solver, args.variable_limiter,
                        args.transverse_amplitude)
               for contrast in args.contrasts
               for layout in ("uniform_coarse", "uniform_fine",
                              "amr_aligned", "amr_shifted")]
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for row in summary:
        print(f"{row['case']}: dEr/Er={row['relative_change']:.9e} "
              f"Newton L2={row['nr_final_l2']:.3e}")
    if args.require_conservation and any(abs(row["relative_change"]) > args.tolerance
                                         for row in summary):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
