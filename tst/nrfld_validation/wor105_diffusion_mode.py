#!/usr/bin/env python3
"""Check the optically thick AMR FLD face sync against a 1D diffusion mode.

The initial Er values are exact cell averages of one periodic cosine mode.
The reference is its continuum backward-Euler update with D=c/(3 kappa rho).
This isolates the spatial AMR error from the intended time discretization.
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess

import numpy as np

from wor102_profile_comparison import compose


def run_case(binary, source, root, n, opacity, end_time, cfl, ranks, mode):
    case = root / f"n{n}_{mode}_r{ranks}"
    case.mkdir(parents=True)
    input_path = case / "input.used"
    input_path.write_text(source)
    command = [str(binary), "-i", str(input_path), "-d", str(case),
               f"mesh/nx1={n}", f"mesh/nx2={n//4}", f"mesh/nx3={n//4}",
               "problem/density_high=1", "problem/exact_cosine_mode=true",
               "problem/transverse_amplitude=0", f"fld/const_opacity={opacity}",
               f"time/tlim={end_time}", f"time/cfl_number={cfl}",
               "time/nlim=1"]
    env = os.environ.copy()
    for name in ("ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE",
                 "ATHENA_NRFLD_LEGACY_INTERFACE",
                 "ATHENA_NRFLD_CONSERVATIVE_SHARED_COEFF",
                 "ATHENA_NRFLD_CONSERVATIVE_DIFFERENCE_JV",
                 "ATHENA_NRFLD_EXPERIMENTAL_TANGENTIAL_RECONSTRUCTION",
                 "ATHENA_NRFLD_EXPERIMENTAL_SERIES_RESISTANCE"):
        env.pop(name, None)
    if mode in ("conservative", "series"):
        env.update(ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE="1",
                   ATHENA_NRFLD_LEGACY_INTERFACE="0",
                   ATHENA_NRFLD_CONSERVATIVE_SHARED_COEFF="1",
                   ATHENA_NRFLD_CONSERVATIVE_DIFFERENCE_JV="0",
                   ATHENA_NRFLD_EXPERIMENTAL_TANGENTIAL_RECONSTRUCTION="1")
    else:
        env["ATHENA_NRFLD_LEGACY_INTERFACE"] = "1"
    env["ATHENA_NRFLD_EXPERIMENTAL_SERIES_RESISTANCE"] = (
        "1" if mode == "series" else "0")
    env["ATHENA_NRFLD_ENERGY_LEDGER"] = "1"
    env["ATHENA_NRFLD_INTERFACE_LEDGER"] = "1"
    if ranks > 1:
        command = ["mpirun", "-np", str(ranks)] + command
    result = subprocess.run(command, cwd=binary.parent.parent, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, timeout=240, check=False)
    (case / "run.log").write_text(result.stdout)
    if result.returncode or "Terminating on cycle limit" not in result.stdout:
        raise RuntimeError(f"{mode} N={n} failed: {case / 'run.log'}")
    statuses = re.findall(r"\[NR\] status=(\w+)", result.stdout)
    if not statuses or statuses[-1] != "converged":
        raise RuntimeError(f"Newton did not converge: {case / 'run.log'}")
    time, initial, _, _ = compose(case, (2*n, n//2, n//2),
                                   (1.0, 0.25, 0.25), cycle=0)
    elapsed, field, _, blocks = compose(case, (2*n, n//2, n//2),
                                         (1.0, 0.25, 0.25), cycle=1)
    if abs(time) > 1e-14:
        raise RuntimeError("Initial output time is not zero")
    # FLD unit conversion follows FLD::FLD and the Test C input file.
    c_code = 2.99792458e10 / math.sqrt(1.0e7 / 1.0e-7)
    sigma_code = opacity * 4.0e10 * 1.0e-7
    diffusivity = c_code / (3.0 * sigma_code)
    wave = 2.0 * math.pi
    fine_width = 1.0 / (2*n)
    centers = (np.arange(2*n) + 0.5) * fine_width
    sinc = math.sin(wave*fine_width/2) / (wave*fine_width/2)
    amplitude = 0.1 / (1.0 + diffusivity*wave*wave*elapsed)
    exact_fine = np.broadcast_to(
        (1.0 + amplitude*sinc*np.cos(wave*centers))[None, None, :],
        field.shape).copy()
    _, _, projected, _ = compose(case, (2*n, n//2, n//2),
                                  (1.0, 0.25, 0.25), cycle=1,
                                  reference=exact_fine)
    error = np.abs(field.astype(np.float64)-projected.astype(np.float64))
    ledger = re.findall(r"\[NR_ENERGY_LEDGER\].*diffusion_delta=([\d.eE+-]+)",
                        result.stdout)
    face_lines = re.findall(r"^\[NR_INTERFACE_LEDGER\].*$", result.stdout,
                            re.MULTILINE)
    interface_mismatch = {}
    for xface in (0.25, 0.75):
        terms = []
        for line in face_lines:
            values = dict(re.findall(r"(\w+)=([\d.eE+-]+)", line))
            if abs(float(values["x1min"])-xface) < 1e-12:
                terms.append(float(values["xminus_delta"]))
            if abs(float(values["x1max"])-xface) < 1e-12:
                terms.append(float(values["xplus_delta"]))
        scale = sum(map(abs, terms))
        interface_mismatch[str(xface)] = (abs(sum(terms))/scale
                                           if scale else 0.0)
    return {"mode": mode, "base_nx": n, "opacity": opacity,
            "ranks": ranks,
            "time": elapsed, "blocks": blocks, "diffusivity": diffusivity,
            "minimum_Er": float(field.min()), "l1": float(error.mean()),
            "linf": float(error.max()),
            "global_diffusion_ledger": sum(map(float, ledger)) if ledger else None,
            "interface_relative_mismatch": interface_mismatch,
            "initial_range": [float(initial.min()), float(initial.max())]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("bin/athena"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--opacity", type=float, default=4.0)
    parser.add_argument("--end-time", type=float, default=0.001)
    parser.add_argument("--cfl", type=float, default=0.3)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--series-resistance", action="store_true")
    args = parser.parse_args()
    binary = args.binary.resolve()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = (Path(__file__).resolve().parents[2]
              / "inputs/radiation/athinput.nrfld_diff_amr_interface").read_text()
    source = source.replace("file_type = hst",
                            "file_type = hst\ndata_format = %.16e")
    source = source.replace("<time>",
                            "<output2>\nfile_type = vtk\nvariable = uov\n"
                            "id = fields\ndcycle = 1\n\n<time>")
    modes = ("baseline", "conservative", "series") if args.series_resistance else (
        "baseline", "conservative")
    records = [run_case(binary, source, root, n, args.opacity,
                        args.end_time, args.cfl, args.ranks, mode)
               for n in args.resolutions for mode in modes]
    for record in records:
        if record["mode"] == "baseline":
            continue
        if record["minimum_Er"] <= 0.0:
            raise RuntimeError(f"Nonpositive radiation energy: {record}")
        if abs(record["global_diffusion_ledger"]) > 1e-12:
            raise RuntimeError(f"Nonconservative periodic diffusion: {record}")
        if max(record["interface_relative_mismatch"].values()) > 1e-10:
            raise RuntimeError(f"Coarse/fine face mismatch: {record}")
    (root / "summary.json").write_text(json.dumps(records, indent=2) + "\n")
    for record in records:
        print(json.dumps(record))


if __name__ == "__main__":
    main()
