#!/usr/bin/env python3
"""Compare continuous and checkpointed coupled FLD on static and moving AMR.

Build first with: python3 configure.py --prob=nrfld_diff_amr_interface -nrmgfld -mpi -hdf5
The test intentionally compares active fields at the same cycle and time.
"""

import argparse
import os
from pathlib import Path
import re
import subprocess

import h5py
import numpy as np


def run(command, env, log):
    with log.open("w") as out:
        proc = subprocess.run(command, env=env, stdout=out,
                              stderr=subprocess.STDOUT, check=False)
    result = log.read_text()
    if proc.returncode != 0 or "FATAL" in result or "Terminating on cycle limit" not in result:
        raise RuntimeError(f"run failed: {log}")
    for line in result.splitlines():
        if line.startswith("[NR] ") and "dt=0.000000" not in line:
            if "status=converged" not in line:
                raise RuntimeError(f"Newton failed: {log}: {line}")


def make_input(source, output, adaptive):
    content = source.read_text()
    content = content.replace("<time>", """<output2>
file_type = hdf5
variable = uov
id = fields
dcycle = 1

<output3>
file_type = rst
dcycle = 2

<time>""")
    content = content.replace("refinement = static", "refinement = adaptive" if adaptive
                              else "refinement = static")
    content = content.replace("<meshblock>", "numlevel = 2\nderefine_count = 1\n\n<meshblock>")
    content = content.replace("<nrfld>", "<nrfld>\nlinear_solver = fgmres")
    content = content.replace("<problem>", "<problem>\ntest_regrid_cycle = 2")
    if adaptive:
        content = re.sub(r"<refinement1>.*?(?=<hydro>)", "", content, flags=re.S)
    output.write_text(content)


def compare(direct, restarted, tolerance):
    files = [sorted(folder.glob("*.fields.*.athdf"))[-1]
             for folder in (direct, restarted)]
    with h5py.File(files[0]) as a, h5py.File(files[1]) as b:
        time_a = float(a.attrs["Time"])
        time_b = float(b.attrs["Time"])
        if time_a != time_b:
            raise AssertionError(f"restart timestep differs: {time_a} vs {time_b}")
        if not np.array_equal(a["LogicalLocations"][:], b["LogicalLocations"][:]):
            raise AssertionError("restart AMR topology differs")
        x, y = a["user_out_var"][:], b["user_out_var"][:]
        max_diff = float(np.max(np.abs(x-y)))
        if max_diff > tolerance:
            raise AssertionError(f"field mismatch {max_diff} exceeds {tolerance}")
    return time_a, max_diff


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("bin/athena"))
    parser.add_argument("--input", type=Path,
                        default=Path("inputs/radiation/athinput.nrfld_diff_amr_interface"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/wor106_restart_regression"))
    parser.add_argument("--tolerance", type=float, default=1e-10)
    args = parser.parse_args()
    binary, source = args.binary.resolve(), args.input.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    baseline_flags = dict(os.environ)
    baseline_flags["ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE"] = "0"
    candidate_flags = dict(baseline_flags)
    candidate_flags.update(ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE="1",
                 ATHENA_NRFLD_CONSERVATIVE_SHARED_COEFF="1",
                 ATHENA_NRFLD_CONSERVATIVE_DIFFERENCE_JV="0",
                 ATHENA_NRFLD_EXPERIMENTAL_TANGENTIAL_RECONSTRUCTION="1",
                 ATHENA_NRFLD_EXPERIMENTAL_VARIABLE_LIMITER="1",
                 ATHENA_NRFLD_EXPERIMENTAL_REGRID="1",
                 ATHENA_NRFLD_EXPERIMENTAL_COUPLED="1")
    cases = (("baseline", False, 4), ("baseline", True, 4),
             ("candidate", False, 4), ("candidate", True, 4),
             ("candidate", True, 1))
    for mode, adaptive, direct_ranks in cases:
        flags = candidate_flags if mode == "candidate" else baseline_flags
        name = f"{mode}_{'adaptive' if adaptive else 'static'}_{direct_ranks}to4"
        folder = args.output / name
        folder.mkdir(exist_ok=True)
        input_file = folder / "input.used"
        make_input(source, input_file, adaptive)
        direct, restarted = folder / "direct", folder / "restart"
        options = ["time/nlim=5",
                   "fld/is_couple=true", "fld/only_rad=false",
                   "fld/fixed_flux_limiter=false", "fld/const_opacity=0.004",
                   "problem/density_high=1.15", "problem/transverse_amplitude=0.3"]
        if adaptive:
            options.append("problem/test_regrid_cycle=2")
        command = (["mpirun", "-n", str(direct_ranks)] if direct_ranks > 1 else [])
        command += [str(binary), "-i", str(input_file), "-d", str(direct)] + options
        run(command, flags, folder / "direct.log")
        checkpoint = direct / "nrfld_diff_amr_interface.00001.rst"
        if not checkpoint.exists():
            raise RuntimeError(f"cycle-2 checkpoint missing: {checkpoint}")
        command = ["mpirun", "-n", "4", str(binary), "-r", str(checkpoint),
                   "-d", str(restarted), "time/nlim=5"]
        run(command, flags, folder / "restart.log")
        time, difference = compare(direct, restarted, args.tolerance)
        print(f"{name}: time={time:.16g} max_field_diff={difference:.3e}")


if __name__ == "__main__":
    main()
