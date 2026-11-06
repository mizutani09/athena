#!/usr/bin/env python3
"""Validate AMR diffusion across a sharp material interface against an eigenmode.

The periodic slab is symmetric about x=0.5. On 0<=x<=a the eigenfunction
is cos(k*x); on a<=x<=0.5 it is B*cos(k*sqrt(r)*(0.5-x)), where r is the
inner/outer extinction ratio.  Reflecting about 0.5 gives the other half.
The root condition enforces continuous Er and D*dEr/dx at both interfaces.
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

import h5py
import numpy as np
from scipy.optimize import brentq


def eigenpair(ratio, slab_left=0.25):
    root_r = math.sqrt(ratio)
    inner_half_width = 0.5-slab_left

    def determinant(wave):
        outer = wave*slab_left
        inner = wave*root_r*inner_half_width
        return (math.sin(outer)*math.cos(inner)
                + math.cos(outer)*math.sin(inner)/root_r)

    # The determinant is positive immediately above zero.  Locate its first
    # sign change without dividing by cosines at eigenfunction nodes.
    left = 1.0e-8
    for right in np.linspace(0.005, 100.0, 20000):
        if determinant(left)*determinant(right) < 0.0:
            wave = brentq(determinant, left, right, xtol=1e-14)
            break
        left = right
    else:
        raise RuntimeError(f"Could not bracket a layered eigenvalue for {ratio}")
    inner = wave*root_r*inner_half_width
    outer = wave*slab_left
    if abs(math.cos(inner)) > 1e-8:
        inner_amplitude = math.cos(outer)/math.cos(inner)
    else:
        inner_amplitude = -root_r*math.sin(outer)/math.sin(inner)
    if abs(math.cos(outer)-inner_amplitude*math.cos(inner)) > 1e-10:
        raise RuntimeError("Eigenfunction is discontinuous")
    if abs(math.sin(outer)+inner_amplitude*math.sin(inner)/root_r) > 1e-10:
        raise RuntimeError("Eigenfunction flux is discontinuous")
    return wave, inner_amplitude


def cell_average(edges, ratio, wave, inner_amplitude, decay, slab_left):
    centers = 0.5*(edges[:-1]+edges[1:])
    widths = np.diff(edges)
    inner = (centers >= slab_left) & (centers < 1.0-slab_left)
    local_wave = np.where(inner, wave*math.sqrt(ratio), wave)
    amplitude = np.where(inner, inner_amplitude, 1.0)
    distance = np.where(inner, centers-0.5,
                        np.where(centers < slab_left, centers, 1.0-centers))
    arg = 0.5*local_wave*widths
    return 1.0 + 0.05*decay*amplitude*np.sinc(arg/math.pi)*np.cos(local_wave*distance)


def inspect_snapshot(path, ratio, wave, inner_amplitude, decay, slab_left):
    with h5py.File(path) as hdf:
        field = hdf["user_out_var"]
        if field.dtype != np.dtype("float64"):
            raise RuntimeError(f"Expected double-precision HDF5 output: {path}")
        names = [name.decode() for name in hdf.attrs["VariableNames"]]
        er_index = names.index("Er")
        error_sum = 0.0
        interface_error_sum = 0.0
        interface_volume = 0.0
        volume = 0.0
        integral = 0.0
        maximum = 0.0
        minimum_er = math.inf
        for block in range(len(hdf["Levels"])):
            xedges = hdf["x1f"][block]
            yedges = hdf["x2f"][block]
            zedges = hdf["x3f"][block]
            exact = cell_average(xedges, ratio, wave, inner_amplitude, decay,
                                 slab_left)
            values = field[er_index, block]
            cell_volume = (np.diff(zedges)[:, None, None]
                           * np.diff(yedges)[None, :, None]
                           * np.diff(xedges)[None, None, :])
            error = np.abs(values-exact[None, None, :])
            xcenters = 0.5*(xedges[:-1]+xedges[1:])
            band = (np.abs(xcenters-0.25) < 0.05) | (np.abs(xcenters-0.75) < 0.05)
            error_sum += float(np.sum(error*cell_volume))
            interface_error_sum += float(np.sum(error[:, :, band]*cell_volume[:, :, band]))
            interface_volume += float(np.sum(cell_volume[:, :, band]))
            volume += float(np.sum(cell_volume))
            integral += float(np.sum(values*cell_volume))
            maximum = max(maximum, float(np.max(error)))
            minimum_er = min(minimum_er, float(np.min(values)))
        return {"time": float(hdf.attrs["Time"]),
                "cycle": int(hdf.attrs["NumCycles"]),
                "blocks": len(hdf["Levels"]),
                "l1": error_sum/volume,
                "linf": maximum,
                "interface_l1": interface_error_sum/interface_volume,
                "volume_integral": integral,
                "minimum_Er": minimum_er}


def run_case(binary, source, root, resolution, opacity, ratio, tlim, cfl, mode,
             ranks, wave, inner_amplitude, slab_left):
    name = f"a{slab_left:g}_r{ratio:g}_n{resolution}_{mode}_p{ranks}"
    directory = root/name
    directory.mkdir()
    (directory/"input.used").write_text(source)
    transverse_size = 8.0/resolution
    command = [str(binary), "-i", str(directory/"input.used"), "-d", str(directory),
               f"mesh/nx1={resolution}", "mesh/nx2=8", "mesh/nx3=8",
               f"mesh/x2max={transverse_size:.17g}",
               f"mesh/x3max={transverse_size:.17g}",
               f"refinement1/x2max={transverse_size:.17g}",
               f"refinement1/x3max={transverse_size:.17g}",
               f"fld/const_opacity={opacity:.17g}",
               f"problem/density_high={ratio:.17g}",
               f"problem/slab_left={slab_left:.17g}",
               f"problem/slab_right={1.0-slab_left:.17g}",
               "problem/layered_eigen_mode=true",
               f"problem/eigen_outer_wave={wave:.17g}",
               f"problem/eigen_inner_amplitude={inner_amplitude:.17g}",
               "problem/erad_amplitude=0.05",
               f"time/tlim={tlim:.17g}", f"time/cfl_number={cfl:.17g}",
               "time/nlim=1000"]
    if ranks > 1:
        command = ["mpirun", "-np", str(ranks)] + command
    env = os.environ.copy()
    for key in ("ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE",
                "ATHENA_NRFLD_LEGACY_INTERFACE",
                "ATHENA_NRFLD_CONSERVATIVE_SHARED_COEFF",
                "ATHENA_NRFLD_CONSERVATIVE_DIFFERENCE_JV",
                "ATHENA_NRFLD_EXPERIMENTAL_TANGENTIAL_RECONSTRUCTION",
                "ATHENA_NRFLD_EXPERIMENTAL_SERIES_RESISTANCE"):
        env.pop(key, None)
    if mode != "baseline":
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
    started = time.perf_counter()
    result = subprocess.run(command, cwd=binary.parent.parent, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, timeout=600, check=False)
    wall = time.perf_counter()-started
    (directory/"run.log").write_text(result.stdout)
    if result.returncode or "Terminating on time limit" not in result.stdout:
        raise RuntimeError(f"Run failed: {directory/'run.log'}")
    statuses = re.findall(r"\[NR\] status=(\w+)", result.stdout)
    active_statuses = [status for status in statuses
                       if status != "dt_zero_initialization"]
    if not active_statuses or any(status != "converged" for status in active_statuses):
        raise RuntimeError(f"Newton did not converge: {directory/'run.log'}")
    paths = sorted(directory.glob("*.fields.*.athdf"))
    if len(paths) < 3:
        raise RuntimeError(f"Expected multi-step HDF5 output: {directory}")
    c_code = 2.99792458e10/math.sqrt(1.0e7/1.0e-7)
    outer_sigma = opacity*4.0e10*1.0e-7
    eigenvalue = c_code*wave*wave/(3.0*outer_sigma)
    records = []
    previous = 0.0
    decay = 1.0
    for path in paths:
        with h5py.File(path) as hdf:
            current = float(hdf.attrs["Time"])
        if current > previous:
            decay /= 1.0+eigenvalue*(current-previous)
        records.append(inspect_snapshot(path, ratio, wave, inner_amplitude,
                                        decay, slab_left))
        previous = current
    final = records[-1]
    ledger = [float(value) for value in re.findall(
        r"\[NR_ENERGY_LEDGER\].*diffusion_delta=([\d.eE+-]+)", result.stdout)]
    return {"mode": mode, "ratio": ratio, "slab_left": slab_left,
            "base_nx": resolution,
            "ranks": ranks, "wave": wave, "inner_amplitude": inner_amplitude,
            "eigenvalue": eigenvalue, "wall_seconds": wall,
            "steps": len(records)-1, "newton_solves": len(active_statuses),
            "final": final, "initial": records[0],
            "max_l1": max(item["l1"] for item in records[1:]),
            "integral_drift": final["volume_integral"]-records[0]["volume_integral"],
            "diffusion_ledger_sum": sum(ledger),
            "snapshots": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("bin/athena"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.1, 10.0])
    parser.add_argument("--slab-left", type=float, default=0.25)
    parser.add_argument("--opacity", type=float, default=4.0)
    parser.add_argument("--tlim", type=float, default=0.003)
    parser.add_argument("--cfl", type=float, default=1.0)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--modes", nargs="+",
                        choices=("baseline", "shared", "series"),
                        default=("baseline", "shared", "series"))
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    binary = args.binary.resolve()
    source = (Path(__file__).resolve().parents[2]
              / "inputs/radiation/athinput.nrfld_diff_amr_interface").read_text()
    source = source.replace("file_type = hst", "file_type = hst\ndata_format = %.16e")
    source = source.replace("<time>", "<output2>\nfile_type = hdf5\nvariable = uov\n"
                            "id = fields\ndcycle = 1\n\n<time>")
    source = source.replace("fas             = false",
                            "fas             = false\nlinear_solver = fgmres\n"
                            "fgmres_threshold = 1e-10")
    source = source.replace("diagnostic_verbosity = 1", "diagnostic_verbosity = 0")
    records = []
    if not (0.0 < args.slab_left < 0.5):
        parser.error("--slab-left must be between 0 and 0.5")
    for ratio in args.ratios:
        if ratio <= 0.0:
            parser.error("All ratios must be positive")
        wave, inner_amplitude = eigenpair(ratio, args.slab_left)
        for resolution in args.resolutions:
            if resolution % 8:
                parser.error("Base x1 resolutions must be divisible by 8")
            if abs(args.slab_left*resolution-round(args.slab_left*resolution)) > 1e-12:
                parser.error("Material interface must align with root cell faces")
            for mode in args.modes:
                record = run_case(binary, source, root, resolution, args.opacity,
                                  ratio, args.tlim, args.cfl, mode, args.ranks,
                                  wave, inner_amplitude, args.slab_left)
                if record["final"]["minimum_Er"] <= 0.0:
                    raise RuntimeError(f"Nonpositive Er: {record}")
                if mode != "baseline" and abs(record["diffusion_ledger_sum"]) > 1e-11:
                    raise RuntimeError(f"Nonconservative diffusion: {record}")
                records.append(record)
                (root/"summary.json").write_text(json.dumps(records, indent=2)+"\n")
                print(json.dumps({key: record[key] for key in
                                  ("mode", "ratio", "base_nx", "steps", "final",
                                   "integral_drift", "diffusion_ledger_sum")}),
                      flush=True)


if __name__ == "__main__":
    main()
