#!/usr/bin/env python3
"""Compare Test C AMR radiation fields with a uniform-fine local reference.

Run one common physical timestep, then replicate each coarse leaf cell onto
the reference grid before measuring volume-weighted errors. VTK stores Er as
float32, so differences below about 1e-7 should not be interpreted.
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np


def read_block(path):
    with path.open("rb") as stream:
        stream.readline()  # VTK version
        time_line = stream.readline().decode("ascii")
        match = re.search(r"time=([\d.eE+-]+)", time_line)
        if not match or stream.readline().strip() != b"BINARY":
            raise ValueError(f"Unexpected VTK header: {path}")
        if stream.readline().strip() != b"DATASET RECTILINEAR_GRID":
            raise ValueError(f"Expected rectilinear grid: {path}")
        dims = tuple(map(int, stream.readline().split()[1:]))
        edges = []
        for name, count in zip((b"X", b"Y", b"Z"), dims):
            header = stream.readline().split()
            if header[:1] != [name + b"_COORDINATES"] or int(header[1]) != count:
                raise ValueError(f"Unexpected coordinate header: {path}")
            edges.append(np.frombuffer(stream.read(4 * count), dtype=">f4").copy())
            stream.readline()
        if stream.readline().split() != [b"CELL_DATA", str(np.prod(np.array(dims)-1)).encode()]:
            raise ValueError(f"Unexpected cell count: {path}")
        if stream.readline().split()[:2] != [b"SCALARS", b"Er"]:
            raise ValueError(f"Er must be the first VTK scalar: {path}")
        if stream.readline().strip() != b"LOOKUP_TABLE default":
            raise ValueError(f"Unexpected lookup table: {path}")
        shape = tuple(n-1 for n in dims[::-1])
        values = np.frombuffer(stream.read(4 * int(np.prod(shape))), dtype=">f4")
        return float(match.group(1)), edges, values.reshape(shape).copy()


def compose(directory, shape, extent, cycle=1, reference=None):
    grid = np.full(shape[::-1], np.nan, dtype=np.float32)
    covered = np.zeros_like(grid, dtype=np.uint8)
    projected = np.full_like(grid, np.nan) if reference is not None else None
    times = set()
    files = sorted(directory.glob(f"*.fields.{cycle:05d}.vtk"))
    if not files:
        raise FileNotFoundError(f"No cycle-{cycle} VTK blocks in {directory}")
    fine_dx = np.array(extent) / np.array(shape)
    for path in files:
        t, edges, values = read_block(path)
        times.add(t)
        starts = np.rint(np.array([e[0] for e in edges]) / fine_dx).astype(int)
        widths = np.array([(e[-1]-e[0])/(len(e)-1) for e in edges])
        factors = np.rint(widths/fine_dx).astype(int)
        if np.max(np.abs(widths/fine_dx-factors)) > 1e-5 or np.any(factors < 1):
            raise ValueError(f"Block does not align with reference grid: {path}")
        expanded = values
        for axis, factor in enumerate(factors[::-1]):
            expanded = np.repeat(expanded, factor, axis=axis)
        x, y, z = starts
        nz, ny, nx = expanded.shape
        region = np.s_[z:z+nz, y:y+ny, x:x+nx]
        if np.any(covered[region]):
            raise ValueError(f"Overlapping leaf blocks: {path}")
        grid[region] = expanded
        if reference is not None:
            fz, fy, fx = factors[::-1]
            coarse = reference[region].reshape(values.shape[0], fz,
                                               values.shape[1], fy,
                                               values.shape[2], fx).mean(axis=(1, 3, 5))
            for axis, factor in enumerate((fz, fy, fx)):
                coarse = np.repeat(coarse, factor, axis=axis)
            projected[region] = coarse
        covered[region] = 1
    if np.any(covered != 1) or len(times) != 1:
        raise ValueError(f"Incomplete grid or inconsistent output times: {directory}")
    return times.pop(), grid, projected, len(files)


def run(binary, source, out, n, mode, dt, contrast, amplitude):
    out.mkdir()
    source = source.replace("file_type = hst", "file_type = hst\ndata_format = %.16e")
    source = source.replace("<time>", "<output2>\nfile_type = vtk\nvariable = uov\nid = fields\ndcycle = 1\n\n<time>")
    source = source.replace("fixed_flux_limiter        = true", "fixed_flux_limiter        = false")
    source = source.replace("density_low    = 1.0",
                            "transverse_amplitude = 0.0\ndensity_low    = 1.0")
    source = source.replace("fas             = false", "fas             = false\nlinear_solver = fgmres\nfgmres_threshold = 1e-10")
    source = source.replace("diagnostic_verbosity = 1", "diagnostic_verbosity = 0")
    input_path = out / "input.used"
    input_path.write_text(source)
    cells = 2*n if mode == "reference" else n
    command = [str(binary), "-i", str(input_path), "-d", str(out),
               f"mesh/nx1={cells}", f"mesh/nx2={cells//4}",
               f"mesh/nx3={cells//4}",
               f"time/tlim={dt:.17g}", "time/nlim=1",
               f"problem/density_high={contrast:.17g}",
               f"problem/transverse_amplitude={amplitude:.17g}"]
    if mode == "reference":
        command += ["mesh/refinement=none"]
    env = os.environ.copy()
    env.pop("ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE", None)
    env.pop("ATHENA_NRFLD_CONSERVATIVE_SHARED_COEFF", None)
    env.pop("ATHENA_NRFLD_CONSERVATIVE_DIFFERENCE_JV", None)
    if mode != "reference":
        env["ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE"] = "0" if mode == "baseline" else "1"
        env["ATHENA_NRFLD_CONSERVATIVE_SHARED_COEFF"] = "1" if mode == "shared" else "0"
    env["ATHENA_NRFLD_EXPERIMENTAL_VARIABLE_LIMITER"] = "1"
    env["ATHENA_NRFLD_ENERGY_LEDGER"] = "1"
    start = time.perf_counter()
    result = subprocess.run(command, cwd=binary.parent.parent, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, timeout=240, check=False)
    elapsed = time.perf_counter()-start
    (out / "run.log").write_text(result.stdout)
    if result.returncode or "Terminating on cycle limit" not in result.stdout:
        raise RuntimeError(f"{mode} N={n} failed, see {out / 'run.log'}")
    fgmres = [int(v) for v in re.findall(r"\[LINEAR_FGMRES\] iterations=(\d+)", result.stdout)]
    nr = re.findall(r"\[NR\] status=(\w+) iterations=(\d+).*dt=([\d.eE+-]+)", result.stdout)
    if not nr or nr[-1][0] != "converged":
        raise RuntimeError(f"Newton did not converge in {out}")
    ledger = re.findall(r"\[NR_ENERGY_LEDGER\].*diffusion_delta=([\d.eE+-]+)", result.stdout)
    return {"mode": mode, "base_nx": n, "wall_seconds": elapsed,
            "fgmres_iterations": sum(fgmres), "newton_iterations": int(nr[-1][1]),
            "dt": float(nr[-1][2]), "diffusion_delta": float(ledger[-1])}


def plot_existing(root, resolutions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    finest_n = max(resolutions)
    for n in resolutions:
        shape = (2*n, n//2, n//2)
        extent = (1.0, 0.25, 0.25)
        x = (np.arange(shape[0])+0.5)/shape[0]
        _, ref, _, _ = compose(root/f"n{n}_reference", shape, extent)
        _, ref0, _, _ = compose(root/f"n{n}_reference", shape, extent, cycle=0)
        for mode, color in (("baseline", "tab:red"), ("shared", "tab:blue")):
            case = root/f"n{n}_{mode}"
            _, field, projected, _ = compose(case, shape, extent, reference=ref)
            _, initial, projected0, _ = compose(case, shape, extent,
                                                cycle=0, reference=ref0)
            error = np.abs((field.astype(float)-initial.astype(float))
                           -(projected.astype(float)-projected0.astype(float)))
            axes[1].plot(x, np.maximum(error.mean(axis=(0, 1)), 1e-9),
                         color=color, linestyle="-" if n == finest_n else "--",
                         label=f"{mode}, N={n}")
            if n == finest_n:
                iy = int(0.0625/(0.25/shape[1]))
                iz = shape[2]//2
                axes[0].plot(x, field[iz, iy, :], color=color, label=mode)
        if n == finest_n:
            axes[0].plot(x, ref[iz, iy, :], color="black", label="uniform fine")
    for axis in axes:
        axis.axvline(0.25, color="0.45", linestyle=":")
        axis.set_xlim(0.15, 0.35)
        axis.set_xlabel("x1")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_title(f"Radiation energy after one step (N={finest_n})")
    axes[0].set_ylabel("Er at y=0.0625, z=0.125")
    axes[1].set_title("Error in one-step change, averaged over y,z")
    axes[1].set_ylabel("Absolute error versus uniform fine")
    axes[1].set_yscale("log")
    fig.savefig(root/"profile_comparison.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("bin/athena"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--dt", type=float, default=5e-5)
    parser.add_argument("--contrast", type=float, default=0.1)
    parser.add_argument("--transverse-amplitude", type=float, default=0.8)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    binary = args.binary.resolve()
    root = args.output.resolve()
    if args.plot_only:
        plot_existing(root, args.resolutions)
        return
    root.mkdir(parents=True, exist_ok=False)
    source = (Path(__file__).resolve().parents[2] / "inputs/radiation/athinput.nrfld_diff_amr_interface").read_text()
    results = []
    for n in args.resolutions:
        if n % 8:
            raise ValueError("Base x1 resolution must be divisible by 8")
        shape = (2*n, n//2, n//2)
        extent = (1.0, 0.25, 0.25)
        fields = {}
        for mode in ("reference", "baseline", "two_point", "shared"):
            case = root / f"n{n}_{mode}"
            record = run(binary, source, case, n, mode, args.dt,
                         args.contrast, args.transverse_amplitude)
            t, field, projected, blocks = compose(case, shape, extent,
                                                  reference=fields.get("reference"))
            t0, initial, projected_initial, _ = compose(
                case, shape, extent, cycle=0,
                reference=fields.get("reference_initial"))
            record.update(output_time=t, blocks=blocks,
                          minimum=float(field.min()), maximum=float(field.max()))
            if mode != "reference":
                ref = fields["reference"]
                error = np.abs(field.astype(np.float64)-projected.astype(np.float64))
                raw_error = np.abs(field.astype(np.float64)-ref.astype(np.float64))
                initial_error = np.abs(initial.astype(np.float64)
                                       -projected_initial.astype(np.float64))
                increment_error = np.abs((field.astype(np.float64)-initial.astype(np.float64))
                                         -(projected.astype(np.float64)
                                           -projected_initial.astype(np.float64)))
                x = (np.arange(shape[0])+0.5)/shape[0]
                band = (np.abs(x-0.25) <= 0.05) | (np.abs(x-0.75) <= 0.05)
                record.update(mean_abs_error=float(error.mean()),
                              interface_mean_abs_error=float(error[:, :, band].mean()),
                              max_abs_error=float(error.max()),
                              raw_mean_abs_error=float(raw_error.mean()),
                              initial_mean_abs_error=float(initial_error.mean()),
                              increment_mean_abs_error=float(increment_error.mean()),
                              increment_interface_mean_abs_error=float(
                                  increment_error[:, :, band].mean()),
                              increment_max_abs_error=float(increment_error.max()))
                if abs(t-results[-1]["output_time"]) > 1e-10:
                    raise ValueError("Cases have different final times")
            fields[mode] = field
            fields[mode+"_initial"] = initial
            results.append(record)
            print(json.dumps(record), flush=True)
        np.savez_compressed(root / f"n{n}_fields.npz", **fields)
    (root / "summary.json").write_text(json.dumps(results, indent=2)+"\n")
    if args.plot:
        plot_existing(root, args.resolutions)


if __name__ == "__main__":
    main()
