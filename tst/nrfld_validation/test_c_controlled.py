#!/usr/bin/env python3
"""WOR-22: fixed-state, fixed-dt diffusion/AMR controls using the existing Test C.

Builds in an isolated source snapshot; never reconfigures the main checkout.
This is a radiation-only diagnostic, not a protostellar-collapse validation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import h5py

from run_suite import (build_case, binary_configuration, effective_settings,
                       validate_binary_configuration)
from nrfld_validation.io import find_repo_root, read_history, read_athdf
from nrfld_validation.manifest import (base_manifest, sha256_file, utc_now,
                                      write_manifest)


def bounded_run(command, cwd, timeout, memory_gib):
    """Limit the entire command's process group on timeout; disallow core dumps."""
    def limits():
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        cap = int(memory_gib * 1024**3)
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024**2, 32 * 1024**2))

    start = time.monotonic()
    with (cwd / "run.log").open("w") as out:
        proc = subprocess.Popen(command, cwd=cwd, stdout=out, stderr=subprocess.STDOUT,
                                start_new_session=True, preexec_fn=limits)
        timed_out = False
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGKILL)
            code = proc.wait()
    return {"command": command, "raw_status": code, "timeout": timed_out,
            "elapsed_seconds": time.monotonic() - start}


def summarize(case, dt):
    text = (case / "run.log").read_text(errors="replace")
    rows = [dict(re.findall(r"(\w+)=([^\s]+)", line)) for line in text.splitlines()
            if line.startswith("[NR] status=")]
    active = [r for r in rows if float(r.get("dt", 0)) > 0]
    cycles = re.findall(r"^cycle=(\d+) time=([^ ]+)", text, re.M)
    result = {"nr": active, "rejected_vcycles": text.count("Rejecting divergent V-cycle"),
              "backtracks": len(re.findall(r"^\[NR_TRIAL\].*accepted=0", text, re.M)),
              "linear_cap_hits": text.count("Stopping at the configured V-cycle limit:"),
              "last_cycle": int(cycles[-1][0]) if cycles else None,
              "last_time": float(cycles[-1][1]) if cycles else None}
    for key, prefix in (("global_jv", "[NRFLD global Jv]"),
                        ("jv_setup", "[NRFLD Jv setup]"),
                        ("jv_skipped", "[NRFLD Jv skipped]"),
                        ("handoff", "[NRFLD handoff]")):
        result[key] = [dict(re.findall(r"(\w+)=([^\s]+)", line))
                       for line in text.splitlines() if line.startswith(prefix)]
    result["numerically_completed"] = (
        len(active) == 1 and all(r["status"] == "converged" for r in active)
        and math.isclose(float(active[0]["dt"]), dt, rel_tol=1e-12)
        and result["last_cycle"] == 1
        and math.isclose(result["last_time"], dt, rel_tol=1e-12)
        and "FATAL ERROR" not in text)
    rss = case / "resources.txt"
    if rss.exists():
        match = re.search(r"max_rss_kib=(\d+)", rss.read_text())
        result["max_rss_kib"] = int(match[1]) if match else None
    histories = list((case / "output").glob("*.hst"))
    if histories:
        h = read_history(histories[0])
        energy = h["Er_volume"]
        result["radiation_energy_relative_change"] = float((energy[-1]-energy[0])/energy[0])
    return result


def analyze_saved(root):
    """Re-analyze existing outputs without launching or rebuilding Athena."""
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    groups = {}
    for record in manifest["samples"]:
        case = root / record["tag"]
        record.update(summarize(case, manifest["arguments"]["dt"]))
        outputs = sorted((case / "output").glob("*.athdf"))
        if outputs:
            with h5py.File(outputs[0]) as raw:
                levels = raw["Levels"][:]
                faces = raw["x1f"][:]
                record.update(meshblocks=int(raw.attrs["NumMeshBlocks"]),
                              levels=np.unique(levels).astype(int).tolist(),
                              dx_min=float(np.diff(faces, axis=1).min()),
                              dx_max=float(np.diff(faces, axis=1).max()),
                              finest_x_intervals=sorted(set((float(x[0]), float(x[-1]))
                                  for x in faces[levels == levels.max()])))
            initial = read_athdf(outputs[0])
            sigma = initial["sigma_R"]
            record["initial_sigma_R_range"] = [float(sigma.min()), float(sigma.max())]
        record["pass"] = record["raw_status"] == 0 and record["numerically_completed"]
        if record["numerically_completed"] and outputs:
            fields = read_athdf(outputs[-1])
            profile = fields["Er"].mean(axis=(0, 1), dtype=np.float64)
            record.update(Er_min=float(fields["Er"].min()), Er_max=float(fields["Er"].max()),
                          fields_finite=bool(np.isfinite(fields["Er"]).all()),
                          transverse_max_deviation=float(np.max(np.abs(fields["Er"]-profile))))
            record["pass"] &= record["fields_finite"]
            groups.setdefault((record["diffusion_number_fine_low"], record["contrast"]), []).append(
                (record, fields["x1v"], profile))
    for group in groups.values():
        reference = next((g for g in group if g[0]["layout"] == "uniform_fine"), None)
        if reference:
            for record, x, profile in group:
                if len(x) == len(reference[1]) and np.allclose(x, reference[1]):
                    record["relative_L1_vs_uniform_fine"] = float(
                        np.sum(np.abs(profile-reference[2]))/np.sum(np.abs(reference[2])))
    manifest["analysis_script_sha256"] = sha256_file(Path(__file__))
    manifest["reanalyzed_at"] = utc_now()
    manifest["all_cases_pass"] = all(r["pass"] for r in manifest["samples"])
    write_manifest(manifest_path, manifest)
    return 0 if manifest["all_cases_pass"] else 1


def run(args):
    repo = find_repo_root()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, root / Path(__file__).name)
    manifest = base_manifest(repo, "WOR-22_test_C_controlled", args.build_jobs)
    manifest.update(samples=[], arguments=vars(args) | {"output": str(root),
                    "binary": str(args.binary) if args.binary else None})
    records = []
    manifest["commands"]["build"] = records
    manifest["restart"] = "none: deterministic pgen, one fixed-dt solve"
    manifest["scope"] = ("only_rad, frozen hydro, lagged-per-step diffusion coefficients; "
                         "not full collapse; variable limiter is opt-in")
    manifest["harness_sha256"] = sha256_file(Path(__file__))
    manifest_path = root / "manifest.json"
    write_manifest(manifest_path, manifest)
    with (root / "source_changes.patch").open("w") as out:
        subprocess.run(["git", "diff", "HEAD", "--", "src", "configure.py", "Makefile.in"],
                       cwd=repo, stdout=out, check=True)
    if args.binary:
        exe = args.binary.resolve()
        manifest["provenance"]["calculation"] = {"source": "existing_binary",
                                                    "source_revision_verified": False}
    else:
        build = build_case(repo, root, "nrfld_diff_amr_interface", args.build_jobs,
                           args.build_timeout, records)
        exe = build["build_source"] / "bin/athena"
        manifest["provenance"]["calculation"] = {
            "source": "isolated_current_worktree", "source_sha256": build["build_source_sha256"]}
    config = binary_configuration(exe, root, 30, records)
    validate_binary_configuration(config, "nrfld_diff_amr_interface", exe)
    manifest["provenance"]["calculation"].update(binary=str(exe),
                 binary_sha256=sha256_file(exe), configuration=config, mpi_ranks=1)
    source_input = repo / "inputs/radiation/athinput.nrfld_diff_amr_interface"
    shutil.copy2(source_input, root / "input.original")
    manifest["input"] = {"sha256": sha256_file(source_input)}
    # All layouts have the same domain. Uniform-fine matches the AMR finest dx.
    base_nx1 = args.base_nx1
    transverse_n = 8 * args.transverse_blocks
    transverse_extent = transverse_n/base_nx1
    layouts = {
        "uniform_coarse": (base_nx1, transverse_n, "none", .25, .75),
        "uniform_fine": (2*base_nx1, 2*transverse_n, "none", .25, .75),
        "amr_aligned": (base_nx1, transverse_n, "static", .25, .75),
        # Shift by a ROOT MeshBlock (8/32), not a fine-block width: otherwise
        # Athena rounds the request outward and changes the number of blocks.
        # The left material interface is now fine interior, the right is coarse
        # interior. Both are separated from AMR boundaries; full fine is control.
        "amr_shifted": (base_nx1, transverse_n, "static",
                        args.shifted_left, args.shifted_right),
    }
    for number in args.diffusion_numbers:
        for contrast in args.contrasts:
            group = []
            # q = dt D_low / dx_fine^2; D=c_code/(3*kappa*rho_unit*L_unit*rho).
            dx_fine = 1.0/(2*base_nx1)
            kappa = args.dt * (2.99792458e10 / 1e7) / (
                3 * 1e-7 * 4e10 * dx_fine**2 * number)
            for name, (nx, ny, refinement, left, right) in layouts.items():
                if name not in args.layouts:
                    continue
                tag = f"q{number:g}_contrast{contrast:g}_{name}"
                case = root / tag
                case.mkdir()
                (case / "output").mkdir()
                overrides = [
                    f"mesh/nx1={nx}", f"mesh/nx2={ny}", f"mesh/nx3={ny}",
                    f"mesh/x2max={transverse_extent:.17g}",
                    f"mesh/x3max={transverse_extent:.17g}",
                    f"mesh/refinement={refinement}", f"refinement1/x1min={left}",
                    f"refinement1/x1max={right}", "hydro/active=fixed",
                    f"refinement1/x2max={transverse_extent:.17g}",
                    f"refinement1/x3max={transverse_extent:.17g}",
                    f"time/tlim={args.dt:.17g}", "time/nlim=1",
                    f"problem/density_high={contrast:.17g}", f"fld/const_opacity={kappa:.17g}",
                    f"problem/pulse_center={args.pulse_center:.17g}",
                    f"problem/pulse_width={args.pulse_width:.17g}",
                    f"nrfld/linear_max_vcycles={args.linear_max_vcycles}",
                    f"nrfld/mgmode={args.mgmode}",
                    f"nrfld/smoother={args.smoother}",
                    f"nrfld/omega={args.omega:.17g}",
                    "nrfld/nr_max_iterations=64",
                    f"nrfld/nr_threshold={args.nr_threshold:.17g}",
                    "nrfld/nr_accept_l2_threshold=-1", "nrfld/nr_accept_max_threshold=-1",
                    "nrfld/coefficient_diagnostics=true",
                    f"nrfld/coefficient_diagnostic_limit={args.coefficient_diagnostic_limit}",
                    f"nrfld/global_jacobian_check={'true' if args.global_jv else 'false'}",
                    f"nrfld/global_jacobian_iteration={args.jv_iteration}",
                    f"nrfld/handoff_diagnostics={'true' if args.handoff else 'false'}",
                    f"fld/fixed_flux_limiter={'false' if args.variable_limiter else 'true'}",
                    f"nrfld/diagnostic_verbosity={args.verbosity}", "output1/data_format=%.16e",
                    "output2/file_type=hdf5", "output2/variable=uov", "output2/id=fields",
                    "output2/dcycle=1", "output2/dt=-1",
                ]
                if args.symmetric_rb_sweeps is not None:
                    overrides.append(
                        f"nrfld/symmetric_rb_sweeps={args.symmetric_rb_sweeps}")
                if args.cache_coefficient_hierarchy is not None:
                    overrides.append(
                        "nrfld/cache_coefficient_hierarchy="
                        f"{args.cache_coefficient_hierarchy}")
                if args.linear_fixed_vcycles is not None:
                    overrides.extend(["nrfld/threshold=-1",
                                      f"nrfld/niteration={args.linear_fixed_vcycles}"])
                if args.linear_divergence_patience is not None:
                    overrides.extend([
                        f"nrfld/linear_divergence_patience={args.linear_divergence_patience}",
                        f"nrfld/linear_max_transient_growth={args.linear_max_transient_growth}"])
                settings = effective_settings(source_input, overrides)
                sections = {}
                for key, value in settings.items():
                    section, param = key.split("/", 1)
                    sections.setdefault(section, {})[param] = value
                input_path = case / "input.used"
                input_path.write_text("\n".join(f"<{s}>\n" + "\n".join(
                    f"{k} = {v}" for k, v in values.items()) for s, values in sections.items())+"\n")
                command = ["/usr/bin/time", "-f", "max_rss_kib=%M elapsed_s=%e", "-o",
                           str(case / "resources.txt"), str(exe), "-i", str(input_path),
                           "-d", str(case / "output")]
                record = bounded_run(command, case, args.timeout, args.memory_gib)
                record.update(summarize(case, args.dt), tag=tag, layout=name, contrast=contrast,
                              input_sha256=sha256_file(input_path), settings=settings,
                              kappa_cgs=kappa, diffusion_number_fine_low=number,
                              diffusion_number_fine_high=number/contrast,
                              diffusion_number_coarse_low=number/4,
                              post_mg_smooth_default=16 if refinement == "static" else 0)
                record["pass"] = record["raw_status"] == 0 and record["numerically_completed"]
                if record["numerically_completed"]:
                    final = sorted((case / "output").glob("*.athdf"))[-1]
                    fields = read_athdf(final)
                    # HDF5 may be float32; accumulate in double to avoid
                    # mistaking reduction roundoff for transverse variation.
                    profile = fields["Er"].mean(axis=(0, 1), dtype=np.float64)
                    group.append((record, fields["x1v"], profile))
                    record["Er_min"] = float(fields["Er"].min())
                    record["Er_max"] = float(fields["Er"].max())
                    record["transverse_max_deviation"] = float(np.max(np.abs(fields["Er"]-profile)))
                    record["fields_finite"] = bool(np.isfinite(fields["Er"]).all())
                    with h5py.File(final) as raw:
                        record["meshblocks"] = int(raw.attrs["NumMeshBlocks"])
                        record["levels"] = np.unique(raw["Levels"][:]).astype(int).tolist()
                        record["dx_min"] = float(np.diff(raw["x1f"][:], axis=1).min())
                        record["dx_max"] = float(np.diff(raw["x1f"][:], axis=1).max())
                        levels = raw["Levels"][:]
                        faces = raw["x1f"][:]
                        record["finest_x_intervals"] = sorted(set(
                            (float(x[0]), float(x[-1]))
                            for x in faces[levels == levels.max()]))
                    record["pass"] &= record["fields_finite"]
                manifest["samples"].append(record)
                write_manifest(manifest_path, manifest)
                iterations = [int(r["iterations"]) for r in record["nr"]]
                print(f"{tag}: pass={record['pass']} NR={iterations} "
                      f"reject={record['rejected_vcycles']} RSS={record.get('max_rss_kib')} KiB", flush=True)
            reference = next((g for g in group if g[0]["layout"] == "uniform_fine"), None)
            if reference:
                for record, x, profile in group:
                    if len(x) == len(reference[1]) and np.allclose(x, reference[1]):
                        record["relative_L1_vs_uniform_fine"] = float(
                            np.sum(np.abs(profile-reference[2]))/np.sum(np.abs(reference[2])))
            write_manifest(manifest_path, manifest)
    manifest["status"] = "finished"
    manifest["finished_at"] = utc_now()
    manifest["all_cases_pass"] = all(r["pass"] for r in manifest["samples"])
    write_manifest(manifest_path, manifest)
    return analyze_saved(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--reanalyze", action="store_true",
                        help="Only refresh metrics from existing outputs/manifest")
    parser.add_argument("--build-jobs", type=int, default=2)
    parser.add_argument("--build-timeout", type=float, default=600)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--memory-gib", type=float, default=4)
    parser.add_argument("--dt", type=float, default=2e-4)
    parser.add_argument("--nr-threshold", type=float, default=1e-10)
    parser.add_argument("--pulse-center", type=float, default=.25)
    parser.add_argument("--pulse-width", type=float, default=.08)
    parser.add_argument("--base-nx1", type=int, default=32)
    parser.add_argument("--transverse-blocks", type=int, default=1)
    parser.add_argument("--shifted-left", type=float, default=0.)
    parser.add_argument("--shifted-right", type=float, default=.5)
    parser.add_argument("--verbosity", type=int, choices=[1, 2], default=1)
    parser.add_argument("--variable-limiter", action="store_true",
                        help="Use the production flux limiter instead of fixed lambda=1/3")
    parser.add_argument("--global-jv", action="store_true",
                        help="Single-rank Test C boundary-inclusive operator diagnostic")
    parser.add_argument("--jv-iteration", type=int, default=0)
    parser.add_argument("--handoff", action="store_true",
                        help="Compare MG and NR correction residuals (first four solves)")
    parser.add_argument("--linear-fixed-vcycles", type=int,
                        help="Diagnostic: bypass monotone rejection and run exactly N V-cycles")
    parser.add_argument("--linear-max-vcycles", type=int, default=30)
    parser.add_argument("--mgmode", choices=["MGI", "FMG"], default="MGI",
                        help="Existing linearMG solve mode")
    parser.add_argument("--smoother", choices=["jacobi-rb", "jacobi-double", "jacobi"],
                        default="jacobi-rb",
                        help="Existing linearMG smoother selection")
    parser.add_argument("--omega", type=float, default=1.0,
                        help="Existing linearMG smoother relaxation parameter")
    parser.add_argument("--linear-divergence-patience", type=int,
                        help="Allow this many worsening V-cycles before rollback")
    parser.add_argument("--linear-max-transient-growth", type=float, default=2.0,
                        help="Immediate rollback bound relative to the best linear residual")
    parser.add_argument("--symmetric-rb-sweeps", choices=["true", "false"],
                        help="Override the RBGS post-smoothing order for WOR-29 controls")
    parser.add_argument("--cache-coefficient-hierarchy", choices=["true", "false"],
                        help="Override MG coefficient-hierarchy caching for WOR-30 controls")
    parser.add_argument("--coefficient-diagnostic-limit", type=int, default=1,
                        help="Number of linear solves with coefficient fingerprints")
    parser.add_argument("--layouts", nargs="+", default=["uniform_coarse", "uniform_fine",
                        "amr_aligned", "amr_shifted"], choices=["uniform_coarse",
                        "uniform_fine", "amr_aligned", "amr_shifted"])
    parser.add_argument("--diffusion-numbers", type=float, nargs="+", default=[1, 1e4])
    parser.add_argument("--contrasts", type=float, nargs="+", default=[1, 10])
    args = parser.parse_args()
    if not (1 <= args.build_jobs <= 4 and args.build_timeout > 0
            and args.dt > 0 and args.timeout > 0
            and args.memory_gib > 0 and args.nr_threshold > 0 and args.pulse_width > 0
            and args.base_nx1 >= 32 and args.base_nx1 % 8 == 0
            and args.transverse_blocks >= 1
            and args.jv_iteration >= 0
            and args.coefficient_diagnostic_limit >= 0
            and (args.linear_fixed_vcycles is None or args.linear_fixed_vcycles > 0)
            and args.linear_max_vcycles > 0 and 0 < args.omega <= 1
            and (args.linear_divergence_patience is None
                 or args.linear_divergence_patience > 0)
            and args.linear_max_transient_growth >= 1
            and 0 <= args.shifted_left < args.shifted_right <= 1
            and all(q > 0 for q in args.diffusion_numbers)
            and all(c >= 1 for c in args.contrasts)):
        parser.error("Invalid resource limit, dt, diffusion number or density contrast")
    sys.exit(analyze_saved(args.output.resolve()) if args.reanalyze else run(args))
