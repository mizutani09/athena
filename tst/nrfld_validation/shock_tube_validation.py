#!/usr/bin/env python3
"""Compare NR-FLD shock tubes with cached, high-resolution 1D Rad-EOS hydro."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from html import escape
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nrfld-validation")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from diffusion_validation import materialize_input  # noqa: E402
from nrfld_validation.io import find_repo_root, read_athdf  # noqa: E402
from nrfld_validation.report_style import REPORT_TYPOGRAPHY  # noqa: E402
from nrfld_validation.manifest import (base_manifest, sha256_file, tree_sha256,
                                       utc_now, write_manifest)  # noqa: E402
from run_suite import (CommandExecutionError, binary_configuration, checked,
                       effective_settings, prepare_build_source)  # noqa: E402
from shock_tube_reference_adapter import adapt_reference_source  # noqa: E402

R_GAS = 8.3144621e7
A_RAD = 7.5657e-15
T_FINAL = 0.1
OUTPUT_DT = 0.005
REF_N = (2048, 4096)
FLD_N = (64, 128)
FIELDS = ("rho", "velocity", "temperature", "gas_energy", "radiation_energy", "total_pressure")
LABELS = {"rho": r"$\rho$ [g cm$^{-3}$]", "velocity": r"$v_x$ [cm s$^{-1}$]",
          "temperature": "$T$ [K]", "gas_energy": r"$e_{\rm gas}$ [erg cm$^{-3}$]",
          "radiation_energy": r"$E_{\rm rad}$ [erg cm$^{-3}$]",
          "total_pressure": r"$P_{\rm tot}$ [erg cm$^{-3}$]"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/d/output/rad"))
    parser.add_argument("--run-name")
    parser.add_argument("--reference-repo", type=Path,
                        default=Path("/home/kosuke/simulation/mhd/athena_st"))
    parser.add_argument("--reference-cache-root", type=Path,
                        default=Path("/mnt/d/output/rad/reference_cache/shock_tube_rad_eos"))
    parser.add_argument("--import-reference-root", type=Path,
                        help="Import existing reference build and case2048/case4096 outputs into the cache.")
    parser.add_argument("--import-fld-root", type=Path,
                        help="Import a matching FLD build and case64_full/case128_full outputs.")
    parser.add_argument("--max-cores", type=int, default=8)
    norm_mode = parser.add_mutually_exclusive_group()
    norm_mode.add_argument("--strict-backward-error", dest="strict_backward_error",
                           action="store_true", default=True,
                           help="Use the local Newton backward-error norm and require strict convergence (default).")
    norm_mode.add_argument("--legacy-residual", dest="strict_backward_error",
                           action="store_false",
                           help="Reproduce the older normalized-residual run with relaxed Newton acceptance.")
    parser.add_argument("--run-timeout", type=float, default=7200)
    parser.add_argument("--build-timeout", type=float, default=900)
    parser.add_argument("--postprocess", type=Path,
                        help="Regenerate metrics, figures and HTML from this completed run.")
    args = parser.parse_args()
    if args.max_cores < 1 or args.run_timeout <= 0 or args.build_timeout <= 0:
        parser.error("cores and timeouts must be positive")
    return args


def shock_settings(fld_template: Path) -> dict:
    values = effective_settings(fld_template, [])
    get = lambda key: float(values[key])
    for key in ("hydro/rho_unit", "hydro/egas_unit", "hydro/leng_unit",
                "hydro/mu", "hydro/gamma", "problem/rho0_L", "problem/rho0_R",
                "problem/T0_L", "problem/T0_R", "problem/v0_L", "problem/v0_R"):
        if key not in values:
            raise ValueError(f"missing FLD shock input setting: {key}")
    if not math.isclose(get("hydro/gamma"), 5/3, rel_tol=1e-10):
        raise ValueError("reference Rad EOS is fixed to gas gamma=5/3")
    sides = {}
    for side in ("L", "R"):
        rho, temp = get(f"problem/rho0_{side}"), get(f"problem/T0_{side}")
        gas_pressure = rho*R_GAS/get("hydro/mu")*temp
        rad_pressure = A_RAD*temp**4/3
        sides[side] = {"rho": rho, "temperature": temp,
                       "p_gas": gas_pressure, "p_rad": rad_pressure,
                       "p_total": gas_pressure+rad_pressure,
                       "gas_energy": gas_pressure/(get("hydro/gamma")-1),
                       "radiation_energy": A_RAD*temp**4}
    return {"rho_unit": get("hydro/rho_unit"), "egas_unit": get("hydro/egas_unit"),
            "leng_unit": get("hydro/leng_unit"), "mu": get("hydro/mu"),
            "gamma": get("hydro/gamma"), "velocity_unit": math.sqrt(
                get("hydro/egas_unit")/get("hydro/rho_unit")), "sides": sides}


def reference_overrides(n: int, physical: dict) -> list[str]:
    return [f"mesh/nx1={n}", "mesh/nx2=1", "mesh/nx3=1",
            "meshblock/nx1=512", "meshblock/nx2=1", "meshblock/nx3=1",
            "output2/ghost_zones=false", "output2/x2_slice=0.5", "output2/x3_slice=0.5",
            "output3/file_type=hdf5", "output3/id=uov_x_slice", "output3/variable=uov",
            f"output3/dt={OUTPUT_DT}", "output3/x2_slice=0.5", "output3/x3_slice=0.5",
            "output3/ghost_zones=false", f"problem/p0_L={physical['sides']['L']['p_total']:.17g}",
            f"problem/p0_R={physical['sides']['R']['p_total']:.17g}",
            f"job/problem_id=rad_eos_ref{n}"]


def fld_overrides(n: int, strict_backward_error: bool = False) -> list[str]:
    width = 32/n
    return [f"mesh/nx1={n}", "mesh/nx2=32", "mesh/nx3=32",
            "meshblock/nx1=32", "meshblock/nx2=32", "meshblock/nx3=32",
            f"mesh/x2max={width:.17g}", f"mesh/x3max={width:.17g}",
            "output2/ghost_zones=false", "output3/ghost_zones=false",
            f"output2/x2_slice={width/2:.17g}", f"output2/x3_slice={width/2:.17g}",
            f"output3/x2_slice={width/2:.17g}", f"output3/x3_slice={width/2:.17g}",
            "time/ncycle_out=100", "nrfld/linear_solver=fgmres",
            "nrfld/failure_policy=abort", "nrfld/diagnostic_verbosity=0",
            "nrfld/nr_accept_l2_threshold=1e-6", "nrfld/nr_accept_max_threshold=1e-5",
            f"job/problem_id=shock_fld_n{n}_full"] + ([
                "nrfld/backward_error_convergence=true",
                "nrfld/nr_accept_l2_threshold=0",
                "nrfld/nr_accept_max_threshold=0"] if strict_backward_error else [])


def reference_fingerprint(repo: Path, template: Path, physical: dict) -> dict:
    return {"reference_git_commit": __import__("subprocess").check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            "reference_src_sha256": tree_sha256(repo/"src"),
            "reference_template_sha256": sha256_file(template),
            "adapter_sha256": sha256_file(HERE/"shock_tube_reference_adapter.py"),
            "physical": physical, "resolutions": list(REF_N), "final_time": T_FINAL,
            "output_dt": OUTPUT_DT, "configure": ["--prob=shock_tube_hydro",
                                                  "--eos=general/rad", "--flux=hllc",
                                                  "-hdf5", "-mpi", "--cflag=-O3"]}


def fingerprint_key(fingerprint: dict) -> str:
    return sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:20]


def verify_source_snapshot(repo: Path, snapshot: Path, adapted_reference: bool):
    """Reject imported binaries whose source differs from the current checkout."""
    patched = {Path("eos/general/rad.cpp"), Path("pgen/shock_tube_hydro.cpp")}
    for original in (repo/"src").rglob("*"):
        if not original.is_file() or original.name == "defs.hpp":
            continue
        relative = original.relative_to(repo/"src")
        if adapted_reference and relative in patched:
            continue
        target = snapshot/"src"/relative
        if not target.is_file() or sha256_file(original) != sha256_file(target):
            raise ValueError(f"source snapshot differs from checkout: {relative}")
    if adapted_reference:
        with tempfile.TemporaryDirectory(prefix="shock_ref_adapter_check_") as temporary:
            expected = Path(temporary)
            for relative in patched:
                target = expected/"src"/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(repo/"src"/relative, target)
            adapt_reference_source(expected)
            for relative in patched:
                if sha256_file(expected/"src"/relative) != sha256_file(snapshot/"src"/relative):
                    raise ValueError(f"adapted source mismatch: {relative}")


def assert_effective_input(actual: Path, template: Path, overrides: list[str]):
    wanted = effective_settings(template, overrides)
    got = effective_settings(actual, [])
    differences = {}
    for key in wanted.keys() | got.keys():
        expected, actual_value = wanted.get(key), got.get(key)
        if expected == actual_value:
            continue
        try:
            numerically_equal = math.isclose(float(expected), float(actual_value),
                                             rel_tol=1e-12, abs_tol=1e-12)
        except (ValueError, TypeError):
            numerically_equal = False
        if not numerically_equal:
            differences[key] = (expected, actual_value)
    if differences:
        raise ValueError(f"imported input differs from expected: {actual}: {differences}")


def pair_paths(case: Path, index: int) -> tuple[Path, Path]:
    files = []
    for output_id in ("prim_x_slice", "uov_x_slice"):
        paths = sorted((case/"output").glob(f"*.{output_id}.*.athdf"))
        if len(paths) != 21:
            raise ValueError(f"expected 21 {output_id} snapshots in {case}; found {len(paths)}")
        files.append(paths[index])
    return files[0], files[1]


def profile(case: Path, index: int, kind: str, physical: dict) -> dict:
    ppath, upath = pair_paths(case, index)
    prim, uov = read_athdf(ppath), read_athdf(upath)
    n = len(prim["x1v"])
    if n not in REF_N+FLD_N:
        raise ValueError(f"unexpected x resolution {n} in {ppath}")
    if not math.isclose(float(prim["Time"]), float(uov["Time"]), abs_tol=1e-7):
        raise ValueError(f"time mismatch between {ppath} and {upath}")
    def arr(data, name):
        values = np.asarray(data[name], dtype=float).reshape(-1, n)
        return values.mean(axis=0)
    rho = arr(prim, "rho")*physical["rho_unit"]
    vel = arr(prim, "vel1")*physical["velocity_unit"]
    if kind == "fld":
        temp = arr(uov, "T_gas")
        egas = arr(uov, "e_gas")
        erad = arr(uov, "E_rad")
        total_pressure = arr(uov, "P_tot")
    else:
        temp = arr(uov, "temp")
        gas_pressure = rho*R_GAS/physical["mu"]*temp
        egas = gas_pressure/(physical["gamma"]-1)
        erad = A_RAD*temp**4
        total_pressure = arr(prim, "press")*physical["egas_unit"]
    result = {"x": np.asarray(prim["x1v"], dtype=float), "time": float(prim["Time"]),
              "cycles": int(prim["NumCycles"]), "rho": rho, "velocity": vel,
              "temperature": temp, "gas_energy": egas, "radiation_energy": erad,
              "total_pressure": total_pressure}
    if not all(np.all(np.isfinite(result[field])) for field in FIELDS):
        raise ValueError(f"nonfinite field in {ppath}")
    if np.any(rho <= 0) or np.any(temp <= 0) or np.any(egas <= 0) or np.any(erad <= 0):
        raise ValueError(f"nonpositive thermodynamic state in {ppath}")
    return result


def coarsen(reference: dict, n: int) -> dict:
    if len(reference["x"]) % n:
        raise ValueError("reference resolution must be divisible by FLD resolution")
    factor = len(reference["x"])//n
    return {key: value.reshape(n, factor).mean(axis=1) for key, value in reference.items()
            if isinstance(value, np.ndarray)}


def metrics(numerical: np.ndarray, reference: np.ndarray) -> dict:
    error = np.abs(numerical-reference)
    denom = float(np.sum(np.abs(reference)))
    return {"relative_l1": float(np.sum(error)/denom) if denom else None,
            "mean_absolute": float(np.mean(error)), "max_absolute": float(np.max(error)),
            "max_over_range": float(np.max(error)/np.ptp(reference)) if np.ptp(reference) else None}


def save_csv(path: Path, numerical: dict, reference: dict):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x_code"]+[f"{field}_{source}" for field in FIELDS
                                     for source in ("fld", "reference")])
        writer.writerows(zip(numerical["x"], *(a for field in FIELDS
                            for a in (numerical[field], reference[field]))))


def parse_solver_log(path: Path, cycles: int) -> dict:
    log = path.read_text(errors="replace")
    status = re.findall(r"\[NR\] status=([^\s]+)", log)
    active = [s for s in status if s != "dt_zero_initialization"]
    linear = re.findall(r"\[LINEAR_FGMRES\].*?relative_residual=([^\s]+).*?target=([^\s]+)", log)
    l2 = [float(x) for x in re.findall(r"\[NR\] status=[^\n]+? final_l2=([^\s]+)", log)]
    mx = [float(x) for x in re.findall(r"\[NR\] status=[^\n]+? final_max=([^\s]+)", log)]
    good = (len(active) == cycles and all(s in {"converged", "accepted_relaxed"} for s in active)
            and bool(linear) and all(float(a) <= float(b)*1.000001 for a,b in linear))
    return {"good": bool(good), "cycles": cycles, "nr_status_count": len(active),
            "strict_converged": active.count("converged"),
            "relaxed_accepted": active.count("accepted_relaxed"),
            "fgmres_calls": len(linear), "max_final_l2": max(l2) if l2 else None,
            "max_final_point": max(mx) if mx else None}


def runtime_resources(path: Path) -> int | None:
    if not path.is_file():
        return None
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", path.read_text())
    return int(match.group(1)) if match else None


def save_figure(fig, path: Path):
    path.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def make_plots(root: Path, series: dict, summary: dict):
    reference = series["r4096"][-1]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for axis, field in zip(axes.flat, FIELDS):
        axis.plot(reference["x"], reference[field], color="black", lw=1.5,
                  label="Rad EOS hydro, N=4096")
        for n, color in ((64, "#1f77b4"), (128, "#d65c27")):
            q = series[f"n{n}"][-1]
            axis.plot(q["x"], q[field], marker="o", ms=2.5, lw=1,
                      color=color, label=f"NR-FLD N={n}")
        axis.set(xlabel="x / L", ylabel=LABELS[field], title=field.replace("_", " "))
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    save_figure(fig, root/"figures/final_profiles")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for axis, field in zip(axes.flat, FIELDS):
        for n, color in ((64, "#1f77b4"), (128, "#d65c27")):
            q = series[f"n{n}"][-1]
            ref = coarsen(reference, n)
            axis.plot(q["x"], np.abs(q[field]-ref[field]), color=color,
                      label=f"N={n}")
        axis.set(xlabel="x / L", ylabel=f"|Δ{field}|", title=field.replace("_", " "))
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    save_figure(fig, root/"figures/final_errors")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for axis, field in zip(axes.flat, FIELDS):
        for n, color in ((64, "#1f77b4"), (128, "#d65c27")):
            values = [metrics(q[field], coarsen(r, n)[field])["relative_l1"]
                      for q, r in zip(series[f"n{n}"][1:], series["r4096"][1:])]
            times = [q["time"] for q in series[f"n{n}"][1:]]
            axis.plot(times, values, "o-", ms=3, color=color, label=f"N={n}")
        axis.set(xlabel="code time", ylabel="relative L1", title=field.replace("_", " "))
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    save_figure(fig, root/"figures/error_history")


def make_video(root: Path, series: dict, records: list):
    target = root/"video/shock_tube.mp4"
    target.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="shock_tube_frames_") as temp:
        directory = Path(temp)
        for frame in range(0, 21, 2):
            fig, axes = plt.subplots(2, 2, figsize=(11, 8))
            for axis, field in zip(axes.flat, ("rho", "velocity", "temperature", "radiation_energy")):
                ref = series["r4096"][frame]
                axis.plot(ref["x"], ref[field], color="black", lw=1.5,
                          label="Rad EOS hydro N=4096")
                for n, color in ((64, "#1f77b4"), (128, "#d65c27")):
                    q = series[f"n{n}"][frame]
                    axis.plot(q["x"], q[field], color=color, lw=1.2, label=f"NR-FLD N={n}")
                axis.set(xlabel="x / L", ylabel=LABELS[field])
                axis.grid(alpha=.25)
                axis.legend(fontsize=8)
            fig.suptitle(f"Shock tube, code time={series['r4096'][frame]['time']:.3f}")
            fig.tight_layout()
            fig.savefig(directory/f"frame_{frame//2:03d}.png", dpi=125)
            plt.close(fig)
        command = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", "4",
                   "-i", str(directory/"frame_%03d.png"),
                   "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
                   "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                   str(target)]
        records.append(checked(command, root, root/"video/encode.log", 300, "video_encode"))
    return {"path": str(target.relative_to(root)), "sha256": sha256_file(target),
            "frames": 11}


def write_report(root: Path, summary: dict):
    comparison = summary.get("comparison", {})
    rows = []
    for n in FLD_N:
        item = comparison.get(f"n{n}", {})
        for field in FIELDS:
            metric = item.get(field)
            if metric:
                rows.append(f'<tr><th>{n}</th><td>{escape(field)}</td><td><strong>{metric["relative_l1"]:.5g}</strong></td>'
                            f'<td>{metric["mean_absolute"]:.5g}</td><td>{metric["max_absolute"]:.5g}</td>'
                            f'<td>{summary["reference_uncertainty"][field]["relative_l1"]:.5g}</td></tr>')
    figures = ''.join(f'<figure><img src="figures/{name}.png" alt="{escape(label)}" loading="lazy">'
                      f'<figcaption>{escape(label)} · <a href="figures/{name}.pdf">PDF</a></figcaption></figure>'
                      for name, label in (("final_profiles", "Final profiles"),
                                          ("final_errors", "Absolute error versus reference"),
                                          ("error_history", "Relative L1 over time"))
                      if (root/"figures"/f"{name}.png").is_file())
    movie = ('<figure><video controls preload="metadata" playsinline width="100%">'
             '<source src="video/shock_tube.mp4" type="video/mp4"></video>'
             '<figcaption>Evolution · <a href="video/shock_tube.mp4">MP4</a></figcaption></figure>'
             if (root/"video/shock_tube.mp4").is_file() else "")
    links = []
    for part, ns in (("reference", REF_N), ("fld", FLD_N)):
        for n in ns:
            key = f"r{n}" if part == "reference" else f"n{n}"
            case = root/part/"cases"/key
            entries = [("input.used", "input"), ("run.log", "run log"),
                       ("resources.txt", "resources"), ("manifest.json", "manifest")]
            if part == "fld":
                entries.append(("profile_final.csv", "profile/reference CSV"))
            entries += [(f"output/{path.name}", "final HDF5") for path in
                        sorted((case/"output").glob("*.uov_x_slice.*.athdf"))[-1:]]
            links.append(f'<li><strong>{part} {n}</strong>: ' + " · ".join(
                f'<a href="{part}/cases/{key}/{escape(name)}">{escape(label)}</a>'
                for name, label in entries if (case/name).is_file()) + '</li>')
    status = summary["status"]
    css = "warn" if status == "completed_with_solver_caveat" else "pass" if status == "passed" else "fail"
    solver_note = ("<strong>All FLD steps reached the strict 1e−10 Newton target.</strong> "
                   "The local backward-error convergence norm estimates the state correction "
                   "needed to satisfy the gas row when floating-point cancellation prevents "
                   "a smaller raw normalized residual. The raw residual and Jacobian used for "
                   "updates are unchanged. Relaxed acceptance is disabled; FGMRES must reach "
                   "its configured true-residual target."
                   if status == "passed" and summary.get("strict_backward_error") else
                   "<strong>Strict Newton convergence was not reached.</strong> The current FLD "
                   "shock tube stalled with a strict 1e−10 target. The reported FLD runs use "
                   "explicit relaxed acceptance bounds L2≤1e−6 and maximum≤1e−5. Their "
                   "number of relaxed steps and measured residuals are recorded in "
                   "<a href=\"summary.json\">summary JSON</a> and case logs. FGMRES must "
                   "reach its own target. This report provides a physical comparison with "
                   "a <strong>solver caveat</strong>; it does not certify strict nonlinear convergence.")
    resource = summary.get("resource_monitor")
    resource_note = (f"<section><h2>Memory and OOM</h2><p>Sampled Athena rank RSS sum "
                     f"<strong>{resource['peak_athena_rss_sum_kib']/1024:.1f} MiB</strong>; "
                     f"whole process tree {resource['peak_process_tree_rss_kib']/1024:.1f} MiB; "
                     f"OOM classification {escape(resource['oom_classification'])}. "
                     f"<a href=\"suite_resource.json\">Per-rank HWM and cgroup events</a>.</p></section>"
                     if resource else "")
    root.joinpath("index.html").write_text(f'''<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NR-FLD shock-tube validation</title><style>
:root{{font:21px/1.6 system-ui,sans-serif;color:#17212b;background:#f7f9fb}}
body{{max-width:1350px;margin:2rem auto;padding:0 1.2rem}}h1,h2{{line-height:1.25}}
section{{background:white;border:1px solid #dce4eb;border-radius:10px;padding:1.3rem;margin:1.4rem 0}}
.overview{{background:#eaf3fb;border-left:5px solid #1769aa;padding:.8rem 1rem;font-size:1.1rem}}
.warn{{color:#8a5700;background:#fff2cb;padding:.2rem .5rem;font-weight:750}}
.pass{{color:#0b683e;background:#e4f5eb;padding:.2rem .5rem;font-weight:750}}
.fail{{color:#ad261f;background:#fdebea;padding:.2rem .5rem;font-weight:750}}
.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:.96rem}}
th,td{{padding:.55rem .65rem;border-bottom:1px solid #dce4eb;text-align:left;white-space:nowrap}}
thead th{{background:#eef3f7}}figure{{margin:1.3rem 0}}img,video{{max-width:100%;height:auto;display:block}}
figcaption{{font-weight:600;margin:.4rem 0}}a{{color:#075ca8}}
{REPORT_TYPOGRAPHY}</style></head><body><h1>NR-FLD shock tube <span class="{css}">{escape(status.upper())}</span></h1>
<p class="overview"><strong>NR-FLD N=64,128</strong> versus 1D Rad EOS hydro N=4096 at code time 0.1; N=2048 assesses reference resolution.</p>
<p>The reference uses a separate <code>athena_st</code> Athena++ with <code>general/rad</code> EOS and HLLC. Its isolated source snapshot adapts the EOS unit constant to the FLD units and initializes total energy through <code>EgasFromRhoP</code>. The upstream repository is unchanged. The initial left/right density, temperature, velocity, and total pressure are checked against FLD. Reference cell averages are compared on each native FLD cell; <strong>no spatial shift or fitted alignment</strong> is applied. Final profiles share the exact requested time. Intermediate Athena++ output crosses each 0.005 time mark on its next step; matching snapshot indices therefore differ by at most {summary.get('max_reference_fld_time_offset', float('nan')):.2g} code-time units in the movie and error-history figure.</p>
<section><h2>Solver qualification</h2><p>{solver_note}</p>
<div class="scroll"><table><thead><tr><th>N</th><th>Newton steps</th><th>Relaxed accepted</th><th>Max final L2</th><th>Max final point</th><th>FGMRES calls</th></tr></thead><tbody>{''.join(f'<tr><th>{n}</th><td>{summary["solver"][f"n{n}"]["nr_status_count"]}</td><td>{summary["solver"][f"n{n}"]["relaxed_accepted"]}</td><td>{summary["solver"][f"n{n}"]["max_final_l2"]:.4g}</td><td>{summary["solver"][f"n{n}"]["max_final_point"]:.4g}</td><td>{summary["solver"][f"n{n}"]["fgmres_calls"]}</td></tr>' for n in FLD_N if f'n{n}' in summary.get('solver', {}))}</tbody></table></div></section>
{resource_note}
<section><h2>Final comparison</h2><p>Relative L1 = Σ|FLD−reference| / Σ|reference|. Reference uncertainty is the corresponding N=2048 versus N=4096 difference after averaging both onto N=128 cells. The reference is numerical, so its finite resolution contributes to the measured discrepancy.</p>
<div class="scroll"><table><thead><tr><th>FLD N</th><th>Field</th><th>Relative L1</th><th>Mean absolute</th><th>Max absolute</th><th>Reference difference</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<section><h2>Profiles, errors and video</h2>{figures}{movie}</section>
<section><h2>Reproduction</h2><p><a href="manifest.json">Run manifest</a> · <a href="reference/manifest.json">Reference cache manifest</a>. The reference is cached using the source, input, adapter and physical-state hashes and reused only when they match.</p><ul>{''.join(links)}</ul></section>
</body></html>''', encoding="utf-8")


def validate_reference_case(case: Path, n: int, physical: dict):
    first, last = profile(case, 0, "reference", physical), profile(case, -1, "reference", physical)
    if len(first["x"]) != n or not math.isclose(first["time"], 0, abs_tol=1e-12):
        raise ValueError(f"reference N={n} has incorrect initial output")
    if not math.isclose(last["time"], T_FINAL, rel_tol=2e-7):
        raise ValueError(f"reference N={n} did not reach final time")
    if "Terminating on time limit" not in (case/"run.log").read_text(errors="replace"):
        raise ValueError(f"reference N={n} did not terminate on time limit")
    return first, last


def ensure_reference(args, root: Path, repo: Path, physical: dict, manifest: dict) -> None:
    ref_repo = args.reference_repo.expanduser().resolve()
    ref_template = ref_repo/"inputs/radiation/athinput.shock_tube_hydro"
    fingerprint = reference_fingerprint(ref_repo, ref_template, physical)
    key = fingerprint_key(fingerprint)
    cache = args.reference_cache_root.expanduser().resolve()/key
    if cache.is_dir():
        cached = json.loads((cache/"manifest.json").read_text())
        if cached.get("fingerprint") != fingerprint or cached.get("status") != "passed":
            raise ValueError(f"reference cache fingerprint/status mismatch: {cache}")
        if (tree_sha256(cache/"build_source"/"src") != cached["source_sha256"]
                or sha256_file(cache/"build_source"/"bin"/"athena") != cached["binary_sha256"]):
            raise ValueError(f"reference cache source/binary hash mismatch: {cache}")
        manifest["reference_cache"] = {"key": key, "path": str(cache), "reused": True}
    else:
        cache.mkdir(parents=True)
        source = cache/"build_source"
        if args.import_reference_root:
            imported = args.import_reference_root.expanduser().resolve()
            shutil.copytree(imported/"build_source", source)
            verify_source_snapshot(ref_repo, source, adapted_reference=True)
            for n in REF_N:
                shutil.copytree(imported/f"case{n}", cache/"cases"/f"r{n}")
            origin = str(imported)
        else:
            prepare_build_source(ref_repo, source)
            adapt_reference_source(source)
            cfg = ["python3", str(source/"configure.py"), "--prob=shock_tube_hydro",
                   "--eos=general/rad", "--flux=hllc", "-hdf5", "-mpi", "--cflag=-O3"]
            records = manifest["commands"].setdefault("records", [])
            records.append(checked(cfg, source, cache/"configure.log", args.build_timeout, "reference_configure"))
            records.append(checked(["make", f"-j{args.max_cores}"], source,
                                   cache/"build.log", args.build_timeout, "reference_build"))
            origin = "new_calculation"
        exe = source/"bin/athena"
        config = binary_configuration(exe, cache, 30, manifest["commands"].setdefault("records", []))
        for field, expected in (("problem_generator", "shock_tube_hydro"),
                                ("equation_of_state", "general/rad"),
                                ("riemann_solver", "hllc"), ("hdf5_output", "ON")):
            if config.get(field) != expected:
                raise ValueError(f"reference binary {field} mismatch: {config.get(field)}")
        cache_manifest = {"status": "running", "fingerprint": fingerprint,
                          "origin": origin, "source_sha256": tree_sha256(source/"src"),
                          "binary_sha256": sha256_file(exe), "binary_configuration": config,
                          "cases": {}}
        for n in REF_N:
            case = cache/"cases"/f"r{n}"
            case.mkdir(parents=True, exist_ok=True)
            output = case/"output"
            output.mkdir(exist_ok=True)
            input_path = case/"input.used"
            overrides = reference_overrides(n, physical)
            if args.import_reference_root:
                assert_effective_input(input_path, ref_template, overrides)
            else:
                materialize_input(ref_template, input_path, overrides)
                command = ["/usr/bin/time", "-v", "-o", str(case/"resources.txt"),
                           "mpirun", "-np", str(min(args.max_cores, n//512)),
                           str(exe), "-i", str(input_path), "-d", str(output)]
                checked(command, case, case/"run.log", args.run_timeout, f"reference_{n}")
            validate_reference_case(case, n, physical)
            case_manifest = {"status": "passed", "resolution": n,
                             "input_sha256": sha256_file(input_path),
                             "final_prim_sha256": sha256_file(pair_paths(case,-1)[0]),
                             "final_uov_sha256": sha256_file(pair_paths(case,-1)[1]),
                             "max_rss_kib": runtime_resources(case/"resources.txt")}
            cache_manifest["cases"][f"r{n}"] = case_manifest
            write_manifest(case/"manifest.json", case_manifest)
        cache_manifest["status"] = "passed"
        write_manifest(cache/"manifest.json", cache_manifest)
        manifest["reference_cache"] = {"key": key, "path": str(cache), "reused": False}
    for n in REF_N:
        case = cache/"cases"/f"r{n}"
        assert_effective_input(case/"input.used", ref_template, reference_overrides(n, physical))
        validate_reference_case(case, n, physical)
        item = json.loads((case/"manifest.json").read_text())
        if (item["final_prim_sha256"] != sha256_file(pair_paths(case,-1)[0])
                or item["final_uov_sha256"] != sha256_file(pair_paths(case,-1)[1])):
            raise ValueError(f"reference cache HDF5 hash mismatch: {case}")
    shutil.copytree(cache, root/"reference", ignore=shutil.ignore_patterns("*.o"))


def ensure_fld(args, root: Path, repo: Path, physical: dict, manifest: dict):
    build = root/"fld"/"build"
    build.mkdir(parents=True)
    source = build/"build_source"
    if args.import_fld_root:
        imported = args.import_fld_root.expanduser().resolve()
        shutil.copytree(imported/"build_source", source)
        verify_source_snapshot(repo, source, adapted_reference=False)
        manifest["fld_imported_from"] = str(imported)
    else:
        prepare_build_source(repo, source)
        cfg = ["python3", str(source/"configure.py"), "--prob=nrfld_shock_tube",
               "--eos=adiabatic", "-nrmgfld", "-hdf5", "-mpi", "--cflag=-O3"]
        records = manifest["commands"].setdefault("records", [])
        records.append(checked(cfg, source, build/"configure.log", args.build_timeout, "fld_configure"))
        records.append(checked(["make", f"-j{args.max_cores}"], source,
                               build/"build.log", args.build_timeout, "fld_build"))
    exe = source/"bin/athena"
    config = binary_configuration(exe, build, 30, manifest["commands"].setdefault("records", []))
    for field, expected in (("problem_generator", "nrfld_shock_tube"),
                            ("equation_of_state", "adiabatic"),
                            ("fld_with_newton-raphson", "ON"), ("hdf5_output", "ON")):
        if config.get(field) != expected:
            raise ValueError(f"FLD binary {field} mismatch: {config.get(field)}")
    manifest["fld_build"] = {"source_sha256": tree_sha256(source/"src"),
                             "binary_sha256": sha256_file(exe), "binary_configuration": config}
    template = repo/"inputs/radiation/athinput.nrfld_shock_tube"
    for n in FLD_N:
        key = f"n{n}"
        case = root/"fld"/"cases"/key
        if args.import_fld_root:
            shutil.copytree(args.import_fld_root.expanduser().resolve()/f"case{n}_full", case)
            assert_effective_input(case/"input.used", template, fld_overrides(n, args.strict_backward_error))
        else:
            case.mkdir(parents=True)
            output = case/"output"
            output.mkdir()
            materialize_input(template, case/"input.used", fld_overrides(n, args.strict_backward_error))
            command = ["/usr/bin/time", "-v", "-o", str(case/"resources.txt"),
                       "mpirun", "-np", str(min(args.max_cores,n//32)),
                       str(exe), "-i", str(case/"input.used"), "-d", str(output)]
            checked(command, case, case/"run.log", args.run_timeout, f"fld_{n}")
        last = profile(case, -1, "fld", physical)
        if len(last["x"]) != n or not math.isclose(last["time"], T_FINAL, rel_tol=2e-7):
            raise ValueError(f"FLD N={n} did not reach the required final state")
        if "Terminating on time limit" not in (case/"run.log").read_text(errors="replace"):
            raise ValueError(f"FLD N={n} did not terminate on time limit")
        solver = parse_solver_log(case/"run.log", last["cycles"])
        if not solver["good"] or (args.strict_backward_error and solver["relaxed_accepted"]):
            raise ValueError(f"FLD N={n} solver did not satisfy recorded limits: {solver}")
        write_manifest(case/"manifest.json", {"status": "passed" if args.strict_backward_error else "completed_with_solver_caveat",
                        "input_sha256": sha256_file(case/"input.used"),
                        "final_prim_sha256": sha256_file(pair_paths(case,-1)[0]),
                        "final_uov_sha256": sha256_file(pair_paths(case,-1)[1]),
                        "solver": solver, "max_rss_kib": runtime_resources(case/"resources.txt")})


def postprocess(root: Path, physical: dict, manifest: dict, summary: dict) -> int:
    series = {}
    summary["time_schedule"] = {}
    for kind, ns, directory, prefix in (("reference", REF_N, "reference", "r"),
                                         ("fld", FLD_N, "fld", "n")):
        for n in ns:
            case = root/directory/"cases"/f"{prefix}{n}"
            series[f"{prefix}{n}"] = [profile(case, i, kind, physical) for i in range(21)]
            offsets = [abs(p["time"]-i*OUTPUT_DT)
                       for i,p in enumerate(series[f"{prefix}{n}"])]
            summary["time_schedule"][f"{prefix}{n}"] = {"max_offset": max(offsets)}
            if max(offsets) > 2e-4:
                raise ValueError(f"snapshot schedule mismatch: {case}")
    summary["max_reference_fld_time_offset"] = max(
        abs(q["time"]-r["time"]) for n in FLD_N
        for q,r in zip(series[f"n{n}"],series["r4096"]))
    reference = series["r4096"][-1]
    summary["reference_uncertainty"] = {field: metrics(coarsen(series["r2048"][-1],128)[field],
                                                       coarsen(reference,128)[field])
                                        for field in FIELDS}
    summary["comparison"] = {}
    summary["solver"] = {}
    for n in FLD_N:
        key = f"n{n}"
        case = root/"fld"/"cases"/key
        initial, final = series[key][0], series[key][-1]
        ref_initial = coarsen(series["r4096"][0], n)
        ref_final = coarsen(reference, n)
        for field in FIELDS:
            comparison = metrics(initial[field], ref_initial[field])
            if comparison["relative_l1"] is not None and comparison["relative_l1"] > 2e-5:
                raise ValueError(f"initial state mismatch at N={n}, {field}: {comparison}")
        summary["comparison"][key] = {field: metrics(final[field], ref_final[field])
                                      for field in FIELDS}
        summary["solver"][key] = parse_solver_log(case/"run.log", final["cycles"])
        if not summary["solver"][key]["good"]:
            raise ValueError(f"FLD solver check failed in {key}")
        save_csv(case/"profile_final.csv", final, ref_final)
    make_plots(root, series, summary)
    summary["video"] = make_video(root, series, manifest["commands"].setdefault("records", []))
    summary["status"] = ("completed_with_solver_caveat" if any(
        value["relaxed_accepted"] for value in summary["solver"].values()) else "passed")
    resource_path = root/"suite_resource.json"
    if resource_path.is_file():
        resource = json.loads(resource_path.read_text())
        summary["resource_monitor"] = {
            "peak_process_tree_rss_kib": resource["peak_process_tree_rss_kib"],
            "peak_athena_rss_sum_kib": resource["peak_athena_rss_sum_kib"],
            "mpi_rank_hwm_kib": {rank: item["hwm_kib"] for rank, item
                                 in resource["mpi_ranks"].items()},
            "oom_classification": resource["oom_classification"],
            "cgroup_memory_event_delta": resource["cgroup_memory_event_delta"],
            "resource_sha256": sha256_file(resource_path)}
    summary["finished_at"] = utc_now()
    manifest["status"] = summary["status"]
    manifest["finished_at"] = summary["finished_at"]
    manifest["analysis_script_sha256"] = sha256_file(Path(__file__))
    manifest["reference_adapter_sha256"] = sha256_file(HERE/"shock_tube_reference_adapter.py")
    manifest["metrics"] = {"solver": summary["solver"], "comparison": summary["comparison"],
                           "reference_uncertainty": summary["reference_uncertainty"],
                           "resource_monitor": summary.get("resource_monitor")}
    write_manifest(root/"manifest.json", manifest)
    (root/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    write_report(root, summary)
    print(root, flush=True)
    return 0


def main() -> int:
    args = parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    repo = find_repo_root(HERE)
    if args.postprocess:
        root = args.postprocess.expanduser().resolve()
        manifest = json.loads((root/"manifest.json").read_text())
        summary = json.loads((root/"summary.json").read_text())
        physical = summary["physical"]
        return postprocess(root, physical, manifest, summary)
    name = args.run_name or f"validation_shock_tube_{datetime.now(timezone(timedelta(hours=9))):%Y%m%d_%H%M%S}_JST"
    root = args.output_root.expanduser().resolve()/name
    root.mkdir(parents=True, exist_ok=False)
    physical = shock_settings(repo/"inputs/radiation/athinput.nrfld_shock_tube")
    manifest = base_manifest(repo, "shock_tube_validation", args.max_cores)
    summary = {"run": name, "status": "running", "started_at": utc_now(),
               "strict_backward_error": args.strict_backward_error,
               "physical": physical, "reference": REF_N, "fld": FLD_N}
    (root/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    write_manifest(root/"manifest.json", manifest)
    write_report(root, summary)
    try:
        ensure_reference(args, root, repo, physical, manifest)
        write_manifest(root/"manifest.json", manifest)
        print(f"Reference ready: {manifest['reference_cache']}", flush=True)
        ensure_fld(args, root, repo, physical, manifest)
        print("FLD cases ready", flush=True)
        return postprocess(root, physical, manifest, summary)
    except Exception as error:
        summary.update(status="failed", error=str(error), finished_at=utc_now())
        manifest.update(status="failed", error=str(error), finished_at=summary["finished_at"])
        (root/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        write_manifest(root/"manifest.json", manifest)
        print(f"Shock tube validation failed: {error}\n{root}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
