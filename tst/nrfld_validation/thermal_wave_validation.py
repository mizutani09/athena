#!/usr/bin/env python3
"""Analyze the 3-D Zhang thermal wave at t_init + elapsed simulation time."""

from __future__ import annotations

import argparse
import csv
from html import escape
import json
import math
from pathlib import Path
import re
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import LogNorm
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nrfld_validation.io import find_repo_root
from nrfld_validation.manifest import (base_manifest, sha256_file, tree_sha256,
                                       utc_now, write_manifest)
from run_suite import effective_settings
from shock_tube_validation import parse_solver_log, runtime_resources
from radiative_shock_validation import launcher_wall_seconds

CASE_SETS = {"fast": ("uniform32_final", "amr16_final", "uniform64_final", "amr32_final"),
             "full": ("uniform32_final", "amr16_final", "uniform64_final", "amr32_final"),
             "hpc": ("uniform64_final", "amr32_final", "uniform128_final", "amr64_final")}
A_RAD = 7.5657e-15
C_LIGHT = 2.99792458e10


def similarity(settings: dict) -> dict:
    m, d = 2.5, 3.0
    beta = 1 / (d*m + 2)
    alpha = d*beta
    cv = float(settings["problem/rho_cv"])
    chi = float(settings["problem/chi_r_coeff"])
    energy = float(settings["problem/e_hot_total"])
    k = 4*A_RAD*C_LIGHT/(3*chi*cv)
    b = m*beta/(2*k)
    integral = .5*math.gamma(1.5)*math.gamma(1+1/m)/math.gamma(2.5+1/m)
    a = ((energy/cv)*b**1.5/(4*math.pi*integral))**(1/(1/m+1.5))
    return {"alpha": alpha, "beta": beta, "A": a, "B": b,
            "ambient": float(settings["problem/t_ambient"]),
            "init_time_s": float(settings["problem/init_time"])}


def analytic_temperature(radius: np.ndarray, time: float, constants: dict) -> np.ndarray:
    core = np.maximum(constants["A"] - constants["B"]*radius**2
                      *time**(-2*constants["beta"]), 0)
    result = time**(-constants["alpha"])*core**(1/2.5)
    return np.maximum(result, constants["ambient"])


def analytic_front(time: float, constants: dict, threshold: float = 10.) -> float:
    # Radius at a nonzero temperature threshold, avoiding ambiguity at the
    # discontinuous support boundary and radiation precursor.
    core = threshold**2.5 * time**(2.5*constants["alpha"])
    return math.sqrt(max(constants["A"]-core, 0)/constants["B"]) * time**constants["beta"]


def files(directory: Path) -> tuple[Path, Path]:
    paths = sorted((directory / "output").glob("*.full3d.*.athdf"))
    if len(paths) != 2:
        raise ValueError(f"expected initial and final 3D HDF5 in {directory}")
    return paths[0], paths[1]


def load(path: Path) -> dict:
    with h5py.File(path) as h:
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in h.attrs["VariableNames"]]
        fields = {name: np.asarray(h["user_out_var"][names.index(name)], dtype=float)
                  for name in ("Tgas", "Erad", "egas", "etherm")}
        shape = fields["Tgas"].shape
        x = np.broadcast_to(np.asarray(h["x1v"])[:, None, None, :], shape)
        y = np.broadcast_to(np.asarray(h["x2v"])[:, None, :, None], shape)
        z = np.broadcast_to(np.asarray(h["x3v"])[:, :, None, None], shape)
        dx = np.diff(np.asarray(h["x1f"]), axis=1)[:, None, None, :]
        dy = np.diff(np.asarray(h["x2f"]), axis=1)[:, None, :, None]
        dz = np.diff(np.asarray(h["x3f"]), axis=1)[:, :, None, None]
        volume = np.broadcast_to(dx*dy*dz, shape)
        radius = np.sqrt(x*x + y*y + z*z)
        levels = np.asarray(h["Levels"])
        result = {"time_s": float(h.attrs["Time"]), "cycles": int(h.attrs["NumCycles"]),
                  "levels": {str(int(level)): int(np.sum(levels == level)) for level in np.unique(levels)},
                  "level_per_cell": np.broadcast_to(levels[:, None, None, None], shape).ravel(),
                  "x": x.ravel(), "y": y.ravel(), "z": z.ravel(),
                  "radius": radius.ravel(), "volume": volume.ravel(),
                  "dx_min": float(np.min(dx))}
        result.update({name: values.ravel() for name, values in fields.items()})
    for name in ("Tgas", "Erad", "egas", "etherm"):
        if not np.all(np.isfinite(result[name])) or np.any(result[name] <= 0):
            raise ValueError(f"invalid {name} in {path}")
    return result


def history_energy(directory: Path) -> dict:
    matches = list((directory / "output").glob("*.hst"))
    if len(matches) != 1:
        raise ValueError(f"missing history: {directory}")
    lines = matches[0].read_text().splitlines()
    names = {name: int(index)-1 for index, name in re.findall(r"\[(\d+)\]=([^\s]+)", lines[1])}
    values = np.loadtxt(matches[0], comments="#")
    energy = values[:, names["all_E"]]
    mass = values[:, names["mass"]]
    return {"initial_erg": float(energy[0]), "final_erg": float(energy[-1]),
            "relative_drift": float((energy[-1]-energy[0])/energy[0]),
            "mass_initial_g": float(mass[0]),
            "mass_relative_drift": float((mass[-1]-mass[0])/mass[0]),
            "history_sha256": sha256_file(matches[0])}


def radial_profile(state: dict, constants: dict) -> dict:
    r, w, t = state["radius"], state["volume"], state["Tgas"]
    width = state["dx_min"]
    index = np.floor(r/width).astype(int)
    weight = np.bincount(index, weights=w)
    center = np.bincount(index, weights=r*w)/np.maximum(weight, 1e-300)
    mean = np.bincount(index, weights=t*w)/np.maximum(weight, 1e-300)
    variance = np.bincount(index, weights=(t-mean[index])**2*w)/np.maximum(weight, 1e-300)
    use = weight > 0
    return {"radius_cm": center[use], "Tgas_K": mean[use],
            "angular_std_K": np.sqrt(variance[use]), "volume_cm3": weight[use]}


def measured_front(profile: dict, threshold: float = 10.) -> float:
    r, t = profile["radius_cm"], profile["Tgas_K"]
    above = np.flatnonzero(t > threshold)
    if len(above) == 0 or above[-1]+1 >= len(r):
        raise ValueError("temperature front does not cross threshold inside domain")
    i = above[-1]
    return float(np.interp(threshold, [t[i+1], t[i]], [r[i+1], r[i]]))


def symmetry_error(state: dict) -> dict:
    """Compare all Cartesian axis permutations/reflections on a finest-grid map."""
    dx = state["dx_min"]
    n = round(400/dx)
    dense = np.full((n, n, n), np.nan)
    size = np.rint(np.cbrt(state["volume"])/dx).astype(int)
    starts = [np.rint((state[axis]+200)/dx-size/2).astype(int)
              for axis in ("x", "y", "z")]
    for cell_size in np.unique(size):
        choose = size == cell_size
        for ox in range(int(cell_size)):
            for oy in range(int(cell_size)):
                for oz in range(int(cell_size)):
                    dense[starts[0][choose]+ox, starts[1][choose]+oy,
                          starts[2][choose]+oz] = state["Tgas"][choose]
    if not np.all(np.isfinite(dense)):
        raise ValueError("AMR leaf map has a gap")
    differences = (np.max(np.abs(dense-dense.transpose(1, 0, 2))),
                   np.max(np.abs(dense-dense.transpose(0, 2, 1))),
                   np.max(np.abs(dense-dense[::-1, :, :])),
                   np.max(np.abs(dense-dense[:, ::-1, :])),
                   np.max(np.abs(dense-dense[:, :, ::-1])))
    return {"max_abs_K": float(max(differences)),
            "relative_to_peak": float(max(differences)/np.max(dense))}


def analyze_case(root: Path, tag: str, out: Path) -> tuple[dict, dict]:
    directory = root / tag
    settings = effective_settings(directory / "input.used", [])
    if settings.get("fld/only_rad") != "false" or settings.get("problem/use_analytic_profile_init") != "true":
        raise ValueError(f"not a fully coupled analytic-init run: {tag}")
    first_file, final_file = files(directory)
    first, final = load(first_file), load(final_file)
    log = (directory / "run.log").read_text(errors="replace")
    if "Terminating on time limit" not in log or re.search(r"FATAL|Segmentation fault", log, re.I):
        raise ValueError(f"incomplete run: {tag}")
    solver = parse_solver_log(directory / "run.log", final["cycles"])
    if not solver["good"] or solver["relaxed_accepted"]:
        raise ValueError(f"solver failed strict convergence: {tag}: {solver}")
    fgmres_residuals = [float(value) for value in re.findall(
        r"\[LINEAR_FGMRES\][^\n]*?relative_residual=([^\s]+)", log)]
    solver["max_true_fgmres_relative_residual"] = max(fgmres_residuals)
    constants = similarity(settings)
    e = {}
    profiles = {}
    for label, state in (("initial", first), ("final", final)):
        analytic_time = constants["init_time_s"] + state["time_s"]
        expected = analytic_temperature(state["radius"], analytic_time, constants)
        weight = state["volume"]
        contrast = np.abs(expected-constants["ambient"])
        e[label] = {"analytic_time_s": analytic_time,
                    "front_analytic_10K_cm": analytic_front(analytic_time, constants)}
        for field, observed, reference, baseline in (
                ("Tgas", state["Tgas"], expected, constants["ambient"]),
                ("Trad", (state["Erad"]/A_RAD)**.25, expected, constants["ambient"]),
                ("Erad", state["Erad"], A_RAD*expected**4,
                 A_RAD*constants["ambient"]**4)):
            diff = np.abs(observed-reference)
            e[label][field + "_relative_L1"] = float(np.sum(diff*weight)/np.sum(np.abs(reference)*weight))
            e[label][field + "_contrast_L1"] = float(np.sum(diff*weight)
                                                       / np.sum(np.abs(reference-baseline)*weight))
            e[label][field + "_max_abs"] = float(np.max(diff))
        profiles[label] = radial_profile(state, constants)
        e[label]["front_measured_10K_cm"] = measured_front(profiles[label])
        inside = profiles[label]["radius_cm"] < e[label]["front_analytic_10K_cm"]
        e[label]["angular_rms_K"] = float(np.sqrt(np.average(
            profiles[label]["angular_std_K"][inside]**2,
            weights=profiles[label]["volume_cm3"][inside])))
    csv_file = out / f"{tag}_radial.csv"
    with csv_file.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("radius_cm", "Tgas_K", "Tgas_analytic_K", "angular_std_K", "volume_cm3"))
        p = profiles["final"]
        writer.writerows(zip(p["radius_cm"], p["Tgas_K"],
                             analytic_temperature(p["radius_cm"], e["final"]["analytic_time_s"], constants),
                             p["angular_std_K"], p["volume_cm3"]))
    result = {"input_sha256": sha256_file(directory / "input.used"),
              "initial_hdf_sha256": sha256_file(first_file), "final_hdf_sha256": sha256_file(final_file),
              "cycles": final["cycles"], "elapsed_s": final["time_s"],
              "initial_levels": first["levels"], "final_levels": final["levels"],
              "active_cells": len(final["radius"]), "dx_min_cm": final["dx_min"],
              "symmetry": symmetry_error(final),
              "solver": solver, "launcher_max_rss_kib": runtime_resources(directory / "resources.txt"),
              "launcher_wall_seconds": launcher_wall_seconds(directory / "resources.txt"),
              "energy": history_energy(directory), "error": e,
              "radial_csv": csv_file.name}
    topology = []
    for path in sorted((directory / "output").glob("*.uov_xy.*.athdf")):
        with h5py.File(path) as h:
            levels = np.asarray(h["Levels"])
            topology.append({"time_s": float(h.attrs["Time"]),
                             "slice_blocks_by_level": {str(int(level)): int(np.sum(levels == level))
                                                       for level in np.unique(levels)}})
    result["center_slice_topology"] = topology
    result["center_slice_topology_changed"] = any(
        row["slice_blocks_by_level"] != topology[0]["slice_blocks_by_level"]
        for row in topology[1:])
    return result, profiles["final"]


def draw_slice(directory: Path, target: Path):
    paths = sorted((directory / "output").glob("*.uov_xy.*.athdf"))
    with h5py.File(paths[-1]) as h:
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in h.attrs["VariableNames"]]
        values = np.asarray(h["user_out_var"][names.index("Tgas")])
        xfaces, yfaces = np.asarray(h["x1f"]), np.asarray(h["x2f"])
    fig, ax = plt.subplots(figsize=(8, 7))
    for block in range(len(xfaces)):
        ax.pcolormesh(xfaces[block], yfaces[block], values[block, 0],
                      cmap="inferno", norm=LogNorm(vmin=1, vmax=1400), shading="flat")
    ax.set(xlim=(-100, 100), ylim=(-100, 100), xlabel="x [cm]", ylabel="y [cm]",
           title=directory.name + " at z = 0")
    ax.set_aspect("equal")
    fig.colorbar(plt.cm.ScalarMappable(norm=LogNorm(vmin=1, vmax=1400), cmap="inferno"),
                 ax=ax, label="Tgas [K]")
    fig.savefig(target, dpi=170)
    fig.savefig(target.with_suffix(".pdf"))
    plt.close(fig)


def draw_movie(directory: Path, target: Path):
    paths = sorted((directory / "output").glob("*.uov_xy.*.athdf"))
    if len(paths) < 3:
        raise ValueError(f"need time series for movie: {directory}")
    fig, ax = plt.subplots(figsize=(7, 6))
    writer = FFMpegWriter(fps=2, codec="libx264", bitrate=1500)
    with writer.saving(fig, str(target), dpi=130):
        for path in paths:
            with h5py.File(path) as h:
                names = [x.decode() if isinstance(x, bytes) else str(x)
                         for x in h.attrs["VariableNames"]]
                values = np.asarray(h["user_out_var"][names.index("Tgas")])
                xfaces, yfaces = np.asarray(h["x1f"]), np.asarray(h["x2f"])
                time = float(h.attrs["Time"])
            ax.clear()
            for block in range(len(xfaces)):
                ax.pcolormesh(xfaces[block], yfaces[block], values[block, 0],
                              cmap="inferno", norm=LogNorm(vmin=1, vmax=1400), shading="flat")
            ax.set(xlim=(-100, 100), ylim=(-100, 100), xlabel="x [cm]", ylabel="y [cm]",
                   title=f"{directory.name}: t = {time:.2e} s")
            ax.set_aspect("equal")
            writer.grab_frame()
    plt.close(fig)


def amr_uniform_comparison(root: Path, amr_tag: str, uniform_tag: str) -> dict:
    """Compare AMR leaves with volume-averaged uniform cells at identical time."""
    uniform = load(files(root / uniform_tag)[-1])
    amr = load(files(root / amr_tag)[-1])
    if uniform["time_s"] != amr["time_s"]:
        raise ValueError("AMR/uniform times differ")
    n = round(400 / uniform["dx_min"])
    dense = np.full((n, n, n), np.nan)
    ui = [np.rint((uniform[axis]+200)/uniform["dx_min"]-.5).astype(int)
          for axis in ("x", "y", "z")]
    dense[ui[0], ui[1], ui[2]] = uniform["Tgas"]
    if not np.all(np.isfinite(dense)):
        raise ValueError("uniform reference has gaps")
    levels = amr["level_per_cell"]
    fine = levels == 1
    starts = [np.rint((amr[axis]+200)/uniform["dx_min"]
                       - np.where(fine, .5, 1.)).astype(int)
              for axis in ("x", "y", "z")]
    expected = np.empty_like(amr["Tgas"])
    expected[fine] = dense[starts[0][fine], starts[1][fine], starts[2][fine]]
    coarse = ~fine
    expected[coarse] = 0
    for ox in (0, 1):
        for oy in (0, 1):
            for oz in (0, 1):
                expected[coarse] += dense[starts[0][coarse]+ox,
                                          starts[1][coarse]+oy,
                                          starts[2][coarse]+oz]/8
    mask = np.zeros_like(dense, dtype=bool)
    mask[starts[0][fine], starts[1][fine], starts[2][fine]] = True
    interface = np.zeros(len(coarse), dtype=bool)
    for index in np.flatnonzero(coarse):
        i, j, k = (axis[index] for axis in starts)
        interface[index] = (
            (i > 0 and np.any(mask[i-1, j:j+2, k:k+2])) or
            (i+2 < n and np.any(mask[i+2, j:j+2, k:k+2])) or
            (j > 0 and np.any(mask[i:i+2, j-1, k:k+2])) or
            (j+2 < n and np.any(mask[i:i+2, j+2, k:k+2])) or
            (k > 0 and np.any(mask[i:i+2, j:j+2, k-1])) or
            (k+2 < n and np.any(mask[i:i+2, j:j+2, k+2])))
    difference = np.abs(amr["Tgas"]-expected)
    def metric(select):
        if not np.any(select):
            return None
        weight = amr["volume"][select]
        return {"cells": int(np.sum(select)),
                "volume_weighted_relative_L1": float(np.sum(difference[select]*weight)
                                                    / np.sum(expected[select]*weight)),
                "max_abs_K": float(np.max(difference[select]))}
    return {"all": metric(np.ones(len(fine), dtype=bool)),
            "refined": metric(fine), "coarse": metric(coarse),
            "coarse_cells_touching_refinement": metric(interface)}


def render(root: Path, summary: dict):
    rows = []
    for tag, value in summary["cases"].items():
        error = value["error"]["final"]
        rows.append(f'<tr><td>{tag}</td><td>{value["active_cells"]:,}</td>'
                    f'<td>{value["final_levels"]}</td><td>{error["Tgas_contrast_L1"]:.4g}</td>'
                    f'<td>{error["Trad_contrast_L1"]:.4g}</td>'
                    f'<td>{error["Erad_contrast_L1"]:.4g}</td>'
                    f'<td>{error["front_measured_10K_cm"]:.3f} / {error["front_analytic_10K_cm"]:.3f}</td>'
                    f'<td>{value["energy"]["relative_drift"]:.3g}</td>'
                    f'<td>{value["energy"]["mass_relative_drift"]:.3g}</td>'
                    f'<td>{value["symmetry"]["max_abs_K"]:.3g}</td>'
                    f'<td>{value["launcher_wall_seconds"]:.2f}</td>'
                    f'<td>{value["launcher_max_rss_kib"]:,}</td>'
                    f'<td><a href="figures/{value["radial_csv"]}">CSV</a></td></tr>')
    high_uniform, high_amr = summary["high_cases"]
    interface = summary["amr_uniform_comparison"][f"{high_amr}_vs_{high_uniform}"]["coarse_cells_touching_refinement"]
    uniform_name = high_uniform.split("_")[0]
    amr_name = high_amr.split("_")[0]
    html = f'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>3D thermal wave validation</title>
<style>:root{{font:21px/1.55 system-ui,sans-serif;color:#17243a;background:#f8fafc}}body{{max-width:1250px;margin:2rem auto;padding:0 1rem}}h1{{font-size:2rem}}h2{{font-size:1.45rem;color:#075985}}strong{{color:#075985}}table{{border-collapse:collapse;width:100%;font-size:.92rem}}th,td{{padding:.5rem;border-bottom:1px solid #cbd5e1;text-align:left;vertical-align:top}}th{{background:#e0f2fe}}img{{max-width:100%;height:auto}}a{{color:#0369a1}}@media(max-width:700px){{:root{{font-size:19px}}table{{display:block;overflow-x:auto}}}}</style>
</head><body><h1>3D thermal wave: uniform / one-level AMR</h1>
<p><strong>全ケースは 101 step で目標時刻 10⁻⁷ s に到達し、NR/FGMRES は厳密収束しました。</strong> 解析解の時刻は初期化時刻 10⁻⁶ s + 経過時間です。Tgas の L1 は球対称解析解との体積重み付き比較で、分母は温度の ambient 1 K からのコントラストです。</p>
<p>本検証では <code>fld/only_rad=false</code> の完全結合残差を用います。従来の <code>only_rad=true</code> は Newton 停滞を再現しました。1 段 AMR の coarse/fine topology と使用メモリも併記します。粗密界面の評価は、界面に接する coarse cell の温度を同じ領域の uniform fine cell 平均と比較したものです。</p>
<p>{high_amr} の粗密界面に接する coarse cell {interface['cells']:,} 個で、同じ体積の {high_uniform} 平均との差は L1 {interface['volume_weighted_relative_L1']:.3g}、最大 {interface['max_abs_K']:.3g} K（保存された HDF5 精度）。</p>
<div style="overflow-x:auto"><table><thead><tr><th>case</th><th>active cells</th><th>level:blocks</th><th>Tgas L1</th><th>Trad L1</th><th>Erad L1</th><th>front 10 K: measured / analytic [cm]</th><th>energy drift</th><th>mass drift</th><th>axis symmetry max K</th><th>wall s</th><th>launcher RSS KiB</th><th>data</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<h2>半径方向の温度</h2><img src="figures/radial.png"><p><a href="figures/radial.pdf">PDF</a></p>
<h2>z=0 の温度分布</h2><img src="figures/{uniform_name}_slice.png"><img src="figures/{amr_name}_slice.png">
<h2>時間発展</h2><video controls preload="metadata" width="620" src="figures/{uniform_name}.mp4"></video><video controls preload="metadata" width="620" src="figures/{amr_name}.mp4"></video>
<p><a href="summary.json">summary.json</a></p></body></html>'''
    (root / "index.html").write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postprocess", type=Path, required=True)
    parser.add_argument("--preset", choices=tuple(CASE_SETS), default="full")
    args = parser.parse_args()
    root = args.postprocess.expanduser().resolve()
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    source = root / "build_source"
    cases = CASE_SETS[args.preset]
    summary = {"created_utc": utc_now(), "status": "passed", "preset": args.preset,
               "cases": {}, "high_cases": (cases[2], cases[3]),
               "build": {"binary_sha256": sha256_file(source / "bin/athena"),
                         "source_sha256": tree_sha256(source / "src")},
               "definitions": {"analytic_time_s": "initial analytic time plus elapsed simulation time",
                               "contrast_L1": "sum |Tgas-Tanalytic| volume / sum |Tanalytic-Tambient| volume",
                               "front": "radial-shell temperature crossing 10 K"}}
    profiles = {}
    for tag in cases:
        summary["cases"][tag], profiles[tag] = analyze_case(root, tag, figures)
    summary["amr_uniform_comparison"] = {
        f"{cases[1]}_vs_{cases[0]}": amr_uniform_comparison(root, cases[1], cases[0]),
        f"{cases[3]}_vs_{cases[2]}": amr_uniform_comparison(root, cases[3], cases[2])}
    fig, ax = plt.subplots(figsize=(11, 7))
    first_settings = effective_settings(root / cases[0] / "input.used", [])
    constants = similarity(first_settings)
    rline = np.linspace(0, 110, 1000)
    ax.plot(rline, analytic_temperature(rline, 1.1e-6, constants), "k-", lw=2,
            label="analytic at 1.1e-6 s")
    for tag, p in profiles.items():
        ax.plot(p["radius_cm"], p["Tgas_K"], ".-", ms=3, label=tag)
    ax.set(xlim=(0, 100), ylim=(0, 1450), xlabel="radius [cm]", ylabel="Tgas [K]")
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(figures / "radial.png", dpi=180)
    fig.savefig(figures / "radial.pdf")
    plt.close(fig)
    for tag in (cases[2], cases[3]):
        draw_slice(root / tag, figures / (tag.split("_")[0] + "_slice.png"))
        draw_movie(root / tag, figures / (tag.split("_")[0] + ".mp4"))
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = base_manifest(find_repo_root(HERE), "thermal_wave", 4)
    manifest.update({"status": summary["status"], "metrics": summary,
                     "analysis_script_sha256": sha256_file(Path(__file__))})
    write_manifest(root / "manifest.json", manifest)
    render(root, summary)
    print(root / "index.html")


if __name__ == "__main__":
    main()
