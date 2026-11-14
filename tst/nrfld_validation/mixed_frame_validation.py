#!/usr/bin/env python3
"""Compare implicit and explicit placement of mixed-frame FLD drift terms.

The simulation directory contains immutable inputs, logs and HDF5 files. This
program only reads them and writes an index, plots and machine-readable metrics.
Use --postprocess after the case matrix described in README.md has completed.
"""

from __future__ import annotations

import argparse
import csv
from html import escape
import json
import math
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from advection_pulse_validation import (FINAL_TIME_CODE, align_moving, comparison, physical_time_unit,
                                        profile, outputs as pulse_outputs)
from diagnose_mach5_convergence import measure, reference_at
from nrfld_validation.io import find_repo_root
from nrfld_validation.manifest import (base_manifest, sha256_file, tree_sha256,
                                       utc_now, write_manifest)
from plots.plot_radiative_shock import (jump_center, load_analytic_profile,
                                        load_numerical_profile)
from radiative_shock_validation import files as shock_files, T_FINAL, launcher_wall_seconds
from run_suite import effective_settings
from shock_tube_validation import parse_solver_log, runtime_resources

SHOCK_REFERENCE = {2: "mach2_semianalytic_bvp.csv", 5: "mach5_semianalytic.csv"}
SHOCK_MATRIX = {"fast": ((64, .3),),
                "full": ((64, .3), (128, .3), (64, .15)),
                "hpc": ((256, .3), (512, .3), (256, .15))}
PULSE_MATRIX = {"fast": ((32, .03),),
                "full": ((64, .03), (64, .015), (32, .03)),
                "hpc": ((512, .03), (512, .015), (256, .03))}


def case_status(directory: Path, kind: str) -> dict:
    settings = effective_settings(directory / "input.used", [])
    paths = shock_files(directory) if kind == "shock" else pulse_outputs(directory)
    with __import__("h5py").File(paths[-1]) as data:
        cycles = int(data.attrs["NumCycles"])
        time = float(data.attrs["Time"])
    log = (directory / "run.log").read_text(errors="replace")
    if "Terminating on time limit" not in log or re.search(r"FATAL|Segmentation fault", log, re.I):
        raise ValueError(f"incomplete or fatal run: {directory}")
    solver = parse_solver_log(directory / "run.log", cycles)
    if not solver["good"] or solver["relaxed_accepted"]:
        raise ValueError(f"strict NR/FGMRES check failed: {directory}: {solver}")
    expected_switch = directory.name.endswith("implicit")
    if settings.get("fld/implicit_mixed_frame_drift", "true").lower() != str(expected_switch).lower():
        raise ValueError(f"mixed-frame switch/input mismatch: {directory}")
    solver["nr_iterations_total"] = sum(int(value) for value in re.findall(
        r"\[NR\] status=converged iterations=(\d+)", log))
    solver["fgmres_iterations_total"] = sum(int(value) for value in re.findall(
        r"\[LINEAR_FGMRES\] iterations=(\d+)", log))
    first_step = re.search(r"^cycle=0 time=[^\n]+? dt=([^\s]+)", log, re.M)
    return {"cycles": cycles, "time_code": time, "solver": solver,
            "first_dt_code": float(first_step[1]) if first_step else None,
            "launcher_max_rss_kib": runtime_resources(directory / "resources.txt"),
            "wall_seconds": launcher_wall_seconds(directory / "resources.txt"),
            "input_sha256": sha256_file(directory / "input.used"),
            "final_sha256": sha256_file(paths[-1]), "settings": settings,
            "final_file": str(paths[-1])}


def raw_shock_error(numerical: dict, analytic: dict) -> dict:
    x = numerical["x"]
    center = jump_center(analytic["x"], analytic["rho"])
    inside = (x >= analytic["x"][0]) & (x <= analytic["x"][-1])
    result = {}
    for field in ("rho", "tgas", "trad"):
        expected = reference_at(x, analytic, field, center)
        result[field] = float(100 * np.sum(np.abs(numerical[field][inside] - expected[inside]))
                              / np.sum(np.abs(expected[inside])))
    return result


def shock_analysis(root: Path, figures: Path, summary: dict, preset: str):
    for mach in (2, 5):
        reference_file = HERE / "reference/radiative_shock" / SHOCK_REFERENCE[mach]
        analytic = load_analytic_profile(reference_file)
        center = jump_center(analytic["x"], analytic["rho"])
        summary["reference"][f"mach{mach}"] = {"file": str(reference_file),
                                                  "sha256": sha256_file(reference_file)}
        fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
        for ax, field in zip(axes, ("rho", "tgas", "trad")):
            ax.plot(analytic["x"] - center, analytic[field], "k--", label="semi-analytic")
            ax.set_ylabel(field)
            ax.grid(alpha=.2)
        for n, cfl in SHOCK_MATRIX[preset]:
            for implicit in (True, False):
                tag = f"shock{mach}_n{n}_cfl{cfl:g}_{'implicit' if implicit else 'explicit'}"
                directory = root / tag
                if not directory.is_dir() and n == 64 and cfl == .3 and mach == 2:
                    directory = root / f"shock2_n64_{'implicit' if implicit else 'explicit'}"
                status = case_status(directory, "shock")
                settings = status["settings"]
                numerical = load_numerical_profile(shock_files(directory)[-1])
                if len(numerical["x"]) != n:
                    raise ValueError(f"shock output grid mismatch: {tag}")
                input_grid_matches = int(settings["mesh/nx1"]) == n
                input_cfl_matches = math.isclose(float(settings["time/cfl_number"]), cfl,
                                                 rel_tol=1e-12)
                if not input_cfl_matches:
                    baseline_tag = f"shock{mach}_n{n}_cfl0.3_{'implicit' if implicit else 'explicit'}"
                    baseline = summary["cases"].get(baseline_tag)
                    if (baseline is None or status["first_dt_code"] is None
                            or not math.isclose(status["first_dt_code"] / baseline["first_dt_code"],
                                                cfl/.3, rel_tol=.02)):
                        raise ValueError(f"shock CFL not confirmed by input or first-step dt: {tag}")
                status["grid_cfl_provenance"] = ("input_and_hdf" if input_grid_matches
                                                 and input_cfl_matches else
                                                 "hdf_and_first_step_dt_legacy_cli_overrides_not_recorded")
                if not math.isclose(numerical["time"], T_FINAL[f"mach{mach}"], rel_tol=1e-6):
                    raise ValueError(f"wrong shock final time: {tag}")
                aligned, arrays = measure(shock_files(directory)[-1], analytic,
                                          40. if mach == 2 else 60.,
                                          (-1000., 500.) if mach == 2 else (-4000., 2000.))
                status.update({"raw_L1_percent": raw_shock_error(numerical, analytic),
                               "aligned_L1_percent": {f: aligned["agreement"][f]["global_relative_L1_percent"]
                                                      for f in ("rho", "tgas", "trad")},
                               "aligned_max_relative_percent": {
                                   f: float(100 * np.max(np.abs(arrays[f] - arrays[f + "_reference"])
                                                        / np.abs(arrays[f + "_reference"])))
                                   for f in ("rho", "tgas", "trad")},
                               "shock_raw_cm": aligned["shock_raw_cm"],
                               "alignment_shift_cm": aligned["shock_alignment_shift_cm"],
                               "shock_dx_cm": aligned["shock_dx_cm"]})
                summary["cases"][tag] = status
                label = f"N{n} CFL {cfl:g} {'implicit' if implicit else 'explicit'}"
                for ax, field in zip(axes, ("rho", "tgas", "trad")):
                    ax.plot(arrays["x_from_shock_cm"], arrays[field], lw=1, label=label)
        for ax in axes:
            ax.set_xlim(-150 if mach == 2 else -300, 150 if mach == 2 else 300)
            ax.set_xlabel("x - measured shock [cm]")
            ax.legend(fontsize=8, ncol=2)
        fig.savefig(figures / f"shock_mach{mach}.png", dpi=170)
        plt.close(fig)


def pulse_analysis(root: Path, figures: Path, summary: dict, preset: str):
    for n, cfl in PULSE_MATRIX[preset]:
        base = f"pulse_n{n}_cfl{cfl:g}_"
        static_tag = base + "static_implicit"
        static_dir = root / static_tag
        static_status = case_status(static_dir, "pulse")
        static = profile(pulse_outputs(static_dir)[-1])
        if int(static_status["settings"]["mesh/nx1"]) != n or not math.isclose(
                float(static_status["settings"]["time/cfl_number"]), cfl, rel_tol=1e-12):
            raise ValueError(f"static grid/CFL input mismatch: {static_tag}")
        if not math.isclose(static["time_code"], FINAL_TIME_CODE, rel_tol=1e-6):
            raise ValueError(f"static pulse wrong final time: {static_tag}")
        summary["cases"][static_tag] = static_status
        fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
        for ax, field in zip(axes, ("Tgas", "Trad", "Erad")):
            ax.plot(static["x"], static[field], "k-", label="static")
            ax.set_ylabel(field)
            ax.grid(alpha=.2)
        for implicit in (True, False):
            tag = base + "moving_" + ("implicit" if implicit else "explicit")
            directory = root / tag
            status = case_status(directory, "pulse")
            moving = profile(pulse_outputs(directory)[-1])
            settings = status["settings"]
            if (int(settings["mesh/nx1"]) != n or not math.isclose(
                    float(settings["time/cfl_number"]), cfl, rel_tol=1e-12)
                    or not math.isclose(float(settings["problem/v0"]), 1e6, rel_tol=1e-12)):
                raise ValueError(f"moving pulse input mismatch: {tag}")
            if not math.isclose(moving["time_code"], static["time_code"], rel_tol=1e-8):
                raise ValueError(f"pulse final time mismatch: {tag}")
            shift = 1e6 * moving["time_code"] * physical_time_unit(status["settings"])
            shifted, method = align_moving(static, moving, shift)
            metrics, _ = comparison(static, shifted, 1e6)
            status.update({"shift_cm": shift, "shift_method": method, "comparison": metrics})
            summary["cases"][tag] = status
            label = "implicit" if implicit else "explicit"
            for ax, field in zip(axes, ("Tgas", "Trad", "Erad")):
                ax.plot(static["x"], shifted[field], label=label)
        for ax in axes:
            ax.set_xlabel("x [cm], moving profile shifted by v0 t")
            ax.legend()
        fig.savefig(figures / f"pulse_n{n}_cfl{cfl:g}.png", dpi=170)
        plt.close(fig)


def closed_box_analysis(root: Path, summary: dict):
    for implicit in (True, False):
        tag = "box_" + ("implicit" if implicit else "explicit")
        directory = root / tag
        settings = effective_settings(directory / "input.used", [])
        if (settings.get("fld/implicit_mixed_frame_drift", "true").lower()
                != str(implicit).lower() or
                not math.isclose(float(settings["problem/v0"]), 1e6, rel_tol=1e-12) or
                not math.isclose(float(settings["time/tlim"]), 1., rel_tol=1e-12)):
            raise ValueError(f"closed-box input mismatch: {tag}")
        # Closed-box runs use history output, not pulse HDF output names.
        text = (directory / "run.log").read_text(errors="replace")
        if "Terminating on time limit" not in text or re.search(r"FATAL|Segmentation fault", text, re.I):
            raise ValueError(f"closed-box run incomplete: {tag}")
        history = sorted((directory / "output").glob("*.hst"))
        if len(history) != 1:
            raise ValueError(f"closed-box history missing: {tag}")
        names = re.findall(r"\[(\d+)\]=([^\s]+)", history[0].read_text().splitlines()[1])
        data = np.loadtxt(history[0], comments="#")
        columns = {name: data[:, int(index)-1] for index, name in names}
        cycles = len(data)-1
        solver = parse_solver_log(directory / "run.log", cycles)
        if not solver["good"] or solver["relaxed_accepted"]:
            raise ValueError(f"closed-box solver did not strictly converge: {tag}: {solver}")
        energy_name = next(name for name in columns if "box_energy" in name)
        mass_name = next(name for name in columns if "box_mass" in name)
        summary["closed_box"][tag] = {"energy_rel_drift": float((columns[energy_name][-1] - columns[energy_name][0]) / columns[energy_name][0]),
                                      "mass_rel_drift": float((columns[mass_name][-1] - columns[mass_name][0]) / columns[mass_name][0]),
                                      "cycles": cycles, "solver": solver,
                                      "input_sha256": sha256_file(directory / "input.used"),
                                      "history_sha256": sha256_file(history[0])}


def diagnostic_analysis(root: Path, figures: Path, summary: dict):
    locations = {mode: root / f"jacobian_{mode}" for mode in ("implicit", "explicit")}
    if not any(path.is_dir() for path in locations.values()):
        summary["diagnostic"] = {"status": "not_run_in_imported_matrix"}
        return
    if not all(path.is_dir() for path in locations.values()):
        raise ValueError("mixed-frame Jacobian audit is incomplete")
    result = {}
    for mode, directory in locations.items():
        log = (directory / "run.log").read_text(errors="replace")
        if "Terminating on cycle limit" not in log or "FATAL ERROR" in log:
            raise ValueError(f"Jacobian audit run incomplete: {mode}")
        solver = parse_solver_log(directory / "run.log", 1)
        if not solver["good"] or solver["relaxed_accepted"]:
            raise ValueError(f"Jacobian audit solver failed: {mode}")
        lines = log.splitlines()
        jacobian = [dict(re.findall(r"(\w+)=([^\s]+)", line))
                    for line in lines if line.startswith("[NRFLD Jacobian]")]
        explicit = [dict(re.findall(r"(\w+)=([^\s]+)", line))
                    for line in lines if line.startswith("[MIXED_EXPLICIT_LEDGER]")]
        implicit = [dict(re.findall(r"(\w+)=([^\s]+)", line))
                    for line in lines if line.startswith("[MIXED_IMPLICIT_LEDGER]")]
        if len(jacobian) != 3 or not explicit or not implicit:
            raise ValueError(f"missing Jacobian or source-ledger lines: {mode}")
        metrics = {"solver": solver,
                   "jacobian": [{key: float(row[key]) for key in
                                 ("epsilon", "gas_l2_rel", "gas_max_rel",
                                  "radiation_l2_rel", "radiation_max_rel")}
                                for row in jacobian],
                   "ledger": {"lambda_abs_volume": sum(float(row["lambda_abs_volume"]) for row in explicit),
                              "explicit_q_abs_volume": sum(float(row["q_abs_volume"]) for row in explicit),
                              "implicit_q_abs_volume": sum(float(row["q_abs_volume"]) for row in implicit),
                              "explicit_pair_balance_volume": sum(float(row["pair_balance_volume"]) for row in explicit),
                              "explicit_floor_volume": sum(float(row["radiation_floor_volume"]) for row in explicit),
                              "implicit_pair_balance_volume": sum(float(row["pair_balance_volume"]) for row in implicit)},
                   "input_sha256": sha256_file(directory / "input.used"),
                   "run_log_sha256": sha256_file(directory / "run.log")}
        for row in metrics["jacobian"]:
            if not all(math.isfinite(value) for value in row.values()):
                raise ValueError(f"nonfinite Jacobian audit: {mode}")
        ledger = metrics["ledger"]
        exchange_scale = ledger["lambda_abs_volume"] + ledger["explicit_q_abs_volume"]
        if abs(ledger["explicit_pair_balance_volume"]
               - ledger["explicit_floor_volume"]) > max(1e-12, 1e-12*exchange_scale):
            raise ValueError(f"gas/radiation source pair imbalance: {mode}")
        ledger["pair_balance_relative_to_exchanged_energy"] = abs(
            ledger["explicit_pair_balance_volume"] - ledger["explicit_floor_volume"]
        ) / exchange_scale
        if (mode == "implicit" and (ledger["implicit_q_abs_volume"] <= 0
                                     or ledger["explicit_q_abs_volume"] != 0)):
            raise ValueError("implicit q placement ledger mismatch")
        if (mode == "explicit" and (ledger["explicit_q_abs_volume"] <= 0
                                     or ledger["implicit_q_abs_volume"] != 0)):
            raise ValueError("explicit q placement ledger mismatch")
        result[mode] = metrics
    summary["diagnostic"] = {"status": "passed", "modes": result,
                             "definitions": "local fixed-coefficient gas/radiation finite-difference Jv; first two source updates, code-volume integrals"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for mode in ("implicit", "explicit"):
        rows = result[mode]["jacobian"]
        axes[0].loglog([x["epsilon"] for x in rows],
                       [x["gas_l2_rel"] for x in rows], "o-", label=f"{mode} gas")
        axes[0].loglog([x["epsilon"] for x in rows],
                       [x["radiation_l2_rel"] for x in rows], "s--", label=f"{mode} radiation")
    axes[0].set(xlabel="finite-difference epsilon", ylabel="relative Jv error")
    axes[0].grid(alpha=.25)
    axes[0].legend()
    positions = np.arange(2)
    axes[1].bar(positions-.2, [result[m]["ledger"]["implicit_q_abs_volume"] for m in ("implicit", "explicit")],
                width=.4, label="implicit q")
    axes[1].bar(positions+.2, [result[m]["ledger"]["explicit_q_abs_volume"] for m in ("implicit", "explicit")],
                width=.4, label="explicit q")
    axes[1].set_xticks(positions, ("implicit", "explicit"))
    axes[1].set_ylabel("sum |q energy| [code energy × code volume]")
    axes[1].legend()
    fig.savefig(figures / "jacobian_ledger.png", dpi=170)
    plt.close(fig)


def render(root: Path, summary: dict):
    rows = []
    for tag, case in summary["cases"].items():
        if tag.startswith("shock"):
            error = case["aligned_L1_percent"]["trad"]
            result = f"Trad L1 {error:.3g}% (aligned); raw {case['raw_L1_percent']['trad']:.3g}%; shock x={case['shock_raw_cm']:.4g} cm"
        elif "moving" in tag:
            error = case["comparison"]["Erad"]["pulse_contrast_L1"]
            result = f"Erad pulse-contrast L1 {error:.3g}; shift {case['shift_cm']:.4g} cm"
        else:
            result = "static reference"
        rows.append(f"<tr><td>{escape(tag)}</td><td>{case['cycles']}</td><td>{escape(result)}</td><td>{case['solver']['fgmres_iterations_total']}</td><td>{case['wall_seconds']}</td><td>{case['launcher_max_rss_kib']}</td></tr>")
    images = "".join(f'<figure><img src="figures/{escape(path.name)}"><figcaption>{escape(path.stem)}</figcaption></figure>'
                     for path in sorted((root / "figures").glob("*.png")))
    legacy_count = sum(case.get("grid_cfl_provenance", "").startswith("hdf_and")
                       for case in summary["cases"].values())
    legacy_note = (f"<p>旧データ {legacy_count} 件の shock 入力ファイルには実行時の格子・CFL 上書きが保存されていません。"
                   "HDF の格子数とログの初回 dt 比で確認し、各ケースの provenance を JSON に記録しました。</p>"
                   if legacy_count else "")
    diagnostic = summary.get("diagnostic", {})
    diagnostic_text = ("<strong>局所 gas/radiation Jacobian の有限差分 Jv と、q 項の陰的/陽的収支ログを検証済み。</strong> "
                       + ", ".join(f"{mode}: ε=10⁻⁶ の最大相対 Jv 誤差 "
                                   f"{max(value['jacobian'][-1]['gas_max_rel'], value['jacobian'][-1]['radiation_max_rel']):.3g}, "
                                   f"収支相対誤差 {value['ledger']['pair_balance_relative_to_exchanged_energy']:.3g}"
                                   for mode, value in diagnostic['modes'].items())
                       if diagnostic.get("status") == "passed" else
                       "局所 Jacobian と項別収支の追加 audit はこの imported matrix では未実行。")
    body = f'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>Mixed-frame implicit placement</title>
<style>:root{{font:21px/1.55 system-ui,sans-serif;color:#17243a;background:#f8fafc}}body{{max-width:1250px;margin:2rem auto;padding:0 1rem}}h1{{font-size:2rem}}h2{{font-size:1.45rem;color:#064e7a}}strong{{color:#075985}}table{{border-collapse:collapse;width:100%;font-size:.95rem}}th,td{{padding:.5rem;border-bottom:1px solid #cbd5e1;text-align:left;vertical-align:top}}th{{background:#e0f2fe}}img{{max-width:100%;height:auto}}figure{{margin:1.5rem 0}}a{{color:#0369a1}}@media(max-width:700px){{:root{{font-size:19px}}table{{display:block;overflow-x:auto}}}}</style>
</head><body><h1>Mixed-frame Lorentz 項の陰的配置比較</h1>
<p><strong>全ケースの NR と FGMRES が厳密収束し、指定の物理時刻まで完走しました。</strong> ON は q v·∇E を NR 残差・Jacobian に含み、OFF は同じ係数・差分で陽的に評価します。固定 flux limiter のみ対応します。</p>
<p>衝撃波の形状誤差は密度ジャンプ位置を平行移動して比較し、<strong>元の位置と平行移動前の誤差も併記</strong>しています。パルスは物理移動距離 v₀t だけ周期シフトして静止計算と比較します。閉じた周期箱の総エネルギードリフト: ON {summary['closed_box']['box_implicit']['energy_rel_drift']:.3g}, OFF {summary['closed_box']['box_explicit']['energy_rel_drift']:.3g}。</p>
<p>{diagnostic_text}</p>
{legacy_note}
<h2>結果</h2><table><thead><tr><th>Case</th><th>steps</th><th>comparison</th><th>FGMRES iterations</th><th>wall s</th><th>launcher RSS KiB</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>図</h2>{images}<p><a href="summary.json">summary.json</a></p></body></html>'''
    (root / "index.html").write_text(body, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postprocess", type=Path, required=True)
    parser.add_argument("--preset", choices=tuple(SHOCK_MATRIX), default="full")
    args = parser.parse_args()
    root = args.postprocess.expanduser().resolve()
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    summary = {"created_utc": utc_now(), "status": "passed", "preset": args.preset, "cases": {},
               "closed_box": {}, "reference": {},
               "definitions": {"shock_shape": "density jump aligned to semi-analytic jump",
                               "shock_raw": "fixed physical x coordinates without postprocessing shift",
                               "pulse": "moving solution shifted by v0*t, velocity subtracts v0"}}
    for kind, candidates in (("shock", ("shock_build_source",)),
                             ("pulse", ("pulse_build_source", "build_source"))):
        source = next((root / name for name in candidates if (root / name / "bin/athena").is_file()), None)
        if source is None:
            raise ValueError(f"missing {kind} isolated build source in {root}")
        summary.setdefault("build", {})[kind] = {
            "binary_sha256": sha256_file(source / "bin/athena"),
            "source_sha256": tree_sha256(source / "src")}
    diagnostic_build = root / "diagnostic_build_source"
    if diagnostic_build.is_dir():
        summary["build"]["diagnostic"] = {
            "binary_sha256": sha256_file(diagnostic_build / "bin/athena"),
            "source_sha256": tree_sha256(diagnostic_build / "src")}
    shock_analysis(root, figures, summary, args.preset)
    pulse_analysis(root, figures, summary, args.preset)
    closed_box_analysis(root, summary)
    diagnostic_analysis(root, figures, summary)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = base_manifest(find_repo_root(HERE), "mixed_frame", 4)
    manifest.update({"status": summary["status"], "metrics": summary,
                     "analysis_script_sha256": sha256_file(Path(__file__))})
    write_manifest(root / "manifest.json", manifest)
    render(root, summary)
    print(root / "index.html")


if __name__ == "__main__":
    main()
