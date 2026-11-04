#!/usr/bin/env python3
"""Compare matched solar-convection inflow-entropy restart experiments."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from solar_convection_report import (effective_problem_bool,
                                     effective_problem_float, input_values,
                                     physical_time_unit,
                                     read_history)


def explicit_entropy(run_dir: Path) -> float:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    values = [float(item.split("=", 1)[1])
              for item in manifest["problem_overrides"]
              if item.startswith("s_in=")]
    if len(values) != 1 or "s_in_from_profile=false" not in manifest[
            "problem_overrides"]:
        raise ValueError(f"{run_dir}: expected one explicit inflow entropy")
    return values[0]


def rolling_mean(y: np.ndarray, samples: int) -> np.ndarray:
    kernel = np.ones(samples)
    return np.convolve(y, kernel, mode="same") / np.convolve(
        np.ones(len(y)), kernel, mode="same")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-continuation", type=Path, action="append",
                        default=[], help="Consecutive checkpoint restart of the "
                        "baseline; repeat for interrupted segments")
    parser.add_argument("--baseline-s-in", type=float, required=True,
                        help="Effective profile-derived entropy from the original-run log")
    parser.add_argument("--variant", nargs=2, action="append", required=True,
                        metavar=("LABEL", "RUN_DIR"))
    parser.add_argument("--variant-continuation", nargs=2, action="append",
                        default=[], metavar=("LABEL", "RUN_DIR"),
                        help="Consecutive checkpoint continuation of a variant")
    parser.add_argument("--header-from", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-flux", type=float, default=6.3e10)
    args = parser.parse_args()
    paths = [("baseline", args.baseline), *[(label, Path(path))
                                           for label, path in args.variant]]
    variant_continuations: dict[str, list[Path]] = {}
    for label, path in args.variant_continuation:
        if label not in {name for name, _ in paths[1:]}:
            raise ValueError(f"Unknown variant continuation label: {label}")
        variant_continuations.setdefault(label, []).append(Path(path))
    manifests = {label: json.loads((path / "manifest.json").read_text())
                 for label, path in paths}
    baseline_manifest = manifests["baseline"]
    continuation_manifests = [json.loads((path / "manifest.json").read_text())
                              for path in args.baseline_continuation]
    previous = args.baseline
    for path, manifest in zip(args.baseline_continuation, continuation_manifests):
        if Path(manifest.get("restart", "")).parent.resolve() != previous.resolve():
            raise ValueError(f"{path}: baseline continuation must use the prior checkpoint")
        for key in ("binary_sha256", "base_input_sha256", "data_sha256",
                    "generated_input_sha256", "problem_overrides"):
            if manifest.get(key) != baseline_manifest.get(key):
                raise ValueError(f"{path}: baseline continuation {key} differs")
        previous = path
    if not effective_problem_bool(args.baseline / "athinput.generated",
                                  baseline_manifest.get("problem_overrides", []),
                                  "s_in_from_profile"):
        raise ValueError("Baseline must derive its inflow entropy from the profile")
    numerical_prefixes = ("mesh/nx1=", "mesh/nx2=", "mesh/nx3=",
                          "time/cfl_number=")
    def numerical_overrides(manifest: dict) -> list[str]:
        return sorted(arg for arg in manifest["command"]
                      if arg.startswith(numerical_prefixes))
    for label, path in paths[1:]:
        manifest = manifests[label]
        for key in ("binary_sha256", "base_input_sha256", "data_sha256",
                    "restart"):
            if manifest.get(key) != baseline_manifest.get(key):
                raise ValueError(f"{label}: {key} differs from baseline")
        if numerical_overrides(manifest) != numerical_overrides(baseline_manifest):
            raise ValueError(f"{label}: grid or CFL differs from baseline")
        for key in ("mass_balance_relax_time", "mass_balance_gain",
                    "mass_balance_rate_gain", "Cdmp"):
            reference = effective_problem_float(
                args.baseline / "athinput.generated",
                baseline_manifest.get("problem_overrides", []), key)
            candidate = effective_problem_float(
                path / "athinput.generated",
                manifest.get("problem_overrides", []), key)
            if reference != candidate:
                raise ValueError(f"{label}: {key} differs from baseline")
    for label, path in paths[1:]:
        previous = path
        original = manifests[label]
        for continuation in variant_continuations.get(label, []):
            manifest = json.loads((continuation / "manifest.json").read_text())
            if Path(manifest.get("restart", "")).parent.resolve() != previous.resolve():
                raise ValueError(f"{continuation}: not restarted from prior variant segment")
            for key in ("binary_sha256", "base_input_sha256", "data_sha256",
                        "generated_input_sha256", "problem_overrides"):
                if manifest.get(key) != original.get(key):
                    raise ValueError(f"{continuation}: changed {key}")
            if numerical_overrides(manifest) != numerical_overrides(original):
                raise ValueError(f"{continuation}: grid or CFL changed")
            previous = continuation
        for key in ("mass_balance_on", "gravity_work_face_flux"):
            reference = effective_problem_bool(
                args.baseline / "athinput.generated",
                baseline_manifest.get("problem_overrides", []), key)
            candidate = effective_problem_bool(
                path / "athinput.generated",
                manifest.get("problem_overrides", []), key)
            if reference != candidate:
                raise ValueError(f"{label}: {key} differs from baseline")
    histories = {label: read_history(next(path.glob("*.hst")), args.header_from)
                 for label, path in paths}
    for path in args.baseline_continuation:
        current = read_history(next(path.glob("*.hst")), args.header_from)
        prior = histories["baseline"]
        if set(current) != set(prior) or current["time"][0] <= prior["time"][0]:
            raise ValueError(f"{path}: incompatible baseline history")
        keep = prior["time"] < current["time"][0]
        histories["baseline"] = {key: np.r_[values[keep], current[key]]
                                 for key, values in prior.items()}
    for label, continuations in variant_continuations.items():
        for path in continuations:
            current = read_history(next(path.glob("*.hst")), args.header_from)
            prior = histories[label]
            if set(current) != set(prior) or current["time"][0] <= prior["time"][0]:
                raise ValueError(f"{path}: incompatible variant history")
            keep = prior["time"] < current["time"][0]
            histories[label] = {key: np.r_[values[keep], current[key]]
                                for key, values in prior.items()}
    start = max(float(h["time"][0]) for h in histories.values())
    end = min(float(h["time"][-1]) for h in histories.values())
    if end - start < 5.0:
        raise ValueError("Need at least five code-time units of common history")
    baseline = histories["baseline"]
    units = input_values(args.baseline / "athinput.generated")
    area = ((units["x1max"]-units["x1min"])
            * (units["x2max"]-units["x2min"]) * units["leng_unit"]**2)
    velocity_unit = np.sqrt(units["egas_unit"]/units["rho_unit"])
    time_unit = physical_time_unit(units, velocity_unit)
    span = min(25.0, 0.5*(end-start))
    late_half_start = start + 0.5*(end-start)
    rows = []
    curves = []
    for label, path in paths[1:]:
        history = histories[label]
        selected = (history["time"] >= start) & (history["time"] <= end)
        t = history["time"][selected]
        if len(t) < 20:
            raise ValueError(f"{path}: too few samples in common interval")
        delta = {}
        for key in ("FhydBottom", "FtotTop", "EtotGrav"):
            delta[key] = history[key][selected] - np.interp(
                t, baseline["time"], baseline[key])
        early = t < start + span
        late = t >= end - span
        late_half = t >= late_half_start
        entropy = explicit_entropy(path)
        row = {
            "label": label, "run_dir": str(path), "s_in_cgs": entropy,
            "delta_s_in_cgs": entropy - args.baseline_s_in,
            "early_delta_top_over_solar": float(np.mean(delta["FtotTop"][early])
                                               / args.reference_flux),
            "late_delta_top_over_solar": float(np.mean(delta["FtotTop"][late])
                                              / args.reference_flux),
            "late_half_delta_top_over_solar": float(np.mean(
                delta["FtotTop"][late_half])/args.reference_flux),
            "late_half_delta_bottom_hydro_over_solar": float(np.mean(
                delta["FhydBottom"][late_half])/args.reference_flux),
            "early_delta_bottom_hydro_over_solar": float(np.mean(
                delta["FhydBottom"][early])/args.reference_flux),
            "late_delta_bottom_hydro_over_solar": float(np.mean(
                delta["FhydBottom"][late])/args.reference_flux),
            "delta_stored_energy_rate_over_solar": float(
                (delta["EtotGrav"][-1]-delta["EtotGrav"][0])
                / (area*args.reference_flux*(t[-1]-t[0])*time_unit)),
            "late_delta_stored_energy_rate_over_solar": float(
                (delta["EtotGrav"][late][-1]-delta["EtotGrav"][late][0])
                / (area*args.reference_flux
                   *(t[late][-1]-t[late][0])*time_unit)),
            "late_absolute_top_over_solar": float(np.mean(
                history["FtotTop"][selected][late])/args.reference_flux),
            "late_absolute_bottom_hydro_over_solar": float(np.mean(
                history["FhydBottom"][selected][late])/args.reference_flux),
            "sample_count": int(len(t)),
        }
        final_path = variant_continuations.get(label, [path])[-1]
        row["final_run_dir"] = str(final_path)
        summary = final_path / "summary.json"
        if summary.is_file():
            item = json.loads(summary.read_text())
            row["tau23_last_K"] = item.get("surface3d_last", {}).get(
                "T_tau23_mean_K")
            row["solver_completed"] = bool(item.get("log", {}).get("completed"))
        rows.append(row)
        curves.append((label, t-start, history["FtotTop"][selected]
                       / args.reference_flux, delta))
    baseline_selected = (baseline["time"] >= start) & (baseline["time"] <= end)
    t_base = baseline["time"][baseline_selected]
    baseline_late = t_base >= end - span
    result = {
        "description": "Matched-restart transient entropy sensitivity; not a solar calibration",
        "baseline_run_dir": str(args.baseline),
        "baseline_continuation_dirs": [str(path) for path in
                                       args.baseline_continuation],
        "variant_continuation_dirs": {
            label: [str(path) for path in paths]
            for label, paths in variant_continuations.items()},
        "baseline_s_in_cgs": args.baseline_s_in,
        "common_start_code": start, "common_end_code": end,
        "early_window_code": [start, start+span],
        "late_window_code": [end-span, end],
        "late_half_window_code": [late_half_start, end],
        "baseline_late_top_over_solar": float(np.mean(
            baseline["FtotTop"][baseline_selected][baseline_late])
            / args.reference_flux),
        "variants": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    plt.rcParams.update({"font.size": 13})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes[0, 0].plot(t_base-start,
                    baseline["FtotTop"][baseline_selected]/args.reference_flux,
                    label="baseline", color="black")
    for index, (label, t, top, delta) in enumerate(curves):
        color = f"C{index}"
        axes[0, 0].plot(t, top, label=label, color=color)
        top_delta = delta["FtotTop"]/args.reference_flux
        axes[0, 1].plot(t, top_delta, alpha=0.18, lw=0.8, color=color)
        top_samples = max(1, int(round(25.0/np.median(np.diff(t)))))
        axes[0, 1].plot(t, rolling_mean(top_delta, top_samples), label=label,
                        color=color)
        samples = max(1, int(round(5.0/np.median(np.diff(t)))))
        axes[1, 0].plot(t, rolling_mean(delta["FhydBottom"]
                                        / args.reference_flux, samples), label=label,
                        color=color)
        axes[1, 1].plot(t, (delta["EtotGrav"]-delta["EtotGrav"][0])
                        / (area*args.reference_flux*3600.0), label=label,
                        color=color)
    axes[0, 0].set(title="Top flux", ylabel=r"$F/F_\odot$")
    axes[0, 1].set(title="Top-flux response (25-code running mean)",
                   ylabel=r"$\Delta F/F_\odot$")
    axes[1, 0].set(title="Bottom hydro-flux response (5-code mean)",
                   ylabel=r"$\Delta F/F_\odot$")
    axes[1, 1].set(title="Stored-energy response", ylabel="solar-flux hours")
    for axis in axes.flat:
        axis.set_xlabel("code time since matched restart")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.savefig(args.output_dir / "entropy_sensitivity.png", dpi=150)
    plt.close(fig)
    cells = []
    for row in rows:
        name = Path(row["run_dir"]).name
        links = f"<a href='../{html.escape(name)}/index.html'>{html.escape(row['label'])}</a>"
        if row["final_run_dir"] != row["run_dir"]:
            continuation_name = Path(row["final_run_dir"]).name
            links += (f" <a href='../{html.escape(continuation_name)}/index.html'>"
                      "(continuation)</a>")
        cells.append("<tr>" + "".join(f"<td>{value}</td>" for value in (
            links,
            f"{row['delta_s_in_cgs']:+.2e}",
            f"{row['early_delta_bottom_hydro_over_solar']:+.3f}",
            f"{row['late_delta_bottom_hydro_over_solar']:+.3f}",
            f"{row['early_delta_top_over_solar']:+.4f}",
            f"{row['late_delta_top_over_solar']:+.4f}",
            f"{row['late_half_delta_top_over_solar']:+.4f}",
            f"{row['delta_stored_energy_rate_over_solar']:+.3f}",
            f"{row['late_absolute_top_over_solar']:.3f}",
        )) + "</tr>")
    headers = ("Case", "Δs [erg g⁻¹ K⁻¹]", "Early ΔFhyd,bottom / solar",
               "Late ΔFhyd,bottom / solar", "Early ΔFtop / solar",
               "Late 25-code ΔFtop / solar", "Late half ΔFtop / solar",
               "Δ stored energy rate / solar", "Late Ftop / solar")
    table_head = "".join(f"<th>{name}</th>" for name in headers)
    baseline_link = (f"<a href='../{html.escape(args.baseline.name)}/index.html'>"
                     "Baseline case report</a>")
    (args.output_dir / "index.html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<title>Solar convection entropy sensitivity</title><style>"
        "body{font:19px/1.5 system-ui;max-width:1300px;margin:2rem auto;"
        "padding:0 1rem;color:#172b3a}table{border-collapse:collapse}"
        "th,td{padding:.5rem .75rem;border-bottom:1px solid #ddd;text-align:left}"
        "img{max-width:100%}</style><h1>Inflow entropy sensitivity</h1>"
        "<p><strong>Transient matched-restart comparison.</strong> A response "
        "at the bottom does not establish a calibrated solar top flux. "
        "The late short and late-half averages should agree in sign before "
        "interpreting a persistent upper response. Each run's stationarity "
        "and solver status are in its linked report. The stored-energy rate "
        "is the variant-minus-baseline endpoint change over the full common "
        "interval; oscillation phase can affect it.</p>"
        f"<p>Common interval: {start:.1f}–{end:.1f} code time; "
        f"baseline s_in = {args.baseline_s_in:.9e} erg g⁻¹ K⁻¹; "
        f"baseline late top flux = {result['baseline_late_top_over_solar']:.3f} "
        f"solar. Late half = {late_half_start:.1f}–{end:.1f} code; "
        f"late short window = {end-span:.1f}–{end:.1f}. {baseline_link}</p>"
        f"<table><tr>{table_head}</tr>{''.join(cells)}</table>"
        "<img src='entropy_sensitivity.png' alt='entropy sensitivity'>"
        "<p><a href='summary.json'>Numerical results</a></p></html>"
    )
    print(args.output_dir / "index.html")


if __name__ == "__main__":
    main()
