#!/usr/bin/env python3
"""Compare sustained lower flux in matched fixed-entropy solar branches."""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from solar_convection_report import (input_values, read_history,
                                     sampled_energy_ledger,
                                     total_bottom_flux, total_top_flux)
from solar_convection_entropy_transport import snapshot

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE / "scripts"))
from build_solar_initial_profile import AthenaEosTable  # noqa: E402


F_SUN = 6.3e10
CASES = ("fixed", "plus1e6", "plus3e6")


def average(time: np.ndarray, values: np.ndarray,
            start: float, end: float) -> float:
    selected = (time > start) & (time < end)
    t = np.r_[start, time[selected], end]
    y = np.r_[np.interp(start, time, values),
              values[selected], np.interp(end, time, values)]
    return float(np.trapezoid(y, t) / (end-start))


def mean_snapshots(rows: list[dict], start: float, end: float) -> dict:
    selected = [row for row in rows if start-1e-8 <= row["time_code"] <= end+1e-8]
    if not selected:
        raise ValueError(f"No full-volume snapshots in t={start}..{end}")
    keys = ("upflow_entropy_cgs", "downflow_entropy_cgs", "entropy_contrast_cgs",
            "area_mean_temperature_K", "upward_mass_flux_cgs",
            "downward_mass_flux_cgs")
    return {"count": len(selected),
            **{key: float(np.mean([row[key] for row in selected])) for key in keys}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--header-from", type=Path, required=True)
    parser.add_argument("--eos-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output_dir or root / "comparison").resolve()
    output.mkdir(parents=True, exist_ok=True)
    state = json.loads((root / "state.json").read_text())
    if state["status"] != "completed":
        parser.error("All three branches must finish before comparison")
    case_paths = {name: root / name for name in CASES}
    manifests = {name: json.loads((path / "manifest.json").read_text())
                 for name, path in case_paths.items()}
    first_manifest = manifests["fixed"]
    for name in CASES[1:]:
        for key in ("binary_sha256", "base_input_sha256",
                    "generated_input_sha256", "data_sha256", "restart"):
            if manifests[name][key] != first_manifest[key]:
                raise ValueError(f"{name}: {key} differs from fixed case")
        a = [x for x in first_manifest["problem_overrides"]
             if not x.startswith("s_in=")]
        b = [x for x in manifests[name]["problem_overrides"]
             if not x.startswith("s_in=")]
        if a != b:
            raise ValueError(f"{name}: controls differ from fixed case")
    histories = {name: read_history(next(path.glob("*.hst")), args.header_from)
                 for name, path in case_paths.items()}
    start = max(float(h["time"][0]) for h in histories.values())
    end = min(float(h["time"][-1]) for h in histories.values())
    if end-start < 250:
        raise ValueError("Need at least 250 common code-time units")
    early = (start, start+50.0)
    late = (end-100.0, end)
    units = input_values(case_paths["fixed"] / "athinput.generated")
    eos = AthenaEosTable(args.eos_table.resolve())
    velocity_unit = np.sqrt(units["egas_unit"] / units["rho_unit"])
    result = {"start_code": start, "end_code": end,
              "early_window": list(early), "late_window": list(late),
              "cases": {}}
    for name, path in case_paths.items():
        h = histories[name]
        if "FhydUpBottom" not in h or "FhydDownBottom" not in h:
            raise ValueError(f"{name}: directional flux diagnostics missing")
        closure = float(np.max(np.abs(h["FhydUpBottom"] + h["FhydDownBottom"]
                                      - h["FhydBottom"])) / F_SUN)
        if closure > 1e-8:
            raise ValueError(f"{name}: lower directional flux does not close")
        records = []
        uov = {p.name.split(".uov_full.")[-1]: p
               for p in path.glob("*.uov_full.*.athdf")}
        entropy = next(float(item.split("=", 1)[1])
                       for item in manifests[name]["problem_overrides"]
                       if item.startswith("s_in="))
        for prim in sorted(path.glob("*.prim_full.*.athdf")):
            suffix = prim.name.split(".prim_full.")[-1]
            if suffix not in uov:
                raise ValueError(f"{name}: missing paired full volume {prim}")
            item = snapshot(prim, uov[suffix], eos, units["rho_unit"],
                            velocity_unit, entropy)
            if item is not None:
                records.append(item)
        report = json.loads((path / "summary.json").read_text())
        log = report["log"]
        if (not log["completed"] or log["fatal_count"] or
                log["slow_mg_warning_count"] or
                set(log["nr_status_counts"]) != {"converged"}):
            raise ValueError(f"{name}: solver checks failed")
        windows = {}
        for label, (left, right) in (("early", early), ("late", late)):
            t = h["time"]
            indices = np.flatnonzero((t >= left) & (t <= right))
            ledger = sampled_energy_ledger(h, indices, units, F_SUN)
            windows[label] = {
                "top_total_over_solar": average(t, total_top_flux(h), left, right)/F_SUN,
                "bottom_total_over_solar": average(t, total_bottom_flux(h), left, right)/F_SUN,
                "bottom_hydro_over_solar": average(t, h["FhydBottom"], left, right)/F_SUN,
                "bottom_up_hydro_over_solar": average(t, h["FhydUpBottom"], left, right)/F_SUN,
                "bottom_down_hydro_over_solar": average(t, h["FhydDownBottom"], left, right)/F_SUN,
                "stored_energy_trend_over_solar": ledger["stored_energy_trend_over_solar"],
                "stored_energy_rate_over_solar": ledger["stored_energy_rate_over_solar"],
                "eos": mean_snapshots(records, left, right)}
        result["cases"][name] = {
            "s_in_cgs": entropy, "windows": windows,
            "directional_closure_over_solar": closure,
            "newton_converged": log["nr_status_counts"]["converged"],
            "slow_mg_warning_count": log["slow_mg_warning_count"],
            "fatal_count": log["fatal_count"],
            "mass_drift_late": report["late_window"]["max_abs_mass_drift"],
            "snapshots": records}
    baseline = result["cases"]["fixed"]
    for name in CASES[1:]:
        item = result["cases"][name]
        item["late_minus_fixed"] = {
            key: item["windows"]["late"][key] - baseline["windows"]["late"][key]
            for key in ("top_total_over_solar", "bottom_total_over_solar",
                        "bottom_up_hydro_over_solar",
                        "bottom_down_hydro_over_solar",
                        "stored_energy_trend_over_solar")}

    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    plt.rcParams.update({"font.size": 15})
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True,
                             constrained_layout=True)
    colors = {"fixed": "#1b6ca8", "plus1e6": "#dc8120", "plus3e6": "#ad3f57"}
    for name in CASES:
        h = histories[name]
        x = h["time"] - 9100
        for axis, values in ((axes[0], total_bottom_flux(h)/F_SUN),
                             (axes[1], total_top_flux(h)/F_SUN)):
            samples = max(3, round(10 / np.median(np.diff(h["time"]))))
            weights = np.ones(samples)
            smoothed = (np.convolve(values, weights, mode="same") /
                        np.convolve(np.ones(len(values)), weights, mode="same"))
            axis.plot(x, smoothed,
                      color=colors[name], label=name, lw=1.6)
        records = result["cases"][name]["snapshots"]
        axes[2].plot([row["time_code"]-9100 for row in records],
                     [row["entropy_contrast_cgs"]/1e6 for row in records],
                     "o-", color=colors[name], label=name)
    axes[0].axhline(1, color="black", ls="--", lw=1, label="solar")
    axes[0].set_ylabel("Bottom total / solar")
    axes[1].set_ylabel("Top total / solar")
    axes[2].set_ylabel("Up−down entropy\n[10⁶ erg g⁻¹ K⁻¹]")
    axes[2].set_xlabel("Code time since t=9100")
    for axis in axes:
        axis.axvspan(late[0]-9100, late[1]-9100, color="gray", alpha=.08)
        axis.grid(alpha=.25)
        axis.legend()
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)

    def table_row(name: str) -> str:
        item = result["cases"][name]
        early_row, late_row = item["windows"]["early"], item["windows"]["late"]
        link = html.escape(os.path.relpath(case_paths[name] / "index.html", output))
        return (f"<tr><td><a href='{link}'>{name}</a></td>"
                f"<td>{(item['s_in_cgs']-baseline['s_in_cgs'])/1e6:+.1f}</td>"
                f"<td>{early_row['bottom_total_over_solar']:.3f}</td>"
                f"<td>{late_row['bottom_total_over_solar']:.3f}</td>"
                f"<td>{late_row['top_total_over_solar']:.3f}</td>"
                f"<td>{late_row['stored_energy_trend_over_solar']:+.3f}</td>"
                f"<td>{late_row['eos']['entropy_contrast_cgs']/1e6:.3f}</td></tr>")

    rows = "".join(table_row(name) for name in CASES)
    html_page = ("<!doctype html><html lang='ja'><meta charset='utf-8'>"
                 "<title>固定流入エントロピーの同一checkpoint比較</title>"
                 "<style>body{font:22px/1.55 system-ui;max-width:1250px;"
                 "margin:2rem auto;padding:0 1rem;color:#17324a}"
                 "h1{font-size:2rem}strong{color:#184c76}"
                 "table{border-collapse:collapse;width:100%}"
                 "th,td{padding:.55rem .7rem;border-bottom:1px solid #ccc;"
                 "text-align:right}th:first-child,td:first-child{text-align:left}"
                 "img{max-width:100%}</style>"
                 "<h1>固定流入エントロピーの同一checkpoint比較</h1>"
                 f"<p>全ケースはt=9100の同じcheckpointからt={end:.0f}まで計算。"
                 "CFL、格子、バイナリ、EOS、HD2境界制御は共通です。"
                 f"最初の50コード時間（t={early[0]:.1f}～{early[1]:.1f}）と"
                 f"最後の100コード時間（t={late[0]:.0f}～{late[1]:.0f}）を比較します。"
                 "</p><table><tr><th>ケース</th><th>Δs [10⁶]</th>"
                 "<th>初期下端 / Fsun</th><th>後半下端 / Fsun</th>"
                 "<th>後半上端 / Fsun</th><th>後半蓄熱 / Fsun</th>"
                 "<th>後半上下流Δs [10⁶]</th></tr>" + rows + "</table>"
                 "<p><strong>下端が太陽値に近づいても、蓄熱が大きければ"
                 "持続的な定常入力とはまだ判定できません。</strong>"
                 "上下流エントロピーは下端第1セルのEOS値を質量流束で重み付けした診断です。"
                 "方向別hydro流束は実際のGodunov面流束です。</p>"
                 "<img src='comparison.png' alt='Flux and entropy comparison'>"
                 "<p><a href='summary.json'>全数値と方法</a> · "
                 "<a href='../state.json'>実行記録</a></p></html>")
    (output / "index.html").write_text(html_page + "\n")
    print(output / "index.html")


if __name__ == "__main__":
    main()
