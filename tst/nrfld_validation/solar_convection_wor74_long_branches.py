#!/usr/bin/env python3
"""Compare matched WOR-74 96 km branches over several crossing times."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from solar_convection_report import (  # noqa: E402
    combine_histories, input_values, parse_log, sampled_energy_ledger,
    total_bottom_flux, total_top_flux,
)
from solar_convection_downflow_profiles import profile, read_volume  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build_solar_initial_profile import AthenaEosTable  # noqa: E402
from plot_simple_convection_tau import units_from_input  # noqa: E402

FSUN = 6.3e10
WINDOWS = [(24010, 24150), (24150, 24310)]
CASES = {"baseline": "s_in fixed, alpha=0.5",
         "splus_10e6": "s_in +1e7, alpha=0.5",
         "alpha_100": "s_in fixed, alpha=1.0"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def time_average(h: dict[str, np.ndarray], values: np.ndarray,
                 start: float, end: float) -> float:
    time = h["time"]
    inner = (time > start) & (time < end)
    ts = np.r_[start, time[inner], end]
    ys = np.interp(ts, time, values)
    return float(np.trapezoid(ys, ts)/(end-start))


def window(h: dict[str, np.ndarray], start: float, end: float,
           units: dict[str, float]) -> dict:
    idx = np.flatnonzero((h["time"] >= start) & (h["time"] <= end))
    ledger = sampled_energy_ledger(h, idx, units, FSUN)
    return {
        "start_code": start, "end_code": end,
        "top_over_solar": time_average(h, total_top_flux(h), start, end)/FSUN,
        "bottom_over_solar": time_average(h, total_bottom_flux(h), start, end)/FSUN,
        "bottom_hydro_over_solar": time_average(h, h["FhydBottom"], start, end)/FSUN,
        "bottom_radiation_over_solar": time_average(
            h, h["FradBottom"] + h.get("FadvRadBottom", 0.0), start, end)/FSUN,
        "stored_trend_over_solar": ledger["stored_energy_trend_over_solar"],
        "ledger_residual_over_solar": ledger["residual_equivalent_solar_flux"],
        "max_mass_drift": float(np.max(np.abs(h["MassDrift"][idx]))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in CASES:
        parser.add_argument(f"--{key.replace('_', '-')}-short", type=Path,
                            required=key != "alpha_100")
    parser.add_argument("--long-root", type=Path, required=True)
    parser.add_argument("--header-from", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--eos-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = {name: label for name, label in CASES.items()
                if getattr(args, f"{name}_short") is not None}
    args.output.mkdir(parents=True, exist_ok=True)
    units = input_values(args.input)
    profile_units = units_from_input(args.input)
    eos = AthenaEosTable(args.eos_table)
    histories = {}
    summary = {"scope": "Matched t=24000 restart; 310 code-time transient",
               "windows": WINDOWS, "cases": {}}
    for name, label in selected.items():
        short = getattr(args, f"{name}_short")
        long = args.long_root / f"{name}_t24310"
        if name == "splus_10e6":
            long = args.long_root / "splus_10e6_t24310"
        paths = [short / "simple_convection_general_6Mm_box.hst",
                 long / "simple_convection_general_6Mm_box.hst"]
        h = combine_histories(paths, args.header_from)
        if abs(h["time"][-1] - 24310) > 1e-6:
            raise ValueError(f"Incomplete history: {name}, t={h['time'][-1]}")
        log = parse_log(long / "run.log")
        if not log["completed"] or log["fatal_count"] or log["slow_mg_warning_count"]:
            raise ValueError(f"Incomplete or suspicious solver log: {name}")
        histories[name] = h
        dumps = sorted(long.glob("*.prim_full.*.athdf"))
        if not dumps:
            raise FileNotFoundError(f"No full 3D output in {long}")
        snapshots = []
        for prim in dumps:
            uov = long / prim.name.replace(".prim_full.", ".uov_full.")
            if not uov.exists():
                raise FileNotFoundError(uov)
            prof, _ = profile(prim, uov, eos, profile_units)
            uov_data, _, _ = read_volume(uov, "user_out_var")
            snapshots.append({
                "time_code": float(prof["time_code"]),
                "prim": str(prim), "prim_sha256": sha256(prim),
                "uov": str(uov), "uov_sha256": sha256(uov),
                "bottom_mean_temperature_K": float(prof["mean_temperature_K"][0]),
                "top_mean_temperature_K": float(prof["mean_temperature_K"][-1]),
                "top_mean_radiation_energy_cgs": float(np.mean(uov_data[1, -1])),
                "top_mean_radiation_temperature_K": float(
                    np.mean(uov_data[3, -1])),
                "bottom_cell_up_entropy_cgs": float(prof["up_entropy_cgs"][0]),
                "bottom_cell_down_entropy_cgs": float(prof["down_entropy_cgs"][0]),
                "bottom_cell_entropy_contrast_cgs": float(
                    prof["entropy_contrast_cgs"][0]),
                "bottom_cell_up_mass_flux_cgs": float(prof["up_mass_flux_cgs"][0]),
                "bottom_cell_down_mass_flux_cgs": float(
                    prof["down_mass_flux_cgs"][0]),
            })
        manifest = json.loads((long / "manifest.json").read_text())
        if manifest["exit_code"] != 0 or manifest["guard_reason"]:
            raise ValueError(f"Incomplete or guarded run: {name}")
        summary["cases"][name] = {
            "label": label, "history_paths": list(map(str, paths)),
            "history_sha256": [sha256(p) for p in paths],
            "log": log, "manifest": manifest,
            "windows": [window(h, a, b, units) for a, b in WINDOWS],
            "snapshots": snapshots,
        }

    baseline = summary["cases"]["baseline"]["windows"]
    for name, case in summary["cases"].items():
        for row, ref in zip(case["windows"], baseline):
            for metric in ("top_over_solar", "bottom_over_solar", "stored_trend_over_solar"):
                row["delta_" + metric] = row[metric] - ref[metric]
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    plt.rcParams.update({"font.size": 15, "axes.labelsize": 17, "legend.fontsize": 13})
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    timeline = np.arange(24010, 24310.0001, 0.2)
    base = histories["baseline"]
    base_top = np.interp(timeline, base["time"], total_top_flux(base))/FSUN
    base_bottom = np.interp(timeline, base["time"], total_bottom_flux(base))/FSUN
    colors = {"baseline": "#2563eb", "splus_10e6": "#dc2626", "alpha_100": "#16a34a"}
    for name, label in selected.items():
        h = histories[name]
        top = np.interp(timeline, h["time"], total_top_flux(h))/FSUN
        bottom = np.interp(timeline, h["time"], total_bottom_flux(h))/FSUN
        axes[0, 0].plot(timeline-24000, top, label=label, color=colors[name])
        axes[0, 1].plot(timeline-24000, bottom, label=label,
                        color=colors[name], alpha=.8)
        if name != "baseline":
            axes[1, 0].plot(timeline-24000, top-base_top, label=label,
                            color=colors[name])
            axes[1, 1].plot(timeline-24000, bottom-base_bottom, label=label,
                            color=colors[name])
    for ax, ylabel in zip(axes.flat, ("Top / F_sun", "Bottom / F_sun",
                                      "Top − baseline / F_sun",
                                      "Bottom − baseline / F_sun")):
        ax.set(xlabel="t − 24000 [code]", ylabel=ylabel)
        ax.grid(alpha=.3)
        ax.legend()
    fig.suptitle("WOR-74: matched 96 km branch responses", fontsize=20)
    fig.tight_layout()
    fig.savefig(args.output / "branch_response.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for name, case in summary["cases"].items():
        series = case["snapshots"]
        ts = [row["time_code"]-24000 for row in series]
        for ax, key in zip(axes, (
                "bottom_cell_entropy_contrast_cgs", "bottom_mean_temperature_K",
                "top_mean_radiation_energy_cgs")):
            ax.plot(ts, [row[key] for row in series], "o-",
                    label=case["label"], color=colors[name])
    for ax, ylabel in zip(axes, ("Bottom entropy contrast [cgs]",
                                      "Bottom mean T [K]", "Top E_rad [erg cm⁻³]")):
        ax.set(xlabel="t − 24000 [code]", ylabel=ylabel)
        ax.grid(alpha=.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "thermal_response.png", dpi=160)
    plt.close(fig)

    rows = []
    for i, (start, end) in enumerate(WINDOWS):
        for name, case in summary["cases"].items():
            row = case["windows"][i]
            rows.append(f"<tr><th>{start}–{end}</th><td>{html.escape(case['label'])}</td>"
                        f"<td>{row['top_over_solar']:.3f}</td>"
                        f"<td>{row['bottom_over_solar']:.3f}</td>"
                        f"<td>{row['stored_trend_over_solar']:+.3f}</td>"
                        f"<td>{row['delta_top_over_solar']:+.3f}</td>"
                        f"<td>{row['delta_bottom_over_solar']:+.3f}</td></tr>")
    snapshot_rows = []
    for name, case in summary["cases"].items():
        for snapshot in case["snapshots"]:
            snapshot_rows.append(
                f"<tr><th>{snapshot['time_code']:.1f}</th>"
                f"<td>{html.escape(case['label'])}</td>"
                f"<td>{snapshot['bottom_cell_up_entropy_cgs']/1e9:.6f}</td>"
                f"<td>{snapshot['bottom_cell_down_entropy_cgs']/1e9:.6f}</td>"
                f"<td>{snapshot['bottom_cell_entropy_contrast_cgs']/1e6:.3f}</td>"
                f"<td>{snapshot['bottom_mean_temperature_K']:.0f}</td>"
                f"<td>{snapshot['top_mean_radiation_energy_cgs']:.3f}</td></tr>")
    page = f"""<!doctype html><html lang="ja"><meta charset="utf-8">
<title>WOR-74: 96 km の長めの境界対照</title><style>
body{{font:21px/1.55 system-ui,sans-serif;max-width:1300px;margin:2rem auto;padding:0 1rem;color:#172033}}
h1{{font-size:2rem}}h2{{font-size:1.5rem}}strong{{color:#075985}}
table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #cbd5e1;padding:.6rem;text-align:right}}
th:first-child,td:nth-child(2){{text-align:left}}img{{max-width:100%;height:auto}}
.note{{background:#eff6ff;border-left:5px solid #0284c7;padding:1rem}}
</style><h1>WOR-74：96 km の境界対照</h1>
<p class="note"><strong>同一 t=24000 restart</strong>から、流入エントロピーを
変更。約310 code-time（鉛直横断約2.7回）の応答であり、
全箱熱時間約2万 code-timeの定常応答ではありません。</p>
<p>基準状態の上端放射エネルギーは約<strong>2.65 erg cm⁻³</strong>。
同じMarshak係数0.5でF☉を出す境界面の目安は
<strong>4.20 erg cm⁻³</strong>（放射温度約4855 K）です。
セル中心値と境界面値は同一ではないため、この数値は較正の目安です。</p>
<table><tr><th>時間窓</th><th>条件</th><th>上端 / F☉</th><th>下端 / F☉</th>
<th>蓄熱 / F☉</th><th>上端差</th><th>下端差</th></tr>{''.join(rows)}</table>
<img src="branch_response.png" alt="境界対照の流束時系列">
<h2>下端内側セルの熱状態</h2><p>最終3D出力の上昇・下降流エントロピーとその差。
セル中心診断なので下端面Godunov流束とは区別してください。</p>
<table><tr><th>t</th><th>条件</th><th>s↑ / 10⁹</th><th>s↓ / 10⁹</th>
<th>差 / 10⁶</th><th>下端 T / K</th><th>上端 E_rad / erg cm⁻³</th></tr>
{''.join(snapshot_rows)}</table>
<img src="thermal_response.png" alt="下端エントロピー差と上下層の温度の時間変化">
<p><a href="summary.json">JSON</a> には入出力のハッシュ、メモリ監視、流束収支と質量流束を収録。</p>
<p>どの枝でも上端が太陽値に到達し、かつ蓄熱が十分小さい状態を確認するまでは較正完了としません。</p>
</html>"""
    (args.output / "index.html").write_text(page)
    print(args.output / "index.html")


if __name__ == "__main__":
    main()
