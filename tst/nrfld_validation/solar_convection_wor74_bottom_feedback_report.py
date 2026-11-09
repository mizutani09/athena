#!/usr/bin/env python3
"""Summarize a completed 96 km lower-flux entropy-controller pilot."""

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
    combine_histories, input_values, sampled_energy_ledger,
    total_bottom_flux, total_top_flux,
)

FSUN = 6.3e10
THERMAL_TIME_CODE = 20600.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(h: dict[str, np.ndarray], y: np.ndarray,
         start: float, end: float) -> float:
    t = h["time"]
    inside = (t > start) & (t < end)
    xs = np.r_[start, t[inside], end]
    ys = np.interp(xs, t, y)
    return float(np.trapezoid(ys, xs)/(end-start))


def flux_window(h: dict[str, np.ndarray], units: dict[str, float],
                start: int, end: int) -> dict:
    if h["time"][0] > start or h["time"][-1] < end:
        raise ValueError(f"History does not cover {start}–{end}")
    indices = np.flatnonzero((h["time"] >= start) & (h["time"] <= end))
    ledger = sampled_energy_ledger(h, indices, units, FSUN)
    if ledger is None:
        raise ValueError(f"Missing energy ledger for {start}–{end}")
    return {
        "start_code": start, "end_code": end,
        "top_over_solar": mean(h, total_top_flux(h), start, end)/FSUN,
        "bottom_over_solar": mean(h, total_bottom_flux(h), start, end)/FSUN,
        "stored_trend_over_solar": ledger["stored_energy_trend_over_solar"],
        "sampled_ledger_residual_over_solar":
            ledger["residual_equivalent_solar_flux"],
        "max_mass_drift": float(np.max(np.abs(h["MassDrift"][indices]))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state_path = args.state.resolve()
    root = state_path.parent
    state = json.loads(state_path.read_text())
    if state["status"] != "completed" or state["config"].get("feedback_face") != "bottom":
        raise ValueError("Expected a completed bottom-flux pilot")
    completed = state["completed"]
    switch = state["control_switch"]["at_time_code"]
    end = completed[-1]["end_time_code"]
    if switch != 24000 or end != 26000:
        raise ValueError(f"Unexpected study interval: {switch}–{end}")
    if any(row["slow_mg_warning_count"] or
           set(row["newton_status_counts"]).difference({"converged"})
           for row in completed if row["end_time_code"] > switch):
        raise ValueError("Pilot has nonconvergent or slow-MG segments")
    header = root / completed[0]["case"] / completed[0]["history_name"]
    selected = [row for row in completed if row["end_time_code"] >= 22000]
    paths = [root / row["case"] / row["history_name"] for row in selected]
    history = combine_histories(paths, header)
    units = input_values(root / "athinput.campaign")
    windows = [flux_window(history, units, a, b) for a, b in (
        (22000, 23000), (23000, 24000),
        (24000, 24500), (24500, 25000),
        (25000, 25500), (25500, 26000),
        (24000, 25000), (25000, 26000))]
    by_interval = {(r["start_code"], r["end_code"]): r for r in windows}
    early = by_interval[(23000, 24000)]
    late = by_interval[(25000, 26000)]
    profiles = json.loads(args.profiles.read_text())
    before = profiles["windows"]["early"]["mean"]
    after = profiles["windows"]["late"]["mean"]
    new_rows = [r for r in completed if r["end_time_code"] > switch]
    switch_row = next(r for r in completed if r["end_time_code"] == switch)
    summary = {
        "state": str(state_path), "state_sha256": sha256(state_path),
        "profiles": str(args.profiles.resolve()),
        "profiles_sha256": sha256(args.profiles),
        "status": state["status"], "switch_time_code": switch,
        "end_time_code": end, "thermal_time_code_estimate": THERMAL_TIME_CODE,
        "elapsed_fraction_of_thermal_time": (end-switch)/THERMAL_TIME_CODE,
        "windows": windows,
        "late_minus_before": {
            name: late[name]-early[name] for name in (
                "top_over_solar", "bottom_over_solar", "stored_trend_over_solar")},
        "entropy": {
            "initial_at_switch_cgs": switch_row["entropy_cgs"],
            "last_applied_cgs": completed[-1]["entropy_cgs"],
            "next_if_continued_cgs": completed[-1]["next_entropy_cgs"],
            "bottom_up_change_cgs": after["up_entropy_cgs"][0]-before["up_entropy_cgs"][0],
            "bottom_down_change_cgs": after["down_entropy_cgs"][0]-before["down_entropy_cgs"][0],
            "bottom_contrast_before_cgs": before["entropy_contrast_cgs"][0],
            "bottom_contrast_after_cgs": after["entropy_contrast_cgs"][0],
            "bottom_temperature_change_K":
                after["mean_temperature_K"][0]-before["mean_temperature_K"][0],
            "top_temperature_change_K":
                after["mean_temperature_K"][-1]-before["mean_temperature_K"][-1],
        },
        "solver": {
            "newton_converged": sum(r["newton_status_counts"].get("converged", 0)
                                    for r in new_rows),
            "slow_mg_warnings": sum(r["slow_mg_warning_count"] for r in new_rows),
            "max_mass_drift": max(r["max_mass_drift"] for r in new_rows),
            "min_available_mib": state["min_available_memory_mib"],
            "peak_rank_rss_mib": state["peak_rank_rss_mib"],
            "peak_swap_mib": state["peak_swap_used_mib"],
        },
        "limits": [
            "The controller changes s_in; early and late windows are not a fixed-entropy matched control.",
            "Only 2000 code times, about 0.10 whole-box thermal time, elapsed after the switch.",
            "The sampled HST energy ledger is a diagnostic, not an exact stage-integrated conservation test.",
        ],
    }
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    plt.rcParams.update({"font.size": 18, "axes.labelsize": 20,
                         "legend.fontsize": 16})
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True,
                             constrained_layout=True)
    t = history["time"]
    recent = (t >= 22000) & (t <= 26000)
    thin = np.flatnonzero(recent)[::50]
    axes[0].plot(t[thin], total_top_flux(history)[thin]/FSUN,
                 label="top", color="#2563eb")
    bottom_flux = total_bottom_flux(history)
    axes[0].plot(t[thin], bottom_flux[thin]/FSUN,
                 label="bottom samples", color="#dc2626", alpha=.28)
    rolling_t = np.arange(22500, 26000.1, 10)
    rolling_bottom = [mean(history, bottom_flux, x-500, x)/FSUN
                      for x in rolling_t]
    axes[0].plot(rolling_t, rolling_bottom,
                 label="bottom, 500-code mean", color="#b91c1c", lw=2.5)
    axes[0].axhline(1, color="black", ls="--", label="solar")
    axes[0].axvline(switch, color="gray", ls=":")
    axes[0].set_ylabel("Flux / F_sun")
    axes[0].legend()
    active_rows = [r for r in completed if r["end_time_code"] >= 22000]
    axes[1].plot([r["end_time_code"] for r in active_rows],
                 [(r["entropy_cgs"]-state["config"]["initial_entropy_cgs"])/1e6
                  for r in active_rows], "o-", ms=3, color="#16a34a")
    axes[1].axvline(switch, color="gray", ls=":")
    axes[1].set(xlabel="Code time", ylabel="Inflow entropy offset / 10⁶ cgs")
    for ax in axes:
        ax.grid(alpha=.3)
    fig.savefig(out / "feedback_fluxes.png", dpi=100)
    plt.close(fig)

    rows = "".join(
        f"<tr><th>{w['start_code']}–{w['end_code']}</th>"
        f"<td>{w['top_over_solar']:.4f}</td>"
        f"<td>{w['bottom_over_solar']:.4f}</td>"
        f"<td>{w['stored_trend_over_solar']:+.4f}</td>"
        f"<td>{w['sampled_ledger_residual_over_solar']:+.5f}</td></tr>"
        for w in windows)
    e = summary["entropy"]
    solver = summary["solver"]
    page = f"""<!doctype html><html lang="ja"><meta charset="utf-8">
<title>WOR-74: 96 km 下端流束制御の t=26000 評価</title><style>
body{{font:21px/1.55 system-ui,sans-serif;max-width:1300px;margin:2rem auto;padding:0 1rem;color:#172033}}
h1{{font-size:2rem}}h2{{font-size:1.5rem}}strong{{color:#075985}}
table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #cbd5e1;padding:.6rem;text-align:right}}
th:first-child{{text-align:left}}img{{max-width:100%;height:auto}}
.note{{background:#eff6ff;border-left:5px solid #0284c7;padding:1rem}}
</style><h1>WOR-74：96 km下端流束制御の評価</h1>
<p class="note"><strong>計算はt=26000まで正常完了。太陽流束は未達。</strong>
直近1000 code-timeの上端は<strong>{late['top_over_solar']:.3f} F☉</strong>、
下端は<strong>{late['bottom_over_solar']:.3f} F☉</strong>、
蓄熱率は<strong>{late['stored_trend_over_solar']:+.3f} F☉</strong>。
下端への供給を増やした分は主に箱に蓄積し、上端はまだ追随していません。</p>
<p>t=24000からの経過は全箱熱時間の約{100*summary['elapsed_fraction_of_thermal_time']:.1f}%です。
固定エントロピーの同時刻対照ではなく、前後比較です。</p>
<img src="feedback_fluxes.png" alt="上下流束と流入エントロピーの推移">
<h2>流束と収支</h2>
<table><tr><th>時間 [code]</th><th>上端 / F☉</th><th>下端 / F☉</th>
<th>蓄熱 / F☉</th><th>標本台帳残差 / F☉</th></tr>{rows}</table>
<h2>下端の熱状態</h2>
<p>前後の3D snapshot各21枚をEOS表で解析すると、下端セルの上昇流・下降流エントロピーは
それぞれ<strong>+{e['bottom_up_change_cgs']:.0f}</strong>、
<strong>+{e['bottom_down_change_cgs']:.0f}</strong> erg g⁻¹ K⁻¹。
差は{e['bottom_contrast_before_cgs']/1e6:.3f}→
{e['bottom_contrast_after_cgs']/1e6:.3f}×10⁶とほぼ一定です。
下端平均温度は{e['bottom_temperature_change_K']:+.1f} K、
上端平均温度は{e['top_temperature_change_K']:+.1f} K。
<a href="downflow_profiles/index.html">鉛直プロファイル</a>を参照。</p>
<p>新しい20区間のNewton収束は{solver['newton_converged']:,}件、遅いMG警告
{solver['slow_mg_warnings']}件、質量ずれ最大{100*solver['max_mass_drift']:.3f}% 、
利用可能メモリ最低{solver['min_available_mib']/1024:.2f} GiB。
<a href="summary.json">数値JSONと根拠</a> ·
<a href="../index.html">実行全体</a></p>
</html>"""
    (out / "index.html").write_text(page)
    print(out / "index.html")


if __name__ == "__main__":
    main()
