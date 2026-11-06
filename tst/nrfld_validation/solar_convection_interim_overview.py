#!/usr/bin/env python3
"""Summarize a completed prefix of a solar entropy-feedback continuation."""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--combined-summary", type=Path, required=True)
    parser.add_argument("--through-time", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    state = json.loads(args.state.read_text())
    report = json.loads(args.combined_summary.read_text())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = [row for row in state["completed"]
            if state["control_switch"]["at_time_code"] < row["end_time_code"]
            <= args.through_time]
    if len(rows) < 2 or rows[-1]["end_time_code"] != args.through_time:
        parser.error("The requested end time is not a completed segment")
    if abs(report["last_history_time_code"] - args.through_time) > 1e-8:
        parser.error("Combined report and requested snapshot time differ")

    base = state["config"]["initial_entropy_cgs"]
    cap = state["config"]["max_total_entropy_change_cgs"]
    t = [row["end_time_code"] for row in rows]
    plt.rcParams.update({"font.size": 14})
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                             constrained_layout=True)
    axes[0].plot(t, [row["top_radiative_flux_over_solar"] for row in rows],
                 lw=1, alpha=.55, label="top: segment mean")
    axes[0].plot(t, [row["rolling_top_flux_over_solar"] for row in rows],
                 lw=2.5, label="top: 500-code mean")
    observed_top = ([row["top_radiative_flux_over_solar"] for row in rows]
                    + [row["rolling_top_flux_over_solar"] for row in rows])
    axes[0].set_ylim(min(observed_top)-.003, max(observed_top)+.003)
    axes[0].text(.98, .96, "Solar target = 1.0 (above plot)",
                 transform=axes[0].transAxes, ha="right", va="top")
    axes[0].set_ylabel("Top flux / solar")
    axes[0].legend(loc="best")
    axes[1].plot(t, [row["rolling_bottom_total_flux_over_solar"] for row in rows],
                 lw=2, color="#1769a5", label="bottom total: 500-code mean")
    axes[1].set_ylabel("Bottom flux / solar")
    storage_axis = axes[1].twinx()
    storage_axis.plot(t, [row["rolling_stored_energy_trend_over_solar"]
                          for row in rows], lw=2, color="#d36b18",
                      label="stored energy trend: 500-code window")
    storage_axis.set_ylabel("Storage trend / solar")
    handles1, labels1 = axes[1].get_legend_handles_labels()
    handles2, labels2 = storage_axis.get_legend_handles_labels()
    axes[1].legend(handles1+handles2, labels1+labels2, loc="best")
    axes[2].plot(t, [(row["entropy_cgs"]-base)/1e6 for row in rows], lw=2)
    axes[2].axhline(cap/1e6, ls="--", color="#b44523",
                    label="cumulative entropy limit")
    axes[2].set_ylabel("Input entropy change\n[$10^6$ erg g$^{-1}$ K$^{-1}$]")
    axes[2].set_xlabel("Code time")
    axes[2].legend(loc="best")
    for axis in axes:
        axis.grid(alpha=.25)
    fig.savefig(output / "controller.png", dpi=150)
    plt.close(fig)

    early = report["early_late_window"]
    late = report["late_window"]
    ledger = report["sampled_energy_ledger_full"]
    log = report["log"]
    previous = state.get("forked_from")
    previous_report = (Path(previous) / "combined" / "index.html") if previous else None
    previous_link = (f"<a href='{html.escape(os.path.relpath(previous_report, output))}'>"
                     "t=0～5000 の結果</a>" if previous_report and
                     previous_report.is_file() else "")
    movie_cards = []
    movie_manifest = output / "movies" / "manifest.json"
    if movie_manifest.is_file():
        movies = json.loads(movie_manifest.read_text()).get("movies", {})
        for key, label in (("xz", "鉛直 xz 面と τ=1 面"),
                           ("xy", "水平 xy 面の鉛直速度"),
                           ("xy_fluct", "水平面平均を除いた鉛直速度")):
            item = movies.get(key)
            if item and (output / "movies" / item["file"]).is_file():
                movie_cards.append(
                    f"<figure><video controls preload='metadata' "
                    f"poster='movies/{html.escape(item['poster'])}'>"
                    f"<source src='movies/{html.escape(item['file'])}' "
                    "type='video/mp4'></video>"
                    f"<figcaption>{label}（{item['first_time_code']:.0f}～"
                    f"{item['last_time_code']:.0f} コード時間）</figcaption></figure>")
    if not movie_cards:
        movie_cards.append("<p>動画は生成中です。後で再読み込みしてください。</p>")

    def row(label: str, item: dict) -> str:
        return (f"<tr><td>{label}</td><td>{item['start_time']:.0f}～"
                f"{item['end_time']:.0f}</td>"
                f"<td>{item['top_total_flux_over_solar']:.4f}</td>"
                f"<td>{item['bottom_hydro_flux_over_solar']:.4f}</td>"
                f"<td>{item['stored_energy_trend_over_solar']:+.4f}</td>"
                f"<td>{item['vertical_crossing_times']:.2f}</td></tr>")

    last = rows[-1]
    offset = last["entropy_cgs"] - base
    cap_remaining = cap - (last["next_entropy_cgs"] - base)
    generated = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime(
        "%Y-%m-%d %H:%M JST")
    verdict = ("太陽流束の基準は未達" if not report["stationarity_candidate"]
               else "定常性の暫定基準を通過")
    page = f"""<!doctype html><html lang='ja'><meta charset='utf-8'>
<title>太陽対流・流入エントロピー制御の途中経過</title>
<style>body{{font:22px/1.55 system-ui;max-width:1250px;margin:2rem auto;
padding:0 1rem;color:#193047}}h1{{font-size:2rem}}h2{{font-size:1.5rem}}
strong{{color:#174c78}}.status{{font-weight:bold;color:#a32923}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:.55rem .7rem;
border-bottom:1px solid #ccc;text-align:right}}th:first-child,td:first-child{{text-align:left}}
img,video{{max-width:100%;width:100%}}figure{{margin:1.5rem 0}}
figcaption{{font-weight:bold}}</style>
<h1>太陽対流・流入エントロピー制御の途中経過</h1>
<p><strong>確定済みの t=5000～{args.through_time:.0f} を解析</strong>。
作成時刻 {generated}。計算は t=20000 を目標に継続中です。</p>
<p class='status'>{verdict}：直近の上端流束は約
{late['top_total_flux_over_solar']:.3f} F<sub>sun</sub> で、合格範囲
0.9～1.1 F<sub>sun</sub> には届いていません。</p>
<p>直近2つの1000コード時間窓では、流束と蓄熱率は以下の通りです。
各窓は約8.7回の鉛直横断時間に相当します。</p>
<table><thead><tr><th>窓</th><th>コード時間</th><th>上端 / Fsun</th>
<th>下端 hydro / Fsun</th><th>蓄熱傾き / Fsun</th><th>鉛直横断回数</th>
</tr></thead><tbody>{row('前半', early)}{row('後半', late)}</tbody></table>
<p>全区間の蓄熱率は {ledger['stored_energy_trend_over_solar']:+.4f} F<sub>sun</sub>、
収支残差は {ledger['residual_equivalent_solar_flux']:.2g} F<sub>sun</sub>。
Newton法は {log['nr_status_counts'].get('converged', 0):,} 回すべて収束し、
slow-MG警告 {log['slow_mg_warning_count']} 件、fatal {log['fatal_count']} 件です。</p>
<p>流入エントロピーはこの区間末で初期値から +{offset/1e6:.3f}×10<sup>6</sup>
erg g<sup>−1</sup> K<sup>−1</sup>。次の更新を含めても、累積上限まで
{cap_remaining/1e6:.3f}×10<sup>6</sup> 残っています。
上端流束への長期応答はまだ確定していません。</p>
<h2>流束・蓄熱・制御の履歴</h2><img src='controller.png' alt='Controller history'>
<h2>計算の空間分布</h2>{''.join(movie_cards)}
<h2>詳細</h2><p><a href='index.html'>結合した収支判定と各区間へのリンク</a> ·
<a href='history.png'>全履歴の図</a> ·
<a href='energy_budget.png'>エネルギー収支の図</a> ·
<a href='../index.html'>現在の実行状況</a> · {previous_link}</p>
</html>"""
    (output / "overview.html").write_text(page)
    print(output / "overview.html")


if __name__ == "__main__":
    main()
