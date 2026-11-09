#!/usr/bin/env python3
"""Build a readable WOR-74 index from completed local and shared diagnostics."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from solar_convection_report import (  # noqa: E402
    combine_histories, input_values, sampled_energy_ledger,
    total_bottom_flux, total_top_flux,
)

FSUN = 6.3e10


def window(h: dict[str, np.ndarray], left: int, right: int,
           units: dict[str, float]) -> dict:
    samples = np.flatnonzero((h["time"] >= left) & (h["time"] <= right))
    ledger = sampled_energy_ledger(h, samples, units, FSUN)
    return {"start_code": left, "end_code": right, "samples": len(samples),
            "top_over_solar": float(np.mean(total_top_flux(h)[samples]) / FSUN),
            "bottom_over_solar": float(np.mean(total_bottom_flux(h)[samples]) / FSUN),
            "stored_trend_over_solar": ledger["stored_energy_trend_over_solar"],
            "ledger_residual_over_solar": ledger["residual_equivalent_solar_flux"],
            "max_mass_drift": float(np.max(np.abs(h["MassDrift"][samples])))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--header-from", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sweep-report", type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    state = json.loads((args.campaign / "state.json").read_text())
    completed = state["completed"]
    last = completed[-1]
    last_segment = int(last["end_time_code"] // 100)
    if last_segment < 240:
        raise ValueError("t=24000 local campaign is not complete")
    paths = [args.campaign / f"feedback_{n:03d}_t{n*100}"
             / "simple_convection_general_6Mm_box.hst"
             for n in range(200, 241)]
    h = combine_histories(paths, args.header_from)
    units = input_values(args.campaign / "feedback_240_t24000/athinput.generated")
    local_windows = [window(h, a, b, units)
                     for a, b in ((20000, 22000), (22000, 24000))]
    entropy = json.loads((args.root / "bottom_entropy/summary.json").read_text())
    if not (args.root / "downflow_profiles/summary.json").exists():
        raise FileNotFoundError("Run the vertical-profile report before the overview")
    xd = json.loads((args.root / "xd_24km/summary.json").read_text())
    sweep = json.loads((args.sweep_report / "summary.json").read_text())
    recent = [row for row in completed if 20000 < row["end_time_code"] <= 24000]
    newton_ok = sum(row["newton_status_counts"].get("converged", 0) for row in recent)
    bad = sum(sum(count for status, count in row["newton_status_counts"].items()
                  if status != "converged") for row in recent)
    overview = {
        "local_last_complete_time_code": last["end_time_code"],
        "local_status": state["status"], "local_stop_reason": state.get("stop_reason"),
        "local_windows": local_windows,
        "local_recent_newton_converged": newton_ok,
        "local_recent_newton_bad": bad,
        "local_recent_slow_mg_warnings": sum(row["slow_mg_warning_count"] for row in recent),
        "local_last_s_in_cgs": last["entropy_cgs"],
        "entropy_late_minus_early": entropy["late_minus_early"],
        "xd_last_history_time_code": xd["xd_24km"]["last_history_time_code"],
        "xd_late_window": xd["xd_24km"]["windows"][-1],
        "sweep_common_window_code": sweep["common_window_code"],
        "sweep_delta_from_baseline": {
            key: row["difference_from_baseline"] for key, row in sweep["cases"].items()},
        "limits": ["t≈40000 local series is not present in this workspace",
                   "shared 24 km history is an early, non-stationary snapshot",
                   "10-code perturbations are transient, not equilibrium responses"],
    }
    (args.root / "summary.json").write_text(
        json.dumps(overview, indent=2, ensure_ascii=False) + "\n")

    rows = "".join(
        f"<tr><th>{r['start_code']}–{r['end_code']}</th>"
        f"<td>{r['top_over_solar']:.4f}</td>"
        f"<td>{r['bottom_over_solar']:.4f}</td>"
        f"<td>{r['stored_trend_over_solar']:+.4f}</td></tr>" for r in local_windows)
    early = entropy["early_window"]
    late = entropy["late_window"]
    ds = entropy["late_minus_early"]
    xw = xd["xd_24km"]["windows"][-1]
    plus = sweep["cases"]["splus_1e6"]["difference_from_baseline"]
    minus = sweep["cases"]["sminus_1e6"]["difference_from_baseline"]
    sweep_rel = "../wor74_coarse_sweep_20261109/report/index.html"
    page = f"""<!doctype html><html lang="ja"><meta charset="utf-8">
<title>WOR-74: 粗格子とXD共有データの診断</title><style>
body{{font:21px/1.55 system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#172033}}
h1{{font-size:2rem}}h2{{font-size:1.5rem}}strong{{color:#075985}}
table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #cbd5e1;padding:.6rem;text-align:right}}
th:first-child{{text-align:left}}img{{max-width:100%;height:auto}}
.note{{background:#eff6ff;border-left:5px solid #0284c7;padding:1rem}}
</style><h1>WOR-74：太陽対流の流束不足を切り分ける</h1>
<p class="note"><strong>現時点の判定：</strong>96 kmの完了済み系列は t={last['end_time_code']:.0f} までで、
上端・下端とも約0.63 F☉、蓄熱は小さい準熱平衡です。
XDの24 km共有データは t={xd['xd_24km']['last_history_time_code']:.1f} までの初期経過で、
上端は約0.56 F☉。24 kmの長時間定常値や解像度単独の効果はまだ判定できません。</p>
<h2>96 km：完了済み t=24000 まで</h2>
<table><tr><th>時間窓 [code]</th><th>上端 / F☉</th><th>下端 / F☉</th><th>蓄積傾向 / F☉</th></tr>
{rows}</table>
<p>直近40区間のNewton収束は {newton_ok:,} 件、非収束 {bad} 件、slow-MG警告
{overview['local_recent_slow_mg_warnings']} 件。質量ずれは各窓で0.03%以下。
この環境の state は <strong>{html.escape(state['status'])}</strong>（{html.escape(state.get('stop_reason',''))}）。
t≈40000の別環境データはこのworkspaceにありません。</p>
<h2>下降流と境界感度</h2>
<p>t={early['start_code']:.0f}–{early['end_code']:.0f} から
t={late['start_code']:.0f}–{late['end_code']:.0f} にかけて指定s<sub>in</sub>は
<strong>+{ds['input_entropy_cgs']:.0f}</strong> erg g⁻¹ K⁻¹、下端内側の上昇流・下降流は
+{ds['upflow_entropy_cgs']:.0f} / +{ds['downflow_entropy_cgs']:.0f}。
上下流の差は {early['entropy_contrast_cgs']/1e6:.3f}→{late['entropy_contrast_cgs']/1e6:.3f}×10⁶で増えていません。</p>
<p>同じ t=24000 checkpointの10 code-time試験では、s<sub>in</sub>を±10⁶変えると下端の正味流束は
{minus['bottom_total_over_solar']:+.3f} / {plus['bottom_total_over_solar']:+.3f} F☉、
上端は約±10⁻⁵ F☉。この区間は鉛直横断時間の約{100*sweep['window_fraction_of_vertical_crossing']:.1f}%で、
長期の上端感度ではありません。</p>
<h2>XD 24 km：共有された初期経過</h2>
<p>最新の1000–1103 code-time窓は上端 {xw['top_flux_over_solar']:.4f}、
下端 {xw['bottom_flux_over_solar']:.4f}、蓄積傾向 {xw['storage_trend_over_solar']:+.4f} F☉。
箱内はまだ冷却中です。24 kmはprofile由来s<sub>in</sub>固定、96 kmはfeedback実行中で、条件が異なります。</p>
<img src="xd_24km/early_flux_context.png" alt="XDとローカルの初期流束比較">
<h2>詳細</h2><ul>
<li><a href="bottom_entropy/index.html">下端エントロピーの時間変化</a></li>
<li><a href="downflow_profiles/index.html">下降流の高さ別プロファイル</a></li>
<li><a href="{sweep_rel}">同一restartの境界摂動試験</a></li>
<li><a href="xd_24km/index.html">XD 24 km共有HSTと96 km初期経過の比較</a></li>
<li><a href="summary.json">概要の数値JSON</a></li>
</ul></html>"""
    (args.root / "index.html").write_text(page)
    print(args.root / "index.html")


if __name__ == "__main__":
    main()
