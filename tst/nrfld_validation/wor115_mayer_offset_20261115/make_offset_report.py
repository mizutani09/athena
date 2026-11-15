#!/usr/bin/env python3
"""Generate a standalone timing comparison for Mayer offset controls."""

import argparse
import base64
import csv
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


AU = 1.495978707e13
YEAR = 365.25 * 86400
CODE_TO_KYR = AU / np.sqrt(1e-9 / 1e-18) / YEAR / 1000


def read_hst(paths):
    rows = {}
    for path in paths:
        with open(path) as stream:
            for line in stream:
                parts = line.split()
                if line.startswith("#") or len(parts) != 39:
                    continue
                row = np.array([float(v) for v in parts])
                old = rows.get(row[0])
                if old is not None and not np.array_equal(old, row):
                    raise ValueError(f"Conflicting history row at {row[0]} in {path}")
                rows[row[0]] = row
    if not rows:
        raise ValueError("No 39-column history data")
    return np.array([rows[t] for t in sorted(rows)])


def first_crossing(rows, threshold):
    rho = rows[:, 15]
    above = np.flatnonzero(rho >= threshold)
    if not above.size:
        return None
    i = int(above[0])
    if i == 0:
        return float(rows[i, 0] * CODE_TO_KYR)
    time = np.interp(np.log(threshold), np.log(rho[i - 1:i + 1]),
                     rows[i - 1:i + 1, 0])
    return float(time * CODE_TO_KYR)


def series_metrics(rows):
    m0 = rows[0, 22]
    return {
        "rows": int(len(rows)),
        "first_last_code_time": [float(rows[0, 0]), float(rows[-1, 0])],
        "last_time_kyr": float(rows[-1, 0] * CODE_TO_KYR),
        "last_rho_max_g_cm3": float(rows[-1, 15]),
        "mass_relative_drift_max": float(np.max(np.abs(rows[:, 22] / m0 - 1))),
        "pressure_min_dyn_cm2": float(np.min(rows[:, 29])),
        "jeans_min_cells": float(np.min(rows[:, 34])),
        "background_T_relative_error_max": float(np.max(rows[:, 36])),
        "background_T_error_rows_over_1percent": int(np.count_nonzero(rows[:, 36] > .01)),
        "threshold_times_kyr": {f"{v:.0e}": first_crossing(rows, v)
                                for v in [1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10]},
    }


def make_plot(series, path):
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    for label, rows, color, style in series:
        ax.semilogy(rows[:, 0] * CODE_TO_KYR, rows[:, 15],
                    color=color, linestyle=style, linewidth=2, label=label)
    ax.axvline(29.09, color="#656565", linestyle=":", label="Mayer 29.09 kyr")
    ax.axhline(1e-10, color="#7d4e3c", linestyle="--", linewidth=1)
    ax.set(xlabel="Physical time [kyr]", ylabel="Maximum density [g cm$^{-3}$]",
           xlim=(20, 38), ylim=(5e-18, 3e-10))
    ax.grid(alpha=.28, which="both")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def read_profile(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    inner = np.array([float(row["r_inner_au"]) for row in rows])
    outer = np.array([float(row["r_outer_au"]) for row in rows])
    radius = np.sqrt(inner * outer)
    radius[0] = outer[0] / 2
    return {
        "r": radius,
        "rho": np.array([float(row["mean_rho_g_cm3"]) for row in rows]),
        "T": np.array([float(row["mass_weighted_temp_K"]) for row in rows]),
    }


def steepest_density_drop_radius(profile):
    selected = (profile["r"] >= 2.0) & (profile["r"] <= 20.0)
    selected &= profile["rho"] > 0.0
    radius = profile["r"][selected]
    slope = np.gradient(np.log(profile["rho"][selected]), np.log(radius))
    return float(radius[np.argmin(slope)])


def profile_comparison(profiles, paper_path, path):
    with paper_path.open(newline="") as stream:
        paper = list(csv.DictReader(stream))
    paper_r = np.array([float(row["r_au"]) for row in paper])
    paper_rho = np.array([float(row["rho_est_g_cm3"]) for row in paper])
    paper_t = np.array([float(row["Tgas_est_K"]) for row in paper])
    colors = {"420 K validated": "#17627f", "14 K pressure trial": "#bd633b",
              "Post-AMR reset, same resolution": "#7f6495",
              "Late high resolution": "#3b7858"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), constrained_layout=True)
    for label, profile in profiles.items():
        for ax, field in zip(axes, ("rho", "T")):
            ax.plot(profile["r"], profile[field], color=colors[label],
                    linewidth=2, label=label)
    for ax, values in zip(axes, (paper_rho, paper_t)):
        ax.scatter(paper_r, values, facecolors="none", edgecolors="#7d4e8b",
                   s=45, label="Mayer Fig. 7 approximate read-off")
        ax.set(xscale="log", yscale="log", xlim=(.6, 300), xlabel="Radius [au]")
        ax.grid(alpha=.25, which="both")
    axes[0].set_ylabel("Density [g cm$^{-3}$]")
    axes[1].set_ylabel("Gas temperature [K]")
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    summary = {}
    for label, profile in profiles.items():
        summary[label] = {}
        for r, rho_ref, t_ref in zip(paper_r, paper_rho, paper_t):
            rho = float(np.interp(r, profile["r"], profile["rho"]))
            temp = float(np.interp(r, profile["r"], profile["T"]))
            summary[label][str(float(r))] = {
                "rho_g_cm3": rho, "T_K": temp,
                "rho_over_figure_read_off": rho / rho_ref,
                "T_over_figure_read_off": temp / t_ref,
            }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-history-dir", type=Path, required=True)
    parser.add_argument("--pressure-preflight-hst", type=Path, required=True)
    parser.add_argument("--pressure-main-hst", type=Path, required=True)
    parser.add_argument("--old-preflight-hst", type=Path, required=True)
    parser.add_argument("--old-main-hst", type=Path, required=True)
    parser.add_argument("--limiter-audit", type=Path, required=True)
    parser.add_argument("--gravity-audit", type=Path, required=True)
    parser.add_argument("--cell-mass-audit", type=Path, required=True)
    parser.add_argument("--tracer-audit", type=Path, required=True)
    parser.add_argument("--highres-audit", type=Path)
    parser.add_argument("--baseline-profile", type=Path)
    parser.add_argument("--pressure-profile", type=Path)
    parser.add_argument("--postamr-profile", type=Path)
    parser.add_argument("--highres-profile", type=Path)
    parser.add_argument("--mayer-read-off", type=Path)
    parser.add_argument("--pressure-solver", type=Path)
    parser.add_argument("--postamr-solver", type=Path)
    parser.add_argument("--highres-solver", type=Path)
    parser.add_argument("output_html", type=Path)
    args = parser.parse_args()
    baseline = read_hst(sorted(args.baseline_history_dir.glob("job_*.hst")))
    pressure = read_hst([args.pressure_preflight_hst, args.pressure_main_hst])
    old_all = read_hst([args.old_preflight_hst, args.old_main_hst])
    bad = np.flatnonzero((old_all[:, 29] <= 1e-18) | (old_all[:, 34] < 16))
    old = old_all[:int(bad[0])] if bad.size else old_all
    metrics = {"baseline_420K": series_metrics(baseline),
               "pressure_control_14K": series_metrics(pressure),
               "old_diagnostic_before_floor": series_metrics(old)}
    metrics["limiter_audit"] = json.loads(args.limiter_audit.read_text())
    metrics["gravity_audit"] = json.loads(args.gravity_audit.read_text())
    metrics["cell_mass_audit"] = json.loads(args.cell_mass_audit.read_text())
    metrics["tracer_audit"] = json.loads(args.tracer_audit.read_text())
    if args.highres_audit:
        metrics["highres_audit"] = json.loads(args.highres_audit.read_text())
    solver_paths = {"pressure": args.pressure_solver,
                    "postamr": args.postamr_solver,
                    "highres": args.highres_solver}
    metrics["solver_summary"] = {
        key: json.loads(path.read_text()) for key, path in solver_paths.items()
        if path is not None}
    out = args.output_html
    out.parent.mkdir(parents=True, exist_ok=True)
    plot = out.with_name(out.stem + "_timing.png")
    make_plot([
        ("Validated 420 K ambient, pressure equilibrium", baseline, "#17627f", "-"),
        ("Only ambient pressure reduced: 14 K", pressure, "#bd633b", "-"),
        ("Earlier 14 K path (diagnostic, before floor)", old, "#817792", "--"),
    ], plot)
    times = ["1e-16", "1e-15", "1e-14", "1e-13", "1e-12", "1e-11", "1e-10"]
    rows_html = ""
    for value in times:
        row = [value]
        for name in ("baseline_420K", "pressure_control_14K", "old_diagnostic_before_floor"):
            time = metrics[name]["threshold_times_kyr"][value]
            row.append("—" if time is None else f"{time:.4f}")
        rows_html += "<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"
    pressure_m = metrics["pressure_control_14K"]
    core_limiter = metrics["limiter_audit"]["bands_au"]["0-7"]
    dense_mass = metrics["cell_mass_audit"]["densest_cell"]
    image = base64.b64encode(plot.read_bytes()).decode("ascii")
    highres_html = ""
    if args.highres_audit:
        highres_html = "<section><h2>追加解像度試験</h2><pre>" + html.escape(
            json.dumps(metrics["highres_audit"], indent=2)) + "</pre></section>"
    profile_html = ""
    if args.baseline_profile and args.mayer_read_off:
        profiles = {"420 K validated": read_profile(args.baseline_profile)}
        if args.pressure_profile:
            profiles["14 K pressure trial"] = read_profile(args.pressure_profile)
        if args.postamr_profile:
            profiles["Post-AMR reset, same resolution"] = read_profile(args.postamr_profile)
        if args.highres_profile:
            profiles["Late high resolution"] = read_profile(args.highres_profile)
        metrics["shock_radius_steepest_density_drop_au"] = {
            label: steepest_density_drop_radius(profile)
            for label, profile in profiles.items()
        }
        profile_plot = out.with_name(out.stem + "_profile.png")
        metrics["profile_comparison"] = profile_comparison(
            profiles, args.mayer_read_off, profile_plot)
        profile_image = base64.b64encode(profile_plot.read_bytes()).decode("ascii")
        ratio_rows = ""
        for radius in ("1.0", "3.0", "5.0", "10.0", "30.0", "100.0"):
            item = metrics["profile_comparison"]["420 K validated"][radius]
            ratio_rows += (f"<tr><td>{radius}</td>"
                           f"<td>{item['rho_over_figure_read_off']:.3f}</td>"
                           f"<td>{item['T_over_figure_read_off']:.3f}</td></tr>")
        profile_html = ("<figure><img src=\"data:image/png;base64," + profile_image
                        + "\" alt=\"密度・温度の半径プロファイル比較\"><figcaption>"
                        + "基準・背景圧力・AMR後reset曲線は最初にρmax≥1e−10 g/cm³に達した状態。"
                        + "高解像度曲線は閾値到達後の最寄り保存区間終端（ρmaxは閾値より0.8%高い）。"
                        + "Mayer点はFig. 7からの概読値であり数値データではない。"
                        + "高解像度試験はc1420からの継続のみで、全経路の解像度収束試験ではない。"
                        + "</figcaption></figure><section><h2>見かけの倍率差</h2>"
                        + f"<p>基準計算の密度の最急減少位置は"
                        + f"{metrics['shock_radius_steepest_density_drop_au']['420 K validated']:.2f} au。"
                        + "Mayerが述べる第一コア境界6–7 auと整合する。</p>"
                        + "<p>420 K検証経路とFigure 7概読点の比は半径で変わる。"
                        + "単一のEoS定数または単位換算倍率では説明できない。</p>"
                        + "<table><tr><th>半径 [au]</th><th>密度比</th><th>温度比</th></tr>"
                        + ratio_rows + "</table></section>")
        if args.postamr_profile:
            effect = {}
            effect_rows = ""
            for radius in ("1.0", "3.0", "5.0", "10.0", "30.0", "100.0"):
                base = metrics["profile_comparison"]["420 K validated"][radius]
                fixed = metrics["profile_comparison"][
                    "Post-AMR reset, same resolution"][radius]
                effect[radius] = {"rho_ratio": fixed["rho_g_cm3"] / base["rho_g_cm3"],
                                  "T_ratio": fixed["T_K"] / base["T_K"]}
                effect_rows += (f"<tr><td>{radius}</td>"
                                f"<td>{effect[radius]['rho_ratio']:.5f}</td>"
                                f"<td>{effect[radius]['T_ratio']:.5f}</td></tr>")
            metrics["postamr_effect_at_radii"] = effect
            profile_html += ("<section><h2>AMR後resetのみの影響</h2>"
                             "<p>同じc1420から解像度を変えずに継続した結果を"
                             "既存の検証済み経路で割った値。</p><table>"
                             "<tr><th>半径 [au]</th><th>密度比</th><th>温度比</th></tr>"
                             + effect_rows + "</table></section>")
        if args.highres_profile:
            effect = {}
            effect_rows = ""
            for radius in ("1.0", "3.0", "5.0", "10.0", "30.0", "100.0"):
                base = metrics["profile_comparison"]["420 K validated"][radius]
                refined = metrics["profile_comparison"]["Late high resolution"][radius]
                effect[radius] = {
                    "rho_ratio": refined["rho_g_cm3"] / base["rho_g_cm3"],
                    "T_ratio": refined["T_K"] / base["T_K"],
                }
                effect_rows += (f"<tr><td>{radius}</td>"
                                f"<td>{effect[radius]['rho_ratio']:.5f}</td>"
                                f"<td>{effect[radius]['T_ratio']:.5f}</td></tr>")
            metrics["highres_effect_at_radii"] = effect
            max_t_difference = max(abs(item["T_ratio"]-1) for item in effect.values())
            mass = metrics["highres_audit"]["densest_cell"] if args.highres_audit else None
            mass_note = (f"最大密度セルの質量は{mass['mass_msun']:.3e} M☉、"
                         f"Mayer目標の{mass['mass_over_mayer_target']:.2f}倍。"
                         if mass else "")
            profile_html += ("<section><h2>後半だけ解像度を上げた影響</h2>"
                             f"<p>c1420からJeans目標32→48、最大AMR level 12→13。"
                             f"1–100 auの温度差は最大{max_t_difference:.2%}。"
                             + mass_note
                             + "この試験は後半だけの変更なので、初期からの解像度依存性は未評価。</p>"
                             + "<table><tr><th>半径 [au]</th><th>密度比</th><th>温度比</th></tr>"
                             + effect_rows + "</table></section>")
    solver_html = ""
    if metrics["solver_summary"]:
        solver_rows = ""
        for name, item in metrics["solver_summary"].items():
            solver_rows += (f"<tr><td>{html.escape(name)}</td>"
                            f"<td>{item['newton_converged']}/{item.get('newton_active_total', item['newton_total'])}</td>"
                            f"<td>{item['fgmres_iterations_total']}</td>"
                            f"<td>{item['fgmres_iterations_max']}</td>"
                            f"<td>{item['fgmres_true_relative_max']:.2e}</td>"
                            f"<td>{item['elapsed_seconds']/60:.1f}</td>"
                            f"<td>{item['maxrss_kib']/1024/1024:.1f}</td>"
                            f"<td>{html.escape(item['slurm_state'])}</td></tr>")
        solver_html = ("<section><h2>ソルバと計算資源</h2><table><tr>"
                       "<th>試験</th><th>Newton収束</th><th>FGMRES総反復</th>"
                       "<th>1 solve最大反復</th><th>最大true相対残差</th>"
                       "<th>時間 [分]</th><th>MaxRSS [GiB]</th><th>Slurm</th>"
                       "</tr>" + solver_rows + "</table></section>")
    document = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>WOR-115 Mayer §5 offset isolation</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;max-width:1040px;margin:auto;padding:1.4rem;background:#f3f7f9;color:#263743}}section,figure{{background:white;padding:1rem;margin:1rem 0;border:1px solid #dce5e8;border-radius:8px}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #dde7e8;padding:.4rem;text-align:left}}a{{color:#17627f}}</style></head><body>
<h1>WOR-115: Mayer §5との時刻・プロファイル差の切り分け</h1>
<p>同じAthena++バイナリと初期球を用い、背景圧力だけを変更した感度試験。
理想気体のため背景温度とその後のgas温度固定先も同時に420 Kから14 Kへ変わる。
初期圧力の力学作用と継続的な熱固定の寄与は、この試験だけでは分離できない。
14 K背景はMayer設定として確定した値ではない。Mayer §5と引用元§3.1からは
密度1/30・初期圧力平衡が示唆されるため、物理比較の基準は420 K背景の検証済み経路。</p>
<figure><img src="data:image/png;base64,{image}" alt="最大密度の時刻履歴"><figcaption>旧14 K経路は最初の圧力floor発生前で打ち切った診断用曲線。</figcaption></figure>
<section><h2>最大密度の閾値到達時刻 [kyr]</h2><p>前後の受理履歴行をlogρで内挿した推定値。</p><table><tr><th>ρ [g/cm³]</th><th>420 K・圧力平衡</th><th>背景圧力のみ低下</th><th>旧14 K診断経路</th></tr>{rows_html}</table>
<p>背景圧力対照の最終時刻 {pressure_m['last_time_kyr']:.4f} kyr、ρmax={pressure_m['last_rho_max_g_cm3']:.4e} g/cm³。
質量相対変化最大 {pressure_m['mass_relative_drift_max']:.3e}、Pmin={pressure_m['pressure_min_dyn_cm2']:.3e} dyn/cm²、Jeans最小={pressure_m['jeans_min_cells']:.2f} cells。</p></section>
<section><h2>背景温度の例外</h2><p>背景圧力対照では再細分化直後の背景温度相対誤差が最大{pressure_m['background_T_relative_error_max']:.3%}となる受理行が1行あった。他の行は1%以内。時刻比較はこの数値上の例外を含む近似的な感度結果として扱う。</p></section>
<section><h2>AMR後resetの局所修正</h2><p>再細分化後に作られた背景セルへ既存の温度固定を再適用した。元の検証済み履歴には背景温度誤差が1%を超える受理行が3行ある。c0→c10の修正試験では最初の例外が丸め誤差に減った。同じc1420状態から第一コアまで解像度を変えずに進めた試験では、背景温度誤差は最大7.77e−16で、最終c1560の密度・圧力・速度・放射エネルギー・tracerが基準と全セルでビット単位に一致した。したがって後半の修正は安全だが、c0からの全履歴への影響は未測定。</p></section>
<section><h2>他の原因候補</h2><p>初期一様球はμ=2.381、γ=5/3、κP=κR=1 cm²/g、ρ=5.2515854e−18 g/cm³、α=Ethermal/|Egrav|=0.41330。Mayer §5の公表値と精度内で一致する。初期重力加速度は解析値と約0.05%以内（5–95%範囲）。
最終7 au以内でLP/Minerbo limiter係数の相対差は最大 {100*core_limiter['max']:.3f}%（同じスナップショット上の局所評価）。
最大密度セルの質量は {dense_mass['mass_msun']:.3e} M☉、Mayerの目標の{dense_mass['mass_over_mayer_target']:.2f}倍。</p>
<p>最終snapshotの1000 au以内には背景tracer≥0.5のセルがなく、100 au以内の最大tracerは
{metrics['tracer_audit']['bands_au']['100']['max_tracer']:.2e}。420 Kの背景温度固定が第一コア内部に直接適用されているわけではない。</p>
<p>第一コアの最終状態でgas/放射温度は1 auで522.07/522.06 K、3 auで384.15/383.70 K、5 auで236.08/234.79 Kと近い。最終時点のガス放射非平衡だけではMayer温度差を説明できない。</p>
<p>同じ最終状態の球殻平均と+x軸の温度差は1 auと10 auで0.5%未満。抽出方法だけで図の大きな差は説明できない。</p>
<p>背景放射14 Kの毎step固定を外す試験はc0→10に成功したが、続くc26付近で放射温度57.38 K、AMRレベル7、64 GiBメモリ超過となり終了。
これはMayerの放射境界が未記載であることへの感度試験であり、物理的な一致/不一致の判定には使わない。</p></section>
<section><h2>背景放射を420 Kにした短時間感度試験</h2><p>背景ガス420 Kに放射場も420 Kを設定した試験では、AMR後resetを有効にしてもc1で最高温度が441 Kになり、c2の途中で64 GiBを超過した。c2の受理状態はない。高コストな継続計算は行わない。この420 K放射浴がMayerの条件であるとは確認できていない。</p></section>
{highres_html}
{profile_html}
{solver_html}
<section><h2>再現情報</h2><p>バイナリSHA-256 b64491aa31da79216c99dbfcc6eb2b9cba805e4c385a4a67d092c303a622b5cb。
背景圧力入力SHA-256 5bd3e640c4a731ac8d87cfd825845db995c460a23be4f1de68ccae78da99d5fd。
XD jobs 340378/340381、放射境界感度 jobs 340384/340386。
原著 <a href="https://arxiv.org/html/2510.12620#S5">Mayer et al. §5</a>、
<a href="https://arxiv.org/html/2410.07016#S3">引用元§3.1</a>。
原著は比較用の数値データを公開しておらず、合理的な依頼に応じて共有すると記載している。したがってFigure 7概読値は厳密な誤差評価に使わない。
数値表は隣接するJSONに保存。</p></section></body></html>"""
    out.write_text(document)
    out.with_name(out.stem + "_summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(out)
    print(json.dumps({k: metrics[k] for k in ("baseline_420K", "pressure_control_14K",
                                             "old_diagnostic_before_floor")}, indent=2))


if __name__ == "__main__":
    main()
