#!/usr/bin/env python3
"""Make a standalone ideal-EOS collapse comparison report from validated CSVs."""

import base64
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent


def read(name):
    with (ROOT / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def profile(name):
    rows = read(name)
    inner = np.array([float(x["r_inner_au"]) for x in rows])
    outer = np.array([float(x["r_outer_au"]) for x in rows])
    r = np.sqrt(inner * outer)
    r[0] = outer[0]/2
    return r, *[np.array([float(x[col]) for x in rows]) for col in (
        "mean_rho_g_cm3", "mass_weighted_temp_K", "mass_weighted_vrad_km_s")]


def make_plot(out):
    ideal = profile("wor118_ideal_final_radial.csv")
    table = profile("c1080_wor118_full_profile.csv")
    ideal_earlier = profile("wor118_ideal_c1090_radial.csv")
    table_earlier = profile("wor118_table_c1060_radial.csv")
    paper = read("mayer_fig7_approx_read_off.csv")
    paper_r = np.array([float(x["r_au"]) for x in paper])
    paper_values = [np.array([float(x[col]) for x in paper]) for col in (
        "rho_est_g_cm3", "Tgas_est_K", "vr_est_km_s")]
    history = read("wor118_ideal_mayer5_history.csv")
    t = np.array([float(x["time_kyr"]) for x in history])
    hrho = np.array([float(x["rho_max_g_cm3"]) for x in history])
    ht = np.array([float(x["Tgas_max_K"]) for x in history])
    mass = np.array([float(x["mass_rel_from_c0"]) for x in history])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, field, ylab in ((axes[0, 0], 1, r"Density [g cm$^{-3}$]"),
                            (axes[0, 1], 2, "Gas temperature [K]"),
                            (axes[1, 0], 3, "Infall speed [km/s]")):
        transform = (lambda y: -y) if field == 3 else (lambda y: y)
        ax.plot(ideal[0], transform(ideal[field]), color="#165c96", lw=2,
                label="Ideal EOS + constant opacity, final")
        ax.plot(table[0], transform(table[field]), color="#b5672e", lw=1.8,
                label="Table EOS + Semenov, c1080")
        ax.scatter(paper_r, transform(paper_values[field-1]), s=35, facecolors="none",
                   edgecolors="#8c3d71", label="Mayer Fig. 7 approximate read-off")
        ax.set_xscale("log")
        ax.set_xlim(.6, 300)
        ax.set_xlabel("Radius [au]")
        ax.set_ylabel(ylab)
        ax.grid(alpha=.23, which="both")
        ax.axvspan(6, 7, color="#8c3d71", alpha=.08)
    axes[0, 0].set_yscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 0].legend(frameon=False, fontsize=7)
    axes[1, 1].semilogy(t, hrho, color="#165c96", lw=1.8)
    axes[1, 1].axhline(1e-10, ls=":", color="#8c3d71")
    axes[1, 1].set_xlim(0, max(t)*1.01)
    axes[1, 1].set_xlabel("Time [kyr]")
    axes[1, 1].set_ylabel(r"Maximum density [g cm$^{-3}$]", color="#165c96")
    twin = axes[1, 1].twinx()
    twin.plot(t, ht, color="#b5672e", lw=1.4)
    twin.set_ylabel("Maximum gas temperature [K]", color="#b5672e")
    fig.suptitle("Matched ideal-EOS collapse: final structure and full-path history")
    fig.savefig(out, dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), constrained_layout=True)
    for ax, field, ylabel in ((axes[0], 1, r"Density [g cm$^{-3}$]"),
                              (axes[1], 2, "Gas temperature [K]")):
        ax.plot(ideal_earlier[0], ideal_earlier[field], color="#165c96", lw=2,
                label="Ideal c1090, rho max 9.20e-11")
        ax.plot(table_earlier[0], table_earlier[field], color="#b5672e", lw=2,
                label="Table c1060, rho max 9.47e-11")
        ax.set_xscale("log")
        ax.set_xlim(.6, 50)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Radius [au]")
        ax.grid(alpha=.23, which="both")
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_yscale("log")
    fig.savefig(out.with_name("wor118_ideal_mayer5_matched_density.png"), dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 3.1), constrained_layout=True)
    ax.plot(t, mass, color="#3b786c", lw=1.5)
    ax.axhline(0, color="black", lw=.7)
    ax.set_xlim(0, max(t)*1.01)
    ax.set_xlabel("Time [kyr]")
    ax.set_ylabel("Gas mass relative to c0")
    ax.grid(alpha=.23)
    fig.savefig(out.with_name("wor118_ideal_mayer5_mass.png"), dpi=170)
    plt.close(fig)


def main():
    final = json.loads((ROOT / "wor118_ideal_final_radial.json").read_text())
    axis = json.loads((ROOT / "wor118_ideal_final_axis.json").read_text())
    history = read("wor118_ideal_mayer5_history.csv")
    last = history[-1]
    mass_drift = max(abs(float(x["mass_rel_from_c0"])) for x in history)
    min_jeans = min(float(x["jeans_min_cells"]) for x in history)
    fig = ROOT / "wor118_ideal_mayer5_report.png"
    make_plot(fig)
    images = [fig, ROOT / "wor118_ideal_mayer5_matched_density.png",
              ROOT / "wor118_ideal_mayer5_mass.png"]
    rendered = [base64.b64encode(path.read_bytes()).decode("ascii") for path in images]
    shock = axis["steepest_positive_x_density_drop_3_to_15_au"]
    html = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WOR-92 matched ideal EOS / Mayer §5 comparison</title>
<style>body{{font:16px/1.7 system-ui,sans-serif;max-width:1100px;margin:auto;padding:1.5rem;background:#f4f8fa;color:#24343e}}section,figure{{background:white;border:1px solid #d7e3e8;border-radius:10px;padding:1rem;margin:1rem 0}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:.42rem;text-align:left;border-bottom:1px solid #d7e3e8}}.small{{font-size:.9rem;color:#526873}}a{{color:#165c96}}</style></head><body>
<h1>理想気体・一定opacityの第一コア比較</h1>
<p>WOR-118修正後の初期状態から計算した理想気体γ=5/3・κP=κR=1 cm²/g経路。元の計算はc1095の局所ガス方程式で停止し、同方程式の直接解法をideal EOSにも有効にした後、c1090のrestartから再開しました。c0→c1090と以降ではバイナリが異なります。</p>
<section><h2>到達点</h2><table><tbody>
<tr><th>最終cycle・時刻</th><td>{final['cycle']}・{float(last['time_kyr']):.5f} kyr</td></tr>
<tr><th>最大密度</th><td>{final['rho_max_g_cm3']:.6e} g/cm³</td></tr>
<tr><th>最高温度</th><td>{final['temperature_max_K']:.2f} K</td></tr>
<tr><th>衝撃波候補</th><td>{shock['inner_au']:.3f}–{shock['outer_au']:.3f} au、密度比{shock['inner_over_outer_density']:.2f}、流入速度 内側{shock['inner_infall_km_s']:.2f}/外側{shock['outer_infall_km_s']:.2f} km/s</td></tr>
<tr><th>中心流入速度</th><td>{-final['rho_max_velocity_radial_km_s']:.3f} km/s</td></tr>
<tr><th>全受理stepの最小Jeans解像度</th><td>{min_jeans:.3e} cells（c1095–1097の圧力floor、要注意）</td></tr>
<tr><th>最終stepのJeans解像度</th><td>{float(last['jeans_min_cells']):.2f} cells</td></tr>
<tr><th>最大gas質量相対変化</th><td>{mass_drift:.2e}（15桁history）</td></tr>
</tbody></table></section>
<figure><img src="data:image/png;base64,{rendered[0]}" alt="半径方向の密度・温度・流入速度と時間発展"><figcaption>青: ideal EOS、橙: table EOS、丸: Mayer et al. Figure 7の概読値。Mayerの6–7 auを薄紫で表示。図からの概読点は厳密な数値表ではありません。</figcaption></figure>
<section><h2>密度を近づけたEOS比較</h2><p>ideal c1090のρmaxは9.20e−11、table c1060は9.47e−11 g/cm³（差約2.9%）。この段階でideal中心温度は600.6 K、tableは416.8 K、衝撃波候補はそれぞれ6.775–7.141 auと7.507–7.874 auです。EOSとopacityを同時に変えた比較で、差を一方だけには帰属できません。</p></section>
<figure><img src="data:image/png;base64,{rendered[1]}" alt="密度を近づけた理想気体と表EOSの半径プロファイル"></figure>
<section><h2>短い圧力floor事象</h2><p>修正版のc1095–1097には、shock付近の半径6.887 auで最小圧力が数値floorに達しました。c1095の3D出力では最細levelの8セル、合計6.48e−8 M☉に限られ、ambient固定セルではありません。c1098に回復し、最終c1140にはfloorセルがありません。10 cycleごとの停止ゲートでは検出できないため、全経路でJeans基準を満たしたとは主張しません。同じc1090 restartからのCFL半減（338758）、混合系輸送OFF（338759）、Riemann solverの輻射圧OFF（338760）のいずれでもfloorを再現しました。これらの設定変更は本計算に採用していません。原因の次の切り分けは、該当8セルのhydro flux更新前後と暗黙FLD更新前後のgas energyを直接記録することです。</p></section>
<section><h2>停止と修正の切り分け</h2><p>元のjob 338718はc1095でNewtonのL2残差1.69e−7まで低下しましたが、単一セルのgas最大残差が約1のままで停止しました。入力のexact_local_gas_solve=trueはtable EOSでのみ有効だったため、ideal EOSでは線形化ガス補正が残っていました。修正commit 582acd59はideal EOSで同じ局所直接解法を使います。同じc1090 restart・64 rankの元バイナリ対照job 338745はc1095で同じ失敗を再現しました。修正版job 338740はc1100に正常到達し、元の失敗段階のgas最大残差は1.07e−11まで下がりました。FGMRESの最大true relative residualは対照1.80e−8、修正版1.62e−8です。</p></section>
<figure><img src="data:image/png;base64,{rendered[2]}" alt="全経路のgas質量相対変化"></figure>
<section><h2>文献との関係</h2><p><a href="https://arxiv.org/html/2510.12620#S5">Mayer et al. §5</a>は同じ初期質量1 M☉、半径3000 au、14 K、μ=2.381、γ=5/3、固定κ=1を採用し、約29.09 kyr、密度1e−10 g/cm³超の最初のsnapshotで6–7 auの第一コアを報告します。今回のAthena++は29.144 kyr、衝撃波候補6.775–7.141 auです。Figure 7の概読点に対する今回の温度は1 auで約0.70倍、100 auで約0.62倍と低く、完全一致ではありません。Athena++はCartesian Jeans AMR、MayerはAREPOのLagrangian質量refinementであり、背景圧実装も異なります。概読点は公開数値表ではないため点ごとの合否には用いません。</p></section>
<section><h2>再現情報</h2><p>ベースsource ff60f8f0、局所ガス修正582acd59。元長時間job 338718、修正版短時間job 338740、継続job 338744。元のc1095は失敗したstepであり、質量・Jeans・密度の履歴グラフには含めていません。ソルバの反復履歴・true residual・Slurm実行時間とメモリは各jobのログと実験記録を参照してください。</p><p class="small"><a href="wor118_ideal_mayer5_history.csv">history CSV（各stepの最小圧力を含む）</a> · <a href="wor118_ideal_final_radial.csv">final radial CSV</a> · <a href="wor118_ideal_final_axis.csv">final axis CSV</a> · <a href="wor118_mayer5_matched_run_20261110.md">実験記録</a></p></section>
</body></html>"""
    (ROOT / "wor118_ideal_mayer5_report.html").write_text(html)
    print(ROOT / "wor118_ideal_mayer5_report.html")


if __name__ == "__main__":
    main()
