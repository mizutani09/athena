#!/usr/bin/env python3
"""Package the verified table-EOS first-core run as a limited application example."""

import base64
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "wor92_firstcore_20261109"


def read(name):
    with (DATA / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def column(rows, name):
    return np.array([float(row[name]) for row in rows])


def main():
    history = read("wor118_full_c0_history.csv")
    profile = read("c1080_wor118_full_profile.csv")
    time = column(history, "time_kyr")
    radius = np.sqrt(column(profile, "r_inner_au") * column(profile, "r_outer_au"))
    radius[0] = column(profile, "r_outer_au")[0] / 2

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4), constrained_layout=True)
    axes[0, 0].semilogy(time, column(history, "rho_max_g_cm3"), color="#17627f")
    axes[0, 0].axhline(1e-10, color="#8d4e39", linestyle="--", linewidth=1)
    axes[0, 0].set(xlabel="Time [kyr]", ylabel=r"Maximum density [g cm$^{-3}$]",
                   xlim=(0, 29.2), title="(a) Collapse history")
    axes[0, 1].plot(time, column(history, "Tgas_max_K"), color="#b35630")
    axes[0, 1].set(xlabel="Time [kyr]", ylabel="Maximum gas temperature [K]",
                   xlim=(27, 29.2), title="(b) Central heating")
    mask = (radius >= 0.25) & (radius <= 100)
    axes[1, 0].loglog(radius[mask], column(profile, "mean_rho_g_cm3")[mask],
                      color="#17627f")
    axes[1, 0].axvspan(7.507, 7.874, color="#8d4e39", alpha=.15)
    axes[1, 0].set(xlabel="Radius [au]", ylabel=r"Density [g cm$^{-3}$]",
                   title="(c) First-core structure")
    axes[1, 1].semilogx(radius[mask], column(profile, "mass_weighted_temp_K")[mask],
                        color="#b35630", label="Gas temperature")
    axes[1, 1].axvspan(7.507, 7.874, color="#8d4e39", alpha=.15)
    axes[1, 1].set(xlabel="Radius [au]", ylabel="Gas temperature [K]",
                   title="(d) Radial heating")
    for ax in axes.flat:
        ax.grid(alpha=.22, which="both")
    png = HERE / "table_eos_application_candidate.png"
    pdf = HERE / "table_eos_application_candidate.pdf"
    fig.savefig(png, dpi=175)
    fig.savefig(pdf)
    plt.close(fig)

    embedded = base64.b64encode(png.read_bytes()).decode("ascii")
    document = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WOR-115 table-EOS collapse application gate</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;max-width:990px;margin:auto;padding:1.5rem;color:#263743;background:#f4f7f8}}
section,figure{{background:white;border:1px solid #dce5e8;border-radius:8px;padding:1rem;margin:1rem 0}}
img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #dce5e8;padding:.4rem;text-align:left}}
a{{color:#17627f}}</style></head><body>
<h1>WOR-115: table EOS・可変opacityのcollapse実証</h1>
<p>掲載判定: 第一コアまでの数値的実行例として採用可能。Mayer et al.の理想気体・一定opacity計算との点ごとの物理的一致は主張しない。最終原稿への採否と統合ソース固定はWOR-88/114で扱う。</p>
<figure><img src="data:image/png;base64,{embedded}" alt="Table EOS collapse history and first-core radial profiles">
<figcaption>候補caption: Three-dimensional protostellar collapse with a tabulated H/He equation of state and density- and temperature-dependent opacity. Starting from a uniform, non-rotating, non-magnetized 1 M☉ sphere, the calculation reaches ρmax=1.044×10⁻¹⁰ g cm⁻³ at 29.110 kyr. The density break at 7.51–7.87 au and the rising central temperature identify a first-core candidate. Shaded bands mark the accretion-shock interval inferred from the +x-axis profile. These results demonstrate stable evolution of this numerical configuration; they are not a matched-condition reproduction of Mayer et al. (2025).</figcaption></figure>
<section><h2>検証済みの主張</h2><table>
<tr><th>物理・数値設定</th><td>Tomida系table EOS、Semenov opacity、Cartesian Jeans AMR、CFL=0.3、Newton–FGMRES–MG、5-node継続</td></tr>
<tr><th>到達点</th><td>c0→c1080、ρmax=1.04414476×10⁻¹⁰ g cm⁻³、29.11046 kyr、Tgas,max=434.524 K</td></tr>
<tr><th>構造</th><td>shock候補7.507–7.874 au、中心半径速度−0.0138 km/s、約0.03034 M☉をshock内に含む</td></tr>
<tr><th>健全性</th><td>全履歴のgas mass最大相対偏差5.28×10⁻¹⁵、Jeans最小31.59セル、最終HDFのEOS表外セル0、opacity診断異常0、Newton収束</td></tr>
</table></section>
<section><h2>制限と次のゲート</h2><p>線形solveにはbounded-inexact上限1×10⁻⁷を使用し、c10→c500の最大受理true相対残差は6.70×10⁻⁸。背景温度の受理行にはc10→c500で最大14.9%の一過性例外が1行あり、最終c500→c1080区間の最大は0.0024%。背景の温度・放射固定は意図的なエネルギー源/吸収源なので、総エネルギー一定は合格条件ではない。高温表域、H₂解離、第二コアは未検証。MayerのSection 5は理想気体・一定opacity・AREPO移動格子であり、図の概読点との温度差は残る。図の厳密一致は本実証の完了条件ではない。</p>
<p>論文用単一ソース版の固定と統合版の短時間回帰はWOR-114/95、Mayerと同条件の定量比較はWOR-96の任意研究課題。<a href="wor115_mayer_offset_report.html">Mayer差の切り分け</a>と<a href="../wor92_firstcore_20261109/wor118_full_c0_report.html">全経路の詳細HTML</a>を参照。</p></section>
<section><h2>再現情報</h2><table>
<tr><th>source commit</th><td>ff60f8f07842acb00e7753932bcfe30867e3a66e</td></tr>
<tr><th>XD binary SHA-256</th><td>908d5def56faf5e0993526f1c709b2593eec814a2a26ffd83d4a0133b6ac0810</td></tr>
<tr><th>input SHA-256</th><td>aad81db83087d27351db59ef70b9b14a68e8a2dc3e6fc164a4cb73e8d413e7e6</td></tr>
<tr><th>EOS / opacity SHA-256</th><td>742a3456923d4c7056c3705accbcc793af3e0c79408a2f597a0cd9271b13ba43 / 6c42445452bca6aeca1253f31ab049b2ff107dea3601f946b4c61c93db022f94</td></tr>
<tr><th>final HDF SHA-256</th><td>13767189008f53ffb9e802b7ef66dcd97faff7045adbdd59051d19269a0354ac</td></tr>
<tr><th>XD jobs</th><td>338469、338485、338486、いずれもCOMPLETED 0:0</td></tr>
</table></section></body></html>"""
    output = HERE / "table_eos_application_candidate.html"
    output.write_text(document)
    print(output)


if __name__ == "__main__":
    main()
