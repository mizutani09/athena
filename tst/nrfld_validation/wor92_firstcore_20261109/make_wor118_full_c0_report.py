#!/usr/bin/env python3
"""Plot and package the validated WOR-118 cycle-0 first-core result."""

import base64
import csv
from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent


def read_csv(name):
    with (ROOT / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def profile_arrays(name):
    rows = read_csv(name)
    inner = np.array([float(row["r_inner_au"]) for row in rows])
    outer = np.array([float(row["r_outer_au"]) for row in rows])
    radius = np.sqrt(inner * outer)
    radius[0] = outer[0] / 2
    rho = np.array([float(row["mean_rho_g_cm3"]) for row in rows])
    temp = np.array([float(row["mass_weighted_temp_K"]) for row in rows])
    infall = -np.array([float(row["mass_weighted_vrad_km_s"]) for row in rows])
    return radius, rho, temp, infall


def make_plot(path):
    old = profile_arrays("c1010_profile.csv")
    new = profile_arrays("c1080_wor118_full_profile.csv")
    mass_rows = read_csv("wor118_full_c0_history.csv")
    time = np.array([float(row["time_kyr"]) for row in mass_rows])
    mass = np.array([float(row["mass_rel_from_c0"]) for row in mass_rows])
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), constrained_layout=True)
    labels = ((old, "Previous trajectory, c1010", "#a96c31", "--"),
              (new, "Corrected full c0 run, c1080", "#087e78", "-"))
    for arrays, label, color, style in labels:
        r, rho, temp, infall = arrays
        mask = (r > 0.5) & (r < 25)
        axes[0, 0].plot(r[mask], rho[mask], label=label, color=color,
                        ls=style, lw=2)
        axes[0, 1].plot(r[mask], temp[mask], color=color, ls=style, lw=2)
        axes[1, 0].plot(r[mask], infall[mask], color=color, ls=style, lw=2)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel(r"Density [g cm$^{-3}$]")
    axes[0, 0].legend(frameon=False, fontsize=9)
    axes[0, 1].set_ylabel("Gas temperature [K]")
    axes[1, 0].set_ylabel("Infall speed [km/s]")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xscale("log")
        ax.set_xlim(0.5, 25)
        ax.set_xlabel("Radius [au]")
        ax.grid(alpha=0.25)
    axes[1, 1].plot(time, mass, color="#3761a6", lw=1.5)
    axes[1, 1].axhline(0, color="black", lw=0.8, alpha=0.5)
    axes[1, 1].set_xlabel("Time [kyr]")
    axes[1, 1].set_ylabel(r"$(M_{gas}-M_{gas,0})/M_{gas,0}$")
    axes[1, 1].set_xlim(0, max(time) * 1.01)
    axes[1, 1].grid(alpha=0.25)
    fig.suptitle("WOR-118: corrected cycle-0 run reaches a first-core structure")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def make_html(path, figure):
    s = json.loads((ROOT / "wor118_full_c0_summary.json").read_text())
    shock = s["axis_shock"]
    image = base64.b64encode(figure.read_bytes()).decode("ascii")
    rows = "".join(
        "<tr><td>{}</td><td>{}→{}</td><td>{}</td><td>{}</td></tr>".format(
            job["id"], job["start_cycle"], job["end_cycle"],
            job["nodes"], job["state"])
        for job in s["jobs"])
    html = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WOR-118 修正後c0→第一コア | 2026-11-10</title>
<style>body{{font:16px/1.7 system-ui,'Noto Sans JP',sans-serif;color:#1a3440;
background:#f4f8f9;max-width:1050px;margin:0 auto;padding:1.5rem}}
h1,h2{{line-height:1.35}}section,figure{{background:white;border:1px solid #dce7e9;
border-radius:10px;padding:1rem;margin:1rem 0}}img{{max-width:100%;height:auto}}
table{{width:100%;border-collapse:collapse}}th,td{{border-bottom:1px solid #dce7e9;
padding:.45rem;text-align:left}}.lead{{font-size:1.13rem}}.small{{font-size:.9rem;color:#536b74}}
a{{color:#086b83}}</style></head><body>
<h1>WOR-118 修正後の初期状態から第一コアまで</h1>
<p class="lead">5ノード計算を含むcycle 0→1080の全経路で、密度閾値、降着衝撃波、低速中心、質量保存を確認しました。</p>
<section><h2>到達点</h2><table><tbody>
<tr><th>最大密度</th><td>{s['rho_max_g_cm3']:.6e} g/cm³</td></tr>
<tr><th>時刻</th><td>{s['time_years']/1000:.5f} kyr</td></tr>
<tr><th>最高温度</th><td>{s['temperature_max_K']:.2f} K</td></tr>
<tr><th>Jeans最小解像度</th><td>{s['jeans_min_cells']:.2f} cells（停止閾値16）</td></tr>
<tr><th>衝撃波候補</th><td>{shock['inner_au']:.3f}–{shock['outer_au']:.3f} au、密度比{shock['inner_over_outer_density']:.2f}、流入速度は内側{shock['inner_infall_km_s']:.2f}／外側{shock['outer_infall_km_s']:.2f} km/s</td></tr>
<tr><th>中心近傍の半径速度</th><td>{s['central_rho_max_velocity_km_s']:.3f} km/s</td></tr>
<tr><th>質量の全経路相対変化</th><td>{s['gas_mass_relative_change']:.2e}（15桁history）</td></tr>
<tr><th>opacity／EOS表外</th><td>opacity異常0、最終3DのEOS表外セル0</td></tr>
</tbody></table></section>
<figure><img src="data:image/png;base64,{image}" alt="修正後c0全経路の第一コア半径プロファイルとgas質量保存">
<figcaption>実線：修正後のc0→c1080。破線：以前の経路のc1010。最大密度は約3.6%異なるため、二つの曲線を厳密な一変数A/Bとして解釈しません。右下は修正後の全経路のgas質量。</figcaption></figure>
<section><h2>ソルバと再現情報</h2><p>XDジョブは全てCOMPLETED 0:0。長時間区間c500→1080は41:50で、Newton {s['main_job_newton_converged']}件が収束、FGMRES {s['main_job_fgmres_solves']} solve／{s['main_job_fgmres_iterations']}反復、最大true linear relative residual {s['main_job_max_true_linear_relative_residual']:.3e}、opacity異常0。</p>
<table><thead><tr><th>ジョブ</th><th>cycle</th><th>ノード</th><th>Slurm</th></tr></thead><tbody>{rows}</tbody></table>
<p class="small">数値ソースcommit: <code>{escape(s['source_commit'])}</code><br>XD binary SHA-256: <code>{escape(s['binary_sha256'])}</code><br>入力SHA-256: <code>{escape(s['input_sha256'])}</code><br>最終3D HDF5 SHA-256: <code>{escape(s['final_hdf_sha256'])}</code></p></section>
<section><h2>比較の範囲</h2><p>Mayer et al. (2025) 第5節は固定γ＝5/3・固定opacityのAREPO計算です。こちらはTomida系table EOS、Semenov opacity、Jeans AMRを使用します。形成時刻とshock半径は近いものの、半径プロファイルの完全一致を主張しません。全エネルギーには背景温度固定のsource/sinkが入るため、厳密保存判定はgas質量で行いました。放射圧はRiemann solverで有効のままです。<a href="https://arxiv.org/html/2510.12620#S5">原著第5節</a></p></section>
</body></html>"""
    path.write_text(html)


def main():
    figure = ROOT / "wor118_full_c0_summary.png"
    html = ROOT / "wor118_full_c0_report.html"
    make_plot(figure)
    make_html(html, figure)
    print(figure, html)


if __name__ == "__main__":
    main()
