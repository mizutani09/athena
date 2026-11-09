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
G = 6.67430e-8
KB = 1.380649e-16
MH = 1.6735575e-24
AU = 1.495978707e13
MSUN = 1.98847e33
YEAR = 31557600.0


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


def physics_metrics():
    profile = read_csv("c1080_wor118_full_profile.csv")
    history = read_csv("wor118_full_c0_history.csv")
    paper = read_csv("mayer_fig7_approx_read_off.csv")
    baseline = read_csv("baseline_first_core_xline.csv")
    radius = np.array([(float(row["r_inner_au"]) + float(row["r_outer_au"])) / 2
                       for row in profile])
    rho = np.array([float(row["mean_rho_g_cm3"]) for row in profile])
    temp = np.array([float(row["mass_weighted_temp_K"]) for row in profile])
    outer = np.array([float(row["r_outer_au"]) for row in profile])
    enclosed = np.array([float(row["enclosed_mass_g"]) for row in profile])
    initial_rho = 3 * MSUN / (4 * np.pi * (3000 * AU)**3)
    tff_kyr = np.sqrt(3 * np.pi / (32 * G * initial_rho)) / YEAR / 1000
    sound = np.sqrt(KB * 14 / (2.381 * MH))
    lp = 8.86 * sound**2 / (4 * np.pi * G * (radius * AU)**2)
    envelope = (radius >= 20) & (radius <= 300)
    time = np.array([float(row["time_kyr"]) for row in history])
    max_rho = np.array([float(row["rho_max_g_cm3"]) for row in history])
    max_temp = np.array([float(row["Tgas_max_K"]) for row in history])
    mass_rel = np.array([float(row["mass_rel_from_c0"]) for row in history])
    crossings = {}
    for threshold in (1e-15, 1e-13, 1e-12, 1e-11, 1e-10):
        index = int(np.flatnonzero(max_rho >= threshold)[0])
        crossings[f"{threshold:.0e}"] = {"time_kyr": float(time[index]),
                                          "temperature_K": float(max_temp[index])}
    mayer = {}
    base_x = np.array([float(row["x_au"]) for row in baseline])
    base_t = np.array([float(row["Tgas_K"]) for row in baseline])
    positive = base_x > 0
    for row in paper:
        r = float(row["r_au"])
        model_rho = 10**np.interp(np.log10(r), np.log10(radius), np.log10(rho))
        model_temp = np.interp(r, radius, temp)
        mayer[str(int(r))] = {
            "model_rho_over_approx_paper": float(model_rho / float(row["rho_est_g_cm3"])),
            "model_temperature_over_approx_paper": float(model_temp / float(row["Tgas_est_K"])),
            "earlier_ideal_xaxis_temperature_over_approx_paper": float(
                np.interp(r, base_x[positive], base_t[positive]) / float(row["Tgas_est_K"]))}
    result = {
        "source": "c1080_wor118_full_profile.csv and wor118_full_c0_history.csv",
        "initial_uniform_density_g_cm3": initial_rho,
        "pressureless_freefall_time_kyr": float(tff_kyr),
        "snapshot_time_over_freefall": float(time[-1] / tff_kyr),
        "first_core_shock_radius_midpoint_au": 7.6904296875,
        "enclosed_gas_mass_at_shock_msun": float(np.interp(7.6904296875, outer, enclosed) / MSUN),
        "envelope_window_au": [20, 300],
        "envelope_density_power_law_slope": float(np.polyfit(np.log10(radius[envelope]), np.log10(rho[envelope]), 1)[0]),
        "envelope_median_rho_over_Larson_Penston_isothermal_asymptote": float(np.median(rho[envelope] / lp[envelope])),
        "envelope_temperature_range_K": [float(np.min(temp[envelope])), float(np.max(temp[envelope]))],
        "maximum_absolute_gas_mass_relative_drift": float(np.max(np.abs(mass_rel))),
        "density_crossings": crossings,
        "mayer_section5_approximate_figure7_ratios": mayer,
        "reference_limitations": "Mayer Figure 7 points were read approximately from a raster figure; Section 5 has different EOS, opacity, mesh, and ambient pressure. The Larson-Penston curve is an isothermal asymptote, not an exact RHD solution. The core mass interpolates cumulative spherical-shell mass and is approximate."
    }
    (ROOT / "wor118_full_c0_physics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def make_physics_plot(path):
    radius, rho, temp, infall = profile_arrays("c1080_wor118_full_profile.csv")
    paper = read_csv("mayer_fig7_approx_read_off.csv")
    baseline = read_csv("baseline_first_core_xline.csv")
    x = np.array([float(row["x_au"]) for row in baseline])
    right = (x > 0) & (x <= 300)
    x = x[right]
    baseline_rho = np.array([float(row["rho_g_cm3"]) for row in baseline])[right]
    baseline_temp = np.array([float(row["Tgas_K"]) for row in baseline])[right]
    baseline_v = -np.array([float(row["vr_km_s"]) for row in baseline])[right]
    paper_r = np.array([float(row["r_au"]) for row in paper])
    paper_rho = np.array([float(row["rho_est_g_cm3"]) for row in paper])
    paper_temp = np.array([float(row["Tgas_est_K"]) for row in paper])
    paper_v = -np.array([float(row["vr_est_km_s"]) for row in paper])
    sound = np.sqrt(KB * 14 / (2.381 * MH))
    lp = 8.86 * sound**2 / (4 * np.pi * G * (radius * AU)**2)
    history = read_csv("wor118_full_c0_history.csv")
    t = np.array([float(row["time_kyr"]) for row in history])
    hrho = np.array([float(row["rho_max_g_cm3"]) for row in history])
    ht = np.array([float(row["Tgas_max_K"]) for row in history])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, current, old, points, ylabel in (
        (axes[0, 0], rho, baseline_rho, paper_rho, r"Density [g cm$^{-3}$]"),
        (axes[0, 1], temp, baseline_temp, paper_temp, "Gas temperature [K]"),
        (axes[1, 0], infall, baseline_v, paper_v, "Infall speed [km/s]")):
        ax.plot(radius, current, color="#087e78", lw=2, label="Corrected table EOS, 3D shells")
        ax.plot(x, old, color="#a96c31", lw=1.5, ls="--", label="Earlier ideal EOS, +x axis")
        ax.scatter(paper_r, points, facecolors="none", edgecolors="#ad3333",
                   label="Mayer Sec. 5, approximate Fig. 7 read-off")
        ax.axvspan(6, 7, color="#ad3333", alpha=.08)
        ax.axvspan(7.5073, 7.8735, color="#087e78", alpha=.10)
        ax.set_xscale("log")
        ax.set_xlim(.6, 300)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Radius [au]")
        ax.grid(alpha=.22, which="both")
    axes[0, 0].plot(radius, lp, color="#755c9a", ls=":", label="14 K Larson–Penston asymptote")
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend(frameon=False, fontsize=7)
    axes[0, 1].set_yscale("log")
    axes[1, 1].semilogy(t, hrho, color="#087e78", lw=1.7, label="Maximum density")
    axes[1, 1].set_xlim(27.5, 29.12)
    axes[1, 1].set_xlabel("Time [kyr]")
    axes[1, 1].set_ylabel(r"Maximum density [g cm$^{-3}$]", color="#087e78")
    axes[1, 1].tick_params(axis="y", labelcolor="#087e78")
    axes[1, 1].grid(alpha=.22)
    twin = axes[1, 1].twinx()
    twin.plot(t, ht, color="#a96c31", lw=1.5, label="Maximum gas temperature")
    twin.set_ylabel("Maximum gas temperature [K]", color="#a96c31")
    twin.tick_params(axis="y", labelcolor="#a96c31")
    fig.suptitle("Corrected first core: radial structure and literature context")
    fig.savefig(path, dpi=170)
    plt.close(fig)


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


def make_html(path, figure, physics_figure, physics):
    s = json.loads((ROOT / "wor118_full_c0_summary.json").read_text())
    shock = s["axis_shock"]
    image = base64.b64encode(figure.read_bytes()).decode("ascii")
    physics_image = base64.b64encode(physics_figure.read_bytes()).decode("ascii")
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
<section><h2>構造と物理的な比較</h2>
<p>初期一様球の密度は{physics['initial_uniform_density_g_cm3']:.4e} g/cm³、圧力なし自由落下時間は{physics['pressureless_freefall_time_kyr']:.4f} kyr。終了時刻はその{physics['snapshot_time_over_freefall']:.5f}倍です。降着衝撃波の中間半径までに含まれるgasは約{physics['enclosed_gas_mass_at_shock_msun']:.4f} M☉（球殻累積質量の補間）です。</p>
<p>20–300 auの密度勾配はr<sup>{physics['envelope_density_power_law_slope']:.2f}</sup>、14 K等温Larson–Penston外層漸近密度との比の中央値は{physics['envelope_median_rho_over_Larson_Penston_isothermal_asymptote']:.2f}。ただしその領域の温度は{physics['envelope_temperature_range_K'][0]:.1f}–{physics['envelope_temperature_range_K'][1]:.1f} Kであり、等温解は厳密な参照解ではありません。</p>
<p><a href="https://arxiv.org/html/2510.12620#S5">Mayer et al. §5</a>では約29.09 kyr、衝撃波半径6–7 au。<a href="https://arxiv.org/html/1102.2921v1#S4.SS1">Commerçon et al. Table 1</a>の別初期条件のFLD計算では半径8.1 au、中心温度411 K、第一コア質量0.0233 M☉。今回の半径・温度・質量は同じ桁ですが、物理・境界条件の異なる厳密な一致試験ではありません。</p>
<p>Mayer Figure 7をラスタ図から概読した値と比べると、今回の温度は1 auで約{physics['mayer_section5_approximate_figure7_ratios']['1']['model_temperature_over_approx_paper']:.2f}倍、100 auで約{physics['mayer_section5_approximate_figure7_ratios']['100']['model_temperature_over_approx_paper']:.2f}倍です。以前の理想気体基準計算でも同じ概読値に対してそれぞれ約{physics['mayer_section5_approximate_figure7_ratios']['1']['earlier_ideal_xaxis_temperature_over_approx_paper']:.2f}倍、{physics['mayer_section5_approximate_figure7_ratios']['100']['earlier_ideal_xaxis_temperature_over_approx_paper']:.2f}倍でした。したがって差を表EOSだけに帰属できません。概読点は公開数値表ではなく、図は球殻平均と+x軸値も併記するため点ごとの合否判定には使いません。</p></section>
<figure><img src="data:image/png;base64,{physics_image}" alt="修正後第一コアの密度・温度・流速を文献概読値と比較し、時間発展を示す図">
<figcaption>緑：修正後table EOSの3D球殻平均。橙破線：以前の理想気体基準の+x軸値。赤丸：Mayer §5 Figure 7の概読点。陰影はMayerのshock半径6–7 auと今回の7.507–7.874 au。紫点線は14 K等温Larson–Penston外層漸近形です。温度・密度差の物理的原因はこの比較だけでは同定できません。</figcaption></figure>
<section><h2>ソルバと再現情報</h2><p>XDジョブは全てCOMPLETED 0:0。長時間区間c500→1080は41:50で、Newton {s['main_job_newton_converged']}件が収束、FGMRES {s['main_job_fgmres_solves']} solve／{s['main_job_fgmres_iterations']}反復、最大true linear relative residual {s['main_job_max_true_linear_relative_residual']:.3e}、opacity異常0。</p>
<table><thead><tr><th>ジョブ</th><th>cycle</th><th>ノード</th><th>Slurm</th></tr></thead><tbody>{rows}</tbody></table>
<p class="small">数値ソースcommit: <code>{escape(s['source_commit'])}</code><br>XD binary SHA-256: <code>{escape(s['binary_sha256'])}</code><br>入力SHA-256: <code>{escape(s['input_sha256'])}</code><br>最終3D HDF5 SHA-256: <code>{escape(s['final_hdf_sha256'])}</code></p></section>
<section><h2>比較の範囲</h2><p>Mayer et al. (2025) 第5節は固定γ＝5/3・固定opacityのAREPO計算です。こちらはTomida系table EOS、Semenov opacity、Jeans AMRを使用します。形成時刻とshock半径は近いものの、半径プロファイルの完全一致を主張しません。全エネルギーには背景温度固定のsource/sinkが入るため、厳密保存判定はgas質量で行いました。放射圧はRiemann solverで有効のままです。<a href="https://arxiv.org/html/2510.12620#S5">原著第5節</a></p></section>
</body></html>"""
    path.write_text(html)


def main():
    figure = ROOT / "wor118_full_c0_summary.png"
    physics_figure = ROOT / "wor118_full_c0_physics.png"
    html = ROOT / "wor118_full_c0_report.html"
    physics = physics_metrics()
    make_plot(figure)
    make_physics_plot(physics_figure)
    make_html(html, figure, physics_figure, physics)
    print(figure, physics_figure, html)


if __name__ == "__main__":
    main()
