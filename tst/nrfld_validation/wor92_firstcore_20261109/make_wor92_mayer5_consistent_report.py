#!/usr/bin/env python3
"""Build a standalone HTML audit for the consistent c0 Mayer ideal-EOS run."""

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
YEAR = 365.25 * 86400.0
MSUN = 1.98847e33
CODE_TO_KYR = AU / np.sqrt(1e-9 / 1e-18) / YEAR / 1000.0
MAYER_KYR = 29.09
SOURCE_BASE_COMMIT = "582acd5999c1a5681ab1bc1d1af292676642ed40"
SOURCE_BINARY_SHA = "9e7c2faa80832a7cd5d8aa3fd4689aed1648a7f346d4d492c2c0742217020b7f"
SOURCE_INPUT_SHA = "382ab6df9ca41769a96475c52548fa008f486e3a43b5693118b378d991fa5ebb"


def read_hst(paths):
    rows = {}
    total = 0
    for path in paths:
        with path.open() as stream:
            for line in stream:
                fields = line.split()
                if len(fields) != 39 or line.lstrip().startswith("#"):
                    continue
                row = np.array([float(x) for x in fields], dtype=np.float64)
                total += 1
                key = row[0]
                if key in rows and not np.array_equal(rows[key], row):
                    raise ValueError(f"Conflicting HST rows at time {key}: {path}")
                rows[key] = row
    if not rows:
        raise ValueError("No 39-column HST rows found")
    return np.stack([rows[t] for t in sorted(rows)]), total


def read_csv(path):
    with path.open(newline="") as stream:
        lines = list(csv.DictReader(stream))
    if not lines:
        raise ValueError(f"Empty CSV: {path}")
    return {key: np.array([float(row[key]) if row[key] else np.nan for row in lines])
            for key in lines[0]}


def make_history_plot(rows, out):
    kyr = rows[:, 0] * CODE_TO_KYR
    rho = rows[:, 15]
    mass = (rows[:, 22] - rows[0, 22]) / rows[0, 22]
    fig, ax = plt.subplots(2, 1, figsize=(8.8, 6.4), sharex=True,
                            constrained_layout=True)
    ax[0].semilogy(kyr, rho, color="#14638b", linewidth=1.8)
    ax[0].axhline(1e-10, color="#995446", linestyle="--", linewidth=1)
    ax[0].axvline(MAYER_KYR, color="#777777", linestyle=":", linewidth=1)
    ax[0].set_ylabel(r"Maximum density [g cm$^{-3}$]")
    ax[0].grid(alpha=.25)
    ax[1].plot(kyr, mass, color="#3f7867", linewidth=1.4)
    ax[1].axhline(0, color="black", linewidth=.7)
    ax[1].set_ylabel("Gas mass change / initial")
    ax[1].set_xlabel("Time from initial state [kyr]")
    ax[1].grid(alpha=.25)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def make_profile_plot(radial, axis, paper, out, prior):
    inner = radial["r_inner_au"]
    outer = radial["r_outer_au"]
    r = np.sqrt(inner * outer)
    r[0] = outer[0] / 2
    fields = (("mean_rho_g_cm3", "rho_est_g_cm3", r"Density [g cm$^{-3}$]"),
              ("mass_weighted_temp_K", "Tgas_est_K", "Gas temperature [K]"),
              ("mass_weighted_vrad_km_s", "vr_est_km_s", "Infall speed [km/s]"))
    fig, ax = plt.subplots(3, 1, figsize=(8, 8.6), sharex=True,
                           constrained_layout=True)
    for i, (our_field, paper_field, ylabel) in enumerate(fields):
        sign = -1 if i == 2 else 1
        for name, previous in prior:
            r_previous = np.sqrt(previous["r_inner_au"] * previous["r_outer_au"])
            r_previous[0] = previous["r_outer_au"][0] / 2
            ax[i].plot(r_previous, sign * previous[our_field],
                       color="#83aabc", linewidth=1.1, alpha=.8,
                       label=name)
        ax[i].plot(r, sign * radial[our_field], color="#14638b", linewidth=2,
                   label="Latest Athena++ radial shell mean")
        ax[i].scatter(paper["r_au"], sign * paper[paper_field],
                      facecolors="none", edgecolors="#9b577d", s=42,
                      label="Mayer Fig. 7 approximate read-off")
        ax[i].axvspan(6, 7, color="#9b577d", alpha=.08)
        ax[i].set_xscale("log")
        ax[i].set_xlim(.5, 300)
        ax[i].set_ylabel(ylabel)
        ax[i].grid(alpha=.25, which="both")
    ax[0].set_yscale("log")
    ax[1].set_yscale("log")
    ax[0].legend(frameon=False, fontsize=8)
    ax[2].set_xlabel("Radius [au]")
    fig.savefig(out, dpi=170)
    plt.close(fig)

    shock = out.with_name(out.stem + "_shock.png")
    positive = (axis["x_au"] > 0) & (axis["x_au"] < 20)
    fig, ax = plt.subplots(2, 1, figsize=(8, 5.7), sharex=True,
                           constrained_layout=True)
    ax[0].semilogy(axis["x_au"][positive], axis["rho_g_cm3"][positive],
                   color="#14638b", linewidth=1.7)
    ax[0].set_ylabel(r"Axis density [g cm$^{-3}$]")
    ax[1].plot(axis["x_au"][positive], -axis["vx_km_s"][positive],
               color="#a85e45", linewidth=1.7)
    ax[1].set_ylabel("Infall speed [km/s]")
    ax[1].set_xlabel("Positive x [au]")
    for a in ax:
        a.set_xlim(0, 20)
        a.grid(alpha=.25)
    fig.savefig(shock, dpi=170)
    plt.close(fig)
    return shock


def image_data(path):
    return base64.b64encode(path.read_bytes()).decode("ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history_dir", type=Path)
    parser.add_argument("radial_csv", type=Path)
    parser.add_argument("radial_json", type=Path)
    parser.add_argument("axis_csv", type=Path)
    parser.add_argument("axis_json", type=Path)
    parser.add_argument("output_html", type=Path)
    parser.add_argument("--prior-radial", type=Path, action="append", default=[],
                        help="Earlier radial CSV to overlay; may be repeated")
    parser.add_argument("--solver-metrics", type=Path,
                        help="JSON with final Slurm and Newton/FGMRES audit")
    parser.add_argument("--continuation-binary-sha", default="",
                        help="SHA-256 of a separately validated continuation binary")
    args = parser.parse_args()
    paths = sorted(args.history_dir.glob("job_*.hst"))
    rows, total_rows = read_hst(paths)
    radial = read_csv(args.radial_csv)
    axis = read_csv(args.axis_csv)
    prior = [(path.stem.split("_")[0], read_csv(path)) for path in args.prior_radial]
    paper = read_csv(Path(__file__).with_name("mayer_fig7_approx_read_off.csv"))
    radial_stats = json.loads(args.radial_json.read_text())
    axis_stats = json.loads(args.axis_json.read_text())
    solver_metrics = (json.loads(args.solver_metrics.read_text())
                      if args.solver_metrics else None)
    final_time = rows[-1, 0] * CODE_TO_KYR
    initial_mass = rows[0, 22]
    mass_drift = float(np.max(np.abs(rows[:, 22] - initial_mass) / initial_mass))
    pressure_min = float(np.min(rows[:, 29]))
    jeans_min = float(np.min(rows[:, 34]))
    rho_max = float(np.max(rows[:, 15]))
    shock = axis_stats["steepest_positive_x_density_drop_3_to_15_au"]
    shock_mid = (shock["inner_au"] + shock["outer_au"]) / 2
    core_mass = float(np.interp(shock_mid, radial["r_outer_au"],
                                radial["enclosed_mass_g"]) / MSUN)
    r_shell = np.sqrt(radial["r_inner_au"] * radial["r_outer_au"])
    r_shell[0] = radial["r_outer_au"][0] / 2
    comparisons = []
    for r_paper, rho_paper, temp_paper, vr_paper in zip(
            paper["r_au"], paper["rho_est_g_cm3"], paper["Tgas_est_K"],
            paper["vr_est_km_s"]):
        log_r = np.log(r_shell)
        rho_model = float(np.exp(np.interp(np.log(r_paper), log_r,
                                           np.log(radial["mean_rho_g_cm3"]))))
        temp_model = float(np.interp(np.log(r_paper), log_r,
                                      radial["mass_weighted_temp_K"]))
        infall_model = float(-np.interp(np.log(r_paper), log_r,
                                         radial["mass_weighted_vrad_km_s"]))
        comparisons.append({"radius_au": float(r_paper),
                            "rho_model_over_paper": rho_model / float(rho_paper),
                            "temperature_model_over_paper": temp_model / float(temp_paper),
                            "infall_model_over_paper": infall_model / -float(vr_paper)})
    # Athena HDF5 stores Time as float32 here; compare within its spacing.
    time_roundoff = max(1e-6, 2 * float(np.spacing(np.float32(rows[-1, 0]))))
    if radial_stats["time_code"] > rows[-1, 0] + time_roundoff:
        raise ValueError("3D snapshot is newer than HST history")
    out = args.output_html
    out.parent.mkdir(parents=True, exist_ok=True)
    history_png = out.with_name(out.stem + "_history.png")
    profile_png = out.with_name(out.stem + "_profile.png")
    make_history_plot(rows, history_png)
    shock_png = make_profile_plot(radial, axis, paper, profile_png, prior)
    summary = {
        "history_rows": int(len(rows)), "duplicate_rows_verified": total_rows - len(rows),
        "history_last_time_kyr": float(final_time), "history_rho_max_g_cm3": rho_max,
        "mass_relative_drift_max": mass_drift, "pressure_min_dyn_cm2": pressure_min,
        "jeans_min_cells": jeans_min, "mass_gate_pass": mass_drift <= 1e-8,
        "pressure_gate_pass": pressure_min > 1e-18, "jeans_gate_pass": jeans_min >= 16,
        "rho_target_pass": rho_max >= 1e-10,
        "snapshot_cycle": radial_stats["cycle"],
        "snapshot_time_kyr": radial_stats["time_years"] / 1000,
        "snapshot_rho_max_g_cm3": radial_stats["rho_max_g_cm3"],
        "snapshot_center_temperature_K": radial_stats["temperature_at_rho_max_K"],
        "snapshot_center_vrad_km_s": radial_stats["rho_max_velocity_radial_km_s"],
        "shock_au": [shock["inner_au"], shock["outer_au"]],
        "shock_density_ratio": shock["inner_over_outer_density"],
        "shock_infall_inner_outer_km_s": [shock["inner_infall_km_s"],
                                              shock["outer_infall_km_s"]],
        "mass_inside_shock_msun": core_mass,
        "mayer_reference_time_kyr": MAYER_KYR,
        "mayer_reference_shock_radius_au": [6, 7],
        "mayer_figure7_approx_read_off_ratios": comparisons,
        "source_base_commit": SOURCE_BASE_COMMIT,
        "source_binary_sha256": SOURCE_BINARY_SHA,
        "source_input_sha256": SOURCE_INPUT_SHA,
    }
    if args.continuation_binary_sha:
        summary["continuation_binary_sha256"] = args.continuation_binary_sha
    if solver_metrics is not None:
        summary["solver_metrics"] = solver_metrics
    out.with_name(out.stem + "_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    solver_ok = (solver_metrics is None or
                 (solver_metrics["slurm_state"] == "COMPLETED" and
                  solver_metrics["newton_nonconverged"] == 0 and
                  solver_metrics["fatal_lines"] == 0))
    status = "PASS" if all((summary["mass_gate_pass"], summary["pressure_gate_pass"],
                             summary["jeans_gate_pass"], summary["rho_target_pass"],
                             solver_ok)) else "INCOMPLETE"
    if solver_metrics is None:
        solver_html = "<p>最終ジョブのソルバ集計は未入力。</p>"
    else:
        solver_html = (
            "<table>"
            f"<tr><th>Job・Slurm</th><td>{solver_metrics['job_id']}・{html.escape(solver_metrics['slurm_state'])}</td></tr>"
            f"<tr><th>Newton</th><td>{solver_metrics['newton_converged']} / {solver_metrics['newton_total']} 収束、最大反復{solver_metrics['newton_iterations_max']}</td></tr>"
            f"<tr><th>FGMRES</th><td>{solver_metrics['fgmres_solves']} solves / {solver_metrics['fgmres_iterations']} iterations、最大true相対残差{solver_metrics['fgmres_true_relative_max']:.3e}</td></tr>"
            f"<tr><th>実行時間・最大RSS</th><td>{solver_metrics['elapsed_seconds']/60:.1f} min・{solver_metrics['maxrss_kib']/1024/1024:.2f} GiB</td></tr>"
            "</table>")
    continuation_note = (f"""<section><h2>cycle 1416 のAMR細分化と修正</h2>
<p>元の計算では降着衝撃波に接する新規level-11の2セルで圧力が
1e-19 dyn/cm²の下限へ落ちた。細分化を1 stepだけ抑えた対照では再現しなかった。
同一修正バイナリの補間制限OFFでも再現し、ONではc1420まで圧力下限セル0。
保存変数の8子セルを共通係数で制限し、粗格子の親平均を保ちながら
内部エネルギー由来の圧力を正にした。磁場ゼロのideal EOS条件に限って有効化。</p>
<table><tr><th>条件</th><th>c1416 最小圧力</th><th>Jeans最小値</th></tr>
<tr><td>元の演算・制限OFF</td><td>1.00e-19 dyn/cm²</td><td>1.49e-6 cells</td></tr>
<tr><td>細分化なし対照</td><td>5.22e-10 dyn/cm²</td><td>31.99 cells</td></tr>
<tr><td>正値制限ON</td><td>5.22e-10 dyn/cm²</td><td>32.04 cells</td></tr></table>
<p class="small">修正ソースcommit 9d007f26、継続バイナリ SHA-256
{html.escape(args.continuation_binary_sha)}。修正前に失敗したc1416行は
有効経路の履歴に含めない。</p></section>"""
                         if args.continuation_binary_sha else "")
    run_title = ("WOR-92: Mayer §5 ideal EOS・一定opacityのc0からの修正継続計算"
                 if args.continuation_binary_sha else
                 "WOR-92: Mayer §5 ideal EOS・一定opacityのc0一貫計算")
    comparison_rows = "".join(
        f"<tr><td>{entry['radius_au']:g}</td><td>{entry['rho_model_over_paper']:.3f}</td>"
        f"<td>{entry['temperature_model_over_paper']:.3f}</td>"
        f"<td>{entry['infall_model_over_paper']:.3f}</td></tr>"
        for entry in comparisons)
    document = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(run_title)}</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;max-width:1020px;margin:auto;padding:1.3rem;color:#263743;background:#f3f7f9}}section,figure{{background:white;border:1px solid #dbe5e8;border-radius:9px;padding:1rem;margin:1rem 0}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;border-bottom:1px solid #dbe5e8;padding:.4rem}}a{{color:#14638b}}.small{{font-size:.9rem;color:#526673}}</style></head><body>
<h1>{html.escape(run_title)}</h1>
<p>数値ゲート: <strong>{status}</strong>。初期一様球1 M☉・3000 au・14 K、γ=5/3、μ=2.381、κP=κR=1 cm²/g。Athena++のCartesian Jeans AMR計算を、Mayer et al. (2025) §5のAREPO計算と比較する。背景ガスは密度1/30・圧力平衡による420 K、背景放射は14 Kという明示的仮定。</p>
<section><h2>履歴と保存検査</h2><table>
<tr><th>最終履歴時刻</th><td>{final_time:.5f} kyr</td></tr>
<tr><th>最大密度</th><td>{rho_max:.6e} g/cm³</td></tr>
<tr><th>質量相対変化の最大値</th><td>{mass_drift:.3e}</td></tr>
<tr><th>最小圧力</th><td>{pressure_min:.6e} dyn/cm²</td></tr>
<tr><th>最小Jeans解像度</th><td>{jeans_min:.3f} cells</td></tr>
<tr><th>照合した履歴</th><td>{len(rows)} unique rows、重複{total_rows-len(rows)}行は完全一致</td></tr>
</table></section>
<section><h2>最終継続ジョブのソルバ監査</h2>{solver_html}</section>
{continuation_note}
<figure><img src="data:image/png;base64,{image_data(history_png)}" alt="最大密度と質量の時間履歴"><figcaption>破線は目標密度10⁻¹⁰ g/cm³、縦点線はMayer論文の29.09 kyr。</figcaption></figure>
<section><h2>3D leaf-cell物理量（cycle {radial_stats['cycle']}）</h2><table>
<tr><th>時刻</th><td>{radial_stats['time_years']/1000:.5f} kyr</td></tr>
<tr><th>中心最大密度・温度</th><td>{radial_stats['rho_max_g_cm3']:.6e} g/cm³、{radial_stats['temperature_at_rho_max_K']:.2f} K</td></tr>
<tr><th>最大密度セルの半径方向速度</th><td>{radial_stats['rho_max_velocity_radial_km_s']:.4f} km/s</td></tr>
<tr><th>衝撃波候補</th><td>{shock['inner_au']:.3f}–{shock['outer_au']:.3f} au、密度比{shock['inner_over_outer_density']:.2f}</td></tr>
<tr><th>衝撃波内・外の流入速度</th><td>{shock['inner_infall_km_s']:.3f} / {shock['outer_infall_km_s']:.3f} km/s</td></tr>
<tr><th>衝撃波半径内の質量</th><td>{core_mass:.4f} M☉</td></tr>
</table></section>
<figure><img src="data:image/png;base64,{image_data(profile_png)}" alt="半径方向プロファイルとMayer論文概読値"><figcaption>青: 当計算の最新体積重み球殻平均。淡青: 先行cycle。紫丸: Mayer Figure 7のラスター図からの概読値。概読点は公開数値表でなく、厳密なacceptance gateには使わない。</figcaption></figure>
<section><h2>Mayer Figure 7 概読点との比</h2><table><tr><th>半径 [au]</th><th>密度比</th><th>温度比</th><th>流入速度比</th></tr>{comparison_rows}</table><p class="small">いずれも当計算/Mayer概読値。ラスター図からの概読値と球殻平均の比較であり、精密な収束判定ではない。</p></section>
<figure><img src="data:image/png;base64,{image_data(shock_png)}" alt="正のx軸の衝撃波構造"><figcaption>3D出力の正のx軸に沿った密度と流入速度。</figcaption></figure>
<section><h2>比較の限界と再現情報</h2><p>Mayerの基準時刻は約29.09 kyr、第一コア半径は6–7 au。当計算の密度閾値時刻は{final_time:.2f} kyrで約{final_time/MAYER_KYR:.3f}倍遅い。衝撃波半径の近さだけで物理解の一致とは判定しない。当計算にはCartesian Jeans AMRとAREPO質量精緻化の差があり、背景放射14 Kの初期値は論文に明記されていない。時刻や半径の差をソルバの単一原因へ帰属しない。</p>
<p class="small">Source base commit {html.escape(SOURCE_BASE_COMMIT)}<br>Binary SHA-256 {html.escape(SOURCE_BINARY_SHA)}<br>Input SHA-256 {html.escape(SOURCE_INPUT_SHA)}<br>References: <a href="https://arxiv.org/html/2510.12620#S5">Mayer et al. 2025 §5</a>; <a href="https://arxiv.org/html/2410.07016#S3">precursor §3.1</a>.</p></section>
</body></html>"""
    out.write_text(document)
    print(out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
