#!/usr/bin/env python3
"""Create a self-contained HTML report for the adopted WOR-92 first-core run."""

import base64
import csv
from html import escape
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MSUN_G = 1.98847e33


def read_json(name):
    return json.loads((ROOT / name).read_text())


def tr(*cells):
    return "<tr>" + "".join(f"<td>{escape(str(cell))}</td>" for cell in cells) + "</tr>"


def render():
    stages = {cycle: read_json(f"c{cycle}_profile.json") for cycle in (670, 780, 1010)}
    axes = {cycle: read_json(f"c{cycle}_axis.json") for cycle in (670, 780, 1010)}
    jobs = [read_json(f"job_{job}_summary.json") for job in (338156, 338171)]
    cost = list(csv.DictReader((ROOT / "adopted_path_cost.csv").open()))
    node_hours = sum(int(row["allocated_nodes"]) * int(row["slurm_elapsed_seconds"])
                     for row in cost) / 3600
    rank_hours = sum(int(row["allocated_nodes"]) * int(row["mpi_ranks_per_node"])
                     * int(row["slurm_elapsed_seconds"]) for row in cost) / 3600
    assert stages[1010]["rho_max_g_cm3"] >= 1e-10
    assert all(job["newton_status_counts"] == {"converged": job["newton_status_counts"]["converged"]}
               and not job["opacity_nonzero_diagnostic_segments"] for job in jobs)
    assert abs(node_hours - 5.067222222222222) < 1e-9
    chart = base64.b64encode((ROOT / "first_core_profiles.png").read_bytes()).decode("ascii")
    stage_rows = "".join(tr(cycle,
                            f"{stages[cycle]['time_years']/1000:.4f}",
                            f"{stages[cycle]['rho_max_g_cm3']:.3e}",
                            f"{stages[cycle]['temperature_max_K']:.1f}",
                            f"{axes[cycle]['steepest_positive_x_density_drop_3_to_15_au']['inner_au']:.2f}–"
                            f"{axes[cycle]['steepest_positive_x_density_drop_3_to_15_au']['outer_au']:.2f}",
                            f"{stages[cycle]['octant_density_relative_spread']['5_to_10_au']*100:.3f}")
                         for cycle in (670, 780, 1010))
    cost_rows = "".join(tr(row["job"],
                           f"{row['used_start_cycle']}→{row['used_end_cycle']}",
                           row["allocated_nodes"],
                           f"{int(row['slurm_elapsed_seconds'])/60:.1f}",
                           f"{int(row['allocated_nodes'])*int(row['slurm_elapsed_seconds'])/3600:.3f}",
                           row["job_final_state"])
                        for row in cost)
    mass_drift = ((stages[1010]["gas_mass_g"] - stages[670]["gas_mass_g"])
                  / stages[670]["gas_mass_g"])
    solver_newton = sum(job["newton_status_counts"]["converged"] for job in jobs)
    solver_fg = sum(job["fgmres_solves"] for job in jobs)
    solver_it = sum(job["fgmres_iterations_total"] for job in jobs)
    solver_true = max(job["fgmres_max_true_relative_residual"] for job in jobs)
    max_newton = max(job["newton_converged_max_final"] for job in jobs)
    shock = axes[1010]["steepest_positive_x_density_drop_3_to_15_au"]
    html = f'''<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>WOR-92 第一コア形成の解析 | 2026-11-09</title>
<style>
:root{{font-family:system-ui,-apple-system,"Noto Sans JP",sans-serif;color:#17313a;background:#f3f7f8}}
*{{box-sizing:border-box}}body{{margin:0;line-height:1.65}}header{{background:#123b4a;color:white;padding:2rem max(1.2rem,calc((100vw - 1120px)/2))}}
header h1{{font-size:clamp(1.7rem,3vw,2.5rem);margin:.25rem 0}}header p{{color:#dcebef;max-width:850px}}
nav{{background:#dcebef;padding:.6rem 1rem;display:flex;flex-wrap:wrap;gap:1rem;justify-content:center}}a{{color:#086b83}}nav a{{font-weight:600}}
main{{max-width:1120px;margin:auto;padding:1rem 1.2rem 3rem}}h2{{margin:2rem 0 .7rem;border-bottom:2px solid #b8d3dc;padding-bottom:.3rem}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(205px,1fr));gap:.75rem}}.card,figure,.panel{{background:white;border:1px solid #d7e3e7;border-radius:10px;padding:1rem}}
.card strong{{font-size:1.35rem;display:block}}.good{{border-left:5px solid #147d69}}.caution{{border-left:5px solid #bd6a25}}
figure{{margin:1rem 0}}figure img{{max-width:100%;height:auto;display:block;margin:auto}}figcaption,.small{{font-size:.9rem;color:#536b74}}
.scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;background:white}}th,td{{text-align:left;padding:.55rem .65rem;border-bottom:1px solid #dfe9ec;vertical-align:top}}th{{background:#e2edf1}}
code{{font-size:.9em;overflow-wrap:anywhere}}.note{{background:#fff6e9;border-left:4px solid #bd6a25;padding:.8rem 1rem}}
.split{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1rem}}footer{{max-width:1120px;margin:auto;padding:0 1.2rem 2rem;color:#536b74}}
</style></head><body>
<header><div>ATHENA++ / WOR-92 / XD2000</div><h1>Table EOS・opacityで第一コアの目標密度に到達</h1><p>2026-11-09 JST。Tomida系EOSとSemenov opacityを使ったCFL=0.3の3D AMR計算。採用した計算経路の到達点と、物理解釈上の限界を分けて示します。</p></header>
<nav><a href="#result">到達点</a><a href="#structure">3D構造</a><a href="#solver">収束</a><a href="#conservation">保存量</a><a href="#cost">計算量</a><a href="#comparison">比較と残件</a><a href="#evidence">再現情報</a></nav>
<main>
<section id="result"><h2>到達点</h2><div class="cards">
<div class="card good"><strong>1.008×10⁻¹⁰ g/cm³</strong>cycle 1010で目標密度1×10⁻¹⁰ g/cm³を超え、正常終了。</div>
<div class="card good"><strong>29.1095 kyr</strong>3D HDF5出力に記録された計算時刻。</div>
<div class="card good"><strong>36.45 cells</strong>最終区間のJeans長解像度。停止閾値16を上回る。</div>
<div class="card caution"><strong>WOR-92は検証継続</strong>第一コア形成は確認。opacityの物理定義と質量増分の切り分けが残る。</div>
</div><p>job 338156はcycle 590→930、job 338171は930→1010を実行し、ともにSlurm COMPLETED 0:0。予備338183は先行到達を確認してスキップしました。</p></section>
<section id="structure"><h2>第一コアの3D構造</h2>
<figure><img src="data:image/png;base64,{chart}" alt="cycle 670、780、1010の球殻平均密度、ガス温度、流入速度" loading="lazy"><figcaption>3D leafセルの体積・質量重み平均。密度段差と流入速度の急増が外側へ移る。元のPNGは<a href="first_core_profiles.png">こちら</a>。</figcaption></figure>
<div class="scroll"><table><thead><tr><th>cycle</th><th>時刻 [kyr]</th><th>ρmax [g/cm³]</th><th>Tmax [K]</th><th>正のx軸の密度段差 [au]</th><th>5–10 auの8象限密度差</th></tr></thead><tbody>{stage_rows}</tbody></table></div>
<p>最終時点の中心ガス温度は431.253 K、放射温度は431.258 K。密度最大セルの半径方向速度は−0.0134 km/s。正のx軸{shock['inner_au']:.3f}–{shock['outer_au']:.3f} auで密度比{shock['inner_over_outer_density']:.3f}、内外流入速度は{shock['inner_infall_km_s']:.3f}/{shock['outer_infall_km_s']:.3f} km/s。7.013 au内のgas massは0.02845 M☉、8.272 au内は0.03150 M☉。継続する降着衝撃波、低速中心、ほぼ球対称の構造を確認しました。</p></section>
<section id="solver"><h2>ソルバと表の有効範囲</h2><div class="cards"><div class="card good"><strong>{solver_newton}/{solver_newton}</strong>最後の420 cycleのNewton statusは全件収束。</div>
<div class="card good"><strong>{solver_fg} solve</strong>FGMRESは計{solver_it}反復。</div>
<div class="card good"><strong>{solver_true:.3e}</strong>独立true linear relative residualの最大。</div>
<div class="card good"><strong>0</strong>42 segmentのopacity異常、最終3DのEOS表外セル。</div></div>
<p>Newtonの最終最大残差は高々{max_newton:.3e}（要求1×10⁻⁵）。最終3Dは3704 leaf MeshBlock、約189万セル。補正済みNewton line searchと、保存的粗細面中心化を有効にした実行binaryのSHA-256は<code>ac13cce7ac7d7281c12b863239d3f0c42e69d5b6396511145c410513cf496803</code>。</p></section>
<section id="conservation"><h2>保存量</h2><div class="split"><div class="panel"><h3>Gas mass</h3><table><tbody>
{''.join(tr(f'cycle {cycle}',f'{stages[cycle]["gas_mass_g"]:.10e} g') for cycle in (670,780,1010))}
</tbody></table><p>cycle 670→1010の相対変化は<code>+{mass_drift:.3e}</code>。完全な保存とは判定しません。</p></div>
<div class="panel"><h3>FLDと全エネルギー</h3><p>cycle 46の同一checkpoint A/Bでは保存的粗細面を有効にすると面積重み付きflux不一致が各方向約6×10⁻⁷から約2×10⁻¹⁷に下がりました。最終高密度状態での直接面監査は未実施です。</p><p>ambientのガス・放射温度を固定する源項があるため、全エネルギーは厳密な保存量として評価しません。</p></div></div></section>
<section id="cost"><h2>計算量の目安</h2><p class="panel"><strong>cycle 47保存点→目標密度: {node_hours:.2f}ノード時間、約{rank_hours:.0f} MPI rank時間。</strong> 最後のcycle 590→1010はMノード5台で31分18秒、2.608ノード時間。キュー待ち、cycle 0→46の初期保存点作成、重複した診断計算は含めません。</p>
<details><summary>採用したcheckpoint列の13ジョブを表示</summary><div class="scroll"><table><thead><tr><th>job</th><th>採用区間</th><th>ノード</th><th>実行時間 [分]</th><th>ノード時間</th><th>ジョブ結果</th></tr></thead><tbody>{cost_rows}</tbody></table></div><p class="small">job 338133はcycle 530まで進み、cycle 500から高解像度へ分岐。338144はcycle 588で失敗し、検証済みcycle 580から修正再開。両ジョブの実際の消費時間を含めます。</p></details></section>
<section id="comparison"><h2>基準計算との比較と残件</h2><div class="scroll"><table><thead><tr><th>指標</th><th>理想気体・一定opacity基準</th><th>今回のtable EOS・Semenov</th></tr></thead><tbody>
{tr('目標密度時刻','約29.1585 kyr','29.1095 kyr')}{tr('中心温度','約694.6 K','431.3 K')}{tr('衝撃波の位置','6.23–6.96 au','7.51–7.87 au')}
</tbody></table></div><p class="note">EOS、opacity、AMR解像度、計算経路が異なります。この表だけで差を単一の物理要因に帰属しません。SemenovのPlanck meanは文献上「extinction」と記述され、FLDの吸収係数として使うことはpilot近似です。高密度状態の粗細面flux、mass drift、独立したopacityモデルとの比較を継続します。</p></section>
<section id="evidence"><h2>証拠・再現情報</h2><ul>
<li><a href="README.md">全検証記録、入力・Git hash、制限</a> / <a href="runs.csv">ジョブ別の反復・残差・資源</a></li>
<li><a href="c1010_profile.csv">最終3Dの球殻平均</a> / <a href="c1010_axis.csv">正負x軸</a> / <a href="adopted_path_cost.csv">採用経路のコスト</a></li>
<li><a href="job_338156_summary.json">job 338156集計</a> / <a href="job_338171_summary.json">job 338171集計</a></li>
<li><a href="analyze_core_profile.py">3D球殻解析</a> / <a href="analyze_axis_profile.py">軸解析</a> / <a href="plot_first_core_profiles.py">図の生成</a></li>
</ul><p>最終HDF5のSHA-256: <code>bdd4c72640d16f1872725904c3a66bcd74e56577016f4593e114a3e91e25974f</code>。XD上の元データ: <code>/work/mizutaniks/output/fld/collapse/wor92_xd_20261107/long/job_338171/</code>。Gitブランチ: <code>codex/wor92-firstcore-integration-20261109</code>。このHTMLは図を埋め込んでおり単独で閲覧できます。</p></section>
</main><footer>Generated from local CSV/JSON and PNG evidence. This is a fixed analysis snapshot, not a live job monitor.</footer></body></html>'''
    output = ROOT / "report.html"
    output.write_text(html, encoding="utf-8")
    return output


if __name__ == "__main__":
    print(render())
