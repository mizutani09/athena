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
    comparison_chart = base64.b64encode(
        (ROOT / "reference_axis_comparison.png").read_bytes()).decode("ascii")
    opacity_chart = base64.b64encode(
        (ROOT / "opacity_replay_comparison.png").read_bytes()).decode("ascii")
    eos_chart = base64.b64encode(
        (ROOT / "firstcore_eos_audit.png").read_bytes()).decode("ascii")
    opacity_means_chart = base64.b64encode(
        (ROOT / "firstcore_opacity_means.png").read_bytes()).decode("ascii")
    absorption_branch_chart = base64.b64encode(
        (ROOT / "absorption_branch_comparison.png").read_bytes()).decode("ascii")
    firstcore_reference_chart = base64.b64encode(
        (ROOT / "firstcore_reference_comparison.png").read_bytes()).decode("ascii")
    shock_chart = base64.b64encode(
        (ROOT / "wor118_shock_tube_regression.png").read_bytes()).decode("ascii")
    eos_audit = read_json("firstcore_eos_audit.json")
    opacity_replay = read_json("opacity_replay_comparison.json")
    opacity_table = read_json("semenov_planck_absorption_trial.json")
    opacity_means = read_json("firstcore_opacity_means.json")
    absorption_branch = read_json("absorption_branch_comparison.json")
    thermal = read_json("c1010_thermal_coupling.json")
    conserved_mass = read_json("c930_c940_conserved_mass.json")
    hydro_flux = read_json("c930_hydro_mass_flux.json")
    hydro_faces = read_json("c930_hydro_mass_faces.json")
    hydro_controls = read_json("hydro_mass_controls.json")
    hydro_xorder1 = read_json("c930_hydro_mass_faces_xorder1.json")
    hydro_ambient = read_json("c930_hydro_mass_ambient.json")
    hydro_sinkoff = read_json("c930_hydro_mass_faces_sinkoff.json")
    hydro_no_radp = read_json("c930_hydro_mass_faces_no_radp.json")
    hydro_face_closure = read_json("c930_hydro_mass_faces_face_closure_trial.json")
    hydro_rad_order = read_json("c930_hydro_mass_faces_rad_order_trial.json")
    hydro_boundary_only = read_json("c930_hydro_mass_faces_boundary_only_trial.json")
    repaired_mass = read_json("c930_c940_boundary_only_conserved_mass.json")
    repaired_run = read_json("wor118_c930_c1010_repaired_run.json")
    repaired_vl2 = read_json("wor118_c930_c940_vl2.json")
    uniform_one_rank = read_json("wor118_uniform_one_rank_short.json")
    shock_regression = read_json("wor118_shock_tube_regression.json")
    firstcore_reference = read_json("firstcore_reference_comparison.json")
    ambient_regions = hydro_ambient["stages"]["0.00616113437895963317"]
    ambient_by_region = {region: sum(value["net_code"] for name, value in ambient_regions.items()
                                     if name.endswith(":" + region))
                         for region in ("ambient", "mixed", "core")}
    faces = {cycle: read_json(f"c{cycle}_face_audit.json")[0]
             for cycle in (930, 1000)}
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
    shock_rows = "".join(tr(n[1:], f"{q['fields']['rho']['baseline_reference_relative_l1']:.6f}",
                            f"{q['fields']['rho']['repaired_reference_relative_l1']:.6f}",
                            f"{q['fields']['rho']['front_baseline_variation_over_range']:.6f}→"
                            f"{q['fields']['rho']['front_repaired_variation_over_range']:.6f}",
                            f"{q['fields']['rho']['front_baseline_significant_extrema']}→"
                            f"{q['fields']['rho']['front_repaired_significant_extrema']}")
                         for n, q in shock_regression["resolutions"].items())
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
<nav><a href="#result">到達点</a><a href="#structure">3D構造</a><a href="#solver">収束</a><a href="#conservation">保存量</a><a href="#shock">shock tube</a><a href="#opacity">opacity試験</a><a href="#eos">EOS監査</a><a href="#cost">計算量</a><a href="#comparison">基準計算</a><a href="#evidence">再現情報</a></nav>
<main>
<section id="result"><h2>到達点</h2><div class="cards">
<div class="card good"><strong>1.008×10⁻¹⁰ g/cm³</strong>cycle 1010で目標密度1×10⁻¹⁰ g/cm³を超え、正常終了。</div>
<div class="card good"><strong>29.1095 kyr</strong>3D HDF5出力に記録された計算時刻。</div>
<div class="card good"><strong>36.45 cells</strong>最終区間のJeans長解像度。停止閾値16を上回る。</div>
<div class="card good"><strong>WOR-92 第一コア達成</strong>吸収表の全経路感度とEOS・保存量監査まで完了。hydroの厳密質量保存は別課題WOR-118。</div>
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
</tbody></table><p>cycle 670→1010の相対変化は<code>+{mass_drift:.3e}</code>、絶対量で{(stages[1010]['gas_mass_g']-stages[670]['gas_mass_g'])/MSUN_G:.3e} M☉（最終7 au内のコア質量の約{100*(stages[1010]['gas_mass_g']-stages[670]['gas_mass_g'])/(0.02845*MSUN_G):.3f}%）。完全な保存とは判定しません。</p></div>
<div class="panel"><h3>FLDと全エネルギー</h3><p>cycle 46の同一checkpoint A/Bでは保存的粗細面を有効にすると面積重み付きflux不一致が各方向約6×10⁻⁷から約2×10⁻¹⁷に下がりました。高密度checkpointのc930、c1000から1 stepずつ64 rankで監査し、各軸それぞれ280・296組の粗細面に対して、重み付き不一致は最大でも{max(faces[c]['faces'][f'x{i}:coarse_fine']['weighted_relative_mismatch'] for c in faces for i in (1,2,3)):.2e}でした。暗黙FLDの符号付き全域収支は最大{max(faces[c]['global_signed_balance_fraction'] for c in faces):.2e}。</p><p>c1000のSlurm終了状態だけは、実行スクリプトが正しく到達した1001を誤って931と比較したためFAILED。物理計算・64 rank監査ログ・final restartは正常です。ambientのガス・放射温度を固定する源項があるため、全エネルギーは厳密な保存量として評価しません。</p></div></div>
<p class="note">c930→940を15桁のhistoryで再実行すると、AMRのMeshBlock配置が同一でもgas massはc931→940で{opacity_replay['reference']['mass_change_c931_to_c940_relative']:.3e}増加。保存変数<code>cons/dens</code>を別途c930/c940で積分しても相対増分は{conserved_mass['relative_conserved_mass_change']:.3e}で、実行中のMeshBlock生成・破棄は0回。長期の質量増分をAMR再配置やhistory表示差だけでは説明できません。</p>
<h3>Hydro質量収支の切り分け</h3><p>隔離した診断binaryでc930→931の1ステップを監査。historyの質量増分{hydro_flux['history_mass_change_g']:.4e} gに対し、RK2両段階のhydro面flux発散の合計は{hydro_flux['predicted_flux_mass_change_g']:.4e} g（比{hydro_flux['predicted_over_observed']:.8f}）。差の大部分は<strong>RK2最終段階の同一level面と周期境界</strong>にある。粗細面は各軸で約4×10⁻¹⁸の相対不一致でした。</p>
<div class="scroll"><table><thead><tr><th>RK2段階</th><th>全域質量変化 [code]</th><th>同一level面のnet [code, x1/x2/x3]</th><th>周期境界のnet [code, x1/x2/x3]</th><th>粗細面の重み付き相対不一致</th></tr></thead><tbody>
{''.join(tr(label, f'{stage["block_net_delta_code"]:.4e}', ' / '.join(f'{stage["faces"][f"x{i}:same"]["net_delta_code"]:.3g}' for i in (1,2,3)), ' / '.join(f'{stage["faces"][f"x{i}:boundary"]["net_delta_code"]:.3g}' for i in (1,2,3)), f'{max(stage["faces"][f"x{i}:coarse_fine"]["weighted_relative_mismatch"] for i in (1,2,3)):.2e}') for label, stage in (("第1段階", hydro_faces["stages"][1]), ("最終段階", hydro_faces["stages"][0])))}
</tbody></table></div><p class="small">同一level面のnetのほぼ全部は外側level 3の面。原因となるghost状態やhydroの面flux共有処理は未確定で、今回のFLD保存的粗細面改良に原因を帰属しません。</p>
<h3>積分法と再構成次数の対照試験</h3><div class="scroll"><table><thead><tr><th>c930からの試験</th><th>終点</th><th>質量相対変化</th><th>解釈</th></tr></thead><tbody>
{tr('RK2・2次（元設定）','c940',f'{hydro_controls["controls"]["rk2_xorder2"]["relative_change_from_c930"]:+.3e}','参照')}
{tr('VL2・2次','c940',f'{hydro_controls["controls"]["vl2_xorder2"]["relative_change_from_c930"]:+.3e}','積分法の変更だけでは改善しない')}
{tr('RK2・1次','c931',f'{hydro_controls["controls"]["rk2_xorder1"]["relative_change_from_c930"]:+.3e}','外側同一level面の不一致が大きく減少')}
</tbody></table></div><p>1次再構成ではRK2最終段階のhydro面flux合計が{hydro_xorder1['stages'][0]['block_net_delta_code']:.2f} code単位に下がる一方、高密度側の同一level面に残差が残ります。1次化は精度を落とすため本線に採用していません。診断用の別binaryのみで試しました。</p>
<h3>WOR-118: ambient固定と放射圧の切り分け</h3><p>c930のトレーサーで、RK2最終段階の面に接するセルを分類しました。面flux差の符号付き和はambientのみ{ambient_by_region['ambient']:.3f} code（{100*ambient_by_region['ambient']/hydro_faces['stages'][0]['block_net_delta_code']:.1f}%）、coreのみ{ambient_by_region['core']:.3f} code（{100*ambient_by_region['core']/hydro_faces['stages'][0]['block_net_delta_code']:.1f}%）、混在{ambient_by_region['mixed']:.3f} code。周期境界だけでなく内部の同一level面にも差があり、物理的な境界流入ではありません。トレーサーはステップ開始時の分類であり、長期誤差の空間伝播を証明するものではありません。</p>
<div class="scroll"><table><thead><tr><th>XD M-debug対照</th><th>c930→931最終RK段階の面flux差 [code]</th><th>判定</th></tr></thead><tbody>
{tr('元設定・job 338248',f'{hydro_faces["stages"][0]["face_net_delta_code"]:.6f}','不保存')}
{tr('ambient温度固定OFF・job 338285',f'{hydro_sinkoff["stages"][0]["face_net_delta_code"]:.6f}','同一')}
{tr('Riemann放射圧OFF・job 338286',f'{hydro_no_radp["stages"][0]["face_net_delta_code"]:.3e}','丸め程度')}
{tr('面limiter共通化・job 338310',f'{hydro_face_closure["stages"][0]["face_net_delta_code"]:.3f}','外側の大きな差を解消、中心に残差')}
{tr('境界タスク順序のみ・job 338325',f'{hydro_rad_order["stages"][0]["face_net_delta_code"]:.6f}','効果なし、採用しない')}
{tr('境界面closureと放射source順序・job 338337',f'{hydro_boundary_only["stages"][0]["face_net_delta_code"]:.3e}','丸め程度')}
</tbody></table></div><p>温度固定OFFの10ステップ試験job 338284でも質量相対増分は+7.705×10⁻⁸（元設定+7.691×10⁻⁸）。温度固定は質量を直接書き換えません。放射圧OFFは原因の切り分け用で、物理モデルの恒久設定には採用しません。</p>
<p>原因は二つありました。(1) MeshBlock端のghost closureの傾きに必要な第3 ghost層が存在せず、同じ面のlimiterが両側で一致しないこと。(2) 放射境界交換の後に混合系sourceが実セルのE<sub>rad</sub>だけを変え、次のRK段階には古いghostが残ること。通常のimplicit mixed driftではsourceを放射境界交換より先に実行し、MeshBlock端だけλとa<sub>rad</sub>を隣接セル中心値から共通化しました。c930→931の同一level・周期境界の質量flux差は各軸で丸め程度。10ステップのjob 338338ではhistoryの全域gas massは15桁で不変、保存変数HDF5の体積積分差は{repaired_mass['relative_conserved_mass_change']:+.2e}（float32出力量子化を含む）。元設定の+7.691×10⁻⁸から改善しました。</p>
<h3>修正版の第一コア到達試験</h3><p>同じc930 restartからc1010まで放射圧を有効にしたまま5ノード・320 rankで再実行したjob 338342は、Slurm COMPLETED 0:0（4:54）、80/80 Newton収束、FGMRES {repaired_run['fgmres_solves']} solve・{repaired_run['fgmres_iterations_total']}反復、最大true residual {repaired_run['fgmres_max_true_relative_residual']:.3e}でした。高精度historyのgas mass相対変化は{repaired_run['history_relative_gas_mass_change']:+.2e}。目標密度{repaired_run['final_density_max_g_cm3']:.3e} g/cm³に到達し、衝撃波位置{repaired_run['shock_inner_au']:.2f}–{repaired_run['shock_outer_au']:.2f} au、EOS表外セル0、最終AMRは{repaired_run['meshblocks']} blockでした。元計算と最大密度は同じHDF5出力桁、最大温度差は相対{repaired_run['hdf_temperature_max_relative_to_original']:+.2e}。元計算の質量増分だけを除去できました。使用binaryとログSHAは<a href="wor118_c930_c1010_repaired_run.json">集計JSON</a>に記録しています。</p>
<p>別積分法VL2のc930→940・64 rank（job 338425）も{repaired_vl2['newton_converged']}/10 Newton収束、質量相対変化{repaired_vl2['gas_mass_relative_change']:+.1e}。元VL2の+7.390×10⁻⁸の増分が消えました。1 rankのuniform shock tube短時間試験は{uniform_one_rank['newton_converged']}ステップすべて収束し、t={uniform_one_rank['final_time_code']}で正常終了。振動の最終比較は2/4 rankのN64/N128で実施します。</p></section>
<section id="shock"><h2>放射圧を含むshock tubeの振動回帰</h2><p>Riemann solverの放射圧を有効にした同じ入力で、修正前後のFLD解をt=0.1まで実行しました。uniform N64は2 rank、N128は4 rankで、いずれもNewton全ステップ収束。独立の1D Rad-EOS HLLC N4096を参照し、圧縮波周辺x=0.60–0.88を拡大しています。</p>
<figure><img src="data:image/png;base64,{shock_chart}" alt="N64とN128のshock tubeの密度、ガス温度、放射エネルギーを修正前後と独立参照解で比較" loading="lazy"><figcaption>旧版と修正版の曲線はほぼ重なる。参照解との幅の差は両方に共通するFLD数値拡散を含む。元図は<a href="wor118_shock_tube_regression.png">こちら</a>。</figcaption></figure>
<div class="scroll"><table><thead><tr><th>解像度</th><th>密度L1・旧</th><th>密度L1・修正後</th><th>圧縮波周辺の正規化全変動</th><th>有意な極値数</th></tr></thead><tbody>{shock_rows}</tbody></table></div>
<p>N64/N128の密度L1はそれぞれ0.100440→0.100413、0.061106→0.061074。波面周辺の密度極値数はどちらも1→1、温度・放射エネルギーは0→0。N128の密度全変動の変化は約1.6×10⁻⁶で、視認できる新たな振動はありません。5物理量の新旧最大点差は参照解の全振幅の0.076%以下。入力・binary・ソースSHA、残差、実行時間、メモリ、全物理量の誤差は<a href="wor118_shock_tube_regression.json">比較JSON</a>と各実験ディレクトリのsummaryに記録しました。</p></section>
<section id="opacity"><h2>Planck吸収係数の感度</h2>
<p>Semenov公開表のPlanck平均は消散として記述されます。公開単色データの吸収/輸送消散比から第一コア温度域だけに適用する<strong>診断用</strong>Planck吸収表を生成しました。ρ=10⁻¹⁰ g/cm³における試験表/元表の係数比は14 Kで{opacity_table['correction_near_14_200_350_431K_at_rho1e-10']['14']:.3f}、350 Kで{opacity_table['correction_near_14_200_350_431K_at_rho1e-10']['350']:.3f}、431 Kで{opacity_table['correction_near_14_200_350_431K_at_rho1e-10']['431']:.3f}。Rosseland平均は同一です。</p>
<figure><img src="data:image/png;base64,{opacity_means_chart}" alt="第一コア共通範囲でのSemenov元表・吸収試験表・OptabのPlanck/Rosseland平均比較" loading="lazy"><figcaption>ρ=10⁻¹⁰ g/cm³、T=316–431 K。431 KのPlanck平均は元Semenov 5.91、診断表3.09、Optab 3.06 cm²/gで、後二者の比は{opacity_means['trial_planck_over_optab_at_431K']:.3f}。一方Rosseland平均はSemenov 5.01、Optab 4.15×10⁻⁸ cm²/gで、表の物理モデルは一致しない。</figcaption></figure>
<figure><img src="data:image/png;base64,{opacity_chart}" alt="cycle930からの吸収係数変更試験の密度履歴と差" loading="lazy"><figcaption>同一c930 restart・同一binary・64 rankで10 cycle。実線と破線はほぼ重なるため、下段は10⁻⁹単位の相対差を表示。</figcaption></figure>
<div class="scroll"><table><thead><tr><th>指標 c940</th><th>元表</th><th>診断用吸収表</th><th>相対差</th></tr></thead><tbody>
{''.join(tr(name, f'{opacity_replay["reference"]["final"][name]:.10e}', f'{opacity_replay["absorption_trial"]["final"][name]:.10e}', f'{opacity_replay["trial_over_reference_final_relative"][name]:+.3e}') for name in ("rho_max_g_cm3", "Tgas_max_K", "Erad_total_erg", "gas_mass_g"))}
</tbody></table></div><p>両試験ともNewtonは10/10収束、FGMRESは30 solve・292反復、最大true relative residualは1.98×10⁻⁸以下。最終3Dでは中心5 au内の|Tgas−Trad|は99%点{thermal['regions']['core_r_below_5au_and_rho_above_1e-14']['p99_abs_Tgas_minus_Trad_K']:.3f} K、最大{thermal['regions']['core_r_below_5au_and_rho_above_1e-14']['max_abs_Tgas_minus_Trad_K']:.3f} KでほぼLTEです。</p>
<h3>c590→1010の全経路感度</h3><p>同一c590 restartからPlanck表だけを変更した5ノードの二つのジョブ338230・338233は合計31:20で目標密度に正常到達。Rosseland表、binary、EOS、CFL、AMR上限は同一です。元表の経路と同じ474/474 Newton収束、1202 FGMRES solve・12063反復、opacity範囲逸脱0。最終c1010のAMR配置も3704 blockで完全一致しました。</p>
<figure><img src="data:image/png;base64,{absorption_branch_chart}" alt="Planck吸収補正を入れたc1010の正のx軸密度・温度とppm単位の差" loading="lazy"><figcaption>c1010の正のx軸。上二段は密度とガス温度、下段は診断用吸収表/元表−1をppm表示。</figcaption></figure>
<div class="scroll"><table><thead><tr><th>c1010指標</th><th>元表</th><th>診断用吸収表</th><th>相対差</th></tr></thead><tbody>
{''.join(tr(label, f'{absorption_branch["final"][key]["reference"]:.10e}', f'{absorption_branch["final"][key]["trial"]:.10e}', f'{absorption_branch["final"][key]["trial_over_reference_minus_one"]:+.3e}') for key,label in (("rho_max_g_cm3","最大密度 [g/cm³]"),("temperature_max_K","最大ガス温度 [K]"),("gas_mass_g","全域gas mass [g]")))}
</tbody></table></div><p>最大密度差は0.794 ppm、最大温度差は0.394 ppm。衝撃波位置は両方7.507–7.874 au。3D HDF5でρ&gt;10⁻¹⁴ g/cm³の486,172セルを比較した密度の99%点相対差は{absorption_branch['hdf_comparison']['1010']['fields']['rho']['p99_absolute_relative']:.2e}、最大{absorption_branch['hdf_comparison']['1010']['fields']['rho']['max_absolute_relative']:.2e}。第一コア形成までの結論はこの吸収係数の定義変更に対し安定しています。診断表は高温ガスopacity域の検証済み製品表ではありません。</p></section>
<section id="eos"><h2>第一コア温度域のEOS監査</h2><figure><img src="data:image/png;base64,{eos_chart}" alt="第一コア範囲のEOS熱容量" loading="lazy"><figcaption>表を実行ソルバとは別のPythonで読み、ρ=10⁻¹⁶〜10⁻¹⁰ g/cm³、T=14〜431.3 Kの480点を調査。破線は低温H₂/Heの並進熱容量。</figcaption></figure>
<p>H₂/He理想混合気体の解析式 P=ρk<sub>B</sub>T(X/2+Y/4)/m<sub>H</sub>（X=0.70、Y=0.28）に対する最大相対差は{eos_audit['pressure_max_absolute_relative_error']:.3e}。表のP→e逆変換の最大相対差は{eos_audit['energy_p_to_e_inverse_max_absolute_relative_error']:.3e}、差分で求めたc<sub>v</sub>は全点で正でした。温度のlog微分dlnT/dlneは472点で、セル内の解析傾きと独立差分の最大絶対差{eos_audit['temperature_log_derivative_max_absolute_error']:.2e}。対象は第一コア形成までで、H₂解離・第二崩壊のEOS妥当性を示すものではありません。</p></section>
<section id="cost"><h2>計算量の目安</h2><p class="panel"><strong>cycle 47保存点→目標密度: {node_hours:.2f}ノード時間、約{rank_hours:.0f} MPI rank時間。</strong> 最後のcycle 590→1010はMノード5台で31分18秒、2.608ノード時間。キュー待ち、cycle 0→46の初期保存点作成、重複した診断計算は含めません。</p>
<details><summary>採用したcheckpoint列の13ジョブを表示</summary><div class="scroll"><table><thead><tr><th>job</th><th>採用区間</th><th>ノード</th><th>実行時間 [分]</th><th>ノード時間</th><th>ジョブ結果</th></tr></thead><tbody>{cost_rows}</tbody></table></div><p class="small">job 338133はcycle 530まで進み、cycle 500から高解像度へ分岐。338144はcycle 588で失敗し、検証済みcycle 580から修正再開。両ジョブの実際の消費時間を含めます。</p></details></section>
<section id="comparison"><h2>基準計算との比較と残件</h2><figure><img src="data:image/png;base64,{comparison_chart}" alt="従来基準とWOR-92の密度、温度、流入速度の正のx軸比較" loading="lazy"><figcaption>両計算ともρmax≈10⁻¹⁰ g/cm³の時点。元の正のx軸データを比較し、球殻平均と混同しない。</figcaption></figure><div class="scroll"><table><thead><tr><th>指標</th><th>理想気体・一定opacity基準</th><th>今回のtable EOS・Semenov</th></tr></thead><tbody>
{tr('目標密度時刻','約29.1585 kyr','29.1095 kyr')}{tr('中心温度','約694.6 K','431.3 K')}{tr('衝撃波の位置','6.23–6.96 au','7.51–7.87 au')}
</tbody></table></div><p><a href="https://arxiv.org/html/2510.12620#S5">Mayer et al. のSection 5</a>は非回転・非磁化の1 M☉球状雲で、ρmax&gt;10⁻¹⁰ g/cm³の時刻を約29.09 kyr、第一コア半径を6–7 auと報告しています。今回の到達時刻29.1095 kyrと衝撃波位置7.51–7.87 auは同じ規模です。ただし論文のSection 5は固定EOS・opacity、AREPOの移動メッシュであり、ここでの計算はtable EOS・Semenov表・Athena++ AMRです。独立した厳密一致試験とは解釈しません。</p>
<h3>原著の半径プロファイルと解析的な極限</h3><figure><img src="data:image/png;base64,{firstcore_reference_chart}" alt="第一コアの球殻平均密度、温度、流入速度とMayer論文Figure 7の概読点、等温崩壊の解析漸近形" loading="lazy"><figcaption>実線は今回のc1010 3D球殻平均。丸印はMayer et al. Figure 7の図面からの概読値で数値表ではない。破線はLarson–Penston等温崩壊の外側漸近密度、点線はSIS。元図は<a href="firstcore_reference_comparison.png">こちら</a>。</figcaption></figure>
<div class="scroll"><table><thead><tr><th>比較</th><th>文献・解析基準</th><th>今回</th><th>解釈</th></tr></thead><tbody>
{tr('一様球の自由落下時間',f'{firstcore_reference["pressureless_uniform_sphere_tff_kyr"]:.3f} kyr',f'{firstcore_reference["snapshot_time_kyr"]:.3f} kyr','時刻比1.002。圧力なし一様球の尺度であり第一コアの厳密解ではない')}
{tr('外層密度の傾き（20–300 au）','Larson–Penston: r⁻²',f'r^{firstcore_reference["analytic_envelope"]["fitted_density_power_law"]:.3f}','漸近形との振幅比中央値1.282、RMS 0.099 dex。20 auで約65 Kなので等温極限ではない')}
{tr('第一コア半径','Mayer: 6–7 au / Commerçon FLD: 8.1 au','7.51–7.87 au','異なる初期条件・EOS・opacityの間で同程度')}
{tr('中心温度','Commerçon FLD: 411 K',f'{firstcore_reference["commercon_2011"]["model_peak_temperature_K"]:.1f} K','Mayer概読の中心温度はより高い。論文間でも物理設定が違う')}
{tr('第一コア質量','Commerçon FLD: 0.0233 M☉',f'{firstcore_reference["commercon_2011"]["model_mass_within_7p8au_msun"]:.4f} M☉ (r&lt;7.8 au)','質量定義・初期状態が異なり厳密一致は要求しない')}
</tbody></table></div><p><a href="https://arxiv.org/abs/0904.3568">Gong &amp; Ostriker (2009)</a>のLarson–Penston外層漸近形ρ≈8.86 a²/(4πGr²)を14 Kの音速で評価しました。Mayerの概読点との半径別比較は<a href="firstcore_reference_comparison.json">JSON</a>に記録。1–100 auの温度は本計算の方が低く、表EOS・opacityと固定係数の違いを含むため一致したとは主張しません。<a href="https://arxiv.org/html/1102.2921v1#S4.SS1">Commerçon et al. (2011), Table 1</a>も10 K開始の別条件です。</p><p class="note">EOS、opacity、AMR解像度（基準の衝撃波付近0.732 au、今回0.366 au）、計算経路が異なります。差を単一の物理要因に帰属しません。公開単色データで補正したPlanck吸収表のc590→1010全経路感度はppm以下の最終量差でした。厳密なhydro質量保存は<a href="https://linear.app/work-on-fld/issue/WOR-118/amr-collapseのhydro同一level面周期境界の質量flux不一致を解消する">WOR-118</a>で追跡します。高温ガスopacityと第二崩壊は今回の検証範囲外です。</p></section>
<section id="evidence"><h2>証拠・再現情報</h2><ul>
<li><a href="README.md">全検証記録、入力・Git hash、制限</a> / <a href="runs.csv">ジョブ別の反復・残差・資源</a></li>
<li><a href="c1010_profile.csv">最終3Dの球殻平均</a> / <a href="c1010_axis.csv">正負x軸</a> / <a href="adopted_path_cost.csv">採用経路のコスト</a></li>
<li><a href="job_338156_summary.json">job 338156集計</a> / <a href="job_338171_summary.json">job 338171集計</a></li>
<li><a href="c930_face_audit.json">c930粗細面監査</a> / <a href="c1000_face_audit.json">c1000粗細面監査</a> / <a href="audit_high_density_faces.py">監査コード</a></li>
<li><a href="c930_c940_conserved_mass.json">保存変数の質量増分</a> / <a href="audit_conserved_mass.py">体積積分コード</a></li>
<li><a href="c930_hydro_mass_flux.json">hydro質量flux合計</a> / <a href="c930_hydro_mass_faces.json">面別内訳</a> / <a href="audit_hydro_mass_faces.py">面監査コード</a></li>
<li><a href="hydro_mass_controls.json">VL2・1次の対照試験</a> / <a href="c930_hydro_mass_faces_xorder1.json">1次の面別監査</a></li>
<li><a href="c930_hydro_mass_ambient.json">ambient/core別の面監査</a> / <a href="audit_hydro_mass_ambient.py">分類コード</a> / <a href="c930_hydro_mass_faces_sinkoff.json">温度固定OFF</a> / <a href="c930_hydro_mass_faces_no_radp.json">Riemann放射圧OFF</a> / <a href="c930_hydro_mass_faces_face_closure_trial.json">面limiter試験</a> / <a href="c930_hydro_mass_faces_rad_order_trial.json">タスク順序試験</a></li>
<li><a href="c930_hydro_mass_faces_boundary_only_trial.json">採用候補の面監査</a> / <a href="c930_c940_boundary_only_conserved_mass.json">10ステップ保存変数</a> / <a href="wor118_mdebug_boundary_only_c930to940.sbatch">XD再現入力</a></li>
<li><a href="wor118_c930_c1010_repaired_run.json">修正版c930→1010の反復・質量・最終量</a> / <a href="wor118_mtest_production_c930to1010.sbatch">XD再現入力</a> / <a href="summarize_wor118_repaired_run.py">集計コード</a></li>
<li><a href="wor118_c930_c940_vl2.json">修正版VL2の10ステップ</a> / <a href="wor118_mdebug_vl2_c930to940.sbatch">XD再現入力</a> / <a href="wor118_uniform_one_rank_short.json">1 rank uniform短時間試験</a></li>
<li><a href="wor118_shock_tube_regression.json">shock tube N64/N128新旧比較</a> / <a href="wor118_shock_tube_regression.png">圧縮波の拡大図</a> / <a href="compare_wor118_shock_tube.py">比較コード</a></li>
<li><a href="wor118_rad_sync_diagnostic.txt">RK段階の放射ghost追跡</a> / <a href="wor118_mdebug_rad_sync_diag_c930to931.sbatch">XD再現入力</a></li>
<li><a href="opacity_replay_comparison.json">Planck補正A/B</a> / <a href="semenov_planck_absorption_trial.json">診断表の生成記録</a> / <a href="firstcore_eos_audit.json">EOS独立監査</a></li>
<li><a href="absorption_branch_comparison.json">c590→1010の吸収表感度</a> / <a href="semenov_planck_absorption_trial.h5">診断用HDF5表</a> / <a href="compare_absorption_branch.py">比較・作図コード</a></li>
<li><a href="firstcore_opacity_means.json">Semenov・Optab共通範囲の表比較</a> / <a href="compare_firstcore_opacity_means.py">比較図コード</a></li>
<li><a href="c1010_thermal_coupling.json">最終3Dのガス・放射温度差</a> / <a href="audit_thermal_coupling.py">温度差監査コード</a></li>
<li><a href="reference_axis_comparison.json">基準軸との比較</a> / <a href="baseline_first_core_xline.csv">基準の元x軸CSV</a> / <a href="compare_reference_axis.py">比較図コード</a></li>
<li><a href="firstcore_reference_comparison.json">Mayer・Commerçon・解析漸近形</a> / <a href="mayer_fig7_approx_read_off.csv">Mayer図面概読点</a> / <a href="compare_firstcore_references.py">比較図コード</a></li>
<li><a href="analyze_core_profile.py">3D球殻解析</a> / <a href="analyze_axis_profile.py">軸解析</a> / <a href="plot_first_core_profiles.py">図の生成</a></li>
</ul><p>最終HDF5のSHA-256: <code>bdd4c72640d16f1872725904c3a66bcd74e56577016f4593e114a3e91e25974f</code>。XD上の元データ: <code>/work/mizutaniks/output/fld/collapse/wor92_xd_20261107/long/job_338171/</code>。Gitブランチ: <code>codex/wor92-firstcore-integration-20261109</code>。このHTMLは図を埋め込んでおり単独で閲覧できます。</p></section>
</main><footer>Generated from local CSV/JSON and PNG evidence. This is a fixed analysis snapshot, not a live job monitor.</footer></body></html>'''
    output = ROOT / "report.html"
    output.write_text(html, encoding="utf-8")
    return output


if __name__ == "__main__":
    print(render())
