# RSG star-in-a-box commissioning の検証記録

2026-10-19 (JST)。計画・物理近似は [調査と再現計画](freytag_star_in_box.md) を参照。

**実装接続と短時間の数値検証を完了。RSG の定量再現・長時間安定性は未達成。**

## 実行環境と再現性

- 親 HEAD: `709910113f160c80a9353bf0ef717c64e6a8e913`。未コミットの FLD 変更を含むソースをコピーした。
- 隔離 build: `work_runs/freytag_rsg/build_source/`。
- 設定: `--prob=nrfld_star_in_box --eos=adiabatic -nrmgfld -hdf5 -mpi --cxx=g++`。
- ビルド: `make -j2`。流体は `hllc_fld`、double、HDF5 出力は single precision。
- 検証: `python3 tst/nrfld_validation/star_in_box.py --binary work_runs/freytag_rsg/build_source/bin/athena --output work_runs/freytag_rsg/validation_final --max-cores 2`。
- [機械可読の結果](../work_runs/freytag_rsg/validation_final/summary.json): 72 判定が通過。各入力・ログ・場も同ディレクトリに保存。
- 親 `Makefile`, `src/defs.hpp`, `bin/athena` と作業中の FLD 4 ファイルは保存した SHA256 と一致した。

実行バイナリ SHA256:
`0b0f66993e50b74c487f83152906769397c6aaf0c70d1da892eaaa9f0dd154c3`

入力テンプレート SHA256:
`086437c9503659e341efcf00bdea394ff12f5621a7fc29780c970babcf163d9a`

`work_runs/` は git 管理外。長期保存には build_source と結果一式を保管する。
通常の `--binary` なしの実行は各ソースの hash と build command も summary に残す。

## 通過した検証

| 検証 | 結果 | 範囲 |
|---|---|---|
| 16³ 初期質量 | 3.00000017 M☉ | HDF5 の single precision で積分。全立方体を含む |
| 初期加熱 | 90,000.0085 L☉ | 離散 W の規格化。初期・進化後で不変 |
| MeshBlock サイズ | 初期場の差 0 | 8³ block と 16³ block、同じ 16³ root mesh |
| MPI | 場の差 0 | 1 rank と 2 ranks、4 step 後の single-precision HDF5 |
| restart | 最大相対差 1.061×10⁻⁶ | 2+2 step 対 4 step。各変数の最大絶対値で規格化 |
| drag | 累積損失が正かつ単調、restart 後も一致 | コアの運動エネルギー除去量 |
| opacity table 接続 | 場の差 0 | 合成定数 HDF5 table と κP=κR=0.01 の直接指定 |
| Newton | 記録された status と final_max が許容範囲 | 初期 dt=0 または converged、最終 max residual ≤10⁻⁷、warning なし |
| 境界設定の誤り | 起動時に拒否 | `linearMG_ix1_bc=zerofixed` を混在させたケース |
| HSE 対照 | RMS Mach 0.03602 → 0.01269 | 16³ → 32³、n=3、t=0.01、加熱・drag・摂動・放射交換・拡散 off |

HSE の 2 解像度は単調改善を示すが、空間収束次数の確定ではない。
32³ の値は将来の Mach_rms<0.01 という提案条件にも未到達であり、
t=0.01 は 1 dynamical time より短い。長時間の well-balanced 性は主張しない。
MPI/table の「差 0」は保存精度での結果であり、全内部状態の bitwise identity を意味しない。

## 32³ の添付入力を使った実行

`work_runs/freytag_rsg/default32/` に、添付入力相当で t=10⁻⁴ まで 36 steps の結果を保存。
全 Newton solve が収束し、warning はなかった。

| 量 | 結果 |
|---|---:|
| 物理経過時間 | 約 1,826 s（約 30 分） |
| ガス質量 | 3.00000005 M☉ |
| 加熱 | 90,000.0058 L☉ |
| 最大 Mach | 0.005894 |
| 質量重み RMS Mach | 0.0008458 |

[診断 JSON](../work_runs/freytag_rsg/default32/diagnostic.json)、
[球殻 CSV](../work_runs/freytag_rsg/default32/diagnostic.csv)、
[診断図](../work_runs/freytag_rsg/default32/diagnostic.png) を生成した。
この時間は stellar turnover / pulsation period に比べて極めて短い。

## 実測で分かった制限

最初の 16³、CFL=0.2、dt≈0.01157 では Newton backtracking が枯渇した。
初期構造は gas HSE + LTE interior であり、定数 κ に対する放射平衡ではない。
密度の低い外層と大きな放射勾配を、最初から流体刻みだけで緩和できるとは限らない。
この失敗を NR の許容誤差緩和や opacity の逆算で隠さず、
添付の動作試験では **CFL=10⁻⁴** を使った（16³ で dt≈5.78×10⁻⁶）。

これは長時間本番の推奨 CFL ではない。次の実装段階は物理 EOS/opacity に基づく
初期放射構造の整合と立ち上げ手順の改善である。startup の timestep convergence、
解像度・外境界・drag 感度を確認してから、本番向けの時間刻みに移る。
現 pgen の drag は hydro/FLD の後の演算子分割なので、コアでは全体が
RK2 と同じ時間精度を持つと仮定しない。

保存的な放射 face luminosity、流出する質量/エネルギー、floor 補填の積算を含む
完全なエネルギー収支検証は未実装。長時間の成熟対流、周期、光球半径、
91,932 L☉ の放出光度への一致も未検証である。

## 2026-10-24: Linear WOR-13 仕様への本番経路追加

- `rsg_star_in_box` を `--eos=general/eos_table -nrmgfld` で隔離ビルドし、完走した。
- 本番 entry point は HSE profile、opacity table、`eos_table_clamp=false` を強制する。
- 手元の Athena EOS 表（SHA256 `bb71f42...fa560`）で、12 M☉ 固定 Freytag
  ポテンシャル、R=830 R☉、Tsurface=3487 K、Menv=3 M☉ の shooting が完走した。
- 低解像度生成 profile は正値、LTE 誤差 0、球内質量 3.0000044 M☉。
  32³ mapping は 2.9848972 M☉で、物理 run では `--grid 235` feedback を必須とする。
- `--grid 235` の独立実行では写像質量 2.9895434 M☉（目標から -0.349%）で、
  0.5% gate を通過した。profile は 320 点、LTE 最大相対誤差 0、内部 HSE 残差の
  median は 0.00661。有限差分 p99 は 1.83 と大きく、温度 profile の遷移と
  atmosphere floor を含むため、長時間 HSE acceptance の代用にはしない。
- 16³、実 EOS、既存テスト opacity、生成 profile の cycle=0 startup が通過し、
  opacity の out-of-domain/clamp/nonfinite はすべて 0。テスト opacity は接続試験専用。
- 同条件の 1 step は CFL=10⁻⁶ でも linear MG が相対 defect 約 0.984 で停滞し、
  Newton 100 回後の total-energy max defect 約 6.98×10⁻⁴ で失敗した。これは
  物理 opacity/HSE を用いた production evolution の通過を意味しないため、成功扱いにしない。
- 原著どおりの 235³ root grid と 47³ MeshBlock は 125 blocks として受理された。
- Python unit tests は重力の解析微分・極限、歴史的入力値、117³/470³ variant を検査する。

未完了なのは PHOENIX/COMA+OPAL 相当 opacity の確定、face-flux を使う
Marshak/vacuum 境界、adiabatic interior + gray atmosphere 初期モデル、完全なエネルギー収支、117³ の複数 dynamical-time run、
235³ の数年緩和、470³ convergence である。

## 2026-10-26: 指定NATA EOS・OPTAB opacityによる再検証

- NATA raw table
  `/home/kosuke/simulation/nata_eos/runs/athena_solar_fine_safe/nata_eos_rho_u_table.dat`
  から6-field Athena表を生成した。変換器は
  `tools/rsg/make_athena_eos_from_nata.py`、出力SHA256は
  `837819aaa71ba3fa5342cdc0849904fdfb04ae50fe4dccf5dfec788aa3460a06`。
- 指定opacityのSHA256は
  `cc5912dd3896c39c1ea3fde0d315dc16c136303a8c363ab58fa56b705b909aaf`。
  128×128、log T=3.0..6.3、log ρ=-23.300..-1.193で、両平均は有限・正値。
  初期状態と32³/64³の1-stepで範囲外、clamp、負値、非有限値はいずれも0。
- 現NATA raw出力はPtot、u、Sに光子寄与を含む。FLDのErと二重計上するため、
  このEOSは数値接続試験には使用できるがproduction物理runには不適格。
  material-only NATA表の生成をproduction gateとした。
- density floorを1e-12 g cm^-3として生成したHSE profileは、128³写像質量
  2.9985642 M☉（-0.0479%）、LTE最大相対誤差0、HSE残差中央値0.00164、
  p99=0.0983。SHA256は
  `6e50b1d9ac49b35f2707b0f0418a61d3b6efa0dc06e5a9cb6ae3d2aa03deea83`。
- 128³ cycle=0 startupは成功。最大RSS 4.48 GiB。成果物は
  `work_runs/freytag_rsg/nata_optab_20261026/cycle0_128/`。
- CFL=1e-6のcoupled 1-stepは32³でNewton 5反復、final max
  6.83e-9、64³で6反復、final max 5.27e-9として収束した。64³最大RSSは
  575 MiB。成果物は同ディレクトリの`one_step_32/`, `one_step_64/`。
- 入力既定のCFL=0.3（32³、dt=12.7827 code units）はNewton 12反復で
  `backtracking_exhausted`、final total-energy max=1となり失敗した。
  小CFLの成功を通常立ち上げ成功とは扱わない。成果物は`one_step_32_cfl03/`。
- 実行バイナリSHA256は
  `bead47bf68683624aa173a7c96b0e326841ef507fbd05a6043ada04038ad957f`。
  ユーザーの通常build設定は検証後に復元した。
- 文献の117/235/470格子は、39/47セル幅のMeshBlockではmesh-only検査後の
  NR-MG生成時にpower-of-two制約で停止した。このため実行系列を128/256/512へ変更し、
  `materialize_rsg_input.py`で不適合variantを事前拒否する。

この再検証で「指定テーブルを読み、範囲外参照なしでcoupled FLDを1 step解く」
ところまでは通過した。物理的RSG再現、通常CFL、長時間HSE、face-flux放射境界、
energy budget、material-only EOS、opacityの組成・由来確定は未完了である。

## 2026-10-29: WOR-53 uniform強拡散solver

- multi-MeshBlock/MPI対応の restarted right-preconditioned FGMRES を追加した。
  既存のV-cycleをflexible preconditionerとして使い、fine-grid matvecと内積は
  MeshBlock境界通信・MPI global reductionを通る。
- coarse operatorにGalerkin積 `R A_h P` を追加した。productionの`galerkin`は既存の
  8-child volume-average restrictionとscaled-transposeになるpiecewise-constant
  prolongationを使う。この組では7点fine operatorの厳密なRAPも7点なので、階層を
  7成分だけで構築し、edge/corner couplingが生じた場合は実行時に停止する。
  旧27点builderは削除せず、`coarse_operator=galerkin_27_experimental`を明示した
  場合だけ確保・実行する。trilinear transferの比較実験用でありRSG標準設定ではない。
- Galerkinはuniform topology限定である。非周期物理境界では、RSG callbackのように
  correction ghost-centreを0にすることを
  `galerkin_zero_physical_boundaries=true` で明示する必要がある。AMR/SMRは対象外。
- 32³、CFL=0.3、NATA EOS接続表＋指定gray opacityの1 stepは、single block、
  16³×8 blocks/1 rank、同/2 ranksのすべてでstrict convergenceした。FGMRESは各
  Newton段で16--19反復、相対残差は全て1e-8未満。1 rankと2 ranksの最終primitive
  HDF5は全要素一致した。
- 2 ranks、8 blocksの10 step試験では`max_update_fraction=0.5`はcycle 3で
  gas residualのbacktrackingに失敗したが、0.2では全10 stepが収束した。161回の
  線形solveは15--21反復（平均16.87）、最大相対残差9.94e-9。opacity診断は全項0。
- RSG入力はこの検証構成（FGMRES、Galerkin、constant prolongation、weighted
  Jacobi `omega=0.7`、update cap 0.2）を選択する。solver全体の既定値は従来の
  `mg`＋`rediscretized`のままであり、他問題の仕様は自動変更しない。

成果物は
`work_runs/freytag_rsg/nata_optab_20261026/wor53_fgmres_galerkin_constant_*`
に保存した。10 stepはsolverの継続動作確認であって、RSGの物理的長時間安定性や
完全なエネルギー収支を証明するものではない。

### 7点production Galerkinへの整理

同日、constant prolongationでは厳密な`R A_h P`が7点supportを保つことを代数的に
再確認し、`coarse_operator=galerkin`の確保・構築・適用を7点へ変更した。各fine rowを
8-child volume averageでaggregateし、coarse column offsetがface stencil外になれば
即時停止するため、edge/corner項が黙って切り捨てられることはない。

27点実装は削除していない。`coarse_operator=galerkin_27_experimental`と明示した場合だけ
27成分を確保し、起動時にもexperimental警告を表示する。trilinear transfer比較用で、
RSG標準入力は引き続き`galerkin`＋`constant`である。

32³ one-block、CFL=0.3、update cap=0.2で7点productionと27点experimental＋constantを
同一バイナリ内で比較した。両者は14回のNewton段、各FGMRES反復数、最終primitive
HDF5全配列が完全一致した。したがって、この構成の27点配列のedge/corner 20成分は
数値解に寄与していない。単回計測では7点版がCPU 1.252 s、最大RSS 119368 KiB、
27点互換版が2.109 s、126936 KiBで、7点化によりCPU 40.6%、全プロセスRSS 6.0%を
削減した（matrix配列単体の成分数は27から7、74.1%減）。

16³×8 blocksの1 rank/2 ranks one-stepはともに14 Newton段、FGMRES合計237反復で、
最終primitive HDF5は全要素一致した。2 ranks、10 stepでは161 linear solveが全て収束し、
15--21反復（平均16.87）、最大相対残差9.94e-9、CPU 10.213 s、opacity診断は全項0。
旧27成分production計測の14.649 sに対して30.3%短縮した。RSG unit test 7件、
`nrmgfld/general_eos_exchange_jacobian`、`nrmgfld/coupled_residual_norms`も合格した。

7点成果物は
`work_runs/freytag_rsg/nata_optab_20261026/wor53_galerkin7_*`に保存した。
