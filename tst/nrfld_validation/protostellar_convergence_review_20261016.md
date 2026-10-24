# Protostellar core FLD: 実装整理と収束性調査の記録

対象: このスレッドの利用可能な会話、添付ログ、repository の履歴と
`validation/` に残る結果。作業開始時 HEAD は `9e2d30a2`。
本資料は「実装で確認できること」「実測」「原因候補」を分離する。
過去の全 assistant 回答・全 remote 生データが揃っているわけではないため、
存在しない測定値や、再現していない本番成功を補完しない。

## 1. 今回戻した範囲

次の最近の実験追加を working tree 上で撤去した。履歴は書き換えていない。

| commit | 撤去したもの |
|---|---|
| `9e2d30a2` | completed V-cycle damping と回復用 virtual hook、runtime parameters |
| `c3f56825` | completed V-cycle の alpha 探索診断 |
| `e4c503da` | 発散 V-cycle の level/interior/AMR-boundary 専用診断 |

変更した production source 4ファイルと Mayer input は `09422a69` 時点と一致。
lagged Jacobian (`3ca41cc9`) と mixed-frame drift treatment 比較 (`9d943692`)
はそれぞれ既に revert 済みであり復活させていない。
前の回答で提案した `vcycle_damping_max_cycles=1,2,4` は未実装で、今回も追加しない。

「基本実装」は upstream Athena++ そのものではなく、実験追加前の FLD 実装である。
以下は維持した既存仕様・修正:

- end-of-step の陰的更新1回。subcycling は採用しない。
- 同じ Newton 方向を保持する line search。
- 発散 V-cycle の rollback、既存 post smoother、既存 symmetric red/black 設定。
- boundary-aware direct coarse solve、FLD 固有の係数制限処理。
- `09422a69` の mixed-frame coarse-row balance 修正。これは対角と面の
  1/dx スケーリングを整合させる修正で、今回の主因と証明されたわけではない。
- HLLD-FLD、ideal EOS、constant opacity、AMR、background heat sink。

`src/gravity/mg_gravity.cpp` は変更していない。
Mayer input の `post_mg_smooth=16` も変更していない。
過去の診断ログ・restart・ジョブスクリプトは証拠として残す。
旧 damping スクリプトを新しい input にそのまま使用すると、削除した runtime
parameter が見つからず起動エラーになる。古いバイナリも自動更新されないため再ビルドが必要。

## 2. 問題設定と、これまでに行った作業

物理設定は 1 Msun、半径3000 au、14 K、mu=2.381、gamma=5/3、
kappa_P=kappa_R=1 cm^2/g。Section 5 比較では回転・磁場なし。
比較点は rho_max が初めて 1e-10 g/cm^3 を超える snapshot。
会話内の短期ジョブ成功は、この点までの collapse 完走や論文との一致を意味しない。

| 段階 | 内容と到達点 |
|---|---|
| HLLD-FLD / pgen | radiation-negligible と B=0 の比較、protostellar_core_fld を導入。旧結果は `validation/hlld_fld_ot_tiny`, `hlld_ot_3d`, `hllc_b0_out`, `hlld_fld_b0_fixed` に存在。今回この solver 極限試験は再実行していない |
| benchmark 準備 | Jeans criterion、必要に応じ cell-mass refinement、初期 tau/alpha/beta、history 診断を追加。uniform grid は first core 解像度の代替にならない |
| background | 現在は passive scalar で選択した ambient の温度を固定する heat sink。sink particle ではない。pgen 冒頭にも sink particles は対象外と明記 |
| HPC 初期問題 | Intel SIMD 内 throw、build/restart/input の不整合、OOM、wall-time、巨大ログ、MPI終了エラーを順に切り分け。物理・数値非収束とは別分類 |
| Newton / MG | max_iterations、slow-convergence、backtracking_exhausted が発生。iteration cap 増加で通るケースもあったが、その後の失敗は解決せず |
| local 縮小試験 | uniform/AMR、CFL、smoother/omega、coarse direct、境界、Schur 対角、subcycles、post smoother を調査。小格子成功だけでは大格子 AMR の保証にならない |
| 更新順序 | 陽的ステージ後の陰的更新と primitive 同期を整理。陰的更新1回を採用。subcycles/post smoother の追加調整はユーザー意向で採用しない |
| fixed Newton direction | backtracking のたびに線形解を作り直して別方向になる問題を修正。coarse retry は default 0、smoothing-only fallback は false |
| test4–test6 | seed/restart と refinement を段階的に検証。1128→4488 MeshBlocks、P群大メモリを利用。`numlevel=9` 設定だけでは実際に level9 まで refine した証明にならない |
| rank 比較 | 同じ cycle7 restart を56/112 ranksで実行し、両者55反復収束。後述の残差差はあるが、rank依存の重大バグをこの結果だけで断定できない |
| test7 | mixed-frame、prolongation、coarse scale、CFL、固定 V-cycle 数、completed-cycle damping を比較。単一ステップ改善が連続計算で維持されず、今回実験追加を撤去 |

## 3. どの方程式・どの場所で問題になるか

`src/nr_multigrid/NRFLD.cpp` の `CalculateDefect` は、概略として

```
Fg = (eg - eg_old) + dt*(S - M)
Fr = (Er - Er_old) - dt*(S + L_D(Er) - M)
S  = c*rho*kappa_P*(a*T(eg)^4 - Er)
L_D(Er) = div(D grad Er),   D = c*lambda/(rho*kappa_R)
M  = mixed-frame drift term (面係数と近傍 Er から評価)
Fg + Fr = delta_eg + delta_Er - dt*L_D(Er)
```

を評価する。実装では総エネルギー残差を直接計算し、stiff な S の引き算での
桁落ちを避ける。coupled mode の停止判定は gas と total energy の L2/max。
ログの `radiation_l2=0` は独立 radiation 残差をこの判定に使用しないという意味であり、
放射方程式が完全に解けた意味ではない。

gas correction を消去した放射の線形系では、局所部分の Schur 対角は

```
B = dt*c*rho*kappa_P * 4*a*T^3 * dT/deg
C = dt*c*rho*kappa_P
A_local = 1 + C/(1+B)
```

（exchange のみの場合）となる。これに拡散 stencil と mixed-frame stencil が加わる。
コードでは `DCCS` が局所対角（および drift/該当時Robin）、`DCCF` と `DXMF` 等が
dt/dx^2 を掛ける拡散部分、`DXMS` 等が既に dt/dx を含む drift 部分を保持する。

問題が観測された経路は次のとおり。

1. `NRFLD::CalculateMatrix` が gas/radiation から Schur 線形系を組み立てる。
2. `linearMGDriver::Solve` が係数階層を構成する。
3. `MultigridDriver::SolveIterative` の V-cycle 後に線形残差が増える。
4. 基本実装はその cycle を棄却し、cycle 前へ rollback して線形 solve を終了する。
5. `linearMGDriver::Solve` の既存 fine-grid post smoothing 後の方向を Newton に返す。
6. `NewtonRaphsonDriver::SolveIterative` が物理残差の減少を検査する。
   改善が遅ければ iteration cap/stagnation、許容する step_scale を試しても
   減少が得られなければ `backtracking_exhausted` になる。

したがって slow-convergence は「Newton の T^4 非線形性が強い」という説明だけでは足りない。
内側の MG が有用な線形解を十分得られないことが実測されている。
ただし、線形残差の減少だけでは外側の正規化非線形残差の減少を保証しない。
line search の打切り範囲より小さい step まで数学的に無効と証明したわけでもない。

## 4. 現在の原因判定

### 確認できた事実

- cycle7、CFL=.02 の最初の completed V-cycle:
  `global_old=0.4495986723`, `global_new=0.6401443542`, factor=`1.4238128216`。
  これは線形反復であり、非線形 line search より前に生じる。
- 同じログの physical AMR level 別 RMS factor:

  | level（内部 absolute level 表記） | interior | AMR-boundary |
  |---|---:|---:|
  | 3 | 0.309 | 0.285 |
  | 4 | 1.763 | 1.203 |
  | 5 | 1.401 | 1.738 |

  「AMR-boundary」は粗密差のある MeshBlock 面の隣接セルであり、物理領域の外端ではない。
  interior にも増加があり、「外端だけ無視すればよい」という結論は支持されない。
  これだけで星中心・core/ambient界面のどちらが原因かは特定できない。
- 56/112 ranks 比較は両者55反復、初期残差一致。
  final L2=2.12348e-8 / 2.12489e-8、max=7.23946e-7 / 6.92771e-7。
  fields 全体の rank independence はまだ検証されていない。
- 別の local coefficient 診断 (`validation/coefficient_levels_baseline/run.log`) では
  local Schur 相当1.02–2.96に対して拡散対角4.45e4–4.45e6、面係数 contrast 約100。
  行和誤差は~5e-16。これは拡散優勢・大きな係数差を示すが、別設定の測定値であり
  cycle7 の全セル・全AMR階層を証明する数値ではない。

### 支持の弱い、または否定された説明

- **mixed-frame drift が cycle7 の主因**:
  implicit/off/explicit の3試験すべて55反復。最初の factor は
  1.423812821609 / 1.423812821566 / 1.423812821564。
  この状態で主要因とは考えにくい。高速流・後期collapse一般まで除外したわけではない。
- **lagged Jacobian にすれば解決**: exact/lagged とも55反復、55 V-cycle棄却。
- **prolongationをconstantにすれば解決**: trilinearの最初のfactor1.424に対し
  constantは4.368、constant試験は128反復で失敗。
- **iteration上限だけ不足**: cycle10の失敗は3反復でbacktracking exhausted。
  上限128→192ではこの種の失敗を解決しない。
- **MPI終了時の134が数値失敗の原因**: 134だけでは区別不能。
  `status=converged` と正常終了・restartを確認する必要がある。
  Job325328では明示的に非線形失敗しており、環境問題として除外してはいけない。

### 残る有力候補（未確定）

拡散優勢の Schur 系で、粗密レベル間の係数制限・補間・境界演算子と
細格子側の残差演算子の整合性が不十分である可能性がある。
`D ~ 1/(rho*kappa_R)` と `dt/dx^2` により低密度 ambient と細格子で
係数が大きくなりやすく、対角と隣接項の差として残る小さな局所項の扱いが重要になる。
ただし flux limiter が効くため、全領域で常に単純な 1/rho スケーリングとは限らない。

core/ambient の密度差、background-temperature fixing、熱交換と拡散の組合せ、
MGとNewtonでのghost値の扱い、係数階層更新、正規化meritとfloorの影響は候補として残る。
どれか一項の削除で連続失敗が解消することはまだ証明されていない。
過去に「1回で線形 solve を終了することが原因」と述べた説明も仮説であり、
完了V-cycleを複数回に増やす修正を実証済みの解決策として扱わない。

streaming/Gaussian/熱交換テストに合格することと、可変係数・stiff exchange・
AMR・背景sinkが同時にある問題の収束性は別の検証である。
物理的な球対称性は、Cartesian AMR上の線形系の容易さを保証しない。

## 5. 最近の数値試験を横並びで比較

すべて同じ条件とは限らない。特に古い build、input、rank数、停止許容値、
同じ物理時間まで進めたかを区別する。表のCPU秒はAthena表示値で、ジョブ総wall-timeではない。

| cycle7 試験 | Newton | 結果・CPU秒 |
|---|---:|---|
| CFL .02 baseline | 55 | PASS, 81.1 |
| CFL .01 baseline | 30 | PASS, 48.3 |
| CFL .005 baseline | 10 | PASS, 192.0（内側MGの費用が大きい） |
| CFL .0025 baseline | 9 | PASS, 77.0 |
| coarse scale .75 | 117 | backtracking exhausted |
| coarse scale .50 | 128 | max_iterations |
| coarse scale .25 | 13 | stagnation |
| 固定V-cycle 1回, CFL .02 | 63 | PASS, 77.7、backtracking115 |
| 固定V-cycle 2回, CFL .02 | 2 | backtracking exhausted |
| 固定V-cycle 4回, CFL .02 | 128 | max_iterations |
| completed damping, CFL .02 | 67 | PASS, 98.2、backtracking5 |
| completed damping, CFL .01 | 13 | PASS, 29.5、backtracking4 |

completed damping の連続試験 Job325328 は以下。

| cycle | dt | Newton | status |
|---|---:|---:|---|
| 7 | .126635 | 13 | converged |
| 8 | .126519 | 70 | converged |
| 9 | .126417 | 91 | converged |
| 10 | .126169 | 3 | backtracking_exhausted |

alpha=.5採用13回、alpha=.25採用72回、回復85回、棄却93回、backtracking133回。
cycle10 の final gas L2=.02966、total-energy L2=.47283、max=1。
初期 total-energy L2~.562 から許容値まで減少せず、単なる roundoff 停滞ではない。
dtはほぼ一定なので、dt急増が原因という説明もこの系列には当てはまらない。
alphaの集計だけから各cycle内の全採用時刻は復元できない。
final.rst はなく、この失敗runを正常checkpointとして使用しない。

## 6. 今後、原因を確定するために必要な最小調査

新しいsolver調整を増やす前に、失敗直前の同一状態を固定して調べる。

1. 同一binary/input/restart、同一dtで線形operatorと右辺を特定する。
   cycle7で通ることとcycle10で通ることは別。失敗前の正常checkpointが必要。
2. 既存 `jacobian_check`、`diagnostic_verbosity=2`、係数診断を小格子に限定して使い、
   gas/total residual、最大セルの座標・rho・T・Er・refinement levelを記録する。
3. 同じセルで Newton Jacobian作用と MG細格子operator作用を照合する。
   内部・同level block境界・coarse/fine境界・物理境界を区別する。
4. 次に coarse coefficient の行和・定数モード、R A_h P と A_H の違いを測定する。
   非Galerkin MGの差は即バグではなく、誤差増幅との関連を定量化する。
5. 新鮮な小問題で uniform/AMR、密度差あり/なし、heat sinkあり/なしを1条件ずつ
   比較する。物理条件を簡単にした試験は原因分離であり、本番の代用ではない。

今回のロールバックと精度試験ではこれらの本番原因確定までは実施していない。
許容値緩和・警告続行で「成功」に変えることもしない。
`warn_keep_old` は失敗した陰的更新をskipするので、将来のvalidation/runは
`failure_policy=abort` を明示して評価する。

## 7. 新しい拡散精度validation

`diffusion_convergence.py` を追加。詳細な実行方法は README に記載。
Eerror = Cx*dx^2 + Ct*dt とすると、diffusion number一定で dt∝dx^2 なので
誤差はdxに対して2次、dtに対して1次に見える。これは時間精度の独立証明ではない。
そこで同じNのdt/2計算を使い時間誤差の先頭項を外挿で除いた空間試験と、
N固定でdtを半分ずつにする時間自己収束試験を併記する。

Gaussian analytic solution が一定Dを仮定するため固定lambda=1/3を使用する。
flux-limited streamingや非線形熱交換の精度をこの試験で主張しない。
計算は既存 `nrfld_diff` と既存inputを再利用し、物理solverを変更しない。

今回、差し戻したソースからisolated buildし、最大2コアで実行した。

| 試験 | 結果 |
|---|---|
| 既存 diffusion、16^3、92 cycles | PASS、既存history相対L1=0.00425233 (<0.01) |
| 既存 coupling、100 cycles | PASS、最大log gas-energy誤差=0.000429071 (<0.005) |
| 新規固定diffusion number、N=16/32/64 | PASS、相対L1=9.04781e-3 / 2.30923e-3 / 5.77028e-4 |
| 同系列のdx次数 | 1.97016 / 2.00070 |
| 同系列のdt次数（混合精度） | 0.98508 / 1.00035 |
| 時間外挿後の空間次数 | 1.98512 / 2.00480 |
| 固定N=32の時間次数 | 0.98902 / 0.99454 |

固定diffusion numberは約0.0799447、tlim=0.02、CFLは
0.103280 / 0.051640 / 0.025820。実測dtは0.0005 / 0.000125 / 0.00003125。
時間試験のdtは0.0005 / 0.00025 / 0.000125 / 0.0000625。
新規試験のNR許容値1e-11、線形許容値1e-12。丸め誤差による最後の極小dtは記録する。

既存2試験: `validation/baseline_reset_20261016/summary.json`。
新規試験の全設定・profile・解析結果・図は
`validation/diffusion_convergence_reset_v3_20261016/` に保存。
初回の試験runnerは単精度HDF5のTimeを倍精度として厳密比較して停止したが、
計算自体はtlim到達済みだった。現在は倍精度の終了ログとHDF5時刻を併用する。
これはvalidation側の時刻読取り修正で、solver変更ではない。

以上は一様格子、一定係数拡散・局所熱交換の検証である。
AMR collapseの完走、HLLD-FLDの全回帰試験、大規模restartの再検証は今回行っていない。

## 8. 証拠の所在

- rank比較、mixed-frame implicit/off/explicit、Job325328: この会話のユーザー提示ログ。
- AMR level位置診断・prolongation比較: 添付 `f79bd26b-9222-424c-a361-63cc6f7fdf96`。
- coarse scale .75/.50: 添付 `56ff2f79-29a0-41c3-916f-0889b336f859`。
- coarse scale .25: 添付 `1bdcafd0-25b5-4b1c-b764-37c8318b9613`。
- CFL系列・alphaスキャン: 添付 `e5abcc7f-7308-4530-aa9c-ec2e76cda832`。
- CFL系列CPU時間: 添付 `1257aa5a-f2a3-4f87-9a97-c7a071990ff9`。
- 固定V-cycle試験: 添付 `f68e297e-1bc3-4756-8115-912de5e4b013`。
- damping単ステップ: 添付 `4b37688a-3308-44c8-947d-81db64cfe23b`。
- local旧係数診断: `validation/coefficient_levels_baseline/run.log`。
- 以前のgeneral-EOS/総エネルギー残差調査:
  `long3_failure_validation.md`, `total_energy_residual_validation.md`。
  これらはsolar系の別設定で、今回のideal-EOS collapse失敗の直接原因として混同しない。
