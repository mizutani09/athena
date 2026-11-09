# WOR-92 first-core 継続検証（2026-11-09）

**2026-11-10更新:** WOR-118の保存性修正を含むソースで初期状態c0から再計算し、XD jobs 338469/338485/338486がすべて正常終了した。c1080（29.11046 kyr）の最大密度は`1.04414476e-10 g cm^-3`、降着衝撃波候補は7.507–7.874 au、gas mass相対変化は`-4.8394e-15`。現在の修正後全経路の結果は[独立HTML](wor118_full_c0_report.html)、[再現記録](wor118_c0_full_run_20261109.md)、[数値要約](wor118_full_c0_summary.json)を参照。検証ソースcommitは`ff60f8f0`、この記録は別ブランチ`codex/wor118-radiation-face-consistency-20261109`。以下のc1010・質量漏れの記述は修正前経路の履歴である。

**最新結果:** cycle1010でρmax=`1.00819e-10 g cm^-3`、Jeans最小36.45 cells、第一コア形成を3D密度・温度・速度構造から確認。採用したcycle47保存点以降の計算量は約5.07ノード時間。[単独で閲覧できるHTMLレポート](report.html)に3D構造・ソルバ収束・保存量・計算量をまとめた。最終詳細は[第一コア密度到達・最終解析](#第一コア密度到達最終解析2026-11-09-1251-jst)、元の密度・温度・流速図は[PNG](first_core_profiles.png)を参照。途中の「未到達」「継続中」は各時点の履歴である。

**最終判定:** c590→1010のPlanck吸収補正A/B、第一コアEOS微分、c930/c1000のFLD粗細面、hydro質量収支まで診断した。WOR-92の第一コア形成・CFL0.3という達成範囲は完了。gas massの厳密な保存は満たしておらず、相対`+2.930e-6`の残存誤差を[WOR-118](https://linear.app/work-on-fld/issue/WOR-118/amr-collapseのhydro同一level面周期境界の質量flux不一致を解消する)で切り分け・修正する。第一コア後の高温ガスopacityや第二崩壊は今回検証していない。

## コードと入力

- 隔離 worktree: `/home/kosuke/simulation/athena-wor92-firstcore-integration`
- ブランチ: `codex/wor92-firstcore-integration-20261109`、修正 commit `f9dc679330bf76dcef0addcbc83b02d5d9121107`（push 済み）。元の `athena` worktree と main は変更していない。
- 元の WOR-92 適応的 inexact Newton コード: `96a4d442b9b6ceae71188229ef46f1d5e8aa5ee1`。WOR-24 保存的粗細面中心化候補 `e786abc4` と WOR-25 丸め誤差ゲート候補 `2e2017d2` を opt-in で取り込んだ。
- XD 入力 SHA-256: `95d9fca3409705006947ea26da2f9abd3e36a4907a1c2335cfde805502f8cbb6`（`wor92_core_symmetry_cycle_output_gravity_20261107.in`）。Tomida EOS `742a3456923d4c7056c3705accbcc793af3e0c79408a2f597a0cd9271b13ba43`、Semenov opacity `6c42445452bca6aeca1253f31ab049b2ff107dea3601f946b4c61c93db022f94`。
- XD binary SHA-256: 改良前 `ee22f78590c003827911a95192c116c6115649bb38252f5114b66b537200d7df`、中心化試験 `8250bdde16e2ac2ad23e734f95144b33012e865cc3afd8ff7350b573cb69f6f8`、丸めゲート試験 `aa61a2e8bbc4feb7e2c0ce4f3f6e6823284f98b63c45d8c7b25285f7ad381376`、最終修正 `53aeab998d431f0722d13c33927d52987fd9a87ffdfc7a834d38c8bfec3b5e72`。最終修正 binary の変更ファイルは commit と SHA-256 が一致する。
- 全 XD 実験は64 MPI rank、1 node、P-debug、CFL 0.3、FGMRES restart 80、最大600反復、通常 MG3、Newton しきい値 `1e-5`。各実験の restart、環境変数、source/input/table SHA は XD の `debug/job_ID/provenance.txt` と同梱 `.sbatch` に記録した。`runs.csv` は Slurm 状態、反復数、最大 true residual、経過時間、step MaxRSS、履歴上の質量変化を示す。元のスクリプトに残る `SOURCE_BASE_COMMIT=873a...` は**統合前の基点**であり、最終 source commit は上記 `f9dc...` である。

## 保存的粗細面と Newton

cycle 46→47 の同一 restart A/B（337947 OFF、337948 ON、960 leaf blocks）は両方 exit 0、Newton 2 回。面積重み付き粗細面 flux 不一致は x1/x2/x3 で OFF の `6.070e-7 / 5.913e-7 / 6.574e-7` から ON の `2.027e-17 / 2.083e-17 / 2.286e-17` に低下した。暗黙 FLD の全域符号付きエネルギー収支誤差は `3.65e-10` と `3.93e-10`。保存 restart の最大相対差は hydro 全エネルギー `5.89e-10`、放射エネルギー `2.39e-9`、密度・運動量は同一。数値全体のエネルギー保存は ambient 温度・放射温度を毎ステップ固定する源項があるため、ここでは硬い合否条件にしていない。詳細は `centered_c46_face_summary.json` と `centered_c46_state_summary.json`。

中心化 ON の cycle 47→95（337950）は exit 0、51 個の Newton status はすべて収束。cycle 95→150 の物理状態は中心化 ON の修正前 337962 と修正後 337974 でほぼ同じで、修正後は 55 step、61 個の Newton status、110 個の FGMRES solve、最大 Newton `final_max=3.223e-6`、最大 true linear 相対残差 `1.983e-8`、opacity domain 問題なし、exit 0。元の 337962 は物理 step 終了後の destructor で FAIL した。

WOR-25 丸めゲートは `nrfld/fgmres_roundoff_cap=2e-7` の opt-in を MG2 cycle 46→47（337959）で確認。2 回目の線形 solve は真の相対残差 `1.7194e-7`、要求補正からの予測 `8.2045e-9`、実現補正との差の増幅 `1.7045e-7`、closure `2.0295e-8` で受理され、Newton 2 回で `final_max=8.367e-6`。長時間計算では既定 OFF とし、MG3 と bounded inexact cap `1e-7` を使用する。

## 終了時 SIGSEGV の原因と修正

旧 binary では cycle 95→150（337949）、cycle 95→120（337961）、中心化 ON cycle 95→150（337962）、最大 AMR level 10 cycle 150→160（337963）が、全物理 step と restart 出力後に exit 255。対照として restart 150 の読み込み即終了（337956）、cycle 145→150（337958）、cycle 120→150（337960）は exit 0。したがって restart の読み込み破損ではない。

隔離診断ビルドの 337968 で SIGSEGV stack を得た。`free` → `MultigridDriver::~MultigridDriver()` → `MGGravityDriver::~MGGravityDriver()` → `Mesh::~Mesh()`。命令位置 `0x69edd3` は AMR octet 配列の**空の matrix AthenaArray** の解放。`AthenaArray` の copy/move constructor は空入力時に出力の `pdata_` と `state_` を未初期化のままにしていた。AMR octet vector が伸びると未初期化ポインタを解放し得る。両 constructor の null/empty 初期化のみを本線に修正した。

`test_empty_athena_array.cpp` は空配列の配置 new、copy/move、octet 相当 vector 拡張を検査する。旧ヘッダでは assertion 失敗（exit 134）、修正後は AddressSanitizer と UndefinedBehaviorSanitizer で exit 0。XD 修正済み binary の cycle 95→120（337972）、最大 AMR level 10 の cycle 150→160（337973、168 blocks 増加）、中心化 ON cycle 95→150（337974）はすべて exit 0。後二者の比較では修正前後の Newton/FGMRES 回数は同一で、物理演算を変えずに終了処理を直した。

## 質量と長時間計算

cycle 95→150 の `HistoryMass`（primitive density の体積積分）は約 `+9.24e-5` 相対増加。FLD 粗細面 flux の一致とは別の指標で、原因は未確定。P-large の progress に記録し、first core の結論にはこの drift と AMR 解像度を併記する。energy drift には ambient reset の寄与が含まれる。

P-large-a 337975 を修正済み binary・cycle 160 checkpoint（337973）から投入した。最大 AMR level 10、中心化 ON、MG3、10-cycle 毎の restart と真の線形収束のソルバ内部検査を使用する。`rho_max >= 1e-10 g cm^-3` を first-core *候補* 到達点とし、Jeans cells <16、opacity domain逸脱、tlim、wall limitで停止する。密度だけでは first core の形成を断定せず、到達後に中心構造・温度・速度を評価する。energy 保存の硬い停止条件は入れていない。337974 成功により依存は解除されたが、提出時点で P-large-a ノード待ち。`wor92_plarge_firstcore_l10_continuation.sbatch` が実行スクリプト。

P-debug の追加先行計算 337976 は cycle 160→200、最大 AMR level 10、中心化 ON で exit 0。224 block 増加して合計 1856、最大 level 7、`rho_max=3.77335e-16 g cm^-3`、最小 Jeans cells `32.1281`、最大 Tgas `14.0464 K`。42 個の Newton status はすべて収束し、80 個の FGMRES solve の最大 true 相対残差は `1.9731e-8`、最大 Newton `final_max=7.513e-7`。opacity domain 問題なし。履歴上の質量増分はこの区間で約 `3.96e-5`。まだ first core 形成密度から遠いので、長時間計算が必要である。

XD 詳細ログ archive: `/work/mizutaniks/output/fld/collapse/wor92_xd_20261107/scripts/wor92_firstcore_pdebug_evidence_20261109.tar.gz`、SHA-256 `1ffc372c3b894b4c2d8f36da856cc75581c836192c662df28218bb8afcef834d`。ローカルコピー: `/home/kosuke/simulation/test/rad_nr/wor92_xd_20261107/firstcore_integration_evidence/`。画像や HTML は新たに作らず、ログと JSON・CSV で再現可能な記録を残した。

## M キューでのメモリ・速度比較（2026-11-09 11:55 JST）

`M-test-a` は最大30分の試験キューであり、`M-debug` と同じ M 系ノードを使用する。修正済み binary `53aeab99...`、同じ入力 `95d9fca3...`、同じ cycle 280 checkpoint（338101）から cycle 320 まで、CFL 0.3・AMR 最大 level 10・MG3 で比較した。各ジョブの完全な環境・ファイル SHA は XD の `debug/job_ID/provenance.txt` または `test/job_ID/provenance.txt` に保存している。試験スクリプトはこのディレクトリの `wor92_mdebug_*` と `wor92_mtest_5node_*`。スクリプト内の `source_restart_gravity_trial` SHA は旧ソースの補助記録であり、**実行 binary は全試験とも上記の修正済み binary** である。

| job | キュー、構成 | 280→320 wall | step MaxRSS | Newton / FGMRES | 最大 Newton final_max / 線形真残差 | 結果 |
|---|---|---:|---:|---:|---:|---|
| 338114 | M-debug、1 node、64 rank | 7:55 | 32,126,148 K | 44 / 80 solve、1168 iteration | 1.517e-7 / 1.920e-8 | PASS |
| 338127 | M-debug、1 node、128 rank | 4:57 | 46,339,468 K | 44 / 80、1168 | 1.502e-7 / 1.920e-8 | PASS |
| 338128 | M-debug、1 node、192 rank | 3:42 | 51,847,844 K | 44 / 80、1168 | 1.526e-7 / 1.920e-8 | PASS |
| 338122 | M-test-a、5 node、各64 rank | 2:05 | 17,372,448 K | 44 / 80、1168 | 1.568e-7 / 1.920e-8 | PASS |

全試験の最終履歴は ρmax=`1.22169e-14`、Tgas,max=`14.453` K、gas mass=`2.27373e33` g、Erad=`1.68443e42` erg、最大 level 9 で一致した（Jeans min は 192 rank の `35.3804` を除き `35.3806`）。cycle 280→320 の gas mass 相対変化は各 `+1.3194e-5`、opacity の out-of-domain/clamp/negative/zero/nonfinite は全て0。5 node の checkpoint byte hash は1 node と異なるが、rank 分割に伴う浮動小数点の演算順序差があり、上記の履歴と真残差で物理解の一致を確認した。M-test-a 5 node は1 node 64 rank 比3.8倍速い。

長時間計算は先に M-large-a 1 node job **338117** を cycle 320 から開始し、cycle 330 の成功（ρmax=`1.74513e-14`、Jeans min=`34.3736`、opacity正常）後に P-large-a 予備 **337975** を取り消した。M-large-a 5 node job **338133** は job 338117 の cycle 330 checkpoint を hash 照合してコピーし、cycle 340、350 を正常完了。cycle 350 の検証済み checkpoint が1 node の cycle 340 を上回った時点で job 338117 を取り消した。job 338133 は cycle 530 で ρmax=`1.81906e-11`、Tgas,max=`208.132` K、Jeans min=`15.0764` となり、設計どおり `STOP_UNDERRESOLVED`（exit 0）。first-core 密度候補には未到達。スクリプトは `wor92_mlarge_5node_firstcore_l10_continuation.sbatch`、実行記録は XD の `long/job_338133/`。

## AMR 解像度を増やした継続試験

旧 `mesh/numlevel=10` の cycle 450→530 で Jeans min は27.94→15.08へ下がった。解像度不足を放置して密度到達だけを主張しないため、同一 binary・CFL 0.3・MG3・保存的面中心化のまま `mesh/numlevel=11` を試した。

| job | 入力 checkpoint | 構成 | wall / step MaxRSS | 最終 Jeans min / ρmax | Newton / FGMRES / 最大真残差 | 結果 |
|---|---|---|---|---|---|---|
| 338135 | 338133 の cycle 450 | M-debug 1 node、64 rank、→460 | 2:06 / 31,464,408 K | 32.6842 / `2.17114e-12` | 12 / 20 solve、191 iter / `1.838e-8` | PASS、opacity異常0 |
| 338136 | 338133 の cycle 480 | M-test-a 5 node、各64 rank、→490 | 0:52 / 18,119,644 K | 33.3963 / `5.85437e-12` | 12 / 20 solve、172 iter / `1.871e-8` | PASS、opacity異常0 |

旧解像度での cycle 460 / 490 の Jeans min は25.4584 / 20.0715であり、追加レベルにより32.6842 / 33.3963へ回復した。物理状態は解像度変更により変わるので、旧解像度と bitwise 一致を要求しない。AMR11 の長時間 job **338144** は、旧解像度で Jeans min18.5565を保った cycle 500 restartから M-test-a 5 node で開始し、最初の cycle 510 を正常終了した。旧 checkpoint 起点の M-large-a 待機 job **338139** は、338144の成功を確認して重複回避のため取消。さらに解像度が必要な場合は、`select_safe_restart.py` が直前の成功済み segmentでJeans min≥20の restartを選ぶ依存 job **338147**（`mesh/numlevel=12`）を登録した。338144が先に first-core 密度候補へ到達した場合、338147は計算せず終了する。

`analyze_core_profile.py` は Athena++ leaf MeshBlock のセル体積と Tomida EOS の P→e→T 双一次補間から密度・温度・流入速度の半径プロファイルを作る。cycle 390 の HDF5 ではρmax=`1.63233578e-13`、最大温度=`19.6991339` K、gas mass=`2.27375538e33` gとなり、`.hst` の最大温度`19.6991` Kとgas mass`2.27376e33` gを再現した。EOS table外セルは0。密度しきい値到達後はこの解析で中心構造とshock候補を評価する。

## Newton line search の根本原因と修正（cycle 588）

AMR11 の M-test-a 5 node job **338144** は cycle 580→590 の cycle 588 で Newton 停滞（job `FAILED`、5:11、最大 step MaxRSS 18,263,396 K）。直前 iterate の L2=`7.922663e-10`、max=`1.007729e-5`。FGMRES の真の相対残差は約`1.8e-8`で、最初の**全幅 Newton 補正**は L2=`7.926707e-10`（直前比 `1.000510`）、max=`4.250063e-9`だった。L2 は0.051%増えたが、L2とmaxは双方とも停止基準`1e-5`を大幅に満たす。旧 line search は L2 の単調減少だけを試行受理条件としたため、この収束済み解を拒否した。縮小補正では最大残差が基準をわずかに超え続け、最終 max=`1.007729e-5`で失敗した。局所ガスソルブ残差は `~6e-16`、失敗セルの signed gas defect は `-4.39e-11` で、局所ガスソルブの発散ではない。依存の AMR12 job **338147** は前段の solver failure を検出し、計算を開始せず停止した。

最小修正は `Newton_Raphson_driver.cpp` の line search で、有限かつ L2・max の**両方**が要求された停止基準内の trial を受理すること。それ以外の trial には従来の「丸め誤差を超えるL2減少」を維持する。固定反復モードの受理規則は変更しない。XD 隔離ソース `source_newton_converged_trial` でビルドした binary SHA-256 は `ac13cce7ac7d7281c12b863239d3f0c42e69d5b6396511145c410513cf496803`、変更ファイル SHA-256 は `b1b1871b7d0706cd119be66acd0b76d3393d4b66598e6f8f1d5a3f80d0251e47`。同じ cycle 580 checkpoint を用いた M-test-a 5 node job **338150** は cycle 588 の該当 trial を `converged_trial=1`、`accepted=1` とし、cycle 590 まで exit 0（0:48、step MaxRSS 14,476,384 K）。Newton 11件、FGMRES 22 solve・191 iteration、最大真の線形相対残差 `1.84275e-8`、最大 Newton final_max `9.69769e-6`、opacity異常0、最終 ρmax=`2.61888e-11`、Tgas,max=`243.288` K、Jeans min=`27.0286`。修正は単に最大残差の許容幅を広げるものではない。

修正 binary から AMR 上限 `mesh/numlevel=12` に上げた M-test-a 5 node job **338156** は cycle 590→600 の最初の segment を exit 0 で通過。Jeans min=`35.9154`、ρmax=`2.84516e-11`、最大 step MaxRSS `17,936,908 K`、opacity異常0。first-core 密度候補の到達と構造判定は継続中。投入スクリプトは `wor92_mtest_5node_newton_converged_l12_continuation.sbatch`。

同ジョブの cycle 670 3D HDF5（code time `1937.3640`、約29.042 kyr、3536 leaf MeshBlocks、HDF5 `MaxLevel=11`）を `analyze_core_profile.py` と `analyze_axis_profile.py` で解析した。ρmax=`4.09842e-11`、中心温度=`295.085` K、中心放射温度=`295.093` K（差0.008 K）、gas mass=`2.27377591e33` g、EOS table 外セル0。半径5.95 au以内のgas massは約`0.0145 M_sun`。正のx軸の5.676–6.226 auで隣接セル密度比は2.886、外側の流入速度1.198 km/sに対し内側0.350 km/s。半径5–10 auの8象限平均密度の相対範囲は0.319%、正負x軸の半径3–15 auの最大密度差は0.293%。この時点でほぼ球対称の降着衝撃波**候補**が現れている。ρmaxはまだ所定の`1e-10`に達していないため、最終判定は後続出力で行う。数値は `c670_profile.csv/json` と `c670_axis.csv/json` に保存した。比較対象の理想気体・一定opacity計算では密度しきい値近傍のshockが6.23–6.96 auだったが、EOSとopacityが異なるため半径の一致だけで表計算を検証済みとはしない。

cycle 780 の3D HDF5（code time `1938.8917`、約29.065 kyr、3200 leaf MeshBlocks）ではρmax=`5.07516e-11`、中心Tgas=`322.580` K、Trad=`322.590` K、中心流速=`0.0041` km/s。正のx軸6.958–7.690 auの密度比は2.881、内側/外側の流入速度は0.799/1.592 km/s。半径7.01 au内のgas massは`0.0203 M_sun`、5–10 auの8象限平均密度の相対範囲は0.266%。cycle670からshock候補が持続して外へ移り、中心がほぼ静止していることは**物理的な第一コア形成の強い証拠**である。EOS table外セル0。密度`1e-10`の比較基準には未到達なので継続計算する。`c780_profile.csv/json` と `c780_axis.csv/json` に保存した。

## 第一コア密度到達・最終解析（2026-11-09 12:51 JST）

M-test-a 5 node × 64 MPI rank の修正 binary `ac13cce7...` による job **338156** は cycle590→930 を24:42で完了し、計画どおり wall budget 停止。検証済み cycle930 restartを引き継いだ job **338171** は cycle930→1010を6:36で完了し、`rho_max >= 1e-10 g cm^-3` で `FIRST_CORE_DENSITY_CANDIDATE` として正常停止。予備 job **338183** は先行到達を検知して計算せず終了した。全ジョブは Slurm `COMPLETED 0:0`。使用 binary、入力、EOS/opacity表、checkpoint の SHA-256 は各 XD `long/job_ID/provenance.txt` に記録。最終 restart は `long/job_338171/output/cfl030_flux_precision16_local.final.rst`（cycle1010、約697 MB）。最終3D HDF5 `prim.00237.athdf` の SHA-256 は `bdd4c72640d16f1872725904c3a66bcd74e56577016f4593e114a3e91e25974f`。元データは XD の `/work/mizutaniks/output/fld/collapse/wor92_xd_20261107/`、解析用 HDF5 のローカルコピーは `/home/kosuke/simulation/test/rad_nr/wor92_xd_20261107/firstcore_integration_evidence/profiles/c1010.athdf`。

最後の420 cycleについて Newton status **474/474 converged**、FGMRES **1202 solve / 12063 iteration**。独立真線形相対残差の最大は`1.99813e-8`、収束した Newton の最終最大残差は高々`9.79968e-6`（要求`1e-5`）。opacity 範囲逸脱・clamp・負値・非有限診断は全42 segmentで0。最終 segment の Jeans 解像度は**36.4486 cells**、ジョブ338156全体の最小は31.9891、338171の最小は32.0018で、停止基準16を十分上回る。最大 Slurm step MaxRSS は338156で`18,336,664 K`、338171で`19,488,060 K`、両ジョブともノード当たり110 Gのメモリ指定内で終了した。詳細集計は `job_338156_summary.json`、`job_338171_summary.json`、`runs.csv`。

cycle1010 の独立3D HDF5解析（3704 leaf blocks、8³ cells/block、約189万 leaf cells、code time1941.8418、**29.1095 kyr**）では最大密度`1.00818888e-10 g cm^-3`、そのセルのガス温度`431.253 K`、放射温度`431.258 K`。EOS表外セルは0。正のx軸の7.507–7.874 auで隣接セル密度比`4.000`、内外の流入速度`0.434/1.665 km s^-1`。密度最大セルの半径方向流速は`-0.0134 km s^-1`。7.013 au内の質量は`0.02845 M_sun`、8.272 au内は`0.03150 M_sun`。半径5–10 auの8象限平均密度の相対範囲は`0.251%`、正負x軸の3–15 au最大密度差は`0.618%`。cycle670/780から持続し外へ進む降着衝撃波、低速の中心、ほぼ球対称の構造とJeans解像度がそろったので、**第一コア形成および従来の目標密度到達を確認**した。図 `first_core_profiles.png` は同じ3D出力の体積重み半径プロファイルを重ねたもの。再生成は `python3 plot_first_core_profiles.py first_core_profiles.png`。

3D leaf-cell積分のgas massはcycle670で`2.2737759073e33 g`、780で`2.2737782486e33 g`、1010で`2.2737825688e33 g`。cycle670→1010の相対増分は**`2.930e-6`**、旧AMR経路を含むcycle390→1010では`1.196e-5`。完全な質量保存とは報告しない。FLDの粗細面 flux 一致はcycle46 A/Bで直接検証したが、最終高密度状態の面ごとの監査は未実施。全エネルギーはambient温度固定源項があるため厳密保存量として扱わない。

理想気体・一定opacity基準の目標密度時刻約29.1585 kyr、中心温度約694.6 K、shock 6.23–6.96 auと比べると、今回のtable EOS/Semenovでは約29.1095 kyr、431 K、7.5–7.9 au。物理表・AMR解像度・計算経路が異なるため、差を単独でEOS/opacityに帰属しない。SemenovのPlanck extinctionとOptab absorptionのモデル選択、最終高密度状態での直接粗細面監査、質量増分の原因は追加検証項目として残す。

### 人に伝える計算量の目安

**「XD2000で第一コアの目標密度まで、採用したcycle47保存点から約5ノード時間。最後の高密度420ステップはMノード5台で約31分」**。この5.07ノード時間は採用したcheckpoint列を作った各ジョブの実測wall×ノード数の合計で、約324 MPI rank時間（各ノード64 rank）。途中のjob338133（cycle530まで進行後、cycle500から高解像度へ分岐）とjob338144（cycle588失敗後、cycle580から修正再開）の実使用時間も含む。最後のjob338156+338171だけなら31:18×5=**2.608ノード時間**、約167 MPI rank時間。node-hourはキュー待ちを含まず、cycle0→46の初期保存点作成とA/B検証・失敗原因調査・重複試験は含まない。したがって初期状態からの全研究コストの数字としては使わない。

## 第一コア後続監査（2026-11-09）

すべての図を埋め込んだ [HTMLレポート](report.html) に、従来基準との正のx軸比較、EOS独立監査、Planck吸収の感度、c930/c1000の粗細面監査を追加した。HTMLは単体閲覧可能であり、生成コードは `make_report.py`。元PNG、JSON、スクリプトも同じディレクトリに置いた。

### 高密度でのFLD面収支

同じ最終binary SHA `ac13cce7...`、元のEOS/opacity表、保存的面中心化を用い、M-debug 1 node・64 rankでc930→931（job **338226**、Slurm COMPLETED、39秒、step MaxRSS `35,638,392 K`）とc1000→1001（job **338228**、物理step完了、37秒、step MaxRSS `36,870,452 K`）を実行。各rankの `[NR_INTERFACE_LEDGER]` と `[NR_ENERGY_LEDGER]` を `audit_high_density_faces.py` で完全数監査した。

| 開始cycle | leaf blocks | 各方向の粗細面group | x1/x2/x3 面積重み相対不一致 | 暗黙FLDの全域符号付き収支比 | 判定 |
|---:|---:|---:|---|---:|---|
| 930 | 3536 | 280 | `6.030e-18 / 6.068e-18 / 6.207e-18` | `3.992e-14` | PASS |
| 1000 | 3704 | 296 | `5.976e-18 / 5.259e-18 / 6.071e-18` | `6.213e-14` | 物理・監査PASS、Slurm statusはスクリプトの終了比較ミス |

job338228はfinal restartのcycleが正しく1001で、Newtonは収束、opacity診断すべて0、64 rankのledgerと3704 blockが揃った。ただしsbatchの最後だけ `1001` を古い `931` と比較し、`FAIL_WRONG_END_CYCLE` を書いてexit1した。この失敗は物理計算の失敗として扱わない。実行時の誤判定はXDの `debug/job_338228/status.txt` とarchiveに保存し、repoのスクリプトは判定値を修正済み。rankログのローカルarchiveは `firstcore_integration_evidence/job_338226_face_ledger.tar.gz`（SHA `ee356b49ab0c27277d265214a49dffb7db1dd16420ff4854c4ea761332712f75`）と `job_338228_face_ledger.tar.gz`（SHA `25adab5591c9f41806eca5b1a9c86634ea13b1d7dcf748b8fe281b4a1711e9f4`）。集計JSONは `c930_face_audit.json`、`c1000_face_audit.json`。

### 質量の高精度historyとPlanck吸収感度

M-debug 64 rank の同一c930 restartからc940まで、元表の参照（job **338222**、Slurm COMPLETED、3:15、step MaxRSS `35,742,588 K`）と診断用吸収表（job **338225**、Slurm COMPLETED、3:10、step MaxRSS `35,720,036 K`）を同じbinaryで実行。両方で `output1/data_format=%22.15e` とし、10/10 Newton収束、30 FGMRES solve・292反復、最大true relative residualは各 `1.970175e-8` / `1.970198e-8`、opacity異常0。参照のc931→c940 gas mass は `2.273781246448054e33`→`2.273781403761505e33 g`、相対 `+6.91858e-8`。HDF5のc930/c940は3536 blocksで `Levels` と `LogicalLocations` が完全一致し、実行ログも「0 created, 0 destroyed」を記録。mesh再配置が**一度もない**区間でも増分が続く。密度floorの閾値より最小密度は十分高い。全期間c670→1010の質量増分`+2.93e-6`をAMR再配置のみでは説明できない。放射FLDの面収支は上の直接試験で機械精度なので、hydro側の質量収支が次の切り分け対象。全エネルギーはambient固定源項のため厳密保存量ではない。

Semenovの公開 `opacity.f` はPlanck/Rosseland meanを消散と表記する。著者の単色NRM複合凝集体データ `nrm.zip` の列は、合計=吸収+散乱およびPlanck積分と公開mean表の照合から、輸送消散・全消散・吸収・散乱と推定できる。この単色比をPlanck表へ適用し、著者コードと同形の蒸発遷移を持たせた**診断用**表を `build_semenov_absorption_trial.py` で作成。元表SHA `6c424454...`、単色archive SHA `a7d2358280c5f4598c4bb0b931e036af2c481f572c75c4b52b87436b9b32a945`、診断表SHA `347baa6e7b0d4303eca1abf762b2fde5042137e42739ef65961eeb68b88dad05`。ρ=1e-10 g/cm³で診断/元のκPは14/350/431 Kで`0.999/0.691/0.523`、κRは同一。共通域431 KではκPが元Semenov `5.909`、診断表`3.093`、Optab`3.057 cm²/g`で、定義差がPlanckの隔たりの大部分を説明する。ただしOptabの低温Rosselandはダストモデルが異なり、同一の物理表として扱わない。高温ガス域の吸収定義もこの表では未検証。

同一c930 restartからの10 cycle A/Bではc940のρmaxは`7.281453102655851e-11`→`7.281453131229435e-11 g/cm³`（相対`+3.924e-9`）、最大Tgasは`375.401214507`→`375.401215138 K`（相対`+1.681e-9`）、Erad総量は相対`-3.257e-8`、gas mass差は`1.0e-14`。短い高密度区間では解にほぼ影響しないが、全collapse経路の感度は別に確認する。数値・反復は `opacity_replay_comparison.json`、図は `opacity_replay_comparison.png`、ログarchiveは `wor92_c930_opacity_replays_338222_338225.tar.gz`（SHA `d973a3a438cde6f046073de5395ebff8c09b049a77bafc34ee519e07a59d507d`）。

さらにHDF5の**保存変数** `cons/dens` を用いて、同じc930 restartを進めないsnapshot job **338236** とc930→940の10-cycle job **338232** をM-debug 64 rankで比較した。両方Slurm COMPLETED、wallは各0:16/3:10、step MaxRSSは`31,732,612 K`/`35,757,508 K`。出力に記録されたcycleは930/940、AMRの `Levels`・`LogicalLocations` は完全一致。float32保存値をfloat64に変換してから面座標由来の体積で積分したgas massは `2.273781229032851e33`→`2.273781403703997e33 g`、相対`+7.681968e-8`。従って増分はprimitive historyだけの表示差ではなく保存変数にも存在する。HDF量子化の影響は各massでhistoryとの差約3e-11相対以下で、この増分より十分小さい。局所原因はまだ特定していない。

#### Hydro面fluxの診断

実行binaryの**隔離コピー**にのみ `hydro_mass_flux_ledger.patch` を当てた診断binary SHA `61e7dda7651d5419e0cb5fc51bb8db53e209cf6d14a134d6402f19f4a18ac480` をM-debugで構築（job **338241**、元binaryは変更なし）。job **338242**（M-debug 1 node・64 rank、c930→931、COMPLETED 0:36、step MaxRSS `35,658,172 K`）では、RK2の2段階・3536 MeshBlockそれぞれのhydro `IDN` flux発散をrank別ログに記録。元binaryのc931高精度historyと診断binaryのc931はgas massを15桁表示で完全再現。実行前c930の高精度historyはjob338236から取得。

高精度historyの質量増分は `1.7553286e25 g`（相対`7.71987e-9`）。診断の面flux合計はRK2第1段階（重み0.01232227）が `1.08616e21 g`、最終段階（重み0.00616113）が `1.7552743e25 g`、合計 `1.75538293e25 g`（history比`1.00003095`）。従って**増分のほぼ全量はhydro最終段階のflux発散**で説明され、FLD・history表示・密度floorが主因という仮説は支持されない。診断archive `job_338242_hydro_mass_ledger.tar.gz` のSHAは `a72bbfe48e48381c8852aaf496fcf1010f99d718fb1fb59039f2b7e61d36327e`。結果 `c930_hydro_mass_flux.json`、解析 `audit_hydro_mass_flux.py`、高精度history `c930_cons_snapshot.hst` と `c931_hydro_flux_trial.hst`。

さらに隔離コピーで6面ごとの質量fluxを出す `hydro_mass_face_ledger.patch`（診断binary SHA `439f22a970b468f5c20fc65a04b4e57d3dea970f300500516193f43809361cac`）を構築し、job **338248**（M-debug 1 node・64 rank、c930→931、COMPLETED 0:36、step MaxRSS `35,658,172 K`）で完全数監査。最終RK2段階の280組/方向の**hydro粗細面**fluxは重み付き相対不一致 x1/x2/x3=`4.286e-18 / 4.305e-18 / 4.311e-18`、net≈0。同一level面2784組/方向は`1.451e-5 / 1.412e-5 / 1.473e-5`で、net質量変化は各`1338 / 1302 / 1359` code単位。境界104組/方向のnetは各`413 / 425 / 406` code単位。合計`5242.866` code単位で、block内部flux発散和との差は`2.0e-10` code単位。第1RK2段階の同一level面の重み付き不一致は各約`5.85e-10`で、net質量差は最終段階より4桁小さい。同一level面のnetの大半は外側level3の面。したがって**この高密度1ステップではhydroの同一level面と周期境界**に主因がある。具体的にどのghost状態・共有処理が不一致を作るかは未確定で、推測でproduction修正はしない。archive `job_338248_hydro_mass_faces.tar.gz` SHA `6e33fb5afb56a6a6f0c72a332ea5bdb390640eb3eeb90dabee0021d4b1e82486`、結果 `c930_hydro_mass_faces.json`、解析 `audit_hydro_mass_faces.py`。ジョブ338245は転送スクリプトの参照patch名ミスで先の簡易ledgerを再ビルドし、依存338246は開始直後に重複と判明して取消。修正後の構築はjob338247、監査が338248。

第一コア到達c1010の3D HDF5では、ρ>1e-14 g/cm³かつ中心5 au以内の81,608セルの|Tgas−Trad|は99%点`0.01528 K`、最大`0.01685 K`。5–12 au衝撃波領域155,136セルでは99%点`0.00847 K`、最大`0.01082 K`。`audit_thermal_coupling.py` と `c1010_thermal_coupling.json` に元HDF/EOS SHAと領域条件を保存した。この近LTE状態が、c930→940におけるPlanck吸収表差の小ささを説明する可能性があるが、全経路への外挿はしない。

第一コアのOptab共通域ρ=1e-10 g/cm³、316.3–431.3 Kを `compare_firstcore_opacity_means.py` で再比較した。431 KのκPは元Semenov `5.909`、診断用吸収 `3.093`、Optab `3.057 cm²/g`。診断用/Optabは`1.012`。一方κRはSemenov/診断用`5.006`、Optab`4.145e-8 cm²/g`で桁違いであり、Planckの一点一致を同一物理モデルの証明として扱わない。Optabの元表・sidecar SHAと定義を含め `firstcore_opacity_means.json/png` に記録した。

### EOSと基準計算

表SHA `742a3456...` をソルバから独立したPythonの双一次補間で読み、ρ=1e-16〜1e-10 g/cm³、T=14〜431.3 Kの480点を調べた。H₂/He理想混合気体の解析圧力 `P=ρk_B T(X/2+Y/4)/m_H`、X=0.70、Y=0.28に対する最大相対差は`1.743e-12`。P→e逆変換の最大相対差は`5.296e-5`、差分で求めた`c_v`は`5.197e7`〜`8.100e7 erg g^-1 K^-1`で全点正。温度のlog微分`d ln T/d ln e`は472点で双一次セル内の解析傾きと独立中心差分の最大絶対差`1.075e-9`。これは第一コア温度域の単位・補間・熱的安定性の検証であり、H₂解離や第二崩壊への外挿はしない。結果 `firstcore_eos_audit.json/png`、コード `audit_firstcore_eos.py`。

旧理想気体・一定opacity計算job335063の元x軸CSVと今回c1010のx軸CSVを、同程度の最大密度で `compare_reference_axis.py` により重ねた。`reference_axis_comparison.png/json`。旧計算の最大軸温度712.595 K、今回431.255 K。旧shockの6.226〜6.958 au、今回7.507〜7.874 auは持続する降着衝撃波に整合するが、EOS、opacity、解像度、実行経路が異なるため単一原因へ帰属しない。旧shock付近のセル幅0.732 au、今回0.366 au。

### 高密度の追加対照と全経路感度

#### Hydro質量増分の追加対照（2026-11-09 14:18 JST）

同一c930 restart・元のopacity表・同一最終binary SHA `ac13cce7...` で時間積分のみ `rk2`→`vl2` に変更した M-debug 64 rank job **338249** はc940までSlurm COMPLETED 0:0（3:09）。c930の高精度質量 `2.273781228894768e33 g` からc940で `2.273781396922404e33 g`、相対`+7.389789e-8`。元のRK2・2次 job338222の同じ区間は`+7.690570e-8`であり、VL2への変更だけでは質量増分は残る。元表・同一restart・同一CFLの短い対照であり、VL2を本線には採用しない。job338249のinput、binary、restart、表SHAはXD `debug/job_338249/provenance.txt`、高精度HSTはローカル外部evidenceの `profiles/c930_c940_vl2.hst`（SHA `43efae610093555368cad1181ec41a48d12396cd7e5ae92754af3c2264f4c64b`）。

同じc930 restartからhydro面診断binary SHA `439f22a9...` で再構成だけ `time/xorder=1` に変えたM-debug 64 rank job **338258** はc931までSlurm COMPLETED 0:0（0:35、step MaxRSS `35,665,816 K`）。c931の高精度gas massは `2.273781228668356e33 g`、c930比`-9.957510e-11`で、元のRK2・2次の1ステップ`+7.719866e-9`より約78倍小さい。ただし符号が反転して完全保存ではない。RK2第1段階の同一level面不一致は機械精度、最終段階の外側level3のnetは各方向`4.1e-5 code`まで減少する一方、内側level12のnetは各約`-16.9 code`残る。全域のhydro面flux発散は最終段階`-67.627 code`であり、元の`+5242.866 code`から大幅に減少した。この結果は2次の再構成境界状態が外側の不一致に大きく寄与することを示すが、ghost通信そのものの不具合の証明ではない。1次化は空間精度を落とすため採用せず、コード修正は面状態の比較後に限定する。全64 rankのログarchiveは `job_338258_hydro_mass_faces_xorder1.tar.gz`、SHA `e9bb3165e21a276dbf9a035ed2cabf1cc2766f1e0c15d24f0585f4cfcb305687`。集計 `c930_hydro_mass_faces_xorder1.json`、対照表 `hydro_mass_controls.json`。

追加の`time/xorder=2c`（特性変数での2次再構成）job **338262** は実行開始時に`General EOS does not support characteristic reconstruction`と明示的に停止（Slurm FAILED、0:13）。この構成はソルバで禁止されているため物理試験のFAILとは数えず、解決策として採用しない。隔離診断binary・元表のみを使い、元の計算には変更がない。

Mayer et al. (2025) Section 5 の[原著](https://arxiv.org/html/2510.12620#S5)では非回転・非磁化、1 M☉、14 Kの球状雲から`ρmax>1e-10 g/cm³`の第一コアへ約29.09 kyr、衝撃波から得た半径6–7 auと報告。今回の29.1095 kyr・7.51–7.87 auは同規模だが、論文Section 5はAREPO移動メッシュ・固定EOS/opacityであり、table EOS/opacityとAthena++ AMRを使う本計算との一致を要求しない。

元表と診断用吸収表の物理経路感度をc590から比較した M-test-a 5 node job **338230** はc590→930をSlurm COMPLETED 0:0（24:49、最大step MaxRSS `18,310,744 K`）。継続の **338233** はc930→1010をCOMPLETED 0:0（6:31、最大step MaxRSS `19,544,476 K`）で終え、ρmax=`1.00819e-10 g/cm³`、Tgas,max=`431.263 K`、Jeans min=`36.4486`で所定の目標に正常停止。全42 segmentのNewtonは**474/474収束**、FGMRESは**1202 solve/12063反復**で、元表の経路と反復数まで同一。最大true linear relative residualは`1.998173e-8`、最大収束Newton final_maxは`9.797649e-6`、opacity out-of-domain/clamp/負/非有限はすべて0。AMR配置はc930（3536 block）、c1010（3704 block）で元表の経路と完全一致。binary・元EOS・checkpoint・診断表SHAと環境変数は各jobの`provenance.txt`に記録。ジョブログarchive SHAはjob338230 `054472991f65cbccbe412fbd341d219fd066e22b72085ad564009b0f2aab94da`、job338233 `606cf02887506a8ccb43460729a620766dfaed259bcd3c2356e9bcfed64fbdc8`。実行表そのものは `semenov_planck_absorption_trial.h5`（174,264 bytes、SHA `347baa6e...`）として同梱した。

c1010の独立3D解析では元表→診断吸収表で最大密度`1.00818888e-10`→`1.00818968e-10 g/cm³`（相対`+7.935e-7`）、最大温度`431.263254`→`431.263424 K`（相対`+3.941e-7`）、gas mass相対差`+5.633e-12`。正のx軸shock位置は両方7.507–7.874 auで一致。ρ>1e-14 g/cm³の486,172 leaf cellsにおける密度相対差99%点は`1.299e-6`、最大`1.086e-5`。前者・後者にはfloat32 HDF5保存量子化も含む。比較は `absorption_branch_comparison.json/png` と `compare_absorption_branch.py`。これにより**第一コア形成まで**の計算結果は、検出したPlanck消散/吸収定義差に対して頑健と確認した。高温ガス域の診断表は未検証で、第二崩壊へ外挿しない。

### Mayer et al.・解析的極限との比較（2026-11-09）

`compare_firstcore_references.py` はc1010の3D球殻平均を、[Mayer et al. (2025), Section 5 / Figure 7](https://arxiv.org/html/2510.12620#S5)、[Commerçon et al. (2011), Table 1](https://arxiv.org/html/1102.2921v1#S4.SS1)、[Gong & Ostriker (2009)](https://arxiv.org/abs/0904.3568) のLarson–Penston外層漸近形および一様球の自由落下時間と比較する。図 `firstcore_reference_comparison.png` と数値 `firstcore_reference_comparison.json` を生成し、`report.html` に埋め込んだ。Mayer Figure 7の数点は従来のラスタPDF概読から転記した `mayer_fig7_approx_read_off.csv` であり、論文公開数値表ではない。曲線の厳密なacceptance gateには使わない。

一様球のρ0=`5.2515854e-18 g cm^-3`から圧力なし自由落下時間は29.0475 kyr、c1010時刻29.1095 kyrとの比1.00214。これは時間尺度の確認でありRHD第一コアの厳密解ではない。14 Kの音速から計算したLarson–Penston外層漸近密度との20–300 au比の中央値は1.282、rのべき指数は−2.069、RMSは0.099 dex。対象領域は14.1–64.7 Kで厳密等温ではない。MayerのFigure 7概読では1–100 auの温度が本計算より高く、密度・速度も一致しない。Table EOS/opacityとMayerの固定値を混同しない。Commerçonの別初期条件のFLD値は第一コア半径8.1 au、中心T411 K、質量0.0233 M☉。今回のshock7.507–7.874 au、Tmax431.263 K、r<7.8 auで0.03035 M☉は同程度だが、初期温度・雲半径・質量の定義が異なるため厳密一致を主張しない。

### WOR-118のambient・放射圧切り分け（2026-11-09）

`audit_hydro_mass_ambient.py` でc930 3D tracer `r0` の面隣接セルを分類。RK2最終段階の面flux差+5242.866048 codeのうち、ambientのみの面が+4536.593（86.53%）、coreのみ+140.268（2.68%）、混在+566.006（10.80%）。ambientのみには内部の同一level面と周期境界の両方があり、外部からの物理流入だけでは説明できない。これはc930開始時のtracer分類であり、長期のコアへの誤差伝播を証明するものではない。

XD M-debug 64 rankの同一c930 restart対照では、ambient温度固定をOFFにしたjob **338285** の面flux差は元のjob338248と表示17桁で同一。10ステップのjob **338284** でもgas mass相対増分は+7.704779e-8で、元の+7.690570e-8とほぼ同じ（差0.185%）。温度固定はgas/radiation energyを変更するがmassを直接変更しない。`fld/radiation_pressure_in_riemann=false` のjob **338286** では最終段階の面flux差が+4.142e-11 codeまで低下した。放射圧OFFは原因特定の対照であり、物理モデルの採用設定ではない。各ジョブのinput、binary/restart/table SHA、環境はXD `debug/job_ID/provenance.txt`、面監査は `c930_hydro_mass_faces_sinkoff.json` と `c930_hydro_mass_faces_no_radp.json`。job338284/285/286はSlurm COMPLETED 0:0、面診断の各step MaxRSSは約35.6 GB。

対向する同一level face（外側level 3、gid2218/2219）を直接出力したjob **338307/338308** ではRK2最終段階のgas密度・速度・圧力が同一でも、面左側のlimiter λは0.1849148075対0.2163368599で異なり、質量fluxは−0.0073208967対−0.0031334988。Er face自体の差は数×10^-15。セル中心のλは互いに約0.2163368599で一致した。`CalculateRadiationFaceStates` の2次再構成はghost側のλ傾きに外側2層目のclosureを使うが、そのclosureの中心差分は存在しない3層目のErを必要とし、コードは端の配列値へclampしている。この状態は同じ面の反対側と等価ではない。隣接セルのλとARADの平均を共通面値とする隔離trial job **338310** では面差が−36.1001 codeへ減り、level3/周期境界寄与はほぼ消失。中心近傍の残差を調査中。trialのソースを検証済み設定へはまだ適用しない。ユーザー指示により、修正を採用する前にshock tubeの振動回帰を行う。

境界設定タスクが自ブロックの送信完了も待つようにした**順序だけ**の対照job **338325** は元設定と全方向・全levelで表示桁まで同じ+5242.866048 code。単独の依存関係変更は修正にならず採用しない。さらに中心level11の対向face（gid400/769）を出力したjob **338315** では、RK2最終段階のgas面状態は同一だが、左側Er faceは1538.681/1534.333、右側Er faceは1534.328/1538.727で入れ替わり、質量fluxが−10.5007/−11.0742へ分かれた。job **338327** の `wor118_rad_sync_diagnostic.txt` によれば、放射境界交換直後のghostは対向実セルと完全一致する。その後、`SRC_TERM` 内の `FLD::AddExplicitSourceTerms` が実セルErだけを更新し、ghostは更新前のまま次のRK段階のRiemann solverへ渡る。通常設定の `implicit_mixed_frame_drift=true` ではこのsourceは現在の放射ghostを読まないため、sourceを放射境界交換より先に実行するタスク順序を別試験中。明示的drift設定はghostを読むので同じ順序変更を無条件には適用できない。

### WOR-118の局所修正と実ソルバ検証（2026-11-09）

独立ブランチ `codex/wor118-radiation-face-consistency-20261109` で二点だけ変更した。`CalculateRadiationFaceStates` はMeshBlock端の面に限り、隣接する二つのセル中心closure（λと`a_rad`）の平均を左右共通の面値へ代入する。これにより、必要な第3 ghost層のないghost closure傾きから生じる左右差を避ける。通常設定 `implicit_mixed_frame_drift=true` では `SRC_TERM` の後に放射境界を送受信・設定し、更新された実セル`E_rad`を次のRK段階のghostへ渡す。明示drift設定はsourceが放射ghostを読むので元の順序を維持する。放射圧はRiemann solver内で有効のままで、shock tube回帰も別に確認する。

同一c930 restart・64 MPI rank M-debugの1ステップ面監査job **338337** では、最終RK段階の全域質量flux差は`-2.59687e-11 code`、同一level面と周期境界は全軸0、粗細面も丸め程度。元の`+5242.866048 code`に対し実質的に解消。全faceを平均する広い変更job **338335** も`-1.83035e-11 code`だったが、採用候補はMeshBlock端だけの狭い修正である。c930→940の10ステップM-debug job **338338** はSlurm COMPLETED 0:0（3:14、step MaxRSS `35,756,948 KiB`）、10/10 Newton収束、30 FGMRES solve・292反復、最大true linear relative residual `1.958e-8`、opacity診断0。高精度historyのgas massはc930とc940でともに`2.273781228894768e33 g`、元設定では相対`+7.690570e-8`。独立の`cons/dens` HDF5体積積分では相対`-1.000814e-10`で、float32出力量子化の範囲。3536 blockのAMR配置は同じ。入力、binary、restart、表SHAはXDの各job `provenance.txt`、面監査は `c930_hydro_mass_faces_boundary_only_trial.json`、HDF監査は `c930_c940_boundary_only_conserved_mass.json`。

同じc930 restartからc1010まで5ノード・320 rank M-test-aで続けたjob **338342** はSlurm COMPLETED 0:0（4:54、step MaxRSS `20,268,088 KiB`）。実行binary SHA `908d5def56faf5e0993526f1c709b2593eec814a2a26ffd83d4a0133b6ac0810`。80/80 Newton収束、240 FGMRES solve・2277反復、最大true linear relative residual `1.995988e-8`、最大Newton final_max `1.873906e-8`、opacity範囲逸脱0。c930→1010の高精度historyのgas massは`2.273781228894768e33`→`2.273781228894767e33 g`、相対`-3.33e-16`。最大密度`1.0081888964e-10 g/cm³`、Tmax`431.26325 K`、Jeans最小`36.4486 cells`、最終AMR3704 block、EOS表外セル0。正のx軸shockは元計算と同じ`7.507–7.874 au`で、HDF5最大温度差は相対`-7.90e-8`。修正版の最終質量は元計算より相対`-5.893e-7`で、元計算の質量増分を除く量に相当する。結果とSHAは `wor118_c930_c1010_repaired_run.json`、生成コードは `summarize_wor118_repaired_run.py`。元HST、HDF5、ログはローカルの `firstcore_integration_evidence/profiles/` と `job_338342_wor118_c930to1010.tar.gz` に保存。全経路c0からの再実行ではなく、検証対象は高密度c930→1010と同一設定の短い面監査である。

### WOR-118の積分法・少数rank確認

修正版production binary（SHA `908d5def...`）を使用したXD M-debug 64 rank・VL2のc930→940 job **338425** はCOMPLETED 0:0（3:18、step MaxRSS `35,756,144 KiB`）。10/10 Newton収束、30 FGMRES solve・292反復、最大true linear residual `1.982921e-8`。高精度historyのgas massはc930/c940とも`2.273781228894768e33 g`で、修正前VL2の相対増分`+7.389789e-8`が消えた。入力とログSHAを `wor118_c930_c940_vl2.json`、投入スクリプトを `wor118_mdebug_vl2_c930to940.sbatch` に記録。ローカル証拠archiveは `job_338425_wor118_vl2.tar.gz`、HSTは `profiles/c930_c940_wor118_vl2.hst`。

修正版と同じソースを用いたuniform shock tube N64をローカル1 MPI rank・`time/tlim=0.005`で短時間実行。55/55 Newton収束、200 FGMRES solve/200反復、t=0.005へ正常終了、95.2秒、最大RSS 195,036 KiB。入力・binary・ログSHAは `wor118_uniform_one_rank_short.json`。短時間の実行可否を確認する試験であり、振動判定は2/4 rankのN64/N128をt=0.1まで完走した比較で行う。

### 放射圧Riemann solverのshock tube回帰（2026-11-09）

ユーザー指示に従い、Riemann solverの放射圧を有効にしたまま、元commit `533ac4dca59b4901ba9a5c3d4e9435a25ca659fc` の旧ソースと、今回の二ファイルのみ修正したソースを同一N64/N128入力でt=0.1まで実行。uniform N64は2 MPI rank、N128は4 MPI rank。両方の実験は検証スクリプト `tst/nrfld_validation/shock_tube_validation.py` で`passed`、N64は1331/1331、N128は2666/2666 Newton収束、relaxed acceptは0。独立したRad-EOS HLLC N4096参照解を同じキャッシュから使用。入力SHAは双方`1858a89f6db7eb6d9bf40284063b86747ba0808e78f8e3a6f752ee20607d5b9d`。修正版の二つのソースSHAはXD本計算と完全一致（`d9655adb...`/`2b1091aa...`）。旧版のソースSHAは元commitと一致。各binary、最終出力、メモリ、solver回数は `wor118_shock_tube_regression.json` 内のmanifestに記録した。

| 解像度 | 密度L1旧→修正後 | 圧縮波x=0.60–0.88の密度全変動旧→修正後 | 有意な密度極値数 | 最大新旧差/参照解振幅（全5変数） |
|---|---:|---:|---:|---:|
| N64 | 0.100440→0.100413 | 1.882952→1.882937 | 1→1 | 0.0375% |
| N128 | 0.061106→0.061074 | 1.895114→1.895115 | 1→1 | 0.0754% |

温度と放射エネルギーの圧縮波周辺の有意極値数も両解像度で0→0。N128の密度全変動の増加は`1.57e-6`（正規化値）で、振動の新たな増加とは判定しない。N64ではradiation energyと全圧の参照L1がそれぞれ約0.10%/0.12%悪化し、他物理量とN128のL1は概ね改善。差はいずれも非常に小さく、修正前後の曲線は図でほぼ重なる。N4096との大きなL1差は両版に共通する数値拡散を含み、WOR-118の局所修正で解消されたとは主張しない。結果 `wor118_shock_tube_regression.json/png`、比較コード `compare_wor118_shock_tube.py`、完全ログ・動画等の外部証拠 `/home/kosuke/simulation/test/rad_nr/wor118_shock_tube_20261109/`。

### Riemann放射圧ON/OFFの因果切り分け

同じ修正版N64・2 rank binary SHA `623c9f87978e3f9c7107e85d77f324703d34d5a794771766c2dbd2ccb949784c`で、Riemann solver内の`fld/radiation_pressure_in_riemann`だけをfalseにした対照をt=0.1まで実施。放射力など他の設定はONのままなので、物理的一貫性のある代替モデルではなく局所寄与の診断である。OFFの入力SHAは`7fce54aa94d57cffc6812955556298c3026f23ce94cab4ef6f274fefebfd14c8`、1381/1381 Newton収束、4179 FGMRES solve/4179反復、wall 11:48.98。最初に行ったコマンドライン上だけの指定はAthena++が未登録パラメータとして計算前に拒否し、`setup_failed.log`として外部証拠へ保管。入力ファイルへ設定を明記した再実行が本対照である。

最終t=0.1の独立N4096参照への相対L1はON→OFFで密度`0.100413→0.100674`、速度`0.038046→0.060868`、放射エネルギー`0.021414→0.034805`、全圧`0.016783→0.029898`。密度の圧縮波周辺の有意極値数は両方1で、OFFでも破局的な密度振動は再現しない。一方、全21時刻の系列にはOFFで各物理量に最大2個の追加有意極値が見られる時刻があり、t≈0.015の温度・放射エネルギーの谷もOFFで深い。これらは放射圧ONが解の一致と一部波形を改善する根拠だが、OFFは物理モデルも変えるため、歴史的な振動の唯一の解消要因とまでは断定しない。新旧修正の放射圧ON回帰がPASSであることが今回の採用判定で、OFFは採用しない。全指標と入力・log SHAは `wor118_riemann_radp_ablation.json`、図は `wor118_riemann_radp_ablation.png`、再生成コードは `compare_riemann_radp_ablation.py`。生ログは外部証拠 `wor118_shock_tube_20261109/radp_off_n64_control/`。
