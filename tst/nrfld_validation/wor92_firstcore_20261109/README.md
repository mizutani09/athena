# WOR-92 first-core 継続検証（2026-11-09）

**最新結果:** cycle1010でρmax=`1.00819e-10 g cm^-3`、Jeans最小36.45 cells、第一コア形成を3D密度・温度・速度構造から確認。採用したcycle47保存点以降の計算量は約5.07ノード時間。最終詳細は[第一コア密度到達・最終解析](#第一コア密度到達最終解析2026-11-09-1251-jst)、密度・温度・流速図は[PNG](first_core_profiles.png)を参照。途中の「未到達」「継続中」は各時点の履歴である。

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
