# WOR-92 first-core 継続検証（2026-11-09）

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
