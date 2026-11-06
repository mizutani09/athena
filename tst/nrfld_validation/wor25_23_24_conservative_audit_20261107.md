# WOR-25 / WOR-23 / WOR-24: 保存的 AMR 演算子のローカル再検証（2026-11-07）

## 基点と測定条件

検証基点は merge commit `84061d74b866b2a42f7a0203cfb3ea0c804aad88`。この履歴には WOR-101 の粗細面台帳、WOR-102 の共通面フラックス試作、WOR-105 の Test C/Mach 5 既定化 (`f4f6ec41`)、WOR-106 の保存量ゴースト同期と restart dt 修正 (`df944aff`, `1e9e0b6d`) が含まれる。元の `nr_mgrad_dev` と既存 worktree は変更していない。`ATHENA_NRFLD_LEGACY_INTERFACE=1` だけが同じ実行ファイルで旧演算子を明示選択する。

専用 worktree: `codex/wor25-23-24-conservative-audit-20261107`。実補正診断の確定ソースは commit `55bd4f77`。隔離した MPI/O3/HDF5 ビルドの source SHA-256 は `46037058efe80f54ed14725eb8719eaca2b2f6d6cb73a14a47bb34758cfe2d35`、binary SHA-256 は `783adb1e6004bf88438930b1ce9db4042b76472105a086dc88b87a8a14f719ca`。ビルド時の manifest は基点 `84061d74` と dirty diff を記録し、その C++ diff を `55bd4f77` に確定した。HDF5 出力は **single precision**。全 Test C ケースでこのバイナリを再使用した。`/usr/bin/time` の RSS は MPI 全 rank の合計ではなく最大プロセス値。ケースの入力設定・入力 SHA-256・Git 基点・binary SHA-256・Newton trial 履歴・FGMRES の明示再計算した真残差・面/総量台帳・時間・RSS・PASS/FAIL は [機械可読結果](wor25_23_24_conservative_audit_20261107.json) にある。元のログと HDF は各レコードの `/tmp/...` パスに保持した。

この測定でソルバ完走は 67 ケース中 54 PASS、13 FAIL。FAIL は以下の MG 単独対照 9 件と閾値 `1e-10` の探索 4 件であり、診断クラッシュではない。別の保存性判定は台帳が揃った 38 件中、保存的経路など 36 PASS、明示した旧 AMR 経路 2 FAIL。台帳がない／MPI 出力が欠けた 29 件は保存性未判定とした。1 ケースの最大実行時間は 7.60 秒、最大 RSS は 269680 KiB。診断 ON/OFF の固定係数 3 配置は同じバイナリで Newton 結果、history bytes、HDF 全データが一致した。移植した変更は実補正を測る診断だけで、ソルバ方針は変更していない。

## WOR-25: 実補正モードと V-cycle

元の `[NRFLD Galerkin]` は FAS で粗格子状態と補正を取り違え、相対差をほぼ 1 と誤表示した。今回の診断は実際に prolong した補正を `A_H e_H` と `R A_h P e_H` の双方へ渡す。`root→octet` は複製された階層全体、MeshBlock 内は rank 0 のブロックを記録する。二 rank の MeshBlock 値は全ブロック集計ではない。

固定 limiter・同一 36 leaf MeshBlocks・`q=1e7`・contrast 1 の初回 V-cycle は次の通り。`block 2/1, 3/2` は同じ実補正に対する作用素差の重み付き相対 L2。`factor` は V-cycle 後／前の残差比。

| 配置 | root→octet | block 2/1 | block 3/2 | factor | MG Newton |
| --- | ---: | ---: | ---: | ---: | --- |
| uniform fine | 対象外 | 0.04480 | 0.05540 | 0.10584 | 2 回、PASS |
| AMR aligned | 3.198e-6 | 0.04459 | 0.05530 | 0.10579 | 5 回、PASS（V-cycle 棄却 4） |
| AMR shifted | 3.133e-6 | 0.04857 | 0.05802 | 0.12031 | 0 回、backtracking exhausted |

aligned と shifted の作用素差と初回残差減少は近い。shifted では MG 内の最初の補正で full-step 最小値が `-19.552` になるので、差の大きさだけを失敗原因としない。係数診断では各 MeshBlock level の拡散対角が coarsening ごとにほぼ 1/4、row-sum 相対誤差は `1.2e-16` 以下で、単純な level scaling 欠落は見つからない。octet→MeshBlock 受け渡しは段階残差を記録したが、その箇所だけの直接 `R A_h P` 比較は未実装。

固定 limiter・contrast `1e4` では初回 root→octet 差が aligned `0.0281`、shifted `0.0268`、MeshBlock 内の差が `0.445–0.508`、初回 factor が uniform／aligned／shifted `39.2/48.4/65.9`。MG 単独は 3 配置とも FAIL。FGMRES は同じ係数とメッシュで 10／14／17 反復、Newton 1 回で全 PASS。したがって粗格子不整合は強拡散時の前処理品質に関係するが、FGMRES が使える補正方向へ回復できる。

可変 limiter の空間変化係数では、`q=5e5` の MG 初回 factor は uniform／aligned／shifted `0.936/0.926/0.938` で全 PASS、`q=7e5` は `1.021/1.007/1.018` で全 FAIL。MeshBlock 内の実補正差は両系列とも概ね `0.69–0.73`、root→octet は `~2.1–2.3e-4`。大きな Galerkin 不一致だけでは成否を予測できない。係数は Newton step 中に lagged である。FGMRES は前者 12／12／14、後者 27／27／33 総反復で全 PASSし、1/2 rank で反復数が一致した。ここでの coarse-level 補正は低周波成分の実際の入力だが、誤差のスペクトル分解自体は行っていない。

## WOR-23 / WOR-24: Jv と補正受け渡し

固定係数 `q=1e7` の frozen-coefficient 全領域有限差分 Jv は `h=1e-4` で相対 L2 `~3e-10`、shifted 粗細面領域は `1.0e-9`。aligned の粗細面領域だけ相対値 `3.17e-4` だが、`h` を 10 倍小さくすると概ね 10 倍に増え、同じ face の差分 tangent 対照は `5.2e-17`。この領域の小さい基準信号に対する差分丸めと整合し、有限の `h→0` 不一致という結論は出せない。可変係数でも `h=1e-4` の全領域は `~2e-9～9e-9`。この global Jv 診断は periodic・only-radiation Test C の frozen 系を対象とし、coupled Schur Jv や物理境界を直接検証しない。

同じ binary の FGMRES matvec 加法性は保存的 AMR で `4.9e-16～1.2e-15`。明示した旧 AMR 経路は aligned `1.86e-4`、shifted `1.52e-4`。FGMRES の最終 `relative_residual` は候補補正へ演算子を再適用した真残差で、PASS ケースは `1e-8` 以下。固定 `q=1e7` の 1/2 rank 対照は uniform／aligned／shifted の総反復 `13/17/18`、Newton 2 回で一致する。HDF の全データも一致するが single precision なので、ログの最終 L2 は rank 間で最後の数桁が異なる。

MG 単独の最初の補正を同一 RHS・係数で追うと、共通基準 `||b-Aδ||/||b||` は次の通り。`MG end` と `retrieved` の一致、境界更新が最初の残差ジャンプであること、負の shifted 方向がそれ以前に存在することを分けて扱う。1/2 rank で数値は一致した。

| 配置 | MG end | retrieved | boundary | fine sweeps | full-step 最小値 |
| --- | ---: | ---: | ---: | ---: | ---: |
| uniform fine | 1.00e-11 | 1.00e-11 | 1.00e-11 | 1.00e-11 | 1.0100 |
| AMR aligned | 4.28e-11 | 4.28e-11 | 9.22e-7 | 4.19e-7 | 1.0114 |
| AMR shifted | 5.55e-11 | 5.55e-11 | 1.637e-2 | 8.669e-3 | -19.552 |

保存的経路の FGMRES では aligned／shifted とも 2 Newton、17／18 総反復で PASS。post-MG 補正 ghost を `minmod` と `tangent` に切り替えた対照は反復数、記録された受け渡し残差、最終 HDF が一致した。この縮小条件で補間の一方だけを変更する根拠はない。結合 Test C の 1/2 rank × uniform/aligned/shifted × fixed/variable limiter の 12 ケースも全 PASS。各 rank 対で Newton 20 回、FGMRES 総反復 118–202、HDF 全データが一致し、FGMRES 加法性は機械精度。coupled global Jv の有限差分比較は診断の対象外で未完了。

## 保存性、旧演算子対照、restart

同一 binary・同一入力で明示した旧経路と保存的既定経路を比較した固定 `q=1e7` の結果:

| 配置 | 経路 | FGMRES 総反復 | 粗細面最大相対不収支 | 周期領域 ΔEr/Er |
| --- | --- | ---: | ---: | ---: |
| aligned | 旧 | 29 | 1.03e-2 | -5.143e-5 |
| aligned | 保存的 | 17 | 1.44e-10 | 1.68e-12 |
| shifted | 旧 | 66 | 2.05e-2 | -5.199e-5 |
| shifted | 保存的 | 18 | 1.51e-9 | 2.07e-12 |

保存的高コントラスト条件では粗細面相対不収支が `~2.5–2.9e-13` の面もある。1/2 rank 結合ケースのガス＋放射の周期領域変化は最大 `~4.4e-14`、粗細面相対不収支は最大 `2.4e-10`。MPI stdout が台帳行を途中で混ぜた少数ケースでは、そのケースの面台帳を不完全と明示して数値を採用しない。完全な 1 rank 台帳と 12 結合ケースの 1/2 rank 台帳で保存性を確認した。

WOR-106 の source では陰的更新後に hydro conserved buffer と放射を同期し、その後 primitive を同期する。restart 時は保存済みの次 `dt` を上限に再評価する。短い結合 static/adaptive・legacy/conservative・1→2/2→2 rank の 6 組は連続実行と restart の cycle 5 時刻 `0.006229967810213566`、AMR topology、HDF 3 user fields が一致した。各 run の入力 hash、NR/FGMRES 履歴、時間、RSS は [restart 結果](wor106_restart_local_20261107.json) にある。今回の HDF 比較は single precision の `Er/rho_profile/sigma_R` であり、hydro 全保存量の double precision 一致を新たに証明したものではない。既存の WOR-106 double-HDF5 4→4/1→4 rank 回帰は別報告に保持する。

## WOR-29 と残る実験

新しい保存的演算子の未測定だった **2 rank・shifted AMR・高コントラスト** で RBGS の色順序を再比較した。default／post 逆順／root/octet 色統一の FGMRES 総反復は固定係数 `17/18/17`、可変 limiter `33/34/32`。全 6 ケースで Newton 収束し、総エネルギーは許容範囲内。1 反復の差はあるが、既定値変更を支持する一貫した改善はない。現行 FGMRES は柔軟前処理なので、色順序から V-cycle 全体の対称性は主張しない。

未完了: (1) octet→MeshBlock で同期済み ghost と実補正を使った直接 `A_H` 対 `R A_h P` 比較、(2) coupled 2 変数／Schur の全領域差分 Jv と物理境界、(3) collapse 同一 checkpoint の精度・positivity・restart double-HDF 回帰、(4) 最粗段からの低周波誤差投影。今回の `fgmres_threshold=1e-10` 探索は 2 回目の solve が真残差 `~1e-9` で頭打ちとなり 200 反復で FAIL、`nr_threshold=1e-10` の探索も `~1e-9` で停滞した。これは許容値を上げた既定 PASS と混ぜず、数値精度・停止規則の次の実験として残す。WOR-25/23/24/29 は Done としない。
