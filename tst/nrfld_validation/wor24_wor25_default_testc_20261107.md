# WOR-24 / WOR-25: Test C 既定経路の実ソルバ検証

## 判定

周期・Cartesian・cubic の multilevel Test C では、共通中心差分の粗細面項を Newton residual、補正境界、FGMRES matvec に同時に使う経路を既定化した。binary64 の格納補正による残差床が検査で確認された場合だけ、FGMRES の true residual に `10 × nrfld/fgmres_threshold` の上限を適用する。FGMRES の報告値は実際の true residual のままで、Newton の非線形残差判定は変更しない。Mach 5 と uniform の既定経路は変更しない。

既定化のソース commit は `b0849726084aec3fc055adf3da31f36dc72404e2`（親 `2e2017d2e14286725895b1c531c9e70910256c2c`）、ブランチ `codex/wor24-wor25-default-testc-20261107`。親に含まれる WOR-24 中心面候補は `e786abc429684cf413d23f423b6d48df5838437d`、WOR-25 の opt-in 丸め判定は `2e2017d2`。独立 worktree `/home/kosuke/simulation/athena-wor24-wor25-default-testc`、隔離ビルド `/tmp/wor24_wor25_default_testc_build`、binary SHA-256 `62c3a7f2ebc15af56062c9d90669deae7b5e60890ca33d3a51d413fd35c4d121`。ビルドは `nrfld_diff_amr_interface`, Cartesian, adiabatic, `hllc_fld`, `-nrmgfld -mpi -hdf5 -h5double`, `mpicxx -O3`。元の dirty worktree は変更していない。

## 初期化順序の修正

最初の既定化案は、線形ソルバ生成時に `NRFLDPrototype::Mode` がまだ設定されていないため、中心面演算子が後で選択されても `fgmres_roundoff_cap=0` として実行された。この二つの準備試験は途中停止し、判定には使わなかった（旧 binary SHA-256 `5a7b840bf1b2c4c3938a6c5345f861a76ed733b89a80c52eaf95779006da62e9`、ログは evidence に保存）。修正では constructor 時点で確定している multilevel mesh、Test C problem generator、operator の明示的 override から上限の既定値を決める。`GetOrAddReal` で値を保持するため、checkpoint に `fgmres_roundoff_cap = 1e-07` が保存される。

## 試験と数値

Test C 入力 `testc_threelevel_shifted_20261107.in`（SHA-256 `c1ffedae69f670d6f17740805077b6741c7cefdaf2baf4639cc7ab5f1fa38a74`）、再起動用 `wor24_wor25_default_restart_testc_20261107.in`（`12bb7515b1318591fd04f3d998d0c4e18a8d93d9af9eea3c72e24388208228c4`）、uniform 入力 `wor25_k1_uniform_20261107.in`（`79662176bb853f6c71c70e541b4b037a8141b6cc3997195a05f5783a2357ff99`）。Test C は3 level shifted AMR、coupled gas/radiation、周期境界、`fgmres_threshold=1e-8`。表の true residual は `||b-Ax||/||b||` の最大値。詳細なコマンド、環境、全反復履歴、各面対、保存量、実行時間、メモリ、HDF digest は JSON に記録した。

| run | rank / limiter / step | Newton | FGMRES solve / 反復 / 丸め採用 | 最大 true residual | 面対 / 最大相対流束差 | 最大エネルギー相対差 | wall / peak RSS | 判定 |
|---|---|---|---|---:|---:|---:|---|---|
| default_fixedcap_testc_r1 | 1 / variable / 2 | 16+7 | 23 / 364 / 6 | 5.649e-8 | 20 / 5.79e-17 | 5.57e-9 | 14:39 / 464380 KiB | PASS |
| default_fixedcap_testc_r2 | 2 / variable / 2 | 16+7 | 23 / 364 / 6 | 5.728e-8 | 20 / 5.79e-17 | 6.37e-9 | 12:22 / 256044 KiB | PASS |
| default_fixed_limiter_r2 | 2 / fixed / 1 | 16 | 16 / 176 / 15 | 3.476e-8 | 10 / 0 | 2.38e-9 | 8:14 / 254244 KiB | PASS |
| default_restart_seed_r2 | 2 / variable / 1 | 16 | 16 / 287 / 0 | 7.885e-9 | 10 / 5.79e-17 | 1.06e-9 | 9:35 / 254180 KiB | PASS |
| default_restart_continue_r2 | 2 / variable / 1追加 | 7 | 7 / 77 / 6 | 5.623e-8 | 10 / 0 | 6.30e-9 | 1:29 / 253360 KiB | PASS |
| optout_old_face_r1 | 1 / variable / 1、旧面演算子 | 16 | 16 / 288 / 0 | 8.091e-9 | 10 / **3.45e-3** | 5.13e-10 | 8:21 / 459288 KiB | solver PASS、面保存 FAIL |
| default_uniform_r1 | 1 / fixed / 1 | 2 | 2 / 12 / 0 | 3.766e-9 | 対象面なし | 1.01e-11 | 0:01 / 55128 KiB | PASS |
| base_uniform_r1 | 1 / fixed / 1、変更前バイナリ | 2 | 2 / 12 / 0 | 3.766e-9 | 対象面なし | 1.01e-11 | 0:02 / 54956 KiB | PASS |

既定 Test C の2 rank 最終 HDF 全数値配列は、明示的に同じ中心面候補と上限を有効化した検証済みバイナリの結果と完全一致。uniform の変更前後も全数値配列が完全一致。Test C の1/2 rank 最終放射場差は最大 `2.842e-14`、相対 `1.904e-16`。1ステップ checkpoint 作成前の HDF は連続実行と完全一致し、2ステップ目を再開した最終放射場は連続実行との差が最大 `2.842e-14`、相対 `1.904e-16`。checkpoint SHA-256 は `713cb7dd030184b2da69f1dea9a8f1ed8f72474cd0edf3ae29bf0556c7a901f4`、上限 `1e-7` が保存された。

旧面演算子への明示的 opt-out は Newton が収束しても局所面対が相対 `3.45e-3` と不保存であり、新しい中心面経路では `5.79e-17`。この対照は局所面保存の改善を演算子選択に結びつける。global Jv、4段階 MG→Newton handoff、regrid と1→2/2→2 rank restart は親候補の[報告](wor24_centered_wor25_roundoff_20261107.md)を参照。FGMRES 丸め差の診断、5-step 1/2 rank、固定 limiter 1/2 rank、strict cap OFF 対照は[WOR-25 の報告](wor25_roundoff_gate_20261107.md)を参照。

## 適用範囲と操作

`nrfld/fgmres_roundoff_cap=0` は厳密停止に戻す。`ATHENA_NRFLD_CENTERED_FACE_TRIAL=0` は旧面演算子の A/B 診断にのみ使用する。上限で採用された solve を線形 `1e-8` 達成と呼ばず、線形 true residual と Newton true residual を別々に報告する。binary64 で格納する補正の量子化は、単純な和の順序変更や補償和だけでは取り除けなかった。より高い精度の全経路や他の問題生成器への一般化は今回の対象外。MG 単独改善はユーザー指示により Canceled。実行時間最適化は今回の Done 条件から除外した。

`wor24_wor25_default_testc_20261107.json` と `wor24_wor25_default_testc_evidence_20261107.tar.gz` に各試験の入力、Git hash、コマンド、反復、true residual、保存量、実行時間、メモリ、PASS/FAIL と raw logs/checkpoint/HDF を保存した。
