# WOR-25: binary64 更新差を確認した FGMRES 停止判定

## 対象と結論

- ソース: `2e2017d2e14286725895b1c531c9e70910256c2c`、ブランチ `codex/wor25-roundoff-gate-20261107`。WOR-24 の中心差分粗細面候補 `e786abc429684cf413d23f423b6d48df5838437d` を含む。旧 AMR 演算子は対象外。
- 独立 worktree `/home/kosuke/simulation/athena-wor25-roundoff-gate`、隔離ビルド `/tmp/wor25_roundoff_gate_build`。元の dirty worktree は変更していない。
- ビルド: `nrfld_diff_amr_interface`, Cartesian, adiabatic, `hllc_fld`, `-nrmgfld -mpi -hdf5 -h5double`, `mpicxx -O3`, binary64。実行バイナリ SHA-256 `3ebe0aff6addf0b51956d3eedf4483601b9c59cc75f3ed040d930673eb864bf8`。
- Test C の強拡散では、FGMRES が要求した補正と格納された binary64 補正との差に `A` を作用させた量が、true residual の92%以上を説明する。残差恒等式の閉包誤差は true residual の最大約 `3.3e-9` 倍。式の評価順序変更だけで厳密 `1e-8` を保証する根拠はない。以前の long double 和と跨ぎ restart 補償更新の試験は改善しなかった。
- `nrfld/fgmres_roundoff_cap=1e-7` を指定した場合だけ、projected residual と要求補正からの予測 residual が `1e-8` 以下で、実補正との丸め差が主因であることを検査し、true residual `1e-7` 以下の補正を Newton に渡す。Newton の非線形残差判定は変更しない。**これは厳密な線形 `1e-8` 達成ではない。** cap 未指定では従来の判定を維持する。
- MG 単独改善はユーザー指示により Canceled。MG は FGMRES 前処理として使用した。

## 判定の具体式

restart 開始時の補正を `x_old`、Krylov が要求した更新を `d`、実際に格納された更新を `d_real=x_new-x_old`、`g=d-d_real` とする。`r_pred=r_old-A d`、`r_true=b-A x_new` を同じ演算子で計算する。条件は `||r_true||/||b||≤cap`、projected と `||r_pred||/||b||≤target`、`||A g||/||b||≥0.5||r_true||/||b||`、`||r_true-r_pred-A g||/||b||≤0.25||r_true||/||b||`、有限値。どれかを満たさなければ従来どおり反復・失敗する。cap は `target` 以上 `10×target` 以下だけ受け付ける。

## 実ソルバ結果

入力は本ディレクトリの `wor25_roundoff_gate_testc_20261107.in`（SHA-256 `cc5b526b71f387be301cf33b953a9a94ea026caa85a834e9c05870c3b0214d81`）、uniform は `wor25_roundoff_gate_uniform_20261107.in`（`d911ecd8c61ea0444e6b17febc994303019f55acd7412514b1ccdcb286910024`）。全 Test C は3 level shifted AMR、coupled gas/radiation、周期境界、中心面候補 ON、`fgmres_threshold=1e-8`。表の時間と RSS は `/usr/bin/time -v`。詳細なコマンド、全 Newton/FGMRES 履歴、face 対、保存量、最終 HDF checksum は JSON に記録した。

| run | rank / limiter / step | Newton 反復 | FGMRES solve / 反復 / 丸め採用 | 最大 true residual | 面対 / 最大相対差 | 最大エネルギー相対差 | wall / peak RSS | 判定 |
|---|---|---|---|---:|---:|---:|---|---|
| strict_gate_valid_r1 | 1 / variable / 2 | 16+7 | 23 / 364 / 6 | 5.649e-8 | 20 / 5.79e-17 | 5.57e-9 | 6:04 / 464300 KiB | PASS |
| strict_gate_valid_r2 | 2 / variable / 2 | 16+7 | 23 / 364 / 6 | 5.728e-8 | 20 / 5.79e-17 | 6.37e-9 | 4:25 / 256196 KiB | PASS |
| five_gate_r1 | 1 / variable / 5 | 16+7+5+4+3 | 35 / 496 / 18 | 5.649e-8 | 50 / 5.79e-17 | 5.57e-9 | 18:03 / 464344 KiB | PASS |
| five_gate_r2 | 2 / variable / 5 | 16+7+5+4+3 | 35 / 496 / 18 | 5.728e-8 | 50 / 5.79e-17 | 6.37e-9 | 10:50 / 256040 KiB | PASS |
| fixed_gate_r1 | 1 / fixed / 1 | 16 | 16 / 177 / 15 | 2.945e-8 | 10 / 0 | 2.29e-9 | 9:01 / 461112 KiB | PASS |
| fixed_gate_r2 | 2 / fixed / 1 | 16 | 16 / 176 / 15 | 3.476e-8 | 10 / 0 | 2.38e-9 | 5:29 / 254524 KiB | PASS |
| fixed_two_gate_r2 | 2 / fixed / 2 | 16+7 | 23 / 254 / 22 | 3.638e-8 | 20 / 0 | 5.26e-9 | 5:19 / 256368 KiB | PASS |
| strict_control_r1 | 1 / variable / 2、cap OFF | 1 step 収束後、2 step目失敗 | 17成功 solve + 80反復の失敗 solve、計378 | 3.695e-8 | 10 / 5.79e-17 | 6.30e-10 | 10:29 / 463204 KiB | 期待した FAIL、exit 134 |
| uniform_gate_r1/r2 | 1/2 / fixed / 1 | 2 / 2 | 2 / 12 / 0 | 3.766e-9 | 対象面なし | 2.49e-12 | 0:03 / 58464・45652 KiB | PASS |
| uniform_control_r1 | 1 / fixed / 1、cap OFF | 2 | 2 / 12 / 0 | 3.766e-9 | 対象面なし | 2.49e-12 | 0:03 / 58160 KiB | PASS |

5-step variable の1/2 rank は最終 HDF の全数値配列が完全一致。2-step variable と fixed 1-step の rank 差は放射場最大 `2.842e-14`、相対約 `1.9e-16`。uniform の cap ON/OFF と1/2 rank は全数値配列が完全一致。`cap=1e-9` の不正設定は起動時に想定どおり拒否された（Athena の fatal 出力はあるが MPI wrapper の exit status は0のため、ログ文言で判定）。

## 適用範囲

この commit は opt-in の停止判定のみである。粗細面中心差分候補も Test C で opt-in のまま。Mach 5、長時間実規模、別の limiter/幾何での自動適用はこの結果からは主張しない。WOR-24 と合わせた既定経路化は別 worktree で検証する。実行時間最適化はユーザー指示により今回の Done 条件から外す。

`wor25_roundoff_gate_20261107.json` と `wor25_roundoff_gate_evidence_20261107.tar.gz` に全測定を保存する。実行時には `OMP_NUM_THREADS=1`, `ATHENA_NRFLD_CENTERED_FACE_TRIAL=1`, `ATHENA_NRFLD_FGMRES_RESTART_AUDIT=1`, `ATHENA_NRFLD_INTERFACE_LEDGER=1`, `ATHENA_NRFLD_ENERGY_LEDGER=1` を設定した。uniform は conservative prototype/shared coeff を明示し、元の入力の対照にも同じ設定を使った。
