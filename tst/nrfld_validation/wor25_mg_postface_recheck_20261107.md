# WOR-25: 中心面演算子導入後の MG 単独再確認

## 問いと判定

新しい保存的粗細面演算子によって、従来は FGMRES でしか解けなかった条件が MG 単独でも解けるかを、同一バイナリで面演算子だけ切り替えて確認した。**条件によっては外側の Newton が MG 単独でも完走するようになった。MG 線形 solve の厳密収束が改善したという結果ではない。**

- ソース commit `b0849726084aec3fc055adf3da31f36dc72404e2`、ブランチ `codex/wor24-wor25-default-testc-20261107`、Test C MPI/HDF double ビルド binary SHA-256 `62c3a7f2ebc15af56062c9d90669deae7b5e60890ca33d3a51d413fd35c4d121`。独立 worktree と隔離ビルドを使用し、元の dirty worktree・XD2000・main は触れていない。
- 旧面演算子対照は同じバイナリで `ATHENA_NRFLD_CENTERED_FACE_TRIAL=0` のみ変更。`linear_solver=mg` では FGMRES の丸め差停止判定は使用されない。
- 元の MG 単独失敗条件は fixed limiter・shifted AMR・強拡散 `q=1e7`、入力 `wor25_mg_shifted_fixed_q1e7_20261107.in`、SHA-256 `ba333aee43531158fe9d34b50a6e9cc97ffbfa70fdb43a1ae5dcbc4045fdf8d2`。連成変係数は既定3-level shifted Test C 入力 `testc_threelevel_shifted_20261107.in`、SHA-256 `c1ffedae69f670d6f17740805077b6741c7cefdaf2baf4639cc7ab5f1fa38a74`。両入力とも `nr_threshold=1e-7`。

## 結果

| 条件 | 面演算子・solver | rank / step | Newton | MG内部の停止 | 最大面流束相対差 | wall / peak RSS | 判定 |
|---|---|---|---|---|---:|---|---|
| fixed `q=1e7` | 中心面・MG | 1 / 1 | 初回 `backtracking_exhausted` | 初回補正後の共通真残差 `8.67e-3` | 出力前に停止 | 0:12 / 192556 KiB | FAIL |
| fixed `q=1e7` | 旧面・MG | 1 / 1 | 同上 | 同残差 `8.67e-3` | 出力前に停止 | 0:10 / 192556 KiB | FAIL |
| fixed `q=1e7` | 中心面・FGMRES | 1 / 1 | 2反復、最終L2 `1.01e-9` | 9+9反復、true residual `6.64e-9 / 9.47e-9` | 0 | 0:11 / 198964 KiB | PASS |
| coupled variable | 中心面・MG | 1 / 1 | 16反復、L2 `2.23e-9` | 16 solveすべて30 V-cycle上限 | `1.08e-16` | 0:47 / 443744 KiB | 外側 PASS、線形目標未達 |
| coupled variable | 旧面・MG | 1 / 1 | 16反復、L2 `2.38e-9` | 同上 | `4.93e-3` | 0:47 / 443716 KiB | 外側 PASS、面保存 FAIL |
| coupled variable | 中心面・MG | 1/2 / 2 | 両rank 16+8反復、最終L2 `8.41e-9`、max `4.88e-8` | 各rank 16回の30 V-cycle上限と8回の増大検出 | `1.08e-16`（20面対） | 1:09 / 447020 KiB、1:37 / 247716 KiB | 外側・面保存 PASS |
| coupled variable | 旧面・MG | 1/2 / 2 | 両rank 1 step目16反復収束、2 step目12反復後 `backtracking_exhausted` | 各rank 16回の30 V-cycle上限と8回の増大検出 | `4.93e-3`（完了した10面対） | 1:27 / 447652 KiB、1:49 / 247732 KiB | FAIL |

固定 `q=1e7` では、新旧のMGが初回補正に約 `-20.56` の体積平均を返し、全18432セルで full step の放射エネルギーが非正となった。中心面経路でMG終了時の共通相対残差は `5.50e-11` でも、境界反映後 `1.637e-2`、追加 fine sweeps 後 `8.669e-3`。旧面経路の値も同程度。初回 Newton は両者で失敗した。FGMRES の初回補正平均は約 `-7.89e-6` で正値性を保ち、2反復で外側 Newton が収束した。この既知失敗を中心面修正が消したとは言えない。

連成変係数の2 stepでは、中心面＋MGの1/2 rank 最終 HDF 全数値配列は完全一致。既定 FGMRES 経路の1 rank最終放射場との差は最大 `3.210e-9`、場の最大値に対する比 `2.151e-11`。中心面＋MGの暗黙エネルギー台帳の最大相対差は1/2 rankで `4.40e-9 / 5.08e-9`。この条件での外側 Newton の改善と面保存は再現したが、MG各 solve の厳密な線形真残差を達成したとは判定しない。

## 解釈

新しい面演算子は Newton residual と補正境界の整合性・面保存を改善し、連成変係数の2 stepでは外側 Newton がMG補正でも完走した。固定強拡散の失敗では、粗細面境界反映後の残差と大きな負の定数補正が残り、MG単独は引き続き不適。MGの coarse operator/transfer をこの追試で変更しておらず、MG線形収束性能の一般的な改善やFGMRES置換を主張しない。検証済みの通常経路は引き続き FGMRES＋MG前処理。

全入力/Git・binary hash、コマンド、Newton・FGMRES履歴、MG停止回数、4段階handoff、保存面対、エネルギー台帳、実行時間・RSS、PASS/FAIL、HDF比較は `wor25_mg_postface_recheck_20261107.json`、raw logs/HDFは `wor25_mg_postface_recheck_evidence_20261107.tar.gz` に保存した。
