# Related FLD Linear issues (WOR-114)

Snapshot 2026-11-09. This is a scope index for FLD and RSG-related WOR issues visible in the Work team. Commit-level decisions are in `fld_code_integration_manifest.md` and `fld_branch_inventory.md`. Issue status records project progress, not a PASS on the integration SHA.

| WOR | Status | Scope | Branch named in issue text |
|---|---|---|---|
| [WOR-9](https://linear.app/work-on-fld/issue/WOR-9) | Done | FLD検証テストの結果と再現条件を整理する | — |
| [WOR-10](https://linear.app/work-on-fld/issue/WOR-10) | Done | Mach 5 radiative shockのT_rad局所ピークを診断する | — |
| [WOR-11](https://linear.app/work-on-fld/issue/WOR-11) | Done | FLDソルバーの数値近似と保存性をレビューする | — |
| [WOR-12](https://linear.app/work-on-fld/issue/WOR-12) | Done | FLD実装と検証の論文構成を準備する | — |
| [WOR-13](https://linear.app/work-on-fld/issue/WOR-13) | Backlog | FreytagらのRSG star-in-a-box手法を文献から再構成する | — |
| [WOR-14](https://linear.app/work-on-fld/issue/WOR-14) | Backlog | RSG用の初期平衡モデルとpgen設計を確定する | — |
| [WOR-15](https://linear.app/work-on-fld/issue/WOR-15) | Backlog | Athena++で最小RSG star-in-a-box試験計算を構築する | — |
| [WOR-16](https://linear.app/work-on-fld/issue/WOR-16) | Backlog | RSGの対流構造と文献結果を定量比較する | — |
| [WOR-20](https://linear.app/work-on-fld/issue/WOR-20) | In Progress | FLD・HD/MHD・自己重力の実行時間内訳と並列性能を評価する | — |
| [WOR-21](https://linear.app/work-on-fld/issue/WOR-21) | In Progress | Protostellar collapseのNR–linearMG収束不良：原因特定と検証の管理 | — |
| [WOR-22](https://linear.app/work-on-fld/issue/WOR-22) | Done | 収束不良の固定状態・ビルド・小規模対照実験を確立する | — |
| [WOR-23](https://linear.app/work-on-fld/issue/WOR-23) | Done | AMR ghost補間込みでNewton JacobianのJv整合性を検証する | — |
| [WOR-24](https://linear.app/work-on-fld/issue/WOR-24) | Done | MG内部からNewtonへ戻す各段階の補正と線形残差を照合する | `codex/wor24-wor25-default-testc-20261107` |
| [WOR-25](https://linear.app/work-on-fld/issue/WOR-25) | Done | 係数restrictionと粗格子演算子の整合性をRAhPで評価する | `codex/wor25-roundoff-gate-20261107` |
| [WOR-26](https://linear.app/work-on-fld/issue/WOR-26) | Done | Schur右辺のFg+Fr相殺と全エネルギー残差の桁落ちを測定する | — |
| [WOR-27](https://linear.app/work-on-fld/issue/WOR-27) | In Progress | 原因特定後の最小修正を回帰試験・collapse継続・精度で評価する | — |
| [WOR-28](https://linear.app/work-on-fld/issue/WOR-28) | Done | Primitive出力にE_radを追加し、F_radの出力仕様を決定する | — |
| [WOR-29](https://linear.app/work-on-fld/issue/WOR-29) | Backlog | Red–Black緩和法の順序反転が対称性・収束性に与える影響を検証する | — |
| [WOR-30](https://linear.app/work-on-fld/issue/WOR-30) | Done | FLD係数の固定・再計算タイミングとMG行列更新処理を監査する | — |
| [WOR-51](https://linear.app/work-on-fld/issue/WOR-51) | Backlog | RSG star-in-a-box用Tomida系RealisticEoS表を検証・正規化する | — |
| [WOR-52](https://linear.app/work-on-fld/issue/WOR-52) | Done | 硬い物質–放射LTE交換項を漸近安定な局所解法へ置換する | — |
| [WOR-53](https://linear.app/work-on-fld/issue/WOR-53) | Done | Uniform RSGの強拡散・全エネルギー結合におけるNR–MG収束不良を解消する | — |
| [WOR-54](https://linear.app/work-on-fld/issue/WOR-54) | In Progress | FGMRESをAMR NR-FLDへ正式対応しcollapse収束性を検証する | — |
| [WOR-55](https://linear.app/work-on-fld/issue/WOR-55) | Done | Optabで金属・分子を含むRSG用opacity tableを生成する | — |
| [WOR-56](https://linear.app/work-on-fld/issue/WOR-56) | Done | FLD validation を正規化し全問題の再実行・可視化・HTML 報告を一括化する | — |
| [WOR-59](https://linear.app/work-on-fld/issue/WOR-59) | Done | Streaming front の位置遅れと広がりの原因を調査・改善する | — |
| [WOR-60](https://linear.app/work-on-fld/issue/WOR-60) | Done | Shock tube の Newton 残差停滞を調査し厳格収束を確認する | — |
| [WOR-61](https://linear.app/work-on-fld/issue/WOR-61) | Done | Mixed-frame Lorentz 補正項の陰的含有を切り替えて物理解と収束性を比較する | — |
| [WOR-62](https://linear.app/work-on-fld/issue/WOR-62) | Done | FLD追加検証を順次実装する：3D熱波・保存性・AMR再起動・mixed-frame・運用 | — |
| [WOR-63](https://linear.app/work-on-fld/issue/WOR-63) | Done | 3D熱波をuniformと一段AMRで再計算し解析解・保存性を検証する | — |
| [WOR-64](https://linear.app/work-on-fld/issue/WOR-64) | Done | 完全結合FLDの周期箱で質量・全エネルギー保存と許容誤差依存を測る | — |
| [WOR-65](https://linear.app/work-on-fld/issue/WOR-65) | Done | AMR-FGMRESの再起動・MPIランク・格子更新で解と真残差を照合する | — |
| [WOR-66](https://linear.app/work-on-fld/issue/WOR-66) | Done | FLD validationの短時間・全件・HPC層とrank別メモリ/OOM診断を整備する | — |
| [WOR-67](https://linear.app/work-on-fld/issue/WOR-67) | Done | Mach 5 radiative shock のガス温度スパイクを高解像度で検証する | — |
| [WOR-68](https://linear.app/work-on-fld/issue/WOR-68) | Backlog | Mayer5 CFL=0.3のcore外縁でNewton方向が非下降になりrestartで成否が変わる原因を特定する | — |
| [WOR-69](https://linear.app/work-on-fld/issue/WOR-69) | Canceled | NR-FLDの非立方セルと曲線座標に方向別・計量対応を実装する | — |
| [WOR-70](https://linear.app/work-on-fld/issue/WOR-70) | In Progress | 太陽対流層の局所箱計算をFLD検証として再現・定量評価する | — |
| [WOR-71](https://linear.app/work-on-fld/issue/WOR-71) | Done | HD2 下端境界の初期圧力を境界面で正しく設定する | — |
| [WOR-72](https://linear.app/work-on-fld/issue/WOR-72) | In Progress | 太陽対流の開放境界・重力込みエネルギー収支の差を切り分ける | — |
| [WOR-73](https://linear.app/work-on-fld/issue/WOR-73) | In Progress | 太陽対流 HD2 下端の約4分周期振動と振幅増大を切り分ける | — |
| [WOR-74](https://linear.app/work-on-fld/issue/WOR-74) | In Progress | 太陽対流の下端上昇流エントロピーを定常太陽流束に較正する | — |
| [WOR-75](https://linear.app/work-on-fld/issue/WOR-75) | In Progress | 太陽対流の初期熱構造と離散FLD境界を整合させ長い熱緩和を短縮する | — |
| [WOR-76](https://linear.app/work-on-fld/issue/WOR-76) | Done | 太陽対流のcheckpoint再起動と連続実行の初期差を追跡する | — |
| [WOR-77](https://linear.app/work-on-fld/issue/WOR-77) | Backlog | NRの成分別更新量制限がNewton方向を歪めAMR更新後の収束を阻害する | — |
| [WOR-78](https://linear.app/work-on-fld/issue/WOR-78) | Backlog | 【D論審査・長期準備】解析式の手計算・典型値・恒星／連星進化のスケール則を整理する | — |
| [WOR-79](https://linear.app/work-on-fld/issue/WOR-79) | Done | 【完了履歴】FLD論文の構成とVerification対象を確定 | — |
| [WOR-80](https://linear.app/work-on-fld/issue/WOR-80) | Done | 【完了履歴】Grey FLD・mixed-frame方程式と日本語Methodノートの整理 | — |
| [WOR-81](https://linear.app/work-on-fld/issue/WOR-81) | Done | 【完了履歴】CodexによるMethod現行実装のread-only監査 | — |
| [WOR-82](https://linear.app/work-on-fld/issue/WOR-82) | Done | 【完了履歴】fld-paper GitHub・Overleaf共同編集環境の構築 | — |
| [WOR-83](https://linear.app/work-on-fld/issue/WOR-83) | Done | 【完了履歴】Methodの実装整合改訂・PR #1をmainへmerge | — |
| [WOR-84](https://linear.app/work-on-fld/issue/WOR-84) | Done | Verification本文を11月の最新validationへ更新（Draft PR #2） | — |
| [WOR-85](https://linear.app/work-on-fld/issue/WOR-85) | Done | 【Codex次タスク】Verificationの最新PDF図を収集し、run provenanceとcaption案を記録 | `codex/verification-figures` |
| [WOR-86](https://linear.app/work-on-fld/issue/WOR-86) | Done | Method統合版をOverleafでPull／コンパイルし数式・図・引用を確認 | — |
| [WOR-87](https://linear.app/work-on-fld/issue/WOR-87) | In Progress | Verification最新版の図・caption・本文を統合（Draft PR #6） | — |
| [WOR-88](https://linear.app/work-on-fld/issue/WOR-88) | In Progress | Applicationsの採用結果・掲載図・物理的主張を確定する | — |
| [WOR-89](https://linear.app/work-on-fld/issue/WOR-89) | Todo | Introduction・Discussion・Abstract・Conclusionsを最新成果と整合させる | — |
| [WOR-90](https://linear.app/work-on-fld/issue/WOR-90) | Todo | 全文最終QA：再現性・引用・LaTeX・図・コード版・投稿形式を確認 | — |
| [WOR-91](https://linear.app/work-on-fld/issue/WOR-91) | Todo | XD2000で太陽対流の48 km以上の解像度・長時間・流束較正計算を行う | — |
| [WOR-92](https://linear.app/work-on-fld/issue/WOR-92) | In Progress | Protostellar collapseをtable EoS・opacityへ拡張しCFL=0.3を検証する | — |
| [WOR-93](https://linear.app/work-on-fld/issue/WOR-93) | Done | 【完了記録】理想気体・一定opacityの第一コア形成をCFL=0.3で確認 | — |
| [WOR-94](https://linear.app/work-on-fld/issue/WOR-94) | Done | HLLD-FLDの微小輻射・B=0極限を最終版ソースで再検証する | — |
| [WOR-95](https://linear.app/work-on-fld/issue/WOR-95) | In Progress | 最終版collapse設定のJeans AMR発火とdiffusion次数を短時間回帰する | — |
| [WOR-96](https://linear.app/work-on-fld/issue/WOR-96) | Backlog | Mayer Section 5との第一コア構造の定量差を切り分ける | — |
| [WOR-97](https://linear.app/work-on-fld/issue/WOR-97) | Done | Section 2–3 の著者レビュー修正と表記統一（PR #4） | — |
| [WOR-98](https://linear.app/work-on-fld/issue/WOR-98) | Done | GitHub ActionsでPRごとのAASTeX PDFプレビューを自動生成 | — |
| [WOR-99](https://linear.app/work-on-fld/issue/WOR-99) | In Review | 投稿前にpaperpile.bibのBibTeX構文エラーと欠落引用を修復する | — |
| [WOR-100](https://linear.app/work-on-fld/issue/WOR-100) | Backlog | Mach 5 radiative shock の N4096 収束性と AMR/SMR 同等性を検証する | — |
| [WOR-101](https://linear.app/work-on-fld/issue/WOR-101) | Done | Mach 5 AMR粗細境界の陰的FLD非保存を特定し保存的修正を判断する | `codex/wor67-xd-20261105` |
| [WOR-102](https://linear.app/work-on-fld/issue/WOR-102) | Done | AMR粗細面の陰的FLD拡散を保存的に実装・検証する | — |
| [WOR-103](https://linear.app/work-on-fld/issue/WOR-103) | Done | 【Codex次タスク】Streaming・Mach 2/5・Closed-boxの不足図を既存データからPDF化する | — |
| [WOR-104](https://linear.app/work-on-fld/issue/WOR-104) | In Progress | 論文Figureの文字・線・マーカー書式を統一（著者フォント校正→Codex一括作図） | — |
| [WOR-105](https://linear.app/work-on-fld/issue/WOR-105) | Done | AMR陰的FLDの粗細面保存を対応構成で既定化する | — |
| [WOR-106](https://linear.app/work-on-fld/issue/WOR-106) | Done | 結合FLDのAMR再起動で粗細境界状態と次ステップがずれる原因を修正する | — |
| [WOR-107](https://linear.app/work-on-fld/issue/WOR-107) | In Review | 【Codex指示】Verification図の再設計・統一書式・取捨選択・本文同期 | — |
| [WOR-108](https://linear.app/work-on-fld/issue/WOR-108) | Done | 【Codex事前調査】CASTROのstatic equilibriumと既存FLD出力を照合 | — |
| [WOR-109](https://linear.app/work-on-fld/issue/WOR-109) | Done | 太陽対流3Dスナップショットを波長・角度依存放射輸送で後処理し観測量を比較する | — |
| [WOR-110](https://linear.app/work-on-fld/issue/WOR-110) | Todo | 太陽対流の波長別観測比較へ向け上層・鉛直解像度・非灰色熱構造を検証する | — |
| [WOR-111](https://linear.app/work-on-fld/issue/WOR-111) | In Progress | 太陽局所対流のMHD+FLD比較で磁場と上端流束への影響を検証する | — |
| [WOR-112](https://linear.app/work-on-fld/issue/WOR-112) | Done | general EOSのMHD+FLD更新で磁気エネルギーを全エネルギーに保持する | — |
| [WOR-113](https://linear.app/work-on-fld/issue/WOR-113) | Done | 太陽MHD+FLD上端の流出境界でHLLDが流入させる問題を修正する | — |
| [WOR-114](https://linear.app/work-on-fld/issue/WOR-114) | Todo | 【Codex】FLD論文用Athena++コード統合・再現版固定・本家upstream準備 | — |
| [WOR-115](https://linear.app/work-on-fld/issue/WOR-115) | Todo | 【Application掲載ゲート】collapseの参照計算・物理精度・再現版を確定する | — |
| [WOR-116](https://linear.app/work-on-fld/issue/WOR-116) | Todo | 【Application掲載ゲート】太陽局所対流の熱収支・定常性・掲載範囲を確定する | — |
| [WOR-117](https://linear.app/work-on-fld/issue/WOR-117) | Backlog | 【将来のupstream統合】PrincetonUniversity/athena向けFLD実装の分割・互換性・提案PR準備 | — |

Branch names are unverified pointers that may refer to trials or historical outputs; no adoption is implied. Untracked local files and unpublished commits are listed by worktree in the branch inventory.

