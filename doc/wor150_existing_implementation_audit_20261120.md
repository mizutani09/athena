# WOR-150 既存実装・文献監査

更新日: 2026-11-20 (JST)

## 結論

modified Lane--Emden + BSCF の機構は、元の BSCF を直接改変せず、独立した
`/home/kosuke/simulation/mhd/BSCF-rsg` に C++ で既に組み立てられている。
WOR-153--159 の通常テスト 32 件、および 6 executable を
`-Wall -Wextra -Wpedantic -Werror -fsanitize=address,undefined` で再ビルドした
同じ 32 件はすべて通過した。元の BSCF は commit
`e67837574aca40ed172366118ed1c4d3255e3698` の clean なままである。

ただし、現在の固定指数 `n_rad=3`, `n_conv=1.5` モデルは数値的には閉じるが、
WOR-151 の MESA 構造には合わない。したがって「二層 SCF 機構がない」のではなく、
「機構は検証済みだが、production 用の MESA-fit EOS が未実装」という状態である。
次のクリティカルパスは、MESA の `P(rho)` / enthalpy を保存する一般化 barotrope
または table EOS を C++ で実装し、同じ SCF に差し替えることである。

Athena++ 側では HD-only の RK4 + Berta WENO-Z (`4wz`) が branch
`wor-120-hd-4wz-memory2`, commit `e3b071b9d` で production 判定を通過している。
これは FLD、MHD、相対論、4次精度 passive-scalar transport を承認する結果ではない。
FLD 実装自体は存在するが、MESA-fit SCF 解との結合、Planck mean opacity、光度輸送の
熱平衡 gate は別段階である。

## 監査対象と分離

| 対象 | commit / branch | 扱い |
|---|---|---|
| 元 BSCF | `e678375`, `master` | 読み取り専用の参照実装。変更なし |
| BSCF-rsg | `fe3f380`, `wor-159-equilibrium-audits` | 新規 C++ modified LE / SCF 実装。remote は意図的に未設定 |
| MESA baseline | `5a6f9dde7`, `wor-151-mesa-baseline` | C++17 解析器と再現可能な profile manifest |
| HD 4wz | `e3b071b9d`, `wor-120-hd-4wz-memory2` | HD-only production baseline |
| 本監査 | `wor-150-implementation-audit` | `5a6f9dde7` から切った独立 worktree |

`BSCF-rsg/PROVENANCE.md` は clone 元、元 commit、remote を置かない理由を固定している。
legacy Fortran は参照用に保持され、新しい softened core、modified LE、piecewise EOS、
axisymmetric SCF、rotation continuation は C++ である。Python は plot / acceptance の
wrapper に限定され、production の平衡機構ではない。

## 一次文献から固定する物理

### 回転二層 SCF

[Kadam et al. (2016)](https://doi.org/10.1093/mnras/stw1814) の一様回転平衡は

\[
H_i + \Phi + \Omega^2\Psi=C_i,\qquad \Psi=-R^2/2,
\]

\[
H_i=(1+n_i)K_i\rho_i^{1/n_i}.
\]

元 BSCF は A/B/interface の格子点から `c1`, `c2`, `omega_sq` を決める。
したがって `R_eq`, axis ratio, `Omega`, `J` をすべて独立入力にはできない。
元コードの `W` は正の重力エネルギー絶対値として蓄積され、表示する
`abs(2*T-W+3*P)/abs(W)` は、符号付き `W_signed<0` を使う通常の
`abs(2*T+W_signed+3*Pi)/abs(W_signed)` と同値である。

### softened core と modified Lane--Emden

[Ohlmann et al. (2017)](https://arxiv.org/abs/1612.00008) に従い、切り出した core は
compact cubic-spline potential とし、gas self-gravity は別に解く。`u=r/h` に対して
`h` は Newtonian recovery radius そのものであり、`r>=h` で厳密に点質量へ戻る。
`BSCF-rsg/core_softening.hpp` は potential、inward acceleration、effective density、
enclosed effective mass を同じ kernel から定義し、core を gas Poisson source に
二重計上しない。

球対称機械平衡は

\[
\frac{dP}{dr}=-\rho\left(\frac{G M_{\rm gas}(<r)}{r^2}+g_{\rm core}(r)\right),
\qquad
\frac{dM_{\rm gas}}{dr}=4\pi r^2\rho
\]

である。原点で gas density は有限、加速度は 0 となる。

### RSG の層境界

[Hirai & Mandel (2022)](https://arxiv.org/abs/2209.05328) が示すように、He core と
convective envelope の間の radiative layer は薄い質量でも強く束縛され得る。
WOR-151 では formal He-core boundary をそのまま cut にせず、nuclear shell の外側で
熱的に安定かつ光度が photosphere に整合する operational cut を採用した。

採用した MESA r24.08.1 baseline は 21 Msun, 895.489 Rsun, 3563.88 K,
116548.54 Lsun であり、`M_c=5.7032361251 Msun`, `M_b=6.2966365079 Msun`、
残す radiative layer は 0.5934003828 Msun である。`L_cut=116315.98 Lsun`。
saved model と 1,625-zone profile の SHA-256 は
`analysis/wor151/source_manifest.json` に保存した。

## 元 BSCF の現物監査

| 場所 | 実装 | 判定 |
|---|---|---|
| `main.F90` | A/B/interface から Bernoulli constants と `omega_sq` を更新 | 原論文の基準として再利用可 |
| `main.F90:90-91` | `d_c1`, `d_c2`, `d_omega_sq` のみ終了条件。`VC` はコメントアウト | virial は外部 acceptance gate が必要 |
| `potential_solver.F90`, `helmadi.F90` | isolated boundary と ADI Poisson iteration | regression reference。production C++ へ直接移植しない |
| `virial.F90`, `getinfo.F90` | `T`, `W`, `3Pi`, `VC`, 構造量 | 診断式の参照として可 |
| `runhydro_template.h` | 120x120 default case | `runhydro.h` がない clean checkout は build 不可 |
| makefile / README | executable は `bin/bscf` | README の `./bscf` と不一致 |

2026-11-20 に元 repo の一時 local clone を作り、そこだけで
`runhydro_template.h -> runhydro.h` として MESA SDK gfortran
`-O3 -fdefault-real-8` の build と default smoke を再実行した。元 repo には書き込んで
いない。build は 3.38 s、default case は 17 iterations、wall 103.48 s、CPU time
1.7183 min、exit 0 となった。最終値は `d_c1=8.0540e-6`, `d_c2=8.7857e-6`,
`d_omega_sq=8.6418e-4`, `VC=1.51468e-4` である。

受入上の注意は、`eps=1e-3` が反復変数にだけ作用し、virial を停止条件にしない点で
ある。default case の小さい `VC` はこの case の測定結果であり、一般保証ではない。

## BSCF-rsg 実装監査

| issue / commit | C++ 機構 | 数値判定 |
|---|---|---|
| WOR-153 / `ceddc9d` | compact spline `Phi`, `g`, effective density/mass | support 外 Newtonian、原点有限、通常/sanitizer test 通過 |
| WOR-154 / `89a5ebd` | 単層 modified LE | 機械平衡は成立、MESA 構造には不適合 |
| WOR-155 / `e40073e` | continuous piecewise EOS と `P(H)`, `rho(H)` | enthalpy round-trip `1.6e-15`、pressure `2.44e-15` |
| WOR-156 / `f7a12e4` | 二層球対称 fitting | residual `2.65e-10`, virial `1.85e-5`; 固定指数 fit は棄却 |
| WOR-157 / `0b5d9b6` | external core + gas-only Poisson の axisymmetric SCF | 20 iter、mass error 0、球対称 profile と一致 |
| WOR-158 / `f887bed`, `c1eb395` | uniform-rotation continuation | 25 models、最終 `Omega/Omega_K=0.8635`、軸比 0.709 |
| WOR-159 / `fe3f380` | virial / force / Poisson / angular momentum audit | 2048 radial の virial `8.59e-6`; zero-core limit 正常 |

WOR-156 の固定指数二層解は、表面質量誤差 `1.85e-10`、interface 質量誤差
`1.91e-10` まで数値的に閉じる。一方、MESA に対して interface radius ratio 13.63、
density ratio 0.00356、pressure ratio 0.000412 である。この差は解像度不足ではなく、
二つの固定 polytropic index では放射圧を含む MESA の barotrope を表せないことを示す。
softening radius だけを変えてもこの構造差は解消しない。

## Athena++ / FLD / 高次精度の境界

| 機能 | 現状 | 本系列への適用判定 |
|---|---|---|
| MESA profile reader | `nrfld_conv_star.cpp` が radial profile と opacity を読む | schema adapter と checksum gate が必要 |
| fixed radial gravity | discrete-HSE 補正を含む | SCF potential / density mapping と同じ重力規約へ統一が必要 |
| central luminosity | volumetric source を持つ | `L_cut` を inner source とし、人工 preparation heating と分離する |
| Rosseland opacity | profile または table から供給可 | WOR-151 の `kappa_R` を利用可 |
| Planck opacity | table または constant 入力は可能 | MESA profile には存在しないため、別の物理 table が必要 |
| FLD | NR/MG-FLD と FGMRES path は存在 | MESA-fit equilibrium と未結合。HD 判定から承認を外挿しない |
| HD 4wz | RK4 + WENO-Z、ghost=4、HD acceptance 済み | relaxation の第一候補 |
| 4wz + FLD | 明示的に未承認 | FLD source / implicit solve を含む独立 convergence が必要 |
| passive scalars | accepted branch は conservative PLM | 4次 scalar を必要とするなら別 issue |

`nrfld_conv_star.cpp` の companion spline は support convention が中心 core と異なる。
両方を単に `softening` と呼んで数値を共有せず、kernel 名、support radius、Newtonian
recovery radius を profile manifest に必須とする。

## canonical 単位・符号・保存量

- profile table は cgs を正本とし、mass/radius/luminosity に限り Msun/Rsun/Lsun を併記する。
- potential は 0 at infinity、負値。保存する `g` は inward magnitude を正とし、接続時に
  `g=dPhi/dr` または signed vector の明示テストを行う。
- Poisson source は gas density のみ。analytic softened core potential は Bernoulli potential
  に一度だけ加える。
- pressure は total pressure とし、`P_rad/P_total`, `Gamma1`, composition を別列にする。
- `H(P)=integral dP/rho` は全層で連続にし、piecewise/table branch の積分定数を保存する。
- `kappa_R` と `kappa_P` を別列にする。欠損 `kappa_P` は NaN/null であり、暗黙に
  `kappa_P=kappa_R` としない。
- core energy、gas self-gravity、internal/radiation/recombination energy は分離して保存する。
- softening metadata は kernel、support radius、Newtonian recovery radius、core mass を含む。

## 受入テスト計画

1. EOS: `rho -> P -> H -> rho` round-trip、単調性、層境界連続性、table 範囲外拒否。
2. 球対称: HSE residual、surface/interface mass、energy/virial、softening 解像度収束。
3. SCF: gas-only Poisson、core 一回加算、zero-rotation / zero-core limit、解像度収束。
4. mapping: cgs manifest と checksum、保形質量積分、Athena++ 重力との discrete balance。
5. HD relaxation: まず accepted RK4+4wz で Mach、mass、energy、COM、virial を gate。
6. FLD: `L_cut`, `kappa_R`, 独立 `kappa_P`、diffusion/free-streaming limit、FGMRES residual、
   luminosity drift を HD から独立に gate。
7. production: restart、MPI decomposition、MeshBlock layout、解像度、table checksum を固定。

## 変更予定ファイルと issue 境界

次に変更すべき場所は元 BSCF ではなく `BSCF-rsg` である。

1. `piecewise_polytrope.hpp` の interface を一般化し、MESA-fit barotrope/table を同じ
   `P(rho)`, `rho(H)`, `H(P)` contract で供給する C++ class を追加する。
2. `piecewise_modified_le.cpp` と `axisymmetric_scf.cpp` は EOS interface だけを通じて使い、
   solver 本体の物理を二重実装しない。
3. MESA profile の `M_c`, `M_b`, pressure/density/enthalpy を C++ fitting input へ変換し、
   fit config と SHA-256 を結果に埋め込む。
4. spherical fit が MESA 構造・binding energy gate を通ってから rotation continuation を
   再実行する。
5. その後に Athena++ mapper/pgen と結合し、HD relaxation、最後に FLD を進める。

WOR-120 は FLD 統合を含むため継続、WOR-151 と WOR-153--159 は各 scope で完了である。
WOR-150 は本監査で完了し、次の実装 issue は「MESA-fit 一般化 EOS と球対称再 fit」を
独立 scope にするのがよい。既存 WOR-152 の共通 diagnostics/schema はこの実装と並行して
整理できるが、EOS の物理選定を diagnostics 作業で隠さない。

## 未解決事項

- production 用に校正した MESA evolution model は未選定。現在は再現可能な構造 baseline。
- MESA profile は `kappa_P` を持たず、Planck mean の供給元が未決定。
- 一般化 EOS を piecewise-polytropic fit と monotone table のどちらにするか。第一候補は、
  近似誤差を明示できる monotone enthalpy table とする。
- core support radius は MESA cut の 0.7369 Rsun を初期基準にするが、Athena++ の最小格子幅
  に対して十分な cells を持つ production 値は mapping 後に再決定する。
- SCF は barotropic equilibrium であり、`L(r)` を運ぶ熱平衡、対流駆動、FLD 平衡を
  自動的には保証しない。
- FLD 光球付近の closure 誤差と 4wz + FLD の安定性は独立検証が必要。
