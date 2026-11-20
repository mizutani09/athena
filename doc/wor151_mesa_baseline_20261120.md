# WOR-151 MESA RSG 基準モデルと質量座標

更新日: 2026-11-20 (JST)

## 結論

modified Lane–Emden + BSCF の物理基準には、MESA r24.08.1 test suite `gyre_in_mesa_rsg/rsg_21M.mod` を20 steps再開した、現在質量21 Msun、Z=0.02のRSGを採用する。

| 量 | 採用値 |
|---|---:|
| 現在質量 | 21.0000 Msun |
| photospheric radius | 895.489 Rsun |
| effective temperature | 3563.88 K |
| photospheric luminosity | 116548.54 Lsun |
| MESA formal He-core mass | 5.369679 Msun |
| operational core cut `M_c` | 5.703236 Msun |
| radiative–convective boundary `M_b` | 6.296637 Msun |
| retained radiative-layer mass | 0.593400 Msun |
| outer convective-layer mass | 14.703363 Msun |

`M_c < M_b` を満たし、core excision後にも0.5934 Msunの放射層を明示的に残す。WOR-153/154へ渡す中心値は `M_c=5.703236 Msun`、感度帯は `5.60–5.91 Msun` とする。

## 候補棚卸し

C++ inventory modeで、手元の `binary_school`、`case_0255` と採用profileの計118 profilesを同一基準で走査した。RSG候補条件は `M>=8 Msun`、`Teff<=5000 K`、`R>=100 Rsun`、`L>=1000 Lsun`。条件を満たしたのは採用profileだけだった。

代表的な不採用候補:

| profile | 状態 | 判定 |
|---|---:|---|
| `binary_school/.../mesa_profile_for_tests/profile10.data` | 15.0 Msun, 26252 K, 9.95 Rsun | hot star |
| `binary_school/.../part1/profile34.data` | 20.739 Msun, 23793 K, 20.21 Rsun | hot star |
| `case_0255` の114 profiles | 最大でもRSG候補条件を満たさない | binary/hot-star系列 |
| legacy 12/20 Msun modified-LE products | raw MESA profileまたは一貫したprovenanceがない | 再現不能 |

棚卸しの全行は `analysis/wor151/mesa_profile_inventory.csv` に保存した。これは2026-11-20時点のローカル資産snapshotであり、将来追加されるproduction MESA系列を排除するものではない。

## provenance と再生成

採用したMESA saved modelのSHA-256は

`3396de468c09a788017d6ec16708e90d73c4fbd8f4afd9ada324a1e95cedeced`

20 steps再開後の `LOGS/profile1.data` は1,625 zonesで、SHA-256は

`b2f40ff36c3cd54bdd1fb19d4d6d4187db65db185928b3d259481170270c371c`

である。test suiteを再生成する例:

```bash
cp -a /home/kosuke/simulation/mesa/star/test_suite/gyre_in_mesa_rsg /tmp/wor151_mesa_export
cd /tmp/wor151_mesa_export
MESA_SKIP_OPTIONAL=1 ./mk
OMP_NUM_THREADS=8 MESA_SKIP_OPTIONAL=1 ./rn
```

解析・unit test・reference gate:

```bash
cd /home/kosuke/simulation/athena-wor151-mesa-baseline
tests/mesa/run_wor151_cpp.sh \
  /tmp/wor151_mesa_export/LOGS/profile1.data \
  analysis/wor151
```

解析器、inventory、cut感度、SVG生成はすべて `tools/mesa/analyze_rsg_profile.cpp` に実装した。Python/NumPy/Matplotlibには依存しない。

test inlistはmodel numberとageを再設定するため、profile headerの `initial_mass=1` と `star_age=10895 yr` は元のZAMS進化史を表さない。現在構造の基準には使えるが、校正済みの恒星進化系列とは主張しない。

## 放射–対流境界 `M_b`

`mix_type==1` の連続領域を列挙し、質量幅が最大の外側領域をconvective envelopeと定義する。内側にはconvective burning coreがあるため、最初のconvective regionを選ばない。外側領域の内端近傍で `nabla_rad-nabla_ad` の符号反転を線形補間し、

`M_b = 6.2966365079 Msun`

を得た。

境界状態は `r_b=5.09055 Rsun`、`rho_b=2.38327e-4 g cm^-3`、`P_b=1.78681e11 dyn cm^-2`、`T_b=2.51140e6 K`、`H_P=1.12512e11 cm`、`Gamma1=1.41280`、`P_rad/P_total=0.561551`、`L=116551.24 Lsun`、`kappa_R=0.404497 cm^2 g^-1` である。

## operational core cut `M_c`

dominant H-burning peakは `m=5.218653 Msun`。その外側で、次の条件を連続8 zonesで初めて満たす点をcutとした。

- `nabla_rad-nabla_ad < 0`
- `eps_nuc/eps_peak <= 1e-4`
- `|L(m)-L_phot|/L_phot <= 2e-3`
- `m < M_b`

結果は `M_c=5.7032361251 Msun`。これは組成だけで定めるformal He coreではなく、excisionと光度source配置のためのoperational boundaryである。

cut状態は `r=0.736857 Rsun`、`rho=0.756405 g cm^-3`、`P=2.46885e15 dyn cm^-2`、`T=2.45839e7 K`、`H_P=1.13323e10 cm`、`Gamma1=1.45513`、`P_rad/P_total=0.373114`、`L_cut=116315.98 Lsun`。

16通りのthreshold感度では `M_c=5.60008–5.90639 Msun`、残る放射層は `0.39025–0.69656 Msun`。全ケースで `M_c<M_b` であり、formal-core fallbackは発生しない。

## 光度と時間尺度

production inner luminosityの基準は `L_cut=L(M_c)=116315.98 Lsun`。photospheric luminosityとの差は232.56 Lsun、0.1995%。cut外の直接積分した核発熱は0.59280 Lsunだけなので、差を残存shell burningだけに帰属させない。time-dependent/gravitational residualを別項として扱う。

| 診断 | `M_c` | `M_b` |
|---|---:|---:|
| local dynamical time | 421.9 s | 7290.8 s |
| local diffusion time | 2.284e9 s | 1.221e8 s |
| exterior thermal time | 1.471e10 s | 1.122e9 s |
| exterior binding energy | 6.561e48 erg | 5.006e47 erg |
| optical depth | 3.581e9 | 5.625e7 |

## 制限とhandoff

- profileにはRosseland mean `kappa_R` はあるがPlanck mean `kappa_P` はない。両者を同一視せず、`kappa_P=null/NaN` として保存する。
- 採用profileは再現可能な構造baselineであり、mass loss、mixing、ZAMS historyを科学目的に校正したproduction系列ではない。
- `M_c` は固定物質面ではなく、後続modelや時間発展で再評価する診断境界である。
- WOR-153は同じ `M_c` とsoftening conventionで spline coreを作り、WOR-154は `M_c`、`L_cut`、`M_b` と上記状態を使ってmodified Lane–Emdenの自由度を監査する。
- piecewise EOS fittingでは `P_rad/P_total` が37–56%と無視できないため、単純な理想気体polytropeを最終物理EOSとは扱わない。
