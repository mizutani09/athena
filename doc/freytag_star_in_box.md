# Freytag 系 star-in-a-box の調査と Athena++ / NR-FLD 再現計画

調査日: 2026-10-19 (JST)。ユーザー指定により **RSG を優先**する。

**現時点の成果は、再現計画、commissioning 用 pgen、および表 EOS・外部 HSE
profile・実 opacity table を必須とする本番用 pgen である。**
物理データの確定前の短時間計算を、論文の対流・脈動の定量再現とは呼ばない。
既存の開発中 NR/MG-FLD をそのまま使う。ソルバー本体、既存 RSG/CE pgen、
親ディレクトリのビルド設定は変更しない。
実測の成功条件・失敗条件は [検証記録](freytag_star_in_box_validation.md) に記載した。

## 調査範囲と文献の位置づけ

著者公開ページ、arXiv、出版社の論文を起点とし、Freytag の第一著者論文に加えて
共著の RSG / AGB 大域モデル、観測との比較まで追跡した。局所表面対流の
**box-in-a-star** は大域計算と区別した。全出版物の網羅的書誌ではない。
下表の「本文」は数値設定を本文で確認、「要旨」は主な役割を要旨から確認したもの。
検索結果の日付を出版年として使わず、原著の書誌情報を使った。

| 文献（一次資料） | 確認 | 再現への意味 |
|---|---|---|
| [Freytag, Steffen & Dorch 2002, *Spots on the surface of Betelgeuse*, AN 323, 213](https://doi.org/10.1002/1521-3994(200208)323:3/4%3C213::AID-ASNA213%3E3.0.CO;2-H) | 要旨・本文抜粋 | RSG の大域対流、人工コア、巨大セルの出発点。現代の高解像度モデルの代用にはしない。 |
| [Freytag & Höfner 2008, *Three-dimensional simulations of the atmosphere of an AGB star*, A&A 483, 571](https://doi.org/10.1051/0004-6361:20078096) ([著者公開原稿](https://www.astro.uu.se/~bf/publications/aaagb1_submitted.pdf)) | 本文（著者原稿） | 人工コア、灰色輸送、対流と大気衝撃波、ダストの非球対称性。初期 AGB 3D 計算と 1D wind 計算を混同しない。3D 側は放射圧を含まない。 |
| [Chiavassa, Plez, Josselin & Freytag 2009, RSG I, A&A 506, 1351](https://arxiv.org/abs/0907.1860) | 要旨 | 3D 構造から干渉計観測量への変換。FLD の Er を強度画像と見なさない。 |
| [Chiavassa et al. 2010, RSG II, A&A 515, A12](https://arxiv.org/abs/1003.1407) | 要旨 | Betelgeuse の可視〜近赤外干渉計との比較。セルの大きさと視線依存性を診断対象にする。 |
| [Chiavassa et al. 2011, RSG III, A&A 528, A120](https://arxiv.org/abs/1012.5234) | 要旨 | 光度・光中心の変動。重心と光中心は異なる診断量。 |
| [Chiavassa, Freytag, Masseron & Plez 2011, RSG IV, A&A 535, A22](https://arxiv.org/abs/1109.3619) | 本文 §2–3, Table 1 | **最初の RSG 基準モデル**。灰色と非灰色、解像度の影響を切り分ける。 |
| [Freytag et al. 2012, *Simulations of stellar convection with CO5BOLD*, JCP 231, 919](https://arxiv.org/abs/1110.6844) | 本文、とくに §3.2.2 / Eq. 41 | 重力・コア・境界の実装根拠。EOS、非局所輸送、重力エネルギーの扱いを確認。 |
| [Freytag 2013, *Advances in the hydrodynamics solver of CO5BOLD*, Mem. SAIt Suppl. 24, 26](https://ui.adsabs.harvard.edu/abs/2013MSAIS..24...26F/abstract) | 書誌確認（2017/2024 本文の引用） | 数値流束の版の違いを再現条件に記録する。本文未取得の係数を推測して移植しない。 |
| [Freytag & Chiavassa 2013, *Global radiation-hydrodynamics simulations of red supergiant stars*](https://www.cambridge.org/core/journals/european-astronomical-society-publications-series/article/global-radiationhydrodynamics-simulations-of-red-supergiant-stars/708486C49832242527E80CC1105BBFCA) | 要旨 | RSG 大域モデルの位置づけを補う会議論文。 |
| [Freytag, Liljegren & Höfner 2017, A&A 600, A137](https://arxiv.org/abs/1702.05433) | 本文 | 自励脈動、質量流束による球殻平均、長時間統計、回転・箱サイズ・解像度依存。AGB の比較用であり RSG のパラメータには流用しない。 |
| [Liljegren, Höfner, Freytag & Bladh 2018, *Atmospheres and wind properties of non-spherical AGB stars*](https://doi.org/10.1051/0004-6361/201833203) | 本文 §2, §4 | CO5BOLD と DARWIN の接続。大気の浮上だけでは持続的質量放出を証明しない。 |
| [Höfner & Freytag 2019, A&A 623, A158](https://arxiv.org/abs/1902.04074) | 要旨 | M 型 AGB の clumpy dust。温度だけでなく密度履歴・粒成長が必要。 |
| [Höfner & Freytag 2022, *Explaining the winds of AGB stars: Recent progress*](https://arxiv.org/abs/2204.09728) | 要旨 | star-and-wind-in-a-box への拡張を概観。今回の RSG 実装とは別段階。 |
| [Freytag & Höfner 2023, A&A 669, A155](https://arxiv.org/abs/2301.11836) | 本文・要旨 | 非灰色ガス輸送、時間依存 silicate dust、wind acceleration zone。灰色 FLD と定数 κ だけではこの wind を再現できない。 |
| [Ahmad, Freytag & Höfner 2023, A&A 669, A49](https://arxiv.org/abs/2211.07682) | 要旨 | AGB/RSG の自励脈動の診断基準。半径の定義を明示し、平均密度と周期を比較する。 |
| [Ma et al. 2024, ApJL 962, L36](https://arxiv.org/abs/2311.16885) | 要旨 | 非回転 RSG の大域対流が観測で回転に見える場合。双極速度場だけで回転を主張しない。 |
| [Freytag, Höfner, Aringer & Chiavassa 2024, A&A 692, A223](https://arxiv.org/abs/2411.17561) | 本文 §2, Table 1、要旨 | 冷たい分子ガスによる減光、衝撃波・plume・rebound。後期の RSG 目標。 |
| [Ahmad, Freytag & Höfner 2025, A&A 699, A148](https://arxiv.org/abs/2502.11978) | 要旨 | 多重モード・mode switching。単一の短い FFT ピークでは不十分。 |
| [Jadlovský et al. 2026, A&A 711, A259](https://arxiv.org/abs/2604.21621) | 要旨（v2） | R Car / VX Sgr の時系列干渉計と CO5BOLD 比較。photosphere と分子層の半径・位相差を区別する。 |

補助比較として、Freytag 共著ではない
[Goldberg, Jiang & Bildsten 2022, ApJ 929, 156](https://arxiv.org/abs/2110.03261)
も確認した。Athena++ の RSG 放射流体計算だが、球座標の envelope 計算であり、
今回の Cartesian star-in-a-box / FLD と同一の計算ではない。

## 最初の比較対象

2011 RSG IV Table 1 の **st35gm03n07** を主対象とする。

| 量 | 論文の値 | 今回の扱い |
|---|---:|---|
| 重力質量 Mpot | 12 M☉ | 固定外場のパラメータ |
| 計算ガス質量 Menv | 3 M☉ | 全立方体の離散積分を規格化 |
| 平均放出光度 | 91,932 ± 1,400 L☉ | 本番入力は 91,932 L☉。commissioning 入力だけ 90,000 L☉ |
| 平均 Teff | 3,487 ± 12 K | 初期温度の目安。進化後に固定しない |
| 平均半径 | 830 ± 2 R☉ | 初期参照半径。進化後の測定値ではない |
| grid | 235³、8.6 R☉/cell | 文献値は保持。NR-MG実行は128³ commissioning、256³ baseline、512³ convergence |
| relaxed time | 5 yr | 初期緩和に加えて統計期間を確保 |

Table 1 の微小な誤差を「異なる物理近似を使う Athena++ が到達すべき許容誤差」
とは解釈しない。非灰色対照の st35gm03n13 は別モデルとして比較する。
初期 n=1.5 のポリトロープは論文から復元した 1D profile ではない。

## 物理・数値対応と差分

Freytag et al. (2012), Eq. 41 の固定外場を実装した:

\[
 A(r)=r_0^4+\frac{r^4}{\sqrt{1+(r/r_1)^8}},\qquad
 \Phi=-GM_{\rm pot} A^{-1/4},\qquad
 g=\frac{GM_{\rm pot}r^3}{[1+(r/r_1)^8]^{3/2} A^{5/4}}.
\]

今回は r0=0.2 R、r1=1.2 R。中心では g→0、外層ではポテンシャルが平坦化する。
自己重力を追加しない（12 M☉ と 3 M☉ を足して重力質量にしない）。
外層平坦化は wind の escape 判定には不適切なので、将来 wind を扱う際には再設計する。

| 要素 | 新しい baseline | 定量的再現に向けた差分 |
|---|---|---|
| 流体 | 3D Cartesian、RK2、既存 hllc_fld | CO5BOLD の Roe/CTU と数値散逸が異なる。低 Mach 流束の対照実験も行う |
| 重力 | 上式の解析微分、外場 | 初期状態に合わせて重力を補正しない。離散 HSE 誤差を測る |
| 加熱 | Q=L W/Σ(W ΔV), W=(1-r²/rheat²)²、コア内 | 正確な総加熱。W は今回の選択であり論文の未公開プロファイルを復元したものではない |
| コア drag | 運動量を指数減衰、除去した運動エネルギーを記録 | 密度・熱・放射を初期値に固定しない。係数感度を測定 |
| 放射 | 開発中 NR/MG-FLD、既存 LP limiter、物理的 c | CO5BOLD の short characteristics とは光球付近で異なる。灰色同士でも同一ではない |
| 放射力 | baseline では off | 2011 の流体方程式を出発点とする。on には放射支持と整合する初期構造が必要 |
| EOS | NATA X=0.70, Y=0.28, Z=0.02 の6-field表を接続試験に使用 | 現NATA出力は光子のP/U/Sを含みFLDと二重計上する。物理runにはmaterial-only出力が必須 |
| 不透明度 | 指定OPTAB表、128×128、log T=3.0..6.3、log ρ=-23.30..-1.19 | 数値範囲は初期RSGを覆う。組成・原データ・利用条件をmanifestへ補完する |
| 流体外境界 | 6 面 open、比内部エネルギーを保ち密度をポテンシャル差で外挿 | スケール高さ係数の感度を調べる。流入を禁止しない |
| 放射外境界 | ghost cell 中心の一定正値、T_bath=1000 K | 外部 bath の Dirichlet 条件。厳密な free streaming でも no-incoming intensity でもない |
| Newton/MG 外境界 | δEr の ghost 値を 0 にする専用 callback | built-in zerofixed は face に対する反対称条件なのでこの Dirichlet 条件と混同しない |
| ダスト・回転・磁場 | 未導入 | RSG の非回転対流 baseline を確立後に個別追加 |

FLD で τ≲1 の角度情報や複数光線の交差は復元できない。光球の冷却、対流の駆動、
衝撃波後の温度に影響するため、単なる表示方法の違いではない。
FLD を保持して比較する場合も、同一 snapshot の formal transfer による
光度・スペクトル・像の後処理、または別の transfer 計算との熱交換率比較が必要。

## 実装したファイルと初期条件

- `src/pgen/nrfld_star_in_box.cpp`: commissioning と共有する実装本体。
- `src/pgen/rsg_star_in_box.cpp`: 表 EOS・HSE profile・opacity table・clamp 禁止を強制する本番 entry point。
- `inputs/radiation/athinput.nrfld_star_in_box`: 32³ の短時間試験用。
- `inputs/rsg/athinput.st35gm03n07_fld*`: NR-MG互換の128³ / 256³ / 512³系列。117/235/470は文献値・拒否試験として保持。
- `tools/rsg/make_hse_model.py`, `verify_hse_model.py`: 1D HSE 生成、Cartesian 質量 feedback、独立検査。
- `scripts/*_rsg.sh`, `repro/`, `data/*/MANIFEST`: build/run/restart と provenance の固定手順。
- `tst/nrfld_validation/star_in_box.py`: ソースを隔離コピーしてビルド・検証。
- `utils/diagnose_star_in_box.py`: HDF5 の球殻平均、質量・加熱・Mach、convective luminosity、図。

内部の初期 gas HSE は、Rg=kB/(μ mu_atomic) として
T(r)=Ts+[Φ(R)-Φ(r)]/[(n+1)Rg]、ρ(r)=ρs[T(r)/Ts]^n、P=ρ Rg T。
外側は Ts の等温 HSE。ρs は **箱の全セル（角を含む）** の積分で Mgas に合わせる。
内部 Er=aT⁴、外側 Er=aTs⁴(R/r)² とする。この初期構造は **放射平衡ではない**。
初期冷却・膨張を対流の成熟や脈動と解釈しない。
初期摂動は座標だけから決め、MeshBlock ID や MPI rank を使わない。

加熱の離散規格化が壊れる AMR/SMR、非等間隔格子、放射力 on、general EOS、MHD は
この版では明示的に拒否する。これらはソルバー全体の制限ではなく新 pgen の制限。
定数 opacity を既存の平衡逆算 opacity で置き換えて「物理 opacity」と呼ぶことはしない。
UserOpacityTable 利用時は実テーブルを指定し、`opacity_table_domain_policy=error` を維持する。
本実装では当該物理テーブルを取得・同梱していない。
`fld/opacity_table_file` にファイルを指定し、HDF5 の `axes/log10_temperature`,
`axes/log10_density`, `kappa/log10_planck`, `kappa/log10_rosseland` という
既存 reader の形式を使用できる。テスト内の定数 table は接続試験専用。

入力の単位は ρu=10⁻⁸ g/cm³、Eu=10⁵ erg/cm³、lu=830 R☉。
vu=√(Eu/ρu)、tu=lu/vu ≈ 1.826×10⁷ s ≈ 0.579 yr。
`core_drag_timescale` と `time/tlim` は code time。
添付入力は CFL=10⁻⁴、tlim=10⁻⁴（約 30 分）の動作試験。
CFL=0.2 の初回試行では Newton が収束せず、初期放射緩和の改善が必要と分かった。
UOV の Tgas/Trad は K、それ以外は code units。history は全て code units。
Edrag_removed は restart に保存される累積散逸エネルギー（熱には戻さない）。
Epotential のゼロ点は上の Φ に従い、無限遠で 0 とはしていない。

## 実行

既存の実行ファイルを保持する推奨手順:

```bash
python3 tst/nrfld_validation/star_in_box.py --max-cores 2
```

`work_runs/star_in_box_*/` 以下に build_source、各ケースの input.used、ログ、
HDF5、restart、summary.json（ソース/実行ファイルのハッシュ、コマンド、検証値）を保存する。
テストは順番に実行し、ビルド・MPI とも上記コア数を上限とする。
既に作った対応バイナリを使う場合:

```bash
python3 tst/nrfld_validation/star_in_box.py \
  --binary /absolute/path/to/build_source/bin/athena --max-cores 2
```

単独の 32³ 短時間試験（binary は当 pgen でビルドしたものを指定）:

```bash
mkdir -p /absolute/path/to/rsg_run
mpiexec -n 2 /absolute/path/to/build_source/bin/athena \
  -i /absolute/path/to/athena/inputs/radiation/athinput.nrfld_star_in_box \
  -d /absolute/path/to/rsg_run
python3 utils/diagnose_star_in_box.py /absolute/path/to/rsg_run \
  --input inputs/radiation/athinput.nrfld_star_in_box \
  --output /absolute/path/to/rsg_run/diagnostic --plot
```

実行時にパラメータを変更した場合、診断の `--input` にも同じ変更を記録した入力を渡す。
診断の convective enthalpy flux は
⟨ρ vr h⟩−⟨ρ vr⟩⟨ρ h⟩/⟨ρ⟩、乱流圧は
⟨ρ vr²⟩−⟨ρ⟩(⟨ρ vr⟩/⟨ρ⟩)²。全球殻が箱内に入る半径まで計算する。
空のビンは NaN。mass は角も含む。FLD の保存的な face flux や観測強度はこの診断から推定しない。

## 再現までの工程と判定基準

以下の許容値は本プロジェクトの提案であり、原著の保証値ではない。

| 段階 | 作業 | 次へ進む条件 |
|---|---|---|
| 0: 接続・数値検証 | 今回の理想気体 pgen、質量・加熱規格化、MPI/restart、HSE 対照 | 有限・正値、NR/MG warning なし、分割と restart の整合、HSE 誤差の解像度依存を確認 |
| 1: 物理データ | H/He/H2 EOS と OPAL+PHOENIX 相当 opacity のデータ・組成・単位・利用条件を確定。既存 general EOS / UserOpacityTable を再利用 | EOS 熱力学微分と κP/κR の単体確認、必要な (ρ,T) 域を全て覆い範囲外アクセスなし |
| 2: 初期構造 | `make_hse_model.py --grid 128` で表 EOS HSE を生成し、専用 reader で写像 | ρ,P,Er の正値、LTE、写像質量 0.5% 以内。無駆動で Mach_rms<0.01 を目標に少なくとも 1 dynamical time |
| 3: RSG 緩和 | 128³ commissioning、256³ baseline、512³ convergence。core drag を初期に taper、箱・浴温・r0/r1・流束の感度比較 | R、E、平均放出光度が secular drift せず、複数 turnover time で統計安定。加熱・流出・放射・drag・floor の収支誤差を入力 L の 1% 未満へ |
| 4: 論文との比較 | gas/entropy profile、乱流圧、Mach、対流セルの低次球面調和スペクトル、衝撃波の半径時系列、脈動 | 同じ半径定義、時間窓、重みで比較。256→512 と境界変更の平均量差を評価。個々の chaotic pattern 一致は要求しない |
| 5: 観測量 | snapshot の formal transfer、分子 opacity、視線・波長・位相を変えた像と visibility/closure phase | FLD の場をそのまま表面強度に置換せず、RSG I–IV、Ma 2024、Jadlovský 2026 と比較 |
| 6: 発展 | 2024 減光シナリオ、回転/磁場、AGB dust/wind | 個々に物理と領域を追加し対照計算。灰色 FLD だけで分子バンドや dust wind を定量再現したとは主張しない |

本番コストは 128³ で計測した cell-updates/s、implicit iteration 数、メモリから見積もる。
原著の235³（47³ MeshBlock）はmesh-only検査までは通るが、NR-MG初期化は
MeshBlock一辺が2の冪でないため拒否する。実行系列128/256/512のΔxはそれぞれ
15.79/7.895/3.947 R☉で、文献117/235/470より約8%細かい。
現入力の 32³ はコア直径約 5 cells の動作試験であり、対流解析には不足。
3D snapshot の保存は pilot でサイズを測定し、histories / shell averages を高頻度、
full 3D を低頻度に分ける。NR/MG 許容誤差を緩める前に厳密解との対照を取る。

自励脈動は数 cycle の初期過渡から判定しない。球殻の ⟨ρvr⟩/⟨ρ⟩ を時系列化し、
長期 drift を除き、少なくとも 10 周期程度と複数時間窓で周波数・位相の持続性を確認する。
時間刻みが不等間隔なら再サンプルまたは Lomb–Scargle を使う。photospheric radius は
τ=2/3、entropy minimum、L=4πR²σT⁴ を混同せず、比較先に合わせて併記する。

まだ必要な外部入力は、物理 EOS/opacity の実テーブルと、長時間計算用 HPC の資源条件。
論文の 3D snapshot / 初期 profile / 正確なコア係数を入手できれば比較精度が上がるが、
論文に書かれていない値を再現済みとは扱わない。現段階の成果と残作業をこの境界で管理する。
