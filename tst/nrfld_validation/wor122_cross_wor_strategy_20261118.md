# WOR-14 と三層 RSG 計画（WOR-150–173）の対応

2026-11-18 時点。ここでは「Freytag/Chiavassa 型の全星 star-in-a-box」と「MESA に基づく点源コア・放射層・対流層の CE 用モデル」を異なる科学目標として扱う。

## 判定

三層計画の中心重力、MESA 由来の質量座標、保存的 3D 写像、総圧の整合検査は WOR-14 の中心流や力学初期化を改善し得る。しかし現時点で測定された WOR-14 の *人工的な表層熱源を切った直後の失敗* は中心からの流れでは説明できず、三層化だけでは解消しない。三層計画でも表層解像度、放射境界、3D 対流流束の立ち上がり、定常 FLD の熱収支が別途必要である。

WOR-150–173 はまだ物理的に成功した RSG の実証ではない。二層 piecewise polytrope の機械的平衡と SCF→Athena++ HD 写像は試作済みだが、WOR-156 の MESA 比較は不合格であり、WOR-163–167 の FLD 熱平衡・短期 pilot・長期進化は未実証。Linear 上の一部 issue が Backlog でもローカルには先行試作があるため、状態と artifact の両方を見る。

## 観測事実

| 要素 | WOR-14 / 122 現行 | WOR-150–173 側 | 含意 |
|---|---|---|---|
| 目標星 | Chiavassa et al. 2011 の灰色モデルを模した固定ポテンシャル質量 12 M☉、計算ガス約 3 M☉、R=830 R☉、L=91,932 L☉ | WOR-151 の MESA baseline は 21 M☉、R=895.5 R☉、L=116,549 L☉、cut=5.703 M☉、放射/対流界面=6.297 M☉ | この baseline を移植しても 12 M☉ の Freytag モデル再現にはならない |
| 中心・重力 | Freytag 型の球対称平滑固定ポテンシャル、熱源半径 0.2R、中心の damping/高次重力を試験 | compact spline core + gas self-gravity、Poisson/energy と core の二重計上を監査 | 中心ドリフトの独立改善候補。ただし二つの softening は同一ではない |
| 層構造 | 合成 1D HSE/FLD profile。深部の局所 MLT 流束容量が不足 | MESA で質量境界を定義、piecewise EOS と modified LE/SCF で HD 平衡を試作 | 恒星構造の物理根拠は強まるが、輸送や対流の成立を保証しない |
| 表層・熱輸送 | 1D 輻射光度 10–90% の変化幅 3.35 R☉、256³ の Δx=7.89 R☉。3D 初期 Fconv≈0。熱補助を止めると 2.55 日で箱の Lrad/Lheat 1.042→0.831、813–817 R☉ の Tgas がさらに約 11% 低下 | WOR-163–165 で総圧、定常 FLD、放射境界、源光度を検証する計画。WOR-161 の HD 試作も表層力残差が支配的 | 三層化だけでは主症状を解決しない。有限体積 FLD と 3D 対流を含む閉じた準備が必要 |
| 本番の成立 | 256³・33 日まで計算したが、人工的殻熱源停止後の安定性は不合格。Henyey MLT の 1D 試作は遷移幅約 0.9 R☉でさらに薄い | WOR-160/161 は 1 cycle / 1 step の HD smoke test と力学残差まで。WOR-162 の HD 緩和、WOR-163–167 の RHD は未受理 | 三層計画を成功例として WOR-14 にコピーする段階ではない |

WOR-14 の初期離散 FLD 監査では 800–813 R☉ の輻射光度推定が 1D の 0.00938 L★ に対し 256³ で 0.1359 L★、同殻の厳密な離散拡散 power は −0.502 L★。初期写像と格子の不整合であり、これだけでは FLD solver のバグは示さない。t=120→130 の matched 熱源 on/off では中心 0–700 R☉ の速度がほぼ一致し、差は表層に現れた。

WOR-151 の MESA 放射層は r=0.737–5.091 R☉、幅約 4.35 R☉。この構造は全星 256³・Δx≈7.9 R☉ では一セル未満になる。WOR-156 の fixed-index piecewise fit は半径 69.37 R☉ に界面を移し、MESA 界面半径比 13.63、密度比 0.00356、圧力比 0.000412、binding energy も 1.13e48 対 MESA 6.56e48 erg。機械的残差が小さくても MESA の三層を保ったとは言えない。WOR-160/161 の細分化で core h≈5.09 R☉ は分解されたが、表層の圧力 scale height は root Δx=34.375 R☉ に対し ρ=1e-10 g/cm³ で 5.47 R☉、ρ=1e-11 で 1.16 R☉。WOR-161 の質量重み付き力残差 L2 は 0.062、主な残差は表面付近。

## Freytag との関係と最短の方針

Chiavassa et al. (2011), A&A 535 A22, §2–3/Table 1: [arXiv full text](https://arxiv.org/html/1109.3619v1)。灰色モデルは 235³、Δx=8.6 R☉、12 M☉ fixed spherical smoothed potential、3 M☉ envelope、開放境界、非局所 ray 輸送を使用する。初期球は前モデルから弱い速度を継承し、光球下の鋭い温度遷移は粗格子で明瞭な格子模様を作るが、数年をかけて対流状態へ緩和した。従って「同程度のセル幅なら 3D 静的初期条件から数十日で定常になる」は文献の主張ではない。WOR-14 はポテンシャル形状と大局的な星の量は近いが、灰色 FLD と非局所 ray 輸送の違い、EOS/opacity/境界と初期流の違いを残す。特に光学的に薄い層で FLD が ray 法と同じ構造を与える保証はない。

結果を早く得るなら次の二経路を分ける。

1. **まず成立する外層対流を示す。** Goldberg, Jiang & Bildsten (2022) は球座標の外層領域だけを取り、内側境界から光度を与えて長時間 3D RHD を得た（[原論文](https://arxiv.org/html/2110.03261v2)）。中心と MESA の薄い放射層を切り離す比較実験として、内側境界を 300–400 R☉程度に置いた外層 domain で、同じ opacity/EOS の 1D 定常 flux、離散 FLD flux、表層解像度、energy closure を先に検証する。これは全星 Freytag の厳密再現ではないが、表層問題と中心問題を分離し、FLD で結果が出せるか最短で判定できる。
2. **全星モデルは検証ゲートを通してから延ばす。** 12 M☉ 用の入力構造を作り、放射力込みの 1D HSE と定常エネルギー輸送を同時に満たし、Cartesian 写像後の gas+radiation force と FLD face flux を測る。表層に複数セルを割く局所細分化または表面処方を比較し、短時間 3D 準備で Fconv の成長と人工熱源の減少が釣り合うかを測る。熱源なし短期対照で Lout、T、radius、mass/energy drift が許容範囲に入った候補だけ年規模へ進む。失敗が FLD の光球輸送に局在するなら、灰色 ray/角度輸送との frozen-state 比較を行い、FLD の適用限界を判定する。

三層計画からは WOR-151 の物理量 schema、WOR-160 の保存的写像、WOR-161 の重力/雰囲気/境界の離散診断を再利用する。一方、21 M☉ の fit と compact-spline core を Freytag 12 M☉ と同一視せず、WOR-163–166 の熱平衡検証を先に完了させる。長時間ジョブの追加投入はこの判定に寄与しにくい。

## 参照

- `wor122_rsg_discrete_initial_fld_audit_20261115.md`
- `wor122_prep256_release_result_20261118.md`
- `wor122_henyey_mlt_20261118.md`
- `/home/kosuke/simulation/athena/doc/wor151_mesa_baseline_20261118.md`
- `/home/kosuke/simulation/mhd/BSCF-rsg/analysis/wor156/comparison_metrics.json`
- `/home/kosuke/simulation/athena-wor161/doc/wor161_scf_gravity_alignment_20261118.md`
