# WOR-120 surface refinement readiness (2026-11-14)

## Question and controlled test

The LTE/FLD RSG surface luminosity rises from 0.1 to 0.9 Lstar across only 3.35 Rsun in the current continuum profile, while a uniform 256³ cell is 7.89 Rsun wide. We checked whether a locally refined FLD mesh can be used as the next short RSG gate.

The existing `test_c_controlled.py` harness was run locally at diffusion number `q=1`, density contrast 1, one fixed-dt step, **variable** FLD limiter, and three grids: uniform fine, aligned one-level static refinement, and shifted one-level static refinement. All tests are radiation-only with fixed hydro and periodic boundaries. Two isolated source builds were used:

* Current WOR-120 branch `wor-120-rsg-hse-balance@429a5659` (legacy nonconservative coarse/fine operator).
* Existing WOR-105 candidate `codex/wor105-castro-face-sync-20261106@7e977117` (supported-default common-face conservative operator for Test C).

Both used `--layouts uniform_fine amr_aligned amr_shifted --diffusion-numbers 1 --contrasts 1 --variable-limiter --timeout 30`. The full settings, source/binary hashes, exact input and run records are in [current manifest](wor120_smr_readiness_20261114/current_manifest.json) and [candidate manifest](wor120_smr_readiness_20261114/conservative_manifest.json). The aligned-case logs are [current](wor120_smr_readiness_20261114/current_amr_aligned.log) and [candidate](wor120_smr_readiness_20261114/conservative_amr_aligned.log).
The complete generated build and field outputs were moved out of the source trees to `/home/kosuke/simulation/test/wor120_smr_readiness_20261114/{current,conservative}`; paths embedded in the saved manifests reflect their original run locations.

| Grid | WOR-120 `ΔEr/Er` | WOR-105 `ΔEr/Er` | Newton iterations, current/candidate | L1 versus uniform fine, current/candidate |
|---|---:|---:|---:|---:|
| Uniform fine | +1.09×10⁻¹⁵ | +1.09×10⁻¹⁵ | 1 / 1 | 0 / 0 |
| AMR aligned | −1.81×10⁻⁵ | +3.05×10⁻¹⁵ | 4 / 3 | 7.55×10⁻⁴ / 7.49×10⁻⁴ |
| AMR shifted | −2.42×10⁻⁶ | +3.92×10⁻¹⁵ | 3 / 3 | 1.51×10⁻⁵ / 1.52×10⁻⁵ |

All six Newton solves converged. The current branch reproduces a closed-box AMR energy defect despite this convergence. The conservative candidate removes that defect in this reduced test without a material change in the measured L1. This is a comparison between two branch snapshots, **not** a claim that a patch has already been ported or validated for RSG.

An additional **coupled** ideal-gas check reused the four uniform-fine/aligned-AMR input and binary pairs, overriding `fld/is_couple=true` and `fld/only_rad=false` for one fixed-dt step. Hydro remained frozen, while gas and radiation exchanged energy. All four Newton solves converged in 23 iterations. The periodic-box relative change of `history(tot-E + Er_volume)` was +1.13×10⁻¹⁵ (current uniform), −1.22×10⁻⁷ (current AMR), +7.53×10⁻¹⁶ (candidate uniform), and +7.53×10⁻¹⁶ (candidate AMR). [Coupled summary](wor120_smr_readiness_20261114/coupled_summary.json) includes the gas/radiation energies and binary/input hashes. This is still constant-opacity ideal-gas Test C; it does not cover table EOS, RSG boundaries, motion, or RSG opacity.

## RSG blockers and next gate

1. The current RSG input enables `fld/fourth_order_radiation_force=true`. `FLD::FLD` explicitly rejects `multilevel` with that option. The 5³ quartic force stencil and its Godunov-face pressure-balance anchor have only been validated on a uniform mesh; coarse/fine prolongation and flux correction need a separate force-balance test.
2. The WOR-105 conservative operator defaults only for Test C and the validated Mach 5 configuration. It explicitly rejects other AMR problem generators unless a legacy, nonconservative route is requested. Its boundary guards are periodic Test C or Mach 5 user-x1/periodic-transverse, whereas the RSG has six user/Marshak faces. RSG general EOS and physical opacity are also outside its recorded validation envelope.
3. The current RSG problem generator has no radial AMR tagging callback. A single Cartesian static-refinement cube would also refine much of the stellar interior; targeted surface refinement needs a new region/tagging design and a block/memory estimate before a production-sized run.

Do not start a long RSG SMR/AMR run from either branch. The next implementable short gate is a small, coupled RSG-like multi-level box that first checks a common-face radiation-energy ledger, Newton convergence, table-EOS/opacity validity, and physical-boundary behavior with second-order radiation force. Only after that passes should fourth-order force across refinement interfaces be implemented and compared on a smooth HSE/force-balance problem. Then a surface-only RSG pilot can test whether finer cells actually reduce the early cooling. No XD job was submitted for this readiness audit.
