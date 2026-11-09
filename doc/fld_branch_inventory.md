# FLD branch and worktree audit (WOR-114)

Local snapshot 2026-11-09: 41 `origin` heads, 26 local heads, 18 Athena worktrees. Remote count independently checked with `git ls-remote --heads origin` (41). Ahead/behind is **baseline-only commits / branch-only commits** against `origin/codex/wor92-firstcore-integration-20261109`; it is not a test result. Source/test count is changed paths from merge-base, and an ancestor branch may have zero branch-only paths. Remote-only branches still have local tracking objects.

## GitHub fork remote branches

| Branch | SHA | behind/ahead | source/test paths; sample source | Decision |
|---|---|---:|---|---|
| `codex/wor-103-default-pdfs` | `c2549ff8` | 111/3 | 0/4; — | 別PR |
| `codex/wor10-stage-audit` | `6936b4fb` | 109/0 | 0/0; — | 祖先・取込済 |
| `codex/wor101-amr-flux-20261106` | `210b0ab7` | 80/0 | 0/0; — | 祖先・取込済 |
| `codex/wor102-conservative-fld-20261106` | `4339929a` | 72/0 | 0/0; — | 祖先・取込済 |
| `codex/wor102-mach5-ab-20261106` | `84061d74` | 53/1 | 0/0; — | WOR105実装取込済 |
| `codex/wor102-normal-reconstruction-trial-20261106` | `c79d530f` | 73/1 | 6/2; bvals_refine.cpp, bvals_cc.cpp, fld.cpp +3 | 試作・保留 |
| `codex/wor105-castro-face-sync-20261106` | `ebf7634a` | 53/0 | 0/0; — | 祖先・取込済 |
| `codex/wor109-solar-spectral-postprocess-20261109` | `42d1b63e` | 111/103 | 2/33; mesh.cpp, simple_convection_lhllc_fld.cpp | 太陽系列・保留 |
| `codex/wor11-numerics-audit` | `3ded4d77` | 111/1 | 5/2; add_mixed_frame_source.cpp, fld.cpp, fld.hpp +2 | 個別審査 |
| `codex/wor111-solar-mhd-fld-20261109` | `c79e78a8` | 111/104 | 3/35; fld.cpp, mesh.cpp, simple_convection_lhllc_fld.cpp | 太陽系列・保留 |
| `codex/wor12-paper-plan` | `f5c94254` | 111/1 | 0/0; — | 個別審査 |
| `codex/wor20-pilot-20261107` | `961d9197` | 111/1 | 0/5; — | 個別審査 |
| `codex/wor24-centered-wor25-roundoff-20261107` | `2ecbea05` | 53/73 | 15/54; fld.cpp, mg_gravity.cpp, mg_gravity.hpp +12 | 数値方針競合・保留 |
| `codex/wor24-wor25-default-testc-20261107` | `bd29527c` | 53/78 | 16/70; fld.cpp, mg_gravity.cpp, mg_gravity.hpp +13 | 数値方針競合・保留 |
| `codex/wor25-23-24-conservative-audit-20261107` | `143b3403` | 53/3 | 3/6; multigrid.cpp, multigrid.hpp, multigrid_driver.cpp | 数値方針競合・保留 |
| `codex/wor25-23-24-followup-20261107` | `3f88c881` | 53/4 | 5/10; multigrid.cpp, multigrid.hpp, multigrid_driver.cpp +2 | 数値方針競合・保留 |
| `codex/wor25-23-24-resolution-20261107` | `2d682123` | 53/68 | 12/44; mg_gravity.cpp, mg_gravity.hpp, linearMG.cpp +9 | 数値方針競合・保留 |
| `codex/wor25-28-10-29-23-24` | `a970e643` | 110/0 | 0/0; — | 祖先・取込済 |
| `codex/wor25-compensated-x-20261107` | `54a8ddbe` | 53/74 | 15/59; fld.cpp, mg_gravity.cpp, mg_gravity.hpp +12 | 試作・保留 |
| `codex/wor25-roundoff-gate-20261107` | `89131700` | 53/75 | 16/60; fld.cpp, mg_gravity.cpp, mg_gravity.hpp +13 | 数値方針競合・保留 |
| `codex/wor67-spike-convergence` | `9419d224` | 103/0 | 0/0; — | 祖先・取込済 |
| `codex/wor67-xd-20261105` | `4b6f2b8f` | 82/0 | 0/0; — | 祖先・取込済 |
| `codex/wor70-solar-convection` | `ed2827d9` | 111/73 | 2/17; mesh.cpp, simple_convection_lhllc_fld.cpp | 太陽系列・保留 |
| `codex/wor74-entropy-calibration` | `74891e64` | 111/96 | 2/27; mesh.cpp, simple_convection_lhllc_fld.cpp | 太陽系列・保留 |
| `codex/wor75-initial-balance-20261106` | `efd439a2` | 111/97 | 2/28; mesh.cpp, simple_convection_lhllc_fld.cpp | 太陽系列・保留 |
| `codex/wor75-relaxed-3d-seed-20261106` | `99ba5528` | 111/101 | 2/31; mesh.cpp, simple_convection_lhllc_fld.cpp | 太陽系列・保留 |
| `codex/wor92-adaptive-inexact-20261108` | `96a4d442` | 4/0 | 0/0; — | 祖先・取込済 |
| `codex/wor92-domain-guard-trial-20261107` | `873a1b4c` | 10/0 | 0/0; — | 祖先・取込済 |
| `codex/wor92-firstcore-integration-20261109` | `8e0ee6aa` | 0/0 | 0/0; — | 祖先・取込済 |
| `codex/wor92-local-diagnosis-20261107` | `a7370166` | 111/1 | 0/5; — | WOR92系列・要判定 |
| `codex/wor92-sink-ledger-trial-20261107` | `6002babe` | 10/9 | 1/38; protostellar_core_fld.cpp | 試作・保留 |
| `codex/wor92-wor105-retest-20261107` | `6ac9288f` | 20/0 | 0/0; — | 祖先・取込済 |
| `codex/wor92-xd-restart-sync-20261107` | `a718ee47` | 6/0 | 0/0; — | 祖先・取込済 |
| `codex/wor92-xd-steptrace-20261108` | `2d056f34` | 7/1 | 1/0; linearMG.cpp | WOR92系列・要判定 |
| `codex/wor95-validation-20261107` | `b78d1b89` | 111/2 | 0/8; — | 検証資産のみ採用 |
| `master` | `053bc738` | 347/241 | 87/30; athena.hpp, bvals_rad.cpp, bvals_rad.hpp +84 | 個別審査 |
| `mdot_l1` | `58c98770` | 224/1 | 1/0; mdot_through_L1.cpp | 個別審査 |
| `mgrad` | `8bd0c9d5` | 271/3 | 4/0; multigrid_driver.cpp, calculate_fld_fluxes.cpp, mg_rad_fld.cpp +1 | 個別審査 |
| `nr_mgrad` | `ac86ae46` | 263/0 | 0/0; — | 祖先・取込済 |
| `nr_mgrad_dev` | `209e2239` | 111/0 | 0/0; — | 祖先・取込済 |
| `only_rad` | `95972695` | 330/0 | 0/0; — | 祖先・取込済 |

## Local-only branches

| Branch | SHA | behind/ahead | source/test paths; sample source | Decision |
|---|---|---:|---|---|
| `codex/wor114-fld-paper-integration-20261109` | `8e0ee6aa` | 0/0 | 0/0; — | 祖先・取込済 |
| `codex/wor92-background-relax-trial-20261107` | `7fd53c00` | 10/1 | 1/6; protostellar_core_fld.cpp | 試作・保留 |
| `codex/wor92-general-eos-local-root-20261107` | `edd8a636` | 26/0 | 0/0; — | 祖先・取込済 |
| `codex/wor92-local-root-clean-trial-20261107` | `9c584cc1` | 22/0 | 0/0; — | 祖先・取込済 |
| `codex/wor92-operator-audit-20261107` | `143b3403` | 53/3 | 3/6; multigrid.cpp, multigrid.hpp, multigrid_driver.cpp | WOR92系列・要判定 |

## Worktrees and uncommitted state

Uncommitted files are excluded from the integration branch. This table records names only, not file contents. `?` means untracked. No stash, reset, clean or checkout was run in these worktrees.

| Worktree directory | Branch | HEAD | Status |
|---|---|---|---|
| `athena` | `nr_mgrad_dev` | `209e2239` | `? inputs/radiation/athinput.nrfld_eos_table`<br>`? inputs/radiation/athinput.nrfld_simple_convection`<br>`? inputs/radiation/athinput.nrfld_solar_convection`<br>`? problem_parameters.txt`<br>`? scripts/plot_solar_convection_athdf.py`<br>`? src/pgen/nrfld_eos_table.cpp`<br>`? src/pgen/nrfld_simple_convection.cpp`<br>`? src/pgen/nrfld_solar_convection.cpp` |
| `athena_fld_audit` | `codex/wor67-spike-convergence` | `9419d224` | clean |
| `athena_wor103` | `codex/wor-103-default-pdfs` | `c2549ff8` | clean |
| `athena_wor105_review` | `codex/wor105-castro-face-sync-20261106` | `ebf7634a` | clean |
| `athena_wor11` | `codex/wor11-numerics-audit` | `3ded4d77` | clean |
| `athena_wor114` | `codex/wor114-fld-paper-integration-20261109` | `8e0ee6aa` | `? doc/fld_code_integration_manifest.md` |
| `athena_wor12` | `codex/wor12-paper-plan` | `f5c94254` | clean |
| `athena_wor20_pilot` | `codex/wor20-pilot-20261107` | `961d9197` | clean |
| `athena_wor70` | `codex/wor111-solar-mhd-fld-20261109` | `c79e78a8` | clean |
| `athena_wor92_background_relax_trial` | `codex/wor92-background-relax-trial-20261107` | `7fd53c00` | `M src/pgen/protostellar_core_fld.cpp` |
| `athena_wor92_diagnosis` | `codex/wor92-local-diagnosis-20261107` | `a7370166` | `M src/linear_solver/linearMG/linearMG.cpp`<br>` M src/linear_solver/linearMG/linearMG.hpp`<br>` M src/nr_multigrid/NRFLD.cpp`<br>` M src/nr_multigrid/NRFLD.hpp`<br>` M src/pgen/protostellar_core_fld.cpp` |
| `athena_wor92_domain_guard_trial` | `codex/wor92-domain-guard-trial-20261107` | `873a1b4c` | `M src/nr_multigrid/NRFLD.cpp` |
| `athena_wor92_general_eos_local_root` | `codex/wor92-general-eos-local-root-20261107` | `edd8a636` | `M src/nr_multigrid/NRFLD.cpp` |
| `athena_wor92_localroot_clean_trial` | `codex/wor92-local-root-clean-trial-20261107` | `9c584cc1` | `M src/fld/fld.cpp`<br>` M src/linear_solver/linearMG/linearMG.cpp`<br>` M src/linear_solver/linearMG/linearMG.hpp`<br>` M src/nr_multigrid/NRFLD.cpp`<br>` M src/nr_multigrid/NRFLD.hpp`<br>` M src/pgen/protostellar_core_fld.cpp` |
| `athena_wor92_operator_audit` | `codex/wor92-operator-audit-20261107` | `143b3403` | `M src/fld/fld.cpp`<br>` M src/linear_solver/linearMG/linearMG.cpp`<br>` M src/linear_solver/linearMG/linearMG.hpp`<br>` M src/nr_multigrid/NRFLD.cpp`<br>` M src/nr_multigrid/NRFLD.hpp`<br>` M src/pgen/protostellar_core_fld.cpp` |
| `athena_wor92_sink_ledger_trial` | `codex/wor92-sink-ledger-trial-20261107` | `6002babe` | clean |
| `athena_wor92_wor105_retest` | `codex/wor92-wor105-retest-20261107` | `6ac9288f` | `M src/fld/fld.cpp`<br>` M src/linear_solver/linearMG/linearMG.cpp`<br>` M src/linear_solver/linearMG/linearMG.hpp`<br>` M src/nr_multigrid/NRFLD.cpp`<br>` M src/nr_multigrid/NRFLD.hpp`<br>` M src/pgen/protostellar_core_fld.cpp` |
| `athena_wor95_validation` | `codex/wor95-validation-20261107` | `b78d1b89` | clean |

## PR scope

- `mizutani09/athena#1`: open draft, PDF figure export, base `nr_mgrad_dev`; keep separate.
- `mizutani09/athena#2`: merged to `codex/wor102-mach5-ab-20261106` at `84061d74`; the selected baseline contains its head `ebf7634a`, not the merge commit.
- `mizutani09/fld-paper#6` and `#9`: open draft manuscript PRs; `#9` is stacked on `#6` and neither is a code merge target.

Historical test summaries are branch-specific and are indexed in the main integration manifest. This inventory is a graph/file survey; a branch name or Done issue does not certify the selected source.
