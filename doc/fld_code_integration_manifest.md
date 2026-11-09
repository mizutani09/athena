# FLD paper source integration manifest (WOR-114)

Audit date: 2026-11-09. This document distinguishes a candidate source tree from a validated publication release. No paper release tag has been made.

## Phase 0 decision

Use `8e0ee6aa2261e5cb4f4b53ca94752c4caa549b80` (`codex/wor92-firstcore-integration-20261109`) as the parent of the isolated integration branch. It descends from `nr_mgrad_dev` at `209e2239` and contains the WOR-105 head `ebf7634a` plus the subsequent WOR-92 table EOS/opacity, local gas root, AMR, FGMRES, and first-core work. The PR #2 merge commit `84061d74` is not itself an ancestor; its feature head is. `master` at `053bc738` tracks PrincetonUniversity/athena and diverges by 236/241 commits from `nr_mgrad_dev`; it is not an FLD source baseline.

This choice minimizes rebasing of the in-progress collapse path. It does **not** certify the table-EOS first-core calculation or a paper source. The baseline remains under test. The alternative `codex/wor24-wor25-default-testc-20261107` is a validated Test C branch, but diverges at `ebf7634a` and lacks later WOR-92 changes.

During Draft PR preparation, the moving WOR-92 branch advanced to `3518882f` and added `54657730`, which changes Newton line-search acceptance in `src/newton_raphson/Newton_Raphson_driver.cpp`. The measured tests below use the earlier fixed parent `8e0ee6aa`. PR #3 therefore targets the new fixed review base `codex/wor114-fixed-base-8e0ee6aa` at that SHA. The newer Newton policy requires separate review and reruns before adoption.

## Adoption and overlap

| WOR / source | Relevant commit or branch | Change and existing evidence | Decision |
|---|---|---|---|
| 23, 24, 25 | `6bc87e3d`, `b0849726`; `codex/wor24-wor25-default-testc-20261107` `bd29527c` | Boundary Jv audit, centered common AMR face and bounded FGMRES stopping; Test C 1/2 rank, five steps and restart reported in WOR-24/25 | **Hold code** pending exact same-state A/B against WOR-92. The later `f9dc6793` independently edits the face and stopping code. |
| 105, 106 | `ebf7634a`, `df944aff`; PR #2 merged into `codex/wor102-mach5-ab-20261106` | Conservative shared faces in supported Test C/Mach 5; coupled AMR boundary and next-dt restart regression | **Already present by ancestry** in chosen base; no duplicate cherry-pick. Distinguish from solar restart below. |
| 92 | `f9dc6793`, `8e0ee6aa` | Table EOS/opacity collapse, guarded local root, FGMRES controls, centered-face trial, empty AthenaArray copy/move fix; first-core continuation in progress | **Baseline candidate**, not yet scientifically qualified. `ATHENA_WOR92_CENTERED_FACE_TRIAL` is an opt-in trial, whereas WOR-24 makes centered Test C default. |
| 95 | `3952b4ff`, `b78d1b89`; `codex/wor95-validation-20261107` | Diffusion/Jeans regression driver and saved results | **Apply isolated validation-only commits**, then rerun on integration SHA. Historical output is evidence for its original source, not a PASS of this branch. |
| 54 | `9c587284`; `nr_mgrad_dev` | FGMRES with MG preconditioner | **Already present** in parent; verify mixed-frame/AMR modes. |
| 61, 62 | `09422a69`, `a4f5045f`; `nr_mgrad_dev` | Mixed-frame transport and validation paths; an earlier drift comparison was reverted | **Already present**; preserve selected physical discretization, rerun mixed-frame checks. |
| 70, 74–76, 109, 111–113 | `50079d71`, `c79e78a8`; `codex/wor111-solar-mhd-fld-20261109` | Solar HD restart, boundary and entropy work; MHD magnetic-energy fix and HLLD boundary pilot. WOR-111 short pilot reports 139/139 converged through t=10, not thermal stationarity | **Hold source changes**; `src/fld/fld.cpp` and `src/mesh/mesh.cpp` overlap WOR-92. Isolate HD and MHD paths and redo restart/energy tests before adoption. Solar steady flux remains unproven. |
| 103 | PR #1 `codex/wor-103-default-pdfs` | Figure export only; open draft to `nr_mgrad_dev` | **Keep separate** to avoid duplicating another review. |
| 57–58, `mdot_l1` | separate Pre-CE branch | Mass-transfer/spherical output, not FLD paper | **Exclude** from integration. |

The WOR-24/25 branch and WOR-92 modify `src/linear_solver/linearMG/linearMG.cpp`, `src/nr_multigrid/NRFLD.cpp`, and face flags. WOR-92's `f9dc6793` adds an opt-in centered-face control and a distinct roundoff audit; WOR-24/25's `b0849726` defaults centered Test C and applies a bounded stopping gate. A clean textual merge would still change numerical policy. The rejected compensated-update and normal-reconstruction trial branches must remain historical controls.

## Verification matrix

| Gate | Configuration | Measurement / acceptance | Status on integration SHA |
|---|---|---|---|
| Build | clean GNU serial, MPI 1/2, HDF5; hydro-only control | compile and run, recorded commands/toolchain | MPI/HDF5 builds PASS; serial and hydro-only pending |
| Core FLD | diffusion, low/high coupling, streaming, Mach 2/5, closed box | Newton/FGMRES true residual, energy, reference errors, mixed-frame flux | Diffusion and low coupling PASS; other cases pending |
| AMR operator | Test C fixed/variable limiter, aligned/shifted, 2/3 levels, 1/2 ranks | coarse/fine face balance, full-boundary Jv, FGMRES true residual; compare WOR-24 default vs WOR-92 trial | Pending |
| Restart/regrid | coupled AMR 1→1, 2→2, changed rank; solar HD | first step and next dt; mass/energy and field differences | WOR-106 six cases PASS with explicit 2-ULP time bound; strict equality fails in two static cases; solar pending |
| Collapse | ideal-gas constant opacity and table EOS local/AMR, Jeans trigger | source/input/table SHA256, CFL, bounded-inexact decisions, end reason; first core is a separate physical gate | Table uniform 1-step smoke PASS with precision-floor acceptance; table AMR guarded; other gates pending |
| Solar | HD then MHD 0/weak B; serial/MPI | restart, div B, energy, boundary flux; report transient separately from steady-state | Pending |
| Upstream | PrincetonUniversity/athena current main | non-FLD build/tests, split patches and compatibility | WOR-117 |

No prior long-run or XD output is represented as a run of this integration commit. External EOS/opacity tables and existing raw data require hashes before a reproducible release. Figures affected by face/default or mixed-frame changes need source-specific review in WOR-87/88/90.

Measured test values, toolchain, binary hashes and current limitations are in `fld_reproducibility_status.md`.

## Integration sequence and release rule

1. Keep this worktree independent and commit the Phase 0 manifest before adopting code.
2. Add only isolated regression tooling whose source behavior is unchanged, and rerun it on the integration SHA.
3. Review the WOR-24/25 versus WOR-92 operator and stopping policy with the same checkpoint/input and both MPI sizes. Adopt one **explicit** production policy only after Jv, face balance, true residual, restart and accuracy pass.
4. Rebase solar HD/restart and MHD source changes by logical unit, resolving `fld.cpp`/`mesh.cpp` conflicts with physics and energy ledgers, not by choosing a whole-file side.
5. Freeze a candidate SHA with build/input/table hashes, exact solver options, known failures and figure provenance. Draft PR review precedes any tag or merge.

The upstream implementation split, non-FLD compatibility, Princeton main tracking and proposal work are handed to WOR-117; publication qualification of this fork remains WOR-114.
