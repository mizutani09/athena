# WOR-92 continuation: branches, restart, MG, CFL and conservation

**Longer-run follow-up:** Three-cycle CFL 0.05 convergence did not persist;
the Semenov table run failed at cycle 5. See the
[EOS/opacity control matrix](wor92_opacity_controls_20261107.md) for the
ten-cycle and matched-opacity comparisons.

2026-11-07 JST. This report supersedes the initial inference in
`wor92_wor105_retest_20261107.md` that the tangential reconstruction itself
was the cause of the coupled AMR failure. It remains an uncommitted
experimental source route, not a production solver fix.

## Provenance and related work

The base branch is `codex/wor92-wor105-retest-20261107`, starting at WOR-105
`ebf7634a8e877edfd58724de1e3810638a76ca7e`. The exact uncommitted
WOR-92 integration patch is at
`/home/kosuke/simulation/test/rad_nr/wor92_wor105_retest_20261107/source_integration.patch`,
SHA-256 `6e507ec3ad350a0fb6a3d1e3f06a73ebaf4998f55c214df3593fb95f766d388e`.
The isolated table/ideal binaries have SHA-256
`cf1e1520ca23334a08615439d8c616b6e88551801ac64101cd6c454f63d3d27a`
and `ff9887c40d80c1a2a64a1a77bb104992298b917dd28921ca84c37f1db89877c0`.
The source patch includes the explicit protostar conservative-prototype guard;
no source changes from this trial were committed.

`git fetch origin --prune` also found
`origin/codex/wor25-23-24-conservative-audit-20261107@143b3403324db869b9690a2b33569252fdb5cb80`.
That branch adds actual coarse-operator and MG diagnostics after WOR-105;
it does not change the production FGMRES policy. WOR-23/24/25 remain open
diagnostic issues. WOR-105's validated default covers Test C/Mach 5, not
protostellar collapse. `origin/nr_mgrad_dev@209e22398` lacks the WOR-105
change. Remote source search did not find the trial flux-form/floor options
committed elsewhere; the successful WOR-93 long run cannot be used as exact
source provenance for this table trial.

The audit branch `codex/wor92-operator-audit-20261107` is isolated at
`143b3403` plus the same uncommitted source patch. Its ideal-gas binary SHA-256
is `98267f45c34e7dc4b69c8349beb72691715f74fea789bb9a80dd19d8d49e6e42`.

## Controlled 32³ root, 8³ block AMR results

All runs used the same Tomida EoS and Semenov opacity files as the earlier
bundle, 56 level-0 and 64 level-1 leaf blocks, periodic Cartesian cubic
domain, flux-form matvec, mixed-frame and exact local gas trial unless a case
name says otherwise. The target is true relative residual ≤ `2e-8` and
FGMRES max iterations 300. `strict` disables the precision-floor acceptance.
The first table `dt` at CFL 0.3 was `31.114819909`; ideal `dt` was
`31.084769248` in code units. The 3-cycle results are short local checks,
not first-core formation.

| Case | Cycles / 3 | Linear solves | Maximum true residual | Floor accepts |
| --- | ---: | ---: | ---: | ---: |
| Ideal, tangent on, restart 10, MG1 | fails cycle 0 | 2 | 0.622 | 0 |
| Ideal, tangent off, restart 10, MG1 | 3 | 13 | 1.971e-8 | 0 |
| Ideal, tangent on, restart 40, MG1 | 3 | 13 | 1.990e-8 | 0 |
| Ideal, tangent on, restart 40, MG2 | 3 | 13 | 1.968e-8 | 0 |
| Table, tangent on, restart 40, MG1 | fails cycle 0 | 2 | 0.295 | 0 |
| Table, tangent on, restart 40, MG2 | 3 | 7 | 3.880e-8 | 1 |
| Table, same, strict | fails cycle 0 | 2 | 3.880e-8 | 0 |
| Table, restart 80, MG2, strict | fails cycle 0 | 2 | 4.345e-8 | 0 |
| Table, restart 40, MG3, strict | fails cycle 0 | 2 | 2.579e-8 | 0 |
| Table, unbatched reductions, strict | fails cycle 0 | 2 | 5.037e-8 | 0 |
| Table, mixed frame off, strict | fails cycle 0 | 1 | 3.339e-8 | 0 |
| Table, original matvec, strict | fails cycle 0 | 2 | 6.692e-8 | 0 |
| Table, CFL 0.1, restart 40, MG2, strict | fails cycle 1 | 4 | 2.261e-8 | 0 |
| Table, CFL 0.05, restart 40, MG2, strict, 2 ranks | 3 | 6 | 1.981e-8 | 0 |
| Table, same, 4 ranks | 3 | 6 | 1.981e-8 | 0 |
| Table, all level-1 leaves, restart 40, MG2, strict | fails cycle 0 | 2 | 9.119e-8 | 0 |
| Table, exact local gas solve off, restart 40, MG2, strict | fails cycle 0 | 2 | 4.272e-8 | 0 |
| Table, operator linearity checked each solve, strict | fails cycle 0 | 2 | 3.880e-8 | 0 |
| Table, max 600 iterations, restart 40, MG2, strict | fails cycle 0 | 2 | 3.896e-8 | 0 |
| Table, WOR-53 residual-minimizing prolongation, strict | fails cycle 0 | 1 | 2.301e-8 | 0 |

All completed cases also met the configured Newton criterion. CFL 0.05
had first `dt=5.185803318` and reached `t=15.402771269`; 2 and 4 ranks
agreed in the printed final mass, gas energy, maximum density, radiation
energy, and temperature. The history files are rounded, so that agreement
is at output precision only. Restart 40 rescues the ideal tangent-on case,
which refutes the earlier inference of an intrinsic tangential defect.
For table EoS at CFL 0.3, MG2 removes the large stall but one solve reaches
only `3.8796578e-8`; its local-stencil backward-error proxy was
`3.08639e-15`. This proxy omits the assembled AMR ghost contribution and
does not replace the required true-residual test. MG3 improves that one
solve to `2.5788308e-8`, still above target. The controls show that the
precision floor depends on the coupled configuration and CFL; mixed frame
alone, reductions, or flux-form alone do not explain it.
Turning off the trial exact local gas solve, with all other settings fixed,
left the first FGMRES result identical (`1.2189e-8`) and made the second
stop at `4.2721e-8` versus `3.8797e-8` with it on. The exact solve helps
slightly in this pilot but does not by itself deliver strict CFL-0.3
convergence. The on-path logged a local gas residual/scale maximum of
`3.47e-13`; the off-path reported zero exact-solve cells.
Checking additivity and homogeneity separately in both linear solves gave
`(2.966e-16, 5.011e-16)` and `(2.968e-16, 5.014e-16)` respectively; the
second solve's `3.8797e-8` true residual was unchanged. This rules out a
gross nonlinearity in the sampled frozen linear operator, but it is not a
global finite-difference Jacobian check at AMR boundaries.
Doubling the maximum FGMRES iterations from 300 to 600, with all other
settings fixed, left the second solve above target (`3.8958e-8` at 600
versus `3.8797e-8` at 300). This confirms stagnation rather than an
iteration cap reached just before convergence in this configuration.
WOR-53's existing `residual_minimizing_prolongation=true` setting was
also tried with the same CFL-0.3 table input. It stopped in the **first**
FGMRES solve at `2.30084e-8` after 300 iterations, whereas the default
preconditioner reached `1.21891e-8` in that solve. Thus that uniform-RSG
coarse-correction control is not a drop-in fix for this coupled AMR case.

## Operator and conservation diagnostics

The WOR-25 diagnostic branch reproduced the ideal restart-10 failure with
tangent on (`0.62247` second-solve true residual) and the tangent-off
one-cycle success (five strict solves). Its first root-to-octet `RA_hP`
versus `A_H` relative mismatches were `0.1599–0.1784` with tangent on and
`0.1707–0.1905` in the first corresponding tangent-off samples. Local
coarsening mismatches ranged `0.145–1.401` and `0.142–1.318` respectively.
The mismatch is real for these sampled correction modes but is not unique to
the failed path. Rediscretized MG need not equal the Galerkin operator, so
this measurement alone is not a bug diagnosis. It does justify a focused
preconditioner/error-mode investigation under WOR-25.

A separate table tangent-on, restart-40, MG2, CFL-0.3 ledger repeat
reproduced 3 cycles and the same one floor acceptance. The x1 coarse/fine
interface audit found six matched rows and maximum relative flux mismatch
`7.68874e-8`, versus `0.0848757` for the old legacy route and
`1.97747e-7` for the earlier tangent-off prototype. Rank-summed implicit
energy-ledger balances by cycle were `-3112.38`, `-1115.85`, and `+162.20`
code-energy units, at up to `3.23e-9` relative rank-level balance. The
domain interior diffusion sums were `+257.69`, `-5.83`, and `+109.08`.
The old route's first-cycle sum was `-2.78088e8`. These are substantial
improvements but not roundoff-level conservation. That older parser groups
whole x1 planes and can hide opposite face errors; the complete 3D audit
below supersedes its interface maximum.

## Complete 3D face audit with direct rank logs

To avoid dropped/interleaved MPI stdout lines, a further diagnostic-only
patch wrote one complete ledger per rank directly to files. It was built on
`143b3403` in the isolated audit worktree, with source-patch SHA-256
`0fdf2d0d770b924e9f47c34051c366fb729998cc289ff39f49a696bbde9319d8`
and binary SHA-256
`ca8ef1ca6f78f0a75f854369d0f251d47085a2598d083288b4d754d94e281447`.
The same table CFL-0.3, restart-40/MG2, floor-enabled run again completed
3 cycles. Each cycle had exactly 120 complete leaf records, 56 coarse and
64 fine; no leaf-count change occurred in this short interval.

The new parser matches each block face by its two-dimensional geometric
overlap, aggregating a coarse face with its actual fine neighbors. It
checks equal matched areas and independently reconstructs the domain
interior diffusion sum. Across the three cycles:

| Axis | Same-level matched components | Coarse/fine components | Max coarse/fine relative mismatch | Weighted coarse/fine mismatch |
| --- | ---: | ---: | ---: | ---: |
| x1 | 252 | 24 | 5.436e-7 | 1.133e-7 |
| x2 | 252 | 24 | 1.129e-6 | 1.431e-7 |
| x3 | 252 | 24 | 5.376e-7 | 1.953e-7 |

Every same-level component balanced exactly in the logged values, and the
net physical-boundary contribution was zero in each axis. The sum of all
matched internal face balances by cycle was `+257.6942945`, `-5.82857895`,
and `+109.0842696`, agreeing with the independently summed energy ledger
to within `1.5e-7` code-energy units. Thus the remaining global diffusion
imbalance is localized to coarse/fine faces in this 3-cycle pilot. The
largest local relative mismatch is `1.129e-6`, whereas the earlier x1
whole-plane value `7.69e-8` understated the worst face group because of
cancellation. The discrepancy is not roundoff, though the solve's Newton
criterion is of order `1e-5`; this check alone does not establish
production-level conservation.

The [3D audit JSON](wor92_interface_3d_results_20261107/summary.json),
[CSV](wor92_interface_3d_results_20261107/axis_summary.csv),
[figure](wor92_interface_3d_results_20261107/coarse_fine_mismatch_3d.png),
and [direct rank logs/source patch/input](wor92_interface_3d_results_20261107/rank_ledger_evidence.tar.gz)
are generated by
`python3 tst/nrfld_validation/wor92_interface_3d_audit_20261107.py`.
The representative run command above adds
`ATHENA_NRFLD_ENERGY_LEDGER=1 ATHENA_NRFLD_INTERFACE_LEDGER=1` and
`ATHENA_NRFLD_RANK_LEDGER_PREFIX=/path/to/fresh/ledger`.

## Precision and mesh-localization follow-up

An isolated diagnostic build on the WOR-25 audit branch recomputed the final
flux-form `Ax` row in long double before subtracting it from `b`; its
acceptance gate was unchanged. With the strict table CFL-0.3 input, the
second FGMRES solve remained at 300 iterations. The normal true residual
was `3.8796577724025381e-8`; the higher-precision final evaluation was
`3.8796577723996370e-8`. Rounding `Ax` to `Real` contributed only
`5.25382e-17` relative to the RHS norm. Diagnostic ON/OFF gave identical
iterations and normal residuals. Final `Ax` rounding alone therefore does
not explain the plateau. The tiny local-stencil backward-error proxy
`3.08639e-15` instead reflects a different row-scaled criterion and does
not cover an assembled AMR global row.

The failed solve's level-2 and level-3 contributions to the global true
residual were `1.25813e-8` and `3.67000e-8`; level 3 contributes about
89.5% of the squared norm. Block-edge cells account for about 65.1% of
the level-3 squared residual, leaving 34.9% in block interiors. These are
block edges, including same-level and physical boundaries, rather than a
specific coarse/fine classification.

A second controlled run changed only the meshblock from 8³ to 16³, yielding
64 level-1 leaf blocks and **no coarse/fine leaf face**, with the same
table physics, CFL 0.3, restart 40, MG2 and strict gate. It still failed
the second solve at `9.1188031e-8` after 300 iterations. Thus a coarse/fine
face is not necessary for this precision-limited failure. The smaller
8³ mixed-level case remains the better reproducer for combined AMR checks.

## Reproduce and next experiments

Full logs, exact inputs, source patch and isolated builds are under
`/home/kosuke/simulation/test/rad_nr/wor92_wor105_retest_20261107/` and
`/home/kosuke/simulation/test/rad_nr/wor92_operator_audit_20261107/`.
The companion [CSV](wor92_wor105_continuation_results_20261107/runs.csv),
[JSON](wor92_wor105_continuation_results_20261107/summary.json), and
[figure](wor92_wor105_continuation_results_20261107/true_residual_matrix.png)
and [compressed raw logs/inputs](wor92_wor105_continuation_results_20261107/logs_inputs.tar.gz)
are generated with
`python3 tst/nrfld_validation/wor92_wor105_continuation_20261107.py`.
The [precision/level JSON](wor92_wor105_continuation_results_20261107/precision_diagnostics.json)
records the diagnostic binary and source-patch hashes. The archive contains
both high-precision diagnostic patches and their ON/OFF logs.
Each record contains the exact input and log SHA-256. The prior report
contains the configure/build commands. Representative run from the bundle
directory (relative table paths resolve there):

```bash
cd /home/kosuke/simulation/test/rad_nr/wor92_20261107/bundle
env ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE=1 \
  ATHENA_NRFLD_CONSERVATIVE_SHARED_COEFF=1 \
  ATHENA_NRFLD_EXPERIMENTAL_TANGENTIAL_RECONSTRUCTION=1 \
  ATHENA_NRFLD_EXPERIMENTAL_VARIABLE_LIMITER=1 \
  ATHENA_NRFLD_EXPERIMENTAL_COUPLED=1 \
  ATHENA_NRFLD_EXPERIMENTAL_REGRID=1 \
  ATHENA_NRFLD_CONSERVATIVE_DIFFERENCE_JV=0 \
  mpirun -np 2 /path/to/isolated/table/bin/athena \
  -i /home/kosuke/simulation/test/rad_nr/wor92_wor105_retest_20261107/input_mixed_amr_restart40_mg2_strict.used \
  -d /path/to/fresh/output
```

The next useful work is to inspect the precision-limited solve using an
assembled AMR-aware componentwise backward error, while logging per-rank
boundaries without MPI stdout interleaving. Compare MG correction/error
modes at restart 10 and 40; test CFL 0.1 and 0.3 after a targeted numerical
change, then rerun the 3D audit, regrid/restart and longer Newton regression.
The Semenov/Optab opacity-definition difference remains a separate physical
model issue. WOR-92 remains **In Progress**; CFL 0.3 strict convergence and
protostellar production readiness are unproven. No XD2000 job was run.
