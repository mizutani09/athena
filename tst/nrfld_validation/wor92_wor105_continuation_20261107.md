# WOR-92 continuation: branches, restart, MG, CFL and conservation

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
improvements but not roundoff-level conservation, and the current parser
only pairs x1 faces. Full 3D and clean per-rank logging remain required.

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
assembled AMR-aware componentwise backward error and repeat the true residual
with higher-precision accumulation, while logging per-rank boundaries
without MPI stdout interleaving. Compare MG correction/error modes at
restart 10 and 40; test CFL 0.1 and 0.3 after a targeted numerical change,
then rerun full 3D conservation, regrid/restart and longer Newton regression.
The Semenov/Optab opacity-definition difference remains a separate physical
model issue. WOR-92 remains **In Progress**; CFL 0.3 strict convergence and
protostellar production readiness are unproven. No XD2000 job was run.
