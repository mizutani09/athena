# WOR-124: local monotone coarse/fine face fallback (2026-11-20)

## Result

The first candidate fix is implemented and passes the known strong-field
cycle-964 safety gate.  It detects an inadmissible converged Newton direction,
marks only coarse/fine faces on the crossing fine MeshBlocks for the monotone
two-point form, rejects the direction before changing the iterate, and
restarts the complete nonlinear solve.  State ghosts, correction ghosts,
shared interface moments, FGMRES matvec, and MG all read the same frozen face
mask.

The fix is a numerical-safety success but not yet a performance success.  The
mixed local operator converges through cycle 970 without a global fallback,
but requires substantially more FGMRES work than globally switching to the
two-point operator.  It therefore remains opt-in on the isolated branch and
is not proposed as the default.

Branch and implementation commit:

- `exp/wor124-local-face-monotone-20261120`
- `30ac4f59b` (`nrfld: retry unsafe AMR faces with local monotone stencil`)

Enable with `ATHENA_NRFLD_LOCAL_TANGENTIAL_FALLBACK=1`.  The global
`ATHENA_NRFLD_TANGENTIAL_FULL_SOLVE_RETRY` was disabled in the decisive gate.

## Implementation

Each Newton object owns a six-face fine-side mask.  Zero retains the
tangential reconstruction and one selects the two-point face form.  A
converged linear direction is checked against the same radiation positivity
condition used by the state update.  For every crossing fine MeshBlock, all
fine subfaces adjoining a real coarser neighbor are marked.  Physical
boundaries (`nblevel=-1`) are explicitly excluded.

The rejected direction is never clipped or committed.  `Solve_general()`
re-enters the full solve, so `LoadVariables()`, the nonlinear residual,
coefficients, interface moments, and MG caches are rebuilt for one consistent
operator.  Up to eight monotonic mask expansions are allowed; a crossing that
adds no face fails safely instead of applying an inconsistent correction.

## Local regression

Both optimized Test C and debug table-EOS/MHD/MPI builds completed.  The
dedicated Test C input uses N64 x N16 x N16, static AMR, a 10:1 coefficient
jump, transverse amplitude 0.8, variable limiter, and FGMRES.

One- and two-rank runs both used Newton 1 and FGMRES 8.  Their true relative
linear residuals were `8.3581537763e-11` and `8.3581537782e-11`, respectively,
against a `1e-10` target.  The history files were byte-identical (SHA-256
`9e1c19ef...d6a0ab`) and the global diffusion integral cancelled to roundoff.
No fallback fired.  The earlier candidate run also reproduced every VTK file
from the validated tangential Test C, showing that the enabled but inactive
gate does not change the safe high-order path.

Local artifacts:
`/home/kosuke/simulation/test/rad_nr/wor124_local_face_monotone_20261120`.

## Exact strong-field restart gate

XD job `343876` used five `M-test-a` nodes, 320 MPI ranks, CFL 0.15, the
immutable cycle-964 restart, table EOS, variable opacity, MHD, rotation, AMR,
and the same FGMRES tolerance (`2e-8`) as the previous diagnosis.  It advanced
six accepted steps to cycle 970.

At cycle 964 the original tangential operator reproduced the known FGMRES 628
solve.  The direction check found 287 crossing cells in 28 MeshBlocks with a
minimum admissible scale of `0.1514528157`.  It marked 1,792 fine face cells,
requested one full nonlinear restart, and never invoked the global two-point
fallback.  The restarted cycle and cycles 965--969 all converged in two Newton
corrections.  There were no further crossing events.

The 13 FGMRES solves used
`628,394,407,417,441,256,261,414,419,255,256,394,394` iterations, totaling
4,936.  Every true relative residual met `2e-8`.  The maximum final Newton L2
was `1.14568e-8`, and the maximum normalized component was `6.53339e-6`.

The c970 audit found 2,412,544 cells, maximum level 11, zero non-finite or
non-positive cells, and zero opacity-table violations.  Density was
`2.87580e-20`--`9.12187e-14 g cm^-3`; temperature was
`6.99996`--`22.40895 K`.  From c964, gas mass changed by `+4.55200e-9`
relative and z angular momentum by `+9.25030e-8`, essentially the same budgets
as the validated global recovery.

The solver interval was 720 s, Slurm elapsed time 12 min 35 s, and the MPI
step MaxRSS was 33,507.5 MiB.  The final HDF SHA-256 is
`7415cb27...8a38fc6`; the final restart SHA-256 is
`6205273c...077f128`.

Job `343874` was an infrastructure-only failed attempt: the isolated source
snapshot lacked the restart's relative table paths and exited before entering
the solver.  The reproducible script now symlinks the immutable validated
tables and job `343876` is the numerical result.

## Comparison and decision

The previous global two-point recovery job `343472` needed 767 total FGMRES
iterations and 208 solver seconds.  The local mixed operator needed 4,936
iterations (6.44 times as many) and 720 seconds (3.46 times as long).  Thus the
first candidate preserves tangential reconstruction away from 28 troubled
blocks, but the mixed stencil is much harder for the current MG
preconditioner.

The c970 maximum density differs from the global recovery by only about
`8.1e-10` relative and maximum temperature by `1.72e-4` relative.  Gas mass
and Lz differ by `3.9e-15` and `4.9e-13` relative.  Other radiation history
quantities differ more (up to 3.67% for a small-valued quantity), which is
expected because this run intentionally retains tangential reconstruction on
most interfaces instead of following the globally two-point trajectory.

Decision:

- retain the implementation and evidence on the dedicated experimental
  branch;
- do not merge or make it the default yet;
- keep the validated global whole-solve two-point recovery as the faster
  operational fallback;
- if local accuracy retention is required, next test a globally two-point MG
  preconditioner for the locally mixed FGMRES operator rather than weakening
  Newton/FGMRES tolerances or reducing CFL.

## Unvalidated scope

The mask is process-local and is not serialized into restart files.  The
decisive run created and destroyed zero MeshBlocks, so mask handling across an
actual regrid is not validated.  A new process or reconstructed MeshBlock
starts unmasked and may trigger the bounded detection/restart sequence again.
These limitations prevent a default-production recommendation despite the
successful c964--970 safety result.
