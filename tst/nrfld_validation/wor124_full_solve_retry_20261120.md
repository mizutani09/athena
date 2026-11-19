# WOR-124: whole-solve tangential fallback (2026-11-20)

## Scope

This report evaluates an opt-in recovery for the strong-field, rotating,
table-EOS collapse failure at cycle 964.  The conservative tangential
coarse/fine reconstruction remains the configured default.  If its first
Newton direction would cross the radiation positivity boundary, the driver
rejects the direction before changing the iterate, restarts the complete
nonlinear solve with the robust two-point coarse/fine face form, and retains
that form for the rest of the process.

The experiment is isolated on branch
`exp/wor124-full-solve-early-retry-20261119`.  It is not merged into the
collaboration freeze or `main`.  The behavior is disabled unless
`ATHENA_NRFLD_TANGENTIAL_FULL_SOLVE_RETRY=1`.

## Why the complete solve must be restarted

The exact cycle-964 audit found that the tangential reconstruction produced a
raw radiation correction crossing zero in fine cells next to the failing
coarse/fine face.  Applying independent positivity floors destroyed the
shared-face linear moment identity.  Switching the face form inside an
already-started Newton solve did not remove the residual and was rejected.

`Solve_general()` does not commit the implicit state to hydro until the solve
succeeds.  Re-entering the solve through `LoadVariables()` and
`BeginTimeStep()` therefore reconstructs the state, nonlinear right-hand side,
coefficients, boundary data, and MG caches consistently.  The recovery does
not accept a failed or inexact Newton result.

## Implementation under test

Commits:

- `99395a3e4`: complete nonlinear replay with tangential reconstruction off;
- `1a5ef599c`: retain the recovered two-point mode until process exit;
- `496b80f0e`: inspect the first converged linear direction and request the
  replay before applying it if the limited update would cross `TINY_NUMBER`;
- `eb74ee14c`: exact c964--970 XD gate.

The direction check uses the same `LimitNewtonCorrection()` rule as the state
update.  It is active only for multilevel NR-FLD with the opt-in retry flag and
tangential reconstruction enabled.  Counts and the global minimum admissible
scale are reduced over MPI ranks.  Non-finite initial/final failures are not
retried.  A failed recovery restores the configured mode; a successful
recovery latches the two-point form only in the current executable process.

## Local regression

The table-EOS, MHD, MPI build completed successfully.  A two-rank N32 AMR Test
C was run with the retry flag enabled and disabled.  Every non-timing metric
was identical.  The normal tangential path used FGMRES 6 and Newton 1, and the
implicit diffusion ledger was `1.0323214044661785e-20`; no fallback was
requested.  This demonstrates that merely enabling the gate does not alter a
safe direction.

Local artifacts are in
`/home/kosuke/simulation/test/rad_nr/wor124_full_solve_early_retry_20261120`.

## Exact c964 single-step replay

XD job `343420` first tested the full-solve replay after the ordinary Newton
failure:

- primary tangential solve: FGMRES `628, 605, 576, 576`, then stagnation;
- complete two-point replay: FGMRES `12, 11`, Newton 2;
- final L2 `1.1364e-9`, final maximum normalized component `7.4571e-8`;
- positive finite c965 state, zero opacity-domain cells.

This proved that a complete replay removes the history dependence that made
the mid-Newton switch fail.  It still spent all four expensive primary linear
solves before recovery.

## Latched c964--970 replay

XD job `343466` tested the process-local latch.  It recovered once and then
kept the robust form for six accepted steps.  All later FGMRES solves used
11--12 iterations and every step converged in two Newton corrections.  It
completed in 7 min 43 s (431 s solver interval).  The final physical history
agreed with the explicit two-point control; the remaining divB difference was
about `3.2e-14` at an absolute value near `9e-13`.

## Early direction rejection: final gate

XD job `343472` used five M-large-a nodes and 320 MPI ranks, CFL 0.15, from
the immutable cycle-964 restart through cycle 970.  It completed successfully:

- Slurm elapsed: 3 min 57 s; solver interval: 208 s;
- maximum Slurm-step RSS: 34,777,148 KiB (33.17 GiB);
- direction rejection count: 1; successful whole-solve retry count: 1;
- first primary FGMRES: 628 iterations, true relative residual
  `1.9926897992719832e-8`;
- rejected direction: 287 crossing cells, minimum admissible scale
  `0.15145281574226735`;
- robust FGMRES after the rejection: 12 solves, 11--12 iterations, total 139,
  maximum true relative residual `1.865531152400878e-8` for target `2e-8`;
- six accepted steps, each Newton 2;
- maximum final Newton L2 `1.1392323075625262e-9`;
- maximum final normalized component `4.4059119293061186e-6`.

Compared with the late-retry job, rejecting the known-unsafe direction avoided
1,757 FGMRES iterations (`605+576+576`), reduced the total count from 2,524 to
767, and reduced the solver interval from 431 to 208 s.  The expensive first
628-iteration solve is still required to construct the direction being
tested.

The final c970 audit found:

- 2,412,544 cells, maximum level 11;
- zero non-finite or non-positive cells;
- zero cells outside the opacity table;
- density `2.8758000283e-20`--`9.1218658603e-14 g cm^-3`;
- temperature `5.1002607`--`22.4128056 K`;
- gas-mass change from c964: `+4.5519979e-9` relative;
- z-angular-momentum change from c964: `+9.2502544e-8` relative.

The c970 history agrees with job `343466` to a maximum relative difference
`3.36e-11` after excluding divB; gas mass and z angular momentum agree exactly
at history precision.  It agrees with the explicit two-point job `343388` to
`4.20e-11` by the same measure.  The divB absolute differences are
`4.84e-14` and `8.04e-14`, respectively, at a value of order `1e-12`,
consistent with MPI reduction noise rather than a physical trajectory change.

Hashes:

- c970 HDF SHA-256:
  `fabf4bbe08a1215edc2bf0a37e4be926ea561b90c5f0beb65e0056b4e066a7a3`;
- final restart SHA-256:
  `7f0987a00a6a5b37c7c2889cddca8184e629ddf825941b759d6b97e1938b61c2`;
- EOS table SHA-256:
  `742a3456923d4c7056c3705accbcc793af3e0c79408a2f597a0cd9271b13ba43`;
- opacity table SHA-256:
  `6c42445452bca6aeca1253f31ab049b2ff107dea3601f946b4c61c93db022f94`.

## Decision and limitation

The early whole-solve retry is a validated, opt-in recovery for this
strong-field AMR trajectory.  It is preferable to reducing CFL because it
identifies the actual non-monotone coarse/fine correction and rebuilds one
consistent nonlinear problem.  It does not weaken Newton or FGMRES tolerances.

It is not a global replacement for tangential reconstruction.  AMR Test C at
N64 previously measured a 24.3-fold coarse/fine-band accuracy advantage for
the tangential form, while both forms conserved the diffusion integral.
Therefore the global default remains unchanged.

The process-local latch is not serialized into an Athena restart.  A fresh
job again follows the configured tangential mode.  Segmented strong-field
production runs should explicitly set the validated two-point form after a
recorded recovery, or deliberately re-enable this gate.  Persisting a
problem-specific solver choice in the physical restart is not justified as a
global source change by this single trajectory.

