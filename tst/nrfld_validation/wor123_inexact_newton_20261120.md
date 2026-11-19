# WOR-123: bounded inexact FGMRES handoff

Date: 2026-11-20 JST.  This audit starts from commit `78e45de0d` in the
isolated branch `exp/wor123-fixed-inexact-pilot-20261120`.  It does not change
the default solver path or the outer Newton convergence test.

## Decision

An unconditional loose FGMRES target is useful as a diagnostic and completed
the RSG interval, but is not the recommended default: it reduced linear work
while increasing most Newton solves from three to four iterations.  The safer
candidate is an opt-in restart-boundary rule.  The ordinary `1e-4` target is
kept; a correction may be returned early only when all of the following hold:

1. at least two restart intervals have completed;
2. the independently recomputed true relative residual is finite and no
   breakdown occurred;
3. that true residual is at or below an explicit cap; and
4. the latest restart interval reduced it by less than the configured amount.

The tested RSG values are cap `1e-2` and progress ratio `0.9` (less than 10%
reduction).  The cap is hard-limited to `0.1`.  The feature is disabled by
default (`nrfld/fgmres_inexact_true_cap=0`).  The outer Newton merit, line
search, final L2, final maximum residual, positivity handling, and accepted
time-step decision are unchanged.  A projected Krylov residual alone can
never activate the rule.

This is a bounded inexact-Newton heuristic, not an Eisenstat--Walker forcing
term: the Schur residual and the complete coupled nonlinear residual are
different norms.  The prior WOR-92 exact-checkpoint campaign nevertheless
provides relevant independent evidence for this same restart-ratio rule: on a
seven-cycle AMR collapse segment it reduced 3049 to 997 FGMRES iterations
without increasing the two Newton iterations per step, while rejecting a
case above its explicit true-residual cap.

## XD RSG comparison

All runs start from
`/work/mizutaniks/wor122/prep256_outer_t20c/wor122_prep256_outer_t20c.00002.rst`
(SHA-256 `cc60130d36d9bbe09d4e614f2e7cf341a0aa804dba30f78d78f3731336c6b41d`),
use one M-bulk-a node and 64 MPI ranks, and end at code time `17.25`.  The
unchanged EOS table SHA-256 is
`7051d1c5222f3ca7dddbbba1c48d473cb39bc5e76badb869b082625473a71576`;
the opacity table SHA-256 is
`00c517e858f33e85fb3fab26618600f43335c445b99ea01d3393549fc27eaefb`.

| Case | Job | FGMRES solves / iterations | Median / max | Newton steps and iterations | Slurm elapsed | Result |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| strict r20, max500, target 1e-4 | 340619 | 87 / 2845 | 31 / 124 | 29, all 3 | 22:43 | PASS |
| strict r40, max500, target 1e-4 | 340609 | 96 / 2385 | 25 / 37 | 32, all 3 | 20:03 | PASS |
| fixed target 1e-2 | 343489 | 108 / 1847 | 19 / 60 | 28: 3 x 4, 4 x 24 | 15:55 | PASS |
| target 1e-4 plus bounded plateau rule | 343496 | 87 / 2845 | 31 / 124 | 29, all 3 | 22:33 solver | PASS, no trigger |

The fixed `1e-2` run used 35.1% fewer FGMRES iterations and 29.9% less
elapsed time than strict r20; relative to strict r40 the reductions were 22.6%
and 20.6%.  Its maximum converged Newton final L2 was `9.32365e-11`, maximum
component residual `5.64570e-9`, and all 28 physical steps met the unchanged
outer criterion.  It completed cycle 315--343 with zero opacity-domain,
clamp, negative, zero, and nonfinite events.  Its final restart SHA-256 is
`f4705cf2ab2d318e23407b115d7b48875034fa5226bd3edb4d4e6475cbe65f2b`.

The fixed run's mass change over the common history window was
`-2.36552e-9` relative, bracketed by the strict controls (`-2.14662e-9` and
`-1.76492e-9`).  Representative energy-history relative changes were also
inside or at the scale of the two strict trajectories.  At the common final
time, mass differed from strict r20/r40 by `2.19e-10`/`6.01e-10` relative.
Bulk energy columns differed at approximately `1e-10`--`2e-7` relative.
Point extrema are chaotic: the two strict runs themselves differ, so those
columns are not used as a bitwise acceptance gate.

The common-time `prim` HDF5 fields were also compared cell by cell (4096
blocks, `256^3` cells; the output datatype is float32).  Mesh metadata,
logical locations, and levels are identical.  Fixed `1e-2` versus strict
r20/r40 relative L2 differences are respectively `2.58e-8`/`4.38e-8` for
density, `3.30e-7`/`6.94e-7` for pressure, `3.95e-8`/`1.27e-8` for radiation
energy, and `4.92e-5`/`1.18e-4` for each velocity component.  The strict
r20--r40 velocity difference is already `6.98e-5`; the fixed-run maximum
absolute velocity difference normalized by the control maximum is at most
`5.50e-4`.  Density sums differ by only `2.14e-10`/`5.98e-10` relative and
radiation-energy sums by `1.78e-8`/`2.86e-9`.  All fixed-run density,
pressure, and radiation values are finite and positive (minima
`1.00e-13`, `1.41e-15`, and `1.86e-6` in code units).

The bounded-rule replay made zero inexact acceptances: every restart continued
to make more than 10% progress and every solve reached the strict `1e-4`
target.  This is a useful negative result, not a failed solver run.  It exactly
reproduced strict r20: the 13x30 history array is bitwise identical (common
SHA-256 `6d455c13b3afacddb039aa5821335f1d086af42a5bc63db8bda78534b4652c92`),
all final primary-HDF5 datasets and attributes compare exactly, and the 87
solve iteration counts, 29 Newton iteration counts, residual maxima, and final
time are identical.  Opacity diagnostics are all zero and the srun step exits
zero.  Slurm labels job 343496 `FAILED` only because the first wrapper version
deliberately returned 1 when no inexact trigger occurred; `status.csv` records
solver exit 0, time-limit completion, no fatal, and valid opacity.  Commit
`0fe79fff0` removes that inappropriate wrapper-only requirement; no expensive
rerun is needed for a scheduler-label change.

The width comparison favors r40 over r20 in this interval (11.7% shorter and
16.2% fewer iterations), but it is not a universal fallback: it consumes
roughly the same per-rank peak memory (`3.22` versus `3.08` GiB reported for
the srun step; fixed `1e-2` is `3.14` GiB) and does not address a bounded correction that Newton could
use.  MG2 had already regressed on the matched RSG control, so it was not
reintroduced as an automatic fallback.

No Eisenstat--Walker implementation is added in this branch.  A true forcing
term would require an explicit, verified mapping between the coupled Newton
norm and the Schur right-hand side, plus retry state in the Newton driver.
The fixed pilot already measures the loose end of that design and shows its
cost (extra Newton iterations); the selective true-residual plateau rule
addresses the observed wasted inner work with fewer new mechanisms.  Adding a
second adaptive policy before the simpler rule has a matched RSG result would
expand the solver surface without evidence of a better total time or accuracy.

## Local default, MPI, restart, and conservation gates

The final code was built with MPI and exercised with Test C on both uniform
and static-AMR meshes, fixed and variable limiter, and one/two ranks.

* Default-off and opt-in/no-trigger two-rank histories are bitwise identical.
* A forced-trigger static-AMR comparison agrees between one and two ranks to
  `5.82e-15` maximum absolute history difference and `8.66e-14` maximum
  relative difference for quantities above `1e-12`.
* A one-cycle restart followed by cycle two is bitwise identical in history to
  the uninterrupted two-cycle run.  The cap and ratio persist in the restart.
* The forced static-AMR variable-limiter run conserves mass and hydro energy
  bitwise; radiation-volume drift is `1.04e-11` relative.  The uniform case is
  `1.47e-12`; static AMR with fixed limiter is `1.31e-15`.
* Every forced-trigger run still satisfies the full Newton final test.  The
  deliberately extreme local test returned after four FGMRES iterations at
  true relative residual `2.99728e-6`, then reached Newton final L2
  `3.24958e-11` and maximum `9.98314e-11`.
* With cap `1.1e-16`, below the observed true-residual floor but above the
  requested `1e-16` target, the solver reaches 20 iterations with true
  residual `7.15102e-16`, records `inexact_accept=0`, and raises the ordinary
  FGMRES failure.  It never misclassifies the result as converged.
* Cap values above `0.1`, nonfinite values, and ratios outside `(0,1)` are
  rejected at initialization.

These forced settings are branch-coverage tests, not recommended physical
tolerances.  They demonstrate that the acceptance decision is bounded by the
true residual and that the complete nonlinear solve remains authoritative.

## Roundoff and scope

Changing algebraic evaluation order can shift a binary64 residual floor, and
two-pass MGS plus long-double reductions already improve the projected Krylov
calculation.  They cannot generally eliminate a representation gap in a
high-contrast operator: the accumulated correction, its stored binary64
representation, and a fresh operator application need not reproduce the
projected residual.  The WOR-92 cycle-46 audit measured this mechanism
directly.  Therefore the principled practical response is not to label every
missed inner target as success, but to bound the independently recomputed
residual and require the outer nonlinear solve to pass unchanged tests.

The RSG `1.95e-3` failure is not claimed to be pure roundoff.  It is a
trajectory-sensitive, preconditioned linear convergence event for which the
outer Newton solve may still make useful progress.  The fixed-target pilot
shows that a correction at the `1e-2` scale is sufficient on a matched short
interval.  The bounded plateau rule is deliberately selective so ordinary
solves still reach `1e-4`.

The original failure occurred during a continuous segment after the last
saved checkpoint; no restart exists at its failing Newton state.  Replays from
the time-16 checkpoint follow a reproducible but slightly different chaotic
trajectory, even across different XD nodes.  The strict r20/max500 replay and
the new bounded-rule replay match on that restarted trajectory, whose hardest
solve reaches 124 rather than 200 iterations.  It would be incorrect to call
the absence of the original failure on this restarted path a solver repair,
or to manufacture a trigger by weakening the 0.9 stagnation criterion.  The
fixed `1e-2` run tests whether the unchanged outer Newton can use a bounded
loose correction; the earlier exact-checkpoint WOR-92 campaign tests the
stagnation trigger itself.

## Reproduction and evidence

Batch launchers are
`tst/nrfld_validation/wor123_rsg_fixed_inexact_1e2_xd.sbatch` and
`tst/nrfld_validation/wor123_rsg_plateau_inexact_xd.sbatch`.  Local raw
evidence is under
`/home/kosuke/simulation/test/rad_nr/wor123_rsg_inexact_20261120`; large XD
restart/HDF5 products remain in their immutable job directories.  The feature
is opt-in and no merge to a publication or main branch is performed here.
The compared final primary-HDF5 SHA-256 values are
`969412236aeefa8adf2cddc322e3807c47a3eb161c9c5e7409002f5aba49073f`
(fixed),
`590837e6371b9cd6d6a9906909d3d7495a82904aa320c3b6d54cb0101d344160`
(strict r20), and
`a19f1d9a5f586cdf0f15a28419ffa48c5dd4aafa05d18f479ca5849b9f33402b`
(strict r40).
