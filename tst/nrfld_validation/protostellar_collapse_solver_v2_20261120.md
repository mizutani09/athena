# Protostellar-collapse solver v2 freeze (2026-11-20)

## Frozen candidate

- Branch: `release/protostellar-collapse-solver-v2-20261120`
- Solver source-tree object: `8c4a94cb1b53b0ce6a7b574371351007a98472cb`
- Physical input: `inputs/radiation/athinput.protostellar_firstcore_tabular_v1`
- Runtime environment: `inputs/radiation/protostellar_firstcore_v2.env`
- Smoke gate: `tst/nrfld_validation/collapse_solver_v2_smoke.py`

This is an isolated release candidate.  It is not merged into `main` and does
not modify the user's other worktrees.  The input and smoke gate pin the
source, EOS table, opacity table, and production environment.

## Selected linear-solver policy

The selected input uses FGMRES restart 40, at most 300 iterations, and two MG
V-cycles per preconditioner application.  Normal production additionally
allows an inexact true-residual cap of `1e-7` after stagnation at ratio `0.9`.
The outer Newton residual remains the acceptance criterion; an inexact linear
acceptance does not accept a failed nonlinear solve.

The conservative tangential coarse/fine face form is tried first.  With
`ATHENA_NRFLD_TANGENTIAL_FULL_SOLVE_RETRY=1`, only an unsafe first Newton
direction, an ordinary nonlinear failure, or an FGMRES maximum-iteration
exception causes a complete replay with the robust two-point face form.  The
failed iterate is never committed.  Other runtime errors are rethrown.

This makes the expensive robustness path demand-driven rather than the normal
path.  Once recovery succeeds, the two-point form is retained for the current
executable process.  The latch is intentionally not serialized in a restart.

## Parameter selection

The exact cycle-964 table-EOS, rotating, magnetized checkpoint was used for a
small cost sweep on five XD nodes (320 MPI ranks):

| Restart | MG V-cycles | FGMRES iterations | Solver time | Krylov basis/rank | Decision |
|---:|---:|---:|---:|---:|---|
| 40 | 1 | 300 + 15 + 15 | 50 s | 4,644,864 B | reject: normal real-collapse path reached the cap |
| 40 | 2 | 300 + 11 + 11 | 54 s | 4,644,864 B | select |
| 160 | 2 | 300 + 11 + 11 | 55 s | 18,407,424 B | reject: no speed benefit, about 4x basis memory |

The first 300-iteration solve in this deliberately difficult checkpoint is
the tangential attempt.  The following short solves are the automatic robust
replay.  Tiny Test C alone made MG=1 look 14--15% faster, but a real collapse
step did not converge on the normal path with MG=1; that saving is therefore
not reliable enough for the shared default.

## Release gate

XD job `344231` ran cycles 964 through 970 with restart 40, maximum 300, two
MG V-cycles, and strict FGMRES acceptance (the inexact production variables
were unset):

- successful FGMRES-cap recovery and no fatal error;
- 13 linear solves, 430 total iterations;
- seven converged Newton solves and 12 Newton updates;
- maximum final Newton L2 residual `1.2428914022491265e-9`;
- solver interval 143 s, Slurm elapsed 2 min 54 s;
- maximum Slurm-step RSS 17,310,588 KiB;
- 2,412,544 audited cells, with no non-finite, non-positive, or
  opacity-table-outside cells.

Relative to the earlier restart-2000, maximum-2500, MG-5 recovery job
`343472`, total FGMRES work fell from 767 to 430 (43.9%), solver time from 208 to 143 s
(31.3%), and maximum step RSS from 34,777,148 to 17,310,588 KiB (50.2%).  The
gas-mass-like history field differed by `5.07e-16`; the energy-like field was
identical at printed precision.  Derived extrema differed by at most
`5.8e-7`.  The only larger relative change was an absolute `5.33e-14` change
in near-zero divB.

XD job `344245` repeated the same c964--970 gate with the production inexact
policy (`true cap=1e-7`, stagnation ratio `0.9`).  It also passed: the first
tangential solve reached the 300-iteration cap, the complete two-point replay
converged, all six active Newton solves used two updates, and the opacity audit
again found no invalid or out-of-domain cells.  It used the same 13 FGMRES
solves and 430 total iterations as the strict gate; 12 short replay solves
were marked as inexact accepts.  Thus the policy did not reduce iteration
count on this particular trajectory, but neither did it weaken Newton or
change the physical result materially.  Excluding near-zero divB, all history
columns agreed with the strict gate to `8.3e-12` relative or better; divB
differed by `3.64e-14` absolute.  Solver wall time was 136.6 s and Slurm-step
RSS was 17,521,396 KiB.  The inexact stop remains useful on normal steps that
stagnate below the Newton-adequate cap, while the replay remains the safety
net when the cap is not reached.

## Local gates

- Two-rank one-cycle table-EOS/MHD/AMR smoke: PASS.  The ordinary path did not
  invoke recovery; gas-mass drift was `-3.22e-15`, Newton converged in one
  update, and the history file was byte-identical to the pre-recovery build.
- Post-patch variable-limiter Test C at two MPI ranks: PASS for uniform,
  aligned AMR, and shifted AMR.  With MG=2 the cases used 2, 5, and 5 FGMRES
  iterations, with no rejected corrections.  Artifacts are in
  `/home/kosuke/simulation/test/rad_nr/collapse_solver_v2_testc_postpatch_20261120`.
- Post-patch AMR/regrid restart and MPI-rank suite: PASS.  The direct,
  one-rank restart, two-rank direct, and two-rank restart paths reached the
  same time after a topology change; all reported relative L1 and maximum
  field differences were exactly zero.  Artifacts are in
  `/home/kosuke/simulation/test/rad_nr/collapse_solver_v2_restart_postpatch_double_20261120`.
- Closed-box conservation suite: PASS for all five tolerance/CFL cases, with
  mass and total-energy drift of order `7e-15`.

- Full radiative shock-tube regression: PASS with strict backward-error
  convergence.  N64 completed 1,331 cycles and N128 completed 2,666 cycles;
  all 3,997 Newton solves converged strictly, with no relaxed acceptance or
  collapse fallback.  Relative L1 error versus the N4096 Rad-EOS reference
  improved from N64 to N128 for all six recorded fields: density
  `0.1004 -> 0.0611`, velocity `0.0380 -> 0.0192`, temperature
  `0.0147 -> 0.00820`, gas energy `0.0971 -> 0.0625`, radiation energy
  `0.0214 -> 0.0130`, and total pressure `0.0168 -> 0.00906`.  The N64 and
  N128 runs took 13 min 58 s and 32 min 17 s locally.  The browsable report,
  profiles, CSV data, and movie are in
  `/home/kosuke/simulation/test/rad_nr/collapse_solver_v2_shock_20261120/index.html`.

## Intended use and limits

This candidate freezes the solver behavior needed for collaborator first-core
calculations.  Application-level initial conditions, refinement rules, and
physical tables may still be changed, but those changes require their own
domain and conservation checks.  The automatic replay is validated for the
observed strong-field AMR failure; it is not evidence that the two-point form
should replace the more accurate tangential form globally.

The release does not claim second-core coverage, high-temperature EOS/opacity
coverage beyond the pinned tables, or a universal strong-field stability
boundary.  These are physics/application extensions rather than reasons to
keep changing this first-core solver candidate.
