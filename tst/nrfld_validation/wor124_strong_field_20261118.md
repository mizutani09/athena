# WOR-124: 129 microG strong-field FGMRES recovery

Date: 2026-11-18 JST

## Failure mechanism

The rotation plus 129 microG ideal-MHD trajectory previously stopped in the
cycle-55 step.  With FGMRES restart width 160, maximum 600 iterations and
MG3, the first linear solve ended at true relative residual `1.0186e-5`
instead of the requested `2e-8`.  MG5 reduced it to `2.6095e-6` but still did
not meet the existing `1e-6` bounded inexact cap.  This is a linear-solver
failure before Newton can test a complete correction; it is not a negative
gas-energy, opacity-domain, Jeans-resolution or memory failure.

## Rejected remedies

P-debug job `341792` replayed the same cycle-55 checkpoint on 64 ranks.

- CFL 0.15 let the first half-step converge, but the second half-step's
  second FGMRES solve ended at `5.7390e-5`.  Timestep reduction alone was not
  sustained to the original one-step physical time.
- Fixed threshold `5e-6` with MG5 accepted the first correction after 411
  iterations, but the next solve ended at `3.6605e-2`.
- Fixed threshold `2e-5` with MG3 accepted the first correction after 364
  iterations, but the next solve ended at `2.0996e-2`.
- Fixed threshold `1e-5` with MG3 missed even that target slightly, ending at
  `1.01865e-5` after 600 iterations.

Thus an intentionally loose first correction makes the subsequent Newton
linearization substantially harder in this state.  A globally loose linear
tolerance is rejected for the long run.

## Krylov-width result

P-debug job `341793` retained the strict `2e-8` target and compared larger
Krylov subspaces from the identical checkpoint.

| Configuration | Result | FGMRES iterations / true residual | Newton final max | Wall |
|---|---|---|---:|---:|
| restart 600, MG3, max 600 | PASS c56 | 466 / `1.813e-8`; 572 / `1.659e-8` | `2.675e-6` | 134 s |
| restart 320, MG5, max 600 | PASS c56 | 531 / `1.969e-8`; 600 / `3.628e-7` (bounded inexact accept) | `2.634e-6` | 173 s |
| restart 320, MG5, max 1000 | PASS c56 | 531 / `1.969e-8`; 811 / `1.986e-8` | `2.578e-6` | 204 s |

The restart-600/MG3 case is both the fastest and the only tested case that
keeps both linear solves below the strict target without increasing the
iteration ceiling.  The failure is therefore attributable to discarding a
useful Krylov subspace at width 160, rather than an intrinsically unsolvable
operator at this checkpoint.

The restart-600 and strictly converged restart-320/MG5 outputs have identical
AMR layouts and are bitwise identical in their single-precision HDF density,
total energy, radiation energy, temperature and magnetic fields.  Their
double-precision history values differ by at most `3.12e-10` relative.

The restart-600 c56 HDF contains 491,520 active cells.  It has zero cells
outside the EOS table and zero cells with nonpositive inferred gas energy.
The magnetic total-energy closure discrepancy has median `3.82e-7`, p99
`2.26e-6`, and maximum `2.64e-6`.  Gas mass is unchanged at printed history
precision and `divB_max=6.43e-17`.

## Sustained gate: restart 600 is not sufficient

Two-node M-test job `341799` replayed c55--65 with restart 600, MG3 and the
strict target on 128 ranks.  Cycles 55--58 completed with two Newton
corrections each.  At cycle 59, however, the first solve reached only
`1.13924e-7` at iteration 600 and was accepted by the existing bounded
inexact cap.  The following Newton linearization then ended at
`5.56328e-3`, so the run aborted after 171 s.  This is the same failure mode
as the deliberately loose cycle-55 experiments: a merely cap-eligible first
correction is not sufficiently accurate for the next strong-field
linearization.  A single checkpoint replay was therefore necessary but not
sufficient evidence for production.

The held restart-600 production job `341800` was cancelled without starting.
The next gate, M-test job `341805`, keeps MG3 and the strict target but expands
both the Krylov width and maximum iteration count to 1000.  Replacement
five-node M-large-a job `341806` is submitted with an `afterok:341805`
dependency and a second in-script status-file gate.  It cannot start unless
all ten gate cycles through c65, including c59, pass.
