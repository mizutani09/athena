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

## Long-run result and cycle-874 gate

M-test job `341805` passed c55--65 in 360 s on 128 ranks.  All 20 active
FGMRES solves met the strict `2e-8` target (maximum 747 iterations), all ten
steps converged in two Newton corrections, and there was no fatal or opacity
diagnostic error.  This validated restart/max 1000 for the early failure.

Five-node job `341806` then advanced the immutable c50 restart through the
durable c870 checkpoint in 82 successful ten-cycle segments.  The 1,649
accepted FGMRES solves before the fatal solve were at or below the `1e-6`
inexact cap; only three early stagnation accepts exceeded the strict target,
and those were within `7.1e-11` of it.  All 968 logged active Newton solves
converged in at most two corrections.  No opacity, EOS-domain, gas-energy,
Jeans-resolution, or memory failure occurred.

The next segment advanced transiently through c873.  At c874 its first
linear solve reached `9.82929e-7` after 1000 iterations and was accepted by
the inexact cap; the next Newton linearization ended at `7.26944e-5` and
aborted.  The complete c870 restart remains available with SHA-256
`745301b146ca6dea1464e9a79b42318d6074efc0fe758283357e863a9516a60f`.
The pattern again shows that a cap-eligible but non-strict first correction
is unsafe for this strong-field trajectory.

The c870 HDF has 2,211,840 active cells, no EOS-outside cells, and no
nonpositive inferred gas energy.  Its total-energy closure relative error is
`1.05e-7` median, `1.48e-6` p99, and `4.21e-6` maximum.  The history has
`rho_max=7.68e-14 g cm^-3`, `Tmax=18.23 K`, Jeans minimum 32.14 cells, mass
change `+2.45e-7` relative, and `divB_max=6.69e-13`.  The minimum plasma beta
is 0.0213 globally, 0.129 for `rho>1e-16 g cm^-3`, and 6.43 at the density
maximum.

M-test job `341961` replays c870--880 on the same 320 ranks with restart 1000,
MG3, and maximum iterations 1600.  This preserves the tested basis width but
allows a second restarted phase instead of accepting the c874 correction at
iteration 1000.  Per-cycle restart output is enabled so that any later
failure leaves the last completed state.  The job is queued for resources.
Five-node M-large-a continuation `341971` is submitted with an
`afterok:341961` dependency and repeats the status/cycle checks internally;
it will start from the gate's c880 restart only if the gate passes.

Job `341961` ran for 910 s and failed while stepping from c874.  The first
solve improved from `9.82929e-7` at iteration 1000 to `3.27737e-7` at
iteration 1600, but remained above the strict target and was cap-accepted.
The following solve improved from `4.08331e-5` to `5.14881e-6` over the same
iterations but remained above the cap.  Raising only the maximum iteration
count is therefore insufficient.  Per-cycle output preserved a complete
c874 restart (4,320 MeshBlocks, SHA-256
`94c1d4bf5295bec5bb56de715bcb8bb325b8376b56097e2658168b210a136e7d`).
The unsatisfiable dependent job `341971` was cancelled.

M-test job `342035` now replays c874--880 with restart 1000, maximum 2000,
and MG5.  This targets the measured slow MG convergence (worst factor about
0.982) instead of only extending the outer Krylov iteration.  It uses five
nodes/320 ranks and per-cycle restarts.  At the 13:25 scheduler review it had
a reserved start of 2026-11-18 13:46:46 JST on
`cna047-048,059-061`; no queue migration was justified because both M and P
node pools lacked five immediately idle exclusive nodes.

Job `342035` started at 13:29 and failed after 835 s during the first c874
step.  MG5 was materially stronger than MG3: its first solve decreased from
`9.79248e-7` at iteration 1000 to `3.36460e-8` at iteration 2000.  This was
still just above the strict target and was cap-accepted.  The second solve
ended at `6.10969e-7` and was also cap-accepted; a third Newton solve then
ended at `1.73372e-4`.  No step completed and no new restart was written.
Thus MG5 alone does not remove the unsafe inexact-correction sequence.

M-test `342092` tests one c874 step with MG5, restart width 2000 and maximum
2500 iterations.  It retains the useful Krylov subspace that job `342035`
discarded at iteration 1000.  The measured basis allocation is expected to
be about 213 MB/rank, or 13.6 GB/node, within the 110 GB/node request.  The
job is pending only because another same-user M-test occupies the QOS slot.

Job `342092` passed in 335 s.  Its two FGMRES solves met the strict target at
1115 / `1.73164e-8` and 1128 / `1.99217e-8`; Newton converged in two
corrections with final maximum residual `2.36308e-7`.  This is direct evidence
that retaining the Krylov space across iteration 1000, rather than merely
raising the cap or MG count, resolves the c874 failure.  The c875 restart has
SHA-256 `0884d4dd2747c61c754d5142876429606e574db315f1e755026617f2b3e54810`.

The c875 HDF contains 2,211,840 cells with zero EOS-outside and zero
nonpositive-gas-energy cells.  Its total-energy closure error is
`1.05e-7` median, `1.47e-6` p99 and `4.38e-6` maximum; HDF SHA-256 is
`59c1fd9e2798b14db96d31fd9bf542a03d190e0df4d54b8e7c9fe96b6b5194fb`.

Five-node M-test `342173` now checks c875--880 with the same
restart-2000/max-2500/MG5 configuration and per-cycle restarts.  Five-node
M-large-a continuation `342174` has an `afterok:342173` dependency and an
independent status/cycle gate; it starts from c880 only if the sustain test
passes.
