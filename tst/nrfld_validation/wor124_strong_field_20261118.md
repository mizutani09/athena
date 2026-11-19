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

## Sustained result and strict-only production policy

Job `342173` ended with Slurm `TIMEOUT` after 1821 s; it did not encounter a
numerical failure.  It completed c875--879, leaving complete per-cycle
checkpoints through c879.  The eight completed FGMRES solves for c875--878 all
met the strict `2e-8` target in 1082--1263 iterations, and all four Newton
steps converged in two corrections.  The first c879 solve also met the strict
target at 1166 / `1.97582e-8` before the 30-minute queue limit interrupted its
second solve.  There was no fatal or opacity diagnostic error.  Because Slurm
did not classify the gate as successful, dependent job `342174` became
unsatisfiable and was cancelled without starting.

The recurring mechanism is now sufficiently isolated.  Restart width 1000
discarded a useful Krylov space just before the 1115--1263 iterations required
by this state.  Independently, the experimental `1e-6` inexact cap allowed an
iteration-ceiling result to enter Newton even when it was not accurate enough
for the following linearization.  The production policy is therefore
restart 2000 / maximum 2500 / MG5, with the inexact cap and stagnation accept
unset.  A solve that cannot reach the strict target stops at a complete
checkpoint instead of advancing with a coarse correction.

M-test job `342417` validated this strict-only policy from the complete c879
checkpoint.  It completed c879--880 in 358 s.  Both FGMRES solves met the
strict target at 1166 / `1.97482e-8` and 1193 / `1.93845e-8`, with
`inexact_accept=0` and `inexact_cap=0`; Newton converged in two corrections
with final maximum residual `5.68970e-7`.  Opacity diagnostics were clean.
The c880 restart SHA-256 is
`75b1a599d6036d4ea5fee02dd8d7a8161b33be3825d005f53a816952d7f37dfb`.

The c880 HDF contains 2,211,840 cells, zero EOS-outside cells and zero
nonpositive inferred gas energies.  Its total-energy closure relative error
is `1.05e-7` median, `1.47e-6` p99 and `4.29e-6` maximum; HDF SHA-256 is
`eca6fafbbf0be438ca72bcb4d7839f020d2a8f2eca5ae87825450bd3ba9603af`.

Five-node M-large-a job `342419` started from that validated c880 restart at
2026-11-18 20:40:17 JST.  Its first segment entered the main loop with the
expected restart-2000/max-2500/MG5 configuration and with the inexact cap
disabled.

## Cycle-880 nonlinear localization and timestep result

Job `342419` did not complete its first c880 step.  It failed after 1392 s
with `backtracking_exhausted`, not with an MPI, memory, opacity or linear-solve
error.  All seven FGMRES solves met the strict target in 1253--1282
iterations with `inexact_accept=0`.  Newton nevertheless stalled after six
corrections at combined `L2=2.11855e-5` and maximum `1.0`; the maximum was a
localized total-energy row on level-9 block 1256 at a corner adjoining both
level-8 and level-10 blocks.  The final gas maximum was only `3.27022e-6`.
No step completed and no output was written, so the validated c880 restart
from job `342417` remains the recovery point.

The componentwise FGMRES backward error was initially `5.54e-6` and
`2.76e-6`, then grew to `0.156`, `0.0999`, `0.0685`, `0.473`, and `0.997`
as the right-hand side became localized.  A global relative residual of
`2e-8` therefore did not imply an accurate correction for the single
coarse/fine corner row.  This explains why making the global Krylov solve
strict removed the earlier unsafe inexact handoff but did not by itself make
the next nonlinear step convergent.

Two controls excluded tempting but incorrect explanations.  With the
positive radiation-ghost limiter disabled, job `342453` reproduced the same
localized total-energy maximum and reached componentwise FGMRES error
`0.263`; it was stopped after the third conclusive correction.  With the
limiter restored and `backward_error_convergence=true`, job `342460` reduced
the reported L2 norm to `5.49e-7` but left the localized maximum at `0.161`;
it too was stopped after three corrections.  Thus neither the positivity
limiter nor a different roundoff normalization is the root remedy.

M-test job `342475` replayed the identical c880 checkpoint with only the CFL
number changed from 0.3 to 0.15.  It passed c880--881 in 38 s of solver time.
The two strict FGMRES solves required only 23 iterations each and ended at
`1.72624e-8` and `1.27655e-8`; their componentwise backward errors were
`2.61e-7` and `2.45e-8`.  Newton converged in two corrections to
`L2=1.38712e-9`, maximum `1.91554e-7`, with clean opacity diagnostics.  The
c881 restart SHA-256 is
`38607e1f2e0f2480c298493ca9c2df4abb4ef7c943a57db62509a9ebb106359c`.

The c881 HDF again has 2,211,840 cells, zero EOS-outside and zero nonpositive
gas-energy cells.  Its total-energy closure relative error is `1.05e-7`
median, `1.47e-6` p99 and `4.36e-6` maximum; HDF SHA-256 is
`9adfa43a31f1dddd0d32717f7407861db4a2b35275374a07bb2d004cb26fc7b9`.
The improvement is not a trade of robustness for cost: halving the explicit
CFL made the implicit radiation system much better conditioned and reduced
the solve from roughly 23 minutes without a completed step to 38 seconds for
a completed step.

Ten-step sustain gate `342477` then passed c880--890 in 211 s of solver time
(214 s Slurm elapsed).  All 20 FGMRES solves met the strict target in 22--26
iterations, with maximum true relative residual `1.96603e-8` and no inexact
accept.  All ten Newton steps converged in two corrections; maximum final L2
was `1.38733e-9` and maximum final component was `1.03987e-6`.  Opacity
diagnostics were clean.  The complete c890 restart SHA-256 is
`6eaedecddb24a1aa82b28e79d26d08cc4bb16397554140df003a73ab0268b10b`.
Dependent five-node M-large-a continuation `342478` passed both the Slurm
dependency and its independent in-script gate.  It is pending only for five
M-large-a nodes, with scheduler-estimated start 2026-11-18 22:10:45 JST.

## CFL-0.15 long continuation and opacity-gate refinement

Job `342478` started at 22:34:30 JST and completed normally after 313 s.  It
advanced three ten-cycle segments from c890 to c920.  All 60 FGMRES solves
met the strict target in 24--31 iterations, with maximum true relative
residual `1.99927e-8` and no inexact accept.  All 30 Newton steps converged in
two corrections; maximum final L2 was `1.37062e-9` and maximum final component
was `6.28255e-6`.  Gas mass changed by `+1.78e-8` and angular momentum by
`+4.80e-7` relative over the run.  Jeans resolution remained above 32 cells.
Slurm's maximum step RSS was 37,023,184 KiB (35.3 GiB).

The job stopped intentionally after c920 because its original wrapper treated
any opacity-table clamp as a production stop.  The last segment reported 112
out-of-domain lookups and exactly 112 clamps, with no negative, zero or
nonfinite opacity result.  These were transient Newton or ghost evaluations,
not accepted active cells: the accepted-state guard passed every cycle and
reported minimum `T=5.11440 K`, above the 5 K table bound.  An independent
post-regrid c920 HDF audit found all 2,383,872 cells inside both opacity axes:
`rho=2.873e-20--8.356e-14 g cm^-3` versus table
`1e-21--1e-7 g cm^-3`, and `T=5.114--21.148 K` versus table
`5--9900 K`.  It also found zero EOS-outside and nonpositive-gas-energy cells,
with total-energy closure relative error `9.96e-8` median, `1.44e-6` p99 and
`4.25e-6` maximum.  The HDF SHA-256 is
`b9e2f3c0273324ee957fc84457dbaf86351eabf43f5d9649b54ee3a28badea52`.

The c920 restart is complete, with SHA-256
`8ec753e3e09aa27e4feaa0749d30edc6553dea5158d3c74986e177c14510ce95`.
The continuation wrapper now distinguishes transient trial clamps from an
invalid accepted state.  It requires equal out-of-domain and clamp counts,
zero bad opacity results, ten accepted-state guard records per segment, and a
new post-regrid HDF audit against both table axes.  Any accepted or post-regrid
cell outside the table still stops the run.  Five-node resume job `342607` is
submitted from c920 and is pending M-large-a resources.

Job `342607` subsequently ran from 00:27:13 to 00:29:11 JST and advanced
c920--930 normally.  All 20 strict FGMRES solves converged in 30--36
iterations with maximum true relative residual `1.99854e-8`; all ten Newton
steps converged in two corrections with maximum final L2 `1.41612e-9` and
maximum component `8.46230e-6`.  Gas mass and angular momentum changed by
only `+8.56e-9` and `+1.64e-7` relative.  Slurm maximum step RSS was
33,225,768 KiB (31.7 GiB).  The complete c930 restart SHA-256 is
`72a40625f144808b249f766f1b944e0c348e2ab549b556a12bba3a54d5d0ea53`.

The wrapper stopped after the segment because XD compute-node Python lacks
`h5py`, not because the accepted state failed its audit.  All ten in-solver
accepted-state guards passed, the 16 transient out-of-domain lookups were all
clamped, and no opacity result was negative, zero, or nonfinite.  Independent
audit of the c930 HDF found all 2,383,872 cells inside the opacity and EOS
domains, with positive gas energy.  Its ranges are
`rho=2.874e-20--8.501e-14 g cm^-3` and `T=8.384--22.063 K`; energy closure is
`1.00e-7` median, `1.44e-6` p99, and `4.21e-6` maximum.  HDF SHA-256 is
`cabe8045da89763e4be902610353ed7b123c7c983979698579542ea2ba92a01a`.

The audit now has an XD-native `h5dump` backend and no longer requires Python
HDF5 packages on compute nodes.  It reproduced the `h5py` audit exactly on
the c930 file.  Five-node M-large-a job `342848` resumes from c930 and is
pending resources, with scheduler-estimated start at 10:37:23 JST.

Job `342848` started at 10:33:15 JST.  Its first production segment completed
c930--940 and the new compute-node-native audit passed: all 2,383,872
post-regrid cells are finite, positive, and inside both opacity axes
(`T=5.106--21.709 K`).  The wrapper recorded `REASON=CONTINUE` and proceeded
automatically to c940--950.  This confirms that the c930 stop was solely the
missing Python package and that the replacement audit works in the allocated
compute environment.

## c960 nonlinear localization and CFL recovery test

Job `342848` safely completed and audited c930--960 before failing while
attempting cycle 964.  The completed 30 steps used 60 strict FGMRES solves at
32--718 iterations; no inexact result was accepted.  The run changed gas mass
by `+2.66e-8` and angular momentum by `+3.02e-7` relative through c960.
Maximum Slurm step RSS was 37,949,840 KiB (36.2 GiB).

The c960 HDF has 2,412,544 cells and zero opacity-domain, EOS-domain, or
nonpositive-gas-energy violations.  Its energy-closure relative error is
`1.01e-7` median, `1.43e-6` p99, and `4.31e-6` maximum.  HDF SHA-256 is
`28f41a17db08796233acb2039a3e2fa3085e78f630c7b1fbb69188f7d3574c6d`.
The complete c960 restart SHA-256 is
`d891265a29942c53e58b41daa079101c7a52fdb52cce9e6b98f42d65c3a32377`.

At cycle 964, all four FGMRES corrections met the strict `2e-8` normwise
target in 628, 605, 576, and 576 iterations.  Their local stencil backward
errors nevertheless grew from `6.36e-6` through `5.43e-4` and `7.21e-2` to
`1.0`.  Newton stopped after four corrections at L2 `6.41e-5`, with one
total-energy component still at normalized residual `1.0`.  This is the same
localized strong-field coarse/fine mode seen at c880, not an opacity-domain
failure and not a license to use an inexact acceptance gate.

Five-node M-test job `342919` replays the exact c960 checkpoint through c966
at CFL 0.075 with strict FGMRES and per-cycle output.  It is pending the
per-user M-test QOS slot; no duplicate recovery was submitted.

Recovery `342919` passed c960--966 in 132 s solver time (138 s Slurm).  All
12 strict FGMRES solves converged in 19--22 iterations, versus 296--628 near
the CFL-0.15 failure, and all six Newton steps converged in two corrections.
Maximum final L2 was `5.83e-10`, maximum final component `1.69e-8`, and
maximum local stencil backward error `1.70e-6`.  Cycle 964 therefore passed
cleanly instead of stagnating.  Maximum Slurm step RSS was 34,383,604 KiB
(32.8 GiB).

The c966 post-regrid HDF has zero opacity/EOS-domain and nonpositive-gas-energy
cells.  Its energy-closure error is `1.01e-7` median, `1.43e-6` p99, and
`4.23e-6` maximum.  HDF SHA-256 is
`06787d4eb03ad931211de18ec6e3d1a56f7549d574237ef9bde2efd60a15f22c`;
restart SHA-256 is
`1d1ad46024b010eecd7b88b00d8dd0a4be640bf687a7d7000dcfb2c4fde0c690`.
The smaller CFL is faster here because it restores a well-conditioned
implicit solve; it is not merely paying twice as many equally expensive
steps.

Five-node M-large-a continuation `342930` starts from the verified c966
checkpoint with CFL 0.075.  It is pending the per-user QOS slot.
