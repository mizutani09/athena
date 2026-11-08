# WOR-92: high-resolution inexact Newton stopping audit

Date: 2026-11-09 JST. This is an isolated **experimental** branch, not a
production default. The source baseline is `a718ee47568096db60df35cee94ad4387b48edb4`
on `codex/wor92-xd-restart-sync-20261107`. The only solver change is an opt-in
FGMRES restart-boundary stopping rule in `src/linear_solver/linearMG/linearMG.cpp`.
The XD trial source file SHA-256 is
`44e7b5262181c538880b7cdf0b45c55209271b7886e33027bb13980e419bd1c8`;
the trial binary SHA-256 is
`ee22f78590c003827911a95192c116c6115649bb38252f5114b66b537200d7df`.
The unchanged source/binary with the new environment variable absent reproduces
the prior true residual at cycle 28. The new switch is
`ATHENA_WOR92_INEXACT_STAGNATION_RATIO=0.9`; it is active only together with
`ATHENA_WOR92_INEXACT_TRUE_CAP=1e-7`. After a restart, it stops if the
independently recomputed true residual is within the cap and has decreased by
less than 10% relative to the preceding restart. The ordinary strict
`2e-8` target remains available. No solve with a true residual above the cap
is accepted by this switch.

This rule is a **bounded pilot heuristic**, motivated by the inexact Newton
condition, not a proof of it. The implemented Schur residual is not the same
norm as the combined nonlinear residual used by the Newton driver. See
Eisenstat and Walker, [*Choosing the Forcing Terms in an Inexact Newton
Method*](https://epubs.siam.org/doi/10.1137/0917003), for the mathematical
criterion. Physical accuracy and AMR conservation must be assessed separately.

## Controlled P-debug results

All tests use the saved XD 960-MeshBlock Tomida/Semenov CFL=0.3 state,
conservative AMR FLD and variable limiter, and the same trial binary.
The cycle-28 one-step pair starts from job 337788's `.00019.rst`; the cycle-29
pair starts from job 337871's exact final restart. The seven-cycle pair
starts from the same cycle-28 restart. Each pair has the same physics, rank
count and solver inputs except the stated stopping rule or tolerance.

| Jobs / comparison | Control | Trial | Result |
| --- | ---: | ---: | --- |
| 337871/337872, cycle 28→29, 64 ranks | 600 iterations in the second solve; 35 s | 81; 13 s | Both Newton 2 iterations, final max `7.50e-6`/`7.93e-6`; trial true residual `4.55e-8` |
| 337874/337875, cycle 29→30, 64 ranks | 600; 34 s | 81; 13 s | Both Newton 2 iterations, final max `7.75e-6`/`7.90e-6`; trial true residual `3.92e-8` |
| 337877/337878, cycle 28→35, 64 ranks | 14 linear solves, 3049 total iterations; 2:29 | 14 solves, 997 iterations; 0:59 | All 7 Newton steps converge in 2 iterations each; no opacity-domain event |
| 337879, cycle 13, restart 40/MG2/300, 64 ranks | prior failure `8.288e-7` | same failure and true residual | Correctly refuses a result above the `1e-7` cap |
| 337876, cycle 28 with exact local gas **off** | on: 2 Newton iterations | off: 3 iterations | Local gas root remains useful at this XD state; on logs have 512 exact-root cells per sampled block and local relative matter residual about `1.4e-13` |
| 337890, cycle 28→29, 32 ranks | 64-rank trial: 81 iterations in last solve | 32-rank trial: 84 | Both Newton 2 iterations and reach cycle 29; max final radiation-state difference `2.65e-11` relative |

The seven-cycle double-precision restart comparison shows maximum active-cell
relative differences of `3.01e-11` in density, `4.67e-11` in hydro total
energy, and `9.55e-11` in radiation energy. The matched final times differ
by `1.51e-10` code units. The trial and control combined hydro+radiation
energy integrals differ by `5.67e-11` relative. These are **trajectory
differences**, not an absolute conservation certificate. Density and momentum
were bitwise identical in the one-step cycle-28 pair; its hydro energy and
radiation energy differed by at most `3.83e-14` and `1.40e-13` relative.

The 64-rank direct-log energy/face audit used the same source and start
state with diagnostic flags only (jobs 337881/337882); each job has all
64 rank ledgers and all 960 unique block faces. The logged Newton and FGMRES
lines are identical to their uninstrumented 337871/337872 counterparts.
The global signed implicit energy-balance fractions are `5.15e-10` and
`4.14e-10`; the maximum rank absolute-term fractions are `5.99e-7` and
`5.96e-7`. All 816 same-level face groups per direction match exactly.
Each direction has 32 coarse-fine groups, with weighted relative mismatches
about `6.14–6.24e-7` in the control and `6.28–6.63e-7` in the trial.
The independent face sum closes with the energy-ledger interior sum within
`4.2e-8` code-energy. Thus this stopping rule does not remove the existing
non-roundoff coarse-fine flux mismatch; its scientific impact remains open.

## Inner-tolerance and preconditioner controls

The plateau rule is selective. In the cycle-13 failing solve, the true
residual at restart boundaries decreased by factors around `0.26–0.34`
through 280 iterations, and was still `8.29e-7` at 300. There is no
stagnation to exploit there. On the direct cycle-13 restart, strict
restart-40/MG3 (337889) converged in 3 Newton / 579 FGMRES iterations
and 43 s; the earlier strict restart-80/MG2 control (337790) took 28 s.
Neither is a demonstrated underlying AMR hierarchy repair. The existing
Galerkin option rejected both trial configurations before solve: it requires
constant prolongation and currently supports uniform MeshBlock topology only
(337887/337888). No AMR Galerkin implementation was attempted.

Relaxing the inner target unconditionally is also not an automatic speedup:

| Same cycle-28→35 checkpoint | FGMRES solves/iterations | Newton iterations per step | Elapsed |
| --- | ---: | --- | ---: |
| Strict `2e-8` plus bounded cap, 337877 | 14 / 3049 | 2 each | 2:29 |
| Bounded plateau stop, 337878 | 14 / 997 | 2 each | 0:59 |
| Fixed inner target `1e-5`, 337883 | 21 / 909 | 3 each | 0:55 |
| Fixed inner target `1e-4`, 337884 | 25 / 910 | 3–4 | 0:55 |

All seven steps converged at the configured Newton L2/max target `1e-5`,
but the fixed loose targets required extra nonlinear solves. On the direct
cycle-13 restart, `1e-5` (337885) took 5 Newton / 774 FGMRES iterations
and 41 s; `1e-4` (337886) took 7 Newton / 768 iterations and 41 s.
This is evidence against adopting a single loose inner tolerance as the
final solution.

## Later-state and timestep checks

The cycle-35 continuation with the trial rule and MG2 (337892) passed the
cycle-45 step, then stopped in the **cycle-46 step**: after
600 iterations its independent true relative residual was `1.16660672e-7`,
above the explicit `1e-7` cap. A byte-for-byte rerun of the same script
(337908) reproduced the same failed residual. The many late restarts stayed
near `1.16e-7`, so this is a new precision plateau **outside** the tested
inexact budget. The solver rejected it as intended.

From the saved cycle-45 checkpoint, an MG2 restart (337901; replay 337906)
converged **only the cycle-45 step** in two Newton iterations, with second
linear solve true residual about `3.97e-8`. Both replays had the same FGMRES
iteration counts (32, 81); they do not test the failing cycle-46 step.
The MG3 one-step restart 337895 and its continuations 337898/337903/337913
reached cycle 95, with two Newton iterations on every physical step and no
opacity-domain failure. The last history row has maximum density about
`1.61e-17 g cm^-3`, still far from first-core density. These are useful
continuation controls, **not** a proof that MG3 repairs the underlying
hierarchy or that an entire collapse is robust.
At cycle 45, strict baseline 337897 and experimental-binary opt-out 337900
agreed at true residual `4.21612292e-8`; the diagnostic 337896 measured a
correction representation gap `3.67825450e-8`. This trace follows the
strict/restarted trajectory, not the failing continuous MG2 trajectory.

Output-cadence controls give a cleaner result than the first one-step replay
suggested: periodic output every five cycles (337892, repeated 337908),
primary HDF5 every cycle (337911), restart every cycle (337912), and both
every cycle (337914) all reproduced the **same cycle-46 failure** and
`1.16660672e-7` residual. The cycle-45-to-46 one-step output test 337905
stopped before the failing step. Its cycle-46 checkpoint and 337914's
cycle-46 checkpoint have bitwise identical serialized MeshBlock data; only
two parameter-header bytes differ. Output cadence is therefore ruled out as
the cause. A direct restart from that cycle-46 checkpoint also failed with
MG2 (337916, true residual `1.16299112e-7` after 600 iterations) and passed
with MG3 (337917, `4.98926e-8`, Newton two iterations, final maximum
nonlinear residual `8.38e-6`). The failure is reproducible across continuous
and restarted trajectories and is preconditioner sensitive.

The direct cycle-46 strict step trace (337918) reproduced the failed MG2
residual. At 600 iterations the Krylov projection predicted `9.38e-9`, but
the independently evaluated true residual was `1.16299112e-7`. Its measured
operator gap was `1.16080618e-7`, dominated by a `1.14193686e-7` correction
representation gap; the operator evaluation gap was `1.99880101e-8`.
The requested last step norm was `2.05e-13` and its realization error
`1.37e-18`; the issue is accumulation/representation through the operator,
not a missing update. This is direct evidence for the double-precision
representation mechanism in the actual failing step.

As a **diagnostic only**, the same MG2 cycle-46 checkpoint with a `2e-7`
true-residual cap (337922) stopped at iteration 81 with true residual
`1.17209e-7` and converged Newton in two iterations; final nonlinear max
`8.32e-6` versus MG3/`1e-7`'s `8.38e-6`. Their final radiation-energy
integrals differed by `4.98e-10` relative and combined-energy integrals
by `2.91e-10`. This shows that the particular inner `1e-7` gate is tighter
than the measured state difference requires at this checkpoint. It does
**not** validate an unconditional cap increase at later states. CFL=0.15
MG2/MG3 both passed two substeps to the same time (337920/337921); their
combined-energy integrals differed by `1.67e-11`, while CFL=0.3/MG3 versus
CFL=0.15/MG3 differed by `9.61e-4`. The timestep effect dominates these
linear-solver differences by many orders of magnitude.

Matched-time tests from the same cycle-45 restart used the same MG3 solver
and ended at `t=597.24740576715` with 1, 2, 4, or 8 substeps. All Newton
solves converged. The relative integral differences in radiation energy
between successive resolutions were `1.6447e-3`, `9.9533e-4`, and
`4.5815e-4`; for combined hydro+radiation energy they were `9.7917e-4`,
`5.9348e-4`, and `2.7467e-4`. The differences shrink as the timestep is
halved, but the coarse CFL=0.3 result is still materially different from
the finer integrations. This is an accuracy check, not a conservation test.
At CFL=0.15, MG2 versus MG3 changed the combined-energy integral by only
`5.51e-12` relative, so the timestep differences cannot be attributed to
this preconditioner setting. Local maximum density differences remain near
3% across adjacent timestep comparisons; shock/displacement and AMR
sensitivity need dedicated spatial diagnostics before assigning an order of
convergence to that pointwise statistic.

## Remaining gate

The pilot has reached only an early, low-density collapse state. Before
making this the default or launching a long first-core run, continue the
same-checkpoint tests through later AMR/regrid states, establish a matched-time
CFL and resolution error budget, and decide whether the coarse-fine flux
mismatch and Newton `1e-5` margin are acceptable. Test opacity-table trial
values and accepted-state domain separately. A forcing term tied to actual
Newton progress is the principled next solver candidate; the present
restart-ratio trigger is a measured shortcut for the observed precision
plateau. If later states exceed the true-residual cap, reject the step and
investigate or retry with a different timestep/preconditioner. Do not silently
increase the cap.

Exact batch scripts, `experimental.patch`, `sacct.txt`, double-precision
restart comparison JSON, 3D face audit JSON and the audit scripts accompany
this report. The compact launcher/provenance/history archive is
`wor92_adaptive_inexact_jobs_337892_337922.tar.gz` (SHA-256
`696aafe44f7aa2cbba755f09a49ee1daf1ce1078a10f82c7edfdab1734dae2b7`).
The full restarts remain at their immutable XD job paths and in the isolated
local evidence directory. No main merge or default change.
