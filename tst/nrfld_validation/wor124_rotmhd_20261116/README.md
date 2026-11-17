# WOR-124: rotating and magnetized table-EOS collapse pilot

This experimental branch starts at `protostellar-firstcore-v1`
(`6e658312bb504a90b2e385604193213c2bb07702`), whose solver source tree
was used for the validated nonrotating, nonmagnetic first-core run. The
existing worktrees and `main` are not modified.

## Scope of the code change

The frozen source has the general-EOS MHD energy bug documented in WOR-112:
its FLD exchange read `E_total-E_kinetic` as gas energy and wrote back
`E_gas+E_kinetic`, dropping `B²/2`. We copied the minimal `MagneticEnergyAt`
logic from [WOR-112 commit c79e78a8](https://github.com/mizutani09/athena/commit/c79e78a8e92cbd6eaefe7bed460fd5511b11aed0)
into `src/fld/fld.cpp`. The problem generator's table-EOS+B guard remains
closed unless `problem/enable_tabular_mhd_pilot=true` is explicitly set.
This is an experimental ideal-MHD calculation, not a non-ideal MHD model.

The physical initial sphere matches the prior table run: 1 Msun, 3000 au,
14 K. Optional initial fields are solid-body `Omega=1.48e-13 s^-1` and
uniform vertical `B0=129 microG`, values used in [Mayer et al. (2025),
Sec. 6.2](https://arxiv.org/html/2510.12620v4). The resulting rotation
energy ratio is beta=0.004973. Our EOS, opacity, Cartesian mesh, and ideal
MHD differ from Mayer's non-ideal RMHD calculation.

## Local 2-rank, one-cycle gate

All cases use the same 32³ root, adaptive `numlevel=2`, CFL 0.3,
table EOS, Semenov table opacity, and original FGMRES 40/300 with MG2.
The input is `input_table_mhd_pilot.in` with only the explicit pilot key
added to the frozen input. Command/environment, table hashes, full logs,
resource logs, restart files, HDF5, and per-case JSON live under
`/home/kosuke/simulation/test/collapse_rotmhd_pilot_v2_20261116/`.

| Case | One-cycle result | Wall s | Initial dt, code units | Max FGMRES iterations | Max true relative residual | Bmax G | divBmax |
|---|---|---:|---:|---:|---:|---:|---:|
| Neither | PASS | 68.8 | 31.1148 | 207 | 5.73e-8 | 0 | 0 |
| Rotation | PASS | 83.1 | 31.1148 | 236 | 4.18e-8 | 0 | 0 |
| Magnetic field | PASS | 71.9 | 10.5395 | 118 | 1.91e-8 | 1.310e-4 | 2.96e-18 |
| Both | PASS | 127.2 | 10.5395 | 186 | 1.77e-8 | 1.311e-4 | 3.24e-18 |

The output history has identical gas mass (to the printed precision) for all
four cases. In the magnetic-only and combined HDF outputs, all 61,440 active
cells have positive gas energy. The maximum relative closure error in
`E_total=E_gas+E_kinetic+E_magnetic` is `2.56e-6` and `2.57e-6`, respectively,
at HDF single precision. Without subtracting magnetic energy, the median
error is about 17.3 times the local gas energy because the dilute ambient is
magnetically dominated. The local binary SHA256 was
`bf9c68b7fac5381bb3c36e380affaf1d9070a8bc72cfe412a822358694ec6f8b`.
These one-cycle wall times are a *pilot*, not full-collapse throughput;
the magnetic field already shortens the initial timestep by 2.95×.

## XD solver gates and limits (2026-11-16)

The XD Intel/MPI/HDF5 binary SHA256 is
`f4ba920bb57839a0849b922393062847e708627226a3915f5b2b6cc6a877c588`.
The source tree staged on XD is `cff00992b14e5907d52dc22d8197b7fe98a0625f`.
All jobs use 64 MPI ranks on one node unless noted; source, inputs, tables,
full logs, resources and HDF outputs are under
`/work/mizutaniks/output/fld/collapse/wor124_rotmhd_20261116/`.

- The 129 μG field with and without rotation passed three cycles on XD
  (`340724`, `340725`) but both stopped in the fourth when FGMRES reached
  600 iterations; true relative residuals were `5.52e-5` and `1.85e-5`,
  respectively. Restarting the same state reproduced the latter failure.
  Reducing CFL to 0.1–0.2 cleared that one step but did not sustain later
  steps. Five MG V-cycles or a 160-vector Krylov restart also helped at
  individual steps without providing sustained convergence. These are
  **unsolved strong-field pilot conditions**.
- The initial 129 μG field shortens the CFL 0.3 timestep from 31.11 to
  10.54 code units (factor 2.95), before its additional linear-solver cost.
  One code time unit is 14.991 years. Our field is *ideal MHD*; Mayer's
  §6.2 calculation uses non-ideal MHD.
- The opt-in FGMRES absolute-norm diagnostic at equal initial timestep
  found a combined-case second Newton right-hand-side L2 norm of `0.11477`
  and a 600-iteration absolute residual of `5.81e-6` (relative `5.06e-5`).
  Setting the FGMRES target globally to `1e-4` made Newton exhaust its
  backtracking immediately (`340770`), so this relaxation was rejected.
  The diagnostic-only source edit was reverted; the separate binary and
  logs remain under `source_absdiag/` and `pdebug_abs_B/` on XD.
- Halving B0 to 64.5 μG still stopped before cycle 30 (`340758`). At
  B0=32.25 μG (one quarter of Mayer's field strength), the
  rotation+field case reached cycle 30 (`340773`) with CFL 0.3, FGMRES
  160/600 MG3 and bounded inexact cap `1e-6`: 117 s, 63 linear solves,
  maximum 600 iterations, maximum accepted true relative residual
  `1.30e-7`, all 30 Newton solves converged, and Slurm MaxRSS 4.86 GB.
  Gas mass changed by `-3.2e-15` relative, `divB_max=6.34e-18`, EOS
  outside cells 0, negative gas-energy cells 0, and HDF total-energy
  decomposition error at cycle 30 was at most `3.92e-7` across 61,440 cells.
- Rotation alone passed cycle 5 with the original 80-vector restart, then
  stalled at cycle 6 (`340756`). A 160-vector restart from that checkpoint
  reached cycle 30 (`340772`, 98 s for 25 cycles, all 26 active Newton
  solves converged). Its cycle-30 Jeans minimum was 25.19 cells; the weak
  field case's was 27.23 cells. The two conditions were then continued
  at increased AMR resolution.

The matched 30-cycle XD run `340796` used one M-test-a node, 64 MPI ranks,
one binary, CFL 0.3, 160/600 MG3, and cap `1e-6`. All three cases completed,
with 30 converged Newton solves each. Their largest final Newton component
residuals were `4.84e-6`, `4.51e-6`, and `3.80e-6` (requirement `1e-5`).
This is an **early-collapse, level-3 performance comparison**, where every
case still has more than 24 cells per Jeans length.

| Case | B0 | Wall for 30 cycles | Simulated code time | Wall per code time | Sum/max FGMRES iterations | Max true relative residual |
|---|---:|---:|---:|---:|---:|---:|
| No rotation, no B | 0 | 83 s | 830.18 | 0.100 s | 5953 / 208 | 6.92e-8 |
| Rotation only | 0 | 130 s | 810.47 | 0.160 s | 9780 / 590 | 8.03e-8 |
| Rotation + weak B | 32.25 μG | 116 s | 630.03 | 0.184 s | 8926 / 600 | 2.45e-7 |

Thus rotation costs 1.57× wall time per 30 cycles, and the weak-field
combined case costs 1.40×, relative to the matched baseline. Per *simulated
physical time* the factors are 1.60× and 1.84× because the magnetic CFL
step is shorter. These ratios describe this early, low-resolution setup;
they do not predict first-core throughput after AMR growth.

**Resolution correction.** The short comparison input allows only two
additional AMR levels (actual maximum level 3). Low-resolution exploratory
jobs `340775` (rotation) and `340776` (32.25 μG plus rotation) completed to
code time 2100 but first fell below 16 cells per Jeans length at times
1359.53 and 1364.58. Their terminal Jeans minima were only 1.19 and 1.20.
Their later physical states are **not valid first-core results**; the jobs
provide only solver/performance evidence before that resolution limit.

Starting instead from the cycle-30 checkpoints (Jeans minima above 25),
`mesh/numlevel=11` allowed both cases to reach cycle 40 (`340787`, `340788`).
Each had 848 leaf MeshBlocks, actual maximum level 4, and Jeans minima
46.26 and 47.19. The rotation continuation took 199 s and used 10.79 GB
Slurm MaxRSS; the weak-field continuation took 142 s and used 10.95 GB.
The latter's cycle-40 HDF audit found no negative gas energy or EOS-domain
cells among 434,176 cells and at most `4.56e-7` energy-decomposition error.
Its user-history gas mass fell by `5.70e-7` relative over ten steps;
`divB_max` stayed below `1.36e-17`. These are measured budgets to follow,
not a claim of exact AMR MHD conservation.

The one-node segmented runner passed its cycle-40→50 smoke gate (`340793`):
848 MeshBlocks, Jeans minimum 49.69, no opacity-domain errors. The two-node
rotation high-AMR job is `340791`. The three-node weak-field job `340792` was
cancelled before starting after the 129 μG combined case passed its high-AMR
cycle-20 gate; the strong-field combined long job `340822` now has priority.
Together the rotation and strong-field jobs request five nodes. Each ten-cycle
segment records the restart,
elapsed time, MeshBlock count, density, Jeans resolution, temperature, mass,
angular momentum and magnetic divergence. They stop at the first-core
density candidate (`rho_max≥1e-10 g cm^-3`), Jeans minimum below 16,
opacity-domain issue, solver failure, target time, or wall budget. Their
status and `progress.csv` are under `high_amr_long/` on XD. The 129 μG
strong-field case remains a guarded exploratory ideal-MHD calculation, not a
non-ideal MHD reproduction of Mayer et al.

### 129 μG high-AMR gate and long job

With `mesh/numlevel=11`, B0=129 μG, CFL 0.3, FGMRES 160/600 with MG3,
and bounded inexact cap `1e-6`, the magnetic-only case passed cycle 10
(`340807`, 566 s). The rotation+magnetic case passed cycle 10 (`340810`,
722 s) and cycle 20 (`340815`, 506 s for cycles 10–20). The low-AMR
cycle-4 failure did not recur in either high-AMR case. The combined run's
first ten active Newton solves converged; its 30 FGMRES solves used 6645
iterations in total, at most 514 in one solve, and true relative residual
at most `2.00e-8`. At cycle 10, the HDF audit of 491,520 cells found no
negative gas energy or EOS-domain cells and at most `2.58e-6` energy closure
error at single precision. Its maximum magnetic-to-gas-energy ratio was
18.73, illustrating why the MHD energy-closure correction is needed.

The combined cycle-10→20 gate ended with 960 MeshBlocks, level 4, Jeans
minimum 60.52, `rho_max=5.28e-18 g cm^-3`, gas mass stable to history
precision, `divB_max=3.83e-17`, and no opacity-domain or solver errors.
Its 30 FGMRES solves used 4617 iterations in total, at most 199 in one
solve, and true relative residual at most `2.00e-8`. This is an early
collapse gate; it does not establish behavior at first-core density.

Job `340822` continues the combined case from the verified cycle-20
checkpoint on three M-large-a nodes and 192 MPI ranks for at most 24 h.
It uses the same guarded ten-cycle segments and `mesh/numlevel=11` as the
rotation job, allowing direct comparison of their high-AMR progress and
resources. The script's `WOR124_INITIAL_CYCLE=20` check prevents an
accidental start from another restart. The queued start estimate was
2026-11-16 18:04 JST. The rotation-only job had reached cycle 300,
actual level 10, Jeans minimum 31.96, and `rho_max=3.10e-14 g cm^-3`
without a solver failure by the time of this submission.

At submission, a three-node P-large-a 24-h job was estimated to begin
2026-11-17 03:58 JST, whereas M-large-a was estimated to begin the same
evening. M-large-a's per-node memory limit is 117,760 MB; the measured
one-node cycle-10 combined job's Slurm step MaxRSS was 11.72 GB. The
long jobs record memory in Slurm accounting for later review. Pending
P-debug continuations `340820` and `340821`, the magnetic-only M-test
continuation `340811`, and the redundant weak-field long job `340792`
were cancelled before execution to avoid duplicate resource use.

## Isolated table-MHD temperature-floor trial

The source-only trial on branch `exp/wor124-tabular-energy-floor-20261116`
(commit `48d4ff14`) changes `src/eos/general/general_mhd.cpp`. This MHD EOS
path is used even by the zero-field rotation case because the executable was
built with `-b`. Previously it did not read `hydro/tfloor`; when a conservative
update drove gas energy below the fixed `efloor≈1e-10`, adding that floor to
kinetic energy around `6e5` could leave only a floating-point ULP. FLD's
slightly different kinetic arithmetic then measured zero gas energy. The
trial uses the configured table-EOS temperature floor (`5.02 K`) for such
cells, with an `8ε` roundoff guard, and applies the same temperature bound to
other table-MHD cells. The corresponding table minimum at the failure
density `rho≈56610.7` is about `1.47e4` code energy units.

The trial was built separately on XD under `source_floor_trial/` (binary
SHA256 `2a8b93e1dbfbeb82f03660407f883cd472b8e6db1bded58fefe5af575ea8955c`);
the original XD source and binary were not overwritten. The two-code
comparison used the *same* cycle-770 restart, CFL 0.1, 64 MPI ranks, 160/600
FGMRES with MG3 and cap `1e-6`:

| Job | Source | Result | Last simulated time |
|---:|---|---|---:|
| `340996` | original | FAIL after cycle 773, `FLD::LoadHydroVariables` gas energy 0 | 1950.265 |
| `340995` | temperature-floor trial | PASS cycle 775 | 1950.277 |
| `340998` | temperature-floor trial, continued | PASS cycle 785 | 1950.336 |

The original failed near time 1950.296 in its 128-rank long run. The trial
passed beyond that time. The 64-rank original and trial logs have identical
Newton final residuals and FGMRES iteration/residual histories through cycle
773; the change acts at the failing update rather than perturbing the prior
trajectory. In the trial's cycle-785 HDF output, all 1,810,432 active cells
have positive gas energy and lie inside the EOS domain. Its maximum HDF
single-precision energy-decomposition error is `8.98e-5`, the same scale as
the original cycle-760 HDF (`8.98e-5`). The trial's cycle-775→785 history
has relative gas-mass change `4.4e-16`, Jeans minimum 32.0, no opacity-domain
errors, and every active Newton solve converged.

Local two-rank one-cycle rotation and 129 μG combined pilots both passed with
the trial and retained their original largest FGMRES iteration counts (236
and 186). Sequential local rotation wall times were 82.93 s for the original
and 80.46 s for the trial; combined wall times were 100.04 and 99.64 s.
The respective density, temperature, mass, magnetic-field and linear-residual
outputs match. The trial's combined HDF has zero negative-gas or EOS-domain
cells and a maximum energy-decomposition error of `2.57e-6`, matching the
original output at single precision. This short gate shows no measurable
slowdown.
The original-CFL 0.3 test passed c760→770 with both 64 MPI (`341002`,
188 s) and the original 128 MPI (`341000`, 130 s). Both reached t=1950.367,
beyond the original failure time t≈1950.296. The 128-MPI run had 12/12
converged active Newton solves, 30 FGMRES solves totaling 297 iterations
(maximum 10), no opacity errors, Jeans minimum 32.0025, and unchanged
gas mass at history precision. The 64- and 128-rank density and temperature
agree to better than 2×10⁻⁹ relative at c770. The exact-rank CFL 0.1 test
`340992` passed c770→780 (138 s, t=1950.307): 12/12 active Newton solves
converged, 30 FGMRES solves totaled 279 iterations (maximum 10), mass was
unchanged at history precision, and Jeans minimum was 32.006.

The candidate's 2-node 128-MPI high-AMR continuation `341013` started from
the verified c770 restart with original CFL 0.3 and is guarded in 10-cycle
segments. It passed c790 with `rho_max=7.04e-12 g cm^-3`, Jeans minimum
31.97, 3536 MeshBlocks, no opacity errors, and gas mass unchanged at
history precision. Its c790 HDF snapshot has 1,810,432 positive-gas-energy
cells, zero EOS-domain cells, and maximum single-precision energy-closure
error `8.97e-5`. Its provenance records the candidate MHD source SHA256
`fc526cb4545b11deccd40923d0cf1afbbaea7f8a30a4bb09b8e29865dceaa968`
and binary SHA256 above. No first-core claim is made until its full output
is audited. The trial remains isolated from the frozen solver and main.

The strong-field FGMRES failure is a separate issue: MG5 job `340976`
lowered the cycle-55 true residual from `8.965e-6` to `2.610e-6` at 600
iterations but still exceeded cap `1e-6`. The original-binary 192-MPI
cap-`1e-5` test `340980` also failed at cycle 55 (residual `1.01865e-5`).
After the 32.25 μG combined case passed its c40→50 AMR11 smoke gate,
candidate-code guarded long job `341015` was queued on three M-large-a
nodes to measure magnetized evolution beyond the strong-field blocker.

`collapse_rotmhd_pilot_20261116.py` repeats the local gate;
`collapse_rotmhd_energy_audit_20261116.py` checks HDF energy closure;
`xd_mtest_fourway.sbatch`, `xd_continue.sbatch`, and
`xd_high_amr_segments.sbatch` define the XD gates and continuation.

## High-AMR long-job review (2026-11-16)

The two long jobs **did not reach the first-core density candidate**. Their
last complete checkpoints and failure segments are retained on XD under
`high_amr_long/`; a local offline HTML review with progress plots and the
failure logs is at
`/home/kosuke/simulation/test/collapse_rotmhd_20261116/long_review_20261116/index.html`.

| Case/job | Last complete cycle | Failed segment | Stop | Job elapsed | Slurm step MaxRSS |
|---|---:|---:|---|---:|---:|
| Rotation, 2 nodes/128 ranks, `340791` | 760 | 760–770, after cycle 766 | `FLD::LoadHydroVariables`: conserved `E_total-E_kinetic=0` in four reported cells | 1:35:34 | 33.8 GB |
| Rotation + 129 μG, 3 nodes/192 ranks, `340822` | 50 | 50–60, near cycle 55 | FGMRES 600 iterations, true relative residual `8.965e-6` > inexact cap `1e-6` | 0:10:03 | 13.9 GB |

The rotation-only run reached `rho_max=6.90e-12 g cm^-3`, `Tgas_max=96.95 K`,
level 12, 3368 MeshBlocks, and a Jeans minimum of 32.0 at its last complete
checkpoint. Its active Newton solves all converged (832 in the segment
logs), and 1722 FGMRES solves used 22389 iterations in total, at most 161
for any solve. Over the long run the user-history gas mass changed by about
`-5.6e-16` relative, and angular momentum by `+2.36e-4`. The cycle-760
HDF output has positive `E_total-E_kinetic` in all 1,724,416 active cells;
the four failed cells acquire zero inferred gas energy during subsequent
updates. This is a local energy-state failure, not a Jeans-resolution or
memory limit.

The strong-field combined run reached `rho_max=5.41e-18 g cm^-3`, level 4,
960 MeshBlocks, and a Jeans minimum of 59.8 at its last complete cycle 50.
Its gas mass remained constant to history precision, and the largest
recorded `divB` was `6.33e-17`. In the failure segment, several successful
FGMRES solves already needed more than 500 iterations before the cycle-55
solve stopped at 600. The high-AMR early gate therefore did not establish
sustained strong-field convergence.

The original MHD EOS energy floor in this input is `1e-10` code units, while
the table's minimum gas energy at the reported failure density
`rho≈56610.7` is about `1.47e4` code units. The difference in multiplication
and division ordering between `ConservedToPrimitive` and
`FLD::LoadHydroVariables` can also move a kinetic-energy estimate by one
floating-point ULP, enough to turn an energy-floor-sized difference into
zero. This provides a concrete candidate mechanism, but the exact update
where the four cells lose internal energy has not yet been instrumented.
The completed short probes and the isolated MHD EOS trial are recorded above;
the guarded long continuation is still required before claiming sustained
collapse success.

## Plasma-beta interpretation (2026-11-16)

Here plasma beta means `beta_gas=8*pi*Pgas_cgs/B_G^2`, not the rotational
energy ratio `Erot/|Egrav|=0.004973` above. The table below was measured from
active-cell HDF snapshots. The dense-core mask requires the background
tracer to be below 0.5 and `rho>=1e-18 g cm^-3`; the global minimum includes
the dilute ambient. The analysis code and JSON are under
`/home/kosuke/simulation/test/collapse_rotmhd_20261116/plasma_beta_20261116/`.

| Field / cycle | Global minimum beta | Dense-core minimum beta | Beta at density maximum |
|---|---:|---:|---:|
| 129 μG / 0 | 0.03847 (ambient) | 2.176 | 3.847 |
| 129 μG / 50 | 0.03334 (ambient) | 0.8827 | 3.813 |
| 32.25 μG / 950 | 0.2151 (dense gas) | 0.2151 | 9.549 |
| 32.25 μG / 1600 | 0.05371 (dense gas) | 0.05371 | 14.76 |

The 32.25 μG run reached beta 0.2151 in cloud material at
`rho=1.87e-13 g cm^-3`, about 8.08 au from the center. Its c950 snapshot
has 2,068,480 active cells, zero nonpositive gas-energy cells, zero
EOS-outside cells, and maximum single-precision energy-closure error
`8.96e-5`. Thus this run has already evolved magnetically dominated dense
gas, whereas the 129 μG FGMRES failure occurred while the maximum-density
cell still had beta 3.813. The failure is not explained by a single plasma-
beta threshold. The 129 μG initial field corresponds to the Mayer et al.
mass-to-flux ratio 5 setup, while the 32.25 μG run has a four-times weaker
field and is a distinct physical case. Neither beta minimum establishes
first-core completion; the guarded jobs remain under observation.
At c1600 the same weak-field run reached beta `0.05371` in cloud material
at `rho=1.01e-13 g cm^-3`. The 2,082,816-cell HDF snapshot still has zero
nonpositive gas-energy and EOS-outside cells and maximum single-precision
energy-closure error `8.99e-5`. Its audit is under
`/home/kosuke/simulation/test/collapse_rotmhd_20261117/`.

## Inexact Newton checkpoint comparison (2026-11-17)

The guarded long jobs remain on the original FGMRES target `2e-8`, with
bounded inexact true-residual cap `1e-6`. The log field `inexact_accept=1`
means the final residual is under that cap; it does not establish that the
solver stopped early. Actual early stopping requires `stagnation_stop=1`.
Across completed weak-field long-job segments, six early stops occurred in
cycles 51–85 with true residuals `2.08e-8`–`2.50e-8`, just above the ordinary
target. Every following Newton solve converged. Recent long-job segments
reach `2e-8` normally, with no stagnation stops.

To test a genuinely looser linear solve, M-test-a job `341091` replayed the
same c1580 weak-field+rotation restart for five cycles at 64 MPI ranks.
All conditions used the same trial binary and settings except the FGMRES
relative target:

| Target | PASS | FGMRES sum/max iterations | Max true relative residual | Newton | Wall |
|---:|:---:|---:|---:|---:|---:|
| `2e-8` | yes | 154 / 11 | `1.90e-8` | 6/6 converged | 116 s |
| `1e-7` | yes | 140 / 10 | `9.43e-8` | 6/6 converged | 115 s |
| `1e-6` | yes | 119 / 8 | `8.45e-7` | 6/6 converged | 105 s |

The largest final Newton max norm was about `3.30e-7` in every case,
comfortably below the `1e-5` stopping threshold. At c1585 the strict and
`1e-6` HDF snapshots have the same AMR layout; all 2,082,816 active-cell
density and magnetic-strength values agree at output precision. Nonpositive
gas-energy and EOS-outside counts are both zero, and the maximum HDF
single-precision energy-closure error is `8.997e-5` in both. Their history
mass, density maximum, temperature maximum, angular momentum, and Jeans
minimum also agree at printed precision; the integrated gas internal energy
differs by about `1e-12` relative. Short-run evidence and logs are under
`/home/kosuke/simulation/test/collapse_rotmhd_20261117/inexact_ab_job341091/`.

A 20-cycle strict/`1e-6` continuation from the same checkpoint passed on
M-test-a as job `341098`. The strict and relaxed conditions had 60 FGMRES
solves totaling 622 and 479 iterations, maximum true relative residuals
`1.98e-8` and `8.45e-7`, and 21/21 converged Newton solves in each case.
Their maximum final Newton max norms were `3.29644e-7` and `3.29659e-7`.
Wall times were 435 and 370 s, a 14.9% reduction in this matched replay;
Slurm step MaxRSS was 35.7 and 36.0 GB. Both had zero opacity errors.
The c1600 history mass, angular momentum, maximum density, temperature,
and Jeans minimum agree at printed precision; integrated gas internal energy
differs by `3.61e-12` relative.

The c1600 snapshots have the same AMR layout and 2,082,816 active cells.
The density relative L1 difference is `8.05e-12` (maximum local relative
difference `7.90e-7`); for radiation energy the values are `1.37e-10` and
`5.09e-6`. Both snapshots have zero nonpositive gas-energy and EOS-outside
cells and maximum HDF single-precision energy-closure error `8.99e-5`.
Logs, HST, HDF, and cellwise comparison are under
`/home/kosuke/simulation/test/collapse_rotmhd_20261117/inexact_sustain_job341098/`.

This establishes a promising *fixed linear tolerance* candidate over this
20-cycle interval; it is not yet an adaptive Newton forcing term, nor a
high-density whole-collapse validation. The production long jobs have not
been altered. A further exploratory `1e-5`/`1e-4` M-test job `341109` was
cancelled before execution because the shared account reached its job limit;
there is no result from those thresholds. The two reproducible scripts are
`xd_inexact_newton_ab.sbatch` and `xd_inexact_newton_sustain_ab.sbatch`.

### Resolution margin during the live 32.25 μG run

At c1780 the guarded three-node run had `rho_max=6.26e-11 g cm^-3` and a
minimum of 18.53 Jeans cells, approaching its explicit 16-cell stop limit
before the `1e-10 g cm^-3` density candidate. The recent empirical
Jeans-versus-density slope projects the 16-cell crossing near the density
candidate; this is a warning, not a demonstrated failure. A one-node
M-test-a gate `341113` restarted from the immutable c1780 checkpoint with
`mesh/numlevel=12` and passed through c1790. It created one additional
actual AMR level (13) and 4292 MeshBlocks; the Jeans minimum increased to
32.07 cells. All 13 active Newton solves converged; 30 FGMRES solves used
308 iterations in total, at most 11, with maximum true relative residual
`1.92e-8`. The ten-cycle gas-mass relative change was `-4.19e-9`, opacity
errors were zero, elapsed time 242 s on one M-test-a node/64 MPI ranks, and
Slurm step MaxRSS 44.5 GB. The c1790 HDF audit found 2,197,504 active cells,
zero nonpositive gas-energy and EOS-outside cells, and maximum
single-precision energy-closure error `8.99e-5`. Logs and HDF audit are at
`/home/kosuke/simulation/test/collapse_rotmhd_20261117/amr12_gate_job341113/`.
The higher-resolution c1790 density maximum is `6.49e-11 g cm^-3`, about
2.8% above the original-resolution branch; this is a new resolved trajectory,
not a bitwise continuation of the old physics.

The production job remains on `numlevel=11` with its guard active. A
three-node dependent rescue job `341124` runs only after original job
`341015` ends and its status explicitly says `STOP_UNDERRESOLVED`; it also
checks gate `341113` before reading the final restart and using
`numlevel=12`. If the original job reaches the first-core density candidate
or stops for another reason, this job exits without evolving gas. The
dependent script is `xd_high_amr_segments_floor_rescue.sbatch` and uses the
same isolated trial binary as the original long job.

## Final review of 32.25 μG + rotation continuation (2026-11-17)

M-large-a job `341015` completed normally after 5:00:57 on three nodes and
192 MPI ranks. Its 220 guarded ten-cycle segments advanced c50→2250 and
stopped at the explicit `FIRST_CORE_DENSITY_CANDIDATE` condition:
`rho_max=1.0118777e-10 g cm^-3`, `Tgas_max=338.34 K`, 4572 leaf MeshBlocks,
actual maximum AMR level 12, and minimum Jeans resolution **16.1278 cells**.
The 16-cell guard was not crossed, but this is a narrow margin. The
conditional AMR+1 rescue job `341124` exited with `SKIP_NO_UNDERRESOLUTION`
without advancing the gas. Density threshold alone does not prove a
hydrostatic or resolution-converged first core.

All 2200 active Newton solves converged (299 additional zero-iteration
entries were also converged). Across 6124 FGMRES solves there were 67075
iterations in total, maximum 254, and maximum true relative residual
`2.498e-8`. Six early stagnation accepts were followed by converged Newton
steps. The largest reported Newton final max residual was `9.801e-6`.
All 220 opacity diagnostics reported zero out-of-domain, clamped, negative,
zero, and nonfinite results. The final c2250 HDF contains 2,340,864 active
cells; an independent EOS/energy audit found zero nonpositive gas-energy and
zero EOS-outside cells. At single-precision output accuracy the maximum
relative `E_total=E_gas+E_kinetic+E_magnetic` closure discrepancy is
`8.976e-5` (median `1.511e-7`). The minimum gas plasma β in dense
cloud-derived material is `0.0371`, while β at the density maximum is
`10.18`. From the first complete segment c60 through c2250, history gas
mass changed by `+2.259e-6` relative and `Lz` by `+4.066e-4` relative.
The last reported `divB_max` was `2.760e-10` in code units. Slurm step
MaxRSS over the job was 33.72 GB; elapsed wall represents 15.05 node hours
for this continuation, not the cost of the c0→2250 evolution.

The c2250 density peak is near `(-0.183,-0.183,-0.183) au`. A spherical
volume-mean profile around it has central ~1 au density `8.4e-11 g cm^-3`,
temperature ~315 K, and radial velocity about `-0.04 sqrt(P/rho)`. At ~8 au
the corresponding values are ~`1e-12 g cm^-3`, ~76 K, and
`-0.92 sqrt(P/rho)`. Central inflow has slowed, but this simple profile is
not a force-balance or shock proof. The analysis and clickable HTML review
are at `/home/kosuke/simulation/test/collapse_rotmhd_20261117/status_review/`;
the final HDF, profile script, independent audits, complete progress,
provenance, restart hash, and solver summary are in sibling
`final_B32_job341015/`.

The rotation-only two-node job `341013` was still running at the final
review: its last complete c3080 had `rho_max=1.46935e-11 g cm^-3`,
`Tgas_max=199.65 K`, and Jeans minimum `30.10` cells, with no solver or
opacity failure. Its first-core candidate and final performance remain to
be reviewed. The 129 μG case remains blocked by the previously documented
FGMRES true-residual cap at c55; the weak-field result does not resolve it.

## Why the rotation-only run needs more cycles; literature comparison

As checked at 2026-11-17 09:36 JST, rotation-only job `341013` was still
running normally (c3140, `rho_max=1.4883e-11 g cm^-3`, Jeans minimum
30.05 cells). Its cycle count is not a stalled timestep artifact. At the
matched physical time of the final weak-field state, the rotation-only
c2000 HDF is at `t=1966.2441` code units and the 32.25 μG c2250 HDF at
`t=1966.2499`, only about 0.09 yr apart. Their peak densities differ by
a factor 11.07: `9.143e-12` versus `1.012e-10 g cm^-3`. The recent
rotation-only timestep is about 0.0088 code units/cycle, *larger* than the
weak-field final timestep of about 0.0060. The rotation-only peak density
fell 20.3% from c1050 (`1.0235e-11`) to c1310 (`8.1596e-12`) and then
recovered. This explains much of the extra physical evolution and cycles.

Across the first 234 complete rotation-only segments c770→3110, all 2340
active Newton solves converged; 7020 FGMRES solves used at most 15
iterations, with maximum true relative residual `2.000e-8`, no stagnation
accepts, and zero opacity diagnostic errors. c2000/c3090 HDF audits found
zero nonpositive gas-energy or EOS-outside cells among ~2.56 million cells
each. History gas mass was unchanged to output precision from c780→3100,
while `Lz` changed by `+1.455e-4` relative. Slurm step MaxRSS reached
39.2 GB. No numerical hang, nonlinear failure, or memory fault is evident.
The job has a finite 24 h wall limit and a restart every ten cycles; its
recent density growth rate, if unchanged, would not reach `1e-10` before
that limit. Collapse may accelerate, so this is only a planning estimate.

At the matched time, rotation-only density peaks at cylindrical radius
3.03 au versus 0.26 au for the weak-field run. Within 5 au, rotation-only
contains `0.00457 M_sun` versus `0.01686 M_sun` with weak B, and its
mass-weighted `v_phi/sqrt(P/rho)` is 0.624 versus 0.422. Within 10 au,
mass-weighted specific `j_z` is 15.00 versus 10.55 in common code units.
This is consistent with reduced rotational support and magnetic braking,
but a direct Maxwell-stress/torque budget has not been measured; the two
states also differ in mass distribution. The density decline is physically
plausible given the first-core bounce discussed by Mayer et al. (2025),
§6.3, but is not proven to have the same mechanism.

[Mayer et al. (2025), §§5–6](https://arxiv.org/html/2510.12620v4) use the
same initial 1 Msun, 3000 au, 14 K sphere and the same angular speed
`1.48e-13 s^-1`. Their magnetized §6 model begins with 129 μG (mass-to-flux
ratio 5) and non-ideal MHD; our completed weak-field model begins with
32.25 μG (ratio about 20 for the same geometry) and ideal MHD. Both use
Tomida-type thermodynamics, but the EOS implementations, opacity coverage,
mesh, and resolution are not identical. At `rho≈1e-10 g cm^-3`, our
temperature at the densest cell is 338 K, of the same order as Mayer's
§6 Fig. 8 track. Our 5–7 au density/inflow transition is qualitatively
similar to the 6–7 au first-core boundary described in their simpler §5
model, although §5 uses ideal EOS and constant opacity. The later §6
magnetorotational outflow and second-core comparisons are inapplicable to
our first-core density candidate. Our threshold time is 29.476 kyr versus
29.09 kyr for Mayer's simpler §5 model, a difference of 0.386 kyr (1.3%);
this is contextual because the physics differs. A local illustrated
comparison and exact HDF-derived metrics are at
`/home/kosuke/simulation/test/collapse_rotmhd_20261117/literature_review/index.html`.
