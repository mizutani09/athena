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
rotation and three-node weak-field high-AMR jobs are `340791` and `340792`.
Together they request five nodes. Each ten-cycle segment records the restart,
elapsed time, MeshBlock count, density, Jeans resolution, temperature, mass,
angular momentum and magnetic divergence. They stop at the first-core
density candidate (`rho_max≥1e-10 g cm^-3`), Jeans minimum below 16,
opacity-domain issue, solver failure, target time, or wall budget. Their
status and `progress.csv` are under `high_amr_long/` on XD. The 129 μG
strong-field case remains open separately from this weak-field run.

`collapse_rotmhd_pilot_20261116.py` repeats the local gate;
`collapse_rotmhd_energy_audit_20261116.py` checks HDF energy closure;
`xd_mtest_fourway.sbatch`, `xd_continue.sbatch`, and
`xd_high_amr_segments.sbatch` define the XD gates and continuation.
