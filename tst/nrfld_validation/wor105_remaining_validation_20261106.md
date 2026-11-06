# WOR-105: remaining local validation

All runs here use `80dc6bec` on `codex/wor105-castro-face-sync-20261106`,
double-precision HDF5, MPI-enabled Test C or Mach 5 builds, and a flux
limiter frozen within each Newton solve. The temporary diagnostic changes
described below were reverted. No production default was changed.

## Test C: moving AMR and axis permutations

The radiation-only series-resistance trial used a variable (lagged) limiter,
`kappa=0.004`, density ratio 0.1, and transverse amplitude 0.3. Adaptive
refinement was allowed to change topology after cycle 2. Both one-rank and
four-rank five-cycle runs created 14 blocks and destroyed 14 blocks, ending
with four blocks. Every nonzero-time Newton solve converged. The summed
diffusion ledger and each `x=0.25,0.75` interface balance were within
approximately `2e-18` per step of zero. The maximum serial/MPI field
difference was `2.22e-16`. Restarting the serial cycle-2 checkpoint on four
ranks gave a bitwise-identical final HDF5 field and the same final time as
the uninterrupted serial run. Initialization's zero-`dt` status is not a
failed Newton solve.

The same variable-limiter series path was run with the normal direction
permuted to x2 and x3 on static AMR, one step each. Both directions converged;
their global diffusion ledgers were `-4.4e-19` and `+2e-19`. The maximum
serial/four-rank field differences were zero and `2.22e-16`, respectively.

Outputs: `/tmp/wor105_remaining_testc_v4_20261106/{series_adaptive_serial,series_adaptive_mpi4,series_split_serial,series_restart_mpi4}`
and `/tmp/wor105_axis_series_20261106`. The exact command, environment, and
input for the moving-AMR tests are saved in `command.json` and `input.used`.
After reverting diagnostics and rebuilding, a one-cycle candidate smoke test
again converged at final Newton L2 `4.18e-14` in
`/tmp/wor105_restored_smoke_20261106`.

## Coupled shared-face path

The coupled variable-limiter Test C on four ranks, with adaptive refinement,
ran five cycles. Fourteen blocks were created and 14 destroyed; all five
nonzero-time Newton solves converged, with final L2 at most `4.35e-11`.
The domain diffusion sum was within `5e-16` of zero per cycle. This test used
the shared-coefficient face path, not the series-resistance trial, because
the latter is intentionally restricted to radiation-only Test C.

The local Mach 5 radiative-shock build ran the shared-face path on four ranks
with a physical x1 boundary and moving AMR. In a three-cycle A/B comparison,
the maximum relative coarse/fine flux mismatch was `8.05e-2` with the old
operator and `3.12e-13` with the shared-face operator. Both converged.
The candidate's summed domain diffusion was `54.224812757798`, versus
`54.224812758523` through physical boundaries, a difference of `7.25e-10`
in these units. Its Newton solves used 131 FGMRES iterations in total versus
211 for the baseline over these three cycles. A separate candidate 100-cycle
run created 112 blocks and destroyed 28, ended with 92 blocks,
converged at every step, and kept the maximum coarse/fine mismatch at
`3.12e-13` over 400 recorded interface-stage entries. The cumulative
diffusion and physical-boundary totals differed by about `2.8e-9` against
a total of `5.146e6`. These are conservation and solver tests, not a
steady-shock accuracy verdict. Early A/B temperature profiles differ by up
to approximately 6 K; a same-time, resolved physical reference remains
necessary before making the path default.

Outputs: `/tmp/wor105_remaining_testc_v4_20261106/shared_coupled_adaptive_mpi4`,
`/tmp/wor105_mach5_local_20261106/ranked`, and
`/tmp/wor105_mach5_local_20261106/long_candidate`. The Mach 5 face audit
summaries are in each run's `audit/summary.json`.

## Coupled AMR restart discrepancy

The coupled Test C restart does not reproduce a continuous run. With the
original runtime limit, initialization recomputes `dt` and the restarted
run reaches a different cycle-5 time (`0.01623580` versus `0.01090244`).
`Mesh::Initialize()` calls `NewTimeStep()` after loading the saved step;
`NewTimeStep()` permits a factor-of-two growth. Capping the next step so
the compared outputs have **the same time** still leaves a maximum active
`E_rad` difference of about `1.16e-3` with the shared-face prototype.

The same-time difference also occurs with the conservative prototype **off**:
maximum `E_rad` difference `7.92e-3` on static AMR and on moving AMR. A
uniform-grid control agrees to `2.22e-16`. This isolates a pre-existing
coupled AMR restart problem rather than a new shared-flux error or a
regridding-specific effect. At the cycle-2 checkpoint, active conserved
variables and radiation agree exactly; primitive pressure and cached gas
energy agree to roundoff. With ghost-zone HDF5 output, immediately after
restart the x1 coarse/fine ghosts differ by up to `0.183` in pressure,
`0.947` in `E_rad`, and `0.00141` in velocity, while active cells agree.
The likely issue is reconstruction of the AMR boundary state after restart,
but its exact cause has not been proved. A temporary test that synchronized
the radiation ghosts reduced their difference to zero; pressure ghosts and
the next-step field discrepancy remained. That trial was reverted.

Reproduction artifacts: `/tmp/wor105_static_restart_diag_20261106`,
`/tmp/wor105_uniform_restart_diag_20261106`,
`/tmp/wor105_baseline_restart_diag_20261106`,
`/tmp/wor105_coupled_restart_diag_20261106`, and
`/tmp/wor105_ghost_restart_diag_20261106`.

## Decision and remaining gate

The local conservation, variable-limiter, direction, MPI, moving-AMR, and
physical-boundary checks pass for the tested configurations. The generic
coupled AMR restart discrepancy needs a separate fix and same-time
regression. The series-resistance trial remains opt-in for radiation-only
Test C pending a physically justified variable-limiter face closure.
WOR-105's default switch also requires a resolved Mach 5 accuracy comparison
and explicit guards for unsupported configurations. No XD2000 access was
needed for these local checks.
