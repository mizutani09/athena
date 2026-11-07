# WOR-92: clean CFL=0.3 continuation through cycle 200

This is a bounded local continuation of the Tomida-family table EoS and Semenov opacity pilot. It uses the committed mixed-frame general-EoS exact local gas root, bounded true-residual inexact Newton gate, accepted-state opacity table-domain guard, and mean-preserving positive AMR radiation ghost reconstruction. The source is on the isolated branch `codex/wor92-domain-guard-trial-20261107`: local-root commit `6da81d20`, opacity guard `ff632195`, and positive ghost commit `14633c31`. It does not demonstrate first-core formation.

## Result

From the saved cycle-100 state at `t=2266.625011212` code units, three fresh output directories reach cycle 120, 160, and 200 at CFL=0.3. All **100/100** subsequent implicit Newton stages converge in at most three iterations, with the largest final maximum residual `7.66067735e-6` below the configured `1e-5`. There are **300 FGMRES solves**. Their maximum independently reported true relative residual is `1.98339554e-8`, below the strict `2e-8` target; the opt-in inexact gate is enabled, but **none of these 300 late solves exceeds the strict target**. Thus this continuation tests the late solver path, while the need for the bounded inexact gate in earlier cycles is established separately in [the early pilot](wor92_inexact_newton_20261107.md).

| Cycles | Actual dt range | Min accepted active T | Max local gas residual / gas scale | End ρmax | End Tgas,max |
|---|---:|---:|---:|---:|---:|
| 100–119 | 8.3240–8.8060 | 15.465 K | 1.634e-13 | 4.73882e-14 g/cm³ | 20.0534 K |
| 120–159 | 7.6803–8.3140 | 14.962 K | 1.430e-13 | 5.74911e-14 g/cm³ | 17.8822 K |
| 160–199 | 7.3397–7.6687 | 14.674 K | 1.185e-13 | 6.29933e-14 g/cm³ | 16.4158 K |

The opacity diagnostics report zero out-of-domain or clamped trial lookups in all three continuations. Every accepted active-cell temperature remains above the table's 5 K lower bound. The positive ghost limiter does not activate in these 100 cycles. Each run has 120 MeshBlocks and zero block creations/destructions. These observations do not test the accepted-state guard after a new regrid; its current hook runs before regrid.

The final density is over three orders of magnitude below the `1e-10 g/cm³` first-core comparison threshold. The earlier cycle-87–88 ghost correction and cycle-92 Newton energy-balance sensitivity remain numerically material; see the [ghost ledger](wor92_positive_ghost_ledger_20261107.md) and [same-checkpoint Newton test](wor92_newton_energy_gate_20261107.md). WOR-92 remains **In Progress** pending physical collapse, matched-time CFL/mesh convergence, regrid-domain and conservation checks, and opacity model-definition review. The Semenov/Optab definition difference is a separate physics question.

## Reproduction and evidence

Run `python3 tst/nrfld_validation/wor92_long_continuation_20261107.py` from this worktree. Its assertions check the 100 Newton stages, cycle limits, accepted temperatures, no table clamp, and absence of fatal/ghost events. It produces [summary JSON](wor92_long_continuation_results_20261107/summary.json), [case CSV](wor92_long_continuation_results_20261107/cases.csv), [cycle CSV](wor92_long_continuation_results_20261107/cycles.csv), [history CSV](wor92_long_continuation_results_20261107/history.csv), [figure](wor92_long_continuation_results_20261107/continuation.png), and a [compressed archive](wor92_long_continuation_results_20261107/evidence.tar.gz) of each full log and history. The JSON/CSV list SHA-256 for the binary, input, starting checkpoint, each log, history, and final restart. Binary SHA-256: `0e6af159f3c545d765667ab45a3b7feaad7a3ddbb0c05e8897633822c86f5c1b`.

For each chunk, use the final restart of the previous chunk (`table_domain_mean_limiter_restart74_to100_2rank/cfl030_flux_precision16_local.final.rst` for the first). From `/home/kosuke/simulation/test/rad_nr/wor92_20261107/bundle`, use a fresh output directory:

```bash
env ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE=1 \
  ATHENA_NRFLD_CONSERVATIVE_SHARED_COEFF=1 \
  ATHENA_NRFLD_EXPERIMENTAL_TANGENTIAL_RECONSTRUCTION=1 \
  ATHENA_NRFLD_EXPERIMENTAL_VARIABLE_LIMITER=1 \
  ATHENA_NRFLD_EXPERIMENTAL_COUPLED=1 \
  ATHENA_NRFLD_EXPERIMENTAL_REGRID=1 \
  ATHENA_NRFLD_CONSERVATIVE_DIFFERENCE_JV=0 \
  ATHENA_WOR92_INEXACT_TRUE_CAP=1e-7 \
  ATHENA_WOR92_POSITIVE_NR_GHOST=2 \
  timeout 180s mpirun -np 2 \
  /home/kosuke/simulation/test/rad_nr/wor92_operator_audit_20261107/build_table_domain_guard_trial/bin/athena \
  -r /path/to/previous.final.rst \
  -i /home/kosuke/simulation/test/rad_nr/wor92_wor105_retest_20261107/input_table_domain_guard_cfl030_restart.used \
  -d /path/to/fresh/output time/nlim=120 time/tlim=10000.0
```

Set `time/nlim` to 160 and 200 for the second and third chunks. The archived logs are the direct command output. No XD2000 access or merge occurred.
